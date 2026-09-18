"""Recurrent Quadrature Filter parameterization.

``RQF``
    Atomic parameter module (``nn.Module``). Stores learnable gamma, freq, tau
    and computes discretized coefficients C1, C2. Used internally by
    :class:`~rqf_models.backbones.rqf.block.BaseRQFBlock` — does **not** run a
    scan or recurrence itself.
"""

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

DiscretizationType = Literal["euler", "zoh", "bilinear", "foh"]


def _expm1_div_x_complex(x: torch.Tensor, threshold: float = 1e-4) -> torch.Tensor:
    """Compute (exp(x) - 1) / x for complex x in a numerically stable way.

    For small |x|, uses Taylor series to avoid catastrophic cancellation.
    For larger |x|, uses direct computation.

    Args:
        x: Complex tensor of any shape.
        threshold: Threshold below which to use Taylor series approximation.

    Returns:
        Complex tensor with same shape as x, containing (exp(x) - 1) / x.

    Notes:
        The Taylor series expansion is: 1 + x/2 + x^2/6 + x^3/24 + x^4/120 + ...
        This is accurate for |x| < 0.1 and avoids numerical issues when x ≈ 0.
    """
    x_abs = torch.abs(x)
    small_mask = x_abs < threshold

    # Direct computation for large |x|
    exp_x = torch.exp(x)
    # Avoid division by zero by adding small epsilon where x is small
    x_safe = torch.where(small_mask, torch.ones_like(x), x)
    direct = (exp_x - 1.0) / x_safe

    # Taylor series for small |x|: 1 + x/2 + x^2/6 + x^3/24 + x^4/120
    taylor = 1.0 + x / 2 + x**2 / 6 + x**3 / 24 + x**4 / 120

    return torch.where(small_mask, taylor, direct)


def _expm1_minus_x_div_x2_complex(x: torch.Tensor, threshold: float = 1e-3) -> torch.Tensor:
    """Compute (exp(x) - 1 - x) / x² for complex x in a numerically stable way.

    For small |x|, uses Taylor series to avoid the double cancellation in
    ``exp(x) - 1 - x``. For larger |x|, uses direct computation.

    Args:
        x: Complex tensor of any shape.
        threshold: Threshold below which to use Taylor series approximation.
            Larger than the threshold in ``_expm1_div_x_complex`` because the
            ``exp(x) - 1 - x`` cancellation loses roughly twice as many digits.

    Returns:
        Complex tensor with same shape as x, containing (exp(x) - 1 - x) / x².

    Notes:
        Taylor expansion: 1/2 + x/6 + x²/24 + x³/120 + x⁴/720.

        Used to evaluate the first-moment kernel integral
        Ψ_h = ∫₀ʰ s·e^{λ(h-s)} ds = h² · this(λh),
        which appears in the FOH (first-order-hold) discretization for both
        prospective and non-prospective inputs.
    """
    x_abs = torch.abs(x)
    small_mask = x_abs < threshold

    exp_x = torch.exp(x)
    x_safe = torch.where(small_mask, torch.ones_like(x), x)
    direct = (exp_x - 1.0 - x) / (x_safe * x_safe)

    taylor = 0.5 + x / 6 + x**2 / 24 + x**3 / 120 + x**4 / 720

    return torch.where(small_mask, taylor, direct)


class RQF(nn.Module):
    """Recurrent Quadrature Filter parameterization.

    Stores learnable filter parameters (gamma, freq, tau) and computes
    discretized recurrence coefficients C1, C2 for the linear recurrence::

        x[n+1] = C1 * x[n] + C2 * u[n]

    This is the atomic building block used by :class:`BaseRQFBlock` and its
    subclasses. It does **not** run a scan — the caller is responsible for
    executing the recurrence (via ``rqf_models.ops.scan`` or Triton kernels).

    Args:
        num_filters: Number of parallel filter banks. Each filter bank has
            independent learnable parameters.
        tau: Initial time constant for the filter dynamics. Controls the
            temporal scale of the filter response. Scalar applies to all
            filters; a sequence of length num_filters sets per-filter values,
            enabling multi-scale temporal analysis. If None, defaults to
            dt * 10 with a warning.
        dt: Discrete time step for filter integration. Should match or be
            smaller than the sampling period of input signals. Default: 0.0001.
        learn_tau: Whether to make tau a learnable parameter. If True, tau is
            optimized during training. Default: False.
        gamma: Initial damping coefficient (gamma). If None, initialized at
            the critical damping point per filter. Default: None.
        freq: Initial oscillation frequency in Hz. If None, randomly initialized
            within [min_freq, max_freq]. Default: None.
        min_gamma: Minimum allowed damping coefficient during optimization.
            Scalar or per-filter sequence. Default: 0.01.
        max_gamma: Maximum allowed damping coefficient during optimization.
            Scalar or per-filter sequence. Default: 0.2.
        min_freq: Minimum allowed frequency in Hz during optimization.
            Scalar or per-filter sequence. Default: 1.0.
        max_freq: Maximum allowed frequency in Hz during optimization.
            Scalar or per-filter sequence. Default: 50.0.
        discretization: Method for discretizing continuous-time dynamics.
            - "euler": Forward Euler (default). Simple but may require small dt.
            - "zoh": Zero-Order Hold. Exact discretization, stable for any dt.
            - "bilinear": Bilinear transform (Tustin). Best frequency preservation.
        use_b_matrix: Whether to use B matrix for multi-dimensional input projection.
            If True, enables SSM-style input handling: x[n+1] = C1*x[n] + (C2 ⊙ B)@u[n].
            Default: False (backward compatible).
        input_features: Number of input features when use_b_matrix=True.
            B matrix will have shape (num_filters, input_features).
            Default: 1 (single-channel input).
        clamp_gamma: Whether to dynamically clamp gamma based on learned freq/tau.

    Notes:
        The continuous-time system is: dx/dt = A*x + B*u where:
            α = γ/τ  (damping rate)
            A = -α + i*2π*freq*(1-γ)
            B = α

        The discretization method determines how C1 and C2 are computed:

        Forward Euler (discretization="euler"):
            C1 = 1 + dt*A, C2 = dt*B

        Zero-Order Hold (discretization="zoh"):
            C1 = exp(A*dt), C2 = (exp(A*dt) - 1) / A * B

        Bilinear Transform (discretization="bilinear"):
            C1 = (1 + A*dt/2) / (1 - A*dt/2), C2 = dt * B / (1 - A*dt/2)

        Parameters gamma and frequency are constrained via sigmoid:
            γ = min_gamma + (max_gamma - min_gamma) * sigmoid(gamma_param)
            f = min_freq + (max_freq - min_freq) * sigmoid(freq_param)
    """

    @staticmethod
    def _to_buffer(val: float | Sequence[float], size: int) -> torch.Tensor:
        """Convert scalar or sequence to a 1-D tensor of length ``size``."""
        if isinstance(val, Sequence):
            t = torch.tensor(val, dtype=torch.float32)
            if t.shape[0] != size:
                raise ValueError(f"Expected sequence of length {size}, got {t.shape[0]}")
            return t
        return torch.full((size,), val, dtype=torch.float32)

    def __init__(
        self,
        num_filters: int = 1,
        tau: float | Sequence[float] | None = None,
        dt: float = 0.0001,
        learn_tau: bool = False,
        learn_freq: bool = True,
        learn_gamma: bool = True,
        gamma: float | None = None,
        freq: float | None = None,
        min_gamma: float | Sequence[float] = 0.01,
        max_gamma: float | Sequence[float] = 0.2,
        min_freq: float | Sequence[float] = 1.0,
        max_freq: float | Sequence[float] = 50.0,
        seq_duration: float | None = None,
        filter_max_freq_factor: float = 1.0,
        param_space: str = "sigmoid",
        discretization: DiscretizationType = "euler",
        use_b_matrix: bool = False,
        input_features: int = 1,
        clamp_gamma: bool = False,
    ) -> None:
        """Initialize the RQF parameter module.

        ``param_space`` selects the leaf parametrization:

        - ``"sigmoid"`` (default): logit leaves, gamma/freq bounded via
          sigmoid into ``[min, max]`` at every read.
        - ``"log"``: log-space leaves ``log_theta`` (theta = 2*pi*freq*tau)
          and ``log_gamma``, mirroring ``log_tau``: updates are multiplicative
          and O(1) across the frequency range; bounds are enforced by calling
          :meth:`clamp_parameters` after each optimizer step. With
          ``seq_duration`` set (the resonant recipe), the frequency band is
          derived as ``[1/T, 1/tau]`` (theta in ``[2*pi*tau/T, 2*pi]``), the
          gamma band is ``~[0, 1]``, and ``log_theta`` is initialized
          log-uniformly in ``[log(2*pi*tau/T), log(2*pi*filter_max_freq_factor)]``.
        """
        super().__init__()
        valid_discretizations = ("euler", "zoh", "bilinear", "foh")
        if discretization not in valid_discretizations:
            raise ValueError(
                f"discretization must be one of {valid_discretizations}, got {discretization!r}"
            )
        if param_space not in ("sigmoid", "log"):
            raise ValueError(f"param_space must be 'sigmoid' or 'log', got {param_space!r}")
        if seq_duration is not None and param_space != "log":
            raise ValueError("seq_duration (resonant recipe) requires param_space='log'")

        self.num_filters = num_filters
        self.dt = dt
        self.learn_tau = learn_tau
        self.learn_freq = learn_freq
        self.learn_gamma = learn_gamma
        self.param_space = param_space
        self.seq_duration = seq_duration
        self.filter_max_freq_factor = filter_max_freq_factor
        self.discretization = discretization
        self.use_b_matrix = use_b_matrix
        self.input_features = input_features
        self.clamp_gamma = clamp_gamma
        # Lowers the resting log_gamma ceiling in clamp_parameters. Set by
        # SelectiveController for the log_shift gamma link so the resting
        # value plus the bounded input-dependent shift stays below max_gamma.
        self.gamma_log_headroom = 0.0

        # Default tau to dt * 10 if not specified
        if tau is None:
            tau = dt * 10
            warnings.warn(
                f"tau not specified, defaulting to dt*10 = {tau}",
                stacklevel=2,
            )

        tau_tensor = self._to_buffer(tau, num_filters)

        # Resolve the frequency / damping bounds. In the resonant recipe
        # (seq_duration set) the bands derive from T and tau and the passed
        # Hz/gamma bound args are ignored.
        recipe = seq_duration is not None
        if recipe:
            min_freq_t = self._to_buffer(1.0 / seq_duration, num_filters)
            max_freq_t = 1.0 / tau_tensor  # per-filter; theta clamp ceiling = 2*pi
            min_gamma_t = self._to_buffer(1e-6, num_filters)
            max_gamma_t = self._to_buffer(1.0 - 1e-6, num_filters)
        else:
            min_freq_t = self._to_buffer(min_freq, num_filters)
            max_freq_t = self._to_buffer(max_freq, num_filters)
            min_gamma_t = self._to_buffer(min_gamma, num_filters)
            max_gamma_t = self._to_buffer(max_gamma, num_filters)

        # Store bounds as registered buffers (saved in state_dict, auto device transfer)
        self.register_buffer("_min_gamma", min_gamma_t)
        self.register_buffer("_max_gamma", max_gamma_t)
        self.register_buffer("_min_freq", min_freq_t)
        self.register_buffer("_max_freq", max_freq_t)

        # Time constant parameter (log-transformed, per-filter)
        self.log_tau = nn.Parameter(torch.log(tau_tensor), requires_grad=learn_tau)

        if param_space == "log":
            self._init_log_space(gamma, freq, tau_tensor, recipe)
        else:
            self._init_sigmoid_space(gamma, freq, tau_tensor)

        # Optional B matrix for multi-dimensional input projection
        if use_b_matrix:
            # Initialize with Lecun normal: std = 1 / sqrt(input_features)
            std = 1.0 / np.sqrt(input_features)
            self.B = nn.Parameter(torch.randn(num_filters, input_features) * std)
        else:
            self.B = None

    def _init_sigmoid_space(self, gamma, freq, tau_tensor) -> None:
        """Logit leaves bounded via sigmoid (the original parametrization)."""
        num_filters = self.num_filters
        if freq is None:
            freq_init_val = self._min_freq + torch.rand(num_filters) * (
                self._max_freq - self._min_freq
            )
        else:
            freq_init_val = torch.full((num_filters,), freq)

        scaled_freq = (freq_init_val - self._min_freq) / (self._max_freq - self._min_freq)
        self.freq_param = nn.Parameter(
            torch.logit(torch.clamp(scaled_freq, 1e-6, 1.0 - 1e-6)),
            requires_grad=self.learn_freq,
        )

        # Warn when freq * tau >= 1 (some filters may become low-pass-like)
        with torch.no_grad():
            max_freq_tau = (self._max_freq * tau_tensor).max().item()
            if max_freq_tau >= 1.0:
                warnings.warn(
                    f"freq * tau >= 1 (max_freq={self._max_freq.max().item():.4f}, "
                    f"max_tau={tau_tensor.max().item():.4f}, product={max_freq_tau:.4f}). "
                    f"Some filters may behave as low-pass. Consider reducing tau or max_freq.",
                    UserWarning,
                    stacklevel=2,
                )

        # Damping parameter (logit-transformed for bounded optimization)
        # When gamma is not specified, initialize at gamma_critical per filter
        if gamma is None:
            omega_tau = 2 * np.pi * freq_init_val * tau_tensor
            gamma_crit = omega_tau / (1 + omega_tau)
            gamma_init_val = gamma_crit
        else:
            gamma_init_val = torch.full((num_filters,), gamma)

        scaled_gamma = (gamma_init_val - self._min_gamma) / (self._max_gamma - self._min_gamma)
        self.gamma_param = nn.Parameter(
            torch.logit(torch.clamp(scaled_gamma, 1e-6, 1.0 - 1e-6)),
            requires_grad=self.learn_gamma,
        )

    def _init_log_space(self, gamma, freq, tau_tensor, recipe: bool) -> None:
        """Log-space leaves ``log_theta``/``log_gamma`` (multiplicative updates).

        theta = 2*pi*freq*tau is the dimensionless discrete pole angle, an O(1)
        quantity, so learning ``log_theta`` keeps frequency updates
        multiplicative and the gradient O(1) across the whole resonant range.
        Bounds are enforced post-step by :meth:`clamp_parameters`.
        """
        num_filters = self.num_filters
        if freq is not None:
            log_theta_init = torch.log(2 * np.pi * torch.full((num_filters,), freq) * tau_tensor)
        elif recipe:
            # Log-uniform init: log_theta ~ Uniform[log(2*pi*tau/T),
            # log(2*pi*filter_max_freq_factor)]. The init ceiling is a sub-band
            # of the clamp ceiling 2*pi.
            log_theta_min = torch.log(2 * np.pi * tau_tensor / self.seq_duration)
            log_theta_max = torch.log(
                torch.full((num_filters,), 2 * np.pi * self.filter_max_freq_factor)
            )
            u = torch.rand(num_filters)
            log_theta_init = log_theta_min + u * (log_theta_max - log_theta_min)
        else:
            freq_init = self._min_freq + torch.rand(num_filters) * (self._max_freq - self._min_freq)
            log_theta_init = torch.log(2 * np.pi * freq_init * tau_tensor)

        self.log_theta = nn.Parameter(log_theta_init, requires_grad=self.learn_freq)

        # Damping learned in log-space (gamma = exp(log_gamma)); critical init.
        if gamma is None:
            theta_init = torch.exp(log_theta_init)  # = 2*pi*freq*tau (= omega*tau)
            gamma_init_val = theta_init / (1.0 + theta_init)
        else:
            gamma_init_val = torch.full((num_filters,), gamma)
        gamma_init_clamped = torch.clamp(gamma_init_val, self._min_gamma, self._max_gamma)
        self.log_gamma = nn.Parameter(
            torch.log(gamma_init_clamped.clone().float()),
            requires_grad=self.learn_gamma,
        )

    @property
    def theta(self) -> torch.Tensor:
        """Dimensionless pole angle ``theta = omega*tau = 2*pi*freq*tau``."""
        if self.param_space == "log":
            return torch.exp(self.log_theta)
        return 2 * np.pi * self.freq * self.tau

    @torch.no_grad()
    def clamp_parameters(self) -> None:
        """Project log-space leaves back into their valid bands (post-step).

        No-op in the sigmoid parametrization (bounds are intrinsic there).
        ``log_theta`` is clamped to the [min_freq, max_freq] band (in
        dimensionless ``log(2*pi*f*tau)`` units); ``log_gamma`` to
        [log(min_gamma), log(max_gamma)]. With ``clamp_gamma``, gamma is
        additionally projected strictly inside the dynamic stability band
        [gamma_min(theta), gamma_crit(theta)] so the gradient stays alive for
        filters that touch the boundary.
        """
        if self.param_space != "log":
            return
        tau = self.tau
        self.log_theta.clamp_(
            min=torch.log(2 * np.pi * self._min_freq * tau),
            max=torch.log(2 * np.pi * self._max_freq * tau),
        )
        self.log_gamma.clamp_(min=torch.log(self._min_gamma),
                             max=torch.log(self._max_gamma) - self.gamma_log_headroom)
        if self.clamp_gamma:
            omega_tau = self.log_theta.exp()
            gamma_min = 1 - 1 / torch.sqrt(1 + omega_tau**2)
            gamma_crit = omega_tau / (1 + omega_tau)
            eps = 1e-6
            self.log_gamma.copy_(
                torch.minimum(
                    torch.maximum(self.log_gamma, gamma_min.clamp_min(1e-12).log() + eps),
                    gamma_crit.log() - eps,
                )
            )

    @property
    def tau(self) -> torch.Tensor:
        """Time constant of the filter dynamics.

        Returns:
            Time constant, shape (num_filters,). Always positive via exp
            transformation.
        """
        return torch.exp(self.log_tau)

    @property
    def gamma(self) -> torch.Tensor:
        """Damping coefficient (gamma) controlling filter response.

        Returns:
            Damping coefficient, shape (num_filters,). Sigmoid space: bounded
            in [min_gamma, max_gamma] at every read. Log space:
            ``exp(log_gamma)``, bounded post-step by :meth:`clamp_parameters`.
        """
        if self.param_space == "log":
            return torch.exp(self.log_gamma)
        return self._min_gamma + (self._max_gamma - self._min_gamma) * torch.sigmoid(
            self.gamma_param
        )

    @property
    def freq(self) -> torch.Tensor:
        """Oscillation frequency of the filter in Hz.

        Returns:
            Frequency in Hz, shape (num_filters,). Sigmoid space: bounded in
            [min_freq, max_freq] at every read. Log space: derived from the
            trainable ``theta = 2*pi*freq*tau``.
        """
        if self.param_space == "log":
            return self.theta / (2 * np.pi * self.tau)
        return self._min_freq + (self._max_freq - self._min_freq) * torch.sigmoid(self.freq_param)

    def compute_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute filter coefficients C1 and C2 using the selected discretization.

        Returns:
            Tuple of (C1, C2) complex tensors with shape (num_filters,).
            C1 is the state transition coefficient, C2 is the input gain.
        """
        tau = self.tau
        gamma = self.gamma
        freq = self.freq

        # Dynamic γ clamping based on current learned frequency and tau.
        # γ_min = 1 - 1/√(1 + (ωτ)²), γ_crit = ωτ/(1 + ωτ)
        if self.clamp_gamma:
            omega_tau = 2 * np.pi * freq * tau
            gamma_min = 1 - 1 / torch.sqrt(1 + omega_tau**2)
            gamma_crit = omega_tau / (1 + omega_tau)
            gamma = torch.clamp(gamma, min=gamma_min, max=gamma_crit)

        # Continuous-time coefficients (same for all discretization methods)
        # α = γ/τ  (damping rate)
        # A = -α + i*2π*freq*(1-γ)
        # B = α
        alpha = gamma / tau
        A = -alpha + 1j * 2 * np.pi * freq * (1.0 - gamma)
        B = alpha

        if self.discretization == "euler":
            # Forward Euler: C1 = 1 + dt*A, C2 = dt*B
            dt_ratio = self.dt / tau
            w = 1.0 + 1j * 2 * np.pi * tau * freq
            C1 = 1.0 + dt_ratio * (-1.0 + (1.0 - gamma) * w)
            C2 = dt_ratio * gamma

        elif self.discretization == "zoh":
            # Zero-Order Hold: C1 = exp(A*dt), C2 = (exp(A*dt) - 1) / A * B
            A_dt = A * self.dt
            C1 = torch.exp(A_dt)
            C2 = self.dt * _expm1_div_x_complex(A_dt) * B

        elif self.discretization == "bilinear":
            # Bilinear (Tustin): C1 = (1 + A*dt/2) / (1 - A*dt/2)
            # C2 = dt * B / (1 - A*dt/2)
            A_dt_half = A * self.dt / 2
            denominator = 1.0 - A_dt_half
            C1 = (1.0 + A_dt_half) / denominator
            C2 = self.dt * B / denominator

        else:  # foh
            # FOH is intrinsically a two-tap update on z (eq. 38 for the
            # non-prospective case, eq. 45 for the prospective case). It does
            # not fit the single-tap (C1, C2) signature.
            raise NotImplementedError(
                "discretization='foh' requires a two-tap update; call "
                "compute_two_tap_coefficients(prospective=...) instead."
            )

        return C1, C2

    def compute_two_tap_coefficients(
        self,
        prospective: bool = True,
        mode: str = "tau",
        prospective_tau: float | Sequence[float] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute C1 and two-tap input coefficients ``(tap0, tap1)``.

        Provides the coefficients for the two-tap recurrence::

            y[n+1] = C1 · y[n] + tap0 · u[n] + tap1 · u[n-1]

        This covers the prospective variants of Forward Euler and ZOH
        (impulse form), and the First-Order Hold (FOH) in both prospective
        and non-prospective modes (§4.2, §4.3, §5 of the discretization
        document).

        Args:
            prospective: If ``True``, return taps for a prospective
                (look-ahead) drive. If ``False``, return the non-prospective
                FOH taps (only valid for ``"foh"``).
            mode: Which prospective operator to discretize when
                ``prospective=True``. ``"tau"`` uses the uniform operator
                ``χ_τ = 1 + τ·d/dt`` (§3, §4). ``"gamma"`` uses the
                filter-matched operator ``χ_γ = 1 + (τ/γ)·d/dt`` (§5),
                whose look-ahead equals each neuron's own decay time
                ``τ/γ``. Ignored when ``prospective=False`` (non-prospective
                FOH is mode-invariant — γ only enters through α).
            prospective_tau: Optional independent prospective look-ahead
                timescale in seconds. When provided, it overrides ``mode`` for
                the derivative/look-ahead term only while leaving the RQF
                filter dynamics ``tau`` unchanged. ``prospective_tau=0``
                exactly recovers the corresponding non-prospective two-tap
                form (or the standard single-tap coefficient for Euler/ZOH).

        Returns:
            Tuple of ``(C1, tap0, tap1)`` where:

            - **C1** – state transition coefficient, shape ``(num_filters,)``,
              complex. ``exp(λ·dt)`` for ``"zoh"``/``"foh"`` and the linearised
              ``1 + (dt/τ)·(p−1)`` for ``"euler"``.
            - **tap0** – lag-0 tap on *u* (coefficient of ``u[n]``),
              shape ``(num_filters,)``.
            - **tap1** – lag-1 tap on *u* (coefficient of ``u[n-1]``),
              shape ``(num_filters,)``.

        Raises:
            ValueError: If ``mode`` is not ``"tau"`` or ``"gamma"``.
            NotImplementedError: If ``discretization`` is ``"bilinear"``
                (no two-tap update defined), or if ``discretization`` is
                ``"euler"``/``"zoh"`` and ``prospective=False`` (those use the
                single-tap :meth:`compute_coefficients`).

        Notes:
            With ``phi = (e^{λ·dt} − 1)/(λ·dt)`` and
            ``psi = (e^{λ·dt} − 1 − λ·dt)/(λ·dt)²``, and ``η = dt/τ``:

            With ``ρ = prospective_tau / tau`` when ``prospective_tau`` is
            specified, otherwise ``ρ = 1`` for τ-mode and ``ρ = 1/γ`` for
            γ-mode:

            Forward Euler (prospective only):
                ``tap0 = γ·η + γ·ρ``, ``tap1 = −γ·ρ``

            ZOH / impulse-prospective (prospective only):
                ``tap0 = γ·η·phi + γ·ρ·C1``, ``tap1 = −γ·ρ·C1``

            FOH, prospective:
                ``tap0 = γ·η·psi + γ·ρ·phi``
                ``tap1 = γ·η·(phi − psi) − γ·ρ·phi``

            FOH, non-prospective (mode-invariant):
                ``tap0 = γ·η·psi``,  ``tap1 = γ·η·(phi − psi)``

            γ-mode reduces to τ-mode at γ=1 (the α·(τ/γ)=1 identity
            eliminates the γ prefactor on every prospective tap).
        """
        if mode not in ("tau", "gamma"):
            raise ValueError(f"mode must be one of ('tau', 'gamma'), got {mode!r}")

        tau = self.tau
        gamma = self.gamma
        freq = self.freq

        if self.clamp_gamma:
            omega_tau = 2 * np.pi * freq * tau
            gamma_min = 1 - 1 / torch.sqrt(1 + omega_tau**2)
            gamma_crit = omega_tau / (1 + omega_tau)
            gamma = torch.clamp(gamma, min=gamma_min, max=gamma_crit)

        if prospective_tau is None:
            rho = torch.ones_like(gamma) if mode == "tau" else 1.0 / gamma
        else:
            tau_p = torch.as_tensor(prospective_tau, dtype=tau.dtype, device=tau.device)
            rho = tau_p / tau

        alpha = gamma / tau
        A = -alpha + 1j * 2 * np.pi * freq * (1.0 - gamma)
        A_dt = A * self.dt

        if self.discretization == "bilinear":
            raise NotImplementedError(
                "Two-tap update is not defined for bilinear discretization. "
                "Use 'euler', 'zoh', or 'foh'."
            )

        if self.discretization == "euler":
            if not prospective:
                raise NotImplementedError(
                    "Non-prospective Euler uses the single-tap recurrence; "
                    "call compute_coefficients() instead."
                )
            dt_ratio = self.dt / tau
            w = 1.0 + 1j * 2 * np.pi * tau * freq
            C1 = 1.0 + dt_ratio * (-1.0 + (1.0 - gamma) * w)
            extra = gamma * rho
            tap0 = gamma * dt_ratio + extra
            tap1 = -extra
            return C1, tap0, tap1

        # "zoh" or "foh": exact exponential state evolution
        C1 = torch.exp(A_dt)
        phi = _expm1_div_x_complex(A_dt)
        B_bar = (self.dt / tau) * phi  # γ·B_bar = B_h = (α/λ)(e^{λh}-1)

        if self.discretization == "zoh":
            if not prospective:
                raise NotImplementedError(
                    "Non-prospective ZOH uses the single-tap recurrence; "
                    "call compute_coefficients() instead."
                )
            # Impulse ZOH prospective generalized by ρ:
            # base = γ·B_bar, extra = γ·ρ·C1.
            extra = gamma * rho * C1
            tap0 = gamma * B_bar + extra
            tap1 = -extra
            return C1, tap0, tap1

        # "foh": first-order hold on z across [t_n, t_{n+1}]
        psi = _expm1_minus_x_div_x2_complex(A_dt)
        D_bar = (self.dt / tau) * psi  # γ·D_bar = D_h = (1/h)·α·Ψ_h

        if prospective:
            # FOH prospective generalized by ρ:
            # base0 = γ·D_bar, base1 = γ·(B_bar − D_bar),
            # extra = γ·ρ·phi.
            extra = gamma * rho * phi
            tap0 = gamma * D_bar + extra
            tap1 = gamma * (B_bar - D_bar) - extra
        else:
            # FOH non-prospective (eq. 38 grouped): mode-invariant.
            # tap0 = D_h; tap1 = B_h − D_h
            tap0 = gamma * D_bar
            tap1 = gamma * (B_bar - D_bar)
        return C1, tap0, tap1

    def compute_prospective_taps(
        self,
        mode: str = "tau",
        prospective_tau: float | Sequence[float] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward-compat alias for ``compute_two_tap_coefficients(prospective=True)``.

        Args:
            mode: Prospective-operator timescale, ``"tau"`` (uniform τ) or
                ``"gamma"`` (filter-matched τ/γ). Default ``"tau"`` preserves
                historical behavior.
            prospective_tau: Optional independent prospective look-ahead
                timescale in seconds.

        Returns:
            Tuple of ``(C1, tap0, tap1)`` for the prospective-input two-tap
            recurrence. See :meth:`compute_two_tap_coefficients` for details.
        """
        return self.compute_two_tap_coefficients(
            prospective=True,
            mode=mode,
            prospective_tau=prospective_tau,
        )
