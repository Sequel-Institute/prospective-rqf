"""Base RQF block with shared scan infrastructure.

Provides the common foundation for S4-style, Mamba-style, and Pure RQF blocks:
RQF filter construction, coefficient computation, scan execution,
multi-oscillator C-weight combination, and readout modes.

Subclasses implement their own ``__init__`` projection layers and
``forward()`` method.
"""

import math
from typing import ClassVar

import torch
import torch.nn as nn

from organics.embedding.base import EmbeddingBase
from organics.embedding.rqf.filter import RQF
from organics.embedding.rqf.utils.activations import SplitReLU, SplitTanh
from organics.embedding.rqf.utils.divisive_normalization import DivisiveNormalization
from organics.ops.scan import parallel_scan, sequential_scan, sequential_scan_abs

# Fused multi-oscillator scan (optional, requires Triton + CUDA)
try:
    from organics.ops.triton_scan_kernel import (
        _sequential_scan_multi_osc_forward_abs_raw,
        _sequential_scan_multi_osc_real_forward_abs_raw,
        _triton_parallel_scan_multi_osc,
        _triton_sequential_scan_multi_osc,
        _triton_sequential_scan_multi_osc_real,
    )

    _HAS_FUSED_MULTI_OSC = True
except ImportError:
    _HAS_FUSED_MULTI_OSC = False


def _halfwave_rectify_4ch(z: torch.Tensor) -> torch.Tensor:
    """Half-wave rectified 4-channel decomposition.

    Decomposes complex z into 4 non-negative channels preserving partial
    phase information: [relu(Re), relu(-Re), relu(Im), relu(-Im)].

    Squaring is NOT applied here — callers are responsible for squaring
    when needed (e.g. the ``"4channel"`` readout mode), or delegating to
    :class:`DivisiveNormalization` which squares internally.

    Args:
        z: Complex tensor of shape (..., F).

    Returns:
        Real tensor of shape (..., 4*F).
    """
    re, im = z.real, z.imag
    return torch.cat(
        [
            torch.relu(re),
            torch.relu(-re),
            torch.relu(im),
            torch.relu(-im),
        ],
        dim=-1,
    )


def _halfwave_rectify_2ch(x: torch.Tensor) -> torch.Tensor:
    """Half-wave rectified 2-channel decomposition for real-valued input.

    Splits a real tensor into positive and negative parts:
    [relu(x), relu(-x)]. Use this after taking ``.real`` from a c-weight
    projection, where the imaginary part is redundant.

    Args:
        x: Real tensor of shape (..., F).

    Returns:
        Real tensor of shape (..., 2*F).
    """
    return torch.cat([torch.relu(x), torch.relu(-x)], dim=-1)


def _validate_readout_config(
    readout_mode: str, num_oscillators: int, c_projection: bool, activation: str = "gelu"
) -> None:
    """Validate readout mode, oscillator count, c_projection, and activation compatibility."""
    import warnings

    valid = (
        "magnitude",
        "4channel",
        "2channel",
        "real",
        "divnorm",
        "4channel_divnorm",
        "2channel_divnorm",
        "complex",
    )
    if readout_mode not in valid:
        raise ValueError(f"readout_mode must be one of {valid}, got {readout_mode!r}")

    # Validate activation ↔ readout compatibility
    _complex_acts = {"split_tanh", "split_relu", "identity"}
    _real_acts = {"silu", "gelu", "relu", "identity"}
    if readout_mode == "complex" and activation not in _complex_acts:
        raise ValueError(
            f"readout_mode='complex' requires a complex-compatible activation "
            f"{_complex_acts}, got {activation!r}"
        )
    if readout_mode != "complex" and activation not in _real_acts:
        raise ValueError(
            f"readout_mode={readout_mode!r} (real output) requires a real activation "
            f"{_real_acts}, got {activation!r}"
        )

    if readout_mode == "real" and num_oscillators == 1 and not c_projection:
        warnings.warn(
            "readout_mode='real' with num_oscillators=1 and no c_projection "
            "will produce carrier oscillations. Consider c_projection=True "
            "to add a learned complex weight: Re(c*z).",
            stacklevel=3,
        )
    if c_projection and readout_mode == "magnitude" and num_oscillators == 1:
        warnings.warn(
            "c_projection=True with readout_mode='magnitude' has no effect "
            "for N=1: |c*z| = |c|*|z|. Use readout_mode='real' to benefit "
            "from the learned complex projection: Re(c*z).",
            stacklevel=3,
        )


class BaseRQFBlock(EmbeddingBase):
    """Base class for RQF blocks with shared scan infrastructure.

    Handles RQF filter construction, coefficient computation, scan execution,
    multi-oscillator support, and readout modes. Subclasses add their own
    projection layers and implement ``forward()``.

    Args:
        d_model: Model dimension (input/output size).
        num_filters: Number of RQF filters per direction.
        samp_frequency: Sampling frequency in Hz.
        min_freq: Minimum learnable frequency.
        max_freq: Maximum learnable frequency.
        dropout: Dropout rate.
        activation: Activation function. Real-valued: ``"gelu"``, ``"relu"``,
            ``"silu"``. Complex-compatible: ``"split_tanh"``, ``"split_relu"``,
            ``"identity"``. Must be compatible with ``readout_mode``.
        use_parallel_scan: Use O(log T) Blelloch or O(T) fused.
        bidirectional: Use bidirectional scan.
        discretization: Discretization method (``"euler"``, ``"zoh"``,
            ``"bilinear"``, or ``"foh"``). FOH (first-order hold) always
            uses a two-tap update on ``z`` (equivalent to the prospective
            path in block dispatch). Bilinear is incompatible with
            ``prospective_input=True``.
        tau: Time constant for RQF filters (memory decay, in seconds).
        learn_tau: Whether tau is learnable (False recommended for stability).
        max_gamma: Maximum damping coefficient.
        min_gamma: Minimum damping coefficient.
        sigma_init: Initial sigma for divisive normalization.
        norm_type: Pre-normalization type (``"none"``, ``"layer"``, or
            ``"batch"``).  Default is ``"none"`` (no pre-norm).  When using a
            divisive-normalization readout (``"divnorm"``, ``"4channel_divnorm"``,
            ``"2channel_divnorm"``), pre-norm is discouraged because the readout
            already normalizes the signal.
        filter_scales: Optional per-filter configuration groups.
        readout_mode: Readout nonlinearity (``"magnitude"``, ``"4channel"``,
            ``"2channel"``, ``"real"``, ``"divnorm"``, ``"4channel_divnorm"``,
            ``"2channel_divnorm"``, ``"complex"``). The ``"complex"`` mode
            returns the raw complex scan output without conversion to real.
        num_oscillators: Default parallel oscillators per filter (S4-style N).
        c_projection: Add learnable complex output projection for N=1.
        clamp_gamma: Whether to clamp gamma values in RQF filters.
        d_model_out: Output dimension. When set, a residual projection D
            (``d_model → d_model_out``) is created. Subclasses should target
            ``self.d_model_out`` in their output projections. ``None`` means
            output equals ``d_model``.
        prospective_input: Use two-tap prospective input filter
            ``drive[n] = tap0·u[n] + tap1·u[n-1]``.
            Incompatible with ``"bilinear"`` discretization.
        prospective_mode: Which prospective operator to discretize when
            ``prospective_input=True``. ``"tau"`` (default) uses the uniform
            operator ``χ_τ = 1 + τ·d/dt``. ``"gamma"`` uses the filter-matched
            operator ``χ_γ = 1 + (τ/γ)·d/dt``, whose look-ahead equals each
            neuron's own decay time ``τ/γ``. Raises when set to a non-``"tau"``
            value without ``prospective_input=True``.
        prospective_tau: Optional independent look-ahead timescale in seconds.
            When set, it overrides ``prospective_mode`` for the two-tap
            derivative/look-ahead term only. ``prospective_tau=0`` makes the
            prospective two-tap coefficients collapse exactly to the
            non-prospective input coefficients while keeping the RQF dynamics
            ``tau`` unchanged.
        scan_grad_mode: Gradient mode for the temporal scan. ``"full"``
            backpropagates through the full recurrence. ``"tbptt1"``
            keeps the forward numerically identical to the scan, but
            truncates temporal gradients to one step by rebuilding the
            recurrence with ``h[t-1]`` detached.
        tbptt_delayed_input_mode: Diagnostic-only surrogate graph for the
            lag-1 explicit input tap in two-tap TBPTT(1). ``"strict"`` uses
            ``W(x[t-1].detach())``; ``"literal"`` uses the same projected
            input ``W(x[t-1])`` as the forward scan. This flag is for
            diagnostics and should not be treated as strict spatial BP.
    """

    _activations: ClassVar[dict[str, type[nn.Module]]] = {
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "split_tanh": SplitTanh,
        "split_relu": SplitReLU,
        "identity": nn.Identity,
    }

    @staticmethod
    def _validate_prospective_config(
        prospective_input: bool, prospective_mode: str, discretization: str
    ) -> None:
        """Validate the combination of prospective_input, prospective_mode, and discretization."""
        if prospective_input and discretization == "bilinear":
            raise ValueError(
                "prospective_input=True is not supported with bilinear discretization. "
                "Use 'euler', 'zoh', or 'foh'."
            )
        if prospective_mode not in ("tau", "gamma"):
            raise ValueError(f"prospective_mode must be 'tau' or 'gamma', got {prospective_mode!r}")
        if prospective_mode != "tau" and not prospective_input:
            raise ValueError(
                f"prospective_mode={prospective_mode!r} requires prospective_input=True."
            )

    @staticmethod
    def _validate_scan_grad_mode(scan_grad_mode: str) -> None:
        valid_scan_grad_modes = {"full", "tbptt1"}
        if scan_grad_mode not in valid_scan_grad_modes:
            raise ValueError(
                f"scan_grad_mode must be one of {valid_scan_grad_modes}, got {scan_grad_mode!r}"
            )

    @staticmethod
    def _validate_tbptt_delayed_input_mode(tbptt_delayed_input_mode: str) -> None:
        valid_modes = {"strict", "literal"}
        if tbptt_delayed_input_mode not in valid_modes:
            raise ValueError(
                "tbptt_delayed_input_mode must be one of "
                f"{valid_modes}, got {tbptt_delayed_input_mode!r}"
            )

    def __init__(
        self,
        d_model: int,
        num_filters: int,
        samp_frequency: float,
        min_freq: float,
        max_freq: float,
        dropout: float,
        activation: str,
        use_parallel_scan: bool,
        bidirectional: bool,
        discretization: str,
        tau: float,
        learn_tau: bool,
        max_gamma: float,
        min_gamma: float,
        sigma_init: float,
        norm_type: str,
        filter_scales: list[dict] | None,
        readout_mode: str,
        num_oscillators: int,
        c_projection: bool,
        clamp_gamma: bool = False,
        d_model_out: int | None = None,
        prospective_input: bool = False,
        prospective_mode: str = "tau",
        prospective_tau: float | list[float] | torch.Tensor | None = None,
        scan_grad_mode: str = "full",
        tbptt_delayed_input_mode: str = "strict",
    ) -> None:
        super().__init__()

        self._validate_prospective_config(prospective_input, prospective_mode, discretization)
        self.prospective_input = prospective_input
        self.prospective_mode = prospective_mode
        self.prospective_tau = prospective_tau
        self.discretization = discretization
        # FOH is intrinsically two-tap (eq. 38/45); route it through the 2-tap
        # scan path regardless of whether the user asked for prospective input.
        self._use_two_tap = prospective_input or discretization == "foh"
        self._validate_scan_grad_mode(scan_grad_mode)
        self.scan_grad_mode = scan_grad_mode
        self._validate_tbptt_delayed_input_mode(tbptt_delayed_input_mode)
        self.tbptt_delayed_input_mode = tbptt_delayed_input_mode

        # Output dimension (defaults to d_model when None)
        self.d_model_out = d_model_out if d_model_out is not None else d_model

        # Residual projection when output dimension differs from input
        self.residual_proj: nn.Module | None = (
            nn.Linear(d_model, self.d_model_out, bias=False)
            if self.d_model_out != d_model
            else None
        )

        self.num_filters = num_filters
        self.bidirectional = bidirectional
        self.use_parallel_scan = use_parallel_scan
        self.norm_type = norm_type
        self.num_oscillators = num_oscillators
        self.readout_mode = readout_mode

        _validate_readout_config(readout_mode, num_oscillators, c_projection, activation)

        # Total filters: 2x for bidirectional
        self.total_filters = num_filters * 2 if bidirectional else num_filters

        # Readout width: 4-channel modes output 4x, 2-channel modes output 2x
        if readout_mode in ("4channel", "4channel_divnorm"):
            self.readout_multiplier = 4
        elif readout_mode in ("2channel", "2channel_divnorm"):
            self.readout_multiplier = 2
        else:
            self.readout_multiplier = 1
        self.scan_output_width = self.total_filters * self.readout_multiplier

        # Activation function
        if activation not in self._activations:
            raise ValueError(
                f"Unknown activation: {activation!r}. Must be one of {set(self._activations)}"
            )
        self.act = self._activations[activation]()

        # Normalization (prenorm): LayerNorm, BatchNorm1d, or None
        self._use_batch_norm = norm_type == "batch"
        self.norm = self._create_prenorm(norm_type, readout_mode, d_model)

        # RQF parameters
        dt = 1.0 / samp_frequency

        # Parse filter configuration and oscillator setup
        rqf_params, group_specs = self._parse_filter_config(
            filter_scales,
            num_filters,
            num_oscillators,
            tau,
            min_freq,
            max_freq,
            min_gamma,
            max_gamma,
        )
        rqf_tau, rqf_min_freq, rqf_max_freq, rqf_min_gamma, rqf_max_gamma = rqf_params

        # Per-filter oscillator counts and scan width
        per_filter_osc: list[int] = []
        for count, n_osc in group_specs:
            per_filter_osc.extend([n_osc] * count)
        self.scan_filters = sum(per_filter_osc)
        self._group_specs = group_specs
        self._uniform_osc = len(set(per_filter_osc)) == 1

        # Register per-filter repeat counts for input expansion
        has_multi_osc = self.scan_filters > num_filters
        if has_multi_osc:
            self.register_buffer(
                "_input_repeats",
                torch.tensor(per_filter_osc, dtype=torch.long),
            )

        # Forward RQF (scan_filters oscillators total)
        self.rqf_fwd = RQF(
            num_filters=self.scan_filters,
            dt=dt,
            tau=rqf_tau,
            learn_tau=learn_tau,
            min_freq=rqf_min_freq,
            max_freq=rqf_max_freq,
            min_gamma=rqf_min_gamma,
            max_gamma=rqf_max_gamma,
            discretization=discretization,
            clamp_gamma=clamp_gamma,
        )

        # Backward RQF (if bidirectional)
        if bidirectional:
            self.rqf_bwd = RQF(
                num_filters=self.scan_filters,
                dt=dt,
                tau=rqf_tau,
                learn_tau=learn_tau,
                min_freq=rqf_min_freq,
                max_freq=rqf_max_freq,
                min_gamma=rqf_min_gamma,
                max_gamma=rqf_max_gamma,
                discretization=discretization,
                clamp_gamma=clamp_gamma,
            )

        # C-weights for combining oscillators within each filter
        self._has_c_weights = has_multi_osc or c_projection
        if self._has_c_weights:
            self.c_weights_fwd = self._make_c_weights(group_specs)
            if bidirectional:
                self.c_weights_bwd = self._make_c_weights(group_specs)

        # Divisive normalization readout (when readout_mode="divnorm" or "4channel_divnorm")
        self.divisive_norm: DivisiveNormalization | None = None
        if readout_mode in ("divnorm", "4channel_divnorm"):
            dn_channels = (
                self.num_filters * self.readout_multiplier
                if readout_mode == "4channel_divnorm"
                else self.total_filters
            )
            self.divisive_norm = DivisiveNormalization(
                num_channels=dn_channels,
                sigma_init=sigma_init,
            )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _make_c_weights(group_specs: list[tuple[int, int]]) -> nn.Parameter:
        """Create random unit-phase C-weights for oscillator combination."""
        parts = []
        for count, n_osc in group_specs:
            phase = torch.rand(count * n_osc) * 2 * math.pi
            c = torch.complex(torch.cos(phase), torch.sin(phase))
            if n_osc > 1:
                c = c / (n_osc**0.5)
            parts.append(c)
        return nn.Parameter(torch.view_as_real(torch.cat(parts)))

    @staticmethod
    def _parse_filter_config(
        filter_scales: list[dict] | None,
        num_filters: int,
        num_oscillators: int,
        tau: float,
        min_freq: float,
        max_freq: float,
        min_gamma: float,
        max_gamma: float,
    ) -> tuple[tuple, list[tuple[int, int]]]:
        """Parse filter_scales into per-filter RQF parameter arrays.

        Returns:
            Tuple of (rqf_params, group_specs) where rqf_params is
            (tau, min_freq, max_freq, min_gamma, max_gamma) and group_specs
            is a list of (count, num_oscillators) per group.
        """
        if filter_scales is not None:
            rqf_tau: float | list[float] = []
            rqf_min_freq: float | list[float] = []
            rqf_max_freq: float | list[float] = []
            rqf_min_gamma: float | list[float] = []
            rqf_max_gamma: float | list[float] = []
            group_specs: list[tuple[int, int]] = []
            total_count = 0
            for group in filter_scales:
                n = group["count"]
                n_osc = group.get("num_oscillators", num_oscillators)
                rqf_tau.extend([group.get("tau", tau)] * n)  # type: ignore[union-attr]
                rqf_min_freq.extend([group.get("min_freq", min_freq)] * n)  # type: ignore[union-attr]
                rqf_max_freq.extend([group.get("max_freq", max_freq)] * n)  # type: ignore[union-attr]
                rqf_min_gamma.extend([group.get("min_gamma", min_gamma)] * n)  # type: ignore[union-attr]
                rqf_max_gamma.extend([group.get("max_gamma", max_gamma)] * n)  # type: ignore[union-attr]
                group_specs.append((n, n_osc))
                total_count += n
            if total_count != num_filters:
                raise ValueError(
                    f"filter_scales counts sum to {total_count}, expected {num_filters}"
                )
        else:
            rqf_tau = tau
            rqf_min_freq = min_freq
            rqf_max_freq = max_freq
            rqf_min_gamma = min_gamma
            rqf_max_gamma = max_gamma
            group_specs = [(num_filters, num_oscillators)]

        # Expand per-filter bounds for multi-oscillator configs
        per_filter_osc: list[int] = []
        for count, n_osc in group_specs:
            per_filter_osc.extend([n_osc] * count)
        if sum(per_filter_osc) > num_filters and isinstance(rqf_tau, list):
            rqf_tau = [v for v, n in zip(rqf_tau, per_filter_osc) for _ in range(n)]
            rqf_min_freq = [v for v, n in zip(rqf_min_freq, per_filter_osc) for _ in range(n)]
            rqf_max_freq = [v for v, n in zip(rqf_max_freq, per_filter_osc) for _ in range(n)]
            rqf_min_gamma = [v for v, n in zip(rqf_min_gamma, per_filter_osc) for _ in range(n)]
            rqf_max_gamma = [v for v, n in zip(rqf_max_gamma, per_filter_osc) for _ in range(n)]

        return (rqf_tau, rqf_min_freq, rqf_max_freq, rqf_min_gamma, rqf_max_gamma), group_specs

    @staticmethod
    def _create_prenorm(norm_type: str, readout_mode: str, d_model: int) -> nn.Module | None:
        """Build the prenorm layer and warn if combined with divisive-norm readout."""
        _divnorm_readouts = ("divnorm", "4channel_divnorm", "2channel_divnorm")
        if norm_type != "none" and readout_mode in _divnorm_readouts:
            import warnings

            warnings.warn(
                f"norm_type='{norm_type}' with readout_mode='{readout_mode}': "
                "pre-normalization is discouraged when using divisive normalization "
                "readout, as the signal is already normalized by the readout. "
                "Consider norm_type='none'.",
                stacklevel=3,
            )
        if norm_type == "none":
            return None
        if norm_type == "batch":
            return nn.BatchNorm1d(d_model)
        return nn.LayerNorm(d_model)

    def _prenorm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply prenormalization (LayerNorm, BatchNorm1d, or identity)."""
        if self.norm is None:
            return x
        if self._use_batch_norm:
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)

    def _apply_readout(
        self, y_fwd: torch.Tensor, y_bwd: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract features from complex scan output.

        Handles all readout modes: complex (passthrough), divnorm, 4channel,
        2channel, real, magnitude.
        """
        if self.readout_mode == "complex":
            # Passthrough: return complex tensor as-is
            if y_bwd is not None:
                return torch.cat([y_fwd, y_bwd], dim=-1)
            return y_fwd
        if self.readout_mode == "divnorm":
            return self.divisive_norm(y_fwd, y_bwd)
        if self.readout_mode == "4channel_divnorm":
            # Per-direction DivNorm: normalize fwd/bwd independently, then concat.
            # Shared W matrix — same inhibition pattern for both directions.
            y_fwd_norm = self.divisive_norm(_halfwave_rectify_4ch(y_fwd))
            if y_bwd is not None:
                y_bwd_norm = self.divisive_norm(_halfwave_rectify_4ch(y_bwd))
                return torch.cat([y_fwd_norm, y_bwd_norm], dim=-1)
            return y_fwd_norm
        if self.readout_mode == "2channel_divnorm":
            y_fwd_norm = self.divisive_norm(_halfwave_rectify_2ch(y_fwd.real))
            if y_bwd is not None:
                y_bwd_norm = self.divisive_norm(_halfwave_rectify_2ch(y_bwd.real))
                return torch.cat([y_fwd_norm, y_bwd_norm], dim=-1)
            return y_fwd_norm
        if self.readout_mode == "4channel":
            y_fwd_out = _halfwave_rectify_4ch(y_fwd).square()
            if y_bwd is not None:
                return torch.cat([y_fwd_out, _halfwave_rectify_4ch(y_bwd).square()], dim=-1)
            return y_fwd_out
        if self.readout_mode == "2channel":
            y_fwd_out = _halfwave_rectify_2ch(y_fwd.real).square()
            if y_bwd is not None:
                return torch.cat([y_fwd_out, _halfwave_rectify_2ch(y_bwd.real).square()], dim=-1)
            return y_fwd_out
        if self.readout_mode == "real":
            if y_bwd is not None:
                return torch.cat([y_fwd.real, y_bwd.real], dim=-1)
            return y_fwd.real
        # "magnitude" (default)
        return self._magnitude_readout(y_fwd, y_bwd)

    @staticmethod
    def _magnitude_readout(y_fwd: torch.Tensor, y_bwd: torch.Tensor | None) -> torch.Tensor:
        """Return ``|y|`` (or bidirectional concat of magnitudes).

        Accepts already-magnitude real input from the fused-abs fast path
        dispatched in ``_scan_standard`` (multi-osc eval dispatch).
        """
        y_fwd_mag = y_fwd if not y_fwd.is_complex() else y_fwd.abs()
        if y_bwd is None:
            return y_fwd_mag
        y_bwd_mag = y_bwd if not y_bwd.is_complex() else y_bwd.abs()
        return torch.cat([y_fwd_mag, y_bwd_mag], dim=-1)

    def _c_weighted_sum(
        self,
        y: torch.Tensor,
        c_weights: nn.Parameter,
    ) -> torch.Tensor:
        """Combine oscillators within each filter using learned C-weights.

        Args:
            y: Complex scan output, shape (B, T, scan_filters).
            c_weights: Real-viewed C-weights parameter, shape (scan_filters, 2).

        Returns:
            Complex tensor, shape (B, T, num_filters).
        """
        b, t = y.shape[:2]
        c = torch.view_as_complex(c_weights)

        if self._uniform_osc:
            n_osc = self._group_specs[0][1]
            return (y.view(b, t, self.num_filters, n_osc) * c.view(self.num_filters, n_osc)).sum(
                dim=-1
            )
        else:
            offset = 0
            c_offset = 0
            parts = []
            for count, n_osc in self._group_specs:
                w = count * n_osc
                y_g = y[:, :, offset : offset + w]
                c_g = c[c_offset : c_offset + w]
                if n_osc > 1:
                    parts.append(
                        (y_g.view(b, t, count, n_osc) * c_g.view(count, n_osc)).sum(dim=-1)
                    )
                else:
                    parts.append(y_g * c_g)
                offset += w
                c_offset += w
            return torch.cat(parts, dim=-1)

    def _compute_coefficients(self, rqf: RQF) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute C1, C2 coefficients from RQF parameters."""
        C1, C2 = rqf.compute_coefficients()
        return C1.to(torch.cfloat), C2.to(torch.cfloat)

    @staticmethod
    def _apply_prospective_taps(
        u: torch.Tensor,
        tap0: torch.Tensor,
        tap1: torch.Tensor,
        *,
        delayed_u: torch.Tensor | None = None,
        detach_delayed: bool = False,
    ) -> torch.Tensor:
        """Apply two-tap filter for prospective input.

        Computes ``drive[n] = tap0 · u[n] + tap1 · u[n-1]`` with ``u[-1] = 0``.
        The tap coefficients have gamma pre-folded in.

        Args:
            u: Input tensor, shape ``(batch, time, filters)``.
            tap0: Lag-0 tap on *u* (gamma pre-folded), shape ``(filters,)``.
            tap1: Lag-1 tap on *u* (gamma pre-folded), shape ``(filters,)``.
            delayed_u: Optional source tensor for the lag-1 tap. This is
                numerically expected to match ``u`` but may have a different
                autograd graph, e.g. ``W(x.detach())`` for strict spatial BP.
            detach_delayed: Treat the lag-1 input values as constants. This is
                available for stricter spatial/local variants, but standard
                TBPTT(1) for a two-tap recurrence keeps this explicit input tap
                differentiable and detaches only the recurrent state.

        Returns:
            Drive signal, shape ``(batch, time, filters)``.
        """
        delayed_source = u if delayed_u is None else delayed_u
        if delayed_source.shape != u.shape:
            raise ValueError(
                f"delayed_u must have shape {tuple(u.shape)}, got {tuple(delayed_source.shape)}"
            )
        delayed = delayed_source[:, :-1, :]
        if detach_delayed:
            delayed = delayed.detach()
        u_delayed = torch.cat([torch.zeros_like(u[:, :1, :]), delayed], dim=1)
        return tap0 * u + tap1 * u_delayed

    def _scan_prospective(
        self,
        x: torch.Tensor,
        rqf_module: torch.nn.Module,
        c_weights: nn.Parameter | None,
        delayed_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the two-tap scan path.

        Used for prospective-input discretizations (Euler, ZOH impulse, FOH)
        and for non-prospective FOH — any update of the form
        ``y[n+1] = C1·y[n] + tap0·u[n] + tap1·u[n-1]`` with gamma pre-folded
        into the taps. Scans with ``C2 = 1`` on the precomputed two-tap drive.

        Args:
            x: Input tensor ``(batch, time, filters)``.
            rqf_module: RQF module to compute two-tap coefficients from.
            c_weights: Optional C-weights for oscillator combination.
            delayed_x: Optional projected source for the lag-1 tap. Used by
                strict spatial BP to keep the delayed tap differentiable with
                respect to current-layer weights while cutting temporal
                gradients into the previous timestep activation.

        Returns:
            Complex scan output ``(batch, time, num_filters)``.
        """
        C1, tap0, tap1 = rqf_module.compute_two_tap_coefficients(
            prospective=self.prospective_input,
            mode=self.prospective_mode,
            prospective_tau=self.prospective_tau,
        )
        C1 = C1.to(torch.cfloat)
        tap0 = tap0.to(torch.cfloat)
        tap1 = tap1.to(torch.cfloat)

        # Expand input for multi-oscillator
        if self.scan_filters > self.num_filters:
            x = x.repeat_interleave(self._input_repeats, dim=-1)
            if delayed_x is not None:
                delayed_x = delayed_x.repeat_interleave(self._input_repeats, dim=-1)

        if not x.is_complex():
            x = x.to(torch.cfloat)
        if delayed_x is not None and not delayed_x.is_complex():
            delayed_x = delayed_x.to(torch.cfloat)

        # Two-tap filter on u (gamma folded into taps), then scan with C2=ones.
        y = self._scan_two_tap_with_grad_mode(x, C1, tap0, tap1, delayed_u=delayed_x)

        if self._has_c_weights:
            y = self._c_weighted_sum(y, c_weights)
        return y

    def _scan_standard_multi_osc(
        self,
        x: torch.Tensor,
        C1: torch.Tensor,
        C2: torch.Tensor,
        c_weights: nn.Parameter | None,
        n_osc: int,
    ) -> torch.Tensor:
        """Fused multi-oscillator kernel dispatch (expand + scan + C-weight reduce)."""
        use_fused_abs = (
            self.readout_mode == "magnitude"
            and not torch.is_grad_enabled()
            and not self.use_parallel_scan
        )
        if use_fused_abs:
            if x.is_complex():
                cw_real = c_weights[:, 0].contiguous()
                cw_imag = c_weights[:, 1].contiguous()
                return _sequential_scan_multi_osc_forward_abs_raw(
                    C1, C2, x, cw_real, cw_imag, n_osc
                )
            return _sequential_scan_multi_osc_real_forward_abs_raw(C1, C2, x, c_weights, n_osc)
        if self.use_parallel_scan:
            u = x if x.is_complex() else x.to(torch.cfloat)
            return _triton_parallel_scan_multi_osc(C1, C2, u, c_weights, n_osc)
        if x.is_complex():
            return _triton_sequential_scan_multi_osc(C1, C2, x, c_weights, n_osc)
        return _triton_sequential_scan_multi_osc_real(C1, C2, x, c_weights, n_osc)

    def _scan_standard_fallback(
        self,
        x: torch.Tensor,
        C1: torch.Tensor,
        C2: torch.Tensor,
        c_weights: nn.Parameter | None,
    ) -> torch.Tensor:
        """Standard path: expand → scan → C-weight reduce."""
        if self.scan_filters > self.num_filters:
            x = x.repeat_interleave(self._input_repeats, dim=-1)
        use_abs_standard = (
            self.scan_filters == self.num_filters
            and self.readout_mode == "magnitude"
            and not torch.is_grad_enabled()
            and not self.use_parallel_scan
        )
        y = sequential_scan_abs(C1, C2, x) if use_abs_standard else self._scan(x, C1, C2)
        if self._has_c_weights:
            y = self._c_weighted_sum(y, c_weights)
        return y

    def _scan_standard(
        self,
        x: torch.Tensor,
        rqf_module: torch.nn.Module,
        c_weights: nn.Parameter | None,
    ) -> torch.Tensor:
        """Run standard (non-prospective) scan path.

        Computes C1/C2 coefficients and runs the scan, using fused Triton
        kernels when available on CUDA with uniform multi-oscillator config.

        Args:
            x: Input tensor ``(batch, time, filters)``.
            rqf_module: RQF module to compute coefficients from.
            c_weights: Optional C-weights for oscillator combination.

        Returns:
            Complex scan output ``(batch, time, num_filters)``.
        """
        C1, C2 = self._compute_coefficients(rqf_module)

        if self._use_tbptt1_scan_grad():
            if self.scan_filters > self.num_filters:
                x = x.repeat_interleave(self._input_repeats, dim=-1)
            y = self._scan_with_grad_mode(x, C1, C2)
            if self._has_c_weights:
                y = self._c_weighted_sum(y, c_weights)
            return y

        if (
            self.scan_filters > self.num_filters
            and self._uniform_osc
            and _HAS_FUSED_MULTI_OSC
            and x.is_cuda
        ):
            n_osc = self._group_specs[0][1]
            return self._scan_standard_multi_osc(x, C1, C2, c_weights, n_osc)
        return self._scan_standard_fallback(x, C1, C2, c_weights)

    def _scan_direction(
        self,
        x: torch.Tensor,
        rqf_module: torch.nn.Module,
        c_weights: nn.Parameter | None,
        delayed_x: torch.Tensor | None = None,
        flip: bool = False,
    ) -> torch.Tensor:
        """Run scan for one direction, optionally time-flipped.

        Computes coefficients, optionally expands input for multi-oscillator,
        runs the linear recurrence scan, and optionally reduces via C-weights.
        Dispatches to the two-tap scan path when ``prospective_input`` is set
        or when ``discretization == "foh"`` (FOH is intrinsically two-tap);
        otherwise uses the single-tap standard path.

        Args:
            x: Input tensor ``(batch, time, filters)``.
            rqf_module: RQF module to compute coefficients from.
            c_weights: Optional C-weights for oscillator combination.
            delayed_x: Optional projected source for the lag-1 tap.
            flip: Whether to time-flip input/output (for backward direction).

        Returns:
            Complex scan output ``(batch, time, num_filters)``.
        """
        if flip:
            x = x.flip(1)
            if delayed_x is not None:
                delayed_x = delayed_x.flip(1)

        if self._use_two_tap:
            y = self._scan_prospective(x, rqf_module, c_weights, delayed_x=delayed_x)
        else:
            y = self._scan_standard(x, rqf_module, c_weights)

        if flip:
            y = y.flip(1)
        return y

    def _scan(self, u: torch.Tensor, C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
        """Apply linear recurrence scan."""
        scan_fn = parallel_scan if self.use_parallel_scan else sequential_scan
        return scan_fn(C1, C2, u)

    def _use_tbptt1_scan_grad(self) -> bool:
        """Return whether scan gradients should use the TBPTT(1) surrogate."""
        return self.scan_grad_mode == "tbptt1" and torch.is_grad_enabled()

    @staticmethod
    def _shift_states(states: torch.Tensor) -> torch.Tensor:
        """Return ``h[t-1]`` aligned with timestep ``t`` and zero init at ``t=0``."""
        h_prev = torch.zeros_like(states)
        h_prev[:, 1:, :] = states[:, :-1, :]
        return h_prev

    def _scan_with_grad_mode(
        self, u: torch.Tensor, C1: torch.Tensor, C2: torch.Tensor
    ) -> torch.Tensor:
        """Apply the scan with either full BPTT or TBPTT(1) temporal gradients.

        In ``"tbptt1"`` mode the forward remains numerically identical to the
        original scan, while the backward uses the one-step rebuild
        ``h[t] = C1*h[t-1]_detached + C2*u[t]``.
        """
        if not self._use_tbptt1_scan_grad():
            return self._scan(u, C1, C2)

        with torch.no_grad():
            states = self._scan(u.detach(), C1.detach(), C2.detach())

        h_prev = self._shift_states(states).detach()
        rebuild = C1 * h_prev + C2 * u

        # Forward values come from the exact scan; backward follows the
        # truncated one-step rebuild.
        return states + (rebuild - rebuild.detach())

    def _scan_two_tap_with_grad_mode(
        self,
        u: torch.Tensor,
        C1: torch.Tensor,
        tap0: torch.Tensor,
        tap1: torch.Tensor,
        *,
        delayed_u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply a two-tap scan with full or spatial/TBPTT(1) gradients.

        The forward recurrence is always
        ``h[n] = C1*h[n-1] + tap0*u[n] + tap1*u[n-1]``. In TBPTT(1), the
        exact scan is evaluated without gradients, then a one-step graph is
        rebuilt with ``h[n-1]`` detached. ``delayed_u`` controls the surrogate
        graph for the lag-1 tap; for strict spatial BP it can be
        ``W(x.detach())``, preserving weight gradients while cutting temporal
        activation gradients.
        """
        C2_ones = torch.ones_like(C1)
        if not self._use_tbptt1_scan_grad():
            drive = self._apply_prospective_taps(u, tap0, tap1)
            return self._scan(drive, C1, C2_ones)

        with torch.no_grad():
            drive_forward = self._apply_prospective_taps(
                u.detach(),
                tap0.detach(),
                tap1.detach(),
            )
            states = self._scan(drive_forward, C1.detach(), C2_ones.detach())

        h_prev = self._shift_states(states).detach()
        drive_rebuild = self._apply_prospective_taps(
            u,
            tap0,
            tap1,
            delayed_u=delayed_u,
        )
        rebuild = C1 * h_prev + drive_rebuild
        return states + (rebuild - rebuild.detach())
