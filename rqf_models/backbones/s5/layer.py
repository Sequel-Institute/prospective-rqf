"""PyTorch S5 layer.

This is a direct PyTorch implementation of the original JAX S5 layer's core
diagonal SSM recurrence. It reuses ``rqf_models.ops.scan.parallel_scan``
for the associative scan.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from rqf_models.backbones.s5.init import lecun_std, make_s5_dplr_init
from rqf_models.ops.scan import parallel_scan, sequential_scan


def _complex_from_parts(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    return torch.complex(real, imag)


def _discretize_zoh(
    lambda_c: torch.Tensor,
    b_tilde: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled_lambda = lambda_c * delta.to(lambda_c.dtype)
    lambda_bar = torch.exp(scaled_lambda)
    lambda_safe = torch.where(
        lambda_c.abs() < torch.finfo(lambda_c.real.dtype).eps,
        torch.ones_like(lambda_c),
        lambda_c,
    )
    b_factor = torch.expm1(scaled_lambda) / lambda_safe
    b_factor = torch.where(
        lambda_c.abs() < torch.finfo(lambda_c.real.dtype).eps,
        delta.to(lambda_c.dtype),
        b_factor,
    )
    b_bar = b_factor.unsqueeze(-1) * b_tilde
    return lambda_bar, b_bar


def _discretize_bilinear(
    lambda_c: torch.Tensor,
    b_tilde: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    identity = torch.ones_like(lambda_c)
    bl = 1.0 / (identity - 0.5 * delta * lambda_c)
    lambda_bar = bl * (identity + 0.5 * delta * lambda_c)
    b_bar = (bl * delta).unsqueeze(-1) * b_tilde
    return lambda_bar, b_bar


class S5Layer(nn.Module):
    """Single S5 sequence layer operating on ``(B, T, H)`` real tensors.

    Prospective input uses the exact impulse-ZOH two-tap recurrence for the
    shared physical horizon ``tau = c * input_dt``. Native S5 represents each
    mode on a learned local clock ``delta[p]``; after converting from the
    shared input clock, the coefficient multiplying the boundary impulse is
    therefore ``c * delta[p]``.

    ``prospective_mode="direct"`` leaves the learned continuous-time input
    matrix unchanged. ``prospective_mode="decay_gain"`` interprets that
    learned matrix as ``W_in`` and uses ``diag(-Re(Lambda)) @ W_in``, matching
    the input-gain structure of an RQF.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 48,
        blocks: int = 8,
        dropout: float = 0.0,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        discretization: str = "zoh",
        activation: str = "half_glu1",
        conj_sym: bool = True,
        bidirectional: bool = False,
        clip_eigs: bool = False,
        step_rescale: float = 1.0,
        use_parallel_scan: bool = False,
        prospective_input: bool = False,
        prospective_mode: str = "direct",
        prospective_horizon_multiplier: float = 5.0,
        prospective_initial: str = "zero",
        prospective_feedthrough: bool = False,
        scan_grad_mode: str = "full",
    ) -> None:
        super().__init__()
        if discretization not in {"zoh", "bilinear"}:
            raise ValueError(f"Unsupported S5 discretization: {discretization!r}")
        if prospective_input and discretization != "zoh":
            raise ValueError(
                "prospective input for S5 uses the impulse-ZOH discretization and is "
                "only supported with discretization='zoh'."
            )
        if activation not in {"gelu", "full_glu", "half_glu1", "half_glu2"}:
            raise ValueError(f"Unsupported S5 activation: {activation!r}")
        if prospective_mode not in {"direct", "decay_gain"}:
            raise ValueError(
                "prospective_mode must be 'direct' or 'decay_gain', "
                f"got {prospective_mode!r}"
            )
        if prospective_mode == "decay_gain" and not prospective_input:
            raise ValueError(
                "prospective_mode='decay_gain' defines alphaP-S5 and requires "
                "prospective_input=True."
            )
        if prospective_mode == "decay_gain" and not clip_eigs:
            raise ValueError(
                "prospective_mode='decay_gain' requires clip_eigs=True so "
                "alpha=-Re(Lambda) remains strictly positive."
            )
        if not math.isfinite(prospective_horizon_multiplier):
            raise ValueError("prospective_horizon_multiplier must be finite")
        if prospective_horizon_multiplier <= 0.0:
            raise ValueError("prospective_horizon_multiplier must be positive")
        if prospective_initial not in {"zero", "repeat"}:
            raise ValueError("prospective_initial must be 'zero' or 'repeat'")
        if prospective_feedthrough:
            raise NotImplementedError(
                "S5 prospective_feedthrough=True is not implemented; the notes specify "
                "a non-prospective D skip path by default."
            )
        self._validate_scan_grad_mode(scan_grad_mode)
        if scan_grad_mode == "tbptt1" and not use_parallel_scan:
            raise ValueError(
                "S5 scan_grad_mode='tbptt1' is currently supported only with "
                "use_parallel_scan=True."
            )

        self.d_model = d_model
        self.d_state = d_state
        self.blocks = blocks
        self.discretization = discretization
        self.activation_name = activation
        self.conj_sym = conj_sym
        self.bidirectional = bidirectional
        self.clip_eigs = clip_eigs
        self.step_rescale = step_rescale
        self.use_parallel_scan = use_parallel_scan
        self.prospective_input = prospective_input
        self.prospective_mode = prospective_mode
        self.prospective_horizon_multiplier = prospective_horizon_multiplier
        self.prospective_initial = prospective_initial
        self.prospective_feedthrough = prospective_feedthrough
        self.scan_grad_mode = scan_grad_mode

        lambda_init, v, vinv, local_p = make_s5_dplr_init(
            d_state=d_state,
            blocks=blocks,
            conj_sym=conj_sym,
        )
        p = lambda_init.numel()
        self.p = p
        self.local_p = local_p

        self.Lambda_re = nn.Parameter(lambda_init.real.clone())
        self.Lambda_im = nn.Parameter(lambda_init.imag.clone())

        b_pre = torch.empty(local_p, d_model)
        nn.init.normal_(b_pre, std=lecun_std(local_p))
        b_tilde = vinv @ b_pre.to(torch.complex64)
        self.B_re = nn.Parameter(b_tilde.real.contiguous())
        self.B_im = nn.Parameter(b_tilde.imag.contiguous())

        c_pre = torch.empty(d_model, local_p, 2)
        nn.init.normal_(c_pre, std=lecun_std(local_p))
        c_complex = torch.complex(c_pre[..., 0], c_pre[..., 1])
        c_tilde = c_complex @ v
        c_out = torch.cat([c_tilde, c_tilde.clone()], dim=-1) if bidirectional else c_tilde
        self.C_re = nn.Parameter(c_out.real.contiguous())
        self.C_im = nn.Parameter(c_out.imag.contiguous())

        self.D = nn.Parameter(torch.randn(d_model))
        log_dt = torch.empty(p).uniform_(math.log(dt_min), math.log(dt_max))
        self.log_step = nn.Parameter(log_dt)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        if activation == "full_glu":
            self.out1 = nn.Linear(d_model, d_model)
            self.out2 = nn.Linear(d_model, d_model)
        elif activation in {"half_glu1", "half_glu2"}:
            self.out2 = nn.Linear(d_model, d_model)

    @staticmethod
    def _validate_scan_grad_mode(scan_grad_mode: str) -> None:
        valid_scan_grad_modes = {"full", "tbptt1"}
        if scan_grad_mode not in valid_scan_grad_modes:
            raise ValueError(
                f"scan_grad_mode must be one of {valid_scan_grad_modes}, got {scan_grad_mode!r}"
            )

    def _lambda(self) -> torch.Tensor:
        lambda_re = self.Lambda_re
        if self.clip_eigs:
            lambda_re = torch.clamp(lambda_re, max=-1e-4)
        return _complex_from_parts(lambda_re, self.Lambda_im)

    def _delta(self) -> torch.Tensor:
        return self.step_rescale * torch.exp(self.log_step)

    def _prospective_input_matrix(
        self,
        lambda_c: torch.Tensor,
        learned_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return Model 1's ``B_tilde`` or Model 2's ``G_alpha @ W_in``."""
        if self.prospective_mode == "direct":
            return learned_input
        alpha = (-lambda_c.real).to(learned_input.dtype)
        return alpha.unsqueeze(-1) * learned_input

    def _discretize(self) -> tuple[torch.Tensor, torch.Tensor]:
        lambda_c = self._lambda()
        b_tilde = _complex_from_parts(self.B_re, self.B_im)
        delta = self._delta()
        if self.discretization == "zoh":
            return _discretize_zoh(lambda_c, b_tilde, delta)
        return _discretize_bilinear(lambda_c, b_tilde, delta)

    def _discretize_prospective(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lambda_c = self._lambda()
        learned_input = _complex_from_parts(self.B_re, self.B_im)
        continuous_input = self._prospective_input_matrix(lambda_c, learned_input)
        delta = self._delta()
        lambda_bar, b_bar = _discretize_zoh(lambda_c, continuous_input, delta)

        # On the shared physical input clock tau = c * input_dt. Native S5
        # uses Lambda_eff = diag(delta / input_dt) @ Lambda_0, so the input_dt
        # factors cancel and the exact impulse coefficient is c * delta.
        native_horizon = self.prospective_horizon_multiplier * delta.unsqueeze(-1)
        correction = native_horizon * lambda_bar.unsqueeze(-1) * continuous_input
        b_plus = b_bar + correction
        b_minus = -correction
        return lambda_bar, b_plus, b_minus

    def _scan(
        self,
        lambda_bar: torch.Tensor,
        x: torch.Tensor,
        *,
        b_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ones = torch.ones_like(lambda_bar)
        if self.use_parallel_scan:
            return parallel_scan(lambda_bar, ones, x, B=b_matrix)
        if b_matrix is not None:
            x = torch.einsum("ph,bth->btp", b_matrix, x.to(b_matrix.dtype))
        return sequential_scan(lambda_bar, ones, x)

    def _use_tbptt1_scan_grad(self) -> bool:
        return self.scan_grad_mode == "tbptt1" and torch.is_grad_enabled()

    @staticmethod
    def _shift_states(states: torch.Tensor) -> torch.Tensor:
        h_prev = torch.zeros_like(states)
        h_prev[:, 1:, :] = states[:, :-1, :]
        return h_prev

    @staticmethod
    def _nonprospective_drive(
        b_bar: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("ph,bth->btp", b_bar, x.to(b_bar.dtype))

    def _scan_nonprospective(
        self,
        lambda_bar: torch.Tensor,
        b_bar: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Run the non-prospective S5 scan with full or TBPTT(1) gradients."""
        drive = self._nonprospective_drive(b_bar, x)
        if not self._use_tbptt1_scan_grad():
            return self._scan(lambda_bar, drive)

        with torch.no_grad():
            states = self._scan(
                lambda_bar.detach(),
                drive.detach(),
            )

        h_prev = self._shift_states(states).detach()
        rebuild = lambda_bar * h_prev + drive
        return states + (rebuild - rebuild.detach())

    def _previous_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            return x
        initial = x[:, :1] if self.prospective_initial == "repeat" else torch.zeros_like(x[:, :1])
        return torch.cat([initial, x[:, :-1]], dim=1)

    def _prospective_drive(
        self,
        b_plus: torch.Tensor,
        b_minus: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_complex = x.to(b_plus.dtype)
        x_prev = self._previous_input(x_complex)
        if self._use_tbptt1_scan_grad():
            # Spatial-only backpropagation: the delayed tap u[t-1] is a past
            # time step of the layer below, so its gradient path is cut (the
            # strict delayed-input convention used by the RQF backbones). The
            # current tap u[t] stays live to preserve the within-step spatial
            # chain, and b_plus/b_minus stay live so Lambda, B and log_step
            # receive their exact one-step gradients.
            x_prev = x_prev.detach()
        return torch.einsum("ph,bth->btp", b_plus, x_complex) + torch.einsum(
            "ph,bth->btp", b_minus, x_prev
        )

    def _scan_prospective(
        self,
        lambda_bar: torch.Tensor,
        b_plus: torch.Tensor,
        b_minus: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Run the prospective (two-tap) S5 scan with full or TBPTT(1) gradients."""
        drive = self._prospective_drive(b_plus, b_minus, x)
        if not self._use_tbptt1_scan_grad():
            return self._scan(lambda_bar, drive)

        with torch.no_grad():
            states = self._scan(
                lambda_bar.detach(),
                drive.detach(),
            )

        h_prev = self._shift_states(states).detach()
        rebuild = lambda_bar * h_prev + drive
        return states + (rebuild - rebuild.detach())

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        if self.prospective_input:
            lambda_bar, b_plus, b_minus = self._discretize_prospective()
            states = self._scan_prospective(lambda_bar, b_plus, b_minus, x)
        else:
            lambda_bar, b_bar = self._discretize()
            states = self._scan_nonprospective(lambda_bar, b_bar, x)
        if self.bidirectional:
            if self.prospective_input:
                states_bwd = self._scan_prospective(lambda_bar, b_plus, b_minus, x.flip(1)).flip(1)
            else:
                states_bwd = self._scan_nonprospective(lambda_bar, b_bar, x.flip(1)).flip(1)
            states = torch.cat([states, states_bwd], dim=-1)

        c_tilde = _complex_from_parts(self.C_re, self.C_im)
        y_complex = torch.einsum("hp,btp->bth", c_tilde, states)
        y = 2.0 * y_complex.real if self.conj_sym else y_complex.real
        return y + x * self.D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._ssm(x)
        if self.activation_name == "gelu":
            return self.dropout(torch.nn.functional.gelu(y))
        if self.activation_name == "full_glu":
            y = self.dropout(torch.nn.functional.gelu(y))
            return self.dropout(self.out1(y) * torch.sigmoid(self.out2(y)))
        if self.activation_name == "half_glu1":
            y = self.dropout(torch.nn.functional.gelu(y))
            return self.dropout(y * torch.sigmoid(self.out2(y)))
        y_gate = self.dropout(torch.nn.functional.gelu(y))
        return self.dropout(y * torch.sigmoid(self.out2(y_gate)))


class S5Block(nn.Module):
    """Prenorm/postnorm residual wrapper around :class:`S5Layer`."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 48,
        blocks: int = 8,
        dropout: float = 0.0,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        discretization: str = "zoh",
        activation: str = "half_glu1",
        conj_sym: bool = True,
        bidirectional: bool = False,
        clip_eigs: bool = False,
        step_rescale: float = 1.0,
        use_parallel_scan: bool = False,
        prenorm: bool = True,
        skip_connection: bool = True,
        norm_type: str = "layer",
        prospective_input: bool = False,
        prospective_mode: str = "direct",
        prospective_horizon_multiplier: float = 5.0,
        prospective_initial: str = "zero",
        prospective_feedthrough: bool = False,
        scan_grad_mode: str = "full",
    ) -> None:
        super().__init__()
        self.prenorm = prenorm
        self.skip_connection = skip_connection
        self.norm_type = norm_type
        if norm_type in {"batch", "batchnorm"}:
            self.norm: nn.Module = nn.BatchNorm1d(d_model, momentum=0.05)
        elif norm_type in {"layer", "layernorm", "none"}:
            self.norm = nn.Identity() if norm_type == "none" else nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unsupported S5 norm_type: {norm_type!r}")

        self.layer = S5Layer(
            d_model=d_model,
            d_state=d_state,
            blocks=blocks,
            dropout=dropout,
            dt_min=dt_min,
            dt_max=dt_max,
            discretization=discretization,
            activation=activation,
            conj_sym=conj_sym,
            bidirectional=bidirectional,
            clip_eigs=clip_eigs,
            step_rescale=step_rescale,
            use_parallel_scan=use_parallel_scan,
            prospective_input=prospective_input,
            prospective_mode=prospective_mode,
            prospective_horizon_multiplier=prospective_horizon_multiplier,
            prospective_initial=prospective_initial,
            prospective_feedthrough=prospective_feedthrough,
            scan_grad_mode=scan_grad_mode,
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.norm, nn.BatchNorm1d):
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prenorm:
            y = self.layer(self._norm(x))
            return x + y if self.skip_connection else y
        y = self.layer(x)
        return self._norm(x + y if self.skip_connection else y)
