"""S4D layer and prenorm-residual block.

The layer implements the standard S4D forward (FFT convolution against
the diagonal kernel plus a skip-D term, followed by GELU/GLU mixing).
The block wraps it in a LayerNorm-prenorm residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from organics.embedding.s4d.kernel import S4DKernel


class S4DLayer(nn.Module):
    """Single S4D layer.

    Input and output are real tensors of shape ``(B, L, H)``.

    Args:
        d_model: Channel dimension ``H``.
        d_state: State dimension ``N`` (must be even).
        dropout: Dropout rate applied after activation.
        dt_min: ``dt`` log-uniform lower bound.
        dt_max: ``dt`` log-uniform upper bound.
        kernel_init: ``"diag-inv"`` or ``"diag-lin"``.
        activation: Nonlinearity after FFT-conv (only ``"gelu"`` supported).
        glu: Use a GLU output mix (2H → H via pointwise Conv1d + GLU) when
            ``True``; otherwise a plain pointwise Conv1d.
        prospective_input: Use the prospective-input S4D kernel.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        kernel_init: str = "diag-inv",
        activation: str = "gelu",
        glu: bool = True,
        prospective_input: bool = False,
        prospective_tau: float | torch.Tensor | None = 0.0,
        prospective_tau_multiplier: float | torch.Tensor | None = None,
        prospective_initial: str = "zero",
        prospective_feedthrough: bool = False,
    ) -> None:
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.prospective_input = prospective_input

        self.s4d_kernel = S4DKernel(
            d_model=d_model,
            N=d_state,
            dt_min=dt_min,
            dt_max=dt_max,
            kernel_init=kernel_init,
            prospective_input=prospective_input,
            prospective_tau=prospective_tau,
            prospective_tau_multiplier=prospective_tau_multiplier,
            prospective_initial=prospective_initial,
            prospective_feedthrough=prospective_feedthrough,
        )
        self.D = nn.Parameter(torch.randn(d_model))

        if activation == "gelu":
            self.activation: nn.Module = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation!r}")

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if glu:
            self.output_linear: nn.Module = nn.Sequential(
                nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
                nn.GLU(dim=-2),
            )
        else:
            self.output_linear = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward S4D layer. Input ``(B, L, H)`` → output ``(B, L, H)``."""
        # (B, L, H) → (B, H, L)
        u = u.transpose(1, 2)
        L = u.size(-1)

        k, k_initial = self.s4d_kernel(L, return_initial_kernel=True)  # (H, L)
        k_f = torch.fft.rfft(k, n=2 * L)  # (H, L+1)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B, H, L+1)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B, H, L)

        if (
            self.prospective_input
            and self.s4d_kernel.prospective_initial == "repeat"
            and k_initial is not None
        ):
            y = y + u[..., :1] * k_initial.unsqueeze(0)

        y = y + u * self.D.unsqueeze(-1)
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)

        return y.transpose(1, 2)  # (B, L, H)


class S4DBlock(nn.Module):
    """LayerNorm-prenorm residual wrapper around an :class:`S4DLayer`."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        kernel_init: str = "diag-inv",
        activation: str = "gelu",
        glu: bool = True,
        prenorm: bool = True,
        skip_connection: bool = True,
        prospective_input: bool = False,
        prospective_tau: float | torch.Tensor | None = 0.0,
        prospective_tau_multiplier: float | torch.Tensor | None = None,
        prospective_initial: str = "zero",
        prospective_feedthrough: bool = False,
    ) -> None:
        super().__init__()
        self.prospective_input = prospective_input
        self.prenorm = prenorm
        self.skip_connection = skip_connection
        self.norm = nn.LayerNorm(d_model)
        self.layer = S4DLayer(
            d_model=d_model,
            d_state=d_state,
            dropout=dropout,
            dt_min=dt_min,
            dt_max=dt_max,
            kernel_init=kernel_init,
            activation=activation,
            glu=glu,
            prospective_input=prospective_input,
            prospective_tau=prospective_tau,
            prospective_tau_multiplier=prospective_tau_multiplier,
            prospective_initial=prospective_initial,
            prospective_feedthrough=prospective_feedthrough,
        )
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prenorm residual: ``x + Drop(Layer(Norm(x)))`` (or postnorm).

        When ``skip_connection=False`` the residual addition is dropped
        (useful for res/nores ablations); ``prenorm`` still controls the
        norm placement.
        """
        if self.prenorm:
            y = self.drop(self.layer(self.norm(x)))
            return x + y if self.skip_connection else y
        y = self.drop(self.layer(x))
        return self.norm(x + y if self.skip_connection else y)
