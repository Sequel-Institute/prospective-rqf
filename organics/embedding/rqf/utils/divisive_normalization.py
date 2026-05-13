"""Divisive Normalization for cross-channel competition.

Implements lateral inhibition across frequency channels::

    y = |z|^2 / (sigma^2 + W @ |z|^2)

Replaces both magnitude extraction and CrossChannelMixing with a single
divisive normalization operation.
"""

import torch
import torch.nn as nn

from organics.constraints.parameter import WMatrix


class DivisiveNormalization(nn.Module):
    """Divisive normalization: ``y = |z|^2 / (sigma^2 + W @ |z|^2)``.

    W is a positive matrix (lateral inhibition) whose structure is controlled
    by ``norm_matrix_constraint``. Sigma is a semi-saturation constant
    parametrized in log-space.

    Args:
        num_channels: Number of input/output channels (total_filters for bidir).
        sigma_init: Initial semi-saturation constant.
        learn_sigma: Whether sigma is learnable.
        norm_matrix_constraint: Constraint type for the W matrix.
            ``"ones"`` (default) — fixed matrix of all ones, no learnable params.
            ``"one_rank"`` — W = p ⊗ q, learnable rank-1 with 2N params.
            ``"full_rank"`` — W = exp(log_W), fully learnable N² params.
            ``"fixed_custom"`` — fixed custom matrix.
        eps: Numerical stability constant.

    Example:
        >>> div_norm = DivisiveNormalization(num_channels=64)
        >>> z_fwd = torch.randn(4, 100, 32, dtype=torch.cfloat)
        >>> z_bwd = torch.randn(4, 100, 32, dtype=torch.cfloat)
        >>> div_norm(z_fwd, z_bwd).shape
        torch.Size([4, 100, 64])
    """

    def __init__(
        self,
        num_channels: int,
        sigma_init: float = 0.1,
        learn_sigma: bool = False,
        norm_matrix_constraint: str = "ones",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if sigma_init <= 0:
            raise ValueError(f"sigma_init must be positive, got {sigma_init}")

        self.num_channels = num_channels
        self.norm_matrix_constraint = norm_matrix_constraint
        self.eps = eps

        log_sigma = torch.log(torch.tensor(sigma_init))
        if learn_sigma:
            self.log_sigma = nn.Parameter(log_sigma)
        else:
            self.register_buffer("log_sigma", log_sigma)

        self.w_matrix = WMatrix(num_channels, norm_matrix_constraint=norm_matrix_constraint)

    @property
    def W(self) -> torch.Tensor:
        """Return the positive W matrix."""
        return self.w_matrix()

    @property
    def sigma_sq(self) -> torch.Tensor:
        """Return sigma^2 (always positive)."""
        return torch.exp(2.0 * self.log_sigma)

    def forward(self, z_fwd: torch.Tensor, z_bwd: torch.Tensor | None = None) -> torch.Tensor:
        """Apply divisive normalization to scan outputs.

        Args:
            z_fwd: Forward scan output ``(batch, time, channels)``.
                Complex or real — squaring is always applied internally.
            z_bwd: Optional backward scan output.

        Returns:
            Normalized output ``(batch, time, total_channels)``.
        """
        mag_sq_fwd = z_fwd.real**2 + z_fwd.imag**2 if z_fwd.is_complex() else z_fwd**2

        if z_bwd is not None:
            mag_sq_bwd = z_bwd.real**2 + z_bwd.imag**2 if z_bwd.is_complex() else z_bwd**2
            mag_sq = torch.cat([mag_sq_fwd, mag_sq_bwd], dim=-1)
        else:
            mag_sq = mag_sq_fwd

        inhibition = torch.matmul(mag_sq, self.W.T)
        return mag_sq / (self.sigma_sq + inhibition + self.eps)

    def extra_repr(self) -> str:
        """Extra representation string for print."""
        return (
            f"num_channels={self.num_channels}, "
            f"sigma_init={torch.exp(self.log_sigma).item():.4f}, "
            f"learn_sigma={isinstance(self.log_sigma, nn.Parameter)}, "
            f"constraint={self.norm_matrix_constraint}"
        )
