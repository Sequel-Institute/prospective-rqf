from __future__ import annotations

import torch
import torch.nn as nn


class WMatrix(nn.Module):
    """Positive matrix helper used by divisive normalization."""

    def __init__(
        self,
        dim: int,
        norm_matrix_constraint: str = "ones",
        fixed_matrix: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.norm_matrix_constraint = norm_matrix_constraint
        if norm_matrix_constraint == "ones":
            self.register_buffer("matrix", torch.ones(dim, dim))
        elif norm_matrix_constraint == "fixed_custom":
            if fixed_matrix is None:
                raise ValueError("fixed_custom requires fixed_matrix")
            if tuple(fixed_matrix.shape) != (dim, dim):
                raise ValueError("fixed_matrix must have shape (dim, dim)")
            self.register_buffer("matrix", fixed_matrix.clamp_min(0.0).float())
        elif norm_matrix_constraint == "one_rank":
            self.left_log = nn.Parameter(torch.zeros(dim))
            self.right_log = nn.Parameter(torch.zeros(dim))
        elif norm_matrix_constraint == "full_rank":
            self.log_matrix = nn.Parameter(torch.zeros(dim, dim))
        else:
            raise ValueError(
                "norm_matrix_constraint must be one of "
                "{'ones', 'fixed_custom', 'one_rank', 'full_rank'}"
            )

    def forward(self, X: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_matrix_constraint in {"ones", "fixed_custom"}:
            return self.matrix
        if self.norm_matrix_constraint == "one_rank":
            return torch.outer(self.left_log.exp(), self.right_log.exp())
        return self.log_matrix.exp()
