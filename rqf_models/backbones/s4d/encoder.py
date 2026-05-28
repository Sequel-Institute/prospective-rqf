"""Stacked S4D encoder.

Wraps a stack of :class:`S4DBlock` with input/output linear projections
and a final LayerNorm. Output shape is real-valued ``(B, T, output_dims)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rqf_models.backbones.s4d.block import S4DBlock


class S4DEncoder(nn.Module):
    """Input Linear -> N x S4DBlock -> LayerNorm -> Output Linear."""

    def __init__(
        self,
        input_dims: int = 1,
        output_dims: int = 128,
        d_model: int = 128,
        d_state: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        kernel_init: str = "diag-inv",
        activation: str = "gelu",
        glu: bool = True,
        prenorm: bool = True,
        bidirectional: bool = False,
        skip_connection: bool = True,
        prospective_input: bool = False,
        prospective_tau: float | torch.Tensor | None = 0.0,
        prospective_tau_multiplier: float | torch.Tensor | None = None,
        prospective_initial: str = "zero",
        prospective_feedthrough: bool = False,
    ) -> None:
        super().__init__()
        if bidirectional:
            raise NotImplementedError("bidirectional S4D not implemented in vanilla wrapper")

        self.output_dims = output_dims
        self.input_proj = nn.Linear(input_dims, d_model)
        self.blocks = nn.ModuleList(
            [
                S4DBlock(
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    kernel_init=kernel_init,
                    activation=activation,
                    glu=glu,
                    prenorm=prenorm,
                    skip_connection=skip_connection,
                    prospective_input=prospective_input,
                    prospective_tau=prospective_tau,
                    prospective_tau_multiplier=prospective_tau_multiplier,
                    prospective_initial=prospective_initial,
                    prospective_feedthrough=prospective_feedthrough,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. ``x: (B, T, input_dims)`` -> ``(B, T, output_dims)``."""
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return self.output_proj(h)
