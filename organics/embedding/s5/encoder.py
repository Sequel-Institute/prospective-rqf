"""Stacked PyTorch S5 encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from organics.embedding.s5.layer import S5Block


class S5Encoder(nn.Module):
    """Input Linear → N × S5Block → LayerNorm → Output Linear."""

    def __init__(
        self,
        input_dims: int = 1,
        output_dims: int = 128,
        d_model: int = 128,
        d_state: int = 128,
        blocks: int = 8,
        num_layers: int = 4,
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
        prospective_tau: float | list[float] | torch.Tensor | None = None,
        prospective_tau_multiplier: float | list[float] | torch.Tensor | None = None,
        prospective_initial: str = "zero",
        prospective_feedthrough: bool = False,
        prospective_input_gain: bool = False,
        min_input_gain: float = 1e-6,
        scan_grad_mode: str = "full",
    ) -> None:
        super().__init__()
        self.output_dims = output_dims
        self.input_proj = nn.Linear(input_dims, d_model)
        self.blocks = nn.ModuleList(
            [
                S5Block(
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
                    prenorm=prenorm,
                    skip_connection=skip_connection,
                    norm_type=norm_type,
                    prospective_input=prospective_input,
                    prospective_tau=prospective_tau,
                    prospective_tau_multiplier=prospective_tau_multiplier,
                    prospective_initial=prospective_initial,
                    prospective_feedthrough=prospective_feedthrough,
                    prospective_input_gain=prospective_input_gain,
                    min_input_gain=min_input_gain,
                    scan_grad_mode=scan_grad_mode,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.Identity() if norm_type == "none" else nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return self.output_proj(h)
