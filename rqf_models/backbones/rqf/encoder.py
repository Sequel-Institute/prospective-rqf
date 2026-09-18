"""Base RQF encoder with shared scaffolding for S4 and Mamba variants.

Provides input/output projections, filter_scales parsing, final normalization,
and the shared forward loop. Subclasses set ``_block_class`` to their specific
block type.
"""

from abc import ABC

import torch
import torch.nn as nn


class BaseRQFEncoder(nn.Module, ABC):
    """Abstract base for stacked RQF encoder architectures.

    Handles the shared encoder scaffolding:
    - Input projection (Linear)
    - Per-layer ``filter_scales`` parsing
    - Block stacking via ``_block_class`` and ``_block_kwargs``
    - Final normalization (LayerNorm or BatchNorm1d)
    - Output projection (Linear)
    - Shared ``forward()``

    Subclasses must set ``_block_class`` to their block type and populate
    ``self._block_kwargs`` before calling ``super().__init__()``.

    Args:
        input_dims: Input feature dimension.
        output_dims: Output feature dimension.
        hidden_dims: Hidden dimension used by all blocks. Can be an int
            (same for every layer) or a list of ints (one per layer).
            When dimensions differ between adjacent layers, a linear
            projection is inserted automatically.
        num_layers: Number of blocks to stack.
        norm_type: Normalization type (``"none"``, ``"layer"``, or ``"batch"``).
        filter_scales: Optional per-filter configuration groups. Can be a flat
            list of dicts (same for all layers) or a list of lists (one per
            layer). Resolved to ``layer_filter_scales`` before being passed to
            ``_create_blocks``.
        input_projection: Project inputs before the first block. If false,
            the first block maps raw inputs to the first hidden width.
        output_projection: Create a final linear projection.
    """

    _block_class: type[nn.Module]

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: int | list[int],
        num_layers: int,
        norm_type: str = "none",
        filter_scales: list[dict] | list[list[dict]] | None = None,
        input_projection: bool = True,
        output_projection: bool = True,
    ) -> None:
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.input_projection = input_projection
        self._use_batch_norm = norm_type == "batch"

        # Normalize hidden_dims to per-layer list
        if isinstance(hidden_dims, int):
            self._layer_dims: list[int] = [hidden_dims] * num_layers
        else:
            if len(hidden_dims) != num_layers:
                raise ValueError(
                    f"hidden_dims list must have {num_layers} entries, got {len(hidden_dims)}"
                )
            self._layer_dims = list(hidden_dims)

        # Resolve per-layer filter_scales
        self._layer_filter_scales = self._resolve_filter_scales(filter_scales, num_layers)

        # Input projection
        self.input_proj = (
            nn.Linear(input_dims, self._layer_dims[0]) if input_projection else nn.Identity()
        )

        # Blocks — subclass creates the specific block type
        self.blocks = self._create_blocks(self._layer_filter_scales)

        # Final norm (matches block norm type)
        if norm_type == "none":
            self.final_norm: nn.Module | None = None
        elif norm_type == "batch":
            self.final_norm = nn.BatchNorm1d(self._layer_dims[-1])
        else:
            self.final_norm = nn.LayerNorm(self._layer_dims[-1])

        # Output projection
        self.output_proj = (
            nn.Linear(self._layer_dims[-1], output_dims) if output_projection else nn.Identity()
        )

    @staticmethod
    def _resolve_filter_scales(
        filter_scales: list[dict] | list[list[dict]] | None,
        num_layers: int,
    ) -> list[list[dict] | None]:
        """Parse ``filter_scales`` into a per-layer list.

        Args:
            filter_scales: Flat list (uniform) or nested list (per-layer).
            num_layers: Expected number of layers.

        Returns:
            List of length ``num_layers`` with per-layer filter config or None.
        """
        if (
            filter_scales is not None
            and len(filter_scales) > 0
            and isinstance(filter_scales[0], list)
        ):
            if len(filter_scales) != num_layers:
                raise ValueError(
                    f"per-layer filter_scales must have {num_layers} entries, "
                    f"got {len(filter_scales)}"
                )
            return filter_scales  # type: ignore[return-value]
        # Uniform: same filter_scales for all layers (or None)
        return [filter_scales for _ in range(num_layers)]  # type: ignore[list-item]

    def _create_blocks(self, layer_filter_scales: list[list[dict] | None]) -> nn.ModuleList:
        """Create blocks using ``_block_class`` and ``_block_kwargs``.

        When adjacent layers have different dimensions, passes ``d_model_out``
        so the block handles the dimension change internally (residual
        projection D + adjusted output projection).

        Args:
            layer_filter_scales: Resolved per-layer filter config.

        Returns:
            ``nn.ModuleList`` of blocks.
        """
        blocks = []
        for i in range(self.num_layers):
            d_in = self._layer_dims[i]
            d_out = self._layer_dims[i + 1] if i < self.num_layers - 1 else None
            if not self.input_projection:
                d_in = self.input_dims if i == 0 else self._layer_dims[i - 1]
                d_out = self._layer_dims[i]
            blocks.append(
                self._block_class(
                    d_model=d_in,
                    d_model_out=d_out,
                    filter_scales=layer_filter_scales[i],
                    **self._block_kwargs,
                )
            )
        return nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process through stacked RQF blocks.

        Args:
            x: Input tensor of shape ``(batch, time, features)``.

        Returns:
            Output tensor of shape ``(batch, time, output_dims)``.
        """
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        if self.final_norm is not None:
            if self._use_batch_norm:
                x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = self.final_norm(x)

        return self.output_proj(x)
