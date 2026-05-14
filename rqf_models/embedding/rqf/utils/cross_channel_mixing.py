"""Cross-channel mixing module for frequency band feature detection.

This module provides learnable weighted sums across frequency channels to detect
patterns like formants and harmonics in multi-scale temporal representations.
"""

import torch
import torch.nn as nn


class CrossChannelMixing(nn.Module):
    """Learnable weighted sums across frequency channels.

    Detects patterns across multiple frequency bands (formants, harmonics)
    via pointwise + depthwise convolutions, analogous to 1x1 conv in CNNs.

    Args:
        in_channels: Number of input frequency bands.
        out_channels: Number of output feature combinations.
        kernel_size: Depthwise conv kernel for local frequency grouping.
        groups: Number of groups for the pointwise convolution.

    Example:
        >>> mixer = CrossChannelMixing(in_channels=128, out_channels=256)
        >>> x = torch.randn(4, 1000, 128)
        >>> mixer(x).shape
        torch.Size([4, 1000, 256])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 1,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd to preserve time dimension, got {kernel_size}"
            )

        self.mixer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=groups,
        )
        self.local_grouping = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=out_channels,
        )

        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, time, in_channels)`` → ``(batch, time, out_channels)``."""
        x = x.transpose(1, 2)
        x = self.mixer(x)
        x = self.local_grouping(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return self.act(x)
