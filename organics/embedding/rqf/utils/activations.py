"""Complex-compatible activation functions for RQF blocks."""

import torch
import torch.nn as nn


class SplitTanh(nn.Module):
    """Split tanh activation for complex tensors: tanh(Re) + i*tanh(Im)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(torch.tanh(x.real), torch.tanh(x.imag))


class SplitReLU(nn.Module):
    """Split ReLU activation for complex tensors: relu(Re) + i*relu(Im)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(torch.relu(x.real), torch.relu(x.imag))
