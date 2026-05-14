from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class State(Protocol):
    pass


@dataclass
class BaseState:
    y: torch.Tensor


@dataclass
class ORGaNICsState:
    y: torch.Tensor
    a: torch.Tensor
    b0: torch.Tensor
    b1: torch.Tensor
    z_prev: torch.Tensor | None = None
