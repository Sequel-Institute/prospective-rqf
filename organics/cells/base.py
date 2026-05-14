from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from organics.cells.state import State


class RNNCell(nn.Module, ABC):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._initialize_parameters(device=device, dtype=dtype)

    @abstractmethod
    def _initialize_parameters(self, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def _init_state_buffers(self, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_state_training(self, batch_size: int, **kwargs: Any) -> State:
        raise NotImplementedError

    @abstractmethod
    def step(self, input: torch.Tensor, state: State) -> State:
        raise NotImplementedError

    @abstractmethod
    def output(self, state: State) -> torch.Tensor:
        raise NotImplementedError

    def compute_cache(self) -> None:
        return None

    def forward(self, input: torch.Tensor, state: State) -> State:
        return self.step(input, state)
