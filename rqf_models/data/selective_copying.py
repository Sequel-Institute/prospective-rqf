"""Selective copying with the local-learning repository's exact sample stream.

Blank is 0, data symbols are 1..n_tokens-2, and n_tokens-1 marks recall.
Sequence length includes the recall suffix. Only that suffix is supervised.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SelectiveCopyingDataSpec:
    seq_len: int = 4096
    n_memorize: int = 16
    n_tokens: int = 16
    n_train: int = 64_000
    n_val: int = 1024
    n_test: int = 1024
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("seq_len", "n_memorize", "n_tokens", "n_train", "n_val", "n_test"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.seq_len < 2 * self.n_memorize:
            raise ValueError("seq_len must be >= 2 * n_memorize")
        if self.n_tokens < 3:
            raise ValueError("n_tokens must be >= 3 (blank, data, recall)")

    def build(self) -> tuple[SelectiveCopyingDataset, ...]:
        return tuple(
            SelectiveCopyingDataset(self, split, size)
            for split, size in (("train", self.n_train), ("val", self.n_val), ("test", self.n_test))
        )


class SelectiveCopyingDataset(Dataset[tuple[Tensor, Tensor]]):
    """Lazy samples keyed by seed/split/epoch/index; evaluation stays fixed."""

    def __init__(self, spec: SelectiveCopyingDataSpec, split: str, size: int) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unknown split: {split!r}")
        self.spec, self.split, self.size = spec, split, size
        self.epoch = 0

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if self.split == "train":
            self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if not 0 <= index < self.size:
            raise IndexError(index)
        spec = self.spec
        key = f"{spec.seed}:{self.split}:{self.epoch}:{index}".encode()
        seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        generator = torch.Generator().manual_seed(seed)
        context_len = spec.seq_len - spec.n_memorize
        positions = torch.randperm(context_len, generator=generator)[:spec.n_memorize].sort().values
        tokens = torch.randint(1, spec.n_tokens - 1, (spec.n_memorize,), generator=generator)

        inputs = torch.zeros(spec.seq_len, spec.n_tokens)
        inputs[:context_len, 0] = 1
        inputs[positions, 0] = 0
        inputs[positions, tokens] = 1
        inputs[context_len:, spec.n_tokens - 1] = 1
        target = torch.full((spec.seq_len,), -100, dtype=torch.long)
        target[context_len:] = tokens
        return inputs, target
