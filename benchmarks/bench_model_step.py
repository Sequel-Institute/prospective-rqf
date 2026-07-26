"""End-to-end model-step benchmark (old vs new backend, p vs np).

Times full forward+backward (PureRQF encoder 6x64 + mean-pool linear head + CE)
at speech-commands raw scale (B=16, T=16000). The scan backend is chosen via
ORGANICS_SCAN_BACKEND (sequential|chunked) before import.
"""

from __future__ import annotations

import os
import time

import torch

BACKEND = os.environ.get("ORGANICS_SCAN_BACKEND", "chunked")


def build(prospective: bool):
    from rqf_models.backbones.rqf.pure import PureRQFEncoder

    return PureRQFEncoder(
        input_dims=1,
        output_dims=64,
        hidden_dims=64,
        num_layers=6,
        samp_frequency=16000.0,
        min_freq=60.0,
        max_freq=7800.0,
        tau=5.0 / 16000.0,
        activation="split_relu",
        prospective_input=prospective,
    ).cuda()


def time_step(model, x, iters=15, warmup=5):
    head = torch.nn.Linear(64, 10).cuda()
    labels = torch.zeros(x.shape[0], dtype=torch.long, device="cuda")
    times = []
    for i in range(warmup + iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = model(x)
        if y.is_complex():
            y = y.abs()
        logits = head(y.mean(dim=1).float())
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        model.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def main():
    torch.manual_seed(0)
    x = torch.randn(16, 16000, 1, device="cuda")
    print(f"PureRQF 6x64, B=16 T=16000, backend={BACKEND}")
    for prospective in (False, True):
        model = build(prospective)
        ms = time_step(model, x)
        print(f"  prospective={str(prospective):5s} full-step: {ms:8.2f} ms")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
