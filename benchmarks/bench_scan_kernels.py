"""Benchmark: legacy sequential scan vs chunked scan, and prospective overhead.

Times forward+backward of the temporal scan at speech-commands scale
(T=16000). Run on a CUDA machine:

    python benchmarks/bench_scan_kernels.py

Reports:
  1. legacy one-block-per-(batch,filter) sequential kernel vs the chunked
     two-level scan (the current default backend), and
  2. non-prospective scan vs the fused two-tap prospective scan with the new
     kernels (the prospective input's compute overhead).
"""

from __future__ import annotations

import time

import torch


def _coeffs(filters: int, device: str):
    g = torch.Generator(device="cpu").manual_seed(0)
    mag = 0.8 + 0.19 * torch.rand(filters, generator=g)
    phase = 2 * torch.pi * torch.rand(filters, generator=g)
    C1 = (mag * torch.exp(1j * phase)).to(torch.cfloat).to(device)
    re, im = torch.randn(filters, generator=g), torch.randn(filters, generator=g)
    C2 = (0.1 * torch.complex(re, im)).to(device)
    return C1, C2


def _time_fn(fn, *args, iters: int = 20, warmup: int = 5) -> float:
    """Median wall time (ms) of forward+backward over `iters` runs."""
    times = []
    for i in range(warmup + iters):
        for a in args:
            if a.grad is not None:
                a.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = fn(*args)
        y.real.sum().backward()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    from rqf_models.ops.chunked_scan_kernel import (
        _triton_chunked_scan,
        _triton_chunked_scan_prospective,
    )
    from rqf_models.ops.triton_scan_kernel import _triton_sequential_scan

    device = "cuda"
    batch, time_len, filters = 16, 16000, 64
    C1, C2 = _coeffs(filters, device)
    u = torch.randn(batch, time_len, filters, dtype=torch.cfloat, device=device)
    tap0, tap1 = C2, 0.5 * C2

    def prep(*tensors):
        return [t.clone().requires_grad_(True) for t in tensors]

    print(f"scan fwd+bwd, B={batch} T={time_len} F={filters}, complex64, median of 20")
    t_legacy = _time_fn(_triton_sequential_scan, *prep(C1, C2, u))
    t_chunked = _time_fn(_triton_chunked_scan, *prep(C1, C2, u))
    t_fused_p = _time_fn(_triton_chunked_scan_prospective, *prep(C1, tap0, tap1, u))

    def two_pass_prospective(c1, a0, a1, x):
        drive = a0 * x
        drive = torch.cat([drive[:, :1], drive[:, 1:] + a1 * x[:, :-1]], dim=1)
        return _triton_chunked_scan(c1, torch.ones_like(c1), drive)

    t_twopass_p = _time_fn(two_pass_prospective, *prep(C1, tap0, tap1, u))

    print(f"{'legacy sequential (old default)':40s} {t_legacy:8.2f} ms")
    print(f"{'chunked (new default)':40s} {t_chunked:8.2f} ms   {t_legacy / t_chunked:5.1f}x")
    print(f"{'chunked, two-pass prospective':40s} {t_twopass_p:8.2f} ms")
    print(
        f"{'chunked, fused prospective':40s} {t_fused_p:8.2f} ms   "
        f"overhead vs non-prospective: {100 * (t_fused_p / t_chunked - 1):+.1f}%"
    )


if __name__ == "__main__":
    main()
