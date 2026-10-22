"""CUDA forward/backward timing and peak memory for static/reference/fused RQFs."""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rqf_models.backbones.rqf import PureRQFEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--grad-mode", choices=("full", "tbptt1"), default="full")
    parser.add_argument("--non-prospective", action="store_true")
    args = parser.parse_args()
    if min(args.time, args.batch, args.width, args.depth, args.steps) < 1:
        parser.error("shape dimensions and steps must be positive")
    if not torch.cuda.is_available():
        parser.error("this benchmark requires CUDA and Triton")
    import triton  # noqa: F401

    torch.manual_seed(0)
    kwargs = dict(
        input_dims=1,
        hidden_dims=args.width,
        output_dims=args.width,
        num_layers=args.depth,
        samp_frequency=16000.0,
        tau=5 / 16000,
        param_space="log",
        seq_duration=1.0,
        activation="split_relu",
        discretization="zoh",
        final_activation=False,
        complex_output=True,
        input_proj_bias=False,
        scan_grad_mode=args.grad_mode,
        prospective_input=not args.non_prospective,
    )
    static = PureRQFEncoder(**kwargs).cuda()
    selected = PureRQFEncoder(
        **kwargs,
        selective={
            "write_control": True,
            "frequency_warp": "grouped",
            "num_groups": min(4, args.width),
        },
    ).cuda()
    with torch.no_grad():
        for block in selected.blocks:
            for p in block.selector.parameters():
                p.normal_(std=0.01)
    x = torch.randn(args.batch, args.time, 1, device="cuda")
    results = []
    runs = (
        ("static", static, "1"),
        ("selective_reference", copy.deepcopy(selected), "0"),
        ("selective_fused", selected, "1"),
    )
    for name, model, fused in runs:
        for _, other, _ in runs:
            other.zero_grad(set_to_none=True)
        os.environ["ORGANICS_RQF_FBT"] = fused
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            model(x).abs().square().mean().backward()
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        forward_ms = backward_ms = 0.0
        for _ in range(args.steps):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss = model(x).abs().square().mean()
            torch.cuda.synchronize()
            middle = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            end = time.perf_counter()
            forward_ms += 1000 * (middle - start) / args.steps
            backward_ms += 1000 * (end - middle) / args.steps
        results.append(
            dict(
                backend=name,
                forward_ms=forward_ms,
                backward_ms=backward_ms,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_increment_bytes=torch.cuda.max_memory_allocated() - baseline,
            )
        )
    print(
        json.dumps(
            dict(
                shape=vars(args), device=torch.cuda.get_device_name(), results=results
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
