from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate packaged checkpoints from a pretrained directory."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("pretrained"),
        help="Directory containing packaged .pt checkpoints.",
    )
    parser.add_argument("--pattern", default="*.pt", help="Checkpoint glob pattern.")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pretrained_eval"))
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override passed through to evaluate_checkpoint.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = sorted(args.checkpoint_dir.glob(args.pattern))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints matched {args.checkpoint_dir / args.pattern}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        output_json = args.output_dir / f"{checkpoint.stem}_{args.split}.json"
        cmd = [
            sys.executable,
            "evaluate_checkpoint.py",
            "--checkpoint",
            str(checkpoint),
            "--split",
            args.split,
            "--device",
            args.device,
            "--output-json",
            str(output_json),
        ]
        if args.synthetic:
            cmd.append("--synthetic")
        if args.data_root is not None:
            cmd += ["--data-root", args.data_root]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        for override in args.override:
            cmd += ["--override", override]

        print(f"Evaluating {checkpoint}...")
        subprocess.run(cmd, check=True)
        result = json.loads(output_json.read_text())
        summary.append(
            {
                "checkpoint": str(checkpoint),
                "split": args.split,
                "metrics": result["metrics"],
                "checkpoint_epoch": result.get("checkpoint_epoch"),
                "output_json": str(output_json),
            }
        )

    summary_path = args.output_dir / f"summary_{args.split}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
