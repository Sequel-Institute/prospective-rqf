from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from train import build_datasets, build_model, make_loader, run_epoch, seed_everything, validate_cfg


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dictionary, got {type(checkpoint)!r}")
    return checkpoint


def checkpoint_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None and "model" in checkpoint:
        state_dict = checkpoint["model"]
    if state_dict is None:
        raise KeyError("checkpoint does not contain state_dict, model_state_dict, or model")
    return dict(state_dict)


def checkpoint_overrides(checkpoint: dict[str, Any]) -> list[str]:
    overrides = checkpoint.get("hydra_overrides", [])
    if overrides is None:
        return []
    if not isinstance(overrides, list):
        raise TypeError("checkpoint hydra_overrides must be a list")
    return [str(item) for item in overrides]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on a split.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrained/pure_rqf_mfcc_tbptt_seed0.pt"),
        help="Path to a checkpoint produced by train.py or a packaged checkpoint.",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for a quick smoke test.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override. Can be passed multiple times.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    overrides = ["output_dir=outputs/evaluate_checkpoint"]
    overrides += checkpoint_overrides(checkpoint) + list(args.override)
    if args.batch_size is not None:
        overrides.append(f"data.batch_size={args.batch_size}")
    if args.data_root is not None:
        overrides.append(f"data.root={args.data_root}")
    if args.synthetic:
        overrides.append("data.use_synthetic=true")
        overrides.append("data.synthetic_train_size=64")
        overrides.append("data.synthetic_val_size=32")
        overrides.append("data.synthetic_test_size=32")

    config_dir = Path(__file__).resolve().parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.resolve(cfg)
    validate_cfg(cfg)
    seed_everything(int(cfg.seed))

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    train_data, val_data, test_data = build_datasets(cfg)
    dataset = val_data if args.split == "val" else test_data
    loader = make_loader(dataset, cfg, args.split)

    model = build_model(cfg).to(device)
    missing, unexpected = model.load_state_dict(checkpoint_state_dict(checkpoint), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint did not match model exactly.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )
    metrics = run_epoch(model, loader, device, optimizer=None, cfg=cfg)
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "metrics": metrics,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "overrides": overrides,
    }
    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
