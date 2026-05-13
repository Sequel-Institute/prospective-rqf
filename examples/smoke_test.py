from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import build_model, validate_cfg


def main() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        for model_name in ("pure_rqf", "rqf_s4", "s4d", "s5"):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"model={model_name}",
                    "data.use_synthetic=true",
                    "training.max_epochs=1",
                ],
            )
            validate_cfg(cfg)
            model = build_model(cfg)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{model_name}: trainable_params={params}")


if __name__ == "__main__":
    main()
