"""LRA Pathfinder / Path-X data and the Path-X training recipe (synthetic PNGs, no LRA download)."""

from __future__ import annotations

import math
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Subset

from rqf_models.data.pathfinder import PathfinderDataSpec
from train import (
    ComplexReadoutClassifier,
    build_datasets,
    build_model,
    build_optimizer,
    run_epoch,
    seed_everything,
    validate_cfg,
)

DIFFICULTY = "curv_contour_length_14"


def write_synthetic_pathfinder(
    root: Path,
    *,
    resolution: int = 32,
    n_meta: int = 2,
    per_meta: int = 10,
    with_empty_png: bool = False,
) -> Path:
    """Create the official layout with tiny grayscale PNGs and text metadata."""

    base = root / "lra_release" / "lra_release" / f"pathfinder{resolution}"
    difficulty = base / DIFFICULTY
    generator = torch.Generator().manual_seed(0)
    for meta_idx in range(n_meta):
        img_dir = difficulty / "imgs" / str(meta_idx)
        img_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for sample_idx in range(per_meta):
            name = f"sample_{sample_idx}.png"
            pixels = torch.randint(
                0, 256, (resolution, resolution), generator=generator, dtype=torch.uint8
            )
            Image.fromarray(pixels.numpy()).save(img_dir / name)
            label = int(torch.randint(0, 2, (1,), generator=generator).item())
            lines.append(f"imgs/{meta_idx} {name} {sample_idx} {label} 1.0 14 4.6 2 0.5 1 1")
        if with_empty_png and meta_idx == 0:
            (img_dir / "sample_broken.png").write_bytes(b"")
            lines.append(f"imgs/{meta_idx} sample_broken.png {per_meta} 0 1.0 14 4.6 2 0.5 1 1")
        meta_dir = difficulty / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / f"{meta_idx}.npy").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base


def pathx_config(*overrides: str):
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
    ):
        cfg = compose(config_name="config", overrides=[
            "+experiment=pathx", "output_dir=outputs/test_pathx", *overrides,
        ])
    OmegaConf.resolve(cfg)
    validate_cfg(cfg)
    return cfg


class TempRootTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def spec(self, **kwargs) -> PathfinderDataSpec:
        kwargs.setdefault("verify_complete", False)
        return PathfinderDataSpec(root=str(self.root), resolution=32, **kwargs)


class PathfinderDataTest(TempRootTest):
    def test_shapes_split_and_standardization(self):
        write_synthetic_pathfinder(self.root, n_meta=2, per_meta=10)
        train, val, test = self.spec().build()

        self.assertEqual((len(train), len(val), len(test)), (16, 2, 2))
        x, y = train[0]
        self.assertEqual(x.shape, (1024, 1))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.long)
        train_x = train.tensors[0]
        self.assertLess(abs(train_x.mean().item()), 1e-5)
        self.assertLess(abs(train_x.std().item() - 1.0), 1e-2)

    def test_labels_and_pixels_match_metadata(self):
        base = write_synthetic_pathfinder(self.root, n_meta=1, per_meta=8)
        train, _, _ = self.spec(normalize=False).build()

        # Example 0 of the train split is the first index of the seed-42
        # permutation over the metadata line order.
        lines = (base / DIFFICULTY / "metadata" / "0.npy").read_text().splitlines()
        first = int(torch.randperm(8, generator=torch.Generator().manual_seed(42))[0])
        fields = lines[first].split()
        with Image.open(base / DIFFICULTY / fields[0] / fields[1]) as im:
            pixels = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        x, y = train[0]
        self.assertTrue(torch.equal(x.squeeze(-1), pixels.float() / 255.0))
        self.assertEqual(int(y), int(fields[3]))

    def test_empty_png_is_skipped_with_warning(self):
        write_synthetic_pathfinder(self.root, n_meta=1, per_meta=10, with_empty_png=True)
        with self.assertWarnsRegex(UserWarning, "skipped 1 unreadable"):
            splits = self.spec().build()
        self.assertEqual(sum(len(split) for split in splits), 10)

    def test_cache_roundtrip_avoids_redecode(self):
        base = write_synthetic_pathfinder(self.root, n_meta=1, per_meta=10)
        first, _, _ = self.spec().build()
        self.assertTrue((base / "processed_data" / DIFFICULTY / "images_u8.pt").exists())

        # Without the PNGs, only the cache can serve the second build.
        for png in (base / DIFFICULTY / "imgs").rglob("*.png"):
            png.unlink()
        second, _, _ = self.spec().build()
        self.assertTrue(torch.equal(first.tensors[0], second.tensors[0]))
        self.assertTrue(torch.equal(first.tensors[1], second.tensors[1]))

    def test_cache_without_marker_is_adopted(self):
        # Caches written without complete.json are validated and reused.
        base = write_synthetic_pathfinder(self.root, n_meta=1, per_meta=10)
        images = torch.randint(0, 256, (10, 1024), dtype=torch.uint8)
        labels = torch.tensor([0, 1] * 5)
        cache = base / "processed_data" / DIFFICULTY
        cache.mkdir(parents=True)
        torch.save(images, cache / "images_u8.pt")
        torch.save(labels, cache / "labels.pt")
        for png in (base / DIFFICULTY / "imgs").rglob("*.png"):
            png.unlink()

        train, val, test = self.spec(normalize=False).build()

        self.assertTrue((cache / "complete.json").exists())
        perm = torch.randperm(10, generator=torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(train.tensors[0].squeeze(-1), images[perm[:8]].float() / 255.0))
        self.assertTrue(torch.equal(test.tensors[1], labels[perm[9:]]))
        self.assertEqual((len(train), len(val), len(test)), (8, 1, 1))

    def test_incomplete_cache_is_rebuilt(self):
        base = write_synthetic_pathfinder(self.root, n_meta=1, per_meta=10)
        cache = base / "processed_data" / DIFFICULTY
        cache.mkdir(parents=True)
        (cache / "images_u8.pt").write_bytes(b"truncated")
        (cache / "labels.pt").write_bytes(b"")

        train, _, _ = self.spec().build()

        self.assertEqual(len(train), 8)
        self.assertTrue((cache / "complete.json").exists())
        images = torch.load(cache / "images_u8.pt", weights_only=True)
        labels = torch.load(cache / "labels.pt", weights_only=True)
        self.assertEqual(images.shape, (10, 1024))
        self.assertEqual(labels.shape, (10,))

    def test_concurrent_first_use_decodes_once(self):
        write_synthetic_pathfinder(self.root, n_meta=1, per_meta=10)
        original_decode = PathfinderDataSpec._decode_all
        original_load = PathfinderDataSpec._load_cache
        lock = threading.Lock()
        counts = {"decodes": 0, "active_loads": 0, "max_active_loads": 0}

        def slow_decode(spec):
            with lock:
                counts["decodes"] += 1
            time.sleep(0.05)
            return original_decode(spec)

        def tracked_load(spec, cache_dir, *, require_marker):
            track = require_marker and (cache_dir / "complete.json").exists()
            if track:
                with lock:
                    counts["active_loads"] += 1
                    counts["max_active_loads"] = max(
                        counts["max_active_loads"], counts["active_loads"]
                    )
                time.sleep(0.05)
            try:
                return original_load(spec, cache_dir, require_marker=require_marker)
            finally:
                if track:
                    with lock:
                        counts["active_loads"] -= 1

        with mock.patch.object(PathfinderDataSpec, "_decode_all", slow_decode), \
                mock.patch.object(PathfinderDataSpec, "_load_cache", tracked_load):
            specs = [self.spec() for _ in range(3)]
            with ThreadPoolExecutor(max_workers=3) as pool:
                results = list(pool.map(lambda spec: spec.build(), specs))

        self.assertEqual(counts["decodes"], 1)
        # Waiters load the published cache in parallel, outside the lock.
        self.assertGreaterEqual(counts["max_active_loads"], 2)
        self.assertTrue(torch.equal(results[0][0].tensors[0], results[1][0].tensors[0]))

    def test_verify_complete_rejects_partial_source_and_cache(self):
        write_synthetic_pathfinder(self.root, n_meta=1, per_meta=10)
        with self.assertRaisesRegex(FileNotFoundError, "expected 200"):
            self.spec(verify_complete=True).build()

        # An undersized cache built with verification off is not adopted later.
        self.spec(verify_complete=False).build()
        with self.assertRaisesRegex(FileNotFoundError, "expected 200"):
            self.spec(verify_complete=True).build()

    def test_stale_staging_dir_is_reclaimed(self):
        base = write_synthetic_pathfinder(self.root, n_meta=1, per_meta=10)
        cache_parent = base / "processed_data"
        cache_parent.mkdir(parents=True, exist_ok=True)
        stale = cache_parent / f".{DIFFICULTY}-dead"
        stale.mkdir()
        (stale / "images_u8.pt").write_bytes(b"junk")

        self.spec().build()

        self.assertFalse(stale.exists())

    def test_max_samples_invalid_specs_and_missing_root(self):
        write_synthetic_pathfinder(self.root, n_meta=1, per_meta=10)
        train, val, test = self.spec(max_samples_per_split=2).build()
        self.assertEqual((len(train), len(val), len(test)), (2, 1, 1))

        for kwargs in ({"resolution": 48}, {"train_split": 0.9, "val_split": 0.1},
                       {"max_samples_per_split": 0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                PathfinderDataSpec(**kwargs)
        with self.assertRaises(FileNotFoundError):
            PathfinderDataSpec(root=str(self.root / "nowhere")).build()


class PathXRecipeTest(TempRootTest):
    def test_recipe_matches_source_architecture_and_optimizer(self):
        cfg = pathx_config()
        self.assertEqual((cfg.data.name, cfg.data.resolution, cfg.data.sequence_length),
                         ("pathfinder", 128, 16384))
        model = build_model(cfg)
        trainable = sum(p.numel() * (2 if p.is_complex() else 1)
                        for p in model.parameters() if p.requires_grad)
        # 6 complex W (64x1, 5 x 64x64), log_theta/log_gamma, 256-unit MLP readout.
        self.assertEqual(trainable, 75_394)
        self.assertIsNone(model.recall_steps)
        self.assertEqual(model.encoder.blocks[0].W.shape, (64, 1))
        self.assertIsInstance(model.encoder.input_proj, torch.nn.Identity)

        # Log-uniform init band: [2*pi*tau/T, 2*pi*F] = [5/16384, pi/2].
        for block in model.encoder.blocks:
            theta = block.rqf_fwd.log_theta.exp()
            self.assertGreaterEqual(theta.min().item(), 5 / 16384 * (1 - 1e-5))
            self.assertLessEqual(theta.max().item(), math.pi / 2 * (1 + 1e-5))
            torch.testing.assert_close(block.rqf_fwd.log_gamma.exp(), theta / (1 + theta))

        hidden, logits = model.head[0], model.head[3]
        self.assertAlmostEqual(hidden.weight.std().item(), math.sqrt(2 / 128), delta=0.005)
        self.assertLess(logits.weight.abs().max().item(), 0.01)
        self.assertTrue(bool((hidden.bias == 0).all()) and bool((logits.bias == 0).all()))

        optimizer = build_optimizer(cfg, model)
        self.assertIs(type(optimizer), torch.optim.Adam)
        for group in optimizer.param_groups:
            self.assertEqual((group["lr"], group["weight_decay"], group["eps"]), (0.003, 0.0, 1e-4))
        self.assertTrue(cfg.training.per_step_loss)
        self.assertEqual(cfg.training.scan_grad_mode, "full")
        self.assertEqual((cfg.training.scheduler, cfg.training.min_lr), ("cosine", 0.0))

    def test_one_training_epoch_on_synthetic_pathfinder(self):
        write_synthetic_pathfinder(self.root, n_meta=2, per_meta=10)
        cfg = pathx_config(
            f"data.root={self.root}", "data.resolution=32", "data.sequence_length=1024",
            "data.verify_complete=false", "training.batch_size=2", "device=cpu",
        )
        seed_everything(0)
        train, val, test = build_datasets(cfg)
        self.assertEqual((len(train), len(val), len(test)), (16, 2, 2))
        model = build_model(cfg)
        optimizer = build_optimizer(cfg, model)

        loader = DataLoader(Subset(train, [0, 1]), batch_size=2)
        metrics = run_epoch(model, loader, torch.device("cpu"), optimizer, cfg)
        self.assertTrue(math.isfinite(metrics["loss"]))
        for name, param in model.named_parameters():
            if param.requires_grad:
                with self.subTest(name=name):
                    self.assertIsNotNone(param.grad)
                    self.assertTrue(bool(torch.isfinite(torch.view_as_real(param.grad)
                                                        if param.is_complex() else param.grad).all()))

        # Training returns per-timestep logits; evaluation averages them over time.
        x = torch.stack([train[0][0], train[1][0]])
        model.train()
        self.assertEqual(model(x).shape, (2, 1024, 2))
        model.eval()
        with torch.no_grad():
            self.assertEqual(model(x).shape, (2, 2))
        eval_metrics = run_epoch(model, DataLoader(val, batch_size=2), torch.device("cpu"), None, cfg)
        self.assertTrue(0.0 <= eval_metrics["acc"] <= 1.0)

    def test_validation_rejects_inconsistent_configs(self):
        with self.assertRaisesRegex(ValueError, "resolution"):
            pathx_config("data.resolution=32")
        with self.assertRaisesRegex(ValueError, "one feature and two classes"):
            pathx_config("data.feature_dim=2")
        # Other readouts keep the per-step loss for TBPTT(1) only.
        with initialize_config_dir(
            version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
        ):
            cfg = compose(config_name="config", overrides=[
                "output_dir=outputs/test_pathx", "model=pure_rqf", "training=full_bptt",
                "training.per_step_loss=true",
            ])
        with self.assertRaisesRegex(ValueError, "tbptt1"):
            validate_cfg(cfg)

    def test_readout_hidden_init_option(self):
        encoder = torch.nn.Identity()
        with self.assertRaisesRegex(ValueError, "hidden_init"):
            ComplexReadoutClassifier(encoder, 4, 2, 8, 0.0, False, hidden_init="xavier")
        with self.assertRaisesRegex(ValueError, "hidden layer"):
            ComplexReadoutClassifier(encoder, 4, 2, None, 0.0, False, hidden_init="he_normal")
        torch.manual_seed(0)
        readout = ComplexReadoutClassifier(encoder, 512, 2, 1024, 0.0, False, hidden_init="he_normal")
        self.assertAlmostEqual(readout.head[0].weight.std().item(), math.sqrt(2 / 1024), delta=1e-3)
        self.assertTrue(bool((readout.head[0].bias == 0).all()))


if __name__ == "__main__":
    unittest.main()
