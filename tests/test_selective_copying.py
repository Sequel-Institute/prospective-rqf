"""Task semantics and end-to-end learning for selective copying."""

import math
from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, TensorDataset

from rqf_models.data.selective_copying import SelectiveCopyingDataSpec
from train import (
    build_datasets,
    build_model,
    build_optimizer,
    run_epoch,
    seed_everything,
    validate_cfg,
)


def copying_config(*overrides):
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
    ):
        cfg = compose(config_name="config", overrides=[
            "+experiment=selective_copying", "output_dir=outputs/test_copying", *overrides,
        ])
    OmegaConf.resolve(cfg)
    validate_cfg(cfg)
    return cfg


class SelectiveCopyingDataTest(unittest.TestCase):
    def test_tokens_are_recalled_in_order_and_only_suffix_is_supervised(self):
        for length, memorize, vocab in ((4096, 16, 16), (40, 4, 8), (2, 1, 3)):
            with self.subTest(length=length):
                spec = SelectiveCopyingDataSpec(
                    seq_len=length, n_memorize=memorize, n_tokens=vocab, n_train=2
                )
                dataset, _, _ = spec.build()
                x, y = dataset[0]
                self.assertEqual(x.shape, (length, vocab))
                torch.testing.assert_close(x.sum(-1), torch.ones(length))
                context = x[:-memorize].argmax(-1)
                data = context[context != 0]
                self.assertEqual(data.numel(), memorize)
                self.assertTrue(bool(((data > 0) & (data < vocab - 1)).all()))
                torch.testing.assert_close(y[-memorize:], data)
                self.assertTrue(bool((y[:-memorize] == -100).all()))
                self.assertTrue(bool((x[-memorize:].argmax(-1) == vocab - 1).all()))

    def test_training_resamples_but_evaluation_is_fixed_and_splits_differ(self):
        spec = SelectiveCopyingDataSpec(seq_len=64, n_memorize=4, n_train=2)
        train, val, test = spec.build()
        original = train[0]
        original_val, original_test = val[0], test[0]
        train.set_epoch(1)
        val.set_epoch(1)
        test.set_epoch(1)
        self.assertFalse(torch.equal(original[0], train[0][0]))
        self.assertFalse(torch.equal(original_val[0], original_test[0]))
        for expected, actual in ((original_val, val[0]), (original_test, test[0])):
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        train.set_epoch(0)
        for a, b in zip(original, train[0]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with self.assertRaises(IndexError):
            train[2]

    def test_invalid_task_sizes_fail_early(self):
        for kwargs in ({"seq_len": 31}, {"n_tokens": 2}, {"n_memorize": 0},
                       {"n_train": 0}, {"seed": -1}, {"seq_len": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SelectiveCopyingDataSpec(**kwargs)


class SelectiveCopyingTrainingTest(unittest.TestCase):
    def test_recipe_matches_architecture_and_optimizer(self):
        cfg = copying_config()
        model = build_model(cfg)
        real_parameters = sum(p.numel() * (2 if p.is_complex() else 1)
                              for p in model.parameters() if p.requires_grad)
        self.assertEqual(real_parameters, 119184)
        self.assertEqual([tuple(b.W.shape) for b in model.encoder.blocks],
                         [(64, 16), (64, 64), (64, 64), (64, 64)])
        self.assertEqual(model.encoder.selective.input_source, "layer_output")
        self.assertEqual(model.encoder.selective.max_log_scale, math.log(4))
        self.assertEqual(model.recall_steps, 16)
        self.assertIsInstance(model.encoder.input_proj, torch.nn.Identity)
        self.assertIsInstance(model.encoder.output_proj, torch.nn.Identity)
        opt = build_optimizer(cfg, model)
        self.assertIs(type(opt), torch.optim.Adam)
        for group in opt.param_groups:
            self.assertEqual(group["lr"], 1e-3)
            self.assertEqual(group["eps"], 1e-4)
            self.assertEqual(group["weight_decay"], 0)
        self.assertEqual(cfg.training.scheduler, "none")
        self.assertEqual(cfg.training.max_epochs * cfg.data.n_train // cfg.training.batch_size,
                         100000)

    def test_recall_metrics_ignore_context_and_weight_partial_batches(self):
        cfg = copying_config("data.sequence_length=40", "data.n_memorize=4",
                             "data.n_train=3", "model.depth=2", "model.hidden_width=8")
        seed_everything(0)
        model = build_model(cfg)
        dataset, _, _ = build_datasets(cfg)
        x, y = (torch.stack(items) for items in zip(*(dataset[i] for i in range(3))))
        model.eval()
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, (3, 4, 16))
        expected_loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y[:, -4:].flatten())
        expected_acc = (logits.argmax(-1) == y[:, -4:]).float().mean()
        y[:, :-4] = 0  # Even valid labels in the context must not enter the metrics.
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        actual = run_epoch(model, loader, torch.device("cpu"), None, cfg)
        self.assertAlmostEqual(actual["loss"], float(expected_loss), places=6)
        self.assertAlmostEqual(actual["acc"], float(expected_acc), places=6)
        model.train()
        self.assertEqual(model(x).shape, (3, 4, 16))

    def test_training_updates_all_controllers_and_checkpoint_round_trips(self):
        cfg = copying_config("data.sequence_length=40", "data.n_memorize=4",
                             "data.n_train=2", "model.depth=2", "model.hidden_width=8")
        seed_everything(0)
        model = build_model(cfg)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        dataset, _, _ = build_datasets(cfg)
        loader = DataLoader(dataset, batch_size=2)
        metrics = run_epoch(model, loader, torch.device("cpu"), build_optimizer(cfg, model), cfg)
        self.assertTrue(math.isfinite(metrics["loss"]))
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            with self.subTest(parameter=name):
                self.assertIsNotNone(p.grad)
                self.assertTrue(bool(torch.isfinite(p.grad).all()))
                self.assertGreater(float(p.grad.abs().sum()), 0)
                self.assertFalse(torch.equal(before[name], p))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save({"model": model.state_dict()}, path)
            restored = build_model(cfg)
            restored.load_state_dict(torch.load(path, weights_only=True)["model"], strict=True)
        x, _ = next(iter(loader))
        with torch.no_grad():
            torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
