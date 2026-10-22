"""Reject unsupported selective lag gradients without blocking harmless modes."""

import unittest

import torch
from omegaconf import OmegaConf

from rqf_models.backbones.rqf.pure import PureRQFBlock
from train import validate_cfg


def make_block(**overrides):
    kwargs = dict(
        d_model=2,
        samp_frequency=100.0,
        min_freq=1.0,
        max_freq=5.0,
        dropout=0.0,
        activation="split_relu",
        use_parallel_scan=False,
        bidirectional=False,
        discretization="zoh",
        tau=0.05,
        learn_tau=False,
        max_gamma=0.85,
        min_gamma=0.001,
        sigma_init=0.1,
        norm_type="none",
        filter_scales=None,
        readout_mode="complex",
        num_oscillators=1,
        c_projection=False,
        param_space="log",
        selective={"input_source": "layer_input"},
        prospective_input=True,
        scan_grad_mode="tbptt1",
        tbptt_delayed_input_mode="literal",
    )
    kwargs.update(overrides)
    return PureRQFBlock(**kwargs)


def make_cfg(**overrides):
    cfg = OmegaConf.create({
        "model": {
            "encoder": "pure_rqf",
            "selective": {},
            "prospective_input": True,
            "prospective_tau": None,
        },
        "training": {
            "scan_grad_mode": "tbptt1",
            "tbptt_delayed_input_mode": "literal",
            "per_step_loss": False,
        },
        "data": {"name": "speech_commands"},
    })
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


class TestSelectiveTBPTTValidation(unittest.TestCase):
    def test_direct_block_rejects_literal_with_nonzero_lag(self):
        for mode in ("tau", "gamma"):
            for horizon in (None, 0.1, [0.0, 0.1], torch.tensor([0.0, 0.1])):
                with self.subTest(mode=mode, horizon=horizon):
                    with self.assertRaisesRegex(ValueError, "literal.*nonzero horizon"):
                        make_block(prospective_mode=mode, prospective_tau=horizon)

    def test_direct_block_allows_modes_with_no_unsupported_lag_gradient(self):
        cases = (
            {"scan_grad_mode": "full"},
            {"prospective_input": False},
            {"prospective_tau": 0.0},
            {"prospective_tau": [0.0, 0.0]},
            {"prospective_tau": torch.zeros(2)},
            {"tbptt_delayed_input_mode": "strict"},
        )
        x = torch.randn(1, 4, 2, dtype=torch.complex64)
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                block = make_block(**kwargs)
                self.assertTrue(bool(torch.isfinite(block(x)).all()))

    def test_dispatch_rejects_postconstruction_mode_and_horizon_changes(self):
        block = make_block(tbptt_delayed_input_mode="strict")
        block.tbptt_delayed_input_mode = "literal"
        x = torch.randn(1, 4, 2, dtype=torch.complex64)
        with self.assertRaisesRegex(ValueError, "literal.*nonzero horizon"):
            block(x)
        block.prospective_tau = 0.0
        self.assertTrue(bool(torch.isfinite(block(x)).all()))
        block.prospective_tau = 0.1
        with self.assertRaisesRegex(ValueError, "literal.*nonzero horizon"):
            block(x)

    def test_config_rejects_literal_with_nonzero_lag(self):
        for horizon in (None, 0.1, [0.0, 0.1]):
            with self.subTest(horizon=horizon):
                cfg = make_cfg(**{"model.prospective_tau": horizon})
                with self.assertRaisesRegex(ValueError, "literal.*nonzero horizon"):
                    validate_cfg(cfg)

    def test_config_allows_harmless_and_nonselective_modes(self):
        cases = (
            {"training.scan_grad_mode": "full"},
            {"model.prospective_input": False},
            {"model.prospective_tau": 0.0},
            {"model.prospective_tau": [0.0, 0.0]},
            {"training.tbptt_delayed_input_mode": "strict"},
            {"model.selective": None},
            {"model.selective": None, "model.encoder": "rqf_s4"},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                validate_cfg(make_cfg(**overrides))


if __name__ == "__main__":
    unittest.main()
