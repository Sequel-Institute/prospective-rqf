"""Invariants of the paper-recipe Pure RQF configuration.

Checks that the improved recipe (log-space filter parametrization, log-uniform
frequency band from the clip duration, power-preserving W init, raw complex
state into the readout, bias-free input projection) instantiates with the
documented properties. CPU-only.
"""

from __future__ import annotations

import math
import unittest

import torch

from rqf_models.backbones.rqf.pure import PureRQFEncoder

FS = 16000.0
TAU = 5.0 / FS
SEQ_DUR = 1.0


def _paper_encoder(num_layers: int = 3, hidden: int = 32) -> PureRQFEncoder:
    return PureRQFEncoder(
        input_dims=1,
        output_dims=hidden,
        hidden_dims=hidden,
        num_layers=num_layers,
        samp_frequency=FS,
        min_freq=1.0,
        max_freq=8000.0,
        tau=TAU,
        activation="split_relu",
        prospective_input=True,
        param_space="log",
        seq_duration=SEQ_DUR,
        filter_max_freq_factor=1.0,
        w_init="forward",
        final_activation=False,
        input_proj_bias=False,
        complex_output=True,
    )


class PaperRecipeTest(unittest.TestCase):
    def test_instantiates_and_runs(self):
        torch.manual_seed(0)
        enc = _paper_encoder()
        y = enc(torch.randn(2, 256, 1))
        self.assertTrue(y.is_complex())
        self.assertEqual(tuple(y.shape), (2, 256, 32))

    def test_log_space_leaves_and_band(self):
        torch.manual_seed(0)
        enc = _paper_encoder()
        r = enc.blocks[0].rqf_fwd
        self.assertTrue(hasattr(r, "log_theta") and hasattr(r, "log_gamma"))
        theta = r.theta
        # Recipe band: theta in [2*pi*tau/T, 2*pi*factor]
        self.assertGreaterEqual(float(theta.min()), 2 * math.pi * TAU / SEQ_DUR * 0.999)
        self.assertLessEqual(float(theta.max()), 2 * math.pi * 1.001)
        # Critical-gamma init: gamma = theta / (1 + theta)
        torch.testing.assert_close(r.gamma, theta / (1 + theta), rtol=1e-5, atol=1e-7)

    def test_w_init_power(self):
        torch.manual_seed(0)
        enc = _paper_encoder(num_layers=2, hidden=256)  # wide for tight statistics
        # Layer 0 consumes raw input: E|W|^2 = 2*std^2 = 1/(n_in*G_f) = 2/n_in... with
        # G_f=0.5, G_rho=1: std^2 = 1/(2*n*0.5) = 1/n -> E|W|^2 = 2/n
        n = 256
        p0 = float((enc.blocks[0].W.abs() ** 2).mean())
        self.assertAlmostEqual(p0, 2.0 / n, delta=0.3 * 2.0 / n)
        # Deeper layers compensate split_relu (G_rho=0.5): E|W|^2 doubles.
        p1 = float((enc.blocks[1].W.abs() ** 2).mean())
        self.assertAlmostEqual(p1, 4.0 / n, delta=0.3 * 4.0 / n)

    def test_raw_state_and_no_input_bias(self):
        enc = _paper_encoder()
        self.assertIsNone(enc.input_proj.bias)
        self.assertFalse(enc.blocks[-1].apply_activation)
        for block in enc.blocks[:-1]:
            self.assertTrue(block.apply_activation)

    def test_clamp_parameters_projects_into_band(self):
        torch.manual_seed(0)
        enc = _paper_encoder()
        r = enc.blocks[0].rqf_fwd
        with torch.no_grad():
            r.log_theta += 10.0  # blow past the ceiling
            r.log_gamma += 10.0
        enc.clamp_parameters()
        self.assertLessEqual(float(r.theta.max()), 2 * math.pi * 1.001)
        self.assertLessEqual(float(r.gamma.max()), 1.0)

    def test_sigmoid_default_unchanged(self):
        """The original sigmoid parametrization stays the default."""
        enc = PureRQFEncoder(
            input_dims=1, output_dims=16, hidden_dims=16, num_layers=2,
            samp_frequency=1000.0, min_freq=1.0, max_freq=400.0, tau=0.005,
        )
        r = enc.blocks[0].rqf_fwd
        self.assertTrue(hasattr(r, "freq_param") and hasattr(r, "gamma_param"))
        self.assertFalse(hasattr(r, "log_theta"))


if __name__ == "__main__":
    unittest.main()
