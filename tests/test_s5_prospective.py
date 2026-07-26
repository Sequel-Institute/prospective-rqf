"""Tests for the two exact prospective-input S5 definitions."""

from __future__ import annotations

import math
import unittest

import torch

from rqf_models.backbones.s5.layer import S5Layer, _discretize_zoh


class S5ProspectiveDefinitionsTest(unittest.TestCase):
    @staticmethod
    def _make_layer(*, mode: str, clip_eigs: bool) -> S5Layer:
        torch.manual_seed(0)
        layer = S5Layer(
            d_model=2,
            d_state=4,
            blocks=1,
            activation="gelu",
            conj_sym=False,
            clip_eigs=clip_eigs,
            prospective_input=True,
            prospective_mode=mode,
            prospective_horizon_multiplier=5.0,
        )
        with torch.no_grad():
            layer.Lambda_re.copy_(torch.tensor([-0.5, -1.0, -0.25, -2.0]))
            layer.Lambda_im.copy_(torch.tensor([0.2, 0.4, 0.8, 1.6]))
            layer.log_step.copy_(torch.log(torch.tensor([0.01, 0.02, 0.04, 0.08])))
            layer.B_re.copy_(
                torch.tensor(
                    [
                        [0.1, 0.2],
                        [0.3, 0.4],
                        [0.5, 0.6],
                        [0.7, 0.8],
                    ]
                )
            )
            layer.B_im.copy_(
                torch.tensor(
                    [
                        [-0.1, 0.05],
                        [-0.2, 0.10],
                        [-0.3, 0.15],
                        [-0.4, 0.20],
                    ]
                )
            )
        return layer

    def test_direct_is_default_and_uses_shared_clock_multiplier(self) -> None:
        default_layer = S5Layer(
            d_model=2,
            d_state=4,
            blocks=1,
            conj_sym=False,
            prospective_input=True,
        )
        self.assertEqual(default_layer.prospective_mode, "direct")
        self.assertEqual(default_layer.prospective_horizon_multiplier, 5.0)

        layer = self._make_layer(mode="direct", clip_eigs=False)

        lambda_c = layer._lambda()
        delta = layer._delta()
        learned_input = torch.complex(layer.B_re, layer.B_im)
        expected_a, expected_b = _discretize_zoh(lambda_c, learned_input, delta)
        expected_correction = (
            5.0 * delta.unsqueeze(-1) * expected_a.unsqueeze(-1) * learned_input
        )

        actual_a, actual_plus, actual_minus = layer._discretize_prospective()

        torch.testing.assert_close(actual_a, expected_a)
        torch.testing.assert_close(actual_plus, expected_b + expected_correction)
        torch.testing.assert_close(actual_minus, -expected_correction)
        torch.testing.assert_close(actual_plus + actual_minus, expected_b)

    def test_decay_gain_uses_alpha_times_learned_input(self) -> None:
        layer = self._make_layer(mode="decay_gain", clip_eigs=True)

        lambda_c = layer._lambda()
        delta = layer._delta()
        learned_input = torch.complex(layer.B_re, layer.B_im)
        alpha = -lambda_c.real
        continuous_input = alpha.to(learned_input.dtype).unsqueeze(-1) * learned_input
        expected_a, expected_b = _discretize_zoh(lambda_c, continuous_input, delta)
        expected_correction = (
            5.0 * delta.unsqueeze(-1) * expected_a.unsqueeze(-1) * continuous_input
        )

        actual_a, actual_plus, actual_minus = layer._discretize_prospective()

        torch.testing.assert_close(actual_a, expected_a)
        torch.testing.assert_close(actual_plus, expected_b + expected_correction)
        torch.testing.assert_close(actual_minus, -expected_correction)
        torch.testing.assert_close(actual_plus + actual_minus, expected_b)

    def test_mode_and_shared_multiplier_are_validated(self) -> None:
        common = {
            "d_model": 2,
            "d_state": 4,
            "blocks": 1,
            "conj_sym": False,
            "prospective_input": True,
        }
        with self.assertRaisesRegex(ValueError, "prospective_mode"):
            S5Layer(**common, prospective_mode="per_mode_tau")
        with self.assertRaisesRegex(ValueError, "must be positive"):
            S5Layer(**common, prospective_horizon_multiplier=0.0)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            S5Layer(**common, prospective_horizon_multiplier=math.inf)
        with self.assertRaisesRegex(ValueError, "requires clip_eigs=True"):
            S5Layer(**common, prospective_mode="decay_gain", clip_eigs=False)


if __name__ == "__main__":
    unittest.main()
