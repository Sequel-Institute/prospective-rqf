"""Compiled CUDA parity; optionally run kernel arithmetic in the Triton interpreter.

The module remains importable without Triton. Interpreter runs use CPU tensors
and do not validate GPU compilation, scheduling, or performance.
"""

import copy
import importlib.util
import itertools
import math
import os
import unittest
from unittest import mock

import torch

from rqf_models.backbones.rqf.selective import SelectiveController
from rqf_models.ops.selective_scan import selective_layer_scan
from test_selective_rqf import (
    CONTROLLER_CASES,
    VARIANTS,
    assert_empty_input_support,
    inputs_for,
    make_model,
)


INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
TRITON = importlib.util.find_spec("triton") is not None
AVAILABLE = TRITON and (torch.cuda.is_available() or INTERPRET)


@unittest.skipUnless(AVAILABLE, "requires CUDA + Triton, or a Triton CPU interpreter")
class SelectiveKernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu" if INTERPRET else "cuda")
        if INTERPRET:
            from triton.runtime import interpreter

            cls.convert = interpreter._implicit_cvt
            interpreter._implicit_cvt = lambda v: (
                v if isinstance(v, int) else cls.convert(v)
            )

    @classmethod
    def tearDownClass(cls):
        if INTERPRET:
            from triton.runtime import interpreter

            interpreter._implicit_cvt = cls.convert

    def compare(
        self,
        variant,
        *,
        time=33,
        prospective=True,
        truncated=False,
        activate=True,
        write_raw=True,
        mode="tau",
        horizon=None,
        freeze=False,
        tiny=False,
        collapse=False,
        autocast=False,
        conjugate_cot=False,
        saturate_write=False,
        freeze_base=False,
        clip_warp=False,
        gamma_edges=False,
        chunk=32,
    ):
        block = (
            make_model(
                variant,
                active=True,
                num_layers=1,
                prospective_input=prospective,
                prospective_mode=mode,
                prospective_tau=horizon,
                samp_frequency=4.0 if clip_warp else 100.0,
            )
            .blocks[0]
            .to(self.device)
        )
        if freeze_base:
            block.rqf_fwd.requires_grad_(False)
            block.W.requires_grad_(False)
            config = block.selector.config
            block.selector = SelectiveController(block.W.shape[1], block.rqf_fwd, config).to(self.device)
            with torch.no_grad():
                for parameter in block.selector.parameters():
                    parameter.normal_(std=0.3)
            self.assertTrue(all(p.requires_grad for p in block.selector.parameters()))
        with torch.no_grad():
            if tiny:
                block.rqf_fwd.log_gamma.fill_(-70.0)
                block.rqf_fwd.log_theta.fill_(-50.0)
            if collapse:
                block.rqf_fwd.dt = 1000.0
            if block.selector.W_write is not None:
                block.selector.W_write[0].zero_()
                block.selector.W_write[0, 0] = -1.0
            if clip_warp:
                lower, upper = block.rqf_fwd._theta_bounds()
                block.rqf_fwd.log_theta.copy_((lower * upper).sqrt().log())
                block.selector.W_delta.zero_()
                block.selector.W_delta[:, 0] = 1.0
            if gamma_edges:
                block.rqf_fwd.log_gamma[:2] = -50.0
                block.rqf_fwd.log_gamma[2:] = 50.0
                block.rqf_fwd.clamp_parameters()
                block.selector.W_gamma.zero_()
                margin = block.selector.config.gamma_edge_margin
                offset = math.log((1 - margin) / margin)
                block.selector.W_gamma[:2, 0] = offset
                block.selector.W_gamma[2:, 0] = -offset
        if freeze:
            block.selector.requires_grad_(False)
            block.rqf_fwd.requires_grad_(False)
        reference = copy.deepcopy(block)
        x = torch.randn(
            5, 2, time, dtype=torch.complex64, device=self.device, requires_grad=True
        )
        s = torch.randn_like(x).requires_grad_()
        with torch.no_grad():
            # These force zero/negative gates for affine and finite gates for sigmoid.
            s[0, 0, 0] = 1.0
            if time > 1:
                s[0, 0, 1] = 2.0
            if saturate_write:
                s[0, 0, :3] = torch.tensor([-1000.0, 0.0, 1000.0], device=self.device)
            if clip_warp:
                # Interior and clipped samples both have nonsaturated tanh
                # derivatives, so omitted clipping masks fail gradient parity.
                pattern = torch.tensor([-2.0, 0.1, 2.0], device=self.device)
                s[0] = pattern.repeat((time + 2) // 3)[:time]
            if gamma_edges:
                s[0] = torch.linspace(0.9, 1.1, time, device=self.device)
        xr, sr = (
            x.detach().clone().requires_grad_(),
            s.detach().clone().requires_grad_(),
        )
        kw = dict(
            truncated=truncated, activate=activate, write_raw=write_raw, chunk=chunk
        )
        # Fail if an eligibility regression silently substitutes the reference.
        with (
            torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=autocast),
            mock.patch(
                "rqf_models.ops.selective_scan.reference_layer_scan",
                side_effect=AssertionError("CUDA test fell back to PyTorch"),
            ),
        ):
            actual = selective_layer_scan(x, block.W, inputs_for(block, s), **kw)
        expected = selective_layer_scan(
            xr, reference.W, inputs_for(reference, sr), use_kernel=False, **kw
        )
        actual_outputs, expected_outputs, cots = [], [], []
        for a, e in zip(actual, expected):
            if a is None:
                self.assertIsNone(e)
            else:
                torch.testing.assert_close(a, e, rtol=4e-4, atol=3e-5)
                actual_outputs.append(a)
                expected_outputs.append(e)
                cot = torch.randn_like(a)
                cots.append(cot.conj() if conjugate_cot else cot)
        leaves = [x, s] + [p for p in block.parameters() if p.requires_grad]
        refs = [xr, sr] + [p for p in reference.parameters() if p.requires_grad]
        actual_grads = torch.autograd.grad(
            actual_outputs, leaves, cots, allow_unused=True
        )
        expected_grads = torch.autograd.grad(
            expected_outputs, refs, cots, allow_unused=True
        )
        for a, e in zip(actual_grads, expected_grads):
            if e is None:
                self.assertIsNone(a)
            else:
                self.assertIsNotNone(a)
                torch.testing.assert_close(a, e, rtol=4e-3, atol=3e-4)
        if freeze_base:
            for gradient in actual_grads[2:]:
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum()), 0)

    def test_empty_inputs(self):
        assert_empty_input_support(self, self.device)

    def test_variants_and_gradient_modes(self):
        for variant, prospective, truncated in itertools.product(
            CONTROLLER_CASES, (False, True), (False, True)
        ):
            with self.subTest(
                variant=variant, prospective=prospective, truncated=truncated
            ):
                self.compare(variant, prospective=prospective, truncated=truncated)

    def test_chunk_edges_and_output_consumers(self):
        for time, truncated, outputs in itertools.product(
            (1, 7, 31, 32, 33, 63, 65, 129),
            (False, True),
            ((True, False), (False, True), (True, True)),
        ):
            with self.subTest(time=time, truncated=truncated, outputs=outputs):
                self.compare(
                    VARIANTS[-2],
                    time=time,
                    truncated=truncated,
                    activate=outputs[0],
                    write_raw=outputs[1],
                )
        for time, truncated in itertools.product((1, 255, 256, 257), (False, True)):
            with self.subTest(time=time, truncated=truncated, chunk="automatic"):
                self.compare(VARIANTS[-2], time=time, truncated=truncated, chunk=None)

    def test_sigmoid_write_nondefault_bound_and_saturation(self):
        for prospective, truncated in itertools.product((False, True), (False, True)):
            for maximum, saturated in ((1.25, False), (3.0, True)):
                with self.subTest(
                    prospective=prospective, truncated=truncated,
                    maximum=maximum, saturated=saturated,
                ):
                    self.compare(
                        dict(VARIANTS[-2], write_link="sigmoid", write_max=maximum),
                        prospective=prospective,
                        truncated=truncated,
                        saturate_write=saturated,
                    )

    def test_clipped_frequency_gradients_and_frozen_base_heads(self):
        variant = dict(write_control=True, frequency_warp="global", max_log_scale=math.log(4))
        for truncated, frozen in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(truncated=truncated, frozen=frozen):
                self.compare(variant, truncated=truncated, clip_warp=True, freeze_base=frozen)

    def test_interior_gamma_edges_with_nonzero_offsets(self):
        for truncated in (False, True):
            with self.subTest(truncated=truncated):
                self.compare({"write_control": True}, truncated=truncated, gamma_edges=True)

    def test_horizons_frozen_heads_and_numerical_edges(self):
        for truncated, mode, horizon in itertools.product(
            (False, True), ("tau", "gamma"), (None, 0.0, 0.08)
        ):
            with self.subTest(truncated=truncated, mode=mode, horizon=horizon):
                self.compare(
                    VARIANTS[-1],
                    truncated=truncated,
                    mode=mode,
                    horizon=horizon,
                    freeze=True,
                )
        for truncated in (False, True):
            variant = {"gamma_control": False, "write_control": True}
            self.compare(variant, tiny=True, truncated=truncated)
            self.compare(variant, collapse=True, truncated=truncated)
            self.compare(VARIANTS[-1], conjugate_cot=True, truncated=truncated)

    @unittest.skipUnless(
        torch.cuda.is_available() and not INTERPRET, "requires native CUDA"
    )
    def test_outer_autocast_and_long_sequence(self):
        for truncated in (False, True):
            self.compare(VARIANTS[-1], truncated=truncated, autocast=True)
            self.compare(VARIANTS[-2], truncated=truncated, time=4097, chunk=None)

    @unittest.skipUnless(
        torch.cuda.is_available() and not INTERPRET, "requires native CUDA"
    )
    def test_encoder_dispatch_and_optimizer(self):
        for mode, source in itertools.product(
            ("full", "tbptt1"), ("layer_input", "layer_output")
        ):
            actual = make_model(
                dict(VARIANTS[-1], input_source=source),
                active=True,
                scan_grad_mode=mode,
                skip_connection=True,
            ).cuda()
            expected = copy.deepcopy(actual)
            optimizers = [
                torch.optim.AdamW(m.parameters(), lr=0.001) for m in (actual, expected)
            ]
            x = torch.randn(2, 65, 3, device="cuda")
            for _ in range(3):
                for model, opt, enabled in zip(
                    (actual, expected), optimizers, ("1", "0")
                ):
                    with mock.patch.dict(os.environ, {"ORGANICS_RQF_FBT": enabled}):
                        opt.zero_grad()
                        model(x).abs().square().mean().backward()
                        opt.step()
                        model.clamp_parameters()
            torch.testing.assert_close(actual(x), expected(x), rtol=3e-3, atol=3e-4)


if __name__ == "__main__":
    unittest.main()
