"""Selective dynamics: independent oracles, configuration, and CPU/MPS parity."""

import copy
import itertools
import os
from pathlib import Path
import unittest

import torch

from rqf_models.backbones.rqf import PureRQFEncoder, SelectiveRQFConfig
from rqf_models.ops.selective_scan import selective_coefficients, reference_layer_scan


VARIANTS = [
    {},
    {"gamma_features": "modulus"},
    {"gamma_link": "log_shift"},
    {"gamma_link": "log_shift", "gamma_features": "modulus"},
    {"gamma_control": False, "write_control": True},
    {"gamma_control": False, "frequency_warp": "global"},
    {"gamma_control": False, "frequency_warp": "grouped", "num_groups": 3},
    {"frequency_warp": "grouped", "num_groups": 3, "write_control": True},
    {
        "frequency_warp": "global",
        "write_control": True,
        "gamma_features": "modulus",
        "gamma_link": "log_shift",
    },
]

CONTROLLER_CASES = []
for gamma, write, warp in itertools.product(
    (False, True), (False, True), ("none", "global", "grouped")
):
    if not gamma and not write and warp == "none":
        continue
    links = ("sigmoid", "log_shift") if gamma else ("sigmoid",)
    features = ("real_imag", "modulus") if gamma else ("real_imag",)
    for link, feature in itertools.product(links, features):
        CONTROLLER_CASES.append(
            dict(
                gamma_control=gamma,
                write_control=write,
                frequency_warp=warp,
                num_groups=3,
                gamma_link=link,
                gamma_features=feature,
            )
        )


def make_model(selective=None, *, active=False, **kwargs):
    torch.manual_seed(120)
    config = dict(
        input_dims=3,
        output_dims=5,
        hidden_dims=5,
        num_layers=2,
        samp_frequency=100.0,
        tau=0.05,
        param_space="log",
        seq_duration=2.0,
        discretization="zoh",
        activation="split_relu",
        clamp_gamma=False,
        prospective_input=True,
        complex_output=True,
        input_proj_bias=False,
        final_activation=False,
        skip_connection=False,
        selective=selective,
    )
    config.update(kwargs)
    model = PureRQFEncoder(**config)
    if active:
        with torch.no_grad():
            for block in model.blocks:
                if block.selector is not None:
                    for p in block.selector.parameters():
                        p.normal_(std=0.3)
    return model


def inputs_for(block, source):
    return block.selector(
        source,
        block.rqf_fwd,
        prospective=block.prospective_input,
        mode=block.prospective_mode,
        prospective_tau=block.prospective_tau,
    )


def oracle(model, x):
    """Separate time-major construction; never calls the selective scan helpers."""
    h = model.input_proj(x).to(model.blocks[0].W.dtype)
    source = h
    for block in model.blocks:
        c, r = block.selector, block.rqf_fwd
        selector_source = source if c.config.input_source == "layer_output" else h
        prev = h.new_zeros(h.shape[0], block.W.shape[0])
        out = []
        for t in range(h.shape[1]):
            s = selector_source[:, t]
            v = torch.cat((s.real, s.imag), dim=-1)
            gamma = r.log_gamma.exp()
            if c.W_gamma is not None:
                f = s.abs() if c.config.gamma_features == "modulus" else v
                offset = f @ c.W_gamma.T
                if c.config.gamma_link == "log_shift":
                    gamma = torch.exp(
                        r.log_gamma + c.config.gamma_max_log_scale * offset.tanh()
                    )
                else:
                    span = r._max_gamma - r._min_gamma
                    eps = torch.finfo(gamma.dtype).eps
                    bias = torch.logit(
                        ((gamma - r._min_gamma) / span).clamp(eps, 1 - eps)
                    )
                    gamma = r._min_gamma + span * torch.sigmoid(bias + offset)
            theta = r.log_theta.exp()
            if c.W_delta is not None:
                delta = c.config.max_log_scale * (v @ c.W_delta.T).tanh()
                theta = theta * delta[:, c.warp_groups].exp()
            gate = 1 if c.W_write is None else 1 + v @ c.W_write.T
            a = (-gamma + 1j * (1 - gamma) * theta) * (r.dt / r.tau)
            A = a.exp()
            small = a.abs() < 1e-4
            phi = torch.where(
                small,
                1 + a / 2 + a.square() / 6 + a**3 / 24,
                torch.expm1(a) / torch.where(small, torch.ones_like(a), a),
            )
            p = gamma * (r.dt / r.tau) * phi
            n = torch.zeros_like(p)
            if block.prospective_input:
                ratio = 1 if block.prospective_mode == "tau" else 1 / gamma
                if block.prospective_tau is not None:
                    ratio = (
                        torch.as_tensor(
                            block.prospective_tau, device=h.device, dtype=r.tau.dtype
                        )
                        / r.tau
                    )
                n = -ratio * gamma * A
                p = p - n
            prior = h[:, t - 1] if t else torch.zeros_like(h[:, t])
            if block.scan_grad_mode == "tbptt1":
                prior, prev = prior.detach(), prev.detach()
            prev = A * prev + gate * (
                p * (h[:, t] @ block.W.T) + n * (prior @ block.W.T)
            )
            out.append(prev)
        raw = torch.stack(out, dim=1)
        act = (
            torch.complex(raw.real.relu(), raw.imag.relu())
            if block.apply_activation
            else raw
        )
        skip = model.skip_connection and h.shape == raw.shape
        source = raw + h if skip else raw
        h = act + h if skip else act
    return h


def assert_model_parity(
    case, left, right, x, *, right_oracle=False, rtol=3e-4, atol=3e-6
):
    device = next(right.parameters()).device
    xl = x.clone().requires_grad_()
    xr = x.to(device).clone().requires_grad_()
    yl = left(xl)
    yr = oracle(right, xr) if right_oracle else right(xr)
    torch.testing.assert_close(yl, yr.cpu(), rtol=rtol, atol=atol)
    cot = torch.randn_like(yl)
    yl.backward(cot)
    yr.backward(cot.to(device))
    torch.testing.assert_close(xl.grad, xr.grad.cpu(), rtol=rtol * 10, atol=atol * 10)
    for (name, pl), (nr, pr) in zip(left.named_parameters(), right.named_parameters()):
        case.assertEqual(name, nr)
        if pl.grad is None:
            case.assertIsNone(pr.grad, name)
        else:
            torch.testing.assert_close(
                pl.grad, pr.grad.cpu(), rtol=rtol * 10, atol=atol * 10, msg=name
            )


class SelectiveReferenceTest(unittest.TestCase):
    def test_float64_recurrence_and_detach_semantics(self):
        for variant, prospective, mode in itertools.product(
            CONTROLLER_CASES, (False, True), ("full", "tbptt1")
        ):
            with self.subTest(variant=variant, prospective=prospective, mode=mode):
                net = make_model(
                    variant,
                    active=True,
                    prospective_input=prospective,
                    scan_grad_mode=mode,
                    skip_connection=True,
                ).double()
                # Module.double() keeps complex parameters at complex64.
                net._apply(lambda t: t.to(torch.complex128) if t.is_complex() else t)
                assert_model_parity(
                    self,
                    net,
                    copy.deepcopy(net),
                    torch.randn(2, 7, 3, dtype=torch.float64),
                    right_oracle=True,
                    rtol=1e-11,
                    atol=1e-12,
                )

    def test_all_options_against_independent_oracle(self):
        for variant, source, prospective, mode in itertools.product(
            CONTROLLER_CASES,
            ("layer_input", "layer_output"),
            (False, True),
            ("full", "tbptt1"),
        ):
            with self.subTest(
                variant=variant, source=source, prospective=prospective, mode=mode
            ):
                net = make_model(
                    dict(variant, input_source=source),
                    active=True,
                    prospective_input=prospective,
                    scan_grad_mode=mode,
                    skip_connection=True,
                )
                assert_model_parity(
                    self,
                    net,
                    copy.deepcopy(net),
                    torch.randn(2, 7, 3),
                    right_oracle=True,
                )

    def test_horizons_and_width_changing_skips(self):
        for mode, horizon, truncated in itertools.product(
            ("tau", "gamma"), (None, 0.0, 0.07), (False, True)
        ):
            with self.subTest(mode=mode, horizon=horizon, truncated=truncated):
                net = make_model(
                    VARIANTS[-1],
                    active=True,
                    prospective_mode=mode,
                    prospective_tau=horizon,
                    hidden_dims=[5, 7],
                    skip_connection=True,
                    scan_grad_mode="tbptt1" if truncated else "full",
                )
                assert_model_parity(
                    self,
                    net,
                    copy.deepcopy(net),
                    torch.randn(2, 9, 3),
                    right_oracle=True,
                )

    def test_zero_heads_preserve_static_model_and_rng(self):
        for source, prospective, mode, skip in itertools.product(
            ("layer_input", "layer_output"),
            (False, True),
            ("full", "tbptt1"),
            (False, True),
        ):
            kwargs = dict(
                prospective_input=prospective, scan_grad_mode=mode, skip_connection=skip
            )
            static = make_model(**kwargs)
            selected = make_model(dict(VARIANTS[-2], input_source=source), **kwargs)
            for key, value in static.state_dict().items():
                torch.testing.assert_close(
                    value, selected.state_dict()[key], rtol=0, atol=0
                )
            x = torch.randn(2, 9, 3)
            torch.testing.assert_close(static(x), selected(x), rtol=3e-5, atol=3e-6)

    def test_tbptt_cuts_history_but_preserves_weight_and_selector_gradients(self):
        net = make_model(
            VARIANTS[-2], active=True, num_layers=1, scan_grad_mode="tbptt1"
        )
        reference = copy.deepcopy(net)
        x = torch.randn(2, 7, 3, requires_grad=True)
        y = net(x)
        y[:, -1].abs().square().sum().backward()
        torch.testing.assert_close(
            x.grad[:, :-1], torch.zeros_like(x.grad[:, :-1]), rtol=0, atol=0
        )
        for p in net.blocks[0].selector.parameters():
            self.assertGreater(float(p.grad.abs().sum()), 0)
        xr = x.detach().clone().requires_grad_()
        oracle(reference, xr)[:, -1].abs().square().sum().backward()
        torch.testing.assert_close(net.blocks[0].W.grad, reference.blocks[0].W.grad)
        full = copy.deepcopy(net)
        full.blocks[0].scan_grad_mode = "full"
        torch.testing.assert_close(y, full(x.detach()), rtol=0, atol=0)

    def test_raw_selector_gradient_survives_negative_relu(self):
        source = torch.full((5, 1, 3), -1 - 1j, requires_grad=True)
        for mode in ("layer_output", "layer_input"):
            block = make_model(dict(input_source=mode), active=True).blocks[1]
            act = torch.complex(source.real.relu(), source.imag.relu())
            selected = source if mode == "layer_output" else act
            p = selective_coefficients(inputs_for(block, selected))[1]
            grad = torch.autograd.grad(p.real.sum(), source, retain_graph=True)[0]
            if mode == "layer_output":
                self.assertGreater(float(grad.abs().sum()), 0)
            else:
                self.assertEqual(float(grad.abs().sum()), 0)

    def test_bounds_groups_and_zero_modulus_gradient(self):
        for features in ("real_imag", "modulus"):
            block = make_model(
                dict(VARIANTS[-2], gamma_features=features, gamma_link="log_shift"),
                active=True,
            ).blocks[0]
            r, c = block.rqf_fwd, block.selector
            with torch.no_grad():
                r.log_gamma.fill_(1.0)
            r.clamp_parameters()
            torch.testing.assert_close(
                r.gamma,
                r._max_gamma * torch.exp(torch.tensor(-c.config.gamma_max_log_scale)),
            )
            group_order = c.warp_groups[r.log_theta.argsort(stable=True)]
            self.assertTrue(bool((group_order[1:] >= group_order[:-1]).all()))
            s = torch.zeros(5, 1, 3, dtype=torch.complex64, requires_grad=True)
            inputs = inputs_for(block, s)
            gamma = torch.exp(
                inputs.bias[:, None, None]
                + c.config.gamma_max_log_scale * inputs.gamma_logits.tanh()
            )
            gamma.sum().backward()
            self.assertTrue(bool(torch.isfinite(s.grad).all()))
            if features == "modulus":
                self.assertEqual(float(s.grad.abs().sum()), 0.0)

    def test_signed_gate_current_interval_and_zero_gate_derivative(self):
        block = make_model(
            {"gamma_control": False, "write_control": True}, num_layers=1
        ).blocks[0]
        with torch.no_grad():
            block.selector.W_write.zero_()
            block.selector.W_write[:, 0] = -1
        s = torch.ones(5, 1, 3, dtype=torch.complex64)
        s[0, 0] = torch.tensor([0.0, 1.0, 2.0])
        inputs = inputs_for(block, s)
        A, p, n = selective_coefficients(inputs)
        self.assertEqual(float(p[..., 1].abs().sum()), 0.0)
        torch.testing.assert_close(p[..., 2], -p[..., 0])
        torch.testing.assert_close(n[..., 2], -n[..., 0])
        torch.testing.assert_close(A[..., 2], A[..., 0])
        z = torch.randn(5, 1, 3, dtype=torch.complex64)
        weight = torch.eye(5, dtype=torch.complex64)
        y, _ = reference_layer_scan(
            z, weight, inputs, truncated=False, activate=False, write_raw=True
        )
        torch.testing.assert_close(y[..., 1], A[..., 1] * y[..., 0])
        grad = torch.autograd.grad(y[..., 1].real.sum(), block.selector.W_write)[0]
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_float64_coefficient_and_controller_gradcheck(self):
        for variant, mode in itertools.product(CONTROLLER_CASES, ("tau", "gamma")):
            block = (
                make_model(variant, active=True, num_layers=1, prospective_mode=mode)
                .blocks[0]
                .double()
            )
            source = torch.randn(5, 1, 2, dtype=torch.complex128, requires_grad=True)
            data = inputs_for(block, source)
            indices = [i for i in (0, 1, 2, 3, 4) if data[i].numel()]
            leaves = tuple(data[i].detach().requires_grad_() for i in indices)

            def evaluate(*values):
                fields = list(data)
                for i, value in zip(indices, values):
                    fields[i] = value
                return selective_coefficients(type(data)(*fields))

            self.assertTrue(
                torch.autograd.gradcheck(evaluate, leaves, atol=2e-6, rtol=2e-4)
            )
            self.assertTrue(
                torch.autograd.gradcheck(
                    lambda s: selective_coefficients(inputs_for(block, s)),
                    (source,),
                    atol=2e-6,
                    rtol=2e-4,
                )
            )

    def test_small_pole_limit(self):
        block = make_model({}, num_layers=1).blocks[0].double()
        data = inputs_for(block, torch.zeros(5, 1, 2, dtype=torch.complex128))
        # Independent gamma/theta leaves exercise exprel at exactly zero.
        data = data._replace(
            options=(False, False, False, False, 0.2, 0.2, True, False),
            bias=torch.zeros(5, dtype=torch.float64, requires_grad=True),
            theta=torch.zeros(5, dtype=torch.float64, requires_grad=True),
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda g, th: selective_coefficients(data._replace(bias=g, theta=th)),
                (data.bias, data.theta),
                eps=1e-7,
                atol=1e-7,
            )
        )


class SelectiveIntegrationTest(unittest.TestCase):
    def test_validation(self):
        for options in (
            {"num_groups": 0},
            {"gamma_control": False},
            {"gamma_link": "bad"},
            {"gamma_features": "bad"},
            {"max_log_scale": float("nan")},
            {"gamma_control": False, "gamma_features": "modulus"},
            {"input_source": "previous_time"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SelectiveRQFConfig(**options)
        for kwargs in (
            {"activation": "split_tanh"},
            {"param_space": "sigmoid", "seq_duration": None},
            {"learn_tau": True},
            {"complex_weights": False},
            {"clamp_gamma": True},
            {"discretization": "euler"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_model({}, **kwargs)
        for options in (
            {"frequency_warp": "grouped", "num_groups": 6},
            {"gamma_link": "log_shift", "gamma_max_log_scale": 30.0},
        ):
            with self.assertRaises(ValueError):
                make_model(options)

    def test_config_optimizer_and_static_build(self):
        from hydra import compose, initialize_config_dir
        from train import build_model, build_optimizer, validate_cfg

        config_dir = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            for recipe in ("pure_rqf_paper", "pure_rqf_selective"):
                cfg = compose(config_name="config", overrides=["model=" + recipe])
                if recipe == "pure_rqf_selective":
                    cfg.model.selective.write_control = True
                    cfg.model.selective.frequency_warp = "grouped"
                validate_cfg(cfg)
                model = build_model(cfg)
                opt = build_optimizer(cfg, model)
                for name, p in model.named_parameters():
                    group = next(
                        (
                            g
                            for g in opt.param_groups
                            if any(p is q for q in g["params"])
                        ),
                        None,
                    )
                    if not p.requires_grad:
                        self.assertIsNone(group)
                    elif "W_gamma" in name or "W_delta" in name:
                        self.assertEqual(group["weight_decay"], 0.0)
                        self.assertEqual(
                            group["lr"],
                            cfg.training.learning_rate * cfg.training.rqf_lr_factor,
                        )
                    elif "W_write" in name:
                        self.assertEqual(
                            group["weight_decay"], cfg.training.weight_decay
                        )
                        self.assertEqual(group["lr"], cfg.training.learning_rate)

    def test_checkpoint_resume_and_settings(self):
        net = make_model(VARIANTS[-1], active=True)
        opt = torch.optim.AdamW(net.parameters(), lr=0.001)
        x = torch.randn(2, 5, 3)

        def step(model, optimizer):
            optimizer.zero_grad()
            model(x).abs().square().mean().backward()
            optimizer.step()
            model.clamp_parameters()

        step(net, opt)
        checkpoint = copy.deepcopy(
            {"model": net.state_dict(), "optimizer": opt.state_dict()}
        )
        restored = make_model(VARIANTS[-1])
        restored.load_state_dict(checkpoint["model"])
        opt2 = torch.optim.AdamW(restored.parameters(), lr=0.001)
        opt2.load_state_dict(checkpoint["optimizer"])
        step(net, opt)
        step(restored, opt2)
        torch.testing.assert_close(net(x), restored(x), rtol=0, atol=0)
        changed = make_model(dict(VARIANTS[-1], input_source="layer_input"))
        with self.assertRaisesRegex(ValueError, "settings"):
            changed.load_state_dict(checkpoint["model"])
        legacy = make_model()
        legacy.load_state_dict(copy.deepcopy(legacy.state_dict()), strict=True)
        self.assertFalse(any("selector" in k for k in legacy.state_dict()))


@unittest.skipUnless(torch.backends.mps.is_available(), "requires MPS")
class SelectiveMPSTest(unittest.TestCase):
    def setUp(self):
        self.assertNotEqual(
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "1",
            "MPS parity tests require CPU fallback disabled",
        )

    def test_options_and_gradients(self):
        for variant, source, prospective, mode in itertools.product(
            CONTROLLER_CASES,
            ("layer_output", "layer_input"),
            (False, True),
            ("full", "tbptt1"),
        ):
            with self.subTest(
                variant=variant, source=source, prospective=prospective, mode=mode
            ):
                cpu = make_model(
                    dict(variant, input_source=source),
                    active=True,
                    prospective_input=prospective,
                    scan_grad_mode=mode,
                    skip_connection=True,
                )
                assert_model_parity(
                    self, cpu, copy.deepcopy(cpu).to("mps"), torch.randn(2, 7, 3)
                )

    def test_optimizer_and_checkpoint(self):
        cpu = make_model(VARIANTS[-1], active=True, skip_connection=True)
        mps = copy.deepcopy(cpu).to("mps")
        optimizers = [
            torch.optim.AdamW(m.parameters(), lr=0.001, foreach=False)
            for m in (cpu, mps)
        ]
        x = torch.randn(2, 7, 3)
        for _ in range(3):
            for model, opt in zip((cpu, mps), optimizers):
                opt.zero_grad()
                model(
                    x.to(next(model.parameters()).device)
                ).abs().square().mean().backward()
                opt.step()
                model.clamp_parameters()
        torch.testing.assert_close(cpu(x), mps(x.to("mps")).cpu(), rtol=5e-4, atol=3e-5)
        restored = make_model(VARIANTS[-1], skip_connection=True).to("mps")
        restored.load_state_dict(copy.deepcopy(mps.state_dict()))
        torch.testing.assert_close(
            restored(x.to("mps")), mps(x.to("mps")), rtol=0, atol=0
        )

    def test_complex_gradient_clipping_matches_cpu(self):
        from train import clip_model_grad_norm_

        cpu = make_model(VARIANTS[-1], active=True)
        mps = copy.deepcopy(cpu).to("mps")
        for pc, pm in zip(cpu.parameters(), mps.parameters()):
            if pc.requires_grad:
                pc.grad = torch.randn_like(pc)
                pm.grad = pc.grad.to("mps")
        expected = torch.nn.utils.clip_grad_norm_(cpu.parameters(), 0.1)
        actual = clip_model_grad_norm_(mps, 0.1)
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
        for pc, pm in zip(cpu.parameters(), mps.parameters()):
            if pc.grad is not None:
                torch.testing.assert_close(pm.grad.cpu(), pc.grad, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
