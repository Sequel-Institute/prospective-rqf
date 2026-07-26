"""Tests for TBPTT(1) gradients through the prospective (two-tap) S5 scan."""

from __future__ import annotations

import copy
import unittest

import torch

from rqf_models.backbones.s5.layer import S5Layer


def _make_layer(*, scan_grad_mode: str, prospective_input: bool = True) -> S5Layer:
    torch.manual_seed(0)
    return S5Layer(
        d_model=3,
        d_state=4,
        blocks=1,
        activation="gelu",
        conj_sym=False,
        clip_eigs=False,
        use_parallel_scan=True,
        prospective_input=prospective_input,
        prospective_mode="direct",
        prospective_horizon_multiplier=5.0,
        scan_grad_mode=scan_grad_mode,
    )


class S5Tbptt1ProspectiveTest(unittest.TestCase):
    def test_forward_value_matches_full(self) -> None:
        """The rebuild trick must not change forward values."""
        full = _make_layer(scan_grad_mode="full")
        tbptt = copy.deepcopy(full)
        tbptt.scan_grad_mode = "tbptt1"
        full.train()
        tbptt.train()
        x = torch.randn(2, 7, 3)
        y_full = full(x)
        y_tbptt = tbptt(x)
        torch.testing.assert_close(y_tbptt, y_full)

    def test_spatial_only_gradient_cut(self) -> None:
        """A last-step loss must send input gradients only to the last step.

        Under TBPTT(1) the recurrent state is detached step to step and the
        delayed tap u[t-1] is detached, so d loss / d x[t] for t < T-1 must be
        exactly zero, while the same probe under full BPTT is non-zero.
        """
        for mode, expect_past_grad in (("tbptt1", False), ("full", True)):
            layer = _make_layer(scan_grad_mode=mode)
            layer.train()
            x = torch.randn(2, 6, 3, requires_grad=True)
            lambda_bar, b_plus, b_minus = layer._discretize_prospective()
            states = (
                layer._scan_prospective(lambda_bar, b_plus, b_minus, x)
                if mode == "tbptt1"
                else layer._scan(lambda_bar, layer._prospective_drive(b_plus, b_minus, x))
            )
            loss = states[:, -1, :].abs().sum()
            loss.backward()
            past = x.grad[:, :-1, :]
            last = x.grad[:, -1, :]
            self.assertTrue(torch.any(last != 0))
            if expect_past_grad:
                self.assertTrue(torch.any(past != 0))
            else:
                self.assertTrue(torch.all(past == 0))

    def test_one_step_gradients_match_manual_reference(self) -> None:
        """Autograd through the scan rebuild == an explicit detached loop.

        Reference recurrence (the definition of TBPTT(1) with the strict
        delayed-input convention):
            h[t] = lambda_bar * h[t-1].detach()
                   + b_plus @ u[t] + b_minus @ u[t-1].detach()
        Parameter gradients (Lambda, B, log_step) and the input gradient must
        match exactly.
        """
        layer = _make_layer(scan_grad_mode="tbptt1")
        ref = copy.deepcopy(layer)
        layer.train()
        ref.train()
        torch.manual_seed(1)
        x = torch.randn(2, 5, 3)
        x_layer = x.clone().requires_grad_(True)
        x_ref = x.clone().requires_grad_(True)

        lambda_bar, b_plus, b_minus = layer._discretize_prospective()
        states = layer._scan_prospective(lambda_bar, b_plus, b_minus, x_layer)
        loss = states.abs().sum()
        loss.backward()

        lambda_r, b_plus_r, b_minus_r = ref._discretize_prospective()
        xc = x_ref.to(b_plus_r.dtype)
        h = torch.zeros(2, lambda_r.numel(), dtype=lambda_r.dtype)
        prev = torch.zeros(2, 3, dtype=xc.dtype)
        outs = []
        for t in range(x.shape[1]):
            drive = torch.einsum("ph,bh->bp", b_plus_r, xc[:, t]) + torch.einsum(
                "ph,bh->bp", b_minus_r, prev.detach()
            )
            h = lambda_r * h.detach() + drive
            outs.append(h)
            prev = xc[:, t]
        loss_ref = torch.stack(outs, dim=1).abs().sum()
        loss_ref.backward()

        torch.testing.assert_close(loss, loss_ref)
        for (name, p), (_, p_ref) in zip(
            layer.named_parameters(), ref.named_parameters(), strict=True
        ):
            if p.grad is None and p_ref.grad is None:
                continue
            torch.testing.assert_close(p.grad, p_ref.grad, msg=f"grad mismatch: {name}")
        torch.testing.assert_close(x_layer.grad, x_ref.grad)


if __name__ == "__main__":
    unittest.main()
