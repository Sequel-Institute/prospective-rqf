"""Numerical parity tests between the scan kernel backends.

The chunked two-level scan and the fused two-tap prospective scan must be
numerically interchangeable (up to float tolerance) with the legacy
sequential kernels and with a plain PyTorch reference. All tests require
CUDA + Triton and are skipped otherwise.
"""

from __future__ import annotations

import unittest

import torch

CUDA = torch.cuda.is_available()
if CUDA:
    from rqf_models.ops.chunked_scan_kernel import (
        _triton_chunked_scan,
        _triton_chunked_scan_prospective,
    )
    from rqf_models.ops.scan import _sequential_scan_cpu
    from rqf_models.ops.triton_scan_kernel import _triton_sequential_scan


def _coeffs(num_filters: int, device: str, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mag = 0.8 + 0.19 * torch.rand(num_filters, generator=g)
    phase = 2 * torch.pi * torch.rand(num_filters, generator=g)
    C1 = (mag * torch.exp(1j * phase)).to(torch.cfloat).to(device)
    C2 = (0.1 * torch.randn(num_filters, 2, generator=g)).to(device)
    return C1, torch.complex(C2[:, 0], C2[:, 1])


@unittest.skipUnless(CUDA, "requires CUDA + Triton")
class ChunkedScanParityTest(unittest.TestCase):
    def _run_pair(self, batch, time, filters, seed):
        device = "cuda"
        C1, C2 = _coeffs(filters, device, seed)
        u = torch.randn(batch, time, filters, dtype=torch.cfloat, device=device)

        args_a = [t.clone().requires_grad_(True) for t in (C1, C2, u)]
        args_b = [t.clone().requires_grad_(True) for t in (C1, C2, u)]
        y_chunked = _triton_chunked_scan(*args_a)
        y_seq = _triton_sequential_scan(*args_b)
        torch.testing.assert_close(y_chunked, y_seq, rtol=2e-4, atol=2e-4)

        grad = torch.randn_like(y_chunked)
        y_chunked.backward(grad)
        y_seq.backward(grad)
        for a, b, name in zip(args_a, args_b, ("C1", "C2", "u"), strict=True):
            torch.testing.assert_close(
                a.grad, b.grad, rtol=5e-3, atol=5e-4, msg=f"grad mismatch: {name}"
            )

    def test_forward_backward_parity_long(self):
        self._run_pair(batch=4, time=16000, filters=32, seed=0)

    def test_forward_backward_parity_odd_length(self):
        self._run_pair(batch=2, time=1023, filters=8, seed=1)

    def test_matches_pytorch_reference(self):
        device = "cuda"
        C1, C2 = _coeffs(16, device, seed=2)
        u = torch.randn(2, 777, 16, dtype=torch.cfloat, device=device)
        y = _triton_chunked_scan(C1, C2, u)
        y_ref = _sequential_scan_cpu(C1.cpu(), C2.cpu(), u.cpu(), None)
        torch.testing.assert_close(y.cpu(), y_ref, rtol=2e-4, atol=2e-4)


@unittest.skipUnless(CUDA, "requires CUDA + Triton")
class FusedProspectiveParityTest(unittest.TestCase):
    """The fused two-tap kernel must equal drive materialisation + scan."""

    def test_forward_backward_parity(self):
        device = "cuda"
        filters = 32
        C1, _ = _coeffs(filters, device, seed=3)
        g = torch.Generator(device="cpu").manual_seed(3)
        tap0 = torch.complex(
            0.5 * torch.randn(filters, generator=g), 0.5 * torch.randn(filters, generator=g)
        ).to(device)
        tap1 = torch.complex(
            0.5 * torch.randn(filters, generator=g), 0.5 * torch.randn(filters, generator=g)
        ).to(device)
        u = torch.randn(3, 4096, filters, dtype=torch.cfloat, device=device)

        args_f = [t.clone().requires_grad_(True) for t in (C1, tap0, tap1, u)]
        args_r = [t.clone().requires_grad_(True) for t in (C1, tap0, tap1, u)]

        y_fused = _triton_chunked_scan_prospective(*args_f)

        # Reference: materialise drive[n] = tap0*u[n] + tap1*u[n-1], scan C2=1
        c1_r, tap0_r, tap1_r, u_r = args_r
        drive = tap0_r * u_r
        drive[:, 1:, :] = drive[:, 1:, :] + tap1_r * u_r[:, :-1, :]
        y_ref = _triton_chunked_scan(c1_r, torch.ones_like(c1_r), drive)

        torch.testing.assert_close(y_fused, y_ref, rtol=2e-4, atol=2e-4)

        grad = torch.randn_like(y_fused)
        y_fused.backward(grad)
        y_ref.backward(grad)
        for a, b, name in zip(args_f, args_r, ("C1", "tap0", "tap1", "u"), strict=True):
            torch.testing.assert_close(
                a.grad, b.grad, rtol=5e-3, atol=5e-4, msg=f"grad mismatch: {name}"
            )


@unittest.skipUnless(CUDA, "requires CUDA + Triton")
class FBTEncoderParityTest(unittest.TestCase):
    """The channel-outermost fused encoder path must match the standard path."""

    def _run_encoder_pair(self, activation: str, skip: bool, in_dims: int):
        import os

        from rqf_models.backbones.rqf.pure import PureRQFEncoder

        torch.manual_seed(0)
        enc = PureRQFEncoder(
            input_dims=in_dims,
            output_dims=16,
            hidden_dims=16,
            num_layers=3,
            samp_frequency=1000.0,
            min_freq=1.0,
            max_freq=400.0,
            tau=0.001,
            prospective_input=True,
            activation=activation,
            skip_connection=skip,
        ).cuda()
        x = torch.randn(2, 700, in_dims, device="cuda")

        x_fbt = x.clone().requires_grad_(True)
        x_std = x.clone().requires_grad_(True)

        os.environ["ORGANICS_RQF_FBT"] = "1"
        self.assertTrue(enc._fbt_enabled(x_fbt), "FBT path should be eligible")
        y_fbt = enc(x_fbt)
        os.environ["ORGANICS_RQF_FBT"] = "0"
        y_std = enc(x_std)
        os.environ["ORGANICS_RQF_FBT"] = "1"

        torch.testing.assert_close(y_fbt, y_std, rtol=2e-4, atol=2e-4)

        grad = torch.randn_like(y_fbt)
        params = [p for p in enc.parameters() if p.requires_grad]
        gp_fbt = torch.autograd.grad(y_fbt, [x_fbt, *params], grad, allow_unused=True)
        gp_std = torch.autograd.grad(y_std, [x_std, *params], grad, allow_unused=True)
        for a, b in zip(gp_fbt, gp_std, strict=True):
            if a is None and b is None:
                continue
            torch.testing.assert_close(a, b, rtol=5e-3, atol=5e-4)

    def test_split_relu(self):
        self._run_encoder_pair("split_relu", skip=False, in_dims=1)

    def test_split_tanh_with_skip(self):
        self._run_encoder_pair("split_tanh", skip=True, in_dims=1)

    def test_multichannel_input(self):
        self._run_encoder_pair("split_relu", skip=False, in_dims=13)


@unittest.skipUnless(CUDA, "requires CUDA + Triton")
class DispatchRoutingTest(unittest.TestCase):
    """sequential_scan/parallel_scan must produce backend-independent values."""

    def test_dispatch_parity_real_and_complex(self):
        from rqf_models.ops import scan

        device = "cuda"
        C1, C2 = _coeffs(8, device, seed=4)
        for is_complex in (False, True):
            u = torch.randn(2, 500, 8, device=device)
            if is_complex:
                u = u.to(torch.cfloat)
            y_seq = scan.sequential_scan(C1, C2, u)
            y_par = scan.parallel_scan(C1, C2, u)
            y_ref = _sequential_scan_cpu(
                C1.cpu(), C2.cpu(), u.cpu().to(torch.cfloat), None
            )
            torch.testing.assert_close(y_seq.cpu(), y_ref, rtol=2e-4, atol=2e-4)
            torch.testing.assert_close(y_par.cpu(), y_ref, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
