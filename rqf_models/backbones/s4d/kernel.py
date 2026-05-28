"""S4D diagonal state-space kernel.

Vendored from the reference S4D implementation::

    https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py

Original code is released under the Apache License, Version 2.0
(Copyright (c) 2022 Albert Gu, Karan Goel, Christopher Ré, and the
state-spaces/s4 contributors). A copy of the license is available at
<https://www.apache.org/licenses/LICENSE-2.0>.

Modifications from upstream:

* Removed the ``register`` wrapper and ``_optim`` attribute plumbing.
  Parameters are plain :class:`torch.nn.Parameter` tensors; learning-rate
  groups are configured externally via ``param_group_patterns`` in the
  training loop.
* Replaced the ``einops`` dependency with equivalent ``torch`` ops.
* Exposed ``kernel_init`` switch between S4D-Inv (default) and S4D-Lin
  initializations.
* Added the prospective-input discretization.

The forward math (ZOH diagonal kernel) is unchanged:

    A = -exp(log_A_real) + i * A_imag          # (H, N/2)
    dtA = A * dt                               # (H, N/2)
    K[h, l] = 2 * Re(sum_n C[h,n] * (exp(dtA[h,n]) - 1) / A[h,n]
                      * exp(dtA[h,n] * l))
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class S4DKernel(nn.Module):
    """Diagonal SSM kernel producing a real convolution filter ``K(L)``.

    Args:
        d_model: Number of channels ``H``.
        N: State dimension (must be even; only ``N // 2`` complex poles
            are stored since the filter is forced real-valued).
        dt_min: Minimum of log-uniform ``dt`` initialization.
        dt_max: Maximum of log-uniform ``dt`` initialization.
        kernel_init: ``"diag-inv"`` (S4D-Inv, default) or ``"diag-lin"``
            (S4D-Lin). Both use ``log_A_real = log(0.5)`` and differ only
            in ``A_imag``.
        prospective_input: Use prospective input discretization.
        prospective_tau: Additive prospective lookahead in seconds.
        prospective_tau_multiplier: Extra prospective lookahead in units of
            each channel's initialized/learned ``dt``.
        prospective_initial: Boundary convention for the delayed tap.
            ``"zero"`` uses zero before the first sample, while ``"repeat"``
            repeats the first input sample.
        prospective_feedthrough: Reserved for prospective treatment of the
            D skip path. Currently unsupported.
    """

    def __init__(
        self,
        d_model: int,
        N: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        kernel_init: str = "diag-inv",
        prospective_input: bool = False,
        prospective_tau: float | torch.Tensor | None = 0.0,
        prospective_tau_multiplier: float | torch.Tensor | None = None,
        prospective_initial: str = "zero",
        prospective_feedthrough: bool = False,
    ) -> None:
        super().__init__()
        assert N % 2 == 0, "S4D state dim N must be even"
        if prospective_initial not in {"zero", "repeat"}:
            raise ValueError("prospective_initial must be either 'zero' or 'repeat'")
        if prospective_feedthrough:
            raise NotImplementedError(
                "prospective_feedthrough is not implemented; the D path remains standard"
            )
        self.prospective_input = prospective_input
        self.prospective_tau = prospective_tau
        self.prospective_tau_multiplier = prospective_tau_multiplier
        self.prospective_initial = prospective_initial

        H = d_model
        half = N // 2

        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        C = torch.randn(H, half, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.log_dt = nn.Parameter(log_dt)

        log_A_real = torch.log(0.5 * torch.ones(H, half))
        n_idx = torch.arange(half, dtype=torch.float32)
        if kernel_init == "diag-lin":
            A_imag = math.pi * n_idx
        elif kernel_init == "diag-inv":
            A_imag = (half / math.pi) * (half / (2.0 * n_idx + 1.0) - 1.0)
        else:
            raise ValueError(f"Unknown kernel_init: {kernel_init!r}")
        A_imag = A_imag.unsqueeze(0).expand(H, -1).contiguous()

        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

    def _prospective_tau(self, dt: torch.Tensor) -> torch.Tensor:
        tau = (
            dt.new_tensor(0.0)
            if self.prospective_tau is None
            else torch.as_tensor(self.prospective_tau, dtype=dt.dtype, device=dt.device)
        )
        if tau.ndim == 1 and tau.size(0) == dt.size(0):
            tau = tau.unsqueeze(-1)
        if self.prospective_tau_multiplier is not None:
            multiplier = torch.as_tensor(
                self.prospective_tau_multiplier,
                dtype=dt.dtype,
                device=dt.device,
            )
            dt_aligned = dt.unsqueeze(-1) if dt.ndim == 1 else dt
            tau = tau + multiplier * dt_aligned
        return tau

    def forward(
        self,
        L: int,
        *,
        return_initial_kernel: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Materialize the real kernel ``K`` of shape ``(H, L)``."""
        dt = torch.exp(self.log_dt)  # (H,)
        C = torch.view_as_complex(self.C)  # (H, N/2)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N/2)

        dtA = A * dt.unsqueeze(-1)  # (H, N/2)
        arange_L = torch.arange(L, device=A.device, dtype=dt.dtype)
        powers = torch.exp(dtA.unsqueeze(-1) * arange_L)  # (H, N/2, L)
        dA = torch.exp(dtA)

        if self.prospective_input:
            dB = torch.expm1(dtA) / A
            tau = self._prospective_tau(dt).to(A)
            b_minus = -tau * dA
            b_plus = dB - b_minus

            k_plus = 2 * torch.einsum("hn,hnl->hl", C * b_plus, powers).real
            k_minus = 2 * torch.einsum("hn,hnl->hl", C * b_minus, powers).real
            if L > 1:
                K = torch.cat(
                    [k_plus[..., :1], k_plus[..., 1:] + k_minus[..., :-1]],
                    dim=-1,
                )
            else:
                K = k_plus
        else:
            k_minus = None
            C = C * (dA - 1.0) / A  # (H, N/2)
            K = 2 * torch.einsum("hn,hnl->hl", C, powers).real  # (H, L)

        if return_initial_kernel:
            return K, k_minus
        return K
