"""Initialization helpers for the PyTorch S5 port.

The routines mirror the HiPPO/DPLR setup used by the original JAX S5
implementation closely enough to make our layer topology comparable while
remaining self-contained in PyTorch.
"""

from __future__ import annotations

import math

import torch


def make_hippo(n: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Create the HiPPO-LegS matrix used by S4/S5 initializers."""
    idx = torch.arange(n, dtype=torch.float64, device=device)
    p = torch.sqrt(1.0 + 2.0 * idx)
    a = p[:, None] * p[None, :]
    return -(torch.tril(a) - torch.diag(idx))


def make_nplr_hippo(
    n: int, *, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the normal-plus-low-rank HiPPO components."""
    hippo = make_hippo(n, device=device)
    p = torch.sqrt(torch.arange(n, dtype=torch.float64, device=device) + 0.5)
    b = torch.sqrt(2.0 * torch.arange(n, dtype=torch.float64, device=device) + 1.0)
    return hippo, p, b


def make_dplr_hippo(
    n: int, *, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagonalize the normal HiPPO approximation.

    Returns:
        ``Lambda`` complex eigenvalues, ``V`` eigenvectors, and ``Vinv``.
    """
    a, p, _ = make_nplr_hippo(n, device=device)
    normal = a + p[:, None] * p[None, :]
    lambda_real = torch.diagonal(normal).mean() * torch.ones(n, dtype=torch.float64, device=device)
    lambda_imag, v = torch.linalg.eigh(normal.to(torch.complex128) * (-1j))
    lambda_c = torch.complex(lambda_real, lambda_imag.real)
    vinv = v.conj().transpose(-2, -1)
    return lambda_c.to(torch.complex64), v.to(torch.complex64), vinv.to(torch.complex64)


def make_s5_dplr_init(
    d_state: int,
    blocks: int,
    *,
    conj_sym: bool,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build block-diagonal S5 initialization tensors."""
    if d_state <= 0:
        raise ValueError("d_state must be positive")
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    if d_state % blocks != 0:
        raise ValueError(f"d_state={d_state} must be divisible by blocks={blocks}")

    block_size = d_state // blocks
    lambda_block, v_block, vinv_block = make_dplr_hippo(block_size, device=device)

    if conj_sym:
        if block_size % 2 != 0:
            raise ValueError(
                "conj_sym=True requires each S5 block to have an even size; "
                f"got d_state={d_state}, blocks={blocks}, block_size={block_size}"
            )
        active_block_size = block_size // 2
    else:
        active_block_size = block_size

    lambda_block = lambda_block[:active_block_size]
    v_block = v_block[:, :active_block_size]
    vinv_block = vinv_block[:active_block_size, :]

    lambdas = lambda_block.repeat(blocks)
    v = torch.block_diag(*([v_block] * blocks))
    vinv = torch.block_diag(*([vinv_block] * blocks))
    local_p = d_state if conj_sym else lambdas.numel()
    return lambdas, v, vinv, local_p


def lecun_std(fan_in: int) -> float:
    """Standard deviation for a LeCun-normal style initializer."""
    if fan_in <= 0:
        raise ValueError("fan_in must be positive")
    return 1.0 / math.sqrt(fan_in)
