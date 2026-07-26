"""Two-level chunked complex scan for the diagonal RQF recurrence.

Implements x[n] = C1*x[n-1] + C2*u[n] for a time-invariant diagonal pole with a
chunked associative scan: each chunk runs an in-block Blelloch scan (log-depth
latency) under a zero entry state, then a serial carry across chunks stitches the
boundary state. One program per (batch, filter) keeps C1/C2 as broadcast scalars
and fuses the C2*u drive; the backward runs the same structure in reverse time
and accumulates the coefficient gradients in-kernel.

Complex tensors are laid out with time innermost and real/imag interleaved (via
``torch.view_as_real`` of a ``(batch, filters, time)`` tensor) so the per-chunk
loads are coalesced and no separate real/imag copies are materialised. This is
the fast CUDA path for the scalar (no input-projection matrix) scan, replacing
the O(T) sequential recurrence; see :mod:`rqf_models.ops.scan` for the dispatch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

# Timesteps per chunk. Bounds per-block SRAM and the associative-scan tree depth.
_CHUNK = 2048


def _pick_chunk(filters: int, batch: int) -> int:
    """Chunk size gated on the launch grid size.

    The scan is occupancy/instruction-bound rather than DRAM-bound, so a smaller
    chunk (more programs) raises occupancy once the grid is already large. For
    small grids the larger chunk keeps the serial inter-chunk carry short. Values
    stay powers of two (``tl.arange(0, CHUNK)``).
    """
    programs = filters * batch
    if programs >= 256:
        return 512
    if programs >= 64:
        return 1024
    return _CHUNK


@triton.jit
def _complex_mul(ar, ai, br, bi):  # pragma: no cover
    """Complex product (ar+i*ai)*(br+i*bi)."""
    return ar * br - ai * bi, ar * bi + ai * br


@triton.jit
def _affine_combine(xl_r, xl_i, fl_r, fl_i, xr_r, xr_i, fr_r, fr_i):  # pragma: no cover
    """Associative operator for x[n]=f*x[n-1]+x: compose affine maps (value, gate)."""
    x_r, x_i = _complex_mul(xl_r, xl_i, fr_r, fr_i)
    x_r += xr_r
    x_i += xr_i
    f_r, f_i = _complex_mul(fl_r, fl_i, fr_r, fr_i)
    return x_r, x_i, f_r, f_i


@triton.jit
def _chunked_scan_kernel(  # pragma: no cover
    c1_ptr,
    c2_ptr,
    u_ptr,
    states_ptr,
    seqlen,
    CHUNK: tl.constexpr,
):
    """Forward two-level chunked scan; interleaved real/imag layout, one program
    per (batch, filter). Broadcasts C1/C2 and fuses the C2*u drive."""
    bsz = tl.program_id(axis=0)
    chan = tl.program_id(axis=1)
    sid = tl.num_programs(axis=1) * bsz + chan
    idx = tl.arange(0, CHUNK)

    c1_r = tl.load(c1_ptr + chan * 2)
    c1_i = tl.load(c1_ptr + chan * 2 + 1)
    c2_r = tl.load(c2_ptr + chan * 2)
    c2_i = tl.load(c2_ptr + chan * 2 + 1)

    carry_r = tl.zeros((), dtype=tl.float32)
    carry_i = tl.zeros((), dtype=tl.float32)
    nchunks = tl.cdiv(seqlen, CHUNK)

    for chunk in range(0, nchunks):
        t = chunk * CHUNK + idx
        mask = t < seqlen
        off = (sid * seqlen + t) * 2

        u_r = tl.load(u_ptr + off, mask=mask, other=0.0)
        u_i = tl.load(u_ptr + off + 1, mask=mask, other=0.0)
        # drive b = C2 * u (fused)
        b_r = c2_r * u_r - c2_i * u_i
        b_i = c2_r * u_i + c2_i * u_r
        # gate f = C1 broadcast; identity (1+0j) beyond the sequence end
        f_r0 = tl.where(mask, c1_r, 1.0)
        f_i0 = tl.where(mask, c1_i, 0.0)

        x_r, x_i, g_r, g_i = tl.associative_scan(
            (b_r, b_i, f_r0, f_i0), axis=0, combine_fn=_affine_combine
        )

        last = idx == CHUNK - 1
        block_x_r = tl.sum(tl.where(last, x_r, 0.0), axis=0)
        block_x_i = tl.sum(tl.where(last, x_i, 0.0), axis=0)
        block_g_r = tl.sum(tl.where(last, g_r, 0.0), axis=0)
        block_g_i = tl.sum(tl.where(last, g_i, 0.0), axis=0)

        # apply the entry carry: out[t] = x[t] + gate_prefix[t] * carry
        car_r, car_i = _complex_mul(g_r, g_i, carry_r, carry_i)
        out_r = x_r + car_r
        out_i = x_i + car_i
        tl.store(states_ptr + off, out_r, mask=mask)
        tl.store(states_ptr + off + 1, out_i, mask=mask)

        # advance carry: carry = block_x + block_gate * carry
        nc_r, nc_i = _complex_mul(block_g_r, block_g_i, carry_r, carry_i)
        carry_r = block_x_r + nc_r
        carry_i = block_x_i + nc_i


@triton.jit
def _chunked_scan_backward_kernel(  # pragma: no cover
    c1_ptr,
    c2_ptr,
    u_ptr,
    states_ptr,
    grad_out_ptr,
    grad_u_ptr,
    grad_c1_ptr,
    grad_c2_ptr,
    seqlen,
    CHUNK: tl.constexpr,
):
    """Chunked adjoint: g[t] = grad_out[t] + conj(C1) g[t+1] via a reverse chunked
    scan, then grad_u = conj(C2) g and the in-kernel reductions
    grad_C1 = sum_t conj(x[t-1]) g[t], grad_C2 = sum_t conj(u[t]) g[t]."""
    bsz = tl.program_id(axis=0)
    chan = tl.program_id(axis=1)
    sid = tl.num_programs(axis=1) * bsz + chan
    idx = tl.arange(0, CHUNK)

    c1_r = tl.load(c1_ptr + chan * 2)
    c1_i = tl.load(c1_ptr + chan * 2 + 1)
    c2_r = tl.load(c2_ptr + chan * 2)
    c2_i = tl.load(c2_ptr + chan * 2 + 1)

    carry_r = tl.zeros((), dtype=tl.float32)
    carry_i = tl.zeros((), dtype=tl.float32)
    gc1_r = tl.zeros((), dtype=tl.float32)
    gc1_i = tl.zeros((), dtype=tl.float32)
    gc2_r = tl.zeros((), dtype=tl.float32)
    gc2_i = tl.zeros((), dtype=tl.float32)
    nchunks = tl.cdiv(seqlen, CHUNK)

    for chunk in range(0, nchunks):
        rchunk = nchunks - 1 - chunk
        t = rchunk * CHUNK + idx
        mask = t < seqlen
        off = (sid * seqlen + t) * 2

        go_r = tl.load(grad_out_ptr + off, mask=mask, other=0.0)
        go_i = tl.load(grad_out_ptr + off + 1, mask=mask, other=0.0)
        # reverse gate = conj(C1) inside the sequence, identity at the boundary t+1==seqlen
        sf_r = tl.where(t < seqlen - 1, c1_r, 1.0)
        sf_i = tl.where(t < seqlen - 1, -c1_i, 0.0)

        g_r, g_i, gg_r, gg_i = tl.associative_scan(
            (go_r, go_i, sf_r, sf_i), axis=0, combine_fn=_affine_combine, reverse=True
        )
        first = idx == 0
        block_g_r = tl.sum(tl.where(first, g_r, 0.0), axis=0)
        block_g_i = tl.sum(tl.where(first, g_i, 0.0), axis=0)
        block_gg_r = tl.sum(tl.where(first, gg_r, 0.0), axis=0)
        block_gg_i = tl.sum(tl.where(first, gg_i, 0.0), axis=0)

        car_r, car_i = _complex_mul(gg_r, gg_i, carry_r, carry_i)
        g_r = g_r + car_r
        g_i = g_i + car_i

        # grad_u[t] = conj(C2) * g[t]
        gu_r = c2_r * g_r + c2_i * g_i
        gu_i = c2_r * g_i - c2_i * g_r
        tl.store(grad_u_ptr + off, gu_r, mask=mask)
        tl.store(grad_u_ptr + off + 1, gu_i, mask=mask)

        # grad_C2 += sum_t conj(u[t]) * g[t]
        u_r = tl.load(u_ptr + off, mask=mask, other=0.0)
        u_i = tl.load(u_ptr + off + 1, mask=mask, other=0.0)
        gc2_r += tl.sum(tl.where(mask, u_r * g_r + u_i * g_i, 0.0), axis=0)
        gc2_i += tl.sum(tl.where(mask, u_r * g_i - u_i * g_r, 0.0), axis=0)

        # grad_C1 += sum_t conj(x[t-1]) * g[t]  (x[-1] = 0)
        tprev = t - 1
        pmask = (tprev >= 0) & (tprev < seqlen) & mask
        offp = (sid * seqlen + tprev) * 2
        xr = tl.load(states_ptr + offp, mask=pmask, other=0.0)
        xi = tl.load(states_ptr + offp + 1, mask=pmask, other=0.0)
        gc1_r += tl.sum(tl.where(pmask, xr * g_r + xi * g_i, 0.0), axis=0)
        gc1_i += tl.sum(tl.where(pmask, xr * g_i - xi * g_r, 0.0), axis=0)

        nc_r, nc_i = _complex_mul(block_gg_r, block_gg_i, carry_r, carry_i)
        carry_r = block_g_r + nc_r
        carry_i = block_g_i + nc_i

    tl.atomic_add(grad_c1_ptr + chan * 2, gc1_r)
    tl.atomic_add(grad_c1_ptr + chan * 2 + 1, gc1_i)
    tl.atomic_add(grad_c2_ptr + chan * 2, gc2_r)
    tl.atomic_add(grad_c2_ptr + chan * 2 + 1, gc2_i)


def _to_time_innermost(u: torch.Tensor) -> torch.Tensor:
    """(batch, time, filters) complex -> (batch, filters, time) contiguous."""
    return u.permute(0, 2, 1).contiguous()


def _chunked_scan_forward_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Raw forward pass for the chunked scan (no autograd); returns (batch, time, filters)."""
    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)
    if not u.is_complex():
        u = u.to(torch.cfloat)

    batch, _, filters = u.shape
    u_bft = _to_time_innermost(u)
    time = u_bft.shape[2]
    c1_real = torch.view_as_real(C1.contiguous()).contiguous()
    c2_real = torch.view_as_real(C2.contiguous()).contiguous()
    u_real = torch.view_as_real(u_bft)
    states_real = torch.empty_like(u_real)

    grid = (batch, filters)
    _chunked_scan_kernel[grid](
        c1_real,
        c2_real,
        u_real,
        states_real,
        time,
        CHUNK=_pick_chunk(filters, batch),
        enable_fp_fusion=False,
    )
    states = torch.view_as_complex(states_real)
    return states.permute(0, 2, 1).contiguous()


def _chunked_scan_backward_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    states: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw backward pass for the chunked scan: (grad_C1, grad_C2, grad_u)."""
    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)
    if not u.is_complex():
        u = u.to(torch.cfloat)
    if not states.is_complex():
        states = states.to(torch.cfloat)
    if not grad_output.is_complex():
        grad_output = grad_output.to(torch.cfloat)

    batch, _, filters = u.shape
    u_bft = _to_time_innermost(u)
    time = u_bft.shape[2]
    states_bft = _to_time_innermost(states)
    grad_bft = _to_time_innermost(grad_output)

    c1_real = torch.view_as_real(C1.contiguous()).contiguous()
    c2_real = torch.view_as_real(C2.contiguous()).contiguous()
    u_real = torch.view_as_real(u_bft)
    states_real = torch.view_as_real(states_bft)
    grad_out_real = torch.view_as_real(grad_bft)
    grad_u_real = torch.empty_like(u_real)
    grad_c1 = torch.zeros(filters, 2, device=u.device, dtype=torch.float32)
    grad_c2 = torch.zeros(filters, 2, device=u.device, dtype=torch.float32)

    grid = (batch, filters)
    _chunked_scan_backward_kernel[grid](
        c1_real,
        c2_real,
        u_real,
        states_real,
        grad_out_real,
        grad_u_real,
        grad_c1,
        grad_c2,
        time,
        CHUNK=_pick_chunk(filters, batch),
        enable_fp_fusion=False,
    )
    grad_C1 = torch.view_as_complex(grad_c1.contiguous())
    grad_C2 = torch.view_as_complex(grad_c2.contiguous())
    grad_u = torch.view_as_complex(grad_u_real).permute(0, 2, 1).contiguous()
    return grad_C1, grad_C2, grad_u


class _ChunkedScanFunction(torch.autograd.Function):  # pragma: no cover
    """Autograd wrapper for the two-level chunked scan.

    Forward and backward both use the chunked structure, so training does not
    fall back to an O(T) sequential adjoint.
    """

    @staticmethod
    def forward(
        ctx,
        C1: torch.Tensor,
        C2: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using the chunked scan kernel; saves inputs and states."""
        if not C1.is_complex():
            C1 = C1.to(torch.cfloat)
        if not C2.is_complex():
            C2 = C2.to(torch.cfloat)
        if not u.is_complex():
            u = u.to(torch.cfloat)
        output = _chunked_scan_forward_raw(C1, C2, u)
        ctx.save_for_backward(C1, C2, u, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass via the chunked adjoint; returns grad_C1, grad_C2, grad_u."""
        C1, C2, u, states = ctx.saved_tensors
        grad_C1, grad_C2, grad_u = _chunked_scan_backward_raw(C1, C2, u, states, grad_output)
        return grad_C1, grad_C2, grad_u


def _triton_chunked_scan(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Compute x[n] = C1*x[n-1] + C2*u[n] with a two-level chunked scan.

    Drop-in replacement for the sequential Triton scan on CUDA for the scalar
    (no input-projection matrix) case, with a chunked parallel backward.

    Args:
        C1: Complex tensor of shape (num_filters,) - recurrence coefficient.
        C2: Complex tensor of shape (num_filters,) - input gain.
        u: Complex tensor of shape (batch, time, num_filters) - input signal.

    Returns:
        Complex tensor of shape (batch, time, num_filters) - output states.
    """
    return _ChunkedScanFunction.apply(C1, C2, u)


# ==============================================================================
# Prospective two-tap fused variant
#
# Fuses the prospective input drive drive[n] = tap0*x[n] + tap1*x[n-1] into the
# scan kernel: instead of materialising the drive (a shift + two complex muls in
# PyTorch, whose autograd shift/select dominate the encoder floor), the kernel
# loads x[n] and x[n-1] and forms the drive in-register, then scans
# x_out[n] = C1*x_out[n-1] + drive[n]. The backward reuses the chunked adjoint
# (the adjoint of the scan wrt its drive, and grad_C1) and finishes grad_x /
# grad_tap0 / grad_tap1 with cheap pointwise ops.
# ==============================================================================


@triton.jit
def _chunked_prospective_kernel(  # pragma: no cover
    c1_ptr,
    tap0_ptr,
    tap1_ptr,
    x_ptr,
    states_ptr,
    seqlen,
    CHUNK: tl.constexpr,
):
    """Forward chunked scan with a fused two-tap prospective drive
    drive[n] = tap0*x[n] + tap1*x[n-1]; interleaved real/imag layout."""
    bsz = tl.program_id(axis=0)
    chan = tl.program_id(axis=1)
    sid = tl.num_programs(axis=1) * bsz + chan
    idx = tl.arange(0, CHUNK)

    c1_r = tl.load(c1_ptr + chan * 2)
    c1_i = tl.load(c1_ptr + chan * 2 + 1)
    t0_r = tl.load(tap0_ptr + chan * 2)
    t0_i = tl.load(tap0_ptr + chan * 2 + 1)
    t1_r = tl.load(tap1_ptr + chan * 2)
    t1_i = tl.load(tap1_ptr + chan * 2 + 1)

    carry_r = tl.zeros((), dtype=tl.float32)
    carry_i = tl.zeros((), dtype=tl.float32)
    nchunks = tl.cdiv(seqlen, CHUNK)

    for chunk in range(0, nchunks):
        t = chunk * CHUNK + idx
        mask = t < seqlen
        off = (sid * seqlen + t) * 2
        x_r = tl.load(x_ptr + off, mask=mask, other=0.0)
        x_i = tl.load(x_ptr + off + 1, mask=mask, other=0.0)

        tprev = t - 1
        pmask = (tprev >= 0) & mask
        poff = (sid * seqlen + tprev) * 2
        xp_r = tl.load(x_ptr + poff, mask=pmask, other=0.0)
        xp_i = tl.load(x_ptr + poff + 1, mask=pmask, other=0.0)

        # drive = tap0 * x[t] + tap1 * x[t-1]
        b_r = t0_r * x_r - t0_i * x_i + t1_r * xp_r - t1_i * xp_i
        b_i = t0_r * x_i + t0_i * x_r + t1_r * xp_i + t1_i * xp_r
        # gate = C1 broadcast; identity beyond the sequence end
        f_r0 = tl.where(mask, c1_r, 1.0)
        f_i0 = tl.where(mask, c1_i, 0.0)

        s_r, s_i, g_r, g_i = tl.associative_scan(
            (b_r, b_i, f_r0, f_i0), axis=0, combine_fn=_affine_combine
        )
        last = idx == CHUNK - 1
        block_s_r = tl.sum(tl.where(last, s_r, 0.0), axis=0)
        block_s_i = tl.sum(tl.where(last, s_i, 0.0), axis=0)
        block_g_r = tl.sum(tl.where(last, g_r, 0.0), axis=0)
        block_g_i = tl.sum(tl.where(last, g_i, 0.0), axis=0)

        car_r, car_i = _complex_mul(g_r, g_i, carry_r, carry_i)
        out_r = s_r + car_r
        out_i = s_i + car_i
        tl.store(states_ptr + off, out_r, mask=mask)
        tl.store(states_ptr + off + 1, out_i, mask=mask)

        nc_r, nc_i = _complex_mul(block_g_r, block_g_i, carry_r, carry_i)
        carry_r = block_s_r + nc_r
        carry_i = block_s_i + nc_i


def _chunked_prospective_forward_raw(  # pragma: no cover
    C1: torch.Tensor,
    tap0: torch.Tensor,
    tap1: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Raw forward for the fused two-tap prospective scan; returns (batch, time, filters)."""
    if not x.is_complex():
        x = x.to(torch.cfloat)
    batch, _, filters = x.shape
    x_bft = _to_time_innermost(x)
    time = x_bft.shape[2]
    c1_real = torch.view_as_real(C1.to(torch.cfloat).contiguous()).contiguous()
    t0_real = torch.view_as_real(tap0.to(torch.cfloat).contiguous()).contiguous()
    t1_real = torch.view_as_real(tap1.to(torch.cfloat).contiguous()).contiguous()
    x_real = torch.view_as_real(x_bft)
    states_real = torch.empty_like(x_real)
    grid = (batch, filters)
    _chunked_prospective_kernel[grid](
        c1_real,
        t0_real,
        t1_real,
        x_real,
        states_real,
        time,
        CHUNK=_pick_chunk(filters, batch),
        enable_fp_fusion=False,
    )
    states = torch.view_as_complex(states_real)
    return states.permute(0, 2, 1).contiguous()


class _ChunkedProspectiveScanFunction(torch.autograd.Function):  # pragma: no cover
    """Fused prospective scan x_out[n]=C1 x_out[n-1] + tap0 x[n] + tap1 x[n-1]."""

    @staticmethod
    def forward(ctx, C1, tap0, tap1, x):
        """Forward via the fused prospective kernel; saves coefficients, input, states."""
        C1 = C1.to(torch.cfloat)
        tap0 = tap0.to(torch.cfloat)
        tap1 = tap1.to(torch.cfloat)
        if not x.is_complex():
            x = x.to(torch.cfloat)
        output = _chunked_prospective_forward_raw(C1, tap0, tap1, x)
        ctx.save_for_backward(C1, tap0, tap1, x, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """Backward: adjoint g wrt the drive (chunked) + pointwise grad_x/tap0/tap1."""
        C1, tap0, tap1, x, states = ctx.saved_tensors
        ones = torch.ones_like(C1)
        # g = adjoint of the scan wrt its drive input (grad_u with unit input gain)
        grad_C1, _grad_ones, g = _chunked_scan_backward_raw(C1, ones, x, states, grad_output)
        zeros = torch.zeros_like(g[:, :1, :])
        g_ahead = torch.cat([g[:, 1:, :], zeros], dim=1)  # g[t+1]
        x_delayed = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)  # x[t-1]
        grad_x = tap0.conj() * g + tap1.conj() * g_ahead
        grad_tap0 = (x.conj() * g).sum(dim=(0, 1))
        grad_tap1 = (x_delayed.conj() * g).sum(dim=(0, 1))
        return grad_C1, grad_tap0, grad_tap1, grad_x


def _triton_chunked_scan_prospective(  # pragma: no cover
    C1: torch.Tensor,
    tap0: torch.Tensor,
    tap1: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Fused two-tap prospective scan: x_out[n] = C1 x_out[n-1] + tap0 x[n] + tap1 x[n-1].

    Equivalent to forming drive[n]=tap0 x[n]+tap1 x[n-1] then scanning with unit
    input gain, but fuses the drive into the kernel (no materialised drive / shift).

    Args:
        C1: Complex (num_filters,) - recurrence coefficient.
        tap0: Complex (num_filters,) - current-sample input tap.
        tap1: Complex (num_filters,) - previous-sample input tap.
        x: Complex (batch, time, num_filters) - input signal.

    Returns:
        Complex (batch, time, num_filters) - output states.
    """
    return _ChunkedProspectiveScanFunction.apply(C1, tap0, tap1, x)
