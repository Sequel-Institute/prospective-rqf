"""Channel-outermost fused prospective scan for the Pure RQF encoder.

The shipped :func:`~rqf_models.ops.chunked_scan_kernel._triton_chunked_scan_prospective`
takes ``u`` as ``(batch, time, filters)`` and permutes to time-innermost on both
entry and exit of the forward and backward. When the whole encoder is kept in
channel-outermost ``(filters, batch, time)`` layout, those permutes disappear:
the channel projection becomes a single 2D complex GEMM ``W @ H`` on the free
view ``(filters, batch*time)`` and the scan indexes ``(filters, batch, time)``
directly (row ``chan*batch + b``, coefficients per channel).

This module provides that scan with the two-tap prospective drive
``drive[n] = tap0*x[n] + tap1*x[n-1]`` fused in, and an optional split-relu
epilogue (``y = relu(Re x) + i relu(Im x)``) folded into the store so the
activation needs no separate pass or complex/real view round-trip.

The backward is likewise fully fused: it recovers the split-relu mask from the
sign of the raw recurrence state (``relu'(0)=0``), runs the reverse chunked
adjoint, and forms ``grad_z``, ``grad_C1``, ``grad_tap0`` and ``grad_tap1`` all
in-register (so the drive-adjoint ``g`` is never materialised and there is no
pointwise finisher pass). ``grad_z`` needs ``g[t+1]``, recovered pointwise as
``(g[t] - gbar[t]) / conj(C1)`` from the adjoint recurrence.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from rqf_models.ops.chunked_scan_kernel import _affine_combine, _complex_mul, _pick_chunk


@triton.jit
def _fbt_prospective_kernel(  # pragma: no cover
    c1_ptr,
    tap0_ptr,
    tap1_ptr,
    x_ptr,
    raw_ptr,
    act_ptr,
    seqlen,
    batch,
    CHUNK: tl.constexpr,
    FUSE_ACT: tl.constexpr,
):
    """Forward chunked prospective scan in ``(filters, batch, time)`` layout.

    One program per row ``= chan*batch + b`` so ``row // batch`` is the channel.
    Stores the raw recurrence state (needed by the backward), and when
    ``FUSE_ACT`` also stores the split-relu of that state.
    """
    row = tl.program_id(axis=0)
    chan = row // batch
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
        off = (row * seqlen + t) * 2
        x_r = tl.load(x_ptr + off, mask=mask, other=0.0)
        x_i = tl.load(x_ptr + off + 1, mask=mask, other=0.0)

        tprev = t - 1
        pmask = (tprev >= 0) & mask
        poff = (row * seqlen + tprev) * 2
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
        tl.store(raw_ptr + off, out_r, mask=mask)
        tl.store(raw_ptr + off + 1, out_i, mask=mask)
        if FUSE_ACT:
            a_r = tl.where(out_r > 0.0, out_r, 0.0)
            a_i = tl.where(out_i > 0.0, out_i, 0.0)
            tl.store(act_ptr + off, a_r, mask=mask)
            tl.store(act_ptr + off + 1, a_i, mask=mask)

        nc_r, nc_i = _complex_mul(block_g_r, block_g_i, carry_r, carry_i)
        carry_r = block_s_r + nc_r
        carry_i = block_s_i + nc_i


@triton.jit
def _fbt_prospective_backward_kernel(  # pragma: no cover
    c1_ptr,
    tap0_ptr,
    tap1_ptr,
    z_ptr,
    raw_ptr,
    grad_out_ptr,
    grad_z_ptr,
    grad_c1_ptr,
    grad_tap0_ptr,
    grad_tap1_ptr,
    seqlen,
    batch,
    CHUNK: tl.constexpr,
    FUSE_ACT: tl.constexpr,
):
    """Fully fused reverse chunked adjoint in ``(filters, batch, time)`` layout.

    Runs the reverse scan for the drive-adjoint ``g`` and folds the whole gradient
    finisher into the kernel: ``grad_z``, ``grad_C1``, ``grad_tap0`` and
    ``grad_tap1`` are all formed in-register (tap and pole grads via atomic
    reductions), so ``g`` is never materialised and there is no separate pointwise
    pass. ``grad_z[t] = conj(tap0) g[t] + conj(tap1) g[t+1]`` needs ``g[t+1]``,
    recovered pointwise from the adjoint recurrence ``g[t]=gbar[t]+conj(C1) g[t+1]``
    as ``g[t+1] = (g[t] - gbar[t]) / conj(C1)`` (the ``g[seqlen]=0`` boundary is
    automatic). ``C1`` is a learned pole with no lower bound on its magnitude;
    the recovery stays well conditioned regardless because its error enters
    ``grad_z`` scaled by ``|tap1|/|C1|`` and the prospective taps carry
    ``tap1 ∝ C1`` (ZOH), cancelling the division — the fp64 parity tests pin the
    gradients for ``|C1|`` down to 0.05, and a fully collapsed pole surfaces as
    inf/nan (``|C1|^2`` underflow) rather than silent drift. When ``FUSE_ACT``
    the split-relu mask is applied to ``grad_out`` before the reverse scan.
    """
    row = tl.program_id(axis=0)
    chan = row // batch
    idx = tl.arange(0, CHUNK)

    c1_r = tl.load(c1_ptr + chan * 2)
    c1_i = tl.load(c1_ptr + chan * 2 + 1)
    t0_r = tl.load(tap0_ptr + chan * 2)
    t0_i = tl.load(tap0_ptr + chan * 2 + 1)
    t1_r = tl.load(tap1_ptr + chan * 2)
    t1_i = tl.load(tap1_ptr + chan * 2 + 1)
    # 1/conj(C1) = C1/|C1|^2
    den = c1_r * c1_r + c1_i * c1_i
    inv_r = c1_r / den
    inv_i = c1_i / den

    carry_r = tl.zeros((), dtype=tl.float32)
    carry_i = tl.zeros((), dtype=tl.float32)
    gc1_r = tl.zeros((), dtype=tl.float32)
    gc1_i = tl.zeros((), dtype=tl.float32)
    gt0_r = tl.zeros((), dtype=tl.float32)
    gt0_i = tl.zeros((), dtype=tl.float32)
    gt1_r = tl.zeros((), dtype=tl.float32)
    gt1_i = tl.zeros((), dtype=tl.float32)
    nchunks = tl.cdiv(seqlen, CHUNK)

    for chunk in range(0, nchunks):
        rchunk = nchunks - 1 - chunk
        t = rchunk * CHUNK + idx
        mask = t < seqlen
        off = (row * seqlen + t) * 2

        go_r = tl.load(grad_out_ptr + off, mask=mask, other=0.0)
        go_i = tl.load(grad_out_ptr + off + 1, mask=mask, other=0.0)
        if FUSE_ACT:
            # split-relu mask from the sign of the raw state (relu'(0)=0)
            raw_r = tl.load(raw_ptr + off, mask=mask, other=0.0)
            raw_i = tl.load(raw_ptr + off + 1, mask=mask, other=0.0)
            go_r = tl.where(raw_r > 0.0, go_r, 0.0)
            go_i = tl.where(raw_i > 0.0, go_i, 0.0)

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

        # g[t+1] = (g[t] - gbar[t]) / conj(C1)
        d_r = g_r - go_r
        d_i = g_i - go_i
        gp_r, gp_i = _complex_mul(d_r, d_i, inv_r, inv_i)

        # grad_z = conj(tap0) g[t] + conj(tap1) g[t+1]
        gz_r = t0_r * g_r + t0_i * g_i + t1_r * gp_r + t1_i * gp_i
        gz_i = t0_r * g_i - t0_i * g_r + t1_r * gp_i - t1_i * gp_r
        tl.store(grad_z_ptr + off, gz_r, mask=mask)
        tl.store(grad_z_ptr + off + 1, gz_i, mask=mask)

        tprev = t - 1
        pmask = (tprev >= 0) & (tprev < seqlen) & mask
        offp = (row * seqlen + tprev) * 2
        # grad_C1 += sum_t conj(raw[t-1]) g[t]  (raw[-1] = 0)
        xr = tl.load(raw_ptr + offp, mask=pmask, other=0.0)
        xi = tl.load(raw_ptr + offp + 1, mask=pmask, other=0.0)
        gc1_r += tl.sum(tl.where(pmask, xr * g_r + xi * g_i, 0.0), axis=0)
        gc1_i += tl.sum(tl.where(pmask, xr * g_i - xi * g_r, 0.0), axis=0)
        # grad_tap0 += sum_t conj(z[t]) g[t]
        zr = tl.load(z_ptr + off, mask=mask, other=0.0)
        zi = tl.load(z_ptr + off + 1, mask=mask, other=0.0)
        gt0_r += tl.sum(tl.where(mask, zr * g_r + zi * g_i, 0.0), axis=0)
        gt0_i += tl.sum(tl.where(mask, zr * g_i - zi * g_r, 0.0), axis=0)
        # grad_tap1 += sum_t conj(z[t-1]) g[t]
        zpr = tl.load(z_ptr + offp, mask=pmask, other=0.0)
        zpi = tl.load(z_ptr + offp + 1, mask=pmask, other=0.0)
        gt1_r += tl.sum(tl.where(pmask, zpr * g_r + zpi * g_i, 0.0), axis=0)
        gt1_i += tl.sum(tl.where(pmask, zpr * g_i - zpi * g_r, 0.0), axis=0)

        nc_r, nc_i = _complex_mul(block_gg_r, block_gg_i, carry_r, carry_i)
        carry_r = block_g_r + nc_r
        carry_i = block_g_i + nc_i

    tl.atomic_add(grad_c1_ptr + chan * 2, gc1_r)
    tl.atomic_add(grad_c1_ptr + chan * 2 + 1, gc1_i)
    tl.atomic_add(grad_tap0_ptr + chan * 2, gt0_r)
    tl.atomic_add(grad_tap0_ptr + chan * 2 + 1, gt0_i)
    tl.atomic_add(grad_tap1_ptr + chan * 2, gt1_r)
    tl.atomic_add(grad_tap1_ptr + chan * 2 + 1, gt1_i)


class _FBTProspectiveScan(torch.autograd.Function):  # pragma: no cover
    """Fused two-tap prospective scan on ``(filters, batch, time)`` complex input.

    Returns the split-relu activated output when ``fuse_act`` is set, otherwise
    the raw recurrence state. No permute or copy of the time axis.
    """

    @staticmethod
    def forward(ctx, C1, tap0, tap1, x_fbt, fuse_act):
        C1 = C1.to(torch.cfloat)
        tap0 = tap0.to(torch.cfloat)
        tap1 = tap1.to(torch.cfloat)
        if not x_fbt.is_complex():
            x_fbt = x_fbt.to(torch.cfloat)
        x_fbt = x_fbt.contiguous()
        filters, batch, time = x_fbt.shape

        c1_real = torch.view_as_real(C1.contiguous()).contiguous()
        t0_real = torch.view_as_real(tap0.contiguous()).contiguous()
        t1_real = torch.view_as_real(tap1.contiguous()).contiguous()
        x_real = torch.view_as_real(x_fbt)
        raw_real = torch.empty_like(x_real)
        act_real = torch.empty_like(x_real) if fuse_act else raw_real

        grid = (filters * batch,)
        _fbt_prospective_kernel[grid](
            c1_real,
            t0_real,
            t1_real,
            x_real,
            raw_real,
            act_real,
            time,
            batch,
            CHUNK=_pick_chunk(filters, batch),
            FUSE_ACT=fuse_act,
            enable_fp_fusion=False,
        )
        raw = torch.view_as_complex(raw_real)
        ctx.save_for_backward(C1, tap0, tap1, x_fbt, raw)
        ctx.fuse_act = fuse_act
        return torch.view_as_complex(act_real) if fuse_act else raw

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        C1, tap0, tap1, x, raw = ctx.saved_tensors
        filters, batch, time = x.shape
        if not grad_output.is_complex():
            grad_output = grad_output.to(torch.cfloat)

        c1_real = torch.view_as_real(C1.contiguous()).contiguous()
        t0_real = torch.view_as_real(tap0.contiguous()).contiguous()
        t1_real = torch.view_as_real(tap1.contiguous()).contiguous()
        z_real = torch.view_as_real(x.contiguous())
        raw_real = torch.view_as_real(raw.contiguous())
        grad_out_real = torch.view_as_real(grad_output.contiguous())
        grad_z_real = torch.empty_like(grad_out_real)
        grad_c1 = torch.zeros(filters, 2, device=x.device, dtype=torch.float32)
        grad_t0 = torch.zeros(filters, 2, device=x.device, dtype=torch.float32)
        grad_t1 = torch.zeros(filters, 2, device=x.device, dtype=torch.float32)

        grid = (filters * batch,)
        _fbt_prospective_backward_kernel[grid](
            c1_real,
            t0_real,
            t1_real,
            z_real,
            raw_real,
            grad_out_real,
            grad_z_real,
            grad_c1,
            grad_t0,
            grad_t1,
            time,
            batch,
            CHUNK=_pick_chunk(filters, batch),
            FUSE_ACT=ctx.fuse_act,
            enable_fp_fusion=False,
        )
        grad_C1 = torch.view_as_complex(grad_c1.contiguous())
        grad_tap0 = torch.view_as_complex(grad_t0.contiguous())
        grad_tap1 = torch.view_as_complex(grad_t1.contiguous())
        grad_x = torch.view_as_complex(grad_z_real)
        return grad_C1, grad_tap0, grad_tap1, grad_x, None


def fbt_prospective_scan(  # pragma: no cover
    C1: torch.Tensor,
    tap0: torch.Tensor,
    tap1: torch.Tensor,
    x_fbt: torch.Tensor,
    fuse_act: bool = False,
) -> torch.Tensor:
    """Fused two-tap prospective scan in channel-outermost layout.

    Computes ``x_out[n] = C1 x_out[n-1] + tap0 x[n] + tap1 x[n-1]`` with the
    drive fused into the kernel, on a ``(filters, batch, time)`` complex input,
    without permuting the time axis.

    Args:
        C1: Complex ``(num_filters,)`` recurrence coefficient.
        tap0: Complex ``(num_filters,)`` current-sample input tap.
        tap1: Complex ``(num_filters,)`` previous-sample input tap.
        x_fbt: Complex ``(num_filters, batch, time)`` input signal.
        fuse_act: When ``True`` apply the split-relu epilogue
            ``relu(Re) + i relu(Im)`` inside the kernel and return the activated
            output; otherwise return the raw recurrence state.

    Returns:
        Complex ``(num_filters, batch, time)`` output.
    """
    return _FBTProspectiveScan.apply(C1, tap0, tap1, x_fbt, fuse_act)


@triton.jit
def _fbt_recompute_forward_kernel(  # pragma: no cover
    c1_ptr,
    tap0_ptr,
    tap1_ptr,
    x_ptr,
    act_ptr,
    carry_ptr,
    seqlen,
    batch,
    nchunks,
    CHUNK: tl.constexpr,
):
    """Forward prospective scan storing only the split-relu output + entry carry.

    Identical recurrence to :func:`_fbt_prospective_kernel` with ``FUSE_ACT`` set,
    but instead of writing the full raw ``(filters, batch, time)`` state it stores
    only the activated output and the tiny per-chunk state that enters each chunk
    (``carry_ptr``, ``rows x nchunks`` complex). The backward replays the forward
    recurrence per chunk from that carry to recover the raw state.
    """
    row = tl.program_id(axis=0)
    chan = row // batch
    idx = tl.arange(0, CHUNK)

    c1_r = tl.load(c1_ptr + chan * 2)
    c1_i = tl.load(c1_ptr + chan * 2 + 1)
    t0_r = tl.load(tap0_ptr + chan * 2)
    t0_i = tl.load(tap0_ptr + chan * 2 + 1)
    t1_r = tl.load(tap1_ptr + chan * 2)
    t1_i = tl.load(tap1_ptr + chan * 2 + 1)

    carry_r = tl.zeros((), dtype=tl.float32)
    carry_i = tl.zeros((), dtype=tl.float32)

    for chunk in range(0, nchunks):
        # state entering this chunk, saved for the backward recompute
        tl.store(carry_ptr + (row * nchunks + chunk) * 2, carry_r)
        tl.store(carry_ptr + (row * nchunks + chunk) * 2 + 1, carry_i)

        t = chunk * CHUNK + idx
        mask = t < seqlen
        off = (row * seqlen + t) * 2
        x_r = tl.load(x_ptr + off, mask=mask, other=0.0)
        x_i = tl.load(x_ptr + off + 1, mask=mask, other=0.0)

        tprev = t - 1
        pmask = (tprev >= 0) & mask
        poff = (row * seqlen + tprev) * 2
        xp_r = tl.load(x_ptr + poff, mask=pmask, other=0.0)
        xp_i = tl.load(x_ptr + poff + 1, mask=pmask, other=0.0)

        b_r = t0_r * x_r - t0_i * x_i + t1_r * xp_r - t1_i * xp_i
        b_i = t0_r * x_i + t0_i * x_r + t1_r * xp_i + t1_i * xp_r
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
        a_r = tl.where(out_r > 0.0, out_r, 0.0)
        a_i = tl.where(out_i > 0.0, out_i, 0.0)
        tl.store(act_ptr + off, a_r, mask=mask)
        tl.store(act_ptr + off + 1, a_i, mask=mask)

        nc_r, nc_i = _complex_mul(block_g_r, block_g_i, carry_r, carry_i)
        carry_r = block_s_r + nc_r
        carry_i = block_s_i + nc_i


@triton.jit
def _fbt_recompute_backward_kernel(  # pragma: no cover
    c1_ptr,
    tap0_ptr,
    tap1_ptr,
    z_ptr,
    carry_ptr,
    grad_out_ptr,
    grad_z_ptr,
    grad_c1_ptr,
    grad_tap0_ptr,
    grad_tap1_ptr,
    seqlen,
    batch,
    nchunks,
    CHUNK: tl.constexpr,
):
    """Reverse adjoint that recomputes the raw state from the stored entry carry.

    Per chunk it replays the forward recurrence from ``carry_ptr`` to recover
    ``raw[t]``, then ``raw[t-1] = (raw[t] - drive[t]) / C1`` gives the shifted state
    for ``grad_C1`` (``raw[-1]=0`` falls out at ``t=0``). The split-relu mask comes
    from the sign of the recomputed raw. Gradients are otherwise formed exactly as
    in :func:`_fbt_prospective_backward_kernel`.
    """
    row = tl.program_id(axis=0)
    chan = row // batch
    idx = tl.arange(0, CHUNK)

    c1_r = tl.load(c1_ptr + chan * 2)
    c1_i = tl.load(c1_ptr + chan * 2 + 1)
    t0_r = tl.load(tap0_ptr + chan * 2)
    t0_i = tl.load(tap0_ptr + chan * 2 + 1)
    t1_r = tl.load(tap1_ptr + chan * 2)
    t1_i = tl.load(tap1_ptr + chan * 2 + 1)
    den = c1_r * c1_r + c1_i * c1_i
    # 1/conj(C1) = C1/|C1|^2 for the adjoint g[t+1] recovery
    inv_r = c1_r / den
    inv_i = c1_i / den
    # 1/C1 = conj(C1)/|C1|^2 for the raw[t-1] recovery
    invc_r = c1_r / den
    invc_i = -c1_i / den

    carry_r = tl.zeros((), dtype=tl.float32)
    carry_i = tl.zeros((), dtype=tl.float32)
    gc1_r = tl.zeros((), dtype=tl.float32)
    gc1_i = tl.zeros((), dtype=tl.float32)
    gt0_r = tl.zeros((), dtype=tl.float32)
    gt0_i = tl.zeros((), dtype=tl.float32)
    gt1_r = tl.zeros((), dtype=tl.float32)
    gt1_i = tl.zeros((), dtype=tl.float32)

    for chunk in range(0, nchunks):
        rchunk = nchunks - 1 - chunk
        t = rchunk * CHUNK + idx
        mask = t < seqlen
        off = (row * seqlen + t) * 2

        # recompute raw[t] for this chunk from its stored entry carry
        ec_r = tl.load(carry_ptr + (row * nchunks + rchunk) * 2)
        ec_i = tl.load(carry_ptr + (row * nchunks + rchunk) * 2 + 1)
        z_r = tl.load(z_ptr + off, mask=mask, other=0.0)
        z_i = tl.load(z_ptr + off + 1, mask=mask, other=0.0)
        tprev = t - 1
        pmask = (tprev >= 0) & mask
        poff = (row * seqlen + tprev) * 2
        zp_r = tl.load(z_ptr + poff, mask=pmask, other=0.0)
        zp_i = tl.load(z_ptr + poff + 1, mask=pmask, other=0.0)

        b_r = t0_r * z_r - t0_i * z_i + t1_r * zp_r - t1_i * zp_i
        b_i = t0_r * z_i + t0_i * z_r + t1_r * zp_i + t1_i * zp_r
        f_r0 = tl.where(mask, c1_r, 1.0)
        f_i0 = tl.where(mask, c1_i, 0.0)
        rs_r, rs_i, rg_r, rg_i = tl.associative_scan(
            (b_r, b_i, f_r0, f_i0), axis=0, combine_fn=_affine_combine
        )
        pc_r, pc_i = _complex_mul(rg_r, rg_i, ec_r, ec_i)
        raw_r = rs_r + pc_r
        raw_i = rs_i + pc_i
        # raw[t-1] = (raw[t] - drive[t]) / C1
        d_r = raw_r - b_r
        d_i = raw_i - b_i
        rawp_r, rawp_i = _complex_mul(d_r, d_i, invc_r, invc_i)

        # reverse adjoint on the split-relu-masked upstream gradient
        go_r = tl.load(grad_out_ptr + off, mask=mask, other=0.0)
        go_i = tl.load(grad_out_ptr + off + 1, mask=mask, other=0.0)
        go_r = tl.where(raw_r > 0.0, go_r, 0.0)
        go_i = tl.where(raw_i > 0.0, go_i, 0.0)
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

        # g[t+1] = (g[t] - gbar[t]) / conj(C1)
        d2_r = g_r - go_r
        d2_i = g_i - go_i
        gp_r, gp_i = _complex_mul(d2_r, d2_i, inv_r, inv_i)
        # grad_z = conj(tap0) g[t] + conj(tap1) g[t+1]
        gz_r = t0_r * g_r + t0_i * g_i + t1_r * gp_r + t1_i * gp_i
        gz_i = t0_r * g_i - t0_i * g_r + t1_r * gp_i - t1_i * gp_r
        tl.store(grad_z_ptr + off, gz_r, mask=mask)
        tl.store(grad_z_ptr + off + 1, gz_i, mask=mask)

        # grad_C1 += sum_t conj(raw[t-1]) g[t]  (raw[-1]=0 at t=0)
        gc1_r += tl.sum(tl.where(mask, rawp_r * g_r + rawp_i * g_i, 0.0), axis=0)
        gc1_i += tl.sum(tl.where(mask, rawp_r * g_i - rawp_i * g_r, 0.0), axis=0)
        # grad_tap0 += sum_t conj(z[t]) g[t]
        gt0_r += tl.sum(tl.where(mask, z_r * g_r + z_i * g_i, 0.0), axis=0)
        gt0_i += tl.sum(tl.where(mask, z_r * g_i - z_i * g_r, 0.0), axis=0)
        # grad_tap1 += sum_t conj(z[t-1]) g[t]
        gt1_r += tl.sum(tl.where(pmask, zp_r * g_r + zp_i * g_i, 0.0), axis=0)
        gt1_i += tl.sum(tl.where(pmask, zp_r * g_i - zp_i * g_r, 0.0), axis=0)

        nc_r, nc_i = _complex_mul(block_gg_r, block_gg_i, carry_r, carry_i)
        carry_r = block_g_r + nc_r
        carry_i = block_g_i + nc_i

    tl.atomic_add(grad_c1_ptr + chan * 2, gc1_r)
    tl.atomic_add(grad_c1_ptr + chan * 2 + 1, gc1_i)
    tl.atomic_add(grad_tap0_ptr + chan * 2, gt0_r)
    tl.atomic_add(grad_tap0_ptr + chan * 2 + 1, gt0_i)
    tl.atomic_add(grad_tap1_ptr + chan * 2, gt1_r)
    tl.atomic_add(grad_tap1_ptr + chan * 2 + 1, gt1_i)


class _FBTRecompute(torch.autograd.Function):  # pragma: no cover
    """Split-relu prospective scan that recomputes the raw state in the backward.

    The forward stores only the activated output plus a tiny per-chunk entry-carry
    buffer, not the full raw ``(filters, batch, time)`` recurrence state; the
    backward replays the forward recurrence per chunk to recover it. This trades a
    little recompute for lower peak memory. Always applies the split-relu epilogue.
    """

    @staticmethod
    def forward(ctx, C1, tap0, tap1, x_fbt):
        C1 = C1.to(torch.cfloat)
        tap0 = tap0.to(torch.cfloat)
        tap1 = tap1.to(torch.cfloat)
        if not x_fbt.is_complex():
            x_fbt = x_fbt.to(torch.cfloat)
        x_fbt = x_fbt.contiguous()
        filters, batch, time = x_fbt.shape
        chunk = _pick_chunk(filters, batch)
        nchunks = (time + chunk - 1) // chunk

        c1_real = torch.view_as_real(C1.contiguous()).contiguous()
        t0_real = torch.view_as_real(tap0.contiguous()).contiguous()
        t1_real = torch.view_as_real(tap1.contiguous()).contiguous()
        x_real = torch.view_as_real(x_fbt)
        act_real = torch.empty_like(x_real)
        carry = torch.empty(filters * batch, nchunks, 2, device=x_fbt.device, dtype=torch.float32)

        grid = (filters * batch,)
        _fbt_recompute_forward_kernel[grid](
            c1_real,
            t0_real,
            t1_real,
            x_real,
            act_real,
            carry,
            time,
            batch,
            nchunks,
            CHUNK=chunk,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(C1, tap0, tap1, x_fbt, carry)
        ctx.chunk = chunk
        ctx.nchunks = nchunks
        return torch.view_as_complex(act_real)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        C1, tap0, tap1, x, carry = ctx.saved_tensors
        filters, batch, time = x.shape
        chunk = ctx.chunk
        nchunks = ctx.nchunks
        if not grad_output.is_complex():
            grad_output = grad_output.to(torch.cfloat)

        c1_real = torch.view_as_real(C1.contiguous()).contiguous()
        t0_real = torch.view_as_real(tap0.contiguous()).contiguous()
        t1_real = torch.view_as_real(tap1.contiguous()).contiguous()
        z_real = torch.view_as_real(x.contiguous())
        grad_out_real = torch.view_as_real(grad_output.contiguous())
        grad_z_real = torch.empty_like(grad_out_real)
        grad_c1 = torch.zeros(filters, 2, device=x.device, dtype=torch.float32)
        grad_t0 = torch.zeros(filters, 2, device=x.device, dtype=torch.float32)
        grad_t1 = torch.zeros(filters, 2, device=x.device, dtype=torch.float32)

        grid = (filters * batch,)
        _fbt_recompute_backward_kernel[grid](
            c1_real,
            t0_real,
            t1_real,
            z_real,
            carry,
            grad_out_real,
            grad_z_real,
            grad_c1,
            grad_t0,
            grad_t1,
            time,
            batch,
            nchunks,
            CHUNK=chunk,
            enable_fp_fusion=False,
        )
        grad_C1 = torch.view_as_complex(grad_c1.contiguous())
        grad_tap0 = torch.view_as_complex(grad_t0.contiguous())
        grad_tap1 = torch.view_as_complex(grad_t1.contiguous())
        grad_x = torch.view_as_complex(grad_z_real)
        return grad_C1, grad_tap0, grad_tap1, grad_x


def fbt_recompute_scan(  # pragma: no cover
    C1: torch.Tensor,
    tap0: torch.Tensor,
    tap1: torch.Tensor,
    x_fbt: torch.Tensor,
) -> torch.Tensor:
    """Split-relu prospective scan with a recompute-in-backward memory footprint.

    Same result as ``fbt_prospective_scan(..., fuse_act=True)`` but stores only the
    activated output plus a small per-chunk entry carry instead of the full raw
    recurrence state, lowering peak memory at the cost of a little recompute in the
    backward. Intended for the split-relu activation on large launch grids.

    Args:
        C1: Complex ``(num_filters,)`` recurrence coefficient.
        tap0: Complex ``(num_filters,)`` current-sample input tap.
        tap1: Complex ``(num_filters,)`` previous-sample input tap.
        x_fbt: Complex ``(num_filters, batch, time)`` input signal.

    Returns:
        Complex ``(num_filters, batch, time)`` split-relu activated output.
    """
    return _FBTRecompute.apply(C1, tap0, tap1, x_fbt)


@triton.jit
def _fbt_recompute_planar_forward_kernel(  # pragma: no cover
    c1_ptr,
    tap0_ptr,
    tap1_ptr,
    zr_ptr,
    zi_ptr,
    act_ptr,
    carry_ptr,
    seqlen,
    batch,
    nchunks,
    CHUNK: tl.constexpr,
):
    """Recompute forward scan reading the drive from PLANAR ``(zr, zi)`` buffers.

    Identical recurrence to :func:`_fbt_recompute_forward_kernel` but the input is
    two separate real tensors (real and imaginary parts contiguous, not
    interleaved), which is what the layer-0 real-input stacked GEMM produces. The
    activated output is written interleaved and only the per-chunk entry carry is
    stored (no raw state).
    """
    row = tl.program_id(axis=0)
    chan = row // batch
    idx = tl.arange(0, CHUNK)

    c1_r = tl.load(c1_ptr + chan * 2)
    c1_i = tl.load(c1_ptr + chan * 2 + 1)
    t0_r = tl.load(tap0_ptr + chan * 2)
    t0_i = tl.load(tap0_ptr + chan * 2 + 1)
    t1_r = tl.load(tap1_ptr + chan * 2)
    t1_i = tl.load(tap1_ptr + chan * 2 + 1)

    carry_r = tl.zeros((), dtype=tl.float32)
    carry_i = tl.zeros((), dtype=tl.float32)

    for chunk in range(0, nchunks):
        tl.store(carry_ptr + (row * nchunks + chunk) * 2, carry_r)
        tl.store(carry_ptr + (row * nchunks + chunk) * 2 + 1, carry_i)

        t = chunk * CHUNK + idx
        mask = t < seqlen
        base = row * seqlen + t
        x_r = tl.load(zr_ptr + base, mask=mask, other=0.0)
        x_i = tl.load(zi_ptr + base, mask=mask, other=0.0)

        tprev = t - 1
        pmask = (tprev >= 0) & mask
        pbase = row * seqlen + tprev
        xp_r = tl.load(zr_ptr + pbase, mask=pmask, other=0.0)
        xp_i = tl.load(zi_ptr + pbase, mask=pmask, other=0.0)

        b_r = t0_r * x_r - t0_i * x_i + t1_r * xp_r - t1_i * xp_i
        b_i = t0_r * x_i + t0_i * x_r + t1_r * xp_i + t1_i * xp_r
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
        off = base * 2
        a_r = tl.where(out_r > 0.0, out_r, 0.0)
        a_i = tl.where(out_i > 0.0, out_i, 0.0)
        tl.store(act_ptr + off, a_r, mask=mask)
        tl.store(act_ptr + off + 1, a_i, mask=mask)

        nc_r, nc_i = _complex_mul(block_g_r, block_g_i, carry_r, carry_i)
        carry_r = block_s_r + nc_r
        carry_i = block_s_i + nc_i


@triton.jit
def _fbt_recompute_planar_backward_kernel(  # pragma: no cover
    c1_ptr,
    tap0_ptr,
    tap1_ptr,
    zr_ptr,
    zi_ptr,
    carry_ptr,
    grad_out_ptr,
    grad_zr_ptr,
    grad_zi_ptr,
    grad_c1_ptr,
    grad_tap0_ptr,
    grad_tap1_ptr,
    seqlen,
    batch,
    nchunks,
    CHUNK: tl.constexpr,
):
    """Planar-``(zr, zi)`` counterpart of :func:`_fbt_recompute_backward_kernel`.

    Recomputes the raw state per chunk from the stored entry carry and runs the
    reverse adjoint; ``grad_z`` is written to separate real/imaginary buffers so
    the caller can feed them straight into the real backward GEMM.
    """
    row = tl.program_id(axis=0)
    chan = row // batch
    idx = tl.arange(0, CHUNK)

    c1_r = tl.load(c1_ptr + chan * 2)
    c1_i = tl.load(c1_ptr + chan * 2 + 1)
    t0_r = tl.load(tap0_ptr + chan * 2)
    t0_i = tl.load(tap0_ptr + chan * 2 + 1)
    t1_r = tl.load(tap1_ptr + chan * 2)
    t1_i = tl.load(tap1_ptr + chan * 2 + 1)
    den = c1_r * c1_r + c1_i * c1_i
    inv_r = c1_r / den
    inv_i = c1_i / den
    invc_r = c1_r / den
    invc_i = -c1_i / den

    carry_r = tl.zeros((), dtype=tl.float32)
    carry_i = tl.zeros((), dtype=tl.float32)
    gc1_r = tl.zeros((), dtype=tl.float32)
    gc1_i = tl.zeros((), dtype=tl.float32)
    gt0_r = tl.zeros((), dtype=tl.float32)
    gt0_i = tl.zeros((), dtype=tl.float32)
    gt1_r = tl.zeros((), dtype=tl.float32)
    gt1_i = tl.zeros((), dtype=tl.float32)

    for chunk in range(0, nchunks):
        rchunk = nchunks - 1 - chunk
        t = rchunk * CHUNK + idx
        mask = t < seqlen
        base = row * seqlen + t
        off = base * 2

        ec_r = tl.load(carry_ptr + (row * nchunks + rchunk) * 2)
        ec_i = tl.load(carry_ptr + (row * nchunks + rchunk) * 2 + 1)
        z_r = tl.load(zr_ptr + base, mask=mask, other=0.0)
        z_i = tl.load(zi_ptr + base, mask=mask, other=0.0)
        tprev = t - 1
        pmask = (tprev >= 0) & mask
        pbase = row * seqlen + tprev
        zp_r = tl.load(zr_ptr + pbase, mask=pmask, other=0.0)
        zp_i = tl.load(zi_ptr + pbase, mask=pmask, other=0.0)

        b_r = t0_r * z_r - t0_i * z_i + t1_r * zp_r - t1_i * zp_i
        b_i = t0_r * z_i + t0_i * z_r + t1_r * zp_i + t1_i * zp_r
        f_r0 = tl.where(mask, c1_r, 1.0)
        f_i0 = tl.where(mask, c1_i, 0.0)
        rs_r, rs_i, rg_r, rg_i = tl.associative_scan(
            (b_r, b_i, f_r0, f_i0), axis=0, combine_fn=_affine_combine
        )
        pc_r, pc_i = _complex_mul(rg_r, rg_i, ec_r, ec_i)
        raw_r = rs_r + pc_r
        raw_i = rs_i + pc_i
        d_r = raw_r - b_r
        d_i = raw_i - b_i
        rawp_r, rawp_i = _complex_mul(d_r, d_i, invc_r, invc_i)

        go_r = tl.load(grad_out_ptr + off, mask=mask, other=0.0)
        go_i = tl.load(grad_out_ptr + off + 1, mask=mask, other=0.0)
        go_r = tl.where(raw_r > 0.0, go_r, 0.0)
        go_i = tl.where(raw_i > 0.0, go_i, 0.0)
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

        d2_r = g_r - go_r
        d2_i = g_i - go_i
        gp_r, gp_i = _complex_mul(d2_r, d2_i, inv_r, inv_i)
        gz_r = t0_r * g_r + t0_i * g_i + t1_r * gp_r + t1_i * gp_i
        gz_i = t0_r * g_i - t0_i * g_r + t1_r * gp_i - t1_i * gp_r
        tl.store(grad_zr_ptr + base, gz_r, mask=mask)
        tl.store(grad_zi_ptr + base, gz_i, mask=mask)

        gc1_r += tl.sum(tl.where(mask, rawp_r * g_r + rawp_i * g_i, 0.0), axis=0)
        gc1_i += tl.sum(tl.where(mask, rawp_r * g_i - rawp_i * g_r, 0.0), axis=0)
        gt0_r += tl.sum(tl.where(mask, z_r * g_r + z_i * g_i, 0.0), axis=0)
        gt0_i += tl.sum(tl.where(mask, z_r * g_i - z_i * g_r, 0.0), axis=0)
        gt1_r += tl.sum(tl.where(pmask, zp_r * g_r + zp_i * g_i, 0.0), axis=0)
        gt1_i += tl.sum(tl.where(pmask, zp_r * g_i - zp_i * g_r, 0.0), axis=0)

        nc_r, nc_i = _complex_mul(block_gg_r, block_gg_i, carry_r, carry_i)
        carry_r = block_g_r + nc_r
        carry_i = block_g_i + nc_i

    tl.atomic_add(grad_c1_ptr + chan * 2, gc1_r)
    tl.atomic_add(grad_c1_ptr + chan * 2 + 1, gc1_i)
    tl.atomic_add(grad_tap0_ptr + chan * 2, gt0_r)
    tl.atomic_add(grad_tap0_ptr + chan * 2 + 1, gt0_i)
    tl.atomic_add(grad_tap1_ptr + chan * 2, gt1_r)
    tl.atomic_add(grad_tap1_ptr + chan * 2 + 1, gt1_i)


class _Layer0RealScan(torch.autograd.Function):  # pragma: no cover
    """Real-input layer-0 projection fused with the planar recompute scan.

    The first encoder block projects a real signal (imaginary part identically
    zero) through the complex weight ``W``. Writing ``W = Wr + i Wi``, the complex
    projection ``z = W @ h`` on a real ``h`` is a single stacked real GEMM
    ``[Wr; Wi] @ h`` giving the planar parts ``(zr, zi)`` directly, which the
    planar recompute scan then consumes without an interleave round-trip. The
    backward's real GEMMs likewise halve the work of the complex adjoint. Applies
    the split-relu epilogue (the only activation this fused path serves).
    """

    @staticmethod
    def forward(ctx, w_stacked, h_real, C1, tap0, tap1, batch):
        w_stacked = w_stacked.contiguous()
        h_real = h_real.contiguous()
        C1 = C1.to(torch.cfloat)
        tap0 = tap0.to(torch.cfloat)
        tap1 = tap1.to(torch.cfloat)
        filters = w_stacked.shape[0] // 2
        cols = h_real.shape[1]
        time = cols // batch

        proj = torch.matmul(w_stacked, h_real)  # (2*filters, batch*time)
        zr = proj[:filters].reshape(filters, batch, time).contiguous()
        zi = proj[filters:].reshape(filters, batch, time).contiguous()

        chunk = _pick_chunk(filters, batch)
        nchunks = (time + chunk - 1) // chunk
        c1_real = torch.view_as_real(C1.contiguous()).contiguous()
        t0_real = torch.view_as_real(tap0.contiguous()).contiguous()
        t1_real = torch.view_as_real(tap1.contiguous()).contiguous()
        act_real = torch.empty(filters, batch, time, 2, device=h_real.device, dtype=torch.float32)
        carry = torch.empty(filters * batch, nchunks, 2, device=h_real.device, dtype=torch.float32)

        grid = (filters * batch,)
        _fbt_recompute_planar_forward_kernel[grid](
            c1_real,
            t0_real,
            t1_real,
            zr.reshape(-1),
            zi.reshape(-1),
            act_real,
            carry,
            time,
            batch,
            nchunks,
            CHUNK=chunk,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(w_stacked, h_real, zr, zi, carry, C1, tap0, tap1)
        ctx.chunk = chunk
        ctx.nchunks = nchunks
        ctx.filters = filters
        ctx.batch = batch
        ctx.time = time
        return torch.view_as_complex(act_real)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_act):
        w_stacked, h_real, zr, zi, carry, C1, tap0, tap1 = ctx.saved_tensors
        filters, batch, time = ctx.filters, ctx.batch, ctx.time
        chunk, nchunks = ctx.chunk, ctx.nchunks
        if not grad_act.is_complex():
            grad_act = grad_act.to(torch.cfloat)
        grad_out_real = torch.view_as_real(grad_act.contiguous())
        grad_zr = torch.empty(filters, batch, time, device=h_real.device, dtype=torch.float32)
        grad_zi = torch.empty(filters, batch, time, device=h_real.device, dtype=torch.float32)
        c1_real = torch.view_as_real(C1.contiguous()).contiguous()
        t0_real = torch.view_as_real(tap0.contiguous()).contiguous()
        t1_real = torch.view_as_real(tap1.contiguous()).contiguous()
        grad_c1 = torch.zeros(filters, 2, device=h_real.device, dtype=torch.float32)
        grad_t0 = torch.zeros(filters, 2, device=h_real.device, dtype=torch.float32)
        grad_t1 = torch.zeros(filters, 2, device=h_real.device, dtype=torch.float32)

        grid = (filters * batch,)
        _fbt_recompute_planar_backward_kernel[grid](
            c1_real,
            t0_real,
            t1_real,
            zr.reshape(-1),
            zi.reshape(-1),
            carry,
            grad_out_real,
            grad_zr.reshape(-1),
            grad_zi.reshape(-1),
            grad_c1,
            grad_t0,
            grad_t1,
            time,
            batch,
            nchunks,
            CHUNK=chunk,
            enable_fp_fusion=False,
        )
        # grad flows back through the real stacked GEMM (its two halves also halve).
        grad_proj = torch.cat([grad_zr.reshape(filters, -1), grad_zi.reshape(filters, -1)], 0)
        grad_w_stacked = torch.matmul(grad_proj, h_real.t())
        grad_h = torch.matmul(w_stacked.t(), grad_proj)
        grad_C1 = torch.view_as_complex(grad_c1.contiguous())
        grad_tap0 = torch.view_as_complex(grad_t0.contiguous())
        grad_tap1 = torch.view_as_complex(grad_t1.contiguous())
        return grad_w_stacked, grad_h, grad_C1, grad_tap0, grad_tap1, None


def fbt_layer0_real_scan(  # pragma: no cover
    w_stacked: torch.Tensor,
    h_real: torch.Tensor,
    C1: torch.Tensor,
    tap0: torch.Tensor,
    tap1: torch.Tensor,
    batch: int,
) -> torch.Tensor:
    """Real-input layer-0 projection + split-relu prospective scan.

    Fuses the first block's complex projection on a real input into a single
    stacked real GEMM ``[Wr; Wi] @ h`` (half the work of the complex GEMM, since
    the input's imaginary part is zero) feeding the planar recompute scan. Result
    is bit-equivalent (up to fp32 reduction order) to the complex path
    ``fbt_recompute_scan(C1, tap0, tap1, (Wr+iWi) @ h.to(complex))``.

    Args:
        w_stacked: Real ``(2*num_filters, d_model)`` stack of ``[Wr; Wi]``.
        h_real: Real ``(d_model, batch*time)`` projected input.
        C1: Complex ``(num_filters,)`` recurrence coefficient.
        tap0: Complex ``(num_filters,)`` current-sample input tap.
        tap1: Complex ``(num_filters,)`` previous-sample input tap.
        batch: Batch size (to unflatten the ``batch*time`` axis).

    Returns:
        Complex ``(num_filters, batch, time)`` split-relu activated output.
    """
    return _Layer0RealScan.apply(w_stacked, h_real, C1, tap0, tap1, batch)
