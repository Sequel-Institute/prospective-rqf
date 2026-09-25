"""Fused input-dependent ZOH scans on complex (F,B,T) tensors.

Only logits, drive, outputs and chunk entries cross the scan boundary;
coefficient sequences and their VJPs remain in registers. Full and truncated
backwards share a kernel.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from rqf_models.ops.chunked_scan_kernel import _affine_combine, _complex_mul


@triton.jit
def _controller(
    G,
    D,
    W,
    Bias,
    Theta,
    Groups,
    Lo,
    Span,
    Step,
    Horizon,
    ThetaLo,
    ThetaHi,
    f,
    b,
    t,
    B,
    T,
    mask,
    HAS_GAMMA: tl.constexpr,
    HAS_WARP: tl.constexpr,
    HAS_WRITE: tl.constexpr,
    LOG_SHIFT: tl.constexpr,
    GAMMA_SCALE: tl.constexpr,
    WARP_SCALE: tl.constexpr,
    WRITE_SIGMOID: tl.constexpr,
    WRITE_MAX: tl.constexpr,
    WRITE_BIAS: tl.constexpr,
    BOUND_WARP: tl.constexpr,
):
    gamma = tl.load(Bias + f) + tl.full(t.shape, 0.0, tl.float32)
    sig = tl.full(t.shape, 0.0, tl.float32)
    if HAS_GAMMA:
        v = tl.load(G + (f * B + b) * T + t, mask, 0.0)
        if LOG_SHIFT:
            sig = libdevice.tanh(v)
            gamma = tl.exp(gamma + GAMMA_SCALE * sig)
        else:
            sig = tl.sigmoid(gamma + v)
            gamma = tl.load(Lo + f) + tl.load(Span + f) * sig
    theta = tl.load(Theta + f) + tl.full(t.shape, 0.0, tl.float32)
    warp = tl.full(t.shape, 0.0, tl.float32)
    if HAS_WARP:
        group = tl.load(Groups + f)
        warp = libdevice.tanh(tl.load(D + (group * B + b) * T + t, mask, 0.0))
        theta *= tl.exp(WARP_SCALE * warp)
        if BOUND_WARP:
            theta = tl.minimum(tl.maximum(theta, tl.load(ThetaLo + f)), tl.load(ThetaHi + f))
    gate = tl.full(t.shape, 1.0, tl.float32)
    if HAS_WRITE:
        write = tl.load(W + (f * B + b) * T + t, mask, 0.0)
        if WRITE_SIGMOID:
            gate = WRITE_MAX * tl.sigmoid(write + WRITE_BIAS)
        else:
            gate += write
    return gamma, theta, gate, sig, warp, tl.load(Step + f), tl.load(Horizon + f)


@triton.jit
def _zoh(g, th, h):
    ar, ai = -g * h, (1.0 - g) * th * h
    e = tl.exp(ar)
    c, s = tl.cos(ai), tl.sin(ai)
    Ar, Ai = e * c, e * s
    hs = tl.sin(0.5 * ai)
    er, ei = libdevice.expm1(ar) * c - 2.0 * hs * hs, Ai
    den = ar * ar + ai * ai
    safe = tl.where(den < 1.0e-8, 1.0, den)
    fr, fi = (er * ar + ei * ai) / safe, (ei * ar - er * ai) / safe
    sr, si = _complex_mul(ar, ai, 1.0 / 24.0, 0.0)
    sr, si = _complex_mul(ar, ai, 1.0 / 6.0 + sr, si)
    sr, si = _complex_mul(ar, ai, 0.5 + sr, si)
    return Ar, Ai, tl.where(den < 1.0e-8, 1.0 + sr, fr), tl.where(den < 1.0e-8, si, fi)


@triton.jit
def _coefficients(g, th, h, horizon, PROSPECTIVE: tl.constexpr, GAMMA_HORIZON: tl.constexpr):
    Ar, Ai, fr, fi = _zoh(g, th, h)
    pr, pi = g * h * fr, g * h * fi
    nr, ni = tl.full(g.shape, 0.0, tl.float32), tl.full(g.shape, 0.0, tl.float32)
    if PROSPECTIVE:
        ratio = 1.0 / g if GAMMA_HORIZON else horizon
        lr, li = ratio * g * Ar, ratio * g * Ai
        pr, pi, nr, ni = pr + lr, pi + li, -lr, -li
    return Ar, Ai, pr, pi, nr, ni


@triton.jit
def _coefficient_vjp(
    g, th, h, horizon, dAr, dAi, dpr, dpi, dnr, dni,
    PROSPECTIVE: tl.constexpr, GAMMA_HORIZON: tl.constexpr,
):
    Ar, Ai, fr, fi = _zoh(g, th, h)
    dg = tl.full(g.shape, 0.0, tl.float32)
    if PROSPECTIVE:
        ratio = 1.0 / g if GAMMA_HORIZON else horizon
        qr, qi = dpr - dnr, dpi - dni
        # In gamma mode ratio*gamma == 1, including its total derivative.
        if not GAMMA_HORIZON:
            dg = ratio * (qr * Ar + qi * Ai)
        dAr += ratio * g * qr
        dAi += ratio * g * qi
    ar, ai = -g * h, (1.0 - g) * th * h
    den = ar * ar + ai * ai
    safe = tl.where(den < 1.0e-8, 1.0, den)
    xr, xi = Ar - fr, Ai - fi
    fpr, fpi = (xr * ar + xi * ai) / safe, (xi * ar - xr * ai) / safe
    # Stable derivative of exprel, including the reference's cubic small branch.
    sr, si = (
        tl.full(g.shape, 1.0 / 45360.0, tl.float32),
        tl.full(g.shape, 0.0, tl.float32),
    )
    for j in tl.static_range(7):
        sr, si = _complex_mul(ar, ai, sr, si)
        if j == 0:
            sr += 1.0 / 5760.0
        elif j == 1:
            sr += 1.0 / 840.0
        elif j == 2:
            sr += 1.0 / 144.0
        elif j == 3:
            sr += 1.0 / 30.0
        elif j == 4:
            sr += 1.0 / 8.0
        elif j == 5:
            sr += 1.0 / 3.0
        else:
            sr += 0.5
    fpr, fpi = tl.where(den < 0.015625, sr, fpr), tl.where(den < 0.015625, si, fpi)
    cr, ci = _complex_mul(ar, ai, 1.0 / 8.0, 0.0)
    cr, ci = _complex_mul(ar, ai, 1.0 / 3.0 + cr, ci)
    fpr, fpi = tl.where(den < 1.0e-8, 0.5 + cr, fpr), tl.where(den < 1.0e-8, ci, fpi)
    dar = Ar * dAr + Ai * dAi + g * h * (fpr * dpr + fpi * dpi)
    dai = Ar * dAi - Ai * dAr + g * h * (fpr * dpi - fpi * dpr)
    dg += h * (fr * dpr + fi * dpi - dar - th * dai)
    return dg, h * (1.0 - g) * dai


@triton.jit
def _forward(
    Z,
    G,
    D,
    W,
    Bias,
    Theta,
    Groups,
    Lo,
    Span,
    Step,
    Horizon,
    ThetaLo,
    ThetaHi,
    Raw,
    Act,
    Entries,
    B,
    T,
    NC,
    C: tl.constexpr,
    HAS_GAMMA: tl.constexpr,
    HAS_WARP: tl.constexpr,
    HAS_WRITE: tl.constexpr,
    LOG_SHIFT: tl.constexpr,
    GAMMA_SCALE: tl.constexpr,
    WARP_SCALE: tl.constexpr,
    PROSPECTIVE: tl.constexpr,
    GAMMA_HORIZON: tl.constexpr,
    WRITE_SIGMOID: tl.constexpr,
    WRITE_MAX: tl.constexpr,
    WRITE_BIAS: tl.constexpr,
    BOUND_WARP: tl.constexpr,
    ACTIVATE: tl.constexpr,
    WRITE_RAW: tl.constexpr,
):
    row = tl.program_id(0)
    f, b = row // B, row % B
    ix = tl.arange(0, C)
    yr, yi = tl.full((), 0.0, tl.float32), tl.full((), 0.0, tl.float32)
    for chunk in range(NC):
        tl.store(Entries + 2 * (row * NC + chunk), yr)
        tl.store(Entries + 2 * (row * NC + chunk) + 1, yi)
        t = chunk * C + ix
        mask, off = t < T, 2 * (row * T + t)
        g, th, gate, _, _, h, horizon = _controller(
            G,
            D,
            W,
            Bias,
            Theta,
            Groups,
            Lo,
            Span,
            Step,
            Horizon,
            ThetaLo,
            ThetaHi,
            f,
            b,
            t,
            B,
            T,
            mask,
            HAS_GAMMA, HAS_WARP, HAS_WRITE, LOG_SHIFT, GAMMA_SCALE, WARP_SCALE,
            WRITE_SIGMOID, WRITE_MAX, WRITE_BIAS, BOUND_WARP,
        )
        ar, ai, pr, pi, nr, ni = _coefficients(g, th, h, horizon, PROSPECTIVE, GAMMA_HORIZON)
        pr, pi, nr, ni = gate * pr, gate * pi, gate * nr, gate * ni
        zr, zi = tl.load(Z + off, mask, 0.0), tl.load(Z + off + 1, mask, 0.0)
        ur, ui = _complex_mul(pr, pi, zr, zi)
        if PROSPECTIVE:
            zpr = tl.load(Z + off - 2, mask & (t > 0), 0.0)
            zpi = tl.load(Z + off - 1, mask & (t > 0), 0.0)
            lr, li = _complex_mul(nr, ni, zpr, zpi)
            ur, ui = ur + lr, ui + li
        ur, ui = tl.where(mask, ur, 0.0), tl.where(mask, ui, 0.0)
        sr, si, gr, gi = tl.associative_scan(
            (ur, ui, tl.where(mask, ar, 1.0), tl.where(mask, ai, 0.0)),
            axis=0,
            combine_fn=_affine_combine,
        )
        cr, ci = _complex_mul(gr, gi, yr, yi)
        rawr, rawi = sr + cr, si + ci
        if WRITE_RAW:
            tl.store(Raw + off, rawr, mask)
            tl.store(Raw + off + 1, rawi, mask)
        if ACTIVATE:
            tl.store(Act + off, tl.maximum(rawr, 0.0), mask)
            tl.store(Act + off + 1, tl.maximum(rawi, 0.0), mask)
        yr = tl.sum(tl.where(ix == C - 1, rawr, 0.0), axis=0)
        yi = tl.sum(tl.where(ix == C - 1, rawi, 0.0), axis=0)


@triton.jit
def _backward(
    Z,
    G,
    D,
    W,
    Bias,
    Theta,
    Groups,
    Lo,
    Span,
    Step,
    Horizon,
    ThetaLo,
    ThetaHi,
    Entries,
    GY,
    GX,
    DZ,
    DZW,
    DG,
    DD,
    DW,
    Partial,
    B,
    T,
    F,
    NC,
    C: tl.constexpr,
    HAS_GAMMA: tl.constexpr,
    HAS_WARP: tl.constexpr,
    HAS_WRITE: tl.constexpr,
    LOG_SHIFT: tl.constexpr,
    GAMMA_SCALE: tl.constexpr,
    WARP_SCALE: tl.constexpr,
    PROSPECTIVE: tl.constexpr,
    GAMMA_HORIZON: tl.constexpr,
    WRITE_SIGMOID: tl.constexpr,
    WRITE_MAX: tl.constexpr,
    WRITE_BIAS: tl.constexpr,
    BOUND_WARP: tl.constexpr,
    ACTIVATE: tl.constexpr,
    HAS_GY: tl.constexpr,
    HAS_GX: tl.constexpr,
    TRUNCATED: tl.constexpr,
):
    row = tl.program_id(0)
    f, b = row // B, row % B
    ix = tl.arange(0, C)
    dc_r, dc_i = tl.full((), 0.0, tl.float32), tl.full((), 0.0, tl.float32)
    bias_sum, theta_sum = tl.full((), 0.0, tl.float32), tl.full((), 0.0, tl.float32)
    for reverse_chunk in range(NC):
        chunk = NC - 1 - reverse_chunk
        t = chunk * C + ix
        mask, base = t < T, row * T + t
        off = 2 * base
        g, th, gate, sig, warp, h, horizon = _controller(
            G,
            D,
            W,
            Bias,
            Theta,
            Groups,
            Lo,
            Span,
            Step,
            Horizon,
            ThetaLo,
            ThetaHi,
            f,
            b,
            t,
            B,
            T,
            mask,
            HAS_GAMMA, HAS_WARP, HAS_WRITE, LOG_SHIFT, GAMMA_SCALE, WARP_SCALE,
            WRITE_SIGMOID, WRITE_MAX, WRITE_BIAS, BOUND_WARP,
        )
        ar, ai, bpr, bpi, bnr, bni = _coefficients(g, th, h, horizon, PROSPECTIVE, GAMMA_HORIZON)
        pr, pi, nr, ni = gate * bpr, gate * bpi, gate * bnr, gate * bni
        zr, zi = tl.load(Z + off, mask, 0.0), tl.load(Z + off + 1, mask, 0.0)
        zpr, zpi = tl.full((C,), 0.0, tl.float32), tl.full((C,), 0.0, tl.float32)
        if PROSPECTIVE:
            zpr = tl.load(Z + off - 2, mask & (t > 0), 0.0)
            zpi = tl.load(Z + off - 1, mask & (t > 0), 0.0)
        ur, ui = (
            pr * zr - pi * zi + nr * zpr - ni * zpi,
            pr * zi + pi * zr + nr * zpi + ni * zpr,
        )
        ec_r = tl.load(Entries + 2 * (row * NC + chunk))
        ec_i = tl.load(Entries + 2 * (row * NC + chunk) + 1)
        sr, si, gr, gi = tl.associative_scan(
            (
                tl.where(mask, ur, 0.0),
                tl.where(mask, ui, 0.0),
                tl.where(mask, ar, 1.0),
                tl.where(mask, ai, 0.0),
            ),
            axis=0,
            combine_fn=_affine_combine,
        )
        cr, ci = _complex_mul(gr, gi, ec_r, ec_i)
        rawr, rawi = sr + cr, si + ci
        prev = tl.maximum(ix - 1, 0)
        ypr = tl.where(ix == 0, ec_r, tl.gather(rawr, prev, axis=0))
        ypi = tl.where(ix == 0, ec_i, tl.gather(rawi, prev, axis=0))
        er, ei = tl.full((C,), 0.0, tl.float32), tl.full((C,), 0.0, tl.float32)
        if HAS_GY:
            er += tl.load(GY + off, mask, 0.0)
            ei += tl.load(GY + off + 1, mask, 0.0)
        if HAS_GX:
            er += tl.where(rawr > 0.0, tl.load(GX + off, mask, 0.0), 0.0)
            ei += tl.where(rawi > 0.0, tl.load(GX + off + 1, mask, 0.0), 0.0)
        er, ei = tl.where(mask, er, 0.0), tl.where(mask, ei, 0.0)
        # The neighbor at the next chunk's first sample has its own controllers.
        next_t = (chunk + 1) * C
        gn, thn, wn, _, _, hn, rn = _controller(
            G,
            D,
            W,
            Bias,
            Theta,
            Groups,
            Lo,
            Span,
            Step,
            Horizon,
            ThetaLo,
            ThetaHi,
            f,
            b,
            next_t,
            B,
            T,
            next_t < T,
            HAS_GAMMA, HAS_WARP, HAS_WRITE, LOG_SHIFT, GAMMA_SCALE, WARP_SCALE,
            WRITE_SIGMOID, WRITE_MAX, WRITE_BIAS, BOUND_WARP,
        )
        anr, ani, _, _, nnr, nni = _coefficients(gn, thn, hn, rn, PROSPECTIVE, GAMMA_HORIZON)
        nx = tl.minimum(ix + 1, C - 1)
        a1r = tl.where(ix == C - 1, anr, tl.gather(ar, nx, axis=0))
        a1i = tl.where(ix == C - 1, ani, tl.gather(ai, nx, axis=0))
        n1r = tl.where(ix == C - 1, wn * nnr, tl.gather(nr, nx, axis=0))
        n1i = tl.where(ix == C - 1, wn * nni, tl.gather(ni, nx, axis=0))
        a1r, a1i = tl.where(t + 1 < T, a1r, 0.0), tl.where(t + 1 < T, a1i, 0.0)
        n1r, n1i = tl.where(t + 1 < T, n1r, 0.0), tl.where(t + 1 < T, n1i, 0.0)
        if TRUNCATED:
            dr, di = er, ei
        else:
            dsr, dsi, dgr, dgi = tl.associative_scan(
                (er, ei, tl.where(mask, a1r, 1.0), tl.where(mask, -a1i, 0.0)),
                axis=0,
                combine_fn=_affine_combine,
                reverse=True,
            )
            dcr, dci = _complex_mul(dgr, dgi, dc_r, dc_i)
            dr, di = dsr + dcr, dsi + dci
        dnextr = tl.where(ix == C - 1, dc_r, tl.gather(dr, nx, axis=0))
        dnexti = tl.where(ix == C - 1, dc_i, tl.gather(di, nx, axis=0))
        dc_r = tl.sum(tl.where(ix == 0, dr, 0.0), axis=0)
        dc_i = tl.sum(tl.where(ix == 0, di, 0.0), axis=0)
        dzr, dzi = pr * dr + pi * di, pr * di - pi * dr
        dwr, dwi = dzr, dzi
        if PROSPECTIVE:
            dwr += n1r * dnextr + n1i * dnexti
            dwi += n1r * dnexti - n1i * dnextr
        if TRUNCATED:
            tl.store(DZ + off, dzr, mask)
            tl.store(DZ + off + 1, dzi, mask)
        tl.store(DZW + off, dwr, mask)
        tl.store(DZW + off + 1, dwi, mask)
        dpr, dpi = zr * dr + zi * di, zr * di - zi * dr
        dnr, dni = zpr * dr + zpi * di, zpr * di - zpi * dr
        if HAS_WRITE:
            dgate = bpr * dpr + bpi * dpi + bnr * dnr + bni * dni
            if WRITE_SIGMOID:
                dgate *= gate * (1.0 - gate / WRITE_MAX)
            tl.store(DW + base, dgate, mask)
        dg, dth = _coefficient_vjp(
            g,
            th,
            h,
            horizon,
            ypr * dr + ypi * di,
            ypr * di - ypi * dr,
            gate * dpr,
            gate * dpi,
            gate * dnr,
            gate * dni,
            PROSPECTIVE, GAMMA_HORIZON,
        )
        if HAS_GAMMA:
            if LOG_SHIFT:
                dg *= g
                tl.store(DG + base, dg * GAMMA_SCALE * (1.0 - sig * sig), mask)
            else:
                dg *= tl.load(Span + f) * sig * (1.0 - sig)
                tl.store(DG + base, dg, mask)
        if HAS_WARP:
            warp_scale = tl.exp(WARP_SCALE * warp)
            if BOUND_WARP:
                unclamped_theta = tl.load(Theta + f) * warp_scale
                # torch.clamp propagates the input gradient at both exact edges.
                inside = (unclamped_theta >= tl.load(ThetaLo + f)) & (
                    unclamped_theta <= tl.load(ThetaHi + f)
                )
                dth = tl.where(inside, dth, 0.0)
            tl.store(DD + base, dth * th * WARP_SCALE * (1.0 - warp * warp), mask)
            dth *= warp_scale
        bias_sum += tl.sum(tl.where(mask, dg, 0.0), axis=0)
        theta_sum += tl.sum(tl.where(mask, dth, 0.0), axis=0)
    tl.store(Partial + row, bias_sum)
    tl.store(Partial + F * B + row, theta_sum)


@triton.jit
def _group_reduce(
    Src,
    Groups,
    Dst,
    N: tl.constexpr,
    F: tl.constexpr,
    BF: tl.constexpr,
    BT: tl.constexpr,
):
    group = tl.program_id(0)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    f = tl.arange(0, BF)
    ids = tl.load(Groups + f, f < F, -1)
    values = tl.load(
        Src + f[:, None] * N + t[None, :],
        (f[:, None] < F) & (t[None, :] < N) & (ids[:, None] == group),
        0.0,
    )
    tl.store(Dst + group * N + t, tl.sum(values, axis=0), t < N)


def _real(tensor):
    return None if tensor is None else torch.view_as_real(tensor.resolve_conj())


def selective_forward(z, inputs, activate, write_raw, chunk):
    f, b, t = z.shape
    nc = triton.cdiv(t, chunk)
    raw = torch.empty_like(z) if write_raw else None
    act = torch.empty_like(z) if activate else None
    entries = torch.empty((f * b, nc), dtype=z.dtype, device=z.device)
    _forward[(f * b,)](
        _real(z),
        *inputs[:-1],
        _real(raw),
        _real(act),
        _real(entries),
        b,
        t,
        nc,
        chunk,
        # Triton 3.3 cannot subscript a constexpr tuple in a JIT helper.
        # Pass the controller options as individual compile-time scalars.
        *inputs.options,
        activate,
        write_raw,
        num_warps=4,
    )
    return raw, act, entries


def selective_backward(z, inputs, entries, gy, gx, activate, truncated, chunk):
    f, b, t = z.shape
    opt = inputs.options
    dz = torch.empty_like(z)
    dzw = torch.empty_like(z) if truncated else dz
    dg = torch.empty_like(inputs.gamma_logits) if opt[0] else None
    dd = (
        torch.empty((f, b, t), device=z.device, dtype=torch.float32) if opt[1] else None
    )
    dw = torch.empty_like(inputs.write_logits) if opt[2] else None
    partial = torch.empty((2, f, b), device=z.device, dtype=torch.float32)
    gy = None if gy is None else gy.contiguous()
    gx = None if gx is None else gx.contiguous()
    _backward[(f * b,)](
        _real(z),
        *inputs[:-1],
        _real(entries),
        _real(gy),
        _real(gx),
        _real(dz),
        _real(dzw),
        dg,
        dd,
        dw,
        partial,
        b,
        t,
        f,
        entries.shape[1],
        chunk,
        *opt,
        activate,
        gy is not None,
        gx is not None,
        truncated,
        num_warps=4,
    )
    group_grad = None
    if dd is not None:
        group_grad = torch.empty_like(inputs.warp_logits)
        _group_reduce[(group_grad.shape[0], triton.cdiv(b * t, 64))](
            dd, inputs.groups, group_grad, b * t, f, triton.next_power_of_2(f), 64
        )
    dbias, dtheta = partial.sum(dim=2, dtype=torch.float64).float()
    return dz, dzw, dg, group_grad, dw, dbias, dtheta
