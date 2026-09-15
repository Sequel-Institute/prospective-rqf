"""Reference selective recurrence and autograd boundary for fused CUDA kernels."""

import os
from typing import NamedTuple

import torch
from torch import Tensor


class SelectiveScanInputs(NamedTuple):
    gamma_logits: Tensor
    warp_logits: Tensor
    write_logits: Tensor
    bias: Tensor
    theta: Tensor
    groups: Tensor | None
    minimum: Tensor
    span: Tensor
    step: Tensor
    horizon: Tensor
    options: tuple


def selective_coefficients(
    inputs: SelectiveScanInputs,
) -> tuple[Tensor, Tensor, Tensor]:
    """Explicit ``(A, gated p, gated n)`` in ``(F,B,T)`` layout."""
    gl, dl, wl, bias, theta, groups, minimum, span, step, horizon, opt = inputs
    gamma = bias[:, None, None]
    if opt[0]:
        gamma = (
            torch.exp(gamma + opt[4] * gl.tanh())
            if opt[3]
            else minimum[:, None, None]
            + span[:, None, None] * torch.sigmoid(gamma + gl)
        )
    theta = theta[:, None, None]
    if opt[1]:
        theta = theta * torch.exp(opt[5] * dl.tanh().index_select(0, groups))
    eta = step[:, None, None]
    a = torch.complex(-gamma, (1 - gamma) * theta) * eta
    A = a.exp()
    small = a.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(a), a)
    phi = torch.where(
        small, 1 + a * (0.5 + a * (1 / 6 + a / 24)), torch.expm1(a) / safe
    )
    p = gamma * eta * phi
    n = torch.zeros_like(p)
    if opt[6]:
        ratio = 1 / gamma if opt[7] else horizon[:, None, None]
        extra = ratio * gamma * A
        p, n = p + extra, -extra
    if opt[2]:
        p, n = p * (1 + wl), n * (1 + wl)
    # Frequency-only and write-only controllers may leave A independent of time.
    shape = (bias.numel(), gl.shape[1], gl.shape[2])
    return A.expand(shape), p.expand(shape), n.expand(shape)


def reference_layer_scan(
    x: Tensor,
    weight: Tensor,
    inputs: SelectiveScanInputs,
    *,
    truncated: bool,
    activate: bool,
    write_raw: bool,
) -> tuple[Tensor | None, Tensor | None]:
    """Independent serial autograd reference on CPU, MPS, or CUDA."""
    f_in, batch, time = x.shape
    z = (weight @ x.reshape(f_in, batch * time)).reshape(-1, batch, time)
    lag_z = (
        (weight @ x.detach().reshape(f_in, batch * time)).reshape_as(z)
        if truncated and inputs.options[6]
        else z
    )
    A, p, n = selective_coefficients(inputs)
    previous = z.new_zeros(z.shape[:2])
    values = []
    for t in range(time):
        previous = (
            A[..., t] * (previous.detach() if truncated else previous)
            + p[..., t] * z[..., t]
        )
        if t and inputs.options[6]:
            previous = previous + n[..., t] * lag_z[..., t - 1]
        values.append(previous)
    raw = torch.stack(values, dim=-1) if values else z * 0
    activated = torch.complex(raw.real.relu(), raw.imag.relu()) if activate else None
    return raw if write_raw else None, activated


class _SelectiveLayerScan(torch.autograd.Function):
    """Own the drive GEMM so TBPTT can send the lag cotangent to W only."""

    @staticmethod
    def forward(ctx, x, weight, *args):
        from rqf_models.ops.selective_scan_kernel import selective_forward

        *tensors, options, truncated, activate, write_raw, chunk = args
        inputs = SelectiveScanInputs(*tensors, options)
        f_in, batch, time = x.shape
        with torch.autocast(x.device.type, enabled=False):
            z = (weight @ x.reshape(f_in, batch * time)).reshape(-1, batch, time)
        raw, act, entries = selective_forward(z, inputs, activate, write_raw, chunk)
        ctx.save_for_backward(x, weight, z, entries, *tensors)
        ctx.options, ctx.truncated = options, truncated
        ctx.activate, ctx.chunk = activate, chunk
        ctx.set_materialize_grads(False)
        return raw, act

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_raw, grad_act):
        from rqf_models.ops.selective_scan_kernel import selective_backward

        x, weight, z, entries, *tensors = ctx.saved_tensors
        inputs = SelectiveScanInputs(*tensors, ctx.options)
        dx_drive, dw_drive, dg, dd, dw, dbias, dtheta = selective_backward(
            z,
            inputs,
            entries,
            grad_raw,
            grad_act,
            ctx.activate,
            ctx.truncated,
            ctx.chunk,
        )
        f_in = x.shape[0]
        with torch.autocast(x.device.type, enabled=False):
            dx = (weight.mH @ dx_drive.reshape(weight.shape[0], -1)).reshape_as(x)
            dweight = dw_drive.reshape(weight.shape[0], -1) @ x.reshape(f_in, -1).mH
        grads = (
            dx,
            dweight,
            dg,
            dd,
            dw,
            dbias,
            dtheta,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        return tuple(
            g if need else None for g, need in zip(grads, ctx.needs_input_grad)
        )


def selective_layer_scan(
    x: Tensor,
    weight: Tensor,
    inputs: SelectiveScanInputs,
    *,
    truncated: bool = False,
    activate: bool = True,
    write_raw: bool = True,
    use_kernel: bool = True,
    chunk: int | None = None,
) -> tuple[Tensor | None, Tensor | None]:
    """Complex ``(I,B,T)`` layer, preserving the reference on unsupported devices."""
    if not activate and not write_raw:
        raise ValueError("selective scan requires at least one output")
    device_ok = x.is_cuda or (
        x.device.type == "cpu" and os.environ.get("TRITON_INTERPRET") == "1"
    )
    eligible = (
        use_kernel
        and device_ok
        and x.dtype == torch.complex64
        and weight.dtype == torch.complex64
        and inputs.bias.dtype == torch.float32
        and x.shape[-1] > 0
    )
    if eligible:
        try:
            import triton
        except ImportError:
            eligible = False
    if not eligible:
        return reference_layer_scan(
            x,
            weight,
            inputs,
            truncated=truncated,
            activate=activate,
            write_raw=write_raw,
        )
    if chunk is None:
        chunk = min(256, triton.next_power_of_2(x.shape[-1]))
    if chunk < 1 or chunk & (chunk - 1):
        raise ValueError("chunk must be a positive power of two")
    tensors = [v.contiguous() if isinstance(v, Tensor) else v for v in inputs[:-1]]
    return _SelectiveLayerScan.apply(
        x.contiguous(),
        weight,
        *tensors,
        inputs.options,
        truncated,
        activate,
        write_raw,
        chunk,
    )
