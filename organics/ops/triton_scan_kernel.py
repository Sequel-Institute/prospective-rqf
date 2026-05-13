"""Triton GPU kernels for linear recurrence scans.

This module provides GPU-accelerated implementations of the standard complex
linear recurrence: x[n+1] = C1*x[n] + C2*u[n].

All scans implement THE SAME standard complex recurrence. Coefficients
(C1, C2) and state x are always complex. The ``_real`` variants are a
NUMERICAL specialization of that same recurrence for the case Im(u) == 0:
they avoid materializing an all-zero imaginary input buffer and fold the
``c2r*u_imag`` / ``c2i*u_imag`` terms to zero at compile time. The output
is still complex and numerically identical (up to float roundoff) to the
complex-input path with ``u + 0j``.

This is NOT the deprecated "real-input RQF equation" with an extra
imaginary term / modified Im(x) feedback. That variant complicated the
local-learning derivation and produced non-symmetric filters. The
project-wide rule is: "one equation only -- the complex one; if u is
real, treat it as u + 0j."

Available Implementations
-------------------------

Sequential Scan (DEFAULT - Fastest for most cases):
    ``_triton_sequential_scan()`` - O(T) sequential, O(1) extra memory.
    Single kernel with state in registers. Best for T < 50,000 (typical use cases).
    ``_triton_sequential_scan_real()`` - real-input specialization of the same
    recurrence (complex state/output, no redundant imaginary-input traffic).

Sequential Scan with B Matrix:
    ``_triton_sequential_scan_b_matrix()`` - O(T) sequential for complex inputs
    with input projection through B matrix. Implements:
    x[n] = C1*x[n-1] + (C2 ⊙ B) @ u[n]
    Supports input_features <= 1024.

Fused Multi-Oscillator Sequential Scan:
    ``_triton_sequential_scan_multi_osc()`` - O(T) sequential, fuses N
    independent scans sharing the same input u per physical filter, with
    C-weight reduction. N× less memory than N separate scans.
    ``_triton_sequential_scan_multi_osc_real()`` - real-input specialization
    of the same fused multi-oscillator recurrence.

Parallel Scan (Blelloch):
    ``_triton_parallel_scan()`` - O(log T) parallel depth, single kernel.
    Uses ``tl.associative_scan`` (Blelloch algorithm).
    Best for T <= 16,384 when you need parallel execution.
    ``_triton_parallel_scan_real()`` - parallel real-input specialization.

Fused Multi-Oscillator Parallel Scan:
    ``_triton_parallel_scan_multi_osc()`` - O(log T) parallel, fuses N
    independent scans sharing the same input u per physical filter, with
    C-weight reduction. Uses 2D ``tl.associative_scan`` on (T, N) blocks.
    N× less memory than N separate scans.

Example:
    >>> from organics.ops.triton_scan_kernel import (
    ...     _triton_sequential_scan,
    ...     _triton_parallel_scan,
    ... )
    >>> # Default: O(T) sequential for complex inputs
    >>> output = _triton_sequential_scan(C1, C2, u_complex)
    >>> # Parallel: O(log T) using associative scan
    >>> output = _triton_parallel_scan(C1, C2, u)

References:
    - Blelloch (1990): "Prefix Sums and Their Applications"
    - Mamba: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

# ==============================================================================
# Section 1: Input Preparation Helpers
# ==============================================================================


@dataclass
class ScanInputs:
    """Prepared inputs for scan kernels after validation and dtype conversion.

    Attributes:
        c1_real: Real part of C1 coefficients (num_filters,)
        c1_imag: Imaginary part of C1 coefficients (num_filters,)
        c2_real: Real part of C2 coefficients (num_filters,)
        c2_imag: Imaginary part of C2 coefficients (num_filters,)
        u: Input tensor after dtype conversion (batch, time, features)
        u_real: Real part of u if complex, or u if real
        u_imag: Imaginary part of u if complex, or None if real
        batch_size: Batch dimension
        time_size: Time dimension
        filter_size: Filter/feature dimension
    """

    c1_real: torch.Tensor
    c1_imag: torch.Tensor
    c2_real: torch.Tensor
    c2_imag: torch.Tensor
    u: torch.Tensor
    u_real: torch.Tensor
    u_imag: torch.Tensor | None
    batch_size: int
    time_size: int
    filter_size: int


def _prepare_scan_inputs(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    require_complex_u: bool = True,
) -> ScanInputs:
    """Prepare inputs for scan kernels (validation, dtype conversion, contiguity).

    This helper consolidates the common input preparation logic used by all
    selective scan variants: ensuring correct dtypes, making tensors contiguous,
    and splitting complex coefficients into real/imaginary parts.

    Args:
        C1: Recurrence coefficient tensor (num_filters,)
        C2: Input gain tensor (num_filters,)
        u: Input signal tensor (batch, time, features)
        require_complex_u: If True, convert u to complex. If False, keep u real.

    Returns:
        ScanInputs dataclass with prepared tensors and shape info.
    """
    batch_size, time_size, filter_size = u.shape

    # Ensure coefficients are complex (always required)
    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)

    # Handle u dtype based on requirement
    if require_complex_u:
        if not u.is_complex():
            u = u.to(torch.cfloat)
    else:
        # Real input expected - extract real part if complex
        if u.is_complex():
            u = u.real

    # Make all tensors contiguous
    u = u.contiguous()
    C1 = C1.contiguous()
    C2 = C2.contiguous()

    # Split coefficients into real/imaginary parts
    # For real u, match the u dtype; for complex u, use base dtype
    target_dtype = u.dtype if not u.is_complex() else u.real.dtype
    c1_real = C1.real.to(target_dtype).contiguous()
    c1_imag = C1.imag.to(target_dtype).contiguous()
    c2_real = C2.real.to(target_dtype).contiguous()
    c2_imag = C2.imag.to(target_dtype).contiguous()

    # Split u if complex
    if u.is_complex():
        u_real = u.real.contiguous()
        u_imag = u.imag.contiguous()
    else:
        u_real = u
        u_imag = None

    return ScanInputs(
        c1_real=c1_real,
        c1_imag=c1_imag,
        c2_real=c2_real,
        c2_imag=c2_imag,
        u=u,
        u_real=u_real,
        u_imag=u_imag,
        batch_size=batch_size,
        time_size=time_size,
        filter_size=filter_size,
    )


_SEQUENTIAL_SCAN_BLOCK_F_CONFIGS = [
    triton.Config({"BLOCK_F": 1}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_F": 2}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_F": 4}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_F": 8}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_F": 16}, num_warps=8, num_stages=1),
]

_SEQUENTIAL_SCAN_BLOCK_F_GRAD_RESET_ARGS = [
    "grad_c1r_ptr",
    "grad_c1i_ptr",
    "grad_c2r_ptr",
    "grad_c2i_ptr",
]


# ==============================================================================
# Section 2: Triton JIT Kernels
#
# These are low-level GPU kernels compiled with Triton. Each kernel handles a
# specific scan variant:
#   - _sequential_scan_kernel: Standard complex recurrence
#   - _sequential_scan_b_matrix_kernel: Complex inputs with B matrix
#   - _sequential_scan_multi_osc_kernel: Fused multi-oscillator scan
#   - _associative_combine: Combine operator for parallel scan
#   - _parallel_scan_kernel: Parallel scan using Blelloch algorithm
# ==============================================================================


@triton.jit
def _sequential_scan_kernel(  # pragma: no cover
    # Input pointers
    u_real_ptr,
    u_imag_ptr,
    # Output pointers
    out_real_ptr,
    out_imag_ptr,
    # Coefficient pointers (shape: num_filters)
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    # Dimensions
    batch_size,
    time_size,
    filter_size,
    # Strides
    stride_batch,
    stride_time,
    stride_filter,
    # Block size (unused but required for API)
    BLOCK_SIZE: tl.constexpr,  # noqa: ARG001
    EPILOGUE: tl.constexpr,
):
    """Selective scan kernel - O(T) sequential recurrence.

    Computes x[n+1] = C1 * x[n] + C2 * u[n] for all n.
    Each program processes one (batch, filter) pair sequentially.
    State is kept in registers for maximum efficiency.

    ``EPILOGUE`` is a compile-time switch selecting what lands in HBM:
      0 = COMPLEX: write x_real to out_real_ptr, x_imag to out_imag_ptr.
      1 = ABS:     write sqrt(x_real**2 + x_imag**2) to out_real_ptr only.
                   out_imag_ptr is not dereferenced.

    Note: BLOCK_F tiling (vector state over a filter chunk) was benchmarked on
    A10G for B=8/F=32 and came out neutral-to-worse — the resulting 8 CTAs
    under-fill the device vs the 256 CTAs of the scalar path. Scalar per
    (batch, filter) is the right default for single-osc workloads.
    """
    pid = tl.program_id(0)

    batch_idx = pid // filter_size
    filter_idx = pid % filter_size

    if batch_idx >= batch_size:
        return

    # Load coefficients (constant across time)
    c1_real = tl.load(c1_real_ptr + filter_idx)
    c1_imag = tl.load(c1_imag_ptr + filter_idx)
    c2_real = tl.load(c2_real_ptr + filter_idx)
    c2_imag = tl.load(c2_imag_ptr + filter_idx)

    # State in registers (initialize with correct dtype via load-subtract trick)
    first_offset = batch_idx * stride_batch + 0 * stride_time + filter_idx * stride_filter
    first_u = tl.load(u_real_ptr + first_offset)
    x_real = first_u - first_u  # Zero with correct dtype
    x_imag = first_u - first_u  # Zero with correct dtype

    # Process all timesteps sequentially
    for t in range(time_size):
        offset = batch_idx * stride_batch + t * stride_time + filter_idx * stride_filter

        # Load input
        u_r = tl.load(u_real_ptr + offset)
        u_i = tl.load(u_imag_ptr + offset)

        # Complex multiply: C1 * x
        c1x_real = c1_real * x_real - c1_imag * x_imag
        c1x_imag = c1_real * x_imag + c1_imag * x_real

        # Complex multiply: C2 * u
        c2u_real = c2_real * u_r - c2_imag * u_i
        c2u_imag = c2_real * u_i + c2_imag * u_r

        # Update state: x_new = C1*x + C2*u
        x_real = c1x_real + c2u_real
        x_imag = c1x_imag + c2u_imag

        if EPILOGUE == 0:
            tl.store(out_real_ptr + offset, x_real)
            tl.store(out_imag_ptr + offset, x_imag)
        else:
            tl.store(out_real_ptr + offset, tl.sqrt(x_real * x_real + x_imag * x_imag))


# ==============================================================================
# Section 3: Complex Input Selective Scan
#
# Standard recurrence: x[n] = C1*x[n-1] + C2*u[n]
# - _sequential_scan_forward_raw: Calls kernel, no autograd
# - _SequentialScanFunction: Autograd wrapper
# - _triton_sequential_scan: Public API with autograd
# ==============================================================================


def _sequential_scan_forward_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Raw forward pass for selective scan - calls Triton kernel directly.

    This is an internal function used by the autograd wrapper.
    It does NOT support autograd - use _triton_sequential_scan instead.
    """
    # Prepare inputs (validation, dtype conversion, contiguity)
    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=True)

    # Allocate output
    out_real = torch.zeros_like(inputs.u_real)
    out_imag = torch.zeros_like(inputs.u_imag)

    # Calculate strides
    stride_batch = inputs.u_real.stride(0)
    stride_time = inputs.u_real.stride(1)
    stride_filter = inputs.u_real.stride(2)

    # Scalar grid: one program per (batch, filter). See kernel docstring — BLOCK_F
    # tiling was benchmarked and dropped because it under-fills the SMs on the
    # single-osc F=32 workload we target.
    num_programs = inputs.batch_size * inputs.filter_size
    grid = (num_programs,)

    _sequential_scan_kernel[grid](
        inputs.u_real,
        inputs.u_imag,
        out_real,
        out_imag,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        inputs.batch_size,
        inputs.time_size,
        inputs.filter_size,
        stride_batch,
        stride_time,
        stride_filter,
        BLOCK_SIZE=256,
        EPILOGUE=0,
    )

    return torch.complex(out_real, out_imag)


def _sequential_scan_forward_abs_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Fused-``abs()`` epilogue variant of ``_sequential_scan_forward_raw``.

    Returns a real tensor ``|x[t]|`` directly, skipping the imaginary half of
    the output. Half the output HBM traffic vs the complex path, intended for
    eval / no_grad consumers — backward still needs both halves so this has
    no autograd wrapper; training stays on the complex path.
    """
    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=True)

    out = torch.empty_like(inputs.u_real)
    out_imag_dummy = torch.empty(1, device=inputs.u_real.device, dtype=inputs.u_real.dtype)

    num_programs = inputs.batch_size * inputs.filter_size
    grid = (num_programs,)

    _sequential_scan_kernel[grid](
        inputs.u_real,
        inputs.u_imag,
        out,
        out_imag_dummy,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        inputs.batch_size,
        inputs.time_size,
        inputs.filter_size,
        inputs.u_real.stride(0),
        inputs.u_real.stride(1),
        inputs.u_real.stride(2),
        BLOCK_SIZE=256,
        EPILOGUE=1,
    )

    return out


@triton.autotune(configs=_SEQUENTIAL_SCAN_BLOCK_F_CONFIGS, key=["time_size", "filter_size"])
@triton.jit
def _sequential_scan_block_f_kernel(  # pragma: no cover
    u_real_ptr,
    u_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    batch_size,
    time_size,
    filter_size,
    stride_batch,
    stride_time,
    stride_filter,
    BLOCK_F: tl.constexpr,
    EPILOGUE: tl.constexpr,
):
    """Vectorized single-oscillator recurrence over a contiguous filter chunk.

    Each program handles one ``(batch, filter_block)`` pair and keeps
    ``BLOCK_F`` complex states in registers. ``BLOCK_F=1`` degenerates to the
    scalar path and is included in the autotune search as a safe baseline.
    """
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    if batch_idx >= batch_size:
        return

    f_offsets = block_idx * BLOCK_F + tl.arange(0, BLOCK_F)
    f_mask = f_offsets < filter_size

    c1_real = tl.load(c1_real_ptr + f_offsets, mask=f_mask, other=0.0)
    c1_imag = tl.load(c1_imag_ptr + f_offsets, mask=f_mask, other=0.0)
    c2_real = tl.load(c2_real_ptr + f_offsets, mask=f_mask, other=0.0)
    c2_imag = tl.load(c2_imag_ptr + f_offsets, mask=f_mask, other=0.0)

    first_offset = batch_idx * stride_batch + f_offsets * stride_filter
    first_u = tl.load(u_real_ptr + first_offset, mask=f_mask, other=0.0)
    x_real = first_u - first_u
    x_imag = first_u - first_u

    for t in range(time_size):
        offset = batch_idx * stride_batch + t * stride_time + f_offsets * stride_filter

        u_r = tl.load(u_real_ptr + offset, mask=f_mask, other=0.0)
        u_i = tl.load(u_imag_ptr + offset, mask=f_mask, other=0.0)

        c1x_real = c1_real * x_real - c1_imag * x_imag
        c1x_imag = c1_real * x_imag + c1_imag * x_real

        c2u_real = c2_real * u_r - c2_imag * u_i
        c2u_imag = c2_real * u_i + c2_imag * u_r

        x_real = c1x_real + c2u_real
        x_imag = c1x_imag + c2u_imag

        if EPILOGUE == 0:
            tl.store(out_real_ptr + offset, x_real, mask=f_mask)
            tl.store(out_imag_ptr + offset, x_imag, mask=f_mask)
        else:
            tl.store(
                out_real_ptr + offset,
                tl.sqrt(x_real * x_real + x_imag * x_imag),
                mask=f_mask,
            )


def _sequential_scan_forward_raw_block_f(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Autotuned BLOCK_F forward for the standard complex recurrence."""
    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=True)

    out_real = torch.zeros_like(inputs.u_real)
    out_imag = torch.zeros_like(inputs.u_imag)

    def grid(META: dict) -> tuple[int, int]:
        return (inputs.batch_size, triton.cdiv(inputs.filter_size, META["BLOCK_F"]))

    _sequential_scan_block_f_kernel[grid](
        inputs.u_real,
        inputs.u_imag,
        out_real,
        out_imag,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        inputs.batch_size,
        inputs.time_size,
        inputs.filter_size,
        inputs.u_real.stride(0),
        inputs.u_real.stride(1),
        inputs.u_real.stride(2),
        EPILOGUE=0,
    )

    return torch.complex(out_real, out_imag)


def _sequential_scan_forward_abs_raw_block_f(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """BLOCK_F fused-``abs()`` epilogue for inference-only consumers."""
    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=True)

    out = torch.empty_like(inputs.u_real)
    out_imag_dummy = torch.empty(1, device=inputs.u_real.device, dtype=inputs.u_real.dtype)

    def grid(META: dict) -> tuple[int, int]:
        return (inputs.batch_size, triton.cdiv(inputs.filter_size, META["BLOCK_F"]))

    _sequential_scan_block_f_kernel[grid](
        inputs.u_real,
        inputs.u_imag,
        out,
        out_imag_dummy,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        inputs.batch_size,
        inputs.time_size,
        inputs.filter_size,
        inputs.u_real.stride(0),
        inputs.u_real.stride(1),
        inputs.u_real.stride(2),
        EPILOGUE=1,
    )

    return out


@triton.jit
def _sequential_scan_backward_kernel(  # pragma: no cover
    # Forward inputs (shape: batch, time, num_filters)
    u_real_ptr,
    u_imag_ptr,
    # Forward outputs (shape: batch, time, num_filters) — saved in ctx
    out_real_ptr,
    out_imag_ptr,
    # Coefficient pointers (shape: num_filters)
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    # Grad output (shape: batch, time, num_filters)
    grad_out_real_ptr,
    grad_out_imag_ptr,
    # Output: grad_u (shape: batch, time, num_filters)
    grad_u_real_ptr,
    grad_u_imag_ptr,
    # Output: coefficient grads (shape: num_filters) — atomic add targets
    grad_c1r_ptr,
    grad_c1i_ptr,
    grad_c2r_ptr,
    grad_c2i_ptr,
    # Dimensions
    batch_size,
    time_size,
    filter_size,
    # Strides for (B,T,F) tensors (u, output, grad_out, grad_u share layout)
    stride_batch,
    stride_time,
    stride_filter,
    # Block size (unused but required for API)
    BLOCK_SIZE: tl.constexpr,  # noqa: ARG001
    # Compile-time gating: when False, the matching grad is never touched. Driven
    # by ``ctx.needs_input_grad`` so frozen-recurrent fine-tuning pays no
    # atomic_add contention for C1/C2 and skips the T-length u/out_prev loads
    # when grad_C2 / grad_C1 aren't required.
    COMPUTE_GRAD_C1: tl.constexpr,
    COMPUTE_GRAD_C2: tl.constexpr,
    COMPUTE_GRAD_U: tl.constexpr,
):
    """Fused backward kernel for complex selective scan.

    Single-pass adjoint sweep (t=T-1..0). No Phase-1 forward recompute:
    the forward output x[t] is already saved by the autograd function and
    passed via ``out_real/imag``; ``x[t-1]`` is read directly from it for
    the grad_C1 accumulation. This removes a full O(T) forward loop plus
    the scratch alloc used to re-materialize x.

    Each program handles one (batch, filter) pair. Scalar complex state.
    Tried BLOCK_F tiling here (vector state + vector atomic_add); on the
    SC scale (B=8, F=32, A10G) it was 30–40% SLOWER than the scalar path
    because the resulting 8 CTAs (B*F/BLOCK_F) massively under-fill the
    A10G's 80 SMs — the scalar path keeps 256 CTAs pipelined across the
    device. Keep scalar for the typical single-osc workload; revisit if a
    large-F path shows up.
    """
    pid = tl.program_id(0)
    batch_idx = pid // filter_size
    filter_idx = pid % filter_size

    if batch_idx >= batch_size:
        return

    # Load coefficients (constant across time)
    c1r = tl.load(c1_real_ptr + filter_idx)
    c1i = tl.load(c1_imag_ptr + filter_idx)
    c2r = tl.load(c2_real_ptr + filter_idx)
    c2i = tl.load(c2_imag_ptr + filter_idx)

    # Zero in correct dtype via load-subtract trick
    first_offset = batch_idx * stride_batch + 0 * stride_time + filter_idx * stride_filter
    first_u = tl.load(u_real_ptr + first_offset)
    zero = first_u - first_u

    # ====== Adjoint sweep (t=T-1..0), single pass ======
    adj_r = zero
    adj_i = zero

    # Accumulators for coefficient grads (summed over time, batch via atomic)
    gc1r = zero
    gc1i = zero
    gc2r = zero
    gc2i = zero

    for t_rev in range(time_size):
        t = time_size - 1 - t_rev
        offset = batch_idx * stride_batch + t * stride_time + filter_idx * stride_filter

        # Load grad_output[t]
        go_r = tl.load(grad_out_real_ptr + offset)
        go_i = tl.load(grad_out_imag_ptr + offset)

        # Adjoint: adj[t] = grad[t] + conj(C1) * adj[t+1]
        # conj(C1) * adj = (c1r*adj_r + c1i*adj_i) + j*(c1r*adj_i - c1i*adj_r)
        adj_r_new = c1r * adj_r + c1i * adj_i + go_r
        adj_i_new = c1r * adj_i - c1i * adj_r + go_i

        if COMPUTE_GRAD_U:
            # grad_u[t] = conj(C2) * adj[t]
            # = (c2r*adj_r + c2i*adj_i) + j*(c2r*adj_i - c2i*adj_r)
            gu_r = c2r * adj_r_new + c2i * adj_i_new
            gu_i = c2r * adj_i_new - c2i * adj_r_new
            tl.store(grad_u_real_ptr + offset, gu_r)
            tl.store(grad_u_imag_ptr + offset, gu_i)

        if COMPUTE_GRAD_C2:
            # grad_C2 += adj[t] * conj(u[t])
            ur = tl.load(u_real_ptr + offset)
            ui = tl.load(u_imag_ptr + offset)
            gc2r += adj_r_new * ur + adj_i_new * ui
            gc2i += adj_i_new * ur - adj_r_new * ui

        # grad_C1 += adj[t] * conj(x[t-1]) for t > 0, reading saved output
        if COMPUTE_GRAD_C1 and t > 0:
            offset_prev = (
                batch_idx * stride_batch + (t - 1) * stride_time + filter_idx * stride_filter
            )
            xr_prev = tl.load(out_real_ptr + offset_prev)
            xi_prev = tl.load(out_imag_ptr + offset_prev)
            gc1r += adj_r_new * xr_prev + adj_i_new * xi_prev
            gc1i += adj_i_new * xr_prev - adj_r_new * xi_prev

        adj_r = adj_r_new
        adj_i = adj_i_new

    # Atomic add for coefficient grads (batch dimension reduction)
    if COMPUTE_GRAD_C1:
        tl.atomic_add(grad_c1r_ptr + filter_idx, gc1r)
        tl.atomic_add(grad_c1i_ptr + filter_idx, gc1i)
    if COMPUTE_GRAD_C2:
        tl.atomic_add(grad_c2r_ptr + filter_idx, gc2r)
        tl.atomic_add(grad_c2i_ptr + filter_idx, gc2i)


def _sequential_scan_backward_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    compute_grad_C1: bool = True,
    compute_grad_C2: bool = True,
    compute_grad_u: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Raw backward pass for complex selective scan using fused Triton kernel.

    Reads the forward output saved by the autograd wrapper to avoid a full
    O(T) forward recompute and the associated scratch allocation.

    When any of the ``compute_grad_*`` flags is False the kernel skips the
    corresponding atomic_add / loads / store via a Triton ``constexpr``
    branch (driven by ``ctx.needs_input_grad`` in the autograd wrapper);
    that grad is returned as ``None`` so downstream autograd sees it as
    "not computed" and the caller's ``.grad`` stays untouched.

    Args:
        C1: Complex (num_filters,) - recurrence coefficient.
        C2: Complex (num_filters,) - input gain.
        u: Complex (batch, time, num_filters) - forward input.
        output: Complex (batch, time, num_filters) - forward output x.
        grad_output: Complex (batch, time, num_filters) - upstream gradient.
        compute_grad_C1: If False, skip grad_C1 in kernel and return None.
        compute_grad_C2: If False, skip grad_C2 in kernel and return None.
        compute_grad_u: If False, skip grad_u in kernel and return None.

    Returns:
        ``(grad_C1, grad_C2, grad_u)`` where each entry is ``None`` when the
        matching ``compute_grad_*`` is False.
    """
    if not (compute_grad_C1 or compute_grad_C2 or compute_grad_u):
        return None, None, None

    # Prepare inputs (validation, dtype conversion, contiguity)
    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=True)

    if not grad_output.is_complex():
        grad_output = grad_output.to(torch.cfloat)
    grad_out_real = grad_output.real.contiguous()
    grad_out_imag = grad_output.imag.contiguous()

    if not output.is_complex():
        output = output.to(torch.cfloat)
    out_real = output.real.contiguous()
    out_imag = output.imag.contiguous()

    batch_size = inputs.batch_size
    time_size = inputs.time_size
    filter_size = inputs.filter_size
    target_dtype = inputs.u_real.dtype

    # Allocate output gradient tensors. grad_u allocation is gated on
    # ``compute_grad_u``: it's the only O(B*T*F) buffer and is pure overhead
    # when not requested. grad_C{1,2} are O(F); cheap, always allocate so the
    # kernel pointers are valid — the kernel just won't atomic-add into them.
    if compute_grad_u:
        grad_u_real = torch.empty(
            batch_size, time_size, filter_size, device=u.device, dtype=target_dtype
        )
        grad_u_imag = torch.empty_like(grad_u_real)
    else:
        grad_u_real = torch.empty(1, device=u.device, dtype=target_dtype)
        grad_u_imag = grad_u_real
    grad_c1r = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c1i = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c2r = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c2i = torch.zeros(filter_size, device=u.device, dtype=target_dtype)

    # Strides (u, output, grad_out, grad_u all share this layout — all built from
    # contiguous calls or torch.empty_like on u_real)
    stride_batch = inputs.u_real.stride(0)
    stride_time = inputs.u_real.stride(1)
    stride_filter = inputs.u_real.stride(2)

    # Launch: one program per (batch, filter) pair. Scalar state per program
    # keeps 256 CTAs in flight at the typical SC workload (B=8, F=32), which
    # the A10G's 80 SMs need to hide the per-iteration HBM-load latency.
    num_programs = batch_size * filter_size
    grid = (num_programs,)

    _sequential_scan_backward_kernel[grid](
        inputs.u_real,
        inputs.u_imag,
        out_real,
        out_imag,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        grad_out_real,
        grad_out_imag,
        grad_u_real,
        grad_u_imag,
        grad_c1r,
        grad_c1i,
        grad_c2r,
        grad_c2i,
        batch_size,
        time_size,
        filter_size,
        stride_batch,
        stride_time,
        stride_filter,
        BLOCK_SIZE=256,
        COMPUTE_GRAD_C1=compute_grad_C1,
        COMPUTE_GRAD_C2=compute_grad_C2,
        COMPUTE_GRAD_U=compute_grad_u,
    )

    grad_C1 = torch.complex(grad_c1r, grad_c1i) if compute_grad_C1 else None
    grad_C2 = torch.complex(grad_c2r, grad_c2i) if compute_grad_C2 else None
    grad_u = torch.complex(grad_u_real, grad_u_imag) if compute_grad_u else None

    return grad_C1, grad_C2, grad_u


@triton.autotune(
    configs=_SEQUENTIAL_SCAN_BLOCK_F_CONFIGS,
    key=["time_size", "filter_size"],
    reset_to_zero=_SEQUENTIAL_SCAN_BLOCK_F_GRAD_RESET_ARGS,
)
@triton.jit
def _sequential_scan_backward_block_f_kernel(  # pragma: no cover
    u_real_ptr,
    u_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    grad_out_real_ptr,
    grad_out_imag_ptr,
    grad_u_real_ptr,
    grad_u_imag_ptr,
    grad_c1r_ptr,
    grad_c1i_ptr,
    grad_c2r_ptr,
    grad_c2i_ptr,
    batch_size,
    time_size,
    filter_size,
    stride_batch,
    stride_time,
    stride_filter,
    BLOCK_F: tl.constexpr,
    COMPUTE_GRAD_C1: tl.constexpr,
    COMPUTE_GRAD_C2: tl.constexpr,
    COMPUTE_GRAD_U: tl.constexpr,
):
    """Vectorized backward over a contiguous filter block."""
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    if batch_idx >= batch_size:
        return

    f_offsets = block_idx * BLOCK_F + tl.arange(0, BLOCK_F)
    f_mask = f_offsets < filter_size

    c1r = tl.load(c1_real_ptr + f_offsets, mask=f_mask, other=0.0)
    c1i = tl.load(c1_imag_ptr + f_offsets, mask=f_mask, other=0.0)
    c2r = tl.load(c2_real_ptr + f_offsets, mask=f_mask, other=0.0)
    c2i = tl.load(c2_imag_ptr + f_offsets, mask=f_mask, other=0.0)

    first_offset = batch_idx * stride_batch + f_offsets * stride_filter
    first_u = tl.load(u_real_ptr + first_offset, mask=f_mask, other=0.0)
    zero = first_u - first_u

    adj_r = zero
    adj_i = zero
    gc1r = zero
    gc1i = zero
    gc2r = zero
    gc2i = zero

    for t_rev in range(time_size):
        t = time_size - 1 - t_rev
        offset = batch_idx * stride_batch + t * stride_time + f_offsets * stride_filter

        go_r = tl.load(grad_out_real_ptr + offset, mask=f_mask, other=0.0)
        go_i = tl.load(grad_out_imag_ptr + offset, mask=f_mask, other=0.0)

        adj_r_new = c1r * adj_r + c1i * adj_i + go_r
        adj_i_new = c1r * adj_i - c1i * adj_r + go_i

        if COMPUTE_GRAD_U:
            gu_r = c2r * adj_r_new + c2i * adj_i_new
            gu_i = c2r * adj_i_new - c2i * adj_r_new
            tl.store(grad_u_real_ptr + offset, gu_r, mask=f_mask)
            tl.store(grad_u_imag_ptr + offset, gu_i, mask=f_mask)

        if COMPUTE_GRAD_C2:
            ur = tl.load(u_real_ptr + offset, mask=f_mask, other=0.0)
            ui = tl.load(u_imag_ptr + offset, mask=f_mask, other=0.0)
            gc2r += adj_r_new * ur + adj_i_new * ui
            gc2i += adj_i_new * ur - adj_r_new * ui

        if COMPUTE_GRAD_C1 and t > 0:
            offset_prev = (
                batch_idx * stride_batch + (t - 1) * stride_time + f_offsets * stride_filter
            )
            xr_prev = tl.load(out_real_ptr + offset_prev, mask=f_mask, other=0.0)
            xi_prev = tl.load(out_imag_ptr + offset_prev, mask=f_mask, other=0.0)
            gc1r += adj_r_new * xr_prev + adj_i_new * xi_prev
            gc1i += adj_i_new * xr_prev - adj_r_new * xi_prev

        adj_r = adj_r_new
        adj_i = adj_i_new

    if COMPUTE_GRAD_C1:
        tl.atomic_add(grad_c1r_ptr + f_offsets, gc1r, mask=f_mask)
        tl.atomic_add(grad_c1i_ptr + f_offsets, gc1i, mask=f_mask)
    if COMPUTE_GRAD_C2:
        tl.atomic_add(grad_c2r_ptr + f_offsets, gc2r, mask=f_mask)
        tl.atomic_add(grad_c2i_ptr + f_offsets, gc2i, mask=f_mask)


def _sequential_scan_backward_raw_block_f(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    compute_grad_C1: bool = True,
    compute_grad_C2: bool = True,
    compute_grad_u: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Autotuned BLOCK_F backward for the standard complex recurrence."""
    if not (compute_grad_C1 or compute_grad_C2 or compute_grad_u):
        return None, None, None

    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=True)

    if not grad_output.is_complex():
        grad_output = grad_output.to(torch.cfloat)
    grad_out_real = grad_output.real.contiguous()
    grad_out_imag = grad_output.imag.contiguous()

    if not output.is_complex():
        output = output.to(torch.cfloat)
    out_real = output.real.contiguous()
    out_imag = output.imag.contiguous()

    batch_size = inputs.batch_size
    time_size = inputs.time_size
    filter_size = inputs.filter_size
    target_dtype = inputs.u_real.dtype

    if compute_grad_u:
        grad_u_real = torch.empty(
            batch_size, time_size, filter_size, device=u.device, dtype=target_dtype
        )
        grad_u_imag = torch.empty_like(grad_u_real)
    else:
        grad_u_real = torch.empty(1, device=u.device, dtype=target_dtype)
        grad_u_imag = grad_u_real
    grad_c1r = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c1i = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c2r = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c2i = torch.zeros(filter_size, device=u.device, dtype=target_dtype)

    def grid(META: dict) -> tuple[int, int]:
        return (batch_size, triton.cdiv(filter_size, META["BLOCK_F"]))

    _sequential_scan_backward_block_f_kernel[grid](
        inputs.u_real,
        inputs.u_imag,
        out_real,
        out_imag,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        grad_out_real,
        grad_out_imag,
        grad_u_real,
        grad_u_imag,
        grad_c1r,
        grad_c1i,
        grad_c2r,
        grad_c2i,
        batch_size,
        time_size,
        filter_size,
        inputs.u_real.stride(0),
        inputs.u_real.stride(1),
        inputs.u_real.stride(2),
        COMPUTE_GRAD_C1=compute_grad_C1,
        COMPUTE_GRAD_C2=compute_grad_C2,
        COMPUTE_GRAD_U=compute_grad_u,
    )

    grad_C1 = torch.complex(grad_c1r, grad_c1i) if compute_grad_C1 else None
    grad_C2 = torch.complex(grad_c2r, grad_c2i) if compute_grad_C2 else None
    grad_u = torch.complex(grad_u_real, grad_u_imag) if compute_grad_u else None

    return grad_C1, grad_C2, grad_u


class _SequentialScanFunction(torch.autograd.Function):  # pragma: no cover
    """Autograd function for selective scan with Triton acceleration.

    Forward: Uses Triton kernel for x[n] = C1*x[n-1] + C2*u[n]
    Backward: Uses fused Triton kernel (single-pass adjoint sweep)

    The forward output is saved in ctx. This is effectively free memory-wise
    because the output tensor is already kept alive by the downstream graph
    (e.g. the loss computation); ctx just holds an additional reference.
    In exchange the backward kernel skips the Phase-1 forward recompute and
    the (B*F, T) scratch allocation — roughly a 2x wall-clock win at the
    workloads measured on RTX 3090 with T~16k.
    """

    @staticmethod
    def forward(
        ctx,
        C1: torch.Tensor,
        C2: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using Triton kernel.

        Args:
            ctx: Autograd context for saving tensors.
            C1: Complex tensor of shape (num_filters,) - recurrence coefficient.
            C2: Complex tensor of shape (num_filters,) - input gain.
            u: Complex tensor of shape (batch, time, num_filters) - input signal.

        Returns:
            Complex tensor of shape (batch, time, num_filters) - output states.
        """
        # Ensure complex
        if not C1.is_complex():
            C1 = C1.to(torch.cfloat)
        if not C2.is_complex():
            C2 = C2.to(torch.cfloat)
        if not u.is_complex():
            u = u.to(torch.cfloat)

        # Run forward pass with Triton
        output = _sequential_scan_forward_raw(C1, C2, u)

        # Save tensors needed for backward; saving ``output`` lets the backward
        # skip a full O(T) forward recompute (no extra memory — output is already
        # live in the autograd graph).
        ctx.save_for_backward(C1, C2, u, output)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass using fused Triton kernel.

        Single-pass adjoint sweep; uses the saved forward output in place of
        the deprecated in-kernel forward recompute + scratch buffers.
        ``ctx.needs_input_grad`` is forwarded so frozen-recurrent fine-tuning
        (e.g. only ``u.requires_grad``) pays no atomic_add contention for
        C1/C2 and skips the T-length u / out_prev loads in the kernel.
        """
        C1, C2, u, output = ctx.saved_tensors
        need_C1, need_C2, need_u = ctx.needs_input_grad
        grad_C1, grad_C2, grad_u = _sequential_scan_backward_raw(
            C1,
            C2,
            u,
            output,
            grad_output,
            compute_grad_C1=need_C1,
            compute_grad_C2=need_C2,
            compute_grad_u=need_u,
        )
        return grad_C1, grad_C2, grad_u


def _triton_sequential_scan(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Compute linear recurrence using selective scan with autograd support.

    This function uses Triton for the forward pass and PyTorch for the
    backward pass, enabling fast training on GPU.

    Computes: x[n] = C1 * x[n-1] + C2 * u[n] for all n.

    Args:
        C1: Complex tensor of shape (num_filters,) - recurrence coefficient.
        C2: Complex tensor of shape (num_filters,) - input gain.
        u: Complex tensor of shape (batch, time, num_filters) - input signal.

    Returns:
        Complex tensor of shape (batch, time, num_filters) - output states.
    """
    return _SequentialScanFunction.apply(C1, C2, u)


@triton.jit
def _sequential_scan_real_kernel(  # pragma: no cover
    u_real_ptr,
    out_real_ptr,
    out_imag_ptr,
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    batch_size,
    time_size,
    filter_size,
    stride_batch,
    stride_time,
    stride_filter,
    BLOCK_SIZE: tl.constexpr,  # noqa: ARG001
    EPILOGUE: tl.constexpr,
):
    """Sequential complex recurrence specialized for a real-valued input ``u``.

    Implements the SAME equation as ``_sequential_scan_kernel``:
        x[t] = C1 * x[t-1] + C2 * u[t]    (C1, C2, x complex)

    but with ``u`` stored as a real tensor and ``Im(u) == 0`` folded in at
    compile time. Concretely, the complex kernel computes
        x_real = c1r*x_real - c1i*x_imag + c2r*u_r - c2i*u_i
        x_imag = c1r*x_imag + c1i*x_real + c2r*u_i + c2i*u_r
    and this kernel drops the terms multiplying ``u_i`` because ``u_i == 0``.
    The state (x_real, x_imag) is still fully complex; the output is complex
    by default. ``EPILOGUE`` is a compile-time switch selecting what lands
    in HBM, letting callers in eval / no_grad mode skip the imaginary half
    entirely (backward needs both so the complex mode stays the default):

      0 = COMPLEX: write x_real to out_real_ptr, x_imag to out_imag_ptr.
      1 = ABS:     write sqrt(x_real**2 + x_imag**2) to out_real_ptr only.
                   out_imag_ptr is not dereferenced.

    This is NOT the deprecated real-input equation with an extra imaginary
    term (see module docstring).
    """
    pid = tl.program_id(0)

    batch_idx = pid // filter_size
    filter_idx = pid % filter_size

    if batch_idx >= batch_size:
        return

    c1_real = tl.load(c1_real_ptr + filter_idx)
    c1_imag = tl.load(c1_imag_ptr + filter_idx)
    c2_real = tl.load(c2_real_ptr + filter_idx)
    c2_imag = tl.load(c2_imag_ptr + filter_idx)

    first_offset = batch_idx * stride_batch + filter_idx * stride_filter
    first_u = tl.load(u_real_ptr + first_offset)
    x_real = first_u - first_u
    x_imag = first_u - first_u

    for t in range(time_size):
        offset = batch_idx * stride_batch + t * stride_time + filter_idx * stride_filter
        u_r = tl.load(u_real_ptr + offset)

        c1x_real = c1_real * x_real - c1_imag * x_imag
        c1x_imag = c1_real * x_imag + c1_imag * x_real

        x_real = c1x_real + c2_real * u_r
        x_imag = c1x_imag + c2_imag * u_r

        if EPILOGUE == 0:
            tl.store(out_real_ptr + offset, x_real)
            tl.store(out_imag_ptr + offset, x_imag)
        else:
            tl.store(out_real_ptr + offset, tl.sqrt(x_real * x_real + x_imag * x_imag))


def _sequential_scan_real_forward_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Raw forward pass for the real-input specialization of the standard
    complex selective scan. Not a separate equation — see module docstring."""
    if u.is_complex():
        raise ValueError("_sequential_scan_real_forward_raw expects real input")
    if u.dtype != torch.float32:
        u = u.to(torch.float32)
    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=False)

    out_real = torch.empty_like(inputs.u_real)
    out_imag = torch.empty_like(inputs.u_real)

    _sequential_scan_real_kernel[(inputs.batch_size * inputs.filter_size,)](
        inputs.u_real,
        out_real,
        out_imag,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        inputs.batch_size,
        inputs.time_size,
        inputs.filter_size,
        inputs.u_real.stride(0),
        inputs.u_real.stride(1),
        inputs.u_real.stride(2),
        BLOCK_SIZE=256,
        EPILOGUE=0,
    )

    return torch.complex(out_real, out_imag)


def _sequential_scan_real_forward_abs_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Fused-``abs()`` epilogue variant of ``_sequential_scan_real_forward_raw``.

    Returns a real tensor holding ``|x[t]| = sqrt(Re(x[t])**2 + Im(x[t])**2)``
    directly, skipping the imaginary half of the output entirely. Half the
    output HBM traffic vs the complex path, intended for eval / no_grad
    consumers (RQF magnitude readout, SC inference) — backward still needs
    the complex output to reconstruct ``x[t-1]`` so this has no autograd
    wrapper; training should stay on the complex path.
    """
    if u.is_complex():
        raise ValueError("_sequential_scan_real_forward_abs_raw expects real input")
    if u.dtype != torch.float32:
        u = u.to(torch.float32)
    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=False)

    out = torch.empty_like(inputs.u_real)
    # Dummy single-element buffer for the out_imag pointer; kernel never
    # dereferences it in EPILOGUE=1 mode.
    out_imag_dummy = torch.empty(1, device=u.device, dtype=inputs.u_real.dtype)

    _sequential_scan_real_kernel[(inputs.batch_size * inputs.filter_size,)](
        inputs.u_real,
        out,
        out_imag_dummy,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        inputs.batch_size,
        inputs.time_size,
        inputs.filter_size,
        inputs.u_real.stride(0),
        inputs.u_real.stride(1),
        inputs.u_real.stride(2),
        BLOCK_SIZE=256,
        EPILOGUE=1,
    )

    return out


@triton.jit
def _sequential_scan_backward_real_kernel(  # pragma: no cover
    u_real_ptr,
    out_real_ptr,
    out_imag_ptr,
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    grad_out_real_ptr,
    grad_out_imag_ptr,
    grad_u_ptr,
    grad_c1r_ptr,
    grad_c1i_ptr,
    grad_c2r_ptr,
    grad_c2i_ptr,
    batch_size,
    time_size,
    filter_size,
    stride_batch,
    stride_time,
    stride_filter,
    BLOCK_SIZE: tl.constexpr,  # noqa: ARG001
    COMPUTE_GRAD_C1: tl.constexpr,
    COMPUTE_GRAD_C2: tl.constexpr,
    COMPUTE_GRAD_U: tl.constexpr,
):
    """Fused backward kernel for the real-input specialization of the
    standard complex selective scan.

    Single-pass adjoint sweep (t=T-1..0). The forward output x[t] (complex)
    is passed in via ``out_real/imag`` so the kernel can skip a full
    forward recompute + scratch alloc and read ``x[t-1]`` directly for the
    grad_C1 accumulation. ``Im(u) == 0`` is folded in at compile time.
    ``grad_u`` is real because ``u`` is real.

    The ``COMPUTE_GRAD_*`` flags are driven by ``ctx.needs_input_grad`` and
    let the compiler elide per-t loads and atomic_adds for any grad the
    caller didn't request. Scalar (batch, filter) state per program — the
    BLOCK_F-tiled variant measured slower at B=8 F=32 (not enough CTAs to
    saturate the A10G); see sister complex-backward kernel's docstring.
    """
    pid = tl.program_id(0)
    batch_idx = pid // filter_size
    filter_idx = pid % filter_size

    if batch_idx >= batch_size:
        return

    c1r = tl.load(c1_real_ptr + filter_idx)
    c1i = tl.load(c1_imag_ptr + filter_idx)
    c2r = tl.load(c2_real_ptr + filter_idx)
    c2i = tl.load(c2_imag_ptr + filter_idx)

    first_offset = batch_idx * stride_batch + filter_idx * stride_filter
    first_u = tl.load(u_real_ptr + first_offset)
    zero = first_u - first_u

    adj_r = zero
    adj_i = zero
    gc1r = zero
    gc1i = zero
    gc2r = zero
    gc2i = zero

    for t_rev in range(time_size):
        t = time_size - 1 - t_rev
        offset = batch_idx * stride_batch + t * stride_time + filter_idx * stride_filter

        go_r = tl.load(grad_out_real_ptr + offset)
        go_i = tl.load(grad_out_imag_ptr + offset)

        adj_r_new = c1r * adj_r + c1i * adj_i + go_r
        adj_i_new = c1r * adj_i - c1i * adj_r + go_i

        if COMPUTE_GRAD_U:
            gu = c2r * adj_r_new + c2i * adj_i_new
            tl.store(grad_u_ptr + offset, gu)

        if COMPUTE_GRAD_C2:
            ur = tl.load(u_real_ptr + offset)
            gc2r += adj_r_new * ur
            gc2i += adj_i_new * ur

        if COMPUTE_GRAD_C1 and t > 0:
            offset_prev = (
                batch_idx * stride_batch + (t - 1) * stride_time + filter_idx * stride_filter
            )
            xr_prev = tl.load(out_real_ptr + offset_prev)
            xi_prev = tl.load(out_imag_ptr + offset_prev)
            gc1r += adj_r_new * xr_prev + adj_i_new * xi_prev
            gc1i += adj_i_new * xr_prev - adj_r_new * xi_prev

        adj_r = adj_r_new
        adj_i = adj_i_new

    if COMPUTE_GRAD_C1:
        tl.atomic_add(grad_c1r_ptr + filter_idx, gc1r)
        tl.atomic_add(grad_c1i_ptr + filter_idx, gc1i)
    if COMPUTE_GRAD_C2:
        tl.atomic_add(grad_c2r_ptr + filter_idx, gc2r)
        tl.atomic_add(grad_c2i_ptr + filter_idx, gc2i)


def _sequential_scan_backward_real_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    compute_grad_C1: bool = True,
    compute_grad_C2: bool = True,
    compute_grad_u: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Raw backward pass for the real-input specialization of the standard
    complex selective scan. ``grad_u`` is real; ``grad_C1``/``grad_C2`` stay
    complex. Reads the saved forward output to skip forward recompute.
    Not a separate equation — see module docstring.

    The ``compute_grad_*`` flags (driven by ``ctx.needs_input_grad``) gate
    the matching grad inside the kernel (constexpr branches) and as the
    return value (``None`` when skipped)."""
    if u.is_complex():
        raise ValueError("_sequential_scan_backward_real_raw expects real input")
    if not (compute_grad_C1 or compute_grad_C2 or compute_grad_u):
        return None, None, None

    if u.dtype != torch.float32:
        u = u.to(torch.float32)
    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=False)

    if not grad_output.is_complex():
        grad_output = grad_output.to(torch.cfloat)
    target_dtype = inputs.u_real.dtype
    grad_out_real = grad_output.real.to(target_dtype).contiguous()
    grad_out_imag = grad_output.imag.to(target_dtype).contiguous()

    if not output.is_complex():
        output = output.to(torch.cfloat)
    out_real = output.real.to(target_dtype).contiguous()
    out_imag = output.imag.to(target_dtype).contiguous()

    batch_size = inputs.batch_size
    time_size = inputs.time_size
    filter_size = inputs.filter_size

    if compute_grad_u:
        grad_u = torch.empty(
            batch_size, time_size, filter_size, device=u.device, dtype=target_dtype
        )
    else:
        grad_u = torch.empty(1, device=u.device, dtype=target_dtype)
    grad_c1r = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c1i = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c2r = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c2i = torch.zeros(filter_size, device=u.device, dtype=target_dtype)

    num_programs = batch_size * filter_size
    _sequential_scan_backward_real_kernel[(num_programs,)](
        inputs.u_real,
        out_real,
        out_imag,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        grad_out_real,
        grad_out_imag,
        grad_u,
        grad_c1r,
        grad_c1i,
        grad_c2r,
        grad_c2i,
        batch_size,
        time_size,
        filter_size,
        inputs.u_real.stride(0),
        inputs.u_real.stride(1),
        inputs.u_real.stride(2),
        BLOCK_SIZE=256,
        COMPUTE_GRAD_C1=compute_grad_C1,
        COMPUTE_GRAD_C2=compute_grad_C2,
        COMPUTE_GRAD_U=compute_grad_u,
    )

    grad_C1 = torch.complex(grad_c1r, grad_c1i) if compute_grad_C1 else None
    grad_C2 = torch.complex(grad_c2r, grad_c2i) if compute_grad_C2 else None
    grad_u_out = grad_u if compute_grad_u else None
    return grad_C1, grad_C2, grad_u_out


@triton.autotune(configs=_SEQUENTIAL_SCAN_BLOCK_F_CONFIGS, key=["time_size", "filter_size"])
@triton.jit
def _sequential_scan_real_block_f_kernel(  # pragma: no cover
    u_real_ptr,
    out_real_ptr,
    out_imag_ptr,
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    batch_size,
    time_size,
    filter_size,
    stride_batch,
    stride_time,
    stride_filter,
    BLOCK_F: tl.constexpr,
    EPILOGUE: tl.constexpr,
):
    """Vectorized real-input specialization over a contiguous filter block."""
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    if batch_idx >= batch_size:
        return

    f_offsets = block_idx * BLOCK_F + tl.arange(0, BLOCK_F)
    f_mask = f_offsets < filter_size

    c1_real = tl.load(c1_real_ptr + f_offsets, mask=f_mask, other=0.0)
    c1_imag = tl.load(c1_imag_ptr + f_offsets, mask=f_mask, other=0.0)
    c2_real = tl.load(c2_real_ptr + f_offsets, mask=f_mask, other=0.0)
    c2_imag = tl.load(c2_imag_ptr + f_offsets, mask=f_mask, other=0.0)

    first_offset = batch_idx * stride_batch + f_offsets * stride_filter
    first_u = tl.load(u_real_ptr + first_offset, mask=f_mask, other=0.0)
    x_real = first_u - first_u
    x_imag = first_u - first_u

    for t in range(time_size):
        offset = batch_idx * stride_batch + t * stride_time + f_offsets * stride_filter
        u_r = tl.load(u_real_ptr + offset, mask=f_mask, other=0.0)

        c1x_real = c1_real * x_real - c1_imag * x_imag
        c1x_imag = c1_real * x_imag + c1_imag * x_real

        x_real = c1x_real + c2_real * u_r
        x_imag = c1x_imag + c2_imag * u_r

        if EPILOGUE == 0:
            tl.store(out_real_ptr + offset, x_real, mask=f_mask)
            tl.store(out_imag_ptr + offset, x_imag, mask=f_mask)
        else:
            tl.store(
                out_real_ptr + offset,
                tl.sqrt(x_real * x_real + x_imag * x_imag),
                mask=f_mask,
            )


def _sequential_scan_real_forward_raw_block_f(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Autotuned BLOCK_F forward for the real-input specialization."""
    if u.is_complex():
        raise ValueError("_sequential_scan_real_forward_raw_block_f expects real input")
    if u.dtype != torch.float32:
        u = u.to(torch.float32)
    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=False)

    out_real = torch.empty_like(inputs.u_real)
    out_imag = torch.empty_like(inputs.u_real)

    def grid(META: dict) -> tuple[int, int]:
        return (inputs.batch_size, triton.cdiv(inputs.filter_size, META["BLOCK_F"]))

    _sequential_scan_real_block_f_kernel[grid](
        inputs.u_real,
        out_real,
        out_imag,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        inputs.batch_size,
        inputs.time_size,
        inputs.filter_size,
        inputs.u_real.stride(0),
        inputs.u_real.stride(1),
        inputs.u_real.stride(2),
        EPILOGUE=0,
    )

    return torch.complex(out_real, out_imag)


def _sequential_scan_real_forward_abs_raw_block_f(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """BLOCK_F fused-``abs()`` epilogue for the real-input specialization."""
    if u.is_complex():
        raise ValueError("_sequential_scan_real_forward_abs_raw_block_f expects real input")
    if u.dtype != torch.float32:
        u = u.to(torch.float32)
    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=False)

    out = torch.empty_like(inputs.u_real)
    out_imag_dummy = torch.empty(1, device=u.device, dtype=inputs.u_real.dtype)

    def grid(META: dict) -> tuple[int, int]:
        return (inputs.batch_size, triton.cdiv(inputs.filter_size, META["BLOCK_F"]))

    _sequential_scan_real_block_f_kernel[grid](
        inputs.u_real,
        out,
        out_imag_dummy,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        inputs.batch_size,
        inputs.time_size,
        inputs.filter_size,
        inputs.u_real.stride(0),
        inputs.u_real.stride(1),
        inputs.u_real.stride(2),
        EPILOGUE=1,
    )

    return out


@triton.autotune(
    configs=_SEQUENTIAL_SCAN_BLOCK_F_CONFIGS,
    key=["time_size", "filter_size"],
    reset_to_zero=_SEQUENTIAL_SCAN_BLOCK_F_GRAD_RESET_ARGS,
)
@triton.jit
def _sequential_scan_backward_real_block_f_kernel(  # pragma: no cover
    u_real_ptr,
    out_real_ptr,
    out_imag_ptr,
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    grad_out_real_ptr,
    grad_out_imag_ptr,
    grad_u_ptr,
    grad_c1r_ptr,
    grad_c1i_ptr,
    grad_c2r_ptr,
    grad_c2i_ptr,
    batch_size,
    time_size,
    filter_size,
    stride_batch,
    stride_time,
    stride_filter,
    BLOCK_F: tl.constexpr,
    COMPUTE_GRAD_C1: tl.constexpr,
    COMPUTE_GRAD_C2: tl.constexpr,
    COMPUTE_GRAD_U: tl.constexpr,
):
    """Vectorized backward for the real-input specialization."""
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    if batch_idx >= batch_size:
        return

    f_offsets = block_idx * BLOCK_F + tl.arange(0, BLOCK_F)
    f_mask = f_offsets < filter_size

    c1r = tl.load(c1_real_ptr + f_offsets, mask=f_mask, other=0.0)
    c1i = tl.load(c1_imag_ptr + f_offsets, mask=f_mask, other=0.0)
    c2r = tl.load(c2_real_ptr + f_offsets, mask=f_mask, other=0.0)
    c2i = tl.load(c2_imag_ptr + f_offsets, mask=f_mask, other=0.0)

    first_offset = batch_idx * stride_batch + f_offsets * stride_filter
    first_u = tl.load(u_real_ptr + first_offset, mask=f_mask, other=0.0)
    zero = first_u - first_u

    adj_r = zero
    adj_i = zero
    gc1r = zero
    gc1i = zero
    gc2r = zero
    gc2i = zero

    for t_rev in range(time_size):
        t = time_size - 1 - t_rev
        offset = batch_idx * stride_batch + t * stride_time + f_offsets * stride_filter

        go_r = tl.load(grad_out_real_ptr + offset, mask=f_mask, other=0.0)
        go_i = tl.load(grad_out_imag_ptr + offset, mask=f_mask, other=0.0)

        adj_r_new = c1r * adj_r + c1i * adj_i + go_r
        adj_i_new = c1r * adj_i - c1i * adj_r + go_i

        if COMPUTE_GRAD_U:
            gu = c2r * adj_r_new + c2i * adj_i_new
            tl.store(grad_u_ptr + offset, gu, mask=f_mask)

        if COMPUTE_GRAD_C2:
            ur = tl.load(u_real_ptr + offset, mask=f_mask, other=0.0)
            gc2r += adj_r_new * ur
            gc2i += adj_i_new * ur

        if COMPUTE_GRAD_C1 and t > 0:
            offset_prev = (
                batch_idx * stride_batch + (t - 1) * stride_time + f_offsets * stride_filter
            )
            xr_prev = tl.load(out_real_ptr + offset_prev, mask=f_mask, other=0.0)
            xi_prev = tl.load(out_imag_ptr + offset_prev, mask=f_mask, other=0.0)
            gc1r += adj_r_new * xr_prev + adj_i_new * xi_prev
            gc1i += adj_i_new * xr_prev - adj_r_new * xi_prev

        adj_r = adj_r_new
        adj_i = adj_i_new

    if COMPUTE_GRAD_C1:
        tl.atomic_add(grad_c1r_ptr + f_offsets, gc1r, mask=f_mask)
        tl.atomic_add(grad_c1i_ptr + f_offsets, gc1i, mask=f_mask)
    if COMPUTE_GRAD_C2:
        tl.atomic_add(grad_c2r_ptr + f_offsets, gc2r, mask=f_mask)
        tl.atomic_add(grad_c2i_ptr + f_offsets, gc2i, mask=f_mask)


def _sequential_scan_backward_real_raw_block_f(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    compute_grad_C1: bool = True,
    compute_grad_C2: bool = True,
    compute_grad_u: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Autotuned BLOCK_F backward for the real-input specialization."""
    if u.is_complex():
        raise ValueError("_sequential_scan_backward_real_raw_block_f expects real input")
    if not (compute_grad_C1 or compute_grad_C2 or compute_grad_u):
        return None, None, None

    if u.dtype != torch.float32:
        u = u.to(torch.float32)
    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=False)

    if not grad_output.is_complex():
        grad_output = grad_output.to(torch.cfloat)
    target_dtype = inputs.u_real.dtype
    grad_out_real = grad_output.real.to(target_dtype).contiguous()
    grad_out_imag = grad_output.imag.to(target_dtype).contiguous()

    if not output.is_complex():
        output = output.to(torch.cfloat)
    out_real = output.real.to(target_dtype).contiguous()
    out_imag = output.imag.to(target_dtype).contiguous()

    batch_size = inputs.batch_size
    time_size = inputs.time_size
    filter_size = inputs.filter_size

    if compute_grad_u:
        grad_u = torch.empty(
            batch_size, time_size, filter_size, device=u.device, dtype=target_dtype
        )
    else:
        grad_u = torch.empty(1, device=u.device, dtype=target_dtype)
    grad_c1r = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c1i = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c2r = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c2i = torch.zeros(filter_size, device=u.device, dtype=target_dtype)

    def grid(META: dict) -> tuple[int, int]:
        return (batch_size, triton.cdiv(filter_size, META["BLOCK_F"]))

    _sequential_scan_backward_real_block_f_kernel[grid](
        inputs.u_real,
        out_real,
        out_imag,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        grad_out_real,
        grad_out_imag,
        grad_u,
        grad_c1r,
        grad_c1i,
        grad_c2r,
        grad_c2i,
        batch_size,
        time_size,
        filter_size,
        inputs.u_real.stride(0),
        inputs.u_real.stride(1),
        inputs.u_real.stride(2),
        COMPUTE_GRAD_C1=compute_grad_C1,
        COMPUTE_GRAD_C2=compute_grad_C2,
        COMPUTE_GRAD_U=compute_grad_u,
    )

    grad_C1 = torch.complex(grad_c1r, grad_c1i) if compute_grad_C1 else None
    grad_C2 = torch.complex(grad_c2r, grad_c2i) if compute_grad_C2 else None
    grad_u_out = grad_u if compute_grad_u else None
    return grad_C1, grad_C2, grad_u_out


class _SequentialScanRealFunction(torch.autograd.Function):  # pragma: no cover
    """Autograd wrapper for the real-input specialization of the standard
    complex selective scan. Same equation as the complex path with Im(u)=0."""

    @staticmethod
    def forward(
        ctx,
        C1: torch.Tensor,
        C2: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        if u.is_complex():
            raise ValueError("_SequentialScanRealFunction expects real input")
        if u.dtype != torch.float32:
            u = u.to(torch.float32)
        output = _sequential_scan_real_forward_raw(C1, C2, u)
        ctx.save_for_backward(C1, C2, u, output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward via fused real-input kernel. ``ctx.needs_input_grad`` gates
        kernel constexpr branches so skipped grads cost zero atomic_add /
        per-t loads (e.g. frozen C1/C2 fine-tuning)."""
        C1, C2, u, output = ctx.saved_tensors
        need_C1, need_C2, need_u = ctx.needs_input_grad
        grad_C1, grad_C2, grad_u = _sequential_scan_backward_real_raw(
            C1,
            C2,
            u,
            output,
            grad_output,
            compute_grad_C1=need_C1,
            compute_grad_C2=need_C2,
            compute_grad_u=need_u,
        )
        return grad_C1, grad_C2, grad_u


def _triton_sequential_scan_real(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Compute the standard complex recurrence x[t]=C1*x[t-1]+C2*u[t] with a
    real-valued input ``u`` (treated as ``u + 0j``). Specialization of
    ``_triton_sequential_scan``: same math, same output (complex), just skips
    the redundant all-zero imaginary-input traffic. NOT the deprecated
    real-input equation with an extra imaginary term — see module docstring.
    """
    if u.is_complex():
        raise ValueError("_triton_sequential_scan_real expects real input")
    _, _, in_filters = u.shape
    num_filters = C1.shape[0]
    if in_filters == 1 and num_filters > 1:
        u = u.expand(-1, -1, num_filters).contiguous()
    return _SequentialScanRealFunction.apply(C1, C2, u)


# ==============================================================================
# Section 4: Fused Multi-Oscillator Complex Scan with C-Weight Reduction
#
# For N>1 oscillators per filter, this kernel fuses:
#   1. N independent complex scans sharing the same input u per physical filter
#   2. C-weight combination: y[t] = Σ_n c_n * x_n[t]
#
# Benefits over N separate scans:
#   - Input u loaded once per timestep (not N times) — N× fewer memory reads
#   - Output (B,T,F) instead of (B,T,F*N) — N× less memory
#   - No separate repeat_interleave or _c_weighted_sum passes
#   - Persistent activations saved for backward are N× smaller
#
# Forward: Fused Triton kernel with C-weight reduction
# Backward: Fused Triton kernel (forward recompute + adjoint in one launch)
# ==============================================================================


@triton.jit
def _sequential_scan_multi_osc_kernel(  # pragma: no cover
    # Input pointers (complex, stored as real/imag, shape: batch, time, num_filters)
    u_real_ptr,
    u_imag_ptr,
    # Output pointers (reduced: batch, time, num_filters)
    out_real_ptr,
    out_imag_ptr,
    # Per-oscillator coefficient pointers (shape: scan_filters = num_filters * num_osc)
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    # C-weight pointers (shape: scan_filters, separate real/imag)
    cw_real_ptr,
    cw_imag_ptr,
    # Dimensions
    batch_size,
    time_size,
    num_filters,
    num_osc,
    # Input strides (for u with num_filters channels)
    u_stride_batch,
    u_stride_time,
    u_stride_filter,
    # Output strides (same layout as input, num_filters channels)
    out_stride_batch,
    out_stride_time,
    out_stride_filter,
    # Block size for oscillators (compile-time, >= num_osc, power of 2)
    BLOCK_N: tl.constexpr,
    # Epilogue selector (0 = complex (re, im), 1 = magnitude |y|)
    EPILOGUE: tl.constexpr,
):
    """Fused multi-oscillator scan with C-weight reduction — O(T) sequential.

    Each program handles one (batch, physical_filter) pair.
    Maintains BLOCK_N oscillator states in vector registers.
    Reads u ONCE per timestep (shared across all oscillators in the filter).
    Writes ONE reduced complex output per timestep (after C-weight combination).

    Implements standard complex recurrence for each oscillator:
        x_n[t] = C1_n * x_n[t-1] + C2_n * u[t]

    Then fuses C-weight reduction:
        out[t] = Σ_n (cw_r_n + j*cw_i_n) * (x_r_n[t] + j*x_i_n[t])

    ``EPILOGUE`` is a compile-time switch:
      0 = COMPLEX: write out_r to out_real_ptr, out_i to out_imag_ptr.
      1 = ABS:     write sqrt(out_r**2 + out_i**2) to out_real_ptr only.
                   out_imag_ptr is not dereferenced (pass a dummy 1-element
                   buffer). Intended for eval / no_grad magnitude readouts.
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_filters
    filter_idx = pid % num_filters

    if batch_idx >= batch_size:
        return

    # Oscillator index range for this physical filter (uniform layout)
    osc_start = filter_idx * num_osc
    osc_range = tl.arange(0, BLOCK_N)
    osc_mask = osc_range < num_osc
    coeff_idx = osc_start + osc_range

    # Load per-oscillator coefficients into BLOCK_N-wide vectors
    # Masked loads return 0.0 for unused oscillator slots
    c1r = tl.load(c1_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c1i = tl.load(c1_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2r = tl.load(c2_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2i = tl.load(c2_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)

    # Load C-weights for combination (BLOCK_N-wide)
    cw_r = tl.load(cw_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    cw_i = tl.load(cw_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)

    # Initialize BLOCK_N oscillator states in registers (all zero)
    # Unused slots stay zero because their coefficients are all zero
    x_r = tl.zeros([BLOCK_N], dtype=tl.float32)
    x_i = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Process all timesteps sequentially
    for t in range(time_size):
        u_offset = batch_idx * u_stride_batch + t * u_stride_time + filter_idx * u_stride_filter
        u_r = tl.load(u_real_ptr + u_offset)  # scalar, shared by all N oscillators
        u_i = tl.load(u_imag_ptr + u_offset)  # scalar, shared by all N oscillators

        # Standard complex recurrence: x_new = C1 * x + C2 * u
        x_r_new = c1r * x_r - c1i * x_i + c2r * u_r - c2i * u_i
        x_i_new = c1i * x_r + c1r * x_i + c2i * u_r + c2r * u_i
        x_r = x_r_new
        x_i = x_i_new

        # Fused C-weight reduction: combine N oscillators into 1 complex output
        # out = Σ_n (cw_r_n + j*cw_i_n) * (x_r_n + j*x_i_n)
        #     = Σ_n (cw_r_n*x_r_n - cw_i_n*x_i_n) + j*(cw_r_n*x_i_n + cw_i_n*x_r_n)
        out_r = tl.sum(cw_r * x_r - cw_i * x_i, axis=0)
        out_i = tl.sum(cw_r * x_i + cw_i * x_r, axis=0)

        out_offset = (
            batch_idx * out_stride_batch + t * out_stride_time + filter_idx * out_stride_filter
        )
        if EPILOGUE == 0:
            tl.store(out_real_ptr + out_offset, out_r)
            tl.store(out_imag_ptr + out_offset, out_i)
        else:
            tl.store(out_real_ptr + out_offset, tl.sqrt(out_r * out_r + out_i * out_i))


def _sequential_scan_multi_osc_forward_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    cw_real: torch.Tensor,
    cw_imag: torch.Tensor,
    num_osc: int,
) -> torch.Tensor:
    """Raw forward pass for fused multi-oscillator complex scan - calls Triton kernel.

    Args:
        C1: Complex tensor of shape (scan_filters,) - per-oscillator recurrence.
        C2: Complex tensor of shape (scan_filters,) - per-oscillator input gain.
        u: Complex tensor of shape (batch, time, num_filters).
        cw_real: Real tensor of shape (scan_filters,) - C-weight real parts.
        cw_imag: Real tensor of shape (scan_filters,) - C-weight imag parts.
        num_osc: Number of oscillators per filter (uniform).

    Returns:
        Complex tensor of shape (batch, time, num_filters) - reduced output.
    """
    # Ensure input is complex
    if not u.is_complex():
        u = u.to(torch.cfloat)

    batch_size, time_size, num_filters = u.shape

    # Ensure coefficients are complex
    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)

    u = u.contiguous()

    # Split into real/imaginary parts
    target_dtype = u.real.dtype
    c1_real = C1.real.to(target_dtype).contiguous()
    c1_imag = C1.imag.to(target_dtype).contiguous()
    c2_real = C2.real.to(target_dtype).contiguous()
    c2_imag = C2.imag.to(target_dtype).contiguous()
    u_real = u.real.contiguous()
    u_imag = u.imag.contiguous()
    cw_real = cw_real.to(target_dtype).contiguous()
    cw_imag = cw_imag.to(target_dtype).contiguous()

    # Allocate REDUCED output (num_filters, not scan_filters!)
    out_real = torch.zeros(batch_size, time_size, num_filters, device=u.device, dtype=target_dtype)
    out_imag = torch.zeros_like(out_real)

    # Block size for oscillators (power of 2 >= num_osc)
    BLOCK_N = triton.next_power_of_2(num_osc)

    # Launch: one program per (batch, physical_filter) pair
    grid = (batch_size * num_filters,)

    _sequential_scan_multi_osc_kernel[grid](
        u_real,
        u_imag,
        out_real,
        out_imag,
        c1_real,
        c1_imag,
        c2_real,
        c2_imag,
        cw_real,
        cw_imag,
        batch_size,
        time_size,
        num_filters,
        num_osc,
        u_real.stride(0),
        u_real.stride(1),
        u_real.stride(2),
        out_real.stride(0),
        out_real.stride(1),
        out_real.stride(2),
        BLOCK_N=BLOCK_N,
        EPILOGUE=0,
    )

    return torch.complex(out_real, out_imag)


def _sequential_scan_multi_osc_forward_abs_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    cw_real: torch.Tensor,
    cw_imag: torch.Tensor,
    num_osc: int,
) -> torch.Tensor:
    """Fused-``abs()`` epilogue variant of ``_sequential_scan_multi_osc_forward_raw``.

    Returns a real tensor ``|y[t]|`` directly after the C-weight reduction,
    halving the output HBM traffic and skipping the downstream ``.abs()``
    pass. Backward still needs the full complex output so this has no autograd
    wrapper; training stays on the complex path.
    """
    if not u.is_complex():
        u = u.to(torch.cfloat)

    batch_size, time_size, num_filters = u.shape

    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)

    u = u.contiguous()
    target_dtype = u.real.dtype
    c1_real = C1.real.to(target_dtype).contiguous()
    c1_imag = C1.imag.to(target_dtype).contiguous()
    c2_real = C2.real.to(target_dtype).contiguous()
    c2_imag = C2.imag.to(target_dtype).contiguous()
    u_real = u.real.contiguous()
    u_imag = u.imag.contiguous()
    cw_real = cw_real.to(target_dtype).contiguous()
    cw_imag = cw_imag.to(target_dtype).contiguous()

    out = torch.empty(batch_size, time_size, num_filters, device=u.device, dtype=target_dtype)
    out_imag_dummy = torch.empty(1, device=u.device, dtype=target_dtype)

    BLOCK_N = triton.next_power_of_2(num_osc)
    grid = (batch_size * num_filters,)

    _sequential_scan_multi_osc_kernel[grid](
        u_real,
        u_imag,
        out,
        out_imag_dummy,
        c1_real,
        c1_imag,
        c2_real,
        c2_imag,
        cw_real,
        cw_imag,
        batch_size,
        time_size,
        num_filters,
        num_osc,
        u_real.stride(0),
        u_real.stride(1),
        u_real.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_N=BLOCK_N,
        EPILOGUE=1,
    )

    return out


@triton.jit
def _sequential_scan_multi_osc_backward_kernel(  # pragma: no cover  # noqa: CCR001
    # Forward inputs (complex, stored as real/imag, shape: batch, time, num_filters)
    u_real_ptr,
    u_imag_ptr,
    # Per-oscillator coefficients (shape: scan_filters = num_filters * num_osc)
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    # C-weights (shape: scan_filters)
    cw_real_ptr,
    cw_imag_ptr,
    # Grad output (shape: batch, time, num_filters)
    grad_out_real_ptr,
    grad_out_imag_ptr,
    # Scratch buffers (shape: batch_size * num_filters, time_size, BLOCK_N)
    scratch_r_ptr,
    scratch_i_ptr,
    # Output: grad_u (complex, shape: batch, time, num_filters)
    grad_u_real_ptr,
    grad_u_imag_ptr,
    # Output: coefficient grads (shape: scan_filters) — atomic add targets
    grad_c1r_ptr,
    grad_c1i_ptr,
    grad_c2r_ptr,
    grad_c2i_ptr,
    grad_cwr_ptr,
    grad_cwi_ptr,
    # Dimensions
    batch_size,
    time_size,
    num_filters,
    num_osc,
    # Strides for (B,T,F) tensors (u, grad_out, grad_u share layout)
    stride_batch,
    stride_time,
    stride_filter,
    # Block size for oscillators (constexpr, power of 2 >= num_osc)
    BLOCK_N: tl.constexpr,
    # Compile-time grad gates driven by ctx.needs_input_grad
    COMPUTE_GRAD_C1: tl.constexpr,
    COMPUTE_GRAD_C2: tl.constexpr,
    COMPUTE_GRAD_U: tl.constexpr,
    COMPUTE_GRAD_CW: tl.constexpr,
):
    """Fused backward kernel for multi-oscillator complex scan with C-weight reduction.

    Two-phase approach within a single kernel launch:
    Phase 1 (forward, t=0..T-1): Recompute oscillator states in BLOCK_N-wide
        registers, save to scratch (only when grad_C1 is needed), accumulate
        grad_c_weights (only when grad_CW is needed).
    Phase 2 (backward, t=T-1..0): Adjoint recurrence using conj(C1), compute
        grad_u/grad_C1/grad_C2 selectively.

    ``COMPUTE_GRAD_*`` are ``tl.constexpr`` flags driven by
    ``ctx.needs_input_grad`` in the autograd wrapper, so the frozen-recurrence
    fine-tuning path pays no FLOPs/HBM for the gradients it discards. The
    scratch-write of Phase 1 is additionally gated by ``COMPUTE_GRAD_C1`` —
    no C1 gradient → no scratch footprint.

    Each program handles one (batch, physical_filter) pair.
    Grid: (batch_size * num_filters,).
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_filters
    filter_idx = pid % num_filters

    if batch_idx >= batch_size:
        return

    # --- Load per-oscillator coefficients (BLOCK_N-wide vectors) ---
    osc_start = filter_idx * num_osc
    osc_range = tl.arange(0, BLOCK_N)
    osc_mask = osc_range < num_osc
    coeff_idx = osc_start + osc_range

    c1r = tl.load(c1_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c1i = tl.load(c1_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2r = tl.load(c2_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2i = tl.load(c2_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)
    cw_r = tl.load(cw_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    cw_i = tl.load(cw_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)

    # Initialize oscillator states
    x_r = tl.zeros([BLOCK_N], dtype=tl.float32)
    x_i = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Accumulators for grad_c_weights (summed over time, batch via atomic)
    gcw_r = tl.zeros([BLOCK_N], dtype=tl.float32)
    gcw_i = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Scratch base offset for this program (int64 to avoid overflow)
    scratch_pid_off = pid.to(tl.int64) * (time_size * BLOCK_N)

    # ====== Phase 1: Forward sweep (t=0..T-1) ======
    # Only needed when we must either:
    #   1. stash per-oscillator states for grad_C1, or
    #   2. accumulate grad_cw from x[t].
    # If both are frozen, Phase 2 can run directly from grad_output/u/C*
    # without ever re-materializing x[t].
    if COMPUTE_GRAD_C1 or COMPUTE_GRAD_CW:
        for t in range(time_size):
            btf_off = batch_idx * stride_batch + t * stride_time + filter_idx * stride_filter
            u_r = tl.load(u_real_ptr + btf_off)
            u_i = tl.load(u_imag_ptr + btf_off)

            # Standard complex recurrence across BLOCK_N oscillators
            x_r_new = c1r * x_r - c1i * x_i + c2r * u_r - c2i * u_i
            x_i_new = c1i * x_r + c1r * x_i + c2i * u_r + c2r * u_i
            x_r = x_r_new
            x_i = x_i_new

            # Save to scratch for Phase 2 (only consumed by grad_C1)
            if COMPUTE_GRAD_C1:
                scratch_off = scratch_pid_off + t * BLOCK_N + osc_range
                tl.store(scratch_r_ptr + scratch_off, x_r, mask=osc_mask)
                tl.store(scratch_i_ptr + scratch_off, x_i, mask=osc_mask)

            # Accumulate grad_cw: Σ_t grad_y[t] * conj(x_n[t])
            if COMPUTE_GRAD_CW:
                go_r = tl.load(grad_out_real_ptr + btf_off)
                go_i = tl.load(grad_out_imag_ptr + btf_off)
                gcw_r += go_r * x_r + go_i * x_i
                gcw_i += go_i * x_r - go_r * x_i

    # ====== Phase 2: Backward sweep (t=T-1..0) ======
    # Adjoint recurrence: adj[t] = conj(C1) * adj[t+1] + gs[t]
    # adj_r_new = c1r*adj_r + c1i*adj_i + gs_r
    # adj_i_new = -c1i*adj_r + c1r*adj_i + gs_i

    # Initialize adjoint state (adj[T] = 0)
    adj_r = tl.zeros([BLOCK_N], dtype=tl.float32)
    adj_i = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Accumulators for grad_c1, grad_c2 (per oscillator, summed over time)
    gc1r = tl.zeros([BLOCK_N], dtype=tl.float32)
    gc1i = tl.zeros([BLOCK_N], dtype=tl.float32)
    gc2r = tl.zeros([BLOCK_N], dtype=tl.float32)
    gc2i = tl.zeros([BLOCK_N], dtype=tl.float32)

    for t_rev in range(time_size):
        t = time_size - 1 - t_rev

        btf_off = batch_idx * stride_batch + t * stride_time + filter_idx * stride_filter
        go_r = tl.load(grad_out_real_ptr + btf_off)
        go_i = tl.load(grad_out_imag_ptr + btf_off)

        # Per-oscillator scan gradient from C-weights: conj(cw) * grad_y
        gs_r = cw_r * go_r + cw_i * go_i
        gs_i = cw_r * go_i - cw_i * go_r

        # Adjoint recurrence: adj[t] = conj(C1) * adj[t+1] + gs[t]
        adj_r_new = c1r * adj_r + c1i * adj_i + gs_r
        adj_i_new = -c1i * adj_r + c1r * adj_i + gs_i

        if COMPUTE_GRAD_U:
            # grad_u[t] = Σ_n conj(C2_n) * adj_n[t]
            grad_u_r_val = tl.sum(c2r * adj_r_new + c2i * adj_i_new, axis=0)
            grad_u_i_val = tl.sum(-c2i * adj_r_new + c2r * adj_i_new, axis=0)
            tl.store(grad_u_real_ptr + btf_off, grad_u_r_val)
            tl.store(grad_u_imag_ptr + btf_off, grad_u_i_val)

        if COMPUTE_GRAD_C2:
            u_r = tl.load(u_real_ptr + btf_off)
            u_i = tl.load(u_imag_ptr + btf_off)
            gc2r += adj_r_new * u_r + adj_i_new * u_i
            gc2i += adj_i_new * u_r - adj_r_new * u_i

        if COMPUTE_GRAD_C1 and t > 0:
            scratch_off_prev = scratch_pid_off + (t - 1) * BLOCK_N + osc_range
            x_r_prev = tl.load(scratch_r_ptr + scratch_off_prev, mask=osc_mask, other=0.0)
            x_i_prev = tl.load(scratch_i_ptr + scratch_off_prev, mask=osc_mask, other=0.0)
            gc1r += adj_r_new * x_r_prev + adj_i_new * x_i_prev
            gc1i += adj_i_new * x_r_prev - adj_r_new * x_i_prev

        adj_r = adj_r_new
        adj_i = adj_i_new

    # ====== Write coefficient gradients (atomic add across batch programs) ======
    if COMPUTE_GRAD_C1:
        tl.atomic_add(grad_c1r_ptr + coeff_idx, gc1r, mask=osc_mask)
        tl.atomic_add(grad_c1i_ptr + coeff_idx, gc1i, mask=osc_mask)
    if COMPUTE_GRAD_C2:
        tl.atomic_add(grad_c2r_ptr + coeff_idx, gc2r, mask=osc_mask)
        tl.atomic_add(grad_c2i_ptr + coeff_idx, gc2i, mask=osc_mask)
    if COMPUTE_GRAD_CW:
        tl.atomic_add(grad_cwr_ptr + coeff_idx, gcw_r, mask=osc_mask)
        tl.atomic_add(grad_cwi_ptr + coeff_idx, gcw_i, mask=osc_mask)


def _sequential_scan_multi_osc_backward_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    c_weights: torch.Tensor,
    grad_output: torch.Tensor,
    num_osc: int,
    *,
    compute_grad_C1: bool = True,
    compute_grad_C2: bool = True,
    compute_grad_u: bool = True,
    compute_grad_cw: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Raw backward for fused multi-oscillator complex scan with grad gating.

    The ``compute_grad_*`` flags (driven by ``ctx.needs_input_grad`` in the
    autograd wrapper) gate the per-gradient work inside the kernel and the
    O(B*T*F) grad_u allocation. Specifically:

    - ``compute_grad_C1=False`` additionally skips the Phase-1 scratch writes
      and the Phase-2 scratch reads + atomic_add for grad_C1.
    - ``compute_grad_C1=False`` and ``compute_grad_cw=False`` skips Phase 1
      entirely: no forward recompute is needed when only grad_u / grad_C2
      are requested.
    - ``compute_grad_u=False`` drops the O(B*T*F) grad_u tensor allocation
      entirely (dummy 1-element buffer is passed so the pointer is valid).

    Returns:
        ``(grad_C1, grad_C2, grad_u, grad_c_weights)`` where each entry is
        ``None`` when the matching ``compute_grad_*`` is False.
    """
    if not (compute_grad_C1 or compute_grad_C2 or compute_grad_u or compute_grad_cw):
        return None, None, None, None

    if not u.is_complex():
        u = u.to(torch.cfloat)

    batch_size, time_size, num_filters = u.shape
    scan_filters = C1.shape[0]

    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)
    u = u.contiguous()

    if not grad_output.is_complex():
        grad_output = grad_output.to(torch.cfloat)

    target_dtype = u.real.dtype
    c1_real = C1.real.to(target_dtype).contiguous()
    c1_imag = C1.imag.to(target_dtype).contiguous()
    c2_real = C2.real.to(target_dtype).contiguous()
    c2_imag = C2.imag.to(target_dtype).contiguous()
    cw_real = c_weights[:, 0].to(target_dtype).contiguous()
    cw_imag = c_weights[:, 1].to(target_dtype).contiguous()
    u_real = u.real.contiguous()
    u_imag = u.imag.contiguous()
    grad_out_real = grad_output.real.to(target_dtype).contiguous()
    grad_out_imag = grad_output.imag.to(target_dtype).contiguous()

    # Scratch only needed when grad_C1 is requested (Phase-2 reads it back).
    BLOCK_N = triton.next_power_of_2(num_osc)
    num_programs = batch_size * num_filters
    if compute_grad_C1:
        scratch_r = torch.empty(
            num_programs, time_size, BLOCK_N, device=u.device, dtype=target_dtype
        )
        scratch_i = torch.empty_like(scratch_r)
    else:
        scratch_r = torch.empty(1, device=u.device, dtype=target_dtype)
        scratch_i = scratch_r

    if compute_grad_u:
        grad_u_real = torch.empty(
            batch_size, time_size, num_filters, device=u.device, dtype=target_dtype
        )
        grad_u_imag = torch.empty_like(grad_u_real)
    else:
        grad_u_real = torch.empty(1, device=u.device, dtype=target_dtype)
        grad_u_imag = grad_u_real
    grad_c1r = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_c1i = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_c2r = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_c2i = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_cwr = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_cwi = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)

    grid = (num_programs,)

    _sequential_scan_multi_osc_backward_kernel[grid](
        u_real,
        u_imag,
        c1_real,
        c1_imag,
        c2_real,
        c2_imag,
        cw_real,
        cw_imag,
        grad_out_real,
        grad_out_imag,
        scratch_r,
        scratch_i,
        grad_u_real,
        grad_u_imag,
        grad_c1r,
        grad_c1i,
        grad_c2r,
        grad_c2i,
        grad_cwr,
        grad_cwi,
        batch_size,
        time_size,
        num_filters,
        num_osc,
        u_real.stride(0),
        u_real.stride(1),
        u_real.stride(2),
        BLOCK_N=BLOCK_N,
        COMPUTE_GRAD_C1=compute_grad_C1,
        COMPUTE_GRAD_C2=compute_grad_C2,
        COMPUTE_GRAD_U=compute_grad_u,
        COMPUTE_GRAD_CW=compute_grad_cw,
    )

    grad_C1 = torch.complex(grad_c1r, grad_c1i) if compute_grad_C1 else None
    grad_C2 = torch.complex(grad_c2r, grad_c2i) if compute_grad_C2 else None
    grad_u = torch.complex(grad_u_real, grad_u_imag) if compute_grad_u else None
    grad_c_weights = torch.stack([grad_cwr, grad_cwi], dim=-1) if compute_grad_cw else None

    return grad_C1, grad_C2, grad_u, grad_c_weights


class _SequentialScanMultiOscFunction(torch.autograd.Function):  # pragma: no cover
    """Autograd function for fused multi-oscillator complex scan with C-weight reduction.

    Forward: Uses fused Triton kernel — N oscillators per physical filter,
    shared input u, reduced output via C-weights.

    Backward: Uses fused Triton kernel — two-phase approach (forward recompute
    + adjoint) in a single kernel launch. All oscillator states maintained in
    BLOCK_N-wide registers. No intermediate (B,T,F*N) tensor allocations.

    Saves only (C1, C2, u, c_weights) for backward — persistent activations
    across layers are N× smaller.
    """

    @staticmethod
    def forward(
        ctx,
        C1: torch.Tensor,
        C2: torch.Tensor,
        u: torch.Tensor,
        c_weights: torch.Tensor,
        num_osc: int,
    ) -> torch.Tensor:
        """Forward pass using fused Triton kernel.

        Args:
            ctx: Autograd context.
            C1: Complex (scan_filters,) - per-oscillator recurrence coefficient.
            C2: Complex (scan_filters,) - per-oscillator input gain.
            u: Complex (batch, time, num_filters) - input signal.
            c_weights: Real (scan_filters, 2) - C-weight [real, imag] per oscillator.
            num_osc: int - uniform oscillators per filter.

        Returns:
            Complex (batch, time, num_filters) - reduced output.
        """
        # Ensure types
        if not C1.is_complex():
            C1 = C1.to(torch.cfloat)
        if not C2.is_complex():
            C2 = C2.to(torch.cfloat)
        if not u.is_complex():
            u = u.to(torch.cfloat)

        # Split c_weights into separate real/imag tensors for kernel
        cw_real = c_weights[:, 0].contiguous()
        cw_imag = c_weights[:, 1].contiguous()

        # Run fused forward kernel (output is reduced: num_filters, not scan_filters)
        output = _sequential_scan_multi_osc_forward_raw(C1, C2, u, cw_real, cw_imag, num_osc)

        # Save for backward — NO full oscillator states! Just inputs + weights.
        ctx.save_for_backward(C1, C2, u, c_weights)
        ctx.num_osc = num_osc

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass using fused Triton kernel.

        Single kernel launch handles forward state recomputation and adjoint
        recurrence in two phases, keeping everything in BLOCK_N-wide registers.
        ``ctx.needs_input_grad`` is forwarded so frozen-recurrent fine-tuning
        pays no FLOPs / atomics / HBM for the gradients it would discard.
        """
        C1, C2, u, c_weights = ctx.saved_tensors
        num_osc = ctx.num_osc
        need_C1, need_C2, need_u, need_cw, _ = ctx.needs_input_grad

        grad_C1, grad_C2, grad_u, grad_c_weights = _sequential_scan_multi_osc_backward_raw(
            C1,
            C2,
            u,
            c_weights,
            grad_output,
            num_osc,
            compute_grad_C1=need_C1,
            compute_grad_C2=need_C2,
            compute_grad_u=need_u,
            compute_grad_cw=need_cw,
        )

        return grad_C1, grad_C2, grad_u, grad_c_weights, None  # None for num_osc


def _triton_sequential_scan_multi_osc(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    c_weights: torch.Tensor,
    num_osc: int,
) -> torch.Tensor:
    """Fused multi-oscillator complex scan with C-weight reduction and autograd.

    For N>1 oscillators per physical filter, this fuses:
    1. N independent complex scans sharing the same input u per physical filter
    2. C-weight combination: y[t,f] = Σ_n c_n * x_n[t]

    Memory savings vs separate scan + reduction:
    - Input: no repeat_interleave needed (N× less input memory)
    - Output: (B,T,F) instead of (B,T,F*N) (N× less output memory)
    - Persistent: saved activations across layers are N× smaller

    Args:
        C1: Complex (scan_filters,) - per-oscillator recurrence coefficient.
        C2: Complex (scan_filters,) - per-oscillator input gain.
        u: Complex (batch, time, num_filters) - input signal.
        c_weights: Real (scan_filters, 2) - C-weight [real, imag] per oscillator.
        num_osc: int - uniform number of oscillators per filter.

    Returns:
        Complex (batch, time, num_filters) - C-weight-reduced output.
    """
    return _SequentialScanMultiOscFunction.apply(C1, C2, u, c_weights, num_osc)


@triton.jit
def _sequential_scan_multi_osc_real_kernel(  # pragma: no cover
    # Input pointer (real, shape: batch, time, num_filters)
    u_real_ptr,
    # Output pointers (reduced: batch, time, num_filters)
    out_real_ptr,
    out_imag_ptr,
    # Per-oscillator coefficient pointers (shape: scan_filters = num_filters * num_osc)
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    # C-weight pointers (shape: scan_filters, separate real/imag)
    cw_real_ptr,
    cw_imag_ptr,
    # Dimensions
    batch_size,
    time_size,
    num_filters,
    num_osc,
    # Input strides (for u with num_filters channels)
    u_stride_batch,
    u_stride_time,
    u_stride_filter,
    # Output strides (same layout as input, num_filters channels)
    out_stride_batch,
    out_stride_time,
    out_stride_filter,
    # Block size for oscillators (compile-time, >= num_osc, power of 2)
    BLOCK_N: tl.constexpr,
    # Epilogue selector (0 = complex (re, im), 1 = magnitude |y|)
    EPILOGUE: tl.constexpr,
):
    """Fused multi-oscillator scan with C-weight reduction — real-input
    specialization of the same standard complex recurrence used by
    ``_sequential_scan_multi_osc_kernel``.

    For each oscillator n, applies the same complex equation
    ``x_n[t] = C1_n * x_n[t-1] + C2_n * u[t]`` with ``Im(u) == 0`` folded in
    at compile time, then fuses the C-weight reduction
    ``out[t] = Σ_n (cw_r_n + j*cw_i_n) * x_n[t]``. All oscillator states and
    the reduced output remain complex. Numerically identical to the
    complex-input path called with ``u + 0j``. NOT the deprecated
    real-input equation with an extra imaginary term.

    ``EPILOGUE`` is a compile-time switch:
      0 = COMPLEX: write out_r to out_real_ptr, out_i to out_imag_ptr.
      1 = ABS:     write sqrt(out_r**2 + out_i**2) to out_real_ptr only.
                   out_imag_ptr is not dereferenced.
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_filters
    filter_idx = pid % num_filters

    if batch_idx >= batch_size:
        return

    osc_start = filter_idx * num_osc
    osc_range = tl.arange(0, BLOCK_N)
    osc_mask = osc_range < num_osc
    coeff_idx = osc_start + osc_range

    c1r = tl.load(c1_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c1i = tl.load(c1_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2r = tl.load(c2_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2i = tl.load(c2_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)
    cw_r = tl.load(cw_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    cw_i = tl.load(cw_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)

    x_r = tl.zeros([BLOCK_N], dtype=tl.float32)
    x_i = tl.zeros([BLOCK_N], dtype=tl.float32)

    for t in range(time_size):
        u_offset = batch_idx * u_stride_batch + t * u_stride_time + filter_idx * u_stride_filter
        u_r = tl.load(u_real_ptr + u_offset)

        x_r_new = c1r * x_r - c1i * x_i + c2r * u_r
        x_i_new = c1i * x_r + c1r * x_i + c2i * u_r
        x_r = x_r_new
        x_i = x_i_new

        out_r = tl.sum(cw_r * x_r - cw_i * x_i, axis=0)
        out_i = tl.sum(cw_r * x_i + cw_i * x_r, axis=0)

        out_offset = (
            batch_idx * out_stride_batch + t * out_stride_time + filter_idx * out_stride_filter
        )
        if EPILOGUE == 0:
            tl.store(out_real_ptr + out_offset, out_r)
            tl.store(out_imag_ptr + out_offset, out_i)
        else:
            tl.store(out_real_ptr + out_offset, tl.sqrt(out_r * out_r + out_i * out_i))


def _sequential_scan_multi_osc_real_forward_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    c_weights: torch.Tensor,
    num_osc: int,
) -> torch.Tensor:
    """Raw forward pass for the real-input specialization of the fused
    multi-oscillator complex scan. Not a separate equation — see module
    docstring."""
    if u.is_complex():
        raise ValueError("_sequential_scan_multi_osc_real_forward_raw expects real input")
    if u.dtype != torch.float32:
        u = u.to(torch.float32)

    batch_size, time_size, num_filters = u.shape

    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)

    target_dtype = u.dtype
    c1_real = C1.real.to(target_dtype).contiguous()
    c1_imag = C1.imag.to(target_dtype).contiguous()
    c2_real = C2.real.to(target_dtype).contiguous()
    c2_imag = C2.imag.to(target_dtype).contiguous()
    u_real = u.contiguous()
    cw_real = c_weights[:, 0].to(target_dtype).contiguous()
    cw_imag = c_weights[:, 1].to(target_dtype).contiguous()

    out_real = torch.empty(batch_size, time_size, num_filters, device=u.device, dtype=target_dtype)
    out_imag = torch.empty_like(out_real)

    BLOCK_N = triton.next_power_of_2(num_osc)
    grid = (batch_size * num_filters,)

    _sequential_scan_multi_osc_real_kernel[grid](
        u_real,
        out_real,
        out_imag,
        c1_real,
        c1_imag,
        c2_real,
        c2_imag,
        cw_real,
        cw_imag,
        batch_size,
        time_size,
        num_filters,
        num_osc,
        u_real.stride(0),
        u_real.stride(1),
        u_real.stride(2),
        out_real.stride(0),
        out_real.stride(1),
        out_real.stride(2),
        BLOCK_N=BLOCK_N,
        EPILOGUE=0,
    )

    return torch.complex(out_real, out_imag)


def _sequential_scan_multi_osc_real_forward_abs_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    c_weights: torch.Tensor,
    num_osc: int,
) -> torch.Tensor:
    """Fused-``abs()`` epilogue variant of ``_sequential_scan_multi_osc_real_forward_raw``.

    Returns ``|y[t]|`` directly after the C-weight reduction. Real input,
    real output. Backward still needs the full complex output so this has no
    autograd wrapper; training stays on the complex path.
    """
    if u.is_complex():
        raise ValueError("_sequential_scan_multi_osc_real_forward_abs_raw expects real input")
    if u.dtype != torch.float32:
        u = u.to(torch.float32)

    batch_size, time_size, num_filters = u.shape

    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)

    target_dtype = u.dtype
    c1_real = C1.real.to(target_dtype).contiguous()
    c1_imag = C1.imag.to(target_dtype).contiguous()
    c2_real = C2.real.to(target_dtype).contiguous()
    c2_imag = C2.imag.to(target_dtype).contiguous()
    u_real = u.contiguous()
    cw_real = c_weights[:, 0].to(target_dtype).contiguous()
    cw_imag = c_weights[:, 1].to(target_dtype).contiguous()

    out = torch.empty(batch_size, time_size, num_filters, device=u.device, dtype=target_dtype)
    out_imag_dummy = torch.empty(1, device=u.device, dtype=target_dtype)

    BLOCK_N = triton.next_power_of_2(num_osc)
    grid = (batch_size * num_filters,)

    _sequential_scan_multi_osc_real_kernel[grid](
        u_real,
        out,
        out_imag_dummy,
        c1_real,
        c1_imag,
        c2_real,
        c2_imag,
        cw_real,
        cw_imag,
        batch_size,
        time_size,
        num_filters,
        num_osc,
        u_real.stride(0),
        u_real.stride(1),
        u_real.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_N=BLOCK_N,
        EPILOGUE=1,
    )

    return out


@triton.jit
def _sequential_scan_multi_osc_backward_real_kernel(  # pragma: no cover  # noqa: CCR001
    # Forward inputs (real, shape: batch, time, num_filters)
    u_real_ptr,
    # Per-oscillator coefficients (shape: scan_filters = num_filters * num_osc)
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    # C-weights (shape: scan_filters)
    cw_real_ptr,
    cw_imag_ptr,
    # Grad output (shape: batch, time, num_filters)
    grad_out_real_ptr,
    grad_out_imag_ptr,
    # Scratch buffers (shape: batch_size * num_filters, time_size, BLOCK_N)
    scratch_r_ptr,
    scratch_i_ptr,
    # Output: grad_u (real, shape: batch, time, num_filters)
    grad_u_ptr,
    # Output: coefficient grads (shape: scan_filters) — atomic add targets
    grad_c1r_ptr,
    grad_c1i_ptr,
    grad_c2r_ptr,
    grad_c2i_ptr,
    grad_cwr_ptr,
    grad_cwi_ptr,
    # Dimensions
    batch_size,
    time_size,
    num_filters,
    num_osc,
    # Strides for (B,T,F) tensors (u, grad_out, grad_u share layout)
    stride_batch,
    stride_time,
    stride_filter,
    # Block size for oscillators (constexpr, power of 2 >= num_osc)
    BLOCK_N: tl.constexpr,
    # Compile-time grad gates driven by ctx.needs_input_grad
    COMPUTE_GRAD_C1: tl.constexpr,
    COMPUTE_GRAD_C2: tl.constexpr,
    COMPUTE_GRAD_U: tl.constexpr,
    COMPUTE_GRAD_CW: tl.constexpr,
):
    """Fused backward kernel — real-input specialization of the same adjoint
    recurrence used by ``_sequential_scan_multi_osc_backward_kernel``.

    Same two-phase scheme (forward recompute + adjoint backward sweep), with
    ``Im(u) == 0`` folded in. ``grad_u`` is real; oscillator state scratch and
    coefficient/c-weight gradients remain complex. Not a separate equation —
    see module docstring.

    ``COMPUTE_GRAD_*`` are ``tl.constexpr`` flags driven by
    ``ctx.needs_input_grad`` in the autograd wrapper, gating FLOPs / atomics /
    HBM per gradient. ``COMPUTE_GRAD_C1=False`` additionally drops the Phase-1
    scratch writes and Phase-2 scratch reads used exclusively by ``grad_C1``.
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_filters
    filter_idx = pid % num_filters

    if batch_idx >= batch_size:
        return

    osc_start = filter_idx * num_osc
    osc_range = tl.arange(0, BLOCK_N)
    osc_mask = osc_range < num_osc
    coeff_idx = osc_start + osc_range

    c1r = tl.load(c1_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c1i = tl.load(c1_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2r = tl.load(c2_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2i = tl.load(c2_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)
    cw_r = tl.load(cw_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    cw_i = tl.load(cw_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)

    x_r = tl.zeros([BLOCK_N], dtype=tl.float32)
    x_i = tl.zeros([BLOCK_N], dtype=tl.float32)
    gcw_r = tl.zeros([BLOCK_N], dtype=tl.float32)
    gcw_i = tl.zeros([BLOCK_N], dtype=tl.float32)

    scratch_pid_off = pid.to(tl.int64) * (time_size * BLOCK_N)

    if COMPUTE_GRAD_C1 or COMPUTE_GRAD_CW:
        for t in range(time_size):
            btf_off = batch_idx * stride_batch + t * stride_time + filter_idx * stride_filter
            u_r = tl.load(u_real_ptr + btf_off)

            x_r_new = c1r * x_r - c1i * x_i + c2r * u_r
            x_i_new = c1i * x_r + c1r * x_i + c2i * u_r
            x_r = x_r_new
            x_i = x_i_new

            if COMPUTE_GRAD_C1:
                scratch_off = scratch_pid_off + t * BLOCK_N + osc_range
                tl.store(scratch_r_ptr + scratch_off, x_r, mask=osc_mask)
                tl.store(scratch_i_ptr + scratch_off, x_i, mask=osc_mask)

            if COMPUTE_GRAD_CW:
                go_r = tl.load(grad_out_real_ptr + btf_off)
                go_i = tl.load(grad_out_imag_ptr + btf_off)
                gcw_r += go_r * x_r + go_i * x_i
                gcw_i += go_i * x_r - go_r * x_i

    adj_r = tl.zeros([BLOCK_N], dtype=tl.float32)
    adj_i = tl.zeros([BLOCK_N], dtype=tl.float32)
    gc1r = tl.zeros([BLOCK_N], dtype=tl.float32)
    gc1i = tl.zeros([BLOCK_N], dtype=tl.float32)
    gc2r = tl.zeros([BLOCK_N], dtype=tl.float32)
    gc2i = tl.zeros([BLOCK_N], dtype=tl.float32)

    for t_rev in range(time_size):
        t = time_size - 1 - t_rev

        btf_off = batch_idx * stride_batch + t * stride_time + filter_idx * stride_filter
        go_r = tl.load(grad_out_real_ptr + btf_off)
        go_i = tl.load(grad_out_imag_ptr + btf_off)

        gs_r = cw_r * go_r + cw_i * go_i
        gs_i = cw_r * go_i - cw_i * go_r

        adj_r_new = c1r * adj_r + c1i * adj_i + gs_r
        adj_i_new = -c1i * adj_r + c1r * adj_i + gs_i

        if COMPUTE_GRAD_U:
            grad_u_val = tl.sum(c2r * adj_r_new + c2i * adj_i_new, axis=0)
            tl.store(grad_u_ptr + btf_off, grad_u_val)

        if COMPUTE_GRAD_C2:
            u_r = tl.load(u_real_ptr + btf_off)
            gc2r += adj_r_new * u_r
            gc2i += adj_i_new * u_r

        if COMPUTE_GRAD_C1 and t > 0:
            scratch_off_prev = scratch_pid_off + (t - 1) * BLOCK_N + osc_range
            x_r_prev = tl.load(scratch_r_ptr + scratch_off_prev, mask=osc_mask, other=0.0)
            x_i_prev = tl.load(scratch_i_ptr + scratch_off_prev, mask=osc_mask, other=0.0)
            gc1r += adj_r_new * x_r_prev + adj_i_new * x_i_prev
            gc1i += adj_i_new * x_r_prev - adj_r_new * x_i_prev

        adj_r = adj_r_new
        adj_i = adj_i_new

    if COMPUTE_GRAD_C1:
        tl.atomic_add(grad_c1r_ptr + coeff_idx, gc1r, mask=osc_mask)
        tl.atomic_add(grad_c1i_ptr + coeff_idx, gc1i, mask=osc_mask)
    if COMPUTE_GRAD_C2:
        tl.atomic_add(grad_c2r_ptr + coeff_idx, gc2r, mask=osc_mask)
        tl.atomic_add(grad_c2i_ptr + coeff_idx, gc2i, mask=osc_mask)
    if COMPUTE_GRAD_CW:
        tl.atomic_add(grad_cwr_ptr + coeff_idx, gcw_r, mask=osc_mask)
        tl.atomic_add(grad_cwi_ptr + coeff_idx, gcw_i, mask=osc_mask)


def _sequential_scan_multi_osc_backward_real_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    c_weights: torch.Tensor,
    grad_output: torch.Tensor,
    num_osc: int,
    *,
    compute_grad_C1: bool = True,
    compute_grad_C2: bool = True,
    compute_grad_u: bool = True,
    compute_grad_cw: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Raw backward for the real-input fused multi-oscillator scan with grad
    gating. ``grad_u`` is real; ``grad_C1/grad_C2/grad_c_weights`` stay complex.
    The ``compute_grad_*`` flags (driven by ``ctx.needs_input_grad``) gate
    atomic_add / scratch traffic / the O(B*T*F) grad_u allocation. When both
    ``compute_grad_C1`` and ``compute_grad_cw`` are False, the Phase-1 forward
    recompute is skipped entirely."""
    if u.is_complex():
        raise ValueError("_sequential_scan_multi_osc_backward_real_raw expects real input")
    if not (compute_grad_C1 or compute_grad_C2 or compute_grad_u or compute_grad_cw):
        return None, None, None, None
    if u.dtype != torch.float32:
        u = u.to(torch.float32)

    batch_size, time_size, num_filters = u.shape
    scan_filters = C1.shape[0]

    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)
    if not grad_output.is_complex():
        grad_output = grad_output.to(torch.cfloat)

    target_dtype = u.dtype
    c1_real = C1.real.to(target_dtype).contiguous()
    c1_imag = C1.imag.to(target_dtype).contiguous()
    c2_real = C2.real.to(target_dtype).contiguous()
    c2_imag = C2.imag.to(target_dtype).contiguous()
    cw_real = c_weights[:, 0].to(target_dtype).contiguous()
    cw_imag = c_weights[:, 1].to(target_dtype).contiguous()
    u_real = u.contiguous()
    grad_out_real = grad_output.real.to(target_dtype).contiguous()
    grad_out_imag = grad_output.imag.to(target_dtype).contiguous()

    BLOCK_N = triton.next_power_of_2(num_osc)
    num_programs = batch_size * num_filters
    if compute_grad_C1:
        scratch_r = torch.empty(
            num_programs, time_size, BLOCK_N, device=u.device, dtype=target_dtype
        )
        scratch_i = torch.empty_like(scratch_r)
    else:
        scratch_r = torch.empty(1, device=u.device, dtype=target_dtype)
        scratch_i = scratch_r

    if compute_grad_u:
        grad_u = torch.empty(
            batch_size, time_size, num_filters, device=u.device, dtype=target_dtype
        )
    else:
        grad_u = torch.empty(1, device=u.device, dtype=target_dtype)
    grad_c1r = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_c1i = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_c2r = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_c2i = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_cwr = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_cwi = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)

    grid = (num_programs,)

    _sequential_scan_multi_osc_backward_real_kernel[grid](
        u_real,
        c1_real,
        c1_imag,
        c2_real,
        c2_imag,
        cw_real,
        cw_imag,
        grad_out_real,
        grad_out_imag,
        scratch_r,
        scratch_i,
        grad_u,
        grad_c1r,
        grad_c1i,
        grad_c2r,
        grad_c2i,
        grad_cwr,
        grad_cwi,
        batch_size,
        time_size,
        num_filters,
        num_osc,
        u_real.stride(0),
        u_real.stride(1),
        u_real.stride(2),
        BLOCK_N=BLOCK_N,
        COMPUTE_GRAD_C1=compute_grad_C1,
        COMPUTE_GRAD_C2=compute_grad_C2,
        COMPUTE_GRAD_U=compute_grad_u,
        COMPUTE_GRAD_CW=compute_grad_cw,
    )

    grad_C1 = torch.complex(grad_c1r, grad_c1i) if compute_grad_C1 else None
    grad_C2 = torch.complex(grad_c2r, grad_c2i) if compute_grad_C2 else None
    grad_u_out = grad_u if compute_grad_u else None
    grad_c_weights = torch.stack([grad_cwr, grad_cwi], dim=-1) if compute_grad_cw else None
    return grad_C1, grad_C2, grad_u_out, grad_c_weights


class _SequentialScanMultiOscRealFunction(torch.autograd.Function):  # pragma: no cover
    """Autograd wrapper for the real-input specialization of the fused
    multi-oscillator complex scan. Same equation as the complex path with
    Im(u)=0."""

    @staticmethod
    def forward(
        ctx,
        C1: torch.Tensor,
        C2: torch.Tensor,
        u: torch.Tensor,
        c_weights: torch.Tensor,
        num_osc: int,
    ) -> torch.Tensor:
        if u.is_complex():
            raise ValueError("_SequentialScanMultiOscRealFunction expects real input")
        if u.dtype != torch.float32:
            u = u.to(torch.float32)

        output = _sequential_scan_multi_osc_real_forward_raw(C1, C2, u, c_weights, num_osc)

        ctx.save_for_backward(C1, C2, u, c_weights)
        ctx.num_osc = num_osc
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward via fused real-input kernel. ``ctx.needs_input_grad`` gates
        per-gradient FLOPs / atomics / allocations (see raw for details)."""
        C1, C2, u, c_weights = ctx.saved_tensors
        num_osc = ctx.num_osc
        need_C1, need_C2, need_u, need_cw, _ = ctx.needs_input_grad

        grad_C1, grad_C2, grad_u, grad_c_weights = _sequential_scan_multi_osc_backward_real_raw(
            C1,
            C2,
            u,
            c_weights,
            grad_output,
            num_osc,
            compute_grad_C1=need_C1,
            compute_grad_C2=need_C2,
            compute_grad_u=need_u,
            compute_grad_cw=need_cw,
        )

        return grad_C1, grad_C2, grad_u, grad_c_weights, None


def _triton_sequential_scan_multi_osc_real(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    c_weights: torch.Tensor,
    num_osc: int,
) -> torch.Tensor:
    """Fused multi-oscillator scan with C-weight reduction — real-input
    specialization of ``_triton_sequential_scan_multi_osc``.

    Runs the same standard complex recurrence per oscillator, then the same
    C-weight combination. Treats a real ``u`` as ``u + 0j`` and avoids the
    redundant imaginary-input traffic. NOT the deprecated real-input
    equation — see module docstring."""
    if u.is_complex():
        raise ValueError("_triton_sequential_scan_multi_osc_real expects real input")
    return _SequentialScanMultiOscRealFunction.apply(C1, C2, u, c_weights, num_osc)


# ==============================================================================
# Section 5: B-Matrix Input Projection Variants (With Autograd)
#
# These kernels support input projection through a B matrix:
#   x[n] = C1*x[n-1] + (C2 ⊙ B) @ u[n]
#
# - _sequential_scan_b_matrix_kernel: Triton kernel for complex inputs
# - _sequential_scan_b_matrix_forward_raw: Raw forward pass
# - _SequentialScanBMatrixFunction: Autograd wrapper
# - _triton_sequential_scan_b_matrix: Public API with autograd
# ==============================================================================


@triton.jit
def _sequential_scan_b_matrix_kernel(  # pragma: no cover
    # Input pointers (complex, stored as real/imag)
    u_real_ptr,
    u_imag_ptr,
    # B_complex pointer (shape: num_filters, input_features) - stored as real/imag
    b_real_ptr,
    b_imag_ptr,
    # Output pointers
    out_real_ptr,
    out_imag_ptr,
    # Coefficient pointers (shape: num_filters)
    c1_real_ptr,
    c1_imag_ptr,
    # Dimensions
    batch_size,
    time_size,
    num_filters,
    input_features,
    # Strides for u
    u_stride_batch,
    u_stride_time,
    u_stride_feature,
    # Strides for output
    out_stride_batch,
    out_stride_time,
    out_stride_filter,
    # Strides for B matrix
    b_stride_filter,
    b_stride_feature,
    # Block sizes
    BLOCK_FEATURES: tl.constexpr,
):
    """Selective scan kernel for complex inputs with B matrix.

    Implements: x[n] = C1*x[n-1] + B_complex @ u[n]

    Where B_complex = C2 ⊙ B (precomputed, complex matrix) and u is complex.

    Uses the standard complex recurrence for complex inputs.
    """
    pid = tl.program_id(0)

    batch_idx = pid // num_filters
    filter_idx = pid % num_filters

    if batch_idx >= batch_size:
        return

    # Load C1 coefficients for this filter (constant across time)
    c1r = tl.load(c1_real_ptr + filter_idx)
    c1i = tl.load(c1_imag_ptr + filter_idx)

    # Load B_complex row for this filter (once, reused for all timesteps)
    feature_offsets = tl.arange(0, BLOCK_FEATURES)
    feature_mask = feature_offsets < input_features
    b_offset = filter_idx * b_stride_filter + feature_offsets * b_stride_feature
    b_row_real = tl.load(b_real_ptr + b_offset, mask=feature_mask, other=0.0)
    b_row_imag = tl.load(b_imag_ptr + b_offset, mask=feature_mask, other=0.0)

    # State in registers (initialize with correct dtype via load-subtract trick)
    first_u_offset = batch_idx * u_stride_batch + 0 * u_stride_time
    first_u = tl.load(u_real_ptr + first_u_offset)
    x_r = first_u - first_u  # Zero with correct dtype
    x_i = first_u - first_u  # Zero with correct dtype

    # Process all timesteps sequentially
    for t in range(time_size):
        # Load u[batch, t, :] (complex) for dot product
        u_base = batch_idx * u_stride_batch + t * u_stride_time
        u_offset = u_base + feature_offsets * u_stride_feature
        u_r = tl.load(u_real_ptr + u_offset, mask=feature_mask, other=0.0)
        u_i = tl.load(u_imag_ptr + u_offset, mask=feature_mask, other=0.0)

        # Compute complex dot product: proj = B_complex @ u
        # proj = sum((b_r + j*b_i) * (u_r + j*u_i))
        # proj_r = sum(b_r*u_r - b_i*u_i)
        # proj_i = sum(b_r*u_i + b_i*u_r)
        proj_real = tl.sum(b_row_real * u_r - b_row_imag * u_i, axis=0)
        proj_imag = tl.sum(b_row_real * u_i + b_row_imag * u_r, axis=0)

        # Standard complex recurrence: x_new = C1 * x + proj
        # x_r_new = c1r*x_r - c1i*x_i + proj_r
        # x_i_new = c1i*x_r + c1r*x_i + proj_i
        x_r_new = c1r * x_r - c1i * x_i + proj_real
        x_i_new = c1i * x_r + c1r * x_i + proj_imag

        # Update state
        x_r = x_r_new
        x_i = x_i_new

        # Store output
        out_offset = (
            batch_idx * out_stride_batch + t * out_stride_time + filter_idx * out_stride_filter
        )
        tl.store(out_real_ptr + out_offset, x_r)
        tl.store(out_imag_ptr + out_offset, x_i)


def _sequential_scan_b_matrix_forward_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """Raw forward pass for complex input B-matrix scan - calls Triton kernel.

    This is an internal function used by the autograd wrapper.
    """
    batch_size, time_size, input_features = u.shape
    num_filters = C1.shape[0]

    # Ensure coefficients are complex
    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)

    # Ensure input is complex
    if not u.is_complex():
        u = u.to(torch.cfloat)

    # Precompute B_complex = C2 ⊙ B (element-wise scale rows by C2)
    B_complex = C2.unsqueeze(-1) * B.to(torch.cfloat)

    u = u.contiguous()
    C1 = C1.contiguous()
    B_complex = B_complex.contiguous()

    # Split into real and imaginary parts
    c1_real = C1.real.contiguous()
    c1_imag = C1.imag.contiguous()
    u_real = u.real.contiguous()
    u_imag = u.imag.contiguous()
    b_real = B_complex.real.contiguous()
    b_imag = B_complex.imag.contiguous()

    # Allocate output
    out_real = torch.zeros(batch_size, time_size, num_filters, device=u.device, dtype=u_real.dtype)
    out_imag = torch.zeros(batch_size, time_size, num_filters, device=u.device, dtype=u_real.dtype)

    # Block size for features (must be power of 2, large enough for input_features)
    BLOCK_FEATURES = triton.next_power_of_2(input_features)
    BLOCK_FEATURES = min(BLOCK_FEATURES, 1024)  # Triton limit

    if input_features > BLOCK_FEATURES:
        raise ValueError(
            f"input_features ({input_features}) exceeds maximum supported ({BLOCK_FEATURES}). "
            "Use PyTorch fallback for large input dimensions."
        )

    # Launch: one program per (batch, filter) pair
    grid = (batch_size * num_filters,)

    _sequential_scan_b_matrix_kernel[grid](
        u_real,
        u_imag,
        b_real,
        b_imag,
        out_real,
        out_imag,
        c1_real,
        c1_imag,
        batch_size,
        time_size,
        num_filters,
        input_features,
        u_real.stride(0),
        u_real.stride(1),
        u_real.stride(2),
        out_real.stride(0),
        out_real.stride(1),
        out_real.stride(2),
        b_real.stride(0),
        b_real.stride(1),
        BLOCK_FEATURES=BLOCK_FEATURES,
    )

    return torch.complex(out_real, out_imag)


class _SequentialScanBMatrixFunction(torch.autograd.Function):  # pragma: no cover
    """Autograd function for complex input selective scan with B matrix projection.

    Implements: x[n] = C1*x[n-1] + (C2 ⊙ B) @ u[n]

    Forward: Uses Triton kernel for fast GPU execution
    Backward: Uses adjoint method with chain rule through B matrix projection

    Gradients computed:
    - grad_C1: From adjoint states and output history
    - grad_C2: From grad_B_complex decomposition
    - grad_u: Chain rule through B_complex^H @ adjoint
    - grad_B: Chain rule through C2 scaling
    """

    @staticmethod
    def forward(
        ctx,
        C1: torch.Tensor,
        C2: torch.Tensor,
        u: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using Triton kernel.

        Note: Inputs should already be converted to correct dtypes by the caller.
        """
        # Compute B_complex for backward pass (detach to avoid double backward issues)
        with torch.no_grad():
            B_complex = C2.unsqueeze(-1) * B.to(C2.dtype)

        # Run forward pass with Triton
        output = _sequential_scan_b_matrix_forward_raw(C1, C2, u, B)

        # Save tensors needed for backward
        ctx.save_for_backward(C1, C2, u, B, B_complex, output)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass using adjoint method.

        The adjoint recurrence is: adj[t] = grad[t] + C1* * adj[t+1]
        Then gradients are computed via chain rule through the B projection.
        """
        C1, C2, u, B, B_complex, x = ctx.saved_tensors

        # Ensure grad_output is complex (preserve dtype)
        if not grad_output.is_complex():
            complex_dtype = (
                torch.complex128 if grad_output.dtype == torch.float64 else torch.complex64
            )
            grad_output = grad_output.to(complex_dtype)

        # Compute adjoint states using flip → scan → flip pattern
        C1_conj = C1.conj()
        num_filters = C1.shape[0]
        ones = torch.ones(num_filters, dtype=C1.dtype, device=C1.device)

        grad_flipped = grad_output.flip(1)
        adj_flipped = _sequential_scan_forward_raw(C1_conj, ones, grad_flipped)
        adj = adj_flipped.flip(1)

        # grad_C1 = Σ_{batch,t} adj[t+1] * x[t]*
        x_conj = x.conj()
        grad_C1 = (adj[:, 1:, :] * x_conj[:, :-1, :]).sum(dim=(0, 1))

        # grad_u = B_complex^H @ adj  (for each batch, time)
        # B_complex: (num_filters, input_features), adj: (batch, time, num_filters)
        # grad_u: (batch, time, input_features)
        grad_u = torch.einsum("fi,btf->bti", B_complex.conj(), adj)

        # grad_B_complex = Σ_{batch,t} adj[b,t,:,None] @ u[b,t,None,:]*
        # = adj^T @ u*  summed over batch
        # adj: (batch, time, num_filters), u: (batch, time, input_features)
        # grad_B_complex: (num_filters, input_features)
        u_conj = u.conj()
        grad_B_complex = torch.einsum("btf,bti->fi", adj, u_conj)

        # Decompose grad_B_complex into grad_C2 and grad_B
        # B_complex = C2[:,None] * B, so:
        # grad_C2 = Σ_i grad_B_complex[:,i] * B[:,i]
        grad_C2 = (grad_B_complex * B.to(adj.dtype)).sum(dim=-1)

        # grad_B = Re(grad_B_complex * C2[:,None]*)  (B is real)
        grad_B = (grad_B_complex * C2.conj().unsqueeze(-1)).real

        return grad_C1, grad_C2, grad_u, grad_B


def _triton_sequential_scan_b_matrix(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """Compute linear recurrence for complex inputs with B matrix and autograd support.

    Implements: x[n] = C1*x[n-1] + (C2 ⊙ B) @ u[n]

    This function uses Triton for the forward pass and computes gradients
    via the adjoint method, enabling fast training on GPU.

    Args:
        C1: Complex tensor of shape (num_filters,) - recurrence coefficient.
        C2: Complex tensor of shape (num_filters,) - input gain.
        u: Complex tensor of shape (batch, time, input_features) - input signal.
        B: Real tensor of shape (num_filters, input_features) - input projection.

    Returns:
        Complex tensor of shape (batch, time, num_filters) - output states.
    """
    # Convert to correct dtypes before calling autograd function
    # This ensures gradient flow works correctly
    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)
    if not u.is_complex():
        u = u.to(torch.cfloat)

    return _SequentialScanBMatrixFunction.apply(C1, C2, u, B)


# ==============================================================================
# Section 6: Parallel Associative Scan (O(log T), with Autograd)
#
# Uses Triton's tl.associative_scan (Blelloch algorithm) for O(log T) parallel
# depth. Best for T <= 16,384 when parallel execution is beneficial.
#
# Forward: parallel associative scan via tl.associative_scan
# Backward: reuses the fused backward kernel from Section 2 (sequential adjoint)
# Falls back to _triton_sequential_scan for very long sequences (T > 16,384).
# ==============================================================================


@triton.jit
def _associative_combine(  # pragma: no cover
    A_l_r,
    A_l_i,
    B_l_r,
    B_l_i,
    A_r_r,
    A_r_i,
    B_r_r,
    B_r_i,
):
    """Associative operator for linear recurrence: (A_l, B_l) * (A_r, B_r).

    Result: (A_r * A_l, A_r * B_l + B_r)
    """
    # A_new = A_right * A_left (complex multiply)
    A_new_r = A_r_r * A_l_r - A_r_i * A_l_i
    A_new_i = A_r_r * A_l_i + A_r_i * A_l_r

    # B_new = A_right * B_left + B_right
    AB_r = A_r_r * B_l_r - A_r_i * B_l_i
    AB_i = A_r_r * B_l_i + A_r_i * B_l_r
    B_new_r = AB_r + B_r_r
    B_new_i = AB_i + B_r_i

    return A_new_r, A_new_i, B_new_r, B_new_i


_PARALLEL_SCAN_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=2, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=8, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_PARALLEL_SCAN_AUTOTUNE_CONFIGS, key=["time_size"])
@triton.jit
def _parallel_scan_kernel(  # pragma: no cover
    # Input: u tensor and coefficient vectors
    u_real_ptr,
    u_imag_ptr,
    c1_real_ptr,
    c1_imag_ptr,  # Shape: (num_filters,)
    c2_real_ptr,
    c2_imag_ptr,  # Shape: (num_filters,)
    # Output
    out_real_ptr,
    out_imag_ptr,
    # Dimensions
    batch_size,
    time_size,
    filter_size,
    # Strides for u (and output, same layout)
    stride_batch,
    stride_time,
    stride_filter,
    BLOCK_T: tl.constexpr,
):
    """Associative scan kernel - O(log T) parallel using Blelloch algorithm.

    Fuses all operations into a single kernel:
    - Broadcast C1 to all timesteps (in registers)
    - Compute B = C2 * u (fused)
    - Parallel associative scan via tl.associative_scan
    """
    pid = tl.program_id(0)
    batch_idx = pid // filter_size
    filter_idx = pid % filter_size

    if batch_idx >= batch_size:
        return

    # Load C1 and C2 for this filter (scalars, loaded ONCE)
    c1_r = tl.load(c1_real_ptr + filter_idx)
    c1_i = tl.load(c1_imag_ptr + filter_idx)
    c2_r = tl.load(c2_real_ptr + filter_idx)
    c2_i = tl.load(c2_imag_ptr + filter_idx)

    # Time offsets
    t_offsets = tl.arange(0, BLOCK_T)
    mask = t_offsets < time_size

    # Calculate memory offsets for u
    offsets = batch_idx * stride_batch + t_offsets * stride_time + filter_idx * stride_filter

    # Load u values
    u_r = tl.load(u_real_ptr + offsets, mask=mask, other=0.0)
    u_i = tl.load(u_imag_ptr + offsets, mask=mask, other=0.0)

    # Compute B = C2 * u INSIDE the kernel (no separate allocation!)
    B_r = c2_r * u_r - c2_i * u_i
    B_i = c2_r * u_i + c2_i * u_r

    # A = C1 broadcast to all timesteps (no memory allocation, just registers)
    # Use tl.where to set identity (1+0j) for masked positions
    A_r = tl.where(mask, c1_r, 1.0)
    A_i = tl.where(mask, c1_i, 0.0)

    # Associative scan - the Blelloch algorithm in one call
    _, _, B_scan_r, B_scan_i = tl.associative_scan(
        (A_r, A_i, B_r, B_i),
        axis=0,
        combine_fn=_associative_combine,
    )

    # Store results
    tl.store(out_real_ptr + offsets, B_scan_r, mask=mask)
    tl.store(out_imag_ptr + offsets, B_scan_i, mask=mask)


def _parallel_scan_forward_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Raw forward pass for associative scan (no autograd).

    This is an internal function used by the autograd wrapper.
    It does NOT support autograd — use _triton_parallel_scan instead.

    Args:
        C1: Complex (num_filters,) - recurrence coefficient.
        C2: Complex (num_filters,) - input gain.
        u: Complex (batch, time, num_filters) - input signal.

    Returns:
        Complex (batch, time, num_filters) - output states.
    """
    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=True)

    out_real = torch.empty_like(inputs.u_real)
    out_imag = torch.empty_like(inputs.u_imag)

    BLOCK_T = triton.next_power_of_2(inputs.time_size)
    BLOCK_T = min(BLOCK_T, 16384)  # Triton limit

    grid = (inputs.batch_size * inputs.filter_size,)

    _parallel_scan_kernel[grid](
        inputs.u_real,
        inputs.u_imag,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        out_real,
        out_imag,
        inputs.batch_size,
        inputs.time_size,
        inputs.filter_size,
        inputs.u_real.stride(0),
        inputs.u_real.stride(1),
        inputs.u_real.stride(2),
        BLOCK_T=BLOCK_T,
    )

    return torch.complex(out_real, out_imag)


class _ParallelScanFunction(torch.autograd.Function):  # pragma: no cover
    """Autograd function for parallel associative scan with Triton acceleration.

    Forward: Uses Blelloch parallel scan O(log T) via tl.associative_scan.
    Backward: Reuses fused backward kernel from selective scan (single-pass
    sequential adjoint, reads saved output to skip forward recompute).

    The backward pass is mathematically identical regardless of how the forward
    was computed — it only depends on C1, C2, u, output, and grad_output.
    """

    @staticmethod
    def forward(
        ctx,
        C1: torch.Tensor,
        C2: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using parallel Blelloch scan."""
        if not C1.is_complex():
            C1 = C1.to(torch.cfloat)
        if not C2.is_complex():
            C2 = C2.to(torch.cfloat)
        if not u.is_complex():
            u = u.to(torch.cfloat)

        output = _parallel_scan_forward_raw(C1, C2, u)
        ctx.save_for_backward(C1, C2, u, output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass reusing fused selective scan backward kernel.

        Parallel forward, serial adjoint — same raw backward as
        ``_SequentialScanFunction``; ``ctx.needs_input_grad`` gates the
        kernel's constexpr branches to avoid unneeded atomic_adds / loads.
        """
        C1, C2, u, output = ctx.saved_tensors
        need_C1, need_C2, need_u = ctx.needs_input_grad
        grad_C1, grad_C2, grad_u = _sequential_scan_backward_raw(
            C1,
            C2,
            u,
            output,
            grad_output,
            compute_grad_C1=need_C1,
            compute_grad_C2=need_C2,
            compute_grad_u=need_u,
        )
        return grad_C1, grad_C2, grad_u


def _triton_parallel_scan(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Compute linear recurrence using associative scan (O(log T) parallel).

    Uses tl.associative_scan (Blelloch algorithm) for parallel execution.
    All operations are fused into a single kernel launch. Supports autograd
    via the fused backward kernel from the selective scan.

    Args:
        C1: Complex tensor of shape (num_filters,) - recurrence coefficient.
        C2: Complex tensor of shape (num_filters,) - input gain.
        u: Complex tensor of shape (batch, time, num_filters) - input signal.

    Returns:
        Complex tensor of shape (batch, time, num_filters) - output states.
    """
    # Handle filter broadcasting if needed (before standard preparation)
    _, _, in_filters = u.shape
    num_filters = C1.shape[0]
    if in_filters == 1 and num_filters > 1:
        u = u.expand(-1, -1, num_filters)

    # Check if sequence fits in a single Triton block
    time_size = u.shape[1]
    BLOCK_T = triton.next_power_of_2(time_size)
    if BLOCK_T > 16384:
        # Fall back to selective scan for very long sequences
        return _triton_sequential_scan(C1, C2, u)

    return _ParallelScanFunction.apply(C1, C2, u)


@triton.autotune(configs=_PARALLEL_SCAN_AUTOTUNE_CONFIGS, key=["time_size"])
@triton.jit
def _parallel_scan_real_kernel(  # pragma: no cover
    u_real_ptr,
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    batch_size,
    time_size,
    filter_size,
    stride_batch,
    stride_time,
    stride_filter,
    BLOCK_T: tl.constexpr,
):
    """Parallel associative scan — real-input specialization of the same
    standard complex recurrence used by ``_parallel_scan_kernel``.

    Same equation ``x[t] = C1*x[t-1] + C2*u[t]`` with C1, C2, x complex.
    Since ``Im(u) == 0``, the per-step drive ``B = C2 * u`` reduces to
    ``(c2r*u_r, c2i*u_r)`` and is constructed directly in the kernel; the
    associative combine and final state remain fully complex. Numerically
    identical to the complex-input path called with ``u + 0j``. NOT the
    deprecated real-input equation with an extra imaginary term.
    """
    pid = tl.program_id(0)
    batch_idx = pid // filter_size
    filter_idx = pid % filter_size

    if batch_idx >= batch_size:
        return

    c1_r = tl.load(c1_real_ptr + filter_idx)
    c1_i = tl.load(c1_imag_ptr + filter_idx)
    c2_r = tl.load(c2_real_ptr + filter_idx)
    c2_i = tl.load(c2_imag_ptr + filter_idx)

    t_offsets = tl.arange(0, BLOCK_T)
    mask = t_offsets < time_size
    offsets = batch_idx * stride_batch + t_offsets * stride_time + filter_idx * stride_filter

    u_r = tl.load(u_real_ptr + offsets, mask=mask, other=0.0)

    B_r = c2_r * u_r
    B_i = c2_i * u_r

    A_r = tl.where(mask, c1_r, 1.0)
    A_i = tl.where(mask, c1_i, 0.0)

    _, _, B_scan_r, B_scan_i = tl.associative_scan(
        (A_r, A_i, B_r, B_i),
        axis=0,
        combine_fn=_associative_combine,
    )

    tl.store(out_real_ptr + offsets, B_scan_r, mask=mask)
    tl.store(out_imag_ptr + offsets, B_scan_i, mask=mask)


def _parallel_scan_real_forward_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Raw forward pass for the real-input specialization of the standard
    complex parallel scan. Not a separate equation — see module docstring."""
    if u.is_complex():
        raise ValueError("_parallel_scan_real_forward_raw expects real input")
    if u.dtype != torch.float32:
        u = u.to(torch.float32)
    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=False)

    out_real = torch.empty_like(inputs.u_real)
    out_imag = torch.empty_like(inputs.u_real)
    BLOCK_T = triton.next_power_of_2(inputs.time_size)

    _parallel_scan_real_kernel[(inputs.batch_size * inputs.filter_size,)](
        inputs.u_real,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        out_real,
        out_imag,
        inputs.batch_size,
        inputs.time_size,
        inputs.filter_size,
        inputs.u_real.stride(0),
        inputs.u_real.stride(1),
        inputs.u_real.stride(2),
        BLOCK_T=BLOCK_T,
    )

    return torch.complex(out_real, out_imag)


class _ParallelScanRealFunction(torch.autograd.Function):  # pragma: no cover
    """Autograd wrapper for the real-input specialization of the standard
    complex parallel scan. Same equation as the complex path with Im(u)=0."""

    @staticmethod
    def forward(
        ctx,
        C1: torch.Tensor,
        C2: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        if u.is_complex():
            raise ValueError("_ParallelScanRealFunction expects real input")
        if u.dtype != torch.float32:
            u = u.to(torch.float32)
        output = _parallel_scan_real_forward_raw(C1, C2, u)
        ctx.save_for_backward(C1, C2, u, output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward via fused real-input kernel. ``ctx.needs_input_grad`` gates
        kernel constexpr branches so skipped grads cost zero atomic_add /
        per-t loads (e.g. frozen C1/C2 fine-tuning)."""
        C1, C2, u, output = ctx.saved_tensors
        need_C1, need_C2, need_u = ctx.needs_input_grad
        grad_C1, grad_C2, grad_u = _sequential_scan_backward_real_raw(
            C1,
            C2,
            u,
            output,
            grad_output,
            compute_grad_C1=need_C1,
            compute_grad_C2=need_C2,
            compute_grad_u=need_u,
        )
        return grad_C1, grad_C2, grad_u


def _triton_parallel_scan_real(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Parallel version of the standard complex recurrence x[t]=C1*x[t-1]+C2*u[t]
    with a real-valued input ``u`` (treated as ``u + 0j``). Specialization of
    ``_triton_parallel_scan``: same math, complex output, just skips the
    redundant all-zero imaginary-input traffic. NOT the deprecated real-input
    equation with an extra imaginary term — see module docstring.
    """
    if u.is_complex():
        raise ValueError("_triton_parallel_scan_real expects real input")

    _, _, in_filters = u.shape
    num_filters = C1.shape[0]
    if in_filters == 1 and num_filters > 1:
        u = u.expand(-1, -1, num_filters)

    time_size = u.shape[1]
    BLOCK_T = triton.next_power_of_2(time_size)
    if BLOCK_T > 16384:
        return _triton_sequential_scan_real(C1, C2, u)

    return _ParallelScanRealFunction.apply(C1, C2, u)


# ==============================================================================
# Section 7: Parallel Associative Scan for Multi-Oscillator (O(log T), Fused)
#
# Combines the parallel Blelloch scan (Section 6) with multi-oscillator fusion
# (Section 4). Uses 2D tl.associative_scan with shape (BLOCK_T, BLOCK_N) to
# process N oscillators per physical filter in parallel, with fused C-weight
# reduction after the scan — no intermediate (B,T,F*N) allocation.
#
# Forward: Parallel associative scan via tl.associative_scan on 2D blocks
# Backward: Reuses fused backward kernel from Section 4 (sequential adjoint)
# Falls back to _triton_sequential_scan_multi_osc for T > 16,384.
# ==============================================================================


@triton.jit
def _parallel_scan_multi_osc_kernel(  # pragma: no cover
    # Input: u tensor (batch, time, num_filters) and coefficient vectors
    u_real_ptr,
    u_imag_ptr,
    # Per-oscillator coefficients (shape: scan_filters = num_filters * num_osc)
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    # C-weight pointers (shape: scan_filters, separate real/imag)
    cw_real_ptr,
    cw_imag_ptr,
    # Output (reduced: batch, time, num_filters)
    out_real_ptr,
    out_imag_ptr,
    # Dimensions
    batch_size,
    time_size,
    num_filters,
    num_osc,
    # Strides for (B,T,F) tensors
    stride_batch,
    stride_time,
    stride_filter,
    # Output strides
    out_stride_batch,
    out_stride_time,
    out_stride_filter,
    # Block sizes (compile-time constants)
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused parallel multi-oscillator scan with C-weight reduction.

    Each program handles one (batch, physical_filter) pair.
    Uses 2D tl.associative_scan with shape (BLOCK_T, BLOCK_N) where:
    - axis=0 (time): Blelloch parallel scan
    - axis=1 (oscillators): independent lanes processed in parallel

    Steps:
    1. Load N oscillator coefficients (c1, c2, cw) — shape (BLOCK_N,)
    2. Load input u for all timesteps — shape (BLOCK_T,), broadcast to BLOCK_N
    3. Compute B = C2 * u — shape (BLOCK_T, BLOCK_N)
    4. Broadcast A = C1 — shape (BLOCK_T, BLOCK_N)
    5. tl.associative_scan along axis=0
    6. C-weight reduce: out[t] = Σ_n cw_n * x_n[t] — shape (BLOCK_T,)
    7. Store reduced output
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_filters
    filter_idx = pid % num_filters

    if batch_idx >= batch_size:
        return

    # --- Load per-oscillator coefficients (BLOCK_N-wide) ---
    osc_start = filter_idx * num_osc
    osc_range = tl.arange(0, BLOCK_N)
    osc_mask = osc_range < num_osc
    coeff_idx = osc_start + osc_range

    c1_r = tl.load(c1_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c1_i = tl.load(c1_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2_r = tl.load(c2_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2_i = tl.load(c2_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)
    cw_r = tl.load(cw_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    cw_i = tl.load(cw_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)

    # --- Load input u for all timesteps (BLOCK_T,) ---
    t_offsets = tl.arange(0, BLOCK_T)
    t_mask = t_offsets < time_size
    u_offsets = batch_idx * stride_batch + t_offsets * stride_time + filter_idx * stride_filter
    u_r = tl.load(u_real_ptr + u_offsets, mask=t_mask, other=0.0)  # (BLOCK_T,)
    u_i = tl.load(u_imag_ptr + u_offsets, mask=t_mask, other=0.0)  # (BLOCK_T,)

    # --- Expand to 2D: (BLOCK_T, BLOCK_N) ---
    # Reshape u to column vectors and coefficients to row vectors for broadcast
    u_r_2d = u_r[:, None]  # (BLOCK_T, 1) -> broadcasts to (BLOCK_T, BLOCK_N)
    u_i_2d = u_i[:, None]

    c2_r_2d = c2_r[None, :]  # (1, BLOCK_N) -> broadcasts to (BLOCK_T, BLOCK_N)
    c2_i_2d = c2_i[None, :]

    # B = C2 * u (complex multiply, 2D)
    B_r = c2_r_2d * u_r_2d - c2_i_2d * u_i_2d  # (BLOCK_T, BLOCK_N)
    B_i = c2_r_2d * u_i_2d + c2_i_2d * u_r_2d

    # A = C1 broadcast to all timesteps (2D)
    # Identity (1+0j) for masked (padding) positions
    c1_r_2d = c1_r[None, :]
    c1_i_2d = c1_i[None, :]
    A_r = tl.where(t_mask[:, None], c1_r_2d, 1.0)  # (BLOCK_T, BLOCK_N)
    A_i = tl.where(t_mask[:, None], c1_i_2d, 0.0)

    # --- Parallel associative scan along axis=0 (time) ---
    # The combine_fn operates element-wise, so it works on any shape
    _, _, X_r, X_i = tl.associative_scan(
        (A_r, A_i, B_r, B_i),
        axis=0,
        combine_fn=_associative_combine,
    )

    # --- C-weight reduction: out[t] = Σ_n cw_n * x_n[t] ---
    cw_r_2d = cw_r[None, :]  # (1, BLOCK_N)
    cw_i_2d = cw_i[None, :]
    # Complex multiply cw * x, then sum over oscillator axis
    out_r = tl.sum(cw_r_2d * X_r - cw_i_2d * X_i, axis=1)  # (BLOCK_T,)
    out_i = tl.sum(cw_r_2d * X_i + cw_i_2d * X_r, axis=1)

    # --- Store reduced output ---
    out_offsets = (
        batch_idx * out_stride_batch + t_offsets * out_stride_time + filter_idx * out_stride_filter
    )
    tl.store(out_real_ptr + out_offsets, out_r, mask=t_mask)
    tl.store(out_imag_ptr + out_offsets, out_i, mask=t_mask)


def _parallel_scan_multi_osc_forward_raw(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    cw_real: torch.Tensor,
    cw_imag: torch.Tensor,
    num_osc: int,
) -> torch.Tensor:
    """Raw forward pass for parallel multi-oscillator scan (no autograd).

    Args:
        C1: Complex (scan_filters,) - per-oscillator recurrence coefficient.
        C2: Complex (scan_filters,) - per-oscillator input gain.
        u: Complex (batch, time, num_filters) - input signal.
        cw_real: Real (scan_filters,) - C-weight real parts.
        cw_imag: Real (scan_filters,) - C-weight imag parts.
        num_osc: Uniform number of oscillators per filter.

    Returns:
        Complex (batch, time, num_filters) - reduced output.
    """
    if not u.is_complex():
        u = u.to(torch.cfloat)
    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)

    batch_size, time_size, num_filters = u.shape
    u = u.contiguous()

    target_dtype = u.real.dtype
    c1_real = C1.real.to(target_dtype).contiguous()
    c1_imag = C1.imag.to(target_dtype).contiguous()
    c2_real = C2.real.to(target_dtype).contiguous()
    c2_imag = C2.imag.to(target_dtype).contiguous()
    u_real = u.real.contiguous()
    u_imag = u.imag.contiguous()
    cw_real = cw_real.to(target_dtype).contiguous()
    cw_imag = cw_imag.to(target_dtype).contiguous()

    out_real = torch.zeros(batch_size, time_size, num_filters, device=u.device, dtype=target_dtype)
    out_imag = torch.zeros_like(out_real)

    BLOCK_T = min(triton.next_power_of_2(time_size), 16384)
    BLOCK_N = triton.next_power_of_2(num_osc)

    grid = (batch_size * num_filters,)

    _parallel_scan_multi_osc_kernel[grid](
        u_real,
        u_imag,
        c1_real,
        c1_imag,
        c2_real,
        c2_imag,
        cw_real,
        cw_imag,
        out_real,
        out_imag,
        batch_size,
        time_size,
        num_filters,
        num_osc,
        u_real.stride(0),
        u_real.stride(1),
        u_real.stride(2),
        out_real.stride(0),
        out_real.stride(1),
        out_real.stride(2),
        BLOCK_T=BLOCK_T,
        BLOCK_N=BLOCK_N,
    )

    return torch.complex(out_real, out_imag)


class _ParallelScanMultiOscFunction(torch.autograd.Function):  # pragma: no cover
    """Autograd function for parallel multi-oscillator scan with C-weight reduction.

    Forward: Uses Blelloch parallel scan O(log T) via 2D tl.associative_scan
    on (BLOCK_T, BLOCK_N) blocks, with fused C-weight reduction.

    Backward: Reuses fused backward kernel from Section 4 (sequential adjoint).
    The backward pass is mathematically identical regardless of how the forward
    was computed — it only depends on C1, C2, u, c_weights, and grad_output.
    """

    @staticmethod
    def forward(
        ctx,
        C1: torch.Tensor,
        C2: torch.Tensor,
        u: torch.Tensor,
        c_weights: torch.Tensor,
        num_osc: int,
    ) -> torch.Tensor:
        """Forward pass using parallel Blelloch scan on 2D blocks."""
        if not C1.is_complex():
            C1 = C1.to(torch.cfloat)
        if not C2.is_complex():
            C2 = C2.to(torch.cfloat)
        if not u.is_complex():
            u = u.to(torch.cfloat)

        cw_real = c_weights[:, 0].contiguous()
        cw_imag = c_weights[:, 1].contiguous()

        output = _parallel_scan_multi_osc_forward_raw(C1, C2, u, cw_real, cw_imag, num_osc)

        ctx.save_for_backward(C1, C2, u, c_weights)
        ctx.num_osc = num_osc
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass reusing fused multi-osc backward kernel from Section 4.
        ``ctx.needs_input_grad`` is forwarded for gated atomics / FLOPs /
        allocations in frozen-recurrent fine-tuning."""
        C1, C2, u, c_weights = ctx.saved_tensors
        num_osc = ctx.num_osc
        need_C1, need_C2, need_u, need_cw, _ = ctx.needs_input_grad

        grad_C1, grad_C2, grad_u, grad_c_weights = _sequential_scan_multi_osc_backward_raw(
            C1,
            C2,
            u,
            c_weights,
            grad_output,
            num_osc,
            compute_grad_C1=need_C1,
            compute_grad_C2=need_C2,
            compute_grad_u=need_u,
            compute_grad_cw=need_cw,
        )

        return grad_C1, grad_C2, grad_u, grad_c_weights, None  # None for num_osc


def _triton_parallel_scan_multi_osc(  # pragma: no cover
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    c_weights: torch.Tensor,
    num_osc: int,
) -> torch.Tensor:
    """Fused parallel multi-oscillator scan with C-weight reduction and autograd.

    Combines O(log T) parallel Blelloch scan with multi-oscillator fusion:
    1. N independent complex scans sharing the same input u per physical filter
    2. C-weight combination: y[t,f] = Σ_n c_n * x_n[t]

    All operations are fused into a single kernel (no intermediate allocations).
    Falls back to sequential multi-osc scan for T > 16,384.

    Args:
        C1: Complex (scan_filters,) - per-oscillator recurrence coefficient.
        C2: Complex (scan_filters,) - per-oscillator input gain.
        u: Complex (batch, time, num_filters) - input signal.
        c_weights: Real (scan_filters, 2) - C-weight [real, imag] per oscillator.
        num_osc: int - uniform number of oscillators per filter.

    Returns:
        Complex (batch, time, num_filters) - C-weight-reduced output.
    """
    time_size = u.shape[1]
    BLOCK_T = triton.next_power_of_2(time_size)
    if BLOCK_T > 16384:
        return _triton_sequential_scan_multi_osc(C1, C2, u, c_weights, num_osc)

    return _ParallelScanMultiOscFunction.apply(C1, C2, u, c_weights, num_osc)
