"""Parallel scan algorithms for Recurrent Quadrature Filters.

This module provides O(log T) parallel scan implementations of RQF using
the Blelloch algorithm, as well as O(T) fused linear recurrence fallbacks.

All scans implement THE SAME standard complex recurrence:
    x[n+1] = C1 * x[n] + C2 * u[n]
with C1, C2 and x complex. A real-valued ``u`` is treated as ``u + 0j``:
coefficients and state stay complex, and on CUDA a specialized Triton fast
path runs the same equation without materializing the all-zero imaginary
input channel. This is NOT the deprecated real-input RQF equation with an
extra imaginary term / modified Im(x) feedback; that variant complicated the
local-learning derivation and produced non-symmetric filters.

Key features:
    1. Blelloch parallel scan: O(T) work, O(log T) depth
    2. Automatic GPU acceleration via Triton kernels when available
    3. Memory efficient on GPU (no coefficient expansion in Triton kernels)

Mathematical background:
    RQF recurrence: x[n+1] = C1 * x[n] + C2 * u[n]

    This is a linear recurrence with associative operator::

        (A_i, b_i) * (A_j, b_j) = (A_j * A_i, A_j * b_i + b_j)

    Blelloch scan computes all prefix products in O(log T) parallel steps.

References:
    - Blelloch (1990): "Prefix Sums and Their Applications"
"""

import math
import os

import torch

# Set ORGANICS_DISABLE_TRITON=1 to force PyTorch fallback (useful for debugging)
_TRITON_DISABLED = os.environ.get("ORGANICS_DISABLE_TRITON", "0") == "1"

# Maximum input features supported by Triton B-matrix kernels.
# Limited by Triton's tl.load block size for loading B matrix rows into SRAM.
# For larger input_features, falls back to PyTorch sequential scan.
_TRITON_MAX_INPUT_FEATURES = 1024

try:
    from organics.ops.triton_scan_kernel import (
        _sequential_scan_forward_abs_raw_block_f,
        _sequential_scan_real_forward_abs_raw_block_f,
        _triton_parallel_scan,
        _triton_parallel_scan_real,
        _triton_sequential_scan,
        _triton_sequential_scan_b_matrix,
        _triton_sequential_scan_real,
    )

    _HAS_TRITON = True and not _TRITON_DISABLED
except ImportError:
    _HAS_TRITON = False


def _pad_to_power_of_two(
    tensors: list[torch.Tensor],
    pad_values: list[float],
) -> tuple[list[torch.Tensor], int, int]:
    """Pad tensors along time dimension to power of 2 for Blelloch scan.

    Args:
        tensors: List of tensors with shape (batch, time, filters) to pad.
        pad_values: Padding value for each tensor (e.g., 1.0 for identity, 0.0 for zero).

    Returns:
        Tuple of (padded_tensors, original_time, log_time) where:
            - padded_tensors: List of padded tensors
            - original_time: Original time dimension (for later trimming)
            - log_time: log2(padded_time) for loop bounds
    """
    batch, time, filters = tensors[0].shape

    log_t = int(math.ceil(math.log2(time))) if time > 0 else 0
    padded_time = 2**log_t if log_t > 0 else 0

    if padded_time <= time:
        return tensors, time, log_t

    pad_size = padded_time - time
    padded = [
        torch.nn.functional.pad(t, (0, 0, 0, pad_size), value=v)
        for t, v in zip(tensors, pad_values, strict=True)
    ]
    return padded, time, log_t


def _complex_mul(
    a_real: torch.Tensor,
    a_imag: torch.Tensor,
    b_real: torch.Tensor,
    b_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex multiplication (a_real + i*a_imag) * (b_real + i*b_imag).

    Args:
        a_real: Real part of first operand.
        a_imag: Imaginary part of first operand.
        b_real: Real part of second operand.
        b_imag: Imaginary part of second operand.

    Returns:
        Tuple of (real_part, imag_part) of the product.
    """
    return (
        a_real * b_real - a_imag * b_imag,
        a_real * b_imag + a_imag * b_real,
    )


def _blelloch_scan(
    A_real: torch.Tensor,
    A_imag: torch.Tensor,
    B_real: torch.Tensor,
    B_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blelloch parallel prefix scan for complex linear recurrence.

    Computes the inclusive scan of elements (A, B) with the associative operator::

        (A_i, B_i) * (A_j, B_j) = (A_j * A_i, A_j * B_i + B_j)

    This computes all prefix products in O(log T) parallel depth with O(T) work.

    Args:
        A_real: Real part of A, shape (batch, time, filters).
        A_imag: Imaginary part of A, shape (batch, time, filters).
        B_real: Real part of B, shape (batch, time, filters).
        B_imag: Imaginary part of B, shape (batch, time, filters).

    Returns:
        Tuple of (A_out_real, A_out_imag, B_out_real, B_out_imag) with scanned results.

    Note:
        This is a PyTorch implementation. For maximum GPU efficiency on CUDA devices,
        the Triton kernels in ``triton_scan_kernel.py`` are preferred and will be
        used automatically when available.
    """
    batch, time, filters = A_real.shape
    device = A_real.device

    if time <= 1:
        return A_real, A_imag, B_real, B_imag

    # Clone inputs and pad to power of 2 for Blelloch algorithm
    # A is padded with identity (1+0j), B is padded with zero
    [A_r, A_i, B_r, B_i], orig_time, log_t = _pad_to_power_of_two(
        [A_real.clone(), A_imag.clone(), B_real.clone(), B_imag.clone()],
        [1.0, 0.0, 0.0, 0.0],
    )
    padded_time = 2**log_t

    # Up-sweep (reduce) phase
    for d in range(log_t):
        stride = 2 ** (d + 1)
        half_stride = 2**d

        # Indices for left and right elements
        right_indices = torch.arange(stride - 1, padded_time, stride, device=device)
        left_indices = right_indices - half_stride

        if len(right_indices) == 0:
            continue

        # Load left and right values
        A_left_r = A_r[:, left_indices, :]
        A_left_i = A_i[:, left_indices, :]
        B_left_r = B_r[:, left_indices, :]
        B_left_i = B_i[:, left_indices, :]

        A_right_r = A_r[:, right_indices, :]
        A_right_i = A_i[:, right_indices, :]
        B_right_r = B_r[:, right_indices, :]
        B_right_i = B_i[:, right_indices, :]

        # Apply operator: (A_right, B_right) * (A_left, B_left)
        # A_new = A_right * A_left
        A_new_r, A_new_i = _complex_mul(A_right_r, A_right_i, A_left_r, A_left_i)

        # B_new = A_right * B_left + B_right
        AB_r, AB_i = _complex_mul(A_right_r, A_right_i, B_left_r, B_left_i)
        B_new_r = AB_r + B_right_r
        B_new_i = AB_i + B_right_i

        # Store back
        A_r[:, right_indices, :] = A_new_r
        A_i[:, right_indices, :] = A_new_i
        B_r[:, right_indices, :] = B_new_r
        B_i[:, right_indices, :] = B_new_i

    # Down-sweep phase
    for d in range(log_t - 2, -1, -1):
        stride = 2 ** (d + 1)
        half_stride = 2**d

        # We need to propagate from position (stride-1) to position (stride-1 + half_stride)
        left_indices = torch.arange(stride - 1, padded_time - half_stride, stride, device=device)
        right_indices = left_indices + half_stride

        if len(left_indices) == 0:
            continue

        # Load values
        A_left_r = A_r[:, left_indices, :]
        A_left_i = A_i[:, left_indices, :]
        B_left_r = B_r[:, left_indices, :]
        B_left_i = B_i[:, left_indices, :]

        A_right_r = A_r[:, right_indices, :]
        A_right_i = A_i[:, right_indices, :]
        B_right_r = B_r[:, right_indices, :]
        B_right_i = B_i[:, right_indices, :]

        # Apply operator: (A_right, B_right) * (A_left, B_left)
        A_new_r, A_new_i = _complex_mul(A_right_r, A_right_i, A_left_r, A_left_i)
        AB_r, AB_i = _complex_mul(A_right_r, A_right_i, B_left_r, B_left_i)
        B_new_r = AB_r + B_right_r
        B_new_i = AB_i + B_right_i

        # Store back to right positions
        A_r[:, right_indices, :] = A_new_r
        A_i[:, right_indices, :] = A_new_i
        B_r[:, right_indices, :] = B_new_r
        B_i[:, right_indices, :] = B_new_i

    # Remove padding
    A_r = A_r[:, :time, :]
    A_i = A_i[:, :time, :]
    B_r = B_r[:, :time, :]
    B_i = B_i[:, :time, :]

    return A_r, A_i, B_r, B_i


def _sequential_scan_cpu(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    B_complex: torch.Tensor | None,
) -> torch.Tensor:
    """PyTorch sequential scan fallback for CPU or when Triton is unavailable.

    Args:
        C1: Recurrence coefficient, shape (num_filters,) complex.
        C2: Input gain, shape (num_filters,) complex.
        u: Input signal, shape (batch, time, features) complex.
        B_complex: Optional pre-computed (C2 ⊙ B) matrix, shape (num_filters, input_features)
            complex. C2 element-wise scales each row of B.

    Returns:
        Output states, shape (batch, time, num_filters) complex.
    """
    batch, time, _ = u.shape
    num_filters = C1.shape[0]
    device = u.device
    use_matrix = B_complex is not None

    x_prev = torch.zeros(batch, num_filters, dtype=torch.cfloat, device=device)
    output = torch.empty(batch, time, num_filters, dtype=torch.cfloat, device=device)

    for t in range(time):
        input_proj = torch.matmul(u[:, t, :], B_complex.T) if use_matrix else C2 * u[:, t, :]

        # Standard complex recurrence
        x_new = C1 * x_prev + input_proj

        output[:, t, :] = x_new
        x_prev = x_new

    return output


def sequential_scan(  # noqa: CCR001
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    B: torch.Tensor | None = None,
) -> torch.Tensor:
    """Memory-efficient linear recurrence using fused kernel.

    Uses Triton kernel on GPU (sequential but fused) or PyTorch loop on CPU.
    This is O(1) extra memory vs O(B*T*F) for Blelloch scan.

    Args:
        C1: Recurrence coefficient, shape (num_filters,) complex.
        C2: Input gain, shape (num_filters,) complex - from RQF discretization.
        u: Input signal, shape (batch, time, input_features).
        B: Optional input projection matrix, shape (num_filters, input_features) real.
           If None, uses scalar C2 (backward compatible). If provided, the effective
           projection is (C2 ⊙ B) where C2 scales each row of B element-wise.

    Returns:
        Output states, shape (batch, time, num_filters) complex.

    Note:
        The recurrence with B matrix is: x[n+1] = C1*x[n] + (C2 ⊙ B)@u[n]
        where C2 (complex) element-wise scales the rows of B (real).

        All inputs (real or complex) use the standard complex recurrence:
            x[n+1] = C1*x[n] + C2*u[n]  [without B]
            x[n+1] = C1*x[n] + (C2 ⊙ B)@u[n]  [with B]

        On CUDA, real inputs may use a specialized Triton fast path that
        avoids materializing the all-zero imaginary channel. Semantics are
        unchanged: same standard complex recurrence, same complex output.
        This is NOT the deprecated real-input RQF equation — see module
        docstring.
    """
    batch, time, in_filters = u.shape
    num_filters = C1.shape[0]
    input_is_complex = u.is_complex()
    real_u = u if not input_is_complex else None

    # Ensure coefficients and input are complex before any computation
    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)

    # Handle B matrix (optional for backward compatibility)
    use_matrix = B is not None
    if use_matrix:
        if not input_is_complex:
            u = real_u.to(torch.cfloat)
        # Compute B_complex = (C2 ⊙ B): C2 element-wise scales each row of B
        # Shape: (num_filters, input_features) complex
        B_complex = C2.unsqueeze(-1) * B.to(torch.cfloat)
    else:
        # Backward compatible: expand input to num_filters if needed
        if input_is_complex:
            if in_filters == 1 and num_filters > 1:
                u = u.expand(-1, -1, num_filters).contiguous()
        else:
            if in_filters == 1 and num_filters > 1:
                real_u = real_u.expand(-1, -1, num_filters).contiguous()

    # Triton kernel dispatch rules:
    # - CUDA + Triton available → use Triton kernels (fast, supports autograd)
    # - B matrix with input_features > _TRITON_MAX_INPUT_FEATURES → PyTorch fallback
    # - CPU → PyTorch sequential fallback
    scan_device = u.device if input_is_complex or use_matrix else real_u.device
    if scan_device.type == "cuda" and _HAS_TRITON:
        if use_matrix and in_filters <= _TRITON_MAX_INPUT_FEATURES:
            return _triton_sequential_scan_b_matrix(C1, C2, u, B)
        elif not use_matrix and not input_is_complex:
            return _triton_sequential_scan_real(C1, C2, real_u)
        elif not use_matrix:
            return _triton_sequential_scan(C1, C2, u)
        # Fall through to PyTorch for large input_features

    # PyTorch sequential fallback (used for: CPU, large input_features)
    if not input_is_complex and not use_matrix:
        u = real_u.to(torch.cfloat)
    B_complex_arg = B_complex if use_matrix else None
    return _sequential_scan_cpu(C1, C2, u, B_complex_arg)


def sequential_scan_abs(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    B: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused ``|sequential_scan(...)|`` for inference / no-grad consumers.

    Equivalent to ``sequential_scan(C1, C2, u, B).abs()`` but, when autograd is
    disabled and inputs are on CUDA with ``B is None``, dispatches to fused
    Triton kernels that fold ``sqrt(Re**2 + Im**2)`` into the scan's output
    store. This halves output HBM traffic and skips the downstream ``.abs()``
    pass. On the A10G this is ~5-11% faster than the two-step path for both
    real and complex input, dominated by saved output bandwidth.

    Falls back to ``sequential_scan(...).abs()`` in any case that isn't
    supported by the fused kernel: CPU, Triton disabled, ``B`` matrix present,
    or when any of ``C1/C2/u/B`` requires grad.

    Args:
        C1: Recurrence coefficient, shape (num_filters,) complex.
        C2: Input gain, shape (num_filters,) complex.
        u: Input signal, shape (batch, time, input_features).
        B: Optional input projection matrix (not supported by the fast path —
           forces fallback).

    Returns:
        Magnitude ``|x[t]|``, shape (batch, time, num_filters), real dtype.
    """
    needs_grad = torch.is_grad_enabled() and (
        u.requires_grad
        or C1.requires_grad
        or C2.requires_grad
        or (B is not None and B.requires_grad)
    )
    fast_path = u.device.type == "cuda" and _HAS_TRITON and B is None and not needs_grad
    if not fast_path:
        return sequential_scan(C1, C2, u, B=B).abs()

    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)

    _, _, in_filters = u.shape
    num_filters = C1.shape[0]
    if in_filters == 1 and num_filters > 1:
        u = u.expand(-1, -1, num_filters).contiguous()

    if u.is_complex():
        return _sequential_scan_forward_abs_raw_block_f(C1, C2, u)
    return _sequential_scan_real_forward_abs_raw_block_f(C1, C2, u)


def parallel_scan(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    B: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute linear recurrence x[n+1] = C1*x[n] + C2*u[n] via Blelloch parallel scan.

    This is the core parallelization: instead of iterating t=0..T-1 sequentially,
    we use the associative property of linear recurrences to compute all x[n]
    in O(log T) parallel steps.

    Args:
        C1: Recurrence coefficient, shape (num_filters,) complex.
        C2: Input gain, shape (num_filters,) complex - from RQF discretization.
        u: Input signal, shape (batch, time, input_features).
        B: Optional input projection matrix, shape (num_filters, input_features) real.
           If None, uses scalar C2 (backward compatible). If provided, the effective
           projection is (C2 ⊙ B) where C2 scales each row of B element-wise.

    Returns:
        Output states, shape (batch, time, num_filters) complex.

    Note:
        The recurrence with B matrix is: x[n+1] = C1*x[n] + (C2 ⊙ B)@u[n]
        where (C2 ⊙ B) denotes C2 (complex) element-wise scaling each row of
        B (real).

        On CUDA, real inputs may use a specialized Triton fast path that
        avoids materializing the all-zero imaginary channel. Semantics are
        unchanged: same standard complex recurrence, same complex output.
        This is NOT the deprecated real-input RQF equation — see module
        docstring.
    """
    batch, time, in_filters = u.shape
    num_filters = C1.shape[0]
    input_is_complex = u.is_complex()
    real_u = u if not input_is_complex else None

    # Ensure coefficients and input are complex before any computation
    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)

    # Handle B matrix - project input through B if provided
    use_matrix = B is not None
    if use_matrix:
        if not input_is_complex:
            u = real_u.to(torch.cfloat)
        # Compute B_complex = (C2 ⊙ B): C2 element-wise scales each row of B
        B_complex = C2.unsqueeze(-1) * B.to(torch.cfloat)
        # Matrix multiply: (num_filters, input_features) @ (batch, time, input_features)
        # Use einsum for batched matrix multiply
        u_projected = torch.einsum("nf,btf->btn", B_complex, u)
    else:
        # Backward compatible: expand input to num_filters if needed
        if input_is_complex:
            if in_filters == 1 and num_filters > 1:
                u = u.expand(-1, -1, num_filters).contiguous()
            u_projected = u
        else:
            if in_filters == 1 and num_filters > 1:
                real_u = real_u.expand(-1, -1, num_filters).contiguous()
            u_projected = real_u

    # Use Triton parallel scan kernel on CUDA (only for complex inputs without B matrix)
    # Note: B matrix support exists in sequential_scan via Triton selective scan kernels
    # For parallel scan with B matrix, we use PyTorch Blelloch scan (still O(log T) parallel)
    if u_projected.is_cuda and _HAS_TRITON and not use_matrix:
        if not input_is_complex:
            return _triton_parallel_scan_real(C1, C2, u_projected)
        return _triton_parallel_scan(C1, C2, u_projected)

    # Fallback: PyTorch Blelloch scan (works on CPU)
    # u_projected is already (C2 ⊙ B) @ u if use_matrix, otherwise apply C2 scaling
    if not input_is_complex and not use_matrix:
        u_projected = u_projected.to(torch.cfloat)
    scan_input = u_projected if use_matrix else C2.unsqueeze(0).unsqueeze(0) * u_projected

    A_real = C1.real.unsqueeze(0).unsqueeze(0).expand(batch, time, -1).contiguous()
    A_imag = C1.imag.unsqueeze(0).unsqueeze(0).expand(batch, time, -1).contiguous()

    scan_input_real = scan_input.real.contiguous()
    scan_input_imag = scan_input.imag.contiguous()

    # Run PyTorch Blelloch parallel scan
    _, _, x_real, x_imag = _blelloch_scan(A_real, A_imag, scan_input_real, scan_input_imag)

    return torch.complex(x_real, x_imag)
