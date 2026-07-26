"""Power-preserving weight initialization for RQF blocks.

The complex projection ``W`` of an RQF layer feeds a resonant filter whose DC
power gain at critical damping is ``G_f = 1/2``, followed by a split
activation with power gain ``G_rho``. Solving ``2 sigma^2 n_in G_f G_rho = 1``
gives the per-component std that keeps the layer's output power at its input
power, so deep stacks neither explode nor vanish at initialization.
"""

from __future__ import annotations

import math

import torch

# DC power gain |H(0)|^2 of the critical-gamma resonator: exact for slow,
# band-limited drive, independent of theta.
NOMINAL_FILTER_GAIN: float = 0.5

# Std for terminal/output blocks that should start near zero (logits heads).
NEAR_ZERO_STD: float = 1e-3

# G_rho: power gain E|rho(z)|^2 / E|z|^2 of an inter-layer activation under
# proper complex Gaussian input.
RQF_ACTIVATION_GAINS: dict[str, float] = {
    "identity": 1.0,
    # Exact at the small-signal edge; mildly contractive at unit power.
    "split_tanh": 1.0,
    # Rectifying real and imaginary parts independently keeps half the power.
    "split_relu": 0.5,
}


def rqf_forward_std(n_in: int, in_activation: str = "identity") -> float:
    """Forward-preserving per-component std for an RQF layer's complex ``W``.

    ``sigma = 1 / sqrt(2 * n_in * G_f * G_rho)`` with the nominal
    ``G_f = 1/2``. ``in_activation`` is the activation feeding this layer's
    input; the first layer consumes raw standardized data (``"identity"``).
    """
    if in_activation not in RQF_ACTIVATION_GAINS:
        raise ValueError(
            f"unknown activation {in_activation!r} for the W init criterion; "
            f"expected one of {sorted(RQF_ACTIVATION_GAINS)}"
        )
    g_rho = RQF_ACTIVATION_GAINS[in_activation]
    return 1.0 / math.sqrt(2.0 * n_in * NOMINAL_FILTER_GAIN * g_rho)


def init_complex_weight_(weight: torch.Tensor, std: float) -> torch.Tensor:
    """Fill ``weight`` in place with per-component standard deviation ``std``.

    For a complex tensor the real and imaginary parts are independent
    ``N(0, std^2)``, giving ``E|W_ij|^2 = 2 std^2``.
    """
    if weight.is_complex():
        real = weight.real
        re = torch.randn(real.shape, dtype=real.dtype, device=real.device)
        im = torch.randn(real.shape, dtype=real.dtype, device=real.device)
        with torch.no_grad():
            weight.copy_(torch.complex(re * std, im * std))
    else:
        with torch.no_grad():
            weight.normal_(0.0, std)
    return weight
