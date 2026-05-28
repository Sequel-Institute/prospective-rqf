"""Vanilla S4D encoder package.

Diagonal state-space model (S4D) vendored from state-spaces/s4
(Apache-2.0). See :mod:`rqf_models.backbones.s4d.kernel` for the kernel
implementation and license attribution.
"""

from rqf_models.backbones.s4d.block import S4DBlock, S4DLayer
from rqf_models.backbones.s4d.encoder import S4DEncoder
from rqf_models.backbones.s4d.kernel import S4DKernel

__all__ = [
    "S4DEncoder",
    "S4DBlock",
    "S4DLayer",
    "S4DKernel",
]
