"""Pure RQF block and encoder with complex passthrough.

Minimal RQF architecture: complex weight matrix → ZOH scan → split
activation → complex output.

Unlike S4/Mamba variants, complex values pass directly between layers with
no per-block readout conversion. Real-valued readout happens only at the
encoder output.

Example::

    Input → [PureRQFBlock × N] → Re(y) → output_proj
              complex passthrough
"""

import math
from typing import Any

import torch
import torch.nn as nn

from organics.embedding.rqf.block import BaseRQFBlock
from organics.embedding.rqf.encoder import BaseRQFEncoder


class PureRQFBlock(BaseRQFBlock):
    """Pure RQF block: Complex Linear → ZOH scan → split activation.

    Always outputs complex tensors. Uses ``readout_mode="complex"``
    (passthrough) so the base class scan infrastructure is fully reused
    without real-valued conversion.

    Compared to :class:`~organics.embedding.rqf.s4.S4RQFBlock`, this block:
    - Sets ``num_filters = d_model_out`` (one RQF oscillator per output neuron)
    - Adds a complex-valued weight matrix ``W`` for input projection
    - Has no post-scan processing (no GLU, dropout, skip, normalization)

    Args:
        d_model: Block input dimension.
        complex_weights: Use complex-valued weight matrix. When ``False``,
            a real-valued matrix is used (cast to complex in forward).
        **kwargs: Forwarded to :class:`BaseRQFBlock`. ``num_filters`` is
            automatically set to ``d_model_out`` (or ``d_model`` when
            ``d_model_out`` is not provided).
    """

    def __init__(self, d_model: int, complex_weights: bool = True, **kwargs: Any) -> None:
        d_out = kwargs.get("d_model_out") or d_model
        super().__init__(d_model=d_model, num_filters=d_out, **kwargs)

        if complex_weights:
            std = 1.0 / math.sqrt(d_model)
            self.W = nn.Parameter(
                torch.complex(
                    torch.randn(d_out, d_model) * std,
                    torch.randn(d_out, d_model) * std,
                )
            )
        else:
            self.W = nn.Parameter(torch.randn(d_out, d_model) / math.sqrt(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: W @ input → scan → activation.

        Delegates the scan dispatch (prospective vs standard) to
        :meth:`BaseRQFBlock._scan_direction`, which is the single source
        of truth for the RQF-family prospective implementation. Pure is
        forward-only, so ``flip=False`` is hard-coded.

        Args:
            x: Complex tensor ``(batch, time, d_model)``.

        Returns:
            Complex tensor ``(batch, time, d_model_out)``.
        """
        W_c = self.W.to(torch.complex64) if not self.W.is_complex() else self.W
        delayed_z = None
        if self._use_tbptt1_scan_grad() and self._use_two_tap:
            # Keep the delayed tap differentiable w.r.t. this layer's W,
            # while cutting temporal gradients into the source activation.
            delayed_z = torch.matmul(x.detach(), W_c.t())
        z = torch.matmul(x, W_c.t())

        c_weights = self.c_weights_fwd if self._has_c_weights else None
        y = self._scan_direction(z, self.rqf_fwd, c_weights, delayed_x=delayed_z, flip=False)
        return self.act(y)


class PureRQFEncoder(BaseRQFEncoder):
    """Stacked pure RQF encoder with complex passthrough between layers.

    Wraps :class:`PureRQFBlock` layers with input/output projections.
    Complex values flow through all blocks; ``Re(y)`` is taken only at
    the output before the final linear projection.

    Accepts the same core parameters as :class:`S4RQFEncoder` (frequency,
    tau, gamma, discretization). The only pure-specific parameter is
    ``complex_weights``.

    Args:
        input_dims: Input feature dimension.
        output_dims: Output feature dimension.
        hidden_dims: Hidden dimension per layer (int or list).
        num_layers: Number of stacked blocks.
        samp_frequency: Sampling frequency in Hz (``1 / dt``).
        min_freq: Minimum learnable frequency in Hz.
        max_freq: Maximum learnable frequency in Hz.
        dropout: Dropout rate (default ``0.0`` — pure blocks have no dropout).
        activation: Complex activation (``"split_tanh"``, ``"split_relu"``,
            ``"identity"``).
        discretization: Discretization method (``"zoh"``, ``"euler"``,
            ``"bilinear"``, ``"foh"``).
        tau: RQF time constant (seconds).
        learn_tau: Whether tau is learnable.
        max_gamma: Maximum damping coefficient.
        min_gamma: Minimum damping coefficient.
        norm_type: Final normalization (``"none"``, ``"layer"``, ``"batch"``).
        filter_scales: Per-layer filter config for hierarchical frequency
            bands. See :class:`BaseRQFBlock`.
        clamp_gamma: Dynamically clamp gamma based on learned frequency.
        complex_weights: Use complex weight matrices in blocks.
        skip_connection: Add a complex skip connection between pure RQF
            layers when their shapes match.

    Example:
        >>> encoder = PureRQFEncoder(
        ...     input_dims=1, output_dims=32, hidden_dims=32, num_layers=5,
        ...     samp_frequency=10000, min_freq=100, max_freq=5000,
        ...     tau=5e-5, activation="split_tanh",
        ... )
        >>> encoder(torch.randn(8, 40, 1)).shape
        torch.Size([8, 40, 32])
    """

    _block_class = PureRQFBlock

    def __init__(
        self,
        input_dims: int = 1,
        output_dims: int = 32,
        hidden_dims: int | list[int] = 32,
        num_layers: int = 5,
        samp_frequency: float = 10000.0,
        min_freq: float = 1.0,
        max_freq: float = 50.0,
        dropout: float = 0.0,
        activation: str = "split_tanh",
        discretization: str = "zoh",
        tau: float = 0.001,
        learn_tau: bool = False,
        max_gamma: float = 0.85,
        min_gamma: float = 0.001,
        norm_type: str = "none",
        filter_scales: list[dict] | list[list[dict]] | None = None,
        clamp_gamma: bool = False,
        complex_weights: bool = True,
        prospective_input: bool = False,
        prospective_mode: str = "tau",
        prospective_tau: float | list[float] | None = None,
        skip_connection: bool | None = None,
        scan_grad_mode: str = "full",
        **legacy_kwargs: Any,
    ) -> None:
        residual_connection = legacy_kwargs.pop("residual_connection", None)
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
        if skip_connection is None:
            skip_connection = (
                bool(residual_connection) if residual_connection is not None else False
            )
        elif residual_connection is not None and bool(residual_connection) != bool(skip_connection):
            raise ValueError(
                "Received both skip_connection and residual_connection with different values"
            )

        self._block_kwargs = {
            "samp_frequency": samp_frequency,
            "min_freq": min_freq,
            "max_freq": max_freq,
            "dropout": dropout,
            "activation": activation,
            "use_parallel_scan": False,
            "bidirectional": False,
            "discretization": discretization,
            "tau": tau,
            "learn_tau": learn_tau,
            "max_gamma": max_gamma,
            "min_gamma": min_gamma,
            "sigma_init": 0.1,
            "norm_type": "none",
            "readout_mode": "complex",
            "num_oscillators": 1,
            "c_projection": False,
            "clamp_gamma": clamp_gamma,
            "complex_weights": complex_weights,
            "prospective_input": prospective_input,
            "prospective_mode": prospective_mode,
            "prospective_tau": prospective_tau,
            "scan_grad_mode": scan_grad_mode,
        }
        super().__init__(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            norm_type=norm_type,
            filter_scales=filter_scales,
        )
        self.skip_connection = bool(skip_connection)

    @property
    def residual_connection(self) -> bool:
        """Backward-compatible alias for older Pure RQF callers."""
        return self.skip_connection

    @residual_connection.setter
    def residual_connection(self, value: bool) -> None:
        self.skip_connection = bool(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process through stacked pure RQF blocks with complex passthrough.

        Args:
            x: Input tensor ``(batch, time, input_dims)``.

        Returns:
            Real output tensor ``(batch, time, output_dims)``.
        """
        x = self.input_proj(x)

        h = x.to(torch.complex64)
        with torch.amp.autocast(x.device.type, enabled=False):
            for block in self.blocks:
                out = block(h)
                # Complex skip: only when shape-preserving (uniform hidden_dims).
                h = out + h if self.skip_connection and out.shape == h.shape else out

        h = h.real

        if self.final_norm is not None:
            if self._use_batch_norm:
                h = self.final_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.final_norm(h)

        return self.output_proj(h)
