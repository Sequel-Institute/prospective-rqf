"""Pure RQF block and encoder with complex passthrough.

Standalone RQF architecture: complex weight matrix → ZOH scan → split
activation → complex output.

Unlike S4/Mamba variants, complex values pass directly between layers with
no per-block readout conversion. Real-valued readout happens only at the
encoder output.

Example::

    Input → [PureRQFBlock × N] → Re(y) → output_proj
              complex passthrough
"""

import math
import os
from typing import Any

import torch
import torch.nn as nn

from rqf_models.backbones.rqf.block import BaseRQFBlock
from rqf_models.backbones.rqf.encoder import BaseRQFEncoder
from rqf_models.backbones.rqf.selective import SelectiveController, SelectiveRQFConfig
from rqf_models.backbones.rqf.utils.activations import SplitReLU
from rqf_models.backbones.rqf.utils.init import init_complex_weight_, rqf_forward_std
from rqf_models.ops import scan as _scan_module
from rqf_models.ops.selective_scan import selective_layer_scan

# Channel-outermost fused prospective scan (optional, requires Triton + CUDA).
# Runs each block's projection as one 2D complex GEMM and the two-tap scan in
# (filters, batch, time) layout with the drive (and optionally split-relu)
# fused into the kernel — no per-layer time-axis permutes.
try:
    from rqf_models.ops.fbt_scan_kernel import fbt_prospective_scan, fbt_recompute_scan

    _HAS_FBT_PROSPECTIVE = True
except ImportError:
    _HAS_FBT_PROSPECTIVE = False

# Grid size (filters * batch) above which the fused split-relu scan recomputes
# the raw state in the backward instead of storing it (lower peak memory).
_FBT_RECOMPUTE_MIN_GRID = 256


def validate_selective_tbptt_mode(
    *,
    scan_grad_mode: str,
    tbptt_delayed_input_mode: str,
    prospective_input: bool,
    prospective_tau: float | list[float] | torch.Tensor | None,
) -> None:
    """Reject an unsupported lag gradient only when its tap is nonzero."""
    if (scan_grad_mode != "tbptt1" or tbptt_delayed_input_mode != "literal"
            or not prospective_input):
        return
    # With fixed positive tau, an explicit all-zero horizon removes the lag
    # tap. None selects the nonzero default for either prospective mode.
    if prospective_tau is not None:
        horizon = (prospective_tau if isinstance(prospective_tau, torch.Tensor)
                   else torch.as_tensor(prospective_tau, dtype=torch.float64))
        if not bool(horizon.ne(0).any()):
            return
    raise ValueError(
        "selective Pure RQF does not support tbptt_delayed_input_mode='literal' "
        "with prospective TBPTT(1) and a nonzero horizon; use 'strict' or full BPTT"
    )


class PureRQFBlock(BaseRQFBlock):
    """Pure RQF block: Complex Linear → ZOH scan → split activation.

    Always outputs complex tensors. Uses ``readout_mode="complex"``
    (passthrough) so the base class scan infrastructure is fully reused
    without real-valued conversion.

    Compared to :class:`~rqf_models.backbones.rqf.s4_rqf.S4RQFBlock`, this block:
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

    def __init__(self, d_model: int, complex_weights: bool = True,
                 selective: SelectiveRQFConfig | dict | None = None, **kwargs: Any) -> None:
        d_out = kwargs.get("d_model_out") or d_model
        super().__init__(d_model=d_model, num_filters=d_out, **kwargs)

        # Whether to apply the split activation on this block's output. The
        # encoder clears this on the last block when ``final_activation=False``
        # so the readout consumes the raw complex scan state.
        self.apply_activation = True

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

        self.selective = SelectiveRQFConfig.resolve(selective)
        self.selector = None
        if self.selective is not None:
            rqf = self.rqf_fwd
            if (rqf.discretization != "zoh" or rqf.param_space != "log" or rqf.learn_tau
                    or rqf.clamp_gamma or not complex_weights or self.bidirectional
                    or self.scan_filters != self.num_filters or self._has_c_weights
                    or not isinstance(self.act, SplitReLU)):
                raise ValueError("selective Pure RQF requires ZOH, log parameters, fixed tau, "
                                 "clamp_gamma=False, complex weights, split_relu, and one "
                                 "forward oscillator per channel")
            if (not bool((torch.isfinite(rqf.tau) & (rqf.tau > 0)).all())
                    or not math.isfinite(rqf.dt) or rqf.dt <= 0):
                raise ValueError("selective tau and dt must be positive and finite")
            self._validate_selective_tbptt_mode()
            self.selector = SelectiveController(d_model, rqf, self.selective)

    def _validate_selective_tbptt_mode(self) -> None:
        validate_selective_tbptt_mode(
            scan_grad_mode=self.scan_grad_mode,
            tbptt_delayed_input_mode=self.tbptt_delayed_input_mode,
            prospective_input=self.prospective_input,
            prospective_tau=self.prospective_tau,
        )

    def _forward_selective(self, x: torch.Tensor, source: torch.Tensor, *,
                           write_raw: bool = True) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        self._validate_selective_tbptt_mode()
        inputs = self.selector(source, self.rqf_fwd, prospective=self.prospective_input,
                               mode=self.prospective_mode, prospective_tau=self.prospective_tau)
        use_kernel = (_scan_module._SCAN_BACKEND == "chunked"
                      and os.environ.get("ORGANICS_RQF_FBT", "1") != "0")
        return selective_layer_scan(x, self.W, inputs, truncated=self.scan_grad_mode == "tbptt1",
                                    activate=self.apply_activation, write_raw=write_raw,
                                    use_kernel=use_kernel)

    def forward(self, x: torch.Tensor, source: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass: W @ input → scan → activation.

        Delegates the scan dispatch (prospective vs standard) to
        :meth:`BaseRQFBlock._scan_direction`, which is the single source
        of truth for the RQF-family prospective implementation. Pure is
        forward-only, so ``flip=False`` is hard-coded.

        Args:
            x: Complex tensor ``(batch, time, d_model)``.
            source: Selector source ``(batch, time, d_model)`` for selective
                blocks configured with ``input_source="layer_output"`` (the
                raw, pre-activation previous-layer output plus any skip).
                Ignored for static blocks; for ``layer_input`` selectors the
                block reads ``x`` itself.

        Returns:
            Complex tensor ``(batch, time, d_model_out)``.
        """
        if self.selector is not None:
            if self.selective.input_source == "layer_output":
                if source is None:
                    raise ValueError(
                        "selective block with input_source='layer_output' needs the raw "
                        "previous-layer source: call block(x, source=raw) or run the "
                        "block inside PureRQFEncoder"
                    )
                selector_source = source.permute(2, 0, 1)
            else:
                selector_source = x.permute(2, 0, 1)
            raw, act = self._forward_selective(x.permute(2, 0, 1), selector_source,
                                               write_raw=not self.apply_activation)
            return (act if self.apply_activation else raw).permute(1, 2, 0)
        W_c = self.W.to(torch.complex64) if not self.W.is_complex() else self.W
        delayed_z = None
        if self._use_tbptt1_scan_grad() and self._use_two_tap:
            # Keep the delayed tap differentiable w.r.t. this layer's W,
            # while cutting temporal gradients into the source activation.
            delayed_z = torch.matmul(x.detach(), W_c.t())
        z = torch.matmul(x, W_c.t())

        c_weights = self.c_weights_fwd if self._has_c_weights else None
        y = self._scan_direction(z, self.rqf_fwd, c_weights, delayed_x=delayed_z, flip=False)
        return self.act(y) if self.apply_activation else y

    def _fbt_eligible(self) -> bool:
        """Whether this block supports the channel-outermost fused path.

        Requires the single-oscillator, unidirectional, prospective two-tap,
        full-gradient configuration (no C-weight oscillator combination) with a
        complex weight, which the fused ``(filters, batch, time)`` scan
        implements.
        """
        return (
            self.selective is None
            and self.prospective_input
            and not self.bidirectional
            and not self._has_c_weights
            and self.scan_filters == self.num_filters
            and self.scan_grad_mode == "full"
            and self.W.is_complex()
        )

    def _forward_fbt(self, h: torch.Tensor) -> torch.Tensor:
        """Forward in channel-outermost ``(filters, batch, time)`` layout.

        Projection is a single 2D complex GEMM ``W @ H`` on the free view
        ``(d_model, batch*time)``; the fused scan drives and (for split-relu)
        activates without permuting the time axis. Equivalent to
        :meth:`forward` on the transposed input.
        """
        d_model, batch, time = h.shape
        W_c = self.W.to(torch.complex64)
        d_out = W_c.shape[0]
        z = torch.matmul(W_c, h.reshape(d_model, batch * time)).reshape(d_out, batch, time)

        C1, tap0, tap1 = self.rqf_fwd.compute_two_tap_coefficients(
            prospective=self.prospective_input,
            mode=self.prospective_mode,
            prospective_tau=self.prospective_tau,
        )
        C1 = C1.to(torch.cfloat)
        tap0 = tap0.to(torch.cfloat)
        tap1 = tap1.to(torch.cfloat)
        fuse = self.apply_activation and isinstance(self.act, SplitReLU)
        if fuse and d_out * batch >= _FBT_RECOMPUTE_MIN_GRID:
            # Large grid: recompute the raw state in the backward instead of
            # storing it (lower peak memory, same split-relu output).
            return fbt_recompute_scan(C1, tap0, tap1, z)
        y = fbt_prospective_scan(C1, tap0, tap1, z, fuse_act=fuse)
        # Split activations are pointwise, so they apply unchanged in this
        # layout; split-relu is already folded into the scan when ``fuse``.
        if fuse or not self.apply_activation:
            return y
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
        input_projection: If false, the first RQF directly consumes raw inputs.
        output_projection: May be disabled with ``complex_output=True`` to
            omit the unused real output projection.

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
        param_space: str = "sigmoid",
        seq_duration: float | None = None,
        filter_max_freq_factor: float = 1.0,
        w_init: str | float | None = None,
        final_activation: bool = True,
        input_proj_bias: bool = True,
        complex_output: bool = False,
        selective: SelectiveRQFConfig | dict | None = None,
        input_projection: bool = True,
        output_projection: bool = True,
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

        self.selective = SelectiveRQFConfig.resolve(selective)
        if not output_projection and not complex_output:
            raise ValueError("output_projection=False requires complex_output=True")
        self._block_kwargs = {
            "selective": self.selective,
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
            "param_space": param_space,
            "seq_duration": seq_duration,
            "filter_max_freq_factor": filter_max_freq_factor,
        }
        super().__init__(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            norm_type=norm_type,
            filter_scales=filter_scales,
            input_projection=input_projection,
            output_projection=output_projection,
        )
        self.skip_connection = bool(skip_connection)
        if not input_projection:
            # Pure RQF uses shape-matched identity skips only. The base block's
            # width-changing residual projections are unused in this stack.
            for block in self.blocks:
                block.residual_proj = None
        # complex_output: return the raw complex features (batch, time, F) and
        # skip .real / final_norm / output_proj — for external complex readouts.
        self.complex_output = bool(complex_output)

        # Power-preserving W init recipe. Layer 0 consumes the raw
        # standardized input (identity gain); deeper layers consume the
        # inter-layer split activation.
        if w_init is not None:
            for idx, block in enumerate(self.blocks):
                if isinstance(w_init, int | float):
                    std = float(w_init)
                elif w_init == "forward":
                    in_act = "identity" if idx == 0 else activation
                    std = rqf_forward_std(block.W.shape[1], in_act)
                else:
                    raise ValueError(
                        f"w_init must be None, 'forward', or a float std; got {w_init!r}"
                    )
                init_complex_weight_(block.W, std)

        # Raw complex state into the readout: skip the last block's activation.
        if not final_activation:
            self.blocks[-1].apply_activation = False

        # Bias-free input projection (removes the learnable DC drive).
        if not input_proj_bias and input_projection:
            self.input_proj.register_parameter("bias", None)

    @torch.no_grad()
    def clamp_parameters(self) -> None:
        """Project all blocks' filter parameters into their valid bands.

        Call after each optimizer step when using ``param_space="log"``
        (no-op for the sigmoid parametrization).
        """
        for block in self.blocks:
            block.rqf_fwd.clamp_parameters()
            if getattr(block, "rqf_bwd", None) is not None:
                block.rqf_bwd.clamp_parameters()

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
        if self.selective is not None:
            h = self._forward_selective_stack(x)
        elif self._fbt_enabled(x):
            h = self._forward_fbt_stack(x)
        else:
            x = self.input_proj(x)

            h = x.to(torch.complex64)
            with torch.amp.autocast(x.device.type, enabled=False):
                for block in self.blocks:
                    out = block(h)
                    # Complex skip: only when shape-preserving (uniform hidden_dims).
                    h = out + h if self.skip_connection and out.shape == h.shape else out

        if self.complex_output:
            return h

        h = h.real

        if self.final_norm is not None:
            if self._use_batch_norm:
                h = self.final_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.final_norm(h)

        return self.output_proj(h)

    def _forward_selective_stack(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(x)
        h = projected.permute(2, 0, 1).to(self.blocks[0].W.dtype).contiguous()
        source = h
        raw_selectors = self.selective.input_source == "layer_output"
        with torch.autocast(x.device.type, enabled=False):
            for index, block in enumerate(self.blocks):
                # The raw state feeds the next layer's selectors; the last
                # block only needs it when its own activation is disabled.
                needs_raw = raw_selectors and index < len(self.blocks) - 1
                raw, act = block._forward_selective(
                    h, source if raw_selectors else h,
                    write_raw=needs_raw or not block.apply_activation,
                )
                out = act if block.apply_activation else raw
                skip = self.skip_connection and out.shape == h.shape
                source = (raw + h if skip else raw) if needs_raw else None
                h = out + h if skip else out
        return h.permute(1, 2, 0).contiguous()

    def _fbt_enabled(self, x: torch.Tensor) -> bool:
        """Whether the channel-outermost fused path applies to this input."""
        return (
            _HAS_FBT_PROSPECTIVE
            and x.is_cuda
            and _scan_module._SCAN_BACKEND == "chunked"
            and os.environ.get("ORGANICS_RQF_FBT", "1") != "0"
            and all(block._fbt_eligible() for block in self.blocks)
        )

    def _forward_fbt_stack(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack in channel-outermost ``(filters, batch, time)`` layout.

        The input projection is applied channel-first so the ``(d_model,
        batch, time)`` layout falls straight out of the projection: only the
        tiny ``input_dims`` axis is permuted, not the large projected tensor,
        which removes the encoder-entry transpose. Each block then projects
        with a 2D complex GEMM and runs the fused scan in place (no per-layer
        time-axis permute); the output is transposed back to ``(batch, time,
        filters)`` once at exit. Exactly equivalent to the standard path up to
        linear-op reordering.
        """
        batch, time, in_dims = x.shape
        x_cf = x.permute(2, 0, 1).reshape(in_dims, batch * time)  # (input_dims, B*T)
        if self.input_projection:
            w_in = self.input_proj.weight  # (d_model, input_dims)
            b_in = self.input_proj.bias
            proj = w_in @ x_cf if b_in is None else torch.addmm(b_in[:, None], w_in, x_cf)
        else:
            proj = x_cf
        h = proj.reshape(-1, batch, time).to(torch.complex64)  # (F, B, T)
        with torch.amp.autocast(x.device.type, enabled=False):
            for block in self.blocks:
                out = block._forward_fbt(h)
                # Complex skip: only when shape-preserving (uniform hidden_dims).
                h = out + h if self.skip_connection and out.shape == h.shape else out
        return h.permute(1, 2, 0).contiguous()  # back to (batch, time, filters)
