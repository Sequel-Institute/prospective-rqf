"""S4-style RQF block and encoder.

Single-path architecture with post-scan non-linearities::

    Input -> Norm -> Linear -> Scan -> readout -> [act] -> [expand] -> Dropout -> [GLU] -> [+ residual]

Unlike the Mamba variant, there is no dual-path gating around the scan.
Supports optional GLU output and optional skip connections.
"""

import torch
import torch.nn as nn

from rqf_models.embedding.rqf.block import BaseRQFBlock
from rqf_models.embedding.rqf.encoder import BaseRQFEncoder


class S4RQFBlock(BaseRQFBlock):
    """S4-style RQF block with single-path scan and optional GLU output.

    Inherits RQF filter construction, scan execution, and readout from
    :class:`~rqf_models.embedding.rqf.block.BaseRQFBlock`.
    Adds a single-path input projection, optional post-scan expansion,
    optional GLU output, and optional skip connections.

    Args:
        d_model: Model dimension (input and output width).
        num_filters: RQF filters per direction.
        samp_frequency: Sampling frequency (Hz).
        min_freq: Minimum learnable frequency (Hz).
        max_freq: Maximum learnable frequency (Hz).
        dropout: Dropout probability.
        activation: Activation function (``"gelu"``, ``"relu"``, ``"silu"``).
        use_parallel_scan: Use Blelloch O(log T) scan instead of O(T) sequential.
        bidirectional: Use forward + backward scan.
        expand_factor: Post-scan expansion factor (0 = no expansion).
        discretization: RQF discretization (``"euler"``, ``"zoh"``,
            ``"bilinear"``, ``"foh"``).
        tau: RQF time constant (seconds).
        learn_tau: Make tau learnable.
        max_gamma: Upper bound on damping coefficient.
        min_gamma: Lower bound on damping coefficient.
        sigma_init: Initial sigma for divisive normalization readout.
        norm_type: ``"none"``, ``"layer"``, or ``"batch"``.
        filter_scales: Per-filter group config (see :class:`BaseRQFBlock`).
        readout_mode: ``"magnitude"``, ``"4channel"``, ``"2channel"``,
            ``"real"``, ``"divnorm"``, ``"4channel_divnorm"``, or
            ``"2channel_divnorm"``.
        num_oscillators: Parallel oscillators per filter (S4-style N).
        use_glu: Use GLU output projection. Disable for local learning.
        skip_connection: Add residual connection.
        c_projection: Learned complex output weight per filter for N=1.
        clamp_gamma: Dynamically clamp gamma based on learned frequency.
        d_model_out: See :class:`BaseRQFBlock`.
    """

    def __init__(
        self,
        d_model: int,
        num_filters: int = 16,
        samp_frequency: float = 256.0,
        min_freq: float = 1.0,
        max_freq: float = 100.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_parallel_scan: bool = False,
        bidirectional: bool = True,
        expand_factor: int = 2,
        discretization: str = "zoh",
        tau: float = 0.001,
        learn_tau: bool = False,
        max_gamma: float = 0.1,
        min_gamma: float = 0.01,
        sigma_init: float = 0.1,
        norm_type: str = "none",
        filter_scales: list[dict] | None = None,
        readout_mode: str = "real",
        num_oscillators: int = 1,
        use_glu: bool = True,
        skip_connection: bool = True,
        clamp_gamma: bool = False,
        c_projection: bool = False,
        d_model_out: int | None = None,
        prospective_input: bool = False,
        prospective_mode: str = "tau",
        prospective_tau: float | list[float] | None = None,
        scan_grad_mode: str = "full",
        tbptt_delayed_input_mode: str = "strict",
    ):
        if readout_mode == "complex":
            raise ValueError(
                "S4RQFBlock does not support readout_mode='complex' because its "
                "post-scan layers (Linear, GLU) require real-valued input. "
                "Use PureRQFBlock for complex passthrough."
            )

        super().__init__(
            d_model=d_model,
            num_filters=num_filters,
            samp_frequency=samp_frequency,
            min_freq=min_freq,
            max_freq=max_freq,
            dropout=dropout,
            activation=activation,
            use_parallel_scan=use_parallel_scan,
            bidirectional=bidirectional,
            discretization=discretization,
            tau=tau,
            learn_tau=learn_tau,
            max_gamma=max_gamma,
            min_gamma=min_gamma,
            sigma_init=sigma_init,
            norm_type=norm_type,
            filter_scales=filter_scales,
            readout_mode=readout_mode,
            num_oscillators=num_oscillators,
            c_projection=c_projection,
            clamp_gamma=clamp_gamma,
            d_model_out=d_model_out,
            prospective_input=prospective_input,
            prospective_mode=prospective_mode,
            prospective_tau=prospective_tau,
            scan_grad_mode=scan_grad_mode,
            tbptt_delayed_input_mode=tbptt_delayed_input_mode,
        )

        self.use_glu = use_glu
        self.skip_connection = skip_connection

        self.in_proj = nn.Linear(d_model, self.total_filters, bias=False)

        inner_dim = self.total_filters * expand_factor if expand_factor > 0 else self.total_filters
        if self.scan_output_width != inner_dim:
            self.post_scan_proj = nn.Linear(self.scan_output_width, inner_dim, bias=False)
        else:
            self.post_scan_proj = None

        if use_glu:
            self.output_linear = nn.Sequential(
                nn.Linear(inner_dim, self.d_model_out * 2, bias=False),
                nn.GLU(dim=-1),
            )
        else:
            self.output_linear = nn.Linear(inner_dim, self.d_model_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Norm → Linear → Scan → [act] → [expand] → Dropout → [GLU] → [+ residual].

        Args:
            x: ``(batch, time, d_model)``

        Returns:
            ``(batch, time, d_model_out)``
        """
        residual = x

        x = self._prenorm(x)

        delayed_x = None
        if self._use_tbptt1_scan_grad() and self._use_two_tap:
            if self.tbptt_delayed_input_mode == "strict":
                # Strict spatial BP cuts the temporal edge into the previous
                # timestep activation, but the delayed prospective tap should
                # still train this layer's input projection:
                # u[t-1] = W(x[t-1].detach()).
                delayed_x = self.in_proj(x.detach())
            elif self.tbptt_delayed_input_mode == "literal":
                # Diagnostic-only state-truncated TBPTT: cut h[t-1], but keep
                # the explicit delayed input edge from the forward scan.
                delayed_x = None
            else:  # pragma: no cover - validated by BaseRQFBlock
                raise RuntimeError(
                    f"unsupported tbptt_delayed_input_mode={self.tbptt_delayed_input_mode!r}"
                )

        x = self.in_proj(x)

        # RQF scan (always in float32)
        with torch.amp.autocast(x.device.type, enabled=False):
            x = x.float()
            if delayed_x is not None:
                delayed_x = delayed_x.float()

            cw_fwd = getattr(self, "c_weights_fwd", None)
            cw_bwd = getattr(self, "c_weights_bwd", None)

            if self.bidirectional:
                x_fwd = x[..., : self.num_filters]
                x_bwd = x[..., self.num_filters :]
                delayed_fwd = delayed_x[..., : self.num_filters] if delayed_x is not None else None
                delayed_bwd = delayed_x[..., self.num_filters :] if delayed_x is not None else None
                y_fwd = self._scan_direction(
                    x_fwd,
                    self.rqf_fwd,
                    cw_fwd,
                    delayed_x=delayed_fwd,
                )
                y_bwd = self._scan_direction(
                    x_bwd,
                    self.rqf_bwd,
                    cw_bwd,
                    delayed_x=delayed_bwd,
                    flip=True,
                )
                y = self._apply_readout(y_fwd, y_bwd)
            else:
                y = self._scan_direction(x, self.rqf_fwd, cw_fwd, delayed_x=delayed_x)
                y = self._apply_readout(y)

        # Post-scan: activation → [expand] → dropout → output
        if self.readout_mode not in ("divnorm", "4channel_divnorm"):
            y = self.act(y)
        if self.post_scan_proj is not None:
            y = self.post_scan_proj(y)
        y = self.dropout(y)
        y = self.output_linear(y)

        if self.skip_connection:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            y = y + residual

        return y


class S4RQFEncoder(BaseRQFEncoder):
    """Stacked S4-style RQF encoder.

    Stacks :class:`S4RQFBlock` layers between input and output linear
    projections. All block-level parameters are forwarded; see
    :class:`S4RQFBlock` for their meaning.

    Args:
        input_dims: Input feature dimension.
        output_dims: Output feature dimension.
        hidden_dims: Hidden dimension for all blocks (int) or per-layer
            dimensions (list of ints). When a list is given, its length
            must equal ``num_layers`` and linear projections are inserted
            between layers whose dimensions differ.
        num_layers: Number of stacked blocks.
        filter_scales: Flat list (uniform) or nested list (per-layer) of
            filter group configs. See :class:`BaseRQFBlock`.
        **kwargs: Forwarded to :class:`S4RQFBlock` (``num_filters``,
            ``samp_frequency``, ``bidirectional``, ``expand_factor``, etc.).

    Example:
        >>> encoder = S4RQFEncoder(
        ...     input_dims=20, output_dims=20, hidden_dims=64, num_layers=3
        ... )
        >>> encoder(torch.randn(8, 161, 20)).shape
        torch.Size([8, 161, 20])
    """

    _block_class = S4RQFBlock

    def __init__(
        self,
        input_dims: int = 1,
        output_dims: int = 1,
        hidden_dims: int | list[int] = 64,
        num_layers: int = 4,
        num_filters: int = 16,
        samp_frequency: float = 256.0,
        min_freq: float = 1.0,
        max_freq: float = 100.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_parallel_scan: bool = False,
        bidirectional: bool = True,
        expand_factor: int = 2,
        discretization: str = "zoh",
        tau: float = 0.001,
        learn_tau: bool = False,
        max_gamma: float = 0.1,
        min_gamma: float = 0.01,
        sigma_init: float = 0.1,
        norm_type: str = "none",
        filter_scales: list[dict] | list[list[dict]] | None = None,
        readout_mode: str = "real",
        num_oscillators: int = 1,
        use_glu: bool = True,
        skip_connection: bool = True,
        clamp_gamma: bool = False,
        c_projection: bool = False,
        prospective_input: bool = False,
        prospective_mode: str = "tau",
        prospective_tau: float | list[float] | None = None,
        scan_grad_mode: str = "full",
        tbptt_delayed_input_mode: str = "strict",
    ):
        # Store block kwargs before super().__init__ calls _create_blocks
        self._block_kwargs = {
            "num_filters": num_filters,
            "samp_frequency": samp_frequency,
            "min_freq": min_freq,
            "max_freq": max_freq,
            "dropout": dropout,
            "activation": activation,
            "use_parallel_scan": use_parallel_scan,
            "bidirectional": bidirectional,
            "expand_factor": expand_factor,
            "discretization": discretization,
            "tau": tau,
            "learn_tau": learn_tau,
            "max_gamma": max_gamma,
            "min_gamma": min_gamma,
            "sigma_init": sigma_init,
            "norm_type": norm_type,
            "readout_mode": readout_mode,
            "num_oscillators": num_oscillators,
            "use_glu": use_glu,
            "skip_connection": skip_connection,
            "clamp_gamma": clamp_gamma,
            "c_projection": c_projection,
            "prospective_input": prospective_input,
            "prospective_mode": prospective_mode,
            "prospective_tau": prospective_tau,
            "scan_grad_mode": scan_grad_mode,
            "tbptt_delayed_input_mode": tbptt_delayed_input_mode,
        }

        super().__init__(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            norm_type=norm_type,
            filter_scales=filter_scales,
        )

    def extra_repr(self) -> str:
        """Extra representation string for print."""
        parts = [
            f"num_layers={self.num_layers}",
            f"hidden_dims={self.hidden_dims}",
        ]
        if not self._block_kwargs.get("use_glu", True):
            parts.append("no_glu")
        if not self._block_kwargs.get("skip_connection", True):
            parts.append("no_skip")
        readout = self._block_kwargs.get("readout_mode", "magnitude")
        if readout != "magnitude":
            parts.append(f"readout={readout}")
        return ", ".join(parts)
