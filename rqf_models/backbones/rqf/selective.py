"""Input-dependent RQF damping, write, and frequency controllers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from rqf_models.ops.selective_scan import SelectiveScanInputs


@dataclass(frozen=True)
class SelectiveRQFConfig:
    gamma_control: bool = True
    gamma_link: str = "sigmoid"
    gamma_features: str = "real_imag"
    gamma_max_log_scale: float = math.log(1.25)
    gamma_edge: str = "interior_clamp"
    gamma_edge_margin: float = 1e-3
    write_control: bool = False
    write_link: str = "sigmoid"
    write_max: float = 2.0
    frequency_warp: str = "none"
    warp_bound: str = "band"
    num_groups: int = 4
    max_log_scale: float = math.log(1.25)
    input_source: str = "layer_output"

    def __post_init__(self) -> None:
        for name, choices in (
            ("gamma_link", {"sigmoid", "log_shift"}),
            ("gamma_features", {"real_imag", "modulus"}),
            ("gamma_edge", {"interior_clamp", "st_clamp"}),
            ("write_link", {"affine", "sigmoid"}),
            ("frequency_warp", {"none", "global", "grouped"}),
            ("warp_bound", {"band", "none"}),
            ("input_source", {"layer_input", "layer_output"}),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"selective.{name} must be one of {sorted(choices)}")
        if type(self.gamma_control) is not bool or type(self.write_control) is not bool:
            raise ValueError("selective controller switches must be booleans")
        if type(self.num_groups) is not int or self.num_groups < 1:
            raise ValueError("selective.num_groups must be a positive integer")
        if not math.isfinite(self.write_max) or self.write_max <= 1:
            raise ValueError("selective.write_max must be finite and greater than one")
        if not math.isfinite(self.gamma_edge_margin) or not 0 < self.gamma_edge_margin < 0.5:
            raise ValueError("selective.gamma_edge_margin must be finite and in (0, 0.5)")
        for name in ("gamma_max_log_scale", "max_log_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"selective.{name} must be finite and positive")
        if not self.gamma_control:
            if self.gamma_link != "sigmoid" or self.gamma_features != "real_imag":
                raise ValueError(
                    "non-default gamma links/features require gamma_control"
                )
            if not self.write_control and self.frequency_warp == "none":
                raise ValueError(
                    "selective requires a controller; use None for static RQFs"
                )

    @classmethod
    def resolve(
        cls, value: SelectiveRQFConfig | Mapping[str, Any] | None
    ) -> SelectiveRQFConfig | None:
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**value)
        raise TypeError("selective must be a SelectiveRQFConfig, mapping, or None")


class _SplitProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, weights, sizes):
        ctx.save_for_backward(v, weights)
        ctx.sizes = sizes
        ctx.set_materialize_grads(False)
        return (weights @ v).split(sizes)

    @staticmethod
    def backward(ctx, *grads):
        v, weights = ctx.saved_tensors
        grad = torch.cat(
            [
                g if g is not None else v.new_zeros((n, v.shape[1]))
                for g, n in zip(grads, ctx.sizes)
                if n
            ]
        )
        with torch.autocast(v.device.type, enabled=False):
            return weights.T @ grad, grad @ v.T, None


class SelectiveController(nn.Module):
    def __init__(
        self, n_in: int, rqf: nn.Module, config: SelectiveRQFConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.n_in = n_in
        n_out = rqf.num_filters
        groups = (
            1
            if config.frequency_warp == "global"
            else config.num_groups
            if config.frequency_warp == "grouped"
            else 0
        )
        if groups > n_out:
            raise ValueError("selective.num_groups cannot exceed any layer's width")
        lo, hi = rqf._min_gamma, rqf._max_gamma
        if not bool(((lo > 0) & (hi > lo) & (hi <= 1)).all()):
            raise ValueError(
                "selective gamma bounds require 0 < min_gamma < max_gamma <= 1"
            )
        if config.gamma_link == "log_shift":
            if not bool((lo.log() < hi.log() - config.gamma_max_log_scale).all()):
                raise ValueError("gamma_max_log_scale leaves no resting gamma range")
            rqf.gamma_log_headroom = config.gamma_max_log_scale
        interior_gamma = (
            config.gamma_control
            and config.gamma_link == "sigmoid"
            and config.gamma_edge == "interior_clamp"
        )
        if interior_gamma:
            rqf.gamma_edge_margin = config.gamma_edge_margin
            span = hi - lo
            if not bool(((lo + span * config.gamma_edge_margin > lo)
                         & (hi - span * config.gamma_edge_margin < hi)).all()):
                raise ValueError("gamma_edge_margin is too small for the parameter precision")
        if groups and config.warp_bound == "band":
            rqf.theta_nyquist_bound = True
            lower, upper = rqf._theta_bounds()
            if not bool((torch.isfinite(lower) & torch.isfinite(upper)
                         & (lower > 0) & (upper >= lower)).all()):
                raise ValueError("selective frequency band has no positive Nyquist-safe range")
        if config.gamma_link == "log_shift" or interior_gamma or rqf.theta_nyquist_bound:
            rqf.clamp_parameters()
        self.W_gamma = (
            nn.Parameter(
                torch.zeros(
                    n_out, n_in if config.gamma_features == "modulus" else 2 * n_in
                ),
            )
            if config.gamma_control
            else None
        )
        self.W_write = (
            nn.Parameter(torch.zeros(n_out, 2 * n_in))
            if config.write_control
            else None
        )
        self.W_delta = (
            nn.Parameter(torch.zeros(groups, 2 * n_in))
            if groups
            else None
        )
        ranks = rqf.log_theta.detach().argsort(stable=True).argsort()
        self.register_buffer("warp_groups", ranks * groups // n_out if groups else None)

    def get_extra_state(self) -> dict[str, Any]:
        return asdict(self.config)

    def set_extra_state(self, state: dict[str, Any]) -> None:
        # Checkpoints predating write_link used the signed affine gain.
        stored = dict(state)
        stored.setdefault("write_link", "affine")
        stored.setdefault("gamma_edge", "st_clamp")
        stored.setdefault("warp_bound", "none")
        restored = asdict(SelectiveRQFConfig(**stored))
        current = asdict(self.config)
        for settings in (restored, current):
            if not settings["gamma_control"] or settings["gamma_link"] != "sigmoid":
                settings.pop("gamma_edge")
                settings.pop("gamma_edge_margin")
            elif settings["gamma_edge"] == "st_clamp":
                settings.pop("gamma_edge_margin")
            if settings["frequency_warp"] == "none":
                settings.pop("warp_bound")
            if not settings["write_control"]:
                settings.pop("write_link")
                settings.pop("write_max")
            elif settings["write_link"] == "affine":
                settings.pop("write_max")
        if restored != current:
            raise ValueError("checkpoint selective settings do not match the model")

    def forward(
        self,
        source: Tensor,
        rqf: nn.Module,
        *,
        prospective: bool,
        mode: str,
        prospective_tau: float | list[float] | Tensor | None,
    ) -> SelectiveScanInputs:
        """Project a complex ``(I,B,T)`` selector source in at least FP32."""
        _, batch, time = source.shape
        dtype = torch.float64 if rqf.log_gamma.dtype == torch.float64 else torch.float32
        with torch.autocast(source.device.type, enabled=False):
            v = (
                torch.cat((source.real, source.imag), dim=0)
                .reshape(2 * self.n_in, -1)
                .to(dtype)
            )
            modulus = self.config.gamma_features == "modulus"
            heads = (None if modulus else self.W_gamma, self.W_delta, self.W_write)
            weights = [h.to(dtype) for h in heads if h is not None]
            if weights:
                sizes = tuple(0 if h is None else h.shape[0] for h in heads)
                g, d, w = _SplitProjection.apply(v, torch.cat(weights), sizes)
            else:
                g = d = w = v.new_empty((0, batch * time))
            if modulus:
                features = torch.linalg.vector_norm(v.reshape(2, self.n_in, -1), dim=0)
                g = self.W_gamma.to(dtype) @ features
            lo = rqf._min_gamma.to(dtype)
            span = rqf._max_gamma.to(dtype) - lo
            log_shift = self.config.gamma_link == "log_shift"
            bias = rqf.log_gamma if log_shift else rqf.gamma
            if self.W_gamma is not None and not log_shift:
                resting = (bias - lo) / span
                if self.config.gamma_edge == "st_clamp":
                    # Historical boundary semantics for existing checkpoints.
                    eps = torch.finfo(dtype).eps
                    interior = resting.clamp(eps, 1 - eps)
                    resting = resting + (interior - resting).detach()
                # Interior mode projects the parameter at construction and
                # after updates, so this logit needs no surrogate derivative.
                bias = torch.logit(resting)
            tau = rqf.tau.to(dtype)
            horizon = (
                torch.ones_like(tau)
                if prospective_tau is None
                else torch.as_tensor(prospective_tau, device=tau.device, dtype=dtype)
                / tau
            )
            horizon = torch.broadcast_to(horizon, tau.shape).contiguous()
            # has_gamma, has_warp, has_write, log_shift, gamma_scale, warp_scale,
            # prospective, gamma-dependent horizon, sigmoid write, maximum, bias,
            # bounded frequency warp.
            # Other horizons use a vector.
            options = (
                self.W_gamma is not None,
                self.W_delta is not None,
                self.W_write is not None,
                log_shift,
                self.config.gamma_max_log_scale,
                self.config.max_log_scale,
                prospective,
                mode == "gamma" and prospective_tau is None,
                self.W_write is not None and self.config.write_link == "sigmoid",
                float(self.config.write_max),
                -math.log(self.config.write_max - 1),
                self.W_delta is not None and self.config.warp_bound == "band",
            )
            theta_minimum, theta_maximum = rqf._theta_bounds()
            return SelectiveScanInputs(
                g.reshape(g.shape[0], batch, time),
                d.reshape(d.shape[0], batch, time),
                w.reshape(w.shape[0], batch, time),
                bias.to(dtype),
                rqf.theta.to(dtype),
                self.warp_groups,
                lo,
                span,
                rqf.dt / tau,
                horizon,
                theta_minimum.to(dtype),
                theta_maximum.to(dtype),
                options,
            )
