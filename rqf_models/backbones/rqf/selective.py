"""Input-only RQF controllers, matching local-learning's parameterization."""

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
    write_control: bool = False
    frequency_warp: str = "none"
    num_groups: int = 4
    max_log_scale: float = math.log(1.25)
    input_source: str = "layer_output"

    def __post_init__(self) -> None:
        for name, choices in (
            ("gamma_link", {"sigmoid", "log_shift"}),
            ("gamma_features", {"real_imag", "modulus"}),
            ("frequency_warp", {"none", "global", "grouped"}),
            ("input_source", {"layer_input", "layer_output"}),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"selective.{name} must be one of {sorted(choices)}")
        if type(self.gamma_control) is not bool or type(self.write_control) is not bool:
            raise ValueError("selective controller switches must be booleans")
        if type(self.num_groups) is not int or self.num_groups < 1:
            raise ValueError("selective.num_groups must be a positive integer")
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
        self, n_in: int, rqf: nn.Module, config: SelectiveRQFConfig, learn_write: bool
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
            rqf.clamp_parameters()
        self.W_gamma = (
            nn.Parameter(
                torch.zeros(
                    n_out, n_in if config.gamma_features == "modulus" else 2 * n_in
                ),
                requires_grad=rqf.log_gamma.requires_grad,
            )
            if config.gamma_control
            else None
        )
        self.W_write = (
            nn.Parameter(torch.zeros(n_out, 2 * n_in), requires_grad=learn_write)
            if config.write_control
            else None
        )
        self.W_delta = (
            nn.Parameter(
                torch.zeros(groups, 2 * n_in), requires_grad=rqf.log_theta.requires_grad
            )
            if groups
            else None
        )
        ranks = rqf.log_theta.detach().argsort(stable=True).argsort()
        self.register_buffer("warp_groups", ranks * groups // n_out if groups else None)

    def get_extra_state(self) -> dict[str, Any]:
        return asdict(self.config)

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if SelectiveRQFConfig(**state) != self.config:
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
                # Straight-through clamp: the logit is evaluated strictly inside
                # (0, 1), but the gradient follows the unclamped resting gamma.
                # clamp_parameters projects log_gamma exactly onto the band
                # edges, where a plain clamp would zero d gamma / d log_gamma
                # forever; this keeps the static model's derivative there.
                eps = torch.finfo(dtype).eps
                resting = (bias - lo) / span
                interior = resting.clamp(eps, 1 - eps)
                bias = torch.logit(resting + (interior - resting).detach())
            tau = rqf.tau.to(dtype)
            horizon = (
                torch.ones_like(tau)
                if prospective_tau is None
                else torch.as_tensor(prospective_tau, device=tau.device, dtype=dtype)
                / tau
            )
            horizon = torch.broadcast_to(horizon, tau.shape).contiguous()
            # has_gamma, has_warp, has_write, log_shift, gamma_scale, warp_scale,
            # prospective, gamma-dependent horizon. Other horizons use a vector.
            options = (
                self.W_gamma is not None,
                self.W_delta is not None,
                self.W_write is not None,
                log_shift,
                self.config.gamma_max_log_scale,
                self.config.max_log_scale,
                prospective,
                mode == "gamma" and prospective_tau is None,
            )
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
                options,
            )
