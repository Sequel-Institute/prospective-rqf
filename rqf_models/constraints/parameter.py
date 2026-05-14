from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


class Diagonal(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.diag(torch.diag(x))

    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Clamp(nn.Module):
    def __init__(self, min: float = 0.0, max: float = 1.0) -> None:
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            real = torch.clamp(x.real, self.min, self.max)
            imag = torch.clamp(x.imag, self.min, self.max)
            return torch.complex(real, imag)
        return torch.clamp(x, self.min, self.max)

    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SpectralNorm(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        del weight, n_power_iterations, dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.linalg.matrix_norm(x, ord=2).clamp_min(self.eps)
        return x / sigma.clamp_min(1.0)

    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class IdentityPlusSpectralNorm(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:
        super().__init__()
        del weight, n_power_iterations, dim
        self.eps = eps
        self.register_buffer("identity", torch.eye(kwargs.get("dim", 0) or 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.linalg.matrix_norm(x, ord=2).clamp_min(self.eps)
        x_norm = x / sigma.clamp_min(1.0)
        if self.identity.shape != x.shape:
            identity = torch.eye(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        else:
            identity = self.identity.to(device=x.device, dtype=x.dtype)
        return identity + x_norm

    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Orthogonal(nn.Module):
    def __init__(self, weight: torch.Tensor, use_trivialization: bool = True) -> None:
        super().__init__()
        del weight, use_trivialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, r = torch.linalg.qr(x)
        sign = torch.sign(torch.diagonal(r))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return q * sign

    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class BoundedPositiveSigmoid(nn.Module):
    def __init__(self, max_val: float = 1.0) -> None:
        super().__init__()
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_val * torch.sigmoid(x)

    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class WMatrix(nn.Module):
    def __init__(
        self,
        dim: int,
        norm_matrix_constraint: str = "full_rank",
        init_log_W: torch.Tensor | None = None,
        init_log_p: torch.Tensor | None = None,
        init_log_q: torch.Tensor | None = None,
        init_W_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self.norm_matrix_constraint = norm_matrix_constraint

        if norm_matrix_constraint == "full_rank":
            value = (
                init_log_W.clone().detach()
                if init_log_W is not None
                else 2.5 * torch.randn(self.dim, self.dim, **kwargs)
            )
            if value.shape != (self.dim, self.dim):
                raise ValueError(f"init_log_W must have shape {(self.dim, self.dim)}")
            self.log_W = nn.Parameter(value)
        elif norm_matrix_constraint == "one_rank":
            log_p = (
                init_log_p.clone().detach()
                if init_log_p is not None
                else torch.randn(self.dim, **kwargs)
            )
            log_q = (
                init_log_q.clone().detach()
                if init_log_q is not None
                else 0.1 * torch.randn(self.dim, **kwargs)
            )
            if log_p.shape != (self.dim,) or log_q.shape != (self.dim,):
                raise ValueError("init_log_p and init_log_q must have shape (dim,)")
            self.log_p = nn.Parameter(log_p)
            self.log_q = nn.Parameter(log_q)
        elif norm_matrix_constraint == "ones":
            self.register_buffer("log_p", torch.zeros(self.dim, **kwargs))
            self.register_buffer("log_q", torch.zeros(self.dim, **kwargs))
        elif norm_matrix_constraint == "fixed_custom":
            value = (
                init_W_weight.clone().detach()
                if init_W_weight is not None
                else torch.ones(self.dim, self.dim, **kwargs)
            )
            if value.shape != (self.dim, self.dim):
                raise ValueError(f"init_W_weight must have shape {(self.dim, self.dim)}")
            self.register_buffer("W_weight", value)
        else:
            raise ValueError(f"unsupported WMatrix constraint: {norm_matrix_constraint!r}")

    @property
    def p(self) -> torch.Tensor:
        return torch.exp(self.log_p)

    @property
    def q(self) -> torch.Tensor:
        return torch.exp(self.log_q)

    @property
    def W(self) -> torch.Tensor:
        return torch.exp(self.log_W)

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        del x
        if self.norm_matrix_constraint == "full_rank":
            return self.W
        if self.norm_matrix_constraint in {"one_rank", "ones"}:
            return torch.outer(self.p, self.q)
        if self.norm_matrix_constraint == "fixed_custom":
            return self.W_weight
        raise ValueError(f"unsupported WMatrix constraint: {self.norm_matrix_constraint!r}")

    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


def register_diagonal(module: nn.Module, parameter_name: str) -> None:
    parametrize.register_parametrization(module, parameter_name, Diagonal())


def register_clamp(
    module: nn.Module,
    parameter_name: str,
    min: float = 0.0,
    max: float = 1.0,
) -> None:
    parametrize.register_parametrization(module, parameter_name, Clamp(min=min, max=max))


def register_spectral_norm(
    module: nn.Module,
    parameter_name: str,
    n_power_iterations: int = 1,
    dim: int = 0,
    eps: float = 1e-12,
) -> None:
    param = getattr(module, parameter_name)
    parametrize.register_parametrization(
        module,
        parameter_name,
        SpectralNorm(param, n_power_iterations=n_power_iterations, dim=dim, eps=eps),
    )


def register_identity_plus_spectral_norm(
    module: nn.Module,
    parameter_name: str,
    n_power_iterations: int = 1,
    dim: int = 0,
    eps: float = 1e-12,
    **kwargs,
) -> None:
    param = getattr(module, parameter_name)
    parametrize.register_parametrization(
        module,
        parameter_name,
        IdentityPlusSpectralNorm(
            param,
            n_power_iterations=n_power_iterations,
            dim=dim,
            eps=eps,
            **kwargs,
        ),
    )


def register_orthogonal(
    module: nn.Module,
    parameter_name: str,
    use_trivialization: bool = True,
) -> None:
    param = getattr(module, parameter_name)
    parametrize.register_parametrization(
        module,
        parameter_name,
        Orthogonal(param, use_trivialization=use_trivialization),
    )


def register_bounded_positive_sigmoid(
    module: nn.Module,
    parameter_name: str,
    max_val: float = 1.0,
) -> None:
    parametrize.register_parametrization(
        module,
        parameter_name,
        BoundedPositiveSigmoid(max_val=max_val),
    )
