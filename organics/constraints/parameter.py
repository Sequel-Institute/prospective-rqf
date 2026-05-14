"""Matrix parametrization utilities for constraining tensors.

This module provides a collection of lightweight parametrization classes that
can be used to impose structural constraints on parameter tensors (e.g.,
symmetry, diagonal structure, spectral norm constraints, low-rank structure).

Each class follows the common pattern used by PyTorch parametrizations:
- The forward() method maps an unconstrained tensor to a constrained one.
- The right_inverse() method maps a constrained tensor back to an unconstrained
  representation suitable for assignment when users set the value of a parametrized
  tensor directly. In many cases, the right-inverse is the identity mapping or a
  simple projection that ensures consistency.

Available parametrizations:
    Unconstrained: Identity transformation (no constraints).
    Symmetric: Projects matrices to symmetric space using upper triangular method.
    Diagonal: Projects matrices to diagonal space, zeroing off-diagonal elements.
    Clamp: Elementwise clamping to [min, max] interval (supports complex tensors).
    AbsoluteValue: Elementwise absolute value (supports complex tensors).
    SpectralNorm: Constrains largest singular value to be ≤ 1.
    IdentityPlusSpectralNorm: Computes I + SN(X) for well-conditioned matrices.
    LowRankSVD: Low-rank projection using truncated SVD.
    LowRankFactorized: Low-rank factorization with learnable factors.
    WMatrix: Positive matrix parametrization with multiple constraint types.
    LDSMatrix: Lyapunov Diagonally Stable matrix parametrization.

Note:
    These parametrizations are designed to avoid changing observable behavior.
    Unless explicitly stated otherwise, shapes are assumed to be 2D matrices.
    Batched inputs may not be supported for some parametrizations (see detailed
    docstrings for caveats).
"""

import math
import typing

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.utils.parametrizations import _SpectralNorm


class Unconstrained(nn.Module):
    """Identity parametrization that returns input tensors unchanged.

    This parametrization applies no constraints and passes tensors through
    unchanged in both forward and right_inverse directions. Useful as a
    baseline or when temporarily disabling constraints.
    """

    def __init__(self) -> None:
        """Initialize unconstrained parametrization."""
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return input tensor unchanged.

        Args:
            X: Input tensor of any shape.

        Returns:
            The input tensor X unchanged.
        """
        return X

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Return input tensor unchanged.

        Args:
            X: Input tensor of any shape.

        Returns:
            The input tensor X unchanged.
        """
        return X


class Symmetric(nn.Module):
    """Projects a matrix to the space of symmetric matrices.

    The forward method returns a symmetric matrix by taking the upper triangular
    part of the input (including diagonal) and adding the transpose of the
    strictly upper triangular part. This ensures the result satisfies A = A^T.

    Supports 2D square matrices and batched inputs where the last two dimensions
    represent the matrix dimensions.
    """

    def __init__(self) -> None:
        """Initialize symmetric parametrization."""
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Project matrix to symmetric space.

        Args:
            X: Input tensor of shape (..., n, n) where the last two dimensions
                represent square matrices.

        Returns:
            Symmetric matrix of the same shape as input, where result = result^T.
        """
        return X.triu() + X.triu(1).transpose(-1, -2)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Map from symmetric space back to unconstrained space.

        When assigning a new tensor we expect it to already be symmetric, but
        if not, this returns it unchanged. Applying forward() to the result
        will yield a symmetric matrix.

        Args:
            X: Input tensor, typically a symmetric matrix.

        Returns:
            The input tensor X unchanged.
        """
        return X


class Diagonal(nn.Module):
    """Projects a matrix onto a diagonal matrix preserving the original diagonal.

    The forward method extracts the diagonal elements of the input matrix and
    constructs a new diagonal matrix from them, zeroing out all off-diagonal
    elements. For rectangular matrices of shape (n, m), this results in a
    square matrix of size min(n, m).
    """

    def __init__(self) -> None:
        """Initialize diagonal parametrization."""
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Project matrix to diagonal space.

        Args:
            X: Input matrix tensor of shape (n, m).

        Returns:
            Diagonal matrix of shape (min(n,m), min(n,m)) containing only the
            diagonal elements of X, with all off-diagonal elements set to zero.
        """
        return torch.diag(torch.diag(X))

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Map from diagonal space back to unconstrained space.

        Args:
            X: Input tensor, typically a diagonal matrix.

        Returns:
            The input tensor X unchanged.
        """
        return X


class Clamp(nn.Module):
    """Elementwise clamp to the interval [min, max].

    Applies elementwise clamping to ensure all tensor values fall within the
    specified bounds. Supports both real and complex tensors, applying the
    bounds separately to real and imaginary components for complex inputs.

    Attributes:
        min: Lower bound for clamping.
        max: Upper bound for clamping.
    """

    def __init__(self, min: float = 0.0, max: float = 1.0) -> None:
        """Initialize clamp parametrization.

        Args:
            min: Lower bound for clamping. Defaults to 0.0.
            max: Upper bound for clamping. Defaults to 1.0.
        """
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply elementwise clamping to the input tensor.

        Args:
            X: Input tensor of any shape, real or complex valued.

        Returns:
            Tensor with all elements clamped to [min, max]. For complex tensors,
            real and imaginary parts are clamped separately.
        """
        # torch.clamp does not support complex tensors; clamp real/imag parts separately
        if torch.is_complex(X):
            real = torch.clamp(X.real, self.min, self.max)
            imag = torch.clamp(X.imag, self.min, self.max)
            return torch.complex(real, imag)
        return torch.clamp(X, self.min, self.max)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Map from clamped space back to unconstrained space.

        Args:
            X: Input tensor, typically with values in [min, max].

        Returns:
            The input tensor X unchanged.
        """
        return X


class AbsoluteValue(nn.Module):
    """Elementwise absolute value mapping ensuring all values are non-negative.

    Applies elementwise absolute value operation to the input tensor. For real
    tensors, this ensures all values are non-negative. For complex tensors,
    returns the magnitude as a complex tensor with zero imaginary part to
    preserve the complex dtype.
    """

    def __init__(self) -> None:
        """Initialize absolute value parametrization."""
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply elementwise absolute value to the input tensor.

        Args:
            X: Input tensor of any shape, real or complex valued.

        Returns:
            Tensor with absolute values. For real tensors, returns |X|. For
            complex tensors, returns complex(|X|, 0) to preserve dtype.
        """
        # Preserve dtype for complex tensors by returning magnitude with zero imaginary part
        if torch.is_complex(X):
            mag = torch.abs(X)
            return torch.complex(mag, torch.zeros_like(mag))
        return torch.abs(X)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Map from absolute value space back to unconstrained space.

        Args:
            X: Input tensor, typically with non-negative values.

        Returns:
            The input tensor X unchanged.
        """
        return X


class SpectralNorm(nn.Module):
    """Applies spectral normalization to constrain the largest singular value.

    This is a thin wrapper around PyTorch's internal _SpectralNorm used by the
    official parametrizations. Spectral normalization divides the weight matrix
    by its largest singular value, constraining the spectral norm to be at most 1.
    This helps stabilize training of neural networks, particularly GANs.

    Attributes:
        n_power_iterations: Number of power iterations for approximating the largest singular value.
        dim: Dimension corresponding to output features.
        eps: Small value for numerical stability.
        spectral_norm_object: Internal PyTorch spectral norm implementation.
    """

    def __init__(
        self,
        weight: nn.Parameter,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:
        """Initialize spectral norm parametrization.

        Args:
            weight: Reference weight used to initialize the internal spectral norm state.
            n_power_iterations: Number of power iterations to approximate the largest
                singular value. More iterations give better approximation but are slower.
                Defaults to 1.
            dim: Dimension corresponding to output features. Defaults to 0.
            eps: Small value for numerical stability. Defaults to 1e-12.
        """
        super().__init__()
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps
        self.spectral_norm_object = _SpectralNorm(weight, n_power_iterations, dim, eps)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply spectral normalization to the input matrix.

        Args:
            X: Input weight matrix to be normalized.

        Returns:
            Matrix with spectral norm (largest singular value) constrained to be ≤ 1.
        """
        return self.spectral_norm_object(X)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Map from spectrally normalized space back to unconstrained space.

        Since spectral normalization is idempotent (applying it twice gives the
        same result), the right inverse is the identity mapping.

        Args:
            X: Input tensor, typically already spectrally normalized.

        Returns:
            The input tensor X unchanged.
        """
        return X


class IdentityPlusSpectralNorm(nn.Module):
    """Identity plus spectrally-normalized component: I + SN(X).

    This parametrization computes I + SN(X) where I is an identity matrix and
    SN(X) is the spectrally normalized input. The result has bounded spectral
    norm: spectral_norm(I + A) ≤ 1 + spectral_norm(A) ≤ 2.

    This transformation is useful for ensuring matrices are well-conditioned
    while maintaining spectral control.

    Note:
        This parametrization requires square matrices since it adds an identity
        matrix. Using it with rectangular matrices will cause dimension mismatch errors.

    Attributes:
        n_power_iterations: Number of power iterations for spectral norm approximation.
        dim: Dimension corresponding to output features.
        eps: Small value for numerical stability.
        spectral_norm_object: Internal PyTorch spectral norm implementation.
        identity: Identity matrix buffer matching the weight dimensions.
    """

    def __init__(
        self,
        weight: nn.Parameter,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:
        """Initialize identity plus spectral norm parametrization.

        Args:
            weight: Reference weight used to initialize the internal spectral norm state.
                Must be a square matrix.
            n_power_iterations: Number of power iterations to approximate the largest
                singular value. Defaults to 1.
            dim: Dimension corresponding to output features. Defaults to 0.
            eps: Small value for numerical stability. Defaults to 1e-12.
            **kwargs: Additional keyword arguments for tensor creation (device, dtype).
        """
        super().__init__()
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps
        self.spectral_norm_object = _SpectralNorm(weight, n_power_iterations, dim, eps)
        # Use kwargs if provided, otherwise fall back to weight's device/dtype
        device = kwargs.get("device", weight.device)
        dtype = kwargs.get("dtype", weight.dtype)
        self.register_buffer(
            "identity",
            torch.eye(weight.shape[-1], device=device, dtype=dtype),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply identity plus spectral normalization.

        Args:
            X: Input square matrix to be transformed.

        Returns:
            Result of I + SN(X), where I is identity and SN(X) is spectrally normalized X.

        Raises:
            RuntimeError: If X is not square (due to identity matrix addition).
        """
        return self.identity + self.spectral_norm_object(X)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Map from I+SN space back to unconstrained space.

        Args:
            X: Input tensor, typically of the form I + SN(Y) for some Y.

        Returns:
            The input tensor X unchanged.
        """
        return X


class LowRankSVD(nn.Module):
    """Low-rank projection using truncated singular value decomposition.

    Projects an input matrix to a matrix of rank at most `rank` by computing
    its SVD and keeping only the top-k singular values and corresponding
    singular vectors. This provides the best rank-k approximation in the
    Frobenius norm sense.

    Attributes:
        rank: Target rank for the approximation. Must be a positive integer.
    """

    def __init__(self, rank: int) -> None:
        """Initialize low-rank SVD parametrization.

        Args:
            rank: Desired target rank. Must be a positive integer.

        Raises:
            ValueError: If rank is not a positive integer.
        """
        super().__init__()
        if rank < 1:
            raise ValueError(f"Rank must be a positive integer, but got {rank}")
        self.rank = rank

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the low-rank approximation of X via truncated SVD.

        Performs SVD on the input matrix and reconstructs using only the top
        `rank` singular values. This gives the optimal rank-k approximation
        in terms of minimizing the Frobenius norm of the reconstruction error.

        Args:
            X: Input matrix tensor of shape (m, n).

        Returns:
            Low-rank approximation of X with rank at most `self.rank`.

        Raises:
            ValueError: If input tensor has fewer than 2 dimensions.

        Note:
            This implementation assumes 2D input matrices. Batched inputs with
            more than 2 dimensions are not currently supported.
        """
        # Ensure the input is at least a 2D tensor
        if X.dim() < 2:
            raise ValueError(
                f"LowRank parametrization can only be applied to matrices, "
                f"but got a tensor of dimension {X.dim()}"
            )

        # 1. Decompose the matrix using SVD
        # full_matrices=False is more efficient
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)

        # 2. Truncate to the desired rank
        # U_k will have shape (m, k), S_k will be (k,), Vh_k will be (k, n)
        U_k = U[:, : self.rank]
        S_k = S[: self.rank]
        Vh_k = Vh[: self.rank, :]

        # 3. Reconstruct the low-rank matrix
        # We can use broadcasting (U_k * S_k) for efficiency
        X_low_rank = (U_k * S_k) @ Vh_k

        return X_low_rank

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Map from low-rank space back to unconstrained space.

        Args:
            X: Input tensor, typically a low-rank matrix.

        Returns:
            The input tensor X unchanged.
        """
        return X


class LowRankFactorized(nn.Module):
    """Low-rank factorization via learned factors.

    Replaces a full matrix of shape (n, m) with two smaller learnable parameters
    factor1 and factor2 of shapes (n, rank) and (rank, m) respectively. The
    effective matrix is their matrix product factor1 @ factor2.

    This reduces the number of parameters from n*m to (n+m)*rank, which can be
    beneficial when rank << min(n, m). The factorization is learned end-to-end
    during training.

    Attributes:
        factor1: Left factor matrix of shape (n, rank).
        factor2: Right factor matrix of shape (rank, m).
    """

    def __init__(self, weight: nn.Parameter, rank: int, **kwargs) -> None:
        """Initialize low-rank factorized parametrization.

        Creates two factor matrices that when multiplied together approximate
        the original weight matrix. The factors are initialized using Kaiming
        uniform initialization.

        Args:
            weight: Reference parameter used for shape validation and initialization.
                Must be a 2D tensor.
            rank: Desired internal rank for the factorization. Must satisfy
                1 <= rank <= min(n, m) where (n, m) is the weight shape.
            **kwargs: Additional keyword arguments for tensor creation (device, dtype).

        Raises:
            ValueError: If weight is not 2D or rank is outside valid range.
        """
        super().__init__()
        original_shape = weight.shape
        if len(original_shape) != 2:
            raise ValueError(
                f"LowRankFactorized can only be applied to 2D matrices, "
                f"but got shape {original_shape}"
            )

        n, m = original_shape
        if rank < 1 or rank > min(n, m):
            raise ValueError(f"Rank must be between 1 and min(n, m), but got {rank}")

        # 1. Create the new, smaller parameters
        # These are the tensors that will be added to the state_dict and trained.
        self.factor1 = nn.Parameter(torch.empty(n, rank, **kwargs))
        self.factor2 = nn.Parameter(torch.empty(rank, m, **kwargs))

        # 2. Initialize the new parameters
        # Using a standard initialization similar to nn.Linear
        nn.init.kaiming_uniform_(self.factor1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.factor2, a=math.sqrt(5))

    @typing.override
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Reconstruct the full matrix from the low-rank factors.

        The input X is ignored as required by the parametrization API. The
        actual matrix is computed as the product of the two learned factors.

        Args:
            X: Original tensor under parametrization (values ignored).

        Returns:
            Matrix product factor1 @ factor2 of the same shape as the original weight.
        """
        return self.factor1 @ self.factor2

    def right_inverse(self, Y: torch.Tensor) -> torch.Tensor:
        """Update internal factors to approximate an assigned matrix Y.

        This method is called when a new value is assigned to the parametrized
        tensor (e.g., module.weight = new_tensor). It computes the best rank-k
        approximation of Y using SVD and updates the factors accordingly.

        The singular values are distributed equally between the factors by
        taking their square root, which helps balance the magnitudes.

        Args:
            Y: New matrix value to approximate with the factorization.

        Returns:
            The input tensor Y unchanged.

        Note:
            The parameter is named Y (not X) to indicate it's an externally
            assigned value, following the convention in the reference implementation.
        """
        # Decompose the input matrix Y to find the best low-rank approximation
        U, S, Vh = torch.linalg.svd(Y, full_matrices=False)

        # Truncate to the desired rank
        rank = self.factor1.shape[1]
        U_k = U[:, :rank]
        S_k = S[:rank]
        Vh_k = Vh[:rank, :]

        # Update our factors to represent this new matrix.
        # We distribute the singular values (S) to balance the magnitudes of the factors.
        sqrt_S_k = torch.sqrt(S_k)
        self.factor1.data.copy_(U_k * sqrt_S_k)
        self.factor2.data.copy_(sqrt_S_k.unsqueeze(-1) * Vh_k)

        return Y


class Orthogonal(nn.Module):
    """Orthogonal matrix parametrization wrapper around PyTorch's implementation.

    Constrains a matrix to be orthogonal, meaning its columns (or rows) are
    orthonormal vectors. This is useful for maintaining stable gradients and
    preventing exploding/vanishing gradients in recurrent networks.

    For a matrix W to be orthogonal:
        W^T @ W = I (identity matrix)

    This is a thin wrapper around PyTorch's orthogonal parametrization to maintain
    API consistency with other constraint classes in this module.

    Attributes:
        use_trivialization: Whether to use trivialization for optimization.
            When True, uses a more numerically stable form.
        _orthogonal: Internal PyTorch orthogonal parametrization.
    """

    def __init__(self, weight: nn.Parameter, use_trivialization: bool = True) -> None:
        """Initialize orthogonal parametrization.

        Args:
            weight: Reference weight used to initialize the parametrization.
            use_trivialization: Use trivialized form for better optimization.
                Defaults to True.
        """
        super().__init__()
        self.use_trivialization = use_trivialization
        # Import PyTorch's internal _Orthogonal class
        from torch.nn.utils.parametrizations import _Orthogonal

        self._orthogonal = _Orthogonal(
            weight, orthogonal_map=None, use_trivialization=use_trivialization
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply QR decomposition to enforce orthogonality.

        Args:
            X: Input matrix of shape (..., m, n).

        Returns:
            Orthogonal matrix Q from QR decomposition of X.
        """
        return self._orthogonal(X)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Compute right inverse.

        Args:
            X: Orthogonal matrix.

        Returns:
            The right inverse from PyTorch's implementation.
        """
        return self._orthogonal.right_inverse(X)


class BoundedPositiveSigmoid(nn.Module):
    """Bounded positive parametrization using sigmoid transformation.

    Maps unconstrained parameters to the interval (0, max_val) using sigmoid:
        output = max_val * sigmoid(input)

    This is useful for time constants and other parameters that must be
    positive and bounded.

    Attributes:
        max_val: Maximum value for the output (exclusive upper bound).
    """

    def __init__(self, max_val: float = 1.0) -> None:
        """Initialize bounded positive parametrization.

        Args:
            max_val: Maximum value for the output. The actual output will be
                in the range (0, max_val). Defaults to 1.0.
        """
        super().__init__()
        self.max_val = max_val

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid transformation scaled by max_val.

        Args:
            X: Unconstrained input tensor.

        Returns:
            Tensor with values in (0, max_val).
        """
        return self.max_val * torch.sigmoid(X)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Map from bounded positive space back to unconstrained space.

        Returns input unchanged - assumes parameter is initialized with raw values
        that should be passed through forward() directly (e.g., randn for sigmoid).

        Args:
            X: Unconstrained tensor (raw parameter values).

        Returns:
            The input tensor X unchanged.
        """
        return X


class WMatrix(nn.Module):
    """Positive matrix parametrization supporting multiple constraint types.

    This parametrization constructs a positive matrix W based on the
    specified constraint type. Four constraint modes are supported:
    - full_rank: W = exp(log_W), a full-rank positive matrix
    - one_rank: W = p ⊗ q, a rank-1 positive matrix (outer product)
    - ones: W = 1 ⊗ 1, a matrix of all ones
    - fixed_custom: W is a fixed custom positive matrix

    Attributes:
        dim: Dimension of the square matrix.
        norm_matrix_constraint: One of {'full_rank', 'one_rank', 'ones', 'fixed_custom'}.
        log_W: Parameter for full_rank constraint (learnable log of W).
        log_p: Parameter for one_rank constraint (learnable log of p vector).
        log_q: Parameter for one_rank constraint (learnable log of q vector).
        W_weight: Buffer for fixed_custom constraint (non-learnable W matrix).
    """

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
        """Initialize W matrix parametrization.

        Args:
            dim: Dimension of the square matrix. Must be a positive integer.
            norm_matrix_constraint: One of {'full_rank', 'one_rank', 'ones', 'fixed_custom'}.
                Defaults to 'full_rank'.
            init_log_W: Initial tensor for log_W parameter (used only with 'full_rank' constraint).
                If None, defaults to 2.5 * torch.randn(dim, dim).
                Must have shape (dim, dim).
            init_log_p: Initial tensor for log_p parameter (used only with 'one_rank' constraint).
                If None, defaults to torch.randn(dim) for 'one_rank', or torch.zeros(dim) for 'ones'.
                Must have shape (dim,).
            init_log_q: Initial tensor for log_q parameter (used only with 'one_rank' constraint).
                If None, defaults to 0.1 * torch.randn(dim) for 'one_rank', or torch.zeros(dim) for 'ones'.
                Must have shape (dim,).
            init_W_weight: Initial tensor for W_weight parameter (used only with 'fixed_custom' constraint).
                If None, defaults to torch.ones(dim, dim).
                Must have shape (dim, dim).
            **kwargs: Additional keyword arguments for tensor creation (device, dtype).

        Raises:
            ValueError: If dim is not an integer.
            ValueError: If dim is not positive.
            ValueError: If norm_matrix_constraint is not one of {'full_rank', 'one_rank', 'ones', 'fixed_custom'}.
            ValueError: If any provided initial tensor has incorrect shape.
        """
        super().__init__()
        if not isinstance(dim, int):
            raise TypeError(f"WMatrix requires dim to be an integer, got {type(dim)}")
        if dim <= 0:
            raise ValueError(f"WMatrix requires dim to be a positive integer, got {dim}")
        self.dim = dim
        self.norm_matrix_constraint = norm_matrix_constraint

        # Initialize norm matrix constraint-specific parameters
        match self.norm_matrix_constraint:
            case "full_rank":
                if init_log_W is not None:
                    if init_log_W.shape != (self.dim, self.dim):
                        raise ValueError(
                            f"init_log_W must have shape ({self.dim}, {self.dim}), got {tuple(init_log_W.shape)}"
                        )
                    self.log_W = nn.Parameter(init_log_W.clone().detach())
                else:
                    self.log_W = nn.Parameter(2.5 * torch.randn(self.dim, self.dim, **kwargs))

            case "one_rank":
                if init_log_p is not None:
                    if init_log_p.shape != (self.dim,):
                        raise ValueError(
                            f"init_log_p must have shape ({self.dim},), got {tuple(init_log_p.shape)}"
                        )
                    self.log_p = nn.Parameter(init_log_p.clone().detach())
                else:
                    self.log_p = nn.Parameter(torch.randn(self.dim, **kwargs))

                if init_log_q is not None:
                    if init_log_q.shape != (self.dim,):
                        raise ValueError(
                            f"init_log_q must have shape ({self.dim},), got {tuple(init_log_q.shape)}"
                        )
                    self.log_q = nn.Parameter(init_log_q.clone().detach())
                else:
                    self.log_q = nn.Parameter(0.1 * torch.randn(self.dim, **kwargs))

            case "ones":
                # For 'ones' constraint: p and q are ones vectors (log_p = log_q = 0)
                # These are not trainable parameters
                self.register_buffer("log_p", torch.zeros(self.dim, **kwargs))
                self.register_buffer("log_q", torch.zeros(self.dim, **kwargs))

            case "fixed_custom":
                if init_W_weight is not None:
                    if init_W_weight.shape != (self.dim, self.dim):
                        raise ValueError(
                            f"init_W_weight must have shape ({self.dim}, {self.dim}), got {tuple(init_W_weight.shape)}"
                        )
                    self.register_buffer("W_weight", init_W_weight.clone().detach())
                else:
                    self.register_buffer("W_weight", torch.ones(self.dim, self.dim, **kwargs))

            case _:
                raise ValueError(
                    f"Unsupported norm matrix constraint '{self.norm_matrix_constraint}'. "
                    f"Supported constraints: {{'full_rank', 'one_rank', 'ones', 'fixed_custom'}}"
                )

    @property
    def p(self) -> torch.Tensor:
        """Return exp(log_p) for one_rank and ones constraints.

        Returns:
            Vector p of shape (dim,) with positive elements.
        """
        return torch.exp(self.log_p)

    @property
    def q(self) -> torch.Tensor:
        """Return exp(log_q) for one_rank and ones constraints.

        Returns:
            Vector q of shape (dim,) with positive elements.
        """
        return torch.exp(self.log_q)

    @property
    def W(self) -> torch.Tensor:
        """Return the W matrix.

        Returns:
            Positive matrix W of shape (dim, dim).
        """
        return torch.exp(self.log_W)

    @typing.override
    def forward(self, X: torch.Tensor | None = None) -> torch.Tensor:
        """Return the W matrix based on the constraint type.

        The input X is ignored as required by the parametrization API when
        this is used as a standalone module.

        Args:
            X: Input tensor (ignored).

        Returns:
            Positive matrix W of shape (dim, dim) based on the constraint type:
            - full_rank: W = exp(log_W)
            - one_rank: W = p ⊗ q where p and q are exponentials of learned parameters
            - ones: W = 1 ⊗ 1 where 1 is a vector of ones
            - fixed_custom: W is a fixed custom matrix
        """

        match self.norm_matrix_constraint:
            case "full_rank":
                return self.W

            case "one_rank" | "ones":
                return torch.outer(self.p, self.q)

            case "fixed_custom":
                return self.W_weight

            case _:
                raise ValueError(
                    f"Unsupported norm matrix constraint '{self.norm_matrix_constraint}'"
                )

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Map from W matrix space back to unconstrained space.

        Args:
            X: Input tensor, typically a positive matrix.

        Returns:
            The input tensor X unchanged.

        Note:
            TODO: Implement the right inverse of the WMatrix parametrization.
        """
        return X


class LDSMatrix(nn.Module):
    """Lyapunov Diagonally Stable (LDS) matrix parametrization.

    This parametrization constructs a matrix A that is provably Lyapunov
    Diagonally Stable (LDS). The construction ensures that there exists a
    diagonal positive definite matrix D such that A^T D + D A is negative
    definite, which guarantees asymptotic stability of the dynamical system
    dx/dt = Ax.

    Reference: https://arxiv.org/pdf/2409.18946

    The matrix is constructed as:
        A = diag(1/t) - W @ diag(u) @ diag(v + W @ u)^{-1}

    where t, u, v are positive vectors and W is a positive matrix.

    Attributes:
        dim: Dimension of the square matrix.
        norm_matrix_constraint: Type of W matrix constraint.
        raw_t: Learnable parameter for t vector (t = exp(-|raw_t|)).
        raw_u: Learnable parameter for u vector (u = raw_u^2).
        raw_v: Learnable parameter for v vector (v = raw_v^2).
        _W_base: WMatrix module for constructing the W matrix.
    """

    def __init__(
        self,
        dim: int,
        norm_matrix_constraint: str = "full_rank",
        init_raw_t: torch.Tensor | None = None,
        init_raw_u: torch.Tensor | None = None,
        init_raw_v: torch.Tensor | None = None,
        init_log_W: torch.Tensor | None = None,
        init_log_p: torch.Tensor | None = None,
        init_log_q: torch.Tensor | None = None,
        init_W_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Initialize LDS matrix parametrization.

        Args:
            dim: Dimension of the square matrix. Must be a positive integer.
            norm_matrix_constraint: One of {'full_rank', 'one_rank', 'ones', 'fixed_custom'}.
                Defaults to 'full_rank'.
            init_raw_t: Initial tensor for raw_t parameter. If None, defaults to 0.5 * torch.rand(dim).
                Must have shape (dim,).
            init_raw_u: Initial tensor for raw_u parameter. If None, defaults to torch.rand(dim).
                Must have shape (dim,).
            init_raw_v: Initial tensor for raw_v parameter. If None, defaults to torch.rand(dim).
                Must have shape (dim,).
            init_log_W: Initial tensor for log_W parameter (used only with 'full_rank' constraint).
                If None, defaults to 2.5 * torch.randn(dim, dim).
                Must have shape (dim, dim).
            init_log_p: Initial tensor for log_p parameter (used only with 'one_rank' constraint).
                If None, defaults to torch.randn(dim).
                Must have shape (dim,).
            init_log_q: Initial tensor for log_q parameter (used only with 'one_rank' constraint).
                If None, defaults to 0.1 * torch.randn(dim).
                Must have shape (dim,).
            init_W_weight: Initial tensor for W_weight parameter (used only with 'fixed_custom' constraint).
                If None, defaults to torch.ones(dim, dim).
                Must have shape (dim, dim).
            **kwargs: Additional keyword arguments for tensor creation (device, dtype).

        Raises:
            ValueError: If dim is not an integer.
            ValueError: If dim is not positive.
            ValueError: If norm_matrix_constraint is not one of {'full_rank', 'one_rank', 'ones', 'fixed_custom'}.
            ValueError: If any provided initial tensor has incorrect shape.
        """
        super().__init__()
        if not isinstance(dim, int):
            raise ValueError(f"LDSMatrix requires dim to be an integer, got {type(dim)}")
        if dim <= 0:
            raise ValueError(f"LDSMatrix requires dim to be a positive integer, got {dim}")
        self.dim = dim

        self.norm_matrix_constraint = norm_matrix_constraint
        assert self.norm_matrix_constraint in [
            "full_rank",
            "one_rank",
            "ones",
            "fixed_custom",
        ]

        # Initialize raw_t parameter with validation
        if init_raw_t is not None:
            if init_raw_t.shape != (self.dim,):
                raise ValueError(
                    f"init_raw_t must have shape ({self.dim},), got {tuple(init_raw_t.shape)}"
                )
            self.raw_t = nn.Parameter(init_raw_t.clone().detach())
        else:
            self.raw_t = nn.Parameter(0.5 * torch.rand(self.dim, **kwargs))

        # Initialize raw_u parameter with validation
        if init_raw_u is not None:
            if init_raw_u.shape != (self.dim,):
                raise ValueError(
                    f"init_raw_u must have shape ({self.dim},), got {tuple(init_raw_u.shape)}"
                )
            self.raw_u = nn.Parameter(init_raw_u.clone().detach())
        else:
            self.raw_u = nn.Parameter(torch.rand(self.dim, **kwargs))

        # Initialize raw_v parameter with validation
        if init_raw_v is not None:
            if init_raw_v.shape != (self.dim,):
                raise ValueError(
                    f"init_raw_v must have shape ({self.dim},), got {tuple(init_raw_v.shape)}"
                )
            self.raw_v = nn.Parameter(init_raw_v.clone().detach())
        else:
            self.raw_v = nn.Parameter(torch.rand(self.dim, **kwargs))

        # Register W matrix as a parameter and apply parametrization
        self._W_base = WMatrix(
            dim=self.dim,
            norm_matrix_constraint=norm_matrix_constraint,
            init_log_W=init_log_W,
            init_log_p=init_log_p,
            init_log_q=init_log_q,
            init_W_weight=init_W_weight,
            **kwargs,
        )

    @property
    def t(self) -> torch.Tensor:
        """Return positive vector t.

        Returns:
            Vector t = exp(-|raw_t|) of shape (dim,) with elements in (0, 1].
        """
        return torch.exp(-torch.abs(self.raw_t))

    @property
    def u(self) -> torch.Tensor:
        """Return positive vector u.

        Returns:
            Vector u = raw_u^2 of shape (dim,) with non-negative elements.
        """
        return self.raw_u**2

    @property
    def v(self) -> torch.Tensor:
        """Return positive vector v.

        Returns:
            Vector v = raw_v^2 of shape (dim,) with non-negative elements.
        """
        return self.raw_v**2

    @property
    def W(self) -> torch.Tensor:
        """Return the W matrix.

        Returns:
            Positive matrix W of shape (dim, dim).
        """
        return self._W_base()

    @property
    def z(self) -> torch.Tensor:
        """Return the square root of u.

        Returns:
            Vector sqrt(u) of shape (dim,).
        """
        return torch.sqrt(self.u)

    @property
    def sigma(self) -> torch.Tensor:
        """Return the square root of v.

        Returns:
            Vector sqrt(v) of shape (dim,).
        """
        return torch.sqrt(self.v)

    @typing.override
    def forward(self, X: torch.Tensor | None = None) -> torch.Tensor:
        """Return a Lyapunov Diagonally Stable (LDS) matrix.

        The input X is ignored as required by the parametrization API.

        Args:
            X: Input tensor (ignored).

        Returns:
            LDS matrix A of shape (dim, dim) constructed as:
                A = diag(1/t) - W @ diag(u) @ diag(v + W @ u)^{-1}

            For full_rank and fixed_custom constraints, W is used directly.
            For one_rank and ones constraints, an efficient rank-1 computation
            is used: W @ u = p * (q^T @ u).
        """
        eps = 1e-8
        # Positive margins ensuring strict dominance per row
        t = self.t  # (dim,)
        u = self.u  # (dim,)
        v = self.v  # (dim,)

        match self.norm_matrix_constraint:
            case "full_rank" | "fixed_custom":
                W = self.W  # (dim, dim)
                W_u = W @ u
                scaling_vector = u / (v + W_u + eps)
                second_term = W * scaling_vector
                lds_mat = -second_term
                lds_mat.diagonal().add_(1.0 / t)
            case "one_rank" | "ones":
                p = self._W_base.p
                q = self._W_base.q
                q_dot_u = torch.dot(q, u)
                W_u = p * q_dot_u
                scaling_vector = u / (v + W_u + eps)
                second_term = torch.outer(p, q * scaling_vector)
                lds_mat = -second_term
                lds_mat.diagonal().add_(1.0 / t)
            case _:
                raise ValueError(
                    f"Unsupported norm matrix constraint '{self.norm_matrix_constraint}'"
                )
        return lds_mat

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Map from LDS matrix space back to unconstrained space.

        Args:
            X: Input tensor, typically an LDS matrix.

        Returns:
            The input tensor X unchanged.

        Note:
            TODO: Implement the right inverse of the LyapunovDiagonallyStableMatrix parametrization.
        """
        return X


# Registration helper functions for convenient usage
def register_unconstrained(module: nn.Module, parameter_name: str) -> None:
    """Register an unconstrained parametrization on a module parameter.

    Args:
        module: The module containing the parameter to constrain.
        parameter_name: Name of the parameter to apply unconstrained parametrization to.
    """
    constraint = Unconstrained()
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_symmetric(module: nn.Module, parameter_name: str) -> None:
    """Register a symmetric constraint on a module parameter.

    Args:
        module: The module containing the parameter to constrain.
        parameter_name: Name of the parameter to make symmetric.
    """
    constraint = Symmetric()
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_diagonal(module: nn.Module, parameter_name: str) -> None:
    """Register a diagonal constraint on a module parameter.

    Args:
        module: The module containing the parameter to constrain.
        parameter_name: Name of the parameter to make diagonal.
    """
    constraint = Diagonal()
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_clamp(
    module: nn.Module, parameter_name: str, min: float = 0.0, max: float = 1.0
) -> None:
    """Register a clamp constraint on a module parameter.

    Args:
        module: The module containing the parameter to constrain.
        parameter_name: Name of the parameter to clamp.
        min: Lower bound for clamping. Defaults to 0.0.
        max: Upper bound for clamping. Defaults to 1.0.
    """
    constraint = Clamp(min=min, max=max)
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_absolute_value(module: nn.Module, parameter_name: str) -> None:
    """Register an absolute value constraint on a module parameter.

    Args:
        module: The module containing the parameter to constrain.
        parameter_name: Name of the parameter to apply absolute value to.
    """
    constraint = AbsoluteValue()
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_spectral_norm(
    module: nn.Module,
    parameter_name: str,
    n_power_iterations: int = 1,
    dim: int = 0,
    eps: float = 1e-12,
) -> None:
    """Register a spectral norm constraint on a module parameter.

    Args:
        module: The module containing the parameter to constrain.
        parameter_name: Name of the parameter to apply spectral normalization to.
        n_power_iterations: Number of power iterations for approximating largest
            singular value. Defaults to 1.
        dim: Dimension corresponding to output features. Defaults to 0.
        eps: Small value for numerical stability. Defaults to 1e-12.
    """
    param = getattr(module, parameter_name)
    constraint = SpectralNorm(param, n_power_iterations, dim, eps)
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_identity_plus_spectral_norm(
    module: nn.Module,
    parameter_name: str,
    n_power_iterations: int = 1,
    dim: int = 0,
    eps: float = 1e-12,
    **kwargs,
) -> None:
    """Register an identity plus spectral norm constraint on a module parameter.

    Args:
        module: The module containing the parameter to constrain.
        parameter_name: Name of the parameter to apply I + SN(X) transformation to.
        n_power_iterations: Number of power iterations for approximating largest
            singular value. Defaults to 1.
        dim: Dimension corresponding to output features. Defaults to 0.
        eps: Small value for numerical stability. Defaults to 1e-12.
        **kwargs: Additional keyword arguments for tensor creation (device, dtype).
    """
    param = getattr(module, parameter_name)
    constraint = IdentityPlusSpectralNorm(param, n_power_iterations, dim, eps, **kwargs)
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_low_rank_svd(module: nn.Module, parameter_name: str, rank: int) -> None:
    """Register a low-rank SVD constraint on a module parameter.

    Args:
        module: The module containing the parameter to constrain.
        parameter_name: Name of the parameter to apply low-rank SVD to.
        rank: Target rank for the low-rank approximation.
    """
    constraint = LowRankSVD(rank)
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_low_rank_factorized(
    module: nn.Module, parameter_name: str, rank: int, **kwargs
) -> None:
    """Register a low-rank factorized constraint on a module parameter.

    Args:
        module: The module containing the parameter to constrain.
        parameter_name: Name of the parameter to factorize into low-rank form.
        rank: Internal rank for the factorization.
        **kwargs: Additional keyword arguments for tensor creation (device, dtype).
    """
    param = getattr(module, parameter_name)
    constraint = LowRankFactorized(param, rank, **kwargs)
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_w_matrix(
    module: nn.Module,
    parameter_name: str,
    dim: int,
    norm_matrix_constraint: str = "full_rank",
    init_log_W: torch.Tensor | None = None,
    init_log_p: torch.Tensor | None = None,
    init_log_q: torch.Tensor | None = None,
    init_W_weight: torch.Tensor | None = None,
    **kwargs,
) -> None:
    """Register a W matrix constraint on a module parameter.

    Args:
        module: The module containing the parameter to constrain.
        parameter_name: Name of the parameter to apply W matrix parametrization to.
        dim: Dimension of the square matrix.
        norm_matrix_constraint: One of {'full_rank', 'one_rank', 'ones', 'fixed_custom'}.
            Defaults to 'full_rank'.
        init_log_W: Initial tensor for log_W parameter (used only with 'full_rank' constraint).
            If None, defaults to 2.5 * torch.randn(dim, dim).
        init_log_p: Initial tensor for log_p parameter (used only with 'one_rank' constraint).
            If None, defaults to torch.randn(dim).
        init_log_q: Initial tensor for log_q parameter (used only with 'one_rank' constraint).
            If None, defaults to 0.1 * torch.randn(dim).
        init_W_weight: Initial tensor for W_weight parameter (used only with 'fixed_custom' constraint).
            If None, defaults to torch.ones(dim, dim).
        **kwargs: Additional keyword arguments for tensor creation (device, dtype).
    """
    constraint = WMatrix(
        dim=dim,
        norm_matrix_constraint=norm_matrix_constraint,
        init_log_W=init_log_W,
        init_log_p=init_log_p,
        init_log_q=init_log_q,
        init_W_weight=init_W_weight,
        **kwargs,
    )
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_orthogonal(
    module: nn.Module, parameter_name: str, use_trivialization: bool = True
) -> None:
    """Register an orthogonal constraint on a module parameter.

    Constrains the parameter to be an orthogonal matrix (W^T @ W = I) using
    QR decomposition.

    Args:
        module: The module containing the parameter to constrain.
        parameter_name: Name of the parameter to make orthogonal.
        use_trivialization: Use trivialized form for better numerical stability
            during optimization. Defaults to True.
    """
    param = getattr(module, parameter_name)
    constraint = Orthogonal(param, use_trivialization=use_trivialization)
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_bounded_positive_sigmoid(
    module: nn.Module,
    parameter_name: str,
    max_val: float = 1.0,
) -> None:
    """Register a bounded positive sigmoid constraint on a module parameter.

    Args:
        module: The module containing the parameter to constrain.
        parameter_name: Name of the parameter to apply bounded positive sigmoid to.
        max_val: Maximum value for the output. Defaults to 1.0.
    """
    constraint = BoundedPositiveSigmoid(max_val=max_val)
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_lds_matrix(
    module: nn.Module,
    parameter_name: str,
    dim: int,
    norm_matrix_constraint: str = "full_rank",
    init_raw_t: torch.Tensor | None = None,
    init_raw_u: torch.Tensor | None = None,
    init_raw_v: torch.Tensor | None = None,
    init_log_W: torch.Tensor | None = None,
    init_log_p: torch.Tensor | None = None,
    init_log_q: torch.Tensor | None = None,
    init_W_weight: torch.Tensor | None = None,
    **kwargs,
) -> None:
    """Register an LDS matrix constraint on a module parameter.

    Args:
        module: The module containing the parameter to constrain.
        parameter_name: Name of the parameter to apply LDS matrix parametrization to.
        dim: Dimension of the square matrix.
        norm_matrix_constraint: One of {'full_rank', 'one_rank', 'ones', 'fixed_custom'}.
            Defaults to 'full_rank'.
        init_raw_t: Initial tensor for raw_t parameter. If None, defaults to 0.5 * torch.rand(dim).
        init_raw_u: Initial tensor for raw_u parameter. If None, defaults to torch.rand(dim).
        init_raw_v: Initial tensor for raw_v parameter. If None, defaults to torch.rand(dim).
        init_log_W: Initial tensor for log_W parameter (used only with 'full_rank' constraint).
            If None, defaults to 2.5 * torch.randn(dim, dim).
        init_log_p: Initial tensor for log_p parameter (used only with 'one_rank' constraint).
            If None, defaults to torch.randn(dim).
        init_log_q: Initial tensor for log_q parameter (used only with 'one_rank' constraint).
            If None, defaults to 0.1 * torch.randn(dim).
        init_W_weight: Initial tensor for W_weight parameter (used only with 'fixed_custom' constraint).
            If None, defaults to torch.ones(dim, dim).
        **kwargs: Additional keyword arguments for tensor creation (device, dtype).
    """
    constraint = LDSMatrix(
        dim=dim,
        norm_matrix_constraint=norm_matrix_constraint,
        init_raw_t=init_raw_t,
        init_raw_u=init_raw_u,
        init_raw_v=init_raw_v,
        init_log_W=init_log_W,
        init_log_p=init_log_p,
        init_log_q=init_log_q,
        init_W_weight=init_W_weight,
        **kwargs,
    )
    parametrize.register_parametrization(module, parameter_name, constraint)
