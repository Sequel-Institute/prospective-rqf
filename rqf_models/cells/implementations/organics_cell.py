"""ORGaNICs recurrent cell with two-neuron-type dynamics and adaptive gates.

This module implements the ORGaNICs (Oscillatory Recurrent Gated Neural Integrated Circuits)
cell, which features a two-neuron-type architecture with adaptive normalization dynamics.

Mathematical Formulation:
    The cell implements four coupled differential equations:

        dy/dt = -y + b1*z + (1 - √relu(a))*y_hat
        da/dt = -a + σ²*b0² + W_ay @ (y²*relu(a))
        db0/dt = -b0 + sigmoid(Wbx0@x + Wby0@y + Wba0@a)
        db1/dt = -b1 + sigmoid(Wbx1@x + Wby1@y + Wba1@a)

    where:
        - y: excitatory neuron population (primary hidden state)
        - a: inhibitory neuron population (implementing divisive normalization)
        - b0, b1: adaptive gates for input modulation
        - z = relu(Wzx @ x): processed input
        - y_hat: recurrent pathway with optional nonlinearity
        - σ: semi-saturation constant
        - W_ay: normalization matrix (positive weights)

Architecture:
    - Two neuron types (excitatory y, inhibitory a)
    - Adaptive gating mechanisms (b0, b1)
    - Configurable constraints on recurrent weights
    - Learnable time constants for each variable

Conventions:
    - All state tensors follow batch-first convention: (batch, hidden_size)
    - Time constants are bounded via sigmoid to ensure positivity
    - Supports multiple recurrent weight constraints (spectral norm, orthogonal, etc.)

Example:
    >>> from rqf_models.cells.implementations.organics_cell import ORGaNICsCell
    >>> cell = ORGaNICsCell(input_size=10, hidden_size=64,
    ...                      organics_activation='rectified_recurrence',
    ...                      Wr_constraint='spectral_norm')
    >>> state = cell.init_state_training(batch_size=32)
    >>> input_t = torch.randn(32, 10)
    >>> cell.compute_cache()
    >>> next_state = cell.step(input_t, state)
"""

import math
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...constraints import parameter as constraints
from ..base import RNNCell
from ..state import ORGaNICsState


class ORGaNICsCell(RNNCell):
    r"""ORGaNICs recurrent cell with two-neuron-type dynamics.

    Implements a dynamical system with excitatory (y) and inhibitory (a) populations,
    coupled through adaptive gates (b0, b1) for robust sequence processing with
    divisive normalization.

    Args:
        input_size: Dimension of input vector at each time step
        hidden_size: Dimension of hidden state vectors (y, a, b0, b1)
        organics_activation: Nonlinearity applied in recurrent pathway.
            Options: 'rectified_recurrence' (ReLU before recurrence),
                    'rectified_y' (ReLU on y before recurrence),
                    'no_rectification' (linear recurrence).
            Default: 'rectified_recurrence'
        Wr_identity: Force identity initialization for Wr (for stability). Default: False
        Wr_constraint: Constraint on recurrent matrix Wr. Options: 'unconstrained',
            'identity', 'positive_diagonal', 'spectral_norm',
            'identity_plus_spectral_norm', 'orthogonal'. Default: 'unconstrained'
        Way_constraint: Constraint type for normalization matrix Way. Options:
            'full_rank', 'one_rank', 'ones', 'fixed_custom'. Default: 'full_rank'
        learn_tau: Whether to learn time constants. Default: True
        dt_tau_max_y: Maximum time constant for y. Default: 0.01
        dt_tau_max_a: Maximum time constant for a. Default: 0.01
        dt_tau_max_b: Maximum time constant for b0 and b1. Default: 0.1
        sigma: Semi-saturation constant for a dynamics. Default: 1.0
        n: Exponent in a dynamics (currently unused). Default: 2.0
        prospective_input: Replace the y-equation drive ``b1·z`` with its
            prospective form ``b1·(z + tau_y · dz/dt)``. Discretizing with
            forward Euler yields an additive correction ``b1·(z[n] − z[n-1])``
            on top of the standard update. Requires the state to carry a
            ``z_prev`` buffer (initialized to zeros). Default: False
        device: Device to place tensors on. Default: None
        dtype: Data type for tensors. Default: None

    Attributes:
        Wr: Recurrent weight matrix (hidden_size, hidden_size)
        Wzx: Input projection matrix (hidden_size, input_size)
        Wbx0, Wby0, Wba0: Gate b0 weights
        Wbx1, Wby1, Wba1: Gate b1 weights
        Way_module: WMatrix parametrization module for normalization matrix
        sigma_param: Semi-saturation constant parameter
        dt_tauy, dt_taua, dt_taub0, dt_taub1: Time constant parameters (with bounded positive constraint)
        init_y, init_a, init_b0, init_b1: Initial state buffers

    Notes:
        - Follows batch-first convention: (batch, hidden_size)
        - Time constants bounded via sigmoid to (0, tau_max)
        - Normalization matrix Way uses WMatrix parametrization for positivity
        - Supports various constraints on Wr for stability and expressivity
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        organics_activation: Literal[
            "rectified_recurrence", "rectified_y", "no_rectification"
        ] = "rectified_recurrence",
        Wr_identity: bool = False,
        Wr_constraint: str = "unconstrained",
        Way_constraint: Literal["full_rank", "one_rank", "ones", "fixed_custom"] = "full_rank",
        learn_tau: bool = True,
        dt_tau_max_y: float = 0.01,
        dt_tau_max_a: float = 0.01,
        dt_tau_max_b: float = 0.1,
        sigma: float = 1.0,
        n: float = 2.0,
        prospective_input: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        # Store configuration before calling super().__init__()
        # because parent calls _initialize_parameters which needs these
        if organics_activation not in ["rectified_recurrence", "rectified_y", "no_rectification"]:
            raise ValueError(
                f"organics_activation must be one of 'rectified_recurrence', "
                f"'rectified_y', or 'no_rectification', got '{organics_activation}'"
            )
        self.organics_activation = organics_activation
        self.learn_tau = learn_tau
        self.dt_tau_max_y = dt_tau_max_y
        self.dt_tau_max_a = dt_tau_max_a
        self.dt_tau_max_b = dt_tau_max_b
        self.sigma_val = sigma  # Store for parameter creation
        self.n = n
        self.prospective_input = prospective_input

        # Store Wr_identity flag for initialization control
        self.Wr_identity = Wr_identity

        if Wr_constraint not in {
            "unconstrained",
            "identity",
            "positive_diagonal",
            "spectral_norm",
            "identity_plus_spectral_norm",
            "orthogonal",
        }:
            raise ValueError(
                f"Wr_constraint must be one of 'unconstrained', 'identity', "
                f"'positive_diagonal', 'spectral_norm', 'identity_plus_spectral_norm', "
                f"or 'orthogonal', got '{Wr_constraint}'"
            )
        self.Wr_constraint = Wr_constraint

        # Validate Way_constraint
        if Way_constraint not in {"full_rank", "one_rank", "ones", "fixed_custom"}:
            raise ValueError(
                f"Way_constraint must be one of 'full_rank', 'one_rank', 'ones', "
                f"or 'fixed_custom', got '{Way_constraint}'"
            )
        self.Way_constraint = Way_constraint

        # Call parent constructor which will call _initialize_parameters
        super().__init__(input_size, hidden_size, device, dtype)

    def _initialize_parameters(self, **kwargs: Any) -> None:
        """Initialize all learnable parameters and buffers.

        Creates:
            - Recurrent weight matrix Wr with configured constraint
            - Input projection matrix Wzx
            - Gate weight matrices (Wbx0, Wby0, Wba0, Wbx1, Wby1, Wba1)
            - Normalization matrix Way using WMatrix parametrization
            - Time constant parameters with bounded positive constraint
            - Semi-saturation constant parameter
            - Initial state buffers

        Args:
            **kwargs: Device and dtype specifications for tensor creation
        """
        # Recurrent weight matrix
        self.Wr = nn.Parameter(torch.eye(self.hidden_size, **kwargs))
        self._register_wr_parametrization()

        # Input drive weight matrix
        self.Wzx = nn.Parameter(torch.randn(self.hidden_size, self.input_size, **kwargs))
        self._register_wzx_parametrization()

        # Gate b0 parameters
        self.Wbx0 = nn.Parameter(torch.randn(self.hidden_size, self.input_size, **kwargs))
        self.Wby0 = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size, **kwargs))
        self.Wba0 = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size, **kwargs))

        # Gate b1 parameters
        self.Wbx1 = nn.Parameter(torch.randn(self.hidden_size, self.input_size, **kwargs))
        self.Wby1 = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size, **kwargs))
        self.Wba1 = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size, **kwargs))

        # Normalization matrix using WMatrix parametrization
        # Initialize log_W with sparse pattern (50% zeros)
        # Sparse zeros become exp(-18.42) ≈ 0, non-zeros become uniform(0, 1)
        # Set diagonal to 0 so Way diagonal = exp(0) = 1.0
        sparsity = 0.5  # 50% of off-diagonal elements are masked
        mask = torch.rand(self.hidden_size, self.hidden_size, **kwargs) > sparsity
        uniform_vals = torch.empty(self.hidden_size, self.hidden_size, **kwargs).uniform_(0.0, 1.0)
        vals = uniform_vals * mask.float()
        init_log_W = torch.log(torch.clamp(vals, min=1e-8))
        init_log_W.fill_diagonal_(0.0)
        self.Way_module = constraints.WMatrix(
            dim=self.hidden_size,
            norm_matrix_constraint=self.Way_constraint,
            init_log_W=init_log_W,
            **kwargs,
        )

        # Semi-saturation constant
        self.sigma_param = nn.Parameter(
            self.sigma_val * torch.ones(self.hidden_size, **kwargs),
            requires_grad=False,
        )

        # Time constants with bounded positive parametrization
        self.dt_tauy = nn.Parameter(
            torch.randn(self.hidden_size, **kwargs),
            requires_grad=self.learn_tau,
        )
        constraints.register_bounded_positive_sigmoid(self, "dt_tauy", max_val=self.dt_tau_max_y)

        self.dt_taua = nn.Parameter(
            torch.randn(self.hidden_size, **kwargs),
            requires_grad=self.learn_tau,
        )
        constraints.register_bounded_positive_sigmoid(self, "dt_taua", max_val=self.dt_tau_max_a)

        self.dt_taub0 = nn.Parameter(
            torch.randn(self.hidden_size, **kwargs),
            requires_grad=self.learn_tau,
        )
        constraints.register_bounded_positive_sigmoid(self, "dt_taub0", max_val=self.dt_tau_max_b)

        self.dt_taub1 = nn.Parameter(
            torch.randn(self.hidden_size, **kwargs),
            requires_grad=self.learn_tau,
        )
        constraints.register_bounded_positive_sigmoid(self, "dt_taub1", max_val=self.dt_tau_max_b)

        # Initialize all weights
        self._initialize_weights()

        # Initialize state buffers
        self._init_state_buffers(**kwargs)

    def _register_wr_parametrization(self) -> None:
        """Apply constraint to recurrent weight matrix Wr."""
        if self.Wr_constraint == "unconstrained":
            # No constraint needed
            pass
        elif self.Wr_constraint == "identity":
            self.Wr.requires_grad = False
        elif self.Wr_constraint == "positive_diagonal":
            constraints.register_diagonal(self, "Wr")
            constraints.register_clamp(self, "Wr", min=0.0, max=1.0)
        elif self.Wr_constraint == "orthogonal":
            constraints.register_orthogonal(self, "Wr")
        elif self.Wr_constraint == "spectral_norm":
            constraints.register_spectral_norm(self, "Wr", n_power_iterations=1)
        elif self.Wr_constraint == "identity_plus_spectral_norm":
            constraints.register_identity_plus_spectral_norm(self, "Wr", n_power_iterations=1)

    def _register_wzx_parametrization(self) -> None:
        """Apply spectral norm to input matrix if recurrent path is constrained."""
        if self.Wr_constraint in {"spectral_norm", "identity_plus_spectral_norm", "orthogonal"}:
            constraints.register_spectral_norm(self, "Wzx", n_power_iterations=1)

    def _initialize_weights(self) -> None:
        """Initialize all weight matrices according to their constraints."""
        self._initialize_wr()
        self._initialize_wzx()
        self._initialize_gain_weights()

    def _initialize_wr(self) -> None:
        """Initialize recurrent matrix according to Wr_identity flag and constraint."""
        # Force identity initialization for stability (keeps torch.eye from constructor)
        if self.Wr_identity:
            return

        if self.Wr_constraint in {
            "unconstrained",
            "spectral_norm",
            "identity_plus_spectral_norm",
            "orthogonal",
        }:
            # Orthogonal initialization for stability
            with torch.no_grad():
                nn.init.orthogonal_(self.Wr)
        elif self.Wr_constraint == "positive_diagonal":
            # Positive diagonal entries in [0, 1]
            with torch.no_grad():
                diag = torch.empty(self.hidden_size, device=self.Wr.device, dtype=self.Wr.dtype)
                diag.uniform_(0.0, 1.0)
                self.Wr.copy_(torch.diag(diag))
        # identity: keep identity from constructor

    def _initialize_wzx(self) -> None:
        """Initialize input projection matrix."""
        nn.init.kaiming_uniform_(self.Wzx, a=math.sqrt(5))

        # Normalize spectral norm to 1 if recurrent path is constrained
        if self.Wr_constraint in {"spectral_norm", "identity_plus_spectral_norm", "orthogonal"}:
            with torch.no_grad():
                spectral_norm = torch.linalg.norm(self.Wzx, ord=2)
                if spectral_norm > 0:
                    self.Wzx.data = self.Wzx.data / spectral_norm

    def _initialize_gain_weights(self) -> None:
        """Initialize gate weight matrices with Kaiming uniform."""
        for w in [self.Wbx0, self.Wby0, self.Wba0, self.Wbx1, self.Wby1, self.Wba1]:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

    def _init_state_buffers(self, **kwargs: Any) -> None:
        """Create initial state buffers for all variables.

        Args:
            **kwargs: Device and dtype specifications
        """
        # Normalized random initial y
        init_y = torch.rand(self.hidden_size, **kwargs)
        init_y = init_y / torch.norm(init_y)
        self.register_buffer("init_y", init_y)

        # Random initial a
        self.register_buffer("init_a", torch.rand(self.hidden_size, **kwargs))

        # Initial gates
        self.register_buffer("init_b0", torch.ones(self.hidden_size, **kwargs))
        self.register_buffer("init_b1", torch.rand(self.hidden_size, **kwargs))

        # Prospective-input history buffer — registered unconditionally so the state_dict
        # shape is invariant across the prospective_input flag, but populated into the
        # runtime state only when prospective_input=True.
        self.register_buffer("init_z_prev", torch.zeros(self.hidden_size, **kwargs))

    def init_state_training(self, batch_size: int, **kwargs: Any) -> ORGaNICsState:
        """Create initial batched state for training.

        Args:
            batch_size: Number of sequences in the batch
            **kwargs: Optional device/dtype (usually not needed)

        Returns:
            Initial ORGaNICsState with all variables of shape (batch_size, hidden_size)
        """
        z_prev = (
            self.init_z_prev.unsqueeze(0).expand(batch_size, -1).contiguous()
            if self.prospective_input
            else None
        )
        return ORGaNICsState(
            y=self.init_y.unsqueeze(0).expand(batch_size, -1).contiguous(),
            a=self.init_a.unsqueeze(0).expand(batch_size, -1).contiguous(),
            b0=self.init_b0.unsqueeze(0).expand(batch_size, -1).contiguous(),
            b1=self.init_b1.unsqueeze(0).expand(batch_size, -1).contiguous(),
            z_prev=z_prev,
        )

    @staticmethod
    def _act_y(y: torch.Tensor, organics_activation: str) -> torch.Tensor:
        """Apply activation to y for normalization dynamics.

        Args:
            y: Excitatory population activity
            organics_activation: Type of activation

        Returns:
            Activated y (typically y^2 with optional ReLU)
        """
        if organics_activation in {"rectified_recurrence", "rectified_y"}:
            return torch.relu(y) ** 2
        elif organics_activation == "no_rectification":
            return y**2
        else:
            raise ValueError(f"Invalid organics_activation: {organics_activation}")

    @staticmethod
    def _act_a(a: torch.Tensor) -> torch.Tensor:
        """Apply activation to inhibitory population.

        Args:
            a: Inhibitory population activity

        Returns:
            sqrt(relu(a))
        """
        return torch.sqrt(torch.relu(a))

    @property
    def Way(self) -> torch.Tensor:
        """Return positive normalization matrix.

        Returns:
            Way matrix of shape (hidden_size, hidden_size).
        """
        return self.Way_module()

    def step(
        self,
        x_t: torch.Tensor,
        s: ORGaNICsState,
        *,
        detach_z_prev_source: bool = False,
    ) -> ORGaNICsState:
        """Advance state by one discrete time step.

        Implements the discrete-time ORGaNICs dynamics with forward Euler integration.

        Args:
            x_t: Input at current time step, shape (batch, input_size)
            s: Current state with y, a, b0, b1 of shape (batch, hidden_size)
            detach_z_prev_source: When ``True`` and prospective input is enabled,
                store the next delayed tap as ``relu(Wzx @ stopgrad(x_t))``.
                This preserves gradients to ``Wzx`` through the future lag tap
                while cutting gradients into the source activation.

        Returns:
            Updated ORGaNICsState after one time step
        """
        # Process input
        z = F.relu(F.linear(x_t, self.Wzx, bias=None))

        # Recurrent pathway with optional nonlinearity
        Wr = self.Wr  # Parametrization applied automatically
        if self.organics_activation == "rectified_recurrence":
            y_hat = F.relu(F.linear(s.y, Wr, bias=None))
        elif self.organics_activation == "rectified_y":
            y_hat = F.linear(F.relu(s.y), Wr, bias=None)
        elif self.organics_activation == "no_rectification":
            y_hat = F.linear(s.y, Wr, bias=None)

        # Compute gate targets
        B0 = torch.sigmoid(
            F.linear(x_t, self.Wbx0, bias=None)
            + F.linear(s.y, self.Wby0, bias=None)
            + F.linear(s.a, self.Wba0, bias=None)
        )
        B1 = torch.sigmoid(
            F.linear(x_t, self.Wbx1, bias=None)
            + F.linear(s.y, self.Wby1, bias=None)
            + F.linear(s.a, self.Wba1, bias=None)
        )

        # Update gates
        b0 = s.b0 + self.dt_taub0 * (-s.b0 + B0)
        b1 = s.b1 + self.dt_taub1 * (-s.b1 + B1)

        # Update excitatory population
        y = s.y + self.dt_tauy * (-s.y + b1 * z + (1 - self._act_a(s.a)) * y_hat)

        # Prospective-input correction: forward-Euler discretization of
        # b1·(z + tau_y · dz/dt) adds b1·(z[n] − z[n-1]) on top of the standard update.
        if self.prospective_input:
            # Narrows Optional[Tensor] for TorchScript and fails loudly if a caller
            # enables prospective mode but supplies a state without z_prev.
            assert s.z_prev is not None, (
                "prospective_input=True requires state.z_prev to be populated; "
                "build the initial state via init_state_training() or pass a zero tensor."
            )
            y = y + b1 * (z - s.z_prev)

        # Update inhibitory population (divisive normalization)
        a = s.a + self.dt_taua * (
            -s.a
            + self.sigma_param**2 * (b0**2)
            + F.linear(
                self._act_y(s.y, self.organics_activation) * torch.relu(s.a), self.Way, bias=None
            )
        )

        if self.prospective_input and detach_z_prev_source:
            z_prev = F.relu(F.linear(x_t.detach(), self.Wzx, bias=None))
        else:
            z_prev = z if self.prospective_input else None
        return ORGaNICsState(y=y, a=a, b0=b0, b1=b1, z_prev=z_prev)

    def output(self, state: ORGaNICsState) -> torch.Tensor:
        """Extract output from state.

        Args:
            state: Current ORGaNICsState

        Returns:
            Excitatory population y of shape (batch, hidden_size)
        """
        return state.y
