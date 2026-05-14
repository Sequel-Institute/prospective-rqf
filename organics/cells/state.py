"""
State containers for RNN cells.

This module defines state containers used to manage the hidden state of various
RNN cell implementations. Each state class is a dataclass that holds the tensors
representing the internal state variables of a specific cell type.

All state tensors follow the batch-first convention with shape (batch, hidden_size)
or similar. The State protocol serves as a type hint marker that any dataclass
can satisfy without explicit inheritance.

Classes:
    State: Protocol marker for all state types
    BaseState: Simple RNN state with single hidden vector
    ORGaNICsState: State for ORGaNICs cells with adaptation variables
    LSTMState: State for LSTM cells with hidden and cell states
    LyapossState: State for Lyaposs oscillator cells with position and velocity
    LinearOrganicsState: State for linear ORGaNICs cells with position and velocity

Example:
    >>> from organics.cells.state import LyapossState
    >>> import torch
    >>>
    >>> # Create a state for a batch of 32 sequences with hidden_size=64
    >>> state = LyapossState(
    ...     x=torch.zeros(32, 64),  # position
    ...     v=torch.zeros(32, 64)   # velocity
    ... )
    >>> print(state.x.shape)
    torch.Size([32, 64])
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class State(Protocol):
    """
    Protocol marker for recurrent cell states.

    This protocol serves as a type hint for any state container used with RNN cells.
    Any dataclass with tensor attributes can satisfy this protocol without explicit
    inheritance. This provides flexibility for defining custom state types.

    Note:
        This is a marker protocol with no required attributes or methods. Specific
        cell implementations will use specific State subclasses (e.g., LyapossState)
        that define the actual state variables.

    Example:
        >>> @dataclass
        >>> class MyCustomState:
        ...     z: torch.Tensor  # Custom state variable
        >>>
        >>> # MyCustomState satisfies the State protocol automatically
        >>> def process_state(state: State) -> State:
        ...     return state
    """

    pass


@dataclass
class BaseState:
    """
    Container for basic RNN state.

    This state is used by simple RNN cells (like BaseRNNCell) that maintain
    a single hidden state vector.

    Attributes:
        y: Hidden state tensor of shape (batch, hidden_size)

    Example:
        >>> state = BaseState(y=torch.randn(32, 64))
        >>> state.y.shape
        torch.Size([32, 64])
    """

    y: torch.Tensor  # (batch, hidden_size)


@dataclass
class ORGaNICsState:
    """
    Container for ORGaNICs recurrent state.

    This state is used by two neuron type ORGaNICs cells.

    Attributes:
        y: Excitatory neurons tensor of shape (batch, hidden_size)
        a: Inhibitory neurons variable of shape (batch, hidden_size)
        b0: Baseline input gain of shape (batch, hidden_size)
        b1: Input gain modulation of shape (batch, hidden_size)
        z_prev: Previous-step input drive ``z = relu(Wzx @ x)`` of shape
            ``(batch, hidden_size)``. Populated only when the cell is
            configured with ``prospective_input=True``; otherwise ``None``.

    Example:
        >>> # Standard state (prospective_input disabled)
        >>> state = ORGaNICsState(
        ...     y=torch.randn(32, 64),
        ...     a=torch.randn(32, 64),
        ...     b0=torch.randn(32, 64),
        ...     b1=torch.randn(32, 64),
        ... )
        >>> # State for a cell built with prospective_input=True; z_prev is typically
        >>> # initialized to zeros by ORGaNICsCell.init_state_training.
        >>> prosp_state = ORGaNICsState(
        ...     y=torch.randn(32, 64),
        ...     a=torch.randn(32, 64),
        ...     b0=torch.randn(32, 64),
        ...     b1=torch.randn(32, 64),
        ...     z_prev=torch.zeros(32, 64),
        ... )
    """

    y: torch.Tensor  # (batch, hidden_size)
    a: torch.Tensor  # (batch, hidden_size)
    b0: torch.Tensor  # (batch, hidden_size)
    b1: torch.Tensor  # (batch, hidden_size)
    z_prev: torch.Tensor | None = None  # (batch, hidden_size) when prospective_input=True


@dataclass
class LSTMState:
    """
    Container for LSTM recurrent state.

    This state is used by LSTM cells which maintain both a hidden state (h)
    and a cell state (c) for managing long-term dependencies.

    Attributes:
        h: Hidden state tensor of shape (batch, hidden_size)
        c: Cell state tensor of shape (batch, hidden_size)

    Example:
        >>> state = LSTMState(
        ...     h=torch.randn(32, 64),
        ...     c=torch.randn(32, 64)
        ... )
    """

    h: torch.Tensor  # (batch, hidden_size)
    c: torch.Tensor  # (batch, hidden_size)


@dataclass
class LyapossState:
    """
    Container for Lyaposs oscillator recurrent state.

    This state is used by Lyaposs cells (lyaposs, lyapossEE, lyapossIMEX) which
    model second-order dynamical systems with position and velocity variables.
    The dynamics follow: ddot{x} + H dot{x} + K x = B u(t)

    Attributes:
        x: Position vector of shape (batch, hidden_size)
        v: Velocity vector of shape (batch, hidden_size), where v = dot{x}

    Example:
        >>> state = LyapossState(
        ...     x=torch.randn(32, 64),  # position
        ...     v=torch.randn(32, 64)   # velocity
        ... )
        >>> # Position and velocity are coupled via oscillator dynamics
    """

    x: torch.Tensor  # (batch, hidden_size)
    v: torch.Tensor  # (batch, hidden_size)


@dataclass
class LinearOrganicsState:
    """
    Container for LinearOrganics recurrent state.

    This state is used by linearized ORGaNICs cells that maintain position
    and velocity variables for linear dynamical system modeling.

    Attributes:
        x: Position vector of shape (batch, hidden_size)
        v: Velocity vector of shape (batch, hidden_size)

    Example:
        >>> state = LinearOrganicsState(
        ...     x=torch.randn(32, 64),
        ...     v=torch.randn(32, 64)
        ... )
    """

    x: torch.Tensor  # (batch, hidden_size)
    v: torch.Tensor  # (batch, hidden_size)


def pack_organics_state(state: ORGaNICsState) -> torch.Tensor:
    """Pack ORGaNICsState into a flat tensor for ODE solver.

    This utility function concatenates all state components into a single tensor
    suitable for use with ODE integrators like torchdiffeq.

    Args:
        state: ORGaNICsState with y, a, b0, b1 components of shape (batch, hidden_size).
            ``state.z_prev`` must be ``None``; the discrete-time prospective-input
            mode is not supported by ODE-based wrappers.

    Returns:
        Flat tensor of shape (batch, 4 * hidden_size)

    Raises:
        ValueError: If ``state.z_prev is not None``. Carrying the prospective
            history through an ODE solver is not supported because the two-tap
            discrete correction has no direct continuous-time analog.

    Example:
        >>> state = ORGaNICsState(
        ...     y=torch.randn(32, 64),
        ...     a=torch.randn(32, 64),
        ...     b0=torch.randn(32, 64),
        ...     b1=torch.randn(32, 64)
        ... )
        >>> packed = pack_organics_state(state)
        >>> packed.shape
        torch.Size([32, 256])
    """
    if state.z_prev is not None:
        raise ValueError(
            "pack_organics_state does not support prospective_input states "
            "(state.z_prev is not None). ODE-based wrappers cannot carry the "
            "discrete two-tap correction; use the standard ORGaNICsCell time loop."
        )
    return torch.cat([state.y, state.a, state.b0, state.b1], dim=-1)


def unpack_organics_state(packed: torch.Tensor, hidden_size: int) -> ORGaNICsState:
    """Unpack flat tensor back into ORGaNICsState.

    This utility function reverses pack_organics_state, splitting a flat tensor
    into the individual state components. The returned state has ``z_prev=None``
    since packing rejects prospective-input states.

    Args:
        packed: Tensor of shape (batch, 4 * hidden_size) or (..., 4 * hidden_size)
        hidden_size: Size of each state component

    Returns:
        ORGaNICsState with y, a, b0, b1 of shape (batch, hidden_size) and ``z_prev=None``.

    Example:
        >>> packed = torch.randn(32, 256)
        >>> state = unpack_organics_state(packed, hidden_size=64)
        >>> state.y.shape
        torch.Size([32, 64])
    """
    y, a, b0, b1 = torch.split(packed, hidden_size, dim=-1)
    return ORGaNICsState(y=y, a=a, b0=b0, b1=b1)
