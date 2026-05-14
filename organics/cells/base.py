"""
Base classes for RNN cells.

This module provides the abstract base class for RNN cells that define
the interface for single-timestep computational units and manage hidden
state evolution in coordination with learning rules.

The RNNCell class establishes a standard interface for recurrent computations
that can be used with various learning rules and network architectures. It
supports both online and batch processing modes while maintaining memory
efficiency for hidden state storage.

Classes:
    RNNCell: Abstract base class for all RNN cells

See Also:
    For a simple example implementation of an RNN cell, refer to:
    docs/examples/base_rnn_cell.py
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from .state import State


class RNNCell(nn.Module, ABC):
    """
    Abstract base class for RNN cells used in ORGaNICs networks.

    This class defines the interface for single-timestep computational units
    that manage hidden state evolution and coordinate with learning rules for
    weight updates. Cells are designed to be composable within the ORGaNICs
    framework and work seamlessly with PyTorch Lightning modules.

    The cell architecture supports both online and batch processing modes
    while maintaining memory efficiency for hidden state storage. Subclasses
    must implement parameter initialization, state management, and the core
    step function that advances the state by one time step.

    Required Methods (to be implemented by subclasses):
        _initialize_parameters: Initialize cell-specific parameters
        _init_state_buffers: Create initial state buffers
        init_state_training: Create batched initial states
        step: Perform one forward step
        output: Extract output from state

    Optional Methods (can be overridden):
        compute_cache: Precompute constants for efficiency
        get_learning_state: Provide learning rule coordination info
        reset_parameters: Reset to initial parameter values

    Attributes:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden state

    Note:
        This class follows standard PyTorch RNN cell patterns and is compatible
        with existing PyTorch RNN workflows while adding local learning capabilities.

    Example:
        >>> # Subclasses should implement the required abstract methods
        >>> class MyCell(RNNCell):
        ...     def _initialize_parameters(self, **kwargs):
        ...         self.weight = nn.Parameter(torch.randn(self.hidden_size, self.input_size))
        ...         self._init_state_buffers(**kwargs)
        ...
        ...     def _init_state_buffers(self, **kwargs):
        ...         self.register_buffer("init_h", torch.zeros(self.hidden_size, **kwargs))
        ...
        ...     def init_state_training(self, batch_size, **kwargs):
        ...         return MyState(h=self.init_h.unsqueeze(0).expand(batch_size, -1).contiguous())
        ...
        ...     def step(self, input, state):
        ...         # Custom dynamics here
        ...         return updated_state
        ...
        ...     def output(self, state):
        ...         return state.h
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the RNN cell.

        Args:
            input_size: The number of expected features in the input x
            hidden_size: The number of features in the hidden state h
            device: Device to place tensors on
            dtype: Data type for tensors
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Store factory kwargs for tensor creation
        kwargs = {"device": device, "dtype": dtype}

        # Initialize parameters (to be defined by subclasses)
        self._initialize_parameters(**kwargs)

    @abstractmethod
    def _initialize_parameters(self, **kwargs: Any) -> None:
        """
        Initialize cell parameters.

        This method should be implemented by subclasses to define the specific
        parameters needed for the cell (e.g., weight matrices, biases). It should
        also call _init_state_buffers() to set up initial state values.

        Args:
            **kwargs: Device and dtype specifications for tensor creation,
                            typically {"device": device, "dtype": dtype}

        Example:
            >>> def _initialize_parameters(self, **kwargs):
            ...     self.W = nn.Parameter(torch.randn(self.hidden_size, self.input_size, **kwargs))
            ...     self.b = nn.Parameter(torch.zeros(self.hidden_size, **kwargs))
            ...     self._init_state_buffers(**kwargs)
        """
        raise NotImplementedError

    @abstractmethod
    def _init_state_buffers(self, **kwargs: Any) -> None:
        """
        Initialize hidden state buffers for various variables in the model.

        This method should register buffers using `register_buffer()` for initial
        state values (e.g., initial position, velocity, hidden state). These buffers
        will be automatically moved to the correct device/dtype when the module is
        moved.

        Args:
            **kwargs: Device and dtype specifications for tensor creation

        Note:
            Buffers should be registered, not returned. Use self.register_buffer()
            to ensure proper state management and device placement.

        Example:
            >>> def _init_state_buffers(self, **kwargs):
            ...     self.register_buffer("init_h", torch.zeros(self.hidden_size, **kwargs))
            ...     self.register_buffer("init_c", torch.zeros(self.hidden_size, **kwargs))
        """
        raise NotImplementedError

    @abstractmethod
    def init_state_training(self, batch_size: int, **kwargs: Any) -> State:
        """
        Create an initial batched state for training.

        Returns a state object with the hidden state initialized for all sequences
        in the batch. The state should have batch dimension as the first dimension.

        Args:
            batch_size: Number of sequences in the batch
            **kwargs: Optional device and dtype specifications (usually not needed
                            as buffers already have correct device/dtype)

        Returns:
            Initial state object with all state variables of shape (batch_size, ...)
            The specific State subclass depends on the cell implementation.

        Example:
            >>> state = cell.init_state_training(batch_size=32)
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        input: torch.Tensor,
        state: State,
    ) -> State:
        """
        Perform a single forward step of the RNN cell.

        This method advances the hidden state by one discrete time step given
        the current input and previous state. This is the core recurrent computation
        that defines the cell's dynamics.

        Args:
            input: Input tensor of shape (batch_size, input_size)
            state: Current state object containing all hidden variables.
                  The specific State type depends on the cell implementation.

        Returns:
            Updated state object after one time step, containing the new values
            of all hidden variables. The state type should match the input state type.

        Example:
            >>> cell = lyapossEE(input_size=10, hidden_size=64)
            >>> state = cell.init_state_training(batch_size=32)
            >>> input_t = torch.randn(32, 10)
            >>> next_state = cell.step(input_t, state)
        """
        raise NotImplementedError

    @abstractmethod
    def output(self, state: State) -> torch.Tensor:
        """
        Extract the output tensor from the state for prediction.

        This method defines which hidden variable(s) from the state should be
        used as the cell's output for prediction tasks (e.g., in a readout layer).
        For many cells, this simply returns the main hidden state, but some cells
        may combine multiple state variables or apply a transformation.

        Args:
            state: Current state object containing all hidden variables

        Returns:
            Output tensor of shape (batch_size, hidden_size) or (batch_size, output_size)
            suitable for use in downstream layers or prediction heads.

        Example:
            >>> # For a simple RNN cell
            >>> output = cell.output(state)  # Returns state.h
            >>>
            >>> # For an oscillator cell
            >>> output = cell.output(state)  # Returns state.x (position)
        """
        raise NotImplementedError

    def compute_cache(self) -> None:
        """
        Precompute and cache constants for efficient sequence processing.

        This method is called once before processing a sequence to compute any
        quantities that remain constant during RNN unrolling (e.g., discretized
        system matrices, transformed parameters). Caching these values avoids
        redundant computation at each time step. Please look at the compute_cache() methods in
        in lyapossEE and lyapossIMEX (in organics/cells/implementations/oscillator.py file) for examples.

        Note:
            This is optional. Subclasses only need to override if they have
            parameters that can be precomputed and reused across time steps.
        """
        pass

    def forward(
        self,
        input: torch.Tensor,
        state: State,
    ) -> State:
        """
        Alias for the step method to support standard PyTorch Module calling.

        This allows the cell to be called directly as `cell(input, state)` in
        addition to `cell.step(input, state)`.

        Args:
            input: Input tensor of shape (batch_size, input_size)
            state: Current state object containing all hidden variables

        Returns:
            Updated state object after one time step
        """
        return self.step(input, state)

    def get_learning_state(self) -> dict[str, Any]:
        """
        Get the current learning state of the cell.

        This method returns any internal state that learning rules might need
        to coordinate weight updates. Subclasses can override this to provide
        cell-specific learning state information.

        Returns:
            Dictionary containing learning state information with keys:
                - input_size: Dimension of input features
                - hidden_size: Dimension of hidden state
                - (subclasses may add additional entries)
        """
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
        }

    def reset_parameters(self) -> None:
        """
        Reset cell parameters to initial values.

        This method reinitializes all learnable parameters using the same
        initialization scheme used during construction. Useful for restarting
        training from scratch without recreating the module.

        Note:
            Default implementation is a no-op. Subclasses should override this
            method to implement their specific parameter reset logic.
        """
        # Default implementation - subclasses should override
        pass

    def extra_repr(self) -> str:
        """
        Return extra representation string for debugging.

        This string is included when printing the module to provide useful
        information about the cell configuration.

        Returns:
            String containing input_size and hidden_size
        """
        return f"input_size={self.input_size}, hidden_size={self.hidden_size}"
