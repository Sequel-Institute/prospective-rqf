"""Base classes for sequence backbones.

This module provides the abstract base class for recurrent backbones that
transform input sequences into learned temporal representations.

Backbones transform input signals into feature representations that capture
temporal structure at multiple scales. In this repository, RQFs are not just
preprocessing layers; they are the recurrent substrate of the architecture.

Classes:
    BackboneBase: Abstract base class for all sequence backbones

See Also:
    For an example implementation of a backbone block, refer to:
    rqf_models/backbones/rqf/block.py (BaseRQFBlock)
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class BackboneBase(nn.Module, ABC):
    """Abstract base class for sequence backbones.

    This class defines the interface for modules that transform input sequences
    into learned feature sequences, potentially with different feature
    dimensions.

    Required Methods (to be implemented by subclasses):
        forward: Transform input data into backbone representations

    The forward method should implement the core sequence transformation logic.

    Example:
        >>> # Subclasses should implement the forward method
        >>> class MyBackbone(BackboneBase):
        ...     def __init__(self, input_dim, output_dim):
        ...         super().__init__()
        ...         self.linear = nn.Linear(input_dim, output_dim)
        ...
        ...     def forward(self, x):
        ...         '''Transform input x into a sequence representation.
        ...
        ...         Args:
        ...             x: Input tensor of shape (batch_size, seq_len, input_dim)
        ...
        ...         Returns:
        ...             Tensor of shape (batch_size, seq_len, output_dim)
        ...         '''
        ...         return self.linear(x)
        ...
        >>> backbone = MyBackbone(10, 20)
        >>> x = torch.randn(32, 100, 10)
        >>> y = backbone(x)
        >>> y.shape
        torch.Size([32, 100, 20])

    Notes:
        Unlike RNN cells which maintain hidden states across time steps,
        backbones expose a standard PyTorch module interface and handle batched
        inputs with shape (batch_size, sequence_length, feature_dim).
    """

    @abstractmethod
    def forward(self, x):
        """Transform input data into a sequence representation.

        This method must be implemented by all subclasses to define the specific
        backbone transformation. The transformation should be differentiable to
        allow gradient-based optimization of any learnable parameters.

        Args:
            x: Input tensor. The expected shape depends on the specific backbone
                implementation, but typically follows (batch_size, sequence_length,
                feature_dim) for sequential data.

        Returns:
            Sequence representation of the input. The output shape depends on the
            specific backbone implementation, but should generally preserve the
            batch and sequence dimensions while potentially transforming the
            feature dimension.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            >>> class CustomBackbone(BackboneBase):
            ...     def forward(self, x):
            ...         # Apply custom transformation
            ...         return transformed_x
        """
        raise NotImplementedError(
            "Subclasses must implement the forward() method to define their "
            "backbone transformation. The forward method should:\n"
            "  1. Accept input tensor(s) with appropriate shape\n"
            "  2. Apply the backbone transformation (e.g., learned filters, "
            "projections, etc.)\n"
            "  3. Return the transformed sequence representation\n\n"
            "For sequential data, inputs typically have shape "
            "(batch_size, sequence_length, feature_dim)."
        )
