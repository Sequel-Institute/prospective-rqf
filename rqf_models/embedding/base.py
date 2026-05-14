"""Base classes for embedding layers.

This module provides the abstract base class for embedding layers that transform
raw input data into learned representations suitable for downstream processing.

Embedding layers preprocess and transform input signals into feature
representations that capture relevant patterns at multiple scales or
modalities. They act as learnable preprocessing stages that can adapt to the
specific characteristics of the input data.

Classes:
    EmbeddingBase: Abstract base class for all embedding layers

See Also:
    For an example implementation of an embedding layer, refer to:
    rqf_models/embedding/rqf/block.py (BaseRQFBlock)
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class EmbeddingBase(nn.Module, ABC):
    """Abstract base class for embedding layers.

    This class defines the interface for embedding layers that transform raw
    input data into learned feature representations. Embedding layers serve as
    preprocessing modules that can be composed with sequence models.

    Embeddings are stateless transformations that operate on input sequences
    and produce output sequences, potentially with different feature dimensions.
    They are designed to integrate seamlessly with PyTorch and PyTorch Lightning
    workflows.

    Required Methods (to be implemented by subclasses):
        forward: Transform input data into embedded representations

    The forward method should implement the core embedding transformation logic,
    taking raw input tensors and producing feature-rich representations suitable
    for downstream neural network processing.

    Example:
        >>> # Subclasses should implement the forward method
        >>> class MyEmbedding(EmbeddingBase):
        ...     def __init__(self, input_dim, output_dim):
        ...         super().__init__()
        ...         self.linear = nn.Linear(input_dim, output_dim)
        ...
        ...     def forward(self, x):
        ...         '''Transform input x into embedded representation.
        ...
        ...         Args:
        ...             x: Input tensor of shape (batch_size, seq_len, input_dim)
        ...
        ...         Returns:
        ...             Embedded tensor of shape (batch_size, seq_len, output_dim)
        ...         '''
        ...         return self.linear(x)
        ...
        >>> embedding = MyEmbedding(10, 20)
        >>> x = torch.randn(32, 100, 10)
        >>> y = embedding(x)
        >>> y.shape
        torch.Size([32, 100, 20])

    Notes:
        Unlike RNN cells which maintain hidden states across time steps,
        embedding layers are typically stateless transformations. However,
        they may include learnable parameters that are optimized during training.

        Embedding layers should be designed to handle batched inputs with shape
        (batch_size, sequence_length, feature_dim) and produce outputs of similar
        structure, though the feature dimension may differ.
    """

    @abstractmethod
    def forward(self, x):
        """Transform input data into embedded representation.

        This method must be implemented by all subclasses to define the specific
        embedding transformation. The transformation should be differentiable to
        allow gradient-based optimization of any learnable parameters.

        Args:
            x: Input tensor. The expected shape depends on the specific embedding
                implementation, but typically follows (batch_size, sequence_length,
                feature_dim) for sequential data.

        Returns:
            Embedded representation of the input. The output shape depends on the
            specific embedding implementation, but should generally preserve the
            batch and sequence dimensions while potentially transforming the
            feature dimension.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            >>> class CustomEmbedding(EmbeddingBase):
            ...     def forward(self, x):
            ...         # Apply custom transformation
            ...         return transformed_x
        """
        raise NotImplementedError(
            "Subclasses must implement the forward() method to define their "
            "embedding transformation. The forward method should:\n"
            "  1. Accept input tensor(s) with appropriate shape\n"
            "  2. Apply the embedding transformation (e.g., learned filters, "
            "projections, etc.)\n"
            "  3. Return the embedded representation\n\n"
            "For sequential data, inputs typically have shape "
            "(batch_size, sequence_length, feature_dim)."
        )
