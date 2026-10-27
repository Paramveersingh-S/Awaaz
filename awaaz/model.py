"""Streaming Conformer-CTC model architecture for Awaaz (Phase 4).

Features 2D Conv subsampling (factor 4/8), RoPE attention, causal depthwise convolution,
dynamic-chunk attention masking, and linear CTC head.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ConformerConfig:
    """Configuration class for Conformer model hyperparameters."""

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        num_blocks: int = 12,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        subsample_factor: int = 4,
        vocab_size: int = 1024,
        causal_conv: bool = True,
    ):
        self.n_mels = n_mels
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.conv_kernel_size = conv_kernel_size
        self.dropout = dropout
        self.subsample_factor = subsample_factor
        self.vocab_size = vocab_size
        self.causal_conv = causal_conv


class ConformerCTC(nn.Module):
    """Bilingual Streaming Conformer-CTC speech recognition model."""

    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        chunk_size: Optional[int] = None,
        left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass emitting log-probabilities and output lengths."""
        raise NotImplementedError("Conformer-CTC will be implemented in Phase 4.")
