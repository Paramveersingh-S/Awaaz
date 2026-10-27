"""Streaming Conformer-CTC speech recognition architecture for Awaaz (Phase 4).

Features 2D Conv subsampling (factor 4/8), RoPE relative attention,
causal depthwise convolution, dynamic-chunk attention masking, and linear CTC head.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


@dataclass
class ConformerConfig:
    """Hyperparameter configuration for Conformer-CTC models."""

    name: str = "small"
    n_mels: int = 80
    d_model: int = 256
    num_blocks: int = 12
    num_heads: int = 4
    ffn_dim: int = 1024
    conv_kernel_size: int = 15
    dropout: float = 0.1
    subsample_factor: int = 4
    vocab_size: int = 1024
    causal_conv: bool = True

    @classmethod
    def from_preset(cls, preset_name: str) -> "ConformerConfig":
        """Loads configuration from standard presets: 'tiny', 'small', or 'base'."""
        presets = {
            "tiny": cls(
                name="tiny",
                d_model=144,
                num_blocks=12,
                num_heads=4,
                ffn_dim=576,
                conv_kernel_size=15,
                dropout=0.1,
                subsample_factor=4,
                vocab_size=1024,
                causal_conv=True,
            ),
            "small": cls(
                name="small",
                d_model=256,
                num_blocks=12,
                num_heads=4,
                ffn_dim=1024,
                conv_kernel_size=15,
                dropout=0.1,
                subsample_factor=4,
                vocab_size=1024,
                causal_conv=True,
            ),
            "base": cls(
                name="base",
                d_model=384,
                num_blocks=16,
                num_heads=6,
                ffn_dim=1536,
                conv_kernel_size=15,
                dropout=0.1,
                subsample_factor=4,
                vocab_size=1024,
                causal_conv=True,
            ),
        }
        if preset_name not in presets:
            raise ValueError(f"Unknown preset '{preset_name}'. Choose from: {list(presets.keys())}")
        return presets[preset_name]

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ConformerConfig":
        """Loads configuration from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        model_data = data.get("model", data)
        return cls(
            name=model_data.get("name", "custom"),
            n_mels=model_data.get("n_mels", 80),
            d_model=model_data.get("d_model", 256),
            num_blocks=model_data.get("num_blocks", 12),
            num_heads=model_data.get("num_heads", 4),
            ffn_dim=model_data.get("ffn_dim", 1024),
            conv_kernel_size=model_data.get("conv_kernel_size", 15),
            dropout=model_data.get("dropout", 0.1),
            subsample_factor=model_data.get("subsample_factor", 4),
            vocab_size=model_data.get("vocab_size", 1024),
            causal_conv=model_data.get("causal_conv", True),
        )


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) for relative sequence representation."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cosine and sine rotation matrices for sequence length."""
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]  # (1, 1, seq_len, dim)
        sin = emb.sin()[None, None, :, :]  # (1, 1, seq_len, dim)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE rotation to query or key tensor."""
    return (x * cos) + (rotate_half(x) * sin)


def build_chunk_attention_mask(
    seq_len: int,
    lengths: torch.Tensor,
    chunk_size: Optional[int] = None,
    left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Builds a dynamic-chunk attention mask with strict future blocking and diagonal safety.

    Args:
        seq_len: Sequence length in encoder frames.
        lengths: Valid unpadded lengths per utterance in batch, shape (batch,).
        chunk_size: Attention chunk window size in encoder frames (None = full context).
        left_chunks: Maximum number of past chunks attended to (-1 = unlimited).
        device: Target compute device.

    Returns:
        Additive attention mask of shape (batch, 1, seq_len, seq_len) where
        allowed positions are 0.0 and blocked positions are -1e9.
    """
    batch_size = lengths.shape[0]
    q_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1)
    k_idx = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len)

    # Valid key mask based on utterance lengths
    valid_key = k_idx < lengths.view(-1, 1, 1, 1)

    if chunk_size is not None and chunk_size > 0:
        c_q = q_idx // chunk_size
        c_k = k_idx // chunk_size

        # A frame in chunk c_q may never attend to future chunks (c_k > c_q)
        chunk_allowed = c_k <= c_q
        if left_chunks >= 0:
            chunk_allowed = chunk_allowed & (c_k >= (c_q - left_chunks))

        allowed = valid_key & chunk_allowed
    else:
        allowed = valid_key

    # Softmax NaN safety: guarantee diagonal position (q, q) is allowed for padded rows
    diag = (q_idx == k_idx)
    is_padded_q = q_idx >= lengths.view(-1, 1, 1, 1)
    allowed = allowed | (is_padded_q & diag)

    mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device)
    mask = mask.masked_fill(~allowed, -1e9)
    return mask


class Conv2dSubsampling(nn.Module):
    """2D Convolutional Front-End with factor 4 (or 8) subsampling."""

    def __init__(self, n_mels: int = 80, d_model: int = 256, factor: int = 4):
        super().__init__()
        self.factor = factor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        f_out = (n_mels - 1) // 2 + 1
        f_out = (f_out - 1) // 2 + 1

        if factor == 8:
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
            f_out = (f_out - 1) // 2 + 1
        else:
            self.conv3 = None

        self.proj = nn.Linear(32 * f_out, d_model)

    def reduce_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Applies exact length reduction formula: T_out = (T_in - 1) // 2 + 1."""
        l1 = (lengths - 1) // 2 + 1
        l2 = (l1 - 1) // 2 + 1
        if self.factor == 8:
            return (l2 - 1) // 2 + 1
        return l2

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward subsampling pass.

        Args:
            x: Spectrogram tensor of shape (batch, time, n_mels).
            lengths: Spectrogram lengths of shape (batch,).

        Returns:
            Subsampled representations (batch, time_out, d_model) and updated lengths.
        """
        x = x.unsqueeze(1)  # (batch, 1, time, n_mels)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.conv3 is not None:
            x = F.relu(self.conv3(x))

        b, c, t, f = x.shape
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        projected = self.proj(x)
        out_lengths = self.reduce_lengths(lengths)
        return projected, out_lengths


class RelMultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with Rotary Positional Embeddings (RoPE)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbedding(self.d_k)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / (self.d_k ** 0.5)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(q, t)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(out)


class FeedForwardModule(nn.Module):
    """Macaron Feed-Forward Network Module."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w_1 = nn.Linear(d_model, ffn_dim)
        self.act = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(ffn_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.ln(x)
        res = self.act(self.w_1(res))
        res = self.dropout1(res)
        res = self.w_2(res)
        return self.dropout2(res)


class CausalConvModule(nn.Module):
    """Causal Depthwise Convolution Module with LayerNorm (no BatchNorm)."""

    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.causal = causal

        self.ln1 = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)

        padding = 0 if causal else (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=padding,
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, d = x.shape
        res = self.ln1(x).transpose(1, 2)
        res = F.glu(self.pointwise1(res), dim=1)

        # Zero out padded frames before depthwise convolution
        if lengths is not None:
            valid_mask = (torch.arange(t, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1)
            res = res * valid_mask

        # Causal left-padding: K - 1 frames on the left, 0 on right
        if self.causal:
            res = F.pad(res, (self.kernel_size - 1, 0))

        res = self.depthwise(res)
        res = self.ln2(res.transpose(1, 2))
        res = self.act(res).transpose(1, 2)
        res = self.pointwise2(res).transpose(1, 2)
        return self.dropout(res)


class ConformerBlock(nn.Module):
    """Macaron-style Conformer Block.

    Structure: FFN(0.5) -> SelfAttention -> ConvModule -> FFN(0.5) -> LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        causal_conv: bool = True,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = RelMultiHeadAttention(d_model, num_heads, dropout)
        self.conv = CausalConvModule(d_model, conv_kernel_size, dropout, causal_conv)
        self.ffn2 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn(self.ln_attn(x), mask=mask)
        x = x + self.conv(x, lengths=lengths)
        x = x + 0.5 * self.ffn2(x)
        return self.final_ln(x)


class ConformerEncoder(nn.Module):
    """Stack of Conformer Blocks."""

    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                conv_kernel_size=config.conv_kernel_size,
                dropout=config.dropout,
                causal_conv=config.causal_conv,
            )
            for _ in range(config.num_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask=mask, lengths=lengths)
        return x


class CTCHead(nn.Module):
    """Linear CTC Projection Head producing log-probabilities."""

    def __init__(self, d_model: int, vocab_size: int = 1024):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return F.log_softmax(logits, dim=-1)


class ConformerCTC(nn.Module):
    """Bilingual Streaming Conformer-CTC Speech Recognition Model for Awaaz."""

    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config
        self.subsampling = Conv2dSubsampling(
            n_mels=config.n_mels,
            d_model=config.d_model,
            factor=config.subsample_factor,
        )
        self.encoder = ConformerEncoder(config)
        self.ctc_head = CTCHead(d_model=config.d_model, vocab_size=config.vocab_size)

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        chunk_size: Optional[int] = None,
        left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass emitting log-probabilities and output lengths.

        Args:
            features: Input log-mel spectrogram features of shape (batch, time, n_mels).
            feature_lengths: Unpadded lengths of spectrograms, shape (batch,).
            chunk_size: Streaming attention chunk size in encoder frames (None = full context).
            left_chunks: Number of past chunks attended (-1 = unlimited).

        Returns:
            Tuple of (log_probs, output_lengths) where log_probs has shape
            (batch, time_out, vocab_size) and output_lengths has shape (batch,).
        """
        # 1. 2D Convolution Subsampling
        x, out_lengths = self.subsampling(features, feature_lengths)
        seq_len = x.shape[1]

        # 2. Dynamic-chunk attention mask
        mask = build_chunk_attention_mask(
            seq_len=seq_len,
            lengths=out_lengths,
            chunk_size=chunk_size,
            left_chunks=left_chunks,
            device=x.device,
        )

        # 3. Conformer Encoder blocks
        encoder_out = self.encoder(x, mask=mask, lengths=out_lengths)

        # 4. CTC Projection Head
        log_probs = self.ctc_head(encoder_out)

        return log_probs, out_lengths

    def count_parameters(self) -> Dict[str, int]:
        """Returns parameter counts broken down by module."""
        sub_params = sum(p.numel() for p in self.subsampling.parameters())
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        ctc_params = sum(p.numel() for p in self.ctc_head.parameters())
        total = sub_params + enc_params + ctc_params
        return {
            "subsampling": sub_params,
            "encoder": enc_params,
            "ctc_head": ctc_params,
            "total": total,
        }
