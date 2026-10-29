"""
Base Attention Implementations

Core scaled dot-product attention without learnable parameters.
Educational implementation showing the mathematical foundation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention (from "Attention Is All You Need")

    Args:
        query: (..., L, E) where L is target sequence length, E is embedding dim
        key: (..., S, E) where S is source sequence length
        value: (..., S, Ev) where Ev is value embedding dim
        attn_mask: (..., L, S) or (..., L, S) boolean/float mask
        dropout_p: Dropout probability on attention weights
        is_causal: If True, applies causal mask (upper triangular)
        scale: Custom scaling factor (default: 1/sqrt(E))

    Returns:
        output: (..., L, Ev) attended values
        attn_weights: (..., L, S) attention probabilities after softmax
    """
    # Get embedding dimension from query
    embed_dim = query.size(-1)

    # Default scale: 1/sqrt(d_k)
    if scale is None:
        scale = embed_dim ** -0.5

    # Compute similarity scores: Q @ K^T
    # (..., L, E) @ (..., E, S) -> (..., L, S)
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply causal mask if requested
    if is_causal:
        L, S = query.size(-2), key.size(-2)
        causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).triu(1)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

    # Apply attention mask if provided
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
        else:
            attn_scores = attn_scores + attn_mask

    # Softmax to get attention probabilities
    attn_weights = F.softmax(attn_scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    # Weighted average of values: attn_weights @ V
    # (..., L, S) @ (..., S, Ev) -> (..., L, Ev)
    output = torch.matmul(attn_weights, value)

    return output, attn_weights


class AttentionBase(nn.Module):
    """
    Base class for attention modules.
    Provides common functionality for scaled dot-product attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim ** -0.5

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (B, L, E) -> (B, H, L, Hd) for multi-head attention
        """
        if self.batch_first:
            batch, seq_len, _ = x.shape
        else:
            seq_len, batch, _ = x.shape

        return x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine heads: (B, H, L, Hd) -> (B, L, E)
        """
        if self.batch_first:
            x = x.transpose(1, 2).contiguous()
            batch, seq_len, _, _ = x.shape
        else:
            x = x.transpose(1, 2).contiguous()
            seq_len, batch, _, _ = x.shape

        return x.reshape(batch, seq_len, self.embed_dim)


def attention_without_params(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention without any learnable parameters.
    Uses raw input as Q, K, V for educational purposes.

    Args:
        x: (B, L, E) input tensor

    Returns:
        output: (B, L, E) attended output
        attn_weights: (B, L, L) attention weights
    """
    # Use x as Q, K, V directly (no projections)
    output, attn_weights = scaled_dot_product_attention(x, x, x)
    return output, attn_weights