"""
Attention Mechanism Implementations for Transformers

This package provides modular attention implementations:
- Base scaled dot-product attention
- Multi-head attention with learnable projections
- Flash Attention (fused) via PyTorch's scaled_dot_product_attention
- Causal and padding masks utilities
"""

from attention.base import scaled_dot_product_attention, AttentionBase
from attention.multihead import MultiHeadAttention, CrossAttention
from attention.flash import SelfAttention
from attention.masks import create_causal_mask, create_padding_mask, combine_masks

__all__ = [
    "scaled_dot_product_attention",
    "AttentionBase",
    "MultiHeadAttention",
    "CrossAttention",
    "SelfAttention",
    "create_causal_mask",
    "create_padding_mask",
    "combine_masks",
]

__version__ = "1.0.0"