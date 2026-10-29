"""
Example Usage Script for Attention Module

Demonstrates how to use the attention implementations.
Run with: python examples/basic_usage.py
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from attention.base import scaled_dot_product_attention, attention_without_params
from attention.multihead import MultiHeadAttention, CrossAttention
from attention.flash import SelfAttention
from attention.masks import create_causal_mask, create_padding_mask


def example_basic_attention():
    """Basic scaled dot-product attention without learnable params"""
    print("=" * 50)
    print("Example 1: Basic Attention (No Learnable Params)")
    print("=" * 50)

    B, L, E = 2, 16, 64
    x = torch.randn(B, L, E)

    out, weights = attention_without_params(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weight sum per row: {weights.sum(-1)[0, :5]}")  # Should be ~1.0
    print()


def example_multihead_attention():
    """Multi-head attention with learnable projections"""
    print("=" * 50)
    print("Example 2: Multi-Head Attention")
    print("=" * 50)

    B, L, E = 4, 32, 256
    mha = MultiHeadAttention(embed_dim=E, num_heads=8, dropout=0.1)
    x = torch.randn(B, L, E)

    out, weights = mha(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Weights shape: {weights.shape} (averaged over heads)")
    print(f"Params: {sum(p.numel() for p in mha.parameters()):,}")
    print()


def example_cross_attention():
    """Cross-attention for encoder-decoder"""
    print("=" * 50)
    print("Example 3: Cross-Attention (Encoder-Decoder)")
    print("=" * 50)

    B, L_tgt, L_src, E_dec, E_enc = 2, 10, 15, 128, 256
    cross = CrossAttention(embed_dim=E_dec, num_heads=4, encoder_dim=E_enc)

    decoder_out = torch.randn(B, L_tgt, E_dec)
    encoder_out = torch.randn(B, L_src, E_enc)

    out, weights = cross(decoder_out, encoder_out)
    print(f"Decoder input:  {decoder_out.shape}")
    print(f"Encoder input:  {encoder_out.shape}")
    print(f"Output shape:   {out.shape}")
    print(f"Weights shape:  {weights.shape}")
    print()


def example_flash_attention():
    """Flash Attention (fused) self-attention"""
    print("=" * 50)
    print("Example 4: Flash Attention (Self-Attention)")
    print("=" * 50)

    B, L, E = 2, 64, 512
    attn = SelfAttention(
        embed_dim=E,
        num_heads=8,
        attn_dropout=0.1,
        proj_dropout=0.1,
        causal=False,
        fused_attn=True,
    )
    x = torch.randn(B, L, E)

    out = attn(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Using fused:  {attn.fused}")
    print()


def example_causal_attention():
    """Causal (autoregressive) attention"""
    print("=" * 50)
    print("Example 5: Causal Attention")
    print("=" * 50)

    B, L, E = 1, 20, 128
    attn = SelfAttention(embed_dim=E, num_heads=4, causal=True)
    x = torch.randn(B, L, E)

    out = attn(x, is_causal=True)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Causal:       {attn.causal}")
    print()


def example_with_masks():
    """Attention with padding and causal masks"""
    print("=" * 50)
    print("Example 6: Attention with Masks")
    print("=" * 50)

    B, L, E = 3, 12, 64
    mha = MultiHeadAttention(embed_dim=E, num_heads=4)
    x = torch.randn(B, L, E)

    # Variable length sequences: [8, 10, 12] tokens
    lengths = torch.tensor([8, 10, 12])
    padding_mask = create_padding_mask(lengths, max_len=L)
    causal_mask = create_causal_mask(L)

    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Causal mask shape:  {causal_mask.shape}")

    # Combine masks
    from attention.masks import combine_masks
    combined = combine_masks(padding_mask.unsqueeze(1), causal_mask)
    print(f"Combined mask shape: {combined.shape}")

    out, weights = mha(x, attn_mask=combined)
    print(f"Output shape: {out.shape}")
    print(f"Weights shape: {weights.shape}")
    print()


def example_manual_scaled_dot_product():
    """Manual scaled dot-product attention"""
    print("=" * 50)
    print("Example 7: Manual Scaled Dot-Product Attention")
    print("=" * 50)

    B, L, S, E = 1, 5, 7, 32
    q = torch.randn(B, L, E)
    k = torch.randn(B, S, E)
    v = torch.randn(B, S, E)

    out, weights = scaled_dot_product_attention(q, k, v, scale=E ** -0.5)
    print(f"Q shape: {q.shape}")
    print(f"K shape: {k.shape}")
    print(f"V shape: {v.shape}")
    print(f"Output:  {out.shape}")
    print(f"Weights: {weights.shape}")
    print()


if __name__ == "__main__":
    example_basic_attention()
    example_multihead_attention()
    example_cross_attention()
    example_flash_attention()
    example_causal_attention()
    example_with_masks()
    example_manual_scaled_dot_product()

    print("=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)