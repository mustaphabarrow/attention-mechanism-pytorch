"""
Unit Tests for Attention Implementations
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from attention.base import (
    scaled_dot_product_attention,
    AttentionBase,
    attention_without_params,
)
from attention.multihead import MultiHeadAttention, CrossAttention
from attention.flash import SelfAttention
from attention.masks import (
    create_causal_mask,
    create_padding_mask,
    combine_masks,
    expand_mask_for_heads,
    create_alibi_mask,
)


class TestBaseAttention:
    """Tests for base attention functions"""

    def test_scaled_dot_product_attention_shapes(self):
        B, L, S, E, Ev = 2, 10, 12, 64, 64
        q = torch.randn(B, L, E)
        k = torch.randn(B, S, E)
        v = torch.randn(B, S, Ev)

        out, weights = scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, L, Ev)
        assert weights.shape == (B, L, S)

    def test_attention_weights_sum_to_one(self):
        B, L, S, E = 1, 5, 5, 32
        q = torch.randn(B, L, E)
        k = torch.randn(B, S, E)
        v = torch.randn(B, S, E)

        _, weights = scaled_dot_product_attention(q, k, v)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(B, L))

    def test_causal_mask(self):
        B, L, E = 1, 4, 16
        q = torch.randn(B, L, E)
        k = torch.randn(B, L, E)
        v = torch.randn(B, L, E)

        out, weights = scaled_dot_product_attention(q, k, v, is_causal=True)
        assert torch.allclose(weights.triu(1), torch.zeros_like(weights.triu(1)), atol=1e-6)

    def test_attention_without_params(self):
        x = torch.randn(2, 8, 32)
        out, weights = attention_without_params(x)
        assert out.shape == (2, 8, 32)
        assert weights.shape == (2, 8, 8)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module"""

    def test_self_attention(self):
        B, L, E = 2, 16, 128
        mha = MultiHeadAttention(embed_dim=E, num_heads=8)
        x = torch.randn(B, L, E)

        out, weights = mha(x)
        assert out.shape == (B, L, E)
        assert weights.shape == (B, L, L)

    def test_cross_attention(self):
        B, L, S, E, E_enc = 2, 10, 12, 64, 128
        cross = CrossAttention(embed_dim=E, num_heads=4, encoder_dim=E_enc)
        query = torch.randn(B, L, E)
        encoder_out = torch.randn(B, S, E_enc)

        out, weights = cross(query, encoder_out)
        assert out.shape == (B, L, E)
        assert weights.shape == (B, L, S)

    def test_attention_mask(self):
        B, L, E = 1, 5, 32
        mha = MultiHeadAttention(embed_dim=E, num_heads=4)
        x = torch.randn(B, L, E)

        mask = torch.zeros(L, L, dtype=torch.bool)
        mask[:, -1] = True

        out, weights = mha(x, attn_mask=mask)
        assert weights.shape == (B, L, L)
        assert torch.allclose(weights[:, :, -1], torch.zeros(B, L), atol=1e-5)

    def test_key_padding_mask(self):
        B, L, E = 2, 6, 32
        mha = MultiHeadAttention(embed_dim=E, num_heads=4)
        x = torch.randn(B, L, E)

        key_padding_mask = torch.tensor([
            [False, False, False, False, False, False],
            [False, False, False, True, True, True],
        ])

        out, weights = mha(x, key_padding_mask=key_padding_mask)
        assert torch.allclose(weights[1, :, 3:], torch.zeros(1, L, 3), atol=1e-5)


class TestFlashAttention:
    """Tests for Flash Attention / SelfAttention"""

    def test_self_attention_forward(self):
        B, L, E = 2, 32, 256
        attn = SelfAttention(embed_dim=E, num_heads=8)
        x = torch.randn(B, L, E)

        out = attn(x)
        assert out.shape == (B, L, E)

    def test_causal_self_attention(self):
        B, L, E = 1, 10, 64
        attn = SelfAttention(embed_dim=E, num_heads=4, causal=True)
        x = torch.randn(B, L, E)

        out = attn(x, is_causal=True)
        assert out.shape == (B, L, E)

    def test_with_attention_mask(self):
        B, L, E = 1, 8, 64
        attn = SelfAttention(embed_dim=E, num_heads=4)
        x = torch.randn(B, L, E)

        mask = create_causal_mask(L)
        out = attn(x, attention_mask=mask)
        assert out.shape == (B, L, E)


class TestMasks:
    """Tests for mask utilities"""

    def test_causal_mask_shape(self):
        mask = create_causal_mask(10)
        assert mask.shape == (10, 10)
        assert mask.dtype == torch.bool
        assert mask.triu(1).all()
        assert not mask.tril(0).any()

    def test_padding_mask(self):
        lengths = torch.tensor([5, 3, 7])
        mask = create_padding_mask(lengths, max_len=7)
        assert mask.shape == (3, 7)
        assert mask[0, 5:].all()
        assert not mask[0, :5].any()
        assert mask[1, 3:].all()
        assert not mask[1, :3].any()

    def test_combine_masks(self):
        m1 = torch.tensor([[False, True], [False, False]])
        m2 = torch.tensor([[False, False], [True, False]])
        combined = combine_masks(m1, m2)
        expected = torch.tensor([[False, True], [True, False]])
        assert torch.equal(combined, expected)

    def test_expand_mask_for_heads(self):
        mask = torch.ones(2, 4, 4, dtype=torch.bool)
        expanded = expand_mask_for_heads(mask, num_heads=8)
        assert expanded.shape == (2, 8, 4, 4)

    def test_alibi_mask(self):
        mask = create_alibi_mask(16, 8)
        assert mask.shape == (8, 16, 16)
        for h in range(8):
            assert torch.allclose(mask[h].triu(1), torch.zeros(16, 16))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])