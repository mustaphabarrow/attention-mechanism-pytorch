"""
Multi-Head Attention Implementation

Learnable Q, K, V projections with multi-head attention.
Supports both self-attention and cross-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from attention.base import scaled_dot_product_attention, AttentionBase


class MultiHeadAttention(AttentionBase):
    """Multi-Head Attention as described in "Attention Is All You Need"."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ):
        super().__init__(embed_dim, num_heads, dropout, bias, batch_first)
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if key is None:
            key = query
        if value is None:
            value = key

        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch, tgt_len, _ = query.shape
        src_len = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self._reshape_for_heads(q)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            key_padding_mask = key_padding_mask.expand(-1, self.num_heads, tgt_len, -1)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

        combined_mask = None
        if attn_mask is not None and key_padding_mask is not None:
            combined_mask = attn_mask | key_padding_mask
        elif attn_mask is not None:
            combined_mask = attn_mask
        elif key_padding_mask is not None:
            combined_mask = key_padding_mask

        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v,
            attn_mask=combined_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self.scale,
        )

        attn_output = self._combine_heads(attn_output)
        output = self.out_proj(attn_output)

        if not self.batch_first:
            output = output.transpose(0, 1)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return output, attn_weights
        return output, None


class CrossAttention(MultiHeadAttention):
    """Cross-Attention (Encoder-Decoder Attention)"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        encoder_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            kdim=encoder_dim,
            vdim=encoder_dim,
        )

    def forward(
        self,
        query: torch.Tensor,
        encoder_output: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return super().forward(
            query=query,
            key=encoder_output,
            value=encoder_output,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
            is_causal=False,
        )