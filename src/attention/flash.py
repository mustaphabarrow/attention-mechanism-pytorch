"""
Flash Attention / Fused Attention Implementation

Uses PyTorch's optimized scaled_dot_product_attention (Flash Attention v2)
for memory-efficient and fast attention computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from attention.base import AttentionBase


class SelfAttention(AttentionBase):
    """
    Self-Attention with Flash Attention support.

    Uses PyTorch's F.scaled_dot_product_attention which automatically
    uses Flash Attention kernels when available (CUDA, appropriate dtype).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        causal: bool = False,
        fused_attn: bool = True,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__(embed_dim, num_heads, attn_dropout, bias, batch_first)
        self.causal = causal
        self.fused = fused_attn
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.proj_drop = nn.Dropout(proj_dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.constant_(self.qkv_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional Flash Attention.

        Args:
            x: (B, L, E) input tensor
            attention_mask: (B, L, S) or (L, S) boolean/float mask
            is_causal: Override causal setting for this forward pass

        Returns:
            output: (B, L, E) attended output
        """
        if not self.batch_first:
            x = x.transpose(0, 1)

        batch, seq_len, embed_dim = x.shape

        # Fused QKV projection
        qkv = self.qkv_proj(x)  # (B, L, 3*E)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, L, Hd)

        # Determine causal setting
        use_causal = self.causal if is_causal is None else is_causal

        if self.fused:
            # Use PyTorch's Flash Attention (scaled_dot_product_attention)
            # Automatically uses Flash Attention v2 on CUDA with fp16/bf16
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=use_causal,
            )
        else:
            # Manual implementation (for comparison/debugging)
            attn = (q @ k.transpose(-2, -1)) * self.scale

            if use_causal:
                causal_mask = torch.ones(
                    seq_len, seq_len, dtype=torch.bool, device=x.device
                ).triu(1)
                attn = attn.masked_fill(causal_mask, float('-inf'))

            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    attn = attn.masked_fill(attention_mask, float('-inf'))
                else:
                    attn = attn + attention_mask

            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
            x = attn @ v

        # Combine heads: (B, H, L, Hd) -> (B, L, E)
        x = x.transpose(1, 2).reshape(batch, seq_len, embed_dim).contiguous()

        # Output projection
        x = self.out_proj(x)
        x = self.proj_drop(x)

        if not self.batch_first:
            x = x.transpose(0, 1)

        return x