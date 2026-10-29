"""
Attention Mask Utilities

Helper functions for creating various attention masks:
- Causal masks (autoregressive)
- Padding masks (for variable length sequences)
- Combined masks
"""

import torch
from typing import Optional, Tuple


def create_causal_mask(
    seq_len: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Create causal (upper triangular) mask for autoregressive attention.

    Args:
        seq_len: Sequence length
        device: Device to create mask on
        dtype: Data type (bool for masked_fill, float for additive)

    Returns:
        mask: (seq_len, seq_len) upper triangular mask
              True/1.0 for positions to mask (future tokens)
    """
    mask = torch.ones(seq_len, seq_len, device=device, dtype=dtype).triu(1)
    if dtype == torch.bool:
        return mask
    return mask * torch.finfo(dtype).min


def create_padding_mask(
    lengths: torch.Tensor,
    max_len: Optional[int] = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Create padding mask from sequence lengths.

    Args:
        lengths: (B,) tensor of actual sequence lengths
        max_len: Maximum sequence length (default: max(lengths))
        dtype: Data type for mask

    Returns:
        mask: (B, max_len) True/1.0 for padded positions
    """
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = lengths.max().item()

    # Create position indices
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    # Mask where position >= length
    mask = positions >= lengths.unsqueeze(1)

    if dtype != torch.bool:
        mask = mask.to(dtype) * torch.finfo(dtype).min

    return mask


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Create sliding window (local) attention mask.

    Args:
        seq_len: Sequence length
        window_size: Window size (attend to window_size//2 on each side)
        device: Device
        dtype: Data type

    Returns:
        mask: (seq_len, seq_len) mask for local attention
    """
    mask = torch.ones(seq_len, seq_len, device=device, dtype=dtype)
    half_window = window_size // 2

    for i in range(seq_len):
        start = max(0, i - half_window)
        end = min(seq_len, i + half_window + 1)
        if dtype == torch.bool:
            mask[i, start:end] = False
        else:
            mask[i, start:end] = 0

    return mask


def combine_masks(
    *masks: Optional[torch.Tensor],
    dtype: torch.dtype = torch.bool,
) -> Optional[torch.Tensor]:
    """
    Combine multiple attention masks with logical OR (for bool) or addition (for float).

    Args:
        *masks: Variable number of masks to combine
        dtype: Output dtype

    Returns:
        Combined mask or None if all inputs are None
    """
    valid_masks = [m for m in masks if m is not None]
    if not valid_masks:
        return None

    combined = valid_masks[0]
    for mask in valid_masks[1:]:
        if dtype == torch.bool:
            combined = combined | mask
        else:
            combined = combined + mask

    return combined


def expand_mask_for_heads(
    mask: torch.Tensor,
    num_heads: int,
    target_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Expand mask for multi-head attention.

    Args:
        mask: (B, L, S) or (B, 1, L, S) or (L, S)
        num_heads: Number of attention heads
        target_len: Target sequence length for query dim

    Returns:
        Expanded mask: (B, H, L, S)
    """
    if mask.dim() == 2:
        # (L, S) -> (1, 1, L, S)
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        # (B, L, S) -> (B, 1, L, S)
        mask = mask.unsqueeze(1)

    if mask.shape[1] == 1:
        # Expand to all heads
        mask = mask.expand(-1, num_heads, -1, -1)

    return mask


def create_alibi_mask(
    seq_len: int,
    num_heads: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create ALiBi (Attention with Linear Biases) slopes mask.

    Args:
        seq_len: Sequence length
        num_heads: Number of heads (each gets different slope)
        device: Device

    Returns:
        mask: (num_heads, seq_len, seq_len) ALiBi bias matrix
    """
    # ALiBi slopes: 2^(-8 * i / num_heads) for i=1..num_heads
    slopes = torch.tensor([
        2 ** (-8 * (i + 1) / num_heads)
        for i in range(num_heads)
    ], device=device)

    # Create position difference matrix
    positions = torch.arange(seq_len, device=device)
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)

    # Apply slopes: slope * -|i-j| (causal: only j <= i)
    causal_diff = diff.tril(0)  # Lower triangular including diagonal
    alibi = slopes.unsqueeze(-1).unsqueeze(-1) * causal_diff  # (H, L, L)

    return alibi