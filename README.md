# Attention Mechanism Implementation

A comprehensive educational project implementing attention mechanisms from scratch using PyTorch, progressing from basic scaled dot-product attention to multi-head attention with Flash Attention optimization.

## Project Structure

```
attention-mechanism/
├── notebooks/
│   ├── 01_attention_intro.ipynb      # Introduction to attention concepts
│   └── 02_attention_implementation.ipynb  # Full implementation with executed examples
├── src/
│   └── attention/
│       ├── __init__.py
│       ├── base.py           # Base attention classes
│       ├── multihead.py      # Multi-head attention implementation
│       ├── flash.py          # Flash Attention / fused attention
│       └── masks.py          # Attention mask utilities
├── tests/
│   ├── __init__.py
│   └── test_attention.py     # Unit tests
├── requirements.txt
└── .gitignore
```

## Notebooks

### 01_attention_intro.ipynb
- Introduction to attention networks and Transformers
- Comparison with RNNs
- Basic tensor setup (Batch × Sequence Length × Embed Dim)

### 02_attention_implementation.ipynb
Complete implementation covering:
- **Scaled Dot-Product Attention** (from scratch, no learnable params)
- **Multi-Head Attention** with learnable Q/K/V projections
- **Cross-Attention** (encoder-decoder) with padding masks
- **Causal Masking** for autoregressive models
- **Flash Attention** via `F.scaled_dot_product_attention`
- Working examples with English/French translation tensors

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 2.0+

## Running the Notebooks

```bash
jupyter notebook notebooks/
```

## Key Implementations

### Base Attention (`src/attention/base.py`)
```python
from attention.base import scaled_dot_product_attention

output, attn_weights = scaled_dot_product_attention(q, k, v, mask=None)
```

### Multi-Head Attention (`src/attention/multihead.py`)
```python
from attention.multihead import MultiHeadAttention

mha = MultiHeadAttention(embed_dim=768, num_heads=12)
output = mha(x, attention_mask=mask)
```

### Self-Attention with Flash (`src/attention/flash.py`)
```python
from attention.flash import SelfAttention

self_attn = SelfAttention(embed_dim=768, num_heads=12, fused_attn=True)
output = self_attn(x, attention_mask=mask, is_causal=True)
```

## GitHub Repository

Suggested repo name: **`attention-mechanism-pytorch`** or **`transformer-attention-from-scratch`**

## License

MIT License - Feel free to use for learning and research.