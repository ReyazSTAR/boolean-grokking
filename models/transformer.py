"""
Minimal transformer for boolean grokking experiments.
Designed for full mechanistic interpretability access:
  - Every weight matrix is accessible by name
  - Activations can be hooked at any layer
  - No dropout (deterministic for analysis)
  - No bias terms (cleaner weight analysis)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    vocab_size: int       # number of tokens (e.g. 2 for {0,1} + operators + special)
    seq_len: int          # fixed input length
    d_model: int = 64     # embedding dimension
    n_heads: int = 2      # attention heads
    n_layers: int = 1     # number of transformer blocks
    d_mlp: int = 256      # MLP hidden dimension (4x d_model)
    use_mlp: bool = True  # set False for attention-only ablation


class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.d_head = cfg.d_model // cfg.n_heads

        # Named explicitly for interpretability access
        self.W_Q = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, self.d_head) / math.sqrt(cfg.d_model))
        self.W_K = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, self.d_head) / math.sqrt(cfg.d_model))
        self.W_V = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, self.d_head) / math.sqrt(cfg.d_model))
        self.W_O = nn.Parameter(torch.randn(cfg.n_heads, self.d_head, cfg.d_model) / math.sqrt(cfg.d_model))

        # Stored for analysis
        self.attn_scores = None
        self.attn_pattern = None

    def forward(self, x):
        # x: (batch, seq, d_model)
        B, S, D = x.shape

        # Compute Q, K, V for all heads
        q = torch.einsum('bsd,hde->bshe', x, self.W_Q)  # (B, S, H, d_head)
        k = torch.einsum('bsd,hde->bshe', x, self.W_K)
        v = torch.einsum('bsd,hde->bshe', x, self.W_V)

        # Attention scores
        scores = torch.einsum('bshe,bthe->bhst', q, k) / math.sqrt(self.d_head)
        self.attn_scores = scores.detach()  # save for analysis

        pattern = F.softmax(scores, dim=-1)
        self.attn_pattern = pattern.detach()  # save for analysis

        # Weighted sum of values
        z = torch.einsum('bhst,bthe->bshe', pattern, v)

        # Project back to d_model
        out = torch.einsum('bshe,hed->bsd', z, self.W_O)
        return out


class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.W_in  = nn.Parameter(torch.randn(cfg.d_model, cfg.d_mlp) / math.sqrt(cfg.d_model))
        self.W_out = nn.Parameter(torch.randn(cfg.d_mlp, cfg.d_model) / math.sqrt(cfg.d_mlp))
        self.b_in  = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

        self.pre_act = None  # save for analysis

    def forward(self, x):
        pre = x @ self.W_in + self.b_in
        self.pre_act = pre.detach()
        return F.gelu(pre) @ self.W_out + self.b_out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg) if cfg.use_mlp else None
        self.ln2 = nn.LayerNorm(cfg.d_model) if cfg.use_mlp else None

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class BooleanTransformer(nn.Module):
    """
    Transformer for boolean logic tasks.

    Input format: [a, op, b, =]  -> predict result token
    Example: XOR(1, 0) = [1, XOR_TOKEN, 0, EQ_TOKEN] -> 1
    """
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.W_E = nn.Embedding(cfg.vocab_size, cfg.d_model)  # token embedding
        self.W_pos = nn.Embedding(cfg.seq_len, cfg.d_model)   # positional embedding

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.W_U = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # unembed

        # Residual stream cache (for analysis)
        self.residual_cache = {}

    def forward(self, tokens):
        # tokens: (batch, seq_len)
        B, S = tokens.shape

        pos = torch.arange(S, device=tokens.device).unsqueeze(0)
        x = self.W_E(tokens) + self.W_pos(pos)

        self.residual_cache['embed'] = x.detach()

        for i, block in enumerate(self.blocks):
            x = block(x)
            self.residual_cache[f'block_{i}'] = x.detach()

        x = self.ln_final(x)
        logits = self.W_U(x)  # (B, S, vocab_size)

        # We only care about the prediction at the last position (after "=")
        return logits[:, -1, :]  # (B, vocab_size)

    def get_attention_pattern(self, layer=0):
        """Returns attention pattern for a given layer. Shape: (batch, heads, seq, seq)"""
        return self.blocks[layer].attn.attn_pattern

    def get_residual_stream(self, layer='embed'):
        return self.residual_cache.get(layer)
