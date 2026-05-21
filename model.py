"""Transformer model: backbone + vanilla MHA (Phase A) + diff-attn (Phase B).

Phase A scope: shared backbone + VanillaMHALayer only.
"""
from __future__ import annotations
import math
import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm: x * scale / sqrt(mean(x^2) + eps). Learned scale, no bias.

    Normalizes the LAST dimension only. Apply to (B, T, dim) -> normalize over dim,
    or to (B, H, T, head_dim) -> normalize over head_dim (per-head; no cross-head mixing).
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        in_dtype = x.dtype
        x_fp32 = x.astype(mx.float32)
        rms = mx.sqrt(mx.mean(x_fp32 * x_fp32, axis=-1, keepdims=True) + self.eps)
        out = x_fp32 / rms
        return (out * self.scale.astype(mx.float32)).astype(in_dtype)


class SwiGLU(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x)). All linears bias=False."""
    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.gate = nn.Linear(dim, intermediate, bias=False)
        self.up = nn.Linear(dim, intermediate, bias=False)
        self.down = nn.Linear(intermediate, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class VanillaMHA(nn.Module):
    """Standard MHA: q/k/v/o all (dim, dim), bias=False. RoPE on q/k. Causal mask.

    Uses mx.fast.rope (traditional=False = LLaMA rotate-halves) and
    mx.fast.scaled_dot_product_attention (mask="causal", scale required kw-only).
    """
    def __init__(self, dim: int, n_heads: int, rope_base: float = 10000.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope_base = rope_base
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, dims=self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        k = mx.fast.rope(k, dims=self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.dim)
        return self.o_proj(out)


class Block(nn.Module):
    """Pre-norm transformer block: x = x + attn(norm(x)); x = x + mlp(norm(x))."""
    def __init__(self, dim: int, n_heads: int, mlp_intermediate: int,
                 rope_base: float = 10000.0, rms_eps: float = 1e-5):
        super().__init__()
        self.norm_attn = RMSNorm(dim, eps=rms_eps)
        self.attn = VanillaMHA(dim, n_heads, rope_base=rope_base)
        self.norm_mlp = RMSNorm(dim, eps=rms_eps)
        self.mlp = SwiGLU(dim, mlp_intermediate)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class Transformer(nn.Module):
    """Pre-norm LLaMA-style transformer with tied embeddings.

    forward(token_ids: (B, T)) -> logits: (B, T, vocab_size)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = [
            Block(
                dim=cfg.dim,
                n_heads=cfg.n_heads_vanilla,
                mlp_intermediate=cfg.mlp_intermediate,
                rope_base=cfg.rope_base,
                rms_eps=cfg.rms_eps,
            )
            for _ in range(cfg.n_layers)
        ]
        self.final_norm = RMSNorm(cfg.dim, eps=cfg.rms_eps)
        # Tied embeddings: no separate lm_head linear; forward uses embed.weight.T

    def __call__(self, tokens: mx.array) -> mx.array:
        x = self.tok_embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = x @ self.tok_embed.weight.T
        return logits


def lambda_init_for_layer(layer_idx: int) -> float:
    """Paper-canonical lambda_init depth schedule (1-indexed layer).

        lambda_init = 0.8 - 0.6 * exp(-0.3 * (layer_idx - 1))

    Layer 1 → 0.2; approaches 0.8 with depth. Per design §6.3 / paper §2.2.
    """
    return 0.8 - 0.6 * math.exp(-0.3 * (layer_idx - 1))
