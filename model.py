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

    Uses mx.fast.rope (traditional=True = LLaMA rotate-halves) and
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
        q = mx.fast.rope(q, dims=self.head_dim, traditional=True, base=self.rope_base, scale=1.0, offset=0)
        k = mx.fast.rope(k, dims=self.head_dim, traditional=True, base=self.rope_base, scale=1.0, offset=0)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.dim)
        return self.o_proj(out)


class DiffAttention(nn.Module):
    """Paper-canonical Differential Attention (Ye et al., ICLR 2025).

    Per design §6.3:
    - n_heads_diff = n_heads_vanilla // 2
    - qk_head_dim = D (same as vanilla head_dim)
    - v_head_dim = 2 * D
    - All projections dim → dim (same total widths as vanilla)
    - subln = RMSNorm over 2D applied per-head AFTER differential subtraction
    - lambda = exp(dot(λ_q1, λ_k1)) - exp(dot(λ_q2, λ_k2)) + λ_init  (scalar, per-forward)
    - Output scaled by (1 - λ_init) before o_proj
    - RoPE via mx.fast.rope(traditional=True) on Q1/K1/Q2/K2 independently
    - SDPA via mx.fast.scaled_dot_product_attention(scale=1/√D, mask="causal")

    v0 forward: two SDPA calls with shared V at width 2D, subtract outputs
    (design §7.1 linearity rewrite — no T×T map materialization).
    """
    def __init__(
        self,
        dim: int,
        n_heads_vanilla: int,
        qk_head_dim: int,
        layer_idx: int,
        rope_base: float = 10000.0,
        rms_eps: float = 1e-5,
    ):
        super().__init__()
        assert n_heads_vanilla % 2 == 0, "n_heads_vanilla must be even (paired into diff heads)"
        assert n_heads_vanilla * qk_head_dim == dim, "dim must equal n_heads_vanilla * qk_head_dim"
        self.dim = dim
        self.n_heads_vanilla = n_heads_vanilla
        self.n_heads_diff = n_heads_vanilla // 2
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = 2 * qk_head_dim
        self.layer_idx = layer_idx
        self.rope_base = rope_base
        self.scale = 1.0 / math.sqrt(qk_head_dim)
        self.lambda_init = lambda_init_for_layer(layer_idx)

        # Projections (all dim → dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Lambda vectors (fp32, init randn * 0.1 per design §6.3)
        self.lambda_q1 = mx.random.normal((qk_head_dim,), dtype=mx.float32) * 0.1
        self.lambda_k1 = mx.random.normal((qk_head_dim,), dtype=mx.float32) * 0.1
        self.lambda_q2 = mx.random.normal((qk_head_dim,), dtype=mx.float32) * 0.1
        self.lambda_k2 = mx.random.normal((qk_head_dim,), dtype=mx.float32) * 0.1

        # subln: RMSNorm over the V head width (2D), per-head application
        self.subln = RMSNorm(self.v_head_dim, eps=rms_eps)

    def _compute_lambda(self) -> mx.array:
        """λ = exp(dot(λ_q1, λ_k1)) - exp(dot(λ_q2, λ_k2)) + λ_init (scalar, fp32)."""
        l1 = mx.exp(mx.sum(self.lambda_q1.astype(mx.float32) * self.lambda_k1.astype(mx.float32)))
        l2 = mx.exp(mx.sum(self.lambda_q2.astype(mx.float32) * self.lambda_k2.astype(mx.float32)))
        return l1 - l2 + self.lambda_init

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape
        H = self.n_heads_diff
        D = self.qk_head_dim

        q = self.q_proj(x).reshape(B, T, 2 * H, D).transpose(0, 2, 1, 3)  # (B, 2H, T, D)
        k = self.k_proj(x).reshape(B, T, 2 * H, D).transpose(0, 2, 1, 3)  # (B, 2H, T, D)
        v = self.v_proj(x).reshape(B, T, H, 2 * D).transpose(0, 2, 1, 3)  # (B, H, T, 2D)

        # Paper-canonical split (matches microsoft/unilm reference): the 2H heads are
        # viewed as (H, 2) in row-major order, so diff-head h pairs Q1 = q[2h], Q2 = q[2h+1]
        # (NOT halves split q[:H] vs q[H:] — that's a different layout). See design §7.4
        # cross-check test for the discriminator.
        q_pair = q.reshape(B, H, 2, T, D)
        k_pair = k.reshape(B, H, 2, T, D)
        q1, q2 = q_pair[:, :, 0, :, :], q_pair[:, :, 1, :, :]
        k1, k2 = k_pair[:, :, 0, :, :], k_pair[:, :, 1, :, :]

        q1 = mx.fast.rope(q1, dims=D, traditional=True, base=self.rope_base, scale=1.0, offset=0)
        q2 = mx.fast.rope(q2, dims=D, traditional=True, base=self.rope_base, scale=1.0, offset=0)
        k1 = mx.fast.rope(k1, dims=D, traditional=True, base=self.rope_base, scale=1.0, offset=0)
        k2 = mx.fast.rope(k2, dims=D, traditional=True, base=self.rope_base, scale=1.0, offset=0)

        out1 = mx.fast.scaled_dot_product_attention(q1, k1, v, scale=self.scale, mask="causal")
        out2 = mx.fast.scaled_dot_product_attention(q2, k2, v, scale=self.scale, mask="causal")

        lam = self._compute_lambda()
        out = out1 - lam.astype(out1.dtype) * out2

        out = self.subln(out)
        out = (1.0 - self.lambda_init) * out

        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * 2 * D)  # H*2D = dim
        return self.o_proj(out)


class Block(nn.Module):
    """Pre-norm transformer block. Variant selects attention type.

    variant="vanilla": uses VanillaMHA(dim, n_heads_vanilla)
    variant="diff":    uses DiffAttention(dim, n_heads_vanilla, qk_head_dim, layer_idx)

    qk_head_dim is required and must satisfy dim == n_heads_vanilla * qk_head_dim.
    layer_idx is required for variant="diff" (1-indexed).
    """
    def __init__(
        self,
        dim: int,
        n_heads_vanilla: int,
        qk_head_dim: int,
        mlp_intermediate: int,
        variant: str = "vanilla",
        layer_idx: int | None = None,
        rope_base: float = 10000.0,
        rms_eps: float = 1e-5,
    ):
        super().__init__()
        assert dim == n_heads_vanilla * qk_head_dim, (
            f"dim={dim} != n_heads_vanilla*qk_head_dim={n_heads_vanilla * qk_head_dim}"
        )
        self.norm_attn = RMSNorm(dim, eps=rms_eps)
        if variant == "vanilla":
            self.attn = VanillaMHA(dim, n_heads_vanilla, rope_base=rope_base)
        elif variant == "diff":
            assert layer_idx is not None, "variant='diff' requires layer_idx (1-indexed)"
            self.attn = DiffAttention(
                dim=dim,
                n_heads_vanilla=n_heads_vanilla,
                qk_head_dim=qk_head_dim,
                layer_idx=layer_idx,
                rope_base=rope_base,
                rms_eps=rms_eps,
            )
        else:
            raise ValueError(f"unknown variant {variant!r}; expected 'vanilla' or 'diff'")
        self.norm_mlp = RMSNorm(dim, eps=rms_eps)
        self.mlp = SwiGLU(dim, mlp_intermediate)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class Transformer(nn.Module):
    """Pre-norm LLaMA-style transformer with tied embeddings.

    variant="vanilla" uses VanillaMHA in every block (Phase A baseline).
    variant="diff"    uses DiffAttention in every block (Phase B+).

    forward(token_ids: (B, T)) -> logits: (B, T, vocab_size)
    """
    def __init__(self, cfg, variant: str = "vanilla"):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = [
            Block(
                dim=cfg.dim,
                n_heads_vanilla=cfg.n_heads_vanilla,
                qk_head_dim=cfg.qk_head_dim,
                mlp_intermediate=cfg.mlp_intermediate,
                variant=variant,
                layer_idx=(i + 1),  # 1-indexed for paper's lambda_init schedule
                rope_base=cfg.rope_base,
                rms_eps=cfg.rms_eps,
            )
            for i in range(cfg.n_layers)
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
