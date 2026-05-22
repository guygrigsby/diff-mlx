"""PyTorch port of the MLX model. Mirrors `../model.py` line-for-line.

Same paper-canonical decomposition: interleaved head split, GPT-J-style RoPE,
two-SDPA v0 diff-attn, fp32 lambda math, fp32 RMSNorm internals, fp32 logits.

bf16 training uses ``torch.autocast``; the model itself stays fp32 in storage.
This matches the MLX side's "fp32 params, bf16 cast inside forward" pattern
(`LinearAMP` in ../model.py) but uses PyTorch's idiomatic autocast instead of
a custom Linear subclass. Storage stays fp32 either way.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm over the LAST dim. Learned scale, no bias. Internal fp32, output dtype matches input."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp32 = x.float()
        rms = torch.sqrt((x_fp32 * x_fp32).mean(dim=-1, keepdim=True) + self.eps)
        out = x_fp32 / rms
        return (out * self.scale.float()).to(in_dtype)


class SwiGLU(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x)). bias=False on all linears."""

    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.gate = nn.Linear(dim, intermediate, bias=False)
        self.up = nn.Linear(dim, intermediate, bias=False)
        self.down = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype=torch.float32):
    """Returns (cos, sin) of shape (seq_len, head_dim // 2). fp32 for precision."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """GPT-J-style interleaved RoPE. Matches MLX `mx.fast.rope(traditional=True)`
    and the vendored reference's `apply_rotary_emb(..., interleaved=True)`.

    Args:
      x: (B, H, T, D), D even
      cos, sin: (T, D // 2) in fp32

    Returns:
      (B, H, T, D), same dtype as x.
    """
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    cos_b = cos[None, None, :, :].to(x.dtype)
    sin_b = sin[None, None, :, :].to(x.dtype)
    o0 = x0 * cos_b - x1 * sin_b
    o1 = x0 * sin_b + x1 * cos_b
    return torch.stack([o0, o1], dim=-1).flatten(-2)


class VanillaMHA(nn.Module):
    """Standard MHA: q/k/v/o all (dim, dim), bias=False. RoPE on q/k. Causal."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = _build_rope_cache(T, self.head_dim, self.rope_base, x.device)
        q = apply_rope_interleaved(q, cos, sin)
        k = apply_rope_interleaved(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, T, self.dim)
        return self.o_proj(out)


def lambda_init_for_layer(layer_idx: int) -> float:
    """Paper-canonical depth schedule (1-indexed). λ_init = 0.8 - 0.6 * exp(-0.3 * (layer_idx - 1))."""
    return 0.8 - 0.6 * math.exp(-0.3 * (layer_idx - 1))


class DiffAttention(nn.Module):
    """Paper-canonical Differential Attention (Ye et al., ICLR 2025).

    Same shape discipline as the MLX side (`../model.py`):
      - n_heads_diff = n_heads_vanilla // 2
      - qk_head_dim = D (same as vanilla head_dim)
      - v_head_dim = 2 * D
      - subln = RMSNorm over 2D applied per-head AFTER differential subtraction
      - lambda = exp(λq1·λk1) - exp(λq2·λk2) + λ_init (scalar, fp32)
      - Output scaled by (1 - λ_init) before o_proj
      - RoPE applied to Q1/K1/Q2/K2 independently (interleaved/GPT-J style)
      - v0 forward: two SDPA calls with shared V at width 2D, subtract outputs
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
        assert n_heads_vanilla % 2 == 0
        assert n_heads_vanilla * qk_head_dim == dim
        self.dim = dim
        self.n_heads_vanilla = n_heads_vanilla
        self.n_heads_diff = n_heads_vanilla // 2
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = 2 * qk_head_dim
        self.layer_idx = layer_idx
        self.rope_base = rope_base
        self.scale = 1.0 / math.sqrt(qk_head_dim)
        self.lambda_init = lambda_init_for_layer(layer_idx)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Lambda vectors stay fp32 regardless of forward dtype.
        self.lambda_q1 = nn.Parameter(torch.randn(qk_head_dim, dtype=torch.float32) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(qk_head_dim, dtype=torch.float32) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(qk_head_dim, dtype=torch.float32) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(qk_head_dim, dtype=torch.float32) * 0.1)

        self.subln = RMSNorm(self.v_head_dim, eps=rms_eps)

    def _compute_lambda(self) -> torch.Tensor:
        """λ scalar in fp32. exp(λq1·λk1) - exp(λq2·λk2) + λ_init."""
        l1 = torch.exp((self.lambda_q1.float() * self.lambda_k1.float()).sum())
        l2 = torch.exp((self.lambda_q2.float() * self.lambda_k2.float()).sum())
        return l1 - l2 + self.lambda_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H = self.n_heads_diff
        D = self.qk_head_dim

        q = self.q_proj(x).view(B, T, 2 * H, D).transpose(1, 2)  # (B, 2H, T, D)
        k = self.k_proj(x).view(B, T, 2 * H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, 2 * D).transpose(1, 2)  # (B, H, T, 2D)

        # Paper-canonical interleaved split: the 2H heads are viewed as (H, 2)
        # in row-major order. diff-head h pairs Q1 = q[2h], Q2 = q[2h+1].
        q_pair = q.view(B, H, 2, T, D)
        k_pair = k.view(B, H, 2, T, D)
        q1, q2 = q_pair[:, :, 0], q_pair[:, :, 1]
        k1, k2 = k_pair[:, :, 0], k_pair[:, :, 1]

        cos, sin = _build_rope_cache(T, D, self.rope_base, x.device)
        q1 = apply_rope_interleaved(q1, cos, sin)
        q2 = apply_rope_interleaved(q2, cos, sin)
        k1 = apply_rope_interleaved(k1, cos, sin)
        k2 = apply_rope_interleaved(k2, cos, sin)

        out1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=True, scale=self.scale)
        out2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=True, scale=self.scale)

        lam = self._compute_lambda()
        out = out1 - lam.to(out1.dtype) * out2

        out = self.subln(out)
        out = (1.0 - self.lambda_init) * out

        out = out.transpose(1, 2).reshape(B, T, H * 2 * D)  # = (B, T, dim)
        return self.o_proj(out)


class Block(nn.Module):
    """Pre-norm transformer block. Variant selects attention type."""

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
        assert dim == n_heads_vanilla * qk_head_dim
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class Transformer(nn.Module):
    """Pre-norm LLaMA-style transformer with tied embeddings.

    forward(tokens: (B, T) long) -> logits: (B, T, vocab_size)
    """

    def __init__(self, cfg, variant: str = "vanilla"):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([
            Block(
                dim=cfg.dim,
                n_heads_vanilla=cfg.n_heads_vanilla,
                qk_head_dim=cfg.qk_head_dim,
                mlp_intermediate=cfg.mlp_intermediate,
                variant=variant,
                layer_idx=(i + 1),
                rope_base=cfg.rope_base,
                rms_eps=cfg.rms_eps,
            )
            for i in range(cfg.n_layers)
        ])
        self.final_norm = RMSNorm(cfg.dim, eps=cfg.rms_eps)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok_embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = x @ self.tok_embed.weight.T
        return logits
