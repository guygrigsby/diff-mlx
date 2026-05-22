"""Shape and basic forward-pass sanity tests for the PyTorch port."""
import pytest
import torch

from model import RMSNorm, SwiGLU, VanillaMHA, DiffAttention, Block, Transformer


class _Cfg:
    """Minimal config stub matching the fields Transformer expects."""

    def __init__(self, dim, n_layers, n_heads_vanilla, qk_head_dim, vocab_size,
                 mlp_intermediate, block_size, rope_base=10000.0, rms_eps=1e-5):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads_vanilla = n_heads_vanilla
        self.qk_head_dim = qk_head_dim
        self.vocab_size = vocab_size
        self.mlp_intermediate = mlp_intermediate
        self.block_size = block_size
        self.rope_base = rope_base
        self.rms_eps = rms_eps


def test_rmsnorm_shape_preserved():
    x = torch.randn(2, 8, 16)
    norm = RMSNorm(16)
    y = norm(x)
    assert y.shape == x.shape


def test_rmsnorm_internal_fp32_returns_input_dtype():
    x = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    norm = RMSNorm(16)
    y = norm(x)
    assert y.dtype == torch.bfloat16


def test_swiglu_shape():
    mlp = SwiGLU(dim=32, intermediate=64)
    x = torch.randn(2, 8, 32)
    assert mlp(x).shape == (2, 8, 32)


def test_vanilla_mha_shape():
    attn = VanillaMHA(dim=64, n_heads=4)
    x = torch.randn(2, 16, 64)
    assert attn(x).shape == (2, 16, 64)


def test_diff_attention_shape():
    attn = DiffAttention(dim=64, n_heads_vanilla=4, qk_head_dim=16, layer_idx=1)
    x = torch.randn(2, 16, 64)
    assert attn(x).shape == (2, 16, 64)


def test_diff_attention_lambda_dtype_fp32():
    """Lambda vectors must stay fp32 regardless of forward dtype."""
    attn = DiffAttention(dim=64, n_heads_vanilla=4, qk_head_dim=16, layer_idx=1)
    assert attn.lambda_q1.dtype == torch.float32
    assert attn.lambda_k1.dtype == torch.float32
    assert attn.lambda_q2.dtype == torch.float32
    assert attn.lambda_k2.dtype == torch.float32


def test_block_vanilla_shape():
    b = Block(dim=64, n_heads_vanilla=4, qk_head_dim=16, mlp_intermediate=128, variant="vanilla")
    x = torch.randn(2, 16, 64)
    assert b(x).shape == (2, 16, 64)


def test_block_diff_shape():
    b = Block(dim=64, n_heads_vanilla=4, qk_head_dim=16, mlp_intermediate=128,
              variant="diff", layer_idx=1)
    x = torch.randn(2, 16, 64)
    assert b(x).shape == (2, 16, 64)


def test_transformer_vanilla_forward():
    cfg = _Cfg(dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
               vocab_size=128, mlp_intermediate=128, block_size=32)
    m = Transformer(cfg, variant="vanilla")
    tokens = torch.randint(0, 128, (2, 16))
    logits = m(tokens)
    assert logits.shape == (2, 16, 128)


def test_transformer_diff_forward():
    cfg = _Cfg(dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
               vocab_size=128, mlp_intermediate=128, block_size=32)
    m = Transformer(cfg, variant="diff")
    tokens = torch.randint(0, 128, (2, 16))
    logits = m(tokens)
    assert logits.shape == (2, 16, 128)


def test_transformer_param_counts_match_variants():
    """Vanilla and diff Transformers have the same param count by design (§6.3).
    H_diff halving + V_doubling preserves 4·dim² of attention weights.
    """
    cfg = _Cfg(dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
               vocab_size=128, mlp_intermediate=128, block_size=32)
    mv = Transformer(cfg, variant="vanilla")
    md = Transformer(cfg, variant="diff")
    nv = sum(p.numel() for p in mv.parameters())
    nd = sum(p.numel() for p in md.parameters())
    # Diff has 4 extra lambda vectors per layer (4 * qk_head_dim = 4*16 = 64) and
    # the subln scale (2*qk_head_dim = 32) instead of nothing in vanilla. So diff
    # has (64 + 32) * n_layers = 192 extra params.
    extra_per_layer = 4 * cfg.qk_head_dim + 2 * cfg.qk_head_dim
    expected_extra = extra_per_layer * cfg.n_layers
    assert nd - nv == expected_extra, f"vanilla={nv}, diff={nd}, expected diff = vanilla + {expected_extra}"
