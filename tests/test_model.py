import numpy as np
import mlx.core as mx
from model import RMSNorm, SwiGLU, VanillaMHA, Block, Transformer
from config import ModelConfig


def _flatten_params(d, prefix=""):
    if hasattr(d, "items"):
        for k, v in d.items():
            yield from _flatten_params(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _flatten_params(v, f"{prefix}[{i}]")
    else:
        yield prefix, d


# RMSNorm

def test_rmsnorm_shape_preserved():
    x = mx.random.normal((2, 8, 16), dtype=mx.float32)
    norm = RMSNorm(dim=16)
    y = norm(x)
    assert y.shape == x.shape


def test_rmsnorm_matches_manual_formula():
    x = mx.random.normal((4, 32), dtype=mx.float32)
    norm = RMSNorm(dim=32, eps=1e-5)
    y = norm(x)
    expected = x / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
    assert mx.allclose(y, expected, atol=1e-6).item()


def test_rmsnorm_scale_is_learnable_and_init_to_one():
    norm = RMSNorm(dim=64)
    params = norm.parameters()
    assert "scale" in params
    assert mx.array_equal(params["scale"], mx.ones(64)).item()


# SwiGLU

def test_swiglu_shape_and_dtype():
    cfg_dim, cfg_intermediate = 256, 704
    mlp = SwiGLU(dim=cfg_dim, intermediate=cfg_intermediate)
    x = mx.random.normal((2, 16, cfg_dim), dtype=mx.float32)
    y = mlp(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_swiglu_no_bias():
    mlp = SwiGLU(dim=128, intermediate=352)
    params = mlp.parameters()
    for name in ("gate", "up", "down"):
        assert name in params, f"missing {name}"
        sub = params[name]
        assert "weight" in sub
        assert "bias" not in sub


def test_swiglu_param_count():
    dim, intermediate = 256, 704
    mlp = SwiGLU(dim=dim, intermediate=intermediate)
    total = sum(p.size for _, p in _flatten_params(mlp.parameters()))
    expected = 3 * dim * intermediate
    assert total == expected


# VanillaMHA

def test_vanilla_mha_shape():
    attn = VanillaMHA(dim=256, n_heads=4)
    x = mx.random.normal((2, 32, 256), dtype=mx.float32)
    y = attn(x)
    assert y.shape == x.shape


def test_vanilla_mha_param_count_4dim2():
    attn = VanillaMHA(dim=256, n_heads=4)
    total = sum(p.size for _, p in _flatten_params(attn.parameters()))
    assert total == 4 * 256 * 256


def test_vanilla_mha_no_bias():
    attn = VanillaMHA(dim=128, n_heads=4)
    params = attn.parameters()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert "bias" not in params[name]


def test_vanilla_mha_causal_property():
    """Output at position t must not depend on inputs at positions > t."""
    mx.random.seed(0)
    attn = VanillaMHA(dim=64, n_heads=2)
    x1 = mx.random.normal((1, 16, 64), dtype=mx.float32)
    perturb = mx.random.normal((1, 8, 64), dtype=mx.float32)
    # Build x2 by replacing tokens 8.. of x1 with the perturbation
    x2 = mx.concatenate([x1[:, :8, :], perturb], axis=1)
    y1 = attn(x1)
    y2 = attn(x2)
    assert mx.allclose(y1[0, :8, :], y2[0, :8, :], atol=1e-4).item()


# Block

def test_block_shape():
    block = Block(dim=128, n_heads_vanilla=4, qk_head_dim=32, mlp_intermediate=352)
    x = mx.random.normal((2, 16, 128), dtype=mx.float32)
    y = block(x)
    assert y.shape == x.shape


def test_block_has_norms_attn_mlp():
    block = Block(dim=128, n_heads_vanilla=4, qk_head_dim=32, mlp_intermediate=352)
    params = block.parameters()
    assert "norm_attn" in params
    assert "norm_mlp" in params
    assert "attn" in params
    assert "mlp" in params


def test_block_diff_variant_shape():
    block = Block(
        dim=128, n_heads_vanilla=4, qk_head_dim=32, mlp_intermediate=352,
        variant="diff", layer_idx=1,
    )
    x = mx.random.normal((2, 16, 128), dtype=mx.float32)
    y = block(x)
    assert y.shape == x.shape


def test_block_variant_picks_correct_attn_class():
    from model import DiffAttention as _DiffAttention
    block_v = Block(dim=128, n_heads_vanilla=4, qk_head_dim=32, mlp_intermediate=352)
    block_d = Block(
        dim=128, n_heads_vanilla=4, qk_head_dim=32, mlp_intermediate=352,
        variant="diff", layer_idx=1,
    )
    assert isinstance(block_v.attn, VanillaMHA)
    assert isinstance(block_d.attn, _DiffAttention)


def test_block_diff_requires_layer_idx():
    import pytest
    with pytest.raises(AssertionError, match="layer_idx"):
        Block(
            dim=128, n_heads_vanilla=4, qk_head_dim=32, mlp_intermediate=352,
            variant="diff",  # layer_idx omitted
        )


def test_transformer_diff_variant_forward_shape():
    cfg = ModelConfig.stage0()
    model = Transformer(cfg, variant="diff")
    x = mx.array(np.random.randint(0, cfg.vocab_size, size=(2, 64), dtype=np.int32))
    logits = model(x)
    assert logits.shape == (2, 64, cfg.vocab_size)


def test_transformer_diff_blocks_are_diffattention():
    from model import DiffAttention as _DiffAttention
    cfg = ModelConfig.stage0()
    model = Transformer(cfg, variant="diff")
    assert all(isinstance(b.attn, _DiffAttention) for b in model.blocks)


def test_transformer_diff_layer_idx_is_1_indexed():
    cfg = ModelConfig.stage0()
    model = Transformer(cfg, variant="diff")
    assert model.blocks[0].attn.layer_idx == 1
    assert model.blocks[-1].attn.layer_idx == cfg.n_layers


# Transformer

def test_transformer_stage0_forward_shape():
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    x = mx.array(np.random.randint(0, cfg.vocab_size, size=(2, 64), dtype=np.int32))
    logits = model(x)
    assert logits.shape == (2, 64, cfg.vocab_size)


def test_transformer_stage0_param_count_approx():
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    total = sum(p.size for _, p in _flatten_params(model.parameters()))
    # Stage 0: embed 100277*256 = 25.67M, transformer body ~4.8M, total ~30.5M
    # nn.Embedding stores weight separately, not duplicated as lm_head, so count once
    assert 28_000_000 < total < 32_000_000, f"unexpected param count: {total:,}"


def test_transformer_final_rmsnorm_exists():
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    params = model.parameters()
    assert "final_norm" in params


def test_transformer_tied_embeddings_no_separate_lm_head():
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    params = model.parameters()
    # Tied means no separate lm_head module; embed.weight is used directly via .T
    assert "tok_embed" in params
    assert "lm_head" not in params
