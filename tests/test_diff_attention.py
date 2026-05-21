import math
import numpy as np
import mlx.core as mx
from model import DiffAttention, VanillaMHA


def _flatten(d, prefix=""):
    if hasattr(d, "items"):
        for k, v in d.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, d


def test_diff_attention_shape_preserved():
    attn = DiffAttention(dim=256, n_heads_vanilla=4, qk_head_dim=64, layer_idx=1)
    x = mx.random.normal((2, 32, 256), dtype=mx.float32)
    y = attn(x)
    assert y.shape == x.shape


def test_diff_attention_param_count():
    diff = DiffAttention(dim=256, n_heads_vanilla=4, qk_head_dim=64, layer_idx=1)
    vanilla = VanillaMHA(dim=256, n_heads=4)
    diff_params = sum(p.size for _, p in _flatten(diff.parameters()))
    vanilla_params = sum(p.size for _, p in _flatten(vanilla.parameters()))
    expected_proj = 4 * 256 * 256
    expected_lambdas = 4 * 64
    expected_subln = 2 * 64  # RMSNorm scale over 2D = 128
    assert vanilla_params == expected_proj
    assert diff_params == expected_proj + expected_lambdas + expected_subln


def test_diff_attention_no_bias_on_projections():
    attn = DiffAttention(dim=128, n_heads_vanilla=4, qk_head_dim=32, layer_idx=1)
    params = attn.parameters()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert "bias" not in params[name]


def test_diff_attention_derived_dims():
    attn = DiffAttention(dim=256, n_heads_vanilla=4, qk_head_dim=64, layer_idx=1)
    assert attn.n_heads_diff == 2
    assert attn.v_head_dim == 128


def test_diff_attention_lambda_close_to_lambda_init_at_init():
    mx.random.seed(0)
    attn = DiffAttention(dim=256, n_heads_vanilla=4, qk_head_dim=64, layer_idx=1)
    lam = attn._compute_lambda()
    expected = 0.8 - 0.6 * math.exp(-0.3 * (1 - 1))  # = 0.2
    assert abs(lam.item() - expected) < 0.5


def test_diff_attention_causal_property():
    mx.random.seed(0)
    attn = DiffAttention(dim=64, n_heads_vanilla=2, qk_head_dim=32, layer_idx=1)
    x1 = mx.random.normal((1, 16, 64), dtype=mx.float32)
    perturb = mx.random.normal((1, 8, 64), dtype=mx.float32)
    x2 = mx.concatenate([x1[:, :8, :], perturb], axis=1)
    y1 = attn(x1)
    y2 = attn(x2)
    assert mx.allclose(y1[0, :8, :], y2[0, :8, :], atol=1e-4).item()


def test_diff_attention_matches_sdpa_oracle():
    """v0 forward must equal the paper's explicit SDPA-composed oracle (design §5.2)."""
    mx.random.seed(42)
    attn = DiffAttention(dim=64, n_heads_vanilla=4, qk_head_dim=16, layer_idx=2)
    x = mx.random.normal((2, 8, 64), dtype=mx.float32)
    y_attn = attn(x)

    H = attn.n_heads_diff
    D = attn.qk_head_dim
    B, T, _ = x.shape
    q = attn.q_proj(x).reshape(B, T, 2 * H, D).transpose(0, 2, 1, 3)
    k = attn.k_proj(x).reshape(B, T, 2 * H, D).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(B, T, H, 2 * D).transpose(0, 2, 1, 3)
    # Interleaved split (must match module): heads viewed as (H, 2) -> Q1=q[2h], Q2=q[2h+1].
    q_pair = q.reshape(B, H, 2, T, D)
    k_pair = k.reshape(B, H, 2, T, D)
    q1, q2 = q_pair[:, :, 0, :, :], q_pair[:, :, 1, :, :]
    k1, k2 = k_pair[:, :, 0, :, :], k_pair[:, :, 1, :, :]
    q1 = mx.fast.rope(q1, dims=D, traditional=True, base=10000.0, scale=1.0, offset=0)
    q2 = mx.fast.rope(q2, dims=D, traditional=True, base=10000.0, scale=1.0, offset=0)
    k1 = mx.fast.rope(k1, dims=D, traditional=True, base=10000.0, scale=1.0, offset=0)
    k2 = mx.fast.rope(k2, dims=D, traditional=True, base=10000.0, scale=1.0, offset=0)
    scale = 1.0 / math.sqrt(D)
    out1 = mx.fast.scaled_dot_product_attention(q1, k1, v, scale=scale, mask="causal")
    out2 = mx.fast.scaled_dot_product_attention(q2, k2, v, scale=scale, mask="causal")
    lam = attn._compute_lambda()
    out = out1 - lam.astype(out1.dtype) * out2
    out = attn.subln(out)
    out = (1 - attn.lambda_init) * out
    out = out.transpose(0, 2, 1, 3).reshape(B, T, H * 2 * D)
    y_oracle = attn.o_proj(out)

    assert mx.allclose(y_attn, y_oracle, atol=1e-5).item()
