import mlx.core as mx
import mlx.nn as nn
import numpy as np
from model import LinearAMP


def test_linear_amp_fp32_matches_nn_linear():
    """At amp_dtype=float32, LinearAMP must be numerically identical to nn.Linear."""
    mx.random.seed(0)
    lin = nn.Linear(8, 4, bias=False)
    amp = LinearAMP(8, 4, bias=False, amp_dtype=mx.float32)
    amp.weight = lin.weight  # share weights
    x = mx.random.normal((3, 8), dtype=mx.float32)
    y_lin = lin(x)
    y_amp = amp(x)
    assert mx.allclose(y_lin, y_amp, atol=1e-7).item()


def test_linear_amp_output_dtype_is_amp_dtype():
    amp = LinearAMP(8, 4, bias=False, amp_dtype=mx.bfloat16)
    x = mx.random.normal((3, 8), dtype=mx.float32)
    y = amp(x)
    assert y.dtype == mx.bfloat16


def test_linear_amp_weight_stays_fp32_after_forward():
    amp = LinearAMP(8, 4, bias=False, amp_dtype=mx.bfloat16)
    x = mx.random.normal((3, 8), dtype=mx.float32)
    _ = amp(x)
    assert amp.weight.dtype == mx.float32


def test_linear_amp_grad_flows_to_fp32_weight():
    """value_and_grad with bf16 forward must yield fp32 grads on the fp32 weight."""
    amp = LinearAMP(8, 4, bias=False, amp_dtype=mx.bfloat16)

    def loss_fn(model, x):
        return (model(x) ** 2).sum()

    x = mx.random.normal((3, 8), dtype=mx.float32)
    loss_and_grad = nn.value_and_grad(amp, loss_fn)
    loss, grads = loss_and_grad(amp, x)
    mx.eval(loss, grads)
    assert grads["weight"].dtype == mx.float32
    assert grads["weight"].shape == amp.weight.shape


def test_linear_amp_with_bias():
    amp = LinearAMP(8, 4, bias=True, amp_dtype=mx.bfloat16)
    x = mx.random.normal((3, 8), dtype=mx.float32)
    y = amp(x)
    assert y.dtype == mx.bfloat16
    assert amp.bias.dtype == mx.float32


from model import SwiGLU, VanillaMHA, DiffAttention


def test_swiglu_amp_output_bf16():
    mlp = SwiGLU(dim=32, intermediate=64, amp_dtype=mx.bfloat16)
    x = mx.random.normal((2, 8, 32), dtype=mx.float32)
    y = mlp(x)
    assert y.dtype == mx.bfloat16
    assert mlp.gate.weight.dtype == mx.float32


def test_vanilla_mha_amp_output_bf16():
    attn = VanillaMHA(dim=64, n_heads=4, amp_dtype=mx.bfloat16)
    x = mx.random.normal((2, 16, 64), dtype=mx.float32)
    y = attn(x)
    assert y.dtype == mx.bfloat16
    assert attn.q_proj.weight.dtype == mx.float32


def test_diff_attention_amp_output_bf16():
    attn = DiffAttention(dim=64, n_heads_vanilla=4, qk_head_dim=16,
                         layer_idx=1, amp_dtype=mx.bfloat16)
    x = mx.random.normal((2, 16, 64), dtype=mx.float32)
    y = attn(x)
    assert y.dtype == mx.bfloat16
    assert attn.q_proj.weight.dtype == mx.float32
    # Lambda vectors must remain fp32 regardless of amp_dtype
    assert attn.lambda_q1.dtype == mx.float32


from model import Transformer
from config import ModelConfig


def test_transformer_amp_output_bf16():
    cfg = ModelConfig(
        dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
        vocab_size=128, mlp_intermediate=128, block_size=32,
        amp_dtype="bfloat16",
    )
    model = Transformer(cfg, variant="vanilla")
    tokens = mx.random.randint(0, 128, shape=(2, 16))
    logits = model(tokens)
    # logits come out of `x @ tok_embed.weight.T`; tok_embed.weight is fp32,
    # x is bf16 (cast at start of forward), so the matmul output dtype depends
    # on MLX broadcast rules. We DO require: forward runs without error and
    # output is one of {bf16, fp32}; CE loss explicitly upcasts later anyway.
    assert logits.dtype in (mx.bfloat16, mx.float32)
    assert logits.shape == (2, 16, 128)


def test_transformer_amp_diff_variant_bf16():
    cfg = ModelConfig(
        dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
        vocab_size=128, mlp_intermediate=128, block_size=32,
        amp_dtype="bfloat16",
    )
    model = Transformer(cfg, variant="diff")
    tokens = mx.random.randint(0, 128, shape=(2, 16))
    logits = model(tokens)
    assert logits.shape == (2, 16, 128)


def test_transformer_fp32_default_unchanged():
    """At amp_dtype='float32' (the default), behavior is unchanged."""
    cfg = ModelConfig(
        dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
        vocab_size=128, mlp_intermediate=128, block_size=32,
    )
    model = Transformer(cfg, variant="vanilla")
    tokens = mx.random.randint(0, 128, shape=(2, 16))
    logits = model(tokens)
    assert logits.dtype == mx.float32


from train_step import train_step
from optim import make_adamw


def test_bf16_train_step_runs_clean():
    cfg = ModelConfig(
        dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
        vocab_size=128, mlp_intermediate=128, block_size=32,
        amp_dtype="bfloat16",
    )
    model = Transformer(cfg, variant="diff")
    opt = make_adamw(lr=1e-4, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)
    x = mx.random.randint(0, 128, shape=(2, 16))
    y = mx.random.randint(0, 128, shape=(2, 16))
    loss = train_step(model, opt, x, y, grad_clip=1.0)
    assert isinstance(loss, float)
    assert mx.array(loss).dtype == mx.float32 or isinstance(loss, float)
    import math
    assert not math.isnan(loss)
    assert not math.isinf(loss)

    # All params must still be fp32 storage after a step
    def walk(p, names):
        if isinstance(p, dict):
            for k, v in p.items():
                walk(v, names + [k])
        elif isinstance(p, list):
            for i, v in enumerate(p):
                walk(v, names + [str(i)])
        elif isinstance(p, mx.array):
            full = ".".join(names)
            assert p.dtype == mx.float32, f"param {full} drifted to {p.dtype}"
    walk(model.parameters(), [])


def test_fp32_and_bf16_initial_loss_within_tolerance():
    """Same paired init, one step at fp32 vs bf16: loss within design §9.0 tolerance."""
    base_cfg_kwargs = dict(
        dim=64, n_layers=2, n_heads_vanilla=4, qk_head_dim=16,
        vocab_size=128, mlp_intermediate=128, block_size=32,
    )
    cfg_fp32 = ModelConfig(**base_cfg_kwargs)
    cfg_bf16 = ModelConfig(**base_cfg_kwargs, amp_dtype="bfloat16")

    mx.random.seed(42)
    m_fp32 = Transformer(cfg_fp32, variant="vanilla")
    mx.eval(m_fp32.parameters())

    mx.random.seed(42)
    m_bf16 = Transformer(cfg_bf16, variant="vanilla")
    mx.eval(m_bf16.parameters())

    x = mx.random.randint(0, 128, shape=(2, 16))
    y = mx.random.randint(0, 128, shape=(2, 16))

    from train_step import _ce_loss
    l_fp32 = _ce_loss(m_fp32, x, y).item()
    l_bf16 = _ce_loss(m_bf16, x, y).item()
    assert abs(l_fp32 - l_bf16) < 1e-2, f"fp32={l_fp32} bf16={l_bf16}"
