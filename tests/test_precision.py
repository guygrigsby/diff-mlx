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
