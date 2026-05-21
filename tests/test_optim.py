import mlx.core as mx
from optim import split_params_for_decay, to_bf16_view, to_bf16_dict
from model import Transformer
from config import ModelConfig


def _flatten(d, prefix=""):
    out = {}
    if hasattr(d, "items"):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}.{i}"))
    else:
        out[prefix] = d
    return out


def test_split_params_categorizes_correctly():
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    flat = _flatten(model.parameters())
    decay, no_decay = split_params_for_decay(flat)

    assert any("tok_embed.weight" in name for name in no_decay)
    assert any("norm" in name and "scale" in name for name in no_decay)
    assert any("final_norm.scale" in name for name in no_decay)
    assert any("mlp.gate.weight" in name for name in decay)
    assert any("mlp.up.weight" in name for name in decay)
    assert any("mlp.down.weight" in name for name in decay)
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert any(f"attn.{proj}.weight" in name for name in decay), f"missing {proj} in decay"


def test_split_params_no_overlap():
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    flat = _flatten(model.parameters())
    decay, no_decay = split_params_for_decay(flat)
    assert set(decay).isdisjoint(set(no_decay))
    assert set(decay) | set(no_decay) == set(flat.keys())


def test_to_bf16_view_casts_dtype():
    p_fp32 = mx.random.normal((4, 8), dtype=mx.float32)
    p_bf16 = to_bf16_view(p_fp32)
    assert p_bf16.dtype == mx.bfloat16
    assert p_bf16.shape == p_fp32.shape
    assert mx.allclose(p_bf16.astype(mx.float32), p_fp32, atol=1e-2).item()


def test_to_bf16_dict_recurses():
    d = {
        "a": mx.random.normal((4,), dtype=mx.float32),
        "b": {"c": mx.random.normal((4,), dtype=mx.float32)},
    }
    out = to_bf16_dict(d)
    assert out["a"].dtype == mx.bfloat16
    assert out["b"]["c"].dtype == mx.bfloat16
