from pathlib import Path
import mlx.core as mx
from paired_init import build_paired_models, save_paired_init, load_paired_init
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


def test_paired_init_shared_weights_byte_identical():
    """All weights with shapes that match across variants must be byte-identical."""
    cfg = ModelConfig.stage0()
    vanilla, diff = build_paired_models(cfg, seed=0)
    flat_v = _flatten(vanilla.parameters())
    flat_d = _flatten(diff.parameters())
    shared_names = [n for n in flat_v if n in flat_d and flat_v[n].shape == flat_d[n].shape]
    assert len(shared_names) > 0, "no shared weight names found — check naming"
    for name in shared_names:
        assert mx.array_equal(flat_v[name], flat_d[name]).item(), f"mismatch at {name}"


def test_paired_init_diff_has_extra_lambda_and_subln_params():
    """Diff-side has params that vanilla doesn't: lambda vectors and subln scale."""
    cfg = ModelConfig.stage0()
    _, diff = build_paired_models(cfg, seed=0)
    flat_d = _flatten(diff.parameters())
    assert any("lambda_q1" in n for n in flat_d)
    assert any("lambda_k1" in n for n in flat_d)
    assert any("lambda_q2" in n for n in flat_d)
    assert any("lambda_k2" in n for n in flat_d)
    assert any("subln" in n for n in flat_d)


def test_paired_init_same_seed_gives_same_lambdas():
    """Diff-only params should be deterministic given the same seed."""
    cfg = ModelConfig.stage0()
    _, diff_a = build_paired_models(cfg, seed=42)
    _, diff_b = build_paired_models(cfg, seed=42)
    flat_a = _flatten(diff_a.parameters())
    flat_b = _flatten(diff_b.parameters())
    for n, v in flat_a.items():
        if "lambda_" in n:
            assert mx.array_equal(v, flat_b[n]).item(), f"non-deterministic lambda at {n}"


def test_paired_init_different_seeds_give_different_lambdas():
    cfg = ModelConfig.stage0()
    _, diff_a = build_paired_models(cfg, seed=0)
    _, diff_b = build_paired_models(cfg, seed=1)
    flat_a = _flatten(diff_a.parameters())
    flat_b = _flatten(diff_b.parameters())
    # At least one lambda vector should differ (otherwise seed isn't doing anything)
    any_diff = False
    for n, v in flat_a.items():
        if "lambda_" in n and not mx.array_equal(v, flat_b[n]).item():
            any_diff = True
            break
    assert any_diff, "lambdas should differ across seeds"


def test_save_and_load_paired_init_roundtrip(tmp_path):
    cfg = ModelConfig.stage0()
    vanilla, diff = build_paired_models(cfg, seed=0)
    save_paired_init(vanilla, diff, tmp_path)
    assert (tmp_path / "vanilla.safetensors").exists()
    assert (tmp_path / "diff.safetensors").exists()
    flat_v_before = _flatten(vanilla.parameters())
    flat_d_before = _flatten(diff.parameters())
    v_loaded, d_loaded = load_paired_init(cfg, tmp_path)
    flat_v_after = _flatten(v_loaded.parameters())
    flat_d_after = _flatten(d_loaded.parameters())
    for n, v in flat_v_before.items():
        assert mx.array_equal(v, flat_v_after[n]).item(), f"vanilla mismatch at {n}"
    for n, v in flat_d_before.items():
        assert mx.array_equal(v, flat_d_after[n]).item(), f"diff mismatch at {n}"


def test_paired_init_diff_lambda_init_constants_are_correct():
    """Diff blocks should have layer_idx-derived lambda_init values."""
    cfg = ModelConfig.stage0()
    _, diff = build_paired_models(cfg, seed=0)
    # Block i (0-indexed list) has layer_idx = i+1
    assert diff.blocks[0].attn.layer_idx == 1
    assert diff.blocks[-1].attn.layer_idx == cfg.n_layers
