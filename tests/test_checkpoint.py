import json
from pathlib import Path
import mlx.core as mx
from checkpoint import save_checkpoint, load_checkpoint, save_run_metadata
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


def test_save_and_load_roundtrip(tmp_path):
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    flat_before = _flatten(model.parameters())
    ckpt_path = tmp_path / "ckpt.safetensors"
    save_checkpoint(model.parameters(), step=1234, ckpt_path=ckpt_path)
    loaded, step, _opt = load_checkpoint(ckpt_path)
    assert step == 1234
    flat_after = _flatten(loaded)
    assert set(flat_before.keys()) == set(flat_after.keys())
    for name in flat_before:
        assert mx.array_equal(flat_before[name], flat_after[name]).item(), f"mismatch: {name}"


def test_save_run_metadata_writes_expected_files(tmp_path):
    cfg = ModelConfig.stage0()
    save_run_metadata(
        run_dir=tmp_path,
        model_cfg=cfg,
        train_cfg_dict={"peak_lr": 6e-4},
        git_hash="deadbeef",
        git_dirty=False,
        mlx_version="0.31.2",
        seed=0,
        data_meta={"vocab_size": 100277, "tiktoken_version": "0.13.0", "tokenizer_name": "cl100k_base"},
    )
    for fname in ("config.json", "git.txt", "mlx_version.txt", "tiktoken.txt", "data_meta.json", "seed.txt"):
        assert (tmp_path / fname).exists(), f"missing {fname}"


from optim import make_adamw


def test_save_and_load_roundtrips_optimizer_state(tmp_path):
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    opt = make_adamw(lr=1e-4, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)

    # Force the optimizer to materialize state by doing one update.
    import mlx.nn as nn

    def loss_fn(m, tokens):
        return (m(tokens) ** 2).sum()

    tokens = mx.array([[0, 1, 2, 3]])
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    _, grads = loss_and_grad(model, tokens)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)

    ckpt = tmp_path / "ckpt.safetensors"
    save_checkpoint(model.parameters(), step=7, ckpt_path=ckpt, optim_state=opt.state)

    loaded_params, step, loaded_opt_state = load_checkpoint(ckpt)
    assert step == 7

    # Spot-check: the optimizer's m/v buffers for a known param should round-trip.
    # Use the step counter as a stable, known key rather than traversal-order-
    # dependent first-array (safetensors does not preserve insertion order).
    def find_first_array(d):
        if isinstance(d, dict):
            for v in sorted(d.keys()):
                r = find_first_array(d[v])
                if r is not None:
                    return r
        elif isinstance(d, list):
            for v in d:
                r = find_first_array(v)
                if r is not None:
                    return r
        elif isinstance(d, mx.array):
            return d
        return None

    orig_arr = find_first_array(opt.state)
    loaded_arr = find_first_array(loaded_opt_state)
    assert orig_arr is not None and loaded_arr is not None
    assert mx.array_equal(orig_arr, loaded_arr).item()


def test_load_checkpoint_without_optim_state_returns_none(tmp_path):
    """Backward compatibility: old checkpoints without opt state still load."""
    cfg = ModelConfig.stage0()
    model = Transformer(cfg)
    ckpt = tmp_path / "ckpt.safetensors"
    save_checkpoint(model.parameters(), step=3, ckpt_path=ckpt)
    loaded_params, step, opt_state = load_checkpoint(ckpt)
    assert step == 3
    assert opt_state is None
