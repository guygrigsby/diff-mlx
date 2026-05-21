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
    loaded, step = load_checkpoint(ckpt_path)
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
