import json
import numpy as np
from pathlib import Path
from train import train_run
from config import ModelConfig, TrainConfig


def _make_synthetic_shards(tmp_path: Path, n_tokens: int = 200_000) -> Path:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    arr = (np.arange(n_tokens, dtype=np.uint32)) % 100_277
    arr.tofile(shards_dir / "train-0000.bin")
    arr[:10_000].tofile(shards_dir / "val.bin")
    (shards_dir / "meta.json").write_text(
        '{"vocab_size": 100277, "eot_id": 100257, "tiktoken_version": "0.8.0", '
        '"tokenizer_name": "cl100k_base", "train_token_count": 200000, '
        '"val_token_count": 10000, "n_train_shards": 1, "source_files": []}'
    )
    return shards_dir


def test_train_run_smoke_writes_metrics_and_checkpoint(tmp_path):
    shards_dir = _make_synthetic_shards(tmp_path)
    run_dir = tmp_path / "runs" / "smoke"
    model_cfg = ModelConfig(
        dim=32, n_layers=2, n_heads_vanilla=2, qk_head_dim=16,
        vocab_size=100_277, mlp_intermediate=64, block_size=64,
    )
    train_cfg = TrainConfig(
        peak_lr=1e-3, warmup_steps=10, total_tokens=50 * 64 * 2,  # ~50 steps
        micro_batch=2, eval_every=20, full_eval_every=50,
        monitoring_tokens=500, full_eval_tokens=2000, save_every=25,
    )
    train_run(model_cfg, train_cfg, shards_dir, run_dir, seed=0, variant="vanilla")
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "latest.safetensors").exists()
    lines = (run_dir / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) >= 3
    first = json.loads(lines[0])
    assert "step" in first and "train_loss" in first
    final = json.loads(lines[-1])
    assert np.isfinite(final["train_loss"])
