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
    train_run(model_cfg, train_cfg, shards_dir, run_dir,
              data_seed=0, model_seed=0, variant="vanilla")
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "latest.safetensors").exists()
    lines = (run_dir / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) >= 3
    first = json.loads(lines[0])
    assert "step" in first and "train_loss" in first
    final = json.loads(lines[-1])
    assert np.isfinite(final["train_loss"])


def test_train_run_resumes_from_latest_checkpoint(tmp_path):
    """Run a tiny training for ~2 steps, then resume in a second train_run call.
    The second call should pick up past the first call's last step.
    """
    from train import train_run
    from config import ModelConfig, TrainConfig
    from dataclasses import replace
    import json
    import numpy as np

    model_cfg = ModelConfig(
        dim=32, n_layers=2, n_heads_vanilla=4, qk_head_dim=8,
        vocab_size=128, mlp_intermediate=64, block_size=16,
    )
    train_cfg = replace(
        TrainConfig.stage0(),
        total_tokens=64,         # micro_batch * block_size * 2 steps
        micro_batch=2,
        warmup_steps=0,
        eval_every=10_000,
        full_eval_every=10_000,
        save_every=1,
    )

    # Fake shards (uint32, train-NNN.bin / val.bin per data/loader.py).
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "meta.json").write_text(json.dumps({
        "vocab_size": 128, "tiktoken_version": "0.13.0", "tokenizer_name": "test",
    }))
    np.array(list(range(1024)) * 4, dtype=np.uint32).tofile(shards / "train-000.bin")
    np.array(list(range(1024)) * 4, dtype=np.uint32).tofile(shards / "val.bin")

    run_dir = tmp_path / "run"

    # First call: run from scratch.
    train_run(model_cfg, train_cfg, shards, run_dir,
              data_seed=0, model_seed=0, variant="vanilla")

    metrics_first = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(metrics_first) >= 1
    last_step_first = json.loads(metrics_first[-1])["step"]

    # Second call with more work: should resume past last_step_first.
    train_cfg_resumed = replace(train_cfg, total_tokens=128)
    train_run(model_cfg, train_cfg_resumed, shards, run_dir,
              data_seed=0, model_seed=0, variant="vanilla")

    metrics_second = (run_dir / "metrics.jsonl").read_text().splitlines()
    last_step_second = json.loads(metrics_second[-1])["step"]
    assert last_step_second > last_step_first, (
        f"resume did not advance: first ended at {last_step_first}, second at {last_step_second}"
    )

    # The second call should resume from where the first left off, not restart
    # from step 0. So the step logged immediately after metrics_first[-1] must be
    # last_step_first + 1, not 0.
    steps_added = [json.loads(l)["step"] for l in metrics_second[len(metrics_first):]]
    assert steps_added, "second train_run logged no new steps"
    assert steps_added[0] == last_step_first + 1, (
        f"expected second call to start at step {last_step_first + 1}, "
        f"but first new step was {steps_added[0]} (no resume?)"
    )
