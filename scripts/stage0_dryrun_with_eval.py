"""Stage 0 mini dry-run with eval enabled (~600 steps, exercises full pipeline).

Validates that:
- Monitoring eval triggers at step 100, 200, ..., 600
- Full eval triggers at step 300
- Checkpoint saves at step 200, 400, 600
- Loss curve is monotonic-ish on real data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace
from config import ModelConfig, TrainConfig
from train import train_run


def main():
    model_cfg = ModelConfig.stage0()
    # ~600 steps with frequent eval/save to exercise the full pipeline
    train_cfg = replace(
        TrainConfig.stage0(),
        total_tokens=600 * 16 * 1024,  # ~600 steps
        eval_every=100,
        full_eval_every=300,
        save_every=200,
        monitoring_tokens=500_000,    # smaller to keep eval fast in dry-run
        full_eval_tokens=2_000_000,
    )
    run_dir = Path("runs/stage0-dryrun-eval")
    train_run(model_cfg, train_cfg, Path("data/shards"), run_dir, seed=0, variant="vanilla")


if __name__ == "__main__":
    main()
