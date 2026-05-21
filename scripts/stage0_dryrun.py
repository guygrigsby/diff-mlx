"""Stage 0 dry-run: ~50 steps on real shards. Validates pipeline before committing
to the full ~1-day Stage 0 run. Run from project root with venv activated.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace
from config import ModelConfig, TrainConfig
from train import train_run


def main():
    model_cfg = ModelConfig.stage0()
    train_cfg = replace(TrainConfig.stage0(), total_tokens=50 * 16 * 1024)
    run_dir = Path("runs/stage0-dryrun")
    train_run(model_cfg, train_cfg, Path("data/shards"), run_dir, seed=0, variant="vanilla")


if __name__ == "__main__":
    main()
