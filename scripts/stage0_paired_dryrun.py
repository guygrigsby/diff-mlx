"""Stage 0 paired DRY-RUN (~30 steps each variant, ~30s total).

Validates the orchestration: paired init builds, both runs start, both produce
metrics.jsonl with descending loss. Use before the full ~2.5h paired run.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace
from config import ModelConfig, TrainConfig
from paired_init import build_paired_models
from train import train_run


def main():
    model_cfg = ModelConfig.stage0()
    train_cfg = replace(TrainConfig.stage0(), total_tokens=30 * 16 * 1024)

    vanilla, diff = build_paired_models(model_cfg, seed=0)

    train_run(
        model_cfg, train_cfg, Path("data/shards"), Path("runs/stage0-paired-dryrun-vanilla"),
        data_seed=0, model_seed=0, variant="vanilla", init_state_dict=vanilla.parameters(),
    )
    train_run(
        model_cfg, train_cfg, Path("data/shards"), Path("runs/stage0-paired-dryrun-diff"),
        data_seed=0, model_seed=0, variant="diff", init_state_dict=diff.parameters(),
    )


if __name__ == "__main__":
    main()
