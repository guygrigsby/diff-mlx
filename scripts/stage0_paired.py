"""Stage 0 paired smoke run.

Build vanilla + diff with paired-seed init (design §9.7), train both to
completion at Stage 0 scale, save metrics for paired delta analysis.

Run from project root with venv activated.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from config import ModelConfig, TrainConfig
from paired_init import build_paired_models, save_paired_init
from train import train_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--model_seed", type=int, default=0)
    p.add_argument("--shards_dir", type=Path, default=Path("data/shards"))
    p.add_argument("--out_root", type=Path, default=Path("runs"))
    args = p.parse_args()

    model_cfg = ModelConfig.stage0()
    train_cfg = TrainConfig.stage0()

    init_dir = args.out_root / f"init-seed{args.model_seed}"
    print(f"[paired] building paired init at {init_dir}")
    vanilla, diff = build_paired_models(model_cfg, seed=args.model_seed)
    save_paired_init(vanilla, diff, init_dir)

    vanilla_dir = args.out_root / f"stage0-paired-vanilla-seed{args.model_seed}"
    print(f"[paired] running vanilla at {vanilla_dir}")
    train_run(
        model_cfg, train_cfg, args.shards_dir, vanilla_dir,
        data_seed=args.data_seed, model_seed=args.model_seed,
        variant="vanilla",
        init_state_dict=vanilla.parameters(),
    )

    diff_dir = args.out_root / f"stage0-paired-diff-seed{args.model_seed}"
    print(f"[paired] running diff at {diff_dir}")
    train_run(
        model_cfg, train_cfg, args.shards_dir, diff_dir,
        data_seed=args.data_seed, model_seed=args.model_seed,
        variant="diff",
        init_state_dict=diff.parameters(),
    )

    print(f"[paired] done. Compare final val losses in:")
    print(f"  {vanilla_dir}/metrics.jsonl")
    print(f"  {diff_dir}/metrics.jsonl")


if __name__ == "__main__":
    main()
