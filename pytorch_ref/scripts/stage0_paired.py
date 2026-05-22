"""Stage 0 paired runner. PyTorch port of ../../scripts/stage0_paired.py.

Builds vanilla + diff under the paired-seed init protocol, trains both to
Stage 0 budget (100M tokens) on CUDA, saves metrics for paired delta analysis.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch

from config import ModelConfig, TrainConfig
from paired_init import build_paired_models, save_paired_init
from train import train_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--model_seed", type=int, default=0)
    p.add_argument("--shards_dir", type=Path, default=Path("../data/shards"))
    p.add_argument("--out_root", type=Path, default=Path("runs"))
    p.add_argument("--no_amp", action="store_true")
    args = p.parse_args()

    model_cfg = ModelConfig.stage0()
    train_cfg = TrainConfig.stage0()
    autocast_dtype = None if args.no_amp else torch.bfloat16

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
        init_state_dict=vanilla.state_dict(),
        autocast_dtype=autocast_dtype,
    )

    diff_dir = args.out_root / f"stage0-paired-diff-seed{args.model_seed}"
    print(f"[paired] running diff at {diff_dir}")
    train_run(
        model_cfg, train_cfg, args.shards_dir, diff_dir,
        data_seed=args.data_seed, model_seed=args.model_seed,
        variant="diff",
        init_state_dict=diff.state_dict(),
        autocast_dtype=autocast_dtype,
    )

    print(f"[paired] done. Compare:")
    print(f"  {vanilla_dir}/metrics.jsonl")
    print(f"  {diff_dir}/metrics.jsonl")


if __name__ == "__main__":
    main()
