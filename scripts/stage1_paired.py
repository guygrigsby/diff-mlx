"""Stage 1 paired run (full).

Builds vanilla + diff under the paired-seed init protocol (design §9.7),
runs each to the configured Stage 1 budget (2B tokens). Uses bf16 forward
(Stage 1 ModelConfig default) and resumes from latest.safetensors if a
previous attempt landed checkpoints.

Wrapped by scripts/stage1_paired.sh under caffeinate. Per-variant wall is
~9-12h on M5 Max with bf16, so the paired total is ~18-24h. If killed
mid-way, re-running picks up from the last checkpoint.
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

    model_cfg = ModelConfig.stage1()
    train_cfg = TrainConfig.stage1()

    tps = train_cfg.micro_batch * model_cfg.block_size * train_cfg.grad_accum
    total_steps = train_cfg.total_tokens // tps
    print(f"[stage1-paired] model: dim={model_cfg.dim} layers={model_cfg.n_layers} "
          f"T={model_cfg.block_size} amp_dtype={model_cfg.amp_dtype}")
    print(f"[stage1-paired] train: B={train_cfg.micro_batch} grad_accum={train_cfg.grad_accum} "
          f"peak_lr={train_cfg.peak_lr} warmup={train_cfg.warmup_steps}")
    print(f"[stage1-paired] {total_steps:,} steps/variant × {tps:,} tokens/step "
          f"= {train_cfg.total_tokens / 1e9:.1f}B tokens/variant")

    init_dir = args.out_root / f"init-stage1-seed{args.model_seed}"
    print(f"[stage1-paired] building paired init at {init_dir}")
    vanilla, diff = build_paired_models(model_cfg, seed=args.model_seed)
    save_paired_init(vanilla, diff, init_dir)

    vanilla_dir = args.out_root / f"stage1-paired-vanilla-seed{args.model_seed}"
    print(f"[stage1-paired] running vanilla at {vanilla_dir}")
    train_run(
        model_cfg, train_cfg, args.shards_dir, vanilla_dir,
        data_seed=args.data_seed, model_seed=args.model_seed,
        variant="vanilla",
        init_state_dict=vanilla.parameters(),
    )

    diff_dir = args.out_root / f"stage1-paired-diff-seed{args.model_seed}"
    print(f"[stage1-paired] running diff at {diff_dir}")
    train_run(
        model_cfg, train_cfg, args.shards_dir, diff_dir,
        data_seed=args.data_seed, model_seed=args.model_seed,
        variant="diff",
        init_state_dict=diff.parameters(),
    )

    print(f"[stage1-paired] done. Compare metrics in:")
    print(f"  {vanilla_dir}/metrics.jsonl")
    print(f"  {diff_dir}/metrics.jsonl")


if __name__ == "__main__":
    main()
