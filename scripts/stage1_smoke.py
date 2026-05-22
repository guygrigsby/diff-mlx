"""Stage 1 paired smoke. ~200 steps, both variants.

Validates the bf16 mixed-precision path, optimizer-state checkpointing,
and grad_accum routing at Stage 1 shapes (768 dim, 12 layers, B=32, T=2048)
before committing to a full single-seed run (~18 hours per variant).

NOT a convergence run. With Stage 1's 1000-step warmup, lr at step 200 is
only ~0.2 * peak_lr = ~8e-5. Loss won't move much. Goals:

  - No NaN/Inf in 200 steps under bf16 forward.
  - Peak MLX memory stays inside unified-memory budget.
  - Sensible step times (the throughput sanity check; thermals shouldn't
    matter on a sub-30-minute run, but caffeinate is wrapping anyway).
  - Optimizer-state checkpoint round-trips on save (validated by the
    auto-resume code path; the smoke writes a checkpoint at save_every).

Run from project root with venv activated. Wrapped by stage1_smoke.sh.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from dataclasses import replace
from config import ModelConfig, TrainConfig
from paired_init import build_paired_models, save_paired_init
from train import train_run


SMOKE_STEPS = 200


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--model_seed", type=int, default=0)
    p.add_argument("--shards_dir", type=Path, default=Path("data/shards"))
    p.add_argument("--out_root", type=Path, default=Path("runs/stage1-smoke"))
    p.add_argument("--steps", type=int, default=SMOKE_STEPS,
                   help="Override the smoke step count. Default 200.")
    args = p.parse_args()

    model_cfg = ModelConfig.stage1()
    base_train = TrainConfig.stage1()

    tokens_per_step = base_train.micro_batch * model_cfg.block_size * base_train.grad_accum
    smoke_total_tokens = args.steps * tokens_per_step

    train_cfg = replace(
        base_train,
        total_tokens=smoke_total_tokens,
        eval_every=50,
        full_eval_every=10_000,
        save_every=100,
    )

    print(f"[smoke] Stage 1 paired smoke: {args.steps} steps × "
          f"{tokens_per_step} tokens/step = {smoke_total_tokens / 1e6:.1f}M tokens per variant")
    print(f"[smoke] model_cfg.amp_dtype={model_cfg.amp_dtype}  "
          f"grad_accum={train_cfg.grad_accum}  warmup={train_cfg.warmup_steps}")

    init_dir = args.out_root / f"init-seed{args.model_seed}"
    print(f"[smoke] building paired init at {init_dir}")
    vanilla, diff = build_paired_models(model_cfg, seed=args.model_seed)
    save_paired_init(vanilla, diff, init_dir)

    vanilla_dir = args.out_root / f"stage1-smoke-vanilla-seed{args.model_seed}"
    print(f"[smoke] running vanilla at {vanilla_dir}")
    train_run(
        model_cfg, train_cfg, args.shards_dir, vanilla_dir,
        data_seed=args.data_seed, model_seed=args.model_seed,
        variant="vanilla",
        init_state_dict=vanilla.parameters(),
    )

    diff_dir = args.out_root / f"stage1-smoke-diff-seed{args.model_seed}"
    print(f"[smoke] running diff at {diff_dir}")
    train_run(
        model_cfg, train_cfg, args.shards_dir, diff_dir,
        data_seed=args.data_seed, model_seed=args.model_seed,
        variant="diff",
        init_state_dict=diff.parameters(),
    )

    print(f"[smoke] done. Compare metrics in:")
    print(f"  {vanilla_dir}/metrics.jsonl")
    print(f"  {diff_dir}/metrics.jsonl")


if __name__ == "__main__":
    main()
