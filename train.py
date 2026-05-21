"""Training driver. Single GPU, single seed. Phase A: vanilla MHA only."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import mlx.core as mx

from config import ModelConfig, TrainConfig
from model import Transformer
from data.loader import ShardLoader, sample_batch
from train_step import train_step
from eval import compute_val_loss
from schedule import cosine_lr_with_warmup
from optim import make_adamw
from metrics import MetricsLogger
from checkpoint import save_checkpoint, save_run_metadata
from gitinfo import current_hash, is_dirty


def _mlx_version() -> str:
    try:
        import mlx
        return getattr(mlx, "__version__", "unknown")
    except Exception:
        return "unknown"


def train_run(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    shards_dir: Path,
    run_dir: Path,
    seed: int = 0,
    variant: str = "vanilla",
) -> None:
    """Run one training stage to completion. Writes metrics, checkpoints, metadata."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = Path(shards_dir)

    mx.random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    train_loader = ShardLoader(shards_dir, "train")
    val_loader = ShardLoader(shards_dir, "val")
    data_meta = json.loads((shards_dir / "meta.json").read_text())

    model = Transformer(model_cfg)
    optimizer = make_adamw(
        lr=0.0,
        weight_decay=train_cfg.weight_decay,
        beta1=train_cfg.adam_beta1,
        beta2=train_cfg.adam_beta2,
        eps=train_cfg.adam_eps,
    )

    save_run_metadata(
        run_dir=run_dir, model_cfg=model_cfg,
        train_cfg_dict=asdict(train_cfg),
        git_hash=current_hash(), git_dirty=is_dirty(),
        mlx_version=_mlx_version(),
        seed=seed, data_meta=data_meta,
    )
    (run_dir / "variant.txt").write_text(variant + "\n")

    eff_tokens_per_step = train_cfg.micro_batch * model_cfg.block_size * train_cfg.grad_accum
    total_steps = max(1, train_cfg.total_tokens // eff_tokens_per_step)
    print(f"[train] {total_steps} steps, ~{eff_tokens_per_step} tokens/step, "
          f"{train_cfg.total_tokens / 1e6:.1f}M total tokens")

    logger = MetricsLogger(run_dir / "metrics.jsonl")
    t0 = time.time()
    step = 0
    while step < total_steps:
        lr = cosine_lr_with_warmup(
            step, train_cfg.peak_lr, train_cfg.warmup_steps, total_steps,
            min_lr_frac=0.1,
        )
        optimizer.learning_rate = lr

        x_np, y_np = sample_batch(train_loader, model_cfg.block_size, train_cfg.micro_batch, rng)
        x = mx.array(x_np)
        y = mx.array(y_np)
        loss = train_step(model, optimizer, x, y, grad_clip=train_cfg.grad_clip)

        do_full_eval = (step > 0 and step % train_cfg.full_eval_every == 0)
        do_monitor_eval = (step > 0 and step % train_cfg.eval_every == 0)
        record = {
            "step": step,
            "train_loss": loss,
            "lr": lr,
            "tps": int(eff_tokens_per_step * (step + 1) / max(1e-6, time.time() - t0)),
            "wall": round(time.time() - t0, 1),
        }
        if do_monitor_eval:
            val = compute_val_loss(model, val_loader, model_cfg.block_size,
                                   train_cfg.micro_batch, train_cfg.monitoring_tokens)
            record["val_loss_monitor"] = val
        if do_full_eval:
            val_full = compute_val_loss(model, val_loader, model_cfg.block_size,
                                        train_cfg.micro_batch, train_cfg.full_eval_tokens)
            record["val_loss_full"] = val_full
        logger.log(**record)

        if (step > 0 and step % train_cfg.save_every == 0) or step == total_steps - 1:
            save_checkpoint(model.parameters(), step=step, ckpt_path=run_dir / "latest.safetensors")

        step += 1
    logger.close()
    print(f"[train] done in {time.time() - t0:.1f}s")


def _build_cfgs(stage: str) -> tuple[ModelConfig, TrainConfig]:
    return getattr(ModelConfig, stage)(), getattr(TrainConfig, stage)()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["stage0", "stage1", "stage2"], required=True)
    p.add_argument("--shards_dir", type=Path, default=Path("data/shards"))
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", choices=["vanilla", "diff"], default="vanilla")
    args = p.parse_args()
    if args.variant != "vanilla":
        raise SystemExit("Phase A: only vanilla supported. Diff lands in Phase B.")
    model_cfg, train_cfg = _build_cfgs(args.stage)
    train_run(model_cfg, train_cfg, args.shards_dir, args.run_dir, args.seed, args.variant)


if __name__ == "__main__":
    main()
