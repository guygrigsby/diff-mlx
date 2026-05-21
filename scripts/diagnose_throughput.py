"""Throughput diagnostic for the Stage 0 diff-attn slowdown.

Runs 500 diff steps + logs per-step wall time, MLX peak memory, and macOS
memory pressure indicators. Designed to be short (~5 min) so we can repeat it
under different conditions (clear_cache on/off, smaller block_size, etc.).

Usage:
    python scripts/diagnose_throughput.py --steps 500 --variant diff
    python scripts/diagnose_throughput.py --steps 500 --variant vanilla   # baseline
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import subprocess
import time
from dataclasses import replace

import mlx.core as mx
import numpy as np

from config import ModelConfig, TrainConfig
from model import Transformer
from data.loader import ShardLoader, sample_batch
from train_step import train_step
from optim import make_adamw
from schedule import cosine_lr_with_warmup


def macos_memory_stats() -> dict:
    """Get macOS memory pressure stats via vm_stat. Cheap; ~10ms."""
    try:
        out = subprocess.check_output(["vm_stat"], text=True, timeout=2)
        stats = {}
        for line in out.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip().rstrip(".")
            try:
                stats[k.strip()] = int(v)
            except ValueError:
                pass
        return stats
    except Exception as e:
        return {"error": str(e)}


def mlx_peak_memory_mb() -> float:
    """Peak MLX-tracked Metal memory in MB."""
    try:
        return mx.metal.get_peak_memory() / 1e6
    except Exception:
        return -1.0


def mlx_active_memory_mb() -> float:
    try:
        return mx.metal.get_active_memory() / 1e6
    except Exception:
        return -1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--variant", choices=["vanilla", "diff"], default="diff")
    p.add_argument("--clear_cache_every", type=int, default=0,
                   help="Call mx.metal.clear_cache() every N steps. 0 disables.")
    p.add_argument("--shards_dir", type=Path, default=Path("data/shards"))
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSONL path (default: runs/diag-<variant>-<timestamp>.jsonl)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.out is None:
        ts = int(time.time())
        args.out = Path(f"runs/diag-{args.variant}-{ts}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model_cfg = ModelConfig.stage0()
    train_cfg = replace(TrainConfig.stage0(), total_tokens=args.steps * 16 * 1024)
    train_loader = ShardLoader(args.shards_dir, "train")

    model = Transformer(model_cfg, variant=args.variant)
    optimizer = make_adamw(
        lr=0.0,
        weight_decay=train_cfg.weight_decay,
        beta1=train_cfg.adam_beta1,
        beta2=train_cfg.adam_beta2,
        eps=train_cfg.adam_eps,
    )

    print(f"[diag] variant={args.variant} steps={args.steps} clear_cache_every={args.clear_cache_every}")
    print(f"[diag] output: {args.out}")

    f = args.out.open("w", buffering=1)
    t0 = time.time()
    last_t = t0

    for step in range(args.steps):
        # Snapshot pre-step memory + macOS
        pre_active = mlx_active_memory_mb()
        pre_peak = mlx_peak_memory_mb()

        lr = cosine_lr_with_warmup(
            step, train_cfg.peak_lr, train_cfg.warmup_steps, args.steps,
            min_lr_frac=0.1,
        )
        optimizer.learning_rate = lr

        x_np, y_np = sample_batch(train_loader, model_cfg.block_size, train_cfg.micro_batch, rng)
        x = mx.array(x_np)
        y = mx.array(y_np)

        step_t0 = time.time()
        loss = train_step(model, optimizer, x, y, grad_clip=train_cfg.grad_clip)
        step_wall = time.time() - step_t0

        if args.clear_cache_every > 0 and (step + 1) % args.clear_cache_every == 0:
            try:
                mx.metal.clear_cache()
            except Exception:
                pass

        post_active = mlx_active_memory_mb()
        post_peak = mlx_peak_memory_mb()

        # macOS vm_stat snapshot every 20 steps (it's not free)
        macos = macos_memory_stats() if step % 20 == 0 else None

        record = {
            "step": step,
            "loss": loss,
            "step_wall_ms": round(step_wall * 1000, 1),
            "cum_wall_s": round(time.time() - t0, 1),
            "mlx_active_mb_before": round(pre_active, 1),
            "mlx_active_mb_after": round(post_active, 1),
            "mlx_peak_mb": round(post_peak, 1),
        }
        if macos is not None and "Pages free" in macos:
            page_size_bytes = 16384  # M1+ macOS uses 16K pages
            record["macos_pages_free_mb"] = macos.get("Pages free", 0) * page_size_bytes / 1e6
            record["macos_pages_active_mb"] = macos.get("Pages active", 0) * page_size_bytes / 1e6
            record["macos_pages_wired_mb"] = macos.get("Pages wired down", 0) * page_size_bytes / 1e6
            record["macos_pages_compressed_mb"] = macos.get("Pages occupied by compressor", 0) * page_size_bytes / 1e6
            record["macos_swapouts"] = macos.get("Swapouts", 0)
            record["macos_swapins"] = macos.get("Swapins", 0)

        f.write(json.dumps(record) + "\n")

        if step % 20 == 0 or step_wall > 2.0:
            print(f"  step={step:>4d} loss={loss:.3f} step_wall={step_wall*1000:>7.1f}ms "
                  f"mlx_active={post_active:>6.0f}MB peak={post_peak:>6.0f}MB")

    f.close()
    print(f"[diag] done in {time.time() - t0:.1f}s; data at {args.out}")


if __name__ == "__main__":
    main()
