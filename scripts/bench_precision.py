"""Precision matrix benchmark: fp32 vs bf16 throughput at Stage 1 shapes.

Hypothesis under test: on Apple Silicon, fp32 and bf16 matmul run at similar
rates, so bf16 mostly buys memory bandwidth, not compute. If true, dropping
to bf16 doesn't automatically help us at Stage 1. Measure to find out.

Runs each (amp_dtype, variant, batch) combo through one step of forward +
backward + optimizer.update on a Stage 1 model. Reports ms/step and
tokens/sec. Skips configs that OOM.

Smaller batches than the live run (B=4, 8) to keep memory in budget for the
fp32 path. Per-token cost was confirmed constant across batch sizes in
scripts/bench_step.py, so the comparison stays valid.

Run alongside `powermetrics --samplers gpu_power -i 1000` in another terminal
to get the GPU active-residency reading.

Usage:
    python scripts/bench_precision.py
    python scripts/bench_precision.py --B 4
    python scripts/bench_precision.py --variants diff
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
import traceback
from dataclasses import replace
import mlx.core as mx
import mlx.nn as nn
from config import ModelConfig
from model import Transformer
from optim import make_adamw
from train_step import _ce_loss


def bench_one(amp_dtype: str, variant: str, B: int, T: int = 2048,
              warmup: int = 2, iters: int = 5) -> dict:
    """One config. Returns dict with timing and memory."""
    mx.random.seed(0)
    cfg = replace(ModelConfig.stage1(), amp_dtype=amp_dtype)
    model = Transformer(cfg, variant=variant)
    opt = make_adamw(lr=1e-4, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)
    x = mx.random.randint(0, cfg.vocab_size, shape=(B, T))
    y = mx.random.randint(0, cfg.vocab_size, shape=(B, T))
    mx.eval(model.parameters(), opt.state, x, y)

    lg = nn.value_and_grad(model, _ce_loss)

    def step_fn():
        loss, grads = lg(model, x, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        return loss

    # Warmup
    for _ in range(warmup):
        out = step_fn()
        mx.eval(out)

    # Time
    t0 = time.perf_counter()
    for _ in range(iters):
        out = step_fn()
        mx.eval(out)
    dt = (time.perf_counter() - t0) / iters

    peak_gb = mx.get_peak_memory() / 1e9
    n_tokens = B * T
    return {
        "ms_per_step": dt * 1000,
        "tps": n_tokens / dt,
        "ms_per_token": (dt * 1000) / n_tokens,
        "peak_gb": peak_gb,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="vanilla,diff",
                   help="comma-separated: vanilla,diff or just one")
    p.add_argument("--dtypes", default="float32,bfloat16",
                   help="comma-separated amp_dtype values")
    p.add_argument("--batches", default="4,8",
                   help="comma-separated micro_batch values")
    p.add_argument("--T", type=int, default=2048)
    args = p.parse_args()

    variants = args.variants.split(",")
    dtypes = args.dtypes.split(",")
    batches = [int(b) for b in args.batches.split(",")]

    print(f"[bench-precision] mlx={mx.__version__}  T={args.T}")
    print(f"[bench-precision] variants={variants}  dtypes={dtypes}  batches={batches}")
    print()

    results = []
    for amp_dtype in dtypes:
        for variant in variants:
            for B in batches:
                label = f"{amp_dtype:>9s}  {variant:>8s}  B={B:>2d}"
                try:
                    r = bench_one(amp_dtype, variant, B, T=args.T)
                    print(f"  {label}   "
                          f"step={r['ms_per_step']:>7.1f} ms  "
                          f"tps={r['tps']:>8,.0f}  "
                          f"ms/tok={r['ms_per_token']:.3f}  "
                          f"peak={r['peak_gb']:>5.1f} GB")
                    results.append((amp_dtype, variant, B, r))
                except Exception as e:
                    print(f"  {label}   FAILED ({type(e).__name__}: {str(e)[:80]})")

    print()
    print("[bench-precision] interpretation hints:")
    print("  - If fp32 tps >= bf16 tps at the same B, MLX bf16 path is underperforming;")
    print("    LinearAMP cast or kernel quality is costing more than the bf16 bandwidth win.")
    print("  - If bf16 tps > fp32 tps but only by <30%, the gain is bandwidth-only,")
    print("    not the 2x you'd see on NVIDIA tensor cores.")
    print("  - If bf16 lets you run a B that fp32 OOMs at, that's the real bf16 win:")
    print("    more concurrency per step at constant per-token cost.")


if __name__ == "__main__":
    main()
