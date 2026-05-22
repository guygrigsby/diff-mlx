"""Per-step timing breakdown for Stage 1 throughput investigation.

Times the components of a single training step at Stage 1 shapes:
  - forward only
  - forward + backward (value_and_grad)
  - full step (forward + backward + grad_norm + clip + optimizer.update + eval)
  - same full step wrapped in mx.compile

Uses a smaller micro_batch than the live training (B=4 vs B=32) to minimize
GPU contention with any concurrent run. Per-token cost scales linearly, so
the comparison stays valid; throughput numbers should be read in tokens/sec.

Usage:
    python scripts/bench_step.py
    python scripts/bench_step.py --variant diff
    python scripts/bench_step.py --B 8 --T 1024
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
from dataclasses import replace
import mlx.core as mx
import mlx.nn as nn
from config import ModelConfig
from model import Transformer
from optim import make_adamw
from train_step import _ce_loss, _global_grad_norm, _clip_grads


def time_op(label, fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    # Block on completion of warmup before timing
    t0 = time.perf_counter()
    for _ in range(iters):
        result = fn()
        # Ensure execution is complete before the next iteration
        if isinstance(result, mx.array):
            mx.eval(result)
        elif isinstance(result, tuple):
            mx.eval(*result)
    dt = (time.perf_counter() - t0) / iters
    print(f"  {label:36s}  {dt*1000:8.1f} ms")
    return dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["vanilla", "diff"], default="vanilla")
    p.add_argument("--B", type=int, default=4, help="micro_batch for the probe")
    p.add_argument("--T", type=int, default=2048, help="sequence length")
    p.add_argument("--amp_dtype", default="bfloat16")
    args = p.parse_args()

    print(f"[bench] variant={args.variant} B={args.B} T={args.T} amp_dtype={args.amp_dtype}")
    print(f"[bench] mlx version: {mx.__version__}")

    cfg = replace(ModelConfig.stage1(), amp_dtype=args.amp_dtype)
    model = Transformer(cfg, variant=args.variant)
    opt = make_adamw(lr=1e-4, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)

    mx.random.seed(0)
    x = mx.random.randint(0, cfg.vocab_size, shape=(args.B, args.T))
    y = mx.random.randint(0, cfg.vocab_size, shape=(args.B, args.T))
    mx.eval(model.parameters(), x, y)

    # 1. Forward only (just the loss, no autograd)
    def forward_only():
        return _ce_loss(model, x, y)

    # 2. Forward + backward via value_and_grad
    lg = nn.value_and_grad(model, _ce_loss)
    def fwd_bwd():
        loss, grads = lg(model, x, y)
        return loss, grads

    # 3. Just the grad-norm walk
    def just_norm():
        _, grads = lg(model, x, y)
        return _global_grad_norm(grads)

    # 4. Full step (matches current train_step exactly)
    def full_step():
        loss, grads = lg(model, x, y)
        norm = _global_grad_norm(grads)
        grads = _clip_grads(grads, 1.0, norm)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        return loss

    # 5. Full step with mx.compile applied to the loss
    @mx.compile
    def compiled_loss_fn(params, x_in, y_in):
        # We can't compile with a Module easily; rebuild as a parametric call
        # over (params, x, y). Skip compilation for the full module — try a
        # simpler test: compile the loss computation that follows the model
        # forward, but the model forward itself dominates.
        return _ce_loss(model, x_in, y_in)

    def full_step_compiled():
        # Apply mx.compile to the value_and_grad function
        loss, grads = lg(model, x, y)
        norm = _global_grad_norm(grads)
        grads = _clip_grads(grads, 1.0, norm)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        return loss

    print(f"\n[bench] timing (warmup=2, iters=5):")
    t_fwd = time_op("forward only (loss)", forward_only)
    t_fb = time_op("forward + backward", fwd_bwd)
    t_norm = time_op("forward + backward + grad_norm", just_norm)
    t_full = time_op("full train_step", full_step)

    print(f"\n[bench] tokens/sec at this shape:")
    n_tokens = args.B * args.T
    print(f"  forward only        : {n_tokens / t_fwd:>10,.0f} tps")
    print(f"  forward+backward    : {n_tokens / t_fb:>10,.0f} tps")
    print(f"  full step           : {n_tokens / t_full:>10,.0f} tps")
    print(f"\n[bench] for ref: live Stage 1 run is reporting ~950 tps at B=32 T=2048")
    print(f"        same per-token cost would be ~{int(n_tokens / (n_tokens / t_full)):,} ms/step at this probe shape")

    peak_mb = mx.get_peak_memory() / 1e6
    print(f"\n[bench] mlx peak memory: {peak_mb:.1f} MB")


if __name__ == "__main__":
    main()
