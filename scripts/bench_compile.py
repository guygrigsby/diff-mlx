"""Compare full train_step throughput with and without mx.compile.

The hypothesis: MLX dispatches many small kernels per layer. mx.compile fuses
them into a single graph and can drop dispatch overhead substantially. At
Stage 1 each kernel is large enough that fusion may help less than at Stage 0,
but worth measuring.

Approach: use Stage 1 model at small B (4) to avoid OOM and live-run
interference. Time 5 iterations of (a) the current train_step path and
(b) the same logic with mx.compile applied to the loss-and-grad function.

Usage:
    python scripts/bench_compile.py
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
from train_step import _ce_loss, _global_grad_norm, _clip_grads, train_step


def time_loop(label, fn, warmup=2, iters=5):
    for _ in range(warmup):
        result = fn()
        if isinstance(result, mx.array):
            mx.eval(result)
    t0 = time.perf_counter()
    for _ in range(iters):
        result = fn()
        if isinstance(result, mx.array):
            mx.eval(result)
    dt = (time.perf_counter() - t0) / iters
    print(f"  {label:42s}  {dt*1000:8.1f} ms / step")
    return dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["vanilla", "diff"], default="vanilla")
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--T", type=int, default=2048)
    args = p.parse_args()

    print(f"[compile-bench] variant={args.variant} B={args.B} T={args.T} amp_dtype=bfloat16")

    cfg = replace(ModelConfig.stage1(), amp_dtype="bfloat16")
    model = Transformer(cfg, variant=args.variant)
    opt = make_adamw(lr=1e-4, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)

    mx.random.seed(0)
    x = mx.random.randint(0, cfg.vocab_size, shape=(args.B, args.T))
    y = mx.random.randint(0, cfg.vocab_size, shape=(args.B, args.T))
    mx.eval(model.parameters(), opt.state, x, y)

    # Baseline: existing train_step
    def baseline_step():
        return train_step(model, opt, x, y, grad_clip=1.0)

    print(f"\n[compile-bench] timing (warmup=2, iters=5):")
    t_base = time_loop("baseline train_step", baseline_step)

    # Compiled path: standard MLX training-compile idiom from mlx-examples.
    # Build a fresh model + optimizer so the timing isn't biased by accumulated
    # state from the baseline.
    mx.random.seed(0)
    model2 = Transformer(cfg, variant=args.variant)
    opt2 = make_adamw(lr=1e-4, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)
    mx.eval(model2.parameters(), opt2.state)

    lg2 = nn.value_and_grad(model2, _ce_loss)

    # Capture model + optimizer state for compile to track in/out tensors.
    # This is the canonical MLX training-compile idiom.
    state = [model2.state, opt2.state]

    def _inner(x_in, y_in):
        loss, grads = lg2(model2, x_in, y_in)
        opt2.update(model2, grads)
        return loss

    compiled_inner = mx.compile(_inner, inputs=state, outputs=state)

    def compiled_step2():
        loss = compiled_inner(x, y)
        mx.eval(state, loss)
        return loss

    t_compiled = time_loop("compiled training step", compiled_step2)

    n_tokens = args.B * args.T
    print(f"\n[compile-bench] throughput at this shape:")
    print(f"  baseline                                  {n_tokens / t_base:>10,.0f} tps")
    print(f"  compiled                                  {n_tokens / t_compiled:>10,.0f} tps")
    speedup = t_base / t_compiled
    print(f"  speedup                                   {speedup:>10.2f}x")

    print(f"\n[compile-bench] mlx peak memory: {mx.get_peak_memory()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
