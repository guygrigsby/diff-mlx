"""Quick peak-memory measurement at Stage 0 diff, fp32 vs bf16.

Builds a Transformer + runs N steps at each precision, prints peak
metal memory. Manual sanity check, not a test.

Usage:
    python scripts/measure_peak_bf16.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace
import mlx.core as mx
import numpy as np
from config import ModelConfig, TrainConfig
from model import Transformer
from train_step import train_step
from optim import make_adamw


def measure(amp_dtype: str, steps: int = 20) -> float:
    mx.random.seed(0)
    cfg = replace(ModelConfig.stage0(), amp_dtype=amp_dtype)
    model = Transformer(cfg, variant="diff")
    opt = make_adamw(lr=1e-4, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)
    try:
        mx.metal.reset_peak_memory()
    except Exception:
        pass
    rng = np.random.default_rng(0)
    for _ in range(steps):
        x = mx.array(rng.integers(0, cfg.vocab_size, size=(16, cfg.block_size), dtype=np.int32))
        y = mx.array(rng.integers(0, cfg.vocab_size, size=(16, cfg.block_size), dtype=np.int32))
        train_step(model, opt, x, y, grad_clip=1.0)
    return mx.metal.get_peak_memory() / 1e9


if __name__ == "__main__":
    fp32_peak = measure("float32")
    print(f"fp32 peak: {fp32_peak:.2f} GB")
    bf16_peak = measure("bfloat16")
    print(f"bf16 peak: {bf16_peak:.2f} GB")
    print(f"reduction: {(1 - bf16_peak / fp32_peak) * 100:.1f}%")
