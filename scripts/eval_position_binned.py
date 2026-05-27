"""Position-binned held-out NLL for diff vs vanilla final checkpoints.

Walks the val set in non-overlapping windows of block_size and accumulates
per-position NLL, then bins by position within the window. Tests whether
DiffAttention's benefit is concentrated at later (long-context) positions
even when the averaged NLL is tied.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ModelConfig
from model import Transformer
from checkpoint import load_checkpoint
from data.loader import ShardLoader

SHARDS = Path("data/shards")
BLOCK = 2048
MICRO = 8
MAX_TOKENS = 4_000_000  # ~1950 windows; plenty for stable per-position means
N_BINS = 8

RUNS = {
    "vanilla": "runs/stage1-paired/stage1-paired-vanilla-seed0/latest.safetensors",
    "diff":    "runs/stage1-paired/stage1-paired-diff-seed0/latest.safetensors",
}


def per_position_nll(variant: str, ckpt: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = ModelConfig.stage1()
    model = Transformer(cfg, variant=variant)
    params, step, _ = load_checkpoint(Path(ckpt))
    model.update(params)
    mx.eval(model.parameters())
    print(f"[{variant}] loaded step {step}")

    val = ShardLoader(SHARDS, "val")
    pos_sum = np.zeros(BLOCK, dtype=np.float64)
    pos_cnt = np.zeros(BLOCK, dtype=np.float64)
    offset = 0
    cap = min(val.total_tokens, MAX_TOKENS + BLOCK + 1)
    while offset + BLOCK + 1 <= cap:
        windows = []
        for _ in range(MICRO):
            if offset + BLOCK + 1 > cap:
                break
            windows.append(val.read(offset, BLOCK + 1))
            offset += BLOCK
        if not windows:
            break
        arr = np.stack(windows).astype(np.int32)
        x = mx.array(arr[:, :-1]); y = mx.array(arr[:, 1:])
        logits = model(x).astype(mx.float32)
        lp = nn.log_softmax(logits, axis=-1)
        nll = -mx.take_along_axis(lp, y[..., None], axis=-1).squeeze(-1)  # (B, T)
        nll_np = np.array(nll)  # (B, T)
        pos_sum += nll_np.sum(axis=0)
        pos_cnt += nll_np.shape[0]
    return pos_sum / pos_cnt, pos_cnt


def main():
    results = {}
    for variant, ckpt in RUNS.items():
        results[variant] = per_position_nll(variant, ckpt)[0]

    d = results["diff"]; v = results["vanilla"]
    binsz = BLOCK // N_BINS
    print(f"\n{'pos range':>14} {'diff':>8} {'vanilla':>8} {'delta':>9}")
    for b in range(N_BINS):
        lo, hi = b * binsz, (b + 1) * binsz
        dm = d[lo:hi].mean(); vm = v[lo:hi].mean()
        print(f"{lo:>5}-{hi-1:<5} {dm:>10.4f} {vm:>8.4f} {dm-vm:>+9.4f}")
    print(f"\n{'overall':>14} {d.mean():>10.4f} {v.mean():>8.4f} {d.mean()-v.mean():>+9.4f}")
    # save arrays for plotting
    np.savez("docs/position_binned_nll.npz", diff=d, vanilla=v)
    print("saved docs/position_binned_nll.npz")


if __name__ == "__main__":
    main()
