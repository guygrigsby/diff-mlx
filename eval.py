"""Evaluation: token-level NLL over a deterministic prefix of the val set."""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from data.loader import ShardLoader


def compute_val_loss(
    model: nn.Module,
    val_loader: ShardLoader,
    block_size: int,
    micro_batch: int,
    max_tokens: int,
) -> float:
    """Walk the first ~max_tokens tokens of the val set in non-overlapping windows of
    block_size. Return token-weighted average NLL.

    Deterministic: same loader + same args yields the same loss exactly.
    """
    total_loss = 0.0
    total_tokens = 0
    offset = 0
    cap = min(val_loader.total_tokens, max_tokens + block_size + 1)
    while offset + block_size + 1 <= cap:
        windows = []
        for _ in range(micro_batch):
            if offset + block_size + 1 > cap:
                break
            windows.append(val_loader.read(offset, block_size + 1))
            offset += block_size
        if not windows:
            break
        x = mx.array(np.stack([w[:-1] for w in windows]).astype(np.int32))
        y = mx.array(np.stack([w[1:]  for w in windows]).astype(np.int32))
        logits = model(x).astype(mx.float32)
        log_probs = nn.log_softmax(logits, axis=-1)
        gathered = mx.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
        loss = -gathered.sum().item()
        n = gathered.size
        total_loss += loss
        total_tokens += n
        if offset >= max_tokens:
            break
    return total_loss / max(1, total_tokens)
