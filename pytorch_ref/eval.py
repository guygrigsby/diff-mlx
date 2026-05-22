"""Token-level NLL over a deterministic prefix of the val set. PyTorch port of ../eval.py."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import ShardLoader


@torch.no_grad()
def compute_val_loss(
    model: nn.Module,
    val_loader: ShardLoader,
    block_size: int,
    micro_batch: int,
    max_tokens: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
) -> float:
    """Walk the first ~max_tokens tokens of val in non-overlapping windows."""
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
        x_np = np.stack([w[:-1] for w in windows]).astype(np.int64)
        y_np = np.stack([w[1:] for w in windows]).astype(np.int64)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        if autocast_dtype is not None and x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(x)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.view(B * T, V), y.view(B * T), reduction="sum"
                ).item()
        else:
            logits = model(x)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V), y.view(B * T), reduction="sum"
            ).item()
        total_loss += loss
        total_tokens += B * T
        if offset >= max_tokens:
            break
    return total_loss / max(1, total_tokens)
