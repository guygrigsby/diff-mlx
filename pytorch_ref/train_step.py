"""Single training step + grad accumulation. PyTorch port of ../train_step.py.

bf16 forward via torch.autocast on CUDA; fp32 params + optimizer state.
Logits cast to fp32 before cross-entropy (paper-canonical, design §9.0).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ce_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy. Logits cast to fp32 (design §9.0)."""
    logits = model(x).float()
    # cross_entropy expects (N, C) logits and (N,) targets.
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), y.view(B * T))


def train_step(
    model: nn.Module,
    optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_clip: float = 1.0,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
) -> float:
    """Forward + backward + optimizer.step. Returns scalar loss as Python float.

    autocast_dtype=None disables AMP (used by the cross-check tests).
    """
    optimizer.zero_grad(set_to_none=True)
    if autocast_dtype is not None and x.is_cuda:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            loss = _ce_loss(model, x, y)
    else:
        loss = _ce_loss(model, x, y)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


def train_step_with_accum(
    model: nn.Module,
    optimizer,
    batches,
    grad_clip: float = 1.0,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
) -> float:
    """Accumulate grads over N micro-batches, then one optimizer step.

    Loss returned is the mean across micro-batches. Each micro-batch's backward
    is called immediately so PyTorch's autograd graph doesn't retain activations
    across the whole accum loop.
    """
    n = len(batches)
    assert n >= 1
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for x, y in batches:
        if autocast_dtype is not None and x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                loss = _ce_loss(model, x, y) / n  # scale before backward
        else:
            loss = _ce_loss(model, x, y) / n
        loss.backward()  # grads accumulate into .grad
        total_loss += loss.item() * n  # unscale for logging

    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return total_loss / n
