"""Cosine LR schedule with linear warmup. Pure float math (no MLX dependency)."""
import math


def cosine_lr_with_warmup(
    step: int,
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_frac: float = 0.1,
) -> float:
    """LR at `step`: linear warmup from 0 to peak over warmup_steps, then cosine
    decay from peak to (peak * min_lr_frac) over (total_steps - warmup_steps).

    Holds at the floor for step > total_steps.
    """
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return peak_lr * min_lr_frac
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1.0 - min_lr_frac) * cosine)
