"""AdamW with weight-decay exclusions matching ../optim.py.

Decay applies to linear projection weights only. Excluded:
- token embedding
- all RMSNorms (norm_attn.scale, norm_mlp.scale, final_norm.scale, subln.scale)
- diff-attn lambda vectors (lambda_q1, lambda_k1, lambda_q2, lambda_k2)
"""
from __future__ import annotations
import torch


FLAT_NO_DECAY_NAMES = (
    "tok_embed",
    "norm",
    "lambda_",
)


def split_params_for_decay(model: torch.nn.Module) -> tuple[list, list]:
    """Return (decay_params, no_decay_params) as two lists of nn.Parameter."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(needle in name for needle in FLAT_NO_DECAY_NAMES):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def make_adamw(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> torch.optim.AdamW:
    """Construct AdamW with the project's decay-exclusion policy applied."""
    decay, no_decay = split_params_for_decay(model)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
    )
