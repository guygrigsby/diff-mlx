"""AdamW optimizer wrapper with weight-decay exclusion policy and precision cast helpers.

Per design §9.1:
- Decay: linear projection weights (q/k/v/o, mlp gate/up/down)
- No decay: embeddings, all RMSNorm/subLN scales, lambda vectors (Phase B)

Per design §9.0:
- fp32 master params, bf16 forward cast each step.
"""
from __future__ import annotations
import mlx.core as mx
import mlx.optimizers as optim

# Substrings: any flat param name containing one of these is NO-DECAY.
FLAT_NO_DECAY_NAMES = (
    "tok_embed",       # token embedding matrix
    "norm",            # any RMSNorm (norm_attn, norm_mlp, final_norm, subln)
    "lambda_",         # diff-attn lambda vectors (Phase B)
)

FLAT_DECAY_NAMES = ("weight",)  # informative; actual rule is complement of no-decay


def split_params_for_decay(flat_params: dict) -> tuple[list[str], list[str]]:
    """Given a flat param dict (name -> tensor), return (decay_names, no_decay_names)."""
    decay, no_decay = [], []
    for name in flat_params:
        if any(needle in name for needle in FLAT_NO_DECAY_NAMES):
            no_decay.append(name)
        else:
            decay.append(name)
    return decay, no_decay


def make_adamw(*, lr: float, weight_decay: float, beta1: float, beta2: float, eps: float) -> optim.AdamW:
    """Construct an AdamW optimizer.

    Note: weight-decay exclusions are applied at the training step level (zero-out
    the decay term for excluded params via a custom step or by using two optimizer
    instances), not via the constructor (MLX's AdamW applies a single decay value
    to all params). See train.py for the integration.
    """
    return optim.AdamW(
        learning_rate=lr,
        betas=[beta1, beta2],
        eps=eps,
        weight_decay=weight_decay,
    )


def to_bf16_view(x: mx.array) -> mx.array:
    """Cast an fp32 tensor to bf16. Used to derive forward-pass params from master."""
    return x.astype(mx.bfloat16)


def to_bf16_dict(params: dict) -> dict:
    """Recursively cast all leaf tensors in a parameter dict to bf16.

    Keeps the same nesting structure as the input.
    """
    out = {}
    for k, v in params.items():
        if isinstance(v, dict):
            out[k] = to_bf16_dict(v)
        elif isinstance(v, list):
            out[k] = [to_bf16_dict(x) if isinstance(x, dict) else to_bf16_view(x) for x in v]
        elif isinstance(v, mx.array):
            out[k] = to_bf16_view(v)
        else:
            out[k] = v
    return out
