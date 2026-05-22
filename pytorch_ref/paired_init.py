"""Paired-seed init protocol per design §9.7. PyTorch port of ../paired_init.py.

Build vanilla and diff models with byte-identical shared weights (embed, MLPs,
RMSNorms, ALL attention projections). Diff-only params (lambda vectors, subln
scale) use a separate RNG stream so vanilla/diff weight tensors don't interact.
"""
from __future__ import annotations
from pathlib import Path
import torch

from model import Transformer

LAMBDA_RNG_OFFSET = 1_000_003  # large prime so RNG streams don't overlap


def build_paired_models(cfg, seed: int) -> tuple[Transformer, Transformer]:
    """Build (vanilla, diff) Transformers with byte-identical shared weights.

    Protocol (design §9.7):
    1. Seed torch with `seed`, build vanilla.
    2. Re-seed torch with `seed + LAMBDA_RNG_OFFSET`, build diff. Diff's backbone
       init is throwaway; it gets overwritten in step 3.
    3. Copy vanilla's backbone + all attention projections (matching shapes) into
       diff's state-dict. Diff retains its lambda vectors and subln scale.
    """
    torch.manual_seed(seed)
    vanilla = Transformer(cfg, variant="vanilla")

    torch.manual_seed(seed + LAMBDA_RNG_OFFSET)
    diff = Transformer(cfg, variant="diff")

    v_state = vanilla.state_dict()
    d_state = diff.state_dict()

    # Copy by name + matching shape. Anything not in vanilla (lambda_q1/k1/q2/k2,
    # subln.scale on diff layers) stays as diff initialized it.
    copied = 0
    for name, v_tensor in v_state.items():
        if name in d_state and d_state[name].shape == v_tensor.shape:
            d_state[name] = v_tensor.clone()
            copied += 1
    diff.load_state_dict(d_state)

    assert copied >= 5, f"expected to copy at least the embed + some norms, got {copied}"
    return vanilla, diff


def save_paired_init(vanilla: Transformer, diff: Transformer, out_dir: Path) -> None:
    """Save both state-dicts to out_dir/{vanilla,diff}.safetensors."""
    from safetensors.torch import save_file
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(vanilla.state_dict(), str(out_dir / "vanilla.safetensors"))
    save_file(diff.state_dict(), str(out_dir / "diff.safetensors"))


def load_paired_init(cfg, in_dir: Path) -> tuple[Transformer, Transformer]:
    """Load both state-dicts and return constructed Transformers."""
    from safetensors.torch import load_file
    in_dir = Path(in_dir)
    vanilla = Transformer(cfg, variant="vanilla")
    diff = Transformer(cfg, variant="diff")
    vanilla.load_state_dict(load_file(str(in_dir / "vanilla.safetensors")))
    diff.load_state_dict(load_file(str(in_dir / "diff.safetensors")))
    return vanilla, diff
