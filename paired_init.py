"""Paired-seed init protocol per design §9.7.

Build vanilla and diff models with byte-identical shared weights (embed, MLPs,
RMSNorms, ALL attention projections). Diff-only params (lambda vectors, subln
scale) use a separate RNG stream so vanilla/diff weight tensors don't interact.

Both state-dicts can then be saved/loaded so paired Stage 0/1/2 runs always
start from byte-identical shared init for clean delta analysis.
"""
from __future__ import annotations
from pathlib import Path
import mlx.core as mx

from config import ModelConfig
from model import Transformer
from checkpoint import save_checkpoint, load_checkpoint, _flatten


def _unflatten_to_template(flat: dict, template):
    """Rebuild nested structure matching `template` using values from `flat` (dotted-key paths)."""
    def helper(tmpl, prefix):
        if isinstance(tmpl, dict):
            return {k: helper(v, f"{prefix}.{k}" if prefix else k) for k, v in tmpl.items()}
        if isinstance(tmpl, list):
            return [helper(v, f"{prefix}.{i}" if prefix else str(i)) for i, v in enumerate(tmpl)]
        if isinstance(tmpl, mx.array):
            return flat.get(prefix, tmpl)
        return tmpl
    return helper(template, "")


def _copy_shared_weights_inplace(vanilla_params: dict, diff_params: dict) -> dict:
    """Return a new diff-params-shaped dict where every leaf that exists in
    vanilla_params with the same shape is overwritten by vanilla's value.

    Names not in vanilla (lambda vectors, subln scale) keep diff's value.
    """
    flat_v = _flatten(vanilla_params)
    flat_d = _flatten(diff_params)
    new_flat_d = {}
    for name, val in flat_d.items():
        if name in flat_v and flat_v[name].shape == val.shape:
            new_flat_d[name] = flat_v[name]
        else:
            new_flat_d[name] = val
    return _unflatten_to_template(new_flat_d, diff_params)


def build_paired_models(cfg: ModelConfig, seed: int) -> tuple[Transformer, Transformer]:
    """Build (vanilla, diff) Transformers with byte-identical shared weights.

    Protocol (design §9.7):
    1. Seed MLX random with `seed`, build vanilla. RNG consumed for: embed, MLPs, norms, attn projections.
    2. Re-seed MLX random with `seed + LAMBDA_RNG_OFFSET`, build diff. RNG consumed for: same backbone
       (which will be overwritten) + 4 lambda vectors + subln scale.
    3. Copy vanilla's backbone + ALL attention projections (matching shapes) into diff's state-dict.
       Diff retains its lambda vectors and subln scale (init from its RNG stream).
    """
    LAMBDA_RNG_OFFSET = 1_000_003  # large prime so streams don't accidentally overlap

    # Step 1: vanilla
    mx.random.seed(seed)
    vanilla = Transformer(cfg, variant="vanilla")

    # Step 2: diff (separate RNG stream for lambdas; backbone init is throwaway)
    mx.random.seed(seed + LAMBDA_RNG_OFFSET)
    diff = Transformer(cfg, variant="diff")

    # Step 3: copy shared weights vanilla -> diff (by name + matching shape)
    new_diff_params = _copy_shared_weights_inplace(vanilla.parameters(), diff.parameters())
    diff.update(new_diff_params)
    return vanilla, diff


def save_paired_init(vanilla: Transformer, diff: Transformer, out_dir: Path) -> None:
    """Save both state-dicts to out_dir/{vanilla,diff}.safetensors."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(vanilla.parameters(), step=0, ckpt_path=out_dir / "vanilla.safetensors")
    save_checkpoint(diff.parameters(), step=0, ckpt_path=out_dir / "diff.safetensors")


def load_paired_init(cfg: ModelConfig, in_dir: Path) -> tuple[Transformer, Transformer]:
    """Load both state-dicts and return constructed Transformers."""
    in_dir = Path(in_dir)
    vanilla = Transformer(cfg, variant="vanilla")
    diff = Transformer(cfg, variant="diff")
    v_params, _ = load_checkpoint(in_dir / "vanilla.safetensors")
    d_params, _ = load_checkpoint(in_dir / "diff.safetensors")
    vanilla.update(v_params)
    diff.update(d_params)
    return vanilla, diff
