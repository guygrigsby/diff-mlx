"""Checkpoint save/load using MLX safetensors + per-run metadata files."""
from __future__ import annotations
from pathlib import Path
import json
import mlx.core as mx


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(_flatten(item, f"{key}.{i}"))
                else:
                    out[f"{key}.{i}"] = item
        else:
            out[key] = v
    return out


def _unflatten(flat: dict) -> dict:
    """Inverse of _flatten: turn 'a.b.0.c' keys back into nested dicts/lists."""
    out: dict = {}
    for key, val in flat.items():
        parts = key.split(".")
        cur = out
        for i, part in enumerate(parts[:-1]):
            nxt = parts[i + 1]
            nxt_is_int = nxt.isdigit()
            if part.isdigit():
                part_i = int(part)
                while len(cur) <= part_i:
                    cur.append({} if not nxt_is_int else [])
                if cur[part_i] == {} or (isinstance(cur[part_i], list) and len(cur[part_i]) == 0):
                    cur[part_i] = [] if nxt_is_int else {}
                cur = cur[part_i]
            else:
                if part not in cur:
                    cur[part] = [] if nxt_is_int else {}
                cur = cur[part]
        last = parts[-1]
        if last.isdigit():
            last_i = int(last)
            while len(cur) <= last_i:
                cur.append(None)
            cur[last_i] = val
        else:
            cur[last] = val
    return out


def save_checkpoint(params: dict, step: int, ckpt_path: Path,
                    optim_state=None, rng_state=None) -> None:
    """Save params + optional optimizer state to a single safetensors file.

    Keys are namespaced: `model.<path>` for params, `opt.<path>` for optimizer
    state. Step is recorded in safetensors metadata. rng_state is currently
    accepted but not yet persisted (left as a future hook).
    """
    flat_model = {f"model.{k}": v for k, v in _flatten(params).items()}
    bundle = dict(flat_model)
    if optim_state is not None:
        flat_opt = {f"opt.{k}": v for k, v in _flatten(optim_state).items()
                    if isinstance(v, mx.array)}
        bundle.update(flat_opt)
    metadata = {"step": str(step), "has_opt": "1" if optim_state is not None else "0"}
    mx.save_safetensors(str(ckpt_path), bundle, metadata=metadata)


def load_checkpoint(ckpt_path: Path) -> tuple[dict, int, dict | None]:
    """Load a checkpoint. Returns (params, step, optim_state-or-None).

    Splits the namespaced keys back into the two pytrees. An older checkpoint
    without optimizer state returns None for the third element.
    """
    loaded = mx.load(str(ckpt_path), return_metadata=True)
    tensors, metadata = loaded
    step = int(metadata.get("step", "0"))
    has_opt = metadata.get("has_opt", "0") == "1"

    model_flat = {k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")}
    if not model_flat:
        # legacy checkpoint: no namespace prefix
        model_flat = dict(tensors)
    params = _unflatten(model_flat)

    optim_state: dict | None = None
    if has_opt:
        opt_flat = {k[len("opt."):]: v for k, v in tensors.items() if k.startswith("opt.")}
        optim_state = _unflatten(opt_flat) if opt_flat else None

    return params, step, optim_state


def save_run_metadata(
    run_dir: Path,
    model_cfg,
    train_cfg_dict: dict,
    git_hash: str,
    git_dirty: bool,
    mlx_version: str,
    seed: int,
    data_meta: dict,
) -> None:
    """Snapshot the run's reproducibility-relevant context."""
    from dataclasses import is_dataclass, asdict
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dict = asdict(model_cfg) if is_dataclass(model_cfg) else dict(model_cfg.__dict__)
    full = {"model": model_dict, "train": train_cfg_dict}
    (run_dir / "config.json").write_text(json.dumps(full, indent=2))
    (run_dir / "git.txt").write_text(f"{git_hash}\ndirty={git_dirty}\n")
    (run_dir / "mlx_version.txt").write_text(mlx_version + "\n")
    (run_dir / "tiktoken.txt").write_text(
        f"version={data_meta.get('tiktoken_version', '?')}\n"
        f"encoding={data_meta.get('tokenizer_name', '?')}\n"
        f"vocab_size={data_meta.get('vocab_size', '?')}\n"
    )
    (run_dir / "data_meta.json").write_text(json.dumps(data_meta, indent=2))
    (run_dir / "seed.txt").write_text(str(seed) + "\n")
