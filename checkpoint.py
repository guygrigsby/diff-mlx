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
    """Save params (and optionally optimizer state / RNG state) to safetensors."""
    flat = _flatten(params)
    metadata = {"step": str(step)}
    mx.save_safetensors(str(ckpt_path), flat, metadata=metadata)


def load_checkpoint(ckpt_path: Path) -> tuple[dict, int]:
    """Load a safetensors checkpoint. Returns (params_dict, step)."""
    loaded = mx.load(str(ckpt_path), return_metadata=True)
    tensors, metadata = loaded
    step = int(metadata.get("step", "0"))
    params = _unflatten(dict(tensors))
    return params, step


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
