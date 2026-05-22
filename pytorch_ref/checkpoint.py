"""Checkpoint save/load. Single safetensors file with namespaced keys.

Mirrors ../checkpoint.py: keys are flat strings with "model." or "opt." prefix.
Metadata holds the step and has_opt flag.
"""
from __future__ import annotations
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file


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
    """Inverse of _flatten."""
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


def save_checkpoint(model: torch.nn.Module, optimizer, step: int, ckpt_path: Path) -> None:
    """Save model.state_dict + optimizer.state_dict to a single safetensors file."""
    bundle = {}
    for k, v in model.state_dict().items():
        bundle[f"model.{k}"] = v.contiguous()

    # optimizer.state_dict() returns {"state": {<int>: {<key>: tensor}}, "param_groups": [...]}.
    # We serialize only the tensor parts; param_groups is reconstructed at load
    # time from the existing optimizer's groups (lr, betas, etc. are set by the
    # caller, so we don't need to round-trip them).
    opt_state = optimizer.state_dict()["state"]
    for pid, pstate in opt_state.items():
        for key, val in pstate.items():
            if isinstance(val, torch.Tensor):
                bundle[f"opt.{pid}.{key}"] = val.contiguous()
            elif isinstance(val, int):
                # AdamW's `step` field is an int; encode as a 1-element int tensor.
                bundle[f"opt.{pid}.{key}__int"] = torch.tensor([val], dtype=torch.int64)

    metadata = {"step": str(step), "has_opt": "1"}
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(bundle, str(ckpt_path), metadata=metadata)


def load_checkpoint(ckpt_path: Path) -> tuple[dict, int, dict | None]:
    """Returns (model_state_dict, step, optimizer_state-or-None).

    optimizer_state is shaped to be passed back as the "state" half of an
    Adam-like optimizer.state_dict(); caller is expected to merge it back.
    """
    tensors = load_file(str(ckpt_path))

    # Read metadata. safetensors stores metadata as a string-string dict.
    # safetensors.torch.load_file doesn't return metadata directly; use safe_open.
    from safetensors import safe_open
    with safe_open(str(ckpt_path), framework="pt") as f:
        metadata = f.metadata() or {}

    step = int(metadata.get("step", "0"))
    has_opt = metadata.get("has_opt", "0") == "1"

    model_state = {k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")}
    if not model_state:
        # Legacy: no namespace prefix.
        model_state = dict(tensors)

    opt_state: dict | None = None
    if has_opt:
        opt_flat = {k[len("opt."):]: v for k, v in tensors.items() if k.startswith("opt.")}
        if opt_flat:
            # Group by param-id and key.
            grouped: dict[int, dict] = {}
            for k, v in opt_flat.items():
                # k looks like "<pid>.<key>" or "<pid>.<key>__int"
                first_dot = k.index(".")
                pid = int(k[:first_dot])
                key = k[first_dot + 1:]
                if key.endswith("__int"):
                    key = key[:-len("__int")]
                    grouped.setdefault(pid, {})[key] = int(v.item())
                else:
                    grouped.setdefault(pid, {})[key] = v
            opt_state = grouped

    return model_state, step, opt_state
