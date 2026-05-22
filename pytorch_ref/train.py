"""Training driver. PyTorch port of ../train.py. Single GPU, single seed.

Mirrors the MLX side feature-for-feature: paired-seed init compatibility,
auto-resume from latest.safetensors with optimizer state, grad accumulation,
cosine LR + warmup, JSONL metrics logging, run metadata.
"""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from config import ModelConfig, TrainConfig
from model import Transformer
from data_loader import ShardLoader, sample_batch
from train_step import train_step, train_step_with_accum
from eval import compute_val_loss
from schedule import cosine_lr_with_warmup
from optim import make_adamw
from checkpoint import save_checkpoint, load_checkpoint
from metrics import MetricsLogger


def _torch_version() -> str:
    return torch.__version__


def _git_info(repo_root: Path) -> tuple[str, bool]:
    """Best-effort git hash + dirty flag. Returns ("unknown", False) on failure."""
    import subprocess
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        ).strip()
        return h, bool(status)
    except Exception:
        return "unknown", False


def _save_run_metadata(
    run_dir: Path,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_meta: dict,
    git_hash: str,
    git_dirty: bool,
    data_seed: int,
    model_seed: int,
    variant: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "model": asdict(model_cfg),
        "train": asdict(train_cfg),
    }, indent=2))
    (run_dir / "git.txt").write_text(f"{git_hash}\ndirty={git_dirty}\n")
    (run_dir / "torch_version.txt").write_text(_torch_version() + "\n")
    (run_dir / "tiktoken.txt").write_text(
        f"version={data_meta.get('tiktoken_version', '?')}\n"
        f"encoding={data_meta.get('tokenizer_name', '?')}\n"
        f"vocab_size={data_meta.get('vocab_size', '?')}\n"
    )
    (run_dir / "data_meta.json").write_text(json.dumps(data_meta, indent=2))
    (run_dir / "seed.txt").write_text(f"data={data_seed}\nmodel={model_seed}\n")
    (run_dir / "variant.txt").write_text(variant + "\n")


def train_run(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    shards_dir: Path,
    run_dir: Path,
    data_seed: int = 0,
    model_seed: int = 0,
    variant: str = "vanilla",
    init_state_dict: dict | None = None,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = Path(shards_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  variant={variant}  "
          f"amp_dtype={'fp32' if autocast_dtype is None else str(autocast_dtype)}")

    torch.manual_seed(model_seed)
    rng = np.random.default_rng(data_seed)

    train_loader = ShardLoader(shards_dir, "train")
    val_loader = ShardLoader(shards_dir, "val")
    data_meta = json.loads((shards_dir / "meta.json").read_text())

    model = Transformer(model_cfg, variant=variant).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    optimizer = make_adamw(
        model,
        lr=0.0,
        weight_decay=train_cfg.weight_decay,
        beta1=train_cfg.adam_beta1,
        beta2=train_cfg.adam_beta2,
        eps=train_cfg.adam_eps,
    )

    git_hash, git_dirty = _git_info(Path(__file__).resolve().parent.parent)
    _save_run_metadata(
        run_dir=run_dir, model_cfg=model_cfg, train_cfg=train_cfg,
        data_meta=data_meta, git_hash=git_hash, git_dirty=git_dirty,
        data_seed=data_seed, model_seed=model_seed, variant=variant,
    )

    eff_tokens_per_step = train_cfg.micro_batch * model_cfg.block_size * train_cfg.grad_accum
    total_steps = max(1, train_cfg.total_tokens // eff_tokens_per_step)

    # Auto-resume.
    latest = run_dir / "latest.safetensors"
    start_step = 0
    if latest.exists() and latest.stat().st_size > 0:
        loaded_state, saved_step, loaded_opt = load_checkpoint(latest)
        model.load_state_dict(loaded_state)
        if loaded_opt is not None:
            # Reconstruct optimizer.state_dict() shape: needs "state" and "param_groups".
            optimizer.load_state_dict({
                "state": loaded_opt,
                "param_groups": optimizer.state_dict()["param_groups"],
            })
        start_step = saved_step + 1
        print(f"[train] resuming from step {start_step} ({latest})")

    print(f"[train] {total_steps} steps total, {eff_tokens_per_step} tokens/step, "
          f"{train_cfg.total_tokens/1e6:.1f}M total tokens")

    logger = MetricsLogger(run_dir / "metrics.jsonl")
    t0 = time.time()
    step = start_step
    while step < total_steps:
        lr = cosine_lr_with_warmup(
            step, train_cfg.peak_lr, train_cfg.warmup_steps, total_steps,
            min_lr_frac=0.1,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        if train_cfg.grad_accum <= 1:
            x_np, y_np = sample_batch(train_loader, model_cfg.block_size,
                                       train_cfg.micro_batch, rng)
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            loss = train_step(model, optimizer, x, y,
                              grad_clip=train_cfg.grad_clip,
                              autocast_dtype=autocast_dtype)
        else:
            batches = []
            for _ in range(train_cfg.grad_accum):
                x_np, y_np = sample_batch(train_loader, model_cfg.block_size,
                                           train_cfg.micro_batch, rng)
                batches.append((
                    torch.from_numpy(x_np).to(device),
                    torch.from_numpy(y_np).to(device),
                ))
            loss = train_step_with_accum(model, optimizer, batches,
                                          grad_clip=train_cfg.grad_clip,
                                          autocast_dtype=autocast_dtype)

        do_full_eval = (step > 0 and step % train_cfg.full_eval_every == 0)
        do_monitor_eval = (step > 0 and step % train_cfg.eval_every == 0)
        record = {
            "step": step,
            "train_loss": loss,
            "lr": lr,
            "tps": int(eff_tokens_per_step * (step + 1) / max(1e-6, time.time() - t0)),
            "wall": round(time.time() - t0, 1),
        }
        if do_monitor_eval:
            record["val_loss_monitor"] = compute_val_loss(
                model, val_loader, model_cfg.block_size, train_cfg.micro_batch,
                train_cfg.monitoring_tokens, device, autocast_dtype,
            )
        if do_full_eval:
            record["val_loss_full"] = compute_val_loss(
                model, val_loader, model_cfg.block_size, train_cfg.micro_batch,
                train_cfg.full_eval_tokens, device, autocast_dtype,
            )
        logger.log(**record)

        if (step > 0 and step % train_cfg.save_every == 0) or step == total_steps - 1:
            save_checkpoint(model, optimizer, step=step, ckpt_path=latest)

        step += 1

    logger.close()
    print(f"[train] done in {time.time() - t0:.1f}s")


def _build_cfgs(stage: str) -> tuple[ModelConfig, TrainConfig]:
    return getattr(ModelConfig, stage)(), getattr(TrainConfig, stage)()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["stage0", "stage1", "stage2"], required=True)
    p.add_argument("--shards_dir", type=Path, default=Path("../data/shards"))
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--model_seed", type=int, default=0)
    p.add_argument("--variant", choices=["vanilla", "diff"], default="vanilla")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable bf16 autocast (fp32 training; slower but simpler).")
    args = p.parse_args()
    model_cfg, train_cfg = _build_cfgs(args.stage)
    train_run(
        model_cfg, train_cfg, args.shards_dir, args.run_dir,
        data_seed=args.data_seed, model_seed=args.model_seed,
        variant=args.variant,
        autocast_dtype=None if args.no_amp else torch.bfloat16,
    )


if __name__ == "__main__":
    main()
