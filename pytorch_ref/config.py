"""Mirror of ../config.py. Same Stage 0/1/2 hyperparameters as the MLX side."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    dim: int
    n_layers: int
    n_heads_vanilla: int
    qk_head_dim: int
    vocab_size: int
    mlp_intermediate: int
    block_size: int
    rope_base: float = 10000.0
    rms_eps: float = 1e-5
    tie_embeddings: bool = True
    amp_dtype: str = "float32"  # parity with MLX side; PyTorch uses autocast at train time

    @classmethod
    def stage0(cls) -> "ModelConfig":
        return cls(
            dim=256, n_layers=6, n_heads_vanilla=4, qk_head_dim=64,
            vocab_size=100_277, mlp_intermediate=704, block_size=1024,
        )

    @classmethod
    def stage1(cls) -> "ModelConfig":
        return cls(
            dim=768, n_layers=12, n_heads_vanilla=12, qk_head_dim=64,
            vocab_size=100_277, mlp_intermediate=2048, block_size=2048,
            amp_dtype="bfloat16",
        )

    @classmethod
    def stage2(cls) -> "ModelConfig":
        return cls(
            dim=1024, n_layers=16, n_heads_vanilla=16, qk_head_dim=64,
            vocab_size=100_277, mlp_intermediate=2752, block_size=2048,
            amp_dtype="bfloat16",
        )


@dataclass(frozen=True)
class TrainConfig:
    peak_lr: float
    warmup_steps: int
    total_tokens: int
    micro_batch: int
    grad_accum: int = 1
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    eval_every: int = 500
    full_eval_every: int = 5000
    monitoring_tokens: int = 2_000_000
    full_eval_tokens: int = 75_000_000
    save_every: int = 1000

    @classmethod
    def stage0(cls) -> "TrainConfig":
        # micro_batch=4 grad_accum=4 (was 16/1 on MLX side). Effective batch
        # unchanged at 16. At B=16 on CUDA, the cross-entropy backward
        # materializes a fp32 (B, T, vocab) grad_logits tensor: 16 * 1024 *
        # 100277 * 4 bytes = 6.6 GB. Plus saved bf16 logits + other state, the
        # 8 GB RTX 3070 Ti OOMs. At B=4, grad_logits is 1.6 GB and the run
        # fits with headroom.
        return cls(
            peak_lr=6e-4, warmup_steps=500, total_tokens=100_000_000,
            micro_batch=4, grad_accum=4,
            eval_every=500, full_eval_every=2500, save_every=500,
        )

    @classmethod
    def stage1(cls) -> "TrainConfig":
        # micro_batch=4 here (was 8 on MLX side at 128 GB unified memory).
        # 3070 Ti has 8 GB VRAM; B=8 with Stage 1 activations + logits won't
        # fit. B=4 gives effective batch 32 at grad_accum=8.
        return cls(
            peak_lr=4e-4, warmup_steps=1000, total_tokens=2_000_000_000,
            micro_batch=4, grad_accum=8,
            eval_every=1000, full_eval_every=5000, save_every=1000,
        )

    @classmethod
    def stage2(cls) -> "TrainConfig":
        # micro_batch=2 for 3070 Ti memory budget. Effective batch stays 32.
        return cls(
            peak_lr=3e-4, warmup_steps=2000, total_tokens=4_000_000_000,
            micro_batch=2, grad_accum=16,
            eval_every=1000, full_eval_every=5000, save_every=500,
        )
