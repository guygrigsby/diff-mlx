from dataclasses import dataclass

import mlx.core as mx


def resolve_amp_dtype(s: str) -> "mx.Dtype":
    if s == "float32":
        return mx.float32
    if s == "bfloat16":
        return mx.bfloat16
    raise ValueError(f"unknown amp_dtype {s!r}; expected 'float32' or 'bfloat16'")


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
    amp_dtype: str = "float32"

    @classmethod
    def stage0(cls) -> "ModelConfig":
        return cls(
            dim=256,
            n_layers=6,
            n_heads_vanilla=4,
            qk_head_dim=64,
            vocab_size=100_277,
            mlp_intermediate=704,
            block_size=1024,
        )

    @classmethod
    def stage1(cls) -> "ModelConfig":
        return cls(
            dim=768,
            n_layers=12,
            n_heads_vanilla=12,
            qk_head_dim=64,
            vocab_size=100_277,
            mlp_intermediate=2048,
            block_size=2048,
            amp_dtype="bfloat16",
        )

    @classmethod
    def stage2(cls) -> "ModelConfig":
        return cls(
            dim=1024,
            n_layers=16,
            n_heads_vanilla=16,
            qk_head_dim=64,
            vocab_size=100_277,
            mlp_intermediate=2752,
            block_size=2048,
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
        # save_every=500: ~12 saves over Stage 0's 6,103 total steps.
        # At ~0.7s/step that's ~6 min between saves; total run is ~90 min,
        # so a crash loses at most one short interval.
        return cls(
            peak_lr=6e-4,
            warmup_steps=500,
            total_tokens=100_000_000,
            micro_batch=16,
            grad_accum=1,
            eval_every=500,
            full_eval_every=2500,
            save_every=500,
        )

    @classmethod
    def stage1(cls) -> "TrainConfig":
        # micro_batch=8 + grad_accum=4 (was 32 + 1). At B=32 the working set
        # exceeded M5 Max's 128 GB unified memory and the live run paged into
        # swap, costing ~14x throughput. See scripts/bench_precision.py for
        # the curve. Effective batch stays 32 microbatches per outer step.
        #
        # save_every=500: ~60 saves over Stage 1's ~30,517 total steps.
        # At ~7s/outer-step that's ~58 min between saves; a crash loses at
        # most ~1h on the ~3-day run. Tighter cadence in practice because
        # the GPU gets kill-restarted occasionally for other work; smaller
        # save window keeps restart cost manageable.
        return cls(
            peak_lr=4e-4,
            warmup_steps=1000,
            total_tokens=2_000_000_000,
            micro_batch=8,
            grad_accum=4,
            eval_every=1000,
            full_eval_every=5000,
            save_every=500,
        )

    @classmethod
    def stage2(cls) -> "TrainConfig":
        # micro_batch=8 (was 32). Same reason as stage1: B=32 swaps on M5 Max.
        # grad_accum stays 4; effective batch unchanged.
        #
        # save_every=500: ~122 saves over Stage 2's ~61,035 total steps.
        # At ~16s/outer-step that's ~2.2h between saves; a crash loses at
        # most ~2h on the ~14-day run. Each save is ~1.2 GB (305M params +
        # optimizer state); cumulative write at sustained SSD speed is well
        # under a minute over the whole run.
        return cls(
            peak_lr=3e-4,
            warmup_steps=2000,
            total_tokens=4_000_000_000,
            micro_batch=8,
            grad_accum=4,
            eval_every=1000,
            full_eval_every=5000,
            save_every=500,
        )
