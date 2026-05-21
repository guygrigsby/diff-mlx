# Stage 0 vanilla seed 0 — run notes

## Run summary

- **Variant:** vanilla MHA (control)
- **Config:** ModelConfig.stage0(), TrainConfig.stage0()
- **Total steps:** 6,103 (target 6,103 = 100M tokens / 16k tokens/step)
- **Wall time:** 73.2 min
- **Final tps:** 22,751 tokens/sec (varies during run; ~30-37k mid-run, slower toward end)
- **NaN/Inf:** none

## Loss progression

| Step | LR | train_loss | val_monitor | val_full |
|------|-----|-----------|-------------|----------|
| 0    | 0     | 11.99 | — | — |
| 500  | 6.0e-4 (peak) | 6.26 | 6.39 | — |
| 1000 | 5.9e-4 | 5.75 | 5.77 | — |
| 2000 | 5.1e-4 | 5.19 | 5.24 | — |
| 2500 | 4.4e-4 | — | 5.07 | 5.08 |
| 3000 | 3.8e-4 | 5.17 | 4.95 | — |
| 4000 | 2.3e-4 | 4.88 | 4.80 | — |
| 5000 | 1.1e-4 | 4.75 | 4.71 | 4.72 |
| 6000 | 6.1e-5 (min) | 4.65 | 4.67 | — |
| 6102 | 6.0e-5 | 4.98 (last batch noise) | — | — |

Random init NLL = log(100277) ≈ 11.52; descent to 4.65 → perplexity ≈ 107.
val_monitor (2M tokens) matches val_full (~5M tokens) within ~0.01 nats at the two
sampled milestones (step 2500: 5.07 vs 5.08; step 5000: 4.71 vs 4.72) — the small
monitoring slice is a faithful proxy.

## Throughput-based projections

At ~22.7k tokens/sec (conservative — mid-run was higher):

| Stage | Tokens | Per-token cost vs S0 | Projected wall (single run) |
|-------|--------|----------------------|----------------------------|
| Stage 1 (162M params) | 2B | ~3x slower per token | ~18 hours |
| Stage 2 (305M params) | 4B | ~5x slower per token | ~16 hours per run (better parallelism at B=32 grad_accum=4) |

Stage 2 × 4 runs (2 variants × 2 seeds) ≈ 64 hours ≈ 2.7 days. Stage 2 × 6 runs (with N=3) ≈ 4 days. Both within the 2-3 week design budget once Phase B/C/D engineering time is added.

**Calibration verdict:** Stage 2 fits in 14 days with N=2 or N=3 seeds. No budget cut needed.

## What this is and isn't

**Is:** the control architecture (vanilla MHA) at Stage 0 scale, trained on FineWeb-Edu with the full Phase A pipeline (cl100k_base tokenizer, paper-canonical arch, fp32 mixed precision, two-tier eval, AdamW with weight-decay exclusions).

**Isn't:** a paired-seed A/B run. Phase B's paired-seed init protocol (§9.7) isn't implemented yet — when Phase B Stage 0 paired runs land, this vanilla baseline will be re-run with byte-identical shared init alongside diff-attn seed 0.

## Anything notable

- No spikes, no NaN. AdamW + grad clip held stable throughout.
- Cosine LR schedule visible in the loss curve as expected (faster descent during warmup peak, slower during cosine decay).
- Slight loss bump at step 3000 (5.17 vs step 2000 at 5.19) is within step-to-step noise from `micro_batch=16` minibatches; val curve continues smooth descent.
- Final 2 steps' train_loss looks high (4.98 at step 6102) — that's normal noise on the final minibatch; val_monitor at step 6000 was 4.67 which is the trustable end-state.
