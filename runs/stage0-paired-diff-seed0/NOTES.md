# Stage 0 paired diff seed 0 — run notes

## Run summary
- **Variant:** diff-attn v0 (experimental arm; two SDPA calls + Python subtract)
- **Init source:** paired-seed init protocol (design §9.7) — `runs/init-seed0/diff.safetensors`. Shared backbone + attention projections byte-identical to vanilla; lambda vectors and subln from separate RNG stream.
- **Config:** ModelConfig.stage0() + TrainConfig.stage0()
- **Total steps:** 6,103
- **Wall time:** 347.9 min (5.8 hours) — degraded by display-sleep stalls (see "Throughput anomaly resolved" below)
- **Final tps:** 4,789 tokens/sec (degraded; caffeinated re-run sustained 18,003 tps)
- **NaN/Inf:** none

## Final state
- step 6000 train_loss: 4.6515 (smoothed)
- step 6000 val_monitor: 4.6515
- step 5000 val_full: 4.6999 (perplexity ≈ 109.9)

## Paired delta vs vanilla
**Diff wins at end of training**, with smooth crossover at step ~3000 (~50% of Stage 0). δ = val_full(diff) − val_full(vanilla) = −0.0201 at step 5000 (~2% perplexity gain). δ on val_monitor trends from +0.149 (step 500) → −0.019 (step 6000), monotonically improving after crossover.

This **directionally reproduces** the paper's central claim at small scale, single seed pair.

## Throughput anomaly resolved
**Root cause: display power state.** The run was unattended for 5.8 hours; the external display slept partway through, dropping the GPU into a low-power state and producing massive intermittent stalls (495 s and 23 s outliers observed in the post-hoc diagnostic). Not thermal, scheduler, MLX cache, or memory pressure.

**Validation:** caffeinated re-run at `runs/stage0-paired-caffeinated/stage0-paired-diff-seed0/` with `caffeinate -disu` completed in 92.6 min (3.76× faster). Per-step train_loss reproduced within 1e-4 at step 6000, so the loss curves and paired δ in this directory remain valid.

**Diagnostic data:** `runs/diag-diff-monitor-{on,off}.jsonl` (from `scripts/diagnose_throughput.py`).

**Full writeup:** `docs/2026-05-20-phase-b-retro.md`, "Throughput anomaly resolved".

## RoPE / head-split
- RoPE: `mx.fast.rope(traditional=True)` (paper-canonical interleaved, fixed in design round 7)
- Head-pair split: interleaved (q[2h], q[2h+1] per diff-head) (fixed in design round 8 after PyTorch cross-check exposed the bug). If the head-split bug had landed, this run would have produced WRONG loss numbers with plausible-looking curves.
