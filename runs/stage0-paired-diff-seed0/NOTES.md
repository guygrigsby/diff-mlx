# Stage 0 paired diff seed 0 — run notes

## Run summary
- **Variant:** diff-attn v0 (experimental arm; two SDPA calls + Python subtract)
- **Init source:** paired-seed init protocol (design §9.7) — `runs/init-seed0/diff.safetensors`. Shared backbone + attention projections byte-identical to vanilla; lambda vectors and subln from separate RNG stream.
- **Config:** ModelConfig.stage0() + TrainConfig.stage0()
- **Total steps:** 6,103
- **Wall time:** 347.9 min (5.8 hours) — **see Throughput Anomaly below**
- **Final tps:** 4,789 tokens/sec
- **NaN/Inf:** none

## Final state
- step 6000 train_loss: 4.6515 (smoothed)
- step 6000 val_monitor: 4.6515
- step 5000 val_full: 4.6999 (perplexity ≈ 109.9)

## Paired delta vs vanilla
**Diff wins at end of training**, with smooth crossover at step ~3000 (~50% of Stage 0). δ = val_full(diff) − val_full(vanilla) = −0.0201 at step 5000 (~2% perplexity gain). δ on val_monitor trends from +0.149 (step 500) → −0.019 (step 6000), monotonically improving after crossover.

This **directionally reproduces** the paper's central claim at small scale, single seed pair.

## Throughput anomaly — flagged for Phase D investigation
Mid-run check (during training): 3,291 steps completed in 51.1 min → ~17.6k tps. Projection at that rate: ~95 min total. **Actual final wall: 347.9 min, ~3.7× slower in the second half.**

Possible causes (Phase D prereq to investigate):
- Thermal throttling on M5 Max over a 5+ hour run
- macOS scheduler / background process competing
- MLX compilation cache eviction or graph state growth
- Memory pressure from accumulated intermediate arrays

If this slowdown is reproducible at Stage 1/2 scale, the 2-3 week design budget is at risk. Mitigations to evaluate:
1. Implement the deferred bf16 mixed precision (design §9.0) — halves param/grad/optimizer memory, may relieve thermal/memory pressure
2. Implement optimizer-state checkpoints (design Phase A retro item 2) so long runs can be split into segments with cooldown between
3. Profile the diff run to find which op slows down

## RoPE / head-split
- RoPE: `mx.fast.rope(traditional=True)` (paper-canonical interleaved, fixed in design round 7)
- Head-pair split: interleaved (q[2h], q[2h+1] per diff-head) (fixed in design round 8 after PyTorch cross-check exposed the bug). If the head-split bug had landed, this run would have produced WRONG loss numbers with plausible-looking curves.
