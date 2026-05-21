# Phase A retro

**Date:** 2026-05-20
**Status:** Complete. Vanilla GPT at Stage 0 trained successfully end-to-end on FineWeb-Edu.
**Branch:** `phase-a-infra` (to be merged or tagged before Phase B begins)

## What works

- **Project scaffold + venv** with pinned MLX (0.31.2), tiktoken (0.13.0), pyarrow (24.0.0).
- **Data pipeline:** FineWeb-Edu sample-10BT download → cl100k_base tokenization → uint32 mmap shards → deterministic sampler. Tokenizing 2.15 GB of parquet produced 727M train tokens + 5M val tokens in ~3 minutes.
- **Model architecture:** RMSNorm, SwiGLU, VanillaMHA (with `mx.fast.rope(traditional=False)` and `mx.fast.scaled_dot_product_attention(mask="causal")`), Block, Transformer (tied embeddings, final RMSNorm). All 16 model tests pass; causal attention verified by perturbation test.
- **Optimizer + schedule:** AdamW factory + weight-decay exclusion split (embed, RMSNorms, lambda vectors all excluded); cosine LR with warmup. Pure stdlib schedule, MLX optimizer.
- **Training loop:** value_and_grad backward, global grad-norm clipping (clip=1.0), single optimizer.update per step. Two-tier eval (2M monitoring slice every 500 steps, full ~5M val every 2500 steps). Checkpoint saved every 1000 steps + at final step.
- **Metadata snapshots per run:** config.json, git hash, MLX/tiktoken versions, seed, data_meta. Sufficient for byte-deterministic reruns.
- **51 unit tests, all passing.**

## Stage 0 vanilla seed 0 results

| | |
|---|---|
| Wall time | 73.2 min |
| Final tps | 22.7k (peaked ~37k mid-run) |
| Step 0 train_loss | 11.99 (random init NLL = log(100277) ≈ 11.52) |
| Step 6000 train_loss | 4.65 |
| Step 5000 val_full | 4.72 (perplexity ≈ 112) |
| NaN/Inf | 0 |

Loss curve is smooth, descends as expected, val tracks train within ~0.05-0.1 nats. Two-tier eval works: val_monitor (2M tokens) matches val_full (5M tokens) within 0.01 nats at sampled milestones. Full notes in `runs/stage0-vanilla-seed0/NOTES.md`.

## Throughput projections (from real Stage 0 calibration)

| Stage | Tokens | Estimated wall (single run) | Decision |
|-------|--------|----------------------------|----------|
| Stage 1 (162M) | 2B | ~18h | Proceed; 4 runs ≈ 3 days fits within budget |
| Stage 2 (305M) | 4B | ~16h | Proceed; 4 runs ≈ 2.7 days, 6 runs (N=3 seeds) ≈ 4 days |

Total Phase B+D training compute: ~6-7 days wall over the active stages. Plus Phase C kernel work (~1 week). Fits the 2-3 week design budget.

## Known deviations from design (deferred, scoped)

These are spec gaps that don't bite at Stage 0 scale but need addressing before Phase B (paired-seed) or Phase D (Stage 1/2):

1. **Precision is pure fp32, not bf16-mixed-with-fp32-master as design §9.0 specifies.**
   - **Closed 2026-05-21.** Implemented via `LinearAMP` (option A from
     `docs/2026-05-21-bf16-mixed-precision-design.md`): fp32 params, bf16 cast
     inside forward at op boundaries. Stage 1/2 ModelConfig defaults switched
     to `amp_dtype="bfloat16"`.

2. **Optimizer state (AdamW m, v) is NOT saved in checkpoints.**
   - `save_checkpoint` only writes model params. If a long run crashes mid-way, AdamW state is lost on resume.
   - At Stage 0 (~73 min), restart-from-scratch is acceptable. At Stage 1 (~18h) or Stage 2 (~16h per run), it isn't.
   - **Address before Phase D.**

3. **`grad_accum` field in TrainConfig is ignored by train.py.**
   - Stage 0 uses `grad_accum=1` so this is invisible. Stage 2 wants `grad_accum=4` (256k effective tokens/step).
   - **Address before Phase D Stage 2.**

4. **Single `seed` parameter** seeds both model init AND data ordering.
   - Per design §9.4, these should be separable so a paired-seed comparison can use the same data order across variants while varying only model init.
   - **Address in Phase B alongside the §9.7 paired-seed init protocol.**

5. **`tokenizer.tiktoken_version()` works but the vocab assertion in `test_tokenizer.py` (asserting `100277`) needs revisiting if tiktoken updates change the count.** Tiktoken 0.13.0 reports `n_vocab = 100277` cleanly (consistent with design). Pin holds for now.

## Ready for Phase B?

- [x] All Phase A tests green (51/51)
- [x] Stage 0 vanilla loss curve sane (descending, no NaN, perplexity ~110)
- [x] Throughput projections fit the 2-3 week target
- [x] Run metadata captured (config, git, MLX/tiktoken versions, seed, data_meta)
- [x] Vanilla checkpoint saved at `runs/stage0-vanilla-seed0/latest.safetensors` (will be the starting point for Phase B's paired-seed init protocol once that's built)
- [x] FineWeb-Edu shards on disk (`data/shards/` — 727M train tokens, 5M val)

## What Phase B adds

- `DiffAttention` module (paper-canonical: H_diff = n_heads_vanilla/2, qk_dim = D, v_dim = 2D, all projections dim → dim)
- Lambda parameters + per-layer lambda computation (`exp(dot(λ_q1, λ_k1)) - exp(dot(λ_q2, λ_k2)) + λ_init`)
- subln (per-head RMSNorm over 2D, post-subtraction; shape-discipline note from design §6.3)
- v0 = two `mx.fast.scaled_dot_product_attention` calls + Python subtract (linearity rewrite from design §7.1)
- Paired-seed init protocol (design §9.7): build vanilla, serialize, build diff, copy-by-name, save both
- Reference cross-check vs `microsoft/unilm/Diff-Transformer/multihead_diffattn.py` on a fixed tensor (design §7.4)
- Stage 0 paired smoke run (vanilla seed-0 vs diff seed-0 with byte-identical shared init)
- Optionally: precision spec deviation #1 above (so Stage 1 can train fast)

After Phase B Stage 0 paired passes, Phase C handles the custom Metal kernels (P1 softmax → P2 SDPA → v1) and Phase D handles Stages 1 and 2.
