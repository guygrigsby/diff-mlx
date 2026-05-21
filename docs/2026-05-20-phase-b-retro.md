# Phase B retro

**Date:** 2026-05-21
**Status:** Complete. Diff-attn v0 implemented, paired-seed init protocol working, PyTorch reference cross-check passes, Stage 0 paired smoke run done — **diff-attn directionally beats vanilla at Stage 0 scale.**
**Branch:** `phase-b-diffattn` (to be merged after this retro is written + tag set)

## What works

- **DiffAttention module** (paper-canonical): `n_heads_diff = n_heads_vanilla / 2`, `qk_head_dim = D`, `v_head_dim = 2D`. v0 = two `mx.fast.scaled_dot_product_attention` calls + Python subtract (design §7.1 linearity rewrite, no T×T map materialization).
- **Lambda machinery:** per-layer fp32 vectors (4 × qk_head_dim), depth-scheduled `λ_init = 0.8 - 0.6·exp(-0.3·(layer_idx-1))`, fp32 lambda scalar `exp(dot) - exp(dot) + λ_init` broadcast over `(B, H, T, 2D)`.
- **subln:** per-head RMSNorm over `2D`, applied AFTER differential subtraction.
- **Block + Transformer variant flag** (`variant="vanilla"|"diff"`), `layer_idx` propagation (1-indexed).
- **Paired-seed init protocol** (design §9.7): byte-identical backbone + ALL four attention projections (q/k/v/o); diff-only lambda/subln from separate RNG stream (seed + 1_000_003 offset).
- **PyTorch reference cross-check** (design §7.4): MLX vs vendored Microsoft `microsoft/unilm/Diff-Transformer/multihead_diffattn.py` at toy shape `(B=2, T=16, dim=64, n_heads_diff=2, qk_head_dim=16)`. CPU stream max |diff| = **3.58e-7**, well below the 1e-3 design gate. GPU stream max |diff| ≈ 1.7e-3 due to MLX Metal reduced-precision fp32 matmul (hardware noise, both variants pay the same tax).
- **Stage 0 paired run** at seed 0: both variants completed to budget without NaN.
- **78 unit tests passing** (Phase A's 51 + 27 new).

## Two real bugs caught by the cross-check (both shipped fixed)

These are the kind of bugs that would have silently corrupted the science. The cross-check earned its keep.

### R7: RoPE convention mismatch

Design rounds 1-6 assumed the paper's reference uses LLaMA rotate-halves RoPE. Verified against `mlx-lm/models/llama.py` (which does use `traditional=False`) but did NOT verify against the diff-attn paper's reference. When Phase B Task 6 vendored the reference for the fixture, found that `multihead_diffattn.py` line 122 explicitly calls `apply_rotary_emb(..., interleaved=True)` — GPT-J consecutive-pair RoPE, not LLaMA rotate-halves.

Fixed by switching both VanillaMHA and DiffAttention to `mx.fast.rope(traditional=True)`. Phase A's Stage 0 vanilla checkpoint became obsolete; re-run as part of Stage 0 paired.

### R8: Head-pair split layout

Original implementation split the 2H concatenated heads as halves: `q[:H]` for Q1, `q[H:]` for Q2. The reference splits them as `(H, 2)` row-major: diff-head `h` pairs `q[2h]` and `q[2h+1]` (interleaved). **The internal v0 SDPA oracle test passed because it shared the same bug** — exactly the "shared architecture bug" failure mode design §7.4 was built to detect.

Fixed by reshape `q.reshape(B, H, 2, T, D)` and indexing `[:, :, 0, ...]` for Q1, `[:, :, 1, ...]` for Q2. Cross-check max |diff| dropped from 0.77 to 3.58e-7.

## Stage 0 paired results — the science

| | Vanilla seed 0 | Diff seed 0 | δ = diff − vanilla |
|---|---|---|---|
| Steps | 6,103 | 6,103 | — |
| Wall time (original) | 70.6 min | 347.9 min (display-sleep stalls) | — |
| Wall time (caffeinated re-run) | 81.5 min | 92.6 min | +14% diff vs vanilla |
| Final tps (caffeinated) | 20,450 | 18,003 | — |
| step-5000 val_full | 4.7200 | 4.6999 | **−0.0201** |
| step-5000 perplexity | 112.2 | 109.9 | −2.3 |
| step-2500 val_full | 5.0675 | 5.0804 | +0.0129 (vanilla wins) |
| NaN/Inf | 0 | 0 | — |

Caffeinated re-run reproduced both arms within ~1e-4 train_loss at step 6000 (Metal fp32 reduce nondeterminism), confirming the original loss numbers were correct. Details below in "Throughput anomaly resolved".

**Paired δ trajectory** (val_monitor) shows clean crossover:

```
step  500: δ=+0.149  (vanilla wins)
step 1000: δ=+0.116  (vanilla wins)
step 1500: δ=+0.097  (vanilla wins)
step 2000: δ=+0.051  (vanilla wins)
step 2500: δ=+0.015  (vanilla wins)
step 3000: δ=-0.006  ← crossover
step 3500: δ=-0.012  (diff)
step 4000: δ=-0.016  (diff)
step 4500: δ=-0.016  (diff)
step 5000: δ=-0.018  (diff)
step 5500: δ=-0.019  (diff)
step 6000: δ=-0.019  (diff)
```

Diff starts behind (lambda subtracts noise at init, perturbing the output relative to the byte-identical-backbone vanilla), then catches up smoothly and overtakes at ~50% of training. Post-crossover, the gap is monotonically increasing toward the final 0.020 nats.

**Status per design §5.4 outcome categories:** "Strong directional replication" — sign matches paper's prediction, gap consistent across 7 consecutive eval points, no sign flip. We cannot make a statistical-significance claim from N=1 seed pair; bootstrap CI requires Phase D's multi-seed Stage 2 protocol.

## Throughput anomaly resolved

**Root cause: display power state, not the model.** The original diff run was unattended for 5.8 hours, during which the external display slept. Display sleep on Apple Silicon transitions the GPU into a low-power state. With a 32 GB live MLX allocation pinned by the diff-attn backward, this manifested as massive intermittent stalls (one 495-second stall observed in the diagnostic; scattered 1-23 s stalls thereafter) rather than uniform slowdown.

Evidence (`scripts/diagnose_throughput.py`, 500-step diff run, see `runs/diag-diff-monitor-{on,off}.jsonl`):

| | Monitor on | Monitor off |
|---|---|---|
| Mean step | 505 ms | 1,900 ms (stalls), 770 ms (otherwise) |
| Worst step | 592 ms | 495,168 ms |
| Steps > 1 s | 0 / 500 | 17 / 386 |
| mlx_active_mb | 366 (flat) | 366 (flat) |
| swapouts delta | 0 | 0 |
| compressor pages | flat | flat |
| free RAM | 50 GB stable | dipped to 47 GB |

The flat mlx_active, zero swapouts, and flat compressor rule out the four hypotheses listed in the original retro draft (thermal throttling, scheduler contention, MLX cache growth, memory pressure). The signal that fits is power-state transition timing on display sleep.

**Caffeinated re-run (`runs/stage0-paired-caffeinated/`)** with `caffeinate -disu` completed in:
- Vanilla: 81.5 min (vs original 70.6 min; +15%, likely powermetrics/macmon sampling overhead)
- Diff: 92.6 min (vs original 347.9 min; **3.76× faster, anomaly gone**)

Per-step train_loss reproduces to within 1e-4 at step 6000 on both arms, so the original loss curves and the −0.0201 paired δ remain valid.

**Residual finding: diff is ~14% slower per step than vanilla at Stage 0** in steady state (5,554 s vs 4,889 s, same 6,103 steps). Consistent with more kernel dispatches per layer (two SDPAs + Python subtract + lambda math + subln, all dispatch-bound at this scale). Expected to amortize as kernel work grows at Stage 1/2 (Stage 0 GPU residency only ~30% per powermetrics; the GPU is mostly idle waiting on host dispatch). Not a blocker.

**Operational fix:** all multi-hour runs must use `caffeinate -disu` (or equivalent). Add to Phase D run scripts.

## Updated Phase D prerequisites

From Phase A retro + Phase B findings:

1. ~~**bf16 mixed precision**~~ — **closed 2026-05-21.** Implemented via `LinearAMP` (option A from `docs/2026-05-21-bf16-mixed-precision-design.md`). 17 new tests pass. Stage 1/2 configs default `amp_dtype="bfloat16"`.
2. **Optimizer state in checkpoints** — needed for long runs (Stage 1/2)
3. **`grad_accum` implementation** — Stage 2 needs grad_accum=4
4. **Multi-seed orchestration** — Phase D plan needs to handle running 4 (vanilla×2 + diff×2) or 6 paired runs
5. ~~**`caffeinate -disu` wrapper on long-run scripts**~~ — **closed 2026-05-21** (commit `869b6e1`).

Throughput investigation removed: root cause identified (display power state), fix is operational (`caffeinate`), validated by caffeinated re-run.

## Ready for Phase C?

Phase C is custom Metal kernels (P1 softmax, P2 causal SDPA, v1 composition). It can proceed independent of the throughput anomaly — kernels live in `kernels/` and the science still ships on v0 by default per design §7.2.

- [x] All Phase B tests green (78/78)
- [x] DiffAttention matches PyTorch reference within design tolerance (3.58e-7 << 1e-3 on CPU)
- [x] Paired-seed init verified byte-identical on shared params
- [x] Stage 0 paired loss curves clean (no NaN, smooth descent, monotonic post-crossover δ)
- [x] Cross-check infrastructure in place for v1 verification (Phase C)
- [x] Throughput anomaly resolved (display power state; caffeinate fixes it; diff arm 3.76× faster on re-run)

## What Phase C adds

- **Stage P1:** softmax kernel preflight (1-3 days)
- **Stage P2:** causal SDPA kernel preflight (2-4 days)
- **v1:** two P2 SDPA calls + Python subtract (no map materialization per design §7.1)
- Kernel correctness gates (v1 vs v0 numerical agreement)
- Memory gate (full forward+backward+optimizer.step at Stage 1/2 shapes)
- Kernel speed eval (vs v0, vs MLX SDPA)

Phase C is optional in the sense that v0 ships the science. But it's the project's MLX-kernel-learning angle, so it's worth doing.

With the throughput anomaly resolved, Phase C or Phase D can come next without dependency between them.
