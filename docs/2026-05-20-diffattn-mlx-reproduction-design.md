# diff-mlx: Design document

**Status update 2026-05-22 (revised same day):** Project scope went through a pivot and a partial un-pivot in one day.

1. First pass (morning, `docs/2026-05-22-stage1-pivot-retro.md`): Stage 1 throughput measured at ~950 tps, projecting ~50 days/variant. Descoped to "reduced Stage 1 + Phase C kernels" with Stage 2 dropped.

2. Same day (`docs/2026-05-22-swap-cliff-and-scope-restore.md`): the ~950 tps was actually swap-thrashing at micro_batch=32 (working set exceeded 128 GB unified memory). Dropping to `micro_batch=8 grad_accum=4` + a fix to `train_step_with_accum` + extending `mx.compile` through the accum path together restored throughput to ~14k tps. Full 2B Stage 1 paired is back at ~4 days; Stage 2 paired single-seed at ~14 days.

Active plan: `docs/2026-05-22-phase-c-plan.md` (updated to restore Stage 1/2 ambition while keeping Phase C kernels as the novelty piece). Stage 1/2 token counts in §5 below are now the operating budgets again. Multi-seed Stage 2 at 4+ seeds remains off the table; 2 seeds is the ceiling.

**Status:** Design approved 2026-05-20; revised 2026-05-20 across eight review passes. R1-R6 captured in §16. **R7 (Phase B implementation finding):** while vendoring `microsoft/unilm/Diff-Transformer/multihead_diffattn.py` for the reference cross-check fixture, discovered the reference calls `apply_rotary_emb(..., interleaved=True)` — GPT-J consecutive-pair rotation, NOT LLaMA's rotate-halves. R6's choice of `traditional=False` was self-consistent for "LLaMA convention" but wrong for paper fidelity. Both variants (vanilla + diff) switched to `traditional=True` to match the paper's reference. The internal A/B is still apples-to-apples (both variants use the same convention); the cross-check test can now compare directly without weight pre-permutation. The Phase A Stage 0 vanilla checkpoint trained under `traditional=False` is obsolete and will be re-run. **R8 (Task 7 implementation finding):** the cross-check test caught a second paper-fidelity bug — §6.3 specified `q1, q2 = q[:, :H, :, :], q[:, H:, :, :]` (halves split) but the reference does `attn_weights.view(bsz, num_heads, 2, T, T)` (row-major (H, 2) split, i.e. interleaved Q1=q[2h], Q2=q[2h+1]). Internal v0 oracle agreement masked the bug because the oracle used the same split as the module. §6.3 and `model.DiffAttention` switched to the interleaved split. With interleaved split + RoPE traditional=True + CPU-stream matmul, the cross-check reaches ~1e-7 against the PyTorch reference; on GPU stream it reaches ~1.7e-3 due to Metal's reduced-precision fp32 matmul.
**Project home:** `/Users/guygrigsby/projects/diff-mlx/`
**Owner:** Guy J. Grigsby (`guy@grigsby.dev`)

---

## 0. TL;DR for a cold-starting session

`diff-mlx` is a small-scale ML research project — a **controlled, small-scale reproduction** of the **Differential Transformer** paper (Ye et al., ICLR 2025, [arXiv 2410.05258](https://arxiv.org/abs/2410.05258)) **in MLX** on Apple Silicon, with **custom Metal kernels** for the diff-attention forward pass (v1 composition over a causal-SDPA kernel; fused v2 as stretch). Deliverables: trained checkpoints (vanilla MHA vs diff-attn, at Stage 2 scale ~305M, ~4B tokens each, multi-seed), custom Metal kernels for the diff-attn path, a comparison writeup, and a small reusable MLX package. This is not a direct paper reproduction — scale, corpus, sequence length, and token count all differ from the paper. The internal A/B (vanilla vs diff at matched scale and corpus) is what we control for cleanly.

**Hardware:** M5 Max, 40 GPU cores, 128GB unified RAM. Pure MLX. Single GPU. The Metal kernel is Apple-only.

**Time budget:** target ~2-3 weeks wall time, likely longer if v1 work extends. P1 + P2 preflights alone can take up to 7 days by their own time-boxes (§5.1, §5.1b); Stage 2 is 4-6 full runs at ~305M. Calibration after Stage 0 step 200 (§5.2) is the early-warning signal if total wall is projecting past 4 weeks.

**Math:** all written out below in Section 6, taken directly from the paper's Section 2 and cross-checked against the paper's public reference repository.

---

## 1. Background and motivation

### 1.1 Differential Transformer

The Differential Transformer ([arXiv 2410.05258](https://arxiv.org/abs/2410.05258), ICLR 2025 oral) replaces standard multi-head attention with **differential attention**: each logical attention head computes **two** softmax attention maps and **subtracts** them, weighted by a learned per-layer scalar λ. The subtraction cancels attention noise the way a differential amplifier cancels common-mode signal.

Key claimed advantages:
- Matches vanilla loss at ~65% of training tokens
- Sparser attention patterns
- Better long-context recall
- Reduced activation outliers (quant-friendly)
- Better in-context learning

Follow-up papers worth reading for the writeup:
- [arXiv 2505.16333](https://arxiv.org/pdf/2505.16333) — "Understanding Differential Transformer Unchains Pretrained Self-Attentions" (mechanism analysis)
- [arXiv 2510.06949](https://arxiv.org/pdf/2510.06949) — "Grouped Differential Attention" (head-allocation extension; possible Stage 2 follow-up)

### 1.2 The MLX + Metal kernel angle

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for Apple Silicon. It exposes a custom-Metal-kernel API via [`mx.fast.metal_kernel`](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), which lets you write Metal Shading Language (MSL) source as a string, declare grid/inputs/outputs, and integrate with MLX autograd via `mx.custom_function`.

As of May 2026, prior-art search shows:
- **No MLX implementation of differential attention exists.** GitHub API searches for "differential transformer mlx" and "diff attention mlx" both return 0 repos.
- **PyTorch implementations exist but are dormant**: [`kyegomez/DifferentialTransformer`](https://github.com/kyegomez/DifferentialTransformer) (41⭐, 6 commits, no released weights), [`axolotl-ai-cloud/diff-transformer`](https://github.com/axolotl-ai-cloud/diff-transformer) (7⭐, 16 commits, no released weights).
- **No released pretrained diff-attn weights at 100-300M scale.** The paper's published checkpoints are at 3B+.
- **Adjacent precedent for MLX + novel-attention kernels** exists: [`humanrouter/ddtree-mlx`](https://github.com/humanrouter/ddtree-mlx) wrote first-of-its-kind Metal kernels for tree attention. Working template to learn from.

### 1.3 Fresh on three axes

- First MLX port of differential attention (V1)
- First custom Metal kernel for diff-attn
- First clean small-scale (~300M) A/B reproduction in any framework with released weights

### 1.4 DIFF V2 — out of scope for this project, possible follow-up

Microsoft published a [Differential Transformer V2](https://huggingface.co/blog/microsoft/diff-attn-v2) on 2026-01-20 (HF blog; no arXiv paper yet, full benchmarks still pending). Code at [`microsoft/unilm/tree/master/Diff-Transformer/Diff-Transformer-V2`](https://github.com/microsoft/unilm/tree/master/Diff-Transformer/Diff-Transformer-V2). V2 makes three substantive changes vs the V1 reproduced here:

- **Inference compatibility:** doubles query heads (2H) and keeps KV heads at H (GQA-style). Standard `flash_attn_func` drops in without modification. KV cache loaded once instead of twice.
- **Stability:** drops the per-head `subln` RMSNorm. The blog identifies it as the root cause of gradient spikes at scale — near-uniform attention over long sequences (n=8192) gets a √n ≈ 90× scale-up from the RMSNorm, producing ~100× gradient magnification.
- **Dynamic lambda:** replaces V1's global `λ = exp(dot(λ_q1, λ_k1)) − exp(dot(λ_q2, λ_k2)) + λ_init` with a token-and-head-specific `λᵢ = sigmoid(W·xᵢ)` of shape `(N, H, 1)`. No special init schedule needed; output context bounded in `(0, √2)`.

**Why we're sticking with V1 for this project:**

1. The project's core deliverable is **custom Metal kernels for diff-attn**. V2 uses standard FlashAttention; the kernel-learning angle largely evaporates.
2. **First-of-its-kind status holds.** "First MLX port of differential attention" applies to V1; V2 has no MLX port either, no released checkpoints at 300M, and the original DIFF paper (ICLR 2025 oral) is the reference paper in the literature.
3. **V2's scale concerns don't bite us as hard.** The gradient-spike pathology V2 fixes is driven by long-context near-uniform attention (paper uses 8k+). Stage 2 here is T=2048 with shorter, more peaked attention; subLN amplification is a smaller risk at this scale.
4. **The bulk of this design transfers.** Precision spec, paired-seed init, eval methodology, kernel preflight ladder, training plan — all reusable for a follow-up V2 reproduction if we ever do one.

V1 stays the target. If a future project picks up V2, the natural framing is "V1 vs V2 controlled comparison in MLX" using whatever we ship here as the V1 baseline.

---

## 2. Hypothesis under test

> At ~305M parameters (Stage 2 backbone) and ~4B training tokens, in pure MLX with a single English corpus (FineWeb-Edu) and the paper-matched tokenizer (tiktoken `cl100k_base`), **differential attention matches or beats vanilla multi-head attention** in token-level validation loss, replicating the paper's central claim at small scale.

Secondary hypotheses (worth recording even if Stage 1 weakens them):

- The paper-claimed efficiency (matching vanilla at ~65% tokens) is visible at this scale.
- The signal is detectable above seed-variance with 2-3 seeds per variant.

A **negative result** (diff-attn does not replicate at Stage 2 scale) is itself a publishable finding.

---

## 3. Scope and deliverables

### 3.1 Deliverables at the end

1. **Trained checkpoints** at three stages (~30M / ~160M / ~305M total params, see §6.1), each with both attention variants:
   - Stage 0: ~100M tokens, single seed per variant (smoke test)
   - Stage 1: ~2B tokens × 2 seeds per variant target. **Stage 2 advancement** requires only that one paired seed completes to budget plus the second pair reaches 50% (§5.3 gate). **Final Stage 1 paired-delta report** uses whichever seeds completed: if both pairs are full, paired Δ over 2 seeds; if only the first pair is full, single-seed Δ with a note that the second pair was truncated. Either is acceptable; the deliverable is the curves and the deltas as measured.
   - Stage 2: ~4B tokens × 2-3 seeds per variant (full reproduction)
2. **Custom Metal kernels** for the diff-attention forward path (v0 pure-MLX baseline always; v1 composition over a causal-SDPA kernel; v2 single fused kernel as stretch — see §7).
3. **Loss curves and paired-seed deltas** at Stages 1 and 2; held-out perplexity table at all three stages (gives a 3-point mini scaling curve as a bonus).
4. **Writeup** (README + blog-post style) covering: kernel design, training setup, results, ablations, what we learned.
5. **(Stretch)** HF Hub release of checkpoints and the kernel as a small reusable MLX package.

### 3.2 Out of scope

Listed explicitly so future-me does not get tempted:

- Long-context extension (16k/32k context)
- Instruction tuning, RP, or any post-training
- Multi-GPU, DDP, FSDP, sharded optimizer
- KV block split (Apple Intelligence tech report 2025)
- Sliding-window attention (SWA)
- Mixture of Depths (MoD)
- Grouped-query attention (GQA) — Variant A uses full MHA; Variant B uses the paper's diff-attn head configuration (§6.3), which has its own per-head structure but is not GQA
- LoRA / QLoRA
- Quantization-aware training
- Inference optimization

All intentionally stripped to keep the diff-attn vs vanilla A/B clean: within this project, **only attention type varies between variants**. Note this is the *internal* claim — compared to the original paper, our setup differs in scale, corpus, sequence length, and total training tokens. The writeup should frame this work as a small-scale controlled reproduction of the diff-attn mechanism, not a direct reproduction of the paper's 3B/cl100k runs.

---

## 4. Hardware and software stack

| | |
|---|---|
| Machine | M5 Max, 40 GPU cores, 128GB unified RAM |
| OS | macOS (latest) |
| Framework | MLX (pin version in `pyproject.toml`; do not auto-upgrade mid-project) |
| Python | 3.12+ |
| GPU API | Metal (via MLX), Metal Shading Language (MSL) for custom kernels |
| Tokenizer | tiktoken `cl100k_base` (~100k vocab, paper-canonical), pinned via `tiktoken==<version>` in `pyproject.toml` |
| Corpus | FineWeb-Edu (English, ~6-8B tokens used) |
| Storage | uint32 token shards on local disk (~24-32GB; cl100k_base vocab >65535 requires uint32) |

### 4.1 Sharing the machine during runs

MLX shares GPU and memory with the OS. Implications during multi-day runs:

- Training pins GPU at 80-100% → UI scrolls and animations stutter (workable; not a hard block).
- **Light tasks fine:** code editing, terminal, reading, email, Slack, light browsing.
- **Heavy GPU tasks blocked:** local LLM inference (LM Studio with Anubis 70B will fight for GPU and tank both runs), video editing, games, 3D rendering.
- **Run on AC power.** macOS aggressively throttles GPU on battery.
- **Kill LM Studio's loaded model** during runs (it pins GPU memory).
- **Use `tmux` or `screen`** so the training process survives terminal restarts.
- **Use `caffeinate -di`** so display sleep / lid close does not kill the run.

---

## 5. Staged plan with explicit gates

Each stage's pass criteria must be met before proceeding. Failure of a gate triggers a documented fallback, not a "push through anyway."

### 5.1 Stage P1 — Softmax kernel preflight (1-3 days)

**Goal:** verify the MLX custom-kernel layer (write MSL, wire through `mx.fast.metal_kernel`, register autograd hook via `mx.custom_function`, gradient-check, run on M5 Max) is tractable before committing to diff-attn kernel complexity.

**Target kernel:** Softmax along the last dim (with the max-subtraction stability trick).

**Why softmax for P1:**
- Two reductions (max, then sum), and one element-wise pass — exercises the reduction primitives the diff-attn kernel needs
- Known, well-tested reference (`mx.softmax(x, axis=-1)`)
- Backward is non-trivial (softmax Jacobian) — good autograd practice
- Fallback if softmax stalls: RMSNorm (simpler, one reduction)

**Pass criteria:**
1. Forward output matches `mx.softmax(x, axis=-1)` within `1e-4` (fp32) or `1e-2` (bf16)
2. Backward via `mx.custom_function` matches autograd of pure-MLX softmax
3. Finite-difference gradient check passes on a small tensor (e.g. shape `(2, 4, 8)`)
4. The implementer feels the API and debugging loop is workable, not miserable

**Time box:** 1-3 days. If not over the hump by day 3, **declare the v1/v2 kernel goal too tall**, ship v0-only and rebudget Stages 0-2 to include more analysis or seeds.

**Gate behavior:**
- **Pass** → proceed to Stage P2.
- **Fail / stall** → drop v1/v2, keep v0 (pure-MLX diff-attn) as the only kernel path; skip P2.

### 5.1b Stage P2 — Causal SDPA kernel preflight (2-4 days)

**Goal:** de-risk the parts of the diff-attn kernel that P1 does not touch: tiled QK matmul, causal masking, row-softmax over long `T`, AV accumulation, `(B, H, T, D)` indexing and strides, bf16 inputs with fp32 accumulators, transient-memory pressure.

**Target kernel:** single-map causal scaled dot-product attention. Inputs Q, K, V each `(B, H, T, D)` in bf16; output `(B, H, T, D)` in bf16. Reference: `mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0/sqrt(D), mask="causal")` (note: MLX's API has no `is_causal` flag; `scale` is mandatory keyword-only and `mask="causal"` is the documented way to apply a causal mask).

**Pass criteria:**
1. Forward output matches `mx.fast.scaled_dot_product_attention` within `1e-2` (bf16) on **all** of these shapes:
   - Toy: `(B=2, H=2, T=128, D=32)` — first signal
   - Stage 0 vanilla shape: `(B=16, H=4, T=1024, D=64)`
   - Stage 0 diff sub-head shape (one of the two maps): `(B=16, H=2, T=1024, D=64)`
   - Stage 1 vanilla shape: `(B=32, H=12, T=2048, D=64)`
   - Stage 1 diff sub-head shape: `(B=32, H=6, T=2048, D=64)`
2. Backward via `mx.custom_function` (pure-MLX gradient) matches MLX autograd of the reference SDPA on a 4-layer / 2-head / D=32 toy
3. Peak transient memory measured and reported at the largest of the above shapes; must be within budget for the v1 promotion gate (see §7.2)
4. The kernel runs at least as fast as a pure-MLX SDPA composed of `mx.softmax` + `mx.matmul` (does not have to beat the built-in SDPA)

Note: v1 calls this exact kernel twice with shared V at width `2D` (see §7.1), so the SDPA kernel API **must** accept `head_dim_qk` and `head_dim_v` independently. Add diff-attn shapes to P2's forward-check matrix:

- Stage 0 diff-attn shape: `(B=16, H=2, T=1024, D_qk=64, D_v=128)`
- Stage 1 diff-attn shape: `(B=32, H=6, T=2048, D_qk=64, D_v=128)`

These complete the shape coverage. P2 directly de-risks v1; if these pass, v1's forward path is mostly composition plumbing.

**Gate behavior:**
- **Pass** → v1 is on the table for training use after the v1-specific gate in §7.2.
- **Fail / stall** → v1/v2 stay as correctness artifacts only; all training runs use v0.

P1 and P2 together verify the full kernel toolchain needed for v1. They are independent of the science: if both fail, Stages 0-2 still ship on v0.

### 5.2 Stage 0 — Engineering smoke test (~1 day compute)

**Goal:** kernel forward correctness, gradient flow, training stability, eval scripts work, infra end-to-end.

**Config:**
- ~30M params total (hidden=256, layers=6, `n_heads_vanilla=4`, head_dim=64; embed dominates)
- ~100M tokens
- Single seed per variant
- block_size=1024 (smaller than later stages to speed iteration)
- batch_size=16

**Pass criteria:**
- Both variants (vanilla, diff-attn-v0) train without crashing
- v0 (pure-MLX diff-attn) matches an explicit SDPA-composed oracle within numerical noise on a forward test. **Oracle definition** (do not compare diff-attn to a single SDPA call):
  ```python
  # inputs (post-RoPE, post-reshape):
  #   q1, k1: (B, H_diff, T, D);  q2, k2: (B, H_diff, T, D);  v: (B, H_diff, T, 2D)
  #   lam: scalar;  lambda_init: scalar (depth-scheduled)
  scale = 1.0 / sqrt(D)  # D = qk_head_dim
  attn1 = mx.fast.scaled_dot_product_attention(q1, k1, v, scale=scale, mask="causal")   # (B, H_diff, T, 2D)
  attn2 = mx.fast.scaled_dot_product_attention(q2, k2, v, scale=scale, mask="causal")   # (B, H_diff, T, 2D)
  out   = attn1 - lam * attn2                                                # (B, H_diff, T, 2D)
  out   = subln(out)                                                         # per-head RMSNorm over 2D
  out   = (1 - lambda_init) * out                                            # depth scaling
  # (output projection compared separately; the oracle stops before o_proj)
  ```
  This **is** exactly v0's definition (and v1's, see §7.1) — two SDPA calls sharing V, each with its own softmax over its corresponding QK pair, subtracted with λ. What it is **not** is a single SDPA call: don't try to validate diff-attn by comparing it to one call to `mx.fast.scaled_dot_product_attention`.
- Gradient check on the diff-attn layer (finite-diff vs analytical) passes on a 4-layer / 64-hidden / 2-diff-head toy
- Training loss descends smoothly; no NaNs in ~3,000 steps
- Reference-implementation cross-check (§7.4) passes before this stage begins

**Calibration deliverable:** after step 200, read `tokens/sec` from `metrics.jsonl` and compute projected wall time for Stage 1 and Stage 2. If projection blows past 14 days for Stage 2, cut Stage 2 to 3B tokens or drop a seed.

### 5.3 Stage 1 — Signal pilot (~3-4 days compute)

**Goal:** verify the pipeline holds at 12-layer scale and look at the diff-attn vs vanilla curves to inform Stage 2 decisions. Signal observation is exploratory — not a pass/fail criterion.

**Config:**
- ~162M params total (hidden=768, layers=12, `n_heads_vanilla=12`, head_dim=64)
- ~2B tokens (Chinchilla-ish)
- 2 seeds per variant
- block_size=2048
- batch_size=32, effective 64k tokens/step (no grad accum)
- ~31,000 steps per run

**Pass criteria** (advancement to Stage 2 requires all three; minimum bar is intentionally light to keep Stage 1 cheap):
1. **Correctness:** finite-diff gradient check on the active kernel still passes; v0/v1 numerical agreement still within tolerance (§7.4)
2. **Stability:** at least one paired seed (vanilla + diff at seed `s₁`) completes to full Stage 1 token budget without NaN/Inf; the second seed pair reaches at least 50% of budget without NaN/Inf. Loss descent smooth; grad norms bounded across the runs that completed
3. **Throughput projection:** measured tokens/sec at Stage 1 shapes implies Stage 2 fits within 14 days wall (or a documented Stage 2 budget cut is applied)

If both seed pairs run to full budget, that's better — it gives a richer exploratory picture (§5.3 exploratory output). But all-four-runs-to-budget is **not** a hard gate; pipeline confidence and Stage 2 throughput projection come from the first complete paired seed plus the partial second pair.

**Exploratory output (informs Stage 2 plan, not a gate):**
- Plot val loss vs tokens-trained, both variants, both seeds.
- Report paired-seed deltas (diff-attn seed `s` minus vanilla seed `s` on the same data order).
- Note any seed-flips on the winner; do not treat them as success or failure.

**Decision branches after looking at the curves:**
- **Clear diff-attn signal:** proceed to Stage 2 as planned.
- **No signal or mixed signal:** still proceed to Stage 2 if compute budget allows; the paper's strongest claim is scaling-direction and Stage 1 noise at 162M may not predict Stage 2.
- **Kernel signal suspect (curves look pathological):** rerun gradient/forward checks before Stage 2. If kernel is verified correct, proceed to Stage 2 and treat the negative result as honest data.

### 5.4 Stage 2 — Full reproduction (~9-10 days compute)

**Goal:** paper-style A/B with paired-seed deltas at Stage 2 scale (~305M).

**Config:**
- ~305M params total (hidden=1024, layers=16, `n_heads_vanilla=16`, head_dim=64; embed ≈ 103M with cl100k_base)
- ~4B tokens
- 2-3 seeds per variant (3 if compute budget allows after Stage 0 calibration)
- block_size=2048
- micro-batch 32, grad accum 4, effective batch 256k tokens/step
- ~15,600 steps per run

**Outcome interpretation (no binary pass/fail — paired-seed reproduction with N=2-3):**

With 2-3 training seeds there is no population-level statistical test available. Report the data and characterize the outcome honestly:

- **Sign convention:** `δ_s = val_loss(diff_s) - val_loss(vanilla_s)`. Negative = diff-attn wins on that seed.
- **Strong directional replication:** all paired δ_s share sign in the paper's predicted direction (negative), AND each per-seed bootstrap CI (eval-set resampling, §10.2) excludes zero. This is the closest we can get to "diff-attn beats vanilla at Stage 2 / 4B with this corpus."
- **Weak directional replication:** all paired δ_s share sign in the predicted direction, but some per-seed CIs include zero. Report as "directional agreement, eval-set noise comparable to effect size on at least one seed."
- **Mixed result:** paired δ_s flip sign across seeds. Report all of them, do not claim a winner; the result is "no consistent direction at Stage 2 / 4B with this corpus." This is itself a valid finding for a small-scale reproduction.
- **Reversed:** all δ_s positive (vanilla wins). Report as "diff-attn does not replicate; vanilla wins consistently at this scale." Also a valid finding.
- Held-out perplexity paired deltas should directionally agree with val-loss paired deltas across all outcomes.

**Pre-register N before Stage 2 begins, based on Stage 0/1 throughput only:**
- N=2 if Stage 0 calibration (§5.2) or Stage 1 throughput projection (§5.3) shows Stage 2 fits within ~10 days for two seed pairs.
- N=3 if both calibrations show comfortable headroom for three seed pairs within ~10 days.
- Decision recorded in the run's `config.json` and the README writeup before any Stage 2 training starts. **No post-hoc N change** based on Stage 2 outcomes — that is p-hacking by seed addition.
- If, after Stage 2 completes with the pre-registered N, the result is borderline AND compute remains, an additional seed pair may be run as **labeled exploratory data** — reported separately, not folded into the primary paired-delta table, and called out as exploratory in the writeup.

---

## 6. Architecture detail

### 6.1 Shared backbone (identical between both variants)

| | Stage 0 | Stage 1 | Stage 2 |
|---|---|---|---|
| Hidden size (dim) | 256 | 768 | 1024 |
| Layers | 6 | 12 | 16 |
| MLP intermediate (`ceil(8/3 · dim)` to next multiple of 32) | 704 | 2048 | 2752 |
| `n_heads_vanilla` (Variant A) | 4 | 12 | 16 |
| `n_heads_diff` (Variant B; `= n_heads_vanilla / 2`) | 2 | 6 | 8 |
| `qk_head_dim` (== vanilla head_dim) | 64 | 64 | 64 |
| `v_head_dim` (Variant B; `= 2 * qk_head_dim`) | 128 | 128 | 128 |
| Vocab (`cl100k_base`) | 100,277 | 100,277 | 100,277 |
| Block size | 1024 | 2048 | 2048 |
| Positional encoding | RoPE base=10000 | RoPE base=10000 | RoPE base=10000 |
| Normalization | RMSNorm (pre-norm) | RMSNorm | RMSNorm |
| Activation | SwiGLU | SwiGLU | SwiGLU |
| Tied embeddings | Yes | Yes | Yes |
| Linear bias (all projections) | False | False | False |

Param estimates: ~30M / ~162M / ~305M, with ~26M / ~77M / ~103M in embeddings (cl100k_base vocab dominates the small-stage budgets). Diff-attn variant has the **same** parameter count as the vanilla variant at every stage — the H_diff halving + V doubling preserves `4 · dim²` of attention weights.

**Backbone implementation details (shared between both variants):**

| Detail | Value | Notes |
|---|---|---|
| Block topology | Pre-norm: `x = x + attn(rmsnorm(x)); x = x + mlp(rmsnorm(x))` | Standard LLaMA-style |
| Final norm before LM head | RMSNorm | Same form as the per-block pre-norms |
| RMSNorm epsilon | `1e-5` | Standard; applied as `x * scale / sqrt(mean(x²) + eps)` |
| RMSNorm affine | Learned `scale` of shape `(last_dim,)`, init to 1.0; no bias | Per-block pre-norm, post-MLP norm, final norm, and diff-attn `subln` all use the same form, differing only in `last_dim` |
| RoPE implementation | `mx.fast.rope(x, dims=qk_head_dim, traditional=True, base=10000.0, scale=1.0, offset=0)` — native Metal kernel, expects `(B, H, T, D)` layout. NO manual `cos`/`sin` tables in Python. |
| RoPE layout | **Consecutive-pair / GPT-J interleaved** (matches the Microsoft Diff-Transformer reference): pair `(x_{2i}, x_{2i+1})` for `i ∈ [0, D/2)`, apply 2×2 rotation. **Critical:** `traditional=True` in MLX = consecutive-pair rotation (RoFormer / GPT-J style); `traditional=False` is rotate-halves (GPT-NeoX / LLaMA). The paper's reference (`microsoft/unilm/Diff-Transformer/multihead_diffattn.py`, line 122-123) explicitly calls `apply_rotary_emb(..., interleaved=True)`, so we use `traditional=True` to match. NOTE: this DIFFERS from LLaMA / mlx-lm's default convention; we deliberately use the paper-canonical interleaved convention here for fidelity. Both variants (vanilla MHA + diff-attn) use the same convention so the internal A/B is apples-to-apples. |
| RoPE base | 10000 (passed as `base=10000.0` to `mx.fast.rope`) | §6.1 arch table |
| RoPE precision | Handled inside the native kernel; inputs in bf16, internal trig in fp32, output in bf16 | No precision tuning needed in user code |
| LM head bias | None (tied to embeddings, which have no bias) | |
| Position embedding | None — RoPE only | |

### 6.2 Variant A — Vanilla MHA

Standard scaled-dot-product attention via MLX's built-in `mx.fast.scaled_dot_product_attention`. Reference path; no custom kernel.

Per-layer projections (all `bias=False`):
- `q_proj`: `dim → n_heads_vanilla * head_dim == dim`
- `k_proj`: `dim → n_heads_vanilla * head_dim == dim`
- `v_proj`: `dim → n_heads_vanilla * head_dim == dim`
- `o_proj`: `n_heads_vanilla * head_dim → dim`

Standard scaling: `1/sqrt(head_dim)`. RoPE applied to Q and K. Causal mask. No QK norm.

SwiGLU MLP uses `intermediate = ceil(8/3 · dim)` rounded up to the next multiple of 32, with three linears (gate, up, down), all `bias=False`. Worked examples: dim=256 → 704; dim=768 → 2048; dim=1024 → 2752. Same MLP shape in both variants.

### 6.3 Variant B — Differential attention (per the paper)

**Reference:** paper §2.1 (Eq. 1) and `microsoft/unilm/Diff-Transformer/multihead_diffattn.py`. Math below is paper-canonical and verified against the reference repo.

**Naming convention** (used throughout the rest of this doc):
- `n_heads_vanilla` — head count in Variant A (vanilla MHA)
- `n_heads_diff = n_heads_vanilla // 2` — half as many differential heads
- `qk_head_dim = D` — same as vanilla head_dim (per-sub-head Q/K dimension)
- `v_head_dim = 2 * D` — V head dimension is **double** the Q/K sub-head dimension
- `dim = hidden_size`
- Identity that makes everything fit: `n_heads_diff * v_head_dim == n_heads_vanilla * D == dim`

**Per-layer projections** (all `bias=False`, matching the reference repo):
- `q_proj`: `dim → 2 * n_heads_diff * D == dim` (packs Q1, Q2)
- `k_proj`: `dim → 2 * n_heads_diff * D == dim` (packs K1, K2)
- `v_proj`: `dim → n_heads_diff * 2*D == dim` (single V at doubled head width)
- `o_proj`: `n_heads_diff * 2*D → dim` (input width also `dim`)
- `subln`: RMSNorm over `2*D` (the V head width), applied per-head AFTER differential subtraction

**Net param effect vs vanilla MHA:** identical attention parameter count (`4 * dim^2`). The H_diff halving + V doubling preserves the projection matrices' total width.

**FLOPs are NOT identical, contrary to what the paper text claims.** The two QK matmuls in diff-attn (Q1·K1ᵀ + Q2·K2ᵀ) sum to the same FLOPs as vanilla's single QK matmul, because each diff sub-head is at half-width per the n_heads halving. But the two AV matmuls — both with `V` at width `2D` — do **2× the work** of vanilla's single AV matmul:

| | Vanilla MHA | Diff-attn (paper) | Ratio |
|---|---|---|---|
| QK matmul FLOPs per layer | `B · T² · n_heads_vanilla · D` | same (two halves of half-width sum to the same) | 1.0× |
| AV matmul FLOPs per layer | `B · T² · n_heads_vanilla · D` | `2 · B · T² · n_heads_vanilla · D` | 2.0× |
| Total attn FLOPs per layer | `2 · B · T² · n_heads_vanilla · D` | `3 · B · T² · n_heads_vanilla · D` | 1.5× |

In practice the MLP dominates total per-token FLOPs (~80% at our shapes with 8/3·dim MLP), so diff-attn's per-token wall-time overhead is more like ~8-12% at Stage 2, not 50%. But it is not zero. The doc previously claimed "−25% on attention" — that was wrong. Token-efficiency claims (§10.3) measure *tokens*, not FLOPs; convert with this 1.0× / 1.5× / 2.0× lens before claiming compute-efficiency wins.

**No Q/K RMSNorm before QK matmul.** The reference repo has no `q_norm` / `k_norm`; only model pre-RMSNorm (already in the block) and the per-head `subln` after the differential subtraction. If we want to ablate QK norm later, it's an explicit deviation, not the reproduction path.

**Lambda parameters (per layer):**
- `lambda_q1`, `lambda_k1`, `lambda_q2`, `lambda_k2`: each `nn.Parameter` of shape `(D,)`, init `randn * 0.1` (NOT zero — zero init kills step-0 gradient)
- `_lambda_init` (constant, depth-scheduled): paper formula `0.8 - 0.6 * exp(-0.3 * (layer_idx - 1))` (1-indexed). Starts at 0.2 for layer 1 and rises toward 0.8 with depth.

**Lambda computation at every forward call:**
```python
lam = exp(dot(lambda_q1, lambda_k1)) - exp(dot(lambda_q2, lambda_k2)) + _lambda_init
```

**Forward pass** (canonical shapes; `H = n_heads_diff`, `D = qk_head_dim`, `2D = v_head_dim`; input `x: (B, T, dim)`):
```python
# 1. Project (no Q/K RMSNorm — not in paper) and reshape to (B, T, head_axis, head_dim)
q = q_proj(x).reshape(B, T, 2*H, D)                          # (B, T, 2H, D)
k = k_proj(x).reshape(B, T, 2*H, D)                          # (B, T, 2H, D)
v = v_proj(x).reshape(B, T, H,   2*D)                        # (B, T, H,  2D)

# 2. Transpose to canonical attention layout (B, head_axis, T, head_dim)
q = q.transpose(0, 2, 1, 3)                                  # (B, 2H, T, D)
k = k.transpose(0, 2, 1, 3)                                  # (B, 2H, T, D)
v = v.transpose(0, 2, 1, 3)                                  # (B,  H, T, 2D)

# 3. Split Q/K into the two head-pairs. Paper-canonical layout (matches the
#    microsoft/unilm reference): the 2H heads are viewed as (H, 2) in row-major
#    order, so diff-head h pairs Q1 = q[2h], Q2 = q[2h+1]. NOT halves split
#    q[:H] vs q[H:] (those are different layouts and produce different outputs).
#    See R8 in §16 and the reference cross-check test for the discriminator.
q_pair = q.reshape(B, H, 2, T, D)                            # (B, H, 2, T, D)
k_pair = k.reshape(B, H, 2, T, D)                            # (B, H, 2, T, D)
q1, q2 = q_pair[:, :, 0, :, :], q_pair[:, :, 1, :, :]        # each (B, H, T, D)
k1, k2 = k_pair[:, :, 0, :, :], k_pair[:, :, 1, :, :]        # each (B, H, T, D)

# 4. RoPE on Q1, K1, Q2, K2 independently via the native Metal kernel
#    (consecutive-pair / GPT-J interleaved layout; traditional=True; see §6.1 backbone details)
import mlx.core.fast as mxf
q1 = mxf.rope(q1, dims=D, traditional=True, base=10000.0, scale=1.0, offset=0)
q2 = mxf.rope(q2, dims=D, traditional=True, base=10000.0, scale=1.0, offset=0)
k1 = mxf.rope(k1, dims=D, traditional=True, base=10000.0, scale=1.0, offset=0)
k2 = mxf.rope(k2, dims=D, traditional=True, base=10000.0, scale=1.0, offset=0)

# 5. Two causal SDPA calls sharing V (canonical signature; (B, H, T, ·) shapes)
#    v0 path: built-in SDPA. v1 path: our P2 SDPA kernel. Same algebra either way.
scale = 1.0 / sqrt(D)
out1 = mx.fast.scaled_dot_product_attention(q1, k1, v, scale=scale, mask="causal")   # (B, H, T, 2D)
out2 = mx.fast.scaled_dot_product_attention(q2, k2, v, scale=scale, mask="causal")   # (B, H, T, 2D)

# 6. Differential subtraction (lam is scalar; broadcasts across all axes)
lam = compute_lambda(lambda_q1, lambda_k1, lambda_q2, lambda_k2, _lambda_init)
out = out1 - lam * out2                                                              # (B, H, T, 2D)

# 7. Per-head subLN (RMSNorm over last dim = 2D; head axis preserved — see shape discipline note)
out = subln(out)                                                                     # (B, H, T, 2D)

# 8. (1 - lambda_init) scaling, then merge heads and output projection
out = (1 - _lambda_init) * out                                                       # depth-dependent
out = out.transpose(0, 2, 1, 3).reshape(B, T, H * 2*D)                               # (B, T, dim)
out = o_proj(out)                                                                    # (B, T, dim)
```

All matmuls are expressed as SDPA calls or projection linears; no bare `@` over ambiguous (T, H, D) shapes. The transpose to `(B, H, T, D)` before SDPA is the canonical MLX/PyTorch attention layout and matches what `mx.fast.scaled_dot_product_attention` expects.

**subLN shape discipline.** `subln` is an `RMSNorm(2 * qk_head_dim)`. It normalizes the **last dimension only**, on a tensor whose head axis is still present. In code: apply subln BEFORE flattening heads into the residual stream. Concretely, with `out` shape `(B, T, H, 2D)`, call `subln(out)` — this normalizes along the `2D` axis, independently for each head. **Do not** reshape to `(B, T, H * 2D)` first; that would mix variances across heads and is not what the reference implementation does. No affine mixing across heads at any point in the diff-attn block.

### 6.4 The `(1 - _lambda_init)` scaling factor

See §6.3 step 8. The paper applies this scale to keep the diff-attn output norm comparable to standard attention at init. Depth-dependent (varies with layer per the `_lambda_init` schedule). Cross-check against the paper §2.2 when implementing.

### 6.5 Tied embeddings

`out_proj.weight = embed.weight`. Saves ~103M params at Stage 2 (the embed matrix would otherwise be duplicated as the output projection). Standard small-model practice; especially impactful here because the 100k vocab makes the embed expensive.

---

## 7. Custom Metal kernel design

### 7.1 What goes in the kernel

Naming below uses §6.3's convention: `H = n_heads_diff`, `D = qk_head_dim`, V head width is `2D`.

**Key algebraic identity that shapes v1.** Because the differential subtraction is linear in V:
```
(A1 − λ·A2) · V  ==  A1·V − λ·(A2·V)
```
v1 does not need to materialize the two `T×T` attention maps. It can compute each `Aᵢ·V` directly inside a causal SDPA kernel and subtract the two outputs in Python. The kernel never sees the maps; the temporary footprint is `O(B·H·T·2D)` per output, not `O(B·H·T·T)`.

**v1 = two causal SDPA calls + Python subtract.** The unit kernel is the same one preflighted in Stage P2.

Per-SDPA-call inputs to the kernel (all bf16):
- Q: `(B, H, T, D)` — post-RoPE, post-reshape sub-head tensor
- K: `(B, H, T, D)` — same
- V: `(B, H, T, 2D)` — shared between the two calls (paper-doubled head width)
- causal flag: bool

Per-call output (bf16): `(B, H, T, 2D)`.

v1 Python composition (around the kernel):
```python
scale = 1.0 / sqrt(D)
out1 = sdpa_kernel(q1, k1, v, scale=scale, mask="causal")   # (B, H, T, 2D)
out2 = sdpa_kernel(q2, k2, v, scale=scale, mask="causal")   # (B, H, T, 2D), same kernel, same V
out  = out1 - lam * out2                          # (B, H, T, 2D)
out  = subln(out)                                 # per-head RMSNorm over 2D (§6.3 step 7)
out  = (1 - _lambda_init) * out                   # depth scaling
out  = o_proj(out.reshape(B, T, H * 2*D))         # H * 2D == dim
```

**v2 = single fused kernel** (stretch). Fuses what v1 spreads across two kernel calls plus Python subtract into one Metal kernel using FlashAttention-style online softmax. Same algebra; the maps still never materialize. Operations folded in:
- Two `Q·Kᵀ` accumulations with `-inf` causal mask
- Online softmax per row, per attention map (running max + sum)
- Running `out1 − λ·out2` accumulator with V tile-loaded once
- Output `(B, H, T, 2D)` directly

**Operations always handled in Python OUTSIDE any kernel:**
- RoPE on each of Q1, K1, Q2, K2 (separately, before the kernel call)
- subLN RMSNorm over `2D` (after the kernel call, per head — see §6.3 subLN shape discipline note)
- `(1 - _lambda_init)` scaling (after subLN, before o_proj)
- Output projection (regular MLX linear, after the kernel)
- Lambda computation (in Python, passed in as scalar)

### 7.2 Staged kernel versions

**v0 is the science path. v1/v2 are speed/learning artifacts that must clear correctness AND a measured memory/perf gate before training anything.** This separation guarantees the science completes regardless of how far the Metal work gets.

| Version | What | Role | Promoted to training when |
|---|---|---|---|
| **v0** | Pure-MLX reference: `mx.fast.scaled_dot_product_attention(qᵢ, kᵢ, v, scale=1/√D, mask="causal")` called twice with shared V, then `out1 − λ·out2` (no custom kernel) | Always the ground truth; trains Stage 0/1/2 by default | Default. No promotion needed. |
| **v1** | **Two calls to the P2 causal SDPA Metal kernel + Python subtract.** Same algebra as v0, just with our custom SDPA kernel instead of MLX's built-in. No `T×T` map materialization (see §7.1 identity). | Correctness + first-real-kernel artifact; validates the P2 SDPA kernel under real diff-attn shapes | Forward correctness (§7.4) AND measured memory/perf check at Stage 1 shapes |
| **v2** | **Single fused kernel** (stretch). FlashAttention-style block-tiled, online softmax per attention map, with running `out1 − λ·out2` accumulator. Same I/O as the v1 composition pre-subln. | Speed win; paper-worthy artifact | All v1 gates + measured speedup vs v0 (and ideally vs vanilla `mx.fast.scaled_dot_product_attention`) |

**Why v1 is much cheaper than the earlier "materialize maps" design.** Earlier drafts of this doc had v1 producing two `T×T` softmax maps as explicit intermediates, then subtracting them, then multiplying by V — multi-GB transient per layer call at Stage 1/2. The `(A1−λA2)V = A1V − λA2V` rewrite (§7.1) lets v1 keep only the two SDPA outputs `(B, H, T, 2D)` between the kernel calls and the subtract.

**Real memory math for v1.** Each SDPA output is `B · H · T · 2D · 2` bytes (bf16). Two outputs per layer call, layer-transient:

| Stage | B | H_diff | T | 2D | Two-output bf16 per layer call |
|---|---|---|---|---|---|
| Stage 0 | 16 | 2 | 1024 | 128 | ~17 MB |
| Stage 1 | 32 | 6 | 2048 | 128 | ~201 MB |
| Stage 2 | 32 | 8 | 2048 | 128 | ~268 MB |

These are transient per-layer (freed before the next layer's attention call), so they don't multiply by depth, and they're small relative to activations, gradients, optimizer state, and the pure-MLX backward graph. v1 is no longer a memory-pressure risk.

**v1 promotion gate (must clear before training any run with v1):**
1. Forward and finite-diff backward correctness vs v0 on a 4-layer / 64-hidden / 2-diff-head toy (§7.4)
2. **Measured peak memory at a full training step** (forward + backward + optimizer.step), not just per-call forward peak. Custom-function inputs/outputs and saved tensors may be retained by autograd across all layers until backward runs; the relevant number is the full-step residency. Measure at the target stage's shapes (Stage 1 and Stage 2 separately) and verify the run fits within ~50 GB headroom on top of the 20-30 GB v0 baseline before promoting.
3. End-to-end 500-step Stage 0 smoke run with v1, final loss within numerical noise of v0 on the same seed and data order (parallel comparison, not sequential — see §7.4 gate #3)

If any gate fails, training stays on v0. v1 ships as a kernel artifact in the writeup either way.

**v2 (stretch).** Single fused Metal kernel: online softmax per attention map (running max + sum per row), running `out1 − λ·out2` accumulator, V tile-loaded once. Significantly harder; the science does not depend on it.

### 7.3 Backward pass

For v0: MLX autograd handles it natively (it's built from MLX primitives).

For v1/v2: register the forward kernel via `mx.custom_function`. Implement the backward in **pure MLX ops** (express the gradient computation using regular MLX, not a custom Metal kernel). Slower than a fused backward kernel, but the science doesn't care, and the verification is much easier.

A fused Metal backward kernel is a post-project stretch goal.

### 7.4 Correctness checks (gating)

**Reference-implementation cross-check (Stage 0 gate, runs once on the v0 path):**
Internal v0/v1 agreement is necessary but not sufficient — both can share an architecture bug. Before Stage 0 training begins, run a fixed-tensor forward and backward comparison against the official PyTorch `microsoft/unilm/Diff-Transformer/multihead_diffattn.py` on a tiny toy (`B=2, T=16, dim=64, n_heads_diff=2, qk_head_dim=16`). Vendor or pin the reference repo at a known commit. Tolerance: `1e-3` fp32 forward, `1e-2` bf16 forward. If this fails, fix v0 before doing anything else — a divergence here means the MLX port deviates from the paper.

**Per-kernel-version gates (must pass before training):**

1. **Forward:** `‖v1_out − v0_out‖_∞ < 1e-2` on bf16 inputs (loose tolerance for bf16); same shapes as the reference cross-check
2. **Backward:** finite-difference vs analytical gradient on a 4-layer, 64-hidden, 2-head toy model
3. **End-to-end smoke (parallel comparison, NOT sequential):** run **two independent 500-step training runs** from **byte-identical init** (§9.7 paired-seed protocol) on **identical batch order** (§9.4 data-seed determinism): one with v0, one with v1. After 500 steps, the two runs' loss curves should track within numerical noise (≤ ~0.5% relative gap on each logged step). Do not interpret "train 500 with v0 then 500 with v1" as a sequential continuation of the same run — that would compare different checkpoints and tell you nothing about kernel correctness.

### 7.5 Performance targets (nice-to-have, not gating)

- **Algorithmic floor:** diff-attn does two attention maps; a perfect implementation runs at ~2x of single-map vanilla SDPA. Anything close to this floor means zero kernel overhead beyond the unavoidable extra map.
- **v1 target:** within ~2.5x of MLX's `scaled_dot_product_attention` for vanilla MHA at matching `(B, H, T, D)` shapes (i.e. ≤ 25% overhead beyond the algorithmic floor)
- **v2 target:** approaches the algorithmic floor (≤ 2.1x); matches or beats v0 by a clear margin

The science doesn't depend on hitting these. If neither happens, the comparison is *attention-type quality*, not *throughput*. Throughput is a secondary deliverable.

---

## 8. Data pipeline

### 8.1 Corpus

**FineWeb-Edu (English).** Single source. ~6-8B unique tokens needed (max across all runs; shards are reused across variants and seeds). Total disk ~24-32GB at uint32.

Why FineWeb-Edu:
- Clean, well-documented, widely used
- Single source removes data-quality confounds
- Comparable to what other small-scale repro work uses
- Easy for external reproduction

### 8.2 Tokenizer

**tiktoken `cl100k_base`** (paper-canonical, ~100k vocab).

The paper's 3B language modeling experiments use `cl100k_base`. We match it. Install via `pip install tiktoken` and pin the version in `pyproject.toml`. Load via `tiktoken.get_encoding("cl100k_base")`. No vendoring of tokenizer files into the repo — the encoding ships with the package and is content-addressed by tiktoken itself.

Record the resolved vocab size and `tiktoken` version into each run's metadata for reproducibility.

### 8.3 Tokenization to shards

`data/tokenize.py` reads FineWeb-Edu jsonl/parquet from local cache, tokenizes with `cl100k_base`, and writes **uint32** streams to `data/shards/` (vocab > 65535 → uint16 would truncate). Val set is stratified by deterministic hash on document ID so the same val examples land in the val pool across re-runs.

Output:
- `data/shards/train-{NNN}.bin` — uint32 token streams, mmap-able
- `data/shards/val.bin` — fixed 50-100M-token deterministic subset
- `data/shards/meta.json` — vocab_size, eot_id, tiktoken version, train/val token counts, n_docs, source-list with hashes

### 8.4 MLX-side data loading

Simple path (start here): `numpy.memmap` on uint32 shards. Sample random `(block_size + 1)`-windows. Convert to `mx.array` on demand.

Upgrade only if bottlenecked: `mlx-data` library for pipelined input. Won't bottleneck at Stage 2 scale.

### 8.5 Determinism

- Data order seeded with a separate "data seed" (fixed across runs for clean A/B)
- Per-run "model seed" varies model init only
- Val set is byte-deterministic across all runs

---

## 9. Training plan

### 9.0 Precision spec

Pinned across the project so memory estimates, kernel correctness gates, and reproducibility tests are well-defined. Pure-MLX mixed precision, AMP-style with **fp32 master weights** (standard recipe — without it, bf16 params + fp32 AdamW state silently quantize away small updates because each step casts the fp32 update back into bf16 storage).

| What | Dtype |
|---|---|
| **Master parameters** (canonical storage) | **fp32** |
| Forward parameters (cast from master each step) | bf16 |
| Forward activations | bf16 |
| Attention (Q, K, V tensors and SDPA inputs) | bf16 |
| SDPA softmax accumulation | fp32 (handled inside `mx.fast.scaled_dot_product_attention`; custom P2/v1/v2 kernels must do the same) |
| Lambda vectors (`λ_q1`, `λ_k1`, `λ_q2`, `λ_k2`) | fp32 (master + forward both; small; numerical headroom matters because of `exp(...)`) |
| Lambda scalar `lam` (per-layer, per-forward) | fp32, broadcast |
| Logits (pre-CE) | **cast to fp32** before softmax/cross-entropy — bf16 logits over a 100k vocab risk NaN |
| Loss | fp32 |
| Gradients (from autograd) | bf16 for activations; gradient w.r.t. master params accumulated in fp32 |
| AdamW optimizer state (m, v) | fp32 |
| AdamW update | fp32, applied to fp32 master params |
| LR, weight decay constants | fp32 |
| RMSNorm / subLN scale params | fp32 (master + forward both; small, numerically sensitive) |

**Implementation note.** MLX's built-in optimizers may not natively maintain a separate fp32 master copy when params are bf16. Two options: (a) keep params as fp32 in MLX and explicitly `astype(bf16)` for the forward pass each step (simpler; lets MLX's optimizer apply fp32 updates directly); (b) wrap the optimizer step to maintain a parallel fp32 dict and copy bf16 ↔ fp32 each step. Verify MLX's current behavior in the implementation plan; option (a) is the default unless measurement shows it's untenable.

Tolerances stated elsewhere in this doc (`1e-2` for bf16 forward, `1e-3` for fp32 forward in §7.4 reference cross-check) follow this spec. Numerical gates on custom kernels assume fp32 accumulation; v0 and v1 must match this discipline.

**Memory implication.** At Stage 2 (~305M params) with fp32 master + bf16 forward + bf16 grads + fp32 AdamW state: `4 + 2 + 2 + 8 = 16` bytes per param ≈ ~4.9 GB just for optimizer/grad/master/forward-param residency, on top of activations and the transient SDPA outputs (§7.2). Within budget on 128 GB unified.

### 9.1 Optimizer (paper-matched)

- **AdamW:** β1=0.9, β2=0.95, eps=1e-8, weight_decay=0.1
- **Grad clip:** 1.0
- **LR schedule:** linear warmup → cosine decay to 10% of peak
- **Peak LR by stage:**
  - Stage 0: `6e-4`, warmup 500 steps
  - Stage 1: `4e-4`, warmup 1000 steps
  - Stage 2: `3e-4`, warmup 2000 steps

**Weight-decay exclusions** (decoupled AdamW; `weight_decay=0.1` applies only to the *included* set):

| Parameter group | Decay? | Why |
|---|---|---|
| Linear projection weights (q_proj, k_proj, v_proj, o_proj, MLP gate/up/down) | **Yes** | Standard; the main thing decay is for |
| Token embeddings (and the tied LM head, which is the same matrix) | **No** | Common practice; decay on embeddings can hurt small-vocab/large-vocab balance and isn't in the original paper recipe |
| RMSNorm scale params (pre-norm, post-MLP norm, final RMSNorm, subLN) | **No** | Scale params have no canonical "zero" target; decay pulls them away from the trained scale |
| Diff-attn lambda vectors (`λ_q1`, `λ_k1`, `λ_q2`, `λ_k2`) | **No** | These parameterize the lambda scalar via `exp(dot(...))`; decaying them would push them toward zero, biasing `lam` toward `lambda_init` and degrading the reparameterization |
| Biases | n/a | All linears `bias=False` (§6.2, §6.3) |

Implementation: pre-build the parameter-group split when constructing the optimizer; do not rely on name pattern-matching at update time.

### 9.2 Batch and sequence

| | block_size | micro_batch | grad_accum | eff. tokens/step | total steps |
|---|---|---|---|---|---|
| Stage 0 (~30M) | 1024 | 16 | 1 | ~16k | ~6,250 |
| Stage 1 (~162M) | 2048 | 32 | 1 | ~64k | ~31,000 |
| Stage 2 (~305M) | 2048 | 32 | 4 | ~256k | ~15,600 |

These are estimates. **Calibrate after 200 steps of Stage 0** by reading `tokens/sec` from `metrics.jsonl`. If Stage 2 projects past 14 days wall, cut to 3B tokens or drop a seed.

### 9.3 Eval cadence

Eval has two tiers — a cheap monitoring slice for cadence and an expensive full set for sparse milestones. Running the full ~75M-token val every 1000 steps would consume ~2.3B forward-only tokens at Stage 1 (more than the 2B training budget) and ~1.2B at Stage 2. The split keeps eval cost bounded.

**Tier 1 — monitoring slice (frequent, cheap):**
- Stage 0: every 500 steps
- Stages 1-2: every 1000 steps
- Set: fixed **~2M-token** deterministic subset of the val set (a stable prefix; same chunks every run)
- Metric: token-level NLL → perplexity, for training-curve plots and stability monitoring
- Cost: ~30M-60M forward-only tokens per Stage 1 run (1.5-3% overhead vs train budget); negligible

**Tier 2 — full val (sparse, expensive):**
- Triggered at: end-of-warmup, every ~5000 steps thereafter, and end-of-training
- Set: full **50-100M-token** held-out FineWeb-Edu subset (the set used for paired-delta and bootstrap CIs in §10.2)
- Metric: same as tier 1, plus the per-seed paired-delta and bootstrap CI inputs
- Cost: ~5 evals × 75M = 375M forward-only tokens per Stage 1 run (~15-20% overhead); acceptable

**Both tiers byte-deterministic across runs.** Tier 1 monitoring set is a strict prefix of the tier 2 full set, so tier 2 is a superset of the data tier 1 sees; no separate "monitoring" curation work.

**No early stopping.** All Stage 1 and Stage 2 runs train to the planned token budget. Token-efficiency comparisons (§10.3) require fixed-budget runs; early-stop selection would bias which variant gets truncated and break the cross-variant comparison.

### 9.4 Seeds

- Vary model init only; data ordering deterministic across seeds for clean variance estimation
- Stage 0: 1 seed per variant
- Stage 1: 2 seeds per variant
- Stage 2: 2-3 seeds per variant (3 if compute budget allows)

### 9.5 Instrumentation

`metrics.jsonl` per run (one record per logged step):
```json
{"step": 1000, "train_loss": 4.21, "val_loss": 4.35, "tps": 12345, "lr": 3e-4, "grad_norm": 1.02, "time": 1234.5}
```

Run dir saves at start:
- `config.json` — full ModelConfig + TrainConfig dataclass repr
- `git.txt` — git hash + dirty flag
- `tiktoken.txt` — `tiktoken` package version + encoding name (`cl100k_base`) + resolved vocab size
- `data_meta.json` — shard list + hashes
- `mlx_version.txt` — pinned MLX version
- `seed.txt` — model and data seeds

### 9.6 Failure policy

- **NaN/Inf:** abort, log, inspect manually. **No auto-restart.** Kernel bugs need human attention.
- **Manual resume:** from `latest.pt` saved every 1000 steps.
- **Spike auto-restart:** **not** implemented in this project. Bias toward debugging over self-healing for a research run.
- **No early stopping for Stage 1 / Stage 2.** Reproduction runs train to the fixed token budget. The only conditions that abort a run are NaN/Inf or manual intervention.

### 9.7 Paired-seed init protocol (byte-identical shared weights)

The clean A/B requires that for a given paired seed `s`, the vanilla and diff-attn variants have **byte-identical shared weights** (embeddings, MLPs, all RMSNorms in the backbone, the o_proj output projection). Same RNG seed alone is **not** enough: the diff variant has extra parameters (lambda vectors) and a different module-build order, so naive same-seed init consumes RNG differently and produces different shared-weight tensors.

**Protocol (pre-registered: maximize init sharing; copy every weight with matching shape):**

1. With seed `s`, construct the **vanilla** model and serialize its full state-dict (parameter name → tensor).
2. Construct the **diff-attn** model with the same shape config and a *separate* RNG stream for the diff-only parameters (lambda vectors). Init lambda vectors as `randn(D) * 0.1` from that separate stream so vanilla/diff weight tensors don't interact via shared RNG.
3. Copy from vanilla's state-dict into diff's state-dict by **parameter name AND matching shape** for every weight that exists in both. Concretely:
   - **Shared backbone (always copied):** token embeddings, tied LM head, every RMSNorm scale (pre-attn, pre-MLP, final), every MLP weight (gate, up, down).
   - **Attention projections (also copied — pre-registered choice):** `q_proj`, `k_proj`, `v_proj`, `o_proj`. All four are `(dim, dim)` in both variants — vanilla and diff differ only in the *reshape* applied after projection and in the per-pair RoPE/softmax/subtract that follows. The matrices themselves have the same shape; copying maximizes the init-noise cancellation in paired analysis. This is a deliberate choice over leaving them independently initialized; it is pre-registered so the paired Δ measures attention-mechanism difference rather than mechanism-difference + init-noise.
4. Diff-only weights that have no vanilla counterpart use their own init:
   - Lambda vectors (`λ_q1`, `λ_k1`, `λ_q2`, `λ_k2`): from the separate RNG stream, `randn(D) * 0.1`
   - subLN scale (the per-head RMSNorm over 2D after the differential subtraction): init to 1.0
5. Save both state-dicts under `runs/init-seed{s}/{vanilla,diff}.safetensors`. Both runs load from these files; same starting state, same `data_meta.json`, same data-seed → byte-identical batches in identical order → comparable paired output.

**Verification:** a one-off `tests/test_paired_init.py` loads both state-dicts and asserts byte-identity on the full copied set (backbone + all four attention projections). Run this before every paired-seed Stage 1/2 run; cheap and catches RNG drift immediately.

**What the paired Δ controls for and what it doesn't.** With this protocol, `δ_s = val_loss(diff_s) − val_loss(vanilla_s)` controls for: data order, init of every weight matrix that exists in both variants, optimizer/LR schedule, and eval set. It does **not** control for: the diff-only λ-vector init (separate RNG stream), the subLN scale, or the algorithmic difference itself (which is what we're measuring). This is what makes paired-seed delta analysis (§10.2) meaningful.

---

## 10. Evaluation

### 10.1 Primary metric

Token-level NLL on the held-out FineWeb-Edu val set (`exp(NLL)` reported as perplexity for readability).

### 10.2 Variance reporting via paired-seed analysis

Data ordering is byte-deterministic across seeds (see §9.4), so each seed `s` produces a directly comparable pair `(vanilla_s, diff_s)` trained on identical batches in identical order. Use the pairing.

**Sign convention:** `δ_s = val_loss(diff_s) - val_loss(vanilla_s)`. Lower val loss is better, so **negative δ means diff-attn beats vanilla on that seed**. Positive δ means vanilla wins.

**Per-stage report:**
- Per-seed final val-loss and final perplexity, both variants, all seeds.
- **All paired deltas** `δ_s`, individually — not just the mean. Reader can see whether seeds agree in sign.
- **Bootstrap CI on each per-seed paired delta:** resample the held-out val-token chunks (e.g. 1024-token) with replacement, recompute the loss for the resampled chunks under each checkpoint, recompute `δ_s` for that bootstrap sample. Report 95% CI from 1000 bootstrap samples for each seed pair. This CI measures **eval-set noise** for that particular seed pair, not population-level seed variance.
- **Loss curves:** plot both variants for all seeds (no shaded variance bands — too few seeds for those to mean anything). Annotate with the per-seed final loss.

**What the bootstrap CI does and does not say:**
- **Does say:** for this paired seed `s`, how much of the observed `δ_s` is within-eval-set noise — i.e. if we'd sampled a different val subset of similar size, would the sign of `δ_s` likely flip?
- **Does not say:** population-level statistical significance across training seeds. With 2-3 seeds, that estimate is too noisy to support claims.

**Honest claim framing:**
- "For these N seeds and this eval set, paired deltas were δ₁=..., δ₂=..., δ₃=... All agree in sign / sign flipped between s₁ and s₂. The direction matches / does not match the paper's prediction. The per-seed bootstrap CIs are tight / wide relative to the deltas."
- Avoid "significantly better," "p < 0.05," or any phrasing that implies a population-level test. Report what was measured; it's directional replication, not statistical inference.

### 10.3 Per-token learning-curve comparison

The paper's strongest claim is the **token-efficiency** dimension: diff-attn matches vanilla at fewer tokens. To test this:

- Plot val loss vs tokens-trained for each variant (all seeds)
- For each diff-attn seed: identify the token count at which it first reaches the corresponding vanilla seed's final val loss
- Report per-seed ratios. (The paper's headline claim is ~65%, i.e. diff matches vanilla at ~65% of vanilla's tokens. We measure what we measure — do not anchor the report to the paper's number.)
- If diff never reaches vanilla's final loss within budget, report that explicitly with the achieved-loss gap at end-of-budget

**Token-efficiency is not FLOP-efficiency.** Per §6.3, diff-attn does ~1.5× the attention FLOPs of vanilla per step (the AV matmul is doubled because V is at width `2D`). In our setup the MLP dominates per-token compute, so diff-attn's per-token wall cost is roughly +8-12% at Stage 2. If diff reaches vanilla's loss at X% of vanilla's tokens, the **compute-efficiency** ratio is approximately `X% × (1 + 0.10)` ≈ `X% × 1.1`. Report both numbers: tokens-to-target (what the paper claims) and approximate FLOPs-to-target (what an honest efficiency comparison requires).

### 10.4 Optional downstream eval

At Stage 2 scale (~305M), zero-shot downstream tasks are barely above chance. **Skip them by default** to avoid noise drowning the result. If you want one anyway: LAMBADA accuracy. Don't expect much.

### 10.4b Optional: associative-recall slice

If the main paired-delta result comes out ambiguous, an AR-hit slice can sharpen the comparison. The paper's strongest mechanistic claim is improved associative recall via attention-noise cancellation. Build a small synthetic AR-hit set (e.g. 500 prompts of the form `Key₁=Val₁ Key₂=Val₂ ... Keyₙ=? → Valₙ`) over short contexts; report top-1 hit rate per variant per seed. Cheap to build, cheap to score, and a positive AR signal can rescue a noisy val-loss result.

### 10.5 Kernel speed eval (secondary)

For the writeup, report tokens/sec for:
- Vanilla MHA (MLX built-in SDPA)
- Diff-attn v0 (pure MLX)
- Diff-attn v1 (Metal kernel)
- (Stretch) Diff-attn v2 (tiled Metal)

This is a kernel deliverable, not a science deliverable.

---

## 11. File and code structure

```
diff-mlx/
  README.md                     # project overview + final writeup
  pyproject.toml                # mlx + numpy + tiktoken + huggingface_hub
  config.py                     # ModelConfig (attn_variant flag), TrainConfig, paths
  docs/
    2026-05-20-diffattn-mlx-reproduction-design.md   # this file
    (implementation plan, written next by writing-plans)

  model.py                      # Transformer + DiffAttentionLayer + VanillaMHALayer

  kernels/
    diff_attention.py           # Python wrapper, autograd registration, v0/v1/v2 selector
    diff_attention.metal        # MSL source for v1 (and v2 if reached)
    softmax_p1.py               # Stage P1 softmax wrapper + autograd hook
    softmax_p1.metal            # MSL source for Stage P1 softmax kernel
    sdpa_p2.py                  # Stage P2 causal SDPA wrapper + autograd hook
    sdpa_p2.metal               # MSL source for Stage P2 causal SDPA kernel

  data/
    shards/                     # uint32 FineWeb-Edu shards (gitignored)
    tokenize.py                 # FineWeb-Edu → uint32 shards using tiktoken cl100k_base
    loader.py                   # numpy mmap loader, deterministic batch sampler

  train.py                      # MLX training loop, single-GPU
  eval.py                       # val loss + perplexity computation

  tests/
    test_model.py               # Architecture forward/backward
    test_kernel.py              # Diff-attn correctness (v0 vs v1, gradient check)
    test_preflight.py           # Stage P1 softmax + Stage P2 causal SDPA correctness
    test_data.py                # Loader correctness, val determinism

  runs/                         # gitignored
    preflight/
    stage0/
    stage1-{variant}-seed{N}/
    stage2-{variant}-seed{N}/
```

**Design rules for the file layout:**
- Single `model.py` with both attention variants (toggle via config flag) — single source of truth for non-attention layers. **Byte-identical shared weights between variants requires an explicit init protocol** (§9.7), not just a shared module definition; same RNG seed plus different module graph consumes RNG differently and produces different weights.
- Kernel work isolated to `kernels/` — clean boundary; easy to drop v1/v2 if either preflight fails
- Preflight kernels (P1, P2) have their own self-contained files so they can be skipped cleanly
- `runs/` gitignored; per-stage/per-variant/per-seed naming

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Metal kernel never converges to correctness | Stage P1 → P2 → v1 gating; v0 ships the science regardless |
| Training instability (loss spikes, NaN) at Stages 1-2 | Conservative LR; LLaMA-style proven backbone; NaN abort + manual inspect; emergency: cut LR 2x, double warmup |
| Wall-time blowout (M5 Max slower than estimated) | Calibrate at Stage 0 step 200; if Stage 2 projects past 14 days, drop a seed or cut to 3B tokens |
| Signal absent at Stage 1 | Either: write up "doesn't replicate at Stage 1 scale" as a legit finding, or proceed to Stage 2 (paper's strongest claim is scaling-direction, may emerge at Stage 2); decide after looking at the curves |
| tiktoken version churn | Pin `tiktoken` version in `pyproject.toml`; record version + resolved vocab size in every run's metadata |
| MLX API churn during 2-3 week project | Pin MLX version in `pyproject.toml`; do not auto-upgrade mid-project |
| Mac needed for life during long runs | Each stage is independent; pause between stages fine; `latest.pt` resume works; each variant/seed run is ~3-5 days for Stage 2 |
| Memory pressure (v0 training) | 128GB unified is huge; Stage 2 v0 training peaks ~18-24GB (embed larger with cl100k_base); not a real risk |
| Memory pressure (v1 training) | After §7.1 rewrite, v1 = two SDPA outputs (~200-270 MB transient at Stage 1/2), not multi-GB materialized maps. No longer a real risk; the promotion gate keeps measuring it as a sanity check |
| Lambda reparameterization init bug | Use `randn * 0.1` for lambda vectors (NOT zero); zero init makes grads zero at step 0. See §6.3. |
| RoPE per-pair application | RoPE applied to each of Q1, K1, Q2, K2 independently (after projection and reshape, before QK matmul). No Q/K RMSNorm — paper does not include one. See §6.3 forward-pass pseudocode. |
| Forgetting subLN after subtraction | The paper applies an RMSNorm per-head AFTER the differential subtraction and BEFORE the output projection. See §6.3 step 8. |
| Forgetting the `(1 - lambda_init)` output scale | The paper scales the per-head output by `(1 - lambda_init)` before the output projection (depth-dependent). See §6.4 and paper §2.2 |

---

## 13. References

### 13.1 Papers

- **Original:** Ye et al., "Differential Transformer," ICLR 2025 oral. [arXiv 2410.05258](https://arxiv.org/abs/2410.05258). [PDF](https://arxiv.org/pdf/2410.05258). [OpenReview](https://openreview.net/forum?id=OvoCm1gGhN).
- **Mechanism analysis:** "Understanding Differential Transformer Unchains Pretrained Self-Attentions," May 2025. [arXiv 2505.16333](https://arxiv.org/pdf/2505.16333).
- **Extension:** "Grouped Differential Attention," Oct 2025. [arXiv 2510.06949](https://arxiv.org/pdf/2510.06949).

### 13.2 Reference implementations

- **Paper's official repo:** `microsoft/unilm/tree/master/Diff-Transformer` — authoritative for λ reparameterization, head-split mechanics, and `_lambda_init` schedule
- **PyTorch (community):** [`kyegomez/DifferentialTransformer`](https://github.com/kyegomez/DifferentialTransformer) (dormant, no released weights)
- **PyTorch (Axolotl plugin):** [`axolotl-ai-cloud/diff-transformer`](https://github.com/axolotl-ai-cloud/diff-transformer) (dormant, conversion tool, no released weights)

### 13.3 MLX documentation

- [MLX repo](https://github.com/ml-explore/mlx)
- [Custom Metal Kernels docs](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [WWDC 2025: Get started with MLX](https://developer.apple.com/videos/play/wwdc2025/298/)
- [`humanrouter/ddtree-mlx`](https://github.com/humanrouter/ddtree-mlx) — precedent for novel-attention Metal kernels in MLX

### 13.4 Data and tokenizer

- [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [tiktoken](https://github.com/openai/tiktoken) — `cl100k_base` encoding (paper-canonical)

---

## 14. Context for future sessions

### 14.1 Hardware: Mac is also the daily driver

The M5 Max is the implementer's daily-use machine. Training pins the GPU; UI gets sluggish but works. Light tasks (code, terminal, reading, email) are fine during runs. **Do not run heavy local LLM inference during training** (will fight for GPU and tank both).

### 14.2 Writing style for this repo

- No em or en dashes (use commas, periods, parentheses, or rewrite)
- Drop leading "I" / "I'm" at the start of sentences
- No redundant context (don't write "in this project" inside this project's docs)
- Express thoughts in the fewest words possible without losing precision

### 14.3 What NOT to do

- Don't suggest skipping diff-attn or swapping to vanilla as a shortcut. The diff-attn architecture is precisely the point of this project.
- Don't suggest moving this project to CUDA. MLX + Metal kernels are the learning goal; CUDA hardware doesn't run Metal.
- Don't roll a custom tokenizer. Use tiktoken `cl100k_base` as-is.
- Don't auto-upgrade MLX mid-project. Pin the version in `pyproject.toml`.

---

## 15. Notes for the next Claude session (resume cold)

If a future session lands in this directory cold (no conversation history):

1. **Read this file first** end-to-end. It is the authoritative design.
2. **Read the paper** ([arXiv 2410.05258](https://arxiv.org/abs/2410.05258)), especially §2, before touching attention code.
3. **Check `runs/`** for any existing checkpoints; the project state is in checkpoint metadata + `metrics.jsonl`.
4. **Don't auto-upgrade MLX.** Check the pinned version in `pyproject.toml`.
5. **Stages P1 / P2 status** dictate whether v1/v2 kernel work is live. If either preflight failed or was skipped, only v0 (pure-MLX) is in scope.
6. **The next step after this design doc** is the implementation plan, to be written by the `writing-plans` skill into `docs/2026-05-20-diffattn-mlx-implementation-plan.md`.

If the user says "carry on" with no other context: check `runs/` for the most recent run, read its `metrics.jsonl` to see where training is, and resume from `latest.pt` if applicable. If no runs exist, begin with Stage P1.

---

## 16. Sign-off

Design approved by Guy 2026-05-20. Revised same day across two Codex review passes plus a self-review pass.

**Round 1 Codex review** (`docs/2026-05-20-diffattn-mlx-design-review.md`): corrected diff-attn dimensions to paper-canonical (`n_heads_diff = n_heads_vanilla / 2`, `v_head_dim = 2 * qk_head_dim`, total attn params equal to vanilla); removed non-paper Q/K RMSNorm from §6.3; switched tokenizer to tiktoken `cl100k_base` (and uint32 token shards); reframed v1 as correctness artifact with explicit memory/perf gate; added Stage P2 causal SDPA preflight; tightened Stage 1 gate to correctness/stability/throughput with signal as exploratory output; replaced population-stdev gate with paired-seed deltas; removed early stopping from Stage 1/2 fixed-budget runs; added bias-false statement, reference-implementation cross-check as Stage 0 gate, and optional AR-hit eval slice.

**Self-review pass** (`docs/2026-05-20-diffattn-mlx-design-self-review.md`): flagged §7.1 / §9.2 inconsistencies introduced by Round 1, plus sign convention, outcome framing, and minor cleanups.

**Round 2 Codex review** (consolidated with self-review findings): §7.1 kernel I/O updated to V=2D and Q/K RMSNorm removed there too; v1 explicitly framed as multi-kernel pipeline with named intermediate arrays (not a single fused diff-attn kernel); Stage P2 pass criteria specify actual Stage 0/1 vanilla shapes and call out that the V=2D path is exercised in v1 against v0; Stage 0 SDPA oracle now defined explicitly (`sdpa(q1,k1,v) − λ·sdpa(q2,k2,v)` + subln + scale); §9.2 batch-table labels updated to ~30M / ~162M / ~305M; MLP rounding rule specified as `ceil(8/3·dim)` to next multiple of 32; §10.2 bootstrap framed as eval-set noise (per-seed CIs), with honest "directional replication, not statistical inference" framing for N=2-3; §5.4 outcomes made symmetric (vanilla-wins is a valid finding, not a failure); §12 memory risk split v0 (low) vs v1 (gated); v1 perf target reframed against the 2x algorithmic floor; smaller fixes for sign convention (negative δ = diff-attn wins), `attn @ v` head batching, §6.4 redundancy trim, §10.3 ratio framing, GQA scope wording, and Stage 0 "single seed per variant" disambiguation.

**Round 3 Codex review:** v1 reframed using the `(A1−λA2)V = A1V − λA2V` linearity — v1 is now two causal-SDPA kernel calls (the same kernel preflighted in Stage P2) with shared V at 2D, plus a Python subtract. T×T attention maps are never materialized; transient memory drops from 3-4 GB to ~200-270 MB at Stage 1/2, and v1 is naturally de-risked by P2. §7.1 / §7.2 / §12 memory row / §10 v1 perf framing updated accordingly. Stage 0 oracle text corrected (it **is** v0; what it's not is a single SDPA call). Deliverables softened from "fused diff-attn kernel" to "custom Metal kernels for the diff-attn path; v1 = composition over P2 SDPA, v2 = fused (stretch)." Added explicit subLN shape-discipline note (RMSNorm over last dim, no cross-head mixing). Time budget framed as "target 2-3 weeks, likely longer if v1 work extends." Stage 1 gate relaxed (1 paired seed full + 1 pair half is enough). Project framed in §3.2 as small-scale controlled reproduction, not direct paper reproduction.

**Round 4 (Codex + Gemini review):** **Runtime-bug fix** — MLX `mx.fast.scaled_dot_product_attention` does **not** accept `is_causal=True`; `scale` is mandatory keyword-only and `mask="causal"` is the documented causal-mask syntax. All call sites updated. **Paired-seed init protocol** (new §9.7) — shared-weight byte-identity requires explicit copy-by-name across variants; same seed alone is not enough. **Precision spec** (new §9.0) — bf16 params/activations, fp32 SDPA accumulation, fp32 logits and loss (bf16 logits over 100k vocab risks NaN), fp32 AdamW state, fp32 lambda vectors. **FLOP-matching clarification** — diff-attn has the same QK FLOPs as vanilla but doubled AV FLOPs (V at 2D), giving 1.5× total attention FLOPs; ~8-12% per-token overhead at Stage 2 because MLP dominates. Token-efficiency claims now distinguish from compute-efficiency. **v1 smoke gate** disambiguated: two independent runs from identical init on identical batch order, not a sequential v0→v1 continuation. **AdamW weight-decay exclusions** pinned: decay only on linear projection weights; embeddings, all RMSNorms, and lambda vectors excluded. **Stage 1 deliverable** wording reconciled with relaxed gate. **Backbone details** filled in: pre-norm topology, final RMSNorm before LM head, RMSNorm eps=1e-5 with learned scale (no bias), RoPE rotate-halves layout (LLaMA convention, not RoFormer interleaved), cos/sin in fp32. Gemini also flagged a **DIFF V2 paper** (Jan 2026, microsoft/unilm/Diff-Transformer/Diff-Transformer-V2); §1.4 added explaining why we stick with V1 (project identity = custom Metal kernels; V2 uses standard FlashAttention).

**Round 5 (Codex):** **Eval cadence split** (§9.3) — fixed 75M-token eval every 1000 steps would consume ~2.3B forward-only tokens at Stage 1 (more than the 2B training budget); now a ~2M-token monitoring slice for cadence plus full 50-100M eval at sparse milestones (end-of-warmup, every ~5000 steps, end). **Paired-seed init resolved internal conflict** (§9.7) — pre-register copying attention projections (q/k/v/o all `(dim, dim)` matching shape) along with backbone; eliminates the "copy o_proj" vs "o_proj optional" contradiction. **Third Stage 2 seed pre-registered** before Stage 2 begins (§5.4) — N decided on Stage 0/1 throughput only; any post-hoc 3rd seed labeled exploratory and excluded from primary report. **fp32 master weights pinned** (§9.0) — bf16 forward cast from fp32 master; without it bf16 params + fp32 AdamW updates silently quantize away small updates. **v1 memory gate measures full step** (§7.2) — forward+backward+optimizer peak, not per-call forward; custom-function autograd retains saved tensors across layers. **§6.3 pseudocode rewritten** in canonical `(B, H, T, D)` shapes with explicit `mx.fast.scaled_dot_product_attention` calls — no ambiguous bare `@` over `(T, H, D)` shapes that would mislead the implementation.

**Round 6 (Antigravity, then verified):** Switched RoPE from manual `apply_rope(q, cos, sin)` to native `mx.fast.rope` (C++/Metal-optimized). Verified the `traditional` flag direction: Antigravity's suggestion that `traditional=True` matches LLaMA was backwards — `mlx-lm/models/llama.py` uses `rope_traditional=False`, and the MLX docstring confirms `traditional=True` means consecutive-pair rotation (original RoFormer), `traditional=False` means rotate-halves (GPT-NeoX / LLaMA). §6.1 backbone table and §6.3 forward pseudocode were updated to use `traditional=False` based on the "LLaMA-style" framing. Phase A trained Stage 0 vanilla under this convention.

**Round 7 (Phase B implementation finding):** While vendoring `microsoft/unilm/Diff-Transformer/multihead_diffattn.py` for the Task 6 reference fixture, found that the paper's reference explicitly uses `apply_rotary_emb(..., interleaved=True)` (line 122-123) — GPT-J consecutive-pair RoPE, NOT LLaMA rotate-halves. R6 picked `traditional=False` because we reasoned "LLaMA convention is rotate-halves and we're LLaMA-style." That reasoning was internally consistent but didn't actually match the paper's reference. For paper-fidelity reproduction (the project's stated goal in §1), both variants are switched to `traditional=True` so they match the paper. The internal A/B is still apples-to-apples (both variants use the same RoPE), and the reference cross-check test can now compare directly without weight pre-permutation. **Phase A's Stage 0 vanilla checkpoint (trained under `traditional=False`) is obsolete and being re-run.**

**Round 8 (Phase B Task 7 implementation finding):** Running the cross-check test exposed a second paper-fidelity bug: §6.3 pseudocode used `q1, q2 = q[:, :H, :, :], q[:, H:, :, :]` (halves split), but the reference does `attn_weights.view(bsz, num_heads, 2, T, T)` — row-major split into `(H, 2)`, meaning diff-head h pairs Q1 = q[2h], Q2 = q[2h+1] (interleaved/strided split). The two layouts give numerically very different outputs (max |diff| ≈ 0.77 with halves, ~1e-7 with interleaved on CPU stream). The internal v0 SDPA oracle test passed under the halves split because the oracle used the same (buggy) split — a textbook example of why design §7.4 mandates a cross-check against a different codebase. §6.3 pseudocode and `model.DiffAttention.__call__` switched to the interleaved split; the oracle test was updated to match. Additionally noted: MLX's default GPU/Metal fp32 matmul uses reduced precision (~1e-3 per matmul), which prevents direct fp32 comparison against PyTorch on GPU. The cross-check test runs under `mx.stream(mx.cpu)` for IEEE fp32 to validate the algorithm itself; production training/eval on GPU is unaffected because both variants share the same reduced-precision path.

Next step: re-run Stage 0 vanilla under `traditional=True` AND the corrected (interleaved) head-split, then proceed with the rest of Phase B.
