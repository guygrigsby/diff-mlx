# diff-mlx reproduction design self-review

**Date:** 2026-05-20
**Reviewed document:** `docs/2026-05-20-diffattn-mlx-reproduction-design.md` (revised after Codex review)
**Reviewer:** Claude (independent pass, not the original author)
**Scope:** Catch internal inconsistencies introduced by the revisions; verify the dimension/forward-pass claims against the paper and the official reference implementation; flag weak spots for the next reviewer.

## Summary

Most of Codex's structural critique landed correctly in the revised doc: dimensions in §6.3 are now paper-canonical, Q/K RMSNorm is gone, tokenizer is `cl100k_base`, Stage P2 is added, statistics are paired-seed + bootstrap CI, early stopping is removed for fixed-budget runs. The bigger issues now are **internal consistency between sections that changed and sections that didn't** — most notably §7.1, which still describes the v1 kernel boundary using the old (incorrect) dimensions. There are also a handful of medium and minor cleanups.

## Critical

### C1. §7.1 still uses the old (pre-revision) dimensions

The kernel I/O block in §7.1 says:

- `Q1, K1, Q2, K2: each (B, H, T, D) where D = head_dim`
- `V: (B, H, T, D)`
- **Output (bf16):** `(B, H, T, D)`

But §6.3 now defines `v_head_dim = 2 * qk_head_dim`, so V and the output should be `(B, H_diff, T, 2D)`, not `(B, H, T, D)`. The Q1/K1/Q2/K2 lines are fine if you read `D` as `qk_head_dim`, but the H is `H_diff` (= `n_heads_vanilla / 2`), which the section does not say.

Fix:
- `Q1, K1, Q2, K2: each (B, H_diff, T, qk_head_dim)`
- `V: (B, H_diff, T, v_head_dim)` where `v_head_dim = 2 * qk_head_dim`
- `Output: (B, H_diff, T, v_head_dim)`
- Update the "Operations fused inside the kernel" list: the final matmul `attn @ V` produces `(B, H_diff, T, 2D)`, not `(B, H, T, D)`

### C2. §7.1 still lists Q/K RMSNorm as an outside-kernel step

> Operations handled in Python OUTSIDE the kernel:
> - RoPE on each of Q1, K1, Q2, K2 (separately, before the kernel call)
> - **QK RMSNorm (before the kernel call)**

Q/K RMSNorm was removed from §6.3 because the paper and reference repo don't have it. This bullet is now wrong. Delete it.

## Medium

### M1. §9.2 batch table still tags stages with stale param-count labels

```
| Stage 0 (10M)  | 1024 | 16 | 1 | ~16k | ~6,250  |
| Stage 1 (100M) | 2048 | 32 | 1 | ~64k | ~31,000 |
| Stage 2 (200M) | 2048 | 32 | 4 | ~256k| ~15,600 |
```

The "(10M)", "(100M)", "(200M)" labels are the pre-revision param estimates. After switching to `cl100k_base`, the actual sizes are ~30M / ~162M / ~305M (per §6.1). Either update the parentheticals or drop them and let the stage names stand alone.

### M2. §9.4 "variance estimation" wording inconsistent with §10.2 paired-delta approach

§9.4:
> Vary model init only; data ordering deterministic across seeds **for clean variance estimation**

§10.2:
> Population-stdev across 2-3 seeds. With this many seeds the estimate is too noisy to support claims, and the paired delta is more powerful anyway.

The §9.4 phrasing implies stdev-style variance analysis; §10.2 explicitly disowns that approach. Reword §9.4 as "for clean paired-seed comparison" or "for paired-delta analysis."

### M3. §3.1 Stage 0 seed wording is ambiguous

> Stage 0: ~100M tokens, **single seed** (smoke test)

§5.2 and §9.4 say "single seed per variant" (so 2 runs total, one per attention type). §3.1 should say "single seed per variant" to match — otherwise a reader can interpret it as one run shared across variants, which is wrong.

### M4. §10.2 paired-delta sign convention not stated

`δ_s = val_loss(diff_s) - val_loss(vanilla_s)`

Lower val loss is better, so diff-attn winning means `δ_s < 0`. §5.4 says "all have the same sign in favor of diff-attn" without defining which sign that is. Add a one-liner: "negative δ means diff-attn beats vanilla on that seed."

### M5. §5.4 pass criterion is one-sided

> Paired-seed deltas (...) **all have the same sign in favor of diff-attn**

This frames the pass criterion as a one-sided test. With a paper-strong prior that's defensible, but a clean reproduction should also be willing to declare "vanilla wins" if it does. The §5.4 failure modes implicitly cover this (sign-flip → no winner) but the framing of the *success* criterion still assumes the diff direction is the "right" one.

Suggestion: rewrite as "all paired deltas agree in sign" + "the consistent direction matches the paper's claim" as separate predicates, so a vanilla-wins outcome reads as a paper-non-replication finding rather than a "failure."

## Minor

### m1. §5.1b and §10.4b numbering

The "b" suffix is fine as a transitional fix but reads awkwardly. Renumber Stage P → P1, "5.1b" → "5.2", and shift §5.2–§5.4 down by one. Similarly §10.4b → §10.5 with subsequent shift. Cosmetic but it's a research doc that may get published; cosmetic things matter for that.

### m2. §7.5 "within 2x of MLX SDPA" is a tautology

> v1: within 2x of MLX's `scaled_dot_product_attention` for vanilla MHA at same params

Diff-attn does **two** attention maps. Single-map SDPA is the algorithmic floor; 2x of single-map SDPA is what you get with zero kernel overhead. Either rephrase as "within 2x is the algorithmic floor (two maps); kernel overhead beyond that is the real perf cost" or set a tighter target (e.g., "within 2.5x").

### m3. §6.3 forward pass `attn @ v` is loose about head batching

```python
attn = a1 - lam * a2          # (T, H, T)
out = attn @ v                # (T, H, 2D)
```

The `@` here is a batched matmul over H. Pseudocode is fine, but a real implementation needs explicit `einsum`-or-equivalent. Worth a one-line clarifier ("batched over H").

### m4. §6.4 redundancy with §6.3

§6.3 already states the `(1 - lambda_init)` scaling in the forward-pass pseudocode. §6.4 restates it. Either delete §6.4 or shorten it to a single "see §6.3 step 8" pointer.

### m5. §10.3 example token-efficiency ratio coincidentally matches paper

> "diff-attn reached vanilla's 4B-token loss at 2.6B tokens"

2.6/4.0 = 65%, which is exactly the paper's headline claim. A casual reader will assume this is a target, not an example. Add "(illustrative ratio; actual value will be measured)" or change the example number.

### m6. §3.2 GQA framing

> Grouped-query attention (GQA) — explicitly use full MHA

Variant B (diff-attn) isn't quite "full MHA" in the traditional sense — it has H_diff heads with V dim 2D rather than H heads with V dim D. The bullet is fine if it means "no GQA in either variant," but a reader unfamiliar with diff-attn could misread. Add a parenthetical: "Variant A uses full MHA; Variant B uses the paper's diff-attn head configuration described in §6.3."

## Verification of Codex's earlier findings

Cross-checked the revised §6.3 against the paper §2.1 (Eq. 1) and `microsoft/unilm/Diff-Transformer/multihead_diffattn.py`:

- **Head halving and V doubling:** Correct. `n_heads_diff = n_heads_vanilla / 2`, `v_head_dim = 2 * qk_head_dim`, all four projections width `dim → dim`, attention param count `4 · dim²` equal to vanilla. Matches both paper and reference repo.
- **Lambda parameters:** Shape `(qk_head_dim,)`, one set per layer (not per head). The reference repo's `multihead_diffattn.py` uses `nn.Parameter(torch.zeros(head_dim).normal_(0, 0.1))` — matches our `randn * 0.1` init. Confirmed.
- **Lambda formula:** `exp(dot(λ_q1, λ_k1)) - exp(dot(λ_q2, λ_k2)) + lambda_init`. Matches reference's `lambda_1 = exp(sum(λ_q1 * λ_k1)); lambda_2 = exp(sum(λ_q2 * λ_k2)); lambda_full = lambda_1 - lambda_2 + lambda_init`. Confirmed.
- **subln scope:** RMSNorm over `2 * qk_head_dim` (the V head width), applied per-head after the differential subtraction. Reference repo: `RMSNorm(2 * head_dim)`. Confirmed.
- **No Q/K norm:** Confirmed against reference; only model pre-RMSNorm and post-subtraction subln.
- **lambda_init schedule:** `0.8 - 0.6 * exp(-0.3 * (layer_idx - 1))`. Layer 1 → 0.2; layer 16 → ~0.793; monotonic. Matches paper.

## Things to verify before the next review pass

These are not findings against the doc; they're items the doc should be checked against before a Codex/LLM second pass.

- **Optimizer hyperparams (§9.1):** β1=0.9, β2=0.95, weight_decay=0.1. The doc claims paper-matched. Confirm against the paper's experimental-setup section.
- **MLP intermediate rounding (§6.1):** 704 / 2048 / 2752. The 2752 value (Stage 2) is `round(8/3 · 1024)` to a multiple of 32 but skews slightly high; the reference repo's rounding rule isn't restated. Pin a rule.
- **LR schedule peak values:** 6e-4 / 4e-4 / 3e-4 across stages. These look reasonable for the scale but the doc doesn't justify them. If they're paper-matched, cite §3; if they're our choice, say so.
- **Wall-time estimates:** §5.3 says ~3-4 days for Stage 1, §5.4 says ~9-10 days for Stage 2. These were calibrated against the original (pre-revision) param counts. The new arch has identical transformer-body FLOPs (8/3 MLP either way) but a 3x larger embed; the embed only affects the loss compute, not the bulk forward/backward. Estimates probably still hold but verify with a back-of-envelope check using current numbers.

## Bottom line

Fix the two §7.1 dimension/Q-K-norm bugs (C1, C2) before sending the doc to the next reviewer — they will catch them and waste review budget pointing them out. The medium items (M1–M5) are cleanup that should be done in the same pass. The minor items and the verification list can carry through; flag them in the cover note when you route the doc.
