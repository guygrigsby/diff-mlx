# diff-mlx reproduction design review

**Date:** 2026-05-20
**Reviewed document:** `docs/2026-05-20-diffattn-mlx-reproduction-design.md`
**Scope:** Independent technical review focused on diff-attention math, kernel staging, Stage P preflight, stage gates, and missing/thin areas.

## Summary

The design is directionally sound as a staged reproduction plan, but it has several correctness issues that should be fixed before implementation. The largest issue is that the differential attention dimensions in Section 6.3 do not match the paper or the official implementation. The plan currently halves the value/output width, while the paper halves the number of heads and keeps the concatenated attention output at `d_model`.

The second major issue is that Section 6.3 adds Q/K RMSNorm before RoPE. That is not part of the Differential Transformer paper or the official implementation. The paper uses pre-RMSNorm at the transformer block level and per-head RMSNorm/GroupNorm after differential attention.

The kernel plan is reasonable as a learning path, but the v1 design is underspecified and likely too memory-heavy to use as the Stage 1 training path without another intermediate gate.

## Findings

### 1. Section 6.3 value/output dimensions are wrong

The design says:

- `v_proj`: `dim -> H*D`
- `o_proj`: `H*D -> dim`
- one set of V heads, no doubling
- “HALF input width vs equivalent vanilla”

That does not match the paper. In the paper, each differential head has:

- `Q1, Q2, K1, K2 in R^{N x d}`
- `V in R^{N x 2d}`

The official implementation follows this shape:

- `head_dim = embed_dim // num_heads // 2`
- `q_proj(embed_dim, embed_dim)`
- `k_proj(embed_dim, embed_dim // n_rep)`
- `v_proj(embed_dim, embed_dim // n_rep)`
- `v.view(..., num_kv_heads, 2 * head_dim)`
- `out_proj(embed_dim, embed_dim)`

For full MHA with no GQA, to compare against a vanilla transformer with 16 heads of dim 64 at hidden size 1024, the differential model should use 8 differential heads. Each differential head has Q/K sub-head dimension 64 and V/output dimension 128, so concatenating 8 heads gives 1024 channels.

Consequence: the current doc defines a smaller attention block than the paper. It changes parameter count, FLOPs, residual-stream width into the output projection, and the meaning of the stage architecture table.

Recommended fix:

- Define `H_diff = n_heads // 2` when matching a vanilla model with `n_heads`.
- Keep `D = qk_head_dim`.
- Set `q_proj: dim -> 2 * H_diff * D`.
- Set `k_proj: dim -> 2 * H_diff * D` for MHA.
- Set `v_proj: dim -> H_diff * 2 * D`.
- Set `o_proj: H_diff * 2 * D -> dim`.
- Set `subln` over `2 * D`, not `D`.

### 2. Section 6.3 adds Q/K RMSNorm that is not paper-canonical

The design includes:

- `q_norm`: RMSNorm over head_dim
- `k_norm`: RMSNorm over head_dim
- RMSNorm applied to Q and K before RoPE

The Differential Transformer paper does not include Q/K normalization in the differential attention definition. The official implementation applies RoPE directly after Q/K projection and reshape. The normalization called out by the paper is the per-head normalization after differential attention:

```text
headi = (1 - lambda_init) * LN(headi)
```

where `LN` uses RMSNorm for each head.

Consequence: Q/K RMSNorm materially changes attention-logit statistics and makes the result no longer a clean reproduction.

Recommended fix:

- Remove `q_norm` and `k_norm` from the paper-matched variant.
- Keep only model pre-RMSNorm and per-head `subln`.
- If QK norm is interesting, make it an explicit ablation after the main reproduction.

### 3. Section 8.2 incorrectly calls LLaMA-2 tokenizer paper-matched

The design says LLaMA-2 SentencePiece is paper-matched. The ICLR paper reports using `tiktoken-cl100k_base` in its 3B language modeling setup.

Consequence: this is a documentation accuracy issue and a possible reproduction confound.

Recommended fix:

- Either switch to `tiktoken-cl100k_base`, or keep LLaMA-2 and describe it as a deliberate practical choice rather than paper-matched.
- If keeping LLaMA-2, note that tokenizer mismatch weakens claims of direct reproduction.

### 4. Section 7 v1 kernel staging is underspecified and likely too memory-heavy

The v1 plan says it materializes two `T x T` softmax maps in global memory, “8MB each at T=2048.”

That number is only per map, per batch item, per head, in bf16:

```text
2048 * 2048 * 2 bytes = 8MB
```

The real temporary footprint scales as:

```text
B * H * 2 maps * T * T * bytes
```

At Stage 1/2 batch and head counts, this becomes multiple GB of temporary attention maps per layer call. If stored in fp32 for stable softmax, it doubles again.

The v1 description also treats “global memory” like hidden scratch. MLX custom Metal kernels return declared output arrays. A two-pass design needs explicit intermediate arrays exposed at the Python level or multiple kernel calls with declared outputs.

Consequence: v1 may be unsuitable as the Stage 1 training path even if it is correct on small tensors.

Recommended fix:

- Treat v1 as a correctness artifact first, not a training dependency.
- Add a measured memory gate before using v1 in real training.
- Consider a more incremental kernel ladder:
  1. custom row softmax preflight
  2. custom causal single-map SDPA forward
  3. custom two-map diff-attn forward on small shapes
  4. memory/perf check at Stage 0 shapes
  5. only then consider Stage 1 use

### 5. Stage P softmax is useful but insufficient

Softmax preflight is a good first move because it tests reductions, numerical stability, MLX custom kernel wiring, and custom VJP mechanics.

It does not de-risk the hardest parts of the diff-attn kernel:

- tiled QK matmul
- causal masking
- row-wise softmax over long `T`
- AV accumulation
- `(B, H, T, D)` indexing and strides
- intermediate memory pressure
- bf16 input with fp32 accumulation

Recommended fix:

- Keep Stage P.
- Add Stage P2: causal SDPA forward for one attention map, compared against MLX SDPA.
- Require Stage P2 before committing to v1/v2 as training paths.

### 6. Stage 1 gate is too strict and may reject a valid result

The Stage 1 pass criterion requires a consistent mid-training validation-loss advantage for diff-attn.

That is too strict for a 100M pilot. The paper's strongest claim is token efficiency and scaling behavior, not necessarily mid-training dominance at this scale. A 100M run with two seeds can easily be noisy or delayed in when the advantage appears.

Recommended fix:

- Gate Stage 1 on correctness, stability, throughput feasibility, and whether curves justify Stage 2.
- Do not require mid-training sign as a hard pass/fail condition.
- Use paired seed deltas and confidence intervals rather than a qualitative “no seed flip” gate.

### 7. Stage 2 success criterion is statistically too lax with 2 seeds

The Stage 2 success criterion says:

```text
Diff-attn final val loss <= vanilla final val loss, with separation >= 1 seed-stdev
```

With only two seeds, seed standard deviation is a weak estimate. A one-standard-deviation gap is not strong evidence, especially without paired analysis.

Recommended fix:

- Use paired seeds: compare diff-attn seed `s` against vanilla seed `s`.
- Report per-seed deltas.
- Add bootstrap confidence intervals over validation documents or batches.
- Avoid a binary “success” claim unless paired deltas agree and the mean gap exceeds eval noise.

### 8. Early stopping should not be used for fixed-token reproduction runs

The design includes early stopping after five evals without validation improvement.

This conflicts with the token-budgeted reproduction goal. Early stopping can bias the comparison and make token-efficiency curves harder to interpret.

Recommended fix:

- Disable early stopping for Stage 1 and Stage 2 reproduction runs.
- Keep NaN/Inf aborts.
- Keep manual resume.
- Train each run to the planned token budget unless the run is invalid.

## Missing or thin areas

### Diff head naming

The doc should distinguish:

- `n_heads_vanilla`
- `n_heads_diff`
- `qk_head_dim`
- `v_head_dim = 2 * qk_head_dim`

This would prevent the Section 6.3 dimension bug from reappearing.

### Bias settings

The official implementation uses bias-free linear projections. The design should explicitly state whether all attention and MLP projections are bias-free.

### MLP ratio

The design uses MLP hidden size `4 * hidden`. The paper's augmented Transformer setup uses SwiGLU and a hidden dimension of `8/3 * d_model`. If the project intentionally uses `4 * hidden`, call that out as a deviation.

### Reference tests

The correctness plan should include a tiny fixed-tensor comparison against the official PyTorch implementation, not only internal v0/v1 agreement. Internal agreement can preserve a shared architecture bug.

### Eval slices

The paper reports fine-grained associative-recall style slices. Full replication is not necessary, but adding a small optional AR-hit/Others slice would make the writeup stronger if the main validation-loss result is ambiguous.

## Bottom line

Fix Section 6.3 before implementation. The current diff-attention block is not paper-canonical because it halves V/output width and adds Q/K RMSNorm. After that, keep Stage P but add a causal SDPA preflight before treating custom diff-attn kernels as viable training paths. Relax the Stage 1 signal gate, strengthen the Stage 2 statistical gate, and remove early stopping from fixed-token reproduction runs.
