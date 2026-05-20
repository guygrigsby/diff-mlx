# diff-mlx — Design document

**Status:** Design approved 2026-05-20. Ready for implementation planning.
**Project home:** `/Users/guygrigsby/projects/diff-mlx/`
**Owner:** Guy J. Grigsby (`guy@grigsby.dev`)

---

## 0. TL;DR for a cold-starting session

`diff-mlx` is a small-scale ML research project that reproduces the **Differential Transformer** paper (Ye et al., ICLR 2025, [arXiv 2410.05258](https://arxiv.org/abs/2410.05258)) **in MLX** on Apple Silicon, with a **custom Metal kernel** for the diff-attention forward pass. The deliverables are: trained checkpoints (vanilla MHA vs diff-attn, at 200M, ~4B tokens each, multi-seed), a working Metal kernel for fused diff-attn, a comparison writeup, and a small reusable MLX package.

**Hardware:** M5 Max, 40 GPU cores, 128GB unified RAM. Pure MLX. Single GPU. The Metal kernel is Apple-only.

**Time budget:** ~2-3 weeks wall time, including a kernel pre-flight stage.

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

- First MLX port of differential attention
- First custom Metal kernel for diff-attn
- First clean small-scale (200M) A/B reproduction in any framework with released weights

---

## 2. Hypothesis under test

> At 200M parameters and ~4B training tokens, in pure MLX with a single English corpus (FineWeb-Edu) and a paper-matched tokenizer (LLaMA-2), **differential attention matches or beats vanilla multi-head attention** in token-level validation loss, replicating the paper's central claim at small scale.

Secondary hypotheses (worth recording even if Stage 1 weakens them):

- The paper-claimed efficiency (matching vanilla at ~65% tokens) is visible at this scale.
- The signal is detectable above seed-variance with 2-3 seeds per variant.

A **negative result** (diff-attn does not replicate at 200M) is itself a publishable finding.

---

## 3. Scope and deliverables

### 3.1 Deliverables at the end

1. **Trained checkpoints** at three scales (10M / 100M / 200M), each with both attention variants:
   - 10M / 100M tokens (Stage 0 smoke test)
   - 100M / 2B tokens × 2 seeds per variant (Stage 1 signal pilot)
   - 200M / 4B tokens × 2-3 seeds per variant (Stage 2 full reproduction)
2. **Custom Metal kernel** for fused diff-attention forward (v0/v1/v2 staged; see §7).
3. **Loss curves and variance bars** at 100M and 200M; held-out perplexity table at all three scales (gives a 3-point mini scaling curve as a bonus).
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
- Grouped-query attention (GQA) — explicitly use full MHA
- LoRA / QLoRA
- Quantization-aware training
- Inference optimization

All intentionally stripped to keep the diff-attn vs vanilla A/B clean: only attention type varies between variants.

---

## 4. Hardware and software stack

| | |
|---|---|
| Machine | M5 Max, 40 GPU cores, 128GB unified RAM |
| OS | macOS (latest) |
| Framework | MLX (pin version in `pyproject.toml`; do not auto-upgrade mid-project) |
| Python | 3.12+ |
| GPU API | Metal (via MLX), Metal Shading Language (MSL) for custom kernels |
| Tokenizer | LLaMA-2 32k SentencePiece (vendored from a public HF mirror, e.g. `NousResearch/Llama-2-7b-hf`) |
| Corpus | FineWeb-Edu (English, ~6-8B tokens used) |
| Storage | uint16 token shards on local disk (~12-16GB) |

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

### 5.1 Stage P — Kernel pre-flight (1-3 days)

**Goal:** verify the MLX custom-kernel layer (write MSL, wire through `mx.fast.metal_kernel`, register autograd hook, gradient-check, run on M5 Max) is tractable for the implementer before committing to diff-attn kernel complexity.

**Target kernel:** Softmax along the last dim (with the max-subtraction stability trick).

**Why softmax for pre-flight:**
- Two reductions (max, then sum), and one element-wise pass — close miniature of what the diff-attn kernel needs
- Has a known, well-tested reference (`mx.softmax(x, axis=-1)`)
- Backward is non-trivial (softmax Jacobian) — good autograd practice
- Fallback if softmax stalls: RMSNorm (simpler, one reduction)

**Pass criteria:**
1. Forward output matches `mx.softmax(x, axis=-1)` within `1e-4` (fp32) or `1e-2` (bf16)
2. Backward via `mx.custom_function` matches autograd of pure-MLX softmax
3. Finite-difference gradient check passes on a small tensor (e.g. shape `(2, 4, 8)`)
4. The implementer feels the API and debugging loop is workable, not miserable

**Time box:** 1-3 days. If not over the hump by day 3, **declare the v1/v2 kernel goal too tall**, ship v0-only at the end of the project, and rebudget Stages 0-2 to include more analysis or seeds.

**Gate behavior:**
- **Pass** → proceed to Stage 0 with confidence v1 (diff-attn kernel) is reachable
- **Fail / stall** → drop v1/v2, keep v0 (pure-MLX diff-attn) as the only kernel path

### 5.2 Stage 0 — Engineering smoke test (~1 day compute)

**Goal:** kernel forward correctness, gradient flow, training stability, eval scripts work, infra end-to-end.

**Config:**
- ~10M params (hidden=256, layers=6, n_heads=4, head_dim=64)
- ~100M tokens
- Single seed per variant
- block_size=1024 (smaller than later stages to speed iteration)
- batch_size=16

**Pass criteria:**
- Both variants (vanilla, diff-attn-v0) train without crashing
- v0 (pure-MLX diff-attn) matches `mx.fast.scaled_dot_product_attention`-style reference within numerical noise on a forward test
- Gradient check on the diff-attn layer (finite-diff vs analytical) passes
- Training loss descends smoothly; no NaNs in ~3,000 steps

**Calibration deliverable:** after step 200, read `tokens/sec` from `metrics.jsonl` and compute projected wall time for Stage 1 and Stage 2. If projection blows past 14 days for Stage 2, cut Stage 2 to 3B tokens or drop a seed.

### 5.3 Stage 1 — Signal pilot (~3-4 days compute)

**Goal:** is the diff-attn vs vanilla signal detectable at this scale?

**Config:**
- 100M params (hidden=768, layers=12, n_heads=12, head_dim=64)
- ~2B tokens (Chinchilla-ish)
- 2 seeds per variant
- block_size=2048
- batch_size=32, effective 64k tokens/step (no grad accum)
- ~31,000 steps per run

**Pass criteria:**
- Diff-attn shows a **consistent (not seed-flip) val-loss advantage** at mid-training, in the direction the paper claims. Magnitude doesn't have to be large; sign and consistency matter.

**Failure paths:**
- **Signal absent, kernel correct:** write up "diff-attn does not replicate at 100M, here's the data" as a valid finding. Optionally continue to Stage 2 to see if the signal emerges with scale (the paper's strongest claim is the scaling-direction).
- **Signal absent, kernel suspect:** rerun gradient and forward correctness checks. If kernel is verified correct, accept the negative result.

### 5.4 Stage 2 — Full reproduction (~9-10 days compute)

**Goal:** paper-style A/B with variance bars at 200M scale.

**Config:**
- 200M params (hidden=1024, layers=16, n_heads=16, head_dim=64)
- ~4B tokens
- 2-3 seeds per variant (3 if compute budget allows after Stage 0 calibration)
- block_size=2048
- micro-batch 32, grad accum 4, effective batch 256k tokens/step
- ~15,600 steps per run

**Pass criteria (success):**
- Diff-attn final val loss ≤ vanilla final val loss, with separation ≥ 1 seed-stdev
- Consistent across seeds (no seed-flips on the winner)
- Held-out perplexity gap directionally agrees with val loss gap

**Failure modes:**
- **No separation:** report variance-bound result, write up as "no significant difference at 200M / 4B with this corpus."
- **Variance too large:** add a seed if compute remains; otherwise report wide error bars honestly.

---

## 6. Architecture detail

### 6.1 Shared backbone (identical between both variants)

| | Stage 0 (10M) | Stage 1 (100M) | Stage 2 (200M) |
|---|---|---|---|
| Hidden size | 256 | 768 | 1024 |
| Layers | 6 | 12 | 16 |
| Intermediate (MLP) | 1024 | 3072 | 4096 |
| n_heads | 4 | 12 | 16 |
| Head dim | 64 | 64 | 64 |
| Vocab | 32,000 (LLaMA-2) | 32,000 | 32,000 |
| Block size | 1024 | 2048 | 2048 |
| Positional encoding | RoPE base=10000 | RoPE base=10000 | RoPE base=10000 |
| Normalization | RMSNorm (pre-norm) | RMSNorm | RMSNorm |
| Activation | SwiGLU | SwiGLU | SwiGLU |
| Tied embeddings | Yes | Yes | Yes |

Param estimates: ~10M / ~107M / ~232M including ~8M / ~25M / ~33M of embeddings.

### 6.2 Variant A — Vanilla MHA

Standard scaled-dot-product attention via MLX's built-in `mx.fast.scaled_dot_product_attention`. Reference path; no custom kernel.

Per-layer projections:
- `q_proj`: `hidden → n_heads * head_dim`
- `k_proj`: `hidden → n_heads * head_dim`
- `v_proj`: `hidden → n_heads * head_dim`
- `o_proj`: `n_heads * head_dim → hidden`

Standard scaling: `1/sqrt(head_dim)`. RoPE applied to Q and K. Causal mask. No QK norm in vanilla variant.

### 6.3 Variant B — Differential attention (per the paper)

**Reference:** the paper's Section 2 and the public reference repository at `microsoft/unilm/tree/master/Diff-Transformer`. The math below is paper-canonical.

**Per-layer projections** (let `H = n_diff_heads`, `D = head_dim`, `dim = hidden_size`):
- `q_proj`: `dim → 2*H*D` (packs Q1, Q2 together; total width same as 2× vanilla Q)
- `k_proj`: `dim → 2*H*D` (packs K1, K2 together)
- `v_proj`: `dim → H*D` (HALF — one set of V heads, no doubling)
- `o_proj`: `H*D → dim` (HALF input width vs equivalent vanilla)
- `q_norm`: RMSNorm over head_dim, applied to each sub-head's Q
- `k_norm`: RMSNorm over head_dim, applied to each sub-head's K
- `subln`: RMSNorm over head_dim, applied to the per-head output AFTER differential subtraction

**Net param effect vs vanilla MHA:** −25% on attention (V and o_proj input both halved). The paper-canonical setup. We do **not** widen the diff-attn variant to match vanilla param count — the comparison is "diff-attn at fewer params/FLOPs vs vanilla at more," which is the paper's claim.

**Lambda parameters (per layer):**
- `lambda_q1`, `lambda_k1`, `lambda_q2`, `lambda_k2`: each `nn.Parameter` of shape `(head_dim,)`, init `randn * 0.1` (NOT zero — zero init makes grads zero at step 0)
- `_lambda_init` (constant, depth-scheduled): use the **paper formula** `0.8 - 0.6 * exp(-0.3 * (layer_idx - 1))` (1-indexed). Starts at 0.2 for layer 1 and rises toward 0.8 with depth (exponential, increasing). Verify against paper §2.2 when implementing.

**Lambda computation at every forward call:**
```python
lam = exp(dot(lambda_q1, lambda_k1)) - exp(dot(lambda_q2, lambda_k2)) + _lambda_init
```

**Forward pass** (for a single head, ignoring batch and head dims for clarity):
```python
# 1. Project and split
q = q_proj(x).reshape(T, 2*H, D)              # 2H sub-heads, each D wide
k = k_proj(x).reshape(T, 2*H, D)
v = v_proj(x).reshape(T, H, D)                 # H heads of V, not 2H
# 2. RMSNorm per sub-head
q = q_norm(q)
k = k_norm(k)
# 3. Split into pairs
q1, q2 = q[:, :H], q[:, H:]                    # each (T, H, D)
k1, k2 = k[:, :H], k[:, H:]
# 4. RoPE on Q1, K1, Q2, K2 independently
q1 = apply_rope(q1, cos, sin)
q2 = apply_rope(q2, cos, sin)
k1 = apply_rope(k1, cos, sin)
k2 = apply_rope(k2, cos, sin)
# 5. Two attention maps with causal mask
scale = 1.0 / sqrt(D)
a1 = softmax(q1 @ k1.T * scale + causal_mask, dim=-1)   # (T, H, T)
a2 = softmax(q2 @ k2.T * scale + causal_mask, dim=-1)
# 6. Differential subtraction
lam = compute_lambda(lambda_q1, lambda_k1, lambda_q2, lambda_k2, _lambda_init)
attn = a1 - lam * a2                                     # (T, H, T)
# 7. Multiply by V
out = attn @ v                                            # (T, H, D)
# 8. Per-head subLN (RMSNorm on each head's output)
out = subln(out)
# 9. Output projection (with diff-attn scaling factor; see §6.4)
out = (1 - _lambda_init) * out                            # paper scale
out = o_proj(out.reshape(T, H*D))
```

### 6.4 The `(1 - _lambda_init)` scaling factor

The paper applies a scaling factor of `(1 - lambda_init)` before the output projection, to keep the diff-attn output norm comparable to standard attention at init. This is depth-dependent (since `_lambda_init` varies with layer per the schedule chosen in §6.3). Cross-check against the paper Section 2.2 when implementing — both the scaling form and the `_lambda_init` schedule must agree with the paper.

### 6.5 Tied embeddings

`out_proj.weight = embed.weight`. Saves ~33M params at 200M scale (~14%). Standard small-model practice.

---

## 7. Custom Metal kernel design

### 7.1 What goes in the kernel

**Inputs (all bf16):**
- Q1, K1, Q2, K2: each `(B, H, T, D)` where `D = head_dim`
- V: `(B, H, T, D)`
- λ: scalar, computed in Python from learned vectors per layer
- causal flag: bool

**Output (bf16):** `(B, H, T, D)`

**Operations fused inside the kernel:**
- Two QK matmuls (`q1 @ k1.T`, `q2 @ k2.T`)
- Causal mask (additive `-inf` on future positions)
- Two softmaxes along last dim
- Subtraction: `a1 - λ * a2`
- Final matmul with V

**Operations handled in Python OUTSIDE the kernel:**
- RoPE on each of Q1, K1, Q2, K2 (separately, before the kernel call)
- QK RMSNorm (before the kernel call)
- subLN RMSNorm (after the kernel call, on the output)
- `(1 - _lambda_init)` scaling (after subLN)
- Output projection (regular MLX linear, after the kernel)
- Lambda computation (in Python, passed in as scalar)

### 7.2 Staged kernel versions

| Version | What | Used in stage | Purpose |
|---|---|---|---|
| **v0** | Pure-MLX naive reference using `mx.softmax` + `mx.matmul` (no custom kernel) | Stage 0 + ground truth always | Correctness oracle; ensures science completes even if v1 fails |
| **v1** | Naive Metal kernel: two-pass tiled matmul, materializes both `T×T` softmax maps in global memory (bf16, 8MB each at T=2048), subtracts, multiplies by V | Stage 1 onward (after Stage 0 gate) | First real kernel; verifies API end-to-end |
| **v2** | FlashAttention-style block-tiled: online softmax per attention map, fully fused, no `T×T` materialization | Stretch (post Stage 2) | Speed win; paper-worthy artifact |

**Threadgroup memory math for v1:** at T=2048, a single `T×T` attention matrix in fp32 (needed for stable softmax) is `2048^2 * 4 = 16MB` per head per layer. Too big for threadgroup memory (typically ~32KB per threadgroup). So v1 is **two-pass**:
1. Pass 1: compute `softmax(Q1·K1ᵀ)` and `softmax(Q2·K2ᵀ)`, write both to global memory in bf16 (8MB per map per head, manageable)
2. Pass 2: read both maps, subtract with λ, multiply by V, write output

v1 is therefore "two tiled matmuls + element-wise op + tiled matmul," each pass tiled in the standard way. This is the first real Metal work but uses no online-softmax tricks.

**v2 (stretch):** single fused kernel using FlashAttention-style online softmax. Maintains per-row max and sum running stats. Two online softmaxes computed in parallel (registers permitting), subtracted, multiplied by V in the same kernel. Significantly harder; not needed for the science.

### 7.3 Backward pass

For v0: MLX autograd handles it natively (it's built from MLX primitives).

For v1/v2: register the forward kernel via `mx.custom_function`. Implement the backward in **pure MLX ops** (express the gradient computation using regular MLX, not a custom Metal kernel). Slower than a fused backward kernel, but the science doesn't care, and the verification is much easier.

A fused Metal backward kernel is a post-project stretch goal.

### 7.4 Correctness checks (gating)

Each kernel version must pass these before being used to train:

1. **Forward:** `‖v1_out − v0_out‖_∞ < 1e-2` on bf16 inputs (loose tolerance for bf16)
2. **Backward:** finite-difference vs analytical gradient on a 4-layer, 64-hidden, 2-head toy model
3. **End-to-end smoke:** train ~500 steps with v0, then 500 with v1 on the same data seed; final loss within numerical noise

### 7.5 Performance targets (nice-to-have, not gating)

- v1: within 2x of MLX's `scaled_dot_product_attention` for vanilla MHA at same params
- v2: matches or beats MLX's SDPA

The science doesn't depend on hitting these. If neither happens, the comparison is *attention-type quality*, not *throughput*. Throughput is a secondary deliverable.

---

## 8. Data pipeline

### 8.1 Corpus

**FineWeb-Edu (English).** Single source. ~6-8B unique tokens needed (max across all runs; shards are reused across variants and seeds). Total disk ~12-16GB at uint16.

Why FineWeb-Edu:
- Clean, well-documented, widely used
- Single source removes data-quality confounds
- Comparable to what other small-scale repro work uses
- Easy for external reproduction

### 8.2 Tokenizer

**LLaMA-2 32k SentencePiece** (paper-matched).

Pull from a public HF mirror (e.g. `NousResearch/Llama-2-7b-hf`) to avoid the Meta-gated original. Vendor the `tokenizer.model` file into the repo and **pin its SHA-256** in `config.py`.

Why LLaMA-2 specifically: the Diff Transformer paper uses LLaMA-2's tokenizer in its experiments. Matching removes one confound vs the paper.

### 8.3 Tokenization to shards

`data/tokenize.py` reads FineWeb-Edu jsonl/parquet from local cache, tokenizes with the vendored LLaMA-2 SentencePiece model, and writes uint16 streams to `data/shards/`. Val set is stratified by deterministic hash on document ID so the same val examples land in the val pool across re-runs.

Output:
- `data/shards/train-{NNN}.bin` — uint16 token streams, mmap-able
- `data/shards/val.bin` — fixed 50-100M-token deterministic subset
- `data/shards/meta.json` — vocab_size, eot_id, train/val token counts, n_docs, source-list with hashes

### 8.4 MLX-side data loading

Simple path (start here): `numpy.memmap` on uint16 shards. Sample random `(block_size + 1)`-windows. Convert to `mx.array` on demand.

Upgrade only if bottlenecked: `mlx-data` library for pipelined input. Won't bottleneck at 200M scale.

### 8.5 Determinism

- Data order seeded with a separate "data seed" (fixed across runs for clean A/B)
- Per-run "model seed" varies model init only
- Val set is byte-deterministic across all runs

---

## 9. Training plan

### 9.1 Optimizer (paper-matched)

- **AdamW:** β1=0.9, β2=0.95, eps=1e-8, weight_decay=0.1
- **Grad clip:** 1.0
- **LR schedule:** linear warmup → cosine decay to 10% of peak
- **Peak LR by stage:**
  - Stage 0: `6e-4`, warmup 500 steps
  - Stage 1: `4e-4`, warmup 1000 steps
  - Stage 2: `3e-4`, warmup 2000 steps

### 9.2 Batch and sequence

| | block_size | micro_batch | grad_accum | eff. tokens/step | total steps |
|---|---|---|---|---|---|
| Stage 0 (10M) | 1024 | 16 | 1 | ~16k | ~6,250 |
| Stage 1 (100M) | 2048 | 32 | 1 | ~64k | ~31,000 |
| Stage 2 (200M) | 2048 | 32 | 4 | ~256k | ~15,600 |

These are estimates. **Calibrate after 200 steps of Stage 0** by reading `tokens/sec` from `metrics.jsonl`. If Stage 2 projects past 14 days wall, cut to 3B tokens or drop a seed.

### 9.3 Eval cadence

- Stage 0: every 500 steps
- Stages 1-2: every 1000 steps
- Set: fixed 50-100M-token held-out FineWeb-Edu subset, deterministic across runs
- Metric: token-level NLL (report perplexity = exp(NLL))
- Early stop: 5 evals without val improvement → save and move on

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
- `tokenizer.sha256` — vendored tokenizer file hash
- `data_meta.json` — shard list + hashes
- `mlx_version.txt` — pinned MLX version
- `seed.txt` — model and data seeds

### 9.6 Failure policy

- **NaN/Inf:** abort, log, inspect manually. **No auto-restart.** Kernel bugs need human attention.
- **Manual resume:** from `latest.pt` saved every 1000 steps.
- **Spike auto-restart:** **not** implemented in this project. Bias toward debugging over self-healing for a research run.
- **Plateau early-stop:** 5 evals without val improvement → save final, exit cleanly.

---

## 10. Evaluation

### 10.1 Primary metric

Token-level NLL on the held-out FineWeb-Edu val set (`exp(NLL)` reported as perplexity for readability).

### 10.2 Variance reporting

At each scale and each variant: mean ± stdev across seeds. Report as a table and as overlapping curves with shaded variance bands.

### 10.3 Per-token learning-curve comparison

The paper's strongest claim is the **token-efficiency** dimension: diff-attn matches vanilla at fewer tokens. To test this:

- Plot val loss vs tokens-trained for each variant
- Identify the token count at which diff-attn first reaches vanilla's final val loss
- Report as a ratio (e.g. "diff-attn reached vanilla's 4B-token loss at 2.6B tokens")

### 10.4 Optional downstream eval

At 200M, zero-shot downstream tasks are barely above chance. **Skip them by default** to avoid noise drowning the result. If you want one anyway: LAMBADA accuracy. Don't expect much.

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
  pyproject.toml                # mlx + numpy + tokenizers + huggingface_hub
  config.py                     # ModelConfig (attn_variant flag), TrainConfig, paths
  docs/
    2026-05-20-diffattn-mlx-reproduction-design.md   # this file
    (implementation plan, written next by writing-plans)

  model.py                      # Transformer + DiffAttentionLayer + VanillaMHALayer

  kernels/
    diff_attention.py           # Python wrapper, autograd registration, v0/v1/v2 selector
    diff_attention.metal        # MSL source for v1 (and v2 if reached)
    preflight.py                # Stage P softmax wrapper + autograd hook
    preflight.metal             # MSL source for Stage P softmax kernel

  data/
    tokenizer/
      tokenizer.model           # vendored LLaMA-2 tokenizer (~500KB)
    shards/                     # uint16 FineWeb-Edu shards (gitignored)
    tokenize.py                 # FineWeb-Edu → uint16 shards using LLaMA-2 tokenizer
    loader.py                   # numpy mmap loader, deterministic batch sampler

  train.py                      # MLX training loop, single-GPU
  eval.py                       # val loss + perplexity computation

  tests/
    test_model.py               # Architecture forward/backward
    test_kernel.py              # Diff-attn correctness (v0 vs v1, gradient check)
    test_preflight.py           # Stage P softmax correctness
    test_data.py                # Loader correctness, val determinism

  runs/                         # gitignored
    preflight/
    stage0/
    stage1-{variant}-seed{N}/
    stage2-{variant}-seed{N}/
```

**Design rules for the file layout:**
- Single `model.py` with both attention variants (toggle via config flag) — single source of truth, ensures non-attention layers are byte-identical between variants
- Kernel work isolated to `kernels/` — clean boundary; easy to drop v1/v2 if pre-flight fails
- Pre-flight has its own self-contained files so Stage P can be skipped cleanly
- `runs/` gitignored; per-stage/per-variant/per-seed naming

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Metal kernel never converges to correctness | Stage P gate; staged v0/v1/v2; v0 ships the science regardless |
| Training instability (loss spikes, NaN) at 100-200M | Conservative LR; LLaMA-style proven backbone; NaN abort + manual inspect; emergency: cut LR 2x, double warmup |
| Wall-time blowout (M5 Max slower than estimated) | Calibrate at Stage 0 step 200; if Stage 2 projects past 14 days, drop a seed or cut to 3B tokens |
| Signal absent at Stage 1 | Either: write up "doesn't replicate at 100M" as a legit finding, or proceed to Stage 2 (paper's strongest claim is scaling-direction, may emerge at 200M); decide after looking at the curves |
| LLaMA-2 tokenizer is HF-gated | Use a public mirror (e.g. `NousResearch/Llama-2-7b-hf`); vendor the tokenizer file into the repo and pin its SHA |
| MLX API churn during 2-3 week project | Pin MLX version in `pyproject.toml`; do not auto-upgrade mid-project |
| Mac needed for life during long runs | Each stage is independent; pause between stages fine; `latest.pt` resume works; each variant/seed run is ~3-5 days for Stage 2 |
| Memory pressure | 128GB unified is huge; 200M training peaks ~15-20GB; not a real risk |
| Lambda reparameterization init bug | Use `randn * 0.1` for lambda vectors (NOT zero); zero init makes grads zero at step 0. See §6.3. |
| RoPE / QK norm ordering bug | Q and K each get RMSNormed BEFORE RoPE; RoPE applied to each of Q1, K1, Q2, K2 independently. See §6.3 forward-pass pseudocode. |
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
- [`NousResearch/Llama-2-7b-hf`](https://huggingface.co/NousResearch/Llama-2-7b-hf) — public mirror with the tokenizer (avoids Meta gate)

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
- Don't pretrain the LLaMA-2 tokenizer. Vendor it as-is.
- Don't auto-upgrade MLX mid-project. Pin the version in `pyproject.toml`.

---

## 15. Notes for the next Claude session (resume cold)

If a future session lands in this directory cold (no conversation history):

1. **Read this file first** end-to-end. It is the authoritative design.
2. **Read the paper** ([arXiv 2410.05258](https://arxiv.org/abs/2410.05258)), especially §2, before touching attention code.
3. **Check `runs/`** for any existing checkpoints; the project state is in checkpoint metadata + `metrics.jsonl`.
4. **Don't auto-upgrade MLX.** Check the pinned version in `pyproject.toml`.
5. **Stage P (pre-flight) status** dictates whether v1/v2 kernel work is live. If pre-flight failed or was skipped, only v0 (pure-MLX) is in scope.
6. **The next step after this design doc** is the implementation plan, to be written by the `writing-plans` skill into `docs/2026-05-20-diffattn-mlx-implementation-plan.md`.

If the user says "carry on" with no other context: check `runs/` for the most recent run, read its `metrics.jsonl` to see where training is, and resume from `latest.pt` if applicable. If no runs exist, begin with Stage P.

---

## 16. Sign-off

Design approved by Guy 2026-05-20 in conversation.

Next step: route this doc to Codex for an independent review pass (the user will do this manually). After that, invoke the `writing-plans` skill to produce an implementation plan.
