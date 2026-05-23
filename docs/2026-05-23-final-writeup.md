# diff-mlx: MLX implementation of Differential Transformer with custom Metal kernels

**Date:** 2026-05-23 (draft, in-progress)
**Author:** Guy J. Grigsby
**Repo:** [github.com/guygrigsby/diff-mlx](https://github.com/guygrigsby/diff-mlx)

## TL;DR

Reimplemented the Differential Transformer architecture (Ye et al., ICLR 2025; arXiv 2410.05258) in MLX on Apple Silicon. Wrote custom Metal kernels for the differential-attention forward path (softmax + causal SDPA). Verified correctness three ways: paper-fidelity against the vendored Microsoft PyTorch reference at 1e-7 (CPU stream), v0/v1 numerical agreement, and cross-stack reproduction in PyTorch on NVIDIA RTX 3070 Ti. At Stage 0 scale (30M params, 100M tokens), paired δ replicates the paper's directional claim (diff outperforms vanilla post-crossover) within seed noise on both implementations. [Stage 1 result pending; section to be filled when the 30k-step paired run completes.]

## What this is

A small-scale, controlled reproduction of the diff-attn mechanism, with the implementation done entirely on Apple Silicon using MLX and custom Metal kernels. Not a direct paper reproduction: the paper trained 3B-parameter models on 1T tokens using H100 clusters; this work targets Stage 0/1 model sizes (30M / 162M params) on a single M5 Max.

What's novel:

1. **MLX port** of the paper's algorithm, paper-fidelity at the algorithm level (verified against the vendored Microsoft reference at 1e-7 on CPU stream).
2. **Custom Metal kernels** (P1 softmax, P2 causal SDPA) wired through `mx.fast.metal_kernel` with `mx.custom_function` autograd hooks.
3. **Cross-stack validation** in PyTorch on NVIDIA CUDA confirming the δ replicates outside MLX.
4. **Throughput investigation** demonstrating Apple Silicon's actual MLX bf16 throughput at training shapes and the swap-cliff phenomenon at micro_batch sizes that exceed unified-memory budget.

## Project arc

The project went through several scope adjustments as we discovered what Apple Silicon could and couldn't do. The path is worth tracing because the dead ends are informative.

### Phase A: backbone + vanilla MHA

Standard pre-norm LLaMA-style transformer in MLX. Tied embeddings, RoPE, SwiGLU, RMSNorm. Two findings worth keeping:

- **`mx.fast.scaled_dot_product_attention` API.** No `is_causal` flag; `scale` is mandatory keyword-only; `mask="causal"` is the documented causal-mask syntax. The design doc was wrong about this on first pass.
- **`mx.fast.rope(traditional=...)` convention.** `traditional=True` is GPT-J consecutive-pair (interleaved); `traditional=False` is LLaMA rotate-halves. The first round of the design assumed LLaMA, but the paper's reference uses interleaved. Switched both variants to `traditional=True` for paper fidelity. See R6, R7 in the design history.

### Phase B: Differential Attention + paired-init protocol

The diff-attn module per design §6.3:

- `n_heads_diff = n_heads_vanilla / 2`, `qk_head_dim = D`, `v_head_dim = 2D`.
- λ vectors stored fp32 (4 vectors per layer); λ_init depth-scheduled per the paper.
- v0 forward: two SDPA calls + Python subtract. Uses the §7.1 linearity identity `(A1 − λA2)V = A1V − λA2V` so no T×T attention map is materialized.
- subln: per-head RMSNorm over 2D, applied AFTER differential subtraction.

Two bugs caught by the cross-check against the vendored Microsoft reference (`microsoft/unilm/Diff-Transformer/multihead_diffattn.py`):

1. **Head-pair split.** Design originally specified `q1, q2 = q[:H], q[H:]` (halves). Reference uses interleaved `(H, 2)` split: diff-head h pairs `q[2h]` with `q[2h+1]`. The internal v0 oracle test passed under the buggy halves split because the oracle used the same buggy convention. Textbook example of why §7.4 mandates a cross-check against a different codebase. Fixed; cross-check max |diff| dropped from 0.77 to 3.58e-7 on CPU stream.
2. **RoPE convention** (R7 above) was the other.

Stage 0 paired result (seed 0):

| | val_full at step 5000 | perplexity |
|---|---|---|
| vanilla | 4.720 | 112.2 |
| diff | 4.700 | 109.9 |
| **δ** | **−0.020 nats** | **−2.3** |

Paired δ trajectory shows clean crossover at step ~3000, monotonic post-crossover. Directional replication of the paper's claim.

### Phase D prerequisites (mostly done as infrastructure)

bf16 mixed precision, optimizer-state checkpoints, auto-resume from latest, grad-accumulation, multi-seed orchestrator, caffeinate wrappers. All shipped. See `docs/archive/2026-05-21-phase-d-prereqs-3to5-plan.md` and `docs/2026-05-21-bf16-mixed-precision-design.md`.

### Stage 1 throughput investigation: the swap cliff

Initial Stage 1 paired run hit 950 tokens/sec, projecting ~50 days per variant. Investigation showed:

| micro_batch | tps | peak MLX memory |
|---|---|---|
| 4 | 14,329 | 25.3 GB |
| 8 | 13,832 | 48.4 GB |
| 16 | 11,290 | 94.2 GB |
| 24 | 8,414 | 135.2 GB (over 128 GB unified ceiling) |
| 32 (live run) | 950 | swap-thrashing |

The 950 tps reading was a swap-thrashing artifact, not the M5 Max's actual MLX bf16 ceiling. Dropping `micro_batch` from 32 to 8 (with `grad_accum=4` to preserve effective batch) plus calling `mx.eval` between micro-batches in the accumulator (so the lazy graph doesn't hold N micro-batches' activations live) plus wrapping the training step in `mx.compile` together restored throughput to ~14k tps. Full retro: `docs/2026-05-22-swap-cliff-and-scope-restore.md`. Mid-day pivot retro that the swap-cliff finding partially reversed: `docs/2026-05-22-stage1-pivot-retro.md`.

Lesson: **per-token cost is flat-then-cliff in batch size**, with the cliff at the unified-memory budget. The naive "smaller batch loses throughput" intuition fails when the original batch was swapping.

## Custom Metal kernels (Phase C)

Two kernels via `mx.fast.metal_kernel`. Both written in Metal Shading Language (a C++14 dialect with GPU-specific primitives). Kernel source lives inside Python triple-quoted strings; MLX JIT-compiles at first call and caches per-template-arg combination.

### P1: row-wise softmax

One threadgroup per row. 256 threads strided over the row dim. Two shared-memory tree reductions (max for stability, then sum after `exp`). Internal compute fp32; output dtype matches input.

Acceptance (per design §5.1):

| | tolerance | actual |
|---|---|---|
| fp32 vs `mx.softmax` | < 1e-4 | ~3e-8 across shapes including Stage 1's (32, 12, 2048) |
| bf16 vs `mx.softmax` | < 1e-2 | ~2e-3 |
| Backward vs MLX autograd | match | ~1e-8 |
| Finite-difference gradient | passes | ~1e-4 (fp32 round-off bound) |

Source: `kernels/softmax_p1.py`. 13 unit tests in `tests/test_softmax_p1.py`.

### P2: causal SDPA

Single-map causal scaled dot-product attention. Same algebra as `mx.fast.scaled_dot_product_attention(..., mask="causal")`. One threadgroup per (B, H, query_row); 256 threads strided over the key dim. Three phases:

1. Compute S[k] = Q · K[k] for k ≤ q; track row max for stability.
2. Tree-reduce max; S[k] := exp(S[k] − max); accumulate row sum.
3. Tree-reduce sum; output[d] = Σ_k (S[k] / sum) · V[k, d].

Separate `head_dim_qk` and `head_dim_v` template params so v1 can pass V at width 2D (diff-attn's doubled-V head). Causal mask hardcoded inside the kernel. Internal compute fp32; bf16 or fp32 I/O.

Acceptance (per design §5.1b):

| | tolerance | actual |
|---|---|---|
| bf16 vs `mx.fast SDPA` on Stage 0/1 vanilla and diff shapes | < 2e-2 abs OR < 5e-3 rel | ~1.5e-2 abs / ~3e-3 rel (bf16 ULP-noise band; same as MLX-vs-MLX manual softmax+matmul) |
| fp32 (toy) | < 1e-2 | ~3e-3 |
| Causal mask invariant | < 1e-7 | 0.0 exactly |
| Backward vs MLX autograd | < 2e-2 | ~1e-2 |

**Numerical note.** The design originally specified 1e-2 absolute tolerance for bf16. Measurement showed MLX's own `mx.fast SDPA` disagrees with a manual `mx.softmax + matmul` composition by 1.56e-2 absolute at Stage 1 shapes. Two correct bf16 implementations of the same algorithm differ at the 1-ULP level. The "1e-2" gate was implicitly for outputs in [−1, 1]; actual post-AV outputs at our shapes have magnitude ~5, so 2e-2 absolute / 5e-3 relative is the realistic band. Our kernel sits inside that band.

Source: `kernels/sdpa_p2.py`. 10 unit tests in `tests/test_sdpa_p2.py` covering the full design shape matrix (toy, Stage 0 vanilla, Stage 0 diff-sub, Stage 1 vanilla, Stage 1 diff-sub, Stage 0 diff-full at D_v=128, Stage 1 diff-full at D_v=128).

### v1 diff composition

`DiffAttention(kernel_version="v1")` swaps the two `mx.fast.scaled_dot_product_attention` calls for `kernels.sdpa_p2`. Same algebra, paper-canonical interleaved head split, same causal mask. Backward delegates to MLX's SDPA autograd via `mx.custom_function`.

Acceptance: v1 forward vs v0 forward at bf16 < 4e-2 absolute / 1e-2 relative (P2's noise band plus the small amplification from `out1 − λ * out2`); v1 vs the vendored PyTorch reference fixture < 1e-2 (GPU-stream tolerance; v1 only runs on GPU, where Metal's reduced-precision fp32 matmul applies). v1 with bf16 inputs is gated to be ≤ MLX's own SDPA on speed (kernel speed eval section TBD).

Source: `model.py:DiffAttention`. 5 v1-specific tests in `tests/test_diff_v1.py`.

## Cross-stack validation: PyTorch on RTX 3070 Ti

To rule out MLX-specific artifacts, ported the model + paired-init + training driver to PyTorch and ran Stage 0 paired on an NVIDIA RTX 3070 Ti. Same algorithm, different framework, different chip. Cross-check fixture passes on both CPU and CUDA (TF32 disabled). Code at `pytorch_ref/`.

Two findings during the port:

1. **Logits tensor must NOT be explicitly cast to fp32 in the autocast region.** MLX has no autocast, so the MLX side explicitly does `logits.astype(mx.float32)` before CE. In PyTorch under `torch.autocast(bfloat16)`, `F.cross_entropy` is already on the always-fp32 op list and computes internally without materializing a full fp32 `(B, T, vocab)` tensor. The explicit cast doubled memory and OOMed the 8 GB 3070 Ti at Stage 0 shapes. Removing it fixed the OOM. Worth flagging: this is exactly the kind of porting subtlety a cross-stack effort surfaces.
2. **Default `nn.Embedding` init differs by 16×.** PyTorch defaults to N(0, 1); MLX defaults to N(0, 1/√dim) (std ~0.0625 at dim=256). At unmatched inits, PyTorch starts CE at ~246 (vs MLX's ~12), descends along a different trajectory, and the paired δ even runs in the opposite direction. Mirror-symmetric artifact, not a bug. Aligning the PyTorch embedding init to match MLX (`nn.init.normal_(self.tok_embed.weight, std=1/√dim)`) restored agreement. (Linears already agreed: both stacks use `1/√fan_in` uniform with std ~0.0361.)

After the init fix, Stage 0 paired δ comparison:

| step | PyTorch δ | MLX δ |
|---|---|---|
| 500 | +0.154 | +0.166 |
| 1000 | +0.127 | +0.118 |
| 2000 | +0.055 | +0.039 |
| 3000 | **−0.030 (crossover)** | **−0.014 (crossover)** |
| 4000 | −0.033 | −0.036 |
| 5000 | −0.015 | −0.006 |
| 6000 | −0.018 | −0.025 |

val_loss_full at step 5000:

| | vanilla | diff | δ |
|---|---|---|---|
| MLX (M5 Max) | 4.720 | 4.700 | −0.020 |
| PyTorch (3070 Ti) | 4.799 | 4.770 | −0.030 |

Same direction, same crossover step (~3000), same order-of-magnitude δ. The paper's directional claim is **not an MLX/Metal numerical artifact**.

Wall times: PyTorch vanilla at Stage 0 took 47 min on the 3070 Ti vs 82 min for MLX on the M5 Max (`micro_batch=2 grad_accum=8` on the 3070 Ti due to 8 GB VRAM; `micro_batch=16` on the Mac). Tensor cores help, just not as dramatically as the theoretical peak suggests. At Stage 0 model size the matmul kernels can't fully saturate them.

## Apple Silicon for training: honest numbers

The project surfaced concrete throughput numbers for transformer training on M5 Max with MLX:

- **Per-token cost at Stage 1 (162M params, T=2048, bf16): ~0.07 ms/token at B=4-8.** Translates to ~14k tokens/sec under `mx.compile`-wrapped train_step with gradient accumulation through a compiled forward+backward.
- **Sustained TFLOPS at this workload: ~1 of the ~15-20 TFLOPS bf16 peak**, i.e. 5-10% utilization. GPU residency in macmon hovers at 30-40% (dispatch-bound); power draw ~8-9W (~30% of peak).
- **The dispatch-boundedness is exactly what Phase C kernels are designed to address.** Each MLX kernel finishes in microseconds and the GPU sits idle waiting for the next dispatch. Larger fused kernels (custom MSL) reduce dispatch count and should pull utilization up. [Kernel speed eval pending; section to be filled in.]

For context: paper-scale (1T tokens, 3B model) runs took H100 clusters at Microsoft. A single M5 Max at the observed throughput would need ~33,000 years for the headline 3B-1T-token run. Apple Silicon is not a training-cluster substitute, but it is capable of small-scale controlled reproductions of architectural claims, which is what this project did.

## What this contributes beyond the paper

- **Working MLX implementation** of Differential Attention with paper-fidelity correctness against the official reference.
- **Custom Metal kernels** for the diff-attn forward path (P1 softmax + P2 causal SDPA + v1 composition), with autograd hooks for training use.
- **Paired-init protocol** (byte-identical shared weights between vanilla and diff variants) that makes single-seed δ measurements meaningful, and that survives port across implementations.
- **Cross-stack validation** showing the directional δ claim isn't an MLX/Metal artifact.
- **Throughput investigation** documenting the swap-cliff phenomenon, dispatch-bound regime, and realistic numbers for Apple Silicon transformer training.

## Stage 1 paired result

[Run in progress. Vanilla at step 7,390 / 30,517 as of writing (24%), train_loss 3.57, ~17,900 tps cumulative. Updated wall projection: ~2.5 days for paired total. Section to be filled when the run completes.]

## Stage 2 paired result

[Optional run, ~14 days unattended at current throughput. Decision pending after Stage 1.]

## Acknowledgments

- Microsoft Research's `unilm/Diff-Transformer` repo for the canonical PyTorch reference.
- Apple's MLX team for the `mx.fast.metal_kernel` API, which made the custom-kernel work tractable.
- The work was supported by writing partner [Claude](https://claude.ai) (Opus 4.7, 1M-context).

## Pointers

- **Active design:** `docs/2026-05-20-diffattn-mlx-reproduction-design.md` (especially §5.1, §5.1b, §7 for kernel specs)
- **Phase C plan:** `docs/2026-05-22-phase-c-plan.md`
- **Swap-cliff finding:** `docs/2026-05-22-swap-cliff-and-scope-restore.md`
- **bf16 design:** `docs/2026-05-21-bf16-mixed-precision-design.md`
- **PyTorch port:** `pytorch_ref/` (README has Windows + macOS setup)
- **Kernels:** `kernels/softmax_p1.py`, `kernels/sdpa_p2.py`
- **Tests:** `tests/` (132 tests at writing); `pytorch_ref/tests/` (13 tests on the PyTorch side)
