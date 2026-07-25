# diff-mlx: MLX implementation of Differential Transformer with custom Metal kernels

**Date:** 2026-05-27
**Author:** Guy J. Grigsby
**Repo:** [github.com/guygrigsby/diff-mlx](https://github.com/guygrigsby/diff-mlx)

> **Update 2026-07-25:** the single-seed caveat below is now measured. A
> four-seed rerun band puts the Stage 0 paired δ at −0.016 ± 0.023 nats
> with a sign flip at seed 3, so the Stage 0 −0.020 "win" was seed noise
> and there was no small-scale effect for Stage 1 to kill. See
> [`2026-07-25-stage0-seed-band.md`](2026-07-25-stage0-seed-band.md).
> This document is otherwise left as written.

## TL;DR

Reimplemented the Differential Transformer (Ye et al., ICLR 2025; arXiv 2410.05258) in MLX on Apple Silicon, with custom Metal kernels for the differential-attention forward path (softmax + causal SDPA). Checked correctness three ways: against the vendored Microsoft PyTorch reference at 1e-7 (CPU stream), v0 vs v1 numerical agreement, and a separate PyTorch run on an NVIDIA RTX 3070 Ti.

At Stage 0 (30M params, 100M tokens) the paired δ reproduced the paper's direction: diff beat vanilla by 0.020 nats on held-out val, inside seed noise. At Stage 1 (162M params, 2.0B tokens) it didn't hold. Diff finished 0.11 nats ahead on *train* loss and 0.035 nats behind vanilla on held-out val, and its last leg shows it overfitting. Binned the held-out loss by token position and vanilla was uniformly ahead across the whole 2048-token window, with no widening at later positions, so the long-context advantage diff is supposed to have didn't show up either. So at this scale the train-loss win is just memorization. Stopped the experiment after Stage 1.

## What this is

A small-scale, controlled reproduction of the diff-attn mechanism, implemented end to end on Apple Silicon with MLX and custom Metal kernels. Not a paper reproduction: the paper trained 3B-param models on 1T tokens on H100 clusters. This targets Stage 0/1 sizes (30M / 162M params) on one M5 Max.

What's new here:

1. **MLX port** of the algorithm, paper-faithful (verified against the Microsoft reference at 1e-7 on the CPU stream).
2. **Custom Metal kernels** (P1 softmax, P2 causal SDPA) through `mx.fast.metal_kernel`, with `mx.custom_function` autograd hooks.
3. **Cross-stack check** in PyTorch on NVIDIA CUDA, confirming the Stage 0 δ isn't an MLX artifact.
4. **Throughput, thermal, and power notes** for sustained MLX training on a laptop: the real bf16 numbers, a swap cliff, the thermal envelope, and a power-delivery throttle when charging through a dock.
5. **A scale-up that killed a small-scale positive.** The Stage 0 win didn't survive Stage 1, which is exactly what single-seed small-scale studies tend to miss.

## Project arc

The project moved through a few scope changes as we learned what Apple Silicon could and couldn't do. The dead ends are worth tracing.

### Phase A: backbone + vanilla MHA

Standard pre-norm LLaMA-style transformer in MLX. Tied embeddings, RoPE, SwiGLU, RMSNorm. Two findings worth keeping:

- **`mx.fast.scaled_dot_product_attention` API.** No `is_causal` flag; `scale` is mandatory keyword-only; `mask="causal"` is the documented causal syntax. The design doc had this wrong on the first pass.
- **`mx.fast.rope(traditional=...)` convention.** `traditional=True` is GPT-J consecutive-pair (interleaved); `traditional=False` is LLaMA rotate-halves. The first design assumed LLaMA, but the paper's reference uses interleaved. Switched both variants to `traditional=True` for fidelity. See R6, R7 in the design history.

### Phase B: Differential Attention + paired-init protocol

The diff-attn module per design §6.3:

- `n_heads_diff = n_heads_vanilla / 2`, `qk_head_dim = D`, `v_head_dim = 2D`.
- λ vectors stored fp32 (4 per layer); λ_init depth-scheduled per the paper.
- v0 forward: two SDPA calls plus a Python subtract. Uses the §7.1 linearity identity `(A1 − λA2)V = A1V − λA2V`, so no T×T attention map gets materialized.
- subln: per-head RMSNorm over 2D, applied after the differential subtraction.

The cross-check against the Microsoft reference (`microsoft/unilm/Diff-Transformer/multihead_diffattn.py`) caught two bugs:

1. **Head-pair split.** Design originally said `q1, q2 = q[:H], q[H:]` (halves). The reference uses an interleaved `(H, 2)` split: diff-head h pairs `q[2h]` with `q[2h+1]`. The internal v0 oracle test passed under the buggy halves split because the oracle used the same buggy convention. This is the textbook reason §7.4 mandates a cross-check against a different codebase. Fixed; cross-check max |diff| dropped from 0.77 to 3.58e-7 on the CPU stream.
2. **RoPE convention** (R7 above) was the other.

Stage 0 paired result (seed 0):

| | val_full at step 5000 | perplexity |
|---|---|---|
| vanilla | 4.720 | 112.2 |
| diff | 4.700 | 109.9 |
| **δ** | **−0.020 nats** | **−2.3** |

The paired δ crossed over cleanly around step 3000 and stayed monotonic after. A directional match for the paper at this scale. Stage 1 below didn't hold, which recasts the 0.020 as plausibly seed noise.

### Phase D prerequisites (infrastructure)

bf16 mixed precision, optimizer-state checkpoints, auto-resume from latest, grad accumulation, multi-seed orchestrator, caffeinate wrappers. All shipped. See `docs/archive/2026-05-21-phase-d-prereqs-3to5-plan.md` and `docs/2026-05-21-bf16-mixed-precision-design.md`.

### Stage 1 throughput investigation: the swap cliff

The first Stage 1 paired run hit 950 tokens/sec, which projected to ~50 days per variant. Digging in:

| micro_batch | tps | peak MLX memory |
|---|---|---|
| 4 | 14,329 | 25.3 GB |
| 8 | 13,832 | 48.4 GB |
| 16 | 11,290 | 94.2 GB |
| 24 | 8,414 | 135.2 GB (over the 128 GB unified ceiling) |
| 32 (live run) | 950 | swap-thrashing |

So the 950 tps was a swap-thrashing artifact, not the M5 Max's real bf16 ceiling. Three things together fixed it: drop `micro_batch` from 32 to 8 (with `grad_accum=4` to keep effective batch), call `mx.eval` between micro-batches in the accumulator so the lazy graph doesn't hold N micro-batches of activations live, and wrap the step in `mx.compile`. Back to ~14k tps. Full retro: `docs/2026-05-22-swap-cliff-and-scope-restore.md`. The mid-day pivot retro that this finding partly reversed: `docs/2026-05-22-stage1-pivot-retro.md`.

The lesson: per-token cost is flat, then it falls off a cliff at the unified-memory budget. The usual "smaller batch loses throughput" intuition is backwards when the original batch was swapping.

## Custom Metal kernels (Phase C)

Two kernels through `mx.fast.metal_kernel`. Both in Metal Shading Language (a C++14 dialect with GPU primitives). The kernel source lives in Python triple-quoted strings; MLX JIT-compiles on first call and caches per template-arg combination.

### P1: row-wise softmax

One threadgroup per row. 256 threads strided over the row dim. Two shared-memory tree reductions (max for stability, then sum after `exp`). Internal compute fp32; output dtype matches input.

Acceptance (design §5.1):

| | tolerance | actual |
|---|---|---|
| fp32 vs `mx.softmax` | < 1e-4 | ~3e-8 across shapes including Stage 1's (32, 12, 2048) |
| bf16 vs `mx.softmax` | < 1e-2 | ~2e-3 |
| backward vs MLX autograd | match | ~1e-8 |
| finite-difference gradient | passes | ~1e-4 (fp32 round-off bound) |

Source: `kernels/softmax_p1.py`. 13 tests in `tests/test_softmax_p1.py`.

### P2: causal SDPA

Single-map causal scaled dot-product attention. Same algebra as `mx.fast.scaled_dot_product_attention(..., mask="causal")`. One threadgroup per (B, H, query_row); 256 threads strided over the key dim. Three phases:

1. Compute S[k] = Q · K[k] for k ≤ q; track row max for stability.
2. Tree-reduce max; S[k] := exp(S[k] − max); accumulate row sum.
3. Tree-reduce sum; output[d] = Σ_k (S[k] / sum) · V[k, d].

Separate `head_dim_qk` and `head_dim_v` template params, so v1 can pass V at width 2D (diff-attn's doubled-V head). Causal mask hardcoded in the kernel. Internal compute fp32; bf16 or fp32 I/O.

Acceptance (design §5.1b):

| | tolerance | actual |
|---|---|---|
| bf16 vs `mx.fast SDPA`, Stage 0/1 vanilla and diff shapes | < 2e-2 abs OR < 5e-3 rel | ~1.5e-2 abs / ~3e-3 rel (bf16 ULP-noise band; same as MLX-vs-MLX manual softmax+matmul) |
| fp32 (toy) | < 1e-2 | ~3e-3 |
| causal mask invariant | < 1e-7 | 0.0 exactly |
| backward vs MLX autograd | < 2e-2 | ~1e-2 |

A numerical note worth flagging. The design originally set 1e-2 absolute tolerance for bf16. Then measurement showed MLX's own `mx.fast SDPA` disagrees with a manual `mx.softmax + matmul` by 1.56e-2 absolute at Stage 1 shapes. Two correct bf16 implementations of the same algorithm differ at the 1-ULP level. The "1e-2" gate was implicitly for outputs in [−1, 1]; our post-AV outputs run around magnitude 5, so the realistic band is 2e-2 absolute / 5e-3 relative. The kernel sits inside it.

Source: `kernels/sdpa_p2.py`. 10 tests in `tests/test_sdpa_p2.py` covering the full shape matrix (toy, Stage 0 vanilla, Stage 0 diff-sub, Stage 1 vanilla, Stage 1 diff-sub, Stage 0 diff-full at D_v=128, Stage 1 diff-full at D_v=128).

### v1 diff composition

`DiffAttention(kernel_version="v1")` swaps the two `mx.fast.scaled_dot_product_attention` calls for `kernels.sdpa_p2`. Same algebra, paper-canonical interleaved head split, same causal mask. Backward delegates to MLX's SDPA autograd through `mx.custom_function`.

Acceptance: v1 forward vs v0 forward at bf16 < 4e-2 absolute / 1e-2 relative (P2's noise band plus the small amplification from `out1 − λ * out2`); v1 vs the vendored PyTorch reference fixture < 1e-2 (GPU-stream tolerance; v1 only runs on GPU, where Metal's reduced-precision fp32 matmul applies).

Source: `model.py:DiffAttention`. 5 v1 tests in `tests/test_diff_v1.py`.

## Cross-stack check: PyTorch on RTX 3070 Ti

To rule out MLX-specific artifacts, ported the model, the paired init, and the training driver to PyTorch and ran Stage 0 paired on an NVIDIA RTX 3070 Ti. Same algorithm, different framework, different chip. The cross-check fixture passes on CPU and CUDA (TF32 disabled). Code at `pytorch_ref/`.

Two things surfaced during the port:

1. **Don't cast the logits to fp32 inside the autocast region.** MLX has no autocast, so the MLX side explicitly does `logits.astype(mx.float32)` before CE. Under `torch.autocast(bfloat16)`, `F.cross_entropy` is already on the always-fp32 op list and computes internally without materializing a full fp32 `(B, T, vocab)` tensor. The explicit cast doubled memory and OOMed the 8 GB 3070 Ti at Stage 0 shapes. Dropping it fixed the OOM.
2. **Default `nn.Embedding` init differs by 16×.** PyTorch defaults to N(0, 1); MLX defaults to N(0, 1/√dim) (std ~0.0625 at dim=256). At unmatched inits, PyTorch starts CE at ~246 (MLX starts at ~12), descends a different trajectory, and the paired δ even runs the opposite direction. A mirror-symmetric artifact, not a bug. Matching the PyTorch embedding init to MLX (`nn.init.normal_(self.tok_embed.weight, std=1/√dim)`) restored agreement. The linears already agreed: both use `1/√fan_in` uniform, std ~0.0361.

After the init fix, the Stage 0 paired δ lined up:

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

Same direction, same crossover step (~3000), same order of magnitude at Stage 0. So the Stage 0 result isn't an MLX/Metal numerical artifact. It also didn't survive scale-up; see Stage 1.

Wall times: PyTorch vanilla at Stage 0 took 47 min on the 3070 Ti vs 82 min for MLX on the M5 Max (`micro_batch=2 grad_accum=8` on the 3070 Ti for the 8 GB VRAM; `micro_batch=16` on the Mac). Tensor cores help, just not as much as the theoretical peak implies. At Stage 0 model size the matmul kernels can't saturate them.

## Stage 1 paired result

Both variants trained to 2.0B tokens (30,517 steps, effective batch 32, peak LR 4e-4, 1000-step warmup), seed 0, byte-identical paired init, identical data order. This is the headline experiment.

The train-loss and held-out signals disagree.

| metric | diff | vanilla | δ (diff − vanilla) |
|---|---|---|---|
| final train loss (last 1000-step mean) | 3.0414 | 3.1526 | **−0.111** (diff lower) |
| held-out `val_loss_full` (75M tok) @ step 30000 | 3.3616 | 3.3265 | **+0.035** (vanilla lower) |

Diff wins train loss and loses held-out. The val curves stay within ±0.04 of each other the whole run, and the Stage 0 crossover never reappears on val at Stage 1. The last leg gives it away: from step 25k to 30k, diff's train loss kept falling while its val loss *rose* (3.3285 → 3.3616), and vanilla improved on val over the same span (3.3427 → 3.3265). That's overfitting. The 0.11-nat train advantage is diff memorizing the training stream harder, not learning a better function.

![Train vs val loss, diff vs vanilla](stage1_diff_vs_vanilla.png)

### Position-binned eval: testing the long-context claim

Diff-attn's whole pitch is that cancelling common-mode attention noise helps most with long or noisy context, so its edge should grow with position. So we binned held-out NLL by token position in the 2048-token window (4M val tokens, both final checkpoints; `scripts/eval_position_binned.py`):

| position | diff | vanilla | δ |
|---|---|---|---|
| 0-255 | 3.574 | 3.529 | +0.046 |
| 256-511 | 3.380 | 3.341 | +0.039 |
| 512-767 | 3.339 | 3.304 | +0.035 |
| 768-1023 | 3.335 | 3.296 | +0.039 |
| 1024-1279 | 3.324 | 3.287 | +0.036 |
| 1280-1535 | 3.324 | 3.288 | +0.035 |
| 1536-1791 | 3.333 | 3.297 | +0.035 |
| 1792-2047 | 3.337 | 3.301 | +0.037 |
| **overall** | **3.368** | **3.330** | **+0.038** |

The δ is flat across the window. Vanilla is uniformly ~0.035 to 0.046 nats better at every position, and the gap at 1792-2047 (+0.037) is the same as the gap at 256-511 (+0.039). No widening, no crossover at long range. Overall +0.038 here matches the training-time val gap of +0.035 on a different token budget, which is a clean consistency check.

![Position-binned NLL and delta](position_binned_nll.png)

### What I take from this

At this scale (162M params, 2.0B tokens, 2048 context) diff-attn gives no generalization benefit, positional or averaged. On the Stage 0 → Stage 1 reversal, the honest read is that the Stage 0 −0.020 nat val win was a single-seed small-scale signal that didn't survive scale-up, which fits it being seed noise. The paper's wins are at 3B params / 1T tokens with long-context evals. This study is three orders of magnitude smaller and uses short-context val NLL. So I won't say "Diff Transformer doesn't work." I'll say it shows no advantage in this small-scale, short-context, single-seed regime, and the long-context edge the architecture is built for doesn't appear here.

Caveats: single seed, single pairing, train/val NLL only. To push further you'd want a second and third seed to bound the δ against noise, plus a bigger model with longer context and a real long-context probe (retrieval, many-shot ICL), which is where the mechanism is supposed to pay off. Both are out of scope for a single-laptop study.

Checkpoints for both final models are published; see "Model checkpoints" below.

## Stage 2 paired result

Not run. Stage 2 (4B tokens, ~14 days unattended) only made sense if Stage 1 came back positive or ambiguous enough to be worth scaling. Stage 1 came back a clean negative on generalization, so spending two more weeks of single-laptop compute to stretch a memorization-only gap wasn't worth it. The experiment ends at Stage 1.

## Apple Silicon for training: honest numbers

The project turned up some concrete numbers for transformer training on an M5 Max with MLX:

- **Per-token cost at Stage 1 (162M params, T=2048, bf16): ~0.07 ms/token at B=4-8**, so ~14k tokens/sec under an `mx.compile`-wrapped step with grad accumulation through a compiled forward+backward.
- **Sustained TFLOPS at this workload: ~1 of the ~15-20 TFLOPS bf16 peak**, so 5-10% utilization. The job is dispatch-bound, not compute-bound. macmon's GPU-% is utilization, not throughput; this workload sits ~75% util at full tok/s because of per-step host/sync overhead, so tok/s (sec/step from the metrics log) is the real scoreboard.
- **Dispatch-boundedness** is what the Phase C kernels target: each MLX op finishes in microseconds and the GPU idles between dispatches. Bigger fused kernels cut the dispatch count.

### Thermal and power throttling on a laptop chassis

Two separate non-compute throttles showed up over the multi-day Stage 1 run. Full detail in `docs/2026-05-24-thermal-empirical-notes.md`.

**Thermal.** On a closed-chassis M5 Max under the stock macOS fan curve, sustained MLX training thermal-throttles within ~10 minutes from cold start (GPU residency 90%+ down to 30-40%, sec/step 4.0 up to 8.2). The chip isn't the cap; the laptop's thermal envelope under default fans is. The mitigations stack: open the lid (the keyboard plate is a heatsink), add cross-flow from a desk fan, and force fans to 100% (Macs Fan Control). Aggressive cooling held ~70%+ residency and 4.34 sec/step across the whole run, which also closes the old "verify 8h+ sustained" question. Once cooling was adequate, no degradation crept in.

**Power delivery (new this run).** Mid-run, throughput dropped ~10% while the chip sat cool at 73°C, which ruled out thermal. The cause: the laptop was charging through a Thunderbolt dock (also driving a monitor) that delivers 100W, and macOS picked the dock as the power source even with the 140W MagSafe brick plugged in too. The Mac doesn't sum two inputs; it arbitrates to one, and the dock won. 100W is the USB-C PD ceiling (20V × 5A); the full 140W needs MagSafe or an EPR-rated cable. The giveaway was the combination: cool temps, lower GPU clocks, and a battery slowly draining while macOS reported "charging" (net draw beat supply). Pulling the dock's power input so MagSafe was the only source re-negotiated to 140W and brought throughput back. The lesson: on a laptop, a low temperature doesn't rule out throttling. Confirm the adapter with `system_profiler SPPowerDataType | grep Wattage`.

For benchmark fairness: most "MLX on Apple Silicon" throughput numbers in the wild don't say anything about cooling or power state. A burst benchmark in fresh cooling on a 140W brick can read 2× faster than a sustained job under default fans on a 100W dock. Both numbers are "true." Always state the cooling and power regime.

### Scale context

Paper-scale (1T tokens, 3B model) ran on H100 clusters at Microsoft. One M5 Max at the throughput here would need ~33,000 years for the headline 3B-1T-token run. Apple Silicon isn't a training-cluster substitute, but it can do small-scale controlled reproductions of architectural claims, which is what this did.

## Model checkpoints

Both final Stage 1 checkpoints (162M params, 2.0B tokens, seed 0, bf16, safetensors) are on Hugging Face:

- **Combined repo:** [huggingface.co/guygrigsby/diff-mlx](https://huggingface.co/guygrigsby/diff-mlx). `diff/` and `vanilla/` hold the final `latest.safetensors` plus per-run `config.json` and `metrics.jsonl`.

## What this contributes beyond the paper

- **A working MLX implementation** of Differential Attention, paper-faithful against the official reference.
- **Custom Metal kernels** for the diff-attn forward path (P1 softmax + P2 causal SDPA + v1 composition), with autograd hooks for training.
- **A paired-init protocol** (byte-identical shared weights between variants) that makes single-seed δ meaningful and survives a port across implementations.
- **A cross-stack check** showing the Stage 0 δ wasn't an MLX/Metal artifact, then a Stage 1 scale-up showing the δ didn't generalize. The honest negative is the most useful thing here.
- **Throughput, thermal, and power notes** documenting the swap cliff, the dispatch-bound regime, the laptop thermal envelope, and the dock power throttle, with real numbers for Apple Silicon training.

## Acknowledgments

- Microsoft Research's `unilm/Diff-Transformer` repo for the canonical PyTorch reference.
- Apple's MLX team for the `mx.fast.metal_kernel` API, which made the kernel work tractable.
- Supported by writing partner [Claude](https://claude.ai) (Opus 4.7, 1M-context).

## Pointers

- **Design (kernel specs in §5.1, §5.1b, §7):** `docs/2026-05-20-diffattn-mlx-reproduction-design.md`
- **Thermal + power notes:** `docs/2026-05-24-thermal-empirical-notes.md`
- **Swap-cliff finding:** `docs/2026-05-22-swap-cliff-and-scope-restore.md`
- **bf16 design:** `docs/2026-05-21-bf16-mixed-precision-design.md`
- **PyTorch port:** `pytorch_ref/` (README has Windows + macOS setup)
- **Position-binned eval:** `scripts/eval_position_binned.py`
- **Tests:** `tests/`; `pytorch_ref/tests/` on the PyTorch side
