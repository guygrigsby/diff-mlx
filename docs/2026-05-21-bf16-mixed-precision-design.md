# bf16 mixed precision: implementation design

**Date:** 2026-05-21
**Status:** Drafted; awaiting review.
**Scope:** Phase D prerequisite. Implements design §9.0 (the spec; this doc is the implementation choice on top of it).
**Companion doc:** `docs/2026-05-20-diffattn-mlx-reproduction-design.md` §9.0 (the authoritative precision spec).

## Why

Stage 0 diff held a 32 GB MLX peak on a 256-dim 6-layer model. Stage 1 (768 dim, 12 layers, B=32, T=2048) and Stage 2 (1024 dim, 16 layers) will exceed unified-memory budget without halving activation storage. bf16 forward activations halve the dominant cost. Loss-curve cost: design §9.0 tolerances apply, validated by paired δ continuity.

## What's already in place

- Logits cast to fp32 before CE softmax (`train_step.py:9`).
- Grad-norm accumulation in fp32 (`train_step.py:35`).
- RMSNorm internal fp32 with `astype(in_dtype)` on output (`model.py:23-27`).
- Lambda math fully fp32, cast to output dtype at apply site (`model.py:128-160`).
- `to_bf16_view` / `to_bf16_dict` helpers (`optim.py:51-71`); not currently wired in.

Roughly: every numerically-sensitive op is already protected. The remaining work is switching the bulk forward path to bf16.

## Implementation choice (resolves design §9.0 option a vs b)

**Option A.** fp32 params, cast to bf16 inside forward at op boundaries. Rationale:

- AdamW operates on fp32 master directly; no optimizer wrapper.
- Smallest diff against current code.
- Activations halve (the dominant cost at Stage 1/2).
- Param + optimizer-state storage stays fp32 (~12 B/param). Acceptable: at Stage 2 ~305M params × 12 B = 3.7 GB, vs activations dominating tens of GB.

Option B (bf16 storage + parallel fp32 master) deferred. Revisit only if Stage 2 param+optimizer footprint becomes load-bearing.

## Cast boundaries

Concrete points where dtype transitions happen. `amp_dtype` is the configured forward dtype (default `mx.float32`, set to `mx.bfloat16` for AMP).

| Location | Behavior |
|---|---|
| Token embedding output | `.astype(amp_dtype)` at exit |
| `nn.Linear` (q, k, v, o, mlp gate/up/down, lm_head) | weight + bias cast to `amp_dtype` inside the call; matmul runs in `amp_dtype` |
| RoPE | inputs cast to `amp_dtype` if not already; `mx.fast.rope` handles internal precision |
| SDPA | Q/K/V passed in `amp_dtype`; output in `amp_dtype` |
| RMSNorm | unchanged. Already does fp32 internal and returns input dtype |
| Lambda math (diff-attn) | unchanged. fp32 internal, cast to output dtype at apply (`out1 - lam.astype(out1.dtype) * out2`) |
| `lm_head` output → CE loss | `_ce_loss` already casts to fp32 (unchanged) |
| Grad-norm clip | already fp32-accumulating (unchanged) |

The `nn.Linear` cast is the only structural change. Two ways to do it:

1. **`LinearAMP` module subclass** of `mlx.nn.Linear` overriding `__call__` to cast weight/bias to a configured dtype. Drop-in replacement.
2. **Functional helper** `linear_amp(linear, x, dtype)` called at every Linear use site.

Plan picks **#1 (LinearAMP).** Single point of change in module construction; call sites unchanged. The Phase A retro can mark the §9.0 item closed without touching forward bodies in Block/MHA/DiffAttention/MLP.

## Configuration

- `ModelConfig` gains `amp_dtype: str = "float32"` (string for JSON-serializability). Mapped to `mx.bfloat16` or `mx.float32` at model construction.
- `ModelConfig.stage0()` defaults to `"float32"` (no behavior change for existing Stage 0 reproducibility).
- `ModelConfig.stage1()` and `.stage2()` default to `"bfloat16"`.

## Init compatibility

Paired-init protocol (`paired_init.build_paired_models`) must remain byte-identical on the shared backbone. Since params stay fp32 in storage, the protocol is unaffected. Verify the existing paired-init test still passes with `amp_dtype="bfloat16"`.

## Acceptance criteria

1. **Numerical sanity:** at Stage 0 shapes, one fp32 step vs one bf16 step on the same paired init: loss within `1e-2` absolute (design §9.0 tolerance) and no NaN/Inf in grads.
2. **PyTorch cross-check still passes** at the design's existing tolerances (3.58e-7 fp32, 1.7e-3 bf16) when invoked with `amp_dtype="bfloat16"`.
3. **Memory:** Stage 0 diff `mx.metal.get_peak_memory()` drops from current ~32 GB to roughly half (sanity check; not a strict threshold since allocator behavior varies).
4. **Paired-init test passes** with `amp_dtype="bfloat16"` (byte-identical fp32 storage; bf16 only at forward boundaries).
5. **All existing tests pass** (78/78 from Phase B).

## Out of scope

- Option B (bf16 storage). Defer.
- Loss scaling. Not needed for bf16 (bf16's fp32-equivalent exponent range avoids underflow that fp16 mixed precision needs to handle).
- AMP for the optimizer step itself. AdamW stays fp32 throughout.
- bf16 in checkpoints. Save as fp32 (matches current); no compatibility break.

## Risks

- **MLX `nn.Linear` subclass behavior under `value_and_grad`.** Need to verify the cast inside `__call__` flows grads back to the fp32 weight via the implicit graph cast. Expected fine, but plan should include a tiny grad-flow test before doing the big switch.
- **MLX broadcast rules for mixed bf16/fp32 ops** in places where AMP-converted intermediates meet fp32 buffers (e.g., lambda math producing fp32, multiplied against bf16 attention output). The cast at the multiply (`lam.astype(out1.dtype)`) is already there; verify no other mixed-dtype op silently upcasts and erases the memory win.

## Validation order

1. Build LinearAMP, verify forward + backward identity at `amp_dtype="float32"` (no behavior change).
2. Switch to `amp_dtype="bfloat16"` on a single layer, eval one step, check no NaN.
3. Full Stage 0 short-run (e.g., 200 steps) in bf16, compare paired δ trajectory against the fp32 reference (`runs/stage0-paired-caffeinated/`).
4. If clean, ship.
