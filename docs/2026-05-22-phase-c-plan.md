# Phase C plan: custom Metal kernels + reduced Stage 1

**Date:** 2026-05-22
**Status:** Active. Replaces the Phase D plan after the 2026-05-22 pivot (see `docs/2026-05-22-stage1-pivot-retro.md`).
**Companion specs:** `docs/2026-05-20-diffattn-mlx-reproduction-design.md` §5.1, §5.1b, §7 (kernel design, correctness gates, time-boxes).

## Goal

Build P1 (softmax) and P2 (causal SDPA) Metal kernels via MLX's custom-kernel layer. Compose into a v1 diff-attn forward (two P2 calls + Python subtract, per design §7.1). Verify correctness against the existing PyTorch reference fixture. Eval speed against `mx.fast.scaled_dot_product_attention`. Run a reduced Stage 1 paired (200M tokens) using the resulting stack to extend the paired δ result to the larger model.

## Non-goals

- Full 2B-token Stage 1 or any Stage 2. Descoped (see retro).
- Fused v2 kernel. Out of scope.
- Multi-seed studies. Infrastructure stays; not exercised.
- Beating `mx.fast.scaled_dot_product_attention` on speed. Design §7.5 lists this as a nice-to-have, not gating. Hitting parity is enough.

## Tasks

### Task 0: `mx.compile` in `train_step`  (~0.5 day)

**Why:** Measured 1.41× speedup at B=4 (`scripts/bench_compile.py`). Free lever, applies regardless of Phase C outcome.

**Files:** `train_step.py`, `tests/test_train_step.py`.

**Plan:**
1. Switch `train_step` to use the canonical MLX compile idiom: `state = [model.state, optimizer.state]`, `mx.compile(inner_step, inputs=state, outputs=state)`.
2. The grad-norm walk and grad-clip can't easily live inside compile (pytree traversal not traceable); keep them outside the compiled core, fold them into the outer wrapper.
3. Add a test that verifies bit-close agreement between compiled and uncompiled `train_step` on a Stage 0-sized config after one step.
4. Update `train_step_with_accum` similarly.

**Acceptance:** existing 102 tests still pass. Stage 0 paired run completes with the same loss curve as before within 1e-4.

### Task 1: P1 softmax kernel  (~1-3 days)

**Spec:** design §5.1. Softmax over last dim, max-subtraction trick, both fp32 and bf16 paths.

**Files:**
- Create: `kernels/softmax_p1.metal` (MSL source).
- Create: `kernels/softmax_p1.py` (`mx.fast.metal_kernel` wrapper + `mx.custom_function` autograd hook).
- Test: `tests/test_softmax_p1.py`.

**Plan:**
1. Write the MSL kernel: one block per row, two reductions (max, sum), one element-wise pass.
2. Wrap via `mx.fast.metal_kernel`. Add an `mx.custom_function` backward that does the softmax Jacobian via pure MLX (no custom backward kernel for P1).
3. Test forward output matches `mx.softmax(x, axis=-1)` within 1e-4 fp32 / 1e-2 bf16 (design §5.1 pass criterion #1).
4. Test backward matches autograd of pure-MLX softmax (criterion #2).
5. Finite-difference gradient check on `(2, 4, 8)` (criterion #3).

**Gate (design §5.1):** if not over the hump by day 3, abandon v1/v2 and ship v0-only. Document the abandonment in the retro and skip to Task 5.

### Task 2: P2 causal SDPA kernel  (~2-4 days)

**Spec:** design §5.1b. Single-map causal SDPA with bf16 I/O, fp32 accumulation, separate `head_dim_qk` and `head_dim_v` parameters (so v1 can pass V at width `2D`).

**Files:**
- Create: `kernels/sdpa_p2.metal`.
- Create: `kernels/sdpa_p2.py`.
- Test: `tests/test_sdpa_p2.py`.

**Plan:**
1. Write the MSL kernel: tiled QK matmul, causal mask, row softmax (uses the P1 softmax pattern), AV accumulation. Online (FlashAttention-style) accumulation to keep transient memory bounded.
2. Wrap via `mx.fast.metal_kernel`. Backward via `mx.custom_function` using pure-MLX SDPA gradient (no backward kernel).
3. Test forward against `mx.fast.scaled_dot_product_attention` within 1e-2 (bf16) on the design's full shape matrix (§5.1b criterion #1):
   - Toy: `(B=2, H=2, T=128, D=32)`
   - Stage 0 vanilla: `(B=16, H=4, T=1024, D=64)`
   - Stage 0 diff sub-head: `(B=16, H=2, T=1024, D=64)`
   - Stage 1 vanilla: `(B=32, H=12, T=2048, D=64)`
   - Stage 1 diff sub-head: `(B=32, H=6, T=2048, D=64)`
   - Stage 0 diff full: `(B=16, H=2, T=1024, D_qk=64, D_v=128)`
   - Stage 1 diff full: `(B=32, H=6, T=2048, D_qk=64, D_v=128)`
4. Backward correctness on a 4-layer / 2-head / D=32 toy (criterion #2).
5. Peak transient memory measured at the largest shape (criterion #3); record in the test output.
6. Speed at least as fast as a pure-MLX SDPA composed of `mx.softmax` + `mx.matmul` (criterion #4).

**Gate:** pass → v1 is on the table. Fail → v1/v2 ship as correctness artifacts only, training uses v0.

### Task 3: v1 diff composition  (~1-2 days)

**Spec:** design §7.1, §7.2 (the v1 row). Two P2 calls + Python subtract; `lambda` and `subln` from existing pure-MLX code.

**Files:**
- Modify: `model.py` (add a `diff_kernel_version` flag on `DiffAttention`; switch SDPA call sites to P2 when flag set).
- Test: `tests/test_diff_v1.py` (cross-check vs v0 within numerical noise; cross-check vs PyTorch fixture at design §7.4 tolerance).

**Plan:**
1. Add `DiffAttention(..., kernel_version="v0"|"v1")`. Default stays `"v0"`.
2. When `kernel_version="v1"`, the two SDPA calls in `__call__` go through `sdpa_p2.py` instead of `mx.fast.scaled_dot_product_attention`.
3. Correctness test: v1 vs v0 forward agree within 1e-3 fp32 / 1e-2 bf16 on Stage 1 diff shapes. Both must match the PyTorch reference fixture at the same tolerances.
4. No backward kernel; the autograd hook from Task 2 handles it.

### Task 4: kernel speed eval  (~1 day)

**Files:**
- Create: `scripts/bench_kernels.py`.

**Plan:**
1. Time `mx.fast.scaled_dot_product_attention`, the P2 kernel, and the pure-MLX SDPA composition at Stage 0 and Stage 1 shapes.
2. Time a full forward+backward of `DiffAttention` at `kernel_version="v0"` and `kernel_version="v1"` at Stage 1 shapes.
3. Record results in `docs/2026-05-22-kernel-speed-eval.md` with the actual numbers (paper claims ~3× on the diff variant; report what we got).

This is the empirical answer to "did the kernels actually help." If v1 is meaningfully faster than v0, use v1 for the reduced Stage 1 run. If not, run on v0.

### Task 5: reduced Stage 1 paired (~2 days unattended)

**Files:**
- Reuse: `scripts/stage1_paired.py`, `scripts/stage1_paired.sh` (already built).
- Possibly modify: `scripts/stage1_paired.py` to default `total_tokens=200_000_000` and accept a `--kernel_version` flag.

**Plan:**
1. Override `total_tokens` to 200M (from the 2B Stage 1 default).
2. Run vanilla + diff at one seed under caffeinate, with `mx.compile` enabled and `kernel_version="v1"` if Task 4 showed v1 faster (else v0).
3. Auto-resume catches a crash; we already have the infra.
4. Compare paired δ trajectory to Stage 0's. Expect monotonic post-crossover δ in the diff favor, larger absolute gap than Stage 0 if the paper's scaling claim holds at this scale.
5. Update Stage 1 NOTES with results.

**Acceptance:** clean run (no NaN), paired δ qualitatively in the diff variant's favor by end of run.

### Task 6: writeup  (~0.5-1 day)

**Files:**
- Create: `docs/2026-05-22-final-writeup.md`.

**Plan:**
1. What was built (MLX impl, Metal kernels, paired-init protocol, reduced Stage 1 result).
2. Cross-check evidence (PyTorch reference fixture, v0/v1 numerical agreement).
3. Kernel speed numbers.
4. Stage 0 + reduced Stage 1 paired δ curves.
5. What this contributes that the paper doesn't.

## Total time estimate

| Task | Days |
|---|---|
| 0: mx.compile | 0.5 |
| 1: P1 softmax | 1-3 |
| 2: P2 SDPA | 2-4 |
| 3: v1 composition | 1-2 |
| 4: kernel speed eval | 1 |
| 5: reduced Stage 1 (unattended) | 2 |
| 6: writeup | 0.5-1 |
| **Total** | **~8-14 days** |

Wall could be shorter if Task 5 overlaps with Task 6.

## Risks

- **P1 stalls** (1-3 day timebox). If MLX's custom-kernel layer turns out to be miserable to work with on M5, design §5.1 says drop v1/v2. Then Phase C ships as a partial deliverable (no kernel, but the reduced Stage 1 result with mx.compile still goes through). Project still has a coherent story.
- **P2 forward correctness fails at Stage 1 shapes**. v1 ships as a correctness artifact only; reduced Stage 1 runs on v0. mx.compile alone gives ~1.4×; reduced Stage 1 still completes in ~5 days unattended at that throughput.
- **v1 is slower than v0**. Per design §7.5 we don't need to beat `mx.fast.scaled_dot_product_attention`; just match it on bf16. If v1 ends up materially slower, use v0 for the Stage 1 run and document v1 as a correctness-only artifact.

## Out of scope (descoped from earlier plans)

- Full 2B Stage 1, all Stage 2 work.
- Multi-seed orchestration runs.
- Fused v2 kernel.
- Cloud rental (optional H100 cross-check stays as a deferred nice-to-have at ~$15 if budget appears later).
