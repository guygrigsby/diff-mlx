# Phase C plan: custom Metal kernels + Stage 1 paired + Stage 2

**Date:** 2026-05-22 (revised same day after the swap-cliff finding)
**Status:** Active.
**Companion specs:** `docs/2026-05-20-diffattn-mlx-reproduction-design.md` §5.1, §5.1b, §7 (kernel design, correctness gates, time-boxes).
**Context:** This plan was first written under the 2026-05-22 pivot retro (`docs/2026-05-22-stage1-pivot-retro.md`) when Stage 1 looked unreachable. The follow-up swap-cliff finding (`docs/2026-05-22-swap-cliff-and-scope-restore.md`) restored most of the scope. The plan was updated to match.

## Goal

Build P1 (softmax) and P2 (causal SDPA) Metal kernels via MLX's custom-kernel layer. Compose into a v1 diff-attn forward (two P2 calls + Python subtract, per design §7.1). Verify correctness against the existing PyTorch reference fixture. Eval speed against `mx.fast.scaled_dot_product_attention`. Run full Stage 1 paired at 2B tokens, and Stage 2 paired single-seed at 4B tokens, with the resulting stack. Multi-seed Stage 2 is optional and budget-dependent.

## Non-goals

- Stage 2 paired at 4+ seeds. The Phase D multi-seed orchestrator stays as infrastructure but isn't load-bearing for the writeup. 2 seeds at Stage 2 is the realistic ceiling.
- Fused v2 kernel. Out of scope.
- Beating `mx.fast.scaled_dot_product_attention` on speed. Design §7.5 lists this as a nice-to-have, not gating. Hitting parity is enough.

## Tasks

### Task 0: throughput-enabling work (DONE)

This task captures three commits landed earlier today that together restored Stage 1/2 feasibility. Listed here for the record; no work remaining.

- `13d34c7` (`train: wrap train_step in mx.compile`): `CompiledTrainStep` class; 1.41× measured at B=4.
- `8148f2f` (`fix: stage1/2 micro_batch=8 grad_accum=4 + eval-between-microbatches`): config change to keep working set under the 128 GB unified-memory cliff; `mx.eval` inside accum loop so memory stays at one-microbatch footprint.
- `b02d1f5` (`train: compile the grad_accum path too`): `CompiledTrainStep.step_with_accum`. Stage 1/2 now get the compile speedup; previously only `grad_accum=1` (Stage 0) did.

Net: Stage 1 vanilla throughput from ~950 tps (swap-thrashing live run) to ~14,000 tps. Full bench/finding writeup in `docs/2026-05-22-swap-cliff-and-scope-restore.md`.

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

### Task 5: Stage 1 paired (~4 days unattended)

**Files:**
- Reuse: `scripts/stage1_paired.py`, `scripts/stage1_paired.sh` (already built).
- Possibly modify: `scripts/stage1_paired.py` to accept a `--kernel_version` flag if v1 is ready.

**Plan:**
1. Run vanilla + diff at one seed at the design's 2B-token Stage 1 budget under caffeinate. Use `kernel_version="v1"` if Task 4 showed v1 faster (else v0).
2. Auto-resume catches a crash; infra already in place.
3. Can be kicked off in the background while Task 1-4 (kernels) continues in the foreground. The training is GPU-bound; kernel-dev work that doesn't itself use the GPU heavily can proceed in parallel.
4. Compare paired δ trajectory to Stage 0's. Expect monotonic post-crossover δ in the diff favor, larger absolute gap than Stage 0 if the paper's scaling claim holds.
5. Update Stage 1 NOTES with results.

**Acceptance:** clean run (no NaN), paired δ qualitatively in the diff variant's favor by end of run.

### Task 6: Stage 2 paired single-seed (~14 days unattended)

**Optional but high-value.** Demonstrates δ at the design's largest model (305M params).

**Plan:**
1. After Stage 1 lands, kick off Stage 2 paired at one seed. ~14 days unattended.
2. Same caffeinate + auto-resume pattern.
3. Use v1 kernels if available.

**Acceptance:** clean run, paired δ recorded.

### Task 7: writeup (~1 day)

**Files:**
- Create: `docs/2026-05-22-final-writeup.md`.

**Plan:**
1. What was built (MLX impl, Metal kernels, paired-init protocol, throughput investigation, Stage 0/1/2 paired δ results).
2. Cross-check evidence (PyTorch reference fixture, v0/v1 numerical agreement).
3. Kernel speed numbers.
4. The swap-cliff finding and what it taught about Apple Silicon training.
5. Stage 0 + Stage 1 + (optional) Stage 2 paired δ curves.
6. What this contributes that the paper doesn't.

## Total time estimate

| Task | Days | Notes |
|---|---|---|
| 0: throughput work (done) | 0 | landed 2026-05-22 |
| 1: P1 softmax | 1-3 | foreground |
| 2: P2 SDPA | 2-4 | foreground |
| 3: v1 composition | 1-2 | foreground |
| 4: kernel speed eval | 1 | foreground |
| 5: Stage 1 paired (unattended) | 4 | parallel with 1-4 |
| 6: Stage 2 paired single-seed (optional, unattended) | 14 | parallel with 7 |
| 7: writeup | 1 | sequential |
| **Total foreground wall** | **~6-11 days** | Stage 1 finishes during the kernel work |
| **Total incl. Stage 2** | **~20-25 days** | Mostly unattended waiting |

## Risks

- **P1 stalls** (1-3 day timebox). If MLX's custom-kernel layer turns out to be miserable on M5, design §5.1 says drop v1/v2. Project still has a coherent story: MLX impl, throughput investigation, Stage 0 + Stage 1 + (maybe) Stage 2 paired δ, all on v0.
- **P2 forward correctness fails at Stage 1 shapes.** v1 ships as a correctness artifact only; Stage 1/2 runs use v0.
- **v1 slower than v0.** Per design §7.5 don't need to beat `mx.fast.scaled_dot_product_attention`. If v1 ends up materially slower, use v0 for the runs and document v1 as correctness-only.
- **Stage 1 paired run derails.** Auto-resume catches crashes; the 4-day unattended cost is amortized across a few resumes if needed. The smoke at 50 steps already showed clean loss descent.

## Out of scope

- Stage 2 multi-seed at 4+ seeds (the design's most ambitious column). Two seeds at Stage 2 is the budget ceiling.
- Fused v2 kernel.
- Cloud rental (deferred nice-to-have at ~$15 if budget appears later).
