# diff-mlx

MLX implementation of the Differential Transformer (Ye et al., ICLR 2025) on Apple Silicon, with custom Metal kernels for the diff-attn forward pass. Correctness validated against the vendored PyTorch reference.

## Current direction (as of 2026-05-22, post swap-cliff finding)

Active scope: MLX implementation + custom Metal kernels (Phase C) + Stage 1 paired (2B tokens, ~4 days unattended) + Stage 2 paired single-seed (4B tokens, ~14 days unattended).

Earlier today a pivot retro descoped Stage 1/2 paper-scale runs based on a ~950 tps reading. That reading turned out to be swap-thrashing at `micro_batch=32` (working set over the 128 GB unified-memory ceiling). Dropping to `micro_batch=8 grad_accum=4` plus `mx.eval` between micro-batches plus extending `mx.compile` through the accum path restored throughput to ~14k tps. Most of the descope is reversed. The pivot retro is preserved for history.

- **Swap-cliff finding (current state):** `docs/2026-05-22-swap-cliff-and-scope-restore.md`
- **Pivot retro (historical, partially superseded):** `docs/2026-05-22-stage1-pivot-retro.md`
- **Active plan:** `docs/2026-05-22-phase-c-plan.md`
- **Original design (kernel specs still authoritative):** `docs/2026-05-20-diffattn-mlx-reproduction-design.md`

## What's already done

- MLX implementation of vanilla MHA + Differential Attention.
- Cross-check vs vendored PyTorch reference (`tests/test_diff_reference.py`): 3.58e-7 max diff on CPU stream.
- Paired-seed init protocol (design §9.7).
- bf16 mixed precision via `LinearAMP`. Design: `docs/2026-05-21-bf16-mixed-precision-design.md`.
- Optimizer-state checkpoints, auto-resume, grad_accum (with compile), caffeinate wrappers.
- Stage 0 paired δ: diff beats vanilla by 0.020 nats post-crossover at step 3000.
- Throughput investigation + fixes: B=8 grad_accum=4 + compiled accum gives ~14k tps Stage 1.
- 104 tests passing.

## What's next

Phase C kernels (P1 softmax, P2 causal SDPA, v1 diff composition) in foreground while Stage 1 paired runs in background, then Stage 2 paired single-seed, then writeup. See the plan.

## Active docs

- `docs/2026-05-22-swap-cliff-and-scope-restore.md` (read this first for current state)
- `docs/2026-05-22-phase-c-plan.md` (active plan)
- `docs/2026-05-22-stage1-pivot-retro.md` (historical pivot decision, partially walked back)
- `docs/2026-05-20-diffattn-mlx-reproduction-design.md` (kernel specs in §5.1, §5.1b, §7 are load-bearing)
- `docs/2026-05-21-bf16-mixed-precision-design.md`
- `docs/2026-05-20-phase-a-retro.md`, `docs/2026-05-20-phase-b-retro.md` (historical context)

## Archive

Superseded plans and one-off scripts at `docs/archive/` and `scripts/archive/`. See the README in each for what's there and why.

## Active scripts

- `scripts/stage0_paired.{py,sh}`, `scripts/stage0_vanilla.sh`: Stage 0 paired and vanilla runners.
- `scripts/stage1_paired.{py,sh}`, `scripts/stage1_smoke.{py,sh}`: Stage 1 runners.
- `scripts/bench_step.py`, `scripts/bench_compile.py`, `scripts/bench_precision.py`: throughput diagnostics.
- `scripts/diagnose_throughput.py`: long-form throughput investigation tool.
- `scripts/generate_ref_fixture.py`: regenerates the PyTorch reference fixture.
- `scripts/stage0_dryrun.py`, `scripts/stage0_dryrun_with_eval.py`, `scripts/stage0_paired_dryrun.py`: dry-run validators used during Phase A/B.
