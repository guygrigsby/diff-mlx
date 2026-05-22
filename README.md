# diff-mlx

MLX implementation of the Differential Transformer (Ye et al., ICLR 2025) on Apple Silicon, with custom Metal kernels for the diff-attn forward pass. Correctness validated against the vendored PyTorch reference.

## Current direction (as of 2026-05-22)

Scope pivoted from "paper-scale reproduction" to "MLX port + custom Metal kernels + reduced Stage 1." Stage 1 throughput measurement showed paper-scale runs would take months on M5 Max; the paper's training ran on H100 clusters. The contribution shifts to the MLX implementation, the custom Metal kernel work, and a reduced Stage 1 result that extends the Stage 0 paired δ replication.

- **Pivot retro (read this first):** `docs/2026-05-22-stage1-pivot-retro.md`
- **Active plan:** `docs/2026-05-22-phase-c-plan.md`
- **Original design (kernel specs still authoritative):** `docs/2026-05-20-diffattn-mlx-reproduction-design.md`

## What's already done

- MLX implementation of vanilla MHA + Differential Attention.
- Cross-check vs vendored PyTorch reference (`tests/test_diff_reference.py`): 3.58e-7 max diff on CPU stream.
- Paired-seed init protocol (design §9.7).
- bf16 mixed precision via `LinearAMP`. Design: `docs/2026-05-21-bf16-mixed-precision-design.md`.
- Optimizer-state checkpoints, auto-resume, grad_accum, caffeinate wrappers.
- Stage 0 paired δ: diff beats vanilla by 0.020 nats post-crossover at step 3000.
- 102 tests passing.

## What's next

Phase C kernels (P1 softmax, P2 causal SDPA, v1 diff composition) and reduced Stage 1 paired at 200M tokens. See the plan.

## Active docs

- `docs/2026-05-22-stage1-pivot-retro.md`
- `docs/2026-05-22-phase-c-plan.md`
- `docs/2026-05-20-diffattn-mlx-reproduction-design.md` (kernel specs in §5.1, §5.1b, §7 are load-bearing)
- `docs/2026-05-21-bf16-mixed-precision-design.md`
- `docs/2026-05-20-phase-a-retro.md`, `docs/2026-05-20-phase-b-retro.md` (historical context)

## Archive

Superseded plans and one-off scripts at `docs/archive/` and `scripts/archive/`. See the README in each for what's there and why.

## Active scripts

- `scripts/stage0_paired.{py,sh}`, `scripts/stage0_vanilla.sh`: Stage 0 paired and vanilla runners.
- `scripts/stage1_paired.{py,sh}`, `scripts/stage1_smoke.{py,sh}`: Stage 1 runners. The reduced 200M-token run uses these with overridden `total_tokens`.
- `scripts/bench_step.py`, `scripts/bench_compile.py`: throughput diagnostics.
- `scripts/diagnose_throughput.py`: long-form throughput investigation tool.
- `scripts/generate_ref_fixture.py`: regenerates the PyTorch reference fixture.
- `scripts/stage0_dryrun.py`, `scripts/stage0_dryrun_with_eval.py`, `scripts/stage0_paired_dryrun.py`: dry-run validators used during Phase A/B.
