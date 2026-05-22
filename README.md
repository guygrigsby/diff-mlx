# diff-mlx

MLX implementation of the Differential Transformer (Ye et al., ICLR 2025) on Apple Silicon, with custom Metal kernels for the diff-attn forward pass. Correctness validated against the vendored PyTorch reference.

## Current direction (as of 2026-05-22)

Scope pivoted from "paper-scale reproduction" to "MLX port + custom Metal kernels + reduced Stage 1." Stage 1 throughput measurement showed paper-scale runs would take months on M5 Max; the paper's training ran on H100 clusters. The contribution shifts to the MLX implementation, the custom Metal kernel work, and a reduced Stage 1 result that extends the Stage 0 paired δ replication.

- **Pivot retro (read this first):** `docs/2026-05-22-stage1-pivot-retro.md`
- **Active plan:** `docs/2026-05-22-phase-c-plan.md`
- **Original design (kernel specs still authoritative):** `docs/2026-05-20-diffattn-mlx-reproduction-design.md`

## What's already done

- MLX implementation of vanilla MHA + Differential Attention (Phase A, B).
- Cross-check vs vendored PyTorch reference (`tests/test_diff_reference.py`): 3.58e-7 max diff on CPU stream.
- Paired-seed init protocol (design §9.7).
- bf16 mixed precision via `LinearAMP` (`docs/2026-05-21-bf16-mixed-precision-design.md`).
- Optimizer-state checkpoints, auto-resume, grad_accum, multi-seed orchestrator, caffeinate wrappers.
- Stage 0 paired δ: diff beats vanilla by 0.020 nats post-crossover at step 3000.
- 102 tests passing.

## What's next

Phase C kernels (P1 softmax, P2 causal SDPA, v1 diff composition) and reduced Stage 1 paired at 200M tokens. See the plan.

## Historical docs

- Phase A retro: `docs/2026-05-20-phase-a-retro.md`
- Phase B retro: `docs/2026-05-20-phase-b-retro.md`
- Phase A implementation plan: `docs/2026-05-20-diffattn-mlx-implementation-plan-phase-a.md`
- Phase B implementation plan: `docs/2026-05-20-diffattn-mlx-implementation-plan-phase-b.md`
- bf16 mixed precision design + plan: `docs/2026-05-21-bf16-mixed-precision-{design,implementation-plan}.md`
- Phase D prereqs plan (3-5): `docs/2026-05-21-phase-d-prereqs-3to5-plan.md` (executed; Phase D scaling runs themselves descoped per the pivot)
