# Stage 1 reality check: pivot to kernels-and-correctness

**Date:** 2026-05-22
**Status:** Decided. Replaces the paper-scale Stage 1/2 ambition. New plan at `docs/2026-05-22-phase-c-plan.md`.

## What happened

Kicked off paired Stage 1 at 00:52 on the new stack from yesterday's Phase D prereqs work (bf16, opt-state checkpoints, grad_accum, auto-resume, caffeinate). Run was clean: loss descending from 12.0, no NaN, machine steady. Step time stabilized at ~70 sec/step (950 tps).

That's catastrophically slower than expected. At 950 tps:
- Stage 1 paired (2B tokens): ~50 days
- Stage 2 paired (4B tokens × 1 seed): months
- Multi-seed Stage 2: don't bother

Killed at step 386. 7.5 hours burned, no checkpoint landed (`save_every=1000`).

## The arithmetic

Bench at B=4 and B=8 (no live-run interference) showed:

- Per-token cost is constant at ~1 ms/token across batch sizes. The "70 sec/step" is just 65k tokens at the hardware's actual ceiling.
- GPU utilization is 5-7% (0.3 TFLOPS sustained vs ~5-10 TFLOPS realistic ceiling for transformer matmul on M5 Max bf16).
- Display-sleep, memory pressure, ckpt overhead, allocator thrashing all ruled out. This is MLX bf16 on M5 Max for this workload, full stop.
- `mx.compile` measured at 1.41× at B=4. Real lever but won't close the gap.
- Phase C custom kernels claim ~3× per paper. Combined ~4× still leaves Stage 1 paired at ~12 days, Stage 2 in months.

## Hardware comparison, since it matters

Paper used Nvidia H100-80GB. Headline 3B model: 1T tokens at batch size 4M tokens. That's ~64-128 H100s minimum, multi-node. Single-chip throughput on H100 for a 162M model is ~50,000 tps bf16. M5 Max measured at 950 tps on the same workload. The chip-for-chip gap is **~50×**. Marketing conflates "AI" with "inference"; training is where the gap shows up.

## What got considered

| Path | Wall | Cost |
|---|---|---|
| M5 reduced runs post-Phase-C | ~30 days | $0 |
| H100 rental, full paper-scale | ~12 days | $400-700 |
| Hybrid (kernels on M5, runs on H100) | ~15-20 days | ~$400 |

Budget ceiling is $25-50. Rental paths don't work as primary.

## What got picked

**Reframe.** Contribution shifts from "paper-scale reproduction on Apple Silicon" (arithmetic forbids) to **"MLX implementation of Differential Transformer with custom Metal kernels, correctness-validated against PyTorch reference, directional δ replication at Stage 0 and reduced Stage 1."** Paper's science is established. Novel piece here is the Metal port and the kernels.

## Deliverables under the new framing

Done:
- MLX implementation of vanilla MHA + Differential Attention (Phase A, B).
- Cross-check vs vendored PyTorch reference: max |diff| = 3.58e-7 on CPU stream, 1.7e-3 on Metal stream.
- Paired-seed init protocol (design §9.7), byte-identical on shared params.
- bf16 mixed precision (LinearAMP, option A from `docs/2026-05-21-bf16-mixed-precision-design.md`).
- Optimizer-state checkpoints, auto-resume, grad_accum, multi-seed orchestrator, caffeinate wrappers.
- Stage 0 paired δ: −0.0201 at step 5000, monotonic post-crossover at step ~3000.
- 102 tests, all green.

To ship:
- Phase C custom Metal kernels: P1 softmax preflight, P2 causal SDPA preflight, v1 diff composition.
- Kernel correctness gates (v1 vs v0 numerical agreement) and speed eval vs `mx.fast.scaled_dot_product_attention`.
- Reduced Stage 1 paired at 200M tokens with the kernels in place. Extends δ checkpoint to the larger 162M-param model.
- Writeup.

Optional, if budget permits:
- Single H100 cross-stack validation run (~10h, ~$15): port `model.py` to PyTorch, run reduced Stage 1, compare δ to MLX. Confirms training dynamics match between stacks. Strictly belt-and-suspenders since the cross-check fixture already validates correctness.

## Descoped

- Full 2B-token Stage 1.
- All Stage 2 work (305M params, 4B tokens).
- Multi-seed scaling studies. `scripts/multi_seed_paired.sh` stays as infrastructure but doesn't get exercised.
- "Matches paper's exact loss numbers" framing. Directional δ at increased scale is the claim now.
- The "fused v2 stretch kernel" mentioned in the design doc.

## Honest assessment

Design's "2-3 weeks wall time" budget was wishful. Phase A retro's "Stage 1 ~18h" was never measured. My "1 sec/step" projection yesterday was the same error compounded. Should have run the smoke before kicking off the big run. The 7.5h burn was avoidable.

What we did right: auto-resume + caffeinate + opt-state ckpt means future Stage 1 attempts after Phase C lands will just need `./scripts/stage1_paired.sh` and resume from any crash. Infrastructure is solid; the only thing that was wrong was the wall-time projection.

The kernels are the actual project now. Phase C is no longer optional, and Phase D is descoped.

## Pointers

- New plan: `docs/2026-05-22-phase-c-plan.md`
- Bench evidence: `scripts/bench_step.py`, `scripts/bench_compile.py`
- Original design (kernel specs still load-bearing): `docs/2026-05-20-diffattn-mlx-reproduction-design.md` §7, §11
- Existing Phase D infra (not getting exercised): `scripts/multi_seed_paired.sh`, `scripts/stage1_paired.sh`, auto-resume in `train_run`
