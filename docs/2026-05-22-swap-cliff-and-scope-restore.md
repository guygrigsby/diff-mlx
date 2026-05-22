# Swap cliff: scope restored

**Date:** 2026-05-22 (later same day as the pivot)
**Status:** Active. Follows up `docs/2026-05-22-stage1-pivot-retro.md`. The pivot retro is preserved as history; the decision to descope was correct given what we knew at the time. This doc layers on top with the next-step finding that walked most of it back.

## TL;DR

The "50 days per variant for Stage 1" finding from this morning's pivot retro was a measurement of swap-thrashing, not of the M5 Max's actual ceiling. Stage 1 was running at `micro_batch=32` which pushes the working set past the 128 GB unified-memory budget; the live run was paging into swap. Dropping to `micro_batch=8` with `grad_accum=4` (effective batch unchanged at 32) plus a fix to `train_step_with_accum` to call `mx.eval` between micro-batches plus extending `mx.compile` through the accum path together restore throughput to ~14k tps. **Full Stage 1 paired is now ~4 days, Stage 2 paired ~14 days.** Paper-scale runs are back on the table; the pivot's descope is mostly reversed.

## What we found

`scripts/bench_precision.py` sweeps `micro_batch` at Stage 1 shape with bf16:

| B | tps | peak GB |
|---|---|---|
| 4 | 14,329 | 25.3 |
| 8 | 13,832 | 48.4 |
| 16 | 11,290 | 94.2 |
| 24 | 8,414 | 135.2 (over 128) |
| 32 (live) | 950 | ? |

The cliff is between B=16 and B=24. The 135 GB peak at B=24 means MLX/macOS are paging. The live run at B=32 was deep in swap, which explains the 70 sec/step and the ~50× drop from steady-state throughput. Pure-compute throughput at B=8 is ~13.8k tps; with grad accumulation that holds when memory is properly managed.

## What we fixed

Three commits today:

1. `8148f2f` (`fix: stage1/2 micro_batch=8 grad_accum=4 + eval-between-microbatches`)
   - `config.py`: `stage1.micro_batch 32 → 8`, `grad_accum 1 → 4`. Stage 2 similar.
   - `train_step_with_accum`: `mx.eval(accum_grads, total_loss)` after each micro-batch's accumulation. Without this the lazy graph held activations from all N micro-batches in flight, defeating the micro-batch memory win.

2. `b02d1f5` (`train: compile the grad_accum path too`)
   - `CompiledTrainStep.step_with_accum` reuses the compiled forward+backward in the accum loop. Previously only `grad_accum=1` (Stage 0) got the compile speedup; Stage 1/2 now do too.

Smoke at Stage 1 with all three fixes:

| | wall | tps | sec/step |
|---|---|---|---|
| Vanilla, 50 steps | 394 s | 13,315 (last) / 14,240 (avg) | 7.9 |
| Diff, 50 steps | 486 s | ~10,600 (estimated; 80% of vanilla) | 9.7 |

Both loss curves descend cleanly, no NaN, both cross 10.0 by step 49.

## Updated wall projections

| Run | Old (pre-pivot estimate) | Pivot retro estimate | **Current (post-fixes)** |
|---|---|---|---|
| Stage 1 paired (2B tokens) | ~24h (wishful) | ~50 days | **~4 days** |
| Stage 2 paired (4B tokens) | weeks | months | **~14 days** |
| Stage 2 paired × 2 seeds | hopeless | not on table | **~28 days** |

## What this means for scope

**Back on the table:**
- Full 2B-token Stage 1 paired at one seed (4 days unattended).
- Full 4B-token Stage 2 paired at one seed (2 weeks unattended).
- Stage 2 multi-seed at 2 pairs (~4 weeks; long but tolerable).

**Still off the table:**
- Stage 2 multi-seed at 4 or 6 pairs (months; the design's most ambitious column).
- Paper's full 1T-token headline 3B model (forever, on any single Apple machine; that's a multi-H100-cluster job).

**Phase C kernels stay valuable** but for a different reason. They're no longer the only path to feasibility (the batch fix did that). They're back to being the project's novel technical contribution: a working MLX implementation of Differential Attention with custom Metal kernels for the diff-attn forward path. If the kernels deliver the paper's ~3× claim, Stage 1 paired could drop further to ~1.5 days and Stage 2 paired to ~5 days.

## What the pivot retro got right (still true)

- Apple Silicon is not a training machine for hyperscale runs. Paper's 1T-token 3B model is unreachable here regardless.
- The Phase A retro's "Stage 1 ~18h" line was wishful and never measured. Our current 4-day projection is the honest number.
- The cross-stack PyTorch / H100 rental path is a real option for paper-scale ambitions. We didn't pick it because budget capped at $25-50 and the project is framed around MLX.

## What the pivot retro got wrong

- The descope of full 2B Stage 1 and all of Stage 2 was based on the 950-tps reading, which was a swap-thrashing artifact. Stage 1/2 single-seed are both feasible at the design's token budgets with the current code.
- The "kernels are the actual project now" framing overrotated. Kernels are a contribution but the science (paired δ at the design's intermediate scales) is also reachable.

## Updated plan

Active plan is still `docs/2026-05-22-phase-c-plan.md` but with these changes:

- Task 5 ("Reduced Stage 1 paired at 200M tokens") → restored to full Stage 1 paired at 2B tokens (~4 days).
- New task before P1: just kick off the Stage 1 paired run in the background. It runs while Phase C kernel development continues.
- Stage 2 paired single-seed added as an optional post-Phase-C task.

See the plan doc for the updated task list.

## Lessons

- **Run the smoke before kicking off the big run.** The pivot retro itself noted this; the lesson reapplies here. A 100-step Stage 1 smoke would have surfaced both the swap-cliff symptom (steady-state too slow) and given us the batch-size sweep we eventually needed.
- **Per-token cost is NOT constant across batch sizes when you cross the unified-memory budget.** Earlier in the bench investigation I claimed it was constant; I was looking at runs that all fit in memory. The real shape of the curve is flat-then-cliff at the swap boundary.
- **Frame measurements by working set, not by config knob.** Stage 1's `micro_batch=32` was a sensible number on a machine with more memory or with smaller activations. On this machine with these activations, it crossed a hard cliff. Computing working-set size before picking a batch size would have caught this.
