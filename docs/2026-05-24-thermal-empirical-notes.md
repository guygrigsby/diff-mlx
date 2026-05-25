# Thermal vs throughput on M5 Max for sustained MLX training

**Date:** 2026-05-24 (active, updated as findings accumulate)
**Context:** Stage 1 paired run on M5 Max, 162M-param transformer, bf16 forward via `LinearAMP`, `mx.compile`-wrapped grad-accum training step. Effective batch 32 (`micro_batch=8 grad_accum=4`). Block size 2048.

## TL;DR

On a closed-chassis M5 Max laptop with stock macOS fan curve, sustained MLX training thermal-throttles within ~10 minutes from a cold start. Cold-start GPU utilization hits 90%+ then decays to ~30-40% sustained. The chip is not the cap; **the laptop's thermal envelope under default fan policy is the cap**. Active cooling progressively recovers the gap.

## Throughput vs cooling configuration

All measurements from the same Stage 1 vanilla training process, MLX 0.31.2, M5 Max 40-core GPU, 128 GB unified RAM, ambient ~22°C, lid closed on stand unless noted.

| Cooling configuration | GPU residency (macmon) | Sec/step | Tokens/sec |
|---|---|---|---|
| Cold start, stock fans | 90%+ | ~4.0 | ~16,400 |
| Default config, sustained (~10 min in) | 30-40% | 8.21 | 7,981 |
| Open lid + small desk fan blowing across | 50-60% | 5.54 | 11,800 |
| Open lid + desk fan + Macs Fan Control @ 100% | 70%+ | **4.34** | **15,083** |

## Observations

**1. Default fan curve is conservative.** macOS prefers acoustic comfort over performance. The chip will let itself sit at 90°C+ for minutes before the fans even spin up audibly. By the time fans engage in earnest, the chip has already throttled. Forcing 100% fans pre-emptively prevents the throttle from ever firing.

**2. Surface area matters.** Closing the lid in a vertical stand reduces effective dissipation area by ~50% (only the bottom is exposed). Opening the lid exposes the keyboard plate, which on M-series Macs is a primary heat-conduction surface. The throughput delta from opening the lid alone (with the same fan) was 30-40% → 50-60%.

**3. Active vs passive matters.** Going from passive cooling (open lid, still air) to active airflow (desk fan blowing through) gives ~10-15% additional residency. The gain is real but smaller than the lid-open delta.

**4. Forced fans vs default fans.** Going from "stock fan curve with lid open and desk fan" to "forced 100% fans" added another 10-15%, taking sustained residency from 50-60% to 70%+. The OS fan policy is the largest single factor.

**5. Workload-relative impact.** A ~2x throughput improvement (8.21 → 4.34 sec/step) translates directly to roughly 2x faster Stage 1 paired wall. Stage 1 paired runtime projection went from ~5 days under stock cooling to ~3.5 days from cold start with aggressive cooling.

## Implications

For anyone doing extended MLX training on a MacBook chassis:

- **Run Macs Fan Control or Macs Fan Control Pro** (commodity menu-bar apps; ~$0-15). Set fans to 100% during long training jobs.
- **Open the laptop lid.** Counterintuitive vs the typical "clamshell on a stand" desk setup. The keyboard plate is a heatsink.
- **Add a desk fan blowing across the keyboard.** Doesn't need to be aimed precisely; any cross-flow helps.
- **Check ambient.** Each 1°C of room temperature drop is roughly 1°C of chip headroom before throttle. Cool rooms (or training during cooler hours) measurably help.
- **Don't waste cycles diagnosing MLX or your code.** If sustained throughput drops over time and your workload is constant, it's almost certainly thermals.

For benchmark fairness: most "MLX on Apple Silicon" throughput numbers in the wild don't condition on cooling state. A burst benchmark of a few hundred steps in fresh cooling will look 2x faster than a multi-hour sustained job under default cooling. Both numbers can be "true." Always state the cooling regime when reporting.

## Open questions

- **How does this scale across chassis?** M-series Mac Studio / Mac Pro (desktop) have much better thermal envelopes. Sustained throughput on those chassis would likely match or exceed the cold-start laptop numbers indefinitely. Worth measuring if access to a Mac Studio comes up.
- **Does an external cooler stack with forced fans?** Untested but likely yes. Aftermarket clip-on vacuum coolers exist for MacBooks and could push residency above 80%.
- **Is there a soft-throttle warning before hard throttle?** powermetrics reports `cpu/gpu thermal level`; tracking that against `sec_per_step` over a run would give an early-warning signal. Future investigation.

## Related findings (other throughput limits)

- **Swap cliff at the unified-memory budget.** Stage 1 with `micro_batch=32` had peak working set >128 GB, causing OS-level paging and a separate ~14x throughput collapse. Documented in `docs/2026-05-22-swap-cliff-and-scope-restore.md`. Fix was `micro_batch=8 grad_accum=4`, preserving effective batch.
- **Dispatch-boundedness.** At Stage 1 shapes even at peak utilization (70%+), the GPU is mostly idle between kernels. Phase C custom Metal kernels target this. Pending speed eval; expected to push utilization further at the cost of kernel-development time.

## Status / TODO

- [x] Stock vs forced fans measured.
- [x] Open lid vs closed lid measured.
- [x] Active vs passive cooling measured.
- [ ] Sustained-period verification: hold 100%-fan setup for 8+ hours, confirm no degradation creeps in.
- [ ] powermetrics thermal-level correlation with sec/step.
- [ ] If access: same measurement on a Mac Studio for envelope comparison.
