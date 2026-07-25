# Thermal vs throughput on M5 Max for sustained MLX training

**Date:** 2026-05-24 (active, updated as findings accumulate)

> **Correction 2026-07-25:** the power-delivery section below frames the dock
> incident as 100W dock vs 140W MagSafe, with recovery attributed to
> "re-negotiating to 140W". Wrong for this machine. The 14-inch M5 Max
> (Mac17,7) negotiates at most 100W regardless of adapter; its fast-charge
> spec is 96W and it does not request the 28V EPR profile (140W is a 16-inch
> spec). Verified 2026-07-25 against Apple's tech specs and live
> `system_profiler` on this machine showing a 100W contract. The observation
> stands (throughput drop while cool, battery drain under "charging",
> recovery after switching to the direct brick). The corrected mechanism is a
> dock sharing upstream power with a monitor delivering less than the
> sustained draw, with at most 100W of headroom to begin with.
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

## Power-delivery throttle (added 2026-05-27, during Stage 1 paired)

A second non-compute throttle, distinct from thermal and arguably more insidious because the temperature stays *low*.

Mid-run, sustained throughput dropped ~10% (sec/step 4.76 → ~5.2) while the chip sat at **73°C**. Low temp ruled out thermal. The cause was power delivery:

- The laptop was charging through a **Thunderbolt dock that also drove a monitor**, and the dock delivers **100W**.
- macOS selected the dock as the power source **even with the 140W MagSafe brick also plugged in**. The Mac does not sum two power inputs; it arbitrates to one, and the dock won. Replugging the dock did not hand the role to MagSafe.
- **100W is the USB-C PD ceiling** (20V × 5A). The full 140W requires MagSafe 3 or an EPR-rated (28V) cable. So even a 140W-capable setup negotiates down to 100W over a standard USB-C path.
- Under full GPU load, the SoC plus charging a depleted battery exceeded 100W, so the system pulled from the battery as a buffer: it **drained while macOS reported "charging"** (net draw > supply). As the battery fell, clocks were shaved further.

**Signature:** cool temps + reduced GPU clocks/utilization + battery discharging while plugged in. **Fix:** fully remove the dock's power input so MagSafe is the only source → re-negotiates to 140W → throughput recovers. **Confirm with** `system_profiler SPPowerDataType | grep Wattage` (should read ~140, not 100).

Practical rule: on a laptop, **a low temperature does not rule out throttling.** If sustained throughput drops and the chip is cool, suspect power delivery before code or thermals. And beware docks: convenient single-cable setups silently cap you at 100W.

## Related findings (other throughput limits)

- **Swap cliff at the unified-memory budget.** Stage 1 with `micro_batch=32` had peak working set >128 GB, causing OS-level paging and a separate ~14x throughput collapse. Documented in `docs/2026-05-22-swap-cliff-and-scope-restore.md`. Fix was `micro_batch=8 grad_accum=4`, preserving effective batch.
- **Dispatch-boundedness.** At Stage 1 shapes even at peak utilization (70%+), the GPU is mostly idle between kernels. Phase C custom Metal kernels target this. Pending speed eval; expected to push utilization further at the cost of kernel-development time.

## Status / TODO

- [x] Stock vs forced fans measured.
- [x] Open lid vs closed lid measured.
- [x] Active vs passive cooling measured.
- [x] Sustained-period verification: the multi-day Stage 1 paired run held the aggressive-cooling setup for many hours with no thermal degradation once cooling was adequate. Confirmed.
- [x] Power-delivery throttle identified (dock 100W vs MagSafe 140W), 2026-05-27.
- [ ] powermetrics thermal-level correlation with sec/step.
- [ ] If access: same measurement on a Mac Studio for envelope comparison.
