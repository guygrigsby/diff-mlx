# Stage 0 seed band: the paired delta is seed noise

**Date:** 2026-07-25
**Runner:** `scripts/stage0_seed_band.sh` (seeds 1-4, overnight on the M5 Max, ~60 min per variant)

The final writeup's caveat was single seed, single pairing. This run puts an
empirical noise band on the Stage 0 paired delta: four more paired seeds,
each internally paired (byte-identical shared init, same data order), varying
`data_seed = model_seed = s`.

Data note: shards were rebuilt from FineWeb-Edu sample-10BT on this machine
(6 raw files; val came out 43.8M tokens vs the original build's 75M). Seeds
1-4 share this build, so the band is internally consistent. The seed 0 row
is the recorded result from the original build, not rerun.

## Result

| seed | vanilla val | diff val | delta (diff - vanilla) |
|---|---|---|---|
| 0 (recorded, original build) | 4.720 | 4.700 | -0.020 |
| 1 | 4.7236 | 4.7019 | -0.0216 |
| 2 | 4.7170 | 4.6946 | -0.0224 |
| 3 | 4.7101 | 4.7272 | +0.0171 |
| 4 | 4.7250 | 4.6872 | -0.0378 |

Seeds 1-4: mean delta -0.016, sample std 0.023, range -0.038 to +0.017.
Seed 3 flips sign.

## Read

The original Stage 0 delta of -0.020 sits inside the measured seed-noise
band. A single-seed delta of that magnitude at this scale is noise, which is
what the final writeup suspected and could not show. This also recasts the
Stage 0 to Stage 1 arc: the scale-up did not kill a real small-scale effect;
there was no Stage 0 effect to kill.

The band is measured at Stage 0. It does not transfer numerically to Stage 1
(20x the tokens should shrink run-to-run noise), so the Stage 1 negative
stays a single-seed result with its own caveat.

Per-seed logs: `runs/stage0-seed-band-seed{1..4}.log`; metrics in
`runs/stage0-paired-{vanilla,diff}-seed{1..4}/metrics.jsonl` (gitignored,
table above is the record).
