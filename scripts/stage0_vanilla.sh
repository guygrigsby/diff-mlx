#!/bin/bash
# Stage 0 vanilla baseline. ~6,250 steps, ~1 day on M5 Max.
# Prereqs: data/shards/ already populated; venv activated; no LM Studio loaded.
# caffeinate -disu prevents display/idle/system sleep, which otherwise
# transitions the GPU into a low-power state and stalls long runs.
# (Phase B retro: display-sleep stalls caused a 5x slowdown on the original diff run.)
set -euo pipefail
mkdir -p runs
caffeinate -disu python -u -m train \
  --stage stage0 \
  --shards_dir data/shards \
  --run_dir runs/stage0-vanilla-seed0 \
  --seed 0 \
  --variant vanilla
