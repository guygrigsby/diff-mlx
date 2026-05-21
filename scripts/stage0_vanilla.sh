#!/bin/bash
# Stage 0 vanilla baseline. ~6,250 steps, ~1 day on M5 Max.
# Prereqs: data/shards/ already populated; venv activated; no LM Studio loaded.
set -euo pipefail
mkdir -p runs
python -m train \
  --stage stage0 \
  --shards_dir data/shards \
  --run_dir runs/stage0-vanilla-seed0 \
  --seed 0 \
  --variant vanilla
