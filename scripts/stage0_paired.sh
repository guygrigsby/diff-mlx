#!/bin/bash
# Stage 0 paired runner. Runs vanilla then diff back-to-back, ~3 hours total on M5 Max.
# Prereqs: data/shards/ populated; venv activated; no other GPU consumers.
# caffeinate -disu prevents display/idle/system sleep. Without it, display
# sleep drops the GPU into a low-power state mid-run and stalls catastrophically
# (Phase B retro: caused a 5x slowdown on the original diff arm).
set -euo pipefail
mkdir -p runs

OUT_ROOT="${OUT_ROOT:-runs/stage0-paired}"
DATA_SEED="${DATA_SEED:-0}"
MODEL_SEED="${MODEL_SEED:-0}"

caffeinate -disu python -u scripts/stage0_paired.py \
  --data_seed "$DATA_SEED" \
  --model_seed "$MODEL_SEED" \
  --out_root "$OUT_ROOT" 2>&1 \
  | tee "${OUT_ROOT}.log"
