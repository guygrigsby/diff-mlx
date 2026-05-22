#!/bin/bash
# Stage 1 paired full run. ~18-24h total wall on M5 Max with bf16.
# Auto-resume: if killed or crashed mid-way, re-run this command to continue
# from the last latest.safetensors.
#
# Usage:
#   ./scripts/stage1_paired.sh
#   OUT_ROOT=runs/stage1-paired-take2 ./scripts/stage1_paired.sh
#   DATA_SEED=1 MODEL_SEED=1 ./scripts/stage1_paired.sh
set -euo pipefail
mkdir -p runs

OUT_ROOT="${OUT_ROOT:-runs/stage1-paired}"
DATA_SEED="${DATA_SEED:-0}"
MODEL_SEED="${MODEL_SEED:-0}"

caffeinate -disu python -u scripts/stage1_paired.py \
  --data_seed "$DATA_SEED" \
  --model_seed "$MODEL_SEED" \
  --out_root "$OUT_ROOT" 2>&1 \
  | tee "${OUT_ROOT}.log"
