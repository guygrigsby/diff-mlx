#!/bin/bash
# Stage 1 paired smoke (200 steps each variant). Wrapped in caffeinate so
# display sleep can't degrade the run. See scripts/stage1_smoke.py for what
# this validates (bf16, opt-state checkpoints, grad_accum routing).
#
# Usage:
#   ./scripts/stage1_smoke.sh
#   STEPS=50 ./scripts/stage1_smoke.sh                # shorter sanity check
#   OUT_ROOT=runs/stage1-smoke-take2 ./scripts/stage1_smoke.sh
set -euo pipefail
mkdir -p runs

OUT_ROOT="${OUT_ROOT:-runs/stage1-smoke}"
DATA_SEED="${DATA_SEED:-0}"
MODEL_SEED="${MODEL_SEED:-0}"
STEPS="${STEPS:-200}"

caffeinate -disu python -u scripts/stage1_smoke.py \
  --data_seed "$DATA_SEED" \
  --model_seed "$MODEL_SEED" \
  --out_root "$OUT_ROOT" \
  --steps "$STEPS" 2>&1 \
  | tee "${OUT_ROOT}.log"
