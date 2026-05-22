#!/bin/bash
# Multi-seed paired runs. Loops over N seeds, invoking the paired runner once
# per seed. Each pair writes to OUT_ROOT_BASE-seed$i/ so runs don't collide.
#
# Usage:
#   ./scripts/multi_seed_paired.sh                # default: 4 seeds (0..3)
#   N_SEEDS=6 ./scripts/multi_seed_paired.sh      # 6 seeds (0..5)
#   START_SEED=4 N_SEEDS=2 ./scripts/multi_seed_paired.sh
#
# Each pair already runs under caffeinate via stage0_paired.sh, so display
# sleep can't degrade these long runs.
set -euo pipefail

N_SEEDS="${N_SEEDS:-4}"
START_SEED="${START_SEED:-0}"
OUT_ROOT_BASE="${OUT_ROOT_BASE:-runs/multi-seed-paired}"

end=$(( START_SEED + N_SEEDS - 1 ))
for i in $(seq "$START_SEED" "$end"); do
  echo "[multi-seed] starting pair seed=$i ($((i - START_SEED + 1))/$N_SEEDS)"
  DATA_SEED="$i" MODEL_SEED="$i" OUT_ROOT="${OUT_ROOT_BASE}-seed${i}" \
    ./scripts/stage0_paired.sh
done

echo "[multi-seed] all $N_SEEDS pairs complete"
