#!/bin/bash
# Stage 0 seed band: rerun the paired experiment at several seeds to put an
# empirical noise band on the paired delta (writeup caveat: single seed).
# Each seed is internally paired (shared init, same data order); the spread
# of delta across seeds is the noise band. Seed 0 is the recorded original.
# ~2.75h per seed on M5 Max; seeds 1-4 is an overnight run.
# Prereqs: data/shards/ populated; .venv present; no other GPU consumers.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs

SEEDS="${SEEDS:-1 2 3 4}"

for s in $SEEDS; do
  echo "=== seed $s: $(date) ==="
  caffeinate -disu .venv/bin/python -u scripts/stage0_paired.py \
    --data_seed "$s" --model_seed "$s" --out_root runs \
    2>&1 | tee "runs/stage0-seed-band-seed${s}.log"
done

echo "=== seed band done: $(date) ==="
# Per-seed paired delta from the final val_loss_full of each variant.
.venv/bin/python - <<'EOF'
import json, glob, re
print(f"{'seed':>4} {'vanilla':>8} {'diff':>8} {'delta':>8}")
for vdir in sorted(glob.glob("runs/stage0-paired-vanilla-seed*")):
    seed = re.search(r"seed(\d+)$", vdir).group(1)
    ddir = vdir.replace("vanilla", "diff")
    def last_val(d):
        v = None
        try:
            for line in open(d + "/metrics.jsonl"):
                rec = json.loads(line)
                if "val_loss_full" in rec:
                    v = rec["val_loss_full"]
        except FileNotFoundError:
            return None
        return v
    va, di = last_val(vdir), last_val(ddir)
    if va is None or di is None:
        print(f"{seed:>4} incomplete")
        continue
    print(f"{seed:>4} {va:8.4f} {di:8.4f} {di-va:+8.4f}")
EOF
