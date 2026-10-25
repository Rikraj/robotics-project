#!/usr/bin/env bash
# Train baseline and DR controllers for every seed, same budget. Safe to re-run: finished runs are skipped.
#   SEEDS="1 2 3" NUM_ENVS=4096 ITERS=1500 bash scripts/02_train_all.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SEEDS=${SEEDS:-"1 2 3"}
NUM_ENVS=${NUM_ENVS:-4096}
ITERS=${ITERS:-1500}
cd "$ISAACLAB"

for seed in $SEEDS; do
  for cfg in baseline dr; do
    if [ "$cfg" = "baseline" ]; then task=G1-Flat-Baseline-v0; exp=g1_flat_baseline; else task=G1-Flat-DR-v0; exp=g1_flat_dr; fi
    if ls "$LOGS/$exp"/*_"${cfg}_s${seed}"/model_$((ITERS-1)).pt >/dev/null 2>&1; then
      echo ">> $cfg seed $seed already finished, skipping"; continue
    fi
    echo ">> training $cfg seed $seed  ($task)"
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task "$task" --headless --num_envs "$NUM_ENVS" --max_iterations "$ITERS" \
      --seed "$seed" --run_name "${cfg}_s${seed}" 2>&1 | tee -a "$PROJ/results/train_${cfg}_s${seed}.log"
  done
done
echo "All training done. Checkpoints are under $LOGS"
