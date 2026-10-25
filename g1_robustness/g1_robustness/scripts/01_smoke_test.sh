#!/usr/bin/env bash
# Short training runs of both tasks. Checks registration, gives a speed estimate, and writes to a throw-away run name.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$ISAACLAB"
for task in G1-Flat-Baseline-v0 G1-Flat-DR-v0; do
  echo "=== smoke test: $task ==="
  start=$(date +%s)
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$task" --headless --num_envs 1024 --max_iterations 30 --seed 0 --run_name smoke
  end=$(date +%s)
  echo "$task: 30 iterations at 1024 envs took $((end-start)) s (includes ~1 min start-up)"
done
echo "Check the 'Computation: ... steps/s' line in the output above to extrapolate the full 1500-iteration run."
echo "Remove smoke logs with:  rm -rf $LOGS/*/*_smoke"
