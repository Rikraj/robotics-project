#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. By default the project and IsaacLab checkouts are siblings.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${PROJECT_ROOT%/*}/IsaacLab}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITERS="${MAX_ITERS:-1500}"
SEEDS="${SEEDS:-11 22 33}"
TASK="${TASK:-Isaac-Velocity-Flat-G1}"

if [[ ! -d "$ISAACLAB_ROOT" ]]; then
  echo "Isaac Lab checkout not found: $ISAACLAB_ROOT" >&2
  echo "Set ISAACLAB_ROOT=/absolute/path/to/IsaacLab and rerun." >&2
  exit 2
fi

python "$PROJECT_ROOT/scripts/patch_g1_task.py" --isaaclab-root "$ISAACLAB_ROOT"
cd "$ISAACLAB_ROOT"
mkdir -p "$PROJECT_ROOT/logs"
{
  echo "timestamp=$(date -Is)"
  echo "isaaclab_root=$ISAACLAB_ROOT"
  echo "isaaclab_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "task=$TASK"
  echo "num_envs=$NUM_ENVS"
  echo "max_iterations=$MAX_ITERS"
  echo "seeds=$SEEDS"
  echo
  nvidia-smi || true
} | tee "$PROJECT_ROOT/logs/training_environment.txt"

for mode in baseline randomized; do
  for seed in $SEEDS; do
    run_name="${mode}_seed${seed}"
    echo "============================================================"
    echo "Training: $run_name"
    echo "============================================================"
    G1_DR_MODE="$mode" uv run --extra rsl-rl isaaclab train \
      --rl_library rsl_rl \
      --task "$TASK" \
      --num_envs "$NUM_ENVS" \
      --max_iterations "$MAX_ITERS" \
      --seed "$seed" \
      --run_name "$run_name" \
      physics=newton_mjwarp
  done
done

echo "All training jobs completed. Locate the six run directories under: $ISAACLAB_ROOT/logs/rsl_rl"
echo "Copy each final model_*.pt path into the evaluation command described in README.md."
