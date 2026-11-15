#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${PROJECT_ROOT%/*}/IsaacLab}"
TASK="${TASK:-Isaac-Velocity-Flat-G1}"
cd "$ISAACLAB_ROOT"
python "$PROJECT_ROOT/scripts/patch_g1_task.py" --isaaclab-root "$ISAACLAB_ROOT"

# First ensure the task can reset and step without learning.
uv run isaaclab random_agent --task "$TASK" --num_envs 8 physics=newton_mjwarp

# Then verify that both experiment variants construct and train for a few updates.
for mode in baseline randomized; do
  echo "Smoke test: $mode"
  G1_DR_MODE="$mode" uv run --extra rsl-rl isaaclab train \
    --rl_library rsl_rl --task "$TASK" --num_envs 16 \
    --max_iterations 10 --seed 11 --run_name "smoke_${mode}" \
    physics=newton_mjwarp
done

echo "Smoke tests completed. Do not include smoke-test checkpoints in final result tables."
