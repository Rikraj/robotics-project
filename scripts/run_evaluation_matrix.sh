#!/usr/bin/env bash
set -euo pipefail

# Example:
#   ISAACLAB_ROOT=../IsaacLab ./scripts/run_evaluation_matrix.sh \
#     /path/baseline_seed11/model_1499.pt /path/baseline_seed22/model_1499.pt /path/baseline_seed33/model_1499.pt \
#     /path/randomized_seed11/model_1499.pt /path/randomized_seed22/model_1499.pt /path/randomized_seed33/model_1499.pt

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${PROJECT_ROOT%/*}/IsaacLab}"
TASK="${TASK:-Isaac-Velocity-Flat-G1}"
NUM_ENVS="${NUM_ENVS:-64}"
EPISODES="${EPISODES:-100}"
EVAL_SEED="${EVAL_SEED:-2027}"

if [[ $# -ne 6 ]]; then
  echo "Expected six checkpoints: baseline seed11/22/33, then randomized seed11/22/33." >&2
  exit 2
fi

cd "$ISAACLAB_ROOT"
run_one() {
  local policy="$1" train_seed="$2" checkpoint="$3"
  local output="$PROJECT_ROOT/results/raw/evaluation_${policy}_seed${train_seed}.csv"
  echo "Evaluating $policy training seed $train_seed -> $output"
  uv run --extra rsl-rl python "$PROJECT_ROOT/scripts/evaluate.py" \
    --headless \
    --task "$TASK" \
    --policy "$policy" \
    --training-seed "$train_seed" \
    --eval-seed "$EVAL_SEED" \
    --checkpoint "$checkpoint" \
    --output "$output" \
    --num-envs "$NUM_ENVS" \
    --episodes-per-condition "$EPISODES"
}

run_one baseline 11 "$1"
run_one baseline 22 "$2"
run_one baseline 33 "$3"
run_one randomized 11 "$4"
run_one randomized 22 "$5"
run_one randomized 33 "$6"

echo "Evaluation matrix finished. Run: python $PROJECT_ROOT/scripts/analyze_results.py"
