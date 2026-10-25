#!/usr/bin/env bash
# Evaluate every trained policy on a friction x push grid (one Isaac Sim process per friction value).
#   FRICTIONS="0.1 0.2 ..." PUSHES="0 0.25 ..." ENVS_PER_LEVEL=100 bash scripts/03_eval_all.sh
# Re-running skips friction values whose CSV already exists (delete results/eval_mu*.csv to redo).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
FRICTIONS=${FRICTIONS:-"0.1 0.2 0.3 0.5 0.8 1.0 1.2 1.5"}
PUSHES=${PUSHES:-"0 0.25 0.5 0.75 1.0 1.25 1.5"}
ENVS_PER_LEVEL=${ENVS_PER_LEVEL:-100}

mapfile -t POLS < <(python3 "$PROJ/scripts/list_policies.py" --logs "$LOGS")
echo "Policies to evaluate:"; printf '  %s\n' "${POLS[@]}"

cd "$ISAACLAB"
for mu in $FRICTIONS; do
  out="$PROJ/results/eval_mu${mu}.csv"
  if [ -f "$out" ]; then echo ">> mu=$mu done already, skipping"; continue; fi
  echo ">> evaluating at friction mu=$mu"
  ./isaaclab.sh -p "$PROJ/scripts/evaluate.py" --headless \
    --friction "$mu" --push_levels $PUSHES --envs_per_level "$ENVS_PER_LEVEL" \
    --out "$out" --policies "${POLS[@]}" 2>&1 | tee -a "$PROJ/results/eval.log"
done
echo "Evaluation done: $(ls "$PROJ"/results/eval_mu*.csv | wc -l) CSV files in $PROJ/results"
