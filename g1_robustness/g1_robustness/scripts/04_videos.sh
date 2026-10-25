#!/usr/bin/env bash
# Demo videos for seed 1 of each group: (a) nominal, (b) low friction, (c) strong pushes held-out, (d) both.
#   SEED=1 bash scripts/04_videos.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SEED=${SEED:-1}
mapfile -t POLS < <(python3 "$PROJ/scripts/list_policies.py" --logs "$LOGS")
cd "$ISAACLAB"
for pol in "${POLS[@]}"; do
  name="${pol%%=*}"
  case "$name" in *_s"$SEED") ;; *) continue ;; esac
  for cond in "0.8 0.0" "0.3 0.0" "0.8 1.0" "0.3 1.0"; do
    set -- $cond
    echo ">> video: $name  friction=$1  push=$2"
    ./isaaclab.sh -p "$PROJ/scripts/record_demo.py" --headless --friction "$1" --push "$2" \
      --policy "$pol" --out "$PROJ/videos" 2>&1 | tail -n 3
  done
done
ls -lh "$PROJ/videos"
