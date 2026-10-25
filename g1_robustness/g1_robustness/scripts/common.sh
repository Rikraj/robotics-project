#!/usr/bin/env bash
# Sourced by the other scripts. Finds the project and the Isaac Lab checkout.
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${ISAACLAB:-}" ]; then
  for c in "$HOME/IsaacLab" "$HOME/isaaclab" /workspace/IsaacLab /workspace/isaaclab /opt/IsaacLab /isaac-lab /IsaacLab; do
    if [ -x "$c/isaaclab.sh" ]; then ISAACLAB="$c"; break; fi
  done
fi
if [ -z "${ISAACLAB:-}" ]; then
  # last resort: search the home and /workspace trees
  ISAACLAB="$(dirname "$(find "$HOME" /workspace -maxdepth 4 -name isaaclab.sh 2>/dev/null | head -n1)" 2>/dev/null || true)"
fi
if [ -z "${ISAACLAB:-}" ] || [ ! -x "$ISAACLAB/isaaclab.sh" ]; then
  echo "ERROR: Isaac Lab not found. Run:  export ISAACLAB=/path/to/IsaacLab  and retry." >&2
  exit 1
fi
export ISAACLAB PROJ
export LOGS="$ISAACLAB/logs/rsl_rl"
mkdir -p "$PROJ/results" "$PROJ/figures" "$PROJ/videos"
