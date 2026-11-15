#!/usr/bin/env bash
set -euo pipefail

# Isaac Lab's play command loads the exact trained policy. Set VIZ=kit to use
# Isaac Sim/Kit visualization (requires the Isaac Sim extra); default is Newton GL.
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/model_1499.pt [baseline|randomized]" >&2
  exit 2
fi
CHECKPOINT="$(realpath "$1")"
POLICY="${2:-baseline}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${PROJECT_ROOT%/*}/IsaacLab}"
TASK="${TASK:-Isaac-Velocity-Flat-G1}"
VIZ="${VIZ:-newton_gl}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi
cd "$ISAACLAB_ROOT"
if [[ "$VIZ" == "kit" ]]; then
  G1_DR_MODE="$POLICY" uv run --extra isaacsim --extra rsl-rl --extra video isaaclab play \
    --rl_library rsl_rl --task "$TASK" --checkpoint "$CHECKPOINT" \
    --num_envs 1 --viz kit --video --video_length 600
else
  G1_DR_MODE="$POLICY" uv run --extra rsl-rl --extra video isaaclab play \
    --rl_library rsl_rl --task "$TASK" --checkpoint "$CHECKPOINT" \
    --num_envs 1 --viz "$VIZ" --video --video_length 600
fi
