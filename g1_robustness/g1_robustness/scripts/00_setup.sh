#!/usr/bin/env bash
# One-time setup: link the g1_dr package into Isaac Lab and install plotting dependencies.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CFG_DIR="$ISAACLAB/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config"
[ -f "$CFG_DIR/g1/flat_env_cfg.py" ] || { echo "ERROR: stock G1 flat config not found under $CFG_DIR"; exit 1; }
[ -f "$CFG_DIR/g1/agents/rsl_rl_ppo_cfg.py" ] || { echo "ERROR: stock G1 rsl_rl config not found"; exit 1; }

ln -sfn "$PROJ/g1_dr" "$CFG_DIR/g1_dr"
echo "Linked $PROJ/g1_dr -> $CFG_DIR/g1_dr"

echo "Isaac Lab path : $ISAACLAB"
grep -m1 '^version' "$ISAACLAB/source/isaaclab/config/extension.toml" || true
(cd "$ISAACLAB" && ./isaaclab.sh -p -m pip install --quiet pandas matplotlib tensorboard numpy)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "Registered G1 tasks:"
(cd "$ISAACLAB" && ./isaaclab.sh -p scripts/environments/list_envs.py --keyword G1-Flat 2>&1 | grep -E "G1-Flat|Isaac-Velocity-Flat-G1" || echo "!! G1-Flat-* tasks not listed, see README troubleshooting")
