"""Registers two tasks:  G1-Flat-Baseline-v0  and  G1-Flat-DR-v0.

This package is symlinked into isaaclab_tasks/.../velocity/config/ by scripts/00_setup.sh, so Isaac Lab's
automatic package import registers the tasks for the stock train.py / play.py scripts.
"""

import gymnasium as gym

gym.register(
    id="G1-Flat-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfgs:G1FlatBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfgs:G1FlatBaselinePPORunnerCfg",
    },
)

gym.register(
    id="G1-Flat-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfgs:G1FlatDREnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfgs:G1FlatDRPPORunnerCfg",
    },
)
