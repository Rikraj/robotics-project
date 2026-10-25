"""Custom event terms used only by the evaluation / demo scripts.

Training uses Isaac Lab's built-in ``push_by_setting_velocity``; this file adds a variant that lets ONE
simulation hold several push strengths side by side (env i gets level ``i % len(levels)``), so a whole
push sweep is evaluated in a single process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def push_by_setting_velocity_levels(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    levels: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Add a random horizontal velocity impulse to the robot base.

    Per axis (x, y) the impulse is uniform in [-m, m] where m = levels[env_id % len(levels)] (m/s).
    This mirrors the distribution used for training pushes (uniform square), only the bound differs per env.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    lv = torch.tensor(levels, device=asset.device, dtype=torch.float32)
    mag = lv[env_ids % len(levels)]
    vel_w = asset.data.root_vel_w[env_ids]
    noise = (torch.rand(len(env_ids), 2, device=asset.device) * 2.0 - 1.0) * mag.unsqueeze(1)
    vel_w[:, 0:2] += noise
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
