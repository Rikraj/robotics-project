"""Two Unitree G1 flat-ground velocity-tracking tasks that differ ONLY in friction / push randomization.

Both are derived from Isaac Lab's stock ``G1FlatEnvCfg`` (same observations, actions, rewards, terminations).
To isolate the effect of randomization we also remove the stock mass randomization from BOTH variants.

If you change any constant below, mirror it in ``scripts/analyze.py`` (TRAIN_* constants).
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

# ---- fixed ("nominal") conditions: Isaac Lab's stock G1 values ----
NOMINAL_STATIC_FRICTION = 0.8
NOMINAL_DYNAMIC_FRICTION = 0.6

# ---- domain randomization ranges (training only) ----
DR_STATIC_FRICTION = (0.4, 1.2)   # sampled once per environment at startup
DR_DYNAMIC_FRICTION = (0.3, 0.9)  # ~0.75 x static, as in the evaluation sweep
DR_PUSH_VELOCITY = 0.6            # m/s, uniform per axis in [-v, v] on x and y
DR_PUSH_INTERVAL_S = (6.0, 10.0)  # time between pushes, per env


def _apply_common(cfg) -> None:
    """Settings shared by both variants (so the comparison stays controlled)."""
    cfg.episode_length_s = 20.0
    ranges = cfg.commands.base_velocity.ranges
    ranges.lin_vel_x = (0.0, 0.8)   # low-speed walking only
    ranges.lin_vel_y = (0.0, 0.0)


def _strip_stock_randomization(cfg) -> None:
    """Remove stock events that would add randomization beyond the factor under study."""
    for name in ("add_base_mass", "push_robot", "base_external_force_torque", "base_com"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)


def _set_friction(cfg, static, dynamic) -> None:
    params = cfg.events.physics_material.params
    params["static_friction_range"] = static
    params["dynamic_friction_range"] = dynamic
    params["restitution_range"] = (0.0, 0.0)


@configclass
class G1FlatBaselineEnvCfg(G1FlatEnvCfg):
    """Baseline: fixed friction, no pushes."""

    def __post_init__(self):
        super().__post_init__()
        _apply_common(self)
        _strip_stock_randomization(self)
        _set_friction(
            self,
            (NOMINAL_STATIC_FRICTION, NOMINAL_STATIC_FRICTION),
            (NOMINAL_DYNAMIC_FRICTION, NOMINAL_DYNAMIC_FRICTION),
        )


@configclass
class G1FlatDREnvCfg(G1FlatEnvCfg):
    """Domain randomization: random ground friction + bounded random pushes."""

    def __post_init__(self):
        super().__post_init__()
        _apply_common(self)
        _strip_stock_randomization(self)
        _set_friction(self, DR_STATIC_FRICTION, DR_DYNAMIC_FRICTION)
        v = DR_PUSH_VELOCITY
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=DR_PUSH_INTERVAL_S,
            params={"velocity_range": {"x": (-v, v), "y": (-v, v)}},
        )
