"""Record demonstration videos of one policy under a chosen friction / push strength.

  ./isaaclab.sh -p $PROJ/scripts/record_demo.py --headless --friction 0.3 --push 1.0 \
      --policy baseline_s1=/path/model_1499.pt --out $PROJ/videos
Needs a rendering-capable GPU (L40S, RTX 6000 Ada, A10G ... not A100/H100).
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="G1-Flat-Baseline-v0")
parser.add_argument("--friction", type=float, default=0.8)
parser.add_argument("--push", type=float, default=0.0, help="Max push speed per axis (m/s).")
parser.add_argument("--push_interval", type=float, default=3.0)
parser.add_argument("--cmd_vx", type=float, default=0.6)
parser.add_argument("--video_length", type=int, default=500, help="Steps (0.02 s each).")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--policy", required=True, help="name=checkpoint.pt")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # needed for rgb_array rendering
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1_dr.events import push_by_setting_velocity_levels  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def main():
    name, ckpt = args.policy.split("=", 1)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.episode_length_s = 20.0
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (args.cmd_vx, args.cmd_vx)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    p = cfg.events.physics_material.params
    p["static_friction_range"] = (args.friction, args.friction)
    p["dynamic_friction_range"] = (0.75 * args.friction, 0.75 * args.friction)
    cfg.events.push_robot = EventTerm(
        func=push_by_setting_velocity_levels,
        mode="interval",
        interval_range_s=(args.push_interval, args.push_interval),
        params={"levels": [args.push]},
    )
    # camera follows robot 0
    cfg.viewer.eye = (2.5, 2.5, 1.8)
    cfg.viewer.lookat = (0.0, 0.0, 0.8)
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.resolution = (1280, 720)

    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
    os.makedirs(args.out, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env, video_folder=args.out, step_trigger=lambda s: s == 0, video_length=args.video_length,
        name_prefix=f"{name}_mu{args.friction}_push{args.push}", disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device
    try:
        import importlib.metadata as metadata

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    except Exception:
        pass
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    out = env.get_observations()
    obs = out[0] if isinstance(out, tuple) else out
    with torch.inference_mode():
        for _ in range(args.video_length + 1):
            obs, _, _, _ = env.step(policy(obs))
    env.close()
    print(f"[demo] video saved in {args.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
