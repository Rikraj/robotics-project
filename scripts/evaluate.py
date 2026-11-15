#!/usr/bin/env python3
"""Evaluate one trained G1 PPO policy over a friction x push condition matrix.

This script is intended for Isaac Lab 3.0.0 and its registered G1 flat task.
Run through the Isaac Lab uv environment from the IsaacLab repository root.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Velocity-Flat-G1")
    parser.add_argument("--policy", choices=["baseline", "randomized"], required=True)
    parser.add_argument("--training-seed", type=int, required=True, help="Seed used to train this checkpoint")
    parser.add_argument("--eval-seed", type=int, default=2027, help="Shared seed used for paired evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes-per-condition", type=int, default=100)
    parser.add_argument("--frictions", type=float, nargs="+", default=[0.4, 0.5, 0.875, 1.0, 1.25, 1.4])
    parser.add_argument("--push-strengths", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--success-error-threshold", type=float, default=0.35)
    # AppLauncher contributes --headless, --device, --enable_cameras, etc.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def configure_condition(env_cfg: Any, friction: float, push_strength: float) -> None:
    """Fix all unrelated dynamics and configure a single held-out condition."""
    events = env_cfg.events
    if events is None or events.physics_material is None:
        raise RuntimeError("G1 task has no events.physics_material term; check the task config version.")

    # Remove potentially confounding base-parameter randomization.
    if hasattr(events, "add_base_mass"):
        events.add_base_mass = None
    if hasattr(events, "base_com"):
        events.base_com = None

    material = events.physics_material
    params = material.params
    asset_cfg = params.get("asset_cfg")
    if asset_cfg is None:
        raise RuntimeError("physics_material.params has no asset_cfg; cannot target foot collision bodies safely.")
    asset_cfg.body_names = ".*_ankle_roll_link"
    params["static_friction_range"] = (float(friction), float(friction))
    params["dynamic_friction_range"] = (float(friction), float(friction))

    if push_strength <= 0.0:
        events.push_robot = None
    else:
        if getattr(events, "push_robot", None) is None:
            raise RuntimeError("G1 task does not define events.push_robot; add the standard push_by_setting_velocity event first.")
        events.push_robot.interval_range_s = (5.0, 5.0)
        events.push_robot.params["velocity_range"] = {
            "x": (-float(push_strength), float(push_strength)),
            "y": (-float(push_strength), float(push_strength)),
        }


def get_velocity_error(env: Any) -> np.ndarray:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    actual = robot.data.root_lin_vel_b[:, :2]
    desired = unwrapped.command_manager.get_command("base_velocity")[:, :2]
    err = (actual - desired).norm(dim=1)
    return to_numpy(err).astype(np.float64)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.num_envs < 1 or args.episodes_per_condition < 1:
        raise ValueError("num-envs and episodes-per-condition must be positive")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Isaac/torch/gym imports must happen after Kit/AppLauncher initialization.
    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401 - triggers task registration
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    # Build the evaluation config with the DR event term present, then set each
    # condition to a fixed friction/push value below.
    os.environ["G1_DR_MODE"] = "randomized"
    torch.manual_seed(args.eval_seed)
    np.random.seed(args.eval_seed)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy", "training_seed", "eval_seed", "friction", "push_strength",
        "episode_index", "mean_planar_velocity_error_m_s", "mean_abs_velocity_error",
        "fell", "success", "survival_seconds",
    ]
    # Start a clean file so reruns do not silently duplicate the dataset.
    with output.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    try:
        for friction in args.frictions:
            for push_strength in args.push_strengths:
                # Keep evaluation randomness paired across the two policies.
                condition_code = int(round(friction * 1000)) * 100 + int(round(push_strength * 1000))
                condition_seed = int(args.eval_seed + condition_code)
                torch.manual_seed(condition_seed)
                np.random.seed(condition_seed)

                env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
                if hasattr(env_cfg, "seed"):
                    env_cfg.seed = condition_seed
                configure_condition(env_cfg, friction, push_strength)

                env = None
                raw_env = None
                try:
                    raw_env = gym.make(args.task, cfg=env_cfg, render_mode=None)
                    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
                    agent_cfg.device = args.device
                    clip_actions = getattr(agent_cfg, "clip_actions", None)
                    env = RslRlVecEnvWrapper(raw_env, clip_actions=clip_actions)
                    runner_cfg = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)
                    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=args.device)
                    runner.load(str(checkpoint))
                    policy = runner.get_inference_policy(device=args.device)

                    obs = env.get_observations()
                    n = args.num_envs
                    error_sum = np.zeros(n, dtype=np.float64)
                    step_count = np.zeros(n, dtype=np.int64)
                    rows: list[dict[str, Any]] = []
                    episode_duration = float(env_cfg.episode_length_s)
                    dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
                    min_survival = 0.95 * episode_duration

                    while len(rows) < args.episodes_per_condition:
                        with torch.inference_mode():
                            errors = get_velocity_error(env)
                            error_sum += errors
                            step_count += 1
                            actions = policy(obs)
                            obs, _rewards, dones, infos = env.step(actions)

                        done_np = to_numpy(dones).reshape(-1).astype(bool)
                        timeout_values = infos.get("time_outs") if isinstance(infos, dict) else None
                        if timeout_values is None:
                            raise RuntimeError(
                                "RSL-RL wrapper did not expose infos['time_outs']; cannot distinguish a fall from a time-limit completion. "
                                "Inspect the installed Isaac Lab 3.0 wrapper API before using these results."
                            )
                        timeout_np = to_numpy(timeout_values).reshape(-1).astype(bool)

                        for env_id in np.flatnonzero(done_np):
                            survival = float(step_count[env_id] * dt)
                            mean_error = float(error_sum[env_id] / max(step_count[env_id], 1))
                            fell = int(not timeout_np[env_id])
                            success = int(
                                fell == 0
                                and survival >= min_survival
                                and mean_error <= args.success_error_threshold
                            )
                            rows.append({
                                "policy": args.policy,
                                "training_seed": args.training_seed,
                                "eval_seed": args.eval_seed,
                                "friction": friction,
                                "push_strength": push_strength,
                                "episode_index": len(rows) + 1,
                                "mean_planar_velocity_error_m_s": mean_error,
                                # Compatibility alias used by older plotting notebooks.
                                "mean_abs_velocity_error": mean_error,
                                "fell": fell,
                                "success": success,
                                "survival_seconds": survival,
                            })
                            error_sum[env_id] = 0.0
                            step_count[env_id] = 0
                            if len(rows) >= args.episodes_per_condition:
                                break

                    rows = rows[: args.episodes_per_condition]
                    with output.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerows(rows)
                    successes = sum(int(r["success"]) for r in rows)
                    falls = sum(int(r["fell"]) for r in rows)
                    mean_err = float(np.mean([r["mean_planar_velocity_error_m_s"] for r in rows]))
                    print(
                        f"{args.policy} friction={friction:.3f} push={push_strength:.3f}: "
                        f"success={successes}/{len(rows)} falls={falls}/{len(rows)} mean_error={mean_err:.3f} m/s"
                    )
                finally:
                    if env is not None:
                        env.close()
                    elif raw_env is not None:
                        raw_env.close()
    finally:
        simulation_app.close()

    print(f"Evaluation CSV written: {output}")


if __name__ == "__main__":
    main()
