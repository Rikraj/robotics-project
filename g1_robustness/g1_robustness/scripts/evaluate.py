"""Evaluate trained G1 policies at ONE ground-friction value and a whole sweep of push strengths.

One simulation holds  len(push_levels) x envs_per_level  robots; robot i gets push level i % len(push_levels).
Every robot runs exactly one 20 s episode per policy (first episode only, so there is no survivor bias).

Usage (from the Isaac Lab folder, via isaaclab.sh -p):
  ./isaaclab.sh -p $PROJ/scripts/evaluate.py --headless --friction 0.8 \
      --push_levels 0 0.25 0.5 0.75 1.0 1.25 1.5 --out results/eval_mu0.8.csv \
      --policies baseline_s1=/path/model_1499.pt dr_s1=/path/model_1499.pt
"""

import argparse
import csv
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Robustness evaluation of G1 walking policies.")
parser.add_argument("--task", default="G1-Flat-Baseline-v0", help="Only used to build the env + network shapes.")
parser.add_argument("--friction", type=float, required=True, help="Static friction mu (dynamic = 0.75 mu).")
parser.add_argument("--push_levels", type=float, nargs="+", required=True, help="Max push speed per axis (m/s).")
parser.add_argument("--envs_per_level", type=int, default=100)
parser.add_argument("--push_interval", type=float, default=4.0, help="Seconds between pushes.")
parser.add_argument("--cmd_vx", type=float, default=0.6, help="Forward velocity command (m/s).")
parser.add_argument("--episode_s", type=float, default=20.0)
parser.add_argument("--success_err", type=float, default=0.25, help="Mean lin-vel error threshold for 'success' (m/s).")
parser.add_argument("--no_obs_noise", action="store_true", help="Disable observation noise (default: keep as in training).")
parser.add_argument("--seed", type=int, default=100, help="Evaluation seed (same for every policy).")
parser.add_argument("--out", required=True, help="CSV file to write.")
parser.add_argument("--policies", nargs="+", required=True, help="name=checkpoint.pt  (name like baseline_s1 / dr_s2)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- everything that imports Isaac Lab must come after the app is launched ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  (importing it registers all tasks, incl. the symlinked g1_dr)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1_dr.events import push_by_setting_velocity_levels  # noqa: E402


def _first(x):
    return x[0] if isinstance(x, tuple) else x


def build_env(n_levels: int):
    n_envs = n_levels * args.envs_per_level
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=n_envs)
    cfg.seed = args.seed
    cfg.episode_length_s = args.episode_s
    # fixed, identical command for every robot: walk straight at cmd_vx
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (args.cmd_vx, args.cmd_vx)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    # fixed friction for the whole simulation
    p = cfg.events.physics_material.params
    p["static_friction_range"] = (args.friction, args.friction)
    p["dynamic_friction_range"] = (0.75 * args.friction, 0.75 * args.friction)
    p["restitution_range"] = (0.0, 0.0)
    # per-env push strength
    cfg.events.push_robot = EventTerm(
        func=push_by_setting_velocity_levels,
        mode="interval",
        interval_range_s=(args.push_interval, args.push_interval),
        params={"levels": list(args.push_levels)},
    )
    if args.no_obs_noise:
        cfg.observations.policy.enable_corruption = False
    env = gym.make(args.task, cfg=cfg)
    return RslRlVecEnvWrapper(env), n_envs


def load_policy(env, ckpt):
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device
    try:  # newer Isaac Lab versions convert old config layouts for newer rsl_rl
        import importlib.metadata as metadata

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    except Exception:
        pass
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    return runner.get_inference_policy(device=env.unwrapped.device)


@torch.inference_mode()
def rollout(env, policy, n_envs):
    """Run one episode per env. Returns per-env tensors."""
    u = env.unwrapped
    dev = u.device
    dt = u.step_dt
    max_steps = int(args.episode_s / dt) + 10
    robot = u.scene["robot"]

    obs = _first(env.reset())
    done_first = torch.zeros(n_envs, dtype=torch.bool, device=dev)
    fell = torch.zeros_like(done_first)
    steps = torch.zeros(n_envs, device=dev)       # steps spent in the episode (incl. last)
    ok_steps = torch.zeros(n_envs, device=dev)    # steps with a valid (non-reset) state for error stats
    lin_sum = torch.zeros(n_envs, device=dev)
    yaw_sum = torch.zeros(n_envs, device=dev)

    for _ in range(max_steps):
        actions = policy(obs)
        obs, _, dones, extras = env.step(actions)
        dones = dones.bool()
        if "time_outs" in extras:
            time_out = extras["time_outs"].bool()
        else:
            time_out = u.termination_manager.time_outs.bool()

        active = ~done_first
        ok = active & ~dones                      # env still in its first episode and not just reset
        cmd = u.command_manager.get_command("base_velocity")
        e_lin = torch.norm(robot.data.root_lin_vel_b[:, :2] - cmd[:, :2], dim=1)
        e_yaw = (robot.data.root_ang_vel_b[:, 2] - cmd[:, 2]).abs()
        lin_sum += torch.where(ok, e_lin, torch.zeros_like(e_lin))
        yaw_sum += torch.where(ok, e_yaw, torch.zeros_like(e_yaw))
        ok_steps += ok.float()
        steps += active.float()

        newly = active & dones
        fell |= newly & ~time_out
        done_first |= newly
        if bool(done_first.all()):
            break

    return {
        "finished": done_first.cpu(),
        "fell": fell.cpu(),
        "ep_s": (steps * dt).cpu(),
        "lin_err": (lin_sum / ok_steps.clamp(min=1)).cpu(),
        "yaw_err": (yaw_sum / ok_steps.clamp(min=1)).cpu(),
    }


def summarize(name, res, n_envs):
    group, seed = name.split("_s")[0], int(name.split("_s")[1])
    L = len(args.push_levels)
    level_idx = torch.arange(n_envs) % L
    rows = []
    for j, push in enumerate(args.push_levels):
        m = level_idx == j
        n = int(m.sum())
        fell = res["fell"][m]
        survived = ~fell
        lin = res["lin_err"][m]
        success = survived & (lin < args.success_err)
        ep_time_min = float(res["ep_s"][m].sum()) / 60.0
        rows.append(
            {
                "policy": name, "group": group, "seed": seed,
                "friction": args.friction, "push": push, "n_episodes": n,
                "success_rate": float(success.float().mean()),
                "survival_rate": float(survived.float().mean()),
                "falls": int(fell.sum()),
                "falls_per_min": int(fell.sum()) / max(ep_time_min, 1e-9),
                "lin_err": float(lin.mean()),
                "lin_err_std": float(lin.std()),
                "lin_err_survivors": float(lin[survived].mean()) if survived.any() else float("nan"),
                "yaw_err": float(res["yaw_err"][m].mean()),
                "ep_len_s": float(res["ep_s"][m].mean()),
            }
        )
    return rows


def main():
    env, n_envs = build_env(len(args.push_levels))
    all_rows = []
    for spec in args.policies:
        name, ckpt = spec.split("=", 1)
        print(f"[eval] mu={args.friction}  policy={name}  ckpt={ckpt}")
        policy = load_policy(env, ckpt)
        res = rollout(env, policy, n_envs)
        if not bool(res["finished"].all()):
            print("[warn] some episodes did not finish within the step budget")
        rows = summarize(name, res, n_envs)
        for r in rows:
            print(f"   push={r['push']:.2f}  success={r['success_rate']:.2f}  falls={r['falls']:3d}  lin_err={r['lin_err']:.3f}")
        all_rows += rows

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"[eval] wrote {args.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
