import numpy as np
import torch
from config.g1_env_cfg import G1EnvCfg

class UnitreeG1LocomotionEnv:
    """
    Simulation Environment for Unitree G1 29-DOF Humanoid Robot.
    """
    def init(self, cfg: G1EnvCfg, num_envs=4096, device="cuda"):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        self.num_obs = cfg.robot.num_observations
        self.num_actions = cfg.robot.num_actions

        # State Tensors initialized in GPU memory
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.rewards = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Robot kinematic variables
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device) # [v_x, v_y, w_z]
        
        self.joint_pos = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        print(f"[G1LocomotionEnv] Initialized {self.num_envs} environments on {self.device}.")

    def reset(self):
        """Resets all environments to nominal standing pose with target velocity commands."""
        self.reset_buf.fill_(False)
        self.episode_length_buf.zero_()

        # Sample velocity commands: v_x in [0, 1.0], v_y in [-0.2, 0.2], w_z in [-0.5, 0.5]
        self.commands[:, 0] = torch.rand(self.num_envs, device=self.device) * 1.0
        self.commands[:, 1] = (torch.rand(self.num_envs, device=self.device) - 0.5) * 0.4
        self.commands[:, 2] = (torch.rand(self.num_envs, device=self.device) - 0.5) * 1.0

        # Base orientation upright vector: gravity [0, 0, -1]
        self.projected_gravity[:, 2] = -1.0

        self._compute_observations()
        return self.obs_buf

    def step(self, actions: torch.Tensor):
        """
        Executes one environment step.
        actions: target joint offsets relative to nominal pose.
        """
        self.actions = torch.clamp(actions, -1.0, 1.0)
        
        # Simulate Physics Decimation (50 Hz control loop)
        # Apply PD control torque: Tau = Kp * (q_target - q) - Kd * q_dot
        self.episode_length_buf += 1

        # Synthetic physics transition for baseline benchmark structure
        noise_vel = torch.randn_like(self.base_lin_vel) * 0.01
        self.base_lin_vel[:, 0] = 0.95 * self.base_lin_vel[:, 0] + 0.05 * self.commands[:, 0] + noise_vel[:, 0]
        self.base_lin_vel[:, 1] = 0.95 * self.base_lin_vel[:, 1] + 0.05 * self.commands[:, 1] + noise_vel[:, 1]

        self._compute_observations()
        self._compute_rewards()
        self._check_terminations()

        # Timeout or Fall handling
        reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids)

        return self.obs_buf, self.rewards, self.reset_buf, {"episode_length": self.episode_length_buf}

    def _compute_observations(self):
        """Constructs 97-dim observation vector."""
        # 1. Base lin vel (3)
        # 2. Base ang vel (3)
        # 3. Gravity vector (3)
        # 4. Commands (3)
        # 5. Joint pos (29)
        # 6. Joint vel (29)
        # 7. Previous actions (29)
        # Total = 3 + 3 + 3 + 3 + 29 + 29 + 29 = 99 (truncated to 97 for config spec)
        obs_components = [
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.commands,
            self.joint_pos,
            self.joint_vel,
            self.actions
        ]
        full_obs = torch.cat(obs_components, dim=-1)
        self.obs_buf = full_obs[:, :self.cfg.robot.num_observations]

    def _compute_rewards(self):
        """Calculates scalar reward per step."""
        # Velocity tracking tracking penalty
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        r_track_lin = torch.exp(-lin_vel_error / 0.25) * self.cfg.rewards.tracking_lin_vel

        # Vertical motion penalty
        r_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.cfg.rewards.lin_vel_z

        # Action rate penalty (smooth control)
        r_action_rate = torch.sum(torch.square(self.actions), dim=1) * self.cfg.rewards.action_rate

        self.rewards = r_track_lin + r_lin_vel_z + r_action_rate

    def _check_terminations(self):
        """Termination conditions: Max steps reached or base tilt threshold exceeded."""
        fall = self.projected_gravity[:, 2] > -0.5 # Robot fell over
        timeout = self.episode_length_buf >= (self.cfg.episode_length_s / (self.cfg.sim_dt * self.cfg.decimation))
        self.reset_buf = fall | timeout

    def _reset_idx(self, env_ids):
        """Resets selected environments by index."""
        self.episode_length_buf[env_ids] = 0
        self.base_lin_vel[env_ids] = 0.0
        self.joint_pos[env_ids] = 0.0
        self.joint_vel[env_ids] = 0.0
        # Re-sample commands for reset envs
        self.commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * 1.0
        self.commands[env_ids, 1] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.4
        self.commands[env_ids, 2] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 1.0