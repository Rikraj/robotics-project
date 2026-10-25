"""
Configuration file defining observation space, action space, joint limits,
and reward scale weights for Unitree G1 humanoid walking environment.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class G1RobotCfg:
    num_actions: int = 29          # 29 active motorized joints (legs, waist, arms)
    num_observations: int = 97     # Includes base velocity, orientation, joint positions/velocities, command
    default_joint_angles: dict = None

    def __post_init__(self):
        # Default standing pose (radians)
        self.default_joint_angles = {
            "hip_pitch": -0.2,
            "knee_pitch": 0.4,
            "ankle_pitch": -0.2,
            "hip_roll": 0.0,
            "hip_yaw": 0.0,
            "ankle_roll": 0.0,
            "waist_yaw": 0.0,
            "shoulder_pitch": 0.2,
            "elbow_pitch": 0.6
        }

@dataclass
class RewardScalesCfg:
    # Task tracking rewards
    tracking_lin_vel: float = 1.5      # Reward tracking command velocity v_x, v_y
    tracking_ang_vel: float = 0.8      # Reward tracking command yaw rate w_z

    # Base stability penalties
    lin_vel_z: float = -2.0           # Penalize vertical base bouncing
    ang_vel_xy: float = -0.05         # Penalize roll/pitch wobbling
    orientation: float = -1.0         # Penalize deviation from upright trunk vector

    # Energy & smoothness penalties
    action_rate: float = -0.01        # Penalize smooth action differences (jerky control)
    torques: float = -0.0001          # Penalize high joint motor torques
    joint_acc: float = -2.5e-7        # Penalize rapid joint acceleration
    dof_pos_limits: float = -10.0     # Penalize joint limit violations

    # Contact penalties
    feet_air_time: float = 0.25       # Encourage rhythmic gait (air time reward)
    undesired_contacts: float = -1.0  # Penalize thigh/trunk contact with floor

@dataclass
class G1EnvCfg:
    env_name: str = "Unitree-G1-Flat-Walking-v0"
    robot: G1RobotCfg = G1RobotCfg()
    rewards: RewardScalesCfg = RewardScalesCfg()

    # Simulation timing
    sim_dt: float = 0.005             # 200 Hz simulation physics step
    decimation: int = 4                # 50 Hz control policy frequency (dt = 0.02s)
    episode_length_s: float = 20.0     # Max episode duration (1000 steps at 50Hz)

    # Fixed Command Range (Baseline Nominal)
    lin_vel_x_range: tuple = (0.0, 1.0) # Target forward speed between 0.0 and 1.0 m/s
    lin_vel_y_range: tuple = (-0.2, 0.2)
    ang_vel_z_range: tuple = (-0.5, 0.5)

    # Nominal Physics Parameters (No Domain Randomization for Baseline)
    static_friction: float = 0.8
    dynamic_friction: float = 0.7
    push_interval_s: float = 0.0       # Disabled for baseline
    push_force_max: float = 0.0         # Disabled for baseline