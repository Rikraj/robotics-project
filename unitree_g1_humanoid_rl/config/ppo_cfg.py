"""
PPO (Proximal Policy Optimization) algorithm and network architecture parameters.
"""

from dataclasses import dataclass

@dataclass
class PPOActorCriticCfg:
    actor_hidden_dims: list = None
    critic_hidden_dims: list = None
    activation: str = "elu"  # Exponential Linear Unit for smooth control gradients

    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [512, 256, 128]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [512, 256, 128]

@dataclass
class PPOTrainerCfg:
    algorithm: str = "PPO"
    network: PPOActorCriticCfg = PPOActorCriticCfg()

    # Training Hyperparameters
    learning_rate: float = 1.0e-3
    learning_rate_schedule: str = "adaptive"  # Adaptive KL-based LR decay
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2               # PPO surrogate clipping threshold epsilon
    gamma: float = 0.99                   # Discount factor
    lam: float = 0.95                     # GAE (Generalized Advantage Estimation) lambda
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01            # Small entropy coefficient for exploration
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0

    # Parallelization & Batch Size
    num_envs: int = 4096
    num_steps_per_env: int = 24            # Horizon length per env per iteration
    total_timesteps: int = 15_000_000     # ~1500 iterations