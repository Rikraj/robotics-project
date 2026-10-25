"""
Helper functions for vector reward term computation (Gaussian tracking kernels, stability penalties).
"""

import torch

def compute_tracking_kernel(target: torch.Tensor, current: torch.Tensor, std: float = 0.25) -> torch.Tensor:
    """
    Exponential tracking reward: exp(- || target - current ||^2 / std)
    Returns value close to 1.0 when error is near 0.
    """
    error = torch.sum(torch.square(target - current), dim=-1)
    return torch.exp(-error / (std**2))

    def compute_action_smoothness_penalty(actions: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        """
        Penalizes first difference between successive action vectors to prevent high-frequency jitter.
        """
        return torch.sum(torch.square(actions - prev_actions), dim=-1)