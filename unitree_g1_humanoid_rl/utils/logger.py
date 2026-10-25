"""
Logging and Plotting module for tracking learning curves, tracking errors, and benchmark statistics.
"""

import os
import csv
import matplotlib.pyplot as plt

class MetricsLogger:
    def init(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_file = os.path.join(log_dir, "training_metrics.csv")

        with open(self.csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "Total_Timesteps", "Mean_Reward", "Tracking_Error_Vx", "Fall_Frequency"])

    def log_step(self, iteration, timesteps, reward, error_vx, fall_freq):
        with open(self.csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([iteration, timesteps, reward, error_vx, fall_freq])

    def generate_midterm_plots(self, save_path="logs/baseline_reward_curve.png"):
        """Reads CSV and plots midterm training curves."""
        iterations = []
        rewards = []
        errors = []

        if not os.path.exists(self.csv_file):
            print("No CSV data found to plot.")
            return

        with open(self.csv_file, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iterations.append(int(row["Iteration"]))
                rewards.append(float(row["Mean_Reward"]))
                errors.append(float(row["Tracking_Error_Vx"]))

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color = "tab:blue"
        ax1.set_xlabel("PPO Iterations")
        ax1.set_ylabel("Mean Episode Reward", color=color)
        ax1.plot(iterations, rewards, color=color, linewidth=2, label="Episode Reward")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, linestyle="--", alpha=0.6)

        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Tracking Error Vx (m/s)", color=color)
        ax2.plot(iterations, errors, color=color, linewidth=1.5, linestyle=":", label="Tracking Error")
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title("Unitree G1 PPO Baseline Training Performance (Midterm)")
        fig.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"[Logger] Midterm baseline performance plot saved to {save_path}")