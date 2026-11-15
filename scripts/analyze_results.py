#!/usr/bin/env python3
"""Aggregate completed evaluation CSVs and generate report-ready figures."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
PROCESSED = ROOT / "results" / "processed"
FIGURES = ROOT / "results" / "figures"

REQUIRED = {
    "policy", "training_seed", "friction", "push_strength", "success", "fell",
    "survival_seconds",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW, help="CSV file or directory containing evaluation CSVs")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    parser.add_argument("--episodes-per-condition", type=int, default=100)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow incomplete CSVs for debugging only; never use partial output as final results.",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("evaluation_*.csv"))
    if not files:
        raise SystemExit(f"No evaluation CSVs found in {input_path}. Run the evaluation stage first.")

    if args.episodes_per_condition < 1:
        raise SystemExit("--episodes-per-condition must be positive")

    expected_names = {
        f"evaluation_{policy}_seed{seed}.csv"
        for policy in ("baseline", "randomized")
        for seed in (11, 22, 33)
    }
    found_names = {f.name for f in files}
    if not args.allow_partial and found_names != expected_names:
        missing_files = sorted(expected_names - found_names)
        unexpected_files = sorted(found_names - expected_names)
        raise SystemExit(
            "Final analysis requires all six policy/seed CSV files. "
            f"Missing: {missing_files}; unexpected: {unexpected_files}. "
            "Use --allow-partial only for exploratory debugging."
        )

    frames = []
    for csv_path in files:
        frame = pd.read_csv(csv_path)
        missing_frame = REQUIRED - set(frame.columns)
        if missing_frame:
            raise SystemExit(f"{csv_path.name} is missing columns: {sorted(missing_frame)}")
        if frame.empty:
            raise SystemExit(f"Evaluation file contains only a header/no episode rows: {csv_path}")
        if not args.allow_partial:
            match = re.fullmatch(r"evaluation_(baseline|randomized)_seed(11|22|33)\.csv", csv_path.name)
            if match is None:
                raise SystemExit(f"Unexpected evaluation filename: {csv_path.name}")
            expected_policy, expected_seed = match.group(1), int(match.group(2))
            if set(frame["policy"].dropna().unique()) != {expected_policy}:
                raise SystemExit(f"Policy column does not match filename: {csv_path}")
            if set(pd.to_numeric(frame["training_seed"], errors="raise").unique()) != {expected_seed}:
                raise SystemExit(f"training_seed column does not match filename: {csv_path}")
            expected_rows = 6 * 3 * args.episodes_per_condition
            if len(frame) != expected_rows:
                raise SystemExit(
                    f"{csv_path.name} has {len(frame)} rows; expected {expected_rows} "
                    "(all 18 friction/push conditions completed)."
                )
            counts = frame.groupby(["friction", "push_strength"]).size()
            expected_conditions = {
                (friction, push)
                for friction in (0.4, 0.5, 0.875, 1.0, 1.25, 1.4)
                for push in (0.0, 0.25, 0.5)
            }
            actual_conditions = {
                (float(friction), float(push))
                for friction, push in counts.index.tolist()
            }
            if actual_conditions != expected_conditions or not (counts == args.episodes_per_condition).all():
                raise SystemExit(
                    f"{csv_path.name} does not contain exactly {args.episodes_per_condition} episodes "
                    "for each of the 18 friction/push conditions."
                )
            if "episode_index" in frame.columns:
                duplicates = frame.duplicated(["friction", "push_strength", "episode_index"]).any()
                if duplicates:
                    raise SystemExit(f"Duplicate episode_index found in {csv_path.name}")
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    missing = REQUIRED - set(data.columns)
    if missing:
        raise SystemExit(f"Input data is missing required columns: {sorted(missing)}")
    if data.empty:
        raise SystemExit("Evaluation files contain only headers; no episodes were recorded.")
    if not set(data["policy"].dropna().unique()).issubset({"baseline", "randomized"}):
        raise SystemExit("policy column must contain only baseline/randomized")

    error_col = "mean_planar_velocity_error_m_s" if "mean_planar_velocity_error_m_s" in data.columns else "mean_abs_velocity_error"
    data["success"] = pd.to_numeric(data["success"], errors="raise").astype(int)
    data["fell"] = pd.to_numeric(data["fell"], errors="raise").astype(int)
    data[error_col] = pd.to_numeric(data[error_col], errors="raise")

    out_dir = args.output_dir.expanduser().resolve()
    fig_dir = args.figure_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    consolidated_path = out_dir / "evaluation_all_episodes.csv"
    data.to_csv(consolidated_path, index=False)

    group_cols = ["policy", "friction", "push_strength"]
    summary = (
        data.groupby(group_cols, as_index=False)
        .agg(
            episodes=("success", "size"),
            success_rate=("success", "mean"),
            fall_frequency=("fell", "mean"),
            mean_velocity_error_m_s=(error_col, "mean"),
            std_velocity_error_m_s=(error_col, "std"),
            mean_survival_seconds=("survival_seconds", "mean"),
        )
    )
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    # Preserve training-seed replicates; pair baseline and randomized runs with the same seed.
    seed_cols = ["training_seed", "policy", "friction", "push_strength"]
    per_seed = (
        data.groupby(seed_cols, as_index=False)
        .agg(
            episodes=("success", "size"),
            success_rate=("success", "mean"),
            fall_frequency=("fell", "mean"),
            mean_velocity_error_m_s=(error_col, "mean"),
            mean_survival_seconds=("survival_seconds", "mean"),
        )
    )
    per_seed.to_csv(out_dir / "metrics_by_training_seed.csv", index=False)
    pivot = per_seed.pivot_table(
        index=["training_seed", "friction", "push_strength"],
        columns="policy",
        values=["success_rate", "fall_frequency", "mean_velocity_error_m_s"],
    )
    delta = pd.DataFrame(index=pivot.index).reset_index()
    for metric in ["success_rate", "fall_frequency", "mean_velocity_error_m_s"]:
        if (metric, "baseline") in pivot and (metric, "randomized") in pivot:
            delta[f"randomized_minus_baseline_{metric}"] = (
                pivot[(metric, "randomized")] - pivot[(metric, "baseline")]
            ).to_numpy()
    delta.to_csv(out_dir / "paired_seed_differences.csv", index=False)

    # Plot 1: success rate vs friction at zero push.
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    nominal = summary[np.isclose(summary["push_strength"], 0.0)]
    for policy in ["baseline", "randomized"]:
        s = nominal[nominal["policy"] == policy].sort_values("friction")
        if not s.empty:
            ax.plot(s["friction"], s["success_rate"], marker="o", linewidth=2, label=policy)
    ax.axvspan(0.4, 0.5, alpha=0.10, label="below DR range")
    ax.axvspan(1.25, 1.4, alpha=0.10, label="above DR range")
    ax.set(xlabel="Foot-contact friction coefficient (fixed during each test)", ylabel="Episode success rate", ylim=(-0.03, 1.03), title="Success rate vs friction (no pushes)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "success_vs_friction.png", dpi=180)
    plt.close(fig)

    # Plot 2: tracking error vs push strength at nominal friction.
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    nominal = summary[np.isclose(summary["friction"], 1.0)]
    for policy in ["baseline", "randomized"]:
        s = nominal[nominal["policy"] == policy].sort_values("push_strength")
        if not s.empty:
            ax.plot(s["push_strength"], s["mean_velocity_error_m_s"], marker="o", linewidth=2, label=policy)
    ax.set(xlabel="Maximum x/y root-velocity perturbation (m/s)", ylabel="Mean planar velocity error (m/s)", title="Velocity-tracking error vs push strength (nominal friction)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "velocity_error_vs_push.png", dpi=180)
    plt.close(fig)

    # Plot 3: fall frequency vs push strength at nominal friction.
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    nominal = summary[np.isclose(summary["friction"], 1.0)]
    for policy in ["baseline", "randomized"]:
        s = nominal[nominal["policy"] == policy].sort_values("push_strength")
        if not s.empty:
            ax.plot(s["push_strength"], s["fall_frequency"], marker="o", linewidth=2, label=policy)
    ax.set(xlabel="Maximum x/y root-velocity perturbation (m/s)", ylabel="Fall frequency", ylim=(-0.03, 1.03), title="Fall frequency vs push strength (nominal friction)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fall_frequency_vs_push.png", dpi=180)
    plt.close(fig)

    # Plot 4: two heatmaps, one per policy, for success over full condition grid.
    policies_present = [p for p in ["baseline", "randomized"] if p in set(summary["policy"])]
    fig, axes = plt.subplots(1, len(policies_present), figsize=(6.4 * len(policies_present), 4.8), squeeze=False)
    for ax, policy in zip(axes[0], policies_present):
        s = summary[summary["policy"] == policy]
        table = s.pivot(index="push_strength", columns="friction", values="success_rate").sort_index(ascending=True)
        im = ax.imshow(table.to_numpy(), aspect="auto", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(np.arange(len(table.columns)), [f"{x:g}" for x in table.columns])
        ax.set_yticks(np.arange(len(table.index)), [f"{y:g}" for y in table.index])
        ax.set_xlabel("Foot-contact friction coefficient")
        ax.set_ylabel("Push velocity limit (m/s)")
        ax.set_title(f"{policy.title()} - success rate")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                value = table.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Episode success rate", shrink=0.85)
    fig.tight_layout()
    fig.savefig(fig_dir / "success_heatmap_by_policy.png", dpi=180)
    plt.close(fig)

    print(f"Consolidated episodes: {consolidated_path}")
    print(f"Summary table: {out_dir / 'metrics_summary.csv'}")
    print(f"Figures: {fig_dir}")
    print("IMPORTANT: interpret seed-level variation; do not report a single pooled mean without the three training seeds.")


if __name__ == "__main__":
    main()
