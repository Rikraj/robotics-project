#!/usr/bin/env python3
"""Aggregate evaluation CSVs + TensorBoard logs into figures and tables.

  python analyze.py --results results --logs $ISAACLAB/logs/rsl_rl --figures figures

Outputs (in --figures / --results):
  fig1_training_curves.png   fig2_push_sweep.png   fig3_friction_sweep.png
  fig4_heatmaps.png          fig5_regime_summary.png
  summary_by_regime.csv/.md  summary_by_condition.csv
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---- keep in sync with g1_dr/env_cfgs.py ----
TRAIN_FRICTION = (0.4, 1.2)   # static friction range seen by the DR policy
TRAIN_PUSH = 0.6              # m/s per axis
NOMINAL_FRICTION = 0.8

COLORS = {"baseline": "#0072B2", "dr": "#D55E00"}  # Okabe-Ito blue / vermillion (colour-blind safe)
LABELS = {"baseline": "Baseline (fixed conditions)", "dr": "Domain randomization"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


# ------------------------------------------------------------------ data
def load_eval(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "eval_mu*.csv")))
    if not files:
        raise SystemExit(f"no eval_mu*.csv in {results_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["friction"] = df["friction"].round(3)
    df["push"] = df["push"].round(3)
    df["in_dist"] = df["friction"].between(*TRAIN_FRICTION) & (df["push"] <= TRAIN_PUSH + 1e-9)
    df["nominal"] = (df["friction"] == NOMINAL_FRICTION) & (df["push"] == 0.0)
    return df


def seed_stats(df, by, metric):
    """mean/std over seeds of `metric`, grouped by `by` + group."""
    g = df.groupby(by + ["group", "seed"])[metric].mean().reset_index()
    s = g.groupby(by + ["group"])[metric].agg(["mean", "std", "count"]).reset_index()
    return g, s


# ------------------------------------------------------------------ figures
def line_panel(ax, df, xcol, metric, ylabel, xlabel, cond_mask=None):
    d = df if cond_mask is None else df[cond_mask]
    per_seed, stats = seed_stats(d, [xcol], metric)
    for grp in ("baseline", "dr"):
        ps = per_seed[per_seed.group == grp]
        for _, s in ps.groupby("seed"):
            ax.plot(s[xcol], s[metric], color=COLORS[grp], alpha=0.25, lw=0.8)
        st = stats[stats.group == grp].sort_values(xcol)
        ax.plot(st[xcol], st["mean"], color=COLORS[grp], lw=2, marker="o", ms=4, label=LABELS[grp])
        ax.fill_between(st[xcol], st["mean"] - st["std"].fillna(0), st["mean"] + st["std"].fillna(0),
                        color=COLORS[grp], alpha=0.15, lw=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


METRICS = [("success_rate", "Success rate"), ("falls_per_min", "Falls per simulated minute"),
           ("lin_err", "Velocity-tracking error (m/s)")]


def fig_push_sweep(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    sub = df[df.friction == NOMINAL_FRICTION]
    for ax, (m, yl) in zip(axes, METRICS):
        line_panel(ax, sub, "push", m, yl, "Push strength (max Δv per axis, m/s)")
        ax.axvspan(0, TRAIN_PUSH, color="grey", alpha=0.08)
        ax.axvline(TRAIN_PUSH, color="grey", ls="--", lw=0.8)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="lower left")
    fig.suptitle(f"Push-strength sweep at nominal friction (μ={NOMINAL_FRICTION}); grey = pushes seen in DR training; "
                 "thin lines = individual seeds, band = ±1 std over seeds", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_friction_sweep(df, out):
    pushes = [0.0, 0.5]
    fig, axes = plt.subplots(len(pushes), 3, figsize=(13, 3.6 * len(pushes)))
    for r, p in enumerate(pushes):
        sub = df[df.push == p]
        for ax, (m, yl) in zip(axes[r], METRICS):
            line_panel(ax, sub, "friction", m, yl, "Static friction μ")
            ax.axvspan(*TRAIN_FRICTION, color="grey", alpha=0.08)
            ax.set_title(f"push = {p} m/s", loc="left", fontsize=9)
        axes[r][0].set_ylim(-0.02, 1.02)
    axes[0][0].legend(loc="lower right")
    fig.suptitle("Friction sweep; grey = friction range seen in DR training", fontsize=9, y=1.0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_heatmaps(df, out):
    g = df.groupby(["group", "friction", "push"])["success_rate"].mean().reset_index()
    fr = sorted(df.friction.unique())
    pu = sorted(df.push.unique())

    def grid(grp):
        s = g[g.group == grp].pivot(index="friction", columns="push", values="success_rate")
        return s.reindex(index=fr, columns=pu).values

    base, dr = grid("baseline"), grid("dr")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, data, title, cmap, vmin, vmax in [
        (axes[0], base, "Baseline: success rate", "viridis", 0, 1),
        (axes[1], dr, "Domain randomization: success rate", "viridis", 0, 1),
        (axes[2], dr - base, "Difference (DR − baseline)", "RdBu", -1, 1),
    ]:
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pu)), [f"{p:g}" for p in pu])
        ax.set_yticks(range(len(fr)), [f"{f:g}" for f in fr])
        ax.set_xlabel("Push strength (m/s)")
        ax.set_ylabel("Static friction μ")
        ax.set_title(title)
        ax.grid(False)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isfinite(v):
                    dark = (abs(v) > 0.55) if cmap == "RdBu" else (v < 0.5)
                    ax.text(j, i, f"{v:+.2f}" if cmap == "RdBu" else f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if dark else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Success rate over the friction × push grid (mean over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def regime_table(df):
    d = df.copy()
    d["regime"] = np.where(d.nominal, "nominal", np.where(d.in_dist, "in-distribution (excl. nominal)", "held-out"))
    rows = []
    for (grp, seed, reg), x in d.groupby(["group", "seed", "regime"]):
        rows.append({"group": grp, "seed": seed, "regime": reg, "success_rate": x.success_rate.mean(),
                     "survival_rate": x.survival_rate.mean(), "falls_per_min": x.falls_per_min.mean(),
                     "lin_err": x.lin_err.mean(), "yaw_err": x.yaw_err.mean()})
    per_seed = pd.DataFrame(rows)
    agg = per_seed.groupby(["regime", "group"]).agg(
        n_seeds=("seed", "nunique"),
        success_mean=("success_rate", "mean"), success_std=("success_rate", "std"),
        survival_mean=("survival_rate", "mean"), survival_std=("survival_rate", "std"),
        falls_per_min_mean=("falls_per_min", "mean"), falls_per_min_std=("falls_per_min", "std"),
        lin_err_mean=("lin_err", "mean"), lin_err_std=("lin_err", "std"),
        yaw_err_mean=("yaw_err", "mean"), yaw_err_std=("yaw_err", "std"),
    ).reset_index()
    return per_seed, agg


def fig_regime(per_seed, out):
    regimes = ["nominal", "in-distribution (excl. nominal)", "held-out"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (m, yl) in zip(axes, METRICS):
        for k, grp in enumerate(("baseline", "dr")):
            xs, ms, ss = [], [], []
            for i, reg in enumerate(regimes):
                v = per_seed[(per_seed.group == grp) & (per_seed.regime == reg)][m].values
                if len(v) == 0:
                    continue
                x = i + (k - 0.5) * 0.36
                ax.scatter([x] * len(v), v, color=COLORS[grp], alpha=0.5, s=14, zorder=3)
                xs.append(x); ms.append(v.mean()); ss.append(v.std(ddof=1) if len(v) > 1 else 0)
            ax.bar(xs, ms, width=0.34, color=COLORS[grp], alpha=0.8, yerr=ss, capsize=3, label=LABELS[grp])
        ax.set_xticks(range(len(regimes)), ["nominal", "in-distribution", "held-out"])
        ax.set_ylabel(yl)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower left")
    fig.suptitle("Mean over conditions in each regime; bars = mean over seeds, dots = individual seeds, "
                 "error bars = ±1 std over seeds", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ training curves
def load_tb_scalar(run_dir, tag_hint):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    tag = next((t for t in tags if t == tag_hint), None) or next((t for t in tags if tag_hint.split("/")[-1] in t), None)
    if tag is None:
        return None
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def fig_training(logs, out):
    panels = [("Train/mean_reward", "Mean episode reward"), ("Train/mean_episode_length", "Mean episode length (steps)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    any_data = False
    for ax, (tag, yl) in zip(axes, panels):
        for grp, exp in (("baseline", "g1_flat_baseline"), ("dr", "g1_flat_dr")):
            curves = []
            for run in sorted(glob.glob(os.path.join(logs, exp, f"*_{grp}_s*"))):
                r = load_tb_scalar(run, tag)
                if r is not None:
                    curves.append(r)
            if not curves:
                continue
            any_data = True
            grid = np.linspace(0, min(c[0].max() for c in curves), 300)
            ys = np.stack([np.interp(grid, c[0], pd.Series(c[1]).rolling(10, min_periods=1).mean().values) for c in curves])
            m, s = ys.mean(0), ys.std(0)
            ax.plot(grid, m, color=COLORS[grp], lw=2, label=f"{LABELS[grp]} (n={len(curves)})")
            ax.fill_between(grid, m - s, m + s, color=COLORS[grp], alpha=0.15, lw=0)
        ax.set_xlabel("PPO iteration")
        ax.set_ylabel(yl)
    if not any_data:
        print("[warn] no TensorBoard data found; skipping fig1")
        plt.close(fig)
        return
    axes[0].legend(loc="lower right")
    fig.suptitle("Training curves (each run measured under its OWN training conditions, so DR rewards are not directly "
                 "comparable to baseline)", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--logs", default=None, help="Isaac Lab logs/rsl_rl folder (for training curves)")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args()
    os.makedirs(a.figures, exist_ok=True)

    df = load_eval(a.results)
    print(f"loaded {len(df)} rows; policies: {sorted(df.policy.unique())}")

    fig_push_sweep(df, os.path.join(a.figures, "fig2_push_sweep.png"))
    fig_friction_sweep(df, os.path.join(a.figures, "fig3_friction_sweep.png"))
    fig_heatmaps(df, os.path.join(a.figures, "fig4_heatmaps.png"))
    per_seed, agg = regime_table(df)
    fig_regime(per_seed, os.path.join(a.figures, "fig5_regime_summary.png"))
    if a.logs:
        fig_training(a.logs, os.path.join(a.figures, "fig1_training_curves.png"))

    agg.to_csv(os.path.join(a.results, "summary_by_regime.csv"), index=False)
    per_seed.to_csv(os.path.join(a.results, "summary_by_regime_per_seed.csv"), index=False)
    cond = df.groupby(["group", "friction", "push"]).agg(
        n_seeds=("seed", "nunique"), success_rate=("success_rate", "mean"), survival_rate=("survival_rate", "mean"),
        falls_per_min=("falls_per_min", "mean"), lin_err=("lin_err", "mean"), yaw_err=("yaw_err", "mean")).reset_index()
    cond.to_csv(os.path.join(a.results, "summary_by_condition.csv"), index=False)

    md = agg.copy()
    for m in ("success", "survival", "falls_per_min", "lin_err", "yaw_err"):
        md[m] = md[f"{m}_mean"].map("{:.3f}".format) + " ± " + md[f"{m}_std"].fillna(0).map("{:.3f}".format)
    md = md[["regime", "group", "n_seeds", "success", "survival", "falls_per_min", "lin_err", "yaw_err"]]
    with open(os.path.join(a.results, "summary_by_regime.md"), "w") as f:
        f.write(md.to_markdown(index=False) if hasattr(md, "to_markdown") and _has_tabulate() else md.to_string(index=False))
    print(md.to_string(index=False))
    print(f"figures -> {a.figures}")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
