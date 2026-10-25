# Domain randomization for robust RL humanoid walking (Unitree G1, Isaac Lab)

**Abstract.** (3–4 sentences: question, method, headline numbers from `summary_by_regime.md`, main caveat.)

## 1. Introduction
Motivation (humanoid RL locomotion, sim-to-real gap, why friction and pushes). Research question: *Does training with friction variation and random pushes improve humanoid walking robustness compared with training under fixed conditions?* Scope: simulation only, flat ground, low speed, no new RL algorithm.

## 2. Method
- Environment: Isaac Lab G1 flat velocity task, command 0–0.8 m/s forward, 20 s episodes, 50 Hz control.
- Controllers: feed-forward PPO (rsl_rl), identical network/reward/1500 iterations/4096 envs, seeds 1–3.
- Baseline: friction static 0.8 / dynamic 0.6, no pushes. DR: friction static 0.4–1.2, dynamic 0.3–0.9, pushes ±0.6 m/s every 6–10 s. Stock mass randomization removed from both.
- Evaluation: friction × push grid (Step 5 of README), 100 episodes per cell, held-out = outside training ranges. Metrics: success rate, survival, falls/min, velocity-tracking error.

## 3. Results
- Fig. 1 training curves; Fig. 2 push sweep; Fig. 3 friction sweep; Fig. 4 heatmaps; Fig. 5 regime summary.
- Table: `results/summary_by_regime.md` (mean ± std over seeds).
- Videos: list clips and conditions.

## 4. Discussion
Nominal cost of DR? Gain in-distribution? Gain on held-out friction/push? Which factor seems to matter (note: factors were not ablated separately)? Failure modes seen in videos.

## 5. Limitations
Simulation only; 3 seeds; one robot and one terrain; evaluation pushes are velocity kicks, not force impulses; success threshold (0.25 m/s) is a choice; no sim-to-real claims, no safety guarantees.

## 6. Reproducibility
Code and configs: this repository. Isaac Lab version, Isaac Sim version, GPU, seeds, commands from README Steps 2–7.
