# Humanoid Walking Robustness under Domain Randomization

**Robot:** Unitree G1  
**Simulator/workflow:** Isaac Lab 3.0.0, Isaac Sim 6.2 (optional Kit visualization)  
**RL:** PPO with RSL-RL  
**Primary physics backend:** `newton_mjwarp`  
**Compute:** NVIDIA Brev GPU VM (recommended starting point: 1x NVIDIA L40S)

This project compares two policies trained with the same G1 flat-ground locomotion task and PPO setup:

1. **Baseline:** fixed foot-contact friction coefficient 1.0; no interval pushes.
2. **Domain-randomized (DR):** foot-contact friction sampled from [0.5, 1.25] at environment startup; bounded horizontal root-velocity perturbations every 5 seconds, with x/y limits of +/-0.5 m/s.

The supplied implementation targets the official Isaac Lab 3.0.0 G1 flat task. It keeps the upstream observations, action space, reward terms, PPO agent configuration, simulation time step, and command distribution unchanged. It disables base-mass and base-CoM randomization in both conditions to avoid confounding the comparison.

> **Interpretation note:** the built-in G1 event randomizes material properties on the G1 foot-contact collision bodies (`.*_ankle_roll_link`). The ground plane's material is held at its task default (1.0); with the task's friction-combine configuration this changes the effective foot-ground contact coefficient. This is a practical contact-friction randomization, not a direct rewrite of the terrain mesh's material. In Isaac Lab's Newton backend, the material model uses a single friction coefficient; the dynamic-friction range and PhysX material bucket count are not operative. The push event sets the robot's root velocity to a bounded random value. It is a reproducible disturbance proxy, not a calibrated force measured in Newtons. These details must be stated in the report.

---

## 1. NVIDIA Brev setup

### 1.1 Create the VM

In Brev, create a **VM-mode** GPU instance. NVIDIA's current Isaac Sim Brev guide recommends selecting **1x NVIDIA L40S**. Open the Jupyter notebook/terminal for command-line work. If you need browser-based Isaac Sim streaming, expose ports **49100, 47998, and 8210 only to your own client IP**; do not expose them to `0.0.0.0/0`. Port 8210 is used by the documented web viewer route. Restrict access because a public streaming service is not an appropriate way to expose an unauthenticated simulation endpoint.

Run these checks in the Brev terminal:

```bash
nvidia-smi
uname -a
ldd --version | head -n 1
```

The Isaac Lab 3.0 source workflow uses Python 3.12 and a recent NVIDIA driver. The current install guide lists Linux driver 580.65.06 or newer for the CUDA 13.0 default stack. Confirm that the VM image meets the current Isaac Lab and Isaac Sim requirements before long training runs.

### 1.2 Install Isaac Lab 3.0.0

Use the upstream release branch rather than an arbitrary `main` revision:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone --branch release/3.0.0 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git \
  ~/workspace/robotics-project/IsaacLab
cd ~/workspace/robotics-project/IsaacLab

# RSL-RL is the selected PPO runner; uv resolves the source checkout environment.
uv run --extra rsl-rl isaaclab list_envs --show_presets
```

Isaac Lab 3.0 uses the unified `isaaclab train` / `isaaclab play` commands. The G1 flat task ID in this release is `Isaac-Velocity-Flat-G1` (do not add a `-v0` suffix unless you deliberately use a different release/task registry).

Check the task appears:

```bash
uv run isaaclab list_envs --show_presets | grep -i 'Velocity-Flat-G1'
```

If the task name is absent, stop and resolve the Isaac Lab version/task registry first. Do not silently switch task IDs: the checkpoint and evaluation scripts must use the exact same task.

### 1.3 Isaac Sim installation and role

Isaac Lab 3.0's default Newton workflow can run without launching Isaac Sim. For this project, use **Newton MuJoCo-Warp for training and quantitative evaluation** so that all six trained policies share one physics backend. Install the `isaacsim` and `video` extras when you are ready to use full Kit/Isaac Sim visualization and video recording:

```bash
cd ~/workspace/robotics-project/IsaacLab
uv run --extra isaacsim --extra rsl-rl --extra video isaaclab --help
```

This resolves the Isaac Sim 6.2 integration supported by this release. The initial Isaac Sim launch may take several minutes while extensions are initialized and may prompt for EULA acceptance. For non-interactive execution, NVIDIA documents `OMNI_KIT_ACCEPT_EULA=yes`.

**Do not train the baseline on Newton and the randomized policy on PhysX.** A physics-backend change would add another experimental variable. If you decide to use `physics=isaacsim_physx` instead, use it consistently for both policies and every evaluation condition, and re-run all training/evaluation.

### 1.4 Place this project beside Isaac Lab

Extract this ZIP so the directory structure is:

```text
~/workspace/robotics-project/
  IsaacLab/
  humanoid_dr_project/
```

If your directories differ, export the actual Isaac Lab path before running scripts:

```bash
export ISAACLAB_ROOT="$HOME/workspace/robotics-project/IsaacLab"
```

---

## 2. Understand the experiment configuration

The canonical settings are in `config/experiment.yaml`.

| Setting | Baseline | DR controller |
|---|---:|---:|
| Foot-contact static-friction range | [1.0, 1.0] | [0.5, 1.25] |
| Foot-contact dynamic-friction range | [1.0, 1.0] | [0.5, 1.25] |
| Interval pushes | Disabled | Every 5.0 s |
| x/y root velocity perturbation | None | +/-0.5 m/s per axis |
| PPO iterations | 1500 | 1500 |
| Parallel environments | 64 | 64 |
| Training seeds | 11, 22, 33 | 11, 22, 33 |
| Backend | Newton MuJoCo-Warp | Newton MuJoCo-Warp |

The G1 task includes other randomized terms in its shared configuration. The patch disables base mass and base CoM randomization for **both** groups. Keep all remaining environment and agent settings unchanged.

The evaluation grid is:

- Friction coefficients: `0.4, 0.5, 0.875, 1.0, 1.25, 1.4`.
- Push velocity limits: `0.0, 0.25, 0.5 m/s` per horizontal axis.
- 100 episodes per friction/push condition, per policy checkpoint.
- Fixed evaluation seed: 2027, reused for paired baseline/DR comparisons.

There are 6 training runs (2 training conditions x 3 seeds), 18 evaluation conditions per checkpoint, and 10,800 episode records in the complete evaluation matrix:

`2 policies x 3 training seeds x 6 friction values x 3 push strengths x 100 episodes = 10,800 episodes.`

The values 0.4 and 1.4 are held-out friction conditions outside the DR training interval [0.5, 1.25]. The other friction conditions test in-range and nominal performance.

---

## 3. Patch and smoke-test the G1 task

The patcher inserts a small, marked block into the upstream `G1FlatEnvCfg.__post_init__` after its parent configuration is initialized. It creates a one-time backup ending in `.humanoid_dr.bak`, is idempotent, and parses the modified Python file before writing it.

```bash
cd ~/workspace/robotics-project/humanoid_dr_project
python scripts/patch_g1_task.py \
  --isaaclab-root "$ISAACLAB_ROOT" --dry-run
python scripts/patch_g1_task.py \
  --isaaclab-root "$ISAACLAB_ROOT"
```

Review the actual source diff before proceeding:

```bash
cd "$ISAACLAB_ROOT"
git diff -- source/isaaclab_tasks
```

The patch selects the runtime mode from the environment variable `G1_DR_MODE`. The same patched checkout is used for both experiment conditions. Do not patch once per condition or leave a different code version between training runs.

Run the smoke test first:

```bash
cd ~/workspace/robotics-project/humanoid_dr_project
ISAACLAB_ROOT="$ISAACLAB_ROOT" bash scripts/run_smoke_test.sh
```

This checks task registration/reset, then runs 10 training iterations with 16 environments for each mode. It is a **software check only**; do not mix smoke-test checkpoints into the final statistics. Resolve import, GPU, task-registration, and simulation errors before full training.

---

## 4. Train the six PPO policies

Full training settings are 64 parallel environments, 1500 PPO iterations, and seeds 11, 22, and 33 for each condition.

```bash
cd ~/workspace/robotics-project/humanoid_dr_project
ISAACLAB_ROOT="$ISAACLAB_ROOT" \
  NUM_ENVS=64 MAX_ITERS=1500 SEEDS="11 22 33" \
  bash scripts/run_experiments.sh
```

The script runs these six jobs in sequence:

```text
baseline_seed11
baseline_seed22
baseline_seed33
randomized_seed11
randomized_seed22
randomized_seed33
```

Isaac Lab writes RSL-RL runs and TensorBoard logs under `IsaacLab/logs/rsl_rl/`. Find the checkpoint for each run (the final iteration checkpoint is typically named `model_1499.pt` for a 1500-iteration job):

```bash
find "$ISAACLAB_ROOT/logs/rsl_rl" -type f -name 'model_*.pt' | sort
```

Copy `logs/checkpoint_manifest.template.csv` to `logs/checkpoint_manifest.csv` and fill in the six exact checkpoint paths and iteration numbers. Each checkpoint must map unambiguously to its policy condition and training seed. Keep the final checkpoint even if you also select a best validation checkpoint. Do not select a checkpoint based on held-out test results.

Monitor training with TensorBoard in a second terminal:

```bash
cd "$ISAACLAB_ROOT"
uv run python -m tensorboard.main --logdir logs/rsl_rl --port 6006
```

Use the Brev notebook port-forwarding feature to access TensorBoard. Inspect reward, mean episode length, KL/loss terms, and throughput. Save the training environment metadata in `logs/training_environment.txt`.

If 64 environments exceed available VRAM, reduce `NUM_ENVS` for **both** training groups and all seeds. Report the actual value and rerun both groups using the same count. Do not quietly reduce only one condition.

---

## 5. Evaluate every checkpoint on the same test matrix

The evaluation script instantiates the same task and RSL-RL policy, sets a fixed friction condition and a fixed push condition, and collects episodes until it has 100 complete episode records for that condition. It writes CSV rows as evaluation progresses.

Pass the six model paths in this order: baseline seeds 11, 22, 33, then randomized seeds 11, 22, 33.

```bash
cd ~/workspace/robotics-project/humanoid_dr_project
ISAACLAB_ROOT="$ISAACLAB_ROOT" \
  bash scripts/run_evaluation_matrix.sh \
  /absolute/path/baseline_seed11/model_1499.pt \
  /absolute/path/baseline_seed22/model_1499.pt \
  /absolute/path/baseline_seed33/model_1499.pt \
  /absolute/path/randomized_seed11/model_1499.pt \
  /absolute/path/randomized_seed22/model_1499.pt \
  /absolute/path/randomized_seed33/model_1499.pt
```

Replace those example paths with the real paths. If a run has a different final checkpoint iteration, use that exact path. The evaluation runner uses a shared evaluation seed of 2027; each condition derives a deterministic condition seed from it. The same seed/grid must be used for both policy groups.

### Metric definitions

1. **Mean planar velocity-tracking error (m/s):** time average of the planar Euclidean norm between desired and measured base x/y velocity. The evaluation adapter reads `root_lin_vel_b` and the `base_velocity` command term. Lower is better.
2. **Fall frequency:** fraction of episodes ending in a non-timeout termination. Lower is better.
3. **Survival time (s):** number of control steps before termination multiplied by `sim.dt x decimation`.
4. **Episode success:** no fall, survival of at least 95% of the configured episode duration, and mean planar velocity error <= 0.35 m/s.

The success threshold is a project-defined operational criterion, not an Isaac Lab default or a certified safety bound. State it explicitly in the final report.

### Critical validation: confirm that the CSV is not empty

The previous plotting failure occurred because the raw evaluation CSV had no episode rows. This package's evaluator creates one CSV per policy/training seed and writes after each condition. After evaluation:

```bash
wc -l results/raw/evaluation_*.csv
head -n 3 results/raw/evaluation_baseline_seed11.csv
```

Each completed checkpoint file should contain **1801 lines**: one header plus 18 conditions x 100 episodes. If a file has only one line or fewer than 1,800 data rows, evaluation did not complete. The analysis script rejects header-only files, missing checkpoint CSVs, duplicate episodes, and incomplete condition groups by default. If the evaluator stops with an API mismatch, inspect the installed Isaac Lab 3.0 RSL-RL wrapper, particularly the `time_outs` field, and fix that adapter before treating any CSV as valid. Never fill an empty CSV with made-up rows.

---

## 6. Aggregate the data and create the plots

Install analysis packages in a normal Python environment (or use the Brev system Python if it has compatible packages):

```bash
python -m pip install -r requirements-analysis.txt
```

Run the final analysis (the default is 100 episodes per each of the 18 conditions in all six checkpoint CSVs):

```bash
python scripts/analyze_results.py
```

The analysis script checks that all six expected policy/seed CSV files are present, that each contains 1,800 episode rows, and that every friction/push condition has exactly 100 episodes. For a small exploratory/debug run only, pass `--allow-partial`; never use partial analysis output in the submitted report.

Outputs:

```text
results/processed/evaluation_all_episodes.csv
results/processed/metrics_summary.csv
results/processed/metrics_by_training_seed.csv
results/processed/paired_seed_differences.csv
results/figures/success_vs_friction.png
results/figures/velocity_error_vs_push.png
results/figures/fall_frequency_vs_push.png
results/figures/success_heatmap_by_policy.png
```

Use the seed-level CSV and paired differences in the report. The independent unit for training variability is the training seed, not each of the 10,800 episodes. Report the 3 seed-level values or mean +/- standard deviation across the three training seeds. Episodes within one trained policy are not independent training replicates.

---

## 7. Generate demonstration videos in Isaac Sim

First, verify that the checkpoint loads using the same task and backend:

```bash
cd ~/workspace/robotics-project/humanoid_dr_project
ISAACLAB_ROOT="$ISAACLAB_ROOT" \
  VIZ=kit bash scripts/play_policy.sh \
  /absolute/path/randomized_seed11/model_1499.pt randomized
```

The `kit` visualizer invokes the Isaac Sim/Kit stack. The script requests a 600-step video and the Isaac Lab video extra. Use `VIZ=newton_gl` if you want to use the Newton visualizer instead. Keep physics, observation, action, and task presets aligned with training. Isaac Lab documents both policy playback and video recording through `isaaclab play`.

Make at least two short videos:

- Baseline policy walking under nominal conditions.
- DR policy walking under a held-out friction condition and/or a 0.5 m/s push condition.

For a defensible visual comparison, use the same camera, episode length, command, and initial setup. The videos are illustrative; all quantitative conclusions must come from the evaluation CSVs.

Store them in `results/videos/` and include a short caption with policy, seed, friction, and push setting.

---

## 8. Final deliverable checklist

| Deliverable | Location / evidence |
|---|---|
| Source code and configuration | This project folder; include the patch diff and all scripts |
| Simulator/software versions | `logs/training_environment.txt`, Git commit, Isaac Lab/Sim versions |
| Six trained checkpoints | Completed `logs/checkpoint_manifest.csv` plus `.pt` files copied/backed up from IsaacLab logs |
| Training curves | TensorBoard event files or exported plots |
| Raw evaluation data | Six CSVs in `results/raw/`, 1800 episode rows each |
| Processed metrics | CSVs under `results/processed/` |
| Evaluation plots | Four PNGs under `results/figures/` |
| Demonstration videos | `results/videos/` |
| Final report | `report/` or the supplied report DOCX/PDF |
| Reproducibility metadata | Exact commands, commit IDs, config, seed-to-checkpoint mapping |

Before shutting down the Brev VM, copy the project, checkpoints, evaluation CSVs, TensorBoard logs, and videos to persistent storage. GPU VM local disks may not be durable after an instance is deleted.

---

## 9. Troubleshooting guide

**Task ID not found**  
Confirm that the checkout is the Isaac Lab 3.0.0 release and run `uv run isaaclab list_envs --show_presets`. The G1 flat task is part of the core velocity task registration for that release.

**The G1 patcher refuses to patch**  
Do not force a text substitution. Confirm that `source/isaaclab_tasks/.../g1/flat_env_cfg.py` contains class `G1FlatEnvCfg` and its `__post_init__` calls `super().__post_init__()`. If using a different release, port the patch to that release deliberately and document it.

**Training does not react to G1_DR_MODE**  
The patch reads `G1_DR_MODE` at config construction. Confirm it is set on the same command that launches `isaaclab train`. Check the patched source for the `HUMANOID_DR_PROJECT_PATCH` marker.

**Out of GPU memory**  
Try 32 environments for both groups and all seeds. Keep all other settings the same. If memory pressure occurs only during Kit video rendering, run training/evaluation headless and render videos separately.

**Evaluation CSV has only a header**  
The script did not finish the first test condition or did not receive terminal flags. Check the console traceback and the installed `RslRlVecEnvWrapper` output signature. The evaluator intentionally stops rather than inventing fall/time-out classifications when `infos['time_outs']` is absent.

**Checkpoint shape mismatch**  
Use the same task, registered RSL-RL agent, observation/action presets, and physics backend used during training. Do not load a checkpoint into a different Isaac Lab release or a changed task configuration.

**Brev streaming does not connect**  
Check that ports 49100, 47998, and 8210 are permitted only from the client IP. For long training runs, streaming is unnecessary; use headless simulation and TensorBoard instead.

---

## 10. Scope and limitations

This is a controlled simulation study, not a claim of sim-to-real transfer or guaranteed safety. The experiment uses one humanoid model, flat ground, low-to-moderate walking commands from the upstream G1 task, one PPO architecture, a finite disturbance range, and three training seeds. It does not cover stairs, jumping, manipulation, hardware deployment, or physical robot validation. Randomized friction and pushes are varied together in the DR condition, so the study estimates the effect of the combined randomization package; it cannot attribute gains independently to friction or pushes without additional ablation groups.

The supplied code is a starter implementation tailored to the official Isaac Lab 3.0.0 task. It has been syntax-checked in the report-generation environment, but not executed against your Brev VM, Isaac Sim installation, or GPU. The simulator smoke test and CSV validation on Brev are mandatory before reporting numerical results.

## References and official documentation

1. NVIDIA, *Isaac Lab 3.0 Installation*. https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation
2. NVIDIA, *Isaac Lab 3.0 Reinforcement Learning*. https://docs.nvidia.com/isaaclab/3.0/concepts/reinforcement-learning
3. NVIDIA, *Isaac Lab 3.0 G1 Flat Environment Configuration*. https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0/source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/flat_env_cfg.py
4. NVIDIA, *Isaac Lab 3.0 MDP API: material randomization and push events*. https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab/isaaclab.envs.mdp.html
5. NVIDIA, *Isaac Sim Deployment on NVIDIA Brev*. https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_advanced_cloud_setup_brev.html
6. Gu, X., Wang, Y.-J., & Chen, J. (2024). *Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer*. arXiv:2404.05695. https://arxiv.org/abs/2404.05695
7. Radosavovic, I., Xiao, T., Zhang, B., Darrell, T., Malik, J., & Sreenath, K. (2024). *Real-world humanoid locomotion with reinforcement learning*. Science Robotics, 9, eadi9579. https://doi.org/10.1126/scirobotics.adi9579
