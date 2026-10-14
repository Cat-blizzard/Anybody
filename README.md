# AnyBody: Whole-Body Humanoid Tracking from Arbitrary Keypoint Subsets

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://opensource.org/license/apache-2-0)

[[Website]](https://hazel-hammer.github.io/anybody-project-page/) 

## Overview

**AnyBody** is a training framework for whole-body humanoid motion tracking from sparse, partial keypoint observations. Rather than requiring a fixed sensor configuration, AnyBody trains a policy that can track from *any subset* of body keypoints — enabling the same model to generalize across VR headsets, wrist-worn IMUs, camera-based pose detectors, or any combination thereof.

The training pipeline has four stages:

1. **GMT (teacher)** — A privileged teacher tracker is trained with full state via PPO on a large multi-source [motion dataset](https://bones.studio/datasets).
2. **Latent bottleneck (Stage 1)** — The teacher is distilled online into a deterministic encoder–decoder student. The decoder D(z, proprio) → joint commands becomes a reusable frozen motor prior for subsequent stages.
3. **Keypoint encoder (Stage 2)** — A self-attention transformer over per-keypoint tokens is distilled in *latent space* against the frozen Stage 1 encoder. A 3-phase masking curriculum progressively masks keypoints from fully visible down to sparse semantic subsets (torso, wrists, ankles).
4. **Latent-space RL (Stage 3)** — PPO fine-tunes the latent action space with the decoder frozen as a motor prior. Reward is world-frame position accuracy over *visible* keypoints. Downstream tasks include omnidirectional locomotion, in-air writing, and obstacle-reach.

## Installation

Install Isaac Lab v2.1.0 by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend the conda installation.

```bash
# create virtual environment
conda create -n isaaclab python=3.10 -y
conda activate isaaclab
pip install --upgrade pip

# install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# install IsaacSim 4.5
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
isaacsim  # verify

# install Isaac Lab v2.1.0
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git fetch --all && git checkout v2.1.0
./isaaclab.sh --install
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py  # verify
```

Clone this repository **outside** the `IsaacLab` directory:

```bash
git clone https://github.com/hazel-hammer/Anybody.git
cd Anybody
```

Install the environment and algorithm libraries:

```bash
pip install -e source/whole_body_tracking
pip install -e source/rsl_rl
```

## Data Preparation

Motion data must be preprocessed into `.npz` format before training. We first download human motions from [Bones Studio's website](https://bones.studio/datasets), then retarget these motions into robot reference trajectories. We follow the same retargeting convention as [Unitree's LAFAN1 dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).

**Convert a single motion:**
```bash
python scripts/csv_to_npz.py --input_file {motion}.csv --input_fps 30 --output_name {motion} --headless
```

**Batch convert a directory:**
```bash
python scripts/batch_csv_to_npz.py --input_dir {motion_dir} --input_fps 30 --output_dir {output_dir} --headless
```

**Replay a motion to verify:**
```bash
python scripts/replay_npz.py --motion_file {motion_file.npz}
```


## Training

Training scripts are in `run/train/`. Run the stages in order, setting motion paths and checkpoint paths inside each script.

### Stage 0 — GMT Teacher

```bash
bash run/train/GMT/run_mosaic_gmt.sh
```

Checkpoints are saved to `logs/rsl_rl/g1_flat_mosaic_hybrid/<timestamp>_<run_name>/`.

### Stage 1 — Latent Bottleneck (MUSE-Joint)

```bash
bash run/train/MUSE/run_muse_transformer_distillation.sh
```

Set `TEACHER_CHECKPOINT` to the GMT `.pt` path before running.

### Stage 2 — Keypoint Encoder Distillation (MUSE-Kp)

Distills a keypoint transformer into latent space against the frozen Stage 1 encoder:

```bash
bash run/train/MUSE/run_muse_kp_latent_distillation.sh
```

Set `TEACHER_CHECKPOINT` to the Stage 1 `.pt` path (used to initialize the frozen decoder, frozen JC encoder, and KP encoder warmstart).

### Stage 3 — Latent-Space RL Fine-tuning

Fine-tunes the keypoint encoder via PPO with the decoder frozen as a motor prior:

```bash
# Locomotion / general
bash run/train/MUSE/run_muse_kp_latent_rl.sh

# Writing task
bash run/train/MUSE/run_muse_kp_latent_rl_writing.sh

# Obstacle-reach task
bash run/train/MUSE/run_muse_kp_latent_rl_obstacle_reach.sh
```

Set `ENCODER_DECODER_WARMSTART` to the Stage 2 `.pt` path. The adapter type (default: `full_ft`) can be changed via the `ADAPTER` env var (`lora` or `residual`).

**Multi-GPU training** is supported via `torchrun`:
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/rsl_rl/train.py \
    --task=<task_id> --distributed --num_envs=4096 --motion <motion_dir> --headless
```

## Evaluation

```bash
# Visualize a trained policy
python scripts/rsl_rl/play.py \
    --task=<task_id> \
    --num_envs=1 \
    --load_run=<timestamp_run_name> \
    --checkpoint=model_<N>.pt \
    --video --video_length=200 --headless
```

See scripts under `run/test/` for per-stage evaluation recipes.

## Repo Structure

```
AnyBody/
├── source/
│   ├── whole_body_tracking/        # Isaac Lab task definitions and environment code
│   │   └── whole_body_tracking/
│   │       ├── tasks/
│   │       │   ├── tracking/       # Main motion-tracking task (MDP, configs, rewards)
│   │       │   │   ├── mdp/        # Atomic MDP functions (commands, rewards, obs, events)
│   │       │   │   ├── config/g1/  # G1-specific env + agent configs
│   │       │   │   └── tracking_env_cfg.py
│   │       │   └── vr_tracking/    # VR teleoperation task variant
│   │       ├── robots/             # Robot-specific actuator/joint configs (G1, SMPL)
│   │       ├── collection/         # Expert trajectory collection utilities
│   │       ├── distillation/       # Distillation model wrappers
│   │       ├── motion_tools/       # Motion resampling utilities
│   │       └── synth/              # Synthetic motion generation (writing task)
│   └── rsl_rl/                     # Modified rsl_rl with AnyBody algorithms and modules
│       └── rsl_rl/
│           ├── algorithms/         # PPO, distillation, latent-RL training loops
│           ├── modules/            # Encoder/decoder architectures (PULSE, MUSE-Kp, latent RL)
│           ├── networks/           # Shared network building blocks (transformer, encoder, etc.)
│           ├── runners/            # On-policy runner variants
│           └── storage/            # Rollout storage
├── scripts/
│   ├── rsl_rl/                     # train.py and play.py entry points
│   ├── csv_to_npz.py               # Single-motion preprocessing
│   ├── batch_csv_to_npz.py         # Batch preprocessing
│   ├── replay_npz.py               # Motion replay/visualization
│   └── ...                         # Dataset analysis, statistics, synth tools
└── run/
    ├── train/
    │   ├── GMT/                    # Stage 0: teacher training scripts
    │   ├── PULSE/                  # Stage 1: latent bottleneck distillation scripts
    │   └── MUSE/                   # Stages 2–3: KP encoder distillation + latent RL scripts
    └── test/                       # Evaluation and visualization scripts
```

## Acknowledgements

This project builds on the following open-source works:

- **[MOSAIC](https://github.com/BAAI-Humanoid/MOSAIC)** (BAAI-Humanoid) — the base training infrastructure and the GMT teacher that AnyBody extends
- **[Isaac Lab](https://github.com/isaac-sim/IsaacLab)** — simulation framework
- **[Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)** — physics engine
- **[rsl_rl](https://github.com/leggedrobotics/rsl_rl)** — RL training library (modified and extended)

## Isolated teacher correctness repairs (2026-10-15)

This worktree branches from `c319a6609158fdb7456c065800b9a489b2605f5b`.
The legacy worktrees, existing ABC motion files and checkpoints remain intact.
Old C must continue with `/data/jxc/projects/bones_curated/references/AnyBody_resume_state_fix_20261006`
and its frozen C data. Do not replace shared editable installations or modify
the old worktree to enable these repairs. Select both source packages explicitly
using `PYTHONPATH` in each future run and record their resolved source paths.

All three profiles below include the reset-cache repair and removal of the unused
actor debug path that rejected an all-zero goal. The PPO/network architecture,
815-dimensional teacher observations and existing 30-body NPZ axes are retained.
Original reward functions and task IDs remain available for controlled comparisons.

| Task ID | Reward changes |
|---|---|
| `General-Tracking-Flat-G1-Teacher-Cache-v1` | Original reward recipe, refreshed reset caches |
| `General-Tracking-Flat-G1-Teacher-Wrists-v1` | Upper6/lower8 world-position extension; original VR term retained |
| `General-Tracking-Flat-G1-Teacher-Points-v1` | Wrist profile plus asset-defined virtual head + two wrist origins |

The extension keeps pelvis, hips, knees, ankles and torso in the original lower
group. Adding wrists changes the upper mean denominator from four to six;
unchanged nominal weights do not mean an unchanged objective.

The Points profile records `g1_urdf_nominal_isolated_head_com_v1` and derives the
torso-frame offset `(0.0092305, 0.000299, 0.405869)` metres from the fixed head joint
and isolated-head inertial origin. Reference and actual torso poses use the same
WXYZ rotation and offset. This declared surrogate is neither the merged torso CoM
nor a certified author/headset landmark. It does not substitute torso for head,
add a simulated body, or change the observation interface. Missing required point
names and invalid shapes/values raise errors instead of silently yielding zero.

The single/batch CSV converters default to `--root_velocity_mode link_v2`, using
the native link-state writer to transform
`v_COM = v_link + omega_world x (R_link * r_COM)`.
Their existing `body_lin_vel_w` output key explicitly stores body CoM velocity.
Every conversion requires a **new** output directory and uses exclusive NPZ
creation; existing files/directories and escaping destinations are rejected.
`legacy_com_v1` is an explicit comparison mode and also requires new output.
NPZ metadata records the mode, point axes, CSV/asset/converter/API hashes and units.
No existing NPZ has been regenerated. Keep velocity-corrected data as a separate
factor from reward repairs; a future run must verify its actual NPZ metadata.

CPU checks (use the project's Python environment; no simulator is launched):

```bash
python tests/test_command_reset_cache.py
python tests/test_teacher_point_rewards.py
python tests/test_actor_zero_goal.py
python scripts/csv_to_npz.py --self-test
python scripts/batch_csv_to_npz.py --self-test
python scripts/test_root_velocity_contract.py
python scripts/verify_legacy_teacher_isolation.py
```

The legacy isolation snapshot and delivery evidence are in
`/data/jxc/projects/bones_curated/training/audit/anybody_teacher_correctness_fix_20261015`.
Use the isolation verifier before any old-C continuation. Native simulator reset,
conversion and learning-effect validation are **NOT_RUN** in this repair delivery;
the CPU checks do not establish improved tracking or physics performance. Student
normalization/freezing and paper-preset differences are outside this teacher patch.

## G1 compatibility entry and bounded native reset check

Use this worktree's entry for G1 scripts on the installed Isaac Lab stack:

```bash
python scripts/run_g1_reference_script.py scripts/rsl_rl/train.py [G1 training arguments]
python scripts/run_g1_reference_script.py scripts/csv_to_npz.py [conversion arguments]
python scripts/run_g1_reference_script.py --self-test
python tests/test_g1_entrypoint.py
python scripts/check_teacher_reset_native.py --self-test
```

The entry pins both project packages, rejects conflicting loaded sources and
adapts the metadata pickle helper and RSL observation return containers. It
preserves the underlying tensors, actions, rewards and termination values.
The robot registry advertises only available configurations; missing optional
robots do not receive replacement G1 configurations. The shard launcher uses
this entry. Old C continues to use its original source and launch pipeline.

The bounded native checker takes `--task`, `--motion`, `--num_envs`,
`--output_dir`, `--device` and `--headless`. Its read-only DEV view is identified
by `DEV3_INPUTS.json` in the view's parent directory. Each profile uses seven
control steps and records reset caches, independent NPZ/PhysX pose targets,
first-step reward and termination, untouched rows and frame transitions.
Reset perturbations, pushes and observation noise are disabled for this
diagnostic; startup randomization remains, and a 0.08-second episode exercises
the normal timeout reset path. Actions are reference joint PD targets.
The checker does not measure trained-policy performance or the motion-end
boundary and does not execute CSV conversion. Native receipts, including
failures, are preserved separately from the CPU repair delivery.
