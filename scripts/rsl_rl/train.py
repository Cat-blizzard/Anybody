# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import os
import argparse
import sys

os.environ.setdefault("WANDB_SILENT", "true")

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

if WORLD_SIZE > 1:
    base = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"isaaclab_kit_{os.getuid()}")
    rank_dir = os.path.join(base, f"rank{RANK}")
    os.environ.setdefault("OMNI_USER_DIR", rank_dir)
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(rank_dir, "cache"))
    os.environ.setdefault("XDG_DATA_HOME",  os.path.join(rank_dir, "data"))
    os.environ.setdefault("XDG_CONFIG_HOME",os.path.join(rank_dir, "config"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--motion", type=str, required=True, help="The path to the motion file or motion directory.")
parser.add_argument(
    "--holdout_eval_motion",
    type=str,
    default=None,
    help="Held-out motion directory for intermittent two-bottleneck student eval. Enables holdout_eval when set.",
)
parser.add_argument(
    "--test_prior_quality",
    action="store_true",
    default=False,
    help="Roll out with prior→decoder only (encoder bypassed), eval mode, no learning. "
    "Requires --resume_student_checkpoint (PULSE student checkpoint).",
)
parser.add_argument(
    "--test_prior_quality_fixed_std",
    type=float,
    default=None,
    help="If set, prior rollout uses this fixed per-dim latent std (μ still from prior MLP). "
    "Omit to use the prior's predicted σ.",
)
# VR-Tracking (residual latent action + command-manager EE masking)
parser.add_argument(
    "--prior_checkpoint",
    type=str,
    default=None,
    help="Path to a PULSE-style .pt containing `prior.*` and `student_core.decoder.*` "
    "(required for VR-Tracking-Flat-G1-v0; for VR-Tracking-Joint-Flat-G1-v0 used for policy PULSE init only, "
    "not for the RL obs normalizer).",
)
parser.add_argument(
    "--latent_rl_checkpoint",
    type=str,
    default=None,
    help="Path to VR latent-space PPO checkpoint (ActorCritic model_state_dict) for VR-Tracking-Joint-Flat-G1-v0: "
    "warm-starts latent_actor + critic and supplies obs_normalizer_checkpoint_path (split goal/proprio stats) "
    "when not resuming.",
)
parser.add_argument(
    "--vr_residual_scale",
    type=float,
    default=None,
    help="Override env actions.joint_pos.residual_scale for VR residual latent.",
)
parser.add_argument(
    "--vr_latent_dim",
    type=int,
    default=None,
    help="Override env actions.joint_pos.latent_dim (must match prior/decoder checkpoint).",
)
parser.add_argument(
    "--vr_proprio_history_length",
    type=int,
    default=None,
    help="Override env actions.joint_pos.proprio_history_length (must match prior input layout).",
)
parser.add_argument(
    "--vr_mask_mode_probs",
    type=str,
    default=None,
    help="Comma-separated sampling probabilities for mask modes in cfg order "
        "(default left_ee, right_ee, both_ee). Renormalized if they do not sum to 1.",
)
parser.add_argument(
    "--vr_compact_goal_obs",
    type=str,
    default=None,
    help="Override VR goal observation layout: true keeps only keypoints that can be active; "
    "false keeps full keypoint layout with zero-padded masked entries.",
)
parser.add_argument(
    "--vr_use_full_goal_obs_with_distill_pretrain",
    type=str,
    default="true",
    help="When true, and VR training uses distillation pretrain warmstart, force non-compact "
    "goal observations (full keypoint layout) unless --vr_compact_goal_obs is explicitly set.",
)
parser.add_argument(
    "--warmstart_from_masked_partial_kp_tracker",
    action="store_true",
    default=False,
    help="Warm-start PPO ActorCritic from a masked partial KP (LatentBottleneckAnyBody) "
    "or compatible ActorCritic .pt. Requires --warmstart_checkpoint.",
)
parser.add_argument(
    "--warmstart_checkpoint",
    type=str,
    default=None,
    help="Path to checkpoint for --warmstart_from_masked_partial_kp_tracker (model_state_dict).",
)
parser.add_argument(
    "--warmstart_lock_full_normalizer",
    action="store_true",
    default=False,
    help="Debug mode for warmstart: load full obs normalizer from --warmstart_checkpoint "
    "and freeze it (disable split normalizer updates).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if getattr(args_cli, "warmstart_from_masked_partial_kp_tracker", False):
    ws = getattr(args_cli, "warmstart_checkpoint", None)
    if ws in (None, ""):
        raise ValueError(
            "--warmstart_from_masked_partial_kp_tracker requires --warmstart_checkpoint=/path/to.pt"
        )

if WORLD_SIZE > 1:
    args_cli.distributed = True
if args_cli.distributed:
    args_cli.device = f"cuda:{LOCAL_RANK}"
if args_cli.distributed and RANK != 0:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    args_cli.video = False

# Enable cameras only when explicit training video recording is requested.
args_cli.enable_cameras = bool(args_cli.video)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# CUDA_VISIBLE_DEVICES remaps CUDA ordinals, but Kit still defaults --/renderer/activeGpu to LOCAL_RANK
# (physical GPU index). Append a second activeGpu so Vulkan targets the same card as torch.cuda.
if args_cli.distributed:
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if _cvd:
        _parts = [p.strip() for p in _cvd.split(",") if p.strip()]
        try:
            _phys = [int(p) for p in _parts]
        except ValueError:
            _phys = []
        if len(_phys) > LOCAL_RANK:
            sys.argv.append(f"--/renderer/activeGpu={_phys[LOCAL_RANK]}")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# SimulationApp appends sys.argv to Kit; Hydra later reparses sys.argv and rejects --/... kit keys.
sys.argv = [a for a in sys.argv if not a.startswith("--/renderer/activeGpu=")]

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import torch
from datetime import datetime
import numpy as np

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.mdp.curriculums import attach_curriculum_rollout_hints
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
from whole_body_tracking.utils.vr_policy_warmstart import apply_vr_policy_warmstart

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if getattr(args_cli, "holdout_eval_motion", None) is not None:
        _hm = os.path.abspath(args_cli.holdout_eval_motion)
        setattr(agent_cfg, "holdout_eval_motion", _hm)
        setattr(agent_cfg, "holdout_eval_enabled", True)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set seeds (explicit rank offset for distributed to avoid identical sampling across ranks)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    base_seed = int(agent_cfg.seed)
    rank = int(os.environ.get("RANK", "0"))
    # stride avoids overlaps if some components use multiple RNG draws per step
    seed_stride = int(os.environ.get("SEED_STRIDE", "1000"))
    env_seed = base_seed + rank * seed_stride
    env_cfg.seed = env_seed

    # also seed common RNGs to keep per-rank randomness independent
    random.seed(env_seed)
    np.random.seed(env_seed)
    torch.manual_seed(env_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(env_seed)

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        print(f"[INFO] Distributed seeding: base_seed={base_seed}, rank={rank}, env_seed={env_seed} (stride={seed_stride})")
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    env_cfg.commands.motion.motion = args_cli.motion

    # VR-Tracking: PULSE checkpoint + motion overrides (latent env also wires prior into ResidualLatentAction)
    task_name = str(getattr(args_cli, "task", "") or "")
    if "VR-Tracking" in task_name:
        if getattr(args_cli, "prior_checkpoint", None) in (None, ""):
            raise ValueError(
                f"Task {task_name!r} requires --prior_checkpoint=/path/to/pulse_student.pt "
                "(must contain prior.* and student_core.decoder.* for latent env; for joint env, used for "
                "normalizer + policy PULSE init)."
            )
        if "Joint" not in task_name:
            jp = env_cfg.actions.joint_pos
            jp.prior_checkpoint = os.path.abspath(str(args_cli.prior_checkpoint))
            if getattr(args_cli, "vr_residual_scale", None) is not None:
                jp.residual_scale = float(args_cli.vr_residual_scale)
            if getattr(args_cli, "vr_latent_dim", None) is not None:
                jp.latent_dim = int(args_cli.vr_latent_dim)
            if getattr(args_cli, "vr_proprio_history_length", None) is not None:
                jp.proprio_history_length = int(args_cli.vr_proprio_history_length)
        if "VR-Tracking-Joint" in task_name:
            if getattr(args_cli, "latent_rl_checkpoint", None) in (None, "") and not getattr(
                agent_cfg, "resume", False
            ):
                raise ValueError(
                    f"Task {task_name!r} requires --latent_rl_checkpoint=/path/to/vr_latent_ppo.pt "
                    "when not resuming (warm-starts latent_actor + critic)."
                )
        if getattr(args_cli, "vr_mask_mode_probs", None) not in (None, ""):
            parts = [float(x.strip()) for x in str(args_cli.vr_mask_mode_probs).split(",") if x.strip()]
            if len(parts) == 0:
                raise ValueError("--vr_mask_mode_probs must be a non-empty comma-separated list.")
            env_cfg.commands.motion.mask_mode_probs = tuple(parts)
        if getattr(args_cli, "vr_compact_goal_obs", None) not in (None, ""):
            compact_raw = str(args_cli.vr_compact_goal_obs).strip().lower()
            if compact_raw in ("1", "true", "t", "yes", "y", "on"):
                env_cfg.commands.motion.compact_goal_observation = True
            elif compact_raw in ("0", "false", "f", "no", "n", "off"):
                env_cfg.commands.motion.compact_goal_observation = False
            else:
                raise ValueError("--vr_compact_goal_obs must be a boolean string (true/false).")
        else:
            auto_full_raw = str(getattr(args_cli, "vr_use_full_goal_obs_with_distill_pretrain", "true")).strip().lower()
            if auto_full_raw in ("1", "true", "t", "yes", "y", "on"):
                auto_full_goal_obs = True
            elif auto_full_raw in ("0", "false", "f", "no", "n", "off"):
                auto_full_goal_obs = False
            else:
                raise ValueError(
                    "--vr_use_full_goal_obs_with_distill_pretrain must be a boolean string (true/false)."
                )
            using_distill_pretrain = bool(
                getattr(args_cli, "warmstart_from_masked_partial_kp_tracker", False)
                or getattr(args_cli, "load_pretrain", None) not in (None, "")
            )
            if auto_full_goal_obs and using_distill_pretrain:
                env_cfg.commands.motion.compact_goal_observation = False
                if int(os.environ.get("RANK", "0")) == 0:
                    print(
                        "[INFO] VR goal obs auto-set to full (non-compact) for distillation pretrain compatibility."
                    )
        if int(os.environ.get("RANK", "0")) == 0:
            if "Joint" not in task_name:
                jp = env_cfg.actions.joint_pos
                print(
                    f"[INFO] VR-Tracking env overrides: prior_checkpoint={jp.prior_checkpoint!r}, "
                    f"latent_dim={getattr(jp, 'latent_dim', None)}, residual_scale={getattr(jp, 'residual_scale', None)}, "
                    f"mask_mode_probs={getattr(env_cfg.commands.motion, 'mask_mode_probs', None)}, "
                    f"compact_goal_observation={getattr(env_cfg.commands.motion, 'compact_goal_observation', None)}"
                )
            else:
                print(
                    f"[INFO] VR-Tracking-Joint: prior_checkpoint={os.path.abspath(str(args_cli.prior_checkpoint))!r}, "
                    f"latent_rl_checkpoint={getattr(args_cli, 'latent_rl_checkpoint', None)!r}, "
                    f"mask_mode_probs={getattr(env_cfg.commands.motion, 'mask_mode_probs', None)}, "
                    f"compact_goal_observation={getattr(env_cfg.commands.motion, 'compact_goal_observation', None)}"
                )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Partial-mask curriculum uses ``phase_until_learning_iterations``; RslRlVecEnvWrapper resets in __init__,
    # which runs curriculum before OnPolicyRunner.learn() sets rollout hints — seed the same fields here.
    _spe = int(getattr(agent_cfg, "num_steps_per_env", 0) or 0)
    if _spe > 0:
        attach_curriculum_rollout_hints(env, spe=_spe, learning_iteration=0)

    # VR joint-space policy needs ``proprio_dim`` before runner builds the network (read before VecEnv wrapper).
    vr_joint_proprio_dim: int | None = None
    if "VR-Tracking-Joint" in task_name:
        _core = env.unwrapped
        _jp0 = _core.action_manager.get_term("joint_pos")
        vr_joint_proprio_dim = getattr(_jp0, "_proprio_dim", None)
        if not isinstance(vr_joint_proprio_dim, int) or vr_joint_proprio_dim <= 0:
            raise RuntimeError(
                "VR-Tracking-Joint: could not read ``_proprio_dim`` from action term ``joint_pos`` "
                "(expected VRJointPositionAction)."
            )

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    if int(os.environ.get("RANK", "0")) == 0:
        n_obs = getattr(env, "num_obs", None)
        n_act = getattr(env, "num_actions", None)
        n_priv = getattr(env, "num_privileged_obs", None)
        print(
            f"[INFO] RslRlVecEnvWrapper: num_envs={getattr(env, 'num_envs', '?')}, "
            f"num_obs(policy)={n_obs}, num_privileged_obs={n_priv}, num_actions={n_act}"
        )

    # build train config and ensure CLI overrides are in the dict
    # (configclass/OmegaConf may not always persist nested attr changes through to_dict())
    train_cfg = agent_cfg.to_dict()
    if vr_joint_proprio_dim is not None:
        train_cfg.setdefault("policy", {})
        train_cfg["policy"]["proprio_dim"] = int(vr_joint_proprio_dim)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[INFO] VR-Tracking-Joint: policy proprio_dim={int(vr_joint_proprio_dim)} (from env action term)")
    if getattr(args_cli, "vr_latent_dim", None) is not None and "VR-Tracking-Joint" in task_name:
        train_cfg.setdefault("policy", {})
        train_cfg["policy"]["latent_dim"] = int(args_cli.vr_latent_dim)
    if getattr(args_cli, "vr_residual_scale", None) is not None and "VR-Tracking-Joint" in task_name:
        train_cfg.setdefault("policy", {})
        train_cfg["policy"]["residual_scale"] = float(args_cli.vr_residual_scale)
    if getattr(args_cli, "critic_warmup_itrs", None) is not None and isinstance(train_cfg.get("algorithm"), dict):
        train_cfg["algorithm"]["critic_warmup_itrs"] = max(0, int(args_cli.critic_warmup_itrs))
    if "VR-Tracking-Joint" in task_name:
        if getattr(agent_cfg, "resume", False):
            _norm_resume = getattr(agent_cfg, "resume_checkpoint_path", None)
            if _norm_resume not in (None, ""):
                train_cfg["obs_normalizer_checkpoint_path"] = os.path.abspath(str(_norm_resume))
            else:
                train_cfg["obs_normalizer_checkpoint_path"] = os.path.abspath(
                    get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
                )
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    "[INFO] VR-Tracking-Joint: obs_normalizer_checkpoint_path from resume checkpoint "
                    f"{train_cfg['obs_normalizer_checkpoint_path']!r}"
                )
        else:
            _lc = getattr(args_cli, "latent_rl_checkpoint", None)
            if _lc in (None, ""):
                raise ValueError(
                    "VR-Tracking-Joint requires --latent_rl_checkpoint for obs normalizer stats when not resuming "
                    "(use the latent VR PPO checkpoint that trained the actor, not --prior_checkpoint)."
                )
            train_cfg["obs_normalizer_checkpoint_path"] = os.path.abspath(str(_lc))
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    "[INFO] VR-Tracking-Joint: obs_normalizer_checkpoint_path from latent_rl_checkpoint "
                    f"{train_cfg['obs_normalizer_checkpoint_path']!r}"
                )
    elif "VR-Tracking" in task_name and getattr(args_cli, "prior_checkpoint", None) not in (None, ""):
        train_cfg["obs_normalizer_checkpoint_path"] = os.path.abspath(str(args_cli.prior_checkpoint))
    if (
        "VR-Tracking" in task_name
        and getattr(args_cli, "warmstart_from_masked_partial_kp_tracker", False)
        and getattr(args_cli, "warmstart_lock_full_normalizer", False)
    ):
        ws_norm_path = os.path.abspath(str(args_cli.warmstart_checkpoint))
        train_cfg["obs_normalizer_checkpoint_path"] = ws_norm_path
        train_cfg["enable_rl_split_obs_normalizer"] = False
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[INFO] Warmstart debug normalizer lock enabled: "
                f"loading full obs normalizer from {ws_norm_path!r} and freezing updates."
            )
    if getattr(args_cli, "holdout_eval_motion", None) is not None:
        train_cfg["holdout_eval_motion"] = os.path.abspath(args_cli.holdout_eval_motion)
        train_cfg["holdout_eval_enabled"] = True
    if getattr(args_cli, "learning_rate", None) is not None and "algorithm" in train_cfg:
        train_cfg["algorithm"]["learning_rate"] = args_cli.learning_rate
    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, train_cfg, log_dir=log_dir, device=agent_cfg.device
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # VR joint: warm-start PULSE prior/decoder + latent-space PPO actor/critic (skip when resuming full ckpt).
    if "VR-Tracking-Joint" in task_name and not agent_cfg.resume:
        from rsl_rl.modules.latent_bottleneck_anybody_actor_critic import LatentBottleneckAnyBodyActorCritic

        _pol = runner.alg.policy
        if isinstance(_pol, LatentBottleneckAnyBodyActorCritic):
            _pulse = os.path.abspath(str(args_cli.prior_checkpoint))
            _rl = os.path.abspath(str(args_cli.latent_rl_checkpoint))
            _pol.init_from_pulse_checkpoint(_pulse)
            _pol.init_from_latent_rl_checkpoint(_rl)
            _pol.to(agent_cfg.device)

    # Optional: initialize from a pretrained checkpoint without resuming optimizer/state.
    # This is controlled via --load_pretrain and is intended for initializing PPO
    # from an offline/online distillation run.
    pretrained_path = getattr(agent_cfg, "pretrained_checkpoint_path", None)
    if pretrained_path is not None and not getattr(args_cli, "test_prior_quality", False):
        print(f"[INFO]: Initializing policy from pretrained checkpoint (no optimizer resume): {pretrained_path}")
        # Only warmstart the actor from the distillation checkpoint; keep the critic
        # randomly initialized so it can adapt cleanly to PPO.
        runner.load(pretrained_path, load_optimizer=False, load_critic=False, load_iteration=False)

    if getattr(args_cli, "warmstart_from_masked_partial_kp_tracker", False) and not getattr(
        args_cli, "test_prior_quality", False
    ):
        if agent_cfg.resume:
            if int(os.environ.get("RANK", "0")) == 0:
                print("[INFO]: Skipping VR policy warmstart because resume=True (full checkpoint load follows).")
        else:
            ws_path = os.path.abspath(str(args_cli.warmstart_checkpoint))
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[INFO]: VR policy warmstart from {ws_path!r}")
            apply_vr_policy_warmstart(runner.alg.policy, ws_path)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # Prefer absolute path from --resume_student_checkpoint / student_checkpoint_path so we can
        # init weights from a run under a different experiment_name (e.g. PULSE ckpt -> prior-only).
        resume_path = getattr(agent_cfg, "resume_checkpoint_path", None)
        if resume_path is None:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        else:
            resume_path = os.path.abspath(resume_path)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if getattr(args_cli, "test_prior_quality", False):
            runner.load(resume_path, load_optimizer=False, load_critic=False)
        else:
            # Full resume restores optimizer moments, its LR, and the next update.
            runner.load(resume_path, load_optimizer=True)

    if int(os.environ.get("RANK", "0")) == 0:
        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training 
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
