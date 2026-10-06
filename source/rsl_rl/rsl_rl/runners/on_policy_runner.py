# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import statistics
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.algorithms import (
    PPO,
    LatentPPO,
    Distillation,
    AdvPulseDistillation,
    MuseCoTrainDistillation,
    MuseDistillation,
    MuseKpDistillation,
    MuseKpLatentDistillation,
    PulseDistillation,
    AnyBodyLatentDistillation,
)
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    EmpiricalNormalization,
    StudentTeacher,
    LatentBottleneckAnyBody,
    LatentBottleneckAnyBodyActorCritic,
    LatentBottleneckMUSE,
    LatentBottleneckMUSECoTrain,
    LatentBottleneckMUSETransformer,
    LatentBottleneckMUSEKp,
    LatentBottleneckMUSEKpLatent,
    LatentBottleneckPULSE,
    LatentBottleneckPULSEAdv,
    LatentBottleneck2B,
    LatentRLActorCritic,
)
from rsl_rl.algorithms.mask_utils import sync_policy_keypoint_mask_from_motion_command
from rsl_rl.utils import (
    _build_full_obs_norm_state_from_split,
    freeze_normalizers_on_resume,
    load_and_freeze_full_obs_normalizer,
    load_normalizer_states_on_resume,
    resolve_obs_norm_checkpoint_state,
    save_normalizer_states,
    store_code_state,
    try_build_rl_split_normalizers_from_checkpoint,
)
from rsl_rl.utils.finite_checks import replace_nonfinite_with_zeros
from rsl_rl.utils.checkpoint_resume import (
    checkpoint_progress,
    ppo_optimizer_learning_rate,
    resolve_checkpoint_phase,
    resolve_checkpoint_progress,
)

# Distillation variants: all use teacher as privileged obs; latter three load teacher from checkpoint
DISTILLATION_TYPES = (
    "distillation",
    "latent_consistent_distillation",
    "pulse_distillation",
    "anybody_latent_distillation",
    "muse_kp_latent_distillation",
)
TEACHER_CHECKPOINT_DISTILLATION = (
    "latent_consistent_distillation",
    "pulse_distillation",
    "anybody_latent_distillation",
    "muse_kp_latent_distillation",
)
# Algos that take current_iter in update() (for annealing / stage logic)
UPDATE_WITH_CURRENT_ITER = (
    "pulse_distillation",
    "anybody_latent_distillation",
    "muse_kp_latent_distillation",
)


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "MOSAIC":
            self.training_type = "mosaic"  # MOSAIC has its own training type with teacher action storage
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        elif self.alg_cfg["class_name"] == "LatentConsistentDistillation":
            self.training_type = "latent_consistent_distillation"
        elif self.alg_cfg["class_name"] == "PulseDistillation":
            self.training_type = "pulse_distillation"
        elif self.alg_cfg["class_name"] == "AdvPulseDistillation":
            self.training_type = "pulse_distillation"
        elif self.alg_cfg["class_name"] == "PriorOnlyPulseDistillation":
            self.training_type = "pulse_distillation"
        elif self.alg_cfg["class_name"] == "MuseDistillation":
            # Same rollout/storage structure as PULSE distillation; differs only in update logic.
            self.training_type = "pulse_distillation"
        elif self.alg_cfg["class_name"] == "MuseKpDistillation":
            # Aliased MuseDistillation on the KP-token student; identical rollout/storage layout.
            self.training_type = "pulse_distillation"
        elif self.alg_cfg["class_name"] == "MuseCoTrainDistillation":
            # Same rollout/storage as MUSE distillation: student_obs (=policy combined JC+KP+proprio)
            # + privileged_obs (=teacher) + privileged_actions (a_target). The 50/50 piloting
            # happens inside policy.act(); the runner sees a single action per env per step.
            self.training_type = "pulse_distillation"
        elif self.alg_cfg["class_name"] == "MuseKpLatentDistillation":
            # Latent-space KP distillation vs a frozen JC MUSE-T teacher + frozen shared decoder.
            # Reuses the anybody split-normalizer + latent-storage plumbing (proprio tail frozen
            # from the JC teacher, KP/mask prefix online); proprio-from-teacher copy is JC-goal-
            # offset aware (the JC teacher obs has a trailing mask bit, so proprio is NOT the tail).
            self.training_type = "muse_kp_latent_distillation"
        elif self.alg_cfg["class_name"] == "AnyBodyLatentDistillation":
            self.training_type = "anybody_latent_distillation"
        elif self.alg_cfg["class_name"] == "LatentPPO":
            # Latent-space RL: the PPO action IS the MUSE-Kp latent. Rollout storage is
            # sized to latent_dim (below) and the runner decodes latent->joint via the
            # frozen decoder before env.step (see the rollout loop).
            self.training_type = "latent_rl"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        obs_dict = extras.get("observations", {})
        if "policy" in obs_dict:
            self.policy_obs_type = "policy"
            obs = obs_dict["policy"]
        else:
            self.policy_obs_type = None
        if "teacher" in obs_dict:
            self.teacher_obs_type = "teacher"
        else:
            self.teacher_obs_type = None
        num_obs = obs.shape[1]

        # resolve type of privileged observations
        if self.training_type in ("rl", "latent_rl"):
            # latent_rl is PPO over the latent: same critic-obs resolution as RL.
            # Uses a "critic" obs group if the env provides one (asymmetric critic),
            # else falls back to the student obs (symmetric critic — M1 default).
            if "critic" in obs_dict:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learning, e.g., PPO
            else:
                self.privileged_obs_type = None
        elif self.training_type == "mosaic":
            # MOSAIC uses critic observations for value function when available.
            # Teacher observations are handled separately for teacher BC.
            has_teacher_obs = "teacher" in obs_dict
            has_critic_obs = "critic" in obs_dict
            if has_critic_obs:
                self.privileged_obs_type = "critic"
                print(f"[MOSAIC] Using 'critic' observations for value estimation.")
            elif has_teacher_obs:
                self.privileged_obs_type = "teacher"
                print(f"[MOSAIC] Using 'teacher' observations for value estimation (no critic obs available).")
            else:
                self.privileged_obs_type = None
        elif self.training_type in DISTILLATION_TYPES:
            self.privileged_obs_type = "teacher" if "teacher" in obs_dict else None

        # resolve type of ref_vel_estimator observations (for MOSAIC with velocity estimator)
        if "ref_vel_estimator" in obs_dict:
            self.ref_vel_estimator_obs_type = "ref_vel_estimator"
            num_ref_vel_estimator_obs = obs_dict["ref_vel_estimator"].shape[1]
            print(f"[Runner] Found 'ref_vel_estimator' observations for velocity estimation (dim={num_ref_vel_estimator_obs}).")
        else:
            self.ref_vel_estimator_obs_type = None

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None and self.privileged_obs_type in obs_dict:
            num_privileged_obs = obs_dict[self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs
        if self.teacher_obs_type is not None and self.teacher_obs_type in obs_dict:
            num_teacher_obs = obs_dict[self.teacher_obs_type].shape[1]
        else:
            num_teacher_obs = None

        # Adjust actor input dimension if using velocity estimator (MOSAIC with estimated ref vel)
        # The actor will receive obs_augmented = [obs, estimated_ref_vel] where estimated_ref_vel is 3D
        # IMPORTANT: Keep num_obs unchanged for normalizer initialization!
        # IMPORTANT: For ResidualActorCritic, do NOT adjust num_actor_obs (it handles estimator internally)
        num_actor_obs = num_obs  # Start with policy obs dimension

        # evaluate the policy class
        policy_class = eval(self.policy_cfg.pop("class_name"))

        # Check if using ResidualActorCritic (special handling for estimator dimension)
        from rsl_rl.modules import ResidualActorCritic
        is_residual_policy = policy_class == ResidualActorCritic

        if self.training_type == "mosaic" and self.alg_cfg.get("use_estimate_ref_vel", False):
            if not is_residual_policy:
                # For normal ActorCritic: adjust input dimension to include estimated ref_vel
                num_actor_obs += 3  # Add 3 dimensions for estimated reference velocity (x, y, z)
                print(f"[Runner] Velocity estimator enabled: actor input dimension adjusted to {num_actor_obs} (policy obs {num_obs} + 3D velocity)")
            else:
                # For ResidualActorCritic: keep num_actor_obs unchanged (770)
                # ResidualActorCritic handles estimator internally:
                # - residual_actor uses num_actor_obs (770)
                # - GMT policy uses num_actor_obs + 3 (773)
                print(f"[Runner] Velocity estimator enabled for ResidualActorCritic: residual_actor uses {num_actor_obs} dims, GMT uses {num_actor_obs + 3} dims")
        policy: ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent | LatentBottleneckAnyBody | LatentBottleneckAnyBodyActorCritic | LatentBottleneckMUSE | LatentBottleneckMUSETransformer | LatentBottleneckMUSEKp | LatentBottleneckPULSE | LatentBottleneckPULSEAdv | LatentBottleneck2B = policy_class(
            num_actor_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        if getattr(policy, "_teacher_goal_adapter", None) is not None:
            print(
                f"[Runner] Teacher goal adapter: env teacher obs dim={getattr(policy, 'num_teacher_obs', '?')}, "
                f"teacher_core obs dim={getattr(policy, 'teacher_encoder_obs_dim', '?')} (frozen Linear on goal prefix)."
            )

        # Latent-consistent, 2B partial-obs: optional keypoint (+ ref body vel) masking.
        _mask_training_types = ("latent_consistent_distillation",)
        if self.training_type in _mask_training_types and hasattr(policy, "set_mask_matrix"):
            mask_cfg_dict = self.alg_cfg.get("mask_cfg")
            if mask_cfg_dict and isinstance(mask_cfg_dict, dict) and mask_cfg_dict.get("mode_spec"):
                env_unwrap = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
                if hasattr(env_unwrap, "command_manager") and "motion" in getattr(env_unwrap.command_manager, "_terms", {}):
                    body_names = env_unwrap.command_manager._terms["motion"].cfg.body_names
                    from rsl_rl.algorithms.mask_utils import MaskCfg, build_mask_matrix
                    mode_spec = mask_cfg_dict["mode_spec"]
                    dup_vel = bool(mask_cfg_dict.get("duplicate_for_ref_body_lin_vel", True))
                    body_masks, _ = build_mask_matrix(
                        body_names,
                        mode_spec,
                        device=self.device,
                        duplicate_for_ref_body_lin_vel=dup_vel,
                    )
                    cfg = MaskCfg(mode_spec=mode_spec, mode_probs=mask_cfg_dict.get("mode_probs"))
                    mode_probs = cfg.get_mode_probs(self.device)
                    obs_steps = int(mask_cfg_dict.get("obs_steps", 1))
                    policy.set_mask_matrix(
                        body_masks,
                        mode_probs,
                        obs_steps=obs_steps,
                    )
                    print(
                        f"[Runner] Set mask matrix for {self.training_type}: "
                        f"{body_masks.shape[0]} modes, body_kp_per_step={body_masks.shape[1]}, "
                        f"obs_steps={obs_steps}, duplicate_for_ref_body_lin_vel={dup_vel} "
                        f"(goal tail derived from policy goal/human_dim)"
                    )

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # initialize algorithm
        alg_class_name = self.alg_cfg.pop("class_name")
        alg_class = eval(alg_class_name)
        self.alg = alg_class(
            policy,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        if self.training_type == "rl":
            warmup_iters = self.alg_cfg.get("critic_warmup_itrs", None)
            if warmup_iters is None:
                warmup_iters = self.alg_cfg.get("critic_warmup_itrs", 0)
            warmup_iters = max(0, int(warmup_iters))
            self.alg.critic_warmup_itrs = warmup_iters
            if self.gpu_global_rank == 0:
                status = "enabled" if warmup_iters > 0 else "disabled"
                print(f"[Runner] Critic warmstart {status} (critic_warmup_itrs={warmup_iters})")

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]

        # Check if using ResidualActorCritic (special handling for GMT normalizer)
        from rsl_rl.modules import ResidualActorCritic
        if isinstance(policy, ResidualActorCritic):
            # Use GMT's frozen normalizer for observations
            if policy.gmt_normalizer is not None:
                self.obs_normalizer = policy.gmt_normalizer
                print("[Runner] Using GMT's frozen normalizer for ResidualActorCritic")
            else:
                print("[Runner] WARNING: ResidualActorCritic has no GMT normalizer, using Identity")
                self.obs_normalizer = torch.nn.Identity().to(self.device)

            # Create privileged obs normalizer (for critic)
            if self.empirical_normalization:
                self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                    self.device
                )
            else:
                self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

            # Teacher obs normalizer (not used for residual learning)
            self.teacher_obs_normalizer = torch.nn.Identity().to(self.device)
        elif self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
            if num_teacher_obs is not None:
                self.teacher_obs_normalizer = EmpiricalNormalization(shape=[num_teacher_obs], until=1.0e8).to(
                    self.device
                )
            else:
                self.teacher_obs_normalizer = torch.nn.Identity().to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.teacher_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # Optional: initialize student obs normalization from an external checkpoint (e.g. PULSE prior ckpt).
        # For VR PPO, only the proprio tail should be loaded/frozen while command prefix keeps adapting.
        self._obs_normalizer_loaded_from_checkpoint = False
        obs_norm_ckpt_path, obs_norm_state = resolve_obs_norm_checkpoint_state(
            cfg=self.cfg,
            device=self.device,
            empirical_normalization=self.empirical_normalization,
        )

        # 2B residual-latent: use separate student normalizers
        # - goal prefix: train/update online
        # - proprio tail: loaded from teacher and frozen
        self.student_goal_obs_normalizer = None
        self.student_proprio_obs_normalizer = None
        self._anybody_latent_goal_dim = None
        self._anybody_latent_proprio_dim = None
        # Generic split mode (also used by VR PPO): command prefix + proprio tail.
        (
            self.student_goal_obs_normalizer,
            self.student_proprio_obs_normalizer,
            self._anybody_latent_goal_dim,
            self._anybody_latent_proprio_dim,
        ) = try_build_rl_split_normalizers_from_checkpoint(
            training_type=self.training_type,
            empirical_normalization=self.empirical_normalization,
            obs_norm_state=obs_norm_state,
            cfg=self.cfg,
            env=self.env,
            num_obs=num_obs,
            device=self.device,
        )
        if self.student_goal_obs_normalizer is not None and self.student_proprio_obs_normalizer is not None:
            print(
                f"[Runner] Split obs normalizer enabled from checkpoint: "
                f"goal_dim={self._anybody_latent_goal_dim} (trainable), "
                f"proprio_dim={self._anybody_latent_proprio_dim} (loaded+frozen) from {obs_norm_ckpt_path}"
            )
        elif (
            self.training_type == "rl"
            and bool(self.cfg.get("enable_rl_split_obs_normalizer", False))
            and self.empirical_normalization
            and obs_norm_state is not None
        ):
            print(
                "[Runner] WARNING: Could not infer valid proprio_dim for split obs normalizer. "
                "Falling back to full obs normalizer load/freeze."
            )
        if (
            self.training_type in ("anybody_latent_distillation", "muse_kp_latent_distillation")
            and self.empirical_normalization
        ):
            # Split student normalizer: proprio is the student-obs TAIL in both layouts
            # (anybody: [goal | proprio]; MUSE-Kp: [kp_lookahead | kp_mask | proprio]). The
            # proprio tail is loaded+frozen from the teacher; the prefix updates online.
            proprio_dim = getattr(policy, "proprio_dim", None)
            if isinstance(proprio_dim, int) and 0 < proprio_dim < num_obs:
                goal_dim = num_obs - proprio_dim
                self._anybody_latent_goal_dim = goal_dim
                self._anybody_latent_proprio_dim = proprio_dim
                self.student_goal_obs_normalizer = EmpiricalNormalization(shape=[goal_dim], until=1.0e8).to(
                    self.device
                )
                self.student_proprio_obs_normalizer = EmpiricalNormalization(shape=[proprio_dim], until=1.0e8).to(
                    self.device
                )
                print(
                    f"[Runner] 2B split student normalizers enabled: goal_dim={goal_dim} (trainable), "
                    f"proprio_dim={proprio_dim} (teacher-frozen)."
                )
        elif (
            self.empirical_normalization
            and obs_norm_state is not None
            and not isinstance(self.obs_normalizer, torch.nn.Identity)
            and not self._use_anybody_latent_split_normalizers()
        ):
            # Fallback behavior: load and freeze full normalizer.
            self._obs_normalizer_loaded_from_checkpoint = load_and_freeze_full_obs_normalizer(
                self.obs_normalizer, obs_norm_state
            )
            if self._obs_normalizer_loaded_from_checkpoint:
                print(
                    f"[Runner] Loaded full obs normalizer from checkpoint and froze it: {obs_norm_ckpt_path}"
                )

        # For MOSAIC, use teacher normalizer from checkpoint and freeze it.
        # IMPORTANT: In multi-teacher mode, skip runner-level normalization
        # because each teacher will use its own normalizer in MOSAIC.update()
        if (
            alg_class_name == "MOSAIC"
            and self.teacher_obs_type == "teacher"
        ):
            # Check for multi-teacher mode
            if hasattr(self.alg, "teacher_normalizers") and self.alg.teacher_normalizers is not None:
                # Multi-teacher: skip runner-level normalization
                self.teacher_obs_normalizer = torch.nn.Identity().to(self.device)
                print("[Runner] Multi-teacher mode: skipping runner-level teacher_obs normalization (each teacher uses its own normalizer)")
            elif hasattr(self.alg, "teacher_normalizer") and self.alg.teacher_normalizer is not None:
                # Single teacher: use teacher's normalizer
                self.teacher_obs_normalizer = self.alg.teacher_normalizer
                self.teacher_obs_normalizer.eval()  # Freeze teacher normalizer
                print("[Runner] Using teacher observation normalizer from checkpoint (frozen)")
            # else: keep the EmpiricalNormalization created above

        # For MOSAIC, pass obs_normalizer and privileged_obs_normalizer for teacher BC
        if alg_class_name == "MOSAIC":
            self.alg.obs_normalizer = self.obs_normalizer
            self.alg.privileged_obs_normalizer = self.privileged_obs_normalizer
            print("[Runner] Passed obs_normalizer and privileged_obs_normalizer to MOSAIC for teacher BC")

            # Pass environment's group mapping to MOSAIC for multi-teacher consistency
            env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            if hasattr(env, 'command_manager') and 'motion' in env.command_manager._terms:
                motion_command = env.command_manager._terms['motion']
                if hasattr(motion_command, 'group_name_to_idx'):
                    self.alg.env_group_name_to_idx = motion_command.group_name_to_idx
                    print(f"[Runner] Passed environment's group mapping to MOSAIC: {self.alg.env_group_name_to_idx}")

            # If MOSAIC loaded a teacher critic normalizer, use it for privileged obs
            if hasattr(self.alg, "teacher_critic_normalizer") and self.alg.teacher_critic_normalizer is not None:
                self.privileged_obs_normalizer = self.alg.teacher_critic_normalizer
                print("[Runner] Using teacher critic normalizer for privileged observations")

        # Load teacher from checkpoint for PULSE, latent-consistent, and two-bottleneck distillation
        if self.training_type in TEACHER_CHECKPOINT_DISTILLATION:
            teacher_path = self.alg_cfg.get("teacher_checkpoint_path")

            if teacher_path:
                if not os.path.isabs(teacher_path):
                    teacher_path = os.path.abspath(teacher_path)
                if not os.path.isfile(teacher_path):
                    raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
                teacher_ckpt = torch.load(teacher_path, map_location=self.device, weights_only=False)
                self.alg.policy.load_state_dict(teacher_ckpt["model_state_dict"])
                if "obs_norm_state_dict" in teacher_ckpt and not isinstance(self.teacher_obs_normalizer, torch.nn.Identity):
                    # Load teacher's observation normalizer for teacher inputs.
                    self.teacher_obs_normalizer.load_state_dict(teacher_ckpt["obs_norm_state_dict"])
                    self.teacher_obs_normalizer.eval()
                    if hasattr(self.teacher_obs_normalizer, "until"):
                        self.teacher_obs_normalizer.until = getattr(self.teacher_obs_normalizer, "count", 1e8)
                    print("[Runner] Loaded teacher observation normalizer from teacher checkpoint (frozen).")
                    if self.training_type == "pulse_distillation":
                        # Use same normalizer for policy (student) obs so student and teacher see identical inputs.
                        #self.obs_normalizer = self.teacher_obs_normalizer
                        #self.obs_normalizer.eval()  # Keep fixed: no stat updates during distillation
                        #print("[Runner] Student uses teacher normalizer for policy obs (fixed/frozen).")
                        pass #TODO: fix this
                    elif (
                        self.training_type == "anybody_latent_distillation"
                        and self.student_proprio_obs_normalizer is not None
                    ):
                        copied = self._initialize_anybody_latent_proprio_normalizer_from_teacher()
                        if copied:
                            print(
                                "[Runner] Initialized 2B student proprio normalizer from teacher checkpoint "
                                "and froze it."
                            )
                        else:
                            print(
                                "[Runner] WARNING: Could not initialize 2B student proprio normalizer from "
                                "teacher checkpoint."
                            )
                    elif (
                        self.training_type == "muse_kp_latent_distillation"
                        and self.student_proprio_obs_normalizer is not None
                    ):
                        copied = self._initialize_muse_kp_latent_proprio_normalizer_from_teacher()
                        if copied:
                            print(
                                "[Runner] Initialized MUSE-Kp-latent student proprio normalizer "
                                "from the JC teacher checkpoint (at jc_goal offset) and froze it."
                            )
                        else:
                            print(
                                "[Runner] WARNING: Could not initialize MUSE-Kp-latent student "
                                "proprio normalizer from teacher checkpoint."
                            )
                print(f"[Runner] Loaded teacher policy from: {teacher_path}")

            # Optional: a SECOND checkpoint that warmstarts only the policy weights — no
            # normalizer touched (so the teacher_obs_normalizer just loaded above survives).
            # Use case: MUSE-Kp distillation, where TEACHER_CHECKPOINT must point at the PHC+
            # stage-1 ckpt (for the right teacher_obs_normalizer) but we ALSO want to warmstart
            # the encoder + decoder from a stable MUSE-Transformer ckpt. The policy's own
            # ``load_state_dict`` is responsible for shape-filtering incompatible keys.
            #
            # CRITICAL: the encoder/decoder warmstart ckpt (e.g. a MUSE-Transformer .pt) carries
            # its OWN frozen ``teacher.*`` slot — a DIFFERENT PHC+ snapshot than the one just
            # loaded from TEACHER_CHECKPOINT. The policy's shape-filtering ``load_state_dict``
            # would happily copy those ``teacher.*`` weights and silently OVERWRITE the intended
            # teacher (observed: ~0.04 max-abs weight drift → teacher-pilot reward ~30 vs the
            # real teacher's ~38, ee_body_pos terminations ~10x). Strip every ``teacher.*`` key
            # (and its ``student.N.*`` aliases, defensively) so the TEACHER_CHECKPOINT teacher
            # survives — matching the documented "warmstarts only the policy weights" intent.
            warmstart_path = self.alg_cfg.get("encoder_decoder_warmstart_checkpoint_path")
            if warmstart_path:
                if not os.path.isabs(warmstart_path):
                    warmstart_path = os.path.abspath(warmstart_path)
                if not os.path.isfile(warmstart_path):
                    raise FileNotFoundError(
                        f"Encoder/decoder warmstart checkpoint not found: {warmstart_path}"
                    )
                warm_ckpt = torch.load(warmstart_path, map_location=self.device, weights_only=False)
                warm_sd = warm_ckpt["model_state_dict"]
                teacher_loaded = bool(self.alg_cfg.get("teacher_checkpoint_path"))
                if teacher_loaded:
                    n_before = len(warm_sd)
                    warm_sd = {
                        k: v
                        for k, v in warm_sd.items()
                        if not (k.startswith("teacher.") or ".teacher." in k)
                    }
                    n_stripped = n_before - len(warm_sd)
                    if n_stripped > 0:
                        print(
                            f"[Runner] Stripped {n_stripped} 'teacher.*' keys from the "
                            f"encoder/decoder warmstart ckpt to preserve the teacher loaded from "
                            f"TEACHER_CHECKPOINT (warmstart must not clobber the distillation target)."
                        )
                self.alg.policy.load_state_dict(warm_sd)
                print(f"[Runner] Warmstarted policy weights (model_state_dict only) from: {warmstart_path}")

        # Latent-RL warmstart (independent of the distillation gate above).
        # RL finetuning MUST start from the distilled MUSE-Kp policy — without
        # this the encoder/decoder train from random init, defeating the premise.
        # There is no teacher_checkpoint for RL (no BC), so the policy's loader
        # consumes the full MUSE-Kp model_state_dict (slot-aware kp_proj remap +
        # encoder + decoder + frozen teacher slot handled inside the policy).
        if self.training_type == "latent_rl":
            warmstart_path = self.alg_cfg.get("encoder_decoder_warmstart_checkpoint_path")
            if warmstart_path:
                if not os.path.isabs(warmstart_path):
                    warmstart_path = os.path.abspath(warmstart_path)
                if not os.path.isfile(warmstart_path):
                    raise FileNotFoundError(
                        f"[latent_rl] warmstart checkpoint not found: {warmstart_path}"
                    )
                warm_ckpt = torch.load(warmstart_path, map_location=self.device, weights_only=False)
                self.alg.policy.load_state_dict(warm_ckpt["model_state_dict"])
                # CRITICAL (verified bug 2026-05-17): also load + FREEZE the
                # distilled POLICY obs normalizer. The distilled MUSE-Kp encoder
                # was trained under a converged EmpiricalNormalization; without
                # these stats a fresh normalizer drifts toward the rollout
                # distribution and feeds the encoder out-of-distribution inputs —
                # the wiring-verify run showed the frozen-actor reward decay
                # 36.5 -> 17 (success 0.98 -> 0.59) purely from this drift. Freeze
                # it so the encoder finetune operates on the exact input
                # statistics it was distilled under. The asymmetric critic's
                # privileged normalizer intentionally stays fresh/trainable (the
                # critic is a new head on a different, larger obs group; the
                # distilled privileged_obs_norm is shape-incompatible anyway).
                if (
                    self.empirical_normalization
                    and not isinstance(self.obs_normalizer, torch.nn.Identity)
                ):
                    # The distilled MUSE-Kp uses SPLIT student normalizers (KP/mask goal prefix
                    # online + proprio tail frozen-from-JC), so the ckpt stores
                    # ``student_{goal,proprio}_obs_norm_state_dict`` and NOT ``obs_norm_state_dict``.
                    # Prefer the full key if present (older/non-split ckpts); otherwise reconstruct
                    # the full [goal | proprio] normalizer from the split keys. Without this the
                    # encoder finetunes under a FRESH online normalizer and sees OOD inputs — the
                    # exact drift the freeze below is meant to prevent (verified: frozen-actor
                    # reward 36.5 -> 17, success 0.98 -> 0.59).
                    norm_state = warm_ckpt.get("obs_norm_state_dict")
                    if not isinstance(norm_state, dict):
                        norm_state = _build_full_obs_norm_state_from_split(
                            warm_ckpt.get("student_goal_obs_norm_state_dict"),
                            warm_ckpt.get("student_proprio_obs_norm_state_dict"),
                        )
                    if isinstance(norm_state, dict):
                        loaded = load_and_freeze_full_obs_normalizer(
                            self.obs_normalizer, norm_state
                        )
                        self._obs_normalizer_loaded_from_checkpoint = bool(loaded)
                        src = (
                            "obs_norm_state_dict"
                            if "obs_norm_state_dict" in warm_ckpt
                            else "split student goal+proprio keys"
                        )
                        print(
                            "[Runner] [latent_rl] Loaded + froze distilled policy obs "
                            f"normalizer from {src} ({'ok' if loaded else 'FAILED — check shape'})."
                        )
                    else:
                        print(
                            "[Runner] [latent_rl] WARNING: distilled obs normalizer NOT "
                            "loaded (no obs_norm_state_dict and split goal/proprio keys "
                            "absent/invalid) — warmstart will be unfaithful."
                        )
                else:
                    print(
                        "[Runner] [latent_rl] WARNING: distilled obs normalizer NOT "
                        "loaded (empirical_normalization off) — warmstart will be unfaithful."
                    )
                print(
                    f"[Runner] [latent_rl] Warmstarted policy from distilled MUSE-Kp: {warmstart_path}"
                )
            else:
                print(
                    "[Runner] [latent_rl] WARNING: no encoder_decoder_warmstart_checkpoint_path "
                    "set — RL will start from RANDOM init, NOT a finetune."
                )

        # init storage and model
        if self.training_type == "mosaic":
            self.alg.init_storage(
                self.training_type,
                self.env.num_envs,
                self.num_steps_per_env,
                [num_obs],
                [num_privileged_obs],
                [self.env.num_actions],
                teacher_obs_shape=[num_teacher_obs] if num_teacher_obs is not None else None,
                ref_vel_estimator_obs_shape=[num_ref_vel_estimator_obs] if self.ref_vel_estimator_obs_type is not None else None,
            )
        elif self.training_type in (
            "anybody_latent_distillation",
            "muse_kp_latent_distillation",
        ):
            latent_dim = getattr(self.alg.policy, "latent_dim", 64)
            # RolloutStorage gates its generator allowlist AND the teacher_latent 6-tuple yield
            # on the literal "anybody_latent_distillation" token. The MUSE-Kp-latent storage
            # contract is identical (obs, priv_obs, actions, priv_actions, dones, teacher_latent),
            # so pass that token (the runner keeps self.training_type for its own branching —
            # same indirection the latent_rl path uses with the literal "rl").
            self.alg.init_storage(
                "anybody_latent_distillation",
                self.env.num_envs,
                self.num_steps_per_env,
                [num_obs],
                [num_privileged_obs] if num_privileged_obs else [num_obs],
                [self.env.num_actions],
                latent_dim=latent_dim,
            )
        elif self.training_type == "one_bottleneck_distillation":
            self.alg.init_storage(
                self.training_type,
                self.env.num_envs,
                self.num_steps_per_env,
                [num_obs],
                [num_privileged_obs] if num_privileged_obs else [num_obs],
                [self.env.num_actions],
            )
        elif self.training_type == "latent_rl":
            # Latent-space PPO: the RL action is the latent, so rollout storage
            # (actions / mu / sigma) is sized to latent_dim, NOT env.num_actions.
            # The decoded joint action is transient (consumed by env.step only).
            # RolloutStorage gates its RL buffers + mini_batch_generator on the
            # literal "rl" token, so pass that (the runner still keeps
            # self.training_type == "latent_rl" for its own branching: decode
            # hook, warmstart, obs-type). Storage is otherwise standard PPO.
            latent_dim = getattr(self.alg.policy, "latent_dim", 16)
            self.alg.init_storage(
                "rl",
                self.env.num_envs,
                self.num_steps_per_env,
                [num_obs],
                [num_privileged_obs],
                [latent_dim],
            )
        else:
            # rl, distillation, latent_consistent_distillation, pulse_distillation
            self.alg.init_storage(
                self.training_type,
                self.env.num_envs,
                self.num_steps_per_env,
                [num_obs],
                [num_privileged_obs],
                [self.env.num_actions],
            )

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        # Keep current_learning_iteration's legacy phase/last-completed meaning.
        # A separate counter prevents re-executing the last completed update.
        self._last_completed_learning_iteration = -1
        self._next_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def _use_anybody_latent_split_normalizers(self) -> bool:
        return (
            self.empirical_normalization
            and self.student_goal_obs_normalizer is not None
            and self.student_proprio_obs_normalizer is not None
            and isinstance(self._anybody_latent_goal_dim, int)
            and isinstance(self._anybody_latent_proprio_dim, int)
        )

    @property
    def next_learning_iteration(self) -> int:
        """The next training update; current_learning_iteration retains its phase."""
        return self._next_learning_iteration

    def _normalize_student_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self._use_anybody_latent_split_normalizers():
            out = self.obs_normalizer(obs)
        else:
            goal_dim = self._anybody_latent_goal_dim
            proprio_dim = self._anybody_latent_proprio_dim
            goal_obs = obs[..., :goal_dim]
            proprio_obs = obs[..., goal_dim : goal_dim + proprio_dim]
            goal_obs = self.student_goal_obs_normalizer(goal_obs)
            proprio_obs = self.student_proprio_obs_normalizer(proprio_obs)
            out = torch.cat([goal_obs, proprio_obs], dim=-1)
        # Masked keypoints (NaN in env / after norm) must not reach the MLP; where() zeros grad on those dims.
        return replace_nonfinite_with_zeros(out)

    def _initialize_anybody_latent_proprio_normalizer_from_teacher(self) -> bool:
        if not self._use_anybody_latent_split_normalizers():
            return False
        proprio_dim = self._anybody_latent_proprio_dim
        teacher_mean = getattr(self.teacher_obs_normalizer, "_mean", None)
        teacher_var = getattr(self.teacher_obs_normalizer, "_var", None)
        teacher_std = getattr(self.teacher_obs_normalizer, "_std", None)
        own_mean = getattr(self.student_proprio_obs_normalizer, "_mean", None)
        own_var = getattr(self.student_proprio_obs_normalizer, "_var", None)
        own_std = getattr(self.student_proprio_obs_normalizer, "_std", None)
        if not all(
            isinstance(t, torch.Tensor)
            for t in (teacher_mean, teacher_var, teacher_std, own_mean, own_var, own_std)
        ):
            return False
        teacher_dim = int(teacher_mean.shape[-1])
        if teacher_dim < proprio_dim:
            return False
        with torch.no_grad():
            own_mean.copy_(teacher_mean[..., -proprio_dim:])
            own_var.copy_(teacher_var[..., -proprio_dim:])
            own_std.copy_(teacher_std[..., -proprio_dim:])
            if hasattr(self.student_proprio_obs_normalizer, "count") and hasattr(self.teacher_obs_normalizer, "count"):
                self.student_proprio_obs_normalizer.count.copy_(self.teacher_obs_normalizer.count)
            if hasattr(self.student_proprio_obs_normalizer, "until"):
                self.student_proprio_obs_normalizer.until = getattr(
                    self.student_proprio_obs_normalizer, "count", 0
                )
        self.student_proprio_obs_normalizer.eval()
        return True

    def _initialize_muse_kp_latent_proprio_normalizer_from_teacher(self) -> bool:
        """Seed+freeze the student proprio normalizer from the JC teacher's proprio block.

        Unlike the anybody case (teacher proprio is the tail), the JC MUSE-T teacher obs is
        ``[goal(jc_goal_dim) | proprio(proprio_dim) | mask(1)]`` — proprio is NOT the tail, so the
        copy slices at the JC goal offset ``[jc_goal_dim : jc_goal_dim + proprio_dim]`` (a tail
        slice would be off-by-one: grab the mask bit and drop proprio dim 0). The proprio terms
        (joint_pos/joint_vel/base_ang_vel/actions, H5) are byte-identical between the JC teacher
        obs and the KP student obs, so this is a dim-for-dim copy. The frozen decoder then always
        sees proprio in the exact distribution it was trained under.
        """
        if not self._use_anybody_latent_split_normalizers():
            return False
        proprio_dim = self._anybody_latent_proprio_dim
        jc_goal_dim = getattr(self.alg.policy, "jc_goal_dim", None)
        if not isinstance(jc_goal_dim, int) or jc_goal_dim < 0:
            print("[Runner] WARNING: policy.jc_goal_dim missing/invalid; cannot offset-copy.")
            return False
        teacher_mean = getattr(self.teacher_obs_normalizer, "_mean", None)
        teacher_var = getattr(self.teacher_obs_normalizer, "_var", None)
        teacher_std = getattr(self.teacher_obs_normalizer, "_std", None)
        own_mean = getattr(self.student_proprio_obs_normalizer, "_mean", None)
        own_var = getattr(self.student_proprio_obs_normalizer, "_var", None)
        own_std = getattr(self.student_proprio_obs_normalizer, "_std", None)
        if not all(
            isinstance(t, torch.Tensor)
            for t in (teacher_mean, teacher_var, teacher_std, own_mean, own_var, own_std)
        ):
            return False
        teacher_dim = int(teacher_mean.shape[-1])
        lo, hi = int(jc_goal_dim), int(jc_goal_dim) + int(proprio_dim)
        if hi > teacher_dim:
            print(
                f"[Runner] WARNING: JC proprio slice [{lo}:{hi}] exceeds teacher norm dim "
                f"{teacher_dim}; cannot copy."
            )
            return False
        with torch.no_grad():
            own_mean.copy_(teacher_mean[..., lo:hi])
            own_var.copy_(teacher_var[..., lo:hi])
            own_std.copy_(teacher_std[..., lo:hi])
            if hasattr(self.student_proprio_obs_normalizer, "count") and hasattr(
                self.teacher_obs_normalizer, "count"
            ):
                self.student_proprio_obs_normalizer.count.copy_(self.teacher_obs_normalizer.count)
            if hasattr(self.student_proprio_obs_normalizer, "until"):
                self.student_proprio_obs_normalizer.until = getattr(
                    self.student_proprio_obs_normalizer, "count", 0
                )
        self.student_proprio_obs_normalizer.eval()
        return True

    def _assert_anybody_latent_proprio_alignment(
        self, student_obs: torch.Tensor, teacher_obs: torch.Tensor
    ) -> None:
        """Sanity check: student/teacher proprio tails must match for 2B."""
        if self.training_type != "anybody_latent_distillation":
            return
        proprio_dim = getattr(self.alg.policy, "proprio_dim", None)
        if not isinstance(proprio_dim, int) or proprio_dim <= 0:
            return
        if student_obs.shape[-1] < proprio_dim or teacher_obs.shape[-1] < proprio_dim:
            raise RuntimeError(
                f"[2B] Proprio alignment check failed: obs dims too small "
                f"(student={student_obs.shape[-1]}, teacher={teacher_obs.shape[-1]}, proprio_dim={proprio_dim})."
            )
        '''
        student_tail = student_obs[..., -proprio_dim:]
        teacher_tail = teacher_obs[..., -proprio_dim:]
        
        if not torch.allclose(student_tail, teacher_tail, rtol=1e-5, atol=1e-6):
            max_abs = (student_tail - teacher_tail).abs().max().item()
            raise RuntimeError(
                f"[2B] Student/teacher proprio tails diverged (max_abs_diff={max_abs:.3e}, "
                f"proprio_dim={proprio_dim}). This indicates an observation-layout bug."
            )
        '''

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
        eval_mode: bool = False,
        eval_fixed_mask_mode_idx: int | None = None,
        eval_log_key_prefix: str | None = None,
        eval_log_step_offset: int | None = None,
    ) -> None:  # noqa: C901
        if (
            isinstance(num_learning_iterations, bool)
            or not isinstance(num_learning_iterations, int)
            or num_learning_iterations < 0
        ):
            raise ValueError("num_learning_iterations must be a nonnegative integer")
        if num_learning_iterations == 0:
            return
        # MotionOnPolicyRunner may explicitly advance current for a training
        # chunk. Evaluation retains the checkpoint phase without advancing it.
        start_iter = self.current_learning_iteration if eval_mode else max(
            self.current_learning_iteration, self._next_learning_iteration
        )
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # Pass writer and log_interval to algorithm for logging (needed by MOSAIC)
        if hasattr(self, 'writer') and self.writer is not None:
            self.alg.writer = self.writer
            
        self.alg.log_interval = 1  # Default log interval

        # Check that teacher is loaded for checkpoint-based distillation
        if self.training_type in TEACHER_CHECKPOINT_DISTILLATION and not getattr(self.alg.policy, "loaded_teacher", True):
            raise ValueError(f"Teacher model parameters not loaded for {self.training_type} training.")

        # For MOSAIC multi-teacher: ensure env_group_name_to_idx is set before training
        if self.training_type == "mosaic" and hasattr(self.alg, 'use_multi_teacher') and self.alg.use_multi_teacher:
            if self.alg.env_group_name_to_idx is None:
                print("[Runner] Retrieving environment's group mapping...")
                env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                if hasattr(env, 'command_manager') and 'motion' in env.command_manager._terms:
                    motion_command = env.command_manager._terms['motion']
                    if hasattr(motion_command, 'group_name_to_idx'):
                        self.alg.env_group_name_to_idx = motion_command.group_name_to_idx
                        print(f"[Runner] Successfully retrieved environment's group mapping: {self.alg.env_group_name_to_idx}")
                    else:
                        raise RuntimeError(
                            "[Runner] FATAL: motion_command does not have 'group_name_to_idx' attribute!\n"
                            "Multi-teacher training cannot proceed without environment's group mapping."
                        )
                else:
                    raise RuntimeError(
                        "[Runner] FATAL: Cannot retrieve environment's group mapping!\n"
                        f"Environment type: {type(env)}\n"
                        f"Has command_manager: {hasattr(env, 'command_manager')}\n"
                        "Multi-teacher training cannot proceed."
                    )

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Keypoint-mask curriculum: iteration-phase bounds use virtual env-step = f(common_step_counter).
        env_u_curr = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        spe = int(self.num_steps_per_env)
        setattr(env_u_curr, "curriculum_env_steps_per_learning_iteration", spe)
        setattr(env_u_curr, "curriculum_counter_anchor", int(getattr(env_u_curr, "common_step_counter", 0)))
        setattr(env_u_curr, "curriculum_learning_iteration_anchor", int(start_iter))

        # start learning
        obs, extras = self.env.get_observations()
        #print(obs[0])
        obs_dict = extras.get("observations", {})
        if self.policy_obs_type is not None and self.policy_obs_type in obs_dict:
            obs = obs_dict[self.policy_obs_type]
        privileged_obs = obs_dict.get(self.privileged_obs_type, obs)
        teacher_obs = obs_dict.get(self.teacher_obs_type)
        obs = obs.to(self.device)
        privileged_obs = privileged_obs.to(self.device)
        if teacher_obs is not None:
            teacher_obs = teacher_obs.to(self.device)
        else:
            teacher_obs = privileged_obs
        self._assert_anybody_latent_proprio_alignment(obs, teacher_obs) # TODO: remove this
        # Initialize ref_vel_estimator observations (NO normalization!)
        ref_vel_estimator_obs = obs_dict.get(self.ref_vel_estimator_obs_type)
        if ref_vel_estimator_obs is not None:
            ref_vel_estimator_obs = ref_vel_estimator_obs.to(self.device)

        # Normalize initial observations (same as in training loop)
        obs = self._normalize_student_obs(obs)
        privileged_obs = replace_nonfinite_with_zeros(self.privileged_obs_normalizer(privileged_obs))
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)

        if eval_mode:
            self.eval_mode()  # disable dropout / keep normalizers fixed
        else:
            self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Velocity estimator error tracking
        vel_est_error_buffer = deque(maxlen=100)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Match mask row to motion EE metrics before the first rollout so early episode completions
        # do not log bucket errors with uninitialized mode indices (-1) or poison means with NaNs.
        env_u_pre = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        motion_term_pre = None
        if hasattr(env_u_pre, "command_manager") and "motion" in getattr(env_u_pre.command_manager, "_terms", {}):
            motion_term_pre = env_u_pre.command_manager.get_term("motion")

        if self.training_type == "latent_consistent_distillation" and hasattr(self.alg.policy, "sample_and_set_mask"):
            synced = False
            if motion_term_pre is not None:
                synced = sync_policy_keypoint_mask_from_motion_command(
                    policy=self.alg.policy,
                    motion_term=motion_term_pre,
                    num_envs=self.env.num_envs,
                    fixed_mode_idx=eval_fixed_mask_mode_idx,
                )
            if not synced:
                self.alg.policy.sample_and_set_mask(
                    self.env.num_envs, fixed_mode_idx=eval_fixed_mask_mode_idx
                )

        # Start training
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()

            # In eval_mode we skip alg.update(), which normally clears rollout storage.
            # Clear buffers explicitly to avoid Rollout buffer overflow.
            if eval_mode and hasattr(self.alg, "storage") and getattr(self.alg, "storage", None) is not None:
                self.alg.storage.clear()

            # Resample mask mode once per iteration (latent-consistent, 2B partial-obs w/ mask_cfg),
            # or sync mask rows from a partial-mask motion command (task-side sampling).
            if self.training_type == "latent_consistent_distillation" and hasattr(self.alg.policy, "sample_and_set_mask"):
                env_u_m = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
                motion_term_m = None
                if hasattr(env_u_m, "command_manager") and "motion" in getattr(env_u_m.command_manager, "_terms", {}):
                    motion_term_m = env_u_m.command_manager.get_term("motion")
                synced_m = False
                if motion_term_m is not None:
                    synced_m = sync_policy_keypoint_mask_from_motion_command(
                        policy=self.alg.policy,
                        motion_term=motion_term_m,
                        num_envs=self.env.num_envs,
                        fixed_mode_idx=eval_fixed_mask_mode_idx,
                    )
                if not synced_m:
                    self.alg.policy.sample_and_set_mask(
                        self.env.num_envs, fixed_mode_idx=eval_fixed_mask_mode_idx
                    )

            # MDP failure rate (Isaac Lab ``terminated``: non-``time_out`` terminations only; same as adaptive sampling)
            self._iter_mdp_failures = 0
            self._iter_mdp_episode_ends = 0

            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    if self.training_type == "mosaic":
                        # Extract motion groups for multi-teacher support
                        motion_groups = None
                        # CRITICAL FIX: Use unwrapped env to access command_manager
                        env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                        if hasattr(env, 'command_manager') and 'motion' in env.command_manager._terms:
                            motion_command = env.command_manager._terms['motion']
                            if hasattr(motion_command, 'env_motion_groups'):
                                motion_groups = motion_command.env_motion_groups.clone()

                        actions = self.alg.act(obs, privileged_obs, teacher_obs=teacher_obs, ref_vel_estimator_obs=ref_vel_estimator_obs, motion_groups=motion_groups)

                        # Track velocity estimator error if available
                        if hasattr(self.alg, 'last_estimated_ref_vel') and self.alg.last_estimated_ref_vel is not None:
                            # Get ground truth ref_anchor_lin_vel_b from environment using existing mdp function
                            # This uses anchor body coordinate system to match offline training
                            from whole_body_tracking.tasks.tracking.mdp import observations as mdp
                            gt_ref_vel_b = mdp.ref_base_lin_vel_b(self.env.unwrapped, "motion")  # [N, 3]

                            # Compute MAE (same metric as training validation)
                            # MAE per environment (averaging across 3 velocity dimensions)
                            vel_error = (self.alg.last_estimated_ref_vel - gt_ref_vel_b).abs().mean(dim=-1)  # [N]

                            vel_est_error_buffer.extend(vel_error.cpu().numpy().tolist())
                    elif self.training_type in TEACHER_CHECKPOINT_DISTILLATION:
                        # Distillation (incl. PULSE, latent-consistent, 2b): act(obs, teacher_obs)
                        actions = self.alg.act(obs, teacher_obs)
                    else:
                        actions = self.alg.act(obs, privileged_obs)

                    # Latent-space RL: ``actions`` is the latent (stored/learned on by
                    # PPO). The env needs joint targets, so run the frozen decoder
                    # (deterministic, no-grad) here; the decoded action is transient.
                    if self.training_type == "latent_rl":
                        step_actions = self.alg.policy.decode_for_env(actions, obs)
                    else:
                        step_actions = actions

                    # Step the environment
                    obs, rewards, dones, infos = self.env.step(step_actions.to(self.env.device))
                    # Move to device
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    # Aggregate MDP terminations over the full rollout (logged once per iteration in ``log``)
                    done_mask = dones.bool()
                    if done_mask.any():
                        core = self.env.unwrapped
                        if hasattr(core, "reset_terminated"):
                            term = core.reset_terminated
                            self._iter_mdp_failures += int(term[done_mask].sum().item())
                            self._iter_mdp_episode_ends += int(done_mask.sum().item())
                    obs_dict = infos.get("observations", {})
                    if self.policy_obs_type is not None and self.policy_obs_type in obs_dict:
                        obs = obs_dict[self.policy_obs_type].to(self.device)
                    else:
                        obs = obs.to(self.device)
                    teacher_obs_raw = None
                    if self.teacher_obs_type is not None and self.teacher_obs_type in obs_dict:
                        teacher_obs_raw = obs_dict[self.teacher_obs_type].to(self.device)
                        self._assert_anybody_latent_proprio_alignment(obs, teacher_obs_raw) # TODO: remove this
                    # perform normalization
                    obs = self._normalize_student_obs(obs)
                    if self.privileged_obs_type is not None and self.privileged_obs_type in obs_dict:
                        privileged_obs = replace_nonfinite_with_zeros(
                            self.privileged_obs_normalizer(
                                obs_dict[self.privileged_obs_type].to(self.device)
                            )
                        )
                    else:
                        privileged_obs = obs
                    if teacher_obs_raw is not None:
                        teacher_obs = self.teacher_obs_normalizer(teacher_obs_raw)
                    else:
                        teacher_obs = privileged_obs
                    # Extract ref_vel_estimator observations (NO normalization - must match offline training!)
                    if self.ref_vel_estimator_obs_type is not None and self.ref_vel_estimator_obs_type in obs_dict:
                        ref_vel_estimator_obs = obs_dict[self.ref_vel_estimator_obs_type].to(self.device)
                    else:
                        ref_vel_estimator_obs = None

                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns (training-only)
                # NOTE: "latent_rl" MUST be here. It is a PPO-style on-policy type
                # (rl-shaped RolloutStorage, GAE). Omitting it leaves
                # storage.advantages/returns at their init ZEROS -> surrogate
                # loss ≡ 0, zero policy gradient, actor never updates (entropy
                # frozen, KL≡0, LR pinned), value_loss collapses to a tiny
                # constant (critic chases zero returns). Adapter-independent.
                if not eval_mode and self.training_type in ["rl", "mosaic", "latent_rl"]:
                    self.alg.compute_returns(privileged_obs)

            loss_dict = {}
            stop = time.time()
            learn_time = stop - start

            # Update policy (training-only)
            if not eval_mode:
                self.alg.current_learning_iteration = it
                if self.training_type in UPDATE_WITH_CURRENT_ITER:
                    loss_dict = self.alg.update(current_iter=it)
                else:
                    loss_dict = self.alg.update()

                stop = time.time()
                learn_time = stop - start
                self.current_learning_iteration = it
                self._last_completed_learning_iteration = it
                self._next_learning_iteration = it + 1
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if (not eval_mode) and it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if (not eval_mode) and self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _eval_scalar_log_tag(self, tag: str, locs: dict) -> str:
        p = locs.get("eval_log_key_prefix")
        return f"{p}/{tag}" if p else tag

    def _eval_scalar_log_step(self, locs: dict) -> int | float:
        if locs.get("eval_log_step_offset") is not None:
            return (locs["it"] - locs["start_iter"]) + int(locs["eval_log_step_offset"])
        return locs["it"]

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]
        ls = self._eval_scalar_log_step(locs)

        # Console-log decluttering for MUSE-family distillation runs, detected by the
        # behavior+regularization loss signature (PPO has neither; the muse_kp teacher_eval
        # path has only "behavior" so it is left verbose on purpose). This only trims the
        # printed terminal columns — every key is still written to wandb below.
        _muse_style_log = "behavior" in locs["loss_dict"] and "regularization" in locs["loss_dict"]
        _LOSS_CONSOLE_KEEP = ("behavior", "regularization", "reg_contribution", "reg_to_behavior_ratio")
        _SAMPLING_METRICS_DROP = (
            "sampling_entropy",
            "sampling_top1_prob",
            "sampling_top1_bin",
            "motion_sampling_prob_mean",
            "motion_sampling_prob_std",
            "motion_sampling_prob_min",
            "motion_sampling_prob_max",
            "motion_sampling_prob_entropy",
        )

        def _console_loss_skip(k: str) -> bool:
            return _muse_style_log and k not in _LOSS_CONSOLE_KEEP

        def _console_metric_skip(k: str) -> bool:
            return _muse_style_log and k.split("/")[-1] in _SAMPLING_METRICS_DROP

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            ep_keys: set[str] = set()
            for ep_info in locs["ep_infos"]:
                ep_keys.update(ep_info.keys())
            for key in sorted(ep_keys):
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                if infotensor.numel() == 0:
                    continue
                value = torch.nanmean(infotensor)
                if not torch.isfinite(value).all():
                    continue
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(self._eval_scalar_log_tag(key, locs), value, ls)
                    if not _console_metric_skip(key):
                        ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar(self._eval_scalar_log_tag("Episode/" + key, locs), value, ls)
                    if not _console_metric_skip(key):
                        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        #mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(self._eval_scalar_log_tag(f"Loss/{key}", locs), value, ls)
        self.writer.add_scalar(self._eval_scalar_log_tag("Loss/learning_rate", locs), self.alg.learning_rate, ls)

        # -- Policy
        #self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar(self._eval_scalar_log_tag("Perf/total_fps", locs), fps, ls)
        self.writer.add_scalar(self._eval_scalar_log_tag("Perf/collection time", locs), locs["collection_time"], ls)
        self.writer.add_scalar(self._eval_scalar_log_tag("Perf/learning_time", locs), locs["learn_time"], ls)

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            if self.alg.rnd:
                self.writer.add_scalar(
                    self._eval_scalar_log_tag("Rnd/mean_extrinsic_reward", locs),
                    statistics.mean(locs["erewbuffer"]),
                    ls,
                )
                self.writer.add_scalar(
                    self._eval_scalar_log_tag("Rnd/mean_intrinsic_reward", locs),
                    statistics.mean(locs["irewbuffer"]),
                    ls,
                )
                self.writer.add_scalar(self._eval_scalar_log_tag("Rnd/weight", locs), self.alg.rnd.weight, ls)
            # everything else
            self.writer.add_scalar(self._eval_scalar_log_tag("Train/mean_reward", locs), statistics.mean(locs["rewbuffer"]), ls)
            self.writer.add_scalar(
                self._eval_scalar_log_tag("Train/mean_episode_length", locs), statistics.mean(locs["lenbuffer"]), ls
            )
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar(
                    self._eval_scalar_log_tag("Train/mean_reward/time", locs),
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    self._eval_scalar_log_tag("Train/mean_episode_length/time", locs),
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        # MDP termination success/failure over environments that finished an episode this iteration
        n_mdp_end = getattr(self, "_iter_mdp_episode_ends", 0)
        if n_mdp_end > 0:
            n_mdp_fail = getattr(self, "_iter_mdp_failures", 0)
            fail_rate = n_mdp_fail / float(n_mdp_end)
            self.writer.add_scalar(
                self._eval_scalar_log_tag("Episode/mdp_termination_success_rate", locs),
                1.0 - fail_rate,
                ls,
            )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                #f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                if _console_loss_skip(key):
                    continue
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- Velocity estimator error (if available)
            if 'vel_est_error_buffer' in locs and len(locs['vel_est_error_buffer']) > 0:
                log_string += f"""{'Mean vel_estimator error:':>{pad}} {statistics.mean(locs['vel_est_error_buffer']):.4f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                #f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                if _console_loss_skip(key):
                    continue
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        n_mdp_end = getattr(self, "_iter_mdp_episode_ends", 0)
        if n_mdp_end > 0:
            n_mdp_fail = getattr(self, "_iter_mdp_failures", 0)
            log_string += f"""{'MDP term. failure rate:':>{pad}} {n_mdp_fail / float(n_mdp_end):.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos: dict | None = None) -> None:
        # Check if using ResidualActorCritic (special handling)
        from rsl_rl.modules import ResidualActorCritic
        if isinstance(self.alg.policy, ResidualActorCritic):
            # Save only residual network + critic (GMT is frozen, no need to save)
            model_state_dict = {
                'residual_actor': self.alg.policy.residual_actor.state_dict(),
                'critic': self.alg.policy.critic.state_dict(),
            }
            # Save noise std parameter
            if hasattr(self.alg.policy, 'std'):
                model_state_dict['std'] = self.alg.policy.std
            elif hasattr(self.alg.policy, 'log_std'):
                model_state_dict['log_std'] = self.alg.policy.log_std
        else:
            # Standard save: entire policy
            model_state_dict = self.alg.policy.state_dict()

        # -- Save model
        saved_dict = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "infos": infos,
        }
        saved_dict.update(checkpoint_progress(
            self._last_completed_learning_iteration, self._next_learning_iteration,
            phase=self.current_learning_iteration,
        ))
        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        if getattr(self.alg, "disc_optimizer", None) is not None:
            saved_dict["disc_optimizer_state_dict"] = self.alg.disc_optimizer.state_dict()
        # -- Save observation normalizer if used
        save_normalizer_states(
            saved_dict=saved_dict,
            empirical_normalization=self.empirical_normalization,
            use_split_normalizers=self._use_anybody_latent_split_normalizers(),
            obs_normalizer=self.obs_normalizer,
            privileged_obs_normalizer=self.privileged_obs_normalizer,
            student_goal_obs_normalizer=self.student_goal_obs_normalizer,
            student_proprio_obs_normalizer=self.student_proprio_obs_normalizer,
            training_type=self.training_type,
            teacher_obs_normalizer=self.teacher_obs_normalizer,
        )

        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_optimizer: bool = True, load_critic: bool = True,
        load_iteration: bool = True,
    ) -> dict | None:
        """Restore a checkpoint; weight warmstarts can explicitly retain fresh progress.

        Full PPO resume restores Adam and its scalar LR, then starts at the next
        update. Weight-only evaluation keeps the legacy checkpoint phase and
        configured optimizer/LR. This does not restore simulator or RNG state.
        """
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)

        # Check if using ResidualActorCritic (special handling)
        from rsl_rl.modules import ResidualActorCritic
        if isinstance(self.alg.policy, ResidualActorCritic):
            # Load only residual network + critic (GMT is already loaded in __init__)
            self.alg.policy.residual_actor.load_state_dict(loaded_dict["model_state_dict"]["residual_actor"])
            if load_critic:
                self.alg.policy.critic.load_state_dict(loaded_dict["model_state_dict"]["critic"])
            # Load noise std parameter
            if "std" in loaded_dict["model_state_dict"]:
                self.alg.policy.std.data = loaded_dict["model_state_dict"]["std"].data
            elif "log_std" in loaded_dict["model_state_dict"]:
                self.alg.policy.log_std.data = loaded_dict["model_state_dict"]["log_std"].data
            if load_critic:
                print("[Runner] Loaded residual network + critic from checkpoint (GMT remains frozen)")
            else:
                print("[Runner] Loaded residual network only (skipping critic from checkpoint)")
            resumed_training = True
        else:
            if load_critic:
                # Standard load: entire policy
                resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
            else:
                actor_only_state_dict = {
                    key: value
                    for key, value in loaded_dict["model_state_dict"].items()
                    if not key.startswith("critic.")
                }
                resumed_training = self.alg.policy.load_state_dict(actor_only_state_dict, strict=False)

        # Load RND model if used
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        # Load observation normalizers if used
        if self.empirical_normalization:
            load_normalizer_states_on_resume(
                loaded_dict=loaded_dict,
                load_critic=load_critic,
                resumed_training=resumed_training,
                training_type=self.training_type,
                is_residual_policy=isinstance(self.alg.policy, ResidualActorCritic),
                use_split_normalizers=self._use_anybody_latent_split_normalizers(),
                obs_normalizer=self.obs_normalizer,
                privileged_obs_normalizer=self.privileged_obs_normalizer,
                teacher_obs_normalizer=self.teacher_obs_normalizer,
                student_goal_obs_normalizer=self.student_goal_obs_normalizer,
                student_proprio_obs_normalizer=self.student_proprio_obs_normalizer,
                anybody_latent_proprio_dim=self._anybody_latent_proprio_dim,
                alg=self.alg,
            )
        # -- load optimizer if used
        optimizer_loaded = False
        if load_optimizer and resumed_training:
            if not load_critic:
                print("[Runner] Skipping optimizer load because load_critic=False.")
            else:
                # Validate outside the legacy stage-transfer fallback: silently
                # accepting an invalid LR would corrupt a requested PPO resume.
                if self.training_type == "rl":
                    restored_lr = ppo_optimizer_learning_rate(loaded_dict["optimizer_state_dict"])
                try:
                    # -- algorithm optimizer
                    self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                    optimizer_loaded = True
                    print("[Runner] Loaded optimizer state from checkpoint.")
                except (ValueError, KeyError) as e:
                    if self.training_type == "rl":
                        raise RuntimeError(
                            "Cannot fully resume PPO optimizer; use an explicit weight-only warmstart "
                            "for a different training stage."
                        ) from e
                    # Optimizer state mismatch (e.g., different parameter groups between stages)
                    # This can happen when:
                    # - Stage 1 had frozen critic (optimizer only has actor params)
                    # - Stage 2 unfreezes critic (optimizer has actor + critic params)
                    print(f"[Runner] WARNING: Could not load optimizer state: {e}")
                    print("[Runner] Optimizer will be initialized from scratch (learning rate, momentum, etc. reset)")
                    print("[Runner] This is expected when transitioning between training stages with different frozen parameters.")

                if optimizer_loaded and self.training_type == "rl":
                    # Restore only the scalar consumed by PPO's existing
                    # adaptive rule; do not modify loaded optimizer groups.
                    self.alg.learning_rate = restored_lr

                # -- RND optimizer if used
                if self.alg.rnd:
                    self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

                if getattr(self.alg, "disc_optimizer", None) is not None and "disc_optimizer_state_dict" in loaded_dict:
                    try:
                        self.alg.disc_optimizer.load_state_dict(loaded_dict["disc_optimizer_state_dict"])
                        print("[Runner] Loaded discriminator optimizer state from checkpoint.")
                    except (ValueError, KeyError) as e:
                        print(f"[Runner] WARNING: Could not load discriminator optimizer state: {e}")
        # -- load current learning iteration
        if resumed_training and load_iteration:
            last_completed, next_iteration = resolve_checkpoint_progress(loaded_dict)
            self.current_learning_iteration = resolve_checkpoint_phase(loaded_dict)
            # Progress loading is independent of optimizer loading. Evaluation
            # uses current phase; training never repeats a completed update.
            # An actual fresh warmstart uses load_iteration=False explicitly.
            self._last_completed_learning_iteration = last_completed
            self._next_learning_iteration = next_iteration

        # -- Reset noise std if specified in config (for stage transitions)
        # This allows changing exploration noise when resuming from a checkpoint
        reset_noise = self.cfg.get("reset_noise_std_on_resume", False)
        print(f"[Runner] reset_noise_std_on_resume = {reset_noise}")
        if reset_noise:
            init_noise_std = self.policy_cfg.get("init_noise_std", 1.0)
            noise_std_type = self.policy_cfg.get("noise_std_type", "scalar")
            print(f"[Runner] init_noise_std from config = {init_noise_std}, noise_std_type = {noise_std_type}")
            num_actions = self.alg.policy.std.shape[0] if hasattr(self.alg.policy, 'std') else self.alg.policy.log_std.shape[0]

            if noise_std_type == "scalar":
                self.alg.policy.std.data = torch.ones(num_actions, device=self.device) * init_noise_std
                print(f"[Runner] Reset noise std to {init_noise_std} (reset_noise_std_on_resume=True)")
            elif noise_std_type == "log":
                self.alg.policy.log_std.data = torch.log(torch.ones(num_actions, device=self.device) * init_noise_std)
                print(f"[Runner] Reset log noise std to log({init_noise_std}) (reset_noise_std_on_resume=True)")

        # -- Freeze normalizer if specified in config (for stage transitions)
        # This prevents normalizer statistics from drifting when resuming from distillation
        freeze_normalizer = self.cfg.get("freeze_normalizer_on_resume", False)
        print(f"[Runner] freeze_normalizer_on_resume = {freeze_normalizer}")
        freeze_normalizers_on_resume(
            empirical_normalization=self.empirical_normalization,
            freeze_normalizer=freeze_normalizer,
            use_split_normalizers=self._use_anybody_latent_split_normalizers(),
            obs_normalizer=self.obs_normalizer,
            privileged_obs_normalizer=self.privileged_obs_normalizer,
            student_goal_obs_normalizer=self.student_goal_obs_normalizer,
            student_proprio_obs_normalizer=self.student_proprio_obs_normalizer,
        )

        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
                if self.student_goal_obs_normalizer is not None:
                    self.student_goal_obs_normalizer.to(device)
                if self.student_proprio_obs_normalizer is not None:
                    self.student_proprio_obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self._normalize_student_obs(x))  # noqa: E731
        return policy

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- Normalization
        if self.empirical_normalization:
            if self._use_anybody_latent_split_normalizers():
                self.student_goal_obs_normalizer.train()
                # proprio normalizer stays fixed from teacher for 2B
                self.student_proprio_obs_normalizer.eval()
            else:
                self.obs_normalizer.train()
            if self._obs_normalizer_loaded_from_checkpoint:
                self.obs_normalizer.eval()
            self.privileged_obs_normalizer.train()
            # Teacher normalizer should remain frozen for MOSAIC
            if self.training_type == "mosaic" and hasattr(self, 'teacher_obs_normalizer'):
                if not isinstance(self.teacher_obs_normalizer, torch.nn.Identity):
                    self.teacher_obs_normalizer.eval()  # Keep frozen
            # For teacher-checkpoint distillation, obs_normalizer is the teacher's; keep frozen
            if self.training_type in TEACHER_CHECKPOINT_DISTILLATION:
                self.teacher_obs_normalizer.eval()
                if self.obs_normalizer is self.teacher_obs_normalizer and not isinstance(self.obs_normalizer, torch.nn.Identity):
                    self.obs_normalizer.eval()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- Normalization
        if self.empirical_normalization:
            if self._use_anybody_latent_split_normalizers():
                self.student_goal_obs_normalizer.eval()
                self.student_proprio_obs_normalizer.eval()
            else:
                self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()
            # Teacher normalizer should remain frozen for MOSAIC
            if self.training_type == "mosaic" and hasattr(self, 'teacher_obs_normalizer'):
                if not isinstance(self.teacher_obs_normalizer, torch.nn.Identity):
                    self.teacher_obs_normalizer.eval()  # Keep frozen

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'.")
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")

        # initialize torch distributed
        torch.distributed.init_process_group(
            backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
        )
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)


if __name__ == "__main__":
    from rsl_rl.utils.checkpoint_resume import sanity_check

    sanity_check()
