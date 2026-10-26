import gymnasium as gym

from . import agents

# Load before flat_env_cfg so tracking_env_cfg can import ``mask_modes`` without deadlock
# (flat_env_cfg imports tracking_env_cfg, which references g1.mask_modes).
from . import mask_modes as _mask_modes  # noqa: F401


def _register_g1_tasks() -> None:
    """Register Gym envs; deferred so ``mask_modes`` is importable during flat_env_cfg/tracking_env_cfg load."""
    from . import flat_env_cfg

    ######### 1. RL-based tracking environments #########

    gym.register(
        id="Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
    )

    gym.register(
        id="Tracking-Flat-G1-Wo-State-Estimation-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
    )

    gym.register(
        id="Tracking-Flat-G1-Low-Freq-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1FlatLowFreqEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg",
        },
    )

    gym.register(
        id="General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1FlatGeneralEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
    )

    gym.register(
        id="General-Tracking-Flat-G1-Wo-State-Estimation-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationGeneralEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
    )

    gym.register(
        id="General-Tracking-Flat-G1-Low-Freq-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1FlatLowFreqGeneralEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg",
        },
    )

    gym.register(
        id="Expert-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1FlatExpertGeneralEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
    )

    gym.register(
        id="Expert-General-Tracking-Flat-G1-MOSAIC-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1FlatExpertGeneralEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_mosaic_cfg:G1FlatMOSAICRunnerCfg",
        },
    )

    # MOSAIC GMT
    gym.register(
        id="General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1OneStageTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
    )

    # Separately selectable repair recipes; legacy task IDs remain available.
    for task_id, env_config in (
        ("General-Tracking-Flat-G1-Teacher-Cache-v1", flat_env_cfg.G1OneStageTrackingEnvCfg),
        ("General-Tracking-Flat-G1-Teacher-Wrists-v1", flat_env_cfg.G1OneStageWristRewardV1EnvCfg),
        ("General-Tracking-Flat-G1-Teacher-Points-v1", flat_env_cfg.G1OneStagePointRewardsV1EnvCfg),
        ("General-Tracking-Flat-G1-Teacher-FullB-Points-v1", flat_env_cfg.G1OneStageFullBPointRewardsV1EnvCfg),
    ):
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": env_config,
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:"
                + ("G1FullBPPORunnerCfg" if "Teacher-FullB-" in task_id else "G1FlatPPORunnerCfg"),
            },
        )

    ######### 2. Distillation-based tracking environments #########

    gym.register(
        id="Distillation-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1DistillationTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatDistillationRunnerCfg",
        },
    )

    # 2.1 PULSE Distillation
    gym.register(
        id="PULSE-Distill-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1PULSEDistillationTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPULSEDistillationRunnerCfg",
        },
    )

    gym.register(
        id="PULSE-AdvDistill-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1PULSEDistillationTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatAdvPULSEDistillationRunnerCfg",
        },
    )

    gym.register(
        id="PULSE-PriorOnly-Distill-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1PULSEDistillationTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPriorOnlyPULSEDistillationRunnerCfg",
        },
    )

    # 2.2 MUSE Distillation (delta-command + random goal masking; no separate prior)
    gym.register(
        id="MUSE-Distill-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1MUSEDistillationTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatMUSEDistillationRunnerCfg",
        },
    )

    # 2.2b MUSE Transformer Distillation (per-modality tokens, attention masking, NaN obs)
    gym.register(
        id="MUSE-Transformer-Distill-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1MUSETransformerDistillationTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatMUSETransformerDistillationRunnerCfg",
        },
    )

    # 2.2e MUSE-KP Distillation (KP transformer encoder warmstarted from MUSE-Transformer)
    gym.register(
        id="MUSE-Kp-Distill-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1MUSEKpDistillationTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatMUSEKpDistillationRunnerCfg",
        },
    )

    # 2.2e* MUSE-KP LATENT-space Distillation: KP encoder distilled against a frozen JC
    # MUSE-Transformer teacher (latent μ_jc target + behavior anchor), shared decoder frozen.
    # Single ckpt via --teacher_checkpoint (the JC MUSE-T .pt) drives jc_encoder + frozen
    # decoder + KP backbone warmstart + teacher_obs_normalizer + frozen student proprio-norm slice.
    gym.register(
        id="MUSE-Kp-LatentDistill-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1MUSEKpLatentDistillationTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatMUSEKpLatentDistillationRunnerCfg",
        },
    )

    # 2.2e'' MUSE-Kp5 from-scratch + passive→active pilot anneal (5-body-native, 1s lookahead).
    # No MUSE-Kp warmstart (backbone+decoder from JC MUSE-Transformer via --encoder_decoder_warmstart;
    # KP front-end fresh). No aux predictor. Teacher pilots through the mask ramp, then anneals to
    # student. The headline run to break the ~85% sparse-5-body ceiling.
    gym.register(
        id="MUSE-Kp5-FromScratch-Distill-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1MUSEKp5FromScratchTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatMUSEKp5FromScratchRunnerCfg",
        },
    )

    # 2.2e''' MUSE-Kp Latent-space RL finetune: PPO over the distilled MUSE-Kp latent
    # (frozen decoder = motor prior). Warmstart the distilled MUSE-Kp (KP6, 0.5 s) via
    # --encoder_decoder_warmstart; reward = world-frame visible-POI accuracy (decision D5).
    # M1: adapter="full_ft", unanchored. See docs/latent_rl_finetune_plan.md.
    gym.register(
        id="MUSE-Kp-LatentRL-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1MUSEKpLatentRLKp5TrackingEnvCfg,
            # KP5 env (15-slot sym_sparse_0p5s, 5 bodies, num_teacher_obs=524) ⇒ MUST use the KP5
            # runner. The legacy G1FlatMUSEKpLatentRLRunnerCfg is the stale 12-slot log_0_5s /
            # num_teacher_obs=815 config and mismatches this env (expected_obs_dim 690 != 750).
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatMUSEKpLatentRLKp5RunnerCfg",
        },
    )

    # 2.2e'''-VERIFY  TEMPORARY wiring sanity check (delete after): old distillation
    # reward + latent-RL critic obs + latent-PPO with critic-only warmup. The frozen
    # actor = the warmstarted distilled policy, so reward/success must reproduce the
    # distillation run's numbers or the latent->decode->env plumbing is buggy.
    gym.register(
        id="MUSE-Kp-LatentRL-VERIFY-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1MUSEKpLatentRLVerifyTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatMUSEKpLatentRLRunnerCfg",
        },
    )

    # KP5 latent-RL (canonical 5-point recipe; KP6 above is legacy). Same world-POI
    # reward + privileged critic + residual adapter as the KP6 task, but the encoder is
    # KP5 (sym_sparse_0p5s, 15 slots) to warmstart-load the kp5_latent_distill checkpoint.
    gym.register(
        id="MUSE-Kp-LatentRL-Kp5-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1MUSEKpLatentRLKp5TrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatMUSEKpLatentRLKp5RunnerCfg",
        },
    )

    # KP5 latent-RL WRITING specialization: mask pinned right-wrist-only + reset-to-frame-0
    # (synth writing clips are valid only at frame 0). Train on the npz_synth_pool word pool.
    gym.register(
        id="MUSE-Kp-LatentRL-Kp5-Writing-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1MUSEKpLatentRLKp5WritingTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatMUSEKpLatentRLKp5RunnerCfg",
        },
    )

    # KP5 latent-RL OBSTACLE-REACH demo: reach a fixed point past an obstacle. One task,
    # split specialists — the phase (0 free / 1 above box / 2 into container / 4 under slab)
    # is selected by --motion <pool> (pools from scripts/gen_obstacle_clips.py).
    gym.register(
        id="MUSE-Kp-LatentRL-Kp5-ObstacleReach-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1MUSEKpLatentRLKp5ObstacleReachTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatMUSEKpLatentRLKp5ObstacleReachRunnerCfg",
        },
    )

    # 2.3 Anybody keypoint distillation:
    gym.register(
        id="Residual-Latent-Distill-2B-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1VAEDistillationTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatAnyBodyLatentDistillationRunnerCfg",
        },
    )

    gym.register(
        id="Partial-Masked-Residual-Latent-Distill-2B-General-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": flat_env_cfg.G1PartialMaskedVAEDistillationTrackingEnvCfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPartialMaskedAnyBodyLatentDistillationRunnerCfg",
        },
    )


_register_g1_tasks()
