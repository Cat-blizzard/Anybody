from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlDistillationAlgorithmCfg,
)
from whole_body_tracking.utils.rsl_rl_cfg import (
    RslRlDistillationCfg,
    RslRlMUSEAlgorithmCfg,
    RslRlMUSEDistillationCfg,
    RslRlMUSETransformerDistillationCfg,
    RslRlMUSEKpAlgorithmCfg,
    RslRlMUSEKpDistillationCfg,
    RslRlMUSEKpLatentDistillationCfg,
    RslRlMUSEKpLatentDistillationAlgorithmCfg,
    RslRlLatentRLActorCriticCfg,
    RslRlLatentPPOAlgorithmCfg,
    RslRlPULSEDistillationCfg,
    RslRlPULSEAdvDistillationCfg,
    RslRlPULSEAlgorithmCfg,
    RslRlAdvPulseAlgorithmCfg,
    RslRlPriorOnlyPULSEAlgorithmCfg,
    RslRlAnyBodyLatentDistillationCfg,
    RslRlAnyBodyLatentDistillationAlgorithmCfg,
)
from whole_body_tracking.tasks.tracking.config.g1 import mask_modes
from rsl_rl.modules.latent_bottleneck_muse_kp import (
    kp_layout_by_name,
)

@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_mosaic_hybrid"
    # experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 1024, 512, 512, 256, 256], #[512, 256, 128],
        critic_hidden_dims=[1024, 1024, 512, 512, 256, 256], #[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1FullBPPORunnerCfg(G1FlatPPORunnerCfg):
    """Full-B teacher, unchanged PPO knobs plus explicit DDP correctness gates.

    Normalizer synchronization is a local implementation, not an author preset.
    The stopping threshold keeps the original per-rank sample-count semantics.
    """

    max_iterations = 100000
    save_interval = 1000
    experiment_name = "g1_full_B_ddp8_points_v1"
    distributed_normalizer_sync: bool = True
    normalizer_until_count_scope: str = "per_rank_equivalent"

    def __post_init__(self) -> None:
        self.algorithm.strict_finite_checks = True


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)

@configclass
class G1FlatDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlDistillationCfg(
        class_name="StudentTeacher",
        init_noise_std=1.0,
        student_hidden_dims=[1024, 1024, 512, 256],
        teacher_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length = 15
    )

@configclass
class G1FlatKLDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Configuration for KL-based distillation (improved version).

    This uses KL divergence loss instead of MSE, matching MOSAIC's approach.
    Expected to provide better imitation performance than standard distillation.
    """
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_kl_distillation"
    empirical_normalization = True

    policy = RslRlDistillationCfg(
        class_name="StudentTeacher",
        init_noise_std=1.0,
        student_hidden_dims=[1024, 1024, 512, 256],
        teacher_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
    )



@configclass
class G1FlatPULSEDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    PULSE-style VAE distillation runner on G1.

    This mirrors the VAE latent-consistent runner's architecture (same
    Student/Teacher VAE bottleneck policy) but uses the standard distillation
    algorithm without masking or latent consistency losses.
    """

    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_pulse_distillation"
    empirical_normalization = True
    # When resuming from a distillation checkpoint, keep normalizer statistics fixed.
    # This helps avoid sudden loss spikes caused by normalizer drift.
    freeze_normalizer_on_resume = True
    # Periodic prior-only eval on training envs (no effect on default training logs).
    prior_eval_enabled = True
    prior_eval_rollout_every_itr = 1000
    prior_eval_rollout_itr = 10  # number of prior-eval episodes per trigger
    prior_eval_rollout_steps = 400  # horizon per prior-eval episode
    prior_eval_log_interval = 50
    prior_eval_fixed_latent_std = None
    prior_eval_fall_term_name = "fall"
    prior_eval_fall_body_names = ["torso_link"]
    prior_eval_fall_min_height = 0.35
    # Prior eval: same motion clip assignment across training triggers and across runs; per-eval-episode
    # index varies via ``prior_eval_motion_sample_seed + episode``. Pose/joint randomization still from env.
    prior_eval_deterministic_motion = True
    prior_eval_motion_sample_seed = 42

    policy = RslRlPULSEDistillationCfg(
        class_name="LatentBottleneckPULSE",
        init_noise_std=1.0,
        proprio_dim=450,
        latent_dim=16, 
        encoder_hidden_dims=[512,256,128], #[512,256,128,]
        decoder_hidden_dims=[512,256,128], #[512,256,128,]
        prior_hidden_dims=[512, 256, 128], #[512,256,128,]
        teacher_hidden_dims=[1024, 1024, 512, 512, 256, 256], # [1024, 1024, 512, 256,]
        activation="elu",
        initialize_std=0.1,
        latent_predict_std_min=0.001,
        latent_predict_std_max=1.0,
        fixed_prior_std=0.5,      
        fixed_encoder_std=0.5/1.414, #/sqrt(2)
    )
    algorithm = RslRlPULSEAlgorithmCfg(
        weight_kl=0.02,
        weight_kl_end=0.02, #0.001,
        kl_anneal_start_iter=4000,
        kl_anneal_end_iter=8000,
        weight_regularization=0.005, # TODO: finetune this; #0.005
        kl_loss_upper_bound=1e6, # no clamp
    )


@configclass
class G1FlatMUSEDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """MUSE distillation runner on G1.

    Encoder + decoder only (no prior); training uses BC + temporal latent regularization, no KL.
    Goal block is randomly masked during training with probability ``p_mask`` (curriculum-driven on
    the env side); the encoder learns to act as an implicit prior under masking.
    """

    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_muse_distillation"
    empirical_normalization = True
    freeze_normalizer_on_resume = True

    policy = RslRlMUSEDistillationCfg(
        class_name="LatentBottleneckMUSE",
        init_noise_std=1.0,
        proprio_dim=450,
        latent_dim=16,
        encoder_hidden_dims=[512, 256, 128],
        decoder_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[1024, 1024, 512, 512, 256, 256],
        activation="elu",
        initialize_std=-1.0,  # σ is fixed; no head to initialize.
        latent_predict_std_min=0.001,
        latent_predict_std_max=1.0,
        # Without KL there is nothing to regularize a learned σ; force a constant per-dim σ to keep
        # the decoder learning latent-neighborhood smoothness via reparameterization noise.
        fixed_encoder_std=0.3,
    )
    algorithm = RslRlMUSEAlgorithmCfg(
        weight_regularization=0.005,
    )


@configclass
class G1FlatMUSETransformerDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """MUSE-Transformer distillation, post-refactor to a SAGE-II-shaped encoder layout.

    Encoder: 1 un-masked command token + H=5 proprio tokens (+ [CLS]) → transformer → μ. Env's
    ``goal_mask_history`` (1-step) drops the command token from attention via ``key_padding_mask``;
    curriculum ramps p_mask 0.0 → 0.5 over iter 500 → 4000.

    Decoder consumes the full proprio history flattened (450-d for G1) by default — same shape as
    the MLP-MUSE decoder. Set ``decoder_one_step_proprio=True`` to revert to current-frame proprio
    only.

    Algorithm: BC + fixed-weight cosine smoothness on μ. Deterministic encoder (no reparameterization
    noise) + unit-norm latent (z on the unit hypersphere). Framing: this is input-modal conditioning,
    not output-multimodal sampling — no need for stochastic z or a Gaussian prior. With ‖μ‖=1 the
    L2-delta failure mode (smoothing pressure → encoder amplifies ‖μ‖) is structurally impossible,
    so Kendall-adaptive weighting is unnecessary; a small fixed weight is cleaner.
    """

    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_muse_transformer_distillation"
    empirical_normalization = True
    freeze_normalizer_on_resume = True

    policy = RslRlMUSETransformerDistillationCfg(
        class_name="LatentBottleneckMUSETransformer",
        init_noise_std=1.0,
        proprio_dim=450,  # placeholder; encoder ignores
        history_length=5,
        goal_term_sizes=[58, 6, 3, 3, 3],
        proprio_term_sizes=[29, 29, 3, 29],
        mask_term_size=1,
        latent_dim=16,
        d_model=192, #128,
        nhead=4,
        num_layers=2,
        ffn_dim=768, #256,
        decoder_hidden_dims=[1024,512,256,128],
        teacher_hidden_dims=[1024, 1024, 512, 512, 256, 256],
        activation="gelu",
        encoder_dropout=0.0,
        latent_normalize=True,
        deterministic_encoder=True,
    )
    algorithm = RslRlMUSEAlgorithmCfg(
        # Cosine smoothness on the unit-norm μ trajectory. Bounded in [0, 2]; with w=0.01 the
        # contribution is at most 0.02 — light touch, comparable to BC O(0.01–0.1). Adaptive
        # Kendall is unnecessary now that ‖μ‖ can't drift.
        smoothness_type="cosine",
        weight_regularization=0.1, 
        use_adaptive_regularization=False,
    )


@configclass
class G1FlatMUSEKpDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """MUSE-Kp distillation runner on G1: KP-token student warmstarted from MUSE-Transformer.

    Encoder shape (d_model / nhead / num_layers / ffn_dim / decoder_hidden_dims) **mirrors**
    :class:`G1FlatMUSETransformerDistillationRunnerCfg.policy` exactly so the MUSE-Transformer
    checkpoint loaded via ``--teacher_checkpoint`` warmstarts cleanly into the shared encoder
    + decoder + MLP teacher. KP-only layers (``kp_proj``, ``body_id_emb``, ``modality_emb`` slot 0)
    keep their small random init.

    Loss / training recipe is identical to MUSE-Transformer (BC vs MLP teacher + cosine smoothness
    on unit-norm μ); the only change is the encoder input modality.
    """

    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_muse_kp_distillation"
    empirical_normalization = True
    freeze_normalizer_on_resume = True

    policy = RslRlMUSEKpDistillationCfg(
        class_name="LatentBottleneckMUSEKp",
        init_noise_std=1.0,
        proprio_dim=450,  # placeholder; encoder ignores
        history_length=5,
        proprio_term_sizes=[29, 29, 3, 29],
        # KP encoder (KP6 recipe, 2026-05-16): 6 tokens = KP6_NATIVE_BODIES
        # (pelvis + torso + L/R wrist + L/R ankle). 5-mode OOD-avoidance mask spec
        # (bernoulli + 4 single-point deploy modes) — all subsets of these 6 bodies, so
        # kp_n_bodies is unchanged. MUST equal len(commands.motion.body_names) in
        # G1MUSEKpDistillationTrackingEnvCfg.
        # 0.5 s log-spaced layout (KP_LAYOUT_0_5S, 12 slots, future to ~0.5 s); per-body dim
        # = 12 × 3 = 36. Shifted from the 1 s layout (2026-05-17) — the 0.5 s
        # ``muse_kp_aux_probe_log05s`` setting tracked markedly better. JC warmstart carries
        # no KP front-end, so kp_proj/body_id_emb learn fresh (slot remap is a no-op — no KP ckpt).
        # MUST equal len(commands.motion.body_names) / KP obs slot_offsets in
        # G1MUSEKpDistillationTrackingEnvCfg (both now KP_LAYOUT_0_5S).
        kp_n_bodies=6,
        kp_lookahead_steps=12,
        kp_layout="log_0_5s",
        # Aux predictor: built on the policy when ``aux_predictor_enabled=True`` (algorithm's
        # ``aux_lambda`` + ``aux_stop_grad`` flags weight/gate it). Default False so the
        # unsupervised MUSE-Kp baseline path stays unaffected.
        aux_predictor_enabled=False,
        # Transformer body — must match G1FlatMUSETransformerDistillationRunnerCfg.policy.
        latent_dim=16,
        d_model=192,
        nhead=4,
        num_layers=2,
        ffn_dim=768,
        decoder_hidden_dims=[1024, 512, 256, 128],
        teacher_hidden_dims=[1024, 1024, 512, 512, 256, 256],
        activation="gelu",
        latent_predict_std_min=0.001,
        latent_predict_std_max=10.0,
        fixed_encoder_std=1.0,
        encoder_dropout=0.0,
        decoder_one_step_proprio=False,
        latent_normalize=True,
        deterministic_encoder=True,
        freeze_mode="none",
    )
    algorithm = RslRlMUSEKpAlgorithmCfg(
        smoothness_type="cosine",
        weight_regularization=0.01,
        use_adaptive_regularization=False,
        # Warmup phase: freeze the warmstarted decoder for the first 200 iters so the new KP-input
        # layers (kp_proj on 36-dim per body, body_id_emb) can catch up; then unfreeze
        # (post-warmup freeze_mode = "none"). Mask curriculum's Phase-1 (full visibility) runs
        # 0..1000, so the decoder unfreezes well inside the bootstrap phase.
        warmup_freeze_iters=200,
        warmup_freeze_mode="decoder_only",
        post_warmup_freeze_mode="none",
        # Aux predictor disabled by default (policy.aux_predictor_enabled=False above).
        aux_lambda=0.0,
    )


@configclass
class G1FlatMUSEKpLatentDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """MUSE-Kp **latent-space** distillation runner: KP encoder distilled against a frozen JC
    MUSE-Transformer teacher with the shared decoder frozen.

    The KP encoder backbone (d_model / nhead / num_layers / ffn_dim / latent_dim /
    decoder_hidden_dims) MUST mirror :class:`G1FlatMUSETransformerDistillationRunnerCfg.policy`
    so the JC MUSE-T checkpoint passed via ``--teacher_checkpoint`` (a single ckpt) loads
    cleanly into: the frozen ``jc_encoder`` (teacher latent μ_jc), the frozen shared ``decoder``,
    the warmstarted KP backbone, the frozen ``teacher_obs_normalizer``, and the frozen student
    proprio-normalizer slice (copied at the JC-goal offset). KP front-end (``kp_proj``,
    ``body_id_emb``) learns fresh.

    Lockstep with the env's KP obs (kp_n_bodies / kp_lookahead_steps / kp_layout) is identical to
    :class:`G1FlatMUSEKpDistillationRunnerCfg`. ``latent_normalize=True`` + ``fixed_encoder_std=0.3``
    match the JC MUSE-T checkpoint's training cfg exactly (2026-05-18: JC teacher switched to
    the unit-norm cosine recipe — the frozen decoder now consumes the L2-normalized z).
    """

    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_muse_kp_latent_distillation"
    empirical_normalization = True
    freeze_normalizer_on_resume = True

    policy = RslRlMUSEKpLatentDistillationCfg(
        class_name="LatentBottleneckMUSEKpLatent",
        init_noise_std=1.0,
        proprio_dim=450,
        history_length=5,
        proprio_term_sizes=[29, 29, 3, 29],
        # KP encoder — KP5 (torso + L/R wrist + L/R ankle, NO pelvis), 0.5 s log-spaced.
        # Lockstep: kp_n_bodies MUST equal len(commands.motion.body_names)=len(COTRAIN_KP5_BODIES)=5
        # in G1MUSEKpLatentDistillationTrackingEnvCfg. Base reset stays correct without a pelvis
        # keypoint (commands._resolve_root_full_index → dedicated full-axis root channel).
        kp_n_bodies=5,
        kp_lookahead_steps=15,
        kp_layout="sym_sparse_0p5s",
        aux_predictor_enabled=False,
        # Backbone — MUST match the JC MUSE-T ckpt (2026-05-18 muse_transformer_det_w0.0001).
        latent_dim=16,
        d_model=192,
        nhead=4,
        num_layers=2,
        ffn_dim=768,
        decoder_hidden_dims=[1024, 512, 256, 128],
        teacher_hidden_dims=[1024, 1024, 512, 512, 256, 256],
        activation="gelu",
        latent_predict_std_min=0.001,
        latent_predict_std_max=1.0,
        # JC MUSE-T was trained with fixed_encoder_std=0.3; deterministic_encoder bypasses
        # sampling so this only sets the diagnostic log-σ constant, but kept matched for clarity.
        fixed_encoder_std=0.3,
        encoder_dropout=0.0,
        decoder_one_step_proprio=False,
        # Latent UNIT-NORMALIZED (JC MUSE-T cosine recipe) — frozen decoder consumes z=norm(μ).
        latent_normalize=True,
        deterministic_encoder=True,
        # Decoder always frozen; algorithm's freeze-warmup tightens to
        # decoder_plus_shared_encoder for the first N iters then relaxes back here.
        freeze_mode="decoder_only",
        # JC teacher front-end schema (standalone MUSE-T obs): delta_command(58)+ori(6)+
        # anchor_pos(3)+base_lin_vel(3)+ref_base_lin_vel(3); proprio (29,29,3,29) H5; mask(1).
        jc_goal_term_sizes=[58, 6, 3, 3, 3],
        jc_proprio_term_sizes=[29, 29, 3, 29],
        jc_mask_term_size=1,
    )
    algorithm = RslRlMUSEKpLatentDistillationAlgorithmCfg(
        class_name="MuseKpLatentDistillation",
        num_learning_epochs=5,
        gradient_length=15,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        loss_type="mse",
        latent_loss_type="cosine",
        weight_latent=1.0,
        weight_behavior=0.05,
        # JC teacher pilots the env for the first 200 iters (on-distribution states while the
        # fresh KP front-end catches up); aligned with the freeze warmup so the front-end learns
        # under the clean teacher-piloted distribution before the warmstarted backbone moves.
        teacher_pilot_warmup_iters=0,
        warmup_freeze_iters=0,
        warmup_freeze_mode="decoder_plus_shared_encoder",
        post_warmup_freeze_mode="decoder_only",
    )


@configclass
class G1FlatMUSEKp5FromScratchRunnerCfg(G1FlatMUSEKpDistillationRunnerCfg):
    """5-body-native, 1s-lookahead, from-scratch (no MUSE-Kp ckpt) + passive→active pilot anneal.

    The headline experiment to break the ~85%% sparse-5-body ceiling. Differences vs the
    base MUSE-Kp runner:

    - **Encoder**: ``kp_n_bodies=6`` (the 5 kp5 demo bodies + pelvis — no 14-body tokens;
      pelvis added so the reset can write the robot root-link reference and is a maskable
      root-pose token), ``kp_layout="log_1_0s"`` (13 slots, future to +1.0 s),
      ``kp_lookahead_steps=13``.
    - **From scratch for KP**: warmstart backbone + decoder + frozen teacher from the JC
      MUSE-Transformer ckpt (pass via ``--encoder_decoder_warmstart``); ``kp_proj`` /
      ``body_id_emb`` learn fresh (slot-aware remap is a no-op — no KP ckpt). Freeze-warmup
      OFF (``warmup_freeze_iters=0``) — nothing warmstarted to protect.
    - **No aux predictor** (found useless/harmful): ``aux_predictor_enabled=False``,
      ``aux_lambda=0.0``.
    - **Pilot anneal ON**: teacher pilots 100%% through M1 (iters 0..1500, mask ramp), then
      anneals teacher→student (convex, power=2.0) to a 0.08 floor by iter 4000, per-env
      per-rollout. Supervision target stays the teacher throughout.
    - **Curriculum**: G1MUSEKp5FromScratchTrackingEnvCfg's 2-phase 5-body curriculum
      (M1 bernoulli p_keep 1.0→0.4 until 1500; M2 mode-mix at p_keep=0.4). The pilot
      ``start_iter`` is aligned to the M1→M2 boundary (1500) so mask reaches final
      difficulty before the pilot starts shifting.

    Eval is student-pilot at the M2 mask from iter 0 (alg.act() pilot mix is training-only;
    the runner's eval path calls the policy directly).
    """

    experiment_name = "g1_flat_muse_kp5_fromscratch_pilot"

    policy = RslRlMUSEKpDistillationCfg(
        class_name="LatentBottleneckMUSEKp",
        init_noise_std=1.0,
        proprio_dim=450,
        history_length=5,
        proprio_term_sizes=[29, 29, 3, 29],
        # 6 = KP5 demo bodies + pelvis (KP6_NATIVE_BODIES). Pelvis added so the motion-command
        # reset can write the robot root-link reference (was spawning base at torso → teacher
        # ~0.80). Must equal len(commands.motion.body_names) in G1MUSEKp5FromScratchTrackingEnvCfg.
        kp_n_bodies=6,
        kp_lookahead_steps=13,
        kp_layout="log_1_0s",
        aux_predictor_enabled=False,
        latent_dim=16,
        d_model=192,
        nhead=4,
        num_layers=2,
        ffn_dim=768,
        decoder_hidden_dims=[1024, 512, 256, 128],
        teacher_hidden_dims=[1024, 1024, 512, 512, 256, 256],
        activation="gelu",
        latent_predict_std_min=0.001,
        latent_predict_std_max=10.0,
        fixed_encoder_std=1.0,
        encoder_dropout=0.0,
        decoder_one_step_proprio=False,
        latent_normalize=True,
        deterministic_encoder=True,
        freeze_mode="none",
    )
    algorithm = RslRlMUSEKpAlgorithmCfg(
        smoothness_type="cosine",
        weight_regularization=0.01,
        use_adaptive_regularization=False,
        # Decoder schedule OFF — from scratch, nothing warmstarted to protect.
        warmup_freeze_iters=0,
        # Aux predictor OFF (found useless/harmful).
        aux_lambda=0.0,
        # Passive→active pilot anneal. start_iter aligned to the curriculum's M1→M2 (1500).
        pilot_anneal_enabled=True,
        pilot_teacher_start_iter=1500,
        pilot_teacher_end_iter=4000,
        pilot_teacher_floor=0.08,
        pilot_anneal_power=2.0,
    )


@configclass
class G1FlatMUSEKpLatentRLRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Latent-space RL finetune of a distilled MUSE-Kp (KP6, 0.5 s log-spaced) policy.

    PPO over the MUSE-Kp **latent** (frozen decoder = motor prior). The policy
    wraps a :class:`LatentBottleneckMUSEKp`; every encoder/decoder shape field
    MUST match the distilled checkpoint warmstarted via ``--encoder_decoder_warmstart``
    (the ``muse_kp6_oodmix_0p5s_3phase`` run). Lockstep with the env's KP obs
    layout (kp_n_bodies / kp_lookahead_steps / kp_layout) is identical to
    :class:`G1FlatMUSEKpDistillationRunnerCfg`.

    Reward = world-frame visible-POI accuracy (decision D5) via
    :class:`G1MUSEKpLatentRLTrackingEnvCfg`. M1: ``adapter="full_ft"``,
    unanchored; ~200-iter critic warmup; latent Gaussian std 0.1 (safe start).
    """

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "g1_flat_muse_kp_latent_rl"
    empirical_normalization = True
    freeze_normalizer_on_resume = True

    policy = RslRlLatentRLActorCriticCfg(
        class_name="LatentRLActorCritic",
        init_noise_std=1.0,  # inner-muse unused action std; ignored by the latent policy
        proprio_dim=450,
        history_length=5,
        proprio_term_sizes=[29, 29, 3, 29],
        # KP encoder shape — MUST match the warmstart ckpt + env obs (KP6, 0.5 s).
        kp_n_bodies=5,
        kp_lookahead_steps=12,
        kp_layout="log_0_5s",
        aux_predictor_enabled=False,
        latent_dim=16,
        d_model=192,
        nhead=4,
        num_layers=2,
        ffn_dim=768,
        decoder_hidden_dims=[1024, 512, 256, 128],
        teacher_hidden_dims=[1024, 1024, 512, 512, 256, 256],
        activation="gelu",
        latent_predict_std_min=0.001,
        latent_predict_std_max=10.0,
        fixed_encoder_std=1.0,
        encoder_dropout=0.0,
        decoder_one_step_proprio=False,
        latent_normalize=True,
        deterministic_encoder=True,
        # --- latent-RL knobs ---
        adapter="full_ft",
        init_latent_std=0.1,
        num_teacher_obs=815,
        critic_hidden_dims=[1024, 512, 256, 128],
        critic_activation="elu",
    )
    algorithm = RslRlLatentPPOAlgorithmCfg(
        class_name="LatentPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # No teacher-critic init; warm the fresh critic before actor updates.
        critic_warmup_itrs=0, #100
    )


@configclass
class G1FlatMUSEKpLatentRLKp5RunnerCfg(RslRlOnPolicyRunnerCfg):
    """KP5 latent-space RL finetune runner (canonical 5-point recipe; KP6 is legacy).

    Identical to :class:`G1FlatMUSEKpLatentRLRunnerCfg` EXCEPT the KP encoder is re-pointed
    to the KP5 arch — ``kp_n_bodies=5``, ``kp_lookahead_steps=15``,
    ``kp_layout="sym_sparse_0p5s"``, ``num_teacher_obs=524`` — matching the kp5_latent_distill
    checkpoint warmstarted via ``--encoder_decoder_warmstart`` (the exact arch of
    :class:`G1FlatMUSEKpLatentDistillationRunnerCfg.policy`). The backbone (latent_dim /
    d_model / nhead / num_layers / ffn_dim / decoder_hidden / teacher_hidden) is identical to
    both KP5 distill and KP6 latent-RL. Pairs with
    :class:`G1MUSEKpLatentRLKp5TrackingEnvCfg` / ``...Kp5WritingTrackingEnvCfg``.
    """

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "g1_flat_muse_kp_latent_rl"
    empirical_normalization = True
    freeze_normalizer_on_resume = True

    policy = RslRlLatentRLActorCriticCfg(
        class_name="LatentRLActorCritic",
        init_noise_std=1.0,
        proprio_dim=450,
        history_length=5,
        proprio_term_sizes=[29, 29, 3, 29],
        # KP encoder shape — KP5 (torso + L/R wrist + L/R ankle, no pelvis), sym_sparse 0.5 s.
        # MUST match the kp5_latent_distill warmstart ckpt + the KP5 env obs (15 slots).
        kp_n_bodies=5,
        kp_lookahead_steps=15,
        kp_layout="sym_sparse_0p5s",
        aux_predictor_enabled=False,
        latent_dim=16,
        d_model=192,
        nhead=4,
        num_layers=2,
        ffn_dim=768,
        decoder_hidden_dims=[1024, 512, 256, 128],
        teacher_hidden_dims=[1024, 1024, 512, 512, 256, 256],
        activation="gelu",
        latent_predict_std_min=0.001,
        latent_predict_std_max=1.0,  # diagnostic bound (deterministic_encoder bypasses sampling)
        fixed_encoder_std=0.3,       # matches the KP5 distill cfg (diagnostic-only constant)
        encoder_dropout=0.0,
        decoder_one_step_proprio=False,
        latent_normalize=True,
        deterministic_encoder=True,
        # --- latent-RL knobs (identical to the KP6 runner) ---
        adapter="residual",
        init_latent_std=0.1,
        # KP5 JC MUSE-T teacher obs dim = goal(73) + proprio(90×5=450) + mask(1) = 524.
        num_teacher_obs=524,
        critic_hidden_dims=[1024, 512, 256, 128],
        critic_activation="elu",
    )
    algorithm = RslRlLatentPPOAlgorithmCfg(
        class_name="LatentPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        critic_warmup_itrs=100, # 100
    )


@configclass
class G1FlatMUSEKpLatentRLKp5ObstacleReachRunnerCfg(G1FlatMUSEKpLatentRLKp5RunnerCfg):
    """Obstacle-reach latent-RL runner: identical to the KP5 latent-RL runner, but the
    residual policy AND critic also see the obstacle. The env appends MAX_OBSTACLES(5)×7 = 35
    obstacle dims LAST to the policy obs; the actor-critic strips them (``obstacle_feat_dim``)
    and feeds the residual corrector ``obstacle_n=5`` per-box tokens (option C). Pairs with
    :class:`G1MUSEKpLatentRLKp5ObstacleReachTrackingEnvCfg`."""

    experiment_name = "g1_flat_muse_kp_latent_rl_obstacle"
    save_interval = 200

    def __post_init__(self):
        super().__post_init__()
        # 5 boxes × (center_b 3 + half 3 + valid 1) = 35; mirrors obstacle_reach.MAX_OBSTACLES
        # and the env's obstacle obs term (mdp.obstacle_params_robot_anchor_b).
        self.policy.obstacle_feat_dim = 35
        self.policy.obstacle_n = 5


@configclass
class G1FlatAdvPULSEDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PULSE distillation with diagnostic action discriminator (encoder vs prior decode paths)."""

    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_adv_pulse_distillation"
    empirical_normalization = True
    freeze_normalizer_on_resume = True
    prior_eval_enabled = True
    prior_eval_rollout_every_itr = 1000
    prior_eval_rollout_itr = 10
    prior_eval_rollout_steps = 400
    prior_eval_log_interval = 50
    prior_eval_fixed_latent_std = None
    prior_eval_fall_term_name = "fall"
    prior_eval_fall_body_names = ["torso_link"]
    prior_eval_fall_min_height = 0.35
    prior_eval_deterministic_motion = True
    prior_eval_motion_sample_seed = 42

    policy = RslRlPULSEAdvDistillationCfg(
        class_name="LatentBottleneckPULSEAdv",
        init_noise_std=1.0,
        proprio_dim=450,
        latent_dim=24,
        encoder_hidden_dims=[512, 256, 128],
        decoder_hidden_dims=[512, 256, 128],
        decoder_use_proprio=False, # important
        prior_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[1024, 1024, 512, 512, 256, 256],
        activation="elu",
        initialize_std=0.1,
        latent_predict_std_min=0.001,
        latent_predict_std_max=1.0,
        fixed_prior_std=0.5,
        fixed_encoder_std=0.5 / 1.414,
        disc_hidden_dims=[512, 256, 128],
    )
    algorithm = RslRlAdvPulseAlgorithmCfg(
        weight_kl=0.01,
        weight_kl_end=0.01,
        kl_anneal_start_iter=4000,
        kl_anneal_end_iter=8000,
        weight_regularization=0.005,
        kl_loss_upper_bound=1e6,
        disc_lr=1e-3,
        disc_update_interval=1,
    )

@configclass
class G1FlatPriorOnlyPULSEDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Online prior-only PULSE: same env/rollouts as full PULSE distillation, but only the prior R(s^p)
    is trained (KL toward detached encoder latents). Initialize from a trained PULSE ``.pt`` via
    ``--resume_student_checkpoint``; keep ``--teacher_checkpoint`` for the privileged teacher (PHC+).
    """

    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 300
    experiment_name = "g1_flat_prior_only_pulse"
    empirical_normalization = True
    # Same rationale as full PULSE: avoid normalizer drift when resuming.
    freeze_normalizer_on_resume = True

    policy = RslRlPULSEDistillationCfg(
        class_name="LatentBottleneckPULSE",
        init_noise_std=1.0,
        proprio_dim=450,
        latent_dim=16,
        encoder_hidden_dims=[512, 256, 128],
        decoder_hidden_dims=[512, 256, 128],
        prior_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        latent_predict_std_min=0.01,
        latent_predict_std_max=1.0,
    )
    algorithm = RslRlPriorOnlyPULSEAlgorithmCfg(
        kl_loss_upper_bound=10.0,
    )


@configclass
class G1FlatAnyBodyLatentDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Residual-latent AnyBody latent distillation from pretrained PULSE.
    Student learns delta latent over frozen prior mean; prior/decoder are frozen.
    """

    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 100
    experiment_name = "g1_flat_vae_latent_distillation_2b"
    empirical_normalization = True

    # Holdout motion set: periodic student rollout + MultiMotionCommand metrics (no reward logging).
    # Each holdout pass uses ``num_steps_per_env`` steps (same inner loop as training).
    holdout_eval_enabled: bool = False
    holdout_eval_motion: str | None = None
    holdout_eval_every_itr: int = 1000
    holdout_eval_rollout_itr: int = 1
    holdout_eval_deterministic_motion: bool = True
    holdout_eval_motion_sample_seed: int = 42

    policy = RslRlAnyBodyLatentDistillationCfg(
        class_name="LatentBottleneckAnyBody",
        init_noise_std=1.0,
        proprio_dim=90,  # Match pretrained PULSE proprio split.
        latent_dim=16,     # Match pretrained PULSE latent dim.
        encoder_hidden_dims=[512, 256, 128],
        teacher_encoder_hidden_dims=[256, 128],
        decoder_hidden_dims=[256, 128],
        prior_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = RslRlAnyBodyLatentDistillationAlgorithmCfg(
        class_name="AnyBodyLatentDistillation",
        num_learning_epochs=5,
        gradient_length=15,
        learning_rate=1.0e-3,
        loss_type="mse",
        weight_latent=1.0,
        # Keep a small direct action anchor so latent matching does not drift
        # into action-irrelevant directions under a frozen decoder.
        weight_behavior=0.1,
    )


@configclass
class G1FlatPartialMaskedAnyBodyLatentDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Residual-latent AnyBody latent distillation with optional partial keypoint masking.

    Student learns delta latent over frozen prior mean; prior/decoder are frozen.
    Keypoints in the student observation can be masked (zeroed) according to a
    mode schedule (e.g., full body vs upper body only). Masking is applied after
    normalization so masked zeros do not affect normalizer statistics.
    """

    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_vae_latent_distillation_2b_partial_mask"
    empirical_normalization = True

    # Holdout motion set: periodic student rollout + MultiMotionCommand metrics (no reward logging).
    # Each holdout pass uses ``num_steps_per_env`` steps (same inner loop as training).
    holdout_eval_enabled: bool = True
    holdout_eval_motion: str | None = "/home/lsn/Datasets/SONIC_npzs/g1/npz_splits_filtered_medium/test"
    holdout_eval_every_itr: int = 0 # disable holdout eval
    holdout_eval_rollout_itr: int = 10
    holdout_eval_deterministic_motion: bool = False #True
    holdout_eval_motion_sample_seed: int = 42

    policy = RslRlAnyBodyLatentDistillationCfg(
        class_name="LatentBottleneckAnyBody",
        init_noise_std=1.0,
        proprio_dim=450,  # Match pretrained PULSE proprio split.
        latent_dim=16,     # Match pretrained PULSE latent dim.
        encoder_hidden_dims=[512, 256, 128],
        teacher_encoder_hidden_dims=[512, 256, 128],
        decoder_hidden_dims=[512, 256, 128],
        prior_hidden_dims=[512, 256, 128],
        activation="elu",
        # Match ``LatentBottleneckPULSE`` teacher (``fixed_*`` => no σ heads in ``model_*.pt``).
        latent_predict_std_min=0.001,
        latent_predict_std_max=1.0,
        fixed_prior_std=0.5,
        fixed_encoder_std=0.3536067892503536,
    )
    algorithm = RslRlAnyBodyLatentDistillationAlgorithmCfg(
        class_name="AnyBodyLatentDistillation",
        num_learning_epochs=5,
        gradient_length=15,
        learning_rate=1.0e-3,
        loss_type="mse",
        weight_latent=1.0,
        weight_behavior=0.1,
    )
