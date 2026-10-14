import os

from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from rsl_rl.modules.latent_bottleneck_muse_kp import (
    KP_LAYOUT_0_5S, # legacy
    KP_LAYOUT_SYM_SPARSE_0P5S,
    kp_layout_by_name,
)
from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.mdp import teacher_point_contract as teacher_points
from whole_body_tracking.tasks.tracking.config.g1 import mask_modes
from whole_body_tracking.tasks.tracking.tracking_env_cfg import (
    TrackingEnvCfg,
    GeneralTrackingEnvCfg,
    ExpertGeneralTrackingEnvCfg,
    DistillationTrackingEnvCfg,
    OneStageTrackingEnvCfg,
    VAEDistillationTrackingEnvCfg,
    MUSEDistillationTrackingEnvCfg,
    MUSETransformerDistillationTrackingEnvCfg,
    MUSEKpDistillationTrackingEnvCfg,
    MUSEKp5FromScratchTrackingEnvCfg,
    PULSEDistillationTrackingEnvCfg,
    PartialMaskedVaeDistillationCurriculumCfg,
    PartialMaskedMultiMotionCommandsCfg,
    CurriculumCfg,
    MUSEKpDistillationCurriculumCfg,
    MUSEKpLatentDemoCurriculumCfg,
)


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE


@configclass
class G1FlatGeneralEnvCfg(GeneralTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FlatWoStateEstimationGeneralEnvCfg(G1FlatGeneralEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqGeneralEnvCfg(G1FlatGeneralEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE

@configclass
class G1FlatExpertGeneralEnvCfg(ExpertGeneralTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.observations.policy.motion_anchor_pos_b = None
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

@configclass
class G1DistillationTrackingEnvCfg(DistillationTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.teacher.motion_anchor_pos_b = None
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FLDDistillationTrackingEnvCfg(DistillationTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.teacher.motion_anchor_pos_b = None
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1OneStageTrackingEnvCfg(OneStageTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        # NOTE 2026-05-13: previously this nulled out ``motion_anchor_pos_b`` to match the
        # original "Wo-State-Estimation" teacher design (770-dim obs). We now want the policy
        # to have privileged anchor-pos info — keep the parent's ``motion_anchor_pos_b`` term
        # (785-dim obs). ``base_lin_vel`` was never present on this PolicyCfg, so no-op kept
        # only for ``motion_anchor_pos_b``'s sake — drop both.
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1OneStageWristRewardV1EnvCfg(G1OneStageTrackingEnvCfg):
    """Teacher initialization fixes plus explicit upper6/lower8 wrist coverage.

    All observations, nominal reward weights/stds, PPO and tracked body axes
    are inherited. Only the extension position term selects the new profile.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        teacher_points.validate_g1_partition(self.commands.motion.body_names)
        self.rewards.teleop_body_position_extend.func = mdp.teleop_body_position_extend_g1_v1


@configclass
class G1OneStagePointRewardsV1EnvCfg(G1OneStageWristRewardV1EnvCfg):
    """Wrist profile plus declared virtual head/wrists three-point reward.

    The head is a nominal asset-defined surrogate, not an exact reconstruction
    of the authors' undisclosed VR target. Definition and offset are serialized.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.teleop_vr_3point.func = mdp.teleop_vr_3point_g1_v1
        self.rewards.teleop_vr_3point.params = {
            **self.rewards.teleop_vr_3point.params,
            "head_offset_in_torso": teacher_points.g1_head_offset_from_urdf(),
            "head_point_definition": teacher_points.G1_HEAD_POINT_V1,
        }


@configclass
class G1VAEDistillationTrackingEnvCfg(VAEDistillationTrackingEnvCfg):
    """G1-specific VAE distillation env config."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1PULSEDistillationTrackingEnvCfg(PULSEDistillationTrackingEnvCfg):
    """G1-specific PULSE distillation env config."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1MUSEDistillationTrackingEnvCfg(MUSEDistillationTrackingEnvCfg):
    """G1-specific MUSE distillation env config."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1MUSETransformerDistillationTrackingEnvCfg(MUSETransformerDistillationTrackingEnvCfg):
    """G1-specific transformer-MUSE env config."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1MUSEKpDistillationTrackingEnvCfg(MUSEKpDistillationTrackingEnvCfg):
    """G1 MUSE-Kp env (simplified KP6 recipe, 2026-05-16): KP-token student warmstarted from a
    MUSE-Transformer (JC) ckpt.

    Simplified from the old dense-14-body / 7-mode / 4-phase recipe to:
      - **6 KP points** (:data:`mask_modes.KP6_NATIVE_BODIES` = pelvis + torso + L/R wrist +
        L/R ankle; pelvis FIRST so the motion-command reset writes the robot root-link
        reference — see :func:`mask_modes._resolve_root_body_index`).
      - **5-mode OOD-avoidance spec** — ``bernoulli`` (over all 6) + the 4 single-point
        deploy modes L/R wrist, torso, pelvis (:func:`mask_modes.muse_kp6_ood_mix_mode_spec`);
        ``mask_mode_probs`` is length-5. Phases 1-3 stay bernoulli-only; phase 4 mixes the
        single-point modes so the single-point-visible drag deploy is in-distribution.
      - **0.5 s lookahead** — KP obs use ``KP_LAYOUT_SYM_SPARSE_0P5S`` (12 slots, future to ~0.5 s),
        matching the base ``MUSEKpObservationsCfg`` (override kept explicit). Shifted from
        the 1 s layout (2026-05-17): the 0.5 s ``muse_kp_aux_probe_log05s`` setting tracked
        markedly better. ``kp_lookahead_steps``/``kp_layout`` in
        :class:`G1FlatMUSEKpDistillationRunnerCfg` are kept in lockstep (also 0.5 s).
      - **4-phase curriculum** (:class:`MUSEKpDistillationCurriculumCfg`): iter 0..1000
        p_see=1.0 (all 6 visible), 1000..2000 ramp 1.0→0.4, 2000..5000 hold 0.4
        (all bernoulli-only), 5000..end → 5-way mask-mode sampling (0.2 each). With no
        student resume (JC encoder-decoder warmstart only) a fresh run executes all 4
        phases in order — phases 1-3 are the bernoulli warmup over iter 0..5000.

    JC warmstart unchanged (backbone+decoder+frozen teacher from the MUSE-Transformer ckpt;
    KP front-end ``kp_proj``/``body_id_emb`` learn fresh). The aux task
    (``MUSE-Kp-Aux-Distill-...``) has its own env/curriculum/runner and is unaffected.
    """

    @configclass
    class MUSEKp6Log0p5sObservationsCfg(MUSEKpDistillationTrackingEnvCfg.MUSEKpObservationsCfg):
        @configclass
        class PolicyCfg(MUSEKpDistillationTrackingEnvCfg.MUSEKpObservationsCfg.PolicyCfg):
            # KP terms explicit at the 0.5 s log-spaced layout (== base MUSEKpObservationsCfg);
            # proprio inherits. Kept in lockstep with rsl_rl_ppo_cfg kp_layout="log_0_5s".
            kp_lookahead = ObsTerm(
                func=mdp.ref_body_pos_robot_anchor_b_logspaced,
                params={"command_name": "motion", "slot_offsets": KP_LAYOUT_SYM_SPARSE_0P5S},
            )
            kp_mask_lookahead = ObsTerm(
                func=mdp.partial_kp_mask_logspaced,
                params={"command_name": "motion", "num_slots": len(KP_LAYOUT_SYM_SPARSE_0P5S)},
            )

        policy: PolicyCfg = PolicyCfg()
        teacher: MUSEKpDistillationTrackingEnvCfg.MUSEKpObservationsCfg.TeacherCfg = (
            MUSEKpDistillationTrackingEnvCfg.MUSEKpObservationsCfg.TeacherCfg()
        )

    observations: MUSEKp6Log0p5sObservationsCfg = MUSEKp6Log0p5sObservationsCfg()

    def __post_init__(self):
        # Parent ``__post_init__`` overwrites ``self.commands`` with a fresh
        # ``PartialMaskedMultiMotionCommandsCfg``. Run super first, then re-apply G1-specific config.
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = list(mask_modes.KP6_NATIVE_BODIES)
        # CANONICAL unified 6-mode spec (single source of truth — every MUSE-kp env uses this).
        self.commands.motion.mask_mode_spec = mask_modes.muse_kp6_unified_mode_spec()
        # Initial probs = the final mask-mode mix (single-source constant). The
        # keypoint_mask_mode curriculum overwrites at runtime — this only sets the very first
        # env step's distribution (fresh runs spend P1/P2 in bernoulli-only via the curriculum).
        self.commands.motion.mask_mode_probs = mask_modes.MUSE_KP6_UNIFIED_MIX_PROBS


@configclass
class G1MUSEKpLatentDistillationTrackingEnvCfg(G1MUSEKpDistillationTrackingEnvCfg):
    """G1 MUSE-Kp **latent-space** distillation env.

    Same KP6 0.5 s student obs + 5-mode OOD-mix + 4-phase curriculum + PartialMasked commands as
    :class:`G1MUSEKpDistillationTrackingEnvCfg` (inherited via ``__post_init__``), with two
    deliberate differences for latent distillation against a frozen JC MUSE-Transformer teacher:

      1. **Teacher group = the JC MUSE-Transformer obs** (not the PHC+ MLP contract): the exact
         schema the JC MUSE-T was trained on — ``delta_command_real`` (58) + motion_anchor_ori_b
         (6) + motion_anchor_pos_b (3) + base_lin_vel (3) + ref_base_lin_vel (3) + proprio-H5 +
         goal_mask_history (1) = 524 dims. The JC goal-mask stays 0 (no goal-mask curriculum
         here — only the KP keypoint mask is curriculum-driven), so the JC teacher is the
         privileged, always-unmasked latent/action source.
      2. **No proprio noise on the student policy obs**: the JC decoder is FROZEN and was trained
         on noise-free proprio; injecting Unoise on the student's proprio would feed the frozen
         decoder out-of-distribution proprio it cannot adapt to. KP keypoint masking (the actual
         partial-observability signal) is unchanged.
    """

    @configclass
    class MUSEKpLatentObservationsCfg(
        G1MUSEKpDistillationTrackingEnvCfg.MUSEKp6Log0p5sObservationsCfg
    ):
        @configclass
        class PolicyCfg(
            G1MUSEKpDistillationTrackingEnvCfg.MUSEKp6Log0p5sObservationsCfg.PolicyCfg
        ):
            # Proprio noise OFF — match the frozen JC decoder's noise-free training distribution.
            # kp_lookahead / kp_mask_lookahead (the partial-obs signal) are inherited unchanged.
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=5)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=5)
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=5)
            actions = ObsTerm(func=mdp.last_action, history_length=5)

        policy: PolicyCfg = PolicyCfg()
        # JC MUSE-Transformer obs as the privileged (always-unmasked) teacher group.
        teacher: MUSETransformerDistillationTrackingEnvCfg.MUSETransformerObservationsCfg.PolicyCfg = (
            MUSETransformerDistillationTrackingEnvCfg.MUSETransformerObservationsCfg.PolicyCfg()
        )

    observations: MUSEKpLatentObservationsCfg = MUSEKpLatentObservationsCfg()
    # 8-mode deploy-shaped curriculum (overrides the inherited KP6 MUSEKpDistillationCurriculumCfg,
    # whose length-6 tuples + commented-out mix phase don't apply to the 8-mode spec set below).
    curriculum: MUSEKpLatentDemoCurriculumCfg = MUSEKpLatentDemoCurriculumCfg()

    def __post_init__(self):
        # Parent sets KP6 (KP6_NATIVE_BODIES + muse_kp6_unified_mode_spec). Run it, then
        # re-point to the 5-body set: torso + L/R wrist + L/R ankle, NO pelvis. The base is
        # still reset correctly without a pelvis keypoint — commands._resolve_root_full_index
        # reads a dedicated full-axis root channel, independent of the tracked body set
        # (decouples the prior root-reset bug).
        #
        # 2026-05-22: swapped the 6-mode muse_kp5_unified spec for the 8-mode deploy-shaped
        # demo spec (full / vr / torso / L,R wrist / wrists / ankles / bernoulli). The mode
        # count changes 6 -> 8, so the inherited length-6 probs/curriculum no longer apply:
        # the length-8 mix probs are set here and the 8-mode MUSEKpLatentDemoCurriculumCfg is
        # installed via the class field above. Same 5 bodies, so the warmstarted encoder is
        # unaffected (it is mode-agnostic — visibility lives only in the NaN-masked obs).
        super().__post_init__()
        self.commands.motion.body_names = list(mask_modes.COTRAIN_KP5_BODIES)
        self.commands.motion.mask_mode_spec = mask_modes.muse_kp5_latent_demo_mode_spec()
        # Length-8 initial probs (the final mix); the curriculum overwrites at runtime so a
        # fresh run still spends phases 1-2 bernoulli-only before this 8-way mix in phase 3.
        self.commands.motion.mask_mode_probs = mask_modes.MUSE_KP5_LATENT_DEMO_MIX_PROBS


def _latentrl_ref_weight() -> float:
    """Master strength for the optional weak full-body *reference* bundle in
    :class:`LatentRLRewardsCfg`, read once from ``LATENTRL_REF_W`` (default ``0.0`` =
    OFF → the bundle contributes nothing and the sparse-POI run is reproduced exactly).

    Re-densifies toward the distillation reward at LOW weight so the UNWATCHED bodies
    (legs/torso) stay anchored to the graceful reference clip instead of being contorted
    to squeeze the last bit of POI accuracy (the 2026-05-23 "stumbles backwards while
    tracking the wrist" finding). Scales a fixed base profile (RewardsExpertCfg ratios:
    posture pos/ori @1.0, gait lin/ang vel @1.5). Sweep e.g. 0.05..0.2.

    Applies ONLY to this locomotion latent-RL reward; the wrist-writing reward
    (:class:`LatentRLKp5WritingRewardsCfg`) is a separate class and is intentionally
    untouched (the writing env overwrites ``self.rewards`` with it).
    """
    raw = os.environ.get("LATENTRL_REF_W", "0.0")
    try:
        w = float(raw)
    except ValueError as exc:
        raise ValueError(f"LATENTRL_REF_W must be a float, got {raw!r}") from exc
    if w < 0.0:
        raise ValueError(f"LATENTRL_REF_W must be >= 0, got {w}")
    return w


# Read once at import (same convention as the KP-layout env-var helpers). Used as a
# plain module constant — NOT a class attribute — so the reward manager never sees a
# stray non-RewTerm field on LatentRLRewardsCfg.
_LATENTRL_REF_W = _latentrl_ref_weight()


@configclass
class LatentRLRewardsCfg:
    """Dedicated reward set for the latent-RL task (NOT RewardsExpertCfg + edits).

    Decisions D5 / D5a / §4a, finalized 2026-05-17:
      - **POI world position** + **POI world linear velocity** — visible-KP only,
        world frame (the deployment-true objective the BC teacher never had).
      - **light global anchor** pos(0.7)/ori(0.3) — minimal "don't drift / stay
        on the global trajectory" anchor. Keypoint-only reward is globally
        under-determined (root XY free → moonwalk hack); the frozen decoder keeps
        motion natural but does NOT pin the world root. This is NOT a per-visible
        -body reference term, so it does not re-introduce masked-body unfairness;
        invisible-body naturalness is the frozen decoder's + M1b latent-anchor's job.
      - penalties unchanged from RewardsExpertCfg (D5a scaffolding, verbatim).
      - everything else (motion_body_ori, motion_body_*_vel, anchor_lin_vel, all
        teleop_*) is **dropped entirely** — relative/all-body/not-mask-aware.
      - 2026-05-23 grace fix: an OPTIONAL weak full-body reference bundle
        (motion_body_pos/ori + motion_body_lin/ang_vel) can be re-added via the
        ``LATENTRL_REF_W`` env var (default 0.0 = off). NON-mask-aware on purpose:
        the bodies that go ugly under sparse-POI RL are precisely the INVISIBLE ones
        (legs in a track-wrist mode), so this is a low-weight naturalness PRIOR over
        all bodies — not a task term — and it reverses the "drop per-body reference"
        rationale above only as a gentle regularizer. See _latentrl_ref_weight.
    All weights/std are tunable knobs.
    """

    # --- Task: world-frame visible points-of-interest ---
    poi_pos = RewTerm(
        func=mdp.motion_visible_kp_position_error_exp_world,
        weight=3.0,
        params={"command_name": "motion", "std": 0.3}, #0.3
    )
    poi_lin_vel = RewTerm(
        func=mdp.motion_visible_kp_lin_vel_error_exp_world,
        weight=3.0,
        params={"command_name": "motion", "std": 1.0}, #1.0
    )

    # --- Light global anchor (anti-drift / global trajectory) ---
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=1.0, # 0.5
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=1.0, # 0.5
        params={"command_name": "motion", "std": 0.4},
    )

    # --- Optional weak full-body REFERENCE bundle (2026-05-23 grace fix) -------
    # Default OFF (LATENTRL_REF_W=0.0 → weight 0 → no contribution → current run
    # reproduced). Set LATENTRL_REF_W>0 to re-anchor the unwatched bodies to the
    # graceful reference clip. NON-mask-aware (all bodies) by design: the legs that
    # stumble are the invisible ones in a track-wrist mode, so a visible-only term
    # could not fix it. Base profile = RewardsExpertCfg ratios (posture pos/ori @1.0,
    # gait lin/ang vel @1.5). Keep weak so the strong POI term still drives the
    # watched bodies. Sweep e.g. LATENTRL_REF_W=0.05..0.2.
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0 * _LATENTRL_REF_W,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0 * _LATENTRL_REF_W,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.5 * _LATENTRL_REF_W,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.5 * _LATENTRL_REF_W,
        params={"command_name": "motion", "std": 3.14},
    )

    # --- Penalty scaffolding (D5a; verbatim from RewardsExpertCfg) ---
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)


@configclass
class G1MUSEKpLatentRLObservationsCfg(
    G1MUSEKpDistillationTrackingEnvCfg.MUSEKp6Log0p5sObservationsCfg
):
    """Inherits the EXACT KP6/0.5s ``policy`` + ``teacher`` groups (so the
    warmstarted encoder sees identical obs) and adds an asymmetric ``critic``
    group: student task-framing ⊕ teacher privileged ⊕ absolute world block.

    The runner auto-detects a ``critic`` group → uses it for the value function
    (privileged_obs_type="critic"); its dim is passed as num_critic_obs to
    LatentRLActorCritic (no module/runner change). The critic is privileged →
    no corruption, full unmasked reference, absolute world frame to match the
    world-frame POI reward.
    """

    @configclass
    class CriticCfg(ObsGroup):
        # 1. Student task-framing (same KP6/0.5s terms the actor sees — what's
        #    visible, the anchor-frame KP window, proprio history). No noise
        #    (privileged critic wants a clean state).
        kp_lookahead = ObsTerm(
            func=mdp.ref_body_pos_robot_anchor_b_logspaced,
            params={"command_name": "motion", "slot_offsets": KP_LAYOUT_SYM_SPARSE_0P5S},
        )
        kp_mask_lookahead = ObsTerm(
            func=mdp.partial_kp_mask_logspaced,
            params={"command_name": "motion", "num_slots": len(KP_LAYOUT_SYM_SPARSE_0P5S)},
        )
        proprio_joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=5)
        proprio_joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=5)
        proprio_base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=5)
        proprio_actions = ObsTerm(func=mdp.last_action, history_length=5)

        # 2. Teacher-privileged extras the masked student lacks (unmasked full
        #    reference command + true base velocity + anchor pose).
        full_ref_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})

        # 3. Absolute world block (matches the world-frame POI reward's frame):
        #    ref & actual body world pos/lin-vel (ALL bodies, unmasked) + abs root.
        ref_body_pos_w = ObsTerm(func=mdp.ref_body_pos_w_abs, params={"command_name": "motion"})
        robot_body_pos_w = ObsTerm(func=mdp.robot_body_pos_w_abs, params={"command_name": "motion"})
        ref_body_lin_vel_w = ObsTerm(func=mdp.ref_body_lin_vel_w, params={"command_name": "motion"})
        robot_body_lin_vel_w = ObsTerm(func=mdp.robot_body_lin_vel_w, params={"command_name": "motion"})
        anchor_ori_w = ObsTerm(func=mdp.robot_anchor_ori_w, params={"command_name": "motion"})
        anchor_lin_vel_w = ObsTerm(func=mdp.robot_anchor_lin_vel_w_abs, params={"command_name": "motion"})
        anchor_ang_vel_w = ObsTerm(func=mdp.robot_anchor_ang_vel_w_abs, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False  # privileged critic: clean value targets
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class LatentRLCurriculumCfg(CurriculumCfg):
    """RL-finetune mask curriculum: ONLY the final mixed-mode phase, held from
    iter 0 — no bootstrap/ramp.

    RL finetunes an already-mask-trained distilled checkpoint, so
    :class:`MUSEKpDistillationCurriculumCfg`'s phase-1/2 (all-visible -> p_see
    ramp) would *un-train* occlusion robustness early. This is precisely that
    curriculum's FINAL phase — ``mask_modes.MUSE_KP6_UNIFIED_MIX_PROBS`` (the
    bernoulli + single-point semantic modes) at ``p_keep=0.4`` (decision D4,
    deployment-weighted) — as a single phase held indefinitely
    (``phase_until_learning_iterations=(None,)``). ``mode_probs`` / ``p_keep``
    are the tunable knobs for this task.
    """

    keypoint_mask_mode = CurrTerm(
        func=mdp.curriculums.keypoint_mask_mode_curriculum,
        params={
            "command_name": "motion",
            "phase_until_learning_iterations": (None,),
            "mask_phases": (
                {
                    "mode_probs": mask_modes.MUSE_KP6_UNIFIED_MIX_PROBS,
                    "p_start": 0.4,
                    "p_end": 0.4,
                },
            ),
        },
    )


@configclass
class G1MUSEKpLatentRLTrackingEnvCfg(G1MUSEKpDistillationTrackingEnvCfg):
    """Latent-space RL finetune env. Inherits all G1/KP6 structure from the
    distillation env (robot, action scale, KP6 body set, OOD-mix mask spec,
    ankle-only ``ee_body_pos`` termination) but installs **dedicated** reward,
    observation, and curriculum configs for this task (no per-term
    ``__post_init__`` surgery on the inherited RewardsExpertCfg).

    - ``observations``: dedicated cfg adding the asymmetric absolute ``critic``
      group (Q-decision 2026-05-17). Clean class-attr override (parent never
      mutates observations in ``__post_init__``).
    - ``rewards``: :class:`LatentRLRewardsCfg` (Q-decision 2026-05-17). Installed
      in ``__post_init__`` AFTER ``super()`` — the parent's anchor-weight tweaks
      (teacher-reward-comparability, meaningful only for pure-BC distillation)
      are intentionally discarded; for RL the reward IS the objective and
      LatentRLRewardsCfg is the single source of truth. Re-installing a cfg group
      post-super is the established codebase idiom (cf. ``self.commands = ...``).
    - ``curriculum``: :class:`LatentRLCurriculumCfg` — the distillation
      curriculum's FINAL mixed-mode phase only, held from iter 0 (skips the
      all-visible bootstrap/ramp, which would un-train occlusion robustness on
      an already-mask-trained warmstart). Clean class-attr override (nothing in
      the ``__post_init__`` chain mutates ``curriculum``).
    """

    observations: G1MUSEKpLatentRLObservationsCfg = G1MUSEKpLatentRLObservationsCfg()
    curriculum: LatentRLCurriculumCfg = LatentRLCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards = LatentRLRewardsCfg()


@configclass
class G1MUSEKpLatentRLVerifyTrackingEnvCfg(G1MUSEKpLatentRLTrackingEnvCfg):
    """TEMPORARY wiring-verification env (delete after the sanity check).

    Identical to :class:`G1MUSEKpLatentRLTrackingEnvCfg` (latent-RL critic obs,
    KP6, OOD-mix mask + 4-phase curriculum, ankle-only termination) EXCEPT it
    keeps the **original distillation reward** (RewardsExpertCfg + the MUSE-Kp
    anchor-weight overrides) instead of :class:`LatentRLRewardsCfg`.

    Why: with critic-only warmup the actor stays frozen at the warmstarted
    distilled policy, so — under the *original* reward and in curriculum phase-1
    (all KP visible, matching the distillation run @step~1000 = model_1000) — the
    reward scale + success_rate MUST reproduce the distillation run's numbers
    (Train/mean_reward ~36.5, success ~0.98, error_body_pos ~0.045). A large
    mismatch means the latent->decode->env plumbing is silently wrong.

    Skips :meth:`G1MUSEKpLatentRLTrackingEnvCfg.__post_init__`'s
    ``self.rewards = LatentRLRewardsCfg()`` by invoking the distillation chain
    directly (which installs RewardsExpertCfg + anchor overrides + ankle term).
    """

    def __post_init__(self):
        G1MUSEKpDistillationTrackingEnvCfg.__post_init__(self)


# =====================================================================================
# KP5 latent-RL (the canonical 5-point recipe; KP6 is legacy).
#
# The existing G1MUSEKpLatentRLTrackingEnvCfg / G1FlatMUSEKpLatentRLRunnerCfg are KP6
# (kp_n_bodies=6, log_0_5s, 12 slots). The deployed/demoed policy is KP5
# (kp_n_bodies=5, sym_sparse_0p5s, 15 slots) — produced by the LatentDistill task. The
# classes below are the KP5 analogue: they INHERIT the KP5 obs/body/mask config from
# G1MUSEKpLatentDistillationTrackingEnvCfg (so the warmstarted kp5_latent_distill encoder
# sees identical obs) and apply only the latent-RL diff (world-POI rewards + privileged
# critic group + single-phase mask curriculum), mirroring the KP6 latent-RL classes.
# =====================================================================================


@configclass
class G1MUSEKpLatentRLKp5ObservationsCfg(
    G1MUSEKpLatentDistillationTrackingEnvCfg.MUSEKpLatentObservationsCfg
):
    """KP5 latent-RL obs — fully self-contained (no dependency on the legacy KP6 classes).

    Three groups:
      - ``policy``  : inherited from the KP5 latent-distill obs — sym_sparse 0.5 s, 15 slots,
        proprio-noise-off. Identical to what the warmstarted kp5_latent_distill encoder saw.
      - ``teacher`` : inherited — the JC MUSE-Transformer obs (524-d, frozen privileged source).
      - ``critic``  : the asymmetric privileged value-function group, defined explicitly below
        (a verbatim copy of the KP6 latent-RL critic terms, decoupled so KP5 doesn't break
        when the legacy KP6 cfgs are removed).

    Every critic term reads the motion command's TRACKED bodies (KP5 = 5: torso + L/R wrist +
    L/R ankle — all valid in the writing clips) and is wrapped in ``_ensure_finite_obs``
    (NaN/Inf → 0). The critic is privileged (``enable_corruption=False``) and its dim is
    auto-detected by the runner (num_critic_obs); it never touches the deployed actor.
    """

    @configclass
    class CriticCfg(ObsGroup):
        # 1. Student task-framing (same KP5 terms the actor sees; clean, no noise).
        kp_lookahead = ObsTerm(
            func=mdp.ref_body_pos_robot_anchor_b_logspaced,
            params={"command_name": "motion", "slot_offsets": KP_LAYOUT_SYM_SPARSE_0P5S},
        )
        kp_mask_lookahead = ObsTerm(
            func=mdp.partial_kp_mask_logspaced,
            params={"command_name": "motion", "num_slots": len(KP_LAYOUT_SYM_SPARSE_0P5S)},
        )
        proprio_joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=5)
        proprio_joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=5)
        proprio_base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=5)
        proprio_actions = ObsTerm(func=mdp.last_action, history_length=5)

        # 2. Teacher-privileged extras (unmasked full reference + true base velocity + anchor).
        full_ref_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})

        # 3. Absolute world block (matches the world-frame POI reward's frame).
        ref_body_pos_w = ObsTerm(func=mdp.ref_body_pos_w_abs, params={"command_name": "motion"})
        robot_body_pos_w = ObsTerm(func=mdp.robot_body_pos_w_abs, params={"command_name": "motion"})
        ref_body_lin_vel_w = ObsTerm(func=mdp.ref_body_lin_vel_w, params={"command_name": "motion"})
        robot_body_lin_vel_w = ObsTerm(func=mdp.robot_body_lin_vel_w, params={"command_name": "motion"})
        anchor_ori_w = ObsTerm(func=mdp.robot_anchor_ori_w, params={"command_name": "motion"})
        anchor_lin_vel_w = ObsTerm(func=mdp.robot_anchor_lin_vel_w_abs, params={"command_name": "motion"})
        anchor_ang_vel_w = ObsTerm(func=mdp.robot_anchor_ang_vel_w_abs, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False  # privileged critic: clean value targets
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class LatentRLKp5DemoCurriculumCfg(CurriculumCfg):
    """KP5 latent-RL finetune mask curriculum — the 8-mode demo mix held from iter 0.

    KP5 analogue of :class:`LatentRLCurriculumCfg`, but over the 8-mode deploy-shaped spec
    (:func:`mask_modes.muse_kp5_latent_demo_mode_spec`) that the latent-distill pretrain used,
    so the finetune mask distribution matches pretraining (full / vr / torso / L,R wrist /
    wrists / ankles / bernoulli). Single phase, p_see=0.4, no bootstrap/ramp (RL finetunes an
    already-mask-trained checkpoint). Length-8 ``mode_probs`` — paired with the 8-mode spec.
    """

    keypoint_mask_mode = CurrTerm(
        func=mdp.curriculums.keypoint_mask_mode_curriculum,
        params={
            "command_name": "motion",
            "phase_until_learning_iterations": (None,),
            "mask_phases": (
                {
                    "mode_probs": mask_modes.MUSE_KP5_LATENT_DEMO_MIX_PROBS_NO_BERNOULLI, #mask_modes.MUSE_KP5_LATENT_DEMO_MIX_PROBS,
                    "p_start": 0.4,
                    "p_end": 0.4,
                },
            ),
        },
    )


@configclass
class G1MUSEKpLatentRLKp5TrackingEnvCfg(G1MUSEKpLatentDistillationTrackingEnvCfg):
    """KP5 latent-space RL finetune env (canonical 5-point recipe).

    Inherits the KP5 obs/body/8-mode demo mask spec from the distillation env (matching the
    kp5_latent_distill checkpoint warmstarted via ``--encoder_decoder_warmstart``) and
    installs the latent-RL world-frame visible-POI rewards (:class:`LatentRLRewardsCfg`),
    the privileged absolute critic group, and the single-phase mask curriculum
    (:class:`LatentRLKp5DemoCurriculumCfg` — the 8-mode demo mix). KP5 analogue of
    :class:`G1MUSEKpLatentRLTrackingEnvCfg`.
    """

    observations: G1MUSEKpLatentRLKp5ObservationsCfg = G1MUSEKpLatentRLKp5ObservationsCfg()
    curriculum: LatentRLKp5DemoCurriculumCfg = LatentRLKp5DemoCurriculumCfg()

    def __post_init__(self):
        # super() (distill chain) sets KP5 body set + muse_kp5_unified_mode_spec + the
        # length-6 mix probs. Then swap in the latent-RL reward set (single source of truth
        # for the RL objective; the parent's anchor-weight tweaks are BC-only and discarded).
        super().__post_init__()
        self.rewards = LatentRLRewardsCfg()


@configclass
class LatentRLKp5WritingCurriculumCfg(CurriculumCfg):
    """Pin the keypoint mask to RIGHT-WRIST-ONLY for the wrist-writing finetune.

    The writing env inherits the 8-mode demo spec from the latent-distill chain
    (:func:`mask_modes.muse_kp5_latent_demo_mode_spec`), whose order is
    ``(full, vr, torso, left_wrist, right_wrist, wrists, ankles, bernoulli)``, so
    ``mode_probs=(0,0,0,0,1,0,0,0)`` selects ``right_wrist`` (index 4 — only the right wrist
    visible) every reset, held from iter 0. The policy sees ONLY the right-wrist target and
    is free to move the torso / step to reach the letters. ``p_*`` are irrelevant for an
    explicit single-point mode (bernoulli keep-prob only applies to the bernoulli mode).
    This curriculum is concept-independent of the mask MIX — it just needs the right index;
    it moved 5 -> 4 when the spec gained ``full``/``vr`` ahead of the single-point modes.
    """

    keypoint_mask_mode = CurrTerm(
        func=mdp.curriculums.keypoint_mask_mode_curriculum,
        params={
            "command_name": "motion",
            "phase_until_learning_iterations": (None,),
            "mask_phases": (
                {
                    "mode_probs": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),  # right_wrist only
                    "p_start": 1.0,
                    "p_end": 1.0,
                },
            ),
        },
    )


_WRITING_BODY = "right_wrist_yaw_link"


@configclass
class G1MUSEKpLatentRLKp5WritingObservationsCfg(
    G1MUSEKpLatentDistillationTrackingEnvCfg.MUSEKpLatentObservationsCfg
):
    """Wrist-writing obs — lean, writing-specific critic (no NaN / no constant-zero waste).

    ``policy`` + ``teacher`` are inherited unchanged (the warmstarted KP5 encoder sees the
    same obs it was distilled on). The ``critic`` is rebuilt for this task: the general KP5
    critic carries all-body reference terms that, under ``right_wrist_only`` + writing clips,
    are either NaN-then-zeroed (masked ``kp_lookahead``) or constant (the 4 stay-pose bodies'
    world/anchor references). This critic keeps ONLY informative, privileged signal:

      - robot proprio (joint pos/vel, base ang/lin vel, last action) — full robot config + base
        motion (privileged, clean — no corruption);
      - robot torso (anchor) world orientation + lin/ang vel — upright + base pose/motion;
      - the RIGHT-WRIST reference lookahead (``ref_single_body_pos_robot_anchor_b_logspaced``,
        UNMASKED, single body) — the only moving target, past/current/future in robot frame.

    Every term is a real, varying quantity for the writing task → no wasted dims.
    """

    @configclass
    class CriticCfg(ObsGroup):
        # Robot config + base motion (privileged, clean).
        proprio_joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=5)
        proprio_joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=5)
        proprio_base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=5)
        proprio_base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        proprio_actions = ObsTerm(func=mdp.last_action, history_length=5)
        # Robot torso (anchor) world pose/motion — upright state + global base motion.
        anchor_ori_w = ObsTerm(func=mdp.robot_anchor_ori_w, params={"command_name": "motion"})
        anchor_lin_vel_w = ObsTerm(func=mdp.robot_anchor_lin_vel_w_abs, params={"command_name": "motion"})
        anchor_ang_vel_w = ObsTerm(func=mdp.robot_anchor_ang_vel_w_abs, params={"command_name": "motion"})
        # Right-wrist target trajectory in the robot frame (UNMASKED, single body).
        rwrist_ref_lookahead = ObsTerm(
            func=mdp.ref_single_body_pos_robot_anchor_b_logspaced,
            params={
                "command_name": "motion",
                "body_name": _WRITING_BODY,
                "slot_offsets": KP_LAYOUT_SYM_SPARSE_0P5S,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False  # privileged critic: clean value targets
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class LatentRLKp5WritingRewardsCfg:
    """Writing reward — wrist-centric + ABSOLUTE stability (no fake motion-anchor terms).

    Differences from :class:`LatentRLRewardsCfg`:
      - **drops** ``motion_global_anchor_pos`` / ``motion_global_anchor_ori``. Their
        "reference" is the constant stay-pose torso we inject into the writing clips, so
        rewarding the robot to match it is a "don't move the base" signal that fights the
        locomotion the task needs (and is scored against a fake target). The world-frame
        ``poi_pos`` already pins global placement via the wrist's world target, so no
        anti-drift anchor is needed.
      - **adds** ``upright`` (absolute flat-orientation penalty, robot state only) so the
        robot stays vertical without referencing any motion anchor.
      - keeps the visible-POI pos/vel (right wrist under ``right_wrist_only``) + the penalty
        bundle verbatim.
    All weights/std are tunable.
    """

    # --- Task: world-frame visible point-of-interest (the right wrist) ---
    poi_pos = RewTerm(
        func=mdp.motion_visible_kp_position_error_exp_world,
        weight=12.0,
        params={"command_name": "motion", "std": 0.3},
    )
    poi_lin_vel = RewTerm(
        func=mdp.motion_visible_kp_lin_vel_error_exp_world,
        weight=6.0,
        params={"command_name": "motion", "std": 1.0},
    )

    # --- Absolute stability (robot state only; NO motion-anchor reference) ---
    upright = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # --- Penalty scaffolding (verbatim from LatentRLRewardsCfg) ---
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)


@configclass
class LatentRLKp5WritingTerminationsCfg:
    """Writing terminations — ABSOLUTE fall/tip only (no motion-reference deviation).

    The inherited :class:`TerminationsCfg` terminates on deviation from the motion anchor
    (``bad_anchor_pos_z_only`` / ``bad_anchor_ori``) and the ankle reference
    (``bad_motion_body_pos_z_only``). For the writing task those references are the injected
    stay-poses (fake), so this cfg replaces them with robot-state-only / visible-POI conditions:

      - ``time_out`` / ``motion_end`` — episode / clip bounds (kept, time_out=True).
      - ``fell`` — torso OR pelvis world height below 0.4 m (``fall_to_ground``, robot state).
      - ``tipped`` — robot tilt beyond ~0.9 rad from upright (``bad_orientation``, robot state).
      - ``wrist_diverged`` — FAILURE termination (time_out=False) when the visible wrist's
        world-frame RMS error exceeds 0.5 m after a 60-step (~1.2 s) lift-in grace window.
        Truncates doomed episodes (wrist hopelessly behind the moving target on a wide word)
        so rollout time isn't wasted finishing clips with zero chance of tracking — and the
        cut value-bootstrap teaches the critic those states are low-value. The grace window
        skips the initial standing→first-letter lift-in (~0.4-0.6 m) which would otherwise
        trip it at t=0. Threshold / grace are Hydra-tunable
        (``env.terminations.wrist_diverged.params.{threshold,grace_steps}``).
    """

    motion_end = DoneTerm(func=mdp.motion_end, params={"command_name": "motion"}, time_out=True)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell = DoneTerm(
        func=mdp.fall_to_ground,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "pelvis"]),
            "min_height": 0.4,
        },
    )
    tipped = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.9},
    )
    wrist_diverged = DoneTerm(
        func=mdp.bad_visible_kp_pos_world,
        params={"command_name": "motion", "threshold": 0.5, "grace_steps": 60},
    )


@configclass
class G1MUSEKpLatentRLKp5WritingTrackingEnvCfg(G1MUSEKpLatentRLKp5TrackingEnvCfg):
    """Wrist-writing specialization of the KP5 latent-RL env.

    Differences over the general KP5 latent-RL env:
      - **mask pinned to right-wrist-only** (:class:`LatentRLKp5WritingCurriculumCfg`).
      - **reset always to frame 0** (``start_from_beginning=True``). The synthetic writing
        clips have valid full-body state ONLY at frame 0; frames 1..T-1 are the right-wrist
        trajectory + NaN at every other body, so the default adaptive-frame reset would
        write a NaN root pose to the sim. Frame-0 reset spawns the robot at the seed standing
        pose every episode, then it writes from the start.
      - **lean writing critic** (:class:`G1MUSEKpLatentRLKp5WritingObservationsCfg`) — no
        NaN / constant-zero waste from masked or stay-pose bodies.
      - **wrist-centric reward + absolute terminations** — no fake motion-anchor reference
        (:class:`LatentRLKp5WritingRewardsCfg` / :class:`LatentRLKp5WritingTerminationsCfg`).
    """

    # obs + curriculum are safe as class attrs (no __post_init__ in the chain mutates them).
    observations: G1MUSEKpLatentRLKp5WritingObservationsCfg = G1MUSEKpLatentRLKp5WritingObservationsCfg()
    curriculum: LatentRLKp5WritingCurriculumCfg = LatentRLKp5WritingCurriculumCfg()
    # terminations is NOT a class attr: the distill chain's __post_init__ does
    # ``self.terminations.ee_body_pos.params[...]`` surgery, which would crash on a writing
    # terminations cfg that drops ee_body_pos. So we let super() run that surgery on the
    # inherited cfg, then REPLACE terminations afterward (established post-super idiom).

    def __post_init__(self):
        # Runs the general KP5 latent-RL chain (installs LatentRLRewardsCfg + the inherited
        # ankle-only ee_body_pos surgery), then swaps in the writing-specific reward,
        # terminations, and frame-0 reset.
        super().__post_init__()
        self.rewards = LatentRLKp5WritingRewardsCfg()
        self.terminations = LatentRLKp5WritingTerminationsCfg()
        self.commands.motion.start_from_beginning = True
        self.commands.motion.start_frame = 0
        # Writing clips average ~17 s (max ~32 s); the inherited 10 s episode would truncate
        # 94% of words, so the policy would only ever practice the first ~58% of each word.
        # Cap at 33 s (≥ max clip) so full words are written before time_out. Free: the PPO
        # rollout buffer is num_steps_per_env × num_envs (decoupled from episode length), and
        # motion_end / fall terminations still reset frequently — this only lets a good policy
        # finish words.
        self.episode_length_s = 33.0


# ========================================================================================
# Obstacle-reach downstream demo: reach a fixed point past an obstacle.
# Phases (selected by ``--motion <phase pool dir>`` at launch — split specialists):
#   0 free reach · 1 above a box · 2 into an open container · 3(=#4) under a low slab.
#
# SELF-CONTAINED on purpose: inherits only the general KP5 latent-RL env
# (:class:`G1MUSEKpLatentRLKp5TrackingEnvCfg`) and declares the reach reward / terminations /
# right-wrist mask explicitly below, so the whole task reads in one place rather than
# threading through the wrist-writing specialization.
#
# What makes it a "reach past an obstacle":
#   - the mask is pinned to the RIGHT WRIST only (the policy sees one target point and is
#     free to locomote/bend to reach it);
#   - the reach target is the constant right-wrist position baked into each pool clip
#     (scripts/gen_obstacle_clips.py); world-frame visible-POI reward pulls the wrist there;
#   - :class:`~...mdp.ObstacleReachCommand` attaches each clip's obstacle boxes (from its
#     ``.scene.json`` sidecar); ``obstacle_keepout`` penalizes any body entering them.
#
# The obstacle is OBSERVED by both the critic and the residual policy (option C): the obs cfg
# below appends ``mdp.obstacle_params_robot_anchor_b`` LAST in the policy group (the
# actor-critic strips it before the frozen encoder and feeds it to the residual corrector as
# per-box tokens) and adds it to the critic. See [[obstacle-reach-task-design]].
# ========================================================================================
# Robot bodies kept OUT of the solid obstacle boxes (regex). The open-container interior is
# NOT a box, so reaching the wrist INTO it is unpenalized.
_OBSTACLE_REACH_KEEPOUT_BODIES = ["torso_link", "pelvis", ".*_shoulder.*", ".*_elbow.*", ".*_wrist.*"]
# Right wrist is the reach point of interest. The KP5 latent env uses the 8-mode demo spec
# (full / vr / torso / left_wrist / RIGHT_WRIST / wrists / ankles / bernoulli), so the
# right-wrist-only one-hot is index 4.
_OBSTACLE_REACH_RIGHT_WRIST_ONEHOT = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


@configclass
class ObstacleReachCurriculumCfg(CurriculumCfg):
    """Pin the keypoint mask to RIGHT-WRIST-ONLY every reset (single, never-ending phase)."""

    keypoint_mask_mode = CurrTerm(
        func=mdp.curriculums.keypoint_mask_mode_curriculum,
        params={
            "command_name": "motion",
            "phase_until_learning_iterations": (None,),
            "mask_phases": (
                {"mode_probs": _OBSTACLE_REACH_RIGHT_WRIST_ONEHOT, "p_start": 1.0, "p_end": 1.0},
            ),
        },
    )


@configclass
class ObstacleReachRewardsCfg:
    """Single-goal reach: world-frame POSITION accuracy of the visible POI (the right wrist
    -> the constant target P) + OBB keep-out + the standard smoothness/limit penalties.

    Deliberately MINIMAL vs the writing reward (the target here is a single STATIC point):
      - NO ``poi_lin_vel``: the target is static, so the reference wrist velocity is 0; a
        velocity-match term would reward stillness and fight the approach. Position alone
        drives the reach; the success-hold + action/joint penalties handle settling.
      - NO ``upright`` posture term: several phases REQUIRE bending/leaning, so penalizing
        base tilt is counter-productive. Stability is enforced by the fell/tipped
        terminations + the frozen motor prior (the decoder produces balanced motion).
    All weights/std are Hydra-tunable."""

    # --- Task: reach the right-wrist target (world-frame visible-POI position) ---
    # Coarse-to-fine so there's a real gradient across the full ~1 m walk-up (the exp kernel
    # alone saturates to ~0 beyond ~0.5 m, leaving the approach near-sparse). ``poi_pos`` (fine,
    # std 0.4) sharpens near the goal; ``poi_pos_approach`` (coarse, std 0.8) pulls the wrist in
    # from far out so the policy gets signal while still approaching.
    poi_pos = RewTerm(
        func=mdp.motion_visible_kp_position_error_exp_world,
        weight=12.0, params={"command_name": "motion", "std": 0.4},
    )
    poi_pos_approach = RewTerm(
        func=mdp.motion_visible_kp_position_error_exp_world,
        weight=4.0, params={"command_name": "motion", "std": 0.8},
    )
    # --- Stay out of the obstacle boxes (OBB penetration; negative weight) ---
    obstacle_keepout = RewTerm(
        func=mdp.obstacle_keepout_penalty,
        weight=-10.0,
        params={
            "command_name": "motion",
            "asset_cfg": SceneEntityCfg("robot", body_names=_OBSTACLE_REACH_KEEPOUT_BODIES),
        },
    )
    # --- Smoothness / limit penalties ---
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)


@configclass
class ObstacleReachTerminationsCfg:
    """Success-on-hold + absolute fall/tip + clip/time bounds. NO motion-reference deviation
    cut: the robot starts ~1 m from the station and walks in, so a wrist-divergence cut would
    fire during the approach."""

    # Success: wrist held within tol of the target for ~2 s (100 steps @ 0.02 s). time_out=True
    # so the value is BOOTSTRAPPED (a held reach is high-value; a `terminated` cut would teach
    # the critic the goal is worthless and discourage reaching). tol/hold are Hydra-tunable.
    reached = DoneTerm(
        func=mdp.obstacle_reach_succeeded,
        params={"command_name": "motion", "success_tol": 0.08, "hold_steps": 100},
        time_out=True,
    )
    motion_end = DoneTerm(func=mdp.motion_end, params={"command_name": "motion"}, time_out=True)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell = DoneTerm(
        func=mdp.fall_to_ground,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "pelvis"]), "min_height": 0.4},
    )
    tipped = DoneTerm(func=mdp.bad_orientation, params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.9})


# Reach POI = right wrist (the single visible keypoint under the right-wrist mask pin).
_OBSTACLE_REACH_BODY = "right_wrist_yaw_link"


@configclass
class G1MUSEKpLatentRLKp5ObstacleReachObservationsCfg(G1MUSEKpLatentRLKp5ObservationsCfg):
    """Obstacle-reach obs: inherits the KP5 latent-distill ``policy`` / ``teacher`` groups
    (the frozen encoder's input — unchanged) and declares a LEAN critic explicitly (no
    inheritance), carrying only informative signal for this single-point reach + the obstacle.

    ``obstacle_params_robot_anchor_b`` = MAX_OBSTACLES(5) × 7 = 35 dims (per box: center_b 3 +
    half 3 + valid 1, robot-anchor frame). MAX_OBSTACLES=5 because the phase-2 open container
    is 5 boxes (floor + 4 walls); phases 1/4 use 1 box, phase 0 none — unused slots are masked
    via the box's valid flag. The runner cfg's obstacle_feat_dim=35 / obstacle_n=5 MUST match.
    """

    @configclass
    class PolicyCfg(G1MUSEKpLatentRLKp5ObservationsCfg.PolicyCfg):
        # Per-reset target: build the KP lookahead from the constant reach point P (injected
        # at the wrist in command.body_pos_w) instead of gathering clip frames. Same packing /
        # dim as the inherited kp_lookahead, so the frozen encoder is unaffected; keeps its
        # (first) position via the field override.
        kp_lookahead = ObsTerm(
            func=mdp.obstacle_reach_kp_lookahead,
            params={"command_name": "motion", "slot_offsets": KP_LAYOUT_SYM_SPARSE_0P5S},
        )
        # Obstacle MUST stay LAST: LatentRLActorCritic strips the trailing 35 dims before the
        # frozen encoder and routes them to the residual corrector (option C).
        obstacle = ObsTerm(func=mdp.obstacle_params_robot_anchor_b, params={"command_name": "motion"})

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Lean privileged critic (mirrors the wrist-writing critic, declared explicitly):
        clean robot proprio + torso(anchor) world pose/motion + the RIGHT-WRIST target
        lookahead (the only moving goal) + the obstacle OBBs. No all-body / NaN-masked /
        constant-zero terms (the waste the general KP5 critic carries under a single-point
        mask). Privileged: ``enable_corruption=False``; the value MLP auto-detects the dim."""

        # Robot config + base motion (privileged, clean).
        proprio_joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=5)
        proprio_joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=5)
        proprio_base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=5)
        proprio_base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        proprio_actions = ObsTerm(func=mdp.last_action, history_length=5)
        # Torso (anchor) world orientation + lin/ang vel — upright + global base motion.
        anchor_ori_w = ObsTerm(func=mdp.robot_anchor_ori_w, params={"command_name": "motion"})
        anchor_lin_vel_w = ObsTerm(func=mdp.robot_anchor_lin_vel_w_abs, params={"command_name": "motion"})
        anchor_ang_vel_w = ObsTerm(func=mdp.robot_anchor_ang_vel_w_abs, params={"command_name": "motion"})
        # Right-wrist reach target, robot-anchor frame (UNMASKED single body) — per-reset
        # constant-target variant (sources the wrist ref from command.body_pos_w).
        rwrist_ref_lookahead = ObsTerm(
            func=mdp.obstacle_reach_rwrist_lookahead,
            params={
                "command_name": "motion",
                "body_name": _OBSTACLE_REACH_BODY,
                "slot_offsets": KP_LAYOUT_SYM_SPARSE_0P5S,
            },
        )
        # Obstacle OBBs (privileged value signal).
        obstacle = ObsTerm(func=mdp.obstacle_params_robot_anchor_b, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class G1MUSEKpLatentRLKp5ObstacleReachTrackingEnvCfg(G1MUSEKpLatentRLKp5TrackingEnvCfg):
    """Reach a fixed point past an obstacle (KP5 latent-RL specialist; phase = --motion <pool>)."""

    # Obstacle in policy (stripped to the residual corrector) + critic.
    observations: G1MUSEKpLatentRLKp5ObstacleReachObservationsCfg = G1MUSEKpLatentRLKp5ObstacleReachObservationsCfg()
    # Right-wrist mask pin (overrides the general env's 8-mode demo curriculum).
    curriculum: ObstacleReachCurriculumCfg = ObstacleReachCurriculumCfg()

    def __post_init__(self):
        # Run the general KP5 latent-RL chain (5 bodies + 8-mode spec + base reward/term and
        # its ee_body_pos surgery), THEN swap in the reach-specific reward/terminations and
        # the ObstacleReach command (replace-after-super is required: the inherited
        # __post_init__ mutates self.terminations, which our cfg doesn't share).
        super().__post_init__()
        self.rewards = ObstacleReachRewardsCfg()
        self.terminations = ObstacleReachTerminationsCfg()
        self.commands.motion.class_type = mdp.ObstacleReachCommand
        # SOLID obstacles (option 2): attach FIXED-size KINEMATIC box colliders for this phase
        # so the robot physically cannot pass through (the OBB keep-out reward is now a soft
        # complement). Phase from env var OBSTACLE_REACH_PHASE — the same var the command reads;
        # the command writes obstacle_0..N-1 poses per reset. Free-reach phase adds none.
        _phase = int(os.environ.get("OBSTACLE_REACH_PHASE", "1"))
        for _name, _cfg in mdp.make_obstacle_collider_cfgs(_phase).items():
            setattr(self.scene, _name, _cfg)
        # Standing seed is a fully-finite CONSTANT pose (gen_obstacle_clips freezes all frames);
        # reset to frame 0 (start_from_beginning) — any frame is the same standing pose anyway.
        self.commands.motion.start_from_beginning = True
        self.commands.motion.start_frame = 0
        # Match the generated clip duration (gen_obstacle_clips.py --duration); motion_end /
        # fall still reset earlier.
        self.episode_length_s = 8.0


@configclass
class G1MUSEKp5FromScratchTrackingEnvCfg(MUSEKp5FromScratchTrackingEnvCfg):
    """G1 5-body-native env for the from-scratch pilot-anneal run.

    Encoder sees exactly the 5 kp5 demo bodies (torso + both wrists + both ankles); no 14-body
    tokens. Curriculum is the 2-phase :class:`MUSEKp5FromScratchCurriculumCfg` over the
    5-mode :func:`mask_modes.muse_kp5_curriculum_mode_spec` (bernoulli + 4 demo modes).
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        # KP6 = KP5 + pelvis (pelvis FIRST so it is also the resolved articulation-root index).
        # The motion-command reset writes the robot root-link (pelvis) state; the old KP5 set
        # dropped pelvis and led with torso_link, so the base spawned at the torso reference
        # every reset (teacher read ~0.80 not ~0.99). Mode count is unchanged (5) — pelvis only
        # joins bernoulli/kp6_full; the demo modes are pelvis-free, so deployment is unchanged.
        self.commands.motion.body_names = list(mask_modes.KP6_NATIVE_BODIES)
        # CANONICAL unified 6-mode spec (was muse_kp6_curriculum 5-mode — now unified).
        self.commands.motion.mask_mode_spec = mask_modes.muse_kp6_unified_mode_spec()
        # Initial probs = M1 (bernoulli-only, single-source constant); curriculum overwrites.
        self.commands.motion.mask_mode_probs = mask_modes.MUSE_KP6_UNIFIED_BERNOULLI_PROBS


@configclass
class G1PartialMaskedVAEDistillationTrackingEnvCfg(G1VAEDistillationTrackingEnvCfg):
    """AnyBody latent distillation with partial keypoint masking handled in the motion command + curriculum."""

    commands: PartialMaskedMultiMotionCommandsCfg = PartialMaskedMultiMotionCommandsCfg()
    curriculum: PartialMaskedVaeDistillationCurriculumCfg = PartialMaskedVaeDistillationCurriculumCfg()

    def __post_init__(self):
        # Parent ``VAEDistillationTrackingEnvCfg.__post_init__`` assigns ``self.commands = MultiMotionCommandsCfg()``,
        # which would replace this class's ``PartialMaskedMultiMotionCommandsCfg``. Run super first, then restore.
        super().__post_init__()
        anchor = self.commands.motion.anchor_body_name
        bodies = list(self.commands.motion.body_names)
        self.commands = PartialMaskedMultiMotionCommandsCfg()
        self.commands.motion.anchor_body_name = anchor
        self.commands.motion.body_names = bodies
        spec = mask_modes.partial_masked_2b_g1_mode_spec()
        self.commands.motion.mask_mode_spec = spec

        # Ordering matches ``PARTIAL_MASKED_2B_G1_MODE_NAMES``: 10 semantic modes + trailing ``bernoulli``.
        PHASE1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # full keypoints 
        PHASE2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] # bernoulli sampling: p from 1 to 0.5
        PHASE3 = [1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/15, 1/3] 
        PHASE4 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1/3, 1/3, 1/3, 0.0] 

        #self.commands.motion.mask_mode_probs = CURRENT_MASK_MODE_PROBS
        self.curriculum.keypoint_mask_mode.params = {
            **self.curriculum.keypoint_mask_mode.params,
            # Two phases: exclusive upper bounds in learning-iters; last phase must end with ``None``.
            # Example: whole-body for iters 0–1, then upper-EE — use ``(2, None)``.
            "phase_until_learning_iterations": (3000, 6000, None),
            # Per-phase schedule dict: includes both mode probs and Bernoulli p_start/p_end.
            "mask_phases": (
                {"mode_probs": tuple(PHASE1), "p_start": 0.0, "p_end": 0.0},
                {"mode_probs": tuple(PHASE2), "p_start": 1.0, "p_end": 0.5},
                {"mode_probs": tuple(PHASE4), "p_start": 0.5, "p_end": 0.5},
            ),
        }



'''
            "phase_until_learning_iterations": (None,),
            # Per-phase schedule dict: includes both mode probs and Bernoulli p_start/p_end.
            "mask_phases": (
                {"mode_probs": tuple(PHASE1), "p_start": 0.0, "p_end": 0.0},
                {"mode_probs": tuple(PHASE2), "p_start": 1.0, "p_end": 0.5},
                {"mode_probs": tuple(PHASE3), "p_start": 0.5, "p_end": 0.5},
                {"mode_probs": tuple(PHASE4), "p_start": 0.5, "p_end": 0.5},
            ),

'''
