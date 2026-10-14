"""Robot platform registry for motion replay/conversion scripts.

Goal:
- Centralize robot-specific configs (ArticulationCfg) and joint ordering.
- Make scripts extensible: adding a new robot should only require registering it here.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.assets import ArticulationCfg

from .g1 import G1_CYLINDER_CFG

# Missing optional robots must not make the complete G1 platform unavailable.
# Only suppress absence of the requested module itself. A dependency failure
# inside an existing robot module is a real error and must propagate.
try:
    from .h1_2 import H1_2_CYLINDER_CFG
except ModuleNotFoundError as error:
    if error.name != f"{__package__}.h1_2":
        raise
    H1_2_CYLINDER_CFG = None
try:
    from .adam import ADAM_CYLINDER_CFG
except ModuleNotFoundError as error:
    if error.name != f"{__package__}.adam":
        raise
    ADAM_CYLINDER_CFG = None


@dataclass(frozen=True)
class RobotPlatformSpec:
    """Specification for one robot platform used by replay/conversion scripts."""

    name: str
    cfg: ArticulationCfg
    joint_names: list[str]

    def __post_init__(self) -> None:
        assert self.cfg is not None, "An unavailable robot cannot be registered"
        assert self.joint_names and len(self.joint_names) == len(set(self.joint_names))


ROBOT_PLATFORMS: dict[str, RobotPlatformSpec] = {
    "g1": RobotPlatformSpec(
        name="g1",
        cfg=G1_CYLINDER_CFG,
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    ),
    **({"h1_2": RobotPlatformSpec(
        name="h1_2",
        cfg=H1_2_CYLINDER_CFG,
        joint_names=[
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "torso_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    )} if H1_2_CYLINDER_CFG is not None else {}),
    **({"adam": RobotPlatformSpec(
        name="adam",
        cfg=ADAM_CYLINDER_CFG,
        joint_names=[
            "hipPitch_Left",
            "hipRoll_Left",
            "hipYaw_Left",
            "kneePitch_Left",
            "anklePitch_Left",
            "ankleRoll_Left",
            "hipPitch_Right",
            "hipRoll_Right",
            "hipYaw_Right",
            "kneePitch_Right",
            "anklePitch_Right",
            "ankleRoll_Right",
            "waistRoll",
            "waistPitch",
            "waistYaw",
            "shoulderPitch_Left",
            "shoulderRoll_Left",
            "shoulderYaw_Left",
            "elbow_Left",
            "shoulderPitch_Right",
            "shoulderRoll_Right",
            "shoulderYaw_Right",
            "elbow_Right",
        ],
    )} if ADAM_CYLINDER_CFG is not None else {}),
}


def available_robot_names() -> list[str]:
    """Return available robot platform names (stable order for CLI choices)."""
    return sorted(ROBOT_PLATFORMS.keys())


def get_robot_platform(name: str) -> RobotPlatformSpec:
    """Get robot platform spec by name."""
    if name not in ROBOT_PLATFORMS:
        raise KeyError(f"Unknown robot platform '{name}'. Available: {available_robot_names()}")
    return ROBOT_PLATFORMS[name]


if __name__ == "__main__":
    platform = get_robot_platform("g1")
    assert platform.cfg is G1_CYLINDER_CFG and len(platform.joint_names) == 29
    assert all(spec.cfg is not None for spec in ROBOT_PLATFORMS.values())
    print(f"Registry sanity: available={available_robot_names()}, G1 joint_count=29, PASS")


