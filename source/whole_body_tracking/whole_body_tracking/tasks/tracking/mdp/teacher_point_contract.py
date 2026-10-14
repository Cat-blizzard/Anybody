"""Explicit G1 teacher reward points, independent of the simulator.

Contract: positions (B,N,3) are world link origins in metres; orientations
(B,N,4) are unit WXYZ quaternions in the same body order. The extension
retains the original four shoulder/elbow points and lower eight, adding the
two wrist origins to upper. Its mean denominator changes from four to six.
The virtual head is the nominal isolated-head inertial centre in this G1
URDF, not the merged torso CoM, a headset marker, or a certified author point.
These are versioned reward definitions, not evidence of training improvement.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import torch

G1_UPPER_V1 = (
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
G1_LOWER_V1 = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
)
G1_HEAD_POINT_V1 = "g1_urdf_nominal_isolated_head_com_v1"
G1_URDF = Path(__file__).resolve().parents[3] / "assets/unitree_description/urdf/g1/main.urdf"


def required_body_indexes(names: Sequence[str], required: Sequence[str]) -> list[int]:
    """Reject a missing/duplicate point instead of silently dropping it."""
    if len(set(names)) != len(names) or len(set(required)) != len(required):
        raise ValueError("Reward body names must be unique")
    missing = sorted(set(required) - set(names))
    if missing:
        raise ValueError(f"Missing required reward bodies: {missing}")
    return [names.index(name) for name in required]


def validate_g1_partition(names: Sequence[str]) -> None:
    required_body_indexes(names, G1_UPPER_V1 + G1_LOWER_V1)
    if len(names) != 14 or set(names) != set(G1_UPPER_V1 + G1_LOWER_V1):
        raise ValueError("G1 teacher reward v1 requires exactly the declared 14 body points")
    assert set(G1_UPPER_V1).isdisjoint(G1_LOWER_V1)


def _vector(text: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in text.split())
    assert len(values) == 3 and all(math.isfinite(value) for value in values)
    return values


@lru_cache(maxsize=4)
def g1_head_offset_from_urdf(path: Path = G1_URDF) -> tuple[float, float, float]:
    """Return a named, asset-derived nominal head point in the torso link frame."""
    root = ET.parse(path).getroot()
    joints = [node for node in root.findall("joint") if node.get("name") == "head_joint"]
    assert len(joints) == 1 and joints[0].get("type") == "fixed"
    joint = joints[0]
    assert joint.find("parent").get("link") == "torso_link"
    assert joint.find("child").get("link") == "head_link"
    origin = joint.find("origin")
    assert origin is not None and _vector(origin.get("rpy", "0 0 0")) == (0.0, 0.0, 0.0)
    translation = _vector(origin.get("xyz"))
    heads = [node for node in root.findall("link") if node.get("name") == "head_link"]
    assert len(heads) == 1
    inertial = heads[0].find("inertial/origin")
    assert inertial is not None
    centre = _vector(inertial.get("xyz"))
    offset = tuple(a + b for a, b in zip(translation, centre))
    assert all(math.isfinite(value) for value in offset)
    return offset


def _position_contract(reference: torch.Tensor, actual: torch.Tensor) -> None:
    assert reference.ndim == 3 and reference.shape == actual.shape
    assert reference.shape[0] > 0 and reference.shape[1] > 0 and reference.shape[2] == 3
    assert reference.device == actual.device and reference.dtype == actual.dtype
    assert reference.is_floating_point() and torch.isfinite(reference).all() and torch.isfinite(actual).all()


def point_position_reward(reference: torch.Tensor, actual: torch.Tensor, std: float) -> torch.Tensor:
    """Original coordinate-MSE exponential: mean over XYZ, then body points."""
    _position_contract(reference, actual)
    assert math.isfinite(std) and std > 0
    error = (reference - actual).square().mean(dim=-1).mean(dim=-1)
    reward = torch.exp(-error / std**2)
    assert reward.shape == (reference.shape[0],) and torch.isfinite(reward).all()
    return reward


def extension_position_reward(
    reference: torch.Tensor, actual: torch.Tensor, names: Sequence[str],
    upper_std: float, lower_std: float, upper_weight: float, lower_weight: float,
) -> torch.Tensor:
    _position_contract(reference, actual)
    assert len(names) == reference.shape[1]
    validate_g1_partition(names)
    assert all(math.isfinite(value) and value >= 0 for value in (upper_weight, lower_weight))
    upper = required_body_indexes(names, G1_UPPER_V1)
    lower = required_body_indexes(names, G1_LOWER_V1)
    return (upper_weight * point_position_reward(reference[:, upper], actual[:, upper], upper_std)
            + lower_weight * point_position_reward(reference[:, lower], actual[:, lower], lower_std))


def rotate_wxyz(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    assert quaternion.ndim == point.ndim == 2
    assert quaternion.shape == (point.shape[0], 4) and point.shape[1] == 3
    assert torch.isfinite(quaternion).all() and torch.isfinite(point).all()
    squared_norm = quaternion.square().sum(dim=-1, keepdim=True)
    assert torch.allclose(squared_norm, torch.ones_like(squared_norm), atol=1e-3, rtol=0)
    quaternion = quaternion / squared_norm.sqrt()
    axis = quaternion[:, 1:]
    return point + 2 * torch.linalg.cross(
        axis, torch.linalg.cross(axis, point, dim=-1) + quaternion[:, :1] * point, dim=-1,
    )


def g1_three_points(
    positions: torch.Tensor, orientations: torch.Tensor, names: Sequence[str],
    head_offset_in_torso: Sequence[float], head_point_definition: str,
) -> torch.Tensor:
    assert positions.ndim == orientations.ndim == 3
    assert positions.shape[:2] == orientations.shape[:2]
    assert positions.shape[2] == 3 and orientations.shape[2] == 4 and positions.shape[0] > 0
    assert positions.device == orientations.device and positions.dtype == orientations.dtype
    assert len(names) == positions.shape[1] and torch.isfinite(positions).all()
    assert torch.isfinite(orientations).all()
    if head_point_definition != G1_HEAD_POINT_V1:
        raise ValueError(f"Unsupported head point definition: {head_point_definition}")
    offset = tuple(float(value) for value in head_offset_in_torso)
    assert len(offset) == 3 and all(math.isfinite(value) for value in offset)
    # This profile is explicitly bound to this asset point, not an arbitrary torso proxy.
    assert all(abs(a-b) < 1e-9 for a,b in zip(offset, g1_head_offset_from_urdf()))
    torso, left, right = required_body_indexes(
        names, ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"),
    )
    local = positions.new_tensor(offset).expand(positions.shape[0], 3)
    head = positions[:, torso] + rotate_wxyz(orientations[:, torso], local)
    result = torch.stack((head, positions[:, left], positions[:, right]), dim=1)
    assert result.shape == (positions.shape[0], 3, 3) and torch.isfinite(result).all()
    return result


def sanity_check() -> None:
    names = list(G1_LOWER_V1 + G1_UPPER_V1)
    validate_g1_partition(names)
    offset = g1_head_offset_from_urdf()
    assert all(abs(a-b) < 1e-12 for a,b in zip(offset, (.0092305, .000299, .405869)))
    for batch in (1, 3):
        p = torch.zeros(batch, 14, 3, dtype=torch.float64)
        q = torch.zeros(batch, 14, 4, dtype=p.dtype); q[..., 0] = 1
        points = g1_three_points(p, q, names, offset, G1_HEAD_POINT_V1)
        assert points.shape == (batch, 3, 3)
        assert torch.all(extension_position_reward(p, p, names, .5, .5, 1, 1) == 2)
        assert torch.all(point_position_reward(points, points, .5) == 1)
        a = p.clone(); a[:, names.index("left_wrist_yaw_link"), 0] = .3
        assert torch.all(extension_position_reward(p, a, names, .5, .5, 1, 1) < 2)
        shift = p.new_tensor([2.0, -1.0, .7])
        shifted = g1_three_points(p+shift, q, names, offset, G1_HEAD_POINT_V1)
        assert torch.allclose(shifted, points+shift)
        print(f"G1 point sanity B={batch}: positions={tuple(p.shape)}, points={tuple(points.shape)}, PASS")
    rotation = torch.tensor([[2**-.5, 0., 2**-.5, 0.]], dtype=torch.float64)
    assert torch.allclose(rotate_wxyz(rotation, torch.tensor([[0., 0., 1.]], dtype=torch.float64)),
                          torch.tensor([[1., 0., 0.]], dtype=torch.float64), atol=1e-12)
    for invalid in (names[:-1], names + ["head_link"], names[:-1] + [names[0]]):
        try:
            validate_g1_partition(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid point schema must fail")
    print(f"Head definition={G1_HEAD_POINT_V1}; offset_m={offset}; no simulator used")


if __name__ == "__main__":
    sanity_check()
