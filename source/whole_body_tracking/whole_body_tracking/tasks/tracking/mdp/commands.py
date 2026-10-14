from __future__ import annotations

import copy
import math
import numpy as np
import os
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from collections.abc import Sequence
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    quat_rotate_inverse,
    sample_uniform,
    subtract_frame_transforms,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _refresh_reset_relative_body_poses(
    command: MotionCommand | MultiMotionCommand, env_ids: Sequence[int] | torch.Tensor,
) -> None:
    """Refresh only reset rows, without advancing time or sampling anything.

    World-frame positions are (N, B, 3), quaternions use wxyz (N, B, 4).
    Preserve the existing yaw alignment: robot anchor x/y, reference anchor z,
    and yaw(robot_anchor * inverse(reference_anchor)). Root and joint writes
    invalidate IsaacLab's body-link pose buffers; reading robot_anchor below
    performs lazy forward kinematics before fetching the new torso pose. Do
    not substitute the articulation root pose for the torso/anchor pose, and
    do not call _update_command: it advances frames and updates sampling EMA.
    Cache tensors retain their identities and non-reset rows remain untouched.
    """
    ids = torch.as_tensor(env_ids, device=command.device, dtype=torch.long)
    assert ids.ndim == 1
    if ids.numel() == 0:
        return
    assert bool(((ids >= 0) & (ids < command.num_envs)).all())
    count, bodies = ids.numel(), len(command.cfg.body_names)
    anchor_pos = command.anchor_pos_w[ids, None, :].repeat(1, bodies, 1)
    anchor_quat = command.anchor_quat_w[ids, None, :].repeat(1, bodies, 1)
    robot_anchor_pos = command.robot_anchor_pos_w[ids, None, :].repeat(1, bodies, 1)
    robot_anchor_quat = command.robot_anchor_quat_w[ids, None, :].repeat(1, bodies, 1)
    assert anchor_pos.shape == robot_anchor_pos.shape == (count, bodies, 3)
    assert anchor_quat.shape == robot_anchor_quat.shape == (count, bodies, 4)
    robot_anchor_pos[..., 2] = anchor_pos[..., 2]
    yaw_delta = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
    relative_pos = robot_anchor_pos + quat_apply(yaw_delta, command.body_pos_w[ids] - anchor_pos)
    relative_quat = quat_mul(yaw_delta, command.body_quat_w[ids])
    assert relative_pos.shape == (count, bodies, 3) and torch.isfinite(relative_pos).all()
    assert relative_quat.shape == (count, bodies, 4) and torch.isfinite(relative_quat).all()
    command.body_pos_relative_w[ids] = relative_pos
    command.body_quat_relative_w[ids] = relative_quat


# Keypoint debug markers: the policy observes keypoint *positions* (not rigid-body poses),
# so a sphere is the honest visual primitive (a frame triad implies an orientation the
# agent never sees). Green = robot's current keypoint, red = the command goal keypoint.
_KP_MARKER_RADIUS = 0.025


def _kp_sphere_marker_cfg(rgb: tuple[float, float, float]) -> VisualizationMarkersCfg:
    """A single-sphere VisualizationMarkersCfg in the given color (prim_path set by callers)."""
    return VisualizationMarkersCfg(
        prim_path="/Visuals/Command/pose",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=_KP_MARKER_RADIUS,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
            )
        },
    )


def _get_rank_world_size(shard_by: str = "global") -> tuple[int, int]:
    """Best-effort (rank, world_size) for multi-GPU dataset sharding.

    Priority:
    1) torch.distributed (if initialized)
    2) environment variables (torchrun/accelerate/slurm compatible)
    """

    shard_by = str(shard_by).lower()
    if shard_by not in {"global", "local"}:
        raise ValueError(f"Invalid shard_by={shard_by!r}. Expected 'global' or 'local'.")

    # torch.distributed path
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
            return rank, world_size
    except Exception:
        pass

    # env var path
    def _get_int_env(name: str, default: int) -> int:
        val = os.getenv(name, "")
        if val == "":
            return default
        try:
            return int(val)
        except Exception:
            return default

    if shard_by == "local":
        rank = _get_int_env("LOCAL_RANK", 0)
        world_size = _get_int_env("LOCAL_WORLD_SIZE", _get_int_env("WORLD_SIZE", 1))
    else:
        rank = _get_int_env("RANK", _get_int_env("LOCAL_RANK", 0))
        world_size = _get_int_env("WORLD_SIZE", _get_int_env("LOCAL_WORLD_SIZE", 1))

    world_size = max(int(world_size), 1)
    rank = int(rank) % world_size
    return rank, world_size


def _select_motion_paths_for_rank(
    motion_paths: list[str],
    *,
    max_motions: int | None,
    shard_across_gpus: bool,
    shard_by: str,
    shard_seed: int,
    shard_strategy: str = "chunk",
) -> tuple[list[str], dict[str, int | str | bool]]:
    """Select a (possibly sharded) subset of motion_paths for the current process.

    Design goals:
    - When motions are plentiful and max_motions is small (e.g. <= num_envs), enable
      deterministic *disjoint* subsets across GPUs (when possible), to reduce duplicates.
    - Keep default behavior unchanged unless shard_across_gpus or max_motions is provided.
    """

    total = len(motion_paths)
    info: dict[str, int | str | bool] = {
        "total_motions": total,
        "selected_motions": total,
        "shard_across_gpus": bool(shard_across_gpus),
        "shard_by": str(shard_by),
        "shard_seed": int(shard_seed),
        "shard_strategy": str(shard_strategy),
        "rank": 0,
        "world_size": 1,
        "max_motions": int(max_motions) if max_motions is not None else -1,
    }

    if total == 0:
        return motion_paths, info

    if max_motions is None:
        # No selection requested.
        return motion_paths, info

    max_motions = int(max_motions)
    if max_motions <= 0:
        raise ValueError("max_motions must be a positive integer when provided.")

    rank, world_size = _get_rank_world_size(shard_by=shard_by)
    info["rank"] = rank
    info["world_size"] = world_size

    # Deterministic shuffle of paths to avoid correlated filesystem ordering.
    indices = list(range(total))
    rng = random.Random(int(shard_seed))
    rng.shuffle(indices)

    # If not sharding, just take the first max_motions after shuffle.
    if (not shard_across_gpus) or (world_size <= 1):
        selected_idx = indices[: min(max_motions, total)]
        selected = [motion_paths[i] for i in selected_idx]
        info["selected_motions"] = len(selected)
        return selected, info

    # Sharded selection:
    # If dataset is large enough, create disjoint fixed-size shards (best case).
    if total >= world_size * max_motions:
        start = rank * max_motions
        selected_idx = indices[start : start + max_motions]
        selected = [motion_paths[i] for i in selected_idx]
        info["selected_motions"] = len(selected)
        return selected, info

    # Otherwise, fall back to disjoint partitioning then (optionally) cap.
    shard_strategy = str(shard_strategy).lower()
    if shard_strategy not in {"chunk", "stride"}:
        raise ValueError(f"Invalid shard_strategy={shard_strategy!r}. Expected 'chunk' or 'stride'.")

    if shard_strategy == "stride":
        shard_idx = indices[rank::world_size]
    else:
        # chunk: contiguous chunks after shuffle
        chunk_size = int(math.ceil(total / float(world_size)))
        start = rank * chunk_size
        shard_idx = indices[start : start + chunk_size]

    if len(shard_idx) > max_motions:
        shard_idx = shard_idx[:max_motions]

    selected = [motion_paths[i] for i in shard_idx]
    info["selected_motions"] = len(selected)
    return selected, info


def _maybe_log_motion_shard_to_wandb_summary(
    shard_info: dict[str, int | str | bool], cfg: "MultiMotionCommandCfg"
) -> None:
    """Best-effort: log one-time shard info to Weights&Biases summary.

    Intended behavior:
    - If torch.distributed is initialized, gather per-rank loaded counts and write summary from rank0 only.
    - If wandb is not available or not initialized, do nothing.
    """

    if not getattr(cfg, "motion_dataset_log_wandb_summary", True):
        return

    # Import wandb lazily and safely.
    try:
        import wandb  # type: ignore

        run = getattr(wandb, "run", None)
        if run is None:
            return
    except Exception:
        return

    rank = int(shard_info.get("rank", 0)) if isinstance(shard_info.get("rank", 0), (int, float)) else 0
    world_size = (
        int(shard_info.get("world_size", 1)) if isinstance(shard_info.get("world_size", 1), (int, float)) else 1
    )
    total = (
        int(shard_info.get("total_motions", 0)) if isinstance(shard_info.get("total_motions", 0), (int, float)) else 0
    )
    loaded = (
        int(shard_info.get("selected_motions", 0))
        if isinstance(shard_info.get("selected_motions", 0), (int, float))
        else 0
    )

    # Try distributed gather for per-rank reporting.
    loaded_list: list[int] | None = None
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            t = torch.tensor([loaded], dtype=torch.long, device=device)
            out = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
            dist.all_gather(out, t)
            loaded_list = [int(x.item()) for x in out]
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
    except Exception:
        loaded_list = None

    # Only rank0 writes summary to avoid collisions.
    if rank != 0:
        return

    # Stable keys for easy browsing.
    run.summary["motion_dataset/world_size"] = int(world_size)
    run.summary["motion_dataset/total_motions_seen_by_rank0"] = int(total)
    run.summary["motion_dataset/shard_enabled"] = bool(getattr(cfg, "motion_dataset_shard_across_gpus", False))
    run.summary["motion_dataset/shard_by"] = str(getattr(cfg, "motion_dataset_shard_by", "global"))
    run.summary["motion_dataset/shard_strategy"] = str(getattr(cfg, "motion_dataset_shard_strategy", "chunk"))
    run.summary["motion_dataset/shard_seed"] = int(getattr(cfg, "motion_dataset_shard_seed", 0))
    run.summary["motion_dataset/load_cap"] = (
        int(getattr(cfg, "motion_dataset_load_cap", -1))
        if getattr(cfg, "motion_dataset_load_cap", None) is not None
        else None
    )

    if loaded_list is not None:
        # Store as a compact string to avoid schema issues.
        run.summary["motion_dataset/loaded_motions_per_rank"] = str(loaded_list)
        run.summary["motion_dataset/loaded_motions_sum"] = int(sum(loaded_list))
        run.summary["motion_dataset/loaded_motions_min"] = int(min(loaded_list)) if len(loaded_list) > 0 else 0
        run.summary["motion_dataset/loaded_motions_max"] = int(max(loaded_list)) if len(loaded_list) > 0 else 0
    else:
        run.summary["motion_dataset/loaded_motions_rank0"] = int(loaded)


def _sanitize_joint_metric_name(joint_name: str) -> str:
    """Convert joint names to stable metric-safe tokens."""
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(joint_name))


# Wrist links used as manipulation end-effector keypoints (G1 and matching configs).
# Per-group metrics: mean L2 error over the listed bodies (for both wrists, mean of the two keypoint errors).
_EE_KEYPOINT_METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("left_ee", ("left_wrist_yaw_link",)),
    ("right_ee", ("right_wrist_yaw_link",)),
    ("both_ee", ("left_wrist_yaw_link", "right_wrist_yaw_link")),
)

# Partial-mask mode names (``mask_cfg.mode_spec`` keys) that define which EE scalar to log for partial-mask runs.
_PULSE_VAE_EE_METRIC_LEFT_MODES: frozenset[str] = frozenset({"upper_left", "left_half", "left_end_effector"})
_PULSE_VAE_EE_METRIC_RIGHT_MODES: frozenset[str] = frozenset({"upper_right", "right_half", "right_end_effector"})
_PULSE_VAE_EE_METRIC_BOTH_MODES: frozenset[str] = frozenset({"upper_end_effector", "end_effector"})


@dataclass(frozen=True)
class _PulseVAEEEKeypointMetricContext:
    """Maps mask-matrix row index -> EE bucket id (0 = no EE-specific metric for that mode)."""

    row_to_bucket: torch.Tensor
    """Shape ``[num_modes]``, long, ``0`` none, ``1`` left_ee, ``2`` right_ee, ``3`` both_ee."""

    entries: tuple[tuple[int, str, str, torch.Tensor], ...]
    """``(bucket_id, pos_metric_key, vel_metric_key, body_cols_long[D])`` per active bucket."""


def _make_pulse_vae_ee_metric_context(
    mode_spec: dict[str, list[str]],
    body_names: Sequence[str],
    device: torch.device,
) -> _PulseVAEEEKeypointMetricContext | None:
    """Build EE keypoint metric context only for mask modes present in ``mode_spec`` with required bodies."""
    mode_names = list(mode_spec.keys())
    if not mode_names:
        return None
    available = frozenset(mode_names)
    name_to_i = {str(n): i for i, n in enumerate(body_names)}

    # (suffix, bucket_id, mode_names_for_bucket)
    bucket_defs: list[tuple[str, int, frozenset[str]]] = []
    if available & _PULSE_VAE_EE_METRIC_LEFT_MODES:
        bucket_defs.append(("left_ee", 1, _PULSE_VAE_EE_METRIC_LEFT_MODES))
    if available & _PULSE_VAE_EE_METRIC_RIGHT_MODES:
        bucket_defs.append(("right_ee", 2, _PULSE_VAE_EE_METRIC_RIGHT_MODES))
    if available & _PULSE_VAE_EE_METRIC_BOTH_MODES:
        bucket_defs.append(("both_ee", 3, _PULSE_VAE_EE_METRIC_BOTH_MODES))
    if not bucket_defs:
        return None

    suffix_to_bodies = dict(_EE_KEYPOINT_METRIC_GROUPS)
    entries: list[tuple[int, str, str, torch.Tensor]] = []
    for suffix, bid, _cset in bucket_defs:
        body_subset = suffix_to_bodies[suffix]
        indices: list[int] = []
        ok = True
        for bn in body_subset:
            if bn not in name_to_i:
                ok = False
                break
            indices.append(name_to_i[bn])
        if not ok:
            continue
        idx = torch.tensor(indices, dtype=torch.long, device=device)
        entries.append((bid, f"error_keypoint_pos_{suffix}", f"error_keypoint_vel_{suffix}", idx))

    if not entries:
        return None

    active_bids = {e[0] for e in entries}
    M = len(mode_names)
    row_to_bucket = torch.zeros(M, dtype=torch.long, device=device)
    for r, mn in enumerate(mode_names):
        if mn in _PULSE_VAE_EE_METRIC_LEFT_MODES and (available & _PULSE_VAE_EE_METRIC_LEFT_MODES) and (1 in active_bids):
            row_to_bucket[r] = 1
        elif mn in _PULSE_VAE_EE_METRIC_RIGHT_MODES and (available & _PULSE_VAE_EE_METRIC_RIGHT_MODES) and (2 in active_bids):
            row_to_bucket[r] = 2
        elif mn in _PULSE_VAE_EE_METRIC_BOTH_MODES and (available & _PULSE_VAE_EE_METRIC_BOTH_MODES) and (3 in active_bids):
            row_to_bucket[r] = 3

    return _PulseVAEEEKeypointMetricContext(row_to_bucket=row_to_bucket, entries=tuple(entries))


def _write_pulse_vae_ee_keypoint_metrics(
    metrics: dict,
    ctx: _PulseVAEEEKeypointMetricContext,
    env_mode_idx: torch.Tensor,
    pos_err_per_body: torch.Tensor,
    vel_err_per_body: torch.Tensor,
    *,
    num_envs: int,
    use_copy_: bool,
) -> None:
    """Aggregate EE errors over envs whose mask mode maps to each EE bucket; broadcast scalar per metric."""
    M = int(ctx.row_to_bucket.shape[0])
    valid = (env_mode_idx >= 0) & (env_mode_idx < M)
    bucket = torch.zeros(num_envs, dtype=torch.long, device=pos_err_per_body.device)
    bucket[valid] = ctx.row_to_bucket[env_mode_idx[valid]]

    for bid, pos_key, vel_key, body_idx in ctx.entries:
        m = bucket == int(bid)
        pe = pos_err_per_body.index_select(1, body_idx).mean(dim=-1)
        ve = vel_err_per_body.index_select(1, body_idx).mean(dim=-1)
        if m.any():
            pv = float(pe[m].mean().item())
            vv = float(ve[m].mean().item())
            if use_copy_:
                metrics[pos_key].fill_(pv)
                metrics[vel_key].fill_(vv)
            else:
                metrics[pos_key] = torch.full((num_envs,), pv, device=pos_err_per_body.device, dtype=pe.dtype)
                metrics[vel_key] = torch.full((num_envs,), vv, device=pos_err_per_body.device, dtype=ve.dtype)
        else:
            if use_copy_:
                metrics[pos_key].fill_(float("nan"))
                metrics[vel_key].fill_(float("nan"))
            else:
                metrics[pos_key] = torch.full((num_envs,), float("nan"), device=pos_err_per_body.device, dtype=pe.dtype)
                metrics[vel_key] = torch.full((num_envs,), float("nan"), device=pos_err_per_body.device, dtype=ve.dtype)


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int],
        device: str = "cpu",
        root_body_index: int = 0,
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        # Dedicated root channel: the articulation-root body sliced from the FULL npz body axis,
        # independent of the tracked ``_body_indexes``. The reset path reads these so the base is
        # correctly initialized even when the root (pelvis) is NOT a tracked keypoint (KP5).
        self.root_pos_w = self._body_pos_w[:, root_body_index]
        self.root_quat_w = self._body_quat_w[:, root_body_index]
        self.root_lin_vel_w = self._body_lin_vel_w[:, root_body_index]
        self.root_ang_vel_w = self._body_ang_vel_w[:, root_body_index]
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


def _resolve_video_debug_body_indices(body_names: list[str], cfg: object) -> list[int]:
    """Indices into ``body_names`` for motion-command debug body markers (subset or all)."""
    filt = getattr(cfg, "video_debug_vis_body_names", None)
    if not filt:
        return list(range(len(body_names)))
    body_set = set(body_names)
    out: list[int] = []
    for name in filt:
        if name not in body_set:
            raise ValueError(f"video_debug_vis_body_names contains {name!r} which is not in motion body_names.")
        out.append(body_names.index(name))
    return out


def _resolve_root_full_index(robot: Articulation) -> int:
    """Index of the robot's PhysX articulation root link into the **full** npz body axis.

    The reset path writes the root-link state via ``write_root_state_to_sim``, so it must read
    that exact body's motion reference. The PhysX articulation root is ``robot.body_names[0]``
    (G1: ``"pelvis"``). Motion npz body arrays are stored in ``robot.body_names`` order — both
    loaders slice them with ``robot.find_bodies`` indices — so the root's npz index is its
    position in ``robot.body_names`` (== 0).

    Resolved from the FULL body set, **independent of the tracked keypoint ``body_names``**, and
    fed to the loaders as a dedicated root channel. A sparse keypoint set that drops the root
    (e.g. KP5 = torso + L/R wrist + L/R ankle, no pelvis) therefore still resets the base from
    the correct reference instead of spawning it at the wrong tracked body (the prior
    root-reset bug). Behaviour is identical for sets that DO track the root (same physical
    reference, just sourced from the full array rather than the tracked-subset view).
    """
    root_link = robot.body_names[0]
    idx = list(robot.body_names).index(root_link)
    assert idx == 0, (
        f"Expected the PhysX articulation root {root_link!r} at body index 0, got {idx}. "
        "Motion npz body arrays are assumed robot-body-ordered (loaders slice the npz body "
        "axis with robot.find_bodies indices)."
    )
    return idx


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        # Set before super(): CommandTerm.__init__ calls set_debug_vis -> _set_debug_vis_impl.
        self._video_debug_body_indices = _resolve_video_debug_body_indices(cfg.body_names, cfg)
        self._video_debug_goal_world = bool(getattr(cfg, "video_debug_vis_goal_bodies_world_frame", False))
        self._video_debug_hide_anchor = bool(getattr(cfg, "video_debug_vis_hide_anchor", False))
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # Root reset reads the robot's actual articulation-root reference from a DEDICATED root
        # channel sourced from the full npz body axis — independent of the tracked keypoint
        # ``body_names``. Correct for sparse sets that drop pelvis (KP5); numerically identical
        # for sets that track it. See :func:`_resolve_root_full_index`.
        self._root_full_index = _resolve_root_full_index(self.robot)
        self.motion = MotionLoader(
            self.cfg.motion, self.body_indexes, device=self.device,
            root_body_index=self._root_full_index,
        )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

        self._dense_joint_error_metrics = bool(getattr(self.cfg, "enable_dense_joint_error_metrics", False))
        self._joint_metric_keys: list[tuple[str, str]] = []
        if self._dense_joint_error_metrics:
            self._joint_metric_keys = self._build_dense_joint_metric_keys()
            self._init_dense_joint_metrics()

        self._pulse_vae_ee_ctx: _PulseVAEEEKeypointMetricContext | None = None
        self._pulse_vae_env_mode_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

    def configure_pulse_vae_ee_keypoint_metrics(self, mode_spec: dict[str, list[str]] | None) -> None:
        """Enable partial-mask EE keypoint metrics for mask modes present in ``mode_spec`` (runner-only)."""
        self._pulse_vae_ee_ctx = (
            _make_pulse_vae_ee_metric_context(mode_spec, self.cfg.body_names, self.device) if mode_spec else None
        )
        if self._pulse_vae_ee_ctx is None:
            return
        for _bid, pk, vk, _ in self._pulse_vae_ee_ctx.entries:
            self.metrics[pk] = torch.full((self.num_envs,), float("nan"), device=self.device)
            self.metrics[vk] = torch.full((self.num_envs,), float("nan"), device=self.device)
        print(
            f"[{self.__class__.__name__}] partial-mask EE keypoint metrics: "
            f"{[e[1] for e in self._pulse_vae_ee_ctx.entries]}"
        )

    def set_pulse_vae_ee_env_mode_indices(self, mode_indices: torch.Tensor | None) -> None:
        """Per-env mask row index from the policy's ``sample_and_set_mask`` (same order as ``mode_spec``)."""
        if self._pulse_vae_ee_ctx is None:
            return
        if mode_indices is None:
            self._pulse_vae_env_mode_idx.fill_(-1)
            return
        self._pulse_vae_env_mode_idx.copy_(
            mode_indices.to(device=self.device, dtype=torch.long).view_as(self._pulse_vae_env_mode_idx)
        )

    def _build_dense_joint_metric_keys(self) -> list[tuple[str, str]]:
        keys: list[tuple[str, str]] = []
        for joint_name in self.robot.joint_names:
            safe = _sanitize_joint_metric_name(joint_name)
            keys.append((f"error_joint_pos_{safe}", f"error_joint_vel_{safe}"))
        return keys

    def _init_dense_joint_metrics(self) -> None:
        for pos_key, vel_key in self._joint_metric_keys:
            self.metrics[pos_key] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[vel_key] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def ref_body_pos_robot_anchor_b(self) -> torch.Tensor:
        """Reference motion body positions in the robot-anchor frame (b-frame).

        Same convention as :func:`whole_body_tracking.tasks.tracking.mdp.observations.robot_body_pos_b`,
        but uses reference ``body_pos_w`` / ``body_quat_w`` from the motion (not simulated robot bodies).
        """
        num_bodies = len(self.cfg.body_names)
        pos_b, _ = subtract_frame_transforms(
            self.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
            self.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
            self.body_pos_w,
            self.body_quat_w,
        )
        return pos_b.view(self.num_envs, -1)

    def ref_body_lin_vel_robot_anchor_b(self) -> torch.Tensor:
        """Reference motion body linear velocities in the robot-anchor frame (same layout as ``ref_body_pos_robot_anchor_b``)."""
        num_bodies = len(self.cfg.body_names)
        v_w = self.body_lin_vel_w
        q = self.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)
        v_w_flat = v_w.reshape(-1, 3)
        q_flat = q.reshape(-1, 4)
        v_b = quat_rotate_inverse(q_flat, v_w_flat)
        return v_b.view(self.num_envs, -1)

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)
        if self._dense_joint_error_metrics:
            joint_pos_error = torch.abs(self.joint_pos - self.robot_joint_pos)
            joint_vel_error = torch.abs(self.joint_vel - self.robot_joint_vel)
            for idx, (pos_key, vel_key) in enumerate(self._joint_metric_keys):
                self.metrics[pos_key] = joint_pos_error[:, idx]
                self.metrics[vel_key] = joint_vel_error[:, idx]
        if self._pulse_vae_ee_ctx is not None:
            pos_err_b = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1)
            vel_err_b = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1)
            _write_pulse_vae_ee_keypoint_metrics(
                self.metrics,
                self._pulse_vae_ee_ctx,
                self._pulse_vae_env_mode_idx,
                pos_err_b,
                vel_err_b,
                num_envs=self.num_envs,
                use_copy_=False,
            )

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.cfg.start_from_beginning:
            start_frame = max(int(self.cfg.start_frame), 0)
            start_frame = min(start_frame, max(self.motion.time_step_total - 1, 0))
            self.time_steps[env_ids] = start_frame
        else:
            self._adaptive_sampling(env_ids)

        # Articulation-root reference from the dedicated full-axis root channel (NOT the tracked
        # subset) — correct even when pelvis is not a tracked keypoint. ``root_pos_w`` is raw
        # (no env origin); ``body_pos_w`` added env_origins, so add it here too.
        root_pos = (self.motion.root_pos_w[self.time_steps] + self._env.scene.env_origins).clone()
        root_ori = self.motion.root_quat_w[self.time_steps].clone()
        root_lin_vel = self.motion.root_lin_vel_w[self.time_steps].clone()
        root_ang_vel = self.motion.root_ang_vel_w[self.time_steps].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        _refresh_reset_relative_body_poses(self, env_ids)

    def _update_command(self):
        # ManagerBasedRLEnv clears episode_length_buf after an auto-reset and
        # calls command_manager.compute() in the same step.  _resample_command
        # has already written the reset frame in that case, so advancing here
        # would expose frame k+1 while the robot state is still frame k.  The
        # next step has episode_length_buf == 1 and resumes the normal advance.
        episode_length = self._env.episode_length_buf
        if episode_length.shape != self.time_steps.shape:
            raise RuntimeError(
                "episode_length_buf and time_steps must have identical shapes for reset synchronization"
            )
        self.time_steps += (episode_length != 0).to(dtype=self.time_steps.dtype)
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for bi in self._video_debug_body_indices:
                    name = self.cfg.body_names[bi]
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.current_body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.goal_body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            _show_anchor = not getattr(self, "_video_debug_hide_anchor", False)
            self.current_anchor_visualizer.set_visibility(_show_anchor)
            self.goal_anchor_visualizer.set_visibility(_show_anchor)
            for i in range(len(self._video_debug_body_indices)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self._video_debug_body_indices)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        if not getattr(self, "_video_debug_hide_anchor", False):
            self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
            self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for vis_i, bi in enumerate(self._video_debug_body_indices):
            self.current_body_visualizers[vis_i].visualize(
                self.robot_body_pos_w[:, bi], self.robot_body_quat_w[:, bi]
            )
            if self._video_debug_goal_world:
                self.goal_body_visualizers[vis_i].visualize(self.body_pos_w[:, bi], self.body_quat_w[:, bi])
            else:
                self.goal_body_visualizers[vis_i].visualize(
                    self.body_pos_relative_w[:, bi], self.body_quat_relative_w[:, bi]
                )


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    start_from_beginning: bool = False
    start_frame: int = 0

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    # If enabled, log per-joint absolute errors for position and velocity tracking.
    # Metrics are emitted as: error_joint_pos_<joint_name>, error_joint_vel_<joint_name>.
    enable_dense_joint_error_metrics: bool = False

    # Anchor stays a frame: it is a true root *pose* whose orientation (heading) is meaningful.
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    # Keypoints are positions, not poses -> spheres. Green = robot current, red = command goal.
    current_body_visualizer_cfg: VisualizationMarkersCfg = _kp_sphere_marker_cfg((0.0, 1.0, 0.0))
    goal_body_visualizer_cfg: VisualizationMarkersCfg = _kp_sphere_marker_cfg((1.0, 0.0, 0.0))

    # Optional: restrict debug body markers and draw goal bodies in motion world frame (see play.py).
    video_debug_vis_body_names: list[str] | None = None
    video_debug_vis_goal_bodies_world_frame: bool = False
    # Optional: suppress the anchor frame triads so only the body dot markers render
    # (see play.py --poi5_dot_vis).
    video_debug_vis_hide_anchor: bool = False


class MultiMotionLoader:
    """Preload motion files into contiguous tensors for fast batched sampling."""

    def __init__(
        self,
        motion_dir: str,
        body_indexes: Sequence[int],
        device: str | torch.device = "cpu",
        file_glob: str = "*.npz",
        storage_device: str | torch.device | None = None,
        root_body_index: int = 0,
        *,
        max_motions: int | None = None,
        shard_across_gpus: bool = False,
        shard_by: str = "global",
        shard_seed: int = 0,
        shard_strategy: str = "chunk",
        motion_groups: dict[str, list[str]] | None = None,
    ):
        motion_dir_path = Path(motion_dir).expanduser().resolve()
        assert motion_dir_path.is_dir(), f"Invalid directory path: {motion_dir}"
        all_motion_paths = sorted(str(path) for path in motion_dir_path.rglob(file_glob) if path.is_file())
        assert len(all_motion_paths) > 0, f"No motion files matched in: {motion_dir} with pattern: {file_glob}"
        # Group-aware sharding: keep every GPU seeing every group when possible.
        rank, world_size = _get_rank_world_size(shard_by=shard_by)

        def _assign_group_name(motion_path: str) -> str:
            if motion_groups is None:
                return "default"
            for group_name, folder_patterns in motion_groups.items():
                for pattern in folder_patterns:
                    if pattern in motion_path:
                        return group_name
            return "default"

        use_group_sharding = motion_groups is not None and (shard_across_gpus or max_motions is not None)
        if use_group_sharding:
            group_to_paths: dict[str, list[str]] = {}
            for motion_path in all_motion_paths:
                group_name = _assign_group_name(motion_path)
                group_to_paths.setdefault(group_name, []).append(motion_path)

            # Validate configured groups exist in the dataset.
            missing_groups = []
            for group_name in motion_groups.keys():
                if len(group_to_paths.get(group_name, [])) == 0:
                    missing_groups.append(group_name)
            if missing_groups:
                raise ValueError(
                    "No motions matched for motion_groups: "
                    f"{missing_groups}. Check motion_groups patterns or dataset layout."
                )

            nonempty_groups = [g for g, paths in group_to_paths.items() if len(paths) > 0]
            num_groups = len(nonempty_groups)

            if max_motions is not None:
                max_motions = int(max_motions)
                if max_motions < num_groups:
                    raise ValueError(
                        f"motion_dataset_load_cap={max_motions} is smaller than the number of "
                        f"non-empty groups ({num_groups}). Increase the cap to ensure every group is loaded."
                    )

                total_paths = sum(len(group_to_paths[g]) for g in nonempty_groups)
                # Start with 1 per group to guarantee coverage.
                group_caps = {g: 1 for g in nonempty_groups}
                remaining = max_motions - num_groups
                if remaining > 0 and total_paths > 0:
                    # Distribute remaining capacity proportionally by group size.
                    extras = {}
                    for g in nonempty_groups:
                        extras[g] = int(math.floor(remaining * len(group_to_paths[g]) / total_paths))
                    used = sum(extras.values())
                    leftover = remaining - used
                    # Assign leftover one by one to groups with available capacity.
                    for g in nonempty_groups:
                        if leftover <= 0:
                            break
                        extras[g] += 1
                        leftover -= 1
                    for g in nonempty_groups:
                        group_caps[g] = min(len(group_to_paths[g]), group_caps[g] + extras[g])
            else:
                group_caps = {g: None for g in nonempty_groups}

            selected_paths = []
            shard_info = {
                "total_motions": len(all_motion_paths),
                "selected_motions": 0,
                "shard_across_gpus": bool(shard_across_gpus),
                "shard_by": str(shard_by),
                "shard_seed": int(shard_seed),
                "shard_strategy": str(shard_strategy),
                "rank": int(rank),
                "world_size": int(world_size),
                "max_motions": int(max_motions) if max_motions is not None else -1,
                "group_mode": "per_group",
                "group_shards": {},
            }

            for idx, group_name in enumerate(nonempty_groups):
                group_paths = group_to_paths[group_name]
                group_cap = group_caps[group_name]
                # If group is smaller than world size, avoid empty ranks by disabling sharding for this group.
                group_shard_across = shard_across_gpus
                if group_shard_across and world_size > 1 and len(group_paths) < world_size:
                    group_shard_across = False

                group_selected, group_info = _select_motion_paths_for_rank(
                    group_paths,
                    max_motions=group_cap,
                    shard_across_gpus=group_shard_across,
                    shard_by=shard_by,
                    shard_seed=int(shard_seed) + (idx + 1) * 10007,
                    shard_strategy=shard_strategy,
                )
                shard_info["group_shards"][group_name] = group_info
                selected_paths.extend(group_selected)

            shard_info["selected_motions"] = len(selected_paths)
        else:
            selected_paths, shard_info = _select_motion_paths_for_rank(
                all_motion_paths,
                max_motions=max_motions,
                shard_across_gpus=shard_across_gpus,
                shard_by=shard_by,
                shard_seed=shard_seed,
                shard_strategy=shard_strategy,
            )
        # Expose for debugging/analysis
        self.motion_paths_all = all_motion_paths
        self.motion_paths = selected_paths
        self.shard_info = shard_info

        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device) if storage_device is not None else self.device

        body_idx_tensor = torch.as_tensor(body_indexes, dtype=torch.long, device="cpu")
        if body_idx_tensor.ndim != 1:
            raise ValueError("body_indexes must be a 1D sequence of indices.")
        body_idx_np = body_idx_tensor.cpu().numpy()

        joint_pos_list: list[torch.Tensor] = []
        joint_vel_list: list[torch.Tensor] = []
        body_pos_list: list[torch.Tensor] = []
        body_quat_list: list[torch.Tensor] = []
        body_lin_vel_list: list[torch.Tensor] = []
        body_ang_vel_list: list[torch.Tensor] = []
        # Dedicated root channel: the articulation-root body sliced from the FULL npz body axis
        # (independent of body_idx_np), so base reset is correct even when the root (pelvis) is
        # not a tracked keypoint. Same [T_total, ...] layout as the body_* tensors → gathered by
        # the same global-index path.
        root_idx = int(root_body_index)
        root_pos_list: list[torch.Tensor] = []
        root_quat_list: list[torch.Tensor] = []
        root_lin_vel_list: list[torch.Tensor] = []
        root_ang_vel_list: list[torch.Tensor] = []
        lengths: list[int] = []
        fps_list: list[float] = []

        for motion_path in self.motion_paths:
            with np.load(motion_path) as data:
                fps_value = float(np.asarray(data["fps"]).reshape(-1)[0])
                fps_list.append(fps_value)

                joint_pos_tensor = torch.from_numpy(np.asarray(data["joint_pos"], dtype=np.float32)).to(
                    self.storage_device
                )
                joint_vel_tensor = torch.from_numpy(np.asarray(data["joint_vel"], dtype=np.float32)).to(
                    self.storage_device
                )

                body_pos_tensor = torch.from_numpy(
                    np.asarray(data["body_pos_w"], dtype=np.float32)[:, body_idx_np, :]
                ).to(self.storage_device)
                body_quat_tensor = torch.from_numpy(
                    np.asarray(data["body_quat_w"], dtype=np.float32)[:, body_idx_np, :]
                ).to(self.storage_device)
                body_lin_vel_tensor = torch.from_numpy(
                    np.asarray(data["body_lin_vel_w"], dtype=np.float32)[:, body_idx_np, :]
                ).to(self.storage_device)
                body_ang_vel_tensor = torch.from_numpy(
                    np.asarray(data["body_ang_vel_w"], dtype=np.float32)[:, body_idx_np, :]
                ).to(self.storage_device)

                root_pos_list.append(
                    torch.from_numpy(np.asarray(data["body_pos_w"], dtype=np.float32)[:, root_idx, :]).to(
                        self.storage_device
                    )
                )
                root_quat_list.append(
                    torch.from_numpy(np.asarray(data["body_quat_w"], dtype=np.float32)[:, root_idx, :]).to(
                        self.storage_device
                    )
                )
                root_lin_vel_list.append(
                    torch.from_numpy(np.asarray(data["body_lin_vel_w"], dtype=np.float32)[:, root_idx, :]).to(
                        self.storage_device
                    )
                )
                root_ang_vel_list.append(
                    torch.from_numpy(np.asarray(data["body_ang_vel_w"], dtype=np.float32)[:, root_idx, :]).to(
                        self.storage_device
                    )
                )

                joint_pos_list.append(joint_pos_tensor)
                joint_vel_list.append(joint_vel_tensor)
                body_pos_list.append(body_pos_tensor)
                body_quat_list.append(body_quat_tensor)
                body_lin_vel_list.append(body_lin_vel_tensor)
                body_ang_vel_list.append(body_ang_vel_tensor)
                lengths.append(joint_pos_tensor.shape[0])

        self.joint_pos = torch.cat(joint_pos_list, dim=0)
        self.joint_vel = torch.cat(joint_vel_list, dim=0)
        self.body_pos_w = torch.cat(body_pos_list, dim=0)
        self.body_quat_w = torch.cat(body_quat_list, dim=0)
        self.body_lin_vel_w = torch.cat(body_lin_vel_list, dim=0)
        self.body_ang_vel_w = torch.cat(body_ang_vel_list, dim=0)
        self.root_pos_w = torch.cat(root_pos_list, dim=0)
        self.root_quat_w = torch.cat(root_quat_list, dim=0)
        self.root_lin_vel_w = torch.cat(root_lin_vel_list, dim=0)
        self.root_ang_vel_w = torch.cat(root_ang_vel_list, dim=0)

        # Pin memory only when tensors are on CPU (GPU tensors cannot be pinned).
        if self.storage_device.type == "cpu":
            self.joint_pos = self.joint_pos.pin_memory()
            self.joint_vel = self.joint_vel.pin_memory()
            self.body_pos_w = self.body_pos_w.pin_memory()
            self.body_quat_w = self.body_quat_w.pin_memory()
            self.body_lin_vel_w = self.body_lin_vel_w.pin_memory()
            self.body_ang_vel_w = self.body_ang_vel_w.pin_memory()
            self.root_pos_w = self.root_pos_w.pin_memory()
            self.root_quat_w = self.root_quat_w.pin_memory()
            self.root_lin_vel_w = self.root_lin_vel_w.pin_memory()
            self.root_ang_vel_w = self.root_ang_vel_w.pin_memory()

        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.motion_lengths = lengths_tensor
        self.motion_offsets = torch.cumsum(lengths_tensor, dim=0) - lengths_tensor
        self.motion_fps = torch.tensor(fps_list, dtype=torch.float32, device=self.device)
        self.total_frames = int(lengths_tensor.sum().item())

        # Build motion-to-group mapping for multi-teacher support
        self.motion_to_group: dict[int, str] = {}

        if motion_groups is not None:
            # Map each motion to its group based on path patterns
            for motion_idx, motion_path in enumerate(self.motion_paths):
                group_assigned = False
                # Check each group's folder patterns
                for group_name, folder_patterns in motion_groups.items():
                    for pattern in folder_patterns:
                        if pattern in motion_path:
                            self.motion_to_group[motion_idx] = group_name
                            group_assigned = True
                            break
                    if group_assigned:
                        break

                # If no match, assign to default group
                if not group_assigned:
                    self.motion_to_group[motion_idx] = "default"
        else:
            # If no groups specified, all motions belong to default group
            for motion_idx in range(len(self.motion_paths)):
                self.motion_to_group[motion_idx] = "default"

        # Print motion group distribution
        from collections import Counter
        group_counts = Counter(self.motion_to_group.values())
        print(f"[MultiMotionLoader] Motion group distribution:")
        for group_name in sorted(group_counts.keys()):
            count = group_counts[group_name]
            print(f"  - {group_name}: {count} motions")

    def __len__(self) -> int:
        return len(self.motion_paths)

    def motion_length(self, motion_index: int) -> int:
        return int(self.motion_lengths[motion_index].item())

    def compute_global_indices(self, motion_indices: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        lengths = self.motion_lengths[motion_indices]
        max_valid = torch.clamp(lengths - 1, min=0)
        # Clamp on both sides: upper bound prevents leak past clip end; lower bound (added for the
        # lookback window terms) prevents negative frame indices from indexing into the previous
        # motion's tail in the concatenated buffer. Both edges hold-first/hold-last, which the
        # encoder treats as "reference held steady at the boundary."
        clamped = torch.clamp(frame_indices, min=0)
        clamped = torch.minimum(clamped, max_valid)
        offsets = self.motion_offsets[motion_indices]
        return offsets + clamped

    def gather_from_global(
        self, attr: str, global_indices: torch.Tensor, out_device: torch.device | str
    ) -> torch.Tensor:
        source_tensor = getattr(self, attr)
        if not isinstance(out_device, torch.device):
            out_device = torch.device(out_device)
        if global_indices.device != source_tensor.device:
            local_indices = global_indices.to(source_tensor.device)
        else:
            local_indices = global_indices
        gathered = source_tensor.index_select(0, local_indices)
        if gathered.device != out_device:
            gathered = gathered.to(out_device, non_blocking=True)
        return gathered

    def gather(
        self,
        attr: str,
        motion_indices: torch.Tensor,
        frame_indices: torch.Tensor,
        out_device: torch.device | str,
    ) -> torch.Tensor:
        global_indices = self.compute_global_indices(motion_indices, frame_indices)
        return self.gather_from_global(attr, global_indices, out_device)


class MultiMotionCommand(CommandTerm):
    """Command term that supports loading and training with multiple motions from a folder.
    
    - Samples motions across files according to difficulty-based and novelty-based sampling.
    - Within each motion, time-step sampling is adaptive based on failure counts (same as single-motion logic).
    - Periodically remaps environment ids to a fresh set of motions so all motions get chances to be sampled.
    """

    cfg: "MultiMotionCommandCfg"

    def __init__(self, cfg: "MultiMotionCommandCfg", env: "ManagerBasedRLEnv"):
        # Set before super(): CommandTerm.__init__ calls set_debug_vis -> _set_debug_vis_impl.
        self._video_debug_body_indices = _resolve_video_debug_body_indices(cfg.body_names, cfg)
        self._video_debug_goal_world = bool(getattr(cfg, "video_debug_vis_goal_bodies_world_frame", False))
        self._video_debug_hide_anchor = bool(getattr(cfg, "video_debug_vis_hide_anchor", False))
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        body_index_array = self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0]
        body_index_tensor = torch.as_tensor(body_index_array, dtype=torch.long)
        self.body_indexes = body_index_tensor.to(self.device)
        body_index_list = body_index_tensor.cpu().tolist()
        # See MotionCommand.__init__: reset the base from the robot's articulation-root reference
        # via a dedicated full-axis root channel — independent of the tracked keypoint set, so a
        # pelvis-less KP5 set still resets the base correctly.
        self._root_full_index = _resolve_root_full_index(self.robot)

        preload_device = (
            torch.device(self.cfg.motion_preload_device)
            if self.cfg.motion_preload_device is not None
            else torch.device(self.device)
        )

        keys = ["x","y","z","roll","pitch","yaw"]
        self._pose_ranges = torch.tensor([self.cfg.pose_range.get(k,(0.,0.)) for k in keys],
                                        device=self.device, dtype=torch.float32)
        self._vel_ranges  = torch.tensor([self.cfg.velocity_range.get(k,(0.,0.)) for k in keys],
                                        device=self.device, dtype=torch.float32)

        # Optional: cap number of motions loaded per process (useful with large datasets).
        # Auto cap is only enabled when sharding is enabled and user doesn't specify a cap.
        load_cap = self.cfg.motion_dataset_load_cap
        if load_cap is None and self.cfg.motion_dataset_shard_across_gpus:
            candidates = [int(self.num_envs)]
            k_cfg = getattr(self.cfg, "max_active_motions", None)
            if k_cfg is not None:
                candidates.append(int(k_cfg))
            load_cap = int(min(candidates)) if len(candidates) > 0 else None

        motion_loader = MultiMotionLoader(
            self.cfg.motion,
            body_index_list,
            device=self.device,
            file_glob=self.cfg.file_glob,
            storage_device=preload_device,
            root_body_index=self._root_full_index,
            max_motions=load_cap,
            shard_across_gpus=self.cfg.motion_dataset_shard_across_gpus,
            shard_by=self.cfg.motion_dataset_shard_by,
            shard_seed=self.cfg.motion_dataset_shard_seed,
            shard_strategy=self.cfg.motion_dataset_shard_strategy,
            motion_groups=self.cfg.motion_groups,
        )

        self._rebind_motion_loader(
            motion_loader,
            cfg_motion=self.cfg.motion,
            verbose=True,
            reset_global_sampling_state=True,
            log_shard_to_wandb=True,
            init_pulse_vae_ctx=True,
        )
        # Do not resample here: termination manager may not be ready during managers' construction.
        # Time steps start at zero; sampling and writes happen in _update_command().

        # MUSE goal-block masking: per-env Bernoulli flag, resampled each step in _update_command.
        # ``p_mask`` is mutated by a curriculum (e.g. ``goal_mask_probability_curriculum``); 0.0 disables.
        self.p_mask: float = 0.0
        self.goal_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_goal_mask(self) -> None:
        """Resample the per-env goal-block mask from Bernoulli(``p_mask``). 0.0 → all unmasked."""
        if self.p_mask <= 0.0:
            self.goal_mask.zero_()
            return
        if self.p_mask >= 1.0:
            self.goal_mask.fill_(True)
            return
        probs = torch.full((self.num_envs,), float(self.p_mask), device=self.device)
        self.goal_mask = torch.bernoulli(probs).to(torch.bool)

    def _motion_metric_keys_to_clear(self) -> list[str]:
        keys: list[str] = []
        for k in list(self.metrics.keys()):
            if k.startswith(("error_", "sampling_", "motion_sampling")):
                keys.append(k)
            elif k.startswith("error_joint_pos_") or k.startswith("error_joint_vel_"):
                keys.append(k)
            elif k.startswith("error_keypoint_"):
                keys.append(k)
        return keys

    def _allocate_standard_tracking_metrics(self) -> None:
        for k in self._motion_metric_keys_to_clear():
            del self.metrics[k]
        n = self.num_envs
        dev = self.device
        self.metrics["error_anchor_pos"] = torch.zeros(n, device=dev)
        self.metrics["error_anchor_rot"] = torch.zeros(n, device=dev)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(n, device=dev)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(n, device=dev)
        self.metrics["error_body_pos"] = torch.zeros(n, device=dev)
        self.metrics["error_body_rot"] = torch.zeros(n, device=dev)
        self.metrics["error_body_lin_vel"] = torch.zeros(n, device=dev)
        self.metrics["error_body_ang_vel"] = torch.zeros(n, device=dev)
        self.metrics["error_joint_pos"] = torch.zeros(n, device=dev)
        self.metrics["error_joint_vel"] = torch.zeros(n, device=dev)
        self.metrics["sampling_entropy"] = torch.zeros(n, device=dev)
        self.metrics["sampling_top1_prob"] = torch.zeros(n, device=dev)
        self.metrics["sampling_top1_bin"] = torch.zeros(n, device=dev)
        self.metrics["motion_sampling_prob_mean"] = torch.zeros(n, device=dev)
        self.metrics["motion_sampling_prob_std"] = torch.zeros(n, device=dev)
        self.metrics["motion_sampling_prob_min"] = torch.zeros(n, device=dev)
        self.metrics["motion_sampling_prob_max"] = torch.zeros(n, device=dev)
        self.metrics["motion_sampling_prob_entropy"] = torch.zeros(n, device=dev)

        if self._dense_joint_error_metrics:
            if not self._joint_metric_keys:
                self._joint_metric_keys = self._build_dense_joint_metric_keys()
            self._init_dense_joint_metrics()

        if self._pulse_vae_ee_ctx is not None:
            for _bid, pk, vk, _ in self._pulse_vae_ee_ctx.entries:
                self.metrics[pk] = torch.full((n,), float("nan"), device=dev)
                self.metrics[vk] = torch.full((n,), float("nan"), device=dev)

        self._pulse_vae_env_mode_idx = torch.full((n,), -1, dtype=torch.long, device=dev)

    def _rebind_motion_loader(
        self,
        motion_loader: MultiMotionLoader,
        *,
        cfg_motion: str,
        verbose: bool,
        reset_global_sampling_state: bool,
        log_shard_to_wandb: bool,
        init_pulse_vae_ctx: bool,
    ) -> None:
        """Point this command at a new :class:`MultiMotionLoader` and rebuild motion-sized state."""

        self.motion_dir_loader = motion_loader
        self.cfg.motion = cfg_motion
        self.num_motions_total = len(self.motion_dir_loader)

        self._motion_dataset_shard_info = getattr(self.motion_dir_loader, "shard_info", {}) or {}
        if verbose and getattr(self.cfg, "motion_dataset_log_shard_info", False):
            print(f"[MultiMotionLoader] shard_info={self._motion_dataset_shard_info}")
        if log_shard_to_wandb:
            _maybe_log_motion_shard_to_wandb_summary(self._motion_dataset_shard_info, self.cfg)

        self.sim_dt = self._env.cfg.decimation * self._env.cfg.sim.dt
        self.frames_per_bin = max(1, int(round(1.0 / self.sim_dt)))

        self.motion_lengths = self.motion_dir_loader.motion_lengths.to(self.device)
        self.motion_lengths_minus_one = (self.motion_lengths - 1).clamp(min=0)
        self.motion_length_denominator = self.motion_lengths.clamp(min=1)

        self.motion_bin_counts = (self.motion_lengths // self.frames_per_bin) + 1
        self.motion_bin_counts_float = self.motion_bin_counts.to(torch.float32)
        self.max_bin_count = int(self.motion_bin_counts.max().item())
        self.bin_index_range = torch.arange(self.max_bin_count, device=self.device)
        self.motion_bin_mask = self.bin_index_range.unsqueeze(0) < self.motion_bin_counts.unsqueeze(1)
        self.motion_end_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.bin_failed_count = torch.zeros(
            self.num_motions_total, self.max_bin_count, dtype=torch.float32, device=self.device
        )
        self.current_bin_failed = torch.zeros_like(self.bin_failed_count)

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env_motion_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env_motion_groups = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        unique_groups = set(self.motion_dir_loader.motion_to_group.values())
        self.group_name_to_idx = {}
        self.idx_to_group_name = {}
        for idx, group_name in enumerate(sorted(unique_groups)):
            self.group_name_to_idx[group_name] = idx
            self.idx_to_group_name[idx] = group_name

        if verbose:
            print(f"[MultiMotionCommand] Registered motion groups: {list(self.group_name_to_idx.keys())}")

        self.group_to_motions = {group_name: [] for group_name in self.group_name_to_idx.keys()}
        for motion_idx, group_name in self.motion_dir_loader.motion_to_group.items():
            self.group_to_motions[group_name].append(motion_idx)

        self.group_to_motions_tensor = {}
        for group_name, motion_list in self.group_to_motions.items():
            self.group_to_motions_tensor[group_name] = torch.tensor(motion_list, dtype=torch.long, device=self.device)
            if verbose:
                print(f"[MultiMotionCommand]   - Group '{group_name}': {len(motion_list)} motions")

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(self.cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(self.cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        if self.cfg.resample_motions_every_s <= 0:
            self._resample_motions_every_steps = 0
        else:
            steps = max(1, int(round(self.cfg.resample_motions_every_s / self.sim_dt)))
            self._resample_motions_every_steps = steps

        if reset_global_sampling_state:
            self._global_sim_step = 0
            self._remap_version = 0
            self._env_remap_version = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        if init_pulse_vae_ctx:
            self._dense_joint_error_metrics = bool(getattr(self.cfg, "enable_dense_joint_error_metrics", False))
            self._joint_metric_keys = []
            if self._dense_joint_error_metrics:
                self._joint_metric_keys = self._build_dense_joint_metric_keys()
            self._pulse_vae_ee_ctx = None

        prob_init = 1.0 / max(self.num_motions_total, 1)
        self.motion_sampling_probs = torch.full(
            (self.num_motions_total,), prob_init, dtype=torch.float32, device=self.device
        )
        self.motion_sample_counts = torch.zeros(self.num_motions_total, dtype=torch.float32, device=self.device)
        self.motion_assigned_counts = torch.zeros(self.num_motions_total, dtype=torch.float32, device=self.device)
        self.motion_fail_counts = torch.zeros(self.num_motions_total, dtype=torch.float32, device=self.device)

        self._allocate_standard_tracking_metrics()

        all_envs = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._assign_motions(all_envs)

    def _snapshot_motion_dataset_for_holdout(self) -> dict:
        snap: dict = {
            "cfg_motion": str(self.cfg.motion),
            "motion_dir_loader": self.motion_dir_loader,
            "num_motions_total": int(self.num_motions_total),
            "_motion_dataset_shard_info": copy.deepcopy(self._motion_dataset_shard_info),
            "motion_lengths": self.motion_lengths.clone(),
            "motion_lengths_minus_one": self.motion_lengths_minus_one.clone(),
            "motion_length_denominator": self.motion_length_denominator.clone(),
            "motion_bin_counts": self.motion_bin_counts.clone(),
            "motion_bin_counts_float": self.motion_bin_counts_float.clone(),
            "max_bin_count": int(self.max_bin_count),
            "bin_index_range": self.bin_index_range.clone(),
            "motion_bin_mask": self.motion_bin_mask.clone(),
            "motion_end_buf": self.motion_end_buf.clone(),
            "bin_failed_count": self.bin_failed_count.clone(),
            "current_bin_failed": self.current_bin_failed.clone(),
            "time_steps": self.time_steps.clone(),
            "env_motion_indices": self.env_motion_indices.clone(),
            "env_motion_groups": self.env_motion_groups.clone(),
            "group_name_to_idx": copy.deepcopy(self.group_name_to_idx),
            "idx_to_group_name": copy.deepcopy(self.idx_to_group_name),
            "group_to_motions": {k: list(v) for k, v in self.group_to_motions.items()},
            "group_to_motions_tensor": {k: v.clone() for k, v in self.group_to_motions_tensor.items()},
            "motion_sampling_probs": self.motion_sampling_probs.clone(),
            "motion_sample_counts": self.motion_sample_counts.clone(),
            "motion_assigned_counts": self.motion_assigned_counts.clone(),
            "motion_fail_counts": self.motion_fail_counts.clone(),
            "metrics": {k: v.clone() for k, v in self.metrics.items() if torch.is_tensor(v)},
            "_global_sim_step": int(self._global_sim_step),
            "_remap_version": int(self._remap_version),
            "_env_remap_version": self._env_remap_version.clone(),
            "_pulse_vae_env_mode_idx": self._pulse_vae_env_mode_idx.clone(),
            "body_pos_relative_w": self.body_pos_relative_w.clone(),
            "body_quat_relative_w": self.body_quat_relative_w.clone(),
            "kernel": self.kernel.clone(),
            "_resample_motions_every_steps": int(self._resample_motions_every_steps),
            "sim_dt": float(self.sim_dt),
            "frames_per_bin": int(self.frames_per_bin),
        }
        return snap

    def _restore_motion_dataset_from_holdout_snapshot(self, snap: dict) -> None:
        self.cfg.motion = snap["cfg_motion"]
        self.motion_dir_loader = snap["motion_dir_loader"]
        self.num_motions_total = snap["num_motions_total"]
        self._motion_dataset_shard_info = copy.deepcopy(snap["_motion_dataset_shard_info"])

        self.motion_lengths = snap["motion_lengths"].to(self.device)
        self.motion_lengths_minus_one = snap["motion_lengths_minus_one"].to(self.device)
        self.motion_length_denominator = snap["motion_length_denominator"].to(self.device)
        self.motion_bin_counts = snap["motion_bin_counts"].to(self.device)
        self.motion_bin_counts_float = snap["motion_bin_counts_float"].to(self.device)
        self.max_bin_count = snap["max_bin_count"]
        self.bin_index_range = snap["bin_index_range"].to(self.device)
        self.motion_bin_mask = snap["motion_bin_mask"].to(self.device)
        self.motion_end_buf = snap["motion_end_buf"].to(self.device)
        self.bin_failed_count = snap["bin_failed_count"].to(self.device)
        self.current_bin_failed = snap["current_bin_failed"].to(self.device)
        self.time_steps = snap["time_steps"].to(self.device)
        self.env_motion_indices = snap["env_motion_indices"].to(self.device)
        self.env_motion_groups = snap["env_motion_groups"].to(self.device)

        self.group_name_to_idx = copy.deepcopy(snap["group_name_to_idx"])
        self.idx_to_group_name = copy.deepcopy(snap["idx_to_group_name"])
        self.group_to_motions = {k: list(v) for k, v in snap["group_to_motions"].items()}
        self.group_to_motions_tensor = {k: v.to(self.device) for k, v in snap["group_to_motions_tensor"].items()}

        self.motion_sampling_probs = snap["motion_sampling_probs"].to(self.device)
        self.motion_sample_counts = snap["motion_sample_counts"].to(self.device)
        self.motion_assigned_counts = snap["motion_assigned_counts"].to(self.device)
        self.motion_fail_counts = snap["motion_fail_counts"].to(self.device)

        self._global_sim_step = snap["_global_sim_step"]
        self._remap_version = snap["_remap_version"]
        self._env_remap_version = snap["_env_remap_version"].to(self.device)
        self._pulse_vae_env_mode_idx = snap["_pulse_vae_env_mode_idx"].to(self.device)

        self.body_pos_relative_w = snap["body_pos_relative_w"].to(self.device)
        self.body_quat_relative_w = snap["body_quat_relative_w"].to(self.device)
        self.kernel = snap["kernel"].to(self.device)
        self._resample_motions_every_steps = snap["_resample_motions_every_steps"]
        self.sim_dt = snap["sim_dt"]
        self.frames_per_bin = snap["frames_per_bin"]

        for k, v in snap["metrics"].items():
            t = v.to(self.device)
            if k in self.metrics and self.metrics[k].shape == t.shape and self.metrics[k].dtype == t.dtype:
                self.metrics[k].copy_(t)
            else:
                self.metrics[k] = t.clone()

    def push_holdout_motion_dataset(self, motion_dir: str) -> None:
        """Swap to a holdout motion directory for eval; call :meth:`pop_holdout_motion_dataset` after."""

        if getattr(self, "_holdout_motion_backup", None) is not None:
            raise RuntimeError("Nested push_holdout_motion_dataset is not supported.")

        body_index_list = self.body_indexes.cpu().tolist()
        preload_device = (
            torch.device(self.cfg.motion_preload_device)
            if self.cfg.motion_preload_device is not None
            else torch.device(self.device)
        )
        load_cap = self.cfg.motion_dataset_load_cap
        if load_cap is None and self.cfg.motion_dataset_shard_across_gpus:
            candidates = [int(self.num_envs)]
            k_cfg = getattr(self.cfg, "max_active_motions", None)
            if k_cfg is not None:
                candidates.append(int(k_cfg))
            load_cap = int(min(candidates)) if len(candidates) > 0 else None

        self._holdout_motion_backup = self._snapshot_motion_dataset_for_holdout()

        try:
            holdout_loader = MultiMotionLoader(
                motion_dir,
                body_index_list,
                device=self.device,
                file_glob=self.cfg.file_glob,
                storage_device=preload_device,
                root_body_index=self._root_full_index,
                max_motions=load_cap,
                shard_across_gpus=self.cfg.motion_dataset_shard_across_gpus,
                shard_by=self.cfg.motion_dataset_shard_by,
                shard_seed=self.cfg.motion_dataset_shard_seed,
                shard_strategy=self.cfg.motion_dataset_shard_strategy,
                motion_groups=self.cfg.motion_groups,
            )
            self._rebind_motion_loader(
                holdout_loader,
                cfg_motion=motion_dir,
                verbose=False,
                reset_global_sampling_state=True,
                log_shard_to_wandb=False,
                init_pulse_vae_ctx=False,
            )
        except Exception:
            self._restore_motion_dataset_from_holdout_snapshot(self._holdout_motion_backup)
            self._holdout_motion_backup = None
            raise

    def pop_holdout_motion_dataset(self) -> None:
        """Restore training motion data after :meth:`push_holdout_motion_dataset`."""

        snap = getattr(self, "_holdout_motion_backup", None)
        if snap is None:
            raise RuntimeError("pop_holdout_motion_dataset without push_holdout_motion_dataset.")
        self._restore_motion_dataset_from_holdout_snapshot(snap)
        self._holdout_motion_backup = None

    def _build_dense_joint_metric_keys(self) -> list[tuple[str, str]]:
        keys: list[tuple[str, str]] = []
        for joint_name in self.robot.joint_names:
            safe = _sanitize_joint_metric_name(joint_name)
            keys.append((f"error_joint_pos_{safe}", f"error_joint_vel_{safe}"))
        return keys

    def _init_dense_joint_metrics(self) -> None:
        for pos_key, vel_key in self._joint_metric_keys:
            self.metrics[pos_key] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[vel_key] = torch.zeros(self.num_envs, device=self.device)

    def configure_pulse_vae_ee_keypoint_metrics(self, mode_spec: dict[str, list[str]] | None) -> None:
        """Enable partial-mask EE keypoint metrics for mask modes present in ``mode_spec`` (runner-only)."""
        self._pulse_vae_ee_ctx = (
            _make_pulse_vae_ee_metric_context(mode_spec, self.cfg.body_names, self.device) if mode_spec else None
        )
        if self._pulse_vae_ee_ctx is None:
            return
        for _bid, pk, vk, _ in self._pulse_vae_ee_ctx.entries:
            self.metrics[pk] = torch.full((self.num_envs,), float("nan"), device=self.device)
            self.metrics[vk] = torch.full((self.num_envs,), float("nan"), device=self.device)
        print(
            f"[{self.__class__.__name__}] partial-mask EE keypoint metrics: "
            f"{[e[1] for e in self._pulse_vae_ee_ctx.entries]}"
        )

    def set_pulse_vae_ee_env_mode_indices(self, mode_indices: torch.Tensor | None) -> None:
        """Per-env mask row index from the policy's ``sample_and_set_mask`` (same order as ``mode_spec``)."""
        if self._pulse_vae_ee_ctx is None:
            return
        if mode_indices is None:
            self._pulse_vae_env_mode_idx.fill_(-1)
            return
        self._pulse_vae_env_mode_idx.copy_(
            mode_indices.to(device=self.device, dtype=torch.long).view_as(self._pulse_vae_env_mode_idx)
        )

    # ------------- properties (gathered across envs/motions) -------------
    def _gather_future_by_motion(self, getter: str, horizon: int) -> torch.Tensor:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        motion_indices = self.env_motion_indices
        base_indices = self.time_steps.unsqueeze(1)
        offsets = torch.arange(horizon, device=self.device, dtype=torch.long).view(1, -1)
        frame_indices = base_indices + offsets
        max_valid = self.motion_lengths_minus_one[motion_indices].unsqueeze(1)
        frame_indices = torch.minimum(frame_indices, max_valid)

        flat_motion = motion_indices.unsqueeze(1).expand_as(frame_indices).reshape(-1)
        flat_frames = frame_indices.reshape(-1)
        gathered = self.motion_dir_loader.gather(getter, flat_motion, flat_frames, out_device=self.device)
        new_shape = (self.num_envs, horizon) + gathered.shape[1:]
        return gathered.view(new_shape)

    @property
    def command(self) -> torch.Tensor:
        horizon = self.cfg.motion_horizon
        joint_pos_seq = self._gather_future_by_motion("joint_pos", horizon)
        if self.cfg.command_velocity:
            joint_vel_seq = self._gather_future_by_motion("joint_vel", horizon)
            command_seq = torch.cat([joint_pos_seq, joint_vel_seq], dim=-1)
        else:
            command_seq = joint_pos_seq
        return command_seq.reshape(self.num_envs, -1)

    def _gather_by_motion(self, getter: str) -> torch.Tensor:
        return self.motion_dir_loader.gather(
            getter, self.env_motion_indices, self.time_steps, out_device=self.device
        )
    
    def _gather_by_motion_for_envs(self, getter: str, env_ids: torch.Tensor) -> torch.Tensor:
        motion_idx = self.env_motion_indices[env_ids]
        frame_idx = self.time_steps[env_ids]
        return self.motion_dir_loader.gather(getter, motion_idx, frame_idx, out_device=self.device)

    def _update_sampling_prob_metrics(self):
        probs = self._compute_motion_sampling_probs()
        if probs.numel() == 0:
            zero = 0.0
            self.metrics["motion_sampling_prob_mean"].fill_(zero)
            self.metrics["motion_sampling_prob_std"].fill_(zero)
            self.metrics["motion_sampling_prob_min"].fill_(zero)
            self.metrics["motion_sampling_prob_max"].fill_(zero)
            self.metrics["motion_sampling_prob_entropy"].fill_(zero)
            return

        mean_val = probs.mean().item()
        std_val = probs.std(unbiased=False).item()
        min_val = probs.min().item()
        max_val = probs.max().item()
        entropy = -(probs * (probs + 1e-12).log()).sum().item()
        norm_entropy = entropy / max(math.log(max(probs.numel(), 1)), 1e-12)

        self.metrics["motion_sampling_prob_mean"].fill_(mean_val)
        self.metrics["motion_sampling_prob_std"].fill_(std_val)
        self.metrics["motion_sampling_prob_min"].fill_(min_val)
        self.metrics["motion_sampling_prob_max"].fill_(max_val)
        self.metrics["motion_sampling_prob_entropy"].fill_(norm_entropy)

    def _motion_sampling_progress(self) -> float:
        """Return ramp progress in [0, 1] for motion-level sampling weights.

        Motivation: early in training, fail statistics are noisy (often everything fails), so we
        start from uniform sampling and gradually increase the configured weights.
        """

        warmup_s = float(getattr(self.cfg, "motion_sampling_warmup_s", 0.0))
        ramp_s = float(getattr(self.cfg, "motion_sampling_ramp_s", 0.0))
        warmup_steps = max(0, int(round(warmup_s / max(self.sim_dt, 1e-12))))
        ramp_steps = max(0, int(round(ramp_s / max(self.sim_dt, 1e-12))))

        if self._global_sim_step <= warmup_steps:
            return 0.0
        if ramp_steps <= 0:
            return 1.0

        x = (float(self._global_sim_step - warmup_steps)) / float(ramp_steps)
        x = max(0.0, min(1.0, x))

        schedule = str(getattr(self.cfg, "motion_sampling_schedule", "linear")).lower()
        if schedule == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * x)
        # default: linear
        return x

    def _in_motion_sampling_warmup(self) -> bool:
        """True if we are still in motion-level sampling warmup window."""
        warmup_s = float(getattr(self.cfg, "motion_sampling_warmup_s", 0.0))
        warmup_steps = max(0, int(round(warmup_s / max(self.sim_dt, 1e-12))))
        return self._global_sim_step <= warmup_steps

    def _compute_motion_sampling_probs(self) -> torch.Tensor:
        num_bins = self.num_motions_total
        if num_bins == 0:
            self.motion_sampling_probs = torch.empty(0, device=self.device)
            return self.motion_sampling_probs

        sample_counts = self.motion_sample_counts
        assigned_counts = self.motion_assigned_counts
        fail_counts = self.motion_fail_counts
        
        # fail-based difficulty
        fail_rates = torch.zeros_like(sample_counts)
        valid_mask = sample_counts > 0
        if valid_mask.any():
            fail_rates[valid_mask] = fail_counts[valid_mask] / sample_counts[valid_mask].clamp(min=1e-6)

        mean_fail = fail_rates.mean()
        beta_cap = self.cfg.cap_beta * mean_fail
        if beta_cap > 0:
            capped_rates = torch.minimum(fail_rates, beta_cap)
        else:
            capped_rates = torch.zeros_like(fail_rates)

        capped_sum = capped_rates.sum()
        if capped_sum > 0:
            prob_fail = capped_rates / capped_sum
        else:
            prob_fail = torch.zeros_like(capped_rates)

        # novelty term: prefer less-sampled motions
        novelty = 1.0 / torch.sqrt(assigned_counts + 1.0)
        if novelty.sum() > 0:
            prob_novel = novelty / novelty.sum()
        else:
            prob_novel = torch.zeros_like(novelty)
        
        # uniform term: prefer more uniform sampling
        prob_uniform = torch.full_like(prob_fail, 1.0 / max(num_bins, 1))

        # mix the terms
        progress = self._motion_sampling_progress()
        w_fail_target = float(self.cfg.weight_fail)
        w_novel_target = float(self.cfg.weight_novel)
        w_fail = progress * w_fail_target
        w_novel = progress * w_novel_target

        # keep weights well-formed
        w_sum = w_fail + w_novel
        if w_sum > 1.0:
            w_fail = w_fail / w_sum
            w_novel = w_novel / w_sum
            w_uniform = 0.0
        else:
            w_uniform = max(0.0, 1.0 - w_fail - w_novel)

        probs = w_fail * prob_fail + w_novel * prob_novel + w_uniform * prob_uniform

        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = prob_uniform
        else:
            probs = probs / probs_sum

        self.motion_sampling_probs = probs
        return probs

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._gather_by_motion("joint_pos")

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._gather_by_motion("joint_vel")

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._gather_by_motion("body_pos_w") + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._gather_by_motion("body_quat_w")

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._gather_by_motion("body_lin_vel_w")

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._gather_by_motion("body_ang_vel_w")

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        # Use body_pos_w then index anchor across bodies
        pos = self._gather_by_motion("body_pos_w")
        return pos[:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        quat = self._gather_by_motion("body_quat_w")
        return quat[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        vel = self._gather_by_motion("body_lin_vel_w")
        return vel[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        vel = self._gather_by_motion("body_ang_vel_w")
        return vel[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def ref_body_pos_robot_anchor_b(self) -> torch.Tensor:
        """Reference motion body positions in the robot-anchor frame (b-frame).

        Same convention as :func:`whole_body_tracking.tasks.tracking.mdp.observations.robot_body_pos_b`,
        but uses reference ``body_pos_w`` / ``body_quat_w`` from the motion (not simulated robot bodies).
        """
        num_bodies = len(self.cfg.body_names)
        pos_b, _ = subtract_frame_transforms(
            self.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
            self.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
            self.body_pos_w,
            self.body_quat_w,
        )
        return pos_b.view(self.num_envs, -1)

    def ref_body_lin_vel_robot_anchor_b(self) -> torch.Tensor:
        """Reference motion body linear velocities in the robot-anchor frame (same layout as ``ref_body_pos_robot_anchor_b``)."""
        num_bodies = len(self.cfg.body_names)
        v_w = self.body_lin_vel_w
        q = self.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)
        v_w_flat = v_w.reshape(-1, 3)
        q_flat = q.reshape(-1, 4)
        v_b = quat_rotate_inverse(q_flat, v_w_flat)
        return v_b.view(self.num_envs, -1)

    def _assign_motions(self, env_ids: torch.Tensor):
        """Assign motions to given env ids using difficulty-based and novelty-based sampling."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n_envs = len(env_ids)
        if n_envs == 0:
            return

        # Check if motion group sampling ratios are configured
        if self.cfg.motion_group_sampling_ratios is not None:
            # Use ratio-based sampling: split environments by group and sample from each group's pool
            self._assign_motions_with_ratios(env_ids)
            return

        probs_motion = self._compute_motion_sampling_probs()

        k_cfg = getattr(self.cfg, "max_active_motions", None)
        use_active_pool = (k_cfg is not None) and (self.num_motions_total > k_cfg)

        if use_active_pool:
            # sample a active set from the motion database (no replacement)
            K = min(k_cfg, self.num_motions_total, n_envs)
            active_motions = torch.multinomial(probs_motion, K, replacement=False)

            # using these motions to fill the envs
            # make every motion get about n_envs / K envs
            reps = n_envs // K
            rem = n_envs % K

            base = active_motions.repeat_interleave(reps)
            if rem > 0:
                extra = active_motions[torch.randperm(K, device=self.device)[:rem]]
                sampled = torch.cat([base, extra], dim=0)
            else:
                sampled = base

            # randomize the order of the envs
            perm = torch.randperm(n_envs, device=self.device)
            self.env_motion_indices[env_ids] = sampled[perm]

            # Update motion groups for multi-teacher support
            for i, env_id in enumerate(env_ids):
                motion_idx = self.env_motion_indices[env_id].item()
                group_name = self.motion_dir_loader.motion_to_group.get(motion_idx, "default")
                group_idx = self.group_name_to_idx[group_name]
                self.env_motion_groups[env_id] = group_idx

            # update the motion sample counts
            unique_motions, counts = torch.unique(sampled, return_counts=True)
            self.motion_assigned_counts.index_add_(
                0, unique_motions, counts.to(self.motion_assigned_counts.dtype)
            )

        else:
            if self.cfg.unique_per_batch and self.num_motions_total >= n_envs:
                sampled = torch.multinomial(probs_motion, n_envs, replacement=False)
            else:
                sampled = torch.multinomial(probs_motion, n_envs, replacement=True)

            self.env_motion_indices[env_ids] = sampled

            # Update motion groups for multi-teacher support
            for i, env_id in enumerate(env_ids):
                motion_idx = self.env_motion_indices[env_id].item()
                group_name = self.motion_dir_loader.motion_to_group.get(motion_idx, "default")
                group_idx = self.group_name_to_idx[group_name]
                self.env_motion_groups[env_id] = group_idx

            unique_motions, counts = torch.unique(sampled, return_counts=True)
            self.motion_assigned_counts.index_add_(
                0, unique_motions, counts.to(self.motion_assigned_counts.dtype)
            )

        self._update_sampling_prob_metrics()

    def _assign_motions_with_ratios(self, env_ids: torch.Tensor):
        """Assign motions to environments using motion group sampling ratios.

        This method splits environments according to configured sampling ratios,
        then samples motions from each group's pool separately while maintaining
        difficulty-based and novelty-based sampling within each group.

        Args:
            env_ids: Tensor of environment IDs to assign motions to
        """
        n_envs = len(env_ids)
        ratios = self.cfg.motion_group_sampling_ratios

        # Validate ratios
        ratio_sum = sum(ratios.values())
        if abs(ratio_sum - 1.0) > 1e-5:
            print(f"[MultiMotionCommand] Warning: motion_group_sampling_ratios sum to {ratio_sum:.4f}, not 1.0. Normalizing...")
            ratios = {k: v / ratio_sum for k, v in ratios.items()}

        # Check that all ratio groups exist in motion groups
        for group_name in ratios.keys():
            if group_name not in self.group_name_to_idx:
                raise ValueError(f"Sampling ratio specified for group '{group_name}' but this group doesn't exist. "
                               f"Available groups: {list(self.group_name_to_idx.keys())}")

        # Compute global motion sampling probabilities (for difficulty/novelty weighting)
        probs_motion_global = self._compute_motion_sampling_probs()

        # Split environments by group according to ratios
        group_env_splits = {}
        start_idx = 0
        for group_name, ratio in sorted(ratios.items()):
            n_envs_group = int(round(n_envs * ratio))
            # Ensure we don't exceed total environments
            if group_name == list(sorted(ratios.keys()))[-1]:  # Last group gets remainder
                n_envs_group = n_envs - start_idx

            if n_envs_group > 0:
                group_env_splits[group_name] = env_ids[start_idx:start_idx + n_envs_group]
                start_idx += n_envs_group

        print(f"[MultiMotionCommand] Assigning {n_envs} environments with ratios: "
              f"{', '.join([f'{k}={len(v)}/{n_envs}' for k, v in group_env_splits.items()])}")

        # Sample motions for each group
        for group_name, group_env_ids in group_env_splits.items():
            self._assign_motions_for_group(group_name, group_env_ids, probs_motion_global)

        self._update_sampling_prob_metrics()

    def _assign_motions_for_group(self, group_name: str, env_ids: torch.Tensor, probs_motion_global: torch.Tensor):
        """Assign motions from a specific group to given environments.

        Args:
            group_name: Name of the motion group to sample from
            env_ids: Tensor of environment IDs to assign motions to
            probs_motion_global: Global motion sampling probabilities (for all motions)
        """
        n_envs = len(env_ids)
        if n_envs == 0:
            return

        # Get motion indices for this group
        group_motion_indices = self.group_to_motions_tensor[group_name]
        n_motions_in_group = len(group_motion_indices)

        if n_motions_in_group == 0:
            raise ValueError(f"Group '{group_name}' has no motions!")

        # Extract and renormalize probabilities for this group's motions
        probs_group = probs_motion_global[group_motion_indices]
        probs_group = probs_group / probs_group.sum()

        # Sample motions from this group
        k_cfg = getattr(self.cfg, "max_active_motions", None)
        use_active_pool = (k_cfg is not None) and (self.num_motions_total > k_cfg)

        if use_active_pool:
            # Sample active motions from this group's pool
            K = min(k_cfg, n_motions_in_group, n_envs)
            active_motion_indices_in_group = torch.multinomial(probs_group, K, replacement=False)
            active_motions = group_motion_indices[active_motion_indices_in_group]

            # Distribute these motions across environments
            reps = n_envs // K
            rem = n_envs % K

            base = active_motions.repeat_interleave(reps)
            if rem > 0:
                extra = active_motions[torch.randperm(K, device=self.device)[:rem]]
                sampled = torch.cat([base, extra], dim=0)
            else:
                sampled = base

            # Randomize order
            perm = torch.randperm(n_envs, device=self.device)
            self.env_motion_indices[env_ids] = sampled[perm]

        else:
            # Direct sampling from group
            if self.cfg.unique_per_batch and n_motions_in_group >= n_envs:
                sampled_indices_in_group = torch.multinomial(probs_group, n_envs, replacement=False)
            else:
                sampled_indices_in_group = torch.multinomial(probs_group, n_envs, replacement=True)

            sampled = group_motion_indices[sampled_indices_in_group]
            self.env_motion_indices[env_ids] = sampled

        # Update motion groups (all environments in this batch belong to the same group)
        group_idx = self.group_name_to_idx[group_name]
        self.env_motion_groups[env_ids] = group_idx

        # Update motion sample counts
        unique_motions, counts = torch.unique(sampled, return_counts=True)
        self.motion_assigned_counts.index_add_(
            0, unique_motions, counts.to(self.motion_assigned_counts.dtype)
        )

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Within-motion sampling for the provided environment indices."""
        if len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        motion_indices = self.env_motion_indices[env_ids]

        lengths = self.motion_lengths[motion_indices]
        lengths_minus_one = self.motion_lengths_minus_one[motion_indices]
        denominators = self.motion_length_denominator[motion_indices]
        bin_counts = self.motion_bin_counts[motion_indices]
        bin_counts_float = self.motion_bin_counts_float[motion_indices]
        bin_mask = self.motion_bin_mask[motion_indices]

        # During warmup, we intentionally do NOT update failure statistics to avoid
        # cold-start bias (everything fails early, which makes difficulty estimates noisy).
        if not self._in_motion_sampling_warmup():
            episode_failed = self._env.termination_manager.terminated[env_ids]
            if torch.any(episode_failed):
                fail_envs = env_ids[episode_failed]
                fail_motion_idx = motion_indices[episode_failed]
                fail_bin_counts = bin_counts[episode_failed]
                fail_denominators = denominators[episode_failed]
                fail_bins = torch.clamp(
                    (self.time_steps[fail_envs] * fail_bin_counts) // fail_denominators,
                    max=fail_bin_counts - 1,
                )
                linear_indices = fail_motion_idx * self.max_bin_count + fail_bins
                self.current_bin_failed.view(-1).index_add_(
                    0,
                    linear_indices,
                    torch.ones_like(fail_bins, dtype=self.current_bin_failed.dtype),
                )
                self.motion_fail_counts.index_add_(
                    0,
                    fail_motion_idx,
                    torch.ones_like(fail_motion_idx, dtype=self.motion_fail_counts.dtype),
                )

        prob = self.bin_failed_count[motion_indices]
        uniform_term = (self.cfg.adaptive_uniform_ratio / bin_counts_float).unsqueeze(1)
        prob = (prob + uniform_term) * bin_mask

        kernel_tail = self.kernel.numel() - 1
        if kernel_tail > 0:
            prob = F.conv1d(
                F.pad(prob.unsqueeze(1), (0, kernel_tail), mode="replicate"),
                self.kernel.view(1, 1, -1),
            ).squeeze(1)
        else:
            prob = prob.clone()

        prob = prob * bin_mask
        prob_sum = prob.sum(dim=1, keepdim=True)
        zero_rows = prob_sum <= 0
        if torch.any(zero_rows):
            prob[zero_rows] = bin_mask[zero_rows].float()
            prob_sum = prob.sum(dim=1, keepdim=True)
        prob = prob / prob_sum

        sampled_bins = torch.multinomial(prob, 1).squeeze(1)
        rand_offset = torch.rand(len(env_ids), device=self.device)

        lengths_minus_one_float = lengths_minus_one.to(torch.float32)
        time_steps = torch.where(
            lengths_minus_one == 0,
            torch.zeros_like(lengths_minus_one),
            (
                (sampled_bins.to(torch.float32) + rand_offset)
                / torch.clamp(bin_counts_float, min=1.0)
                * lengths_minus_one_float
            ).long(),
        )
        time_steps = torch.clamp(time_steps, max=lengths_minus_one)
        self.time_steps[env_ids] = time_steps

        entropy = -(prob * (prob + 1e-12).log()).sum(dim=1)
        log_bins = torch.log(torch.clamp(bin_counts_float, min=1.0))
        entropy_norm = torch.where(log_bins > 0, entropy / log_bins, torch.zeros_like(entropy))
        self.metrics["sampling_entropy"][env_ids] = entropy_norm

        pmax, imax = prob.max(dim=1)
        self.metrics["sampling_top1_prob"][env_ids] = pmax
        self.metrics["sampling_top1_bin"][env_ids] = imax.to(torch.float32) / torch.clamp(bin_counts_float, min=1.0)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self.cfg.random_init_frame:
            lengths_minus_one = torch.clamp(
                self.motion_lengths[self.env_motion_indices[env_ids]] - 1, min=0
            )
            rand = torch.rand(len(env_ids), device=self.device)
            self.time_steps[env_ids] = (rand * lengths_minus_one.to(torch.float32)).long()
            self.time_steps[env_ids] = torch.minimum(self.time_steps[env_ids], lengths_minus_one)
        elif self.cfg.start_from_beginning:
            start_frame = max(int(self.cfg.start_frame), 0)
            lengths_minus_one = self.motion_lengths[self.env_motion_indices[env_ids]] - 1
            lengths_minus_one = torch.clamp(lengths_minus_one, min=0)
            start_frame_tensor = torch.full_like(lengths_minus_one, start_frame)
            self.time_steps[env_ids] = torch.minimum(start_frame_tensor, lengths_minus_one)
        else:
            self._adaptive_sampling(env_ids)

        motion_indices = self.env_motion_indices[env_ids]
        self.motion_sample_counts.index_add_(
            0,
            motion_indices,
            torch.ones_like(motion_indices, dtype=self.motion_sample_counts.dtype),
        )

        # Gather current sampled states. The base reset reads the articulation-root reference
        # from the DEDICATED root channel (full npz body axis), NOT the tracked-subset body_*
        # arrays — correct even when pelvis is not a tracked keypoint (KP5). Numerically
        # identical for sets that do track the root.
        jpos = self._gather_by_motion_for_envs("joint_pos", env_ids)
        jvel = self._gather_by_motion_for_envs("joint_vel", env_ids)

        root_pos = (
            self._gather_by_motion_for_envs("root_pos_w", env_ids)
            + self._env.scene.env_origins[env_ids]
        )
        root_ori = self._gather_by_motion_for_envs("root_quat_w", env_ids)
        root_lin_vel = self._gather_by_motion_for_envs("root_lin_vel_w", env_ids).clone()
        root_ang_vel = self._gather_by_motion_for_envs("root_ang_vel_w", env_ids).clone()

        # Random pose/velocity deltas around sampled states
        rand_samples = sample_uniform(self._pose_ranges[:, 0], self._pose_ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)

        rand_samples = sample_uniform(self._vel_ranges[:, 0], self._vel_ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = jpos.clone()
        joint_vel = jvel.clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])

        if self.cfg.reset_base_xy_to_origin:
            root_pos[..., :2] = self._env.scene.env_origins[env_ids][..., :2]

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
            env_ids=env_ids,
        )
        _refresh_reset_relative_body_poses(self, env_ids)

    def _update_metrics(self):
        self.metrics["error_anchor_pos"].copy_(torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1))
        self.metrics["error_anchor_rot"].copy_(quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w))
        self.metrics["error_anchor_lin_vel"].copy_(torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1))
        self.metrics["error_anchor_ang_vel"].copy_(torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1))

        self.metrics["error_body_pos"].copy_(torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1))
        self.metrics["error_body_rot"].copy_(quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1))

        self.metrics["error_body_lin_vel"].copy_(torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1))
        self.metrics["error_body_ang_vel"].copy_(torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1))

        self.metrics["error_joint_pos"].copy_(torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1))
        self.metrics["error_joint_vel"].copy_(torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1))
        if self._dense_joint_error_metrics:
            joint_pos_error = torch.abs(self.joint_pos - self.robot_joint_pos)
            joint_vel_error = torch.abs(self.joint_vel - self.robot_joint_vel)
            for idx, (pos_key, vel_key) in enumerate(self._joint_metric_keys):
                self.metrics[pos_key].copy_(joint_pos_error[:, idx])
                self.metrics[vel_key].copy_(joint_vel_error[:, idx])
        if self._pulse_vae_ee_ctx is not None:
            pos_err_b = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1)
            vel_err_b = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1)
            _write_pulse_vae_ee_keypoint_metrics(
                self.metrics,
                self._pulse_vae_ee_ctx,
                self._pulse_vae_env_mode_idx,
                pos_err_b,
                vel_err_b,
                num_envs=self.num_envs,
                use_copy_=True,
            )
        self._update_sampling_prob_metrics()

    def _update_command(self):
        self._global_sim_step += 1
        # Keep the command frame written by _resample_command on the same-step
        # auto-reset path.  Explicit/initial resets advance normally because
        # episode_length_buf is 1 by the time the next step reaches compute().
        episode_length = self._env.episode_length_buf
        if episode_length.shape != self.time_steps.shape:
            raise RuntimeError(
                "episode_length_buf and time_steps must have identical shapes for reset synchronization"
            )
        self.time_steps += (episode_length != 0).to(dtype=self.time_steps.dtype)

        # Resample goal-block mask once per env step (no-op when p_mask <= 0).
        self._resample_goal_mask()

        # Per-motion episode end detection and resampling
        motion_lengths = self.motion_lengths[self.env_motion_indices]
        ended = self.time_steps >= motion_lengths                                  # (num_envs,)
        self.motion_end_buf[:] = ended
        # envs_to_resample = torch.where(self.time_steps >= motion_lengths)[0]
        # if envs_to_resample.numel() > 0:
        #     self._resample_command(envs_to_resample)

        # Compute relative body poses vs anchor
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # Update per-motion failure statistics using EMA (skip during warmup)
        if not self._in_motion_sampling_warmup():
            mask = self.motion_bin_mask.float()
            if self.current_bin_failed.any():
                self.bin_failed_count = (
                    self.cfg.adaptive_alpha * self.current_bin_failed
                    + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
                ) * mask
        self.current_bin_failed.zero_()

        # Periodic motion remap
        if self._resample_motions_every_steps > 0 and \
            (self._global_sim_step % self._resample_motions_every_steps) == 0:
            self._remap_version += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        
        need = self._env_remap_version[env_ids] < self._remap_version
        envs_need_remap = env_ids[need]
        if envs_need_remap.numel() > 0:
            self._assign_motions(envs_need_remap)
            self._env_remap_version[envs_need_remap] = self._remap_version

        self.motion_end_buf[env_ids] = False
        
        return super().reset(env_ids=env_ids)
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for bi in self._video_debug_body_indices:
                    name = self.cfg.body_names[bi]
                    self.current_body_visualizers.append(
                        VisualizationMarkers(self.cfg.current_body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name))
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(self.cfg.goal_body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name))
                    )

            _show_anchor = not getattr(self, "_video_debug_hide_anchor", False)
            self.current_anchor_visualizer.set_visibility(_show_anchor)
            self.goal_anchor_visualizer.set_visibility(_show_anchor)
            for i in range(len(self._video_debug_body_indices)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self._video_debug_body_indices)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        if not getattr(self, "_video_debug_hide_anchor", False):
            self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
            self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for vis_i, bi in enumerate(self._video_debug_body_indices):
            self.current_body_visualizers[vis_i].visualize(
                self.robot_body_pos_w[:, bi], self.robot_body_quat_w[:, bi]
            )
            if self._video_debug_goal_world:
                self.goal_body_visualizers[vis_i].visualize(self.body_pos_w[:, bi], self.body_quat_w[:, bi])
            else:
                self.goal_body_visualizers[vis_i].visualize(
                    self.body_pos_relative_w[:, bi], self.body_quat_relative_w[:, bi]
                )


class PartialMaskedMultiMotionCommand(MultiMotionCommand):
    """Multi-motion command with per-env keypoint visibility modes (task-side masking).

    Reference keypoint positions/velocities use the full ``body_names`` layout; coordinates for
    bodies that are not visible in the current mode are set to NaN (policy applies a finite mask).
    Mask mode indices and sampling probabilities are owned here (and can be updated by curriculum).
    """

    cfg: "PartialMaskedMultiMotionCommandCfg"
    uses_command_manager_mask_sampling: bool = True

    def __init__(self, cfg: "PartialMaskedMultiMotionCommandCfg", env: "ManagerBasedRLEnv"):
        self._eval_fixed_mode_idx: int | None = None
        super().__init__(cfg, env)

        # World-frame VISIBLE-point-of-interest accuracy metrics (the deployment-true
        # objective for masked-KP tracking; the base ``error_body_pos`` is anchor-relative
        # and averaged over ALL tracked bodies, which hides per-mask accuracy). Under a
        # single-point mask (e.g. right_wrist_only for the writing task) these reduce to the
        # accuracy of that one visible body. Computed in ``_update_metrics``; zero-init here.
        self.metrics["error_body_pos_w_visible"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate_pos_w_2cm"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate_pos_w_5cm"] = torch.zeros(self.num_envs, device=self.device)

        from rsl_rl.algorithms.mask_utils import build_mask_matrix

        self._mode_names = tuple(cfg.mask_mode_spec.keys())
        if len(self._mode_names) == 0:
            raise ValueError("mask_mode_spec must contain at least one mode.")

        self._bernoulli_keep_prob = float(getattr(cfg, "bernoulli_keep_prob", 1.0))
        self._bernoulli_keep_prob = max(0.0, min(1.0, self._bernoulli_keep_prob))
        # Detect ALL bernoulli modes: name == "bernoulli"/"bernouli" OR ends with "_bernoulli"
        # (e.g. "kp5_bernoulli" — a bernoulli scoped to a specific body subset). Each bernoulli mode
        # samples per-body visibility at runtime over the bodies listed in its mask_mode_spec entry;
        # bodies NOT listed stay 0.0 (fixed-masked).
        self._bernoulli_mode_indices: list[int] = []
        for i, name in enumerate(self._mode_names):
            n = str(name).lower()
            if n in ("bernoulli", "bernouli") or n.endswith("_bernoulli"):
                self._bernoulli_mode_indices.append(i)
        # Legacy alias (first bernoulli mode, or None). Kept for backward-compat with any external
        # callers; new logic uses ``_bernoulli_mode_indices``.
        self._bernoulli_mode_idx: int | None = (
            self._bernoulli_mode_indices[0] if self._bernoulli_mode_indices else None
        )

        probs = torch.tensor(list(cfg.mask_mode_probs), dtype=torch.float32, device=self.device)
        if probs.numel() != len(self._mode_names):
            raise ValueError(
                f"mask_mode_probs length ({probs.numel()}) must equal number of modes ({len(self._mode_names)})."
            )
        probs = torch.clamp(probs, min=0.0)
        if torch.all(probs <= 0):
            raise ValueError("mask_mode_probs must contain at least one positive value.")
        self._effective_mode_probs = probs / probs.sum()

        dup_vel = bool(cfg.duplicate_for_ref_body_lin_vel)
        self._mode_goal_masks, _ = build_mask_matrix(
            list(cfg.body_names),
            dict(cfg.mask_mode_spec),
            device=self.device,
            duplicate_for_ref_body_lin_vel=dup_vel,
        )

        num_bodies = len(cfg.body_names)
        self._mode_body_visibility = torch.zeros(
            (len(self._mode_names), num_bodies), dtype=torch.float32, device=self.device
        )
        name_to_idx = {name: i for i, name in enumerate(cfg.body_names)}
        for mode_i, mode_name in enumerate(self._mode_names):
            # Bodies listed in the mode's spec are "potentially visible" (set to 1.0).
            # For deterministic modes this IS the final visibility. For bernoulli modes, this defines
            # the SUBSET that per-body sampling acts on at runtime (bodies outside the subset stay 0.0).
            # Existing configs that listed all bodies under "bernoulli" continue to work unchanged
            # (subset == all bodies → behaviour identical to the old fill-with-1.0 fast path).
            for body_name in cfg.mask_mode_spec[mode_name]:
                if body_name not in name_to_idx:
                    raise ValueError(f"Body {body_name!r} in mode {mode_name!r} is not in cfg.body_names.")
                self._mode_body_visibility[mode_i, name_to_idx[body_name]] = 1.0

        self._env_mode_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._env_body_mask = torch.ones(self.num_envs, num_bodies, dtype=torch.float32, device=self.device)
        self._sample_modes_for_envs(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

        p_cpu = self._effective_mode_probs.detach().cpu()
        lines = [
            "[PartialMaskedMultiMotionCommand] keypoint mask modes (per-env on resample; curriculum can update probs):",
        ]
        for i, name in enumerate(self._mode_names):
            tag = "  (bernoulli)" if i in self._bernoulli_mode_indices else ""
            vis_count = int(self._mode_body_visibility[i].sum().item())
            lines.append(f"  [{i}] {name!r}: p={float(p_cpu[i].item()):.6f}  vis_bodies={vis_count}/{num_bodies}{tag}")
        if self._bernoulli_mode_indices:
            lines.append(f"  bernoulli_keep_prob_init={self._bernoulli_keep_prob:.6f} (shared across all bernoulli modes)")
        lines.append(f"  duplicate_for_ref_body_lin_vel={dup_vel}  num_bodies={num_bodies}")
        print("\n".join(lines), flush=True)

    def _sample_modes_for_envs(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if self._eval_fixed_mode_idx is not None:
            m = int(self._eval_fixed_mode_idx)
            sampled = torch.full((env_ids.numel(),), m, dtype=torch.long, device=self.device)
        else:
            sampled = torch.multinomial(self._effective_mode_probs, env_ids.numel(), replacement=True)
        self._env_mode_idx[env_ids] = sampled
        self._env_body_mask[env_ids] = self._mode_body_visibility[sampled]

        # For each bernoulli mode (there may be more than one — e.g. global ``bernoulli`` over all
        # bodies + ``kp5_bernoulli`` scoped to a demo subset), resample per-body visibility for the
        # envs that drew that mode. Sampling is restricted to the mode's body subset via element-
        # wise multiply with ``_mode_body_visibility[mode_i]`` so bodies outside the subset stay 0.0.
        # All bernoulli modes share the same global ``_bernoulli_keep_prob`` (curriculum-driven).
        p_keep = float(self._bernoulli_keep_prob)
        num_bodies = int(self._env_body_mask.shape[1])
        for mode_i in self._bernoulli_mode_indices:
            bernoulli_env_mask = sampled == mode_i
            if not bool(bernoulli_env_mask.any()):
                continue
            bernoulli_env_ids = env_ids[bernoulli_env_mask]
            mode_vis = self._mode_body_visibility[mode_i]  # [num_bodies], 1.0 for in-subset bodies
            if p_keep <= 0.0:
                # All bodies (in or out of subset) masked.
                self._env_body_mask[bernoulli_env_ids] = 0.0
            elif p_keep >= 1.0:
                # All in-subset bodies visible; out-of-subset bodies stay masked.
                self._env_body_mask[bernoulli_env_ids] = mode_vis.unsqueeze(0).expand(
                    bernoulli_env_ids.numel(), num_bodies
                )
            else:
                vis = (torch.rand((bernoulli_env_ids.numel(), num_bodies), device=self.device) < p_keep).to(
                    dtype=torch.float32
                )
                # Restrict bernoulli draw to the mode's body subset (out-of-subset → 0.0).
                vis = vis * mode_vis.unsqueeze(0)
                self._env_body_mask[bernoulli_env_ids] = vis

    def set_bernoulli_keep_prob(self, p_keep: float) -> None:
        """Update Bernoulli keep probability `p_keep` used when sampling `bernoulli` mode.

        Note: existing envs only update their visibility when their mask mode is re-sampled.
        """
        p = float(p_keep)
        p = max(0.0, min(1.0, p))
        self._bernoulli_keep_prob = p

    def set_mask_mode_probs_tuple(self, probs: tuple[float, ...]) -> None:
        """Update mode sampling distribution (used by curriculum).

        When probabilities change, all parallel envs are resampled immediately. Otherwise only resetting
        envs would redraw modes on :meth:`_resample_command`, and non-reset envs would keep stale
        mode indices until their episode ended (curriculum would look \"stuck\" in the prior phase).
        """
        t = torch.tensor(probs, dtype=torch.float32, device=self.device)
        if t.numel() != len(self._mode_names):
            raise ValueError(
                f"probs length {t.numel()} must match num modes {len(self._mode_names)}."
            )
        t = torch.clamp(t, min=0.0)
        if torch.all(t <= 0):
            raise ValueError("mode_probs must have a positive entry.")
        new_p = t / t.sum()
        if new_p.shape == self._effective_mode_probs.shape and torch.allclose(
            new_p, self._effective_mode_probs, rtol=0.0, atol=1e-6
        ):
            return
        self._effective_mode_probs = new_p
        self._sample_modes_for_envs(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    def set_eval_fixed_mask_mode_idx(self, idx: int | None) -> None:
        """Pin every env to one mode row during evaluation (e.g. play). None = resume stochastic sampling."""
        self._eval_fixed_mode_idx = None if idx is None else int(idx)

    def resample_all_mask_modes(self) -> None:
        """Re-draw mask modes for every env (e.g. after :meth:`set_eval_fixed_mask_mode_idx`)."""
        self._sample_modes_for_envs(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    @property
    def env_keypoint_mask_mode_indices(self) -> torch.Tensor:
        return self._env_mode_idx

    def ref_body_pos_robot_anchor_b(self) -> torch.Tensor:
        pos = super().ref_body_pos_robot_anchor_b().view(self.num_envs, len(self.cfg.body_names), 3)
        vis = self._env_body_mask.unsqueeze(-1)
        nan = torch.full_like(pos, float("nan"))
        out = torch.where(vis > 0.5, pos, nan)
        return out.view(self.num_envs, -1)

    def ref_body_lin_vel_robot_anchor_b(self) -> torch.Tensor:
        vel = super().ref_body_lin_vel_robot_anchor_b().view(self.num_envs, len(self.cfg.body_names), 3)
        vis = self._env_body_mask.unsqueeze(-1)
        nan = torch.full_like(vel, float("nan"))
        out = torch.where(vis > 0.5, vel, nan)
        return out.view(self.num_envs, -1)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # RNG-order alignment with the teacher's training env: ``super()._resample_command``
        # (MultiMotionCommand) selects the motion clip and applies pose/velocity/joint reset
        # randomization via the global torch RNG. ``_sample_modes_for_envs`` ALSO consumes the
        # global RNG (multinomial + bernoulli ``torch.rand``). Running mode-sampling FIRST (the
        # previous order) shifted the RNG stream so every reset drew different clip + init
        # randomization than OneStage — degrading the (deterministic) teacher's closed-loop
        # tracking (~20× more ee_body_pos terminations under teacher-pilot). Reset first so the
        # clip/init RNG draws match OneStage exactly; mode-sampling consumes RNG afterward and
        # only the KP-student mask depends on it (applied later in obs, order-independent here).
        super()._resample_command(env_ids)
        self._sample_modes_for_envs(env_ids_t)

    def _update_metrics(self):
        if self._pulse_vae_ee_ctx is not None:
            self._pulse_vae_env_mode_idx.copy_(self._env_mode_idx)
        super()._update_metrics()

        # World-frame position accuracy of the VISIBLE points of interest (matches the
        # latent-RL ``poi_pos`` reward's frame + visibility selection). For right_wrist_only
        # this is the right-wrist writing error; for bernoulli masks it's the mean over the
        # currently-visible bodies. NaN-safe on all-masked envs (denom clamp).
        vis = self._env_body_mask.to(self.body_pos_w.dtype)  # [N, B], 1=visible
        per_body_err = torch.norm(self.body_pos_w - self.robot_body_pos_w, dim=-1)  # [N, B] world
        denom = vis.sum(dim=-1).clamp_min(1.0)
        err_vis = (per_body_err * vis).sum(dim=-1) / denom  # [N]
        self.metrics["error_body_pos_w_visible"] = err_vis
        self.metrics["success_rate_pos_w_2cm"] = (err_vis < 0.02).to(err_vis.dtype)
        self.metrics["success_rate_pos_w_5cm"] = (err_vis < 0.05).to(err_vis.dtype)

    def set_pulse_vae_ee_env_mode_indices(self, mode_indices: torch.Tensor | None) -> None:
        """Mode indices are owned by this command; runner/policy should not overwrite."""
        del mode_indices
        return

    def _snapshot_motion_dataset_for_holdout(self) -> dict:
        snap = super()._snapshot_motion_dataset_for_holdout()
        snap["_partial_mask_env_mode_idx"] = self._env_mode_idx.clone()
        snap["_partial_mask_env_body_mask"] = self._env_body_mask.clone()
        return snap

    def _restore_motion_dataset_from_holdout_snapshot(self, snap: dict) -> None:
        super()._restore_motion_dataset_from_holdout_snapshot(snap)
        if "_partial_mask_env_mode_idx" in snap:
            self._env_mode_idx.copy_(snap["_partial_mask_env_mode_idx"].to(self.device))
            self._env_body_mask.copy_(snap["_partial_mask_env_body_mask"].to(self.device))


@configclass
class MultiMotionCommandCfg(CommandTermCfg):
    """Configuration for the multi-motion command."""

    class_type: type = MultiMotionCommand

    asset_name: str = MISSING

    motion: str = MISSING
    file_glob: str = "*.npz"
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    motion_preload_device: str | None = 'cuda'

    motion_horizon: int = 1

    # ---- multi-teacher support: motion groups ----
    # Motion groups configuration for multi-teacher support.
    # Maps group names to folder patterns for categorizing motions.
    # Example: {"lafan": ["lafan_npz_10s_without_fall_and_getup"], "fld": ["motions_fld_test"]}
    # If None, all motions belong to a single "default" group.
    motion_groups: dict[str, list[str]] | None = None

    # Motion group sampling ratios - controls proportion of environments assigned to each group.
    # Maps group names to sampling ratios (should sum to 1.0).
    # Example: {"lafan": 0.7, "fld": 0.3} means 70% of envs use lafan motions, 30% use fld motions.
    # If None, environments are assigned uniformly across all available motions (ignoring groups).
    motion_group_sampling_ratios: dict[str, float] | None = None

    # ---- dataset slicing / sharding across GPUs ----
    # If True, each GPU process loads a disjoint subset of motions (when possible),
    # reducing duplicates during multi-GPU training. This is especially useful when
    # total motions >> (num_envs or max_active_motions).
    motion_dataset_shard_across_gpus: bool = True
    # Use 'global' rank/world_size (RANK/WORLD_SIZE) or 'local' (LOCAL_RANK/LOCAL_WORLD_SIZE).
    motion_dataset_shard_by: str = "global"
    # Deterministic shuffle seed before slicing.
    motion_dataset_shard_seed: int = 0
    # Sharding strategy when total < world_size * load_cap: 'chunk' (contiguous) or 'stride' (round-robin).
    motion_dataset_shard_strategy: str = "chunk"
    # Cap number of motions loaded per process. If None and sharding enabled, auto-caps to min(num_envs, max_active_motions).
    motion_dataset_load_cap: int | None = None
    # If True, print shard info once at startup (useful for sanity checks).
    motion_dataset_log_shard_info: bool = False
    # If True, write shard info once into wandb run.summary (rank0 only). No-op if wandb isn't used.
    motion_dataset_log_wandb_summary: bool = True

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    start_from_beginning: bool = False
    start_frame: int = 0

    # If True, sample a uniformly random frame in [0, motion_length-1) for each env on resample.
    # Takes precedence over `start_from_beginning` and `_adaptive_sampling`.
    random_init_frame: bool = False
    # If True, override the sampled root XY to the env's scene origin XY before writing root state.
    # Decouples initial world position from the motion clip's progression — keeps pose+momentum but
    # always spawns the robot in the camera frame. No effect on yaw or relative-frame goal observations.
    reset_base_xy_to_origin: bool = False

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    # If enabled, log per-joint absolute errors for position and velocity tracking.
    # Metrics are emitted as: error_joint_pos_<joint_name>, error_joint_vel_<joint_name>.
    enable_dense_joint_error_metrics: bool = False

    weight_fail: float = 0.5
    weight_novel: float = 0.3
    cap_beta: float = 2.0

    command_velocity: bool = True

    # ---- motion-level sampling weight schedule (uniform -> ramp to weights above) ----
    # insight for the values:
    # - warmup_s: make sure every motion is sampled at least once. (10-20 times of resample_motions_every_s)
    # - ramp_s: make sure the weights are not too small. (20-50 times of resample_motions_every_s)
    # - schedule: "linear" or "cosine" is the schedule type for the ramp. (cosine is better)
    # Warmup: keep motion sampling uniform for this duration (seconds).
    motion_sampling_warmup_s: float = 1000000000.0
    # Ramp: linearly/cosine ramp fail/novel weights from 0 -> target over this duration (seconds).
    motion_sampling_ramp_s: float = 1000000000.0
    # Schedule type for ramp: "linear" or "cosine".
    motion_sampling_schedule: str = "linear"

    # Resampling cadence (seconds) for motion-to-env reassignment (set to 0 or 1e9 to disable)
    resample_motions_every_s: float = 1000000000.0
    # Whether to sample motions without replacement per remap batch when possible
    unique_per_batch: bool = True

    max_active_motions: int | None = 10000

    # Anchor stays a frame: it is a true root *pose* whose orientation (heading) is meaningful.
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    # Keypoints are positions, not poses -> spheres. Green = robot current, red = command goal.
    current_body_visualizer_cfg: VisualizationMarkersCfg = _kp_sphere_marker_cfg((0.0, 1.0, 0.0))
    goal_body_visualizer_cfg: VisualizationMarkersCfg = _kp_sphere_marker_cfg((1.0, 0.0, 0.0))

    # Optional: restrict debug body markers and draw goal bodies in motion world frame (see play.py).
    video_debug_vis_body_names: list[str] | None = None
    video_debug_vis_goal_bodies_world_frame: bool = False
    # Optional: suppress the anchor frame triads so only the body dot markers render
    # (see play.py --poi5_dot_vis).
    video_debug_vis_hide_anchor: bool = False


@configclass
class PartialMaskedMultiMotionCommandCfg(MultiMotionCommandCfg):
    """Like :class:`MultiMotionCommandCfg` but with partial keypoint visibility modes."""

    class_type: type = PartialMaskedMultiMotionCommand

    mask_mode_spec: dict[str, list[str]] = MISSING
    mask_mode_probs: tuple[float, ...] = MISSING
    duplicate_for_ref_body_lin_vel: bool = True
    # Keep probability for the special stochastic `bernoulli` mask mode.
    # Semantics: `p_keep=1.0` => all keypoints visible; `p_keep=0.0` => all hidden.
    # x/y/z coordinates of the same body share one Bernoulli draw.
    bernoulli_keep_prob: float = 1.0
