#!/usr/bin/env python3
"""Bounded G1 teacher reset acceptance; --self-test needs no Isaac or GPU.

Native runs use existing DEV NPZ files only, reference joint PD actions, four
reset cases and at most 12 control steps. No training or conversion occurs.
The target used for a first reward is frozen at reset, before physics moves
the robot. Hooks observe the original manager computations exactly once.
No command update, FK update, reward recomputation or RNG draw is injected.
The 0.08 s episode timeout is a declared lifecycle diagnostic, not a failure
of a learned teacher. Neither profile is a checkpoint-policy evaluation.

Contracts: world positions (N,B,3); unit WXYZ quaternions (N,B,4);
tracked G1 bodies B=14, native bodies=30, joints/actions=29;
returned policy/critic observations (N,815)/(N,289). An independent NumPy
oracle implements the original yaw-relative geometry using raw NPZ and
PhysX link transforms (whose XYZW quaternions are explicitly converted).
Only selected rows are changed by an explicit partial reset. The ordinary
env.reset observation-history update still follows Isaac's own lifecycle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "General-Tracking-Flat-G1-Teacher-Cache-v1",
    "General-Tracking-Flat-G1-Teacher-Points-v1",
)
TOLERANCE = 3.0e-5


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.array(value, copy=True)
    assert np.isfinite(result).all(), "Nonfinite native state"
    return result


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    assert left.shape[-1] == right.shape[-1] == 4
    lw, lv = left[..., :1], left[..., 1:]
    rw, rv = right[..., :1], right[..., 1:]
    return np.concatenate((lw * rw - np.sum(lv * rv, axis=-1, keepdims=True),
                           lw * rv + rw * lv + np.cross(lv, rv)), axis=-1)


def quat_inverse(value: np.ndarray) -> np.ndarray:
    assert value.shape[-1] == 4
    result = value.copy()
    result[..., 1:] *= -1.0
    return result / np.sum(value * value, axis=-1, keepdims=True)


def quat_apply(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    assert quat.shape[-1] == 4 and vector.shape[-1] == 3
    cross = 2.0 * np.cross(quat[..., 1:], vector)
    return vector + quat[..., :1] * cross + np.cross(quat[..., 1:], cross)


def relative_oracle(reference_pos: np.ndarray, reference_quat: np.ndarray,
                    actual_anchor_pos: np.ndarray, actual_anchor_quat: np.ndarray,
                    anchor_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Original yaw alignment; x/y actual anchor and z reference anchor."""
    n, bodies, xyz = reference_pos.shape
    assert xyz == 3 and reference_quat.shape == (n, bodies, 4)
    assert actual_anchor_pos.shape == (n, 3) and actual_anchor_quat.shape == (n, 4)
    assert 0 <= anchor_index < bodies
    assert np.allclose(np.linalg.norm(reference_quat, axis=-1), 1.0, atol=1.0e-4)
    assert np.allclose(np.linalg.norm(actual_anchor_quat, axis=-1), 1.0, atol=1.0e-4)
    anchor = reference_pos[:, anchor_index]
    delta = quat_mul(actual_anchor_quat, quat_inverse(reference_quat[:, anchor_index]))
    w, x, y, z = np.moveaxis(delta, -1, 0)
    angle = np.arctan2(2.0 * (w*z+x*y), 1.0-2.0*(y*y+z*z))
    yaw = np.stack((np.cos(angle/2.0), np.zeros(n), np.zeros(n), np.sin(angle/2.0)), axis=-1)
    aligned_anchor = actual_anchor_pos.copy()
    aligned_anchor[:, 2] = anchor[:, 2]
    pos = aligned_anchor[:, None] + quat_apply(yaw[:, None], reference_pos-anchor[:, None])
    quat = quat_mul(yaw[:, None], reference_quat)
    assert pos.shape == (n, bodies, 3) and quat.shape == (n, bodies, 4)
    assert np.isfinite(pos).all() and np.isfinite(quat).all()
    return pos, quat


def position_reward(target: np.ndarray, actual: np.ndarray, std: float,
                    indices: Sequence[int]) -> np.ndarray:
    assert target.shape == actual.shape and std > 0.0 and len(indices) > 0
    return np.exp(-np.sum((target[:, indices]-actual[:, indices])**2, axis=-1).mean(axis=-1)/std**2)


def orientation_reward(target: np.ndarray, actual: np.ndarray, std: float,
                       indices: Sequence[int]) -> np.ndarray:
    assert target.shape == actual.shape and std > 0.0 and len(indices) > 0
    delta = quat_mul(target[:, indices], quat_inverse(actual[:, indices]))
    angle = 2.0*np.arctan2(np.linalg.norm(delta[..., 1:], axis=-1), np.abs(delta[..., 0]))
    return np.exp(-(angle**2).mean(axis=-1)/std**2)


def z_termination(target: np.ndarray, actual: np.ndarray, threshold: float,
                  indices: Sequence[int]) -> np.ndarray:
    assert target.shape == actual.shape and threshold > 0.0 and len(indices) > 0
    return (np.abs(target[:, indices, 2]-actual[:, indices, 2]) > threshold).any(axis=-1)


def validate_dev_view(motion: Path, manifest: Path) -> list[dict[str, Any]]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["status"] == "READ_ONLY_EXISTING_NPZ_VIEW"
    assert data["new_NPZ_created"] is False and data["conversion_executed"] is False
    entries = data["entries"]
    assert isinstance(entries, list) and 1 <= len(entries) <= 3
    files = sorted(path.resolve() for path in motion.glob("*.npz"))
    expected = sorted(Path(entry["source"]).resolve() for entry in entries)
    assert files == expected, "Motion view differs from its frozen DEV manifest"
    for entry in entries:
        assert Path(entry["view"]).parent.resolve() == motion.resolve()
        assert Path(entry["view"]).resolve() == Path(entry["source"]).resolve()
        assert sha256(Path(entry["source"])) == entry["sha256"]
        with np.load(entry["source"], allow_pickle=False) as payload:
            assert payload["joint_pos"].shape == (entry["frames"], 29)
            assert payload["body_pos_w"].shape == (entry["frames"], 30, 3)
            assert payload["body_quat_w"].shape == (entry["frames"], 30, 4)
            assert float(np.asarray(payload["fps"]).reshape(-1)[0]) == 50.0
            for key in ("joint_pos", "body_pos_w", "body_quat_w"):
                assert np.isfinite(payload[key]).all()
    return entries


def source_binding() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        if name in ("whole_body_tracking", "rsl_rl") or name.startswith(("whole_body_tracking.", "rsl_rl.")):
            filename = getattr(module, "__file__", None)
            if filename is not None:
                path = Path(filename).resolve()
                assert path.is_relative_to(ROOT/"source"), f"Conflicting source: {name}: {path}"
                result[name] = str(path)
    assert "whole_body_tracking.tasks.tracking.mdp.commands" in result
    return result


def save_raw_tensors(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    tensors: dict[str, np.ndarray] = {}
    for index, event in enumerate(events):
        for stage in ("before", "after", "state"):
            state = event.get(stage)
            if isinstance(state, dict):
                for key, value in state.items():
                    if isinstance(value, list):
                        tensors[f"event_{index:03d}_{stage}_{key}"] = array(value)
        for key in ("raw", "joint_target", "live_scale", "live_offset", "total_reward",
                    "reset_frozen_oracle_pos", "reset_frozen_oracle_quat", "old_stale_cache_pos", "old_stale_cache_quat"):
            if key in event:
                tensors[f"event_{index:03d}_{key}"] = array(event[key])
    with path.open("xb") as handle:
        np.savez_compressed(handle, **tensors)


class ResetObserver:
    """Read-only method observers; original mutation methods run once."""

    def __init__(self, env: Any, torch_module: Any, output: Path) -> None:
        self.env, self.torch, self.output = env, torch_module, output
        self.command = env.command_manager.get_term("motion")
        self.robot = env.scene["robot"]
        self.body_names = list(self.command.cfg.body_names)
        self.native_names = list(self.robot.body_names)
        schema_path = ROOT/"source/whole_body_tracking/whole_body_tracking/synth/cache/g1_body_names.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert self.native_names == schema["body_names"], "NPZ/native body axis mismatch"
        assert list(self.robot.joint_names) == schema["joint_names"], "NPZ/native joint axis mismatch"
        assert len(self.body_names) == 14 and len(self.native_names) == 30
        self.indices = [self.native_names.index(name) for name in self.body_names]
        self.anchor = self.body_names.index(self.command.cfg.anchor_body_name)
        self.native_anchor = self.native_names.index(self.command.cfg.anchor_body_name)
        self.paths = [Path(path).resolve() for path in self.command.motion_dir_loader.motion_paths]
        self.payloads: list[dict[str, np.ndarray]] = []
        for path in self.paths:
            with np.load(path, allow_pickle=False) as payload:
                self.payloads.append({key: np.array(payload[key], copy=True) for key in
                                      ("body_pos_w", "body_quat_w", "joint_pos")})
        self.oracle_pos = array(self.command.body_pos_relative_w)
        self.oracle_quat = array(self.command.body_quat_relative_w)
        self.stale_pos, self.stale_quat = self.oracle_pos.copy(), self.oracle_quat.copy()
        self.first_step = np.zeros(env.num_envs, dtype=bool)
        self.events: list[dict[str, Any]] = []
        self.label, self.step_index = "constructed", 0
        self.counts = {"resample": 0, "termination": 0, "reward": 0}
        self.last_terms: dict[str, list[bool]] = {}
        self.failures: list[str] = []
        self.install()

    def rng_hash(self) -> str:
        h = hashlib.sha256()
        h.update(repr(random.getstate()).encode())
        h.update(repr(np.random.get_state()).encode())
        h.update(bytes(self.torch.get_rng_state().cpu().numpy()))
        h.update(bytes(self.torch.cuda.get_rng_state(self.env.device).cpu().numpy()))
        return h.hexdigest()

    def physics_poses(self) -> tuple[np.ndarray, np.ndarray]:
        # Direct PhysX read, no update_articulations_kinematic/forward/step.
        pose = array(self.robot.root_physx_view.get_link_transforms())
        assert pose.shape == (self.env.num_envs, 30, 7)
        return pose[..., :3], pose[..., [6, 3, 4, 5]]

    def reference(self) -> tuple[np.ndarray, np.ndarray]:
        motion_ids, times = array(self.command.env_motion_indices), array(self.command.time_steps)
        origins = array(self.env.scene.env_origins)
        pos, quat = [], []
        for env_id, (motion_id, frame) in enumerate(zip(motion_ids, times)):
            data = self.payloads[int(motion_id)]
            frame = min(max(int(frame), 0), data["body_pos_w"].shape[0]-1)
            pos.append(data["body_pos_w"][frame, self.indices]+origins[env_id])
            quat.append(data["body_quat_w"][frame, self.indices])
        return np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)

    def fresh_oracle(self) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = self.physics_poses()
        ref_pos, ref_quat = self.reference()
        return relative_oracle(ref_pos, ref_quat, pos[:, self.native_anchor], quat[:, self.native_anchor], self.anchor)

    def state(self) -> dict[str, Any]:
        actual_pos, actual_quat = self.physics_poses()
        ref_pos, ref_quat = self.reference()
        return {"label": self.label, "step_index": self.step_index,
                "time_steps": array(self.command.time_steps).tolist(),
                "episode_length": array(self.env.episode_length_buf).tolist(),
                "motion_indices": array(self.command.env_motion_indices).tolist(),
                "actual_link_pos": actual_pos.tolist(), "actual_link_quat_wxyz": actual_quat.tolist(),
                "reference_pos": ref_pos.tolist(), "reference_quat": ref_quat.tolist(),
                "cache_pos": array(self.command.body_pos_relative_w).tolist(),
                "cache_quat": array(self.command.body_quat_relative_w).tolist(),
                "joint_pos": array(self.robot.data.joint_pos).tolist(),
                "root_state": array(self.robot.data.root_state_w).tolist(),
                "sample_counts": array(self.command.motion_sample_counts).tolist(),
                "motion_fail_counts": array(self.command.motion_fail_counts).tolist(),
                "bin_failed_count": array(self.command.bin_failed_count).tolist(),
                "current_bin_failed": array(self.command.current_bin_failed).tolist()}

    def compare_cache(self, ids: np.ndarray, phase: str) -> dict[str, Any]:
        actual_pos = array(self.command.body_pos_relative_w)
        actual_quat = array(self.command.body_quat_relative_w)
        pos_error = np.max(np.abs(actual_pos[ids]-self.oracle_pos[ids]))
        quat_error = np.max(np.minimum(np.linalg.norm(actual_quat[ids]-self.oracle_quat[ids], axis=-1),
                                      np.linalg.norm(actual_quat[ids]+self.oracle_quat[ids], axis=-1)))
        passed = bool(pos_error < TOLERANCE and quat_error < TOLERANCE)
        if not passed:
            self.failures.append(f"{phase}: relative cache mismatch pos={pos_error}, quat={quat_error}")
        return {"pass": passed, "max_position_component_error_m": float(pos_error),
                "max_sign_invariant_quaternion_error": float(quat_error), "env_ids": ids.tolist()}

    def indices_for(self, params: Mapping[str, Any]) -> list[int]:
        selected = params.get("body_names")
        result = [i for i, name in enumerate(self.body_names) if selected is None or name in selected]
        assert result
        return result

    def install(self) -> None:
        original_resample = self.command._resample_command
        original_termination = self.env.termination_manager.compute
        original_reward = self.env.reward_manager.compute

        def resample(env_ids: Any) -> Any:
            ids = array(env_ids).astype(np.int64)
            if ids.size == 0:
                return original_resample(env_ids)
            before = self.state()
            untouched = np.setdiff1d(np.arange(self.env.num_envs), ids)
            old_pos, old_quat = array(self.command.body_pos_relative_w), array(self.command.body_quat_relative_w)
            identities = (id(self.command.body_pos_relative_w), id(self.command.body_quat_relative_w))
            result = original_resample(env_ids)
            rng_after_original = self.rng_hash()
            fresh_pos, fresh_quat = self.fresh_oracle()
            self.stale_pos[ids], self.stale_quat[ids] = old_pos[ids], old_quat[ids]
            self.oracle_pos[ids], self.oracle_quat[ids] = fresh_pos[ids], fresh_quat[ids]
            self.first_step[ids] = True
            after = self.state()
            unchanged = all(np.array_equal(np.asarray(before[key])[untouched], np.asarray(after[key])[untouched])
                            for key in ("time_steps", "motion_indices", "cache_pos", "cache_quat", "joint_pos", "root_state"))
            identity_kept = identities == (id(self.command.body_pos_relative_w), id(self.command.body_quat_relative_w))
            sampler_delta = np.asarray(after["sample_counts"])-np.asarray(before["sample_counts"])
            expected_delta = np.bincount(np.asarray(after["motion_indices"])[ids], minlength=len(self.paths))
            sampling_once = bool(np.array_equal(sampler_delta, expected_delta))
            observer_rng_unchanged = rng_after_original == self.rng_hash()
            assert observer_rng_unchanged, "Observer unexpectedly changed RNG"
            if not (unchanged and identity_kept and sampling_once):
                self.failures.append("Reset row/cache identity/sample count invariant failed")
            self.events.append({"phase": "resample", "before": before, "after": after,
                                "check": self.compare_cache(ids, "resample"),
                                "untouched_rows_equal": unchanged, "cache_identity_preserved": identity_kept,
                                "sampling_once": sampling_once, "observer_rng_unchanged": observer_rng_unchanged,
                                "reset_frozen_oracle_pos": self.oracle_pos[ids].tolist(),
                                "reset_frozen_oracle_quat": self.oracle_quat[ids].tolist(),
                                "old_stale_cache_pos": old_pos[ids].tolist(),
                                "old_stale_cache_quat": old_quat[ids].tolist()})
            self.counts["resample"] += 1
            return result

        def termination() -> Any:
            before = self.state()
            ids = np.flatnonzero(self.first_step)
            checks = self.compare_cache(ids, "first_termination") if ids.size else None
            target = self.oracle_pos.copy()
            cfg = self.env.termination_manager.get_term_cfg("ee_body_pos")
            assert cfg.func.__name__ == "bad_motion_body_pos_z_only"
            actual = np.asarray(before["actual_link_pos"])[:, self.indices]
            indices = self.indices_for(cfg.params)
            expected = z_termination(target, actual, float(cfg.params["threshold"]), indices)
            stale = z_termination(self.stale_pos, actual, float(cfg.params["threshold"]), indices)
            rng_before = self.rng_hash()
            result = original_termination()
            names = list(self.env.termination_manager.active_terms)
            done_values = array(self.env.termination_manager._term_dones)
            self.last_terms = {name: done_values[:, i].tolist() for i, name in enumerate(names)}
            actual_ee = done_values[:, names.index("ee_body_pos")]
            same = bool(np.array_equal(expected, actual_ee))
            if not same:
                self.failures.append("ee_body_pos disagrees with frozen geometric oracle")
            self.events.append({"phase": "termination", "before": before, "first_step_rows": ids.tolist(),
                                "cache_check": checks, "terms": self.last_terms,
                                "ee_frozen_oracle": expected.tolist(), "ee_old_stale_counterfactual": stale.tolist(),
                                "ee_oracle_equal": same, "manager_rng_unchanged": rng_before == self.rng_hash()})
            self.counts["termination"] += 1
            return result

        def reward(dt: float) -> Any:
            before = self.state()
            actual_pos = np.asarray(before["actual_link_pos"])[:, self.indices]
            actual_quat = np.asarray(before["actual_link_quat_wxyz"])[:, self.indices]
            pos_cfg = self.env.reward_manager.get_term_cfg("motion_body_pos")
            quat_cfg = self.env.reward_manager.get_term_cfg("motion_body_ori")
            assert pos_cfg.func.__name__ == "motion_relative_body_position_error_exp"
            assert quat_cfg.func.__name__ == "motion_relative_body_orientation_error_exp"
            oracle_pos = position_reward(self.oracle_pos, actual_pos, float(pos_cfg.params["std"]), self.indices_for(pos_cfg.params))
            oracle_quat = orientation_reward(self.oracle_quat, actual_quat, float(quat_cfg.params["std"]), self.indices_for(quat_cfg.params))
            stale_pos = position_reward(self.stale_pos, actual_pos, float(pos_cfg.params["std"]), self.indices_for(pos_cfg.params))
            stale_quat = orientation_reward(self.stale_quat, actual_quat, float(quat_cfg.params["std"]), self.indices_for(quat_cfg.params))
            sums_before = {name: array(value) for name, value in self.env.reward_manager._episode_sums.items()}
            result = original_reward(dt)
            names = list(self.env.reward_manager.active_terms)
            values = array(self.env.reward_manager._step_reward)
            actual_pos_reward = values[:, names.index("motion_body_pos")]/float(pos_cfg.weight)
            actual_quat_reward = values[:, names.index("motion_body_ori")]/float(quat_cfg.weight)
            pos_error = float(np.max(np.abs(actual_pos_reward-oracle_pos)))
            quat_error = float(np.max(np.abs(actual_quat_reward-oracle_quat)))
            passed = pos_error < TOLERANCE and quat_error < TOLERANCE
            once = all(np.allclose(array(self.env.reward_manager._episode_sums[name])-sums_before[name],
                                   values[:, i]*dt, rtol=1.0e-5, atol=2.0e-6) for i, name in enumerate(names))
            if not (passed and once):
                self.failures.append(f"Reward oracle/once mismatch pos={pos_error}, quat={quat_error}, once={once}")
            self.events.append({"phase": "reward", "before": before, "dt": float(dt),
                                "first_step_rows": np.flatnonzero(self.first_step).tolist(),
                                "terms_weighted_per_second": {name: values[:, i].tolist() for i, name in enumerate(names)},
                                "total_reward": array(result).tolist(), "oracle_pass": passed,
                                "position_max_error": pos_error, "orientation_max_error": quat_error,
                                "position_frozen_oracle": oracle_pos.tolist(), "orientation_frozen_oracle": oracle_quat.tolist(),
                                "position_old_stale_counterfactual": stale_pos.tolist(),
                                "orientation_old_stale_counterfactual": stale_quat.tolist(), "episodic_sum_updated_once": once})
            self.first_step[:] = False
            self.counts["reward"] += 1
            return result

        self.command._resample_command = resample
        self.env.termination_manager.compute = termination
        self.env.reward_manager.compute = reward

    def observe_reset_complete(self, observations: Mapping[str, Any]) -> None:
        self.check_observations(observations)
        ids = np.flatnonzero(self.first_step)
        fresh_pos, fresh_quat = self.fresh_oracle()
        post_forward_pos_error = float(np.max(np.abs(fresh_pos[ids]-self.oracle_pos[ids])))
        post_forward_quat_error = float(np.max(np.minimum(
            np.linalg.norm(fresh_quat[ids]-self.oracle_quat[ids], axis=-1),
            np.linalg.norm(fresh_quat[ids]+self.oracle_quat[ids], axis=-1))))
        if post_forward_pos_error >= TOLERANCE or post_forward_quat_error >= TOLERANCE:
            self.failures.append("Reset-time PhysX oracle changed after native sim.forward")
        self.events.append({"phase": "explicit_reset_return", "state": self.state(),
                            "check": self.compare_cache(ids, "reset_return"),
                            "native_forward_oracle_position_error": post_forward_pos_error,
                            "native_forward_oracle_quaternion_error": post_forward_quat_error})

    def check_observations(self, observations: Mapping[str, Any]) -> None:
        assert array(observations["policy"]).shape == (self.env.num_envs, 815)
        assert array(observations["critic"]).shape == (self.env.num_envs, 289)

    def action(self) -> Any:
        term = self.env.action_manager.get_term("joint_pos")
        target = self.command.joint_pos.clone()[:, term._joint_ids]
        limits = self.robot.data.soft_joint_pos_limits[:, term._joint_ids]
        target = self.torch.clamp(target, limits[..., 0], limits[..., 1])
        scale = self.torch.as_tensor(term._scale, device=self.env.device)
        assert bool((scale.abs() > 1.0e-12).all())
        result = (target-term._offset)/scale
        assert result.shape == (self.env.num_envs, 29) and bool(self.torch.isfinite(result).all())
        self.events.append({"phase": "action", "label": self.label, "kind": "reference_joint_PD_soft_clamped",
                            "raw": array(result).tolist(), "joint_target": array(target).tolist(),
                            "live_scale": array(scale).tolist(), "live_offset": array(term._offset).tolist(),
                            "checkpoint_loaded": False})
        return result

    def step(self) -> None:
        assert self.step_index < 12, "Hard control-step budget exhausted"
        self.step_index += 1
        before_times = array(self.command.time_steps)
        observations, rewards, terminated, truncated, _extras = self.env.step(self.action())
        self.check_observations(observations)
        assert array(rewards).shape == (self.env.num_envs,)
        reset_rows = array(terminated) | array(truncated)
        after_times = array(self.command.time_steps)
        expected_times = np.where(reset_rows, 0, before_times+1)
        frame_pass = bool(np.array_equal(after_times, expected_times))
        if not frame_pass:
            self.failures.append("Natural command frame transition differs from reset0/advance1")
        self.oracle_pos, self.oracle_quat = self.fresh_oracle()
        self.events.append({"phase": "step_return", "state": self.state(), "terminated": array(terminated).tolist(),
                            "truncated": array(truncated).tolist(), "frame_expected": expected_times.tolist(),
                            "frame_pass": frame_pass, "cache_check": self.compare_cache(np.arange(self.env.num_envs), "step_return")})
        if self.failures:
            raise AssertionError("; ".join(self.failures))


def native_main(args: argparse.Namespace) -> int:
    assert args.task in TASKS and 4 <= args.num_envs <= 8
    assert args.motion is not None and args.output_dir is not None
    motion = Path(args.motion).resolve()
    manifest = Path(args.dev_manifest).resolve() if args.dev_manifest else motion.parent/"DEV3_INPUTS.json"
    entries = validate_dev_view(motion, manifest)
    output = Path(args.output_dir).absolute()
    output.mkdir(parents=True, exist_ok=False)  # Before AppLauncher: refuse any prior output.
    exclusive_json(output/"REQUEST.json", {"status": "PREPARED_NATIVE_NOT_RUN", "task": args.task,
                   "num_envs": args.num_envs, "seed": args.seed, "motion": str(motion), "DEV_inputs": entries,
                   "script_sha256": sha256(Path(__file__)), "hard_control_step_cap": 12,
                   "training_updates": 0, "conversion_executed": False, "checkpoint_policy_test": False,
                   "episode_length_s_override": 0.08, "override_reason": "diagnostic natural time_out auto reset",
                   "physics_substeps_cap_per_environment": 48})
    # Simulation-specific imports stay after the explicit native branch.
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    env = None
    observer = None
    try:
        import torch
        import gymnasium as gym
        import whole_body_tracking.tasks  # noqa: F401: register actual profiles
        from isaaclab_tasks.utils import parse_env_cfg
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        cfg.seed = args.seed
        cfg.episode_length_s = 0.08
        cmd = cfg.commands.motion
        original = {name: repr(getattr(cmd, name)) for name in
                    ("pose_range", "velocity_range", "joint_position_range", "start_from_beginning", "random_init_frame",
                     "resample_motions_every_s", "motion_dataset_load_cap", "motion_dataset_shard_across_gpus")}
        cmd.motion, cmd.debug_vis = str(motion), False
        cmd.start_from_beginning, cmd.start_frame, cmd.random_init_frame = True, 0, False
        cmd.pose_range = {name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")}
        cmd.velocity_range = {name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")}
        cmd.joint_position_range = (0.0, 0.0)
        cmd.resample_motions_every_s = 1.0e9
        cmd.motion_dataset_load_cap, cmd.max_active_motions = len(entries), len(entries)
        cmd.motion_dataset_shard_across_gpus = False
        cmd.motion_dataset_log_wandb_summary = False
        # Existing native group assignment gives every read-only DEV class
        # environments; do not retry random draws until coverage looks good.
        cmd.motion_groups = {entry["class"]: [Path(entry["view"]).name] for entry in entries}
        cmd.motion_group_sampling_ratios = {entry["class"]: 1.0/len(entries) for entry in entries}
        cfg.events.push_robot = None
        cfg.observations.policy.enable_corruption = False
        cfg.observations.critic.enable_corruption = False
        assert math.isclose(cfg.decimation*cfg.sim.dt, 0.02, abs_tol=1.0e-9)
        assert cfg.decimation == 4
        exclusive_json(output/"CONFIG.json", {"command_original": original,
                       "effective_config": json.loads(json.dumps(cfg.to_dict(), default=str)),
                       "declared_changes": ["4-step diagnostic episode", "reset pose/velocity/joint perturbations zero",
                                            "push disabled", "observation corruption disabled", "3 read-only DEV motions",
                                            "native group ratios give each DEV class environments",
                                            "start frame zero", "no remap within budget"],
                       "startup_material_default_joint_and_COM_randomization": "preserved with recorded seed"})
        env = gym.make(args.task, cfg=cfg).unwrapped
        assert env.num_envs == args.num_envs and env.action_manager.total_action_dim == 29
        binding = source_binding()
        observer = ResetObserver(env, torch, output)
        exclusive_json(output/"PROVENANCE.json", {"source_commit": subprocess.check_output(
                       ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "loaded_modules": binding,
                       "script_sha256": sha256(Path(__file__)), "body_names": observer.native_names,
                       "tracked_body_names": observer.body_names, "motion_paths_loaded": [str(p) for p in observer.paths],
                       "PhysX_COMs": array(observer.robot.root_physx_view.get_coms()).tolist(),
                       "action_kind": "reference_joint_PD_soft_clamped", "learned_policy_loaded": False,
                       "reward_definitions_and_termination_thresholds": "unchanged",
                       "oracle_anchor_timing": "frozen at reset; reward uses previous command-compute anchor",
                       "oracle_backend": "direct PhysX get_link_transforms, no forced FK call",
                       "partial_reset_observation_history": "native env.reset computes update_history=True for all rows"})
        observer.label = "cold_reset"
        obs, _extras = env.reset(seed=args.seed)
        observer.observe_reset_complete(obs)
        initial_ids = array(observer.command.env_motion_indices)
        assert set(initial_ids.tolist()) == set(range(len(entries))), "All selected DEV clips must actually be assigned"
        observer.step()
        observer.label = "explicit_partial_reset"
        ids = torch.arange(0, args.num_envs, 2, dtype=torch.long, device=env.device)
        obs, _extras = env.reset(env_ids=ids)
        observer.observe_reset_complete(obs)
        observer.step()
        observer.label = "explicit_full_reset"
        obs, _extras = env.reset()
        observer.observe_reset_complete(obs)
        observer.step()
        observer.label = "natural_timeout_auto_reset"
        for _ in range(3):
            observer.step()
        timeout_observed = any(any(event["terms"].get("time_out", [])) for event in observer.events
                               if event["phase"] == "termination" and event["before"]["label"] == observer.label)
        observer.label = "post_auto_reset_first_step"
        observer.step()
        assert observer.counts["reward"] == observer.counts["termination"] == observer.step_index == 7
        assert not observer.failures
        save_raw_tensors(output/"native_tensors.npz", observer.events)
        exclusive_json(output/"native_receipt.json", {"status": "PASS" if timeout_observed else "CACHE_PASS_TIMEOUT_COVERAGE_MISSING",
                       "cache_and_first_reward_done_oracle_pass": True, "natural_timeout_auto_reset_observed": timeout_observed,
                       "control_steps": observer.step_index, "counts": observer.counts, "events": observer.events,
                       "genuine_physical_failure_is_not_cache_failure": True,
                       "genuine_termination_events": [event for event in observer.events if event["phase"] == "termination"
                                                       and any(any(value) for key, value in event["terms"].items()
                                                               if key not in ("time_out", "motion_end"))],
                       "training_updates": 0, "conversion_executed": False,
                       "raw_tensor_file": "native_tensors.npz", "raw_tensor_sha256": sha256(output/"native_tensors.npz"),
                       "motion_end_boundary": "NOT_TESTED: auto reset uses declared short-episode native time_out",
                       "learned_teacher_control_performance": "NOT_TESTED"})
        return 0 if timeout_observed else 2
    except BaseException as error:
        if observer is not None and not (output/"native_tensors.npz").exists():
            save_raw_tensors(output/"native_tensors.npz", observer.events)
        exclusive_json(output/"FAILURE.json", {"status": "FAIL", "error": repr(error), "traceback": traceback.format_exc(),
                       "control_steps": observer.step_index if observer else 0,
                       "events": observer.events if observer else [], "training_updates": 0, "conversion_executed": False})
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def self_test() -> dict[str, Any]:
    for count in (1, 4, 8):
        reference_pos = np.zeros((count, 14, 3), dtype=np.float64)
        reference_pos[:, 7, 2] = 0.7
        reference_pos[:, 10, 0] = 0.3
        quat = np.zeros((count, 14, 4), dtype=np.float64)
        quat[..., 0] = 1.0
        anchor_pos = np.tile([2.0, -3.0, 9.0], (count, 1))
        anchor_quat = np.tile([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], (count, 1))
        pos, ori = relative_oracle(reference_pos, quat, anchor_pos, anchor_quat, 7)
        assert np.allclose(pos[:, 7], [2.0, -3.0, 0.7])
        assert np.allclose(pos[:, 10], [2.0, -2.7, 0.0])
        assert np.allclose(position_reward(pos, pos, 0.3, list(range(14))), 1.0)
        assert np.allclose(orientation_reward(ori, -ori, 0.4, list(range(14))), 1.0)
        actual = pos.copy()
        actual[:, 10, 2] += 0.4
        assert z_termination(pos, actual, 0.25, [10, 13]).all()
        assert not z_termination(pos, pos, 0.25, [10, 13]).any()
        frozen = pos.copy()
        anchor_after_physics = anchor_pos.copy()
        anchor_after_physics[:, 0] += 0.1
        fresh_after_physics, _ = relative_oracle(reference_pos, quat, anchor_after_physics, anchor_quat, 7)
        assert not np.array_equal(frozen, fresh_after_physics)
        assert (position_reward(frozen, fresh_after_physics, 0.3, list(range(14))) < 1.0).all()
        if count > 1:
            cache = np.zeros_like(pos)
            untouched = cache[1::2].copy()
            identity = id(cache)
            cache[::2] = pos[::2]
            assert id(cache) == identity and np.array_equal(cache[1::2], untouched)
    assert "isaaclab" not in sys.modules and "torch" not in sys.modules
    return {"status": "PASS", "batch_sizes": [1, 4, 8], "tracked_bodies": 14,
            "quaternion_convention": "WXYZ", "frozen_reset_anchor_vs_postphysics_drift": "PASS",
            "quaternion_sign_invariance": "PASS", "partial_row_identity": "PASS",
            "native_execution": "NOT_RUN", "GPU_or_torch_imported": False}


def main() -> int:
    if "--self-test" in sys.argv:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--dev_manifest")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20261015)
    # Let AppLauncher define --device/--headless and its native runtime flags.
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return native_main(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
