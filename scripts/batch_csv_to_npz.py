"""Batch convert CSV motions in a folder to NPZ using the same pipeline as csv_to_npz.py.

This script launches one Omniverse/Isaac session, sets up the same robot/scene,
then iterates all matching CSV files in the input folder and writes NPZ files
with a user-specified prefix added to the base filename.
The output directory must be new. Default link_v2 preserves CSV link-origin
kinematics and records native COM velocities. Use --self-test for CPU only.
"""

from __future__ import annotations

import os
import sys


def _sanitize_python_path_for_isaac() -> None:
    """Avoid binary incompatibility by preventing mixed user-site packages.

    Typical failure mode:
    - numpy is imported from ~/.local/... (user site)
    - then IsaacSim/Kit loads another bundled numpy component (pip_prebundle)
    - results in: ValueError: numpy.dtype size changed (binary incompatibility)

    We aggressively remove user-site paths from sys.path before importing isaaclab/isaacsim.
    """

    # Best-effort: also propagate to any subprocesses (doesn't affect current interpreter startup).
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    try:
        import site  # noqa: WPS433 (runtime import by design)

        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    def _is_user_site_path(p: str) -> bool:
        if not p:
            return False
        # canonical user site
        if isinstance(user_site, str) and p == user_site:
            return True
        # common user-site patterns
        if "/.local/lib/python" in p and "site-packages" in p:
            return True
        return False

    sys.path[:] = [p for p in sys.path if not _is_user_site_path(p)]

    # If numpy got imported earlier for any reason, force re-import after sanitization.
    if "numpy" in sys.modules:
        del sys.modules["numpy"]


_sanitize_python_path_for_isaac()

if '--self-test' in sys.argv:
    if sys.argv[1:] != ['--self-test']:
        raise SystemExit('--self-test is CPU-only and must be used alone')
    from root_velocity_contract import self_test

    self_test()
    raise SystemExit(0)

import argparse
import os
from pathlib import Path
from typing import Any, Mapping
import numpy as np

from tqdm import tqdm

from root_velocity_contract import (
    MODES, conversion_metadata, plan_batch_outputs, reserve_output_directory,
    save_npz_exclusive, validate_mode, write_csv_root_state,
)
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Batch convert CSV motions in a folder to NPZ format.")
parser.add_argument("--input_dir", type=str, required=True, help="Folder containing input CSV motion files.")
parser.add_argument("--file_pattern", type=str, default="*.csv", help="Glob pattern to match input files.")
parser.add_argument("--output_prefix", type=str, required=True, help='Prefix to add to output names, e.g. "g1_lafan".')
parser.add_argument("--output_dir", type=str, required=True, help="New output directory; an existing directory is always refused.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motions.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motions.")
parser.add_argument('--root_velocity_mode', choices=MODES, default='link_v2',
                    help='link_v2 transforms link velocity to native COM velocity; legacy_com_v1 preserves the old mismatch only in a new output directory.')
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Optional global frame range applied to every file. Index starts from 1. If omitted, loads all frames.",
)
parser.add_argument(
    "--robot",
    type=str,
    default="g1",
    help="Robot platform name (e.g. g1, h1_2). See whole_body_tracking.robots.robot_registry.ROBOT_PLATFORMS.",
)
parser.add_argument(
    "--bones_seed_g1_csv",
    action="store_true",
    help=(
        "BONES-SEED / named-column G1 CSV: skip header, drop Frame column, root euler XYZ (deg), joints in degrees."
    ),
)
parser.add_argument("--csv_skip_header", action="store_true", help="Skip the first line (column names).")
parser.add_argument(
    "--csv_drop_frame_column",
    action="store_true",
    help="Drop the first numeric column (e.g. Frame index) after loading.",
)
parser.add_argument(
    "--root_rotation",
    type=str,
    choices=("quat_xyzw", "euler_xyz_deg"),
    default="quat_xyzw",
    help="Root orientation: quaternion xyzw (cols 3-6) or euler XYZ degrees (cols 3-5).",
)
parser.add_argument(
    "--joint_angles_in_degrees",
    action="store_true",
    help="Convert joint DOF columns from degrees to radians.",
)
parser.add_argument(
    "--position_scale",
    type=float,
    default=0.01,
    help="Multiply root translation (x,y,z) by this factor (e.g. 0.01 if positions are in centimeters).",
)
parser.add_argument("--log_wandb", action="store_true", help="If set, log each NPZ to wandb as an artifact.")
parser.add_argument(
    "--min_duration",
    type=float,
    default=3.0,
    help="Minimum duration (seconds) for motions to be converted. Motions shorter than this will be skipped. Default: 3.0",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.bones_seed_g1_csv:
    args_cli.csv_skip_header = True
    args_cli.csv_drop_frame_column = True
    args_cli.root_rotation = "euler_xyz_deg"
    args_cli.joint_angles_in_degrees = True

# Freeze the entire output plan and refuse all collisions before native startup.
if args_cli.input_fps <= 0 or args_cli.output_fps <= 0 or not np.isfinite(args_cli.position_scale) or args_cli.position_scale <= 0:
    parser.error('FPS and position_scale must be positive and finite')
if not np.isfinite(args_cli.min_duration) or args_cli.min_duration < 0:
    parser.error('min_duration must be nonnegative and finite')
if args_cli.frame_range is not None and not (1 <= args_cli.frame_range[0] <= args_cli.frame_range[1]):
    parser.error('frame_range requires 1 <= START <= END')
input_dir = Path(args_cli.input_dir).expanduser().resolve(strict=True)
if not input_dir.is_dir():
    parser.error('input_dir must be a directory')
output_dir = Path(args_cli.output_dir).expanduser().absolute()
csv_files = sorted(path for path in input_dir.rglob(args_cli.file_pattern) if path.is_file())
if not csv_files:
    parser.error('No CSV files matched; no output directory or native application was created')
output_plan = plan_batch_outputs(input_dir, csv_files, output_dir, args_cli.output_prefix)
output_run = reserve_output_directory(output_dir, validate_mode(args_cli.root_velocity_mode))

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from motion_csv_loader import MotionLoader
from whole_body_tracking.robots.robot_registry import available_robot_names, get_robot_platform


def create_replay_scene_cfg(robot_cfg: ArticulationCfg) -> type[InteractiveSceneCfg]:
    """Create a scene configuration for replaying motions with the specified robot."""

    @configclass
    class ReplayMotionsSceneCfg(InteractiveSceneCfg):
        """Configuration for a replay motions scene."""

        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )

        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    return ReplayMotionsSceneCfg


def setup_sim_and_scene(robot_cfg: ArticulationCfg) -> tuple[SimulationContext, InteractiveScene]:
    """Setup simulation and scene with one robot."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg_class = create_replay_scene_cfg(robot_cfg)
    scene_cfg = scene_cfg_class(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    return sim, scene


def process_one_file(
    sim: SimulationContext,
    scene: InteractiveScene,
    csv_path: str,
    output_npz_path: str,
    output_name: str,
    joint_names: list[str],
    conversion_run: Mapping[str, Any],
) -> float | None:
    """Replay one CSV and save NPZ (and optionally log to wandb).
    
    Returns:
        Motion duration in seconds if successful, None if skipped (too short).
    """
    # robot and joint indices
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    assert len(robot_joint_indexes) == len(joint_names)
    options = {name:getattr(args_cli,name) for name in ('robot','input_fps','output_fps','frame_range',
               'csv_skip_header','csv_drop_frame_column','root_rotation','joint_angles_in_degrees','position_scale')}
    options['source_joint_names'] = joint_names
    metadata = conversion_metadata(Path(csv_path), options, conversion_run, robot, Path(__file__))

    # motion loader
    motion = MotionLoader(
        motion_file=csv_path,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=tuple(args_cli.frame_range) if args_cli.frame_range is not None else None,
        csv_skip_header=args_cli.csv_skip_header,
        csv_drop_frame_column=args_cli.csv_drop_frame_column,
        root_rotation=args_cli.root_rotation,
        joint_angles_in_degrees=args_cli.joint_angles_in_degrees,
        position_scale=args_cli.position_scale,
        expected_dof_count=len(joint_names),
    )
    
    # Check duration threshold
    if motion.duration < args_cli.min_duration:
        print(f"[SKIP]: Motion duration ({motion.duration:.2f}s) < min_duration ({args_cli.min_duration:.2f}s), skipping: {csv_path}")
        return None

    # logger buffers
    log: dict[str, Any] = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    # iterate the whole motion once
    for (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
    ) in motion.iter_states():
        # root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        write_csv_root_state(robot, root_states, validate_mode(str(conversion_run['root_velocity_mode'])))

        # joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        assert joint_pos.shape == joint_vel.shape == robot.data.default_joint_pos.shape
        assert bool(torch.isfinite(joint_pos).all()) and bool(torch.isfinite(joint_vel).all())
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # no physics stepping; only render and update scene
        sim.render()
        scene.update(sim.get_physics_dt())

        # record one frame
        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_com_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

    # stack and save
    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)
    save_npz_exclusive(Path(output_npz_path), log, metadata)
    print(f"[INFO]: Saved NPZ: {output_npz_path}")

    if args_cli.log_wandb:
        import wandb

        COLLECTION = output_name
        run = wandb.init(project="csv_to_npz", name=COLLECTION, reinit=True)
        print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
        REGISTRY = "motions"
        logged_artifact = run.log_artifact(artifact_or_path=output_npz_path, name=COLLECTION, type=REGISTRY)
        run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
        print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")
        run.finish()
    
    return motion.duration


def main() -> None:
    # Resolve robot platform from registry
    try:
        robot_platform = get_robot_platform(args_cli.robot)
    except KeyError as e:
        raise SystemExit(f"{e}\nAvailable robots: {available_robot_names()}") from None

    sim, scene = setup_sim_and_scene(robot_platform.cfg)
    durations: list[float] = []
    skipped_count = 0
    try:
        all_files_length = len(csv_files)
        for i, (csv_path, output_npz_path) in enumerate(tqdm(output_plan, desc="Converting motions", unit="file")):
            output_name = output_npz_path.stem
            print(f"[INFO]: Converting {i+1}/{all_files_length}: {csv_path} -> {output_npz_path}")
            duration = process_one_file(sim, scene, str(csv_path), str(output_npz_path), output_name,
                                        robot_platform.joint_names, output_run)
            if duration is not None:
                durations.append(duration)
            else:
                skipped_count += 1
        print(f"[INFO]: Skipped {skipped_count} motions (duration < {args_cli.min_duration}s)")
    finally:
        # Print statistics before closing simulator (simulation_app.close() may exit the program)
        print("\n" + "=" * 80)
        print("CONVERSION STATISTICS")
        print("=" * 80)
        print(f"Total files processed: {all_files_length}")
        print(f"Successfully converted: {len(durations)}")
        print(f"Skipped (duration < {args_cli.min_duration}s): {skipped_count}")
        
        if len(durations) > 0:
            durations_array = np.array(durations)
            mean_duration = float(np.mean(durations_array))
            std_duration = float(np.std(durations_array))
            median_duration = float(np.median(durations_array))
            min_duration = float(np.min(durations_array))
            max_duration = float(np.max(durations_array))
            
            print(f"\nMotion Duration Statistics (seconds):")
            print(f"  Mean:   {mean_duration:.3f}")
            print(f"  Std:    {std_duration:.3f}")
            print(f"  Median: {median_duration:.3f}")
            print(f"  Min:    {min_duration:.3f}")
            print(f"  Max:    {max_duration:.3f}")
        else:
            print("\nNo motions were successfully converted.")
        print("=" * 80)
        
        # close sim app (this may exit the program, so do it last)
        # Unsubscribe IsaacLab's STOP rendering callback before stage close.
        # This runs after conversion; it does not step or modify motion data.
        try:
            SimulationContext.clear_instance()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    main()


'''
python /home/lsn/MOSAIC/scripts/batch_csv_to_npz.py \
  --input_dir /home/lsn/Datasets/bones-seed/g1/csv_by_gpu/0 \
  --output_dir /home/lsn/Datasets/bones-seed/g1/npz \
  --output_prefix bones_seed_g1 \
  --input_fps 120 \
  --output_fps 50 \
  --bones_seed_g1_csv \
  --headless
'''
