"""Replay CSV into a fresh NPZ directory using the versioned link/COM contract.

Run ``python csv_to_npz.py --self-test`` for a CPU-only test without Isaac.
The default link_v2 uses link-origin velocity input and native COM velocity
output. Each invocation requires a previously nonexistent output directory.

.. code-block:: bash

    # LAFAN-style (root quat xyzw + joint radians, no header). Headless is the default (servers).
    python csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
        --output_name ./motions/dance1_subject2 --output_fps 50

    # BONES-SEED G1 CSV @ 120 fps
    python csv_to_npz.py --input_file motion.csv --input_fps 120 --bones_seed_g1_csv \
        --output_name ./motions/motion --output_fps 50
"""

from __future__ import annotations

import os
import sys


def _sanitize_python_path_for_isaac() -> None:
    """Avoid binary incompatibility by preventing mixed user-site packages.

    See batch_csv_to_npz.py for the detailed rationale.
    """

    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    try:
        import site  # noqa: WPS433 (runtime import by design)

        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    def _is_user_site_path(p: str) -> bool:
        if not p:
            return False
        if isinstance(user_site, str) and p == user_site:
            return True
        if "/.local/lib/python" in p and "site-packages" in p:
            return True
        return False

    sys.path[:] = [p for p in sys.path if not _is_user_site_path(p)]

    if "numpy" in sys.modules:
        del sys.modules["numpy"]


_sanitize_python_path_for_isaac()

# This path must finish before importing AppLauncher or launching Isaac/Kit.
if '--self-test' in sys.argv:
    if sys.argv[1:] != ['--self-test']:
        raise SystemExit('--self-test is CPU-only and must be used alone')
    from root_velocity_contract import self_test

    self_test()
    raise SystemExit(0)

import argparse
from pathlib import Path
from typing import Any
import numpy as np

from root_velocity_contract import (
    MODES, conversion_metadata, npz_output_path, reserve_output_directory,
    save_npz_exclusive, validate_mode, write_csv_root_state,
)
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="NPZ path whose parent must be a new directory; existing directories are refused.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument('--root_velocity_mode', choices=MODES, default='link_v2',
                    help='link_v2 correctly transforms link velocity to native COM velocity; legacy_com_v1 explicitly reproduces the old mismatch. Both require a new output directory.')
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
        "BONES-SEED / named-column G1 CSV: skip header row, drop leading Frame column, root euler XYZ in degrees,"
        " joint angles in degrees (matches Isaac quat_from_euler_xyz)."
    ),
)
parser.add_argument(
    "--csv_skip_header",
    action="store_true",
    help="Skip the first line (column names).",
)
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
    help="Root orientation: quaternion xyzw columns 3-6, or euler XYZ in degrees columns 3-5.",
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
parser.add_argument(
    "--no_headless",
    action="store_true",
    help="Run with a visible window (disables headless). Default is headless for servers.",
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

# Default headless on for server runs; use --no_headless for a local window
args_cli.headless = not args_cli.no_headless

# Refuse historical/output collisions before starting the native application.
if args_cli.input_fps <= 0 or args_cli.output_fps <= 0 or not np.isfinite(args_cli.position_scale) or args_cli.position_scale <= 0:
    parser.error('FPS and position_scale must be positive and finite')
if args_cli.frame_range is not None and not (1 <= args_cli.frame_range[0] <= args_cli.frame_range[1]):
    parser.error('frame_range requires 1 <= START <= END')
source_csv = Path(args_cli.input_file).expanduser().resolve(strict=True)
if not source_csv.is_file():
    parser.error('input_file must be a regular CSV file')
output_path = npz_output_path(args_cli.output_name)
output_run = reserve_output_directory(output_path.parent, validate_mode(args_cli.root_velocity_mode))

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Robot platform registry
##
from motion_csv_loader import MotionLoader
from whole_body_tracking.robots.robot_registry import available_robot_names, get_robot_platform


def create_replay_scene_cfg(robot_cfg: ArticulationCfg) -> type[InteractiveSceneCfg]:
    """Create a scene configuration for replaying motions with the specified robot."""

    @configclass
    class ReplayMotionsSceneCfg(InteractiveSceneCfg):
        """Configuration for a replay motions scene."""

        # ground plane
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        # lights
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )

        # articulation - set based on robot type
        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    return ReplayMotionsSceneCfg


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]) -> None:
    """Runs the simulation loop."""
    robot = scene['robot']
    options = {name:getattr(args_cli,name) for name in ('robot','input_fps','output_fps','frame_range',
               'csv_skip_header','csv_drop_frame_column','root_rotation','joint_angles_in_degrees','position_scale')}
    options['source_joint_names'] = joint_names
    metadata = conversion_metadata(source_csv, options, output_run, robot, Path(__file__))
    # Load motion without changing its link-origin position/velocity convention.
    motion = MotionLoader(
        motion_file=args_cli.input_file,
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

    # Extract scene entities
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    assert len(robot_joint_indexes) == len(joint_names)

    # ------- data logger -------------------------------------------------------
    log: dict[str, Any] = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        write_csv_root_state(robot, root_states, validate_mode(args_cli.root_velocity_mode))

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        assert joint_pos.shape == joint_vel.shape == robot.data.default_joint_pos.shape
        assert bool(torch.isfinite(joint_pos).all()) and bool(torch.isfinite(joint_vel).all())
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_com_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            save_npz_exclusive(output_path, log, metadata)
            print(f"[INFO]: Motion saved locally: {output_path}")
            return


def main() -> None:
    """Main function."""
    # Resolve robot platform from registry
    try:
        robot_platform = get_robot_platform(args_cli.robot)
    except KeyError as e:
        raise SystemExit(f"{e}\nAvailable robots: {available_robot_names()}") from None

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg_class = create_replay_scene_cfg(robot_platform.cfg)
    scene_cfg = scene_cfg_class(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
        joint_names=robot_platform.joint_names,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        # Unsubscribe IsaacLab's STOP rendering callback before stage close.
        # This runs after conversion; it does not step or modify motion data.
        try:
            SimulationContext.clear_instance()
        finally:
            simulation_app.close()

'''
python scripts/csv_to_npz.py --input_file /home/lsn/Datasets/bones-seed/g1/csv/210531/jump_and_land_heavy_001__A001_M.csv --input_fps 120 --bones_seed_g1_csv \
        --output_name ./motions/jump_and_land_heavy_001__A001_M --output_fps 50
'''
