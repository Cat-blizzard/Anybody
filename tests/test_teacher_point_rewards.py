"""CPU tests of real reward functions and real profile/registration ASTs.

The command fixture provides (B,14,3) world link positions and (B,14,4)
WXYZ orientations. Synthetic errors test reward semantics, not feasible robot
trajectories. Simulator-only configuration constructors are narrow fixtures;
new/legacy post_init methods and registered reward functions are actual code.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
TRACKING = REPO / "source/whole_body_tracking/whole_body_tracking/tasks/tracking"
MDP = TRACKING / "mdp"
spec = importlib.util.spec_from_file_location("teacher_point_contract_CPU", MDP / "teacher_point_contract.py")
assert spec is not None and spec.loader is not None
points = importlib.util.module_from_spec(spec); spec.loader.exec_module(points)


def compile_nodes(nodes: list[ast.stmt], namespace: dict[str, Any], source: Path) -> None:
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(source), "exec"), namespace)


def reward_namespace() -> dict[str, Any]:
    path = MDP / "rewards.py"
    wanted = {"_get_body_indexes", "teleop_body_position_extend", "teleop_vr_3point",
              "teleop_body_position_extend_g1_v1", "teleop_vr_3point_g1_v1"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert len(nodes) == len(wanted)
    namespace: dict[str, Any] = {"torch": torch, "teacher_points": points}
    compile_nodes(nodes, namespace, path)
    return namespace


def config_namespace(functions: dict[str, Any]) -> dict[str, Any]:
    mdp = SimpleNamespace(**{name: function for name, function in functions.items() if callable(function)})
    source = TRACKING / "tracking_env_cfg.py"
    tree = ast.parse(source.read_text())
    expert = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RewardsExpertCfg")
    for call in ast.walk(expert):
        if isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name) and call.value.id == "mdp":
            if not hasattr(mdp, call.attr):
                setattr(mdp, call.attr, lambda *args, **kwargs: None)
    namespace: dict[str, Any] = {
        "mdp": mdp, "teacher_points": points, "configclass": lambda cls: cls,
        "RewTerm": lambda **kwargs: SimpleNamespace(**{"params": {}, **kwargs}),
        "SceneEntityCfg": lambda *args, **kwargs: SimpleNamespace(args=args, **kwargs),
    }
    compile_nodes([expert], namespace, source)
    one_stage = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OneStageTrackingEnvCfg")
    post = next(node for node in one_stage.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    weight_nodes = [node for node in post.body if isinstance(node, ast.Assign)
                    and ast.unparse(node.targets[0]).startswith("self.rewards.")]

    class OneStageFixture:
        def __post_init__(self) -> None:
            self.scene = SimpleNamespace(robot=None)
            self.actions = SimpleNamespace(joint_pos=SimpleNamespace(scale=None))
            self.commands = SimpleNamespace(motion=SimpleNamespace(anchor_body_name=None, body_names=[]))
            self.rewards = SimpleNamespace(**{
                name: copy.deepcopy(value) for name, value in vars(namespace["RewardsExpertCfg"]).items()
                if isinstance(value, SimpleNamespace)
            })
            compile_nodes(weight_nodes, {"self": self}, source)

    namespace.update({"OneStageTrackingEnvCfg": OneStageFixture,
                      "G1_CYLINDER_CFG": SimpleNamespace(replace=lambda **kwargs: ("G1_asset_fixture", kwargs)),
                      "G1_ACTION_SCALE": object()})
    path = TRACKING / "config/g1/flat_env_cfg.py"
    names = {"G1OneStageTrackingEnvCfg", "G1OneStageWristRewardV1EnvCfg", "G1OneStagePointRewardsV1EnvCfg"}
    classes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(classes) == 3
    compile_nodes(classes, namespace, path)
    return namespace


class PointRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.functions = reward_namespace()
        ns = config_namespace(self.functions)
        cfg = ns["G1OneStageTrackingEnvCfg"](); cfg.__post_init__()
        self.names = cfg.commands.motion.body_names
        self.reference = torch.zeros(3, 14, 3, dtype=torch.float64)
        self.reference[:, self.names.index("torso_link"), 2] = .7
        self.orientation = torch.zeros(3, 14, 4, dtype=torch.float64); self.orientation[..., 0] = 1
        self.command = SimpleNamespace(
            cfg=SimpleNamespace(body_names=self.names), body_pos_w=self.reference,
            robot_body_pos_w=self.reference.clone(), body_quat_w=self.orientation,
            robot_body_quat_w=self.orientation.clone(),
        )
        self.env = SimpleNamespace(num_envs=3, device="cpu", command_manager=SimpleNamespace(get_term=lambda _: self.command))

    def new_VR(self) -> torch.Tensor:
        return self.functions["teleop_vr_3point_g1_v1"](
            self.env, "motion", .5, points.g1_head_offset_from_urdf(), points.G1_HEAD_POINT_V1,
        )

    def test_perfect_match_and_wrist_error(self) -> None:
        old_ext = self.functions["teleop_body_position_extend"]
        new_ext = self.functions["teleop_body_position_extend_g1_v1"]
        assert torch.all(new_ext(self.env, "motion") == 2)
        assert torch.all(self.new_VR() == 1)
        for side in ("left", "right"):
            self.command.robot_body_pos_w = self.reference.clone()
            self.command.robot_body_pos_w[:, self.names.index(side+"_wrist_yaw_link"), 0] = 1
            assert torch.all(old_ext(self.env, "motion", .5, .5, 1, 1) == 2)
            assert torch.all(self.functions["teleop_vr_3point"](self.env, "motion", .5) == 0)
            assert torch.allclose(new_ext(self.env, "motion"), torch.full((3,), 1+math.exp(-1/4.5), dtype=torch.float64))
            assert torch.allclose(self.new_VR(), torch.full((3,), math.exp(-1/2.25), dtype=torch.float64))

    def test_head_orientation_has_real_response(self) -> None:
        # Same torso origin, different torso pitch: the virtual head must move.
        q = self.command.robot_body_quat_w
        q[:, self.names.index("torso_link")] = q.new_tensor([math.cos(.15), 0, math.sin(.15), 0])
        reward = self.new_VR()
        assert torch.all(reward < 1) and torch.all(reward > 0)

    def test_batch1_reordered_axes_and_translation(self) -> None:
        baseline = self.new_VR()
        shift = self.reference.new_tensor([5, -2, .4])
        self.command.body_pos_w = self.reference + shift
        self.command.robot_body_pos_w = self.reference + shift
        assert torch.equal(self.new_VR(), baseline)
        order = list(reversed(range(14)))
        self.command.cfg.body_names = [self.names[index] for index in order]
        for name in ("body_pos_w", "robot_body_pos_w", "body_quat_w", "robot_body_quat_w"):
            setattr(self.command, name, getattr(self.command, name)[:1, order])
        self.env.num_envs = 1
        assert self.new_VR().shape == (1,) and self.new_VR().item() == 1
        assert self.functions["teleop_body_position_extend_g1_v1"](self.env, "motion").item() == 2

    def test_invalid_names_shape_values_and_head_contract(self) -> None:
        self.command.cfg.body_names = self.names[:-1]
        with self.assertRaises((ValueError, AssertionError)):
            self.functions["teleop_body_position_extend_g1_v1"](self.env, "motion")
        self.command.cfg.body_names = self.names
        with self.assertRaises(ValueError):
            self.functions["teleop_vr_3point_g1_v1"](self.env, "motion", .5, (0., 0., 0.), "torso_proxy")
        with self.assertRaises(AssertionError):
            self.functions["teleop_vr_3point_g1_v1"](self.env, "motion", .5, (0., 0., 0.), points.G1_HEAD_POINT_V1)
        self.command.robot_body_pos_w[0, 0, 0] = float("nan")
        with self.assertRaises(AssertionError):
            self.functions["teleop_body_position_extend_g1_v1"](self.env, "motion")

    def test_config_profiles_change_only_declared_reward_fields(self) -> None:
        ns = config_namespace(self.functions)
        configs = []
        for name in ("G1OneStageTrackingEnvCfg", "G1OneStageWristRewardV1EnvCfg", "G1OneStagePointRewardsV1EnvCfg"):
            cfg = ns[name](); cfg.__post_init__(); configs.append(cfg)
        old, wrist, three = configs
        for candidate in (wrist, three):
            assert candidate.commands.motion.body_names == old.commands.motion.body_names
            assert set(vars(candidate.rewards)) == set(vars(old.rewards))
            for name in vars(old.rewards):
                previous = getattr(old.rewards, name); repaired = getattr(candidate.rewards, name)
                assert previous.weight == repaired.weight
                if name not in ("teleop_vr_3point", "teleop_body_position_extend"):
                    assert vars(previous) == vars(repaired)
                if name != "teleop_vr_3point" or candidate is wrist:
                    assert previous.params == repaired.params
        assert old.rewards.teleop_vr_3point.func is wrist.rewards.teleop_vr_3point.func
        assert three.rewards.teleop_vr_3point.params["head_point_definition"] == points.G1_HEAD_POINT_V1
        assert three.rewards.teleop_vr_3point.weight == .5
        assert three.rewards.teleop_body_position_extend.weight == 1
        # Execute the actual registration loop in a CPU registry fixture.
        path = TRACKING / "config/g1/__init__.py"
        register = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "_register_g1_tasks")
        loops = [node for node in register.body if isinstance(node, ast.For)]
        assert len(loops) == 1
        registry: list[dict[str, Any]] = []
        compile_nodes(loops, {"gym": SimpleNamespace(register=lambda **kw: registry.append(kw)),
                              "flat_env_cfg": SimpleNamespace(**{name:ns[name] for name in ns if name.startswith("G1OneStage")}),
                              "agents": SimpleNamespace(__name__="agents")}, path)
        assert [item["id"] for item in registry] == ["General-Tracking-Flat-G1-Teacher-Cache-v1", "General-Tracking-Flat-G1-Teacher-Wrists-v1", "General-Tracking-Flat-G1-Teacher-Points-v1"]
        assert all(item["kwargs"]["rsl_rl_cfg_entry_point"].endswith(":G1FlatPPORunnerCfg") for item in registry)


if __name__ == "__main__":
    torch.set_num_threads(1)
    points.sanity_check()
    unittest.main()
