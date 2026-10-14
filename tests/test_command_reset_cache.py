"""CPU regression checks of real reset/update methods, with no Isaac startup.

Compile unchanged method bodies from commands.py and CPU math bodies from the
installed IsaacLab source. The fixture models its documented write-invalidates /
read-performs-FK lifecycle. It is not a simulator or a robot performance test.
Contracts: q/qd=(N,29), root=(N,13), cached body pos=(N,3,3), quat=(N,3,4)
in wxyz, with N=1 or 3. Compare reset state/RNG against unmodified c319a66,
and compare cache geometry against an independent yaw matrix / Hamilton product.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any
from collections.abc import Sequence

import torch


REPO = Path(__file__).resolve().parents[1]
COMMANDS = REPO / 'source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py'
ISAAC = Path('/data/jxc/cache/nvidia/IsaacLab-v2.3.2/source/isaaclab/isaaclab')
MATH = ISAAC / 'utils/math.py'


class SyntheticCommandTerm:
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = torch.arange(self.num_envs) if env_ids is None else env_ids
        self._resample_command(ids)
        return {}


def source_namespace(source: str) -> dict[str, Any]:
    """Execute production methods directly; mock only environment/physics APIs."""
    namespace: dict[str, Any] = {'torch': torch, 'math': math, 'Sequence': Sequence,
                                 'CommandTerm': SyntheticCommandTerm}
    names = {'normalize', 'quat_conjugate', 'quat_inv', 'quat_mul', 'quat_apply',
             'quat_from_euler_xyz', 'yaw_quat', 'sample_uniform'}
    functions = []
    for node in ast.parse(MATH.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []  # JIT decoration needs full Isaac imports; body is unchanged.
            functions.append(node)
    math_module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(math_module), str(MATH), 'exec'), namespace)
    selected: list[ast.stmt] = []
    methods = {'_resample_command', '_update_command', 'reset', 'joint_pos', 'joint_vel',
               'body_pos_w', 'body_quat_w', 'anchor_pos_w', 'anchor_quat_w',
               'robot_anchor_pos_w', 'robot_anchor_quat_w'}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == '_refresh_reset_relative_body_poses':
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in ('MotionCommand', 'MultiMotionCommand', 'PartialMaskedMultiMotionCommand'):
            node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in methods]
            selected.append(node)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(COMMANDS), 'exec'), namespace)
    return namespace


def hamilton(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Independent wxyz product, used only as a geometry oracle."""
    assert left.shape == right.shape and left.shape[-1] == 4
    w, xyz = left[..., :1], left[..., 1:]
    v, uvw = right[..., :1], right[..., 1:]
    return torch.cat([w*v-(xyz*uvw).sum(-1, keepdim=True), w*uvw+v*xyz+torch.cross(xyz, uvw, dim=-1)], dim=-1)


def yaw_matrix(angle: torch.Tensor) -> torch.Tensor:
    assert angle.ndim == 1
    out = torch.zeros(len(angle), 3, 3)
    out[:, 0, 0] = out[:, 1, 1] = angle.cos()
    out[:, 0, 1], out[:, 1, 0], out[:, 2, 2] = -angle.sin(), angle.sin(), 1.0
    return out


class SyntheticRobot:
    """Write setters leave stale link buffers until a lazy FK getter is read."""
    def __init__(self, count: int, namespace: dict[str, Any]) -> None:
        self.count, self.ns = count, namespace
        self.q = torch.zeros(count, 29)
        self.qd = torch.zeros_like(self.q)
        self.root = torch.zeros(count, 13)
        self.root[:, 3] = 1.0
        self.dirty = False
        self.fk_reads = 0
        self.joint_writes = self.root_writes = 0
        self.cached_pos = torch.full((count, 3, 3), -123.0)
        self.cached_quat = torch.zeros(count, 3, 4)
        self.cached_quat[..., 0] = 1.0
        self.data = SyntheticRobotData(self)

    def write_joint_state_to_sim(self, q: torch.Tensor, qd: torch.Tensor, *, env_ids: torch.Tensor) -> None:
        assert q.shape == qd.shape == (len(env_ids), 29)
        self.q[env_ids], self.qd[env_ids] = q.clone(), qd.clone()
        self.dirty = True
        self.joint_writes += 1

    def write_root_state_to_sim(self, state: torch.Tensor, *, env_ids: torch.Tensor) -> None:
        assert state.shape == (len(env_ids), 13) and torch.isfinite(state).all()
        self.root[env_ids] = state.clone()
        self.dirty = True
        self.root_writes += 1

    def forward_kinematics(self) -> None:
        if not self.dirty:
            return
        self.fk_reads += 1
        local = torch.tensor([[0., 0., 0.], [.3, 0., .8], [.7, .1, 1.]]).repeat(self.count, 1, 1)
        local[:, 1, 0] += self.q[:, 0] * .05
        root_quat = self.root[:, None, 3:7].repeat(1, 3, 1)
        self.cached_pos = self.root[:, None, :3] + self.ns['quat_apply'](root_quat, local)
        local_q = torch.zeros(self.count, 3, 4)
        local_q[..., 0] = 1.
        local_q[:, 1, :] = self.ns['quat_from_euler_xyz'](torch.zeros(self.count), torch.zeros(self.count), self.q[:, 0]*.2)
        self.cached_quat = hamilton(root_quat, local_q)
        self.dirty = False


class SyntheticRobotData:
    def __init__(self, robot: SyntheticRobot) -> None:
        self.robot = robot
        self.soft_joint_pos_limits = torch.tensor([[[-.15, .15]] * 29]).repeat(robot.count, 1, 1)

    @property
    def body_pos_w(self) -> torch.Tensor:
        self.robot.forward_kinematics()
        return self.robot.cached_pos

    @property
    def body_quat_w(self) -> torch.Tensor:
        self.robot.forward_kinematics()
        return self.robot.cached_quat


def make_command(namespace: dict[str, Any], name: str, count: int, randomized: bool = False) -> Any:
    cls = namespace[name]
    cmd = cls.__new__(cls)
    cmd.num_envs, cmd.device = count, torch.device('cpu')
    cfg = SimpleNamespace(body_names=['pelvis', 'torso', 'wrist'], start_from_beginning=True, start_frame=2,
                          random_init_frame=False, reset_base_xy_to_origin=False,
                          pose_range={key: (-.1, .1) if randomized else (0., 0.) for key in ('x','y','z','roll','pitch','yaw')},
                          velocity_range={key: (0., 0.) for key in ('x','y','z','roll','pitch','yaw')},
                          joint_position_range=(-.01, .01) if randomized else (0.,0.), adaptive_alpha=.1)
    cmd.cfg = cfg
    cmd._env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.arange(count, dtype=torch.float32)[:,None].expand(count,3)*2),
                               episode_length_buf=torch.zeros(count, dtype=torch.long))
    cmd.robot = SyntheticRobot(count, namespace)
    cmd.robot_anchor_body_index = cmd.motion_anchor_body_index = 1
    cmd.time_steps = torch.arange(count, dtype=torch.long) % 4
    cmd.body_pos_relative_w = torch.full((count,3,3), 12345.)
    cmd.body_quat_relative_w = torch.zeros(count,3,4)
    cmd.body_quat_relative_w[...,0] = 1.
    length = 8
    root_pos = torch.stack([torch.tensor([.1*t, -.2*t, .5]) for t in range(length)])
    root_q = namespace['quat_from_euler_xyz'](torch.full((length,),.12),torch.full((length,),-.15),torch.arange(length)*.1)
    body_q = root_q[:,None,:].repeat(1,3,1)
    body_pos = root_pos[:,None,:] + namespace['quat_apply'](body_q,torch.tensor([[0.,0.,0.],[.3,0.,.8],[.7,.1,1.]]).repeat(length,1,1))
    cmd.motion = SimpleNamespace(time_step_total=length, root_pos_w=root_pos, root_quat_w=root_q,
                                root_lin_vel_w=torch.zeros(length,3),root_ang_vel_w=torch.zeros(length,3),
                                joint_pos=torch.full((length,29),.2),joint_vel=torch.zeros(length,29),
                                body_pos_w=body_pos,body_quat_w=body_q)
    cmd.env_motion_indices = torch.arange(count, dtype=torch.long)%2
    cmd.motion_lengths = torch.full((2,),length, dtype=torch.long)
    cmd.motion_sample_counts = torch.zeros(2)
    cmd._pose_ranges = torch.tensor([cfg.pose_range[key] for key in ('x','y','z','roll','pitch','yaw')])
    cmd._vel_ranges = torch.zeros(6,2)
    cmd.current_bin_failed = torch.ones(2,1)
    cmd.bin_failed_count = torch.full((2,1),7.)
    cmd._current_bin_failed = torch.zeros(1)
    cmd._global_sim_step = 17
    cmd._resample_motions_every_steps = 0
    cmd._remap_version = 0
    cmd._env_remap_version = torch.zeros(count,dtype=torch.int32)
    cmd.motion_end_buf = torch.zeros(count,dtype=torch.bool)
    cmd.mask_samples = cmd.goal_mask_samples = 0
    cmd._in_motion_sampling_warmup = lambda: True
    cmd._resample_goal_mask = lambda: setattr(cmd, 'goal_mask_samples', cmd.goal_mask_samples+1)
    cmd._sample_modes_for_envs = lambda ids: setattr(cmd,'mask_samples',cmd.mask_samples+1)
    def gather(name: str, ids: Sequence[int] | None = None) -> torch.Tensor:
        selected = torch.arange(count) if ids is None else torch.as_tensor(ids)
        return getattr(cmd.motion,name)[cmd.time_steps[selected]].clone()
    cmd._gather_by_motion = lambda name: gather(name)
    cmd._gather_by_motion_for_envs = gather
    return cmd


def expected_cache(cmd: Any, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ref_anchor, ref_quat = cmd.anchor_pos_w[ids], cmd.anchor_quat_w[ids]
    robot_anchor, robot_quat = cmd.robot_anchor_pos_w[ids],cmd.robot_anchor_quat_w[ids]
    inverse = ref_quat.clone();inverse[:,1:] *= -1
    delta = hamilton(robot_quat,inverse)
    w,x,y,z = delta.unbind(-1)
    angle = torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    origin = robot_anchor.clone();origin[:,2] = ref_anchor[:,2]
    relative = origin[:,None,:]+torch.einsum('nij,nbj->nbi',yaw_matrix(angle),cmd.body_pos_w[ids]-ref_anchor[:,None,:])
    yaw = torch.zeros(len(ids),3,4);yaw[...,0]=torch.cos(angle[:,None]/2);yaw[...,3]=torch.sin(angle[:,None]/2)
    orientation = hamilton(yaw,cmd.body_quat_w[ids])
    return relative,orientation


def run_checks() -> list[dict[str, Any]]:
    old = subprocess.check_output(['git','show','c319a66:source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py'],cwd=REPO,text=True)
    new_ns,old_ns = source_namespace(COMMANDS.read_text()),source_namespace(old)
    results=[]
    for name in ('MotionCommand','MultiMotionCommand','PartialMaskedMultiMotionCommand'):
        for count in (1,3):
            for randomized in (False,True):
                new,baseline=make_command(new_ns,name,count,randomized),make_command(old_ns,name,count,randomized)
                ids = torch.tensor([0]) if count == 1 else torch.tensor([0,2])
                untouched=torch.tensor([1]) if count==3 else torch.empty(0,dtype=torch.long)
                pos_identity,quat_identity=id(new.body_pos_relative_w),id(new.body_quat_relative_w)
                before_pos,before_q=new.body_pos_relative_w.clone(),new.body_quat_relative_w.clone()
                before_frames=new.time_steps.clone();stats=new.bin_failed_count.clone();failed=new.current_bin_failed.clone()
                torch.manual_seed(20261015);rng=torch.random.get_rng_state().clone()
                new._resample_command(ids);new_rng=torch.random.get_rng_state().clone()
                torch.random.set_rng_state(rng);baseline._resample_command(ids)
                assert torch.equal(new_rng,torch.random.get_rng_state())
                assert torch.equal(new.robot.q,baseline.robot.q) and torch.equal(new.robot.qd,baseline.robot.qd)
                assert torch.equal(new.robot.root,baseline.robot.root) and torch.equal(new.time_steps,baseline.time_steps)
                assert torch.equal(new.motion_sample_counts,baseline.motion_sample_counts)
                assert id(new.body_pos_relative_w)==pos_identity and id(new.body_quat_relative_w)==quat_identity
                assert torch.equal(new.body_pos_relative_w[untouched],before_pos[untouched])
                assert torch.equal(new.body_quat_relative_w[untouched],before_q[untouched])
                assert torch.equal(new.time_steps[untouched],before_frames[untouched])
                assert torch.equal(new.bin_failed_count,stats) and torch.equal(new.current_bin_failed,failed)
                assert new._global_sim_step==17 and new.goal_mask_samples==0
                assert new.robot.joint_writes==new.robot.root_writes==1 and new.robot.fk_reads==1
                assert new.mask_samples==int(name=='PartialMaskedMultiMotionCommand')
                position,orientation=expected_cache(new,ids)
                assert torch.allclose(new.body_pos_relative_w[ids],position,atol=1e-6,rtol=1e-6)
                assert torch.allclose(new.body_quat_relative_w[ids],orientation,atol=1e-6,rtol=1e-6)
                # Independent torso differs from root; a root-as-anchor shortcut is invalid.
                assert not torch.allclose(new.robot_anchor_pos_w[ids],new.robot.root[ids,:3])
                # Normal auto-reset compute keeps the sampled frame for episode==0.
                new._env.episode_length_buf.fill_(1);new._env.episode_length_buf[ids]=0
                baseline._env.episode_length_buf.copy_(new._env.episode_length_buf)
                new._update_command();baseline._update_command()
                assert torch.equal(new.time_steps,baseline.time_steps)
                assert torch.all(new.time_steps[ids]==2)
                assert torch.allclose(new.body_pos_relative_w,baseline.body_pos_relative_w,atol=1e-6,rtol=1e-6)
                assert torch.allclose(new.body_quat_relative_w,baseline.body_quat_relative_w,atol=1e-6,rtol=1e-6)
                assert new._global_sim_step==baseline._global_sim_step
                writes=new.robot.root_writes;new._resample_command([])
                assert new.robot.root_writes==writes
                results.append({'class':name,'B':count,'randomized_reset':randomized,'relative_pos_shape':[count,3,3],
                                'relative_quat_shape':[count,3,4],'reset_RNG_and_state_equal_old':True,
                                'partial_rows_cache_identity_preserved':True,'lazy_FK_reads':1,'normal_auto_reset_matches_old':True})
    # Exercise the actual Multi.reset / derived Partial resample chain as well
    # as direct reset methods. Base manager resamples exactly once in the fixture.
    for name in ('MultiMotionCommand','PartialMaskedMultiMotionCommand'):
        command=make_command(new_ns,name,3)
        before_stats=command.bin_failed_count.clone()
        before_counter=command._global_sim_step
        untouched_cache=command.body_pos_relative_w[1].clone()
        command.reset([0,2])
        assert command.robot.root_writes==command.robot.joint_writes==1
        assert command.mask_samples==int(name=='PartialMaskedMultiMotionCommand')
        assert command._global_sim_step==before_counter and torch.equal(command.bin_failed_count,before_stats)
        assert torch.equal(command.body_pos_relative_w[1],untouched_cache)
        position,orientation=expected_cache(command,torch.tensor([0,2]))
        assert torch.allclose(command.body_pos_relative_w[[0,2]],position,atol=1e-6,rtol=1e-6)
        assert torch.allclose(command.body_quat_relative_w[[0,2]],orientation,atol=1e-6,rtol=1e-6)
        assert command.motion_sample_counts.sum()==2
        results.append({'class':name,'B':3,'full_reset_chain':True,'partial_sampling_exactly_once':True,
                        'no_step_or_EMA_update':True,'nonreset_cache_unchanged':True})
    assert not torch.cuda.is_initialized()
    return results


if __name__ == '__main__':
    torch.set_num_threads(1)
    checks=run_checks()
    print(json.dumps({'status':'PASS','tests_passed':len(checks),'checks':checks,'commands_sha256':hashlib.sha256(COMMANDS.read_bytes()).hexdigest(),
                      'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'Isaac_math_sha256':hashlib.sha256(MATH.read_bytes()).hexdigest(),
                      'CUDA_initialized':False,'native_validation':'NOT_RUN','new_physics_calls':0},ensure_ascii=False,indent=2))
