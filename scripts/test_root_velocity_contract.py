"""CPU regressions: exact native API math, CLI guard order, and96legacy DEV files.

No AppLauncher, native articulation, GPU, physics or data re-conversion is run.
The native API methods are AST-extracted and executed on a CPU-only state stub;
all existing NPZ are read and rehashed, never written.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np
import torch

from root_velocity_contract import (
    conversion_metadata, imported_source_binding, link_velocity_to_com, reserve_output_directory, save_npz_exclusive,
    self_test, sha256_file, validate_mode, write_csv_root_state,
)

API = Path('/data/jxc/cache/nvidia/IsaacLab-v2.3.2/source/isaaclab/isaaclab/assets/articulation/articulation.py')
INITIAL = Path('/data/jxc/projects/bones_curated/training/runs/stage0_gmt/curation_abc_resume_30000_shared_20261012T195500Z/evaluation/results/DEV/A/seed_20260904/updates_30000/initial_state.pt')
SCRIPTS = Path(__file__).resolve().parent


def quat_apply_cpu(quat: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    assert quat.shape == (len(point),4) and point.ndim == 2 and point.shape[1] == 3
    xyz = quat[:,1:]
    first = torch.linalg.cross(xyz,point,dim=-1)
    return point + 2*quat[:,:1]*first + 2*torch.linalg.cross(xyz,first,dim=-1)


def native_api_math_test() -> dict[str, Any]:
    tree = ast.parse(API.read_text())
    articulation = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Articulation')
    names = {'write_root_link_state_to_sim','write_root_link_velocity_to_sim','write_root_state_to_sim'}
    nodes = [n for n in articulation.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes) == 3
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    namespace: dict[str,Any] = {'torch':torch,'math_utils':SimpleNamespace(quat_apply=quat_apply_cpu)}
    exec(compile(ast.fix_missing_locations(module),str(API),'exec'),namespace)

    class CPUArticulationStub:
        def __init__(self, offset: torch.Tensor) -> None:
            self.data = SimpleNamespace(root_link_quat_w=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64).repeat(len(offset),1),
                                        body_com_pos_b=offset[:,None,:].clone())
            self._data = SimpleNamespace(root_link_vel_w=torch.zeros((len(offset),6),dtype=torch.float64),
                                         _root_link_state_w=SimpleNamespace(data=None))
            self.last_com = torch.zeros((len(offset),6),dtype=torch.float64)

        def write_root_link_pose_to_sim(self, pose: torch.Tensor, env_ids: Any = None) -> None:
            assert env_ids is None and pose.shape == (len(self.last_com),7)
            self.data.root_link_quat_w = pose[:,3:7].clone()

        def write_root_com_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Any = None) -> None:
            assert env_ids is None and root_velocity.shape == self.last_com.shape
            self.last_com = root_velocity.clone()

    for name in names:
        setattr(CPUArticulationStub,name,namespace[name])
    offset = torch.tensor([[0.,0.,-.07603006],[.12,-.03,.01],[0.,0.,0.]],dtype=torch.float64)
    robot = CPUArticulationStub(offset)
    state = torch.zeros((3,13),dtype=torch.float64)
    state[:,3:7] = torch.tensor([[1.,0.,0.,0.],[np.sqrt(.5),0.,0.,np.sqrt(.5)],[1.,0.,0.,0.]],dtype=torch.float64)
    state[:,7:10] = torch.tensor([[.1,.2,.3],[1.,2.,3.],[4.,5.,6.]],dtype=torch.float64)
    state[:,10:13] = torch.tensor([[1.,0.,0.],[0.,0.,2.],[7.,8.,9.]],dtype=torch.float64)
    original = state.clone()
    expected = link_velocity_to_com(state[:,7:10].numpy(),state[:,10:13].numpy(),state[:,3:7].numpy(),offset.numpy())
    write_csv_root_state(robot,state,'link_v2')
    torch.testing.assert_close(robot.last_com[:,:3],torch.from_numpy(expected),rtol=0,atol=1e-14)
    torch.testing.assert_close(robot.last_com[:,3:],state[:,10:13],rtol=0,atol=0)
    torch.testing.assert_close(state,original,rtol=0,atol=0)
    write_csv_root_state(robot,state,'legacy_com_v1')
    torch.testing.assert_close(robot.last_com,state[:,7:],rtol=0,atol=0)
    # Exercise full production provenance creation and non-pickle storage.
    with tempfile.TemporaryDirectory(prefix='converter_metadata_',dir=os.environ.get('ANYBODY_CONVERTER_TEST_TMPDIR')) as temporary:
        base = Path(temporary)
        csv = base/'source.csv'; csv.write_text('synthetic metadata fixture\n')
        run = reserve_output_directory(base/'fresh','link_v2')
        robot.cfg = SimpleNamespace(spawn=SimpleNamespace(asset_path=str(SCRIPTS.parent/'source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf')))
        robot.body_names, robot.joint_names = ['pelvis'], ['synthetic_joint']
        metadata = conversion_metadata(csv,{'robot':'g1','input_fps':30,'output_fps':50},run,robot,Path(__file__))
        assert metadata['runtime_package_source_binding']['status'] == 'sources_not_imported'
        assert metadata['runtime_package_source_binding']['native_import_verification'] == 'NOT_RUN'
        assert metadata['native_root_state_writer'] == 'write_root_link_state_to_sim'
        assert metadata['body_linear_velocity_semantics'] == 'rigid_body_COM_world'
        assert metadata['source_files_sha256'][str(API)] == sha256_file(API)
        arrays = {'fps':np.asarray([50]),'joint_pos':np.zeros((3,1)),'joint_vel':np.zeros((3,1)),
                  'body_pos_w':np.zeros((3,1,3)),'body_quat_w':np.tile([1.,0.,0.,0.],(3,1,1)),
                  'body_lin_vel_w':np.zeros((3,1,3)),'body_ang_vel_w':np.zeros((3,1,3))}
        destination = base/'fresh/nested/synthetic.npz'
        save_npz_exclusive(destination,arrays,metadata)
        with np.load(destination,allow_pickle=False) as archive:
            loaded = json.loads(str(archive['conversion_metadata_json']))
            assert loaded['source_files_sha256'] == metadata['source_files_sha256']
            assert archive['body_names'].tolist() == ['pelvis']
        try:
            save_npz_exclusive(base/'outside.npz',arrays,metadata)
        except ValueError:
            pass
        else:
            raise AssertionError('Output escape was accepted')
        altered = dict(metadata,source_files_sha256={str(API):'0'*64})
        try:
            save_npz_exclusive(base/'fresh/source_drift.npz',arrays,altered)
        except RuntimeError:
            pass
        else:
            raise AssertionError('API identity mismatch was accepted')
        assert not (base/'outside.npz').exists() and not (base/'fresh/source_drift.npz').exists()
    assert not torch.cuda.is_initialized()
    return {'status':'PASS','native_API_source':str(API),'native_API_source_sha256':sha256_file(API),
            'CPU_stub_batch':3,'native_simulator_conversion':'NOT_RUN',
            'production_metadata_and_exclusive_storage':'PASS',
            'tested':'exact source writer dispatch; pose before COM rotation; variable asset offsets; zero offset; unchanged inputs; explicit legacy'}


def cli_collision_test() -> dict[str, Any]:
    """Run actual CLI prefixes with a mock AppLauncher that must never initialize."""
    with tempfile.TemporaryDirectory(prefix='converter_cli_guard_',dir=os.environ.get('ANYBODY_CONVERTER_TEST_TMPDIR')) as temp:
        root = Path(temp)
        (root/'isaaclab').mkdir()
        (root/'isaaclab/__init__.py').write_text('')
        flag = root/'unexpected_app_start'
        (root/'isaaclab/app.py').write_text(
            "import os\nfrom pathlib import Path\n"
            "class AppLauncher:\n"
            "    @staticmethod\n"
            "    def add_app_launcher_args(parser):\n"
            "        parser.add_argument('--device', default='cpu')\n"
            "    def __init__(self, args):\n"
            "        Path(os.environ['CONVERTER_TEST_APP_FLAG']).write_text('unexpected')\n"
            "        raise RuntimeError('Native app must not start in collision tests')\n")
        inputs = root/'inputs'; inputs.mkdir()
        csv = inputs/'sample.csv'; csv.write_text('synthetic only\n')
        old = root/'historical'; old.mkdir()
        old_file = old/'sample.npz'; old_file.write_bytes(b'untouched historical fixture')
        old_sha = sha256_file(old_file)
        env = dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1',CONVERTER_TEST_APP_FLAG=str(flag))
        errors: list[str] = []
        for script, args in [
            ('csv_to_npz.py',['--input_file',str(csv),'--output_name',str(old_file)]),
            ('batch_csv_to_npz.py',['--input_dir',str(inputs),'--output_prefix','prefix','--output_dir',str(old)]),
        ]:
            result = subprocess.run([sys.executable,str(SCRIPTS/script),*args],env=env,text=True,capture_output=True,check=False)
            assert result.returncode != 0 and 'Output directory must be new' in result.stderr
            assert not flag.exists() and sha256_file(old_file) == old_sha
            errors.append(script)
    return {'status':'PASS','actual_script_pre_AppLauncher_collision_rejection':errors,
            'historical_fixture_bytes_unchanged':True,'real_AppLauncher_imported':False}


def actual_script_synthetic_loop_test() -> dict[str, Any]:
    """Execute both real conversion loops on CPU fixtures, with three saved frames."""
    class Robot:
        def __init__(self) -> None:
            self.body_names, self.joint_names = ['synthetic_root'], ['synthetic_joint']
            self.cfg = SimpleNamespace(spawn=SimpleNamespace(asset_path=None,usd_path=None))
            self.data = SimpleNamespace(default_root_state=torch.tensor([[0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]],dtype=torch.float64),
                                        default_joint_pos=torch.zeros((1,1)),default_joint_vel=torch.zeros((1,1)),
                                        body_com_pos_b=torch.tensor([[[0.,0.,-.076]]],dtype=torch.float64),
                                        body_pos_w=torch.zeros((1,1,3)),body_quat_w=torch.tensor([[[1.,0.,0.,0.]]]),
                                        body_com_lin_vel_w=torch.zeros((1,1,3)),body_ang_vel_w=torch.zeros((1,1,3)),
                                        joint_pos=torch.zeros((1,1)),joint_vel=torch.zeros((1,1)))

        def find_joints(self, names: list[str], preserve_order: bool) -> tuple[list[int],list[str]]:
            assert names == self.joint_names and preserve_order
            return [0],names

        def write_root_link_state_to_sim(self, state: torch.Tensor) -> None:
            self.update(state,link=True)

        def write_root_state_to_sim(self, state: torch.Tensor) -> None:
            self.update(state,link=False)

        def update(self, state: torch.Tensor, link: bool) -> None:
            value = state[:,7:10].numpy()
            if link:
                value = link_velocity_to_com(value,state[:,10:].numpy(),state[:,3:7].numpy(),self.data.body_com_pos_b[:,0].numpy())
            self.data.body_com_lin_vel_w = torch.from_numpy(value[:,None].copy())
            self.data.body_pos_w = state[:,:3,None].transpose(1,2).clone()
            self.data.body_quat_w = state[:,None,3:7].clone()
            self.data.body_ang_vel_w = state[:,None,10:].clone()

        def write_joint_state_to_sim(self, position: torch.Tensor, velocity: torch.Tensor) -> None:
            self.data.joint_pos,self.data.joint_vel = position.clone(),velocity.clone()

    class Loader:
        def __init__(self, **kwargs: Any) -> None:
            self.duration = .06
            self.index = 0

        def state(self) -> tuple[torch.Tensor,...]:
            return (torch.zeros((1,3)),torch.tensor([[1.,0.,0.,0.]]),torch.zeros((1,3)),
                    torch.tensor([[1.,0.,0.]]),torch.zeros((1,1)),torch.zeros((1,1)))

        def get_next_state(self) -> tuple[tuple[torch.Tensor,...],bool]:
            self.index += 1
            return self.state(), self.index == 3

        def iter_states(self) -> Iterator[tuple[torch.Tensor,...]]:
            for _ in range(3):
                yield self.state()

    cases: list[dict[str,Any]] = []
    with tempfile.TemporaryDirectory(prefix='converter_loop_',dir=os.environ.get('ANYBODY_CONVERTER_TEST_TMPDIR')) as temporary:
        base = Path(temporary)
        source = base/'source.csv'; source.write_text('synthetic conversion-loop fixture\n')
        for filename,function in [('csv_to_npz.py','run_simulator'),('batch_csv_to_npz.py','process_one_file')]:
            path = SCRIPTS/filename
            tree = ast.parse(path.read_text())
            node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
            module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
            for mode in ['link_v2','legacy_com_v1']:
                run = reserve_output_directory(base/(path.stem+'_'+mode),validate_mode(mode))
                destination = Path(run['output_directory'])/'synthetic.npz'
                robot = Robot()
                scene = {'robot':robot}
                scene_ns = SimpleNamespace(env_origins=torch.zeros((1,3)),update=lambda _:None)

                class Scene:
                    env_origins = scene_ns.env_origins
                    def __getitem__(self, name: str) -> Robot:
                        return scene[name]
                    def update(self, dt: float) -> None:
                        scene_ns.update(dt)

                args = SimpleNamespace(robot='g1',input_file=str(source),input_fps=30,output_fps=50,frame_range=None,
                                       csv_skip_header=False,csv_drop_frame_column=False,root_rotation='quat_xyzw',
                                       joint_angles_in_degrees=False,position_scale=.01,root_velocity_mode=mode,
                                       min_duration=0.,log_wandb=False)
                namespace: dict[str,Any] = {'__file__':str(path),'torch':torch,'np':np,'Path':Path,'args_cli':args,
                    'MotionLoader':Loader,'source_csv':source,'output_run':run,'output_path':destination,
                    'simulation_app':SimpleNamespace(is_running=lambda:True),'conversion_metadata':conversion_metadata,
                    'write_csv_root_state':write_csv_root_state,'validate_mode':validate_mode,'save_npz_exclusive':save_npz_exclusive}
                exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
                simulator = SimpleNamespace(device='cpu',render=lambda:None,get_physics_dt=lambda:.02,set_camera_view=lambda *_:None)
                with contextlib.redirect_stdout(io.StringIO()):
                    if function == 'run_simulator':
                        namespace[function](simulator,Scene(),robot.joint_names)
                    else:
                        namespace[function](simulator,Scene(),str(source),str(destination),'synthetic',robot.joint_names,run)
                with np.load(destination,allow_pickle=False) as archive:
                    assert archive['body_lin_vel_w'].shape == (3,1,3)
                    expected = np.tile([0.,.076 if mode=='link_v2' else 0.,0.],(3,1,1))
                    np.testing.assert_allclose(archive['body_lin_vel_w'],expected,atol=1e-12)
                    assert str(archive['root_velocity_mode']) == mode
                cases.append({'script':filename,'mode':mode,'synthetic_frames':3,'status':'PASS'})
    return {'status':'PASS','CPU_only':True,'actual_source_functions_on_synthetic_fixtures':cases,'native_conversion':'NOT_RUN'}


def shard_launcher_plan_test() -> dict[str, Any]:
    launcher = SCRIPTS/'run_batch_csv_to_npz_shards.sh'
    subprocess.run(['bash','-n',str(launcher)],check=True,capture_output=True)
    plans: dict[str,list[str]] = {}
    with tempfile.TemporaryDirectory(prefix='converter_shards_',dir=os.environ.get('ANYBODY_CONVERTER_TEST_TMPDIR')) as temporary:
        root = Path(temporary)
        csv_parent = root/'inputs'
        for index in [0,1]:
            directory = csv_parent/str(index); directory.mkdir(parents=True)
            (directory/'source.csv').write_text('synthetic launcher fixture\n')
        for mode in ['link_v2','legacy_com_v1']:
            output = root/mode
            env = dict(os.environ,MOSAIC_ROOT=str(SCRIPTS.parent),CSV_PARENT=str(csv_parent),OUT_DIR=str(output),
                       ROOT_VELOCITY_MODE=mode,FIRST_SHARD='0',LAST_SHARD='1',MAX_PARALLEL='2',DRY_RUN='1',PYTHON_BIN=sys.executable,
                       PYTHONPATH='/external/isaac/source:/external/legacy_package')
            result = subprocess.run(['bash',str(launcher)],env=env,text=True,capture_output=True,check=True)
            lines = [line for line in result.stdout.splitlines() if line.startswith('CUDA_VISIBLE_DEVICES=')]
            assert len(lines) == 2 and not output.exists()
            destinations: list[str] = []
            for index,line in enumerate(lines):
                words = shlex.split(line)
                assert words[0] == f'CUDA_VISIBLE_DEVICES={index}'
                expected_binding = f'PYTHONPATH={SCRIPTS.parent}/source/whole_body_tracking:{SCRIPTS.parent}/source/rsl_rl:/external/isaac/source:/external/legacy_package'
                assert words[1] == expected_binding
                assert words[3] == str(SCRIPTS/'batch_csv_to_npz.py')
                assert words[words.index('--root_velocity_mode')+1] == mode
                destination = words[words.index('--output_dir')+1]
                assert destination == str(output/f'shard_{index}')
                destinations.append(destination)
            assert len(set(destinations)) == 2
            plans[mode] = lines
            output.mkdir()
            old = output/'old.npz'; old.write_bytes(b'preserved synthetic existing dataset')
            before = sha256_file(old)
            refused = subprocess.run(['bash',str(launcher)],env=env,text=True,capture_output=True,check=False)
            assert refused.returncode != 0 and 'must be new' in refused.stderr
            assert sha256_file(old) == before
    return {'status':'PASS','bash_syntax':'PASS','DRY_RUN_command_plans':plans,
            'newtree_package_prefixes_before_preserved_external_paths':True,
            'distinct_fresh_shard_directories':True,'existing_output_refused_for_both_modes':True,'native_calls':0}


def imported_binding_test() -> dict[str, Any]:
    package = SCRIPTS.parent/'source/whole_body_tracking/whole_body_tracking'
    names = {'whole_body_tracking':package/'__init__.py',
             'whole_body_tracking.assets':package/'assets/__init__.py',
             'whole_body_tracking.robots.robot_registry':package/'robots/robot_registry.py',
             'whole_body_tracking.robots.g1':package/'robots/g1.py'}
    with patch.dict(sys.modules,{name:None for name in names}):
        missing = imported_source_binding(SCRIPTS.parent,'g1')
        assert missing['status'] == 'sources_not_imported' and missing['native_import_verification']=='NOT_RUN'
    synthetic: dict[str,ModuleType] = {}
    for name,path in names.items():
        module = ModuleType(name); module.__file__ = str(path)
        synthetic[name] = module
    with patch.dict(sys.modules,synthetic):
        match = imported_source_binding(SCRIPTS.parent,'g1')
        assert match['status'] == 'imported_sources_match'
    old = Path('/data/jxc/projects/bones_curated/references/AnyBody_resume_state_fix_20261006')
    rejected: list[str] = []
    for name,path in names.items():
        wrong = ModuleType(name); wrong.__file__ = str(old/path.relative_to(SCRIPTS.parent))
        with patch.dict(sys.modules,{**synthetic,name:wrong}):
            try:
                imported_source_binding(SCRIPTS.parent,'g1')
            except RuntimeError:
                rejected.append(name)
            else:
                raise AssertionError(f'Old source accepted for {name}')
    assert len(rejected) == 4
    return {'status':'PASS','synthetic_module_record_tests':True,'old_tree_module_paths_rejected':rejected,
            'unimported_native_sources_explicitly_NOT_RUN':True,'real_native_imports':'NOT_RUN'}


def existing_DEV_formula_audit() -> dict[str, Any]:
    before_initial = sha256_file(INITIAL)
    saved = torch.load(INITIAL,map_location='cpu',weights_only=False)
    offset = saved['physics']['get_coms'][0,0,:3].numpy()
    cases = {c['motion']:c for c in saved['cases']}
    corrections: list[np.ndarray[Any,Any]] = []
    source_hashes: dict[str,str] = {}
    max_link_derivative_gap = 0.0
    for motion in cases:
        path = Path(motion); before = sha256_file(path)
        with np.load(path,allow_pickle=False) as archive:
            position = archive['body_pos_w'][:,0].copy()
            quat = archive['body_quat_w'][:,0].copy()
            omega = archive['body_ang_vel_w'][:,0].copy()
            old_velocity = archive['body_lin_vel_w'][:,0].copy()
            fps = float(np.asarray(archive['fps']).reshape(-1)[0]); assert fps == 50
            corrected = link_velocity_to_com(old_velocity,omega,quat,np.broadcast_to(offset,old_velocity.shape))
            assert corrected.shape == old_velocity.shape and np.isfinite(corrected).all()
            delta = corrected-old_velocity
            # This is only an offline formula evaluation, no prospective NPZ is saved.
            all_body_corrected = archive['body_lin_vel_w']+delta[:,None,:]
            assert all_body_corrected.shape == archive['body_lin_vel_w'].shape
            corrections.append(np.linalg.norm(delta,axis=-1))
            gap = float(np.max(abs(np.gradient(position,1/fps,axis=0)-old_velocity)))
            max_link_derivative_gap = max(max_link_derivative_gap,gap)
            assert gap < 5e-4
        assert sha256_file(path) == before
        source_hashes[str(path)] = before
    combined = np.concatenate(corrections)
    assert len(cases) == 96 and len(combined) == 35175
    assert sha256_file(INITIAL) == before_initial and not torch.cuda.is_initialized()
    return {'status':'PASS','initial_state':str(INITIAL),'initial_state_sha256':before_initial,
            'unique_existing_DEV_NPZ':len(cases),'existing_reference_frames':len(combined),
            'root_COM_offset_from_existing_native_fixture_m':offset.tolist(),
            'legacy_root_velocity_minus_link_position_derivative_max_mps':max_link_derivative_gap,
            'prospective_COM_correction_norm_mean_mps':float(combined.mean()),
            'prospective_COM_correction_norm_p95_mps':float(np.quantile(combined,.95)),
            'prospective_COM_correction_norm_max_mps':float(combined.max()),
            'all_existing_NPZ_hashes_unchanged':True,'existing_NPZ_sha256':source_hashes,
            'new_dataset_created':False,'new_native_simulator_conversion':'NOT_RUN',
            'performance_effect':'NOT_TESTED'}


def main() -> None:
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    with contextlib.redirect_stdout(io.StringIO()):
        helper = self_test()
    for name in ['csv_to_npz.py','batch_csv_to_npz.py','root_velocity_contract.py',Path(__file__).name]:
        ast.parse((SCRIPTS/name).read_text())
    result = {'status':'PASS','CPU_only':True,'new_physical_calls':0,
              'helper_self_test':helper,'native_API_math':native_api_math_test(),
              'actual_CLI_collision_test':cli_collision_test(),
              'actual_script_synthetic_loops':actual_script_synthetic_loop_test(),
              'shard_launcher':shard_launcher_plan_test(),
              'imported_package_binding':imported_binding_test(),
              'existing_DEV_offline':existing_DEV_formula_audit()}
    assert not torch.cuda.is_initialized()
    result['CUDA_initialized'] = torch.cuda.is_initialized()
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__ == '__main__':
    main()
