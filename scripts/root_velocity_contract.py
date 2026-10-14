"""CSV/NPZ velocity contract and exclusive output storage, without Isaac imports.

CSV position and its derivative refer to the articulation link origin. The
native link-state writer transforms v_COM = v_link + omega_world x R_link*r_COM.
Recorded body positions remain link origins; recorded body linear velocities
remain native rigid-body COM velocities. WXYZ, world axes, metres and seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence, cast

import numpy as np
from numpy.typing import NDArray

RootVelocityMode = Literal['link_v2', 'legacy_com_v1']
SCHEMA_VERSION = 'anybody_csv_npz_v2'
MODES: tuple[RootVelocityMode, ...] = ('link_v2', 'legacy_com_v1')
ARRAY_FIELDS = ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w')


class RootStateWriter(Protocol):
    def write_root_link_state_to_sim(self, root_state: Any) -> None:
        """Native link-pose and link-velocity writer interface."""

    def write_root_state_to_sim(self, root_state: Any) -> None:
        """Explicit legacy link-pose and COM-velocity writer interface."""


def sha256_file(path: Path) -> str:
    with path.open('rb') as stream:
        result = hashlib.file_digest(stream, 'sha256') if hasattr(hashlib, 'file_digest') else None
        if result is not None:
            return result.hexdigest()
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
        return digest.hexdigest()


def validate_mode(mode: str) -> RootVelocityMode:
    if mode not in MODES:
        raise ValueError(f'Unknown root velocity mode: {mode!r}')
    return cast(RootVelocityMode, mode)


def link_velocity_to_com(
    link_velocity_w: NDArray[np.floating[Any]], omega_w: NDArray[np.floating[Any]],
    link_quat_wxyz: NDArray[np.floating[Any]], com_offset_link: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Pure CPU reference: arrays (N,3), (N,3), (N,4), (N,3) -> COM (N,3)."""
    v, omega, quat, offset = [np.asarray(a, dtype=np.float64)
                              for a in (link_velocity_w, omega_w, link_quat_wxyz, com_offset_link)]
    assert v.ndim == 2 and v.shape[1] == 3
    assert omega.shape == offset.shape == v.shape and quat.shape == (len(v), 4)
    assert all(np.isfinite(a).all() for a in (v, omega, quat, offset))
    assert np.max(np.abs(np.linalg.norm(quat, axis=1)-1.0)) < 1e-3
    qxyz = quat[:, 1:]
    cross = np.cross(qxyz, offset)
    world_offset = offset + 2*quat[:, :1]*cross + 2*np.cross(qxyz, cross)
    result = v + np.cross(omega, world_offset)
    assert result.shape == v.shape and np.isfinite(result).all()
    return result


def write_csv_root_state(robot: RootStateWriter, root_state: Any, mode: RootVelocityMode) -> None:
    """Use the native writer, so the current asset COM offset is authoritative."""
    validate_mode(mode)
    assert len(root_state.shape) == 2 and root_state.shape[1] == 13
    if isinstance(root_state, np.ndarray):
        assert np.isfinite(root_state).all()
        assert np.max(np.abs(np.linalg.norm(root_state[:, 3:7], axis=1)-1)) < 1e-3
    else:
        assert bool(root_state.isfinite().all().item())
        assert bool(((root_state[:, 3:7].square().sum(-1).sqrt()-1).abs() < 1e-3).all().item())
    method_name = 'write_root_link_state_to_sim' if mode == 'link_v2' else 'write_root_state_to_sim'
    writer = getattr(robot, method_name, None)
    if not callable(writer):
        raise RuntimeError(f'{method_name} is required for {mode}; no implicit API fallback is permitted')
    writer(root_state)


def npz_output_path(name: str | Path) -> Path:
    path = Path(name).expanduser()
    if path.suffix != '.npz':
        path = Path(str(path)+'.npz')
    return path.absolute()


def plan_batch_outputs(input_dir: Path, files: Sequence[Path], output_dir: Path,
                       prefix: str) -> list[tuple[Path, Path]]:
    """Validate every destination before creating any output or starting Isaac."""
    if not prefix or Path(prefix).name != prefix or '/' in prefix or '\\' in prefix or prefix in {'.', '..'}:
        raise ValueError('output_prefix must be a nonempty filename component')
    target = output_dir.absolute()
    result: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for csv_path in files:
        source = csv_path.absolute()
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = source.relative_to(input_dir.absolute())
        destination = (target/relative).with_name(f'{prefix}_{relative.stem}.npz')
        if destination in seen:
            raise FileExistsError(f'Duplicate output destination: {destination}')
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        seen.add(destination)
        result.append((source, destination))
    return result


def reserve_output_directory(path: Path, mode: RootVelocityMode) -> dict[str, Any]:
    """A conversion run exclusively owns a newly created output directory."""
    validate_mode(mode)
    path = path.expanduser().absolute()
    if os.path.lexists(path):
        raise FileExistsError(f'Output directory must be new, including for legacy mode: {path}')
    path.mkdir(parents=True, exist_ok=False)
    run = {'schema_version': SCHEMA_VERSION, 'run_id': str(uuid.uuid4()),
           'created_utc': datetime.now(timezone.utc).isoformat(), 'root_velocity_mode': mode,
           'output_directory': str(path.resolve())}
    with (path/'conversion_run.json').open('x', encoding='utf-8') as stream:
        json.dump(run, stream, sort_keys=True, indent=2)
        stream.write('\n')
    return run


def imported_source_binding(root: Path, robot_name: str) -> dict[str, Any]:
    """Check loaded packages without importing native code; absence is explicit."""
    package = root.resolve()/'source/whole_body_tracking/whole_body_tracking'
    expected = {'whole_body_tracking':package/'__init__.py',
                'whole_body_tracking.assets':package/'assets/__init__.py',
                'whole_body_tracking.robots.robot_registry':package/'robots/robot_registry.py',
                'whole_body_tracking.robots.g1':package/'robots/g1.py'}
    selected = package/f'robots/{robot_name}.py'
    if selected.is_file():
        expected[f'whole_body_tracking.robots.{robot_name}'] = selected
    checked: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for name, candidate in expected.items():
        module = sys.modules.get(name)
        if module is None:
            missing.append(name)
            continue
        actual = getattr(module,'__file__',None)
        if not isinstance(actual,str) or Path(actual).resolve() != candidate.resolve():
            raise RuntimeError(f'Imported {name} source mismatch: {actual!r}; expected {candidate}')
        checked[name] = {'path':str(candidate.resolve()),'sha256':sha256_file(candidate.resolve())}
    status = 'sources_not_imported' if not checked else ('partial_sources_imported' if missing else 'imported_sources_match')
    return {'status':status,'checked_modules':checked,'modules_not_imported':missing,
            'native_import_verification':'NOT_RUN' if status == 'sources_not_imported' else status}


def conversion_metadata(source_csv: Path, options: Mapping[str, Any], run: Mapping[str, Any],
                        robot: Any, entrypoint: Path) -> dict[str, Any]:
    """Freeze source/data/API identities before reading motion frames."""
    mode = validate_mode(str(run['root_velocity_mode']))
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), entrypoint.resolve(), root/'scripts/motion_csv_loader.py',
             root/'source/whole_body_tracking/whole_body_tracking/robots/robot_registry.py']
    robot_name = str(options['robot'])
    binding = imported_source_binding(root,robot_name)
    config_source = root/f'source/whole_body_tracking/whole_body_tracking/robots/{robot_name}.py'
    if config_source.is_file():
        files.append(config_source)
    asset = getattr(robot.cfg.spawn, 'asset_path', None) or getattr(robot.cfg.spawn, 'usd_path', None)
    if isinstance(asset, str) and Path(asset).is_file():
        if not Path(asset).resolve().is_relative_to(root/'source/whole_body_tracking/whole_body_tracking/assets'):
            raise RuntimeError(f'Local robot asset is outside this worktree: {asset}')
        files.append(Path(asset).resolve())
    method = getattr(robot, 'write_root_link_state_to_sim' if mode == 'link_v2' else 'write_root_state_to_sim')
    api_source = inspect.getsourcefile(method)
    if api_source and Path(api_source).is_file():
        files.append(Path(api_source).resolve())
    source = source_csv.resolve()
    return {'schema_version': SCHEMA_VERSION, 'run_id': run['run_id'], 'root_velocity_mode': mode,
            'output_directory': run['output_directory'],
            'csv_root_position_semantics': 'link_origin_world', 'csv_root_linear_velocity_semantics': 'link_origin_world_derivative',
            'native_root_state_writer': 'write_root_link_state_to_sim' if mode == 'link_v2' else 'write_root_state_to_sim',
            'legacy_mismatch_preserved': mode == 'legacy_com_v1',
            'body_position_semantics': 'link_origin_world', 'body_linear_velocity_semantics': 'rigid_body_COM_world',
            'native_body_linear_velocity_reader': 'body_com_lin_vel_w',
            'angular_velocity_semantics': 'world', 'quaternion_order': 'wxyz', 'length_unit': 'm', 'time_unit': 's',
            'source_csv': str(source), 'source_csv_sha256': sha256_file(source), 'options': dict(options),
            'source_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
            'source_tree_dirty': bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip()),
            'runtime_package_source_binding':binding,
            'source_files_sha256': {str(path):sha256_file(path) for path in files},
            'robot_asset': asset, 'body_names': list(robot.body_names), 'joint_names': list(robot.joint_names),
            'root_COM_offset_in_link_m': robot.data.body_com_pos_b[0,0].detach().cpu().tolist()}


def save_npz_exclusive(path: Path, arrays: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    """Write one fresh NPZ; refuse races/collisions, object arrays, or source drift."""
    if sha256_file(Path(metadata['source_csv'])) != metadata['source_csv_sha256']:
        raise RuntimeError('Source CSV changed while converting; refusing to publish NPZ')
    for source, expected_sha in metadata.get('source_files_sha256', {}).items():
        if sha256_file(Path(source)) != expected_sha:
            raise RuntimeError(f'Converter/asset/API source changed while converting: {source}')
    mode = validate_mode(str(metadata['root_velocity_mode']))
    assert metadata['schema_version'] == SCHEMA_VERSION
    directory = Path(metadata['output_directory']).resolve(strict=True)
    if not path.resolve().is_relative_to(directory):
        raise ValueError('NPZ destination escapes this fresh conversion directory')
    run = json.loads((directory/'conversion_run.json').read_text())
    if run['run_id'] != metadata['run_id'] or run['root_velocity_mode'] != mode or run['schema_version'] != SCHEMA_VERSION:
        raise RuntimeError('Output directory run identity does not match conversion metadata')
    payload = {key:np.asarray(value) for key,value in arrays.items()}
    frames = payload['joint_pos'].shape[0]
    joints, bodies = len(metadata['joint_names']), len(metadata['body_names'])
    assert frames >= 3
    assert payload['joint_pos'].shape == payload['joint_vel'].shape == (frames,joints)
    assert payload['body_pos_w'].shape == payload['body_lin_vel_w'].shape == payload['body_ang_vel_w'].shape == (frames,bodies,3)
    assert payload['body_quat_w'].shape == (frames,bodies,4)
    assert all(np.isfinite(payload[key]).all() for key in (*ARRAY_FIELDS,'fps'))
    assert payload['fps'].size == 1 and float(payload['fps'].reshape(-1)[0]) > 0
    assert np.abs(np.linalg.norm(payload['body_quat_w'],axis=-1)-1).max() < 1e-3
    payload.update(conversion_schema_version=np.asarray(SCHEMA_VERSION), root_velocity_mode=np.asarray(mode),
                   conversion_metadata_json=np.asarray(json.dumps(dict(metadata),sort_keys=True,allow_nan=False)),
                   body_names=np.asarray(metadata['body_names'],dtype=np.str_), joint_names=np.asarray(metadata['joint_names'],dtype=np.str_))
    assert all(value.dtype.kind != 'O' for value in payload.values())
    path.parent.mkdir(parents=True,exist_ok=True)
    # A failed write remains as failure evidence and blocks accidental reuse.
    with path.open('xb') as stream:
        np.savez(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())


def self_test() -> dict[str, Any]:
    """CPU-only mathematical, writer-selection, metadata, and overwrite tests."""
    v = np.zeros((1,3)); omega = np.asarray([[1.,0.,0.]])
    quat = np.asarray([[1.,0.,0.,0.]]); offset = np.asarray([[0.,0.,-.076]])
    np.testing.assert_allclose(link_velocity_to_com(v,omega,quat,offset),[[0.,.076,0.]])
    np.testing.assert_array_equal(link_velocity_to_com(v,v,quat,offset),v)
    yaw = np.asarray([[math.sqrt(.5),0.,0.,math.sqrt(.5)]])
    np.testing.assert_allclose(link_velocity_to_com(v,np.asarray([[0.,0.,1.]]),yaw,np.asarray([[1.,0.,0.]])),[[-1.,0.,0.]],atol=1e-12)
    np.testing.assert_allclose(link_velocity_to_com(v,omega,-quat,offset),[[0.,.076,0.]])
    batch = link_velocity_to_com(np.repeat(v,2,axis=0),np.repeat(omega,2,axis=0),np.repeat(quat,2,axis=0),np.repeat(offset,2,axis=0))
    assert batch.shape == (2,3)

    class FakeWriter:
        def __init__(self) -> None:
            self.calls: list[str] = []
        def write_root_link_state_to_sim(self, root_state: Any) -> None:
            self.calls.append('link')
        def write_root_state_to_sim(self, root_state: Any) -> None:
            self.calls.append('legacy')

    state = np.zeros((1,13)); state[:,3] = 1
    writer = FakeWriter()
    write_csv_root_state(writer,state,'link_v2'); write_csv_root_state(writer,state,'legacy_com_v1')
    assert writer.calls == ['link','legacy']
    failures = 0
    temp_root = os.environ.get('ANYBODY_CONVERTER_TEST_TMPDIR')
    with tempfile.TemporaryDirectory(prefix='root_velocity_contract_',dir=temp_root) as temporary:
        base = Path(temporary); source = base/'input.csv'; source.write_text('synthetic only\n')
        for mode in MODES:
            output = base/mode
            run = reserve_output_directory(output,mode)
            try:
                reserve_output_directory(output,mode)
            except FileExistsError:
                failures += 1
            else:
                raise AssertionError('Existing output directory accepted')
            arrays = {'fps':np.asarray([50]), 'joint_pos':np.zeros((3,1)), 'joint_vel':np.zeros((3,1)),
                      'body_pos_w':np.zeros((3,1,3)), 'body_quat_w':np.tile(quat[:,None,:],(3,1,1)),
                      'body_lin_vel_w':np.zeros((3,1,3)), 'body_ang_vel_w':np.zeros((3,1,3))}
            metadata = {'schema_version':SCHEMA_VERSION,'root_velocity_mode':mode,'run_id':run['run_id'],
                        'output_directory':run['output_directory'],
                        'source_csv':str(source),'source_csv_sha256':sha256_file(source),
                        'joint_names':['joint'],'body_names':['body']}
            path = output/'synthetic.npz'; save_npz_exclusive(path,arrays,metadata); before = sha256_file(path)
            with np.load(path,allow_pickle=False) as archive:
                assert str(archive['root_velocity_mode']) == mode
                assert json.loads(str(archive['conversion_metadata_json']))['run_id'] == run['run_id']
            try:
                save_npz_exclusive(path,arrays,metadata)
            except FileExistsError:
                failures += 1
            else:
                raise AssertionError('Existing NPZ overwritten')
            assert sha256_file(path) == before
        duplicate = base/'input.CSV'; duplicate.write_text('synthetic only\n')
        try:
            plan_batch_outputs(base,[source,duplicate],base/'batch','prefix')
        except FileExistsError:
            failures += 1
        else:
            raise AssertionError('Duplicate batch destinations accepted')
        # A read-only input view may link to an external original CSV.
        view = base/'input_view'; view.mkdir()
        alias = view/'external.csv'; alias.symlink_to(source)
        planned = plan_batch_outputs(view,[alias],base/'external_plan','prefix')
        assert planned == [(alias,base/'external_plan/prefix_external.npz')]
        assert sha256_file(alias.resolve()) == sha256_file(source)
        assert not (base/'external_plan').exists()
        source.write_text('changed synthetic source\n')
        try:
            save_npz_exclusive(base/'drift.npz',arrays,metadata)
        except RuntimeError:
            failures += 1
        else:
            raise AssertionError('Source CSV drift accepted')
        assert not (base/'drift.npz').exists()
    assert failures == 6
    result = {'status':'PASS','CPU_only':True,'synthetic_shape':[1,13],
              'read_only_external_input_CSV_symlink_accepted':True,
              'overwrite_collision_and_source_drift_rejections':failures,'native_conversion':'NOT_RUN'}
    print(json.dumps(result,sort_keys=True))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    self_test()
