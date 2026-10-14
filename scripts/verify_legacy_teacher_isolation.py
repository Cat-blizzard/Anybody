#!/usr/bin/env python3
"""CPU-only, fail-closed preservation check for the accepted legacy teacher.

Contract: never import IsaacLab/torch or change an environment/install/repository.
Capture hashes BOTH old repositories' tracked files, accepted C7000 checkpoints,
their receipts/configurations, ABC manifests/protocols, and editable import hooks.
The legacy C catalog is checked by membership, resolved targets and file stat;
three NPZ samples are checked by NPY headers/fps, NOT full NPZ payload hashes.
This is an isolation check, not fresh simulator/RNG restoration certification.

No arguments runs synthetic self-tests. Explicit --capture/--verify runs real
checks. All report writes are exclusive and confined to the new audit directory.
No old data, checkpoint, receipt, repository, or environment is written.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any, BinaryIO, Sequence
import zipfile

Json = dict[str, Any]
PROJECT = Path('/data/jxc/projects/bones_curated')
AUDIT = PROJECT / 'training/audit/anybody_teacher_correctness_fix_20261015'
LEGACY = PROJECT / 'references/AnyBody_resume_state_fix_20261006'
ORIGINAL = PROJECT / 'references/AnyBody'
NEW_TREE = PROJECT / 'references/AnyBody_teacher_correctness_fix_20261015'
ENVIRONMENT = Path('/data/jxc/envs/pbfm_isaaclab')
RUNS = PROJECT / 'training/runs/stage0_gmt'
ORIGINAL_RUN = RUNS / 'curation_abc_fresh_7000_20261008T090539Z'
REPAIR_RUN = RUNS / 'curation_abc_fresh_7000_repair_20261009T032624Z'
RETRY_RUN = RUNS / 'curation_abc_fresh_7000_retry_seed3_v3_20261010T055246Z'
PUBLICATION_RUN = RUNS / 'curation_abc_cpu_publication_repair_20261011T155844Z'
EXPECTED_HEAD = 'c319a6609158fdb7456c065800b9a489b2605f5b'
ARRAY_KEYS = {'fps', 'joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
              'body_lin_vel_w', 'body_ang_vel_w'}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Json:
    value = json.loads(path.read_text())
    assert isinstance(value, dict), f'Expected JSON object: {path}'
    return value


def file_identity(path: Path) -> Json:
    """Hash actual file bytes; reject a concurrent modification while reading."""
    assert path.is_file(), f'Missing file: {path}'
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return {'sha256': digest.hexdigest(), 'size_bytes': after.st_size,
            'symlink': os.readlink(path) if path.is_symlink() else None}


def git_output(root: Path, arguments: Sequence[str]) -> bytes:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    return subprocess.check_output(['git', '-C', str(root), *arguments], env=env)


def repository_identity(root: Path) -> Json:
    names = git_output(root, ['ls-files', '-z']).split(b'\0')
    files = {os.fsdecode(name): file_identity(root / os.fsdecode(name))
             for name in names if name}
    return {'root': str(root),
            'head': git_output(root, ['rev-parse', 'HEAD']).decode().strip(),
            'tracked_status': git_output(root, ['status', '--porcelain=v1',
                                           '--untracked-files=no']).decode().strip(),
            'full_status_informational': git_output(root, ['status', '--porcelain=v1']).decode().strip(),
            'tracked_file_count': len(files), 'files': files}


def repository_drift(expected: Json, actual: Json) -> list[str]:
    """Untracked cache directories are informational, never silently tracked."""
    keys = ('root', 'head', 'tracked_status', 'tracked_file_count', 'files')
    return [key for key in keys if expected[key] != actual[key]]


def validate_snapshot_contract(value: Json) -> None:
    """A verification baseline must actually identify BOTH designated old repos."""
    assert value['schema_version'] in (1, 2)
    repositories = value['repositories']
    assert isinstance(repositories, list) and len(repositories) == 2
    for repo, root, head, count in zip(
        repositories, (LEGACY, ORIGINAL),
        (EXPECTED_HEAD, '0cffa139eab0bc145c9216223e249f03c67a324c'), (562, 560), strict=True,
    ):
        assert repo['root'] == str(root) and repo['head'] == head
        assert repo['tracked_file_count'] == len(repo['files']) == count
    if value['schema_version'] == 2:
        assert set(value['ABC_manifests']) == {'A', 'B_scope', 'C_semantic_recovery'}
        assert value['legacy_C_catalog']['count'] == 14824
        assert value['C_recovery_additions'] == 433
        jobs = value['legacy_C7000_jobs']
        assert sorted(job['train_seed'] for job in jobs) == [20260904, 20260905, 20260906]
        assert all(job['source_commit'] == EXPECTED_HEAD and job['completed_updates'] == 7000
                   and job['next_learning_iteration'] == 7000 and job['Adam_steps'] == 140000
                   for job in jobs)


def safe_output(path: Path) -> Path:
    resolved = path.resolve()
    assert resolved.is_relative_to(AUDIT.resolve()), f'Output outside new audit: {path}'
    assert resolved != AUDIT.resolve(), 'Output must be a file'
    assert not path.exists() and not path.is_symlink(), f'Refusing overwrite: {path}'
    return resolved


def write_exclusive(path: Path, value: Json) -> None:
    target = safe_output(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def npy_header(handle: BinaryIO) -> Json:
    """Parse only a bounded .npy header; never unpickle object arrays."""
    assert handle.read(6) == b'\x93NUMPY', 'Invalid NPY magic'
    version = handle.read(2)
    assert version in (b'\x01\x00', b'\x02\x00', b'\x03\x00'), version
    width = 2 if version[0] == 1 else 4
    raw_length = handle.read(width)
    assert len(raw_length) == width
    length = struct.unpack('<H' if width == 2 else '<I', raw_length)[0]
    assert 0 < length <= 65536, 'Invalid/oversized NPY header'
    raw = handle.read(length)
    assert len(raw) == length
    header = ast.literal_eval(raw.decode('utf-8' if version[0] == 3 else 'latin1'))
    assert isinstance(header, dict) and set(header) == {'descr', 'fortran_order', 'shape'}
    dtype = header['descr']
    assert isinstance(dtype, str) and len(dtype) >= 3 and dtype[1] in 'fiu', dtype
    assert dtype[0] in '<>=|', dtype
    shape = header['shape']
    assert isinstance(shape, tuple) and all(isinstance(n, int) and n >= 0 for n in shape)
    assert isinstance(header['fortran_order'], bool)
    return {'dtype': dtype, 'shape': list(shape), 'fortran_order': header['fortran_order']}


def npz_schema(path: Path) -> Json:
    """Legacy headers: q(T,29), bodies(T,30,3/4), positive scalar 50 Hz fps."""
    headers: Json = {}
    fps = 0.0
    with zipfile.ZipFile(path) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())), 'Duplicate ZIP members'
        for name in archive.namelist():
            assert name.endswith('.npy') and '/' not in name, name
            with archive.open(name) as handle:
                header = npy_header(handle)
                headers[name[:-4]] = header
                if name == 'fps.npy':
                    assert math.prod(header['shape']) == 1
                    formats = {'i8': 'q', 'i4': 'i', 'u8': 'Q', 'u4': 'I', 'f8': 'd', 'f4': 'f'}
                    dtype = header['dtype']
                    assert dtype[1:] in formats, dtype
                    fmt = ('>' if dtype[0] == '>' else '<') + formats[dtype[1:]]
                    raw = handle.read(struct.calcsize(fmt))
                    assert len(raw) == struct.calcsize(fmt)
                    fps = float(struct.unpack(fmt, raw)[0])
    assert ARRAY_KEYS.issubset(headers), f'Missing loader keys: {ARRAY_KEYS-set(headers)}'
    assert math.isfinite(fps) and fps == 50.0, fps
    length = headers['joint_pos']['shape'][0]
    assert length >= 1
    for key in ('joint_pos', 'joint_vel'):
        assert headers[key]['shape'] == [length, 29], (key, headers[key])
    for key in ('body_pos_w', 'body_lin_vel_w', 'body_ang_vel_w'):
        assert headers[key]['shape'] == [length, 30, 3], (key, headers[key])
    assert headers['body_quat_w']['shape'] == [length, 30, 4]
    return {'headers': headers, 'fps': fps, 'payload_content_hashed': False}


def legacy_loader_contract() -> Json:
    path = LEGACY / 'source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'MultiMotionLoader')
    keys = {node.slice.value for node in ast.walk(cls)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == 'data' and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)}
    assert keys == ARRAY_KEYS, keys
    return {'path': str(path), 'source_sha256': file_identity(path)['sha256'],
            'class_AST_sha256': hashlib.sha256(ast.dump(cls).encode()).hexdigest(),
            'required_NPZ_keys': sorted(keys), 'joint_count': 29, 'full_body_count': 30,
            'quaternion_order': 'WXYZ', 'positions': 'world link origins, meters',
            'linear_velocities': 'native body COM velocity; legacy converter retained',
            'point_semantics_not_repaired_by_this_check': True}


def catalog_identity(root: Path) -> Json:
    entries: Json = {}
    for path in sorted(root.rglob('*.npz')):
        assert path.is_file(), f'Broken legacy motion link: {path}'
        resolved = path.resolve()
        assert not resolved.is_relative_to(NEW_TREE), f'Legacy motion points into repair tree: {path}'
        stat = path.stat()
        entries[str(path.relative_to(root))] = {
            'resolved_path': str(resolved), 'size_bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
            'symlink': os.readlink(path) if path.is_symlink() else None}
    assert len(entries) == 14824, len(entries)
    return {'root': str(root), 'count': len(entries), 'entries': entries,
            'all_NPZ_payloads_hashed': False}


def explicit_legacy_imports(pythonpath: str) -> Json:
    """Resolve top-level packages without importing their executable __init__."""
    paths = [str(Path(p).resolve()) for p in pythonpath.split(os.pathsep) if p]
    result: Json = {}
    for package, relative in [('whole_body_tracking', 'source/whole_body_tracking'),
                              ('rsl_rl', 'source/rsl_rl')]:
        spec = importlib.machinery.PathFinder.find_spec(package, paths)
        expected = (LEGACY / relative / package / '__init__.py').resolve()
        assert spec is not None and spec.origin is not None, f'Cannot resolve {package}'
        assert Path(spec.origin).resolve() == expected, (package, spec.origin, expected)
        result[package] = {'origin': str(expected), 'package_not_imported': True}
    return result


def capture(pre_capture: Path | None) -> Json:
    repos = [repository_identity(LEGACY), repository_identity(ORIGINAL)]
    assert repos[0]['head'] == EXPECTED_HEAD and repos[0]['tracked_file_count'] == 562
    assert repos[0]['tracked_status'] == '', 'Legacy teacher tracked state must stay clean'
    assert repos[1]['head'] == '0cffa139eab0bc145c9216223e249f03c67a324c'
    assert repos[1]['tracked_file_count'] == 560
    assert repos[1]['tracked_status'] == 'M source/rsl_rl/rsl_rl/algorithms/ppo.py'
    if pre_capture is not None:
        baseline = read_json(pre_capture)
        validate_snapshot_contract(baseline)
        assert len(baseline['repositories']) == len(repos)
        for before, after in zip(baseline['repositories'], repos, strict=True):
            assert not repository_drift(before, after), f'Pre-capture drift: {before["root"]}'
    files: Json = {}

    def add(path: Path, expected_sha: str | None = None) -> None:
        record = files.setdefault(str(path), file_identity(path))
        if expected_sha is not None:
            assert record['sha256'] == expected_sha, f'Receipt SHA mismatch: {path}'

    for run in (ORIGINAL_RUN, REPAIR_RUN, RETRY_RUN, PUBLICATION_RUN):
        add(run / 'protocol.json')
    protocol = read_json(REPAIR_RUN / 'protocol.json')
    assert protocol['source_commit'] == EXPECTED_HEAD and protocol['source_repo'] == str(LEGACY)
    manifests: Json = {}
    rows_by_condition: dict[str, list[dict[str, str]]] = {}
    for condition, expected_count in [('A', 16168), ('B_scope', 14391), ('C_semantic_recovery', 14824)]:
        item = protocol['conditions'][condition]
        path = Path(item['manifest'])
        add(path, item['manifest_sha256'])
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        ids = [row['motion_id'] for row in rows]
        assert len(ids) == len(set(ids)) == expected_count
        assert all(len(row['npz_sha256']) == 64 for row in rows)
        rows_by_condition[condition] = rows
        manifests[condition] = {'path': str(path), 'sha256': files[str(path)]['sha256'],
                                'motion_count': len(rows), 'motion_view': item['motion_view']}
    bids = {row['motion_id'] for row in rows_by_condition['B_scope']}
    cids = {row['motion_id'] for row in rows_by_condition['C_semantic_recovery']}
    assert bids < cids and len(cids - bids) == 433
    for name in ('data_contract.json', 'final_data_receipt.json', 'final_data_input_lock.json'):
        add(ORIGINAL_RUN / 'data' / name)
    add(Path(protocol['compatibility_wrapper']))
    master_path = PUBLICATION_RUN / 'training/master_complete.json'
    add(master_path)
    master = read_json(master_path)
    jobs: list[Json] = []
    for job in master['original_training_completions']:
        if job['condition'] != 'C_semantic_recovery':
            continue
        cp_ref, completion_ref = job['native_checkpoint'], job['completion']
        cp, completion_path = Path(cp_ref['path']), Path(completion_ref['path'])
        add(cp, cp_ref['sha256']); add(completion_path, completion_ref['sha256'])
        completion = read_json(completion_path)
        assert completion['status'] == 'passed' and completion['launcher_exit_code'] == 0
        assert completion['completed_updates'] == 7000 and completion['Adam_steps'] == 140000
        assert completion['train_seed'] == job['train_seed']
        ready_ref = completion['milestones_ready']['7000']
        ready_path = Path(ready_ref['path']); add(ready_path, ready_ref['sha256'])
        ready = read_json(ready_path)
        assert ready['source_commit'] == EXPECTED_HEAD and ready['technical_validation'] == 'passed'
        assert ready['completed_updates'] == ready['next_learning_iteration'] == 7000
        assert ready['phase'] == 6999 and ready['condition'] == 'C_semantic_recovery'
        add(Path(ready['checkpoint']), ready['checkpoint_sha256'])
        add(Path(ready['cpu_audit']), ready['cpu_audit_sha256'])
        assert ready['checkpoint_sha256'] == cp_ref['sha256']
        for name in ('env.yaml', 'agent.yaml'):
            add(cp.parent / 'params' / name)
        jobs.append({'train_seed': job['train_seed'], 'native_checkpoint': cp_ref,
                     'completion': completion_ref, 'ready': ready_ref,
                     'source_commit': ready['source_commit'], 'completed_updates': 7000,
                     'next_learning_iteration': 7000, 'Adam_steps': 140000,
                     'full_simulator_RNG_sampler_state_restored': False})
    assert sorted(job['train_seed'] for job in jobs) == [20260904, 20260905, 20260906]
    environment_hooks: list[str] = []
    for site in ENVIRONMENT.glob('lib/python*/site-packages'):
        hooks = list(site.glob('*.pth')) + list(site.glob('__editable__*finder.py'))
        hooks += list(site.glob('*whole*dist-info/direct_url.json'))
        hooks += list(site.glob('*rsl*dist-info/direct_url.json'))
        for path in sorted(set(hooks)):
            add(path); environment_hooks.append(str(path))
    assert environment_hooks, 'Missing shared environment import metadata'
    crows = rows_by_condition['C_semantic_recovery']
    recovery = next(row for row in crows if row['motion_id'] not in bids)
    sample_paths = sorted({row['npz_path'] for row in (crows[0], crows[-1], recovery)})
    samples = {path: npz_schema(Path(path)) for path in sample_paths}
    return {'schema_version': 2, 'created_at_utc': utc_now(), 'status': 'CAPTURED',
            'CPU_only': True, 'new_physical_calls': 0, 'repositories': repos,
            'pre_capture': {'path': str(pre_capture), 'sha256': file_identity(pre_capture)['sha256']}
            if pre_capture else None, 'files': files, 'ABC_manifests': manifests,
            'C_recovery_additions': 433, 'legacy_C7000_jobs': jobs,
            'legacy_loader': legacy_loader_contract(),
            'legacy_C_catalog': catalog_identity(Path(manifests['C_semantic_recovery']['motion_view'])),
            'NPZ_header_samples': samples, 'shared_environment_import_hooks': environment_hooks,
            'legacy_explicit_import_resolution': explicit_legacy_imports(
                str(LEGACY/'source/whole_body_tracking')+os.pathsep+str(LEGACY/'source/rsl_rl')),
            'limitations': ['No full NPZ payload rehash; header/stat/membership checks only.',
                            'Existing CPU checkpoint receipts bound by actual checkpoint bytes; not re-executed.',
                            'No new native simulation, training, import, or resume certification.',
                            'Future C run still needs its own protocol and admission checks.']}


def verify(snapshot: Path, pythonpath: str | None) -> Json:
    expected = read_json(snapshot)
    validate_snapshot_contract(expected)
    checked = 0
    for repo in expected['repositories']:
        actual = repository_identity(Path(repo['root']))
        assert not repository_drift(repo, actual), f'Legacy repository changed: {repo["root"]}'
        checked += len(repo['files'])
    for name, identity in expected.get('files', {}).items():
        assert file_identity(Path(name)) == identity, f'Preserved file changed: {name}'
        checked += 1
    if expected['schema_version'] == 2:
        assert legacy_loader_contract() == expected['legacy_loader'], 'Legacy loader changed'
        catalog = expected['legacy_C_catalog']
        assert catalog_identity(Path(catalog['root'])) == catalog, 'Legacy C catalog changed'
        for name, header in expected['NPZ_header_samples'].items():
            assert npz_schema(Path(name)) == header, f'Legacy NPZ header changed: {name}'
    imports = explicit_legacy_imports(pythonpath) if pythonpath is not None else None
    return {'schema_version': 1, 'created_at_utc': utc_now(), 'status': 'passed',
            'CPU_only': True, 'new_physical_calls': 0,
            'snapshot': {'path': str(snapshot), 'sha256': file_identity(snapshot)['sha256']},
            'content_hashed_files_checked': checked,
            'legacy_C_catalog_checked': expected['schema_version'] == 2,
            'legacy_explicit_import_resolution': imports,
            'full_NPZ_payload_rehash': False, 'new_native_resume_validation': 'NOT_RUN'}


def self_test() -> Json:
    """Synthetic byte mutation, schema, boundary, no-overwrite and path tests."""
    AUDIT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='isolation_selftest_', dir=AUDIT) as folder:
        root = Path(folder); path = root/'example.py'; path.write_text('x = 1\n')
        before = file_identity(path); assert file_identity(path) == before
        path.write_text('x = 2\n'); assert file_identity(path) != before
        fake: Json = {'root': 'synthetic', 'head': 'a', 'tracked_status': '',
                      'tracked_file_count': 1, 'files': {'example.py': before}}
        assert repository_drift(fake, dict(fake)) == []
        for key, value in [('head', 'b'), ('tracked_status', 'M example.py'),
                           ('tracked_file_count', 2), ('files', {})]:
            changed = dict(fake); changed[key] = value
            assert repository_drift(fake, changed) == [key]
        try: validate_snapshot_contract({'schema_version': 1, 'repositories': []})
        except AssertionError: pass
        else: raise AssertionError('Empty baseline accepted')
        target = root/'output.json'; write_exclusive(target, {'finite': 0.0})
        for output in (target, Path('/tmp/forbidden_isolation_output.json')):
            try: safe_output(output)
            except AssertionError: pass
            else: raise AssertionError('Overwrite/path escape accepted')
        for shape, dtype in [((1, 29), '<f4'), ((1,), '<i8')]:
            import io
            header = repr({'descr': dtype, 'fortran_order': False, 'shape': shape}).encode()
            blob = b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header
            assert npy_header(io.BytesIO(blob))['shape'] == list(shape)
        import io
        object_header = repr({'descr': '|O8', 'fortran_order': False, 'shape': (1,)}).encode()
        blob=b'\x93NUMPY\x01\x00'+struct.pack('<H',len(object_header))+object_header
        try: npy_header(io.BytesIO(blob))
        except AssertionError: pass
        else: raise AssertionError('Object array accepted')
        npz = root/'singleton.npz'
        with zipfile.ZipFile(npz, 'w') as archive:
            for key in sorted(ARRAY_KEYS):
                shape = (1,) if key=='fps' else ((1,29) if key.startswith('joint_') else
                        (1,30,4 if key=='body_quat_w' else 3))
                dtype = '<i8' if key=='fps' else '<f4'
                header = repr({'descr':dtype,'fortran_order':False,'shape':shape}).encode()
                payload=struct.pack('<q',50) if key=='fps' else bytes(math.prod(shape)*4)
                archive.writestr(key+'.npy', b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+payload)
        schema=npz_schema(npz); assert schema['fps']==50 and schema['headers']['joint_pos']['shape']==[1,29]
        assert not schema['payload_content_hashed']
        expected_pythonpath=str(LEGACY/'source/whole_body_tracking')+os.pathsep+str(LEGACY/'source/rsl_rl')
        resolved=explicit_legacy_imports(expected_pythonpath)
        assert all(record['package_not_imported'] for record in resolved.values())
        try: explicit_legacy_imports(str(NEW_TREE/'source/whole_body_tracking')+os.pathsep+str(NEW_TREE/'source/rsl_rl'))
        except AssertionError: pass
        else: raise AssertionError('Repair source accepted as legacy import path')
    return {'status':'passed','CPU_only':True,'new_physical_calls':0,
            'synthetic_shape':[1,29],'mutation_and_HEAD_detection':True,
            'object_array_rejected':True,'overwrite_and_path_escape_rejected':True,
            'empty_baseline_rejected':True,
            'legacy_import_identity_checked_without_import':True}


def main() -> None:
    if not __debug__:
        raise RuntimeError('Python -O is forbidden: isolation assertions must be enabled')
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--capture',action='store_true')
    mode.add_argument('--verify',type=Path)
    parser.add_argument('--pre-capture',type=Path)
    parser.add_argument('--legacy-pythonpath',type=str)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.capture or args.verify:
        assert args.output is not None, 'Explicit new audit --output required'
        safe_output(args.output)
        result=capture(args.pre_capture) if args.capture else verify(args.verify,args.legacy_pythonpath)
    else:
        assert args.pre_capture is None and args.legacy_pythonpath is None
        result=self_test()
    if args.output is not None:
        write_exclusive(args.output,result)
    if not (args.capture or args.verify):
        print(json.dumps(result,sort_keys=True))
    print(json.dumps({'status':result['status'],'output':str(args.output) if args.output else None,
                      'self_test':not (args.capture or args.verify),
                      'CPU_only':True,'new_physical_calls':0},sort_keys=True))


if __name__ == '__main__':
    main()
