"""CPU checks of the unmodified production selector and FullB-only overrides.

Inputs:86128 distinct path strings. Eight native selector invocations must form
disjoint10766-member shards, with union exactly the original set. Single-rank
must retain86128. No simulation, payload loading, or RNG mutation is required.
"""
from __future__ import annotations

import ast
import hashlib
import math
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REL = 'source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py'
CONFIG = ROOT / 'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py'


def native_selector() -> Any:
    raw = (ROOT / REL).read_bytes()
    base = subprocess.check_output(['git', '-C', str(ROOT), 'show', '306c5d0:' + REL])
    assert raw == base, 'FullB coverage must not alter public commands.py logic/defaults'
    names = {'_get_rank_world_size', '_select_motion_paths_for_rank'}
    body = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(body) == 2
    ns: dict[str, Any] = {'os': os, 'math': math, 'random': random}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), REL, 'exec'), ns)
    return ns['_select_motion_paths_for_rank']


def profile_values() -> dict[str, Any]:
    tree = ast.parse(CONFIG.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'G1OneStageFullBPointRewardsV1EnvCfg')
    post = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__post_init__')
    assignments = [n for n in post.body if isinstance(n, ast.Assign)]
    values: dict[str, Any] = {}
    for assignment in assignments:
        assert len(assignment.targets) == 1
        target = ast.unparse(assignment.targets[0])
        assert target.startswith('self.commands.motion.')
        values[target.rsplit('.', 1)[1]] = ast.literal_eval(assignment.value)
    assert values == {'motion_dataset_load_cap': 86128, 'motion_dataset_shard_across_gpus': True,
                      'motion_dataset_shard_by': 'global', 'motion_dataset_shard_strategy': 'chunk',
                      'motion_dataset_shard_seed': 0, 'motion_dataset_log_shard_info': True,
                      'motion_preload_device': 'cpu'}
    assert cls.bases[0].id == 'G1OneStagePointRewardsV1EnvCfg'
    return values


def check() -> None:
    select = native_selector()
    cfg = profile_values()
    paths = [f'motion_{i:06d}.npz' for i in range(86128)]
    original_rng = random.getstate()
    previous = {k: os.environ.get(k) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE')}
    kwargs = {'max_motions': cfg['motion_dataset_load_cap'], 'shard_across_gpus': True,
              'shard_by': 'global', 'shard_seed': 0, 'shard_strategy': 'chunk'}
    try:
        shards = []
        for rank in range(8):
            os.environ.update(RANK=str(rank), WORLD_SIZE='8', LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE='8')
            selected, info = select(paths, **kwargs)
            assert len(selected) == len(set(selected)) == 10766
            assert info['rank'] == rank and info['world_size'] == 8
            assert info['total_motions'] == 86128 and info['selected_motions'] == 10766
            shards.append(set(selected))
        assert sum(map(len, shards)) == 86128
        assert set.union(*shards) == set(paths)
        assert all(not shards[i] & shards[j] for i in range(8) for j in range(i))
        os.environ.update(RANK='0', WORLD_SIZE='1', LOCAL_RANK='0', LOCAL_WORLD_SIZE='1')
        single, info = select(paths, **kwargs)
        assert len(single) == 86128 and set(single) == set(paths)
        assert info['rank'] == 0 and info['world_size'] == 1
        # Reproduce the reported default cap omission without changing the selector.
        truncated = []
        for rank in range(8):
            os.environ.update(RANK=str(rank), WORLD_SIZE='8')
            selected, _ = select(paths, **dict(kwargs, max_motions=4096))
            truncated.extend(selected)
        assert len(set(truncated)) == 32768
        assert len(set(paths) - set(truncated)) == 53360
        assert random.getstate() == original_rng
        print('PASS config-only;8x10766=86128 exact disjoint union;single86128;default8x4096 leaves53360;globalRNG unchanged')
        print('commands.py sha256=' + hashlib.sha256((ROOT / REL).read_bytes()).hexdigest())
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == '__main__':
    check()
