"""CPU check of the real Isaac configclass serialization used by Full-B PPO.

Contract: the serialized algorithm dictionary must contain the opt-in strict
gate, while a fresh legacy profile keeps its original fields and values.
Only class definitions are extracted from the installed/configured source to
avoid importing the simulator package initializers; configclass and its
serialization helpers execute their original installed implementations.
No CUDA context, simulator, policy rollout, or training is created here.
"""
from __future__ import annotations

import ast
from dataclasses import MISSING
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Literal


REPO = Path(__file__).resolve().parents[1]
LAB = Path('/data/jxc/cache/nvidia/IsaacLab-v2.3.2/source/isaaclab/isaaclab/utils')
RL_CFG = Path('/data/jxc/cache/nvidia/IsaacLab-v2.3.2/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py')
PROFILE = REPO / 'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py'


def installed_configclass() -> Any:
    """Load the installed utility under a private package, bypassing mesh/pxr."""
    name = '_full_b_cpu_isaac_utils'
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(LAB)]
        sys.modules[name] = package
    full_name = name + '.configclass'
    if full_name not in sys.modules:
        specification = importlib.util.spec_from_file_location(full_name, LAB / 'configclass.py')
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        sys.modules[full_name] = module
        specification.loader.exec_module(module)
    return sys.modules[full_name].configclass


def execute_classes(path: Path, names: tuple[str, ...], namespace: dict[str, Any]) -> None:
    """Execute exact AST class bodies in source order, with postponed hints."""
    parsed = ast.parse(path.read_text(), filename=str(path))
    definitions = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert tuple(node.name for node in definitions) == names, (path, names)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    extracted = ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[]))
    exec(compile(extracted, str(path), 'exec'), namespace)


def self_test() -> None:
    import torch
    assert not torch.cuda.is_initialized()
    module_name = '_full_b_serialization_fixture'
    module = ModuleType(module_name)
    module.__file__ = str(PROFILE)
    sys.modules[module_name] = module
    namespace = module.__dict__
    namespace.update(configclass=installed_configclass(), MISSING=MISSING, Literal=Literal)
    execute_classes(RL_CFG, ('RslRlPpoActorCriticCfg', 'RslRlPpoAlgorithmCfg',
                            'RslRlBaseRunnerCfg', 'RslRlOnPolicyRunnerCfg'), namespace)
    execute_classes(PROFILE, ('G1FlatPPORunnerCfg', 'G1FullBPPORunnerCfg'), namespace)
    legacy_before = namespace['G1FlatPPORunnerCfg']().to_dict()
    full = namespace['G1FullBPPORunnerCfg']()
    actual = full.to_dict()
    legacy_after = namespace['G1FlatPPORunnerCfg']().to_dict()
    assert legacy_before == legacy_after, 'Full-B profile mutated shared legacy config'
    assert 'strict_finite_checks' not in legacy_after['algorithm']
    assert actual['algorithm']['strict_finite_checks'] is True
    algorithm = dict(actual['algorithm'])
    del algorithm['strict_finite_checks']
    assert algorithm == legacy_after['algorithm'], 'Unrequested PPO hyperparameter change'
    allowed = {'max_iterations', 'save_interval', 'experiment_name',
               'distributed_normalizer_sync', 'normalizer_until_count_scope', 'algorithm'}
    assert {key: value for key, value in actual.items() if key not in allowed} == {
        key: value for key, value in legacy_after.items() if key not in allowed}
    assert actual['max_iterations'] == 100000 and actual['save_interval'] == 1000
    assert actual['distributed_normalizer_sync'] is True
    assert actual['normalizer_until_count_scope'] == 'per_rank_equivalent'
    assert actual['num_steps_per_env'] == 24
    assert actual['algorithm']['num_learning_epochs'] * actual['algorithm']['num_mini_batches'] == 20
    assert not torch.cuda.is_initialized()
    files = [PROFILE, RL_CFG, *(LAB / name for name in ('configclass.py', 'dict.py', 'array.py', 'string.py'))]
    print(json.dumps({'status': 'PASS', 'CPU_only': True,
                      'actual_installed_serialization': True,
                      'strict_gate_serialized': True, 'legacy_config_unchanged': True,
                      'PPO_hyperparameters_unchanged': True,
                      'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}},
                     sort_keys=True))


if __name__ == '__main__':
    self_test()
