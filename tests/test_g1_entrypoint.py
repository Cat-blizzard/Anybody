"""CPU-only G1 startup/ABI tests; no native app, conversion, or GPU.

The actual registry, entrypoint and CLI prefixes execute. Native configuration
imports are narrow fixtures. ABI tests use real CPU TensorDict/Tensor objects:
policy(B,815), critic(B,289), actions(B,29), rewards/dones(B,). Identity and RNG
checks distinguish container adaptation from a change in observations/control.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Iterator
import unittest

sys.dont_write_bytecode=True
import torch
from tensordict import TensorDict

ROOT=Path(__file__).resolve().parents[1]
AUDIT=Path('/data/jxc/projects/bones_curated/training/audit/anybody_g1_compat_startup_20261015')
PACKAGE=ROOT/'source/whole_body_tracking/whole_body_tracking'
NAME='whole_body_tracking.robots.robot_registry'
spec=importlib.util.spec_from_file_location('g1_entrypoint_cpu',ROOT/'scripts/run_g1_reference_script.py')
assert spec is not None and spec.loader is not None
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


@contextlib.contextmanager
def registry_fixture() -> Iterator[object]:
    saved={k:v for k,v in sys.modules.items() if k=='whole_body_tracking' or k.startswith('whole_body_tracking.')
           or k=='isaaclab' or k.startswith('isaaclab.')}
    for name in saved:sys.modules.pop(name)
    for name,directory in [('whole_body_tracking',PACKAGE),('whole_body_tracking.robots',PACKAGE/'robots'),
                           ('isaaclab',None),('isaaclab.assets',None)]:
        module=ModuleType(name);module.__path__=[] if directory is None else [str(directory)]
        if directory is not None:module.__file__=str(directory/'__init__.py')
        sys.modules[name]=module
    sys.modules['isaaclab.assets'].ArticulationCfg=object
    sentinel=object();g1=ModuleType('whole_body_tracking.robots.g1')
    g1.__file__=str(PACKAGE/'robots/g1.py');g1.G1_CYLINDER_CFG=sentinel;sys.modules[g1.__name__]=g1
    try:yield sentinel
    finally:
        for name in list(sys.modules):
            if name=='whole_body_tracking' or name.startswith('whole_body_tracking.') or name=='isaaclab' or name.startswith('isaaclab.'):
                sys.modules.pop(name)
        sys.modules.update(saved)


def load_registry(source: str | None=None) -> ModuleType:
    path=PACKAGE/'robots/robot_registry.py'
    spec=importlib.util.spec_from_file_location(NAME,path)
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec);sys.modules[NAME]=module
    if source is None:spec.loader.exec_module(module)
    else:exec(compile(source,str(path),'exec'),module.__dict__)
    return module


class StartupTests(unittest.TestCase):
    def test_actual_registry_missing_and_none_optional_robots(self) -> None:
        for old_stub in (False,True):
            with registry_fixture() as sentinel:
                if old_stub:
                    for name,attr in [('h1_2','H1_2_CYLINDER_CFG'),('adam','ADAM_CYLINDER_CFG')]:
                        module=ModuleType('whole_body_tracking.robots.'+name);setattr(module,attr,None)
                        sys.modules[module.__name__]=module
                registry=load_registry()
                assert registry.available_robot_names()==['g1']
                assert registry.get_robot_platform('g1').cfg is sentinel
                assert len(registry.get_robot_platform('g1').joint_names)==29
                for name in ('h1_2','adam'):
                    with self.assertRaises(KeyError):registry.get_robot_platform(name)

    def test_optional_internal_dependency_error_is_not_swallowed(self) -> None:
        class BrokenLoader(importlib.abc.Loader):
            def create_module(self,spec: Any) -> None:return None
            def exec_module(self,module: ModuleType) -> None:
                raise ModuleNotFoundError('Real robot dependency failure',name='real_robot_dependency')
        class BrokenFinder(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname: str,path: Any,target: Any=None) -> Any:
                if fullname=='whole_body_tracking.robots.h1_2':
                    return importlib.machinery.ModuleSpec(fullname,BrokenLoader())
                return None
        with registry_fixture():
            finder=BrokenFinder();sys.meta_path.insert(0,finder)
            try:
                with self.assertRaises(ModuleNotFoundError) as caught:load_registry()
                assert caught.exception.name=='real_robot_dependency'
            finally:sys.meta_path.remove(finder)

    def test_source_pin_and_explicit_robot_task_guard(self) -> None:
        entry.sanity_check()
        with registry_fixture():
            entry.assert_loaded_package_sources()
            module=sys.modules['whole_body_tracking.robots.g1']
            module.__file__='/data/jxc/projects/bones_curated/references/AnyBody/source/whole_body_tracking/whole_body_tracking/robots/g1.py'
            with self.assertRaises(RuntimeError):entry.assert_loaded_package_sources()
        finder=entry.PinnedPackageFinder()
        with self.assertRaises(RuntimeError):
            finder.find_spec('whole_body_tracking.assets',[str(ROOT.parent/'AnyBody/source/whole_body_tracking/whole_body_tracking')])

    def test_pickle_compatibility_keeps_existing_helper(self) -> None:
        module=ModuleType('isaaclab.utils.io');entry.add_dump_pickle(module)
        first=module.dump_pickle;entry.add_dump_pickle(module);assert module.dump_pickle is first
        with tempfile.TemporaryDirectory(prefix='pickle_fixture_',dir=AUDIT) as folder:
            path=Path(folder)/'params';module.dump_pickle(path,{'value':[1,2]})
            with Path(str(path)+'.pkl').open('rb') as handle:assert pickle.load(handle)=={'value':[1,2]}

    def test_real_tensordict_observation_and_step_identity(self) -> None:
        for batch in (1,3):
            policy=torch.randn(batch,815);critic=torch.randn(batch,289);history=torch.randn(batch,12)
            observations=TensorDict({'policy':policy,'critic':critic,'history':history},batch_size=[batch])
            rewards=torch.randn(batch);dones=torch.zeros(batch,dtype=torch.long);actions=torch.randn(batch,29)
            marker=torch.randn(1);extras={'marker':marker}
            class Wrapper:
                def get_observations(self) -> TensorDict:return observations
                def step(self,received: torch.Tensor) -> tuple[Any,...]:
                    assert received is actions
                    return observations,rewards,dones,extras
            module=ModuleType('isaaclab_rl.rsl_rl.vecenv_wrapper');module.RslRlVecEnvWrapper=Wrapper
            state=torch.random.get_rng_state().clone();entry.adapt_vecenv_observation_abi(module)
            wrapper=Wrapper();primary,info=wrapper.get_observations()
            assert primary is policy
            assert all(info['observations'][key] is observations[key] for key in observations.keys())
            p,r,d,new_extras=wrapper.step(actions)
            assert p is policy and r is rewards and d is dones and new_extras['marker'] is marker
            assert new_extras is not extras and 'observations' not in extras
            assert all(new_extras['observations'][key] is observations[key] for key in observations.keys())
            assert torch.equal(state,torch.random.get_rng_state())
            get,step=Wrapper.get_observations,Wrapper.step;entry.adapt_vecenv_observation_abi(module)
            assert Wrapper.get_observations is get and Wrapper.step is step
            print(f'ABI real TensorDict B={batch}: policy={(batch,815)}, critic={(batch,289)}, actions={(batch,29)}, identity/RNG PASS')

    def test_existing_legacy_tuple_is_returned_unchanged(self) -> None:
        obs=torch.zeros(1,815);legacy_get=(obs,{'observations':{'policy':obs}})
        actions=torch.zeros(1,29);legacy_step=(obs,torch.ones(1),torch.zeros(1),{'x':obs})
        class Wrapper:
            def get_observations(self) -> tuple[Any,...]:return legacy_get
            def step(self,value: torch.Tensor) -> tuple[Any,...]:
                assert value is actions
                return legacy_step
        module=ModuleType('isaaclab_rl.rsl_rl.vecenv_wrapper');module.RslRlVecEnvWrapper=Wrapper
        entry.adapt_vecenv_observation_abi(module)
        assert Wrapper().get_observations() is legacy_get and Wrapper().step(actions) is legacy_step

    def test_actual_converter_help_prefixes_and_shard_plan_no_native(self) -> None:
        with tempfile.TemporaryDirectory(prefix='CLI_fixture_',dir=AUDIT) as folder:
            temp=Path(folder);(temp/'isaaclab').mkdir();(temp/'isaaclab/__init__.py').write_text('')
            flag=temp/'unexpected_native'
            (temp/'isaaclab/app.py').write_text(
                "from pathlib import Path\nimport os\nclass AppLauncher:\n"
                " @staticmethod\n def add_app_launcher_args(parser):\n  parser.add_argument('--headless',action='store_true')\n"
                " def __init__(self,args):\n  Path(os.environ['G1_TEST_NATIVE_FLAG']).write_text('BAD')\n  raise RuntimeError('Native forbidden')\n")
            environment=dict(os.environ,PYTHONPATH=str(temp),PYTHONDONTWRITEBYTECODE='1',G1_TEST_NATIVE_FLAG=str(flag))
            for script in ('csv_to_npz.py','batch_csv_to_npz.py'):
                result=subprocess.run([sys.executable,str(ROOT/'scripts/run_g1_reference_script.py'),str(ROOT/'scripts'/script),'--help'],
                                      env=environment,text=True,capture_output=True)
                assert result.returncode==0,(script,result.stderr)
                assert '--root_velocity_mode' in result.stdout and not flag.exists()
            shards=temp/'csv';(shards/'0').mkdir(parents=True);(shards/'0/sample.csv').write_text('synthetic only\n')
            output=temp/'new_campaign'
            environment.update(DRY_RUN='1',CSV_PARENT=str(shards),OUT_DIR=str(output),FIRST_SHARD='0',LAST_SHARD='0',MOSAIC_ROOT=str(ROOT))
            result=subprocess.run(['bash',str(ROOT/'scripts/run_batch_csv_to_npz_shards.sh')],env=environment,text=True,capture_output=True)
            assert result.returncode==0 and 'run_g1_reference_script.py' in result.stdout,result.stderr
            assert not output.exists() and not flag.exists()


def preserve_original_failure() -> None:
    source=subprocess.check_output(['git','-C',str(ROOT),'show','8b60c1e:source/whole_body_tracking/whole_body_tracking/robots/robot_registry.py'],text=True)
    with registry_fixture():
        try:load_registry(source)
        except ModuleNotFoundError as error:
            assert error.name=='whole_body_tracking.robots.h1_2';failure=str(error)
        else:raise AssertionError('Original bare registry failure not reproduced')
    import datetime
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path=AUDIT/f'original_registry_failure_{stamp}.json'
    with path.open('x') as handle:
        json.dump({'base_commit':'8b60c1ef3cf26ee3185de9ab7efc34bcce4c234c','source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                   'failure':failure,'CPU_fixture_only':True,'native_calls':0},handle,indent=2);handle.write('\n')


if __name__=='__main__':
    AUDIT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    preserve_original_failure()
    unittest.main()
