#!/usr/bin/env python3
"""Run a G1 script with this worktree's packages and metadata/observation ABI.

Usage: python run_g1_reference_script.py SCRIPT [SCRIPT_ARGS ...]
       python run_g1_reference_script.py --self-test

No missing robot is replaced by G1 or by a fake configuration. The registry
only advertises available robot configs. Package imports are pinned to this
worktree and previously imported conflicting sources are rejected before the
script runs. Only missing dump_pickle and the legacy RSL observation return
ABI are adapted; tensors, actions, rewards, dones, sampling and RNG are kept.
This wrapper itself never constructs a simulator or trains a model.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import pickle
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOTS = {
    "whole_body_tracking": ROOT/"source/whole_body_tracking/whole_body_tracking",
    "rsl_rl": ROOT/"source/rsl_rl/rsl_rl",
}


def validate_g1_arguments(arguments: Sequence[str]) -> None:
    """Reject explicit non-G1 targets; the real script retains full parsing."""
    for index, argument in enumerate(arguments):
        for flag in ("--robot", "--task"):
            if argument == flag:
                if index+1 >= len(arguments):
                    raise ValueError(f"Missing value for {flag}")
                value = arguments[index+1]
            elif argument.startswith(flag+"="):
                value = argument.split("=",1)[1]
            else:
                continue
            if flag == "--robot" and value != "g1":
                raise ValueError(f"G1-only entrypoint rejects robot {value!r}")
            if flag == "--task" and "-G1-" not in value:
                raise ValueError(f"G1-only entrypoint rejects task {value!r}")


def assert_loaded_package_sources() -> None:
    for name, module in list(sys.modules.items()):
        for prefix, directory in PACKAGE_ROOTS.items():
            if name != prefix and not name.startswith(prefix+"."):
                continue
            filename = getattr(module,"__file__",None)
            if filename is not None:
                if not Path(filename).resolve().is_relative_to(directory.resolve()):
                    raise RuntimeError(f"Already loaded {name} from conflicting source: {filename}")
            else:
                locations = getattr(module,"__path__",None)
                if not locations or not all(Path(p).resolve().is_relative_to(directory.resolve()) for p in locations):
                    raise RuntimeError(f"Unbound/stub {name} is forbidden in the G1 entrypoint")


class PinnedPackageFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Sequence[str] | None,
                  target: ModuleType | None = None) -> importlib.machinery.ModuleSpec | None:
        for prefix,directory in PACKAGE_ROOTS.items():
            if fullname != prefix and not fullname.startswith(prefix+"."):
                continue
            search = [str(directory.parent)] if fullname == prefix else path
            spec = importlib.machinery.PathFinder.find_spec(fullname,search)
            if spec is None:
                raise ModuleNotFoundError(f"No module named {fullname!r} in pinned G1 worktree",name=fullname)
            if spec.origin is not None:
                if not Path(spec.origin).resolve().is_relative_to(directory.resolve()):
                    raise RuntimeError(f"Import {fullname} escapes pinned source: {spec.origin}")
            else:
                locations=spec.submodule_search_locations
                if not locations or not all(Path(p).resolve().is_relative_to(directory.resolve()) for p in locations):
                    raise RuntimeError(f"Namespace {fullname} escapes pinned source")
            return spec
        return None


def install_source_pin() -> None:
    assert all((path/"__init__.py").is_file() for path in PACKAGE_ROOTS.values())
    assert_loaded_package_sources()
    for directory in reversed(list(PACKAGE_ROOTS.values())):
        parent=str(directory.parent)
        if parent in sys.path: sys.path.remove(parent)
        sys.path.insert(0,parent)
    if not any(isinstance(finder,PinnedPackageFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0,PinnedPackageFinder())


def add_dump_pickle(module: ModuleType) -> None:
    if hasattr(module,"dump_pickle"):
        return

    def dump_pickle(filename: str | Path, data: object) -> None:
        path=Path(filename)
        if path.suffix != ".pkl": path=Path(str(path)+".pkl")
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(data,handle)

    module.dump_pickle=dump_pickle


def adapt_vecenv_observation_abi(module: ModuleType) -> None:
    """Only change return containers of the actual RSL wrapper module.

    Policy/critic/group tensors keep object identity. Existing legacy tuple
    observations and tensor-first step results are returned transparently.
    No action, reward, done, RNG, or observation-computation method is added.
    """
    if module.__name__ != "isaaclab_rl.rsl_rl.vecenv_wrapper":
        raise ValueError("Observation ABI adaptation requires the exact RSL wrapper module")
    cls=getattr(module,"RslRlVecEnvWrapper",None)
    if cls is None or getattr(cls,"_g1_legacy_obs_contract_installed",False):
        return
    original_get,original_step=cls.get_observations,cls.step

    def get_observations(self: Any) -> Any:
        result=original_get(self)
        if isinstance(result,tuple) and len(result)==2:
            return result
        assert hasattr(result,"keys") and len(result.keys())>0
        primary=result["policy"] if "policy" in result else next(iter(result.values()))
        return primary,{"observations":{key:result[key] for key in result.keys()}}

    def step(self: Any, actions: Any) -> Any:
        result=original_step(self,actions)
        if not isinstance(result,tuple) or len(result)!=4:
            return result
        observations,rewards,dones,extras=result
        if hasattr(observations,"keys"):
            assert len(observations.keys())>0
            primary=observations["policy"] if "policy" in observations else next(iter(observations.values()))
            new_extras=dict(extras)
            new_extras["observations"]={key:observations[key] for key in observations.keys()}
            return primary,rewards,dones,new_extras
        return result

    cls.get_observations=get_observations
    cls.step=step
    cls._g1_legacy_obs_contract_installed=True


class CompatibilityLoader(importlib.abc.Loader):
    def __init__(self,wrapped: importlib.abc.Loader,fullname: str) -> None:
        self.wrapped,self.fullname=wrapped,fullname

    def create_module(self,spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        create=getattr(self.wrapped,"create_module",None)
        return create(spec) if create is not None else None

    def exec_module(self,module: ModuleType) -> None:
        self.wrapped.exec_module(module)
        if self.fullname=="isaaclab.utils.io": add_dump_pickle(module)
        else: adapt_vecenv_observation_abi(module)


class CompatibilityFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname: str,path: Sequence[str] | None,
                  target: ModuleType | None = None) -> importlib.machinery.ModuleSpec | None:
        if fullname not in ("isaaclab.utils.io","isaaclab_rl.rsl_rl.vecenv_wrapper"):
            return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is not None and spec.loader is not None:
            spec.loader=CompatibilityLoader(spec.loader,fullname)
        return spec


def install_api_compatibility() -> None:
    for name,adapter in (("isaaclab.utils.io",add_dump_pickle),
                         ("isaaclab_rl.rsl_rl.vecenv_wrapper",adapt_vecenv_observation_abi)):
        module=sys.modules.get(name)
        if module is not None: adapter(module)
    if not any(isinstance(finder,CompatibilityFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0,CompatibilityFinder())


def sanity_check() -> None:
    validate_g1_arguments(["--robot=g1","--task=General-Tracking-Flat-G1-Teacher-Cache-v1"])
    for arguments in (["--robot","h1_2"],["--task=Any-H1-Task"],["--robot"]):
        try: validate_g1_arguments(arguments)
        except ValueError: pass
        else: raise AssertionError("Non-G1/invalid arguments must be rejected")
    finder=PinnedPackageFinder()
    for package,path in PACKAGE_ROOTS.items():
        spec=finder.find_spec(package,None)
        assert spec is not None and Path(spec.origin).resolve()==(path/"__init__.py").resolve()
    print("G1 entrypoint sanity: packages=2, explicit wrong robot/task rejected, CPU-only PASS")


def main() -> None:
    arguments=sys.argv[1:]
    if arguments==["--self-test"]:
        sanity_check();return
    if not arguments or arguments in (["--help"],["-h"]):
        print(__doc__);return
    script=Path(arguments[0]).expanduser().resolve(strict=True)
    assert script.is_file() and script.suffix==".py"
    validate_g1_arguments(arguments[1:])
    install_source_pin()
    install_api_compatibility()
    sys.path.insert(0,str(script.parent))
    sys.argv=[str(script),*arguments[1:]]
    try:
        runpy.run_path(str(script),run_name="__main__")
    finally:
        assert_loaded_package_sources()


if __name__=="__main__":
    main()
