"""CPU regression for converter-only final shutdown ordering.

No Isaac/Kit/GPU import: execute the installed official clear_instance method
in a synthetic class. Check both real converter sources against the frozen
pre-fix commit after removing only the new final teardown wrapper. This proves
that loader, numerical processing, physics writes and loop ASTs are unchanged.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BASELINE = '2a28aa61618752ae109411dec8b619cbb73b3a99'
INSTALLED_SOURCE = Path('/data/jxc/cache/nvidia/IsaacLab-v2.3.2/source/isaaclab/isaaclab/sim/simulation_context.py')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_clear(node: ast.AST) -> bool:
    """Recognize only the exact official teardown call, without accepting extras."""
    return ast.dump(node, include_attributes=False) == ast.dump(
        ast.parse('SimulationContext.clear_instance()').body[0], include_attributes=False)


def is_close(node: ast.AST) -> bool:
    return ast.dump(node, include_attributes=False) == ast.dump(
        ast.parse('simulation_app.close()').body[0], include_attributes=False)


class RemoveOnlyTeardown(ast.NodeTransformer):
    """Restore the old final app close for an exact whole-source AST comparison."""

    def __init__(self) -> None:
        self.removed = 0

    def visit_Try(self, node: ast.Try) -> Any:
        if (len(node.body)==1 and is_clear(node.body[0]) and not node.handlers
                and not node.orelse and len(node.finalbody)==1 and is_close(node.finalbody[0])):
            self.removed += 1
            return node.finalbody[0]
        return self.generic_visit(node)


def shutdown_contract(path: Path, baseline: str) -> ast.Try:
    """Return the actual cleanup AST after checking no other semantic change."""
    original = subprocess.check_output(['git','-C',str(REPO),'show',f'{baseline}:scripts/{path.name}'],text=True)
    actual = ast.parse(path.read_text())
    candidates = [node for node in ast.walk(actual) if isinstance(node,ast.Try)
                  and len(node.body)==1 and is_clear(node.body[0])
                  and not node.handlers and not node.orelse
                  and len(node.finalbody)==1 and is_close(node.finalbody[0])]
    assert len(candidates)==1, f'One final cleanup wrapper required: {path}'
    cleanup = deepcopy(candidates[0])
    transformer = RemoveOnlyTeardown()
    restored = transformer.visit(deepcopy(actual))
    assert transformer.removed==1
    assert ast.dump(restored,include_attributes=False)==ast.dump(ast.parse(original),include_attributes=False), path
    return cleanup


def official_clear_class(source: Path, events: list[str]) -> type:
    """Compile the actual installed API method, preserving its class/super calls."""
    module = ast.parse(source.read_text())
    context = next(node for node in module.body if isinstance(node,ast.ClassDef) and node.name=='SimulationContext')
    method = next(node for node in context.body if isinstance(node,ast.FunctionDef) and node.name=='clear_instance')
    class_node = ast.ClassDef(name='SimulationContext',bases=[ast.Name(id='Base',ctx=ast.Load())],
        keywords=[],body=[deepcopy(method)],decorator_list=[])
    namespace: dict[str,Any] = {'events':events}
    exec('''class Base:
    _instance = None
    @classmethod
    def clear_instance(cls):
        events.append('clear_singleton')
        cls._instance = None
''',namespace)
    code = ast.fix_missing_locations(ast.Module(body=[class_node],type_ignores=[]))
    exec(compile(code,str(source),'exec'),namespace)
    return namespace['SimulationContext']


def execute_shutdown(cleanup: ast.Try, context: type, events: list[str]) -> None:
    """Execute actual converter shutdown statements with observable close."""
    def close() -> None:
        events.append('original_app_close')
    namespace = dict(SimulationContext=context,simulation_app=SimpleNamespace(close=close))
    code = ast.fix_missing_locations(ast.Module(body=[deepcopy(cleanup)],type_ignores=[]))
    exec(compile(code,'<actual_converter_finally>','exec'),namespace)


def self_test(installed_source: Path = INSTALLED_SOURCE, baseline: str = BASELINE) -> dict[str,Any]:
    """Successful/no-instance/no-handle/error cases; original app close always once."""
    assert installed_source.is_file()
    results: dict[str,Any] = {}
    for name in ('csv_to_npz.py','batch_csv_to_npz.py'):
        path = REPO/'scripts'/name
        cleanup = shutdown_contract(path,baseline)
        events: list[str] = ['original_NPZ_write_complete']
        context = official_clear_class(installed_source,events)
        class Handle:
            def __init__(self, fail: bool = False) -> None:
                self.fail = fail
            def unsubscribe(self) -> None:
                events.append('unsubscribe_STOP')
                if self.fail:
                    raise RuntimeError('synthetic_unsubscribe_failure')
        context._instance = SimpleNamespace(_app_control_on_stop_handle=Handle())
        execute_shutdown(cleanup,context,events)
        assert events==['original_NPZ_write_complete','unsubscribe_STOP','clear_singleton','original_app_close']
        assert context._instance is None
        successful_order = list(events)
        for instance in (None, SimpleNamespace(_app_control_on_stop_handle=None)):
            events.clear()
            context._instance = instance
            execute_shutdown(cleanup,context,events)
            assert events==['clear_singleton','original_app_close']
            assert context._instance is None
        events.clear()
        context._instance = SimpleNamespace(_app_control_on_stop_handle=Handle(fail=True))
        try:
            execute_shutdown(cleanup,context,events)
        except RuntimeError as error:
            assert str(error)=='synthetic_unsubscribe_failure'
        else:
            raise AssertionError('Cleanup failure must remain visible')
        assert events==['unsubscribe_STOP','original_app_close']
        results[name] = dict(status='PASS',whole_source_AST_after_removing_cleanup_equals_baseline=True,
            clear_before_close_order=successful_order,no_instance=True,no_STOP_handle=True,
            original_close_once_on_cleanup_error=True,source_sha256=digest(path))
    report = dict(status='PASS',baseline_commit=baseline,installed_API_source=str(installed_source),
        installed_API_source_sha256=digest(installed_source),converters=results,
        physics_steps=0,native='NOT_RUN',active_process_stack='NOT_CAPTURED')
    print(json.dumps(report,sort_keys=True))
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed-source',type=Path,default=INSTALLED_SOURCE)
    parser.add_argument('--baseline',default=BASELINE)
    args=parser.parse_args()
    self_test(args.installed_source,args.baseline)


if __name__=='__main__':
    main()
