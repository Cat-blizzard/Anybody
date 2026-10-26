"""CPU-only synchronization regression; run this file without Isaac/CUDA.

Tests two real Gloo ranks, unequal batches, repeated rollouts, frozen buffers,
per-rank until overshoot, pending-state resume and rank0 boundary checkpoints.
The single-rank path is compared bitwise with the frozen pre-change source.
"""

from __future__ import annotations

import copy
import ast
import importlib.util
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / "AnyBody_teacher_correctness_fix_20261015"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NORMALIZER = _load(ROOT / "source/rsl_rl/rsl_rl/modules/normalizer.py", "local_normalizer")
SYNC = _load(ROOT / "source/rsl_rl/rsl_rl/utils/distributed_normalizer.py", "local_sync")


def _equal_state(left: torch.nn.Module, right: torch.nn.Module) -> None:
    assert left.state_dict().keys() == right.state_dict().keys()
    for name, value in left.state_dict().items():
        assert torch.equal(value, right.state_dict()[name]), name


def _check_reference(normalizer: torch.nn.Module, rows: list[torch.Tensor]) -> None:
    reference = torch.cat(rows).to(torch.float64)
    assert int(normalizer.count) == reference.shape[0]
    assert torch.allclose(normalizer._mean.double(), reference.mean(0, keepdim=True), atol=1e-6, rtol=1e-6)
    assert torch.allclose(normalizer._var.double(), reference.var(0, unbiased=False, keepdim=True), atol=1e-6, rtol=1e-6)
    assert torch.equal(normalizer._std, torch.sqrt(normalizer._var))


def _worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        norm = NORMALIZER.EmpiricalNormalization((2,), until=None)
        helper = SYNC.DistributedNormalizerSync(norm)
        reference: list[torch.Tensor] = []
        for round_index in range(3):
            local_rows = []
            for batch_index in range(2):
                size = rank + batch_index + 1
                x = torch.arange(size * 2, dtype=torch.float32).reshape(size, 2)
                x = x + round_index * 7.0 + rank * 3.0
                norm(x)
                local_rows.append(x)
            gathered: list[Any] = [None, None]
            dist.all_gather_object(gathered, local_rows)
            reference.extend([x for part in gathered for x in part])
            result = helper.synchronize()
            _check_reference(norm, reference)
            assert result["global_count"] == sum(x.shape[0] for x in reference)
            snapshot = copy.deepcopy(norm.state_dict())
            helper.synchronize()  # no pending rows must not duplicate shared history
            for key, value in snapshot.items():
                assert torch.equal(value, norm.state_dict()[key])
            before = int(norm.count)
            refreshed = SYNC.normalize_without_update(norm, local_rows[-1])
            assert refreshed.shape == local_rows[-1].shape and int(norm.count) == before

        # Rank-local snapshots include pending moments, not only public buffers.
        x = torch.tensor([[100.0 + rank, -20.0 - rank]])
        norm(x)
        pending_state = helper.state_dict()
        norm_state = copy.deepcopy(norm.state_dict())
        try:
            helper.state_dict(require_synchronized=True)
        except AssertionError:
            pass
        else:
            raise AssertionError("Unsynchronized rank0 checkpoint was accepted")
        resumed = NORMALIZER.EmpiricalNormalization((2,), until=None)
        resumed.load_state_dict(norm_state)
        resumed_helper = SYNC.DistributedNormalizerSync(resumed, state=pending_state)
        resumed_helper.synchronize()
        gathered_rows: list[Any] = [None, None]
        dist.all_gather_object(gathered_rows, x)
        reference.extend(gathered_rows)
        _check_reference(resumed, reference)

        # A synchronized checkpoint saved by rank0 retains every rank's counter.
        payload: list[Any] = [None]
        if rank == 0:
            payload[0] = {
                "normalizer": copy.deepcopy(resumed.state_dict()),
                "distributed": resumed_helper.state_dict(require_synchronized=True),
            }
        dist.broadcast_object_list(payload, src=0)
        boundary = NORMALIZER.EmpiricalNormalization((2,), until=None)
        boundary.load_state_dict(payload[0]["normalizer"])
        boundary_helper = SYNC.DistributedNormalizerSync(boundary, state=payload[0]["distributed"])
        assert boundary_helper.local_seen == payload[0]["distributed"]["local_counts"][rank]
        boundary_helper.synchronize()
        _equal_state(resumed, boundary)

        limited = NORMALIZER.EmpiricalNormalization((2,), until=5)
        limited_helper = SYNC.DistributedNormalizerSync(limited)
        accepted_rows: list[torch.Tensor] = []
        for round_index in range(3):
            x = torch.full((rank + 3, 2), rank + round_index, dtype=torch.float32)
            before = limited_helper.local_seen
            limited(x)
            local = [x] if limited_helper.local_seen > before else []
            gathered = [None, None]
            dist.all_gather_object(gathered, local)
            accepted_rows.extend([x for part in gathered for x in part])
            info = limited_helper.synchronize()
            _check_reference(limited, accepted_rows)
        assert info["local_accepted_counts"] == [6, 8]
        assert info["global_count"] == 14
        assert info["new_global_rows"] == 0

        # Large DC offsets must not erase small variances by cancellation.
        offset = NORMALIZER.EmpiricalNormalization((2,), until=None)
        offset_helper = SYNC.DistributedNormalizerSync(offset)
        offset_rows: list[torch.Tensor] = []
        for round_index in range(2):
            x = torch.tensor([[1.0e6, -1.0e6], [1.0e6 + 0.0625, -1.0e6 + 0.125]])
            x.add_(rank * 0.0625 + round_index * 0.125)
            offset(x)
            gathered_offset: list[Any] = [None, None]
            dist.all_gather_object(gathered_offset, x)
            offset_rows.extend(gathered_offset)
            offset_helper.synchronize()
            _check_reference(offset, offset_rows)
        assert bool((offset._var > 0).all())

        # eval and until-frozen normalizers remain byte-for-byte unchanged.
        frozen = NORMALIZER.EmpiricalNormalization((2,), until=None).eval()
        frozen._mean.fill_(2.0)
        frozen._var.fill_(3.0)
        frozen._std.copy_(torch.sqrt(frozen._var))
        frozen_state = copy.deepcopy(frozen.state_dict())
        frozen_helper = SYNC.DistributedNormalizerSync(frozen)
        frozen(torch.full((1, 2), 99.0 + rank))
        frozen_helper.synchronize()
        for key, value in frozen_state.items():
            assert torch.equal(value, frozen.state_dict()[key])
        assert limited_helper.local_seen >= limited.until
        before_state = copy.deepcopy(limited.state_dict())
        limited(torch.full((1, 2), -99.0 - rank))
        limited_helper.synchronize()
        for key, value in before_state.items():
            assert torch.equal(value, limited.state_dict()[key])
        limited(torch.tensor([[float("nan") if rank == 0 else 1.0, 2.0]]))
        try:
            limited_helper.synchronize()
        except AssertionError as error:
            assert "Nonfinite" in str(error)
        else:
            raise AssertionError("Nonfinite input escaped checking after until")
        # A bad input on only one rank is rejected by both ranks at the rollout
        # boundary, without a per-forward collective or a stranded other rank.
        bad = NORMALIZER.EmpiricalNormalization((2,), until=None)
        bad_helper = SYNC.DistributedNormalizerSync(bad)
        bad(torch.tensor([[float("nan") if rank == 0 else 1.0, 2.0]]))
        try:
            bad_helper.synchronize()
        except AssertionError as error:
            assert "Nonfinite" in str(error)
        else:
            raise AssertionError("A bad remote-rank batch was accepted")

        # Execute the real runner and PPO on a synthetic CPU VecEnv. Only its
        # CUDA-specific process-group setup is substituted with this live Gloo
        # group; normalization, rollout cadence, GAE, update and checkpoint code
        # all execute from this worktree.
        sys.path.insert(0, str(ROOT / "source/rsl_rl"))
        fixtures = _load(ROOT / "tests/test_checkpoint_resume.py", "runner_fixture")
        Runner = fixtures.OnPolicyRunner
        previous_configure = Runner._configure_multi_gpu

        def configure_cpu(runner: Any) -> None:
            runner.gpu_world_size = 2
            runner.is_distributed = True
            runner.gpu_global_rank = rank
            runner.gpu_local_rank = rank
            runner.multi_gpu_cfg = {"global_rank": rank, "local_rank": rank, "world_size": 2}

        Runner._configure_multi_gpu = configure_cpu
        try:
            config = fixtures.runner_config()
            config["distributed_normalizer_sync"] = True
            config["normalizer_until_count_scope"] = "per_rank_equivalent"
            config["algorithm"]["strict_finite_checks"] = True

            def create_runner() -> Any:
                env = fixtures.SyntheticVecEnv()
                env._obs.add_(rank * 0.7)
                env._critic.add_(rank * 0.7)
                return fixtures.make_runner(env=env, config=config)

            runner = create_runner()
            runner.learn(3)
            assert int(runner.obs_normalizer.count) == 28
            assert int(runner.privileged_obs_normalizer.count) == 28
            checkpoint = str(Path(rendezvous).parent / "runner_checkpoint.pt")
            if rank == 0:
                runner.save(checkpoint)
            dist.barrier()
            restored = create_runner()
            restored.load(checkpoint)
            restored.learn(1)
            assert int(restored.obs_normalizer.count) == 40
            assert int(restored.privileged_obs_normalizer.count) == 40
            assert restored._distributed_normalizer_helpers["actor"].local_seen == 20
            assert not torch.cuda.is_initialized()
        finally:
            Runner._configure_multi_gpu = previous_configure
        if rank == 0:
            print("Gloo 2-rank passed: weighted reference, 3 rounds, no double-counting, cached observations,")
            print("pending resume, rank0 checkpoint counters, frozen buffers, until overshoot [6,8], exact rank equality")
            print("Real runner/PPO CPU Gloo integration passed: 3 updates + checkpoint + 1 update; counts 28 -> 40")
    finally:
        dist.destroy_process_group()


def main() -> None:
    torch.set_num_threads(1)
    # Read the frozen source as data: do not import it or create its pycache.
    original_path = ORIGINAL / "source/rsl_rl/rsl_rl/modules/normalizer.py"
    tree = ast.parse(original_path.read_text())
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EmpiricalNormalization")
    namespace: dict[str, Any] = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(original_path), "exec"), namespace)
    original = SimpleNamespace(EmpiricalNormalization=namespace["EmpiricalNormalization"])
    left = NORMALIZER.EmpiricalNormalization((2,), until=5)
    right = original.EmpiricalNormalization((2,), until=5)
    batches = [torch.zeros((1, 2)), torch.tensor([[1.0, -2.0], [3.0, 7.0], [2.0, 6.0]])]
    batches.extend([torch.full((4, 2), 3.0), torch.full((1, 2), -9.0)])
    for x in batches:
        assert torch.equal(left(x), right(x))
        _equal_state(left, right)
    assert int(left.count) == 8
    print("Single rank bitwise upstream-equivalent: outputs, mean/var/std/count; batch shapes", [tuple(x.shape) for x in batches])
    sys.path.insert(0, str(ROOT / "source/rsl_rl"))
    fixtures = _load(ROOT / "tests/test_checkpoint_resume.py", "single_runner_fixture")
    snapshots = []
    for enabled in (False, True):
        torch.manual_seed(47)
        config = fixtures.runner_config()
        config["algorithm"]["strict_finite_checks"] = True
        config["distributed_normalizer_sync"] = enabled
        config["normalizer_until_count_scope"] = "per_rank_equivalent"
        runner = fixtures.make_runner(config=config)
        runner.learn(3)
        assert int(runner.obs_normalizer.count) == 14
        assert not runner._distributed_normalizer_helpers
        snapshots.append(fixtures.tensor_snapshot(runner))
    assert snapshots[0] == snapshots[1]
    print("Single-rank real runner 3-update opt-in/off bitwise-equivalent: policy, actor/critic statistics and Adam")
    with tempfile.TemporaryDirectory(prefix="anybody_normalizer_gloo_") as temporary:
        mp.spawn(_worker, args=(str(Path(temporary) / "rendezvous"),), nprocs=2, join=True)
        config = fixtures.runner_config()
        config["algorithm"]["strict_finite_checks"] = True
        config["distributed_normalizer_sync"] = True
        evaluator = fixtures.make_runner(config=config)
        evaluator.load(str(Path(temporary) / "runner_checkpoint.pt"), load_optimizer=False)
        before = fixtures.tensor_snapshot(evaluator)
        evaluator.learn(1, eval_mode=True)
        assert before == fixtures.tensor_snapshot(evaluator)
        print("Single-rank evaluation of DDP checkpoint preserves policy/normalizers/Adam exactly")
    print("All distributed normalization CPU tests passed")


if __name__ == "__main__":
    main()
