"""CPU tests of the actual PPO/ActorCritic classes, without Isaac or CUDA.

Rollouts are (24,8,6) actor observations, (24,8,7) critic observations,
and (24,8,3) actions. Each update is 5 epochs x 4 batches = 20 Adam
steps. Gloo ranks deliberately see different samples and initial weights.
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source/rsl_rl"))
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic import ActorCritic

BASE_COMMIT = "306c5d0a4262ff3df435c00247c3cb0a73c34f7e"


def make_policy(actor_class: type = ActorCritic, noise: str = "scalar") -> ActorCritic:
    with contextlib.redirect_stdout(io.StringIO()):
        policy = actor_class(6, 7, 3, actor_hidden_dims=[12, 12], critic_hidden_dims=[12, 12],
                             noise_std_type=noise)
    policy.register_buffer("broadcast_test_integer", torch.tensor([19], dtype=torch.int64))
    return policy


def make_algorithm(policy: ActorCritic, *, rank: int | None = None,
                   strict: bool = True, algorithm_class: type = PPO) -> PPO:
    kwargs: dict[str, Any] = dict(num_learning_epochs=5, num_mini_batches=4,
                                  learning_rate=1e-3, desired_kl=0.01, schedule="adaptive", device="cpu")
    if algorithm_class is PPO:
        kwargs["strict_finite_checks"] = strict
    if rank is not None:
        kwargs["multi_gpu_cfg"] = {"global_rank": rank, "world_size": 2}
    algorithm = algorithm_class(policy, **kwargs)
    algorithm.init_storage("rl", 8, 24, [6], [7], [3])
    return algorithm


def rollout(algorithm: PPO, seed: int) -> torch.Tensor:
    """Populate a genuine storage/GAE rollout, retaining sampled actions."""
    torch.manual_seed(seed)
    sampled = []
    with torch.no_grad():
        for step in range(24):
            observations = torch.randn(8, 6)
            critic_observations = torch.randn(8, 7)
            actions = algorithm.act(observations, critic_observations)
            assert actions.shape == (8, 3) and torch.isfinite(actions).all()
            sampled.append(actions.clone())
            rewards = 0.1 * torch.randn(8) - actions.square().mean(dim=-1)
            dones = torch.zeros(8)
            if step == 11:
                dones[0] = 1
            algorithm.process_env_step(rewards, dones, {})
        algorithm.compute_returns(torch.randn(8, 7))
    return torch.stack(sampled)


def assert_same_tree(left: Any, right: Any, *, bytewise: bool = False) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape
        if bytewise:
            assert torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                               right.contiguous().reshape(-1).view(torch.uint8))
        else:
            assert torch.equal(left, right), "Learner tensors differ"
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_tree(left[key], right[key], bytewise=bytewise)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same_tree(a, b, bytewise=bytewise)
    else:
        assert left == right


def learner_state(algorithm: PPO) -> dict[str, Any]:
    return copy.deepcopy({"policy": algorithm.policy.state_dict(), "optimizer": algorithm.optimizer.state_dict()})


def assert_rank_states_equal(algorithm: PPO) -> None:
    states: list[Any] = [None, None]
    dist.all_gather_object(states, learner_state(algorithm))
    assert_same_tree(states[0], states[1])
    rates: list[Any] = [None, None]
    dist.all_gather_object(rates, algorithm.learning_rate)
    assert rates[0] == rates[1]


def gloo_worker(rank: int, init_file: str, output_dir: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"file://{init_file}", timeout=datetime.timedelta(seconds=120))
    try:
        torch.manual_seed(100 + rank)
        policy = make_policy()
        policy.broadcast_test_integer.fill_(19 + rank)
        algorithm = make_algorithm(policy, rank=rank)
        parameter_ids = [id(parameter) for parameter in policy.parameters()]
        algorithm.broadcast_parameters()
        assert [id(parameter) for parameter in policy.parameters()] == parameter_ids
        assert policy.broadcast_test_integer.item() == 19
        assert_rank_states_equal(algorithm)
        rates = []
        for update in range(3):
            rollout(algorithm, seed=1000 + update * 10 + rank)
            losses = algorithm.update()
            assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
            assert all(int(state["step"].item()) == (update + 1) * 20
                       for state in algorithm.optimizer.state.values())
            assert_rank_states_equal(algorithm)
            rates.append(algorithm.learning_rate)
        assert any(rate != 1e-3 for rate in rates), "Adaptive KL/LR path did not execute"

        # All ranks must fail on each rank-1-only defect, and Adam must not
        # change parameters, moment buffers, or step counters before failure.
        cases = ("actor_input", "raw_mean", "raw_std", "loss", "gradient")
        for case in cases:
            torch.manual_seed(200 + rank)
            candidate = make_algorithm(make_policy(), rank=rank)
            candidate.broadcast_parameters()
            rollout(candidate, seed=2000 + rank)
            candidate.update()
            assert all(int(state["step"].item()) == 20 for state in candidate.optimizer.state.values())
            assert_rank_states_equal(candidate)
            rollout(candidate, seed=3000 + rank)
            hooks = []
            original_coefficient = candidate.value_loss_coef
            if case == "raw_mean" and rank == 1:
                hooks.append(candidate.policy.actor.register_forward_hook(
                    lambda module, inputs, output: output * float("nan")))
            elif case == "raw_std" and rank == 1:
                with torch.no_grad():
                    candidate.policy.std[0] = float("inf")
            elif case == "loss" and rank == 1:
                candidate.value_loss_coef = float("inf")
            elif case == "gradient" and rank == 1:
                hooks.append(next(candidate.policy.actor.parameters()).register_hook(
                    lambda gradient: torch.full_like(gradient, float("nan"))))
            before = learner_state(candidate)
            raised = False
            try:
                if case == "actor_input":
                    obs = torch.zeros(8, 6)
                    if rank == 1:
                        obs[0, 0] = float("nan")
                    candidate.act(obs, torch.zeros(8, 7))
                else:
                    candidate.update()
            except RuntimeError as error:
                assert "collective finite check failed before Adam" in str(error), str(error)
                raised = True
            finally:
                for hook in hooks:
                    hook.remove()
                candidate.value_loss_coef = original_coefficient
            all_raised: list[Any] = [None, None]
            dist.all_gather_object(all_raised, raised)
            assert all(all_raised), f"Only one rank rejected {case}"
            assert_same_tree(before, learner_state(candidate), bytewise=True)
        assert not torch.cuda.is_initialized()
        Path(output_dir, f"rank_{rank}.txt").write_text(
            f"PASS: shared tensor broadcast; 3x20 Adam; rank-exact model/Adam/LR; "
            f"five collective fail-before-Adam cases; LR={rates}\n")
    finally:
        dist.destroy_process_group()


def load_baseline_class(relative_path: str, class_name: str, directory: Path) -> type:
    source = subprocess.run(["git", "show", f"{BASE_COMMIT}:{relative_path}"], cwd=REPO,
                             check=True, text=True, capture_output=True).stdout
    source_path = directory / f"baseline_{class_name}.py"
    source_path.write_text(source)
    specification = importlib.util.spec_from_file_location(f"baseline_{class_name}", source_path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return getattr(module, class_name)


class DistributedPPOTests(unittest.TestCase):
    def test_single_rank_exact_finite_parity_with_frozen_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline_actor = load_baseline_class("source/rsl_rl/rsl_rl/modules/actor_critic.py",
                                                 "ActorCritic", Path(directory))
            baseline_ppo = load_baseline_class("source/rsl_rl/rsl_rl/algorithms/ppo.py", "PPO", Path(directory))
            for noise in ("scalar", "log"):
                torch.manual_seed(41)
                old = make_algorithm(make_policy(baseline_actor, noise), strict=False, algorithm_class=baseline_ppo)
                new = make_algorithm(make_policy(noise=noise))
                new.policy.load_state_dict(old.policy.state_dict())
                for update in range(2):
                    old_actions = rollout(old, 4000 + update)
                    old_losses = old.update()
                    old_rng = torch.get_rng_state().clone()
                    new_actions = rollout(new, 4000 + update)
                    new_losses = new.update()
                    assert torch.equal(old_actions, new_actions)
                    assert old_losses == new_losses
                    assert torch.equal(old_rng, torch.get_rng_state())
                    assert old.learning_rate == new.learning_rate
                    assert_same_tree(learner_state(old), learner_state(new))
                print(f"Single-rank {noise}: 2x20 Adam, exact actions/loss/model/Adam/LR/RNG PASS")
        assert not torch.cuda.is_initialized()

    def test_two_rank_gloo_actual_ppo_and_collective_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(gloo_worker, args=(str(Path(directory) / "process_group"), directory), nprocs=2, join=True)
            for rank in range(2):
                text = Path(directory, f"rank_{rank}.txt").read_text()
                assert text.startswith("PASS:")
                print(f"Gloo rank {rank}: {text.strip()}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
