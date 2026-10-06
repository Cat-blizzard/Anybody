# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Scalar contracts for checkpoint progress and the PPO optimizer learning rate.

``iter`` is the zero-based LAST completed update (-1 before training), whereas
``next_learning_iteration`` is the next update to execute (0 before training).
Legacy runner checkpoints were saved after an update and have only ``iter``.
These helpers never modify policy, normalizer, optimizer tensors, or RNG state.
PPO has one scalar LR: all main optimizer groups must therefore agree.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any


RUNNER_STATE_VERSION = 1


def _integer(value: Any, name: str, minimum: int) -> int:
    """Validate a counter without truncating floats or accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def checkpoint_progress(
    last_completed: int, next_iteration: int, phase: int | None = None,
) -> dict[str, int]:
    """Return versioned, internally consistent progress for a new checkpoint."""
    last = _integer(last_completed, "iter", -1)
    following = _integer(next_iteration, "next_learning_iteration", 0)
    if following != last + 1:
        raise ValueError(f"Checkpoint progress is inconsistent: iter={last}, next={following}")
    progress = {
        "iter": last,
        "runner_state_version": RUNNER_STATE_VERSION,
        "next_learning_iteration": following,
    }
    if phase is not None:
        progress["learning_iteration_phase"] = _integer(phase, "learning_iteration_phase", 0)
    return progress


def resolve_checkpoint_progress(checkpoint: Mapping[str, Any]) -> tuple[int, int]:
    """Read (last completed, next), supporting legacy post-update checkpoints."""
    if "runner_state_version" not in checkpoint and "next_learning_iteration" not in checkpoint:
        last = _integer(checkpoint["iter"], "legacy iter", 0)
        return last, last + 1
    version = _integer(checkpoint.get("runner_state_version"), "runner_state_version", 1)
    if version != RUNNER_STATE_VERSION:
        raise ValueError(f"Unsupported runner_state_version: {version}")
    progress = checkpoint_progress(checkpoint["iter"], checkpoint.get("next_learning_iteration"))
    return progress["iter"], progress["next_learning_iteration"]


def resolve_checkpoint_phase(checkpoint: Mapping[str, Any]) -> int:
    """Retain evaluation phase independently of optimizer training progress."""
    last, _ = resolve_checkpoint_progress(checkpoint)
    return _integer(checkpoint.get("learning_iteration_phase", max(0, last)), "learning_iteration_phase", 0)


def ppo_optimizer_learning_rate(optimizer_state: Mapping[str, Any]) -> float:
    """Validate the main optimizer's finite, positive, shared scalar LR."""
    groups = optimizer_state.get("param_groups")
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("PPO optimizer checkpoint must contain nonempty param_groups")
    rates: list[float] = []
    for index, group in enumerate(groups):
        rate = group.get("lr") if isinstance(group, Mapping) else None
        if isinstance(rate, bool) or not isinstance(rate, Real):
            raise ValueError(f"PPO optimizer group {index} has invalid scalar LR: {rate!r}")
        numeric = float(rate)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"PPO optimizer group {index} LR must be finite and positive")
        rates.append(numeric)
    if any(rate != rates[0] for rate in rates[1:]):
        raise ValueError("PPO scalar learning_rate cannot represent different optimizer group LRs")
    return rates[0]


def sanity_check() -> None:
    """Exercise boundary counters and a real singleton Adam forward/restore."""
    import copy

    import torch

    for last in (0, 99, 199, 999):
        assert resolve_checkpoint_progress({"iter": last}) == (last, last + 1)
        state = checkpoint_progress(last, last + 1)
        assert resolve_checkpoint_progress(state) == (last, last + 1)
    assert resolve_checkpoint_progress(checkpoint_progress(-1, 0)) == (-1, 0)
    assert resolve_checkpoint_phase(checkpoint_progress(198, 199, phase=199)) == 199
    invalid_progress = (
        {"iter": True}, {"iter": 1.5}, {"iter": -1},
        {"iter": 199, "runner_state_version": 1, "next_learning_iteration": 199},
        {"iter": 199, "next_learning_iteration": 200},
        {"iter": 199, "runner_state_version": 2, "next_learning_iteration": 200},
    )
    for state in invalid_progress:
        try:
            resolve_checkpoint_progress(state)
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError(f"Invalid checkpoint progress accepted: {state}")

    with torch.random.fork_rng(devices=[]):
        policy = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)
        value = policy(torch.zeros(1, 3))
        assert value.shape == (1, 2) and torch.isfinite(value).all()
        value.square().mean().backward()
        optimizer.step()
        state = copy.deepcopy(optimizer.state_dict())
        target = torch.optim.Adam(policy.parameters(), lr=1e-3)
        target.load_state_dict(state)
        assert ppo_optimizer_learning_rate(target.state_dict()) == 1e-5
        for key, tensors in state["state"].items():
            for name, tensor in tensors.items():
                assert torch.equal(tensor, target.state_dict()["state"][key][name])
        print(f"checkpoint progress: legacy/new OK; singleton output Shape={tuple(value.shape)}, LR=1e-5")
    for rates in ([], [0.0], [-1.0], [float("nan")], [float("inf")], [1e-5, 1e-3]):
        try:
            ppo_optimizer_learning_rate({"param_groups": [{"lr": rate} for rate in rates]})
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid PPO group rates accepted: {rates}")
    assert ppo_optimizer_learning_rate({"param_groups": [{"lr": 1e-5}, {"lr": 1e-5}]}) == 1e-5
    print("checkpoint progress and optimizer LR sanity checks passed")


if __name__ == "__main__":
    sanity_check()
