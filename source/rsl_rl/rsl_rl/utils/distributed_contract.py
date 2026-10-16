# SPDX-License-Identifier: BSD-3-Clause
"""Collective failure checks for the opt-in strict distributed teacher run.

Each call consumes identically ordered collectives on every rank. Inputs may
have different batch sizes but must describe the same logical check. A bad
value on any rank makes *every* rank raise before the next optimizer step.
No tensors, RNG state, gradients, or learner parameters are modified.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist


def collective_require_finite(tensors: torch.Tensor | Iterable[torch.Tensor], label: str) -> None:
    """Require finite tensors of arbitrary shapes, voting across the live group."""
    values = [tensors] if isinstance(tensors, torch.Tensor) else list(tensors)
    if not values or not all(isinstance(value, torch.Tensor) for value in values):
        raise ValueError("Finite checks require one or more tensors")
    distributed = dist.is_available() and dist.is_initialized()
    if distributed and dist.get_backend() == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    # Combine booleans on their original device before the single scalar vote.
    local_bad = torch.stack([~torch.isfinite(value).all() for value in values]).any()
    failed = local_bad.to(device=device, dtype=torch.int32)
    if distributed:
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if bool(failed.item()):
        raise RuntimeError(f"Non-finite values in {label}; collective finite check failed before Adam")


def _self_test() -> None:
    """CPU boundary checks; real two-rank checks live in test_distributed_ppo.py."""
    torch.set_num_threads(1)
    rng = torch.get_rng_state().clone()
    for tensor in (torch.zeros(1, 1), torch.empty(0), torch.ones(2, 3)):
        original = tensor.clone()
        collective_require_finite(tensor, "finite boundary")
        assert torch.equal(tensor, original)
    for bad in (float("nan"), float("inf"), -float("inf")):
        try:
            collective_require_finite(torch.tensor([bad]), "synthetic bad tensor")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Non-finite tensor was accepted")
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()
    print("collective_require_finite CPU: shapes (1,1)/(0,)/(2,3), finite/NaN/Inf PASS")


if __name__ == "__main__":
    _self_test()
