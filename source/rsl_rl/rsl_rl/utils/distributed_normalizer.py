"""Opt-in, once-per-rollout synchronization of online observation statistics.

This is a local implementation, not an implementation supplied by AnyBody/MOSAIC.
The public normalizer count records the true global sample count.  ``until``
retains the upstream *per-rank* accepted-row threshold, including its whole-final-
batch overshoot; each rank therefore keeps a separate accepted-row counter.
Only new, finite rows since the previous synchronization are reduced.  Shared
history is merged once, rather than being multiplied by world_size each rollout.
Single-process runners do not attach this helper and retain their original path.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


class DistributedNormalizerSync:
    """Synchronize one online normalizer with input shape ``(batch, *features)``.

    Sufficient statistics have shape ``(1, *features)`` and float64 precision.
    Normalization remains in the original buffer dtype.  Collectives run only
    in :meth:`synchronize`, never during forward/update or PPO minibatches.
    """

    def __init__(
        self,
        normalizer: torch.nn.Module,
        *,
        until_count_scope: str = "per_rank_equivalent",
        state: dict[str, Any] | None = None,
    ) -> None:
        assert dist.is_initialized() and dist.get_world_size() > 1
        assert until_count_scope == "per_rank_equivalent"
        assert hasattr(normalizer, "_mean") and hasattr(normalizer, "count")
        assert getattr(normalizer, "_distributed_normalizer_sync", None) is None
        self.normalizer = normalizer
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.until_count_scope = until_count_scope
        self.until = None if normalizer.until is None else float(normalizer.until)
        self.base_count = int(normalizer.count.item())
        self.base_mean = normalizer._mean.detach().to(torch.float64).clone()
        self.base_m2 = normalizer._var.detach().to(torch.float64).clone() * self.base_count
        self.pending_count = 0
        self.pending_mean = torch.zeros_like(self.base_mean)
        self.pending_m2 = torch.zeros_like(self.base_mean)
        self.local_seen = self.base_count
        self.local_counts = [self.local_seen] * self.world_size
        self.invalid_rows = 0
        if state is not None:
            self.load_state_dict(state)
        normalizer._distributed_normalizer_sync = self

    @torch.no_grad()
    def before_update(self, x: torch.Tensor) -> bool:
        """Record accepted finite rows; return whether original update may run."""
        assert x.ndim == self.base_mean.ndim and x.shape[1:] == self.base_mean.shape[1:]
        assert x.device == self.base_mean.device and x.shape[0] > 0
        # Frozen modules normally bypass update in forward; direct calls obey
        # the same protection here as well.
        if not self.normalizer.training:
            return False
        rows = int(x.shape[0])
        if not bool(torch.isfinite(x).all()):
            # Defer the shared failure to the next rollout collective: raising
            # on only one rank here could strand other ranks in PPO collectives.
            # The invalid rollout is rejected before any optimizer update.
            self.invalid_rows += rows
            return False
        if self.until is not None and self.local_seen >= self.until:
            return False
        values = x.detach().to(torch.float64)
        self.local_seen += rows
        batch_mean = values.mean(dim=0, keepdim=True)
        batch_m2 = (values - batch_mean).square().sum(dim=0, keepdim=True)
        total = self.pending_count + rows
        delta = batch_mean - self.pending_mean
        self.pending_mean.add_(delta * (rows / total))
        self.pending_m2.add_(batch_m2 + delta.square() * (self.pending_count * rows / total))
        self.pending_count = total
        return True

    @torch.no_grad()
    def synchronize(self) -> dict[str, Any]:
        """All-reduce pending moments, merge history once, and check equality."""
        n = self.normalizer
        device = self.base_mean.device
        control = torch.tensor(
            [self.pending_count, self.local_seen, int(n.training), self.invalid_rows], dtype=torch.int64, device=device
        )
        controls = [torch.empty_like(control) for _ in range(self.world_size)]
        dist.all_gather(controls, control)
        assert len({int(c[2].item()) for c in controls}) == 1, "Rank normalization modes differ"
        invalid_by_rank = [int(c[3].item()) for c in controls]
        assert not any(invalid_by_rank), f"Nonfinite distributed normalization input: rollout rejected; rows={invalid_by_rank}"
        self.local_counts = [int(c[1].item()) for c in controls]
        accepted = sum(int(c[0].item()) for c in controls)
        # Central moments avoid cancellation in E[x²]-E[x]² when observations
        # have a large offset and small variance. Both reductions happen only
        # here at the rollout boundary.
        mean_new = self.pending_mean * self.pending_count
        dist.all_reduce(mean_new, op=dist.ReduceOp.SUM)
        mean_new.div_(max(accepted, 1))
        m2_new = self.pending_m2 + (self.pending_mean - mean_new).square() * self.pending_count
        dist.all_reduce(m2_new, op=dist.ReduceOp.SUM)
        if accepted:
            assert n.training, "Cannot mutate a frozen normalizer with pending updates"
            assert bool(torch.isfinite(m2_new).all() and (m2_new >= 0).all()), "Invalid central moment"
            count = self.base_count + accepted
            delta = mean_new - self.base_mean
            self.base_mean.add_(delta * (accepted / count))
            self.base_m2.add_(m2_new + delta.square() * (self.base_count * accepted / count))
            self.base_count = count
            variance = self.base_m2 / count
            assert bool(torch.isfinite(self.base_mean).all() and torch.isfinite(variance).all())
            assert bool((variance >= 0).all())
            n._mean.copy_(self.base_mean.to(n._mean.dtype))
            n._var.copy_(variance.to(n._var.dtype))
            n._std.copy_(torch.sqrt(n._var))
            n.count.fill_(count)
        else:
            assert int(n.count.item()) == self.base_count
        self.pending_count = 0
        self.pending_mean.zero_()
        self.pending_m2.zero_()
        # All ranks participate even after their own until threshold is reached.
        state = torch.cat((n._mean.flatten(), n._var.flatten(), n._std.flatten())).to(torch.float64)
        states = [torch.empty_like(state) for _ in range(self.world_size)]
        dist.all_gather(states, state)
        assert all(torch.equal(state, other) for other in states), "Normalizer buffers differ across ranks"
        counts = torch.tensor([int(n.count.item())], dtype=torch.int64, device=device)
        all_counts = [torch.empty_like(counts) for _ in range(self.world_size)]
        dist.all_gather(all_counts, counts)
        assert all(torch.equal(counts, other) for other in all_counts), "Normalizer counts differ across ranks"
        return {
            "until_count_scope": self.until_count_scope,
            "until_per_rank": self.until,
            "global_count": self.base_count,
            "local_accepted_counts": list(self.local_counts),
            "new_global_rows": accepted,
        }

    def state_dict(self, *, require_synchronized: bool = False) -> dict[str, Any]:
        """Save exact shared history and optional rank-local pending state.

        A runner's single-file rank0 checkpoint must use synchronized=True.
        A rank-local snapshot can preserve pending moments using the default.
        """
        if require_synchronized:
            assert self.pending_count == 0, "Checkpoint requires rollout synchronization first"
            assert self.invalid_rows == 0, "Cannot save invalid observation statistics"
            assert int(self.normalizer.count.item()) == self.base_count
        return {
            "version": 1,
            "world_size": self.world_size,
            "rank": self.rank,
            "until_count_scope": self.until_count_scope,
            "until": self.until,
            "base_count": self.base_count,
            "base_mean": self.base_mean.clone(),
            "base_m2": self.base_m2.clone(),
            "local_counts": list(self.local_counts),
            "local_seen": self.local_seen,
            "pending_count": self.pending_count,
            "pending_mean": self.pending_mean.clone(),
            "pending_m2": self.pending_m2.clone(),
            "invalid_rows": self.invalid_rows,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore local pending state or a shared synchronized rank0 snapshot."""
        assert state["version"] == 1 and state["world_size"] == self.world_size
        assert state["until_count_scope"] == self.until_count_scope
        assert state["until"] == self.until, "Normalizer until differs from checkpoint"
        pending = int(state["pending_count"])
        assert pending >= 0
        if pending:
            assert state["rank"] == self.rank, "Pending moments belong to a different rank"
        self.base_count = int(state["base_count"])
        self.base_mean.copy_(state["base_mean"])
        self.base_m2.copy_(state["base_m2"])
        self.local_counts = [int(c) for c in state["local_counts"]]
        assert len(self.local_counts) == self.world_size
        self.local_seen = int(state["local_seen"]) if pending else self.local_counts[self.rank]
        self.pending_count = pending
        self.pending_mean.copy_(state["pending_mean"])
        self.pending_m2.copy_(state["pending_m2"])
        self.invalid_rows = int(state["invalid_rows"])
        assert int(self.normalizer.count.item()) == self.base_count + pending


def normalize_without_update(normalizer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Refresh cached next observations without counting the same batch twice."""
    if isinstance(normalizer, torch.nn.Identity):
        assert bool(torch.isfinite(x).all())
        return x
    result = (x - normalizer._mean) / (normalizer._std + normalizer.eps)
    assert result.shape == x.shape and bool(torch.isfinite(result).all())
    return result


def _self_test() -> None:
    """Pure CPU shape/numerical check; multiprocess tests live in tests/."""
    class Normalizer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._mean = torch.tensor([[1.0, -1.0]])
            self._std = torch.tensor([[2.0, 1.0]])
            self.eps = 0.01

    n = Normalizer()
    x = torch.tensor([[1.0, -1.0]])
    result = normalize_without_update(n, x)
    assert result.shape == (1, 2) and torch.equal(result, torch.zeros_like(result))
    print("distributed_normalizer CPU sanity passed; shape", tuple(result.shape))


if __name__ == "__main__":
    _self_test()
