# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright (c) 2020 Preferred Networks, Inc.

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        distributed_sync = getattr(self, "_distributed_normalizer_sync", None)
        if distributed_sync is not None:
            if not distributed_sync.before_update(x):
                return
        elif self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        w = torch.isfinite(x).to(dtype=x.dtype, device=x.device)
        sum_w = w.sum(dim=0, keepdim=True)
        valid = sum_w > 0
        x0 = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        mean_x = torch.where(
            valid,
            (x0 * w).sum(dim=0, keepdim=True) / sum_w.clamp(min=1.0),
            self._mean,
        )
        centered = torch.where(torch.isfinite(x), x - mean_x, torch.zeros_like(x))
        var_x = torch.where(
            valid,
            ((centered**2) * w).sum(dim=0, keepdim=True) / sum_w.clamp(min=1.0),
            self._var,
        )
        delta_mean = mean_x - self._mean
        self._mean += torch.where(valid, rate * delta_mean, torch.zeros_like(self._mean))
        self._var += torch.where(
            valid,
            rate * (var_x - self._var + delta_mean * (mean_x - self._mean)),
            torch.zeros_like(self._var),
        )
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize
    the scale of the rewards so that the value function can learn quickly. We did this by dividing
    the rewards by a running estimate of the standard deviation of the sum of discounted rewards.
    """

    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = DiscountedAverage(gamma)

    def forward(self, rew):
        if self.training:
            # update discounected rewards
            avg = self.disc_avg.update(rew)

            # update moments from discounted rewards
            self.emp_norm.update(avg)

        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew


class DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t

    Args:
        gamma (float): Discount factor.
    """

    def __init__(self, gamma):
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg


def _sanity_check() -> None:
    """Check finite singleton/boundary batches without a process group."""
    norm = EmpiricalNormalization((2,), until=2)
    initial = torch.tensor([[1.0, -1.0]])
    result = norm(initial)
    assert result.shape == (1, 2) and torch.isfinite(result).all()
    norm(torch.tensor([[3.0, 2.0], [5.0, 3.0]]))
    assert int(norm.count) == 3  # retain the upstream whole-batch overshoot
    snapshot = {name: value.clone() for name, value in norm.state_dict().items()}
    norm(torch.zeros((1, 2)))
    assert all(torch.equal(value, norm.state_dict()[name]) for name, value in snapshot.items())
    assert torch.allclose(norm.inverse(norm(initial)), initial)
    print("normalizer sanity passed: singleton shape", tuple(result.shape), "until-overshoot count", int(norm.count))


if __name__ == "__main__":
    _sanity_check()
