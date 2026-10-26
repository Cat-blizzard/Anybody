# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.utils.finite_checks import require_all_finite
from rsl_rl.utils.distributed_contract import collective_require_finite


class ActorWithSkip(nn.Module):
    """Wrapper module for actor with skip connection."""

    def __init__(self, actor_layer1, actor_activation, actor_remaining, ref_vel_dim):
        super().__init__()
        self.actor_layer1 = actor_layer1
        self.actor_activation = actor_activation
        self.actor_remaining = actor_remaining
        self.ref_vel_dim = ref_vel_dim

    def forward(self, observations):
        """Forward pass with ref_vel skip connection."""
        # Split observations into policy_obs and ref_vel
        policy_obs = observations[:, :-self.ref_vel_dim]
        ref_vel = observations[:, -self.ref_vel_dim:]

        # Pass policy_obs through first layer
        layer1_out = self.actor_activation(self.actor_layer1(policy_obs))

        # Concatenate layer1_out with ref_vel
        layer2_input = torch.cat([layer1_out, ref_vel], dim=-1)

        # Pass through remaining layers
        output = self.actor_remaining(layer2_input)

        return output


class ActorCritic(nn.Module):
    is_recurrent = False
    is_encoding = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        ref_vel_skip_first_layer: bool = False,
        ref_vel_dim: int = 3,
        strict_finite_checks: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.strict_finite_checks = strict_finite_checks
        activation = resolve_nn_activation(activation)

        # Store ref_vel skip connection parameters
        self.ref_vel_skip_first_layer = ref_vel_skip_first_layer
        self.ref_vel_dim = ref_vel_dim

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        if ref_vel_skip_first_layer:
            # Build actor with ref_vel skip connection
            self._build_actor_with_skip(
                mlp_input_dim_a, actor_hidden_dims, num_actions, activation
            )
        else:
            # Build standard actor (backward compatible)
            actor_layers: list[nn.Module] = []
            prev_dim = mlp_input_dim_a
            if actor_hidden_dims:
                actor_layers.append(nn.Linear(prev_dim, actor_hidden_dims[0]))
                actor_layers.append(activation)
                for layer_index in range(len(actor_hidden_dims)):
                    in_dim = actor_hidden_dims[layer_index]
                    if layer_index == len(actor_hidden_dims) - 1:
                        actor_layers.append(nn.Linear(in_dim, num_actions))
                    else:
                        out_dim = actor_hidden_dims[layer_index + 1]
                        actor_layers.append(nn.Linear(in_dim, out_dim))
                        actor_layers.append(activation)
            else:
                actor_layers.append(nn.Linear(prev_dim, num_actions))
            self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers: list[nn.Module] = []
        prev_dim_c = mlp_input_dim_c
        if critic_hidden_dims:
            critic_layers.append(nn.Linear(prev_dim_c, critic_hidden_dims[0]))
            critic_layers.append(activation)
            for layer_index in range(len(critic_hidden_dims)):
                in_dim = critic_hidden_dims[layer_index]
                if layer_index == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(in_dim, 1))
                else:
                    out_dim = critic_hidden_dims[layer_index + 1]
                    critic_layers.append(nn.Linear(in_dim, out_dim))
                    critic_layers.append(activation)
        else:
            critic_layers.append(nn.Linear(prev_dim_c, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Print network architecture
        if self.ref_vel_skip_first_layer:
            print(f"Actor: ref_vel skip connection enabled")
        else:
            print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        self._warned_nonfinite = False
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def _ensure_finite(self, tensor: torch.Tensor, label: str) -> torch.Tensor:
        """Replace NaN/Inf to keep policy distribution valid during training."""
        if torch.isfinite(tensor).all():
            return tensor
        if not self._warned_nonfinite:
            bad = ~torch.isfinite(tensor)
            bad_count = int(bad.sum().item())
            finite_vals = torch.abs(tensor[torch.isfinite(tensor)])
            max_abs = float(finite_vals.max().item()) if finite_vals.numel() > 0 else 0.0
            warnings.warn(
                f"ActorCritic: non-finite values in {label}; count={bad_count}, max_abs={max_abs}. "
                "Replacing NaN/Inf with finite values.",
                stacklevel=2,
            )
            self._warned_nonfinite = True
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_actor_with_skip(self, num_actor_obs, actor_hidden_dims, num_actions, activation):
        """
        Build actor network with ref_vel skip connection.

        Architecture:
            policy_obs (num_actor_obs - ref_vel_dim) → layer1 → layer1_out
            ref_vel (ref_vel_dim) ─────────────────────────────┘
                                                                ↓
            [layer1_out, ref_vel] → layer2 → ... → output
        """
        if not actor_hidden_dims or len(actor_hidden_dims) < 2:
            raise ValueError("ref_vel_skip_first_layer requires at least 2 hidden layers")

        # First layer: only processes policy_obs (without ref_vel)
        policy_obs_dim = num_actor_obs - self.ref_vel_dim
        actor_layer1 = nn.Linear(policy_obs_dim, actor_hidden_dims[0])
        actor_activation = activation

        # Second layer: takes [layer1_out, ref_vel] as input
        layer2_input_dim = actor_hidden_dims[0] + self.ref_vel_dim

        # Build remaining layers
        remaining_layers = []
        remaining_layers.append(nn.Linear(layer2_input_dim, actor_hidden_dims[1]))
        remaining_layers.append(activation)

        for layer_index in range(1, len(actor_hidden_dims)):
            in_dim = actor_hidden_dims[layer_index]
            if layer_index == len(actor_hidden_dims) - 1:
                remaining_layers.append(nn.Linear(in_dim, num_actions))
            else:
                out_dim = actor_hidden_dims[layer_index + 1]
                remaining_layers.append(nn.Linear(in_dim, out_dim))
                remaining_layers.append(activation)

        actor_remaining = nn.Sequential(*remaining_layers)

        # Create wrapper module that combines all components
        self.actor = ActorWithSkip(actor_layer1, actor_activation, actor_remaining, self.ref_vel_dim)

        # Store references for compatibility
        self.actor_layer1 = actor_layer1
        self.actor_activation = actor_activation
        self.actor_remaining = actor_remaining

        print(f"Actor with ref_vel skip connection:")
        print(f"  Layer1: {policy_obs_dim} -> {actor_hidden_dims[0]} (policy_obs only)")
        print(f"  Layer2+: {layer2_input_dim} -> ... -> {num_actions} (layer1_out + ref_vel)")

    def _actor_forward_with_skip(self, observations):
        """
        Forward pass with ref_vel skip connection.

        Args:
            observations: [batch, num_actor_obs] where last ref_vel_dim are ref_vel

        Returns:
            actions: [batch, num_actions]
        """
        # Split observations into policy_obs and ref_vel
        policy_obs = observations[:, :-self.ref_vel_dim]
        ref_vel = observations[:, -self.ref_vel_dim:]

        # Pass policy_obs through first layer
        layer1_out = self.actor_activation(self.actor_layer1(policy_obs))

        # Concatenate layer1_out with ref_vel
        layer2_input = torch.cat([layer1_out, ref_vel], dim=-1)

        # Pass through remaining layers
        output = self.actor_remaining(layer2_input)

        return output

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        if self.strict_finite_checks:
            collective_require_finite(observations, "ActorCritic update_distribution observations")
        else:
            require_all_finite(observations, "ActorCritic update_distribution observations")

        # compute mean (actor.forward handles both standard and skip connection modes)
        mean = self.actor(observations)
        # No unused goal-compaction debug path: an entirely zero finite goal
        # is a valid input and must not fail via reshape(-1, 0).
        if not self.strict_finite_checks:
            mean = self._ensure_finite(mean, "action mean")
        # compute standard deviation
        if self.noise_std_type == "scalar":
            raw_std = self.std
            # Keep std strictly positive to avoid invalid Normal distributions.
            std = torch.clamp_min(self.std, 1e-6).expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
            raw_std = self.log_std
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        if self.strict_finite_checks:
            collective_require_finite((mean, raw_std, std), "ActorCritic raw action mean/std")
        else:
            std = self._ensure_finite(std, "action std")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        if self.strict_finite_checks:
            collective_require_finite(observations, "ActorCritic act_inference observations")
        else:
            require_all_finite(observations, "ActorCritic act_inference observations")
        # actor.forward handles both standard and skip connection modes
        actions_mean = self.actor(observations)
        if self.strict_finite_checks:
            collective_require_finite(actions_mean, "ActorCritic inference action mean")
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        if self.strict_finite_checks:
            collective_require_finite((critic_observations, value), "ActorCritic critic input/value")
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True


if __name__ == "__main__":
    torch.set_num_threads(1)
    for batch_size in (1, 3):
        actor = ActorCritic(4, 5, 2, actor_hidden_dims=[8], critic_hidden_dims=[8], strict_finite_checks=True)
        actions = actor.act(torch.zeros(batch_size, 4))
        assert actions.shape == (batch_size, 2) and torch.isfinite(actions).all()
        values = actor.evaluate(torch.zeros(batch_size, 5))
        assert values.shape == (batch_size, 1) and torch.isfinite(values).all()
        try:
            actor.act(torch.full((batch_size, 4), float("nan")))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Strict actor accepted non-finite observations")
        print(f"ActorCritic strict CPU: obs=({batch_size},4), actions={tuple(actions.shape)}, values={tuple(values.shape)} PASS")
    assert not torch.cuda.is_initialized()
