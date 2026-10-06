#!/usr/bin/env python3
"""Independent CPU tests of the real runner, PPO, and versioned checkpoint API.

Constraints: import only this worktree's rsl_rl source, never an old runtime
adapter; no Isaac simulator, GPU, training-data change, or production mutation.
Synthetic VecEnv observations/actions are (B,D)/(B,29), with B=1 or 2.
The full runner constructor and learn loop execute actual PPO/Adam updates.
For the singleton numerical LR witness only, return normalization is disabled
because unbiased sample std is undefined for one return; PPO.update is intact.
Real checkpoint tests infer MLP dimensions from exact stored layer shapes.
Hash assertions cover every policy, normalizer and nested optimizer tensor,
including shape/dtype, as well as optimizer scalars and original file bytes.
Fresh progress is last=-1,next=0; legacy iter denotes the last complete update.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import re
import sys
import traceback
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'source/rsl_rl'
FORENSICS = Path('/data/jxc/projects/bones_curated/training/runs/stage0_gmt/surrogate_guard_forensics_20260910')
DEFAULT_OUTPUT = FORENSICS / 'resume_state_fix_20261006'
DEFAULT_CHECKPOINTS = (
    FORENSICS / 'data_factor_pilot_20261006/train_full_s20260904/logs/rsl_rl/g1_flat_mosaic_hybrid/2026-10-06_02-13-25_data_factor_full_s20260904/model_199.pt',
    FORENSICS / 'dynamic_followup_20261006/learning_curve/full_s20260904/logs/rsl_rl/g1_flat_mosaic_hybrid/2026-10-06_03-32-08_curve_full_s20260904/model_999.pt',
    FORENSICS.parent / 'logs/rsl_rl/g1_flat_mosaic_hybrid/2026-09-09_01-31-31_bones_curated_gmt_stage0/model_70000.pt',
)

assert 'smoke_runtime' not in sys.modules and 'curve_runtime' not in sys.modules, 'An old observer adapter was imported'
sys.path.insert(0, str(SOURCE))
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils.checkpoint_resume import checkpoint_progress, ppo_optimizer_learning_rate, resolve_checkpoint_phase, resolve_checkpoint_progress

assert Path(inspect.getfile(OnPolicyRunner)).resolve().is_relative_to(SOURCE.resolve())
assert not hasattr(OnPolicyRunner.load, '_curve_resume_adapter')


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(value: Any) -> str:
    """Device-independent hash with strict finite and shape/dtype contracts."""
    digest = hashlib.sha256()

    def visit(node: Any) -> None:
        if isinstance(node, torch.Tensor):
            assert node.device.type == 'cpu' and torch.isfinite(node).all()
            tensor = node.detach().contiguous()
            digest.update(json.dumps(['tensor', str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(node, Mapping):
            digest.update(b'mapping')
            for key in sorted(node, key=lambda item: (type(item).__name__, str(item))):
                visit(key)
                visit(node[key])
        elif isinstance(node, (list, tuple)):
            digest.update(type(node).__name__.encode())
            for item in node:
                visit(item)
        else:
            assert node is None or isinstance(node, (str, bool, int, float)), type(node)
            if isinstance(node, float):
                assert math.isfinite(node)
            digest.update(json.dumps([type(node).__name__, node], allow_nan=False).encode())

    visit(value)
    return digest.hexdigest()


class SyntheticVecEnv(VecEnv):
    """Finite deterministic API fixture, not a robot dynamics experiment."""

    def __init__(self, num_envs: int = 2, actor_dim: int = 3, critic_dim: int = 2) -> None:
        assert num_envs > 0 and actor_dim > 0 and critic_dim > 0
        self.num_envs, self.num_actions = num_envs, 29
        self.actor_dim, self.critic_dim = actor_dim, critic_dim
        self.device = torch.device('cpu')
        self.unwrapped = self
        self.max_episode_length = 8
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.reset_terminated = torch.zeros(num_envs, dtype=torch.bool)
        self.common_step_counter = 0
        self.step_dt = 0.02
        self.cfg: dict[str, Any] = {'synthetic_CPU_fixture': True}
        # The existing actor's debug goal_compact reshape rejects an entirely
        # zero normalized goal. Distinct finite fixture rows exercise the real
        # runner without bypassing or changing that unrelated actor code.
        env_offset = torch.arange(num_envs, dtype=torch.float32).reshape(num_envs, 1) * 0.3
        self._obs = torch.linspace(-0.25, 0.25, actor_dim).reshape(1, actor_dim).repeat(num_envs, 1) + env_offset
        self._critic = torch.linspace(-0.2, 0.2, critic_dim).reshape(1, critic_dim).repeat(num_envs, 1) + env_offset

    def get_observations(self) -> tuple[torch.Tensor, dict[str, Any]]:
        assert self._obs.shape == (self.num_envs, self.actor_dim)
        return self._obs.clone(), {'observations': {'policy': self._obs.clone(), 'critic': self._critic.clone()}}

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        self.episode_length_buf.zero_()
        self._obs.zero_()
        self._critic.zero_()
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        assert actions.shape == (self.num_envs, 29) and actions.device.type == 'cpu' and torch.isfinite(actions).all()
        self.common_step_counter += 1
        self.episode_length_buf.add_(1)
        shift = actions.mean(dim=1, keepdim=True).tanh() * 0.01
        self._obs.add_(shift)
        self._critic.add_(shift)
        reward = 1.0 - actions.square().mean(dim=1) * 0.01
        done = self.episode_length_buf >= self.max_episode_length
        self.episode_length_buf[done] = 0
        obs, extras = self.get_observations()
        assert reward.shape == done.shape == (self.num_envs,) and torch.isfinite(reward).all()
        return obs, reward, done, extras


def runner_config(actor_hidden: list[int] | None = None, critic_hidden: list[int] | None = None,
                  schedule: str = 'adaptive', steps: int = 2) -> dict[str, Any]:
    assert schedule in ('adaptive', 'fixed') and steps > 0
    return {'num_steps_per_env': steps, 'save_interval': 1, 'empirical_normalization': True,
            'logger': 'tensorboard', 'reset_noise_std_on_resume': False,
            'policy': {'class_name': 'ActorCritic', 'actor_hidden_dims': actor_hidden or [8],
                       'critic_hidden_dims': critic_hidden or [8], 'activation': 'elu',
                       'init_noise_std': 1.0, 'noise_std_type': 'scalar'},
            'algorithm': {'class_name': 'PPO', 'num_learning_epochs': 1, 'num_mini_batches': 1,
                          'learning_rate': 1e-3, 'schedule': schedule, 'desired_kl': 0.01,
                          'gamma': 0.99, 'lam': 0.95, 'clip_param': 0.2,
                          'entropy_coef': 0.005, 'max_grad_norm': 1.0}}


def make_runner(log_dir: Path | None = None, env: SyntheticVecEnv | None = None,
                config: dict[str, Any] | None = None) -> OnPolicyRunner:
    runner = OnPolicyRunner(env or SyntheticVecEnv(), copy.deepcopy(config or runner_config()),
                            log_dir=str(log_dir) if log_dir is not None else None, device='cpu')
    runner.git_status_repos = []
    if log_dir is None:
        runner.disable_logs = True
        runner.logger_type = 'tensorboard'
    assert not torch.cuda.is_initialized()
    return runner


def progress(runner: OnPolicyRunner) -> tuple[int, int, int]:
    assert runner.next_learning_iteration == runner._next_learning_iteration
    return (runner.current_learning_iteration, runner._last_completed_learning_iteration,
            runner.next_learning_iteration)


def tensor_snapshot(runner: OnPolicyRunner) -> dict[str, str]:
    return {'policy': state_hash(runner.alg.policy.state_dict()),
            'actor_norm': state_hash(runner.obs_normalizer.state_dict()),
            'critic_norm': state_hash(runner.privileged_obs_normalizer.state_dict()),
            'Adam': state_hash(runner.alg.optimizer.state_dict())}


def expected_snapshot(checkpoint: Mapping[str, Any]) -> dict[str, str]:
    return {'policy': state_hash(checkpoint['model_state_dict']),
            'actor_norm': state_hash(checkpoint['obs_norm_state_dict']),
            'critic_norm': state_hash(checkpoint['privileged_obs_norm_state_dict']),
            'Adam': state_hash(checkpoint['optimizer_state_dict'])}


def cpu_load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location='cpu', weights_only=False)
    assert isinstance(value, dict)
    state_hash(value)
    return value


def expect_error(call: Callable[[], Any], exception_types: tuple[type[Exception], ...]) -> str:
    try:
        call()
    except exception_types as error:
        return f'{type(error).__name__}: {error}'
    raise AssertionError('Invalid resume was accepted')


def record(results: list[dict[str, Any]], name: str, details: Mapping[str, Any]) -> None:
    results.append({'test': name, 'passed': True, **dict(details)})
    print(f'PASS: {name}')


def install_update_trace(runner: OnPolicyRunner, iterations: list[int]) -> None:
    original = runner.alg.update

    def update() -> dict[str, Any]:
        iterations.append(int(runner.alg.current_learning_iteration))
        result = original()
        assert all(math.isfinite(float(value)) for value in result.values())
        state_hash(runner.alg.policy.state_dict())
        state_hash(runner.alg.optimizer.state_dict())
        return result

    runner.alg.update = update


def actual_singleton_update(runner: OnPolicyRunner) -> dict[str, Any]:
    """Run the original one-minibatch PPO update and observe LR at Adam.step."""
    alg = runner.alg
    alg.init_storage('rl', 1, 1, [runner.env.actor_dim], [runner.env.critic_dim], [29])
    actor = torch.zeros((1, runner.env.actor_dim))
    critic = torch.zeros((1, runner.env.critic_dim))
    runner.obs_normalizer.eval()
    runner.privileged_obs_normalizer.eval()
    with torch.no_grad():
        actor = runner.obs_normalizer(actor)
        critic = runner.privileged_obs_normalizer(critic)
        actions = alg.act(actor, critic)
        assert actions.shape == (1, 29) and torch.isfinite(actions).all()
        alg.process_env_step(torch.ones(1), torch.zeros(1), {})
        alg.storage.compute_returns(alg.policy.evaluate(critic).detach(), alg.gamma, alg.lam,
                                    normalize_advantage=False)
    before_lr = float(alg.optimizer.param_groups[0]['lr'])
    before_steps = sorted(set(int(item['step']) for item in alg.optimizer.state.values()))
    observed: list[list[float]] = []
    original_step = alg.optimizer.step

    def step(*args: Any, **kwargs: Any) -> Any:
        observed.append([float(group['lr']) for group in alg.optimizer.param_groups])
        return original_step(*args, **kwargs)

    alg.optimizer.step = step
    try:
        losses = alg.update()
    finally:
        alg.optimizer.step = original_step
    expected_lr = min(1e-2, before_lr * 1.5) if alg.schedule == 'adaptive' else before_lr
    assert len(observed) == 1 and all(math.isclose(lr, expected_lr, rel_tol=1e-12) for lr in observed[0])
    assert all(math.isfinite(float(value)) for value in losses.values())
    after_steps = sorted(set(int(item['step']) for item in alg.optimizer.state.values()))
    expected_steps = [before_steps[0] + 1] if before_steps else [1]
    assert after_steps == expected_steps
    state_hash(alg.policy.state_dict())
    state_hash(alg.optimizer.state_dict())
    assert (alg.policy.std > 0).all()
    return {'action_shape': [1, 29], 'source_PPO_update_called': True,
            'only_singleton_return_normalization_disabled': True,
            'optimizer_lr_before': before_lr, 'optimizer_lr_at_actual_Adam_step': observed[0],
            'expected_source_rule_lr': expected_lr, 'LR_multiplier': observed[0][0] / before_lr,
            'Adam_steps_before': before_steps, 'Adam_steps_after': after_steps,
            'losses': {key: float(value) for key, value in losses.items()}}


def run_synthetic(work: Path, results: list[dict[str, Any]]) -> None:
    work.mkdir(parents=True, exist_ok=False)
    fresh = make_runner()
    assert progress(fresh) == (0, -1, 0)
    initial = work / 'untrained.pt'
    fresh.save(str(initial))
    assert resolve_checkpoint_progress(cpu_load(initial)) == (-1, 0)
    restored = make_runner()
    restored.load(str(initial))
    assert progress(restored) == (0, -1, 0) and tensor_snapshot(restored) == tensor_snapshot(fresh)
    record(results, 'new_untrained_checkpoint_starts_at_zero', {'progress': progress(restored)})

    iterations: list[int] = []
    logged = make_runner(work / 'logged_training')
    install_update_trace(logged, iterations)
    saved_progress: list[dict[str, Any]] = []
    original_save = logged.save

    def save(path: str, infos: Any = None) -> None:
        original_save(path, infos)
        checkpoint = cpu_load(Path(path))
        last, following = resolve_checkpoint_progress(checkpoint)
        assert following == last + 1
        saved_progress.append({'filename': Path(path).name, 'iter': last, 'next': following})

    logged.save = save
    logged.learn(2)
    assert iterations == [0, 1] and progress(logged) == (1, 1, 2)
    assert saved_progress == [{'filename': 'model_0.pt', 'iter': 0, 'next': 1},
                              {'filename': 'model_1.pt', 'iter': 1, 'next': 2},
                              {'filename': 'model_1.pt', 'iter': 1, 'next': 2}]
    record(results, 'periodic_and_final_save_same_last_completed_convention', {'saved_progress': saved_progress})
    before_zero, before_progress = tensor_snapshot(logged), progress(logged)
    rng_before_zero = torch.random.get_rng_state().clone()
    logged.learn(0)
    after_zero = tensor_snapshot(logged)
    assert progress(logged) == before_progress and iterations == [0, 1]
    assert before_zero == after_zero and torch.equal(rng_before_zero, torch.random.get_rng_state())
    record(results, 'learn_zero_does_not_update_Adam_or_progress',
           {'progress': progress(logged), 'all_tensor_states_and_CPU_RNG_unchanged': True})
    for invalid in (-1, True, 1.5):
        message = expect_error(lambda invalid=invalid: logged.learn(invalid), (ValueError,))
        assert progress(logged) == before_progress and tensor_snapshot(logged) == before_zero
        assert torch.equal(rng_before_zero, torch.random.get_rng_state())
        record(results, f'invalid_learn_count_{invalid!r}_rejected_without_state_change', {'error': message})
    logged.learn(2)
    assert iterations == [0, 1, 2, 3] and progress(logged) == (3, 3, 4)
    assert logged.env.curriculum_learning_iteration_anchor == 2
    record(results, 'repeated_learn_has_no_duplicate_or_skipped_updates', {'update_indices': iterations})
    logged.writer.close()

    source = cpu_load(work / 'logged_training/model_3.pt')
    source['iter'] = 199
    source.pop('runner_state_version')
    source.pop('next_learning_iteration')
    source.pop('learning_iteration_phase', None)
    for group in source['optimizer_state_dict']['param_groups']:
        group['lr'] = 1e-5
    legacy = work / 'synthetic_legacy199.pt'
    torch.save(source, legacy)
    restored = make_runner()
    rng_before = torch.random.get_rng_state().clone()
    restored.load(str(legacy))
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert progress(restored) == (199, 199, 200)
    assert tensor_snapshot(restored) == expected_snapshot(source)
    assert restored.alg.learning_rate == 1e-5
    record(results, 'legacy_full_resume_exact_tensor_states_and_RNG', {'progress': progress(restored)})
    resaved = work / 'new_resaved199.pt'
    restored.save(str(resaved))
    second = make_runner()
    second.load(str(resaved))
    assert progress(second) == (199, 199, 200)
    assert tensor_snapshot(second) == expected_snapshot(source)
    second.save(str(work / 'second_resave199.pt'))
    third = make_runner()
    third.load(str(work / 'second_resave199.pt'))
    assert progress(third) == (199, 199, 200)
    record(results, 'load_save_load_versioned_progress_is_idempotent', {'progress': progress(third)})
    resumed_iters: list[int] = []
    install_update_trace(third, resumed_iters)
    third.learn(2)
    assert resumed_iters == [200, 201] and progress(third) == (201, 201, 202)
    assert third.env.curriculum_learning_iteration_anchor == 200
    record(results, 'legacy199_real_learn_begins_at200', {'update_indices': resumed_iters})

    fixed = make_runner(config=runner_config(schedule='fixed'))
    fixed.load(str(legacy))
    assert fixed.alg.learning_rate == 1e-5 and fixed.alg.schedule == 'fixed'
    record(results, 'fixed_schedule_restores_scalar_and_actual_Adam_LR', actual_singleton_update(fixed))
    adaptive = make_runner()
    adaptive.load(str(legacy))
    record(results, 'adaptive_resume_actual_Adam_step_has_1p5_not150_multiplier', actual_singleton_update(adaptive))

    weight_only = make_runner()
    before = tensor_snapshot(weight_only)
    weight_only.load(str(legacy), load_optimizer=False)
    after = tensor_snapshot(weight_only)
    assert after['Adam'] == before['Adam'] and weight_only.alg.learning_rate == 0.001
    assert {k: after[k] for k in ('policy', 'actor_norm', 'critic_norm')} == {k: expected_snapshot(source)[k] for k in ('policy', 'actor_norm', 'critic_norm')}
    assert progress(weight_only) == (199, 199, 200)
    record(results, 'weight_only_preserves_config_Adam_LR_and_legacy_eval_phase', {'progress': progress(weight_only)})
    weight_only_saved = work / 'weight_only_new_stage_save.pt'
    weight_only.save(str(weight_only_saved))
    weight_only_checkpoint = cpu_load(weight_only_saved)
    assert resolve_checkpoint_progress(weight_only_checkpoint) == (199, 200)
    assert resolve_checkpoint_phase(weight_only_checkpoint) == 199
    second_weight_only = make_runner()
    second_weight_only.load(str(weight_only_saved), load_optimizer=False)
    assert progress(second_weight_only) == (199, 199, 200)
    second_weight_only.save(str(work / 'second_weight_only_new_stage_save.pt'))
    second_weight_only_checkpoint = cpu_load(work / 'second_weight_only_new_stage_save.pt')
    assert resolve_checkpoint_progress(second_weight_only_checkpoint) == (199, 200)
    assert resolve_checkpoint_phase(second_weight_only_checkpoint) == 199
    record(results, 'weight_only_resave_preserves_eval_phase_and_checkpoint_progress',
           {'checkpoint_phase_before_and_after': 199, 'last_completed': 199, 'next_update': 200})
    weight_only_iters: list[int] = []
    install_update_trace(second_weight_only, weight_only_iters)
    second_weight_only.learn(1)
    assert weight_only_iters == [200] and progress(second_weight_only) == (200, 200, 201)
    record(results, 'default_weight_only_training_uses_next_instead_of_repeating_checkpoint_update',
           {'update_indices': weight_only_iters, 'fresh_Adam_at_retained_checkpoint_phase': True})

    for load_optimizer in (False, True):
        actor_only = make_runner()
        before = tensor_snapshot(actor_only)
        critic_before = state_hash(actor_only.alg.policy.critic.state_dict())
        actor_only.load(str(legacy), load_optimizer=load_optimizer, load_critic=False)
        after = tensor_snapshot(actor_only)
        assert after['Adam'] == before['Adam'] and after['critic_norm'] == before['critic_norm']
        assert state_hash(actor_only.alg.policy.critic.state_dict()) == critic_before
        assert after['actor_norm'] == expected_snapshot(source)['actor_norm']
        assert state_hash(actor_only.alg.policy.actor.state_dict()) == state_hash({k.removeprefix('actor.'): v for k, v in source['model_state_dict'].items() if k.startswith('actor.')})
        assert progress(actor_only) == (199, 199, 200) and actor_only.alg.learning_rate == 0.001
        record(results, f'actor_only_load_optimizer_{load_optimizer}_retains_critic_and_Adam', {'progress': progress(actor_only)})

    for load_optimizer in (False, True):
        retained = make_runner()
        before_progress = progress(retained)
        retained.load(str(legacy), load_optimizer=load_optimizer, load_iteration=False)
        assert progress(retained) == before_progress == (0, -1, 0)
        expected_lr = 1e-5 if load_optimizer else 0.001
        assert retained.alg.learning_rate == expected_lr
        record(results, f'explicit_load_iteration_false_optimizer_{load_optimizer}', {'progress': progress(retained)})

    evaluated = make_runner(work / 'logged_eval')
    evaluated.load(str(legacy))
    evaluated.eval_mode()
    before = tensor_snapshot(evaluated)
    eval_iterations: list[int] = []
    original_log = evaluated.log

    def log(locs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        eval_iterations.append(int(locs['it']))
        return original_log(locs, *args, **kwargs)

    evaluated.log = log
    evaluated.learn(2, eval_mode=True)
    assert eval_iterations == [199, 200] and progress(evaluated) == (199, 199, 200)
    assert evaluated.env.curriculum_learning_iteration_anchor == 199
    assert tensor_snapshot(evaluated) == before
    assert not list((work / 'logged_eval').glob('model_*.pt'))
    evaluated.writer.close()
    record(results, 'evaluation_uses_checkpoint_phase_and_does_not_save_or_update', {'log_indices': eval_iterations})

    grouped_source = make_runner()
    parameters = list(grouped_source.alg.policy.parameters())
    midpoint = len(parameters) // 2
    grouped_source.alg.optimizer = torch.optim.Adam([{'params': parameters[:midpoint]}, {'params': parameters[midpoint:]}], lr=1e-5)
    grouped_source.alg.learning_rate = 1e-5
    grouped_source.learn(1)
    grouped_path = work / 'two_uniform_groups.pt'
    grouped_source.save(str(grouped_path))
    grouped_checkpoint = cpu_load(grouped_path)
    target = make_runner()
    parameters = list(target.alg.policy.parameters())
    target.alg.optimizer = torch.optim.Adam([{'params': parameters[:midpoint]}, {'params': parameters[midpoint:]}], lr=1e-3)
    target.load(str(grouped_path))
    assert tensor_snapshot(target) == expected_snapshot(grouped_checkpoint)
    assert target.alg.learning_rate == grouped_checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
    record(results, 'two_uniform_groups_restore_exactly', {'group_lrs': [g['lr'] for g in target.alg.optimizer.param_groups]})
    incompatible = make_runner()
    before = tensor_snapshot(incompatible)
    message = expect_error(lambda: incompatible.load(str(grouped_path)), (RuntimeError,))
    assert tensor_snapshot(incompatible)['Adam'] == before['Adam'] and progress(incompatible) == (0, -1, 0)
    assert incompatible.alg.learning_rate == 0.001
    record(results, 'incompatible_full_PPO_Adam_is_not_silent_warmstart', {'error': message})

    bad_lr_cases = {'heterogeneous': [1e-5, 2e-5], 'zero': [0.0, 0.0],
                    'negative': [-1e-5, -1e-5], 'nan': [float('nan'), float('nan')],
                    'inf': [float('inf'), float('inf')], 'bool': [True, True]}
    for name, rates in bad_lr_cases.items():
        bad = copy.deepcopy(grouped_checkpoint)
        for group, rate in zip(bad['optimizer_state_dict']['param_groups'], rates, strict=True):
            group['lr'] = rate
        path = work / f'invalid_LR_{name}.pt'
        torch.save(bad, path)
        for schedule in ('adaptive', 'fixed'):
            target = make_runner(config=runner_config(schedule=schedule))
            before = state_hash(target.alg.optimizer.state_dict())
            message = expect_error(lambda: target.load(str(path)), (ValueError,))
            assert state_hash(target.alg.optimizer.state_dict()) == before and progress(target) == (0, -1, 0)
            assert target.alg.learning_rate == 0.001
            record(results, f'invalid_LR_{name}_{schedule}_rejected_before_Adam_load', {'error': message})

    for name, fields in (('wrong_next', {'iter': 199, 'runner_state_version': 1, 'next_learning_iteration': 199}),
                         ('float_iter', {'iter': 1.5}), ('bool_iter', {'iter': True}),
                         ('unknown_version', {'iter': 199, 'runner_state_version': 2, 'next_learning_iteration': 200})):
        message = expect_error(lambda fields=fields: resolve_checkpoint_progress(fields), (ValueError,))
        record(results, f'invalid_progress_{name}_rejected', {'error': message})
    assert checkpoint_progress(-1, 0)['next_learning_iteration'] == 0
    for bad_phase in (True, -1, 1.5):
        fields = checkpoint_progress(198, 199)
        fields['learning_iteration_phase'] = bad_phase
        message = expect_error(lambda fields=fields: resolve_checkpoint_phase(fields), (ValueError,))
        record(results, f'invalid_phase_{bad_phase!r}_rejected', {'error': message})
    assert ppo_optimizer_learning_rate({'param_groups': [{'lr': 1e-5}, {'lr': 1e-5}]}) == 1e-5
    assert not torch.cuda.is_initialized()


def infer_architecture(state: Mapping[str, torch.Tensor], prefix: str, final_dim: int) -> tuple[int, list[int]]:
    layers = sorted((int(match.group(1)), tensor) for key, tensor in state.items()
                    if (match := re.fullmatch(rf'{prefix}\.(\d+)\.weight', key)))
    assert layers and layers[-1][1].shape[0] == final_dim
    previous = int(layers[0][1].shape[1])
    input_dim = previous
    for _, tensor in layers:
        assert tensor.ndim == 2 and tensor.shape[1] == previous and torch.isfinite(tensor).all()
        previous = int(tensor.shape[0])
    return input_dim, [int(tensor.shape[0]) for _, tensor in layers[:-1]]


def run_real_checkpoint(path: Path, work: Path, results: list[dict[str, Any]]) -> None:
    assert path.exists() and path not in work.parents
    original_file_hash = file_hash(path)
    checkpoint = cpu_load(path)
    original_memory_hash = state_hash(checkpoint)
    actor_dim, actor_hidden = infer_architecture(checkpoint['model_state_dict'], 'actor', 29)
    critic_dim, critic_hidden = infer_architecture(checkpoint['model_state_dict'], 'critic', 1)
    assert actor_dim == 815 and critic_dim == 289
    config = runner_config(actor_hidden, critic_hidden, steps=1)
    runner = make_runner(env=SyntheticVecEnv(1, actor_dim, critic_dim), config=config)
    rng_before = torch.random.get_rng_state().clone()
    runner.load(str(path))
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert tensor_snapshot(runner) == expected_snapshot(checkpoint)
    last, following = resolve_checkpoint_progress(checkpoint)
    assert progress(runner) == (last, last, following)
    checkpoint_lr = ppo_optimizer_learning_rate(checkpoint['optimizer_state_dict'])
    assert runner.alg.learning_rate == checkpoint_lr
    prefix = f'real_{path.stem}_{original_file_hash[:12]}'
    resaved = work / f'{prefix}_new_format.pt'
    runner.save(str(resaved))
    resaved_checkpoint = cpu_load(resaved)
    assert resaved_checkpoint['runner_state_version'] == 1
    assert resolve_checkpoint_progress(resaved_checkpoint) == (last, following)
    assert expected_snapshot(resaved_checkpoint) == expected_snapshot(checkpoint)
    runner.load(str(resaved))
    assert progress(runner) == (last, last, following) and tensor_snapshot(runner) == expected_snapshot(checkpoint)
    record(results, f'{prefix}_exact_CPU_resume_and_new_format_idempotence',
           {'checkpoint': str(path), 'checkpoint_sha256': original_file_hash,
            'progress': progress(runner), 'exact_policy_normalizer_Adam_hashes': tensor_snapshot(runner),
            'original_actor_shape': [1, actor_dim], 'original_critic_shape': [1, critic_dim],
            'checkpoint_learning_rate': checkpoint_lr, 'CPU_load_RNG_unchanged': True})
    update = actual_singleton_update(runner)
    assert math.isclose(update['LR_multiplier'], 1.5, rel_tol=1e-12)
    assert file_hash(path) == original_file_hash and state_hash(checkpoint) == original_memory_hash
    record(results, f'{prefix}_actual_batch1_PPO_update_preserves_original_checkpoint', update)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-checkpoint', '--checkpoint', action='append', type=Path, dest='checkpoints')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--self-test-only', '--synthetic-only', action='store_true', dest='synthetic_only')
    args = parser.parse_args()
    assert not torch.cuda.is_initialized() and int(os.environ.get('WORLD_SIZE', '1')) == 1
    assert 'smoke_runtime' not in sys.modules and 'curve_runtime' not in sys.modules
    torch.set_num_threads(1)
    torch.manual_seed(20261006)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    work = args.output_dir / f'cpu_test_inputs_{stamp}'
    report = args.output_dir / f'cpu_test_report_{stamp}.json'
    results: list[dict[str, Any]] = []
    source_files = [Path(inspect.getfile(OnPolicyRunner)), SOURCE / 'rsl_rl/utils/checkpoint_resume.py']
    source_hashes = {str(path): file_hash(path) for path in source_files}
    error: str | None = None
    try:
        run_synthetic(work, results)
        if not args.synthetic_only:
            for path in args.checkpoints or DEFAULT_CHECKPOINTS:
                run_real_checkpoint(path.resolve(), work, results)
        assert not torch.cuda.is_initialized()
        assert source_hashes == {str(path): file_hash(path) for path in source_files}, 'Source changed during testing'
    except Exception:
        error = traceback.format_exc()
        raise
    finally:
        payload = {'status': 'passed' if error is None else 'failed', 'exception': error,
                   'CPU_only_cuda_initialized': torch.cuda.is_initialized(), 'synthetic_environment_is_not_physics': True,
                   'old_runtime_adapters_imported': False, 'source_hashes': source_hashes,
                   'tests_passed': len(results), 'tests': results, 'retained_synthetic_inputs': str(work),
                   'limits': ['No GPU/distributed execution was tested', 'No simulator/RNG/sampler continuation is claimed',
                              'Batch1 disables return normalization only, not the PPO update or adaptive LR rule']}
        with report.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'status': payload['status'], 'tests_passed': len(results), 'report': str(report)}))


if __name__ == '__main__':
    main()
