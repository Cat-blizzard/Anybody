"""CPU regression of the actual legacy/new actor with identical state and RNG.

Input is (B,815), actions (B,29). This checks the removed unused debug branch,
not robot stability. Real actor layers, distributions and parameters execute.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source/rsl_rl"))
from rsl_rl.modules.actor_critic import ActorCritic

LEGACY = Path("/data/jxc/projects/bones_curated/references/AnyBody_resume_state_fix_20261006/source/rsl_rl/rsl_rl/modules/actor_critic.py")


def load_legacy() -> type:
    spec = importlib.util.spec_from_file_location("legacy_actor_for_CPU_test", LEGACY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.ActorCritic


class ActorZeroGoalTests(unittest.TestCase):
    def test_actual_actor_zero_boundary_and_finite_parity(self) -> None:
        legacy_class = load_legacy()
        for skip in (False, True):
            for noise in ("scalar", "log"):
                kwargs: dict[str, Any] = dict(num_actor_obs=815, num_critic_obs=289, num_actions=29,
                                              actor_hidden_dims=[8, 8], critic_hidden_dims=[8],
                                              ref_vel_skip_first_layer=skip, noise_std_type=noise)
                with contextlib.redirect_stdout(io.StringIO()):
                    old = legacy_class(**kwargs); new = ActorCritic(**kwargs)
                new.load_state_dict(old.state_dict())
                assert set(new.state_dict()) == set(old.state_dict())
                for batch in (1, 3):
                    normal = torch.randn(batch, 815)
                    torch.manual_seed(123); old_actions = old.act(normal)
                    torch.manual_seed(123); new_actions = new.act(normal)
                    assert torch.equal(old_actions, new_actions)
                    assert torch.equal(old.act_inference(normal), new.act_inference(normal))
                    for magnitude in (0., 1e-8):
                        values = torch.full((batch, 815), magnitude)
                        with self.assertRaises(RuntimeError):
                            old.act(values)
                        actions = new.act(values)
                        assert actions.shape == (batch, 29) and torch.isfinite(actions).all()
                        assert torch.isfinite(new.get_actions_log_prob(actions)).all()
                        assert torch.isfinite(new.evaluate(torch.zeros(batch, 289))).all()
                print(f"Actor real CPU: skip={skip}, noise={noise}, zero/near-zero B1/B3 PASS")
        assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
