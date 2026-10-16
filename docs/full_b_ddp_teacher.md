# Full-B distributed teacher profile

This local reproduction uses `General-Tracking-Flat-G1-Teacher-FullB-Points-v1`
and `G1FullBPPORunnerCfg`. Its requested formal run starts from random weights
with seed 20260904 on physical GPUs 1, 3, 4, 5, 6, 7, 8 and 9. It uses 4096
environments per rank, 24 control steps per rollout, and 5 epochs with 4 PPO
minibatches: 32768 total environments and 786432 control samples per update.
The user-selected formal budget is 100000 updates, with complete learner
checkpoints every 1000 completed updates.

## Data coverage

The training catalog contains exactly 86128 Full-B training motions. The
inherited `commands.py` selection/loading/sampling implementation is unchanged
from source commit `306c5d0a4262ff3df435c00247c3cb0a73c34f7e`.
The new environment profile overrides existing configuration fields only:

```text
motion_dataset_load_cap = 86128
motion_dataset_shard_across_gpus = True
motion_dataset_shard_by = "global"
motion_dataset_shard_strategy = "chunk"
motion_dataset_shard_seed = 0
motion_dataset_log_shard_info = True
motion_preload_device = "cpu"
```

The native selector then assigns 10766 distinct motions to each of eight
ranks. A single rank selects all 86128. Native acceptance must check actual
loaded identities, pairwise disjointness and the complete union; selected
counts alone are insufficient. `max_active_motions` still controls the
existing active-motion sampler and does not limit the full loaded catalog.
The teacher retains the original 50-second motion-to-environment reassignment
cadence. Loading every motion proves eligibility, not that every motion has
already contributed training samples; actual execution coverage is audited
separately. A 100-update smoke run contains only 48 simulated seconds.

Input inventory acceptance is separate from conversion acceptance. Of these
motions, 14391 have previously accepted `link_v2` outputs, which are rechecked
against all original 70 gates before reuse. The remaining 71737 require
native conversion by the original clean source306 producer. A training entry
must require the complete conversion receipt, all eight native worker Exit0
statuses, exact catalog membership, and frozen output hashes. Held-out data
are not members of this training catalog.

## Local engineering changes

Observation normalization synchronization is a locally developed repair;
it is not an official AnyBody or MOSAIC implementation. It is enabled only by
the new teacher profile. Online full-observation actor and critic statistics
are synchronized initially and once after each rollout, before GAE/PPO.
Frozen teacher and split-normalizer paths are excluded.

The helper reduces only newly accepted rows using float64 Chan central
moments `(count, mean, M2)`, then merges one copy of shared history. It updates
mean, variance, standard deviation and the true global count, and asserts
exact equality across ranks. The original `until=1e8` semantics remain a
per-rank threshold, including the original whole-final-batch overshoot.
Separate local counters and shared merge history are checkpointed. Cached
next observations are normalized again without counting their rows twice.
Single-process training retains the original normalization path. Changing
training world size on resume requires a separate explicit migration;
single-process evaluation can load a distributed checkpoint with statistics
frozen.

PPO broadcasts live parameters and buffers before learning. The new profile
also enables strict collective checks for raw actor inputs, distribution
parameters, losses and gradients. A nonfinite value on one rank rejects the
update on every rank before the next Adam step. The legacy default for
`strict_finite_checks` remains false. These checks do not tune PPO rewards or
hyperparameters.

The requested native acceptance sequence is single-rank fresh100 updates,
eight-rank fresh100 updates and an eight-rank cold-process full-learner
resume20. Scratch outputs are separate from the fresh formal100000 run.
Resume acceptance covers the model, both normalizers and distributed merge
history, Adam state, learning rate and completed-update accounting; it does
not assert restoration of simulator, RNG or sampler trajectories.

## Reproduction scope and evidence

This profile retains the earlier declared Points-v1 rewards: the upper-body
extension includes wrists, and the three-point reward uses an asset-defined
nominal head point. This is a local reward variant, not a certified exact
reconstruction of the authors' unpublished target definitions. Full-B data,
the 100000-update budget and the installed software stack are also declared
reproduction choices. No official checkpoint or MOSAIC warm start is used.

CPU evidence is stored in the associated run's `audit/` directory:

- `FULL_B_CONFIG_CPU.log`: actual inherited selector, complete/disjoint
  eight-rank and single-rank catalog selection, and the default-cap omission.
- `FULL_B_CONFIG_SERIALIZATION_CPU.log`: actual installed configclass
  serialization includes the strict gate and leaves legacy configuration and
  PPO hyperparameters unchanged.
- `NORMALIZER_DDP_CPU_20261017/CPU_RESULTS.json`: real two-rank Gloo and
  single-rank runner tests, unequal batches, multiple rounds, stable
  high-offset moments, `until`, frozen state, failure and checkpoint cases.
- `PPO_DDP_CPU_20261016T210044_071772Z/VALIDATION.json`: real PPO/Adam tests,
  legacy single-rank equivalence and collective failure rejection.

CPU checks do not constitute native simulator acceptance or evidence of
control-performance improvement. Native conversion, loading, training and
evaluation require their own positive receipts and successful process exits.
An eight-rank NCCL communication probe passed with `NCCL_P2P_DISABLE=1`;
that setting is a verified communication workaround, not a demonstrated
diagnosis of the underlying default-P2P timeout.
