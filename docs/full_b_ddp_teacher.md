# Optional distributed teacher profile

This profile extends the authors' original teacher-training workflow with
local multi-GPU correctness repairs. Its runtime source is
`c23e488af67f7cc9aa57be4c457d4602cd06553d`.

## Configuration

Task: `General-Tracking-Flat-G1-Teacher-FullB-Points-v1`.
Runner configuration: `G1FullBPPORunnerCfg`.

The profile configures the existing motion loader with an 86,128-motion cap,
global contiguous rank sharding, and CPU dataset preloading. This count
belongs to the local Full-B reproduction catalog; configure a different
profile for a different catalog. The inherited loader and sampling logic are
unchanged. Loading a catalog establishes eligibility rather than execution
coverage. The profile retains 24 rollout steps, five PPO epochs, and four
minibatches; its separate training-duration settings are local choices.

## Synchronization

Online full-observation teacher actor and critic statistics are synchronized
initially and after each rollout. Frozen teachers and split-normalizer paths
are excluded. New rows are combined using float64 Chan central moments
(count, mean, M2), then merged with one copy of shared history. The original
`until=1e8` threshold retains per-rank semantics and whole-batch overshoot.
Cached next observations are normalized again without double-counting rows.

Shared history and local counters are checkpointed. Changing world size on
resume requires an explicit migration. Single-process evaluation can load
distributed checkpoints with normalization frozen. Single-process training
retains its original normalization path.

PPO broadcasts live parameters and buffers before learning. The optional
`strict_finite_checks` gate rejects a nonfinite input, loss, or gradient on
all ranks before an Adam update. Legacy profiles keep this gate disabled.

## CPU verification

With PyTorch and the normal package dependencies installed, run:

```bash
CUDA_VISIBLE_DEVICES="" python tests/test_distributed_ppo.py
CUDA_VISIBLE_DEVICES="" python tests/test_distributed_normalizer.py
CUDA_VISIBLE_DEVICES="" python tests/test_full_b_shard_config.py
CUDA_VISIBLE_DEVICES="" python tests/test_full_b_config_serialization.py
```

The historical normalization equivalence test expects a sibling checkout
named `AnyBody_teacher_correctness_fix_20261015` at source `306c5d0`.
The serialization check currently references a locally installed Isaac Lab
2.3.2 utility directory. These two test dependencies need configuration on
a fresh machine. CPU tests establish implementation behavior; they do not
measure humanoid control performance or replace native simulator validation.

The Points-v1 reward profile is a previously declared local reproduction
variant. Original task IDs and rewards remain available. Checkpoints,
training data, run controllers, and custom latent/terrain research are not
included in this publication.
