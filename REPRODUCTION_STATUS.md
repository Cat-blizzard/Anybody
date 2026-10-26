# Local AnyBody reproduction status

This is a reproduction fork of [hazel-hammer/Anybody](https://github.com/hazel-hammer/Anybody),
not an official author release. Original attribution, acknowledgements, and
license files are retained. This document does not relicense upstream components.

## Source identities

| Source | Commit | Reference in this repository |
|---|---|---|
| Authors' public release | `0cffa139eab0bc145c9216223e249f03c67a324c` | Common ancestor of the local repairs |
| Legacy A/B/C teacher source | `c319a6609158fdb7456c065800b9a489b2605f5b` | `reproduction/legacy-c319a66` |
| Frozen New B teacher source | `306c5d0a4262ff3df435c00247c3cb0a73c34f7e` | `reproduction/new-b-306c5d0` |
| Teacher distributed-correctness source | `c23e488af67f7cc9aa57be4c457d4602cd06553d` | Ancestor of the 2026-10-26 publication on `main` |

The original publication added documentation to the frozen New B source.
The 2026-10-26 update also includes the teacher distributed-correctness changes
listed below. Runtime files match the identified distributed-correctness source;
publication documentation is maintained separately. Active training checkouts
and their frozen source identities remain separate from this publication.

## Publication scope: 2026-10-26

The authors' current public `main` was checked against release commit
`0cffa139eab0bc145c9216223e249f03c67a324c`. This update covers the original
teacher-training workflow and its engineering support:

- Broadcast actual policy parameters and buffers consistently across ranks.
- Optionally synchronize online teacher observation statistics, preserving
  single-process behavior and frozen or split-normalizer paths.
- Optionally reject nonfinite inputs, losses, and gradients collectively before
  an optimizer update.
- Provide a separate full-catalog teacher task/profile and CPU regressions.

The authors' original encoder/decoder, latent-RL modules, and downstream tasks
remain available as upstream code. This update does not add our custom latent
analysis, predictive/innovation research, terrain training, Plan v2.4 modules,
experiment controllers, model weights, logs, or raw experimental results.
The full-catalog profile is a local reproduction configuration, not an official
default or a new control-performance claim.

## Changes relative to the authors' release

- Preserve the reference frame on automatic reset, instead of advancing it one
  frame ahead of the newly written robot state.
- Refresh relative body position/orientation caches after cold or explicit reset.
- Restore PPO's scalar learning rate consistently with Adam, record the next
  update separately from the last completed update, and reject an incomplete
  full optimizer restore. Weight-only initialization can retain fresh progress.
- Remove unused actor debugging code that rejected an otherwise valid all-zero
  goal. The actor forward computation and network dimensions are retained.
- Add separate Cache, Wrists, and Points task profiles. Wrists expands the upper
  position group from four to six points; Points also activates a virtual-head
  and two-wrist reward. These are explicit objective variants, not all pure
  engineering fixes. The asset-derived head point has not been established as
  identical to the authors' or a headset's landmark.
- Add `link_v2` conversion, matching root link-origin and center-of-mass velocity
  semantics using `v_COM = v_link + omega_world x (R_link * r_COM)`. Legacy
  conversion remains selectable for comparison. New outputs require independent
  directories and record provenance metadata.
- Add a G1 entrypoint that pins source packages and adapts supported runtime API
  differences, optional-robot imports, and conversion shutdown handling.
- Add CPU regressions, a bounded native reset checker, and legacy-source isolation
  checks.

The original reward functions and task IDs remain available. The GAE/loss
formulas, original PPO configuration, six-hidden-layer teacher network, and
815/289/29 actor/critic/action interfaces are retained. The distributed update
changes synchronization and optional failure checks without adding a new
policy-learning objective. The historical uncommitted CPU-object broadcast
patch is not imported; this update includes the separately tested
tensor-broadcast implementation from the identified source.

## Experimental scope

New B uses the Points-v1 task and an independently converted `link_v2` version of
the same 14,391 B training clips. It is a combined comparison of correctness
repairs, reward variants, and velocity references. Performance differences cannot
be attributed to wrist coverage or velocity conversion alone.

The local experiment uses 1,024 environments on one GPU. This is not equivalent
to the public multi-GPU example's sample budget. Full learner restoration does
not restore simulator, RNG, or motion-sampler state, and is not a claim of
uninterrupted physical continuation. Fixed DEV completion is not automatically
the paper's accuracy-based success rate.

Formal New B teachers start from random initialization and later restore their
own checkpoints. Separately, an official MOSAIC ONNX actor has been migrated and
checked in a local experimental audit. That migration factory, its weights, and
its runtime are not included in this frozen source; it is not a vanilla resume
of an official training checkpoint and has not completed a PPO training chain.

Training data, checkpoints, raw trajectories, experiment controllers, and local
audit results are not included in this source publication.

## Validation and portability

For the 2026-10-26 publication, eight CPU entrypoints passed: finite-value
checks, normalization self-tests, two-rank PPO, distributed normalization,
full-catalog sharding, config serialization, synthetic checkpoint restoration,
and the zero-goal actor regression. All 185 tracked Python files parsed
successfully. Runtime files were checked byte-for-byte against `c23e488`;
the 14 upstream latent/residual modules remain byte-identical to `0cffa13`.
This publication check did not launch a simulator or execute GPU training.

CPU regressions, native reset checks, conversion checks, and a short training /
checkpoint-restore chain were previously run in the local environment. These
checks establish implementation behavior, not a general improvement in control
performance or reproduction of the paper's full results.

The original installation instructions are retained as upstream documentation.
Local compatibility acceptance used Isaac Lab 2.3.2, Isaac Sim 5.1, and PyTorch
2.7. A fresh-machine installation of this fork has not been validated.

Some historical audit tests refer to local Isaac Lab source, older checkouts,
assets, or checkpoints. They are not all portable to a fresh clone. For CPU-only
synthetic checkpoint checks with dependencies installed, use an independent
output directory:

```bash
CUDA_VISIBLE_DEVICES="" python tests/test_checkpoint_resume.py \
    --synthetic-only --output-dir /tmp/anybody-resume-check
CUDA_VISIBLE_DEVICES="" python tests/test_teacher_point_rewards.py
CUDA_VISIBLE_DEVICES="" python scripts/run_g1_reference_script.py --self-test
```

Scripts containing local paths must be configured for the destination machine.
The dated `NOT_RUN` statements in the original repair delivery describe that
delivery stage; later native acceptance was recorded in separate local receipts.
