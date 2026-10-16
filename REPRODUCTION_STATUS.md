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

The publication commit on `main` adds documentation only relative to the frozen
New B source. The active training checkout and its frozen source identity are
kept separate from this publication checkout.

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

The original reward functions and task IDs remain available. The PPO algorithm,
GAE/loss formulas, original PPO configuration, six-hidden-layer teacher network,
and 815/289/29 actor/critic/action interfaces are not changed by these repairs.
The separate uncommitted CPU broadcast patch in the original reference checkout
is not part of this frozen source.

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
