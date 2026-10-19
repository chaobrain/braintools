# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the :class:`~braintools.cogtask.task.Task` orchestrator.

These cover:
- the construction / shape contract of ``sample_trial`` across every pre-built
  task (incl. the variable-duration ones),
- seed → trial reproducibility (per-trial keys are derived from
  ``(seed, index)`` via ``jax.random.fold_in`` so the same pair gives the
  same trial),
- ``batch_sample`` vmap shape + ``start_index`` semantics,
- class-based subclass attribute overrides via kwargs,
- the ``trial_init`` hook is wired correctly,
- ``output_mode='vector'`` allocates a 2-D output buffer.

Variable-length (packed) trials -- see the section at the end of this file:

Regression tests for variable-length / packed-mode trial generation.

Covers the design-spec test matrix:
  1. Fixed-length task is unchanged (no packed code path triggered).
  2. ``phase_tree_is_variable`` detects ``VariableDuration`` anywhere.
  3. ``max_trial_duration`` is a Python ``int`` (safe as a static shape).
  4. Packed ``sample_trial`` returns a mask aligned with the actual length.
  5. Mixed-length batches share buffer shape; mask sums vary per trial.
  6. ``batch_sample(return_mask=True)`` runs under JIT+vmap.
  7. Trailing positions past the per-trial length are zero in ``X``/``Y``.
  8. Migrated builtin tasks (IntervalDiscrimination, ReadySetGo,
     HierarchicalReasoning) work in packed mode.
  9. ``If`` packed branches: both ``then`` and ``else_`` produce shape-stable
     output via ``lax.cond``.
"""

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintools.cogtask as ct
from braintools.cogtask import (
    Task,
    Context,
    Feature,
    Fixation,
    Stimulus,
    Response,
    Delay,
    If,
    VariableDuration,
    concat,
    phase_tree_is_variable,
)
from braintools.cogtask.feature import FeatureSet
from .conftest import ALL_TASKS, FIXED_DURATION_TASKS


# ---------------------------------------------------------------------------
# Construction / shape contract — runs once per pre-built task
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_TASKS, ids=lambda c: c.__name__)
def test_sample_trial_produces_consistent_shapes(cls):
    """sample_trial(0) must succeed for every pre-built task and produce
    shape-consistent (X, Y) for the task's output_mode."""
    task = cls(seed=0)
    X, Y, info = task.sample_trial(0)
    assert X.ndim == 2
    assert X.shape[1] == task.num_inputs
    if task.output_mode == 'categorical':
        assert Y.ndim == 1
    else:
        assert Y.ndim == 2
        assert Y.shape[1] == task.num_outputs
    assert X.shape[0] == Y.shape[0]
    assert 'phase_history' in info and len(info['phase_history']) > 0
    assert info['index'] == 0


def test_seed_reproducibility_per_index():
    """Same (seed, index) → bit-identical trials across instances."""
    a = ct.PerceptualDecisionMaking(seed=42)
    b = ct.PerceptualDecisionMaking(seed=42)
    Xa, Ya, _ = a.sample_trial(7)
    Xb, Yb, _ = b.sample_trial(7)
    np.testing.assert_array_equal(np.asarray(Ya), np.asarray(Yb))
    np.testing.assert_allclose(np.asarray(Xa), np.asarray(Xb), rtol=1e-5, atol=1e-5)


def test_different_indices_yield_different_trials():
    """Successive trial indices must vary or batch_sample is useless."""
    task = ct.PerceptualDecisionMaking(seed=42)
    Y0 = np.asarray(task.sample_trial(0)[1])
    for i in range(1, 10):
        if not np.array_equal(Y0, np.asarray(task.sample_trial(i)[1])):
            return
    raise AssertionError("All trials produced identical Y across 10 indices")


def test_no_seed_still_produces_valid_trials():
    """A task without ``seed=`` falls back to brainstate default RNG."""
    task = ct.PerceptualDecisionMaking()  # no seed
    X, Y, info = task.sample_trial(0)
    assert X.shape[0] == Y.shape[0] > 0


def test_explicit_key_overrides_seed_derivation():
    import jax.random
    task = ct.PerceptualDecisionMaking(seed=0)
    key = jax.random.PRNGKey(999)
    Xa, _, _ = task.sample_trial(0, key=key)
    Xb, _, _ = task.sample_trial(0, key=key)
    np.testing.assert_array_equal(np.asarray(Xa), np.asarray(Xb))


def test_getitem_returns_X_and_Y_only():
    task = ct.PerceptualDecisionMaking(seed=0)
    X, Y = task[3]
    assert X.ndim == 2 and Y.ndim == 1


# ---------------------------------------------------------------------------
# batch_sample
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [
    ct.PerceptualDecisionMaking,
    ct.DelayMatchSample,
    ct.GoNoGo,
    ct.AntiReach,
], ids=lambda c: c.__name__)
def test_batch_sample_vmap_shape(cls):
    task = cls(seed=0)
    X, Y = task.batch_sample(4)
    assert X.shape[1] == 4
    assert Y.shape[1] == 4
    assert X.shape[2] == task.num_inputs


def test_batch_sample_time_first_false_makes_batch_leading():
    task = ct.PerceptualDecisionMaking(seed=0)
    X, Y = task.batch_sample(4, time_first=False)
    # batch on axis 0, time on axis 1.
    assert X.shape[0] == 4
    assert Y.shape[0] == 4


def test_batch_sample_start_index_changes_trials():
    a = ct.PerceptualDecisionMaking(seed=99)
    b = ct.PerceptualDecisionMaking(seed=99)
    _, Ya = a.batch_sample(4, start_index=0)
    _, Yb = b.batch_sample(4, start_index=0)
    np.testing.assert_array_equal(np.asarray(Ya), np.asarray(Yb))
    _, Yc = a.batch_sample(4, start_index=100)
    assert not np.array_equal(np.asarray(Ya), np.asarray(Yc))


def test_batch_sample_with_meta_returns_three_values():
    # DelayMatchSample overrides get_trial_meta to return (sample_idx, test_idx).
    task = ct.DelayMatchSample(seed=0)
    X, Y, meta = task.batch_sample(2, return_meta=True)
    assert X.shape[1] == 2
    assert Y.shape[1] == 2
    assert meta is not None


def test_batch_sample_with_default_meta_drops_string_leaves():
    # Tasks that fall back to the default get_trial_meta return the whole
    # trial_state dict, which includes a string (output_mode). batch_sample
    # must drop such non-JAX leaves so the numeric meta can be vmap-stacked.
    task = ct.PerceptualDecisionMaking(seed=0)
    X, Y, meta = task.batch_sample(4, return_meta=True)
    assert isinstance(meta, dict)
    assert 'output_mode' not in meta  # string leaf dropped
    assert np.asarray(meta['ground_truth']).shape[0] == 4


# ---------------------------------------------------------------------------
# Class-based subclass attribute overrides
# ---------------------------------------------------------------------------

def test_kwargs_override_class_attributes():
    """Kwargs forwarded through Task.__init__ override class attributes set by
    subclass constructors. This covers the long-tail of hyperparameter sweeps."""
    task = ct.PerceptualDecisionMaking(seed=0, num_choices=3)
    assert task.num_choices == 3
    # output_features changes shape accordingly: fix(1) + choice(3) = 4.
    assert task.num_outputs == 4


def test_repr_reports_task_metadata():
    task = ct.PerceptualDecisionMaking(seed=0, name='pdm')
    r = repr(task)
    assert 'pdm' in r
    assert 'inputs=' in r and 'outputs=' in r
    assert 'output_mode=categorical' in r


# ---------------------------------------------------------------------------
# Output mode + custom Task construction (instance-based path)
# ---------------------------------------------------------------------------

def test_categorical_output_buffer_is_one_d():
    task = ct.PerceptualDecisionMaking(seed=0)
    _, Y, _ = task.sample_trial(0)
    assert Y.ndim == 1 and Y.dtype == jnp.int32


def test_vector_output_buffer_is_two_d():
    task = ct.DelayDirectionReproduction(seed=0)
    _, Y, _ = task.sample_trial(0)
    assert Y.ndim == 2
    assert Y.shape[1] == task.num_outputs


def test_unknown_output_mode_rejected():
    fix = ct.Feature(1, 'fix')
    p = ct.Fixation(5 * u.ms, inputs={'fix': 1.0}, outputs={'label': 0})
    with pytest.raises(ValueError, match="output_mode"):
        ct.Task(
            phases=p, input_features=FeatureSet(fix), output_features=FeatureSet(fix),
            output_mode='nonsense', seed=0,
        )


def test_single_feature_accepted_without_featureset_wrapper():
    """A lone Feature (not wrapped in FeatureSet) is accepted as input/output
    features and normalized internally, so a single-channel task is usable."""
    fix = ct.Feature(1, 'fixation')
    task = ct.Task(
        phases=ct.concat([
            ct.Fixation(10 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0}),
            ct.Response(10 * u.ms, inputs={'fixation': 0.0}, outputs={'label': 1}),
        ]),
        input_features=fix,
        output_features=fix,
        seed=0,
    )
    assert isinstance(task.input_features, FeatureSet)
    assert isinstance(task.output_features, FeatureSet)
    X, Y = task[0]
    assert X.shape == (20, 1)
    assert Y.shape == (20,)


def test_create_task_factory_returns_task_instance():
    fix = ct.Feature(1, 'fix')
    out = ct.Feature(1, 'out')
    p = ct.Fixation(5 * u.ms, inputs={'fix': 1.0}, outputs={'label': 0})
    from braintools.cogtask.task import create_task
    task = create_task(p, FeatureSet(fix), fix + out, seed=0)
    assert isinstance(task, ct.Task)
    X, Y = task[0]
    assert X.shape[0] == 5


# ---------------------------------------------------------------------------
# dt (task-level time step, decoupled from the ambient environ dt)
# ---------------------------------------------------------------------------

def _two_phase_task(**kwargs):
    """A 10 ms + 10 ms task used to probe how dt resolves durations."""
    fix = ct.Feature(1, 'fixation')
    p = ct.concat([
        ct.Fixation(10 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0}),
        ct.Response(10 * u.ms, inputs={'fixation': 0.0}, outputs={'label': 1}),
    ])
    return ct.Task(phases=p, input_features=fix, output_features=fix, seed=0, **kwargs)


def test_dt_defaults_to_environ_when_unset():
    """Without dt=, the task resolves durations against the ambient environ dt
    (pinned to 1 ms by the package fixture)."""
    task = _two_phase_task()
    assert task.dt == 1.0 * u.ms
    X, _, info = task.sample_trial(0)
    assert X.shape[0] == 20            # (10 + 10) ms / 1 ms
    assert info['dt'] == 1.0 * u.ms


def test_dt_overrides_environ_for_durations_and_reported_dt():
    """A task-level dt drives buffer sizes and the reported dt, overriding the
    ambient environ dt (1 ms here)."""
    task = _two_phase_task(dt=2.0 * u.ms)
    assert task.dt == 2.0 * u.ms
    X, _, info = task.sample_trial(0)
    assert X.shape[0] == 10            # (10 + 10) ms / 2 ms, despite ambient 1 ms
    assert info['dt'] == 2.0 * u.ms


def test_create_task_forwards_dt():
    fix = ct.Feature(1, 'fix')
    out = ct.Feature(1, 'out')
    p = ct.Fixation(10 * u.ms, inputs={'fix': 1.0}, outputs={'label': 0})
    from braintools.cogtask.task import create_task
    task = create_task(p, FeatureSet(fix), fix + out, seed=0, dt=2.0 * u.ms)
    assert task.dt == 2.0 * u.ms
    X, _ = task[0]
    assert X.shape[0] == 5             # 10 ms / 2 ms


# ---------------------------------------------------------------------------
# num_classes (categorical class count, decoupled from num_outputs)
# ---------------------------------------------------------------------------

def test_num_classes_defaults_to_num_outputs_categorical():
    """Backward-compatible default: categorical num_classes == num_outputs."""
    task = ct.PerceptualDecisionMaking(seed=0)
    assert task.output_mode == 'categorical'
    assert task.num_classes == task.num_outputs


def test_num_classes_is_none_in_vector_mode():
    task = ct.DelayDirectionReproduction(seed=0)
    assert task.output_mode == 'vector'
    assert task.num_classes is None


def test_num_classes_explicit_decouples_from_num_outputs():
    """An explicit num_classes is independent of the output-feature width,
    which otherwise grows with e.g. cue_dim while the label space does not."""
    fix = ct.Feature(1, 'fix')
    p = ct.concat([
        ct.Fixation(5 * u.ms, inputs={'fix': 1.0}, outputs={'label': 0}),
        ct.Response(5 * u.ms, outputs={'label': 1}),
    ])
    task = ct.Task(
        phases=p, input_features=fix, output_features=fix + ct.Feature(4, 'r'),
        num_classes=3, seed=0,
    )
    assert task.num_outputs == 5   # 1 + 4 output-feature dims
    assert task.num_classes == 3   # but only 3 label classes

    from braintools.cogtask.task import create_task
    t2 = create_task(p, fix, fix + ct.Feature(4, 'r'), num_classes=2, seed=0)
    assert t2.num_classes == 2


def test_phases_without_features_raises_clear_error():
    p = ct.Fixation(5 * u.ms, inputs={'fix': 1.0}, outputs={'label': 0})
    with pytest.raises(ValueError, match="input_features.*output_features"):
        ct.Task(phases=p, seed=0)


def test_out_of_range_literal_label_rejected_when_num_classes_declared():
    """A literal int label outside [0, num_classes) is caught at bind time when
    the task declares num_classes (the only sound bound; output dim is not)."""
    fix = ct.Feature(1, 'fix')
    p = ct.concat([
        ct.Fixation(3 * u.ms, inputs={'fix': 1.0}, outputs={'label': 0}),
        ct.Response(3 * u.ms, outputs={'label': 99}),   # 99 >= num_classes (3)
    ])
    with pytest.raises(ValueError, match="out of range"):
        ct.Task(phases=p, input_features=fix, output_features=fix + ct.Feature(2, 'r'),
                num_classes=3, seed=0)


def test_in_range_literal_label_accepted_with_num_classes():
    fix = ct.Feature(1, 'fix')
    p = ct.concat([
        ct.Fixation(3 * u.ms, inputs={'fix': 1.0}, outputs={'label': 0}),
        ct.Response(3 * u.ms, outputs={'label': 2}),    # within [0, 3)
    ])
    task = ct.Task(phases=p, input_features=fix, output_features=fix + ct.Feature(2, 'r'),
                   num_classes=3, seed=0)
    _, Y = task[0]
    assert int(np.asarray(Y).max()) == 2


def test_label_not_validated_without_num_classes():
    """Without an explicit num_classes there is no sound upper bound (output
    dim != class count), so literal labels are left unchecked — preserving
    tasks that encode N classes in fewer output dims."""
    fix = ct.Feature(1, 'fix')
    p = ct.concat([
        ct.Fixation(3 * u.ms, inputs={'fix': 1.0}, outputs={'label': 0}),
        ct.Response(3 * u.ms, outputs={'label': 1}),    # 1 classes into a 1-dim output
    ])
    task = ct.Task(phases=p, input_features=fix, output_features=fix, seed=0)  # num_outputs == 1
    _, Y = task[0]
    assert int(np.asarray(Y).max()) == 1


def test_class_based_task_without_define_features_raises():
    class IncompleteTask(ct.Task):
        pass

    with pytest.raises(NotImplementedError, match="define_features"):
        IncompleteTask()


def test_class_based_task_without_define_phases_raises():
    class HasFeaturesOnly(ct.Task):
        def define_features(self):
            fix = ct.Feature(1, 'fix')
            return fix, fix + ct.Feature(1, 'out')

    with pytest.raises(NotImplementedError, match="define_phases"):
        HasFeaturesOnly()


# ---------------------------------------------------------------------------
# Compound-phase integration (migrated from cogtask_test.py)
# ---------------------------------------------------------------------------

def test_pulse_decision_making_runs_through_repeat():
    """PulseDecisionMaking is a stress test for ``Repeat`` because its inner
    encoder reads ``ctx['repeat_index']`` and writes a different value into
    the cue channel each iteration."""
    task = ct.PulseDecisionMaking(seed=3)
    for i in range(5):
        _, Y, info = task.sample_trial(i)
        gt = int(info['trial_state']['ground_truth'])
        last = int(np.asarray(Y)[-1])
        assert last == gt + 1


# ---------------------------------------------------------------------------
# create_task factory: must be exported from the package and produce a
# sampleable task. Also exercises the (previously broken) docstring pattern
# of a string label spec wrapped in label(...).
# ---------------------------------------------------------------------------

def test_create_task_is_exported_from_package():
    assert hasattr(ct, 'create_task')
    assert 'create_task' in ct.__all__


def test_create_task_builds_sampleable_task_with_string_label():
    task = ct.create_task(
        phases=(
            ct.Fixation(10 * u.ms, inputs={'fixation': 1.0})
            >> ct.Stimulus(20 * u.ms, inputs={'stimulus': ct.circular('direction')})
            >> ct.Response(10 * u.ms, outputs={'label': ct.label('ground_truth')})
        ),
        input_features=ct.Feature(1, 'fixation') + ct.Feature(2, 'stimulus'),
        output_features=ct.Feature(1, 'fixation') + ct.Feature(2, 'choice'),
        trial_init=lambda ctx: ctx.update(
            ground_truth=ctx.rng.choice(2),
            direction=ctx.rng.uniform(0, 2 * 3.14159),
        ),
        seed=0,
    )
    X, Y, info = task.sample_trial(0)
    assert X.shape[0] == Y.shape[0] == 40
    gt = int(info['trial_state']['ground_truth'])
    # Response phase is the last 10 ticks; label == ground_truth there.
    assert (np.asarray(Y)[-10:] == gt).all()


# ---------------------------------------------------------------------------
# Variable-length (packed) trials
# ---------------------------------------------------------------------------

def _make_variable_task(seed: int = 0) -> Task:
    fix = Feature(1, 'fixation')
    stim = Feature(2, 'stim')
    choice = Feature(2, 'choice')

    phases = concat([
        Fixation(50 * u.ms, inputs={'fixation': 1.0}),
        VariableDuration(
            min_duration=20 * u.ms,
            max_duration=200 * u.ms,
            ctx_key='delay',
            inputs={'fixation': 1.0},
        ),
        Stimulus(40 * u.ms, inputs={'stim': lambda c, f: jnp.ones(f.num)}),
        Response(20 * u.ms, outputs={'label': lambda c, f: c['gt']}),
    ])

    def init(ctx):
        ctx['delay'] = ctx.rng.uniform(20.0, 200.0)
        ctx['gt'] = ctx.rng.choice(2).astype(jnp.int32) + 1

    return Task(
        phases=phases,
        input_features=fix + stim,
        output_features=fix + choice,
        trial_init=init,
        seed=seed,
    )


def _make_fixed_task(seed: int = 0) -> Task:
    fix = Feature(1, 'fixation')
    stim = Feature(2, 'stim')
    choice = Feature(2, 'choice')

    phases = concat([
        Fixation(50 * u.ms, inputs={'fixation': 1.0}),
        Stimulus(40 * u.ms, inputs={'stim': lambda c, f: jnp.ones(f.num)}),
        Response(20 * u.ms, outputs={'label': lambda c, f: c['gt']}),
    ])

    def init(ctx):
        ctx['gt'] = ctx.rng.choice(2).astype(jnp.int32) + 1

    return Task(
        phases=phases,
        input_features=fix + stim,
        output_features=fix + choice,
        trial_init=init,
        seed=seed,
    )


def test_fixed_task_is_not_variable():
    task = _make_fixed_task()
    assert task.is_variable_length is False
    X, Y, info = task.sample_trial(0)
    assert info['mask'] is None
    assert X.shape[0] == Y.shape[0]


def test_phase_tree_detection():
    fixed = Fixation(50 * u.ms, inputs={'fixation': 1.0})
    assert phase_tree_is_variable(fixed) is False

    var = VariableDuration(
        min_duration=20 * u.ms, max_duration=100 * u.ms, ctx_key='d',
        inputs={'fixation': 1.0},
    )
    assert phase_tree_is_variable(var) is True
    assert phase_tree_is_variable(concat([fixed, var])) is True


def test_max_trial_duration_is_python_int():
    task = _make_variable_task()
    md = task.max_trial_duration()
    assert isinstance(md, int)
    # 50 + 200 + 40 + 20 ms at dt=1ms = 310 steps.
    assert md == 310


def test_packed_sample_trial_mask_matches_actual_length():
    task = _make_variable_task()
    X, Y, info = task.sample_trial(7)
    mask = info['mask']
    assert mask.shape[0] == X.shape[0] == 310
    assert mask.dtype == jnp.bool_
    actual = int(jnp.sum(mask))
    # 50 + delay + 40 + 20, with delay in [20, 200] → in [130, 310] steps.
    assert 130 <= actual <= 310


def test_packed_trailing_zeros_in_X_and_Y():
    task = _make_variable_task()
    X, Y, info = task.sample_trial(3)
    mask = info['mask']
    inv = jnp.logical_not(mask)
    assert bool(jnp.all(X[inv] == 0))
    assert bool(jnp.all(Y[inv] == 0))


def test_packed_mask_is_packed_at_front():
    """The mask should be True for ``[0, actual)`` and False after."""
    task = _make_variable_task()
    _, _, info = task.sample_trial(5)
    mask = info['mask']
    actual = int(jnp.sum(mask))
    assert bool(jnp.all(mask[:actual]))
    assert bool(jnp.all(jnp.logical_not(mask[actual:])))


def test_packed_batch_sample_jit_vmap():
    task = _make_variable_task()
    X, Y, mask = task.batch_sample(8, return_mask=True)
    # Time-first layout: (T, B, F)
    assert X.shape == (310, 8, 3)
    assert Y.shape == (310, 8)
    assert mask.shape == (310, 8)
    # Each trial should mask at least the fixed portion (110 steps).
    sums = [int(s) for s in mask.sum(axis=0)]
    assert all(110 <= s <= 310 for s in sums)
    # Trials should differ in length (random delay sampling).
    assert len(set(sums)) > 1


def test_packed_batch_sample_without_mask_still_works():
    task = _make_variable_task()
    X, Y = task.batch_sample(4)
    assert X.shape == (310, 4, 3)
    assert Y.shape == (310, 4)


def test_fixed_batch_sample_return_mask_is_all_true():
    task = _make_fixed_task()
    X, Y, mask = task.batch_sample(4, return_mask=True)
    assert mask.shape[1] == 4
    assert bool(jnp.all(mask))


@pytest.mark.parametrize(
    'cls', [ct.IntervalDiscrimination, ct.ReadySetGo, ct.HierarchicalReasoning]
)
def test_migrated_builtin_tasks_packed(cls):
    task = cls(seed=42)
    assert task.is_variable_length is True
    md = task.max_trial_duration()
    assert isinstance(md, int) and md > 0

    X, Y, info = task.sample_trial(0)
    mask = info['mask']
    assert X.shape[0] == md
    assert mask.shape[0] == md
    assert 0 < int(jnp.sum(mask)) <= md

    # batch_sample under JIT+vmap
    X, Y, M = task.batch_sample(4, return_mask=True)
    assert X.shape[0] == md
    assert M.shape == (md, 4)
    sums = [int(s) for s in M.sum(axis=0)]
    assert all(0 < s <= md for s in sums)


def test_if_packed_branches():
    fix = Feature(1, 'fixation')
    stim = Feature(2, 'stim')
    choice = Feature(2, 'choice')

    phases = concat([
        Fixation(20 * u.ms, inputs={'fixation': 1.0}),
        If(
            lambda ctx: ctx['go'],
            then=Stimulus(40 * u.ms, inputs={'stim': lambda c, f: jnp.ones(f.num)}),
            else_=Fixation(40 * u.ms, inputs={'fixation': 0.5}),
        ),
        Response(20 * u.ms, outputs={'label': lambda c, f: c['gt']}),
    ])

    def init(ctx):
        ctx['go'] = ctx.rng.choice(2).astype(jnp.bool_)
        ctx['gt'] = ctx.rng.choice(2).astype(jnp.int32) + 1

    task = Task(
        phases=phases,
        input_features=fix + stim,
        output_features=fix + choice,
        trial_init=init,
        seed=0,
    )
    assert task.is_variable_length is True
    md = task.max_trial_duration()
    assert md == 80  # 20 + max(40, 40) + 20

    X, Y, info = task.sample_trial(0)
    assert X.shape[0] == md
    assert info['mask'].shape[0] == md

    # JIT+vmap path
    X, Y, M = task.batch_sample(8, return_mask=True)
    assert X.shape == (md, 8, 3)
    # All 8 trials should fill the buffer (both branches are 40 ms long).
    assert bool(jnp.all(M))


def test_on_enter_sees_slot_position_in_packed_mode():
    """In packed (variable-length) mode, a leaf's ``on_enter`` hook must see
    ``phase_start``/``phase_end`` pointing at its trial slot — the same
    contract as fixed-mode ``execute_phase`` — not stale values from the
    previous phase."""
    fix = Feature(1, 'fixation')
    seen = {}

    def record(ctx):
        seen['start'] = int(ctx.phase_start)
        seen['end'] = int(ctx.phase_end)

    phases = concat([
        Fixation(50 * u.ms, inputs={'fixation': 1.0}),
        VariableDuration(
            min_duration=20 * u.ms, max_duration=200 * u.ms, ctx_key='delay',
            inputs={'fixation': 1.0}, on_enter=record,
        ),
    ])

    def init(ctx):
        ctx['delay'] = ctx.rng.uniform(20.0, 200.0)

    task = Task(phases=phases, input_features=fix, output_features=Feature(1, 'o'),
                trial_init=init, seed=0)
    _, _, info = task.sample_trial(0)
    # The variable phase starts right after the 50-step fixation.
    assert seen['start'] == 50
    # phase_end reflects the phase's own actual (clipped) length, not 0/stale.
    assert seen['end'] > seen['start']


@pytest.mark.parametrize('value, dt', [
    (500.0, 1.0 * u.ms),             # plain float is milliseconds
    (500 * u.ms, 1.0 * u.ms),
    (0.5 * u.second, 1.0 * u.ms),    # was read as 0.5 ms -> 1 step
    (500 * u.ms, 0.001 * u.second),  # was 500 / 0.001 -> clipped to the slot
    (500.0, 0.001 * u.second),
])
def test_variable_duration_converts_units(value, dt):
    task = Task(
        phases=VariableDuration(
            min_duration=100 * u.ms,
            max_duration=1000 * u.ms,
            ctx_key='d',
            inputs={'x': 1.0},
            outputs={'label': 0},
        ),
        input_features=Feature(1, 'x'),
        output_features=Feature(1, 'y'),
        trial_init=lambda ctx: ctx.update(d=value),
        seed=0,
        num_classes=1,
        dt=dt,
    )
    _, _, info = task.sample_trial(0)
    assert int(np.asarray(info['mask']).sum()) == 500
    _, _, mask = task.batch_sample(2, return_mask=True)
    assert np.asarray(mask.sum(axis=0)).tolist() == [500, 500]


def test_variable_duration_samplers_have_bounds():
    te = ct.TruncExp(500 * u.ms, 200 * u.ms, 1000 * u.ms)
    assert te.is_variable is True
    assert float(te.min_value().to(u.ms).mantissa) == 200.0
    assert float(te.max_value().to(u.ms).mantissa) == 1000.0

    ud = ct.UniformDuration(100 * u.ms, 400 * u.ms)
    assert ud.is_variable is True
    assert float(ud.min_value().to(u.ms).mantissa) == 100.0
    assert float(ud.max_value().to(u.ms).mantissa) == 400.0
