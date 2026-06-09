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
"""

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

import braintools.cogtask as ct
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
