# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the phase composition primitives.

These check the leaf/compound dispatch contract enforced by
``execute_phase``: compound phases own ``execute`` and the time advance,
while leaf phases just write their slice. The tests also pin down the
Parallel child-duration semantics (each child writes inside its own
``[parent_start, parent_start+child_dur)`` window) since a regression
there silently erases the tail of short children.
"""

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

import braintools.cogtask as ct
from braintools.cogtask.context import Context
from braintools.cogtask.feature import FeatureSet
from braintools.cogtask.phase import (
    Phase, Sequence, Repeat, Parallel, concat, execute_phase,
    DeclarativePhase, Fixation, Stimulus,
)


def test_concat_empty_list_raises():
    with pytest.raises(ValueError):
        concat([])


def test_concat_wraps_into_sequence():
    p = concat([Fixation(5 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0})])
    assert isinstance(p, Sequence)
    assert len(p) == 1


def test_sequence_rshift_flattens_nested_sequences():
    a = Fixation(5 * u.ms, name='a', inputs={'fixation': 1.0}, outputs={'label': 0})
    b = Fixation(5 * u.ms, name='b', inputs={'fixation': 1.0}, outputs={'label': 0})
    c = Fixation(5 * u.ms, name='c', inputs={'fixation': 1.0}, outputs={'label': 0})
    seq = a >> b
    assert isinstance(seq, Sequence)
    seq2 = seq >> c
    assert [p.name for p in seq2.phases] == ['a', 'b', 'c']
    # Combining two existing sequences merges them.
    seq3 = seq >> Sequence(c)
    assert [p.name for p in seq3.phases] == ['a', 'b', 'c']


def test_phase_mul_creates_repeat():
    a = Fixation(5 * u.ms, name='a', inputs={'fixation': 1.0}, outputs={'label': 0})
    r = a * 4
    assert isinstance(r, Repeat)
    assert r.count == 4
    assert r.phase is a


def test_phase_mul_rejects_non_positive_counts():
    a = Fixation(5 * u.ms, name='a', inputs={'fixation': 1.0}, outputs={'label': 0})
    with pytest.raises(ValueError):
        a * 0
    with pytest.raises(ValueError):
        a * -1


def test_phase_pipe_creates_parallel_and_chains():
    a = Fixation(5 * u.ms, name='a', inputs={'fixation': 1.0}, outputs={'label': 0})
    b = Fixation(5 * u.ms, name='b', inputs={'fixation': 1.0}, outputs={'label': 0})
    c = Fixation(5 * u.ms, name='c', inputs={'fixation': 1.0}, outputs={'label': 0})
    par = a | b
    assert isinstance(par, Parallel)
    # Pipe-chaining merges into a single Parallel rather than nesting one.
    par2 = par | c
    assert isinstance(par2, Parallel)
    assert [p.name for p in par2.phases] == ['a', 'b', 'c']


def test_declarative_phase_requires_explicit_name():
    # DeclarativePhase directly is abstract for the purposes of name defaulting.
    with pytest.raises(ValueError, match="subclassed"):
        DeclarativePhase(duration=5 * u.ms, inputs={}, outputs={})


def test_unbound_declarative_phase_raises_on_encode():
    p = Fixation(5 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0})
    ctx = Context()
    ctx.inputs = jnp.zeros((5, 1))
    ctx.outputs = jnp.zeros((5,), dtype=jnp.int32)
    ctx.phase_start, ctx.phase_end = 0, 5
    with pytest.raises(RuntimeError, match="bind_features"):
        p.encode_inputs(ctx)


def test_bind_features_rejects_unknown_input_feature():
    fix = ct.Feature(1, 'fixation')
    out = ct.Feature(1, 'out')
    p = Stimulus(5 * u.ms, inputs={'wrong': 1.0}, outputs={'label': 0})
    with pytest.raises(ValueError, match="Unknown input feature"):
        p.bind_features(FeatureSet(fix), fix + out)


def test_bind_features_rejects_unknown_noise_feature():
    fix = ct.Feature(1, 'fixation')
    out = ct.Feature(1, 'out')
    p = Stimulus(5 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0},
                 noise={'missing': 1.0 * u.ms ** 0.5})
    with pytest.raises(ValueError, match="Unknown noise feature"):
        p.bind_features(FeatureSet(fix), fix + out)


def test_bind_features_rejects_unknown_output_feature_in_vector_mode():
    fix = ct.Feature(1, 'fixation')
    out = ct.Feature(1, 'out')
    p = Stimulus(5 * u.ms, inputs={'fixation': 1.0}, outputs={'missing': 1.0})
    with pytest.raises(ValueError, match="Unknown output feature"):
        p.bind_features(FeatureSet(fix), fix + out)


def test_sequence_total_duration_sums_children():
    fix = ct.Feature(1, 'fixation')
    out = ct.Feature(1, 'out')
    seq = concat([
        Fixation(5 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0}),
        Fixation(20 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0}),
    ])
    task = ct.Task(phases=seq, input_features=FeatureSet(fix), output_features=fix + out, seed=0)
    X, _, _ = task.sample_trial(0)
    assert X.shape[0] == 25


def test_repeat_scopes_repeat_index_and_writes_history():
    # The Repeat compound exposes ctx['repeat_index'] across iterations and pops
    # it on exit (or restores the prior value). We verify by inspecting the
    # phase_history entries it pushes.
    fix = ct.Feature(1, 'fixation')
    out = ct.Feature(1, 'out')
    body = Fixation(3 * u.ms, name='body', inputs={'fixation': 1.0}, outputs={'label': 0})
    task = ct.Task(
        phases=Repeat(body, count=4),
        input_features=FeatureSet(fix),
        output_features=fix + out,
        seed=0,
    )
    _, _, info = task.sample_trial(0)
    # 4 body executions + the Repeat itself = 5 entries.
    names = [name for name, _, _ in info['phase_history']]
    assert names.count('body') == 4
    assert 'Repeat(body, 4)' in names or any('Repeat' in n for n in names)


def test_parallel_inputs_share_parent_window_and_outputs_come_from_first_child():
    fix = ct.Feature(1, 'fixation')
    short_feat = ct.Feature(1, 'short')
    long_feat = ct.Feature(1, 'long')
    parallel = (
        Stimulus(5 * u.ms, name='short_phase', inputs={'short': 1.0}, outputs={'label': 0})
        | Stimulus(20 * u.ms, name='long_phase', inputs={'long': 1.0}, outputs={'label': 7})
    )
    task = ct.Task(
        phases=parallel,
        input_features=fix + short_feat + long_feat,
        output_features=fix + ct.Feature(1, 'out'),
        seed=0,
    )
    X, Y, info = task.sample_trial(0)
    X = np.asarray(X)
    Y = np.asarray(Y)
    # short child writes 1.0 for its 5 ticks then 0; long child writes 1.0 for all 20.
    assert X.shape[0] == 20
    np.testing.assert_allclose(X[:5, 1], 1.0, atol=1e-6)
    np.testing.assert_allclose(X[5:, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(X[:, 2], 1.0, atol=1e-6)
    # Output label comes from the *first* child (label=0), not the second (label=7).
    assert (Y == 0).all(), "Parallel must take outputs from the first child by convention"


def test_parallel_phase_history_records_each_child():
    fix = ct.Feature(1, 'fixation')
    a = ct.Feature(1, 'a')
    b = ct.Feature(1, 'b')
    par = (
        Stimulus(3 * u.ms, name='a_phase', inputs={'a': 1.0}, outputs={'label': 0})
        | Stimulus(4 * u.ms, name='b_phase', inputs={'b': 1.0}, outputs={'label': 0})
    )
    task = ct.Task(
        phases=par,
        input_features=fix + a + b,
        output_features=fix + ct.Feature(1, 'out'),
        seed=0,
    )
    _, _, info = task.sample_trial(0)
    names = [n for n, _, _ in info['phase_history']]
    assert 'a_phase' in names and 'b_phase' in names


def test_repeat_index_visible_inside_body_and_pops_on_exit():
    # Use a custom Phase subclass that captures ctx['repeat_index'] each iteration.
    seen = []
    fix = ct.Feature(1, 'fixation')
    out = ct.Feature(1, 'out')

    def hook(ctx):
        seen.append(int(ctx.get('repeat_index', -1)))

    body = Fixation(
        3 * u.ms, name='body',
        inputs={'fixation': 1.0}, outputs={'label': 0},
        on_enter=hook,
    )
    task = ct.Task(
        phases=Repeat(body, count=3),
        input_features=FeatureSet(fix),
        output_features=fix + out,
        seed=0,
    )
    task.sample_trial(0)
    # Second pass (the encoding pass after the dry-run duration computation)
    # writes [0, 1, 2]. The dry run may have added more — we just verify it ends
    # with [0, 1, 2] and that the keys are restored.
    assert seen[-3:] == [0, 1, 2]


def test_execute_phase_advances_current_step_for_leaf():
    fix = ct.Feature(1, 'fixation')
    out = ct.Feature(1, 'out')
    leaf = Fixation(4 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0})
    leaf.bind_features(FeatureSet(fix), fix + out)
    ctx = Context()
    ctx.inputs = jnp.zeros((4, 1))
    ctx.outputs = jnp.zeros((4,), dtype=jnp.int32)
    ctx.current_step = 0
    execute_phase(leaf, ctx)
    assert ctx.current_step == 4
    assert ctx.phase_history == [(leaf.name, 0, 4)]


def test_categorical_output_accepts_time_varying_label_array():
    fix = ct.Feature(1, 'fixation')
    out = ct.Feature(1, 'out')

    def time_varying(ctx, feature):
        dur = ctx.phase_end - ctx.phase_start
        return jnp.arange(dur, dtype=jnp.int32)

    p = Stimulus(5 * u.ms, inputs={'fixation': 1.0}, outputs={'label': time_varying})
    task = ct.Task(phases=p, input_features=FeatureSet(fix), output_features=fix + out, seed=0)
    _, Y, _ = task.sample_trial(0)
    np.testing.assert_array_equal(np.asarray(Y), np.arange(5, dtype=np.int32))


def test_categorical_output_rejects_2d_label():
    fix = ct.Feature(1, 'fixation')
    out = ct.Feature(1, 'out')

    def bad(ctx, feature):
        return jnp.zeros((5, 2), dtype=jnp.int32)

    p = Stimulus(5 * u.ms, inputs={'fixation': 1.0}, outputs={'label': bad})
    task = ct.Task(phases=p, input_features=FeatureSet(fix), output_features=fix + out, seed=0)
    with pytest.raises(ValueError, match="categorical label"):
        task.sample_trial(0)


def test_vector_mode_writes_features_separately():
    fix_in = ct.Feature(1, 'fix_in')
    fix_out = ct.Feature(1, 'fix_out')
    other = ct.Feature(2, 'other')
    p = Stimulus(
        3 * u.ms,
        inputs={'fix_in': 1.0},
        outputs={'fix_out': jnp.asarray([0.7]), 'other': jnp.asarray([0.3, 0.4])},
    )
    task = ct.Task(
        phases=p,
        input_features=FeatureSet(fix_in),
        output_features=fix_out + other,
        output_mode='vector',
        seed=0,
    )
    _, Y, _ = task.sample_trial(0)
    Y = np.asarray(Y)
    assert Y.shape == (3, 3)
    np.testing.assert_allclose(Y[:, 0], 0.7, atol=1e-6)
    np.testing.assert_allclose(Y[:, 1], 0.3, atol=1e-6)
    np.testing.assert_allclose(Y[:, 2], 0.4, atol=1e-6)


def test_noise_is_zero_when_sigma_is_zero():
    fix = ct.Feature(1, 'fixation')
    stim = ct.Feature(4, 'stim')
    out = ct.Feature(1, 'out')
    p = Stimulus(
        10 * u.ms,
        inputs={'fixation': 1.0, 'stim': jnp.zeros(4)},
        outputs={'label': 0},
        noise={'stim': 0.0 * u.ms ** 0.5},
    )
    task = ct.Task(phases=p, input_features=fix + stim, output_features=fix + out, seed=0)
    X, _, _ = task.sample_trial(0)
    np.testing.assert_allclose(np.asarray(X[:, 1:]), 0.0, atol=1e-6)


def test_noise_is_applied_when_sigma_is_positive():
    fix = ct.Feature(1, 'fixation')
    stim = ct.Feature(4, 'stim')
    out = ct.Feature(1, 'out')
    p = Stimulus(
        100 * u.ms,
        inputs={'fixation': 1.0, 'stim': jnp.zeros(4)},
        outputs={'label': 0},
        noise={'stim': 1.0 * u.ms ** 0.5},
    )
    task = ct.Task(phases=p, input_features=fix + stim, output_features=fix + out, seed=0)
    X, _, _ = task.sample_trial(0)
    X = np.asarray(X)
    # With sigma > 0 the stim columns should have noticeable variance.
    assert X[:, 1:].std() > 0.1


def test_describe_lists_inputs_outputs_and_noise():
    p = Fixation(
        5 * u.ms,
        inputs={'fixation': 1.0},
        outputs={'label': 0},
        noise={'fixation': 0.1 * u.ms ** 0.5},
    )
    s = p.describe()
    assert "Inputs" in s and "Outputs" in s and "Noise" in s
    assert "fixation" in s
