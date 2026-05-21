# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for variable-length / packed-mode trial generation.

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


def test_variable_duration_samplers_have_bounds():
    te = ct.TruncExp(500 * u.ms, 200 * u.ms, 1000 * u.ms)
    assert te.is_variable is True
    assert float(te.min_value().to(u.ms).mantissa) == 200.0
    assert float(te.max_value().to(u.ms).mantissa) == 1000.0

    ud = ct.UniformDuration(100 * u.ms, 400 * u.ms)
    assert ud.is_variable is True
    assert float(ud.min_value().to(u.ms).mantissa) == 100.0
    assert float(ud.max_value().to(u.ms).mantissa) == 400.0
