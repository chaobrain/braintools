# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the conditional compound phases.

Earlier in the project's history these were treated as leaf phases by the
execute dispatch and silently produced zero-filled buffers. The tests below
pin down that ``If``/``Switch``/``While`` actually execute their bodies and
expose the right context keys (``switch_key``, ``while_iteration``,
``while_total_iterations``) on the way out.
"""

import brainunit as u
import jax.numpy as jnp
import numpy as np

import braintools.cogtask as ct
from braintools.cogtask.conditional import If, Switch, While
from braintools.cogtask.feature import FeatureSet
from braintools.cogtask.phase import Fixation, Stimulus


# ---------------------------------------------------------------------------
# If
# ---------------------------------------------------------------------------

def _build_if_task(cond_value):
    fix = ct.Feature(1, 'fixation')
    a = ct.Feature(2, 'a')
    b = ct.Feature(2, 'b')
    out = ct.Feature(1, 'out')
    phases = ct.concat([
        Fixation(10 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0}),
        If(
            lambda ctx: ctx['cond'],
            then=Stimulus(20 * u.ms, inputs={'a': jnp.asarray([1.0, 1.0])}, outputs={'label': 5}),
            else_=Stimulus(20 * u.ms, inputs={'b': jnp.asarray([1.0, 1.0])}, outputs={'label': 6}),
        ),
    ])
    return ct.Task(
        phases=phases,
        input_features=fix + a + b,
        output_features=fix + out,
        trial_init=lambda ctx: ctx.update(cond=cond_value),
        seed=0,
    )


def test_if_then_branch_executes_when_condition_true():
    task = _build_if_task(True)
    X, Y, _ = task.sample_trial(0)
    X = np.asarray(X)
    Y = np.asarray(Y)
    # The 'a' channel (cols 1-2) is non-zero in the second half (post-fixation).
    assert X[10:, 1:3].sum() > 0
    assert X[10:, 3:5].sum() == 0
    # Output label inside the conditional region is 5 (then branch label).
    assert (Y[10:] == 5).all()


def test_if_else_branch_executes_when_condition_false():
    task = _build_if_task(False)
    X, Y, _ = task.sample_trial(0)
    X = np.asarray(X)
    Y = np.asarray(Y)
    assert X[10:, 1:3].sum() == 0
    assert X[10:, 3:5].sum() > 0
    assert (Y[10:] == 6).all()


def test_if_without_else_collapses_duration_when_condition_false():
    fix = ct.Feature(1, 'fixation')
    out = ct.Feature(1, 'out')
    phases = ct.concat([
        Fixation(5 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0}),
        If(
            lambda ctx: ctx['go'],
            then=Stimulus(20 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 1}),
        ),
    ])
    task = ct.Task(
        phases=phases, input_features=FeatureSet(fix), output_features=fix + out,
        trial_init=lambda ctx: ctx.update(go=False), seed=0,
    )
    X, _, info = task.sample_trial(0)
    # Packed-mode: buffer is sized to ``max_trial_duration`` (5 + 20) and a
    # mask gates the actual valid timesteps. The false branch contributes 0
    # steps, so only the fixation's 5 timesteps should be valid.
    assert X.shape[0] == 25
    assert int(jnp.sum(info['mask'])) == 5


# ---------------------------------------------------------------------------
# Switch
# ---------------------------------------------------------------------------

def test_switch_dispatches_to_named_case():
    fix = ct.Feature(1, 'fixation')
    a = ct.Feature(1, 'a')
    b = ct.Feature(1, 'b')
    c = ct.Feature(1, 'c')
    out = ct.Feature(1, 'out')
    phases = Switch(
        lambda ctx: ctx['rule'],
        cases={
            'A': Stimulus(5 * u.ms, name='caseA', inputs={'a': 1.0}, outputs={'label': 1}),
            'B': Stimulus(5 * u.ms, name='caseB', inputs={'b': 1.0}, outputs={'label': 2}),
        },
        default=Stimulus(5 * u.ms, name='default', inputs={'c': 1.0}, outputs={'label': 9}),
    )
    task = ct.Task(
        phases=phases,
        input_features=fix + a + b + c,
        output_features=fix + out,
        trial_init=lambda ctx: ctx.update(rule='B'),
        seed=0,
    )
    X, Y, info = task.sample_trial(0)
    X = np.asarray(X)
    Y = np.asarray(Y)
    assert (X[:, 2] == 1.0).all(), "Case 'B' was not encoded"
    assert X[:, 1].sum() == 0 and X[:, 3].sum() == 0
    assert (Y == 2).all()
    # selector key is exposed to subsequent phases.
    assert info['trial_state']['switch_key'] == 'B'


def test_switch_falls_back_to_default_for_unknown_key():
    fix = ct.Feature(1, 'fixation')
    a = ct.Feature(1, 'a')
    c = ct.Feature(1, 'c')
    out = ct.Feature(1, 'out')
    phases = Switch(
        lambda ctx: ctx['rule'],
        cases={'A': Stimulus(5 * u.ms, inputs={'a': 1.0}, outputs={'label': 1})},
        default=Stimulus(5 * u.ms, inputs={'c': 1.0}, outputs={'label': 7}),
    )
    task = ct.Task(
        phases=phases, input_features=fix + a + c, output_features=fix + out,
        trial_init=lambda ctx: ctx.update(rule='nope'), seed=0,
    )
    X, Y, _ = task.sample_trial(0)
    X = np.asarray(X)
    assert X[:, 1].sum() == 0
    assert (X[:, 2] == 1.0).all()
    assert (np.asarray(Y) == 7).all()


def test_switch_with_unknown_key_and_no_default_yields_zero_duration():
    fix = ct.Feature(1, 'fixation')
    a = ct.Feature(1, 'a')
    out = ct.Feature(1, 'out')
    phases = ct.concat([
        Fixation(3 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0}),
        Switch(
            lambda ctx: ctx['rule'],
            cases={'A': Stimulus(10 * u.ms, inputs={'a': 1.0}, outputs={'label': 1})},
        ),
    ])
    task = ct.Task(
        phases=phases, input_features=fix + a, output_features=fix + out,
        trial_init=lambda ctx: ctx.update(rule='nope'), seed=0,
    )
    X, _, info = task.sample_trial(0)
    # Packed-mode: buffer is sized to ``max_trial_duration`` (3 + 10). The
    # Switch resolves to a missing key with no default and contributes 0
    # steps, so only the fixation's 3 timesteps should be marked valid.
    assert X.shape[0] == 13
    assert int(jnp.sum(info['mask'])) == 3


# ---------------------------------------------------------------------------
# While
# ---------------------------------------------------------------------------

def test_while_loop_runs_until_condition_false_and_records_iterations():
    fix = ct.Feature(1, 'fixation')
    out = ct.Feature(1, 'out')

    def bump(ctx):
        ctx['counter'] = int(ctx.get('counter', 0)) + 1

    body = Fixation(1 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0}, on_exit=bump)
    phases = While(
        condition=lambda ctx: int(ctx.get('counter', 0)) < 5,
        body=body,
        max_iterations=10,
    )
    task = ct.Task(
        phases=phases, input_features=FeatureSet(fix), output_features=fix + out,
        trial_init=lambda ctx: ctx.update(counter=0), seed=0,
    )
    _, _, info = task.sample_trial(0)
    # Encoding pass runs the loop 5 times. The dry-run pass also ran the
    # condition but its state was rolled back, so the final total_iterations
    # reflects the second pass.
    assert info['trial_state']['while_total_iterations'] == 5


def test_while_respects_max_iterations_safety_limit():
    fix = ct.Feature(1, 'fixation')
    out = ct.Feature(1, 'out')

    body = Fixation(1 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0})
    phases = While(
        condition=lambda ctx: True,
        body=body,
        max_iterations=3,
    )
    task = ct.Task(
        phases=phases, input_features=FeatureSet(fix), output_features=fix + out, seed=0,
    )
    _, _, info = task.sample_trial(0)
    assert info['trial_state']['while_total_iterations'] == 3


def test_while_get_duration_uses_max_iterations_as_upper_bound():
    fix = ct.Feature(1, 'fixation')
    out = ct.Feature(1, 'out')
    body = Fixation(2 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0})
    phases = While(condition=lambda ctx: True, body=body, max_iterations=4)
    task = ct.Task(
        phases=phases, input_features=FeatureSet(fix), output_features=fix + out, seed=0,
    )
    X, _, _ = task.sample_trial(0)
    # Buffer is allocated for the upper bound: 2 * 4 = 8 timesteps.
    assert X.shape[0] == 8
