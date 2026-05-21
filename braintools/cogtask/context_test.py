# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``braintools.cogtask.context.Context``."""

import brainunit as u
import jax.numpy as jnp
import jax.random
import numpy as np

from braintools.cogtask.context import Context


def test_dict_protocol_roundtrip():
    ctx = Context()
    ctx['ground_truth'] = 3
    assert ctx['ground_truth'] == 3
    assert 'ground_truth' in ctx
    assert ctx.get('missing', 'fallback') == 'fallback'
    assert ctx.get('ground_truth') == 3


def test_update_merges_multiple_keys():
    ctx = Context()
    ctx['a'] = 1
    ctx.update(b=2, c=3)
    assert ctx['a'] == 1
    assert ctx['b'] == 2
    assert ctx['c'] == 3


def test_state_property_returns_copy():
    ctx = Context()
    ctx['x'] = 7
    snapshot = ctx.state
    snapshot['x'] = 99
    assert ctx['x'] == 7, "mutating the returned dict must not affect Context state"


def test_clear_resets_state_and_time_and_history():
    ctx = Context()
    ctx['x'] = 1
    ctx.current_step = 10
    ctx.phase_start = 5
    ctx.phase_end = 10
    ctx.current_phase = 'foo'
    ctx.phase_history.append(('foo', 5, 10))

    ctx.clear()

    assert 'x' not in ctx
    assert ctx.current_step == 0
    assert ctx.phase_start == 0
    assert ctx.phase_end == 0
    assert ctx.current_phase is None
    assert ctx.phase_history == []


def test_phase_duration_and_phase_time_properties():
    ctx = Context()
    ctx.phase_start = 100
    ctx.phase_end = 250
    ctx.current_step = 130
    assert ctx.phase_duration == 150
    assert ctx.phase_time == 30


def test_phase_slice_matches_start_end():
    ctx = Context()
    ctx.phase_start = 5
    ctx.phase_end = 15
    s = ctx.phase_slice()
    assert s == slice(5, 15)


def test_total_steps_reads_input_buffer():
    ctx = Context()
    assert ctx.total_steps == 0, "without inputs total_steps must be 0"
    ctx.inputs = jnp.zeros((42, 3))
    assert ctx.total_steps == 42


def test_dt_reads_from_brainstate_environ():
    ctx = Context()
    # The session fixture pins dt=1.0 ms.
    dt = ctx.dt
    assert hasattr(dt, 'mantissa')
    assert float(dt.mantissa) == 1.0


def test_explicit_key_seeds_rng_deterministically():
    key = jax.random.PRNGKey(123)
    ctx_a = Context(key=key)
    ctx_b = Context(key=key)
    a = float(ctx_a.rng.uniform())
    b = float(ctx_b.rng.uniform())
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_repr_includes_dt_step_and_phase():
    ctx = Context()
    ctx.current_step = 4
    ctx.current_phase = 'stimulus'
    s = repr(ctx)
    assert 'Context' in s
    assert 'step=4' in s
    assert 'stimulus' in s
