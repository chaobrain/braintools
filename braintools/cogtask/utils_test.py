# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``braintools.cogtask.utils``."""

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.cogtask.utils import (
    TruncExp,
    UniformDuration,
    Transform,
    TransformIT,
    initialize,
    initialize2,
    interval_of,
    period_to_arr,
    rate_to_prob,
    firing_rate,
    choice,
)
from braintools.cogtask.context import Context


# ---------------------------------------------------------------------------
# TruncExp
# ---------------------------------------------------------------------------

def test_truncexp_samples_within_bounds():
    rng = np.random.default_rng(0)
    sampler = TruncExp(600 * u.ms, 300 * u.ms, 1500 * u.ms)
    for _ in range(50):
        v = sampler()
        m = float(v.to(u.ms).mantissa)
        assert 300.0 <= m <= 1500.0 + 1e-6


def test_truncexp_min_ge_max_returns_max():
    sampler = TruncExp(100 * u.ms, 200 * u.ms, 100 * u.ms)
    v = sampler()
    assert float(v.to(u.ms).mantissa) == pytest.approx(100.0)


def test_truncexp_repr_describes_parameters():
    s = repr(TruncExp(100 * u.ms, 50 * u.ms, 200 * u.ms))
    assert 'TruncExp' in s
    assert '100' in s and '50' in s and '200' in s


def test_truncexp_uses_ctx_rng_when_provided():
    ctx = Context()
    sampler = TruncExp(500 * u.ms, 100 * u.ms, 1000 * u.ms)
    v = sampler(ctx)
    assert 100.0 <= float(v.to(u.ms).mantissa) <= 1000.0 + 1e-6


# ---------------------------------------------------------------------------
# UniformDuration
# ---------------------------------------------------------------------------

def test_uniform_duration_within_bounds():
    sampler = UniformDuration(200 * u.ms, 400 * u.ms)
    for _ in range(30):
        v = sampler()
        m = float(v.to(u.ms).mantissa)
        assert 200.0 <= m <= 400.0


def test_uniform_duration_repr():
    s = repr(UniformDuration(200 * u.ms, 400 * u.ms))
    assert 'UniformDuration' in s
    assert '200' in s and '400' in s


# ---------------------------------------------------------------------------
# Transform / TransformIT
# ---------------------------------------------------------------------------

def test_transform_is_subclassable():
    class MyT(Transform):
        pass

    assert issubclass(MyT, Transform)


def test_transform_it_applies_both_pipelines():
    t = TransformIT(
        input_transform=lambda x: x * 2,
        target_transform=lambda y: y + 1,
    )
    x, y = t(3, 7)
    assert x == 6 and y == 8


def test_transform_it_skips_none_transforms():
    t = TransformIT(input_transform=lambda x: x * 2)
    x, y = t(3, 7)
    assert x == 6 and y == 7


def test_transform_it_repr_lists_transforms():
    t = TransformIT(input_transform=lambda x: x, target_transform=lambda y: y)
    s = repr(t)
    assert 'TransformIT' in s
    assert 'Input transform' in s
    assert 'Target transform' in s


# ---------------------------------------------------------------------------
# initialize / initialize2
# ---------------------------------------------------------------------------

def test_initialize_passthrough_for_scalars_and_quantities():
    assert initialize(5) == 5
    assert initialize(2.5) == 2.5
    q = 100 * u.ms
    assert initialize(q) is q


def test_initialize_invokes_callable():
    assert initialize(lambda: 42) == 42


def test_initialize_none_raises_unless_allowed():
    with pytest.raises(TypeError):
        initialize(None)
    assert initialize(None, allow_none=True) is None


def test_initialize_unknown_type_raises():
    with pytest.raises(TypeError):
        initialize("not allowed")


def test_initialize2_converts_to_timesteps_with_plain_dt():
    # initialize2 compares ``dt == 0`` directly, so dt must be unit-free here.
    assert initialize2(100, 1) == 100
    assert initialize2(50, 2) == 25


def test_initialize2_invokes_callable():
    assert initialize2(lambda: 30, 1) == 30


def test_initialize2_dt_zero_raises():
    with pytest.raises(ValueError, match="dt"):
        initialize2(100, 0)


def test_initialize2_none_raises_unless_allowed():
    with pytest.raises(TypeError):
        initialize2(None, 1)
    assert initialize2(None, 1, allow_none=True) is None


def test_initialize2_unknown_type_raises():
    with pytest.raises(TypeError):
        initialize2("nope", 1)


# ---------------------------------------------------------------------------
# interval_of / period_to_arr
# ---------------------------------------------------------------------------

def test_interval_of_with_dict():
    periods = {'fixation': 10, 'stimulus': 20, 'delay': 15}
    assert interval_of('stimulus', periods) == slice(10, 30)


def test_interval_of_with_sequence_of_tuples():
    periods = [('a', 5), ('b', 10), ('c', 7)]
    assert interval_of('c', periods) == slice(15, 22)


def test_interval_of_missing_raises():
    with pytest.raises(ValueError):
        interval_of('missing', {'x': 1})


def test_period_to_arr_labels_each_step_by_period_index():
    arr = period_to_arr({'fixation': 3, 'stimulus': 2})
    np.testing.assert_array_equal(np.asarray(arr), [0, 0, 0, 1, 1])


def test_period_to_arr_three_periods():
    arr = period_to_arr({'a': 1, 'b': 2, 'c': 3})
    np.testing.assert_array_equal(np.asarray(arr), [0, 1, 1, 2, 2, 2])


# ---------------------------------------------------------------------------
# rate_to_prob (and its deprecated alias firing_rate)
# ---------------------------------------------------------------------------

def test_rate_to_prob_returns_probability_per_step():
    # 100 Hz, dt=1 ms → 100 * 1 / 1000 = 0.1
    assert rate_to_prob(100.0, 1.0) == pytest.approx(0.1)


def test_rate_to_prob_boundary_is_allowed():
    # base * dt == 1e3 is the largest allowed product → probability 1.
    assert rate_to_prob(1000.0, 1.0) == pytest.approx(1.0)
    assert rate_to_prob(0.0, 1.0) == 0.0


def test_rate_to_prob_rejects_overflow():
    with pytest.raises(ValueError, match="dt"):
        rate_to_prob(1e9, 1.0)


def test_firing_rate_alias_warns_and_forwards():
    from braintools._deprecation import BraintoolsDeprecationWarning
    with pytest.warns(BraintoolsDeprecationWarning, match="rate_to_prob"):
        assert firing_rate(100.0, 1.0) == pytest.approx(0.1)


def test_firing_rate_alias_metadata():
    from braintools._deprecation import registered_deprecations
    assert firing_rate.__name__ == 'firing_rate'
    assert firing_rate.__module__ == 'braintools.cogtask.utils'
    records = [r for r in registered_deprecations()
               if r.old_name == 'braintools.cogtask.utils.firing_rate']
    assert len(records) == 1
    assert records[0].target is rate_to_prob
    assert records[0].new_name == 'braintools.cogtask.utils.rate_to_prob'


def test_firing_rate_alias_exported_from_package():
    import braintools.cogtask as ct
    assert ct.firing_rate is firing_rate
    assert ct.rate_to_prob is rate_to_prob
    assert 'firing_rate' in ct.__all__ and 'rate_to_prob' in ct.__all__


def test_firing_rate_alias_still_validates():
    from braintools._deprecation import BraintoolsDeprecationWarning
    with pytest.warns(BraintoolsDeprecationWarning):
        with pytest.raises(ValueError, match="dt"):
            firing_rate(1e9, 1.0)


# ---------------------------------------------------------------------------
# choice helper
# ---------------------------------------------------------------------------

def test_choice_excludes_specified_index():
    import brainstate.random
    rng = brainstate.random.default_rng(0)
    for _ in range(50):
        v = int(choice(rng, 5, 2))
        assert v in {0, 1, 3, 4}, f"choice returned excluded index {v}"
