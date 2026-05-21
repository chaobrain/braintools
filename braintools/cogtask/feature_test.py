# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``braintools.cogtask.feature``.

These check that:
- composition is immutable (``+`` and ``-`` don't mutate the operands),
- indices are rebased correctly so that consecutive features carve out
  contiguous, non-overlapping slices,
- duplicate names, unknown names, and removing the last feature all raise.
"""

import jax.numpy as jnp
import pytest

from braintools.cogtask.feature import (
    Feature, FeatureSet, CircleFeature, is_feature, as_feature,
)


def test_feature_initial_slice_starts_at_zero():
    f = Feature(5, 'x')
    assert f.num == 5
    assert f.i == slice(0, 5)


def test_is_feature_predicate():
    assert is_feature(Feature(1, 'a')) is True
    assert is_feature("nope") is False
    assert is_feature(None) is False


def test_as_feature_validator():
    f = Feature(2, 'a')
    assert as_feature(f) is f
    with pytest.raises(TypeError):
        as_feature("not a feature")


def test_addition_creates_feature_set_with_rebased_indices():
    a = Feature(1, 'a')
    b = Feature(8, 'b')
    fs = a + b
    assert isinstance(fs, FeatureSet)
    assert fs['a'].i == slice(0, 1)
    assert fs['b'].i == slice(1, 9)
    assert fs.num == 9


def test_addition_is_immutable():
    a = Feature(1, 'a')
    b = Feature(8, 'b')
    _ = a + b
    # Originals must keep their pre-composition slices.
    assert a.i == slice(0, 1)
    assert b.i == slice(0, 8)


def test_feature_set_plus_feature_appends_with_offset():
    fs = Feature(1, 'a') + Feature(2, 'b')
    fs2 = fs + Feature(3, 'c')
    assert fs2['a'].i == slice(0, 1)
    assert fs2['b'].i == slice(1, 3)
    assert fs2['c'].i == slice(3, 6)
    assert fs2.num == 6


def test_feature_set_plus_feature_set_concatenates():
    left = Feature(1, 'a') + Feature(2, 'b')
    right = Feature(3, 'c') + Feature(1, 'd')
    combined = left + right
    assert combined.num == 7
    assert combined['c'].i == slice(3, 6)
    assert combined['d'].i == slice(6, 7)
    # Original sets must remain untouched.
    assert right['c'].i == slice(0, 3)
    assert right['d'].i == slice(3, 4)


def test_pipe_operator_aliases_plus():
    a = Feature(1, 'a')
    b = Feature(2, 'b')
    via_plus = (a + b).num
    via_pipe = (a | b).num
    assert via_plus == via_pipe == 3


def test_duplicate_feature_names_raise():
    a = Feature(1, 'x')
    b = Feature(2, 'x')
    with pytest.raises(ValueError, match="Duplicate"):
        a + b


def test_subtraction_by_name_drops_correct_feature():
    fs = Feature(1, 'a') + Feature(2, 'b') + Feature(3, 'c')
    fs2 = fs - 'b'
    assert [f.name for f in fs2] == ['a', 'c']
    assert fs2.num == 4  # sum of remaining feature dims
    # The first feature's slice is preserved.
    assert fs2['a'].i == slice(0, 1)


def test_subtraction_by_feature_object_yields_lone_feature_when_one_left():
    # When only one feature remains, ``__sub__`` returns the bare Feature
    # rather than a 1-element FeatureSet (because the re-composition loop
    # runs zero iterations).
    a = Feature(1, 'a')
    b = Feature(2, 'b')
    fs = a + b
    fs2 = fs - b
    assert isinstance(fs2, Feature)
    assert fs2.name == 'a'


def test_subtraction_unknown_name_raises_keyerror():
    fs = Feature(1, 'a') + Feature(2, 'b')
    with pytest.raises(KeyError):
        fs - 'missing'


def test_subtraction_last_feature_raises_value_error():
    fs = Feature(1, 'a') + Feature(2, 'b') + Feature(3, 'c')
    fs2 = fs - 'a'
    fs3 = fs2 - 'b'
    # Now fs3 is just a Feature; we need to test the "remove last" rule on a
    # 1-element FeatureSet directly.
    one = FeatureSet(Feature(1, 'only'))
    with pytest.raises(ValueError, match="last feature"):
        one - 'only'


def test_subtraction_bad_type_raises_type_error():
    fs = Feature(1, 'a') + Feature(2, 'b')
    with pytest.raises(TypeError):
        fs - 42


def test_feature_repetition_indexes_names_and_offsets():
    fs = Feature(2, 'choice') * 3
    names = [f.name for f in fs]
    assert names == ['choice_0', 'choice_1', 'choice_2']
    assert fs.num == 6
    assert fs['choice_0'].i == slice(0, 2)
    assert fs['choice_2'].i == slice(4, 6)


def test_feature_repetition_rejects_non_positive_count():
    with pytest.raises(ValueError):
        Feature(1, 'x') * 0
    with pytest.raises(ValueError):
        Feature(1, 'x') * -1


def test_feature_addition_rejects_unsupported_type():
    with pytest.raises(TypeError):
        Feature(1, 'a') + 42  # type: ignore


def test_feature_set_addition_rejects_unsupported_type():
    fs = Feature(1, 'a') + Feature(2, 'b')
    with pytest.raises(TypeError):
        fs + 42  # type: ignore


def test_feature_set_getitem_by_index_and_slice():
    fs = Feature(1, 'a') + Feature(2, 'b') + Feature(3, 'c')
    assert fs[0].name == 'a'
    sliced = fs[1:]
    assert isinstance(sliced, tuple)
    assert [f.name for f in sliced] == ['b', 'c']


def test_feature_set_contains_and_len_and_iter():
    fs = Feature(1, 'a') + Feature(2, 'b')
    assert 'a' in fs
    assert 'missing' not in fs
    assert len(fs) == 2
    assert [f.name for f in fs] == ['a', 'b']


def test_feature_set_copy_does_not_alias_features():
    fs = Feature(1, 'a') + Feature(2, 'b')
    fs_copy = fs.copy()
    fs_copy['a'].shift(100)
    assert fs['a'].i == slice(0, 1)


def test_feature_repr_includes_name_and_num():
    f = Feature(3, 'choice')
    r = repr(f)
    assert 'choice' in r
    assert '3' in r


def test_feature_set_repr_lists_names_and_nums():
    fs = Feature(1, 'a') + Feature(2, 'b')
    r = repr(fs)
    assert "'a'" in r and "'b'" in r
    assert '1' in r and '2' in r


def test_circle_feature_stores_limits_and_range():
    cf = CircleFeature(8, limits=(0.0, jnp.pi), name='angle')
    assert cf.num == 8
    assert float(cf.v_min) == 0.0
    assert float(cf.v_max) == float(jnp.pi)
    assert float(cf.v_range) == float(jnp.pi)


def test_circle_feature_default_limits_are_zero_to_two_pi():
    cf = CircleFeature(4, name='dir')
    assert float(cf.v_min) == 0.0
    assert float(cf.v_max) == pytest.approx(float(2 * jnp.pi))
