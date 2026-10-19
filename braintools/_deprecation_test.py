# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import importlib
import inspect
import pickle
import warnings

import jax
import jax.numpy as jnp
import pytest

import braintools
from braintools._deprecation import (
    BraintoolsDeprecationWarning,
    DeprecationRecord,
    _version_tuple,
    deprecated_class,
    deprecated_function,
    register_deprecation,
    registered_deprecations,
    warn_deprecated,
)


def _add(a, b):
    """Add two numbers."""
    return a + b


class _Point:
    """A point."""

    def __init__(self, x, y=0):
        self.x = x
        self.y = y


# Module-level aliases so that their registered names resolve through import.
plus = deprecated_function(_add, f'{__name__}.plus', f'{__name__}._add')
OldPoint = deprecated_class(_Point, f'{__name__}.OldPoint', f'{__name__}._Point')


def _resolve(dotted: str):
    module_name, _, attr = dotted.rpartition('.')
    return getattr(importlib.import_module(module_name), attr)


class TestDeprecatedFunction:
    def test_warns_on_call_and_forwards(self):
        with pytest.warns(BraintoolsDeprecationWarning, match=r'`.*\.plus` is deprecated since 0\.4\.0'):
            assert plus(1, 2) == 3

    def test_no_warning_on_access(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            fn = _resolve(f'{__name__}.plus')
        assert fn is plus

    def test_metadata(self):
        assert plus.__name__ == 'plus'
        assert plus.__qualname__ == 'plus'
        assert plus.__module__ == __name__
        assert plus.__wrapped__ is _add
        assert plus.__doc__ == _add.__doc__
        assert 'removed in 1.0.0' in plus.__deprecated__

    def test_keyword_arguments_forwarded(self):
        with pytest.warns(BraintoolsDeprecationWarning):
            assert plus(a=2, b=5) == 7

    def test_is_subclass_of_deprecation_warning(self):
        with pytest.warns(DeprecationWarning):
            plus(0, 0)

    def test_warning_attributed_to_caller(self):
        with pytest.warns(BraintoolsDeprecationWarning) as record:
            plus(1, 1)
        assert record[0].filename == __file__

    def test_pickle_roundtrip(self):
        assert pickle.loads(pickle.dumps(plus)) is plus

    def test_warns_at_trace_time_under_jit(self):
        @jax.jit
        def f(x):
            return plus(x, 1.0)

        with pytest.warns(BraintoolsDeprecationWarning):
            out = f(jnp.asarray(1.0))
        assert float(out) == 2.0


class TestDeprecatedClass:
    def test_warns_on_instantiation(self):
        with pytest.warns(BraintoolsDeprecationWarning, match='OldPoint'):
            p = OldPoint(3, y=4)
        assert (p.x, p.y) == (3, 4)

    def test_isinstance_of_target(self):
        with pytest.warns(BraintoolsDeprecationWarning):
            p = OldPoint(1)
        assert isinstance(p, _Point)
        assert issubclass(OldPoint, _Point)

    def test_target_instance_is_instance_of_alias(self):
        # Before the move both names were the same class; keep isinstance
        # checks against the old name working for objects built the new way.
        p = _Point(1)
        assert isinstance(p, OldPoint)
        assert issubclass(_Point, OldPoint)
        assert not isinstance(object(), OldPoint)

    def test_signature_matches_target(self):
        assert str(inspect.signature(OldPoint)) == str(inspect.signature(_Point))

    def test_target_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _Point(1)

    def test_metadata(self):
        assert OldPoint.__name__ == 'OldPoint'
        assert OldPoint.__qualname__ == 'OldPoint'
        assert OldPoint.__module__ == __name__
        assert OldPoint.__doc__ == _Point.__doc__
        assert 'Use `' in OldPoint.__deprecated__

    def test_warning_attributed_to_caller(self):
        with pytest.warns(BraintoolsDeprecationWarning) as record:
            OldPoint(1)
        assert record[0].filename == __file__


class TestRegistry:
    def test_register_and_warn(self):
        record = register_deprecation(
            f'{__name__}.some_argument', 'something else',
            since='0.4.0', removed_in='1.0.0', kind='argument',
        )
        assert record in registered_deprecations()
        with pytest.warns(BraintoolsDeprecationWarning, match='some_argument') as caught:
            warn_deprecated(record, stacklevel=1)
        assert caught[0].filename == __file__

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match='Unknown deprecation kind'):
            register_deprecation('a.b', 'c', since='0.4.0', removed_in='1.0.0', kind='module')

    @pytest.mark.parametrize('removed_in', ['0.4.0', '0.3.9'])
    def test_removal_must_follow_since(self, removed_in):
        with pytest.raises(ValueError, match='must be later'):
            register_deprecation('a.b', 'c', since='0.4.0', removed_in=removed_in, kind='feature')

    def test_record_message(self):
        record = DeprecationRecord('pkg.old', 'pkg.new', '0.4.0', '1.0.0', 'function')
        assert record.message == (
            '`pkg.old` is deprecated since 0.4.0 and will be removed in 1.0.0. Use `pkg.new` instead.'
        )

    @pytest.mark.parametrize(
        'text, expected',
        [('0.4.0', (0, 4, 0)), ('1.0', (1, 0)), ('0.4.0rc1', (0, 4, 0)), ('2.1.dev3', (2, 1))],
    )
    def test_version_tuple(self, text, expected):
        assert _version_tuple(text) == expected


def _library_records():
    # Importing braintools imports every subpackage, which registers every alias.
    return [r for r in registered_deprecations() if r.old_name.startswith('braintools.')]


class TestLibraryDeprecations:
    """Wiring and removal-schedule checks for every deprecation in braintools."""

    def test_aliases_are_exported_under_old_name(self):
        for record in _library_records():
            if record.kind in ('function', 'class'):
                assert _resolve(record.old_name) is record.alias, record.old_name

    def test_aliases_forward_to_target(self):
        for record in _library_records():
            if record.kind == 'function':
                assert record.alias.__wrapped__ is record.target, record.old_name
            elif record.kind == 'class':
                assert issubclass(record.alias, record.target), record.old_name

    def test_not_past_removal_version(self):
        current = _version_tuple(braintools.__version__)
        overdue = [
            r.old_name for r in _library_records()
            if _version_tuple(r.removed_in) <= current
        ]
        assert not overdue, f'Deprecated APIs past their removal version must be removed: {overdue}'
