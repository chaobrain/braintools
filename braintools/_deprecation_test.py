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

import builtins
import importlib
import inspect
import os
import pathlib
import pickle
import re
import subprocess
import sys
import textwrap
import tomllib
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

    def test_is_future_warning(self):
        with pytest.warns(FutureWarning):
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


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestWarningCategory:
    """``BraintoolsDeprecationWarning`` is a ``FutureWarning``, shown by default."""

    def test_is_future_warning_not_deprecation_warning(self):
        assert issubclass(BraintoolsDeprecationWarning, FutureWarning)
        assert not issubclass(BraintoolsDeprecationWarning, DeprecationWarning)

    def test_shown_under_default_filters_from_library_code(self, tmp_path):
        # A module other than ``__main__`` calls a deprecated alias. Under
        # Python's default filters a DeprecationWarning raised there is hidden,
        # while a FutureWarning is shown.
        (tmp_path / 'wpd_libmod.py').write_text(textwrap.dedent(
            """
            import warnings
            from braintools._deprecation import deprecated_function

            plus = deprecated_function(
                lambda a, b: a + b, 'braintools.wpd_fake.plus', 'braintools.wpd_fake.add')

            def call_alias():
                return plus(1, 2)

            def emit_plain_deprecation():
                warnings.warn('plain-deprecation-control', DeprecationWarning, stacklevel=1)
            """
        ))
        main = (
            'import wpd_libmod\n'
            'wpd_libmod.emit_plain_deprecation()\n'
            'assert wpd_libmod.call_alias() == 3\n'
        )
        env = {k: v for k, v in os.environ.items() if k != 'PYTHONWARNINGS'}
        env['PYTHONPATH'] = os.pathsep.join([str(tmp_path), str(_REPO_ROOT)])
        env['JAX_PLATFORMS'] = 'cpu'
        env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        proc = subprocess.run(
            [sys.executable, '-c', main], env=env, cwd=tmp_path,
            capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == 0, proc.stderr
        assert 'BraintoolsDeprecationWarning' in proc.stderr
        assert '`braintools.wpd_fake.plus` is deprecated since 0.4.0' in proc.stderr
        assert 'wpd_libmod.py' in proc.stderr
        # Control: the same interpreter hides a plain DeprecationWarning raised
        # outside ``__main__``, so the filters really were Python's defaults.
        assert 'plain-deprecation-control' not in proc.stderr

    def test_deprecation_warning_error_filter_does_not_catch_it(self):
        with pytest.warns(BraintoolsDeprecationWarning):
            with warnings.catch_warnings():
                warnings.simplefilter('error', DeprecationWarning)
                assert plus(1, 2) == 3

    def test_deprecation_warning_ignore_filter_does_not_silence_it(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            warnings.simplefilter('ignore', DeprecationWarning)
            plus(1, 2)
        assert [w.category for w in caught] == [BraintoolsDeprecationWarning]

    def test_ignore_filter_on_class_silences_it(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            warnings.filterwarnings('ignore', category=BraintoolsDeprecationWarning)
            assert plus(1, 2) == 3
            OldPoint(1)
        assert caught == []

    def test_future_warning_ignore_filter_silences_it(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            warnings.simplefilter('ignore', FutureWarning)
            plus(1, 2)
        assert caught == []

    def test_class_alias_pickles_by_reference(self):
        assert pickle.loads(pickle.dumps(OldPoint)) is OldPoint
        with pytest.warns(BraintoolsDeprecationWarning):
            p = OldPoint(2, y=3)
        q = pickle.loads(pickle.dumps(p))
        assert (q.x, q.y) == (2, 3)


def _pytest_error_filters():
    """Return the ``error`` entries of the pytest ``filterwarnings`` option.

    Each entry is a ``(message regex, warning category)`` pair.
    """
    pyproject = _REPO_ROOT / 'pyproject.toml'
    if not pyproject.is_file():
        pytest.skip('pyproject.toml not found (running from an installed package)')
    with pyproject.open('rb') as f:
        entries = tomllib.load(f)['tool']['pytest']['ini_options']['filterwarnings']
    filters = []
    for entry in entries:
        action, message, category = entry.split(':')[:3]
        if action == 'error':
            filters.append((message, getattr(builtins, category)))
    return filters


class TestPytestFilter:
    """The test suite turns every registered braintools deprecation into an error."""

    def test_every_library_message_is_covered(self):
        filters = _pytest_error_filters()
        uncovered = [
            r.old_name for r in _library_records()
            if not any(
                re.match(message, r.message, re.IGNORECASE)
                and issubclass(BraintoolsDeprecationWarning, category)
                for message, category in filters
            )
        ]
        assert not uncovered, f'Deprecations not turned into errors by pyproject.toml: {uncovered}'

    def test_filter_targets_future_warning(self):
        assert any(category is FutureWarning for _, category in _pytest_error_filters())

    def test_registry_warning_is_an_error_outside_pytest_warns(self):
        # The pytest filter is active here: an unexpected deprecation fails the test.
        with pytest.raises(BraintoolsDeprecationWarning):
            plus(1, 2)


def _shipped_records():
    # Library records without the aliases this test module registers itself.
    return [r for r in _library_records() if not r.old_name.startswith(f'{__name__}.')]


def _migration_guide():
    docs = _REPO_ROOT / 'docs'
    if not docs.is_dir():
        pytest.skip('docs/ not found (running from an installed package)')
    return (docs / 'migration.md').read_text(encoding='utf-8')


def _guide_row(record):
    """The migration-guide table row of ``record``."""
    return (f'| `{record.old_name}` | `{record.new_name}` | {record.kind} '
            f'| {record.since} | {record.removed_in} |')


def _missing_from_guide(records, text):
    """Return the records without a matching table row in ``text``."""
    rows = {line.strip() for line in text.splitlines()}
    return [r.old_name for r in records if _guide_row(r) not in rows]


class TestMigrationGuide:
    """Every registered deprecation is listed in ``docs/migration.md``."""

    def test_library_has_deprecations(self):
        assert len(_shipped_records()) > 0

    def test_every_deprecation_in_guide(self):
        missing = _missing_from_guide(_shipped_records(), _migration_guide())
        assert not missing, f'Add these deprecations to docs/migration.md: {missing}'

    def test_missing_entry_is_detected(self):
        record = DeprecationRecord('braintools.not_in_guide', 'braintools.new', '0.4.0', '1.0.0', 'function')
        assert _missing_from_guide([record], _migration_guide()) == ['braintools.not_in_guide']
        text = '| `braintools.not_in_guide` | `braintools.other` | function | 0.4.0 | 1.0.0 |'
        assert _missing_from_guide([record], text) == ['braintools.not_in_guide']
        text = '| `braintools.not_in_guide` | `braintools.new` | function | 0.4.0 | 1.0.0 |'
        assert _missing_from_guide([record], text) == []

    def test_mention_outside_the_table_is_not_an_entry(self):
        # Only a full table row counts: a name quoted in prose or in a sample
        # warning, or a row whose details differ from the registry, is missing.
        record = DeprecationRecord('braintools.old', 'braintools.new', '0.4.0', '1.0.0', 'function')
        prose = 'Calling `braintools.old` warns; use `braintools.new` instead.\n'
        row = '| `braintools.old` | `braintools.new` | function | 0.4.0 | 1.0.0 |\n'
        wrong_since = '| `braintools.old` | `braintools.new` | function | 0.3.0 | 1.0.0 |\n'
        assert _missing_from_guide([record], prose) == ['braintools.old']
        assert _missing_from_guide([record], wrong_since) == ['braintools.old']
        assert _missing_from_guide([record], prose + row) == []

    def test_guide_documents_how_to_surface_warnings(self):
        text = _migration_guide()
        assert '-W error::FutureWarning' in text
        assert 'BraintoolsDeprecationWarning' in text
