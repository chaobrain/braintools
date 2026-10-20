# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Package-level tests, the one exception to ``foo.py`` <-> ``foo_test.py``: they check ``py.typed`` and the whole public API.

Tests that braintools ships PEP 561 type information and that its public
API carries usable type annotations consumable by static type checkers."""

import importlib.util
import inspect
import re
import sys
import tomllib
import typing
import unittest
from pathlib import Path

import braintools

SPIKE_OPERATIONS = (
    'spike_bitwise_or',
    'spike_bitwise_and',
    'spike_bitwise_iand',
    'spike_bitwise_not',
    'spike_bitwise_xor',
    'spike_bitwise_ixor',
    'spike_bitwise',
)

ENCODERS = (
    'LatencyEncoder',
    'RateEncoder',
    'PoissonEncoder',
    'PopulationEncoder',
    'BernoulliEncoder',
    'DeltaEncoder',
    'StepCurrentEncoder',
    'SpikeCountEncoder',
    'TemporalEncoder',
    'RankOrderEncoder',
)


def _package_dir() -> Path:
    return Path(braintools.__file__).resolve().parent


def _unannotated_params(func) -> list:
    """Return the names of parameters lacking annotations (ignoring self/cls
    and variadic ``*args``/``**kwargs``)."""
    sig = inspect.signature(func)
    missing = []
    for name, param in sig.parameters.items():
        if name in ('self', 'cls'):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.annotation is inspect.Parameter.empty:
            missing.append(name)
    return missing


def _has_return_annotation(func) -> bool:
    return inspect.signature(func).return_annotation is not inspect.Signature.empty


class TestPyTypedMarker(unittest.TestCase):
    """PEP 561: a ``py.typed`` marker must ship inside the package directory."""

    def test_marker_file_present(self):
        marker = _package_dir() / 'py.typed'
        self.assertTrue(marker.is_file(), f'py.typed marker missing at {marker}')

    def test_marker_adjacent_to_init(self):
        # The marker must sit next to __init__.py so type checkers discover it.
        pkg = _package_dir()
        self.assertTrue((pkg / '__init__.py').is_file())
        self.assertTrue((pkg / 'py.typed').is_file())

    def test_pyproject_declares_marker(self):
        # Only present in a source checkout, not an installed wheel.
        pyproject = _package_dir().parent / 'pyproject.toml'
        if not pyproject.is_file():
            self.skipTest('pyproject.toml not available (installed distribution)')
        config = tomllib.loads(pyproject.read_text(encoding='utf-8'))
        package_data = config['tool']['setuptools'].get('package-data', {})
        self.assertIn('py.typed', package_data.get('braintools', []))


class TestPublicApiAnnotations(unittest.TestCase):
    """The public API surface must carry resolvable type annotations."""

    def test_all_exports_resolvable(self):
        for name in braintools.__all__:
            self.assertTrue(
                hasattr(braintools, name),
                f'braintools.__all__ advertises {name!r} but it is not importable',
            )

    def test_spike_operations_fully_annotated(self):
        for name in SPIKE_OPERATIONS:
            func = getattr(braintools, name)
            missing = _unannotated_params(func)
            self.assertEqual(missing, [], f'{name} has unannotated params: {missing}')
            self.assertTrue(_has_return_annotation(func), f'{name} lacks a return annotation')

    def test_tree_functions_fully_annotated(self):
        for name in braintools.tree.__all__:
            func = getattr(braintools.tree, name)
            missing = _unannotated_params(func)
            self.assertEqual(missing, [], f'tree.{name} has unannotated params: {missing}')
            self.assertTrue(_has_return_annotation(func), f'tree.{name} lacks a return annotation')

    def test_encoder_call_annotated(self):
        for name in ENCODERS:
            cls = getattr(braintools, name)
            call = cls.__call__
            self.assertTrue(_has_return_annotation(call), f'{name}.__call__ lacks a return annotation')
            self.assertNotIn('data', _unannotated_params(call), f'{name}.__call__ data param unannotated')

    def test_annotations_resolve_to_real_types(self):
        # get_type_hints evaluates the (possibly stringized) annotations against
        # the defining module globals; failure means a typo/undefined name.
        targets = [getattr(braintools, n) for n in SPIKE_OPERATIONS]
        targets += [getattr(braintools.tree, n) for n in braintools.tree.__all__]
        for func in targets:
            try:
                hints = typing.get_type_hints(func)
            except Exception as exc:  # pragma: no cover - surfaced as failure
                self.fail(f'Could not resolve annotations for {func.__name__}: {exc}')
            self.assertIn('return', hints, f'{func.__name__} missing resolved return type')


def _public_modules():
    """Yield ``(dotted_name, module)`` for every public module reachable through ``__all__``."""
    seen = set()
    stack = [('braintools', braintools)]
    while stack:
        name, mod = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        yield name, mod
        for attr in getattr(mod, '__all__', ()):
            obj = getattr(mod, attr, None)
            if inspect.ismodule(obj) and obj.__name__.startswith('braintools.'):
                stack.append((obj.__name__, obj))


_RST_DIRECTIVE = re.compile(
    r'^\.\. (?:py:)?(currentmodule|module|autosummary|autofunction|autoclass|autodata|data|class|function)'
    r'::\s*(\S*)'
)


def _documented_names(text: str) -> set:
    """Return the fully qualified names an API page documents.

    Entries of ``autosummary`` blocks and ``auto*``/``data``/``class``/``function``
    directives are qualified with the ``currentmodule`` in effect, so a name is
    only counted for the module whose section lists it.
    """
    current = None
    in_summary = False
    names = set()
    for line in text.splitlines():
        match = _RST_DIRECTIVE.match(line)
        if match:
            kind, arg = match.groups()
            in_summary = kind == 'autosummary'
            if kind in ('currentmodule', 'module'):
                current = arg
            elif arg:
                names.add(arg if arg.startswith('braintools.') or current is None else f'{current}.{arg}')
            continue
        if in_summary:
            entry = line.strip()
            if not entry or entry.startswith(':'):
                continue  # blank line or directive option
            if not line[:1].isspace():
                in_summary = False  # dedent ends the block
                continue
            names.add(entry if entry.startswith('braintools.') else f'{current}.{entry}')
    return names


def _undocumented_public_names(documented: set) -> list:
    """Public names (deprecated aliases excluded) missing from ``documented``."""
    from braintools._deprecation import registered_deprecations

    # Deprecated aliases are documented in docs/migration.md instead.
    deprecated = {record.old_name for record in registered_deprecations()}
    missing = []
    for mod_name, mod in _public_modules():
        for attr in getattr(mod, '__all__', ()):
            qualified = f'{mod_name}.{attr}'
            if attr.startswith('__') or inspect.ismodule(getattr(mod, attr, None)) or qualified in deprecated:
                continue
            if qualified not in documented:
                missing.append(qualified)
    return missing


class TestApiDocsCoverage(unittest.TestCase):
    """Every name a public module exports appears in the API reference (``docs/apis``)."""

    def _documented(self) -> set:
        apis = _package_dir().parent / 'docs' / 'apis'
        if not apis.is_dir():
            self.skipTest('docs/apis not available (installed distribution)')
        documented = set()
        for page in apis.glob('*.rst'):
            documented |= _documented_names(page.read_text(encoding='utf-8'))
        return documented

    def test_every_public_name_is_documented(self):
        missing = _undocumented_public_names(self._documented())
        self.assertEqual(missing, [], f'public names missing from docs/apis/*.rst: {missing}')

    def test_names_are_qualified_by_their_section(self):
        text = (
            '.. currentmodule:: braintools.init\n\n'
            '.. autosummary::\n   :toctree: generated/\n\n   Clipped\n   Normal\n\n'
            'Clipped values are described in the prose below.\n'
            '    Indented\n\n'
            '.. currentmodule:: braintools.input\n\n'
            '.. autofunction:: ramp\n'
            '.. autoclass:: braintools.conn.Ring\n'
        )
        self.assertEqual(
            _documented_names(text),
            {'braintools.init.Clipped', 'braintools.init.Normal',
             'braintools.input.ramp', 'braintools.conn.Ring'},
        )

    def test_name_listed_under_another_module_is_missing(self):
        # ``Clipped`` is exported by both ``init`` and ``input``; dropping it
        # from the input page must be detected even though init lists it.
        documented = self._documented()
        self.assertIn('braintools.input.Clipped', documented)
        self.assertIn('braintools.init.Clipped', documented)
        self.assertIn('braintools.input.Clipped', _undocumented_public_names(documented - {'braintools.input.Clipped'}))


class TestStaticTypeChecker(unittest.TestCase):
    """The whole package (tests excluded) is configured for, and clean under, mypy."""

    def _mypy_config(self) -> dict:
        with open(_package_dir().parent / 'pyproject.toml', 'rb') as f:
            return tomllib.load(f)['tool']['mypy']

    def test_mypy_configured_for_whole_package(self):
        config = self._mypy_config()
        self.assertEqual(config['files'], ['braintools'])
        exclude = re.compile(config['exclude'])
        self.assertTrue(exclude.search('braintools/conn/_base_test.py'))
        self.assertTrue(exclude.search('braintools/__init___test.py'))
        self.assertIsNone(exclude.search('braintools/conn/_base.py'))
        self.assertIsNone(exclude.search('braintools/conn/_test_utils.py'))

    def test_mypy_clean_whole_package(self):
        if importlib.util.find_spec('mypy') is None:
            self.skipTest('mypy is not installed')
        from mypy import api

        config = self._mypy_config()
        stdout, stderr, exit_status = api.run([
            str(_package_dir()),
            '--exclude', config['exclude'],
            '--config-file', str(_package_dir().parent / 'pyproject.toml'),
            '--no-error-summary',
        ])
        self.assertEqual(
            exit_status, 0,
            f'mypy reported issues:\n{stdout}\n{stderr}',
        )


if __name__ == '__main__':
    sys.exit(unittest.main())
