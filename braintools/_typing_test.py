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

"""Tests that braintools ships PEP 561 type information and that its public
API carries usable type annotations consumable by static type checkers."""

import importlib.util
import inspect
import sys
import tomllib
import typing
import unittest
from pathlib import Path

import braintools

# Top-level public files whose annotations are actively maintained and are
# verified to be clean under mypy (see ``test_mypy_clean_public_surface``).
TYPED_PUBLIC_MODULES = (
    '__init__.py',
    '_misc.py',
    '_spike_operation.py',
    '_spike_encoder.py',
    'tree.py',
)

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


class TestStaticTypeChecker(unittest.TestCase):
    """Run mypy on the maintained public surface when it is installed."""

    def test_mypy_clean_public_surface(self):
        if importlib.util.find_spec('mypy') is None:
            self.skipTest('mypy is not installed')
        from mypy import api

        pkg = _package_dir()
        files = [str(pkg / name) for name in TYPED_PUBLIC_MODULES]
        stdout, stderr, exit_status = api.run(
            files + ['--follow-imports=skip', '--ignore-missing-imports', '--no-error-summary']
        )
        self.assertEqual(
            exit_status, 0,
            f'mypy reported issues on the public surface:\n{stdout}\n{stderr}',
        )


if __name__ == '__main__':
    sys.exit(unittest.main())
