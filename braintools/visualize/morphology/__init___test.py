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

"""Architectural guards for :mod:`braintools.visualize.morphology`.

**No simulator import.** The package draws neutral data
(:class:`MorphologyData`, :class:`TopologyGraph`) so it never depends on
the simulator whose objects it draws; braincell depends on braintools, so
any ``import braincell`` here would also close a cycle. The AST scan
covers function bodies too: a deferred import is still a dependency.

**Lazy loading.** ``import braintools`` must not load this package nor any
of its optional heavy dependencies, and importing the package itself must
not load networkx or PyVista; they are imported where they are used.

**Exported surface.** ``__all__`` is the module's contract: every name in
it is importable, the list is sorted, and nothing public is left out.
"""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

import braintools.visualize.morphology

_PACKAGE_DIR = Path(braintools.visualize.morphology.__file__).resolve().parent


def _probe(code: str) -> list[str]:
    """Run ``code`` in a fresh interpreter and return the modules it lists."""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return result.stdout.split()


_LOADED = (
    "import sys; {imports}; "
    "print(' '.join(m for m in ('braintools.visualize.morphology', 'networkx', 'pyvista', 'plotly', 'braincell') "
    "if m in sys.modules))"
)


class NoSimulatorImportTest(unittest.TestCase):
    def test_no_module_imports_braincell(self) -> None:
        offenders = []
        for path in sorted(_PACKAGE_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif (
                    isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                ):
                    names = [node.module]
                else:
                    continue
                if any(
                    name == "braincell" or name.startswith("braincell.")
                    for name in names
                ):
                    offenders.append(f"{path.relative_to(_PACKAGE_DIR)}:{node.lineno}")
        self.assertEqual(offenders, [])


class LazyLoadingTest(unittest.TestCase):
    def test_import_braintools_loads_neither_this_package_nor_heavy_dependencies(
        self,
    ) -> None:
        self.assertEqual(_probe(_LOADED.format(imports="import braintools")), [])

    def test_importing_the_package_does_not_load_networkx_or_pyvista(self) -> None:
        loaded = _probe(
            _LOADED.format(imports="import braintools.visualize.morphology")
        )
        self.assertIn("braintools.visualize.morphology", loaded)
        for name in ("networkx", "pyvista", "braincell"):
            self.assertNotIn(name, loaded)


class PublicSurfaceTest(unittest.TestCase):
    """``braintools.visualize.morphology.__all__`` is the module's contract."""

    def test_every_listed_name_is_importable(self) -> None:
        for name in braintools.visualize.morphology.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(braintools.visualize.morphology, name))

    def test_all_is_sorted_and_unique(self) -> None:
        names = list(braintools.visualize.morphology.__all__)
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), len(set(names)))

    def test_no_public_callable_is_exported_without_being_listed(self) -> None:
        exported = set(braintools.visualize.morphology.__all__)
        submodules = {path.stem for path in _PACKAGE_DIR.glob("*.py")} | {
            path.name for path in _PACKAGE_DIR.iterdir() if path.is_dir()
        }
        unlisted = [
            name
            for name in dir(braintools.visualize.morphology)
            if not name.startswith("_")
            and name not in exported
            and name not in submodules
        ]
        self.assertEqual(unlisted, [], f"public but absent from __all__: {unlisted}")

    def test_the_surface_has_the_planned_size(self) -> None:
        # 27 names kept from the plots this package was moved from, the two
        # result types, five data types and two normalisers (VIS-3).
        self.assertEqual(len(braintools.visualize.morphology.__all__), 36)


if __name__ == "__main__":
    unittest.main()
