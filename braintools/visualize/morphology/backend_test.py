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

"""Tests for :mod:`braintools.visualize.morphology.backend` — scene-kind dispatch guards."""

import unittest
from unittest import mock

from braintools.visualize.morphology import plot2d, plot3d
from braintools.visualize.morphology._testing import (
    VisDefaultsResetMixin,
    make_node_tree,
)
from braintools.visualize.morphology.backend_pyvista import PyVistaBackend


class BackendSceneKindTest(VisDefaultsResetMixin, unittest.TestCase):
    def test_plot2d_rejects_pyvista_backend(self) -> None:
        tree = make_node_tree()

        # Force PyVista to report as available so the dispatch reaches the
        # scene-kind validation step even when pyvista isn't installed.
        with mock.patch.object(PyVistaBackend, "available", return_value=True):
            with self.assertRaisesRegex(ValueError, "only supports 3D scenes"):
                plot2d(tree, backend="pyvista")

    def test_plot3d_rejects_matplotlib_backend(self) -> None:
        tree = make_node_tree()

        with self.assertRaisesRegex(ValueError, "only supports 2D scenes"):
            plot3d(tree, backend="matplotlib")


class MissingOptionalBackendTest(VisDefaultsResetMixin, unittest.TestCase):
    """A missing optional backend names the extra that installs it."""

    def test_requested_backends_report_the_install_hint(self) -> None:
        from braintools.visualize.morphology.backend_plotly import PlotlyBackend

        tree = make_node_tree()
        for name, backend in (("pyvista", PyVistaBackend), ("plotly", PlotlyBackend)):
            with self.subTest(backend=name):
                with mock.patch.object(backend, "available", return_value=False):
                    with self.assertRaisesRegex(
                        RuntimeError, r"pip install braintools\[morphology\]"
                    ):
                        plot3d(tree, backend=name)

    def test_default_3d_backend_reports_the_install_hint(self) -> None:
        from braintools.visualize.morphology.backend_plotly import PlotlyBackend

        with mock.patch.object(PyVistaBackend, "available", return_value=False):
            with mock.patch.object(PlotlyBackend, "available", return_value=False):
                with self.assertRaisesRegex(
                    RuntimeError, r"No 3D .*pip install braintools\[morphology\]"
                ):
                    plot3d(make_node_tree())

    def test_3d_movie_without_pyvista(self) -> None:
        import numpy as np

        from braintools.visualize.morphology import plot_movie

        with mock.patch.dict("sys.modules", {"pyvista": None}):
            with self.assertRaisesRegex(
                ImportError, r"pip install braintools\[morphology\]"
            ):
                plot_movie(make_node_tree(), np.zeros((2, 1)), dimensionality="3d")


if __name__ == "__main__":
    unittest.main()
