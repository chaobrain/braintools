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

import importlib
import sys
import types
import unittest

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

import braintools.visualize._figures as figures_mod
import braintools.visualize._style as style_mod
from braintools.visualize import get_figure


class TestGetFigure(unittest.TestCase):
    """Cover braintools.visualize._figures.get_figure."""

    def tearDown(self):
        plt.close('all')

    def test_returns_figure_and_gridspec(self):
        fig, gs = get_figure(1, 1)
        self.assertIsNotNone(fig)
        self.assertEqual(gs.get_geometry(), (1, 1))

    def test_custom_dimensions(self):
        fig, gs = get_figure(2, 3, row_len=2, col_len=4)
        self.assertEqual(gs.get_geometry(), (2, 3))
        # figure size = (col_num * col_len, row_num * row_len)
        w, h = fig.get_size_inches()
        self.assertAlmostEqual(w, 3 * 4)
        self.assertAlmostEqual(h, 2 * 2)

    def test_module_alias(self):
        # get_figure is exported from the _figures module too
        self.assertIs(get_figure, figures_mod.get_figure)


class TestStyleModule(unittest.TestCase):
    """Cover braintools.visualize._style.

    ``scienceplots`` is not installed in this environment, so the module's
    ``try`` block normally fails immediately. We inject a stub ``scienceplots``
    module and a ``notebook`` style entry so the success path (defining
    ``exclude``, registering ``notebook2``, defining ``plot_style1``) runs, then
    separately exercise the failure (``except``) path.
    """

    def setUp(self):
        # Snapshot global state we are about to mutate.
        self._had_scienceplots = 'scienceplots' in sys.modules
        self._saved_scienceplots = sys.modules.get('scienceplots')
        self._had_notebook = 'notebook' in plt.style.library
        self._saved_rc_params = style_mod.rcParams

    def tearDown(self):
        # Restore scienceplots entry.
        if self._had_scienceplots:
            sys.modules['scienceplots'] = self._saved_scienceplots
        else:
            sys.modules.pop('scienceplots', None)
        # Restore the original rcParams reference on the module.
        style_mod.rcParams = self._saved_rc_params
        # Reload the module a final time without the stub so the repository's
        # imported module object is left in its natural (except-path) state.
        sys.modules.pop('scienceplots', None)
        importlib.reload(style_mod)
        plt.close('all')

    def _inject_notebook_style(self):
        if 'notebook' not in plt.style.library:
            base = plt.style.library.get('seaborn-v0_8-notebook', dict(plt.rcParams))
            plt.style.core.update_nested_dict(plt.style.library, {'notebook': dict(base)})

    def test_success_path_defines_helpers(self):
        sys.modules['scienceplots'] = types.ModuleType('scienceplots')
        self._inject_notebook_style()
        importlib.reload(style_mod)
        self.assertTrue(hasattr(style_mod, 'exclude'))
        self.assertTrue(hasattr(style_mod, 'plot_style1'))

    def test_exclude_filters_keys(self):
        from matplotlib import RcParams
        sys.modules['scienceplots'] = types.ModuleType('scienceplots')
        self._inject_notebook_style()
        importlib.reload(style_mod)
        rc = RcParams()
        rc._set('font.size', 10)
        rc._set('axes.titlesize', 12)
        rc._set('lines.linewidth', 1)
        filtered = style_mod.exclude(rc, ['size', 'width'])
        self.assertIsInstance(filtered, RcParams)
        # keys containing the excluded substrings are dropped
        self.assertNotIn('font.size', filtered.keys())
        self.assertNotIn('axes.titlesize', filtered.keys())

    def test_plot_style1_updates_params(self):
        sys.modules['scienceplots'] = types.ModuleType('scienceplots')
        self._inject_notebook_style()
        importlib.reload(style_mod)
        # Swap in a plain dict so the list-valued latex preamble assignment and
        # the final update() call run without modern matplotlib validation.
        style_mod.rcParams = dict(plt.rcParams)
        style_mod.plot_style1(fontsize=20, axes_edgecolor='red',
                              figsize='5,4', lw=2)
        self.assertTrue(style_mod.rcParams['text.usetex'])
        self.assertEqual(style_mod.rcParams['axes.edgecolor'], 'red')
        self.assertEqual(style_mod.rcParams['lines.linewidth'], 2)

    def test_failure_path_when_scienceplots_missing(self):
        # Without the stub, the import fails and the except branch swallows it.
        sys.modules.pop('scienceplots', None)
        importlib.reload(style_mod)
        # Module still imports successfully (except: pass).
        self.assertIsNotNone(style_mod)


if __name__ == '__main__':
    unittest.main()
