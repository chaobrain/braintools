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

import unittest

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

import braintools.visualize._figures as figures_mod
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


if __name__ == '__main__':
    unittest.main()
