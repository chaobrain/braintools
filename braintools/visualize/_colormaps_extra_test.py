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
from matplotlib.colors import LinearSegmentedColormap

import braintools


class TestColormapsExtra(unittest.TestCase):
    """Complementary coverage tests for braintools.visualize._colormaps."""

    def setUp(self):
        # Snapshot the registered colormaps so tests that register new ones
        # (e.g. brain_colormaps) do not leak global state into other tests.
        self._cmaps_before = set(plt.colormaps)

    def tearDown(self):
        plt.close('all')
        for name in set(plt.colormaps) - self._cmaps_before:
            try:
                plt.colormaps.unregister(name)
            except (KeyError, ValueError):
                pass

    # ------------------------------------------------------------------
    # style functions
    # ------------------------------------------------------------------
    def test_publication_style_usetex_false(self):
        braintools.visualize.publication_style(fontsize=8, usetex=False)
        self.assertEqual(plt.rcParams['font.size'], 8)

    def test_colorblind_friendly_style(self):
        braintools.visualize.colorblind_friendly_style()
        cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.assertIn('#1f77b4', cycle_colors)

    # ------------------------------------------------------------------
    # create_neural_colormap
    # ------------------------------------------------------------------
    def test_create_neural_colormap(self):
        cmap = braintools.visualize.create_neural_colormap(
            'extra_test_cmap', ['#000000', '#FFFFFF'], n_bins=64
        )
        self.assertIsInstance(cmap, LinearSegmentedColormap)
        self.assertIn('extra_test_cmap', plt.colormaps)

    def test_create_neural_colormap_idempotent(self):
        # Registering the same name twice must not raise (force=True).
        braintools.visualize.create_neural_colormap('extra_dup', ['#000000', '#FFFFFF'])
        braintools.visualize.create_neural_colormap('extra_dup', ['#FFFFFF', '#000000'])
        self.assertIn('extra_dup', plt.colormaps)

    # ------------------------------------------------------------------
    # brain_colormaps
    # ------------------------------------------------------------------
    def test_brain_colormaps(self):
        braintools.visualize.brain_colormaps()
        for name in ('membrane', 'spikes', 'connectivity', 'brain_activation'):
            self.assertIn(name, plt.colormaps)

    # ------------------------------------------------------------------
    # apply_style
    # ------------------------------------------------------------------
    def test_apply_style_neural(self):
        braintools.visualize.apply_style('neural', fontsize=11)
        self.assertEqual(plt.rcParams['font.size'], 11)

    def test_apply_style_publication(self):
        braintools.visualize.apply_style('publication', fontsize=9)
        self.assertEqual(plt.rcParams['font.size'], 9)

    def test_apply_style_dark(self):
        braintools.visualize.apply_style('dark')
        self.assertEqual(plt.rcParams['figure.facecolor'], '#2E2E2E')

    def test_apply_style_colorblind(self):
        braintools.visualize.apply_style('colorblind')
        cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.assertIn('#1f77b4', cycle_colors)

    def test_apply_style_invalid(self):
        with self.assertRaises(ValueError):
            braintools.visualize.apply_style('nonexistent')

    # ------------------------------------------------------------------
    # get_color_palette
    # ------------------------------------------------------------------
    def test_get_color_palette_dark(self):
        colors = braintools.visualize.get_color_palette('dark')
        self.assertIsInstance(colors, list)
        self.assertTrue(len(colors) > 0)

    def test_get_color_palette_colorblind(self):
        colors = braintools.visualize.get_color_palette('colorblind', n_colors=4)
        self.assertEqual(len(colors), 4)

    def test_get_color_palette_repeat(self):
        # n_colors larger than the base palette -> colors get repeated
        base = braintools.visualize.get_color_palette('dark')
        colors = braintools.visualize.get_color_palette('dark', n_colors=len(base) + 3)
        self.assertEqual(len(colors), len(base) + 3)

    def test_get_color_palette_neural_subset(self):
        colors = braintools.visualize.get_color_palette('neural', n_colors=2)
        self.assertEqual(len(colors), 2)

    def test_get_color_palette_invalid(self):
        with self.assertRaises(ValueError):
            braintools.visualize.get_color_palette('bad_palette')

    # ------------------------------------------------------------------
    # set_default_colors
    # ------------------------------------------------------------------
    def test_set_default_colors(self):
        braintools.visualize.set_default_colors({'spike': '#123456'})
        neural = braintools.visualize.get_color_palette('neural')
        self.assertIn('#123456', neural)


if __name__ == '__main__':
    unittest.main()
