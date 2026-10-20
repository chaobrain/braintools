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
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import braintools
from braintools.visualize import apply_style


class TestColormaps(unittest.TestCase):
    """Test colormap and styling functions."""

    def test_neural_style(self):
        """Test neural style application."""
        braintools.visualize.neural_style(fontsize=12)
        # Test that style is applied
        self.assertEqual(plt.rcParams['font.size'], 12)

    def test_publication_style(self):
        """Test publication style."""
        braintools.visualize.publication_style(fontsize=10, dpi=150)
        self.assertEqual(plt.rcParams['font.size'], 10)
        self.assertEqual(plt.rcParams['figure.dpi'], 150)

    def test_dark_style(self):
        """Test dark style."""
        braintools.visualize.dark_style()
        # Check that background is dark
        self.assertEqual(plt.rcParams['figure.facecolor'], '#2E2E2E')

    def test_color_palettes(self):
        """Test color palette functions."""
        neural_colors = braintools.visualize.get_color_palette('neural')
        self.assertIsInstance(neural_colors, list)
        self.assertTrue(len(neural_colors) > 0)

        colorblind_colors = braintools.visualize.get_color_palette('colorblind', n_colors=5)
        self.assertEqual(len(colorblind_colors), 5)

    def test_brain_colormaps(self):
        """Test brain-specific colormap creation."""
        braintools.visualize.brain_colormaps()
        # Check that custom colormaps are registered
        self.assertIn('membrane', plt.colormaps)


class TestApplyStyleContextManager(unittest.TestCase):
    """B4: apply_style works both as a plain call and as a context manager."""

    def tearDown(self):
        plt.rcdefaults()
        plt.close('all')

    def test_context_manager_restores_rcparams(self):
        plt.rcdefaults()
        base_grid = plt.rcParams['axes.grid']
        with apply_style('publication'):
            pass
        self.assertEqual(plt.rcParams['axes.grid'], base_grid)

    def test_plain_call_applies_and_persists(self):
        plt.rcdefaults()
        apply_style('dark')
        # dark_style sets a dark figure facecolor; it must persist (no restore)
        self.assertEqual(plt.rcParams['figure.facecolor'], '#2E2E2E')

    def test_invalid_style_raises(self):
        with self.assertRaises(ValueError):
            apply_style('does-not-exist')


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
        self.assertIn('#E69F00', cycle_colors)  # Okabe-Ito orange

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
        self.assertIn('#E69F00', cycle_colors)  # Okabe-Ito orange

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


def test_colorblind_palette_is_okabe_ito():
    import braintools
    assert braintools.visualize.get_color_palette('colorblind')[:3] == ['#E69F00', '#56B4E9', '#009E73']


def test_dark_style_uses_accent_color():
    import braintools
    with braintools.visualize.apply_style('dark', accent_color='#123456'):
        assert plt.rcParams['axes.prop_cycle'].by_key()['color'][0] == '#123456'


class TestIntegration(unittest.TestCase):
    """Integration tests for the visualization module."""

    def test_style_integration(self):
        """Test that styles work with plotting functions."""
        # Apply neural style
        braintools.visualize.neural_style()

        # Generate data
        time = np.linspace(0, 1, 100)
        values = np.random.randn(100, 2)

        # Create plot
        ax = braintools.visualize.line_plot(time, values, legend='Test')
        self.assertIsNotNone(ax)
        plt.close()

        # Reset to default
        plt.rcdefaults()
