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
import braintools


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
