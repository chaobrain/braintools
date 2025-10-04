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
import braintools


class TestExistingPlots(unittest.TestCase):
    """Test enhanced existing plotting functions."""

    def setUp(self):
        """Set up test data."""
        self.time = np.linspace(0, 10, 100)
        self.values = np.random.randn(100, 3)
        self.spike_matrix = np.random.binomial(1, 0.05, (100, 10))

    def test_line_plot(self):
        """Test enhanced line plot."""
        ax = braintools.visualize.line_plot(
            self.time,
            self.values,
            plot_ids=[0, 1],
            colors=['red', 'blue'],
            alpha=0.8,
            legend='Signal'
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_line_plot_errors(self):
        """Test line plot error handling."""
        with self.assertRaises(ValueError):
            braintools.visualize.line_plot([], self.values)

        with self.assertRaises(ValueError):
            braintools.visualize.line_plot(self.time[:50], self.values)

    def test_raster_plot(self):
        """Test enhanced raster plot."""
        ax = braintools.visualize.raster_plot(
            self.time,
            self.spike_matrix,
            alpha=0.8,
            color='red'
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_raster_plot_errors(self):
        """Test raster plot error handling."""
        with self.assertRaises(ValueError):
            braintools.visualize.raster_plot(None, self.spike_matrix)

        with self.assertRaises(ValueError):
            braintools.visualize.raster_plot(self.time, [])
