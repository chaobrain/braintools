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
import braintools


class TestInteractiveVisualization(unittest.TestCase):
    """Test interactive visualization functions."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.spike_times = [np.sort(np.random.uniform(0, 1, 10)) for _ in range(5)]
        self.data = np.random.randn(100, 3)

    def test_interactive_spike_raster(self):
        """Test interactive spike raster."""
        try:
            fig = braintools.visualize.interactive_spike_raster(
                self.spike_times,
                color_by='neuron'
            )
            self.assertIsNotNone(fig)
        except ImportError:
            self.skipTest("Plotly not available")

    def test_interactive_line_plot(self):
        """Test interactive line plot."""
        try:
            x = np.linspace(0, 10, 100)
            y = [np.sin(x), np.cos(x)]
            fig = braintools.visualize.interactive_line_plot(
                x, y,
                labels=['sin', 'cos']
            )
            self.assertIsNotNone(fig)
        except ImportError:
            self.skipTest("Plotly not available")
