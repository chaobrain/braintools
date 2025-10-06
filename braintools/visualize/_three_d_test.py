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


class TestThreeDVisualization(unittest.TestCase):
    """Test 3D visualization functions."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.positions = np.random.randn(10, 3)
        self.trajectory = np.random.randn(50, 3)

    def test_neural_network_3d(self):
        """Test 3D neural network visualization."""
        layer_sizes = [5, 10, 3]
        ax = braintools.visualize.neural_network_3d(layer_sizes)
        self.assertIsNotNone(ax)
        plt.close()

    def test_connectivity_3d(self):
        """Test 3D connectivity visualization."""
        source_pos = self.positions[:5]
        target_pos = self.positions[5:]
        connections = np.random.binomial(1, 0.3, (5, 5))

        ax = braintools.visualize.connectivity_3d(
            source_pos, target_pos, connections
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_trajectory_3d(self):
        """Test 3D trajectory visualization."""
        ax = braintools.visualize.trajectory_3d(
            self.trajectory,
            time_colors=True
        )
        self.assertIsNotNone(ax)
        plt.close()
