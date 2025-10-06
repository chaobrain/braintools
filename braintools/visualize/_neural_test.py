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


class TestNeuralVisualization(unittest.TestCase):
    """Test neural-specific visualization functions."""

    def setUp(self):
        """Set up test data."""
        self.n_neurons = 10
        self.n_time_steps = 100
        self.time = np.linspace(0, 1, self.n_time_steps)

        # Generate spike data
        np.random.seed(42)
        self.spike_times = [np.sort(np.random.uniform(0, 1, np.random.randint(5, 15)))
                            for _ in range(self.n_neurons)]

        # Generate population activity
        self.population_data = np.random.randn(self.n_time_steps, self.n_neurons)

        # Generate connectivity matrix
        self.connectivity = np.random.rand(self.n_neurons, self.n_neurons)
        self.connectivity[self.connectivity < 0.7] = 0  # Sparse connectivity

    def test_spike_raster(self):
        """Test spike raster plotting."""
        ax = braintools.visualize.spike_raster(
            self.spike_times,
            show_stats=True,
            figsize=(8, 6)
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_population_activity(self):
        """Test population activity plotting."""
        ax = braintools.visualize.population_activity(
            self.population_data,
            time=self.time,
            method='mean'
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_connectivity_matrix(self):
        """Test connectivity matrix visualization."""
        ax = braintools.visualize.connectivity_matrix(
            self.connectivity,
            show_values=True,
            center_zero=False
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_neural_trajectory(self):
        """Test neural trajectory plotting."""
        trajectory_data = np.random.randn(50, 3)
        ax = braintools.visualize.neural_trajectory(
            trajectory_data,
            time_color=True
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_spike_histogram(self):
        """Test spike histogram (PSTH)."""
        all_spikes = np.concatenate(self.spike_times)
        ax = braintools.visualize.spike_histogram(
            all_spikes,
            bins=20,
            density=True
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_isi_distribution(self):
        """Test inter-spike interval distribution."""
        ax = braintools.visualize.isi_distribution(
            self.spike_times,
            log_scale=False
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_firing_rate_map(self):
        """Test firing rate map visualization."""
        # Create 2D rate map
        rate_map = np.random.rand(20, 20)
        ax = braintools.visualize.firing_rate_map(
            rate_map,
            cmap='hot'
        )
        self.assertIsNotNone(ax)
        plt.close()
