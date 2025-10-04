# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,# distributed under the License is distributed on an "AS IS" BASIS,
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

    def test_comprehensive_dashboard(self):
        """Test creating a comprehensive neural data dashboard."""
        # Generate test data
        spike_times = [np.sort(np.random.uniform(0, 10, np.random.randint(20, 50)))
                       for _ in range(20)]
        time = np.linspace(0, 10, 1000)
        population_activity = np.random.randn(1000)

        # Create multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Spike raster
        braintools.visualize.spike_raster(spike_times, ax=axes[0, 0], title='Spike Raster')

        # Population activity
        braintools.visualize.population_activity(population_activity[:, None], time=time,
                                                 ax=axes[0, 1], title='Population Activity')

        # ISI distribution
        braintools.visualize.isi_distribution(spike_times, ax=axes[1, 0], title='ISI Distribution')

        # Spike histogram
        all_spikes = np.concatenate(spike_times)
        braintools.visualize.spike_histogram(all_spikes, ax=axes[1, 1], title='PSTH')

        plt.tight_layout()
        plt.close()
