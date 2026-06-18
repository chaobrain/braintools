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

"""Comprehensive coverage for braintools.visualize._interactive."""

import unittest
from unittest import mock

import numpy as np

from braintools.visualize import (
    interactive_spike_raster,
    interactive_line_plot,
    interactive_heatmap,
    interactive_3d_scatter,
    interactive_network,
    interactive_histogram,
    interactive_surface,
    interactive_correlation_matrix,
    dashboard_neural_activity,
)
from braintools.visualize import _interactive

skip_no_plotly = unittest.skipUnless(
    _interactive.PLOTLY_AVAILABLE, "Plotly not available"
)


class TestCheckPlotly(unittest.TestCase):
    def test_raises_when_unavailable(self):
        # Force the "plotly missing" branch regardless of the real environment.
        with mock.patch.object(_interactive, 'PLOTLY_AVAILABLE', False):
            with self.assertRaises(ImportError):
                interactive_heatmap(np.random.randn(4, 4))


@skip_no_plotly
class TestInteractiveSpikeRaster(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.times = np.random.rand(200) * 100
        self.ids = np.random.randint(0, 10, 200)

    def test_array_input_color_neuron(self):
        fig = interactive_spike_raster(self.times, self.ids, color_by='neuron')
        self.assertIsNotNone(fig)

    def test_array_input_color_time(self):
        fig = interactive_spike_raster(self.times, self.ids, color_by='time')
        self.assertIsNotNone(fig)

    def test_array_input_no_color(self):
        fig = interactive_spike_raster(self.times, self.ids)
        self.assertIsNotNone(fig)

    def test_list_input(self):
        spikes = [np.sort(np.random.rand(8) * 50) for _ in range(5)]
        fig = interactive_spike_raster(spikes, color_by='neuron')
        self.assertIsNotNone(fig)

    def test_time_and_neuron_filtering(self):
        fig = interactive_spike_raster(self.times, self.ids,
                                       time_range=(10, 50), neuron_range=(2, 7))
        self.assertIsNotNone(fig)

    def test_missing_ids_raises(self):
        with self.assertRaises(ValueError):
            interactive_spike_raster(self.times, None)


@skip_no_plotly
class TestInteractiveLinePlot(unittest.TestCase):
    def test_single_trace(self):
        x = np.linspace(0, 1, 50)
        fig = interactive_line_plot(x, np.sin(x))
        self.assertIsNotNone(fig)

    def test_multi_trace_with_labels(self):
        x = np.linspace(0, 1, 50)
        fig = interactive_line_plot(x, [np.sin(x), np.cos(x)], labels=['s', 'c'])
        self.assertIsNotNone(fig)

    def test_multi_trace_default_labels(self):
        x = np.linspace(0, 1, 50)
        fig = interactive_line_plot(x, [np.sin(x), np.cos(x)])
        self.assertIsNotNone(fig)


@skip_no_plotly
class TestInteractiveHeatmap(unittest.TestCase):
    def test_no_labels(self):
        fig = interactive_heatmap(np.random.randn(8, 8))
        self.assertIsNotNone(fig)

    def test_with_labels(self):
        fig = interactive_heatmap(np.random.randn(3, 3),
                                  x_labels=['a', 'b', 'c'],
                                  y_labels=['x', 'y', 'z'])
        self.assertIsNotNone(fig)


@skip_no_plotly
class TestInteractive3DScatter(unittest.TestCase):
    def setUp(self):
        self.pts = np.random.randn(50, 3)

    def test_minimal(self):
        fig = interactive_3d_scatter(self.pts[:, 0], self.pts[:, 1], self.pts[:, 2])
        self.assertIsNotNone(fig)

    def test_with_color_size_labels(self):
        fig = interactive_3d_scatter(
            self.pts[:, 0], self.pts[:, 1], self.pts[:, 2],
            color=np.random.rand(50), size=np.full(50, 6),
            labels=[f'p{i}' for i in range(50)],
        )
        self.assertIsNotNone(fig)


@skip_no_plotly
class TestInteractiveNetwork(unittest.TestCase):
    def setUp(self):
        self.adj = (np.random.rand(8, 8) > 0.6).astype(float)

    def test_default_layout(self):
        fig = interactive_network(self.adj)
        self.assertIsNotNone(fig)

    def test_explicit_positions_and_styling(self):
        positions = np.random.randn(8, 2)
        fig = interactive_network(
            self.adj, positions=positions,
            node_labels=[f'n{i}' for i in range(8)],
            node_colors=np.arange(8),
            edge_weights=self.adj,
        )
        self.assertIsNotNone(fig)


@skip_no_plotly
class TestInteractiveHistogram(unittest.TestCase):
    def test_single(self):
        fig = interactive_histogram(np.random.randn(500))
        self.assertIsNotNone(fig)

    def test_multi_with_labels(self):
        fig = interactive_histogram([np.random.randn(200), np.random.randn(200)],
                                    labels=['a', 'b'])
        self.assertIsNotNone(fig)


@skip_no_plotly
class TestInteractiveSurface(unittest.TestCase):
    def test_z_only(self):
        x = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x, x)
        fig = interactive_surface(np.sin(np.sqrt(X ** 2 + Y ** 2)))
        self.assertIsNotNone(fig)

    def test_with_xy(self):
        x = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x, x)
        fig = interactive_surface(np.sin(np.sqrt(X ** 2 + Y ** 2)), x=x, y=x)
        self.assertIsNotNone(fig)


@skip_no_plotly
class TestInteractiveCorrelationMatrix(unittest.TestCase):
    def test_pearson(self):
        fig = interactive_correlation_matrix(np.random.randn(100, 5))
        self.assertIsNotNone(fig)

    def test_spearman_two_features(self):
        # regression: spearmanr returns a scalar for exactly two columns
        fig = interactive_correlation_matrix(np.random.randn(100, 2),
                                             method='spearman')
        self.assertIsNotNone(fig)

    def test_spearman_many_features(self):
        fig = interactive_correlation_matrix(np.random.randn(100, 4),
                                             method='spearman', labels=list('abcd'))
        self.assertIsNotNone(fig)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            interactive_correlation_matrix(np.random.randn(50, 3), method='nope')


@skip_no_plotly
class TestDashboardNeuralActivity(unittest.TestCase):
    def test_list_input(self):
        spikes = [np.sort(np.random.rand(15) * 100) for _ in range(8)]
        fig = dashboard_neural_activity(spikes)
        self.assertIsNotNone(fig)

    def test_array_input_with_population(self):
        times = np.sort(np.random.rand(300) * 100)
        ids = np.random.randint(0, 12, 300)
        pop = np.random.rand(100)
        t = np.linspace(0, 100, 100)
        fig = dashboard_neural_activity(times, ids, population_activity=pop, time=t)
        self.assertIsNotNone(fig)

    def test_array_input_no_ids(self):
        times = np.sort(np.random.rand(100) * 100)
        fig = dashboard_neural_activity(times)
        self.assertIsNotNone(fig)

    def test_population_activity_default_time(self):
        # population_activity given but time=None -> default time axis branch
        times = np.sort(np.random.rand(150) * 100)
        ids = np.random.randint(0, 6, 150)
        fig = dashboard_neural_activity(times, ids,
                                        population_activity=np.random.rand(80))
        self.assertIsNotNone(fig)

    def test_single_spike_neuron_zero_rate(self):
        # a neuron with a single spike exercises the rate=0 (else) branch
        times = np.array([1.0, 2.0, 3.0, 50.0])
        ids = np.array([0, 0, 0, 1])  # neuron 1 has a single spike
        fig = dashboard_neural_activity(times, ids)
        self.assertIsNotNone(fig)

    def test_empty_spikes(self):
        # empty input must not crash the statistics table branch
        fig = dashboard_neural_activity(np.array([]), np.array([]))
        self.assertIsNotNone(fig)


if __name__ == '__main__':
    unittest.main()
