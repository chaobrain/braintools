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
import warnings
from unittest import mock

import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from braintools.visualize._neural import (
    spike_raster,
    population_activity,
    connectivity_matrix,
    neural_trajectory,
    spike_histogram,
    isi_distribution,
    firing_rate_map,
    phase_portrait,
    network_topology,
    tuning_curve,
)


class TestSpikeRasterExtra(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.spike_times_flat = np.sort(np.random.uniform(0, 10, 50))
        self.neuron_ids = np.random.randint(0, 8, 50)

    def tearDown(self):
        plt.close('all')

    def test_flat_with_neuron_ids(self):
        fig, ax = plt.subplots()
        out = spike_raster(self.spike_times_flat, neuron_ids=self.neuron_ids,
                           ax=ax, color='red', title='R')
        self.assertIs(out, ax)

    def test_flat_without_neuron_ids_raises(self):
        with self.assertRaises(ValueError):
            spike_raster(self.spike_times_flat)

    def test_time_and_neuron_range(self):
        fig, ax = plt.subplots()
        out = spike_raster(
            self.spike_times_flat, neuron_ids=self.neuron_ids,
            time_range=(2.0, 8.0), neuron_range=(1, 5),
            ax=ax, show_stats=True,
        )
        self.assertIs(out, ax)

    def test_list_input_show_stats(self):
        spikes = [np.sort(np.random.uniform(0, 5, np.random.randint(3, 10)))
                  for _ in range(6)]
        out = spike_raster(spikes, show_stats=True, figsize=(6, 4))
        self.assertIsNotNone(out)

    def test_empty_lists(self):
        # all-empty neuron lists -> no scatter, no stats
        spikes = [np.array([]) for _ in range(4)]
        out = spike_raster(spikes, show_stats=True)
        self.assertIsNotNone(out)


class TestPopulationActivityExtra(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        self.data = np.random.rand(80, 12)

    def tearDown(self):
        plt.close('all')

    def test_1d_data(self):
        out = population_activity(np.random.rand(80))
        self.assertIsNotNone(out)

    def test_methods(self):
        for method in ['mean', 'sum', 'std', 'var', 'median']:
            fig, ax = plt.subplots()
            out = population_activity(self.data, method=method, ax=ax, fill=False)
            self.assertIs(out, ax)
            plt.close(fig)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            population_activity(self.data, method='bogus')

    def test_neuron_ids_window_and_dt(self):
        fig, ax = plt.subplots()
        out = population_activity(
            self.data,
            dt=0.1,
            neuron_ids=np.array([0, 2, 4]),
            window_size=5,
            ax=ax,
            color='green',
            title='pop',
        )
        self.assertIs(out, ax)

    def test_explicit_time(self):
        t = np.linspace(0, 8, 80)
        out = population_activity(self.data, time=t, ylabel='custom')
        self.assertIsNotNone(out)


class TestConnectivityMatrixExtra(unittest.TestCase):

    def setUp(self):
        np.random.seed(2)
        self.w = np.random.randn(5, 5)

    def tearDown(self):
        plt.close('all')

    def test_values_threshold_labels(self):
        fig, ax = plt.subplots()
        out = connectivity_matrix(
            self.w,
            pre_labels=[f'p{i}' for i in range(5)],
            post_labels=[f'q{i}' for i in range(5)],
            center_zero=True,
            show_values=True,
            value_threshold=0.5,
            ax=ax,
            xlabel='post', ylabel='pre',
            title='conn',
        )
        self.assertIs(out, ax)

    def test_no_center_no_colorbar(self):
        out = connectivity_matrix(
            np.abs(self.w), center_zero=False, show_colorbar=False)
        self.assertIsNotNone(out)


class TestNeuralTrajectoryExtra(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_3d_time_color(self):
        data = np.random.randn(40, 4)
        out = neural_trajectory(data, time_color=True, title='3d')
        self.assertIsNotNone(out)

    def test_3d_no_time_color(self):
        data = np.random.randn(40, 3)
        out = neural_trajectory(data, dims=(0, 1, 2), time_color=False)
        self.assertIsNotNone(out)

    def test_2d_no_time_color_with_ax(self):
        data = np.random.randn(40, 2)
        fig, ax = plt.subplots()
        out = neural_trajectory(data, dims=(0, 1), time_color=False, ax=ax)
        self.assertIs(out, ax)

    def test_2d_default_dims(self):
        data = np.random.randn(30, 2)
        out = neural_trajectory(data, time_color=True)
        self.assertIsNotNone(out)


class TestSpikeHistogramExtra(unittest.TestCase):

    def setUp(self):
        np.random.seed(3)
        self.spikes = np.sort(np.random.uniform(0, 10, 200))

    def tearDown(self):
        plt.close('all')

    def test_bin_size_with_time_range(self):
        fig, ax = plt.subplots()
        out = spike_histogram(self.spikes, bin_size=0.5, time_range=(1, 9),
                              ax=ax, title='psth')
        self.assertIs(out, ax)

    def test_bin_size_no_time_range(self):
        out = spike_histogram(self.spikes, bin_size=1.0, density=True)
        self.assertIsNotNone(out)

    def test_list_input(self):
        spikes = [np.random.uniform(0, 5, 20) for _ in range(4)]
        out = spike_histogram(spikes, bins=15)
        self.assertIsNotNone(out)


class TestISIDistributionExtra(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_list_log_scale_max_isi(self):
        spikes = [np.sort(np.random.uniform(0, 10, 30)) for _ in range(5)]
        fig, ax = plt.subplots()
        out = isi_distribution(spikes, max_isi=2.0, log_scale=True, ax=ax,
                               title='isi')
        self.assertIs(out, ax)

    def test_flat_array(self):
        spikes = np.sort(np.random.uniform(0, 10, 100))
        out = isi_distribution(spikes)
        self.assertIsNotNone(out)

    def test_empty_warns(self):
        # single spike per neuron -> no ISIs -> warning + early return
        spikes = [np.array([1.0]) for _ in range(3)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            out = isi_distribution(spikes)
        self.assertIsNotNone(out)
        self.assertTrue(any('inter-spike' in str(x.message).lower() for x in w))


class TestFiringRateMapExtra(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_2d_map_with_ax(self):
        fig, ax = plt.subplots()
        out = firing_rate_map(np.random.rand(15, 15), ax=ax, title='rate',
                              show_colorbar=False)
        self.assertIs(out, ax)

    def test_1d_with_positions(self):
        rates = np.random.rand(100)
        positions = np.random.rand(100, 2) * 10
        out = firing_rate_map(rates, positions=positions, grid_size=(10, 10))
        self.assertIsNotNone(out)

    def test_1d_with_positions_default_grid(self):
        # grid_size None -> defaults to (50, 50)
        rates = np.random.rand(80)
        positions = np.random.rand(80, 2) * 5
        out = firing_rate_map(rates, positions=positions)
        self.assertIsNotNone(out)

    def test_1d_without_positions_raises(self):
        with self.assertRaises(ValueError):
            firing_rate_map(np.random.rand(50))


class TestPhasePortraitExtra(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_2d_input_trajectory(self):
        data = np.random.randn(60, 2)
        out = phase_portrait(data, title='phase')
        self.assertIsNotNone(out)

    def test_xy_with_vector_field(self):
        x = np.random.randn(30)
        y = np.random.randn(30)
        dx = np.random.randn(30)
        dy = np.random.randn(30)
        fig, ax = plt.subplots()
        out = phase_portrait(x, y, dx=dx, dy=dy, trajectory=True,
                             vector_field=True, ax=ax)
        self.assertIs(out, ax)

    def test_single_point(self):
        out = phase_portrait(np.array([1.0]), np.array([2.0]))
        self.assertIsNotNone(out)

    def test_1d_x_no_y_raises(self):
        with self.assertRaises(ValueError):
            phase_portrait(np.random.randn(10))


class TestNetworkTopologyExtra(unittest.TestCase):

    def setUp(self):
        np.random.seed(4)
        self.adj = (np.random.rand(6, 6) > 0.6).astype(float)

    def tearDown(self):
        plt.close('all')

    def test_circular_layout(self):
        out = network_topology(self.adj, layout='circular', title='net')
        self.assertIsNotNone(out)

    def test_random_layout(self):
        out = network_topology(self.adj, layout='random')
        self.assertIsNotNone(out)

    def test_spring_layout(self):
        out = network_topology(self.adj, layout='spring')
        self.assertIsNotNone(out)

    def test_unknown_layout_raises(self):
        with self.assertRaises(ValueError):
            network_topology(self.adj, layout='nope')

    def test_explicit_positions_and_arrays(self):
        n = 6
        positions = np.random.rand(n, 2)
        node_colors = np.arange(n)
        node_sizes = np.full(n, 80.0)
        edge_colors = ['gray'] * (n * n)
        edge_widths = np.ones(n * n)
        fig, ax = plt.subplots()
        out = network_topology(
            self.adj, positions=positions, node_colors=node_colors,
            node_sizes=node_sizes, edge_colors=edge_colors,
            edge_widths=edge_widths, ax=ax,
        )
        self.assertIs(out, ax)


class TestTuningCurveExtra(unittest.TestCase):

    def setUp(self):
        np.random.seed(5)
        self.stim = np.random.uniform(-5, 5, 300)
        self.resp = np.exp(-self.stim ** 2 / 2) + 0.05 * np.random.randn(300)

    def tearDown(self):
        plt.close('all')

    def test_error_bars(self):
        out = tuning_curve(self.stim, self.resp, bins=15, error_bars=True)
        self.assertIsNotNone(out)

    def test_no_error_bars_with_ax(self):
        fig, ax = plt.subplots()
        out = tuning_curve(self.stim, self.resp, bins=12, error_bars=False,
                           ax=ax, title='tune')
        self.assertIs(out, ax)

    def test_gaussian_fit(self):
        out = tuning_curve(self.stim, self.resp, bins=20, fit_curve='gaussian')
        self.assertIsNotNone(out)

    def test_polynomial_fit(self):
        out = tuning_curve(self.stim, self.resp, bins=20, fit_curve='polynomial')
        self.assertIsNotNone(out)

    def test_bins_array(self):
        edges = np.linspace(-5, 5, 16)
        out = tuning_curve(self.stim, self.resp, bins=edges, error_bars=False)
        self.assertIsNotNone(out)

    def test_gaussian_fit_failure_warns(self):
        # Force curve_fit to raise so the except branch (warning) is covered.
        with mock.patch('scipy.optimize.curve_fit',
                        side_effect=RuntimeError('no convergence')):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                out = tuning_curve(self.stim, self.resp, bins=20,
                                   fit_curve='gaussian')
        self.assertIsNotNone(out)
        self.assertTrue(any('gaussian' in str(x.message).lower() for x in w))

    def test_polynomial_fit_failure_warns(self):
        # Force polyfit to raise so the except branch (warning) is covered.
        with mock.patch('numpy.polyfit',
                        side_effect=np.linalg.LinAlgError('singular')):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                out = tuning_curve(self.stim, self.resp, bins=20,
                                   fit_curve='polynomial')
        self.assertIsNotNone(out)
        self.assertTrue(any('polynomial' in str(x.message).lower() for x in w))


if __name__ == '__main__':
    unittest.main()
