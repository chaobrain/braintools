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
import pytest

matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import braintools
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


def _draw(ax):
    """Force a canvas draw so deferred matplotlib validation (e.g. scatter
    color/size mismatches) actually raises."""
    ax.figure.canvas.draw()


class TestSpikeRasterColorFiltering(unittest.TestCase):
    """Issue 2: a per-spike ``color`` array was not filtered with the spikes."""

    def tearDown(self):
        plt.close('all')

    def test_array_color_with_time_range(self):
        st = np.linspace(0, 100, 50)
        nid = np.arange(50) % 10
        colors = np.linspace(0, 1, 50)
        ax = spike_raster(st, nid, color=colors, time_range=(0, 50))
        _draw(ax)  # must not raise a length-mismatch error

    def test_array_color_with_neuron_range(self):
        st = np.linspace(0, 100, 50)
        nid = np.arange(50) % 10
        colors = np.linspace(0, 1, 50)
        ax = spike_raster(st, nid, color=colors, neuron_range=(0, 4))
        _draw(ax)

    def test_array_color_with_both_filters(self):
        st = np.linspace(0, 100, 50)
        nid = np.arange(50) % 10
        colors = np.linspace(0, 1, 50)
        ax = spike_raster(st, nid, color=colors, time_range=(10, 90), neuron_range=(0, 5))
        _draw(ax)

    def test_string_color_with_filter_unaffected(self):
        st = np.linspace(0, 100, 50)
        nid = np.arange(50) % 10
        ax = spike_raster(st, nid, color='red', time_range=(0, 50))
        _draw(ax)


class TestNeuralTrajectoryValidation(unittest.TestCase):
    """Issue 6: cryptic IndexError for < 2 feature columns."""

    def tearDown(self):
        plt.close('all')

    def test_single_feature_raises_valueerror(self):
        with self.assertRaises(ValueError):
            neural_trajectory(np.random.rand(100, 1))

    def test_out_of_range_dims_raise(self):
        with self.assertRaises(ValueError):
            neural_trajectory(np.random.rand(100, 2), dims=(0, 5))

    def test_non_2d_raises(self):
        with self.assertRaises(ValueError):
            neural_trajectory(np.random.rand(100))

    def test_valid_2d_and_3d(self):
        ax2 = neural_trajectory(np.random.rand(50, 2), time_color=False)
        self.assertIsNotNone(ax2)
        ax3 = neural_trajectory(np.random.rand(50, 3), time_color=False)
        self.assertIsNotNone(ax3)


class TestFiringRateMapNonSquare(unittest.TestCase):
    """A4: rectangular grid_size must not raise IndexError."""

    def tearDown(self):
        plt.close('all')

    def test_tall_grid(self):
        rates = np.random.rand(300) * 10
        pos = np.random.rand(300, 2)
        ax = firing_rate_map(rates, positions=pos, grid_size=(10, 25))
        self.assertIsNotNone(ax)

    def test_wide_grid(self):
        rates = np.random.rand(300) * 10
        pos = np.random.rand(300, 2)
        ax = firing_rate_map(rates, positions=pos, grid_size=(25, 10))
        self.assertIsNotNone(ax)

    def test_boundary_points_included(self):
        # points exactly on the max edge must be binned, not dropped
        rates = np.array([1.0, 2.0, 3.0, 4.0])
        pos = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        ax = firing_rate_map(rates, positions=pos, grid_size=(2, 3))
        self.assertIsNotNone(ax)


class TestNeuralTrajectory3DColorbar(unittest.TestCase):
    """D3: 3D time-colored trajectory gets a colorbar like the 2D path."""

    def tearDown(self):
        plt.close('all')

    def test_3d_time_color_adds_colorbar(self):
        fig = plt.figure()
        data = np.random.randn(40, 3)
        ax = neural_trajectory(data, dims=(0, 1, 2), time_color=True)
        # the 3D axes plus a colorbar axes
        self.assertGreaterEqual(len(ax.figure.axes), 2)


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

    def test_int_bins_use_time_range_as_the_extent(self):
        """A scalar ``bins`` combined with ``time_range`` must bin over
        ``time_range``, not over the data's min/max.

        Regression test: ``ax.hist`` was called without ``range=``, so the edges
        silently came from the filtered data instead, which made two datasets
        sharing a ``time_range`` produce non-comparable histograms.
        """
        for time_range in ((0, 10), (2, 8), (4, 6)):
            fig, ax = plt.subplots()
            spike_histogram(self.spikes, bins=10, time_range=time_range, ax=ax)
            edges = np.array([p.get_x() for p in ax.patches])
            widths = np.array([p.get_width() for p in ax.patches])
            counts = np.array([p.get_height() for p in ax.patches])
            np.testing.assert_allclose(
                np.r_[edges, edges[-1] + widths[-1]],
                np.linspace(time_range[0], time_range[1], 11),
            )
            expected, _ = np.histogram(self.spikes, bins=10, range=time_range)
            np.testing.assert_array_equal(counts, expected)
            plt.close('all')

    def test_two_datasets_share_edges_for_the_same_time_range(self):
        """Histograms from different data must be comparable under one time_range."""
        other = np.sort(np.random.uniform(8, 10, 100))

        def edges_of(data):
            fig, ax = plt.subplots()
            spike_histogram(data, bins=10, time_range=(0, 10), ax=ax)
            left = np.array([p.get_x() for p in ax.patches])
            w = np.array([p.get_width() for p in ax.patches])
            result = np.r_[left, left[-1] + w[-1]]
            plt.close('all')
            return result

        np.testing.assert_allclose(edges_of(self.spikes), edges_of(other))


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

    def test_last_bin_includes_the_maximum_stimulus(self):
        """No sample may be dropped: the final bin is closed on the right.

        Regression test: every bin was half-open ``[lo, hi)`` while the last right
        edge equals ``stimulus.max()``, so all samples sitting exactly at the
        maximum were silently discarded -- negligible for continuous stimuli but
        ~7% for integer-valued or clipped ones, which biased the top bin's mean.
        """
        n_bins = 10
        stim = np.repeat(np.arange(0.0, 11.0), 8)   # integer-valued: many == max
        resp = 2.0 * stim
        fig, ax = plt.subplots()
        tuning_curve(stim, resp, bins=n_bins, error_bars=False, ax=ax)
        plotted = np.asarray(ax.lines[0].get_ydata(), float)

        edges = np.linspace(stim.min(), stim.max(), n_bins + 1)
        expected, total = [], 0
        for i in range(n_bins):
            hi = (stim <= edges[i + 1]) if i == n_bins - 1 else (stim < edges[i + 1])
            in_bin = (stim >= edges[i]) & hi
            total += int(in_bin.sum())
            if in_bin.any():
                expected.append(resp[in_bin].mean())

        self.assertEqual(total, len(stim), 'the maximum stimulus samples were dropped')
        np.testing.assert_allclose(plotted, np.array(expected))

    def test_last_bin_keeps_clipped_samples(self):
        """Clipped stimuli pile up at the maximum; the top bin must still see them."""
        n_bins = 10
        stim = np.clip(np.random.RandomState(0).randn(200) * 3 + 5, 0, 10)
        resp = stim.copy()
        pile_up = int((stim == stim.max()).sum())
        self.assertGreater(pile_up, 5)          # the clipped pile-up exists

        fig, ax = plt.subplots()
        tuning_curve(stim, resp, bins=n_bins, error_bars=False, ax=ax)
        plotted = np.asarray(ax.lines[0].get_ydata(), float)

        edges = np.linspace(stim.min(), stim.max(), n_bins + 1)
        top_bin = (stim >= edges[-2]) & (stim <= edges[-1])
        # the closed top bin contains the whole pile-up, not just the samples below it
        self.assertGreaterEqual(int(top_bin.sum()), pile_up)
        self.assertAlmostEqual(float(plotted[-1]), float(resp[top_bin].mean()), places=9)

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


@pytest.mark.parametrize('pref, width, hi', [(90.0, 20.0, 180.0), (1.5, 0.35, np.pi)])
def test_tuning_curve_gaussian_fit_finds_peak_in_degrees_and_radians(pref, width, hi):
    # curve_fit used to start at mu = sigma = 1 and silently converge to a wrong
    # optimum for stimuli outside [0, 1]
    rng = np.random.default_rng(0)
    s = rng.uniform(0, hi, 2000)
    r = 2 + 20 * np.exp(-(s - pref) ** 2 / (2 * width ** 2)) + rng.normal(0, 2, s.size)
    ax = tuning_curve(s, r, bins=18, fit_curve='gaussian')
    x, y = [l for l in ax.get_lines() if l.get_label() == 'Gaussian fit'][0].get_data()
    assert abs(x[np.argmax(y)] - pref) < 0.05 * hi
    assert abs(y.max() - 22.0) < 2.0
    plt.close('all')


@pytest.mark.parametrize('widths', ['matrix', 'flat', 'per_edge'])
def test_network_topology_edge_widths_formats(widths):
    A = np.array([[0, 2., 0], [0, 0, 5.], [1., 0, 0]])
    w = {'matrix': A, 'flat': A.ravel(), 'per_edge': np.array([2., 5., 1.])}[widths]
    ax = network_topology(A > 0, layout='circular', edge_widths=w)
    assert [l.get_linewidth() for l in ax.get_lines()] == [2.0, 5.0, 1.0]
    plt.close('all')


def test_network_topology_rejects_mismatched_edge_attribute():
    A = np.array([[0, 2., 0], [0, 0, 5.], [1., 0, 0]])
    with pytest.raises(ValueError):
        network_topology(A > 0, layout='circular', edge_widths=[1.0, 2.0])
    plt.close('all')


def test_firing_rate_map_unvisited_bins_are_blank_and_axes_in_position_units():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.1, 0.1]])
    rates = np.array([5.0, 0.0, 3.0, 7.0])
    ax = firing_rate_map(rates, positions=positions, grid_size=(2, 2), show_colorbar=False)
    img = ax.get_images()[0]
    data = np.ma.filled(img.get_array().astype(float), np.nan)
    assert data[0, 0] == 6.0          # mean of 5 and 7
    assert data[0, 1] == 0.0          # visited, silent
    assert data[1, 0] == 3.0
    assert np.isnan(data[1, 1])       # never visited
    assert tuple(img.get_extent()) == (0.0, 1.0, 0.0, 2.0)
    plt.close('all')


def test_network_topology_draws_negative_edges_in_blue():
    W = np.array([[0.0, 1.0], [-2.0, 0.0]])
    ax = network_topology(W, layout='circular')
    colors = sorted(str(l.get_color()) for l in ax.get_lines())
    assert colors == sorted(['gray', '#1f77b4'])
    ax = network_topology(W, layout='circular', edge_widths=W)
    assert sorted(l.get_linewidth() for l in ax.get_lines()) == [1.0, 2.0]
    plt.close('all')


class TestIntegration(unittest.TestCase):
    """Integration tests for the visualization module."""

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



class TestSpikeTrainsInput(unittest.TestCase):
    """``spike_raster``, ``spike_histogram`` and ``isi_distribution`` accept a SpikeTrains."""

    def setUp(self):
        import brainunit as u
        from braintools.spike import SpikeTrains
        self.u = u
        self.spikes = np.zeros((100, 4), dtype=bool)
        self.spikes[::10, 0] = True
        self.spikes[5::20, 2] = True
        self.st = SpikeTrains.from_dense(self.spikes, dt=0.1 * u.ms)
        self.times, self.ids = np.nonzero(self.spikes)
        self.times = self.times * 0.1

    def tearDown(self):
        plt.close('all')

    def test_spike_raster_matches_legacy_call(self):
        from braintools.visualize import spike_raster
        ax = spike_raster(self.st)
        offsets = ax.collections[0].get_offsets()
        legacy = spike_raster(self.times, self.ids).collections[0].get_offsets()
        order = np.lexsort((offsets[:, 1], offsets[:, 0]))
        legacy_order = np.lexsort((legacy[:, 1], legacy[:, 0]))
        np.testing.assert_allclose(np.asarray(offsets)[order], np.asarray(legacy)[legacy_order], rtol=1e-5)
        self.assertEqual(ax.get_xlabel(), 'Time (ms)')

    def test_spike_raster_rejects_neuron_ids_with_spike_trains(self):
        from braintools.visualize import spike_raster
        with self.assertRaises(ValueError):
            spike_raster(self.st, neuron_ids=[0, 1])

    def test_spike_raster_unitless_spike_trains_keeps_label(self):
        from braintools.spike import SpikeTrains
        from braintools.visualize import spike_raster
        ax = spike_raster(SpikeTrains.from_dense(self.spikes, dt=0.1), time_range=(0, 5), show_stats=True)
        self.assertEqual(ax.get_xlabel(), 'Time')
        # time_range is closed: the spike at exactly 5.0 (step 50) is shown
        self.assertEqual(len(ax.collections[0].get_offsets()), int(self.spikes[:51].sum()))

    def test_spike_histogram_pools_all_neurons(self):
        from braintools.visualize import spike_histogram
        ax = spike_histogram(self.st, bins=5)
        heights = [p.get_height() for p in ax.patches]
        self.assertEqual(sum(heights), self.spikes.sum())
        self.assertEqual(ax.get_xlabel(), 'Time (ms)')

    def test_isi_distribution_within_neurons(self):
        from braintools.visualize import isi_distribution
        ax = isi_distribution(self.st, bins=3)
        heights = [p.get_height() for p in ax.patches]
        # neuron 0: 9 intervals of 1 ms; neuron 2: 4 intervals of 2 ms
        self.assertEqual(sum(heights), 13)
        values = np.concatenate([np.full(int(h), p.get_x()) for p, h in zip(ax.patches, heights) if h])
        self.assertLessEqual(values.max(), 2.0)

    def test_isi_distribution_empty_spike_trains_warns(self):
        from braintools.spike import SpikeTrains
        from braintools.visualize import isi_distribution
        with self.assertWarns(UserWarning):
            isi_distribution(SpikeTrains.from_dense(np.zeros((10, 2)), dt=1.))


class TestSpikeTrainsQuantityTimeRange(unittest.TestCase):
    """``time_range``/``bin_size`` given as Quantities work with a unit-carrying SpikeTrains."""

    def setUp(self):
        import brainunit as u
        from braintools.spike import SpikeTrains
        self.u = u
        spikes = np.zeros((100, 2), dtype=bool)
        spikes[::10, 0] = True
        self.st = SpikeTrains.from_dense(spikes, dt=0.1 * u.ms)

    def tearDown(self):
        plt.close('all')

    def test_spike_raster_quantity_time_range(self):
        from braintools.visualize import spike_raster
        u = self.u
        ax = spike_raster(self.st, time_range=(0 * u.ms, 3 * u.ms))
        plain = spike_raster(self.st, time_range=(0., 3.))
        np.testing.assert_allclose(ax.collections[0].get_offsets(), plain.collections[0].get_offsets())
        # Converted, not just stripped: 3000 us == 3 ms.
        us = spike_raster(self.st, time_range=(0 * u.us, 3000 * u.us))
        np.testing.assert_allclose(us.collections[0].get_offsets(), plain.collections[0].get_offsets())

    def test_spike_histogram_quantity_time_range_and_bin_size(self):
        from braintools.visualize import spike_histogram
        u = self.u
        ax = spike_histogram(self.st, time_range=(0 * u.ms, 5 * u.ms), bin_size=1 * u.ms)
        plain = spike_histogram(self.st, time_range=(0., 5.), bin_size=1.)
        self.assertEqual([p.get_height() for p in ax.patches], [p.get_height() for p in plain.patches])
