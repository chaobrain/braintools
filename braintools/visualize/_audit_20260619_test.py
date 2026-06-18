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

"""Regression tests for the 2026-06-19 ``braintools.visualize`` audit fixes.

Each test pins a specific bug found and fixed during the audit; see
``docs/braintools-visualize-issues-found-20260619.md``.
"""

import unittest
import warnings

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from braintools.visualize import (
    line_plot, animate_1D, animate_2D,
    spike_raster, neural_trajectory,
    correlation_matrix, regression_plot, confusion_matrix, roc_curve,
    brain_surface_3d, volume_rendering, neural_network_3d,
)


def _draw(ax):
    """Force a canvas draw so deferred matplotlib validation (e.g. scatter
    color/size mismatches) actually raises."""
    ax.figure.canvas.draw()


class TestLinePlotConversion(unittest.TestCase):
    """Issue 1: ``line_plot`` reshaped before converting -> crashed on lists."""

    def tearDown(self):
        plt.close('all')

    def test_nested_list_val_matrix(self):
        # Previously raised AttributeError: 'list' object has no attribute 'reshape'
        ax = line_plot(list(range(5)), [[1], [2], [3], [4], [5]], plot_ids=[0])
        self.assertEqual(len(ax.get_lines()), 1)

    def test_flat_list_inputs(self):
        ax = line_plot([0, 1, 2], [1.0, 2.0, 3.0], plot_ids=[0])
        self.assertEqual(len(ax.get_lines()), 1)

    def test_ndarray_still_works(self):
        ax = line_plot(np.arange(10), np.random.randn(10, 3), plot_ids=[0, 1, 2])
        self.assertEqual(len(ax.get_lines()), 3)


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


class TestAnimate1DNoMutation(unittest.TestCase):
    """Issue 3: ``animate_1D`` mutated the caller's input dictionaries."""

    def tearDown(self):
        plt.close('all')

    def test_single_dict_not_mutated(self):
        d = {'ys': np.random.rand(10, 20)}
        before = set(d)
        animate_1D(d, show=False)
        self.assertEqual(set(d), before)

    def test_list_of_dicts_not_mutated(self):
        v = [{'ys': np.random.rand(5, 8), 'legend': 'a'}]
        before = set(v[0])
        animate_1D(v, show=False)
        self.assertEqual(set(v[0]), before)
        self.assertEqual(v[0]['legend'], 'a')

    def test_ndarray_input_works(self):
        fig = animate_1D(np.random.rand(6, 12), show=False)
        self.assertIsNotNone(fig)

    def test_unknown_dynamic_type_raises(self):
        with self.assertRaises(ValueError):
            animate_1D("not-an-array", show=False)


class TestConfusionMatrixNormalization(unittest.TestCase):
    """Issue 4: normalization divided by zero -> NaN for absent classes."""

    def tearDown(self):
        plt.close('all')

    def _assert_no_nan(self, normalize):
        y_true = np.array([0, 1, 2, 2, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 1, 0])  # class 2 never predicted
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            ax = confusion_matrix(y_true, y_pred, normalize=normalize)
        im = ax.get_images()[0]
        self.assertFalse(np.isnan(im.get_array()).any())

    def test_normalize_pred(self):
        self._assert_no_nan('pred')

    def test_normalize_true(self):
        self._assert_no_nan('true')

    def test_normalize_all(self):
        self._assert_no_nan('all')

    def test_normalize_true_rows_sum_to_one(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        ax = confusion_matrix(y_true, y_pred, normalize='true')
        cm = ax.get_images()[0].get_array()
        np.testing.assert_allclose(cm.sum(axis=1), np.ones(3))


class TestRegressionPlotRSquared(unittest.TestCase):
    """Issue 5: R^2 divided by zero when y is constant."""

    def tearDown(self):
        plt.close('all')

    def test_constant_y_no_warning(self):
        x = np.linspace(0, 10, 50)
        y = np.full(50, 3.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            ax = regression_plot(x, y)
        # The R^2 annotation must be finite (perfect fit on a flat line -> 1.0).
        texts = [t.get_text() for t in ax.texts if 'R' in t.get_text()]
        self.assertTrue(any('1.000' in t for t in texts))

    def test_normal_regression_r2_reasonable(self):
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 + np.random.RandomState(0).randn(100) * 0.1
        ax = regression_plot(x, y)
        texts = [t.get_text() for t in ax.texts if 'R' in t.get_text()]
        self.assertTrue(texts)


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


class TestCorrelationMatrixSingleFeature(unittest.TestCase):
    """Issue 7: single-feature input produced a 0-d scalar -> imshow crash."""

    def tearDown(self):
        plt.close('all')

    def test_single_feature_pearson(self):
        ax = correlation_matrix(np.random.rand(100, 1))
        _draw(ax)
        cm = ax.get_images()[0].get_array()
        self.assertEqual(cm.shape, (1, 1))
        self.assertAlmostEqual(float(cm[0, 0]), 1.0)

    def test_single_feature_spearman(self):
        ax = correlation_matrix(np.random.rand(100, 1), method='spearman')
        _draw(ax)

    def test_single_feature_kendall(self):
        ax = correlation_matrix(np.random.rand(100, 1), method='kendall')
        _draw(ax)

    def test_1d_input_treated_as_single_feature(self):
        ax = correlation_matrix(np.random.rand(100))
        _draw(ax)
        self.assertEqual(ax.get_images()[0].get_array().shape, (1, 1))


class TestAnimate2DShapeHandling(unittest.TestCase):
    """Issue 8: cryptic unpack error for non-2D input / size mismatch."""

    def tearDown(self):
        plt.close('all')

    def test_2d_input(self):
        anim = animate_2D(np.random.rand(10, 25), net_size=(5, 5), show=False)
        self.assertIsNotNone(anim)

    def test_3d_input_accepted(self):
        anim = animate_2D(np.random.rand(10, 5, 5), net_size=(5, 5), show=False)
        self.assertIsNotNone(anim)

    def test_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            animate_2D(np.random.rand(10, 30), net_size=(5, 5), show=False)

    def test_3d_frame_mismatch_raises(self):
        with self.assertRaises(ValueError):
            animate_2D(np.random.rand(10, 4, 4), net_size=(5, 5), show=False)

    def test_bad_ndim_raises(self):
        with self.assertRaises(ValueError):
            animate_2D(np.random.rand(10), net_size=(5, 5), show=False)


class TestRocCurveAUC(unittest.TestCase):
    """Issue 9: AUC was not anchored at the curve endpoints."""

    def tearDown(self):
        plt.close('all')

    @staticmethod
    def _auc(ax):
        for line in ax.get_lines():
            lbl = line.get_label()
            if 'AUC' in lbl:
                return float(lbl.split('=')[1])
        raise AssertionError('No AUC label found')

    def test_perfect_classifier_auc_one(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([.1, .2, .3, .4, .45, .6, .7, .8, .9, .95])
        ax = roc_curve(y_true, y_scores)
        self.assertAlmostEqual(self._auc(ax), 1.0, places=3)

    def test_auc_in_unit_interval(self):
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, 200)
        y_scores = rng.rand(200)
        ax = roc_curve(y_true, y_scores)
        auc = self._auc(ax)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)

    def test_curve_anchored_at_endpoints(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_scores = np.array([.2, .8, .3, .7, .9, .1])
        ax = roc_curve(y_true, y_scores)
        line = ax.get_lines()[0]
        xs, ys = line.get_xdata(), line.get_ydata()
        self.assertAlmostEqual(xs[0], 0.0)
        self.assertAlmostEqual(ys[0], 0.0)
        self.assertAlmostEqual(xs[-1], 1.0)
        self.assertAlmostEqual(ys[-1], 1.0)


class TestBrainSurfaceNormalization(unittest.TestCase):
    """Issue 10: signed values were mis-normalized (/max only)."""

    def tearDown(self):
        plt.close('all')

    def test_signed_values_run(self):
        verts = np.random.RandomState(0).randn(20, 3)
        faces = np.random.RandomState(1).randint(0, 20, (10, 3))
        vals = np.linspace(-5, 5, 20)
        ax = brain_surface_3d(verts, faces, values=vals)
        self.assertIsNotNone(ax)

    def test_constant_values_no_nan(self):
        verts = np.random.RandomState(0).randn(20, 3)
        faces = np.random.RandomState(1).randint(0, 20, (10, 3))
        vals = np.full(20, 2.0)
        ax = brain_surface_3d(verts, faces, values=vals)
        self.assertIsNotNone(ax)


class TestVolumeRenderingCmap(unittest.TestCase):
    """Issue 11: documented ``cmap`` parameter was ignored."""

    def tearDown(self):
        plt.close('all')

    def test_cmap_applied(self):
        vol = np.random.RandomState(0).rand(6, 6, 6)
        ax = volume_rendering(vol, threshold=0.5, cmap='plasma')
        self.assertIsNotNone(ax)

    def test_empty_volume_no_crash(self):
        vol = np.zeros((5, 5, 5))
        ax = volume_rendering(vol, threshold=0.5)
        self.assertIsNotNone(ax)


class TestNeuralNetwork3DReturn(unittest.TestCase):
    """Issue 12: doc example unpacked the single returned Axes."""

    def tearDown(self):
        plt.close('all')

    def test_returns_single_axes(self):
        from mpl_toolkits.mplot3d import Axes3D
        ret = neural_network_3d([4, 3, 2])
        self.assertIsInstance(ret, Axes3D)
        with self.assertRaises(TypeError):
            _a, _b = neural_network_3d([4, 3, 2])  # not unpackable


if __name__ == '__main__':
    unittest.main()
