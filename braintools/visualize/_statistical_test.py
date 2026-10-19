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

import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

import braintools
from braintools.visualize import (
    correlation_matrix, regression_plot, confusion_matrix, roc_curve,
)


class TestStatisticalVisualization(unittest.TestCase):
    """Test statistical visualization functions."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = np.random.randn(100, 5)
        self.labels = [f'Feature {i + 1}' for i in range(5)]

    def test_correlation_matrix(self):
        """Test correlation matrix visualization."""
        ax = braintools.visualize.correlation_matrix(
            self.data,
            labels=self.labels,
            show_values=True
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_distribution_plot(self):
        """Test distribution plotting."""
        ax = braintools.visualize.distribution_plot(
            self.data[:, 0],
            plot_type='hist',
            fit_normal=True
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_qq_plot(self):
        """Test Q-Q plot."""
        ax = braintools.visualize.qq_plot(
            self.data[:, 0],
            distribution='norm'
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_box_plot(self):
        """Test box plot."""
        data_list = [self.data[:, i] for i in range(3)]
        ax = braintools.visualize.box_plot(
            data_list,
            labels=['A', 'B', 'C']
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_regression_plot(self):
        """Test regression plot."""
        x = self.data[:, 0]
        y = 2 * x + np.random.randn(len(x)) * 0.5
        ax = braintools.visualize.regression_plot(
            x, y,
            fit_line=True,
            confidence_interval=True
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_confusion_matrix(self):
        """Test confusion matrix visualization."""
        y_true = np.random.randint(0, 3, 100)
        y_pred = np.random.randint(0, 3, 100)
        ax = braintools.visualize.confusion_matrix(
            y_true, y_pred,
            labels=['A', 'B', 'C']
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_scatter_matrix(self):
        """Test scatter matrix visualization."""
        # Test full scatter matrix (standalone)
        fig = braintools.visualize.scatter_matrix(
            self.data[:, :3],  # Use 3 features for manageable matrix
            labels=self.labels[:3],
            figsize=(8, 8),
            alpha=0.6
        )
        self.assertIsNotNone(fig)
        plt.show()
        plt.close()

        # Test simplified scatter matrix (with ax parameter)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        result_fig = braintools.visualize.scatter_matrix(
            self.data[:, :3],
            labels=self.labels[:3],
            ax=ax,
            alpha=0.6
        )
        self.assertIsNotNone(result_fig)
        plt.show()
        plt.close()


def _draw(ax):
    """Force a canvas draw so deferred matplotlib validation (e.g. scatter
    color/size mismatches) actually raises."""
    ax.figure.canvas.draw()


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


class TestCorrelationSpearmanTwoFeatures(unittest.TestCase):
    """A5: spearman correlation for exactly two features must be a 2x2 matrix."""

    def tearDown(self):
        plt.close('all')

    def test_two_feature_matrix_shape(self):
        ax = correlation_matrix(np.random.randn(100, 2), method='spearman')
        img = ax.images[0].get_array()
        self.assertEqual(img.shape, (2, 2))

    def test_many_feature_spearman(self):
        ax = correlation_matrix(np.random.randn(100, 5), method='spearman')
        self.assertEqual(ax.images[0].get_array().shape, (5, 5))


class TestStatisticalExtra(unittest.TestCase):
    """Complementary coverage tests for braintools.visualize._statistical."""

    def setUp(self):
        np.random.seed(0)
        self.data = np.random.randn(80, 4)
        self.labels = [f'F{i}' for i in range(4)]

    def tearDown(self):
        plt.close('all')

    # ------------------------------------------------------------------
    # correlation_matrix
    # ------------------------------------------------------------------
    def test_correlation_matrix_spearman(self):
        ax = braintools.visualize.correlation_matrix(
            self.data, method='spearman', show_values=True
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_correlation_matrix_kendall(self):
        # The 'kendall' branch builds the matrix pairwise over feature columns.
        ax = braintools.visualize.correlation_matrix(
            self.data[:, :3], method='kendall', show_values=False
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_correlation_matrix_mask_diagonal_and_no_colorbar(self):
        fig, ax = plt.subplots()
        out = braintools.visualize.correlation_matrix(
            self.data,
            labels=self.labels,
            mask_diagonal=True,
            show_colorbar=False,
            title='Corr',
            ax=ax,
        )
        self.assertIs(out, ax)

    def test_correlation_matrix_invalid_method(self):
        with self.assertRaises(ValueError):
            braintools.visualize.correlation_matrix(self.data, method='bad')

    # ------------------------------------------------------------------
    # distribution_plot
    # ------------------------------------------------------------------
    def test_distribution_plot_kde(self):
        ax = braintools.visualize.distribution_plot(
            self.data[:, 0], plot_type='kde'
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_distribution_plot_both_multiple(self):
        datasets = [self.data[:, 0], self.data[:, 1]]
        ax = braintools.visualize.distribution_plot(
            datasets,
            labels=['a', 'b'],
            plot_type='both',
            colors=['red', 'green'],
            fit_normal=True,
            title='Dist',
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_distribution_plot_on_given_ax(self):
        fig, ax = plt.subplots()
        out = braintools.visualize.distribution_plot(
            self.data[:, 2], plot_type='hist', ax=ax
        )
        self.assertIs(out, ax)

    # ------------------------------------------------------------------
    # qq_plot
    # ------------------------------------------------------------------
    def test_qq_plot_norm_default(self):
        ax = braintools.visualize.qq_plot(
            self.data[:, 0], distribution='norm'
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_qq_plot_uniform(self):
        ax = braintools.visualize.qq_plot(
            np.random.rand(50), distribution='uniform'
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_qq_plot_expon(self):
        ax = braintools.visualize.qq_plot(
            np.random.exponential(size=50), distribution='expon', title='QQ'
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_qq_plot_invalid_distribution(self):
        with self.assertRaises(ValueError):
            braintools.visualize.qq_plot(self.data[:, 0], distribution='bad')

    def test_qq_plot_reference_line_is_the_fitted_line(self):
        """The reference line must fit the plotted points, not be ``y = x``.

        Regression test: the sample quantiles are the *raw* data values, so the
        ideal line is ``y = mean + std * x``. The identity line is only correct
        for standardized data; drawing it here makes a genuinely normal sample
        look systematically off.
        """
        rng = np.random.RandomState(0)
        for loc, scale in ((0.0, 1.0), (1.0, 2.0), (50.0, 5.0)):
            data = rng.randn(150) * scale + loc
            fig, ax = plt.subplots()
            braintools.visualize.qq_plot(data, distribution='norm', ax=ax)

            points = np.asarray(ax.collections[0].get_offsets())
            line = [ln for ln in ax.lines
                    if len(np.asarray(ln.get_xdata())) == 2][0]
            xd = np.asarray(line.get_xdata(), dtype=float)
            yd = np.asarray(line.get_ydata(), dtype=float)
            drawn_slope = (yd[1] - yd[0]) / (xd[1] - xd[0])
            drawn_icpt = yd[0] - drawn_slope * xd[0]

            fit_slope, fit_icpt = np.polyfit(points[:, 0], points[:, 1], 1)
            self.assertAlmostEqual(drawn_slope, fit_slope, places=9)
            self.assertAlmostEqual(drawn_icpt, fit_icpt, places=9)
            # and it is emphatically not the identity line for shifted/scaled data
            if loc != 0.0 or scale != 1.0:
                self.assertGreater(abs(drawn_slope - 1.0) + abs(drawn_icpt), 1e-6)
            plt.close('all')

    # ------------------------------------------------------------------
    # box_plot
    # ------------------------------------------------------------------
    def test_box_plot_single_array_with_colors(self):
        ax = braintools.visualize.box_plot(
            self.data[:, 0],
            colors=['orange'],
            notch=True,
            title='Box',
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_box_plot_with_positions(self):
        fig, ax = plt.subplots()
        data_list = [self.data[:, i] for i in range(3)]
        out = braintools.visualize.box_plot(
            data_list,
            labels=['x', 'y', 'z'],
            positions=[1, 2, 3],
            colors=['r', 'g', 'b'],
            ax=ax,
        )
        self.assertIs(out, ax)

    # ------------------------------------------------------------------
    # violin_plot
    # ------------------------------------------------------------------
    def test_violin_plot_basic(self):
        data_list = [self.data[:, i] for i in range(3)]
        ax = braintools.visualize.violin_plot(
            data_list, labels=['a', 'b', 'c'], title='Violin'
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_violin_plot_single_with_colors_and_positions(self):
        fig, ax = plt.subplots()
        out = braintools.visualize.violin_plot(
            self.data[:, 0],
            colors=['purple'],
            positions=[1],
            labels=['only'],
            ax=ax,
        )
        self.assertIs(out, ax)

    # ------------------------------------------------------------------
    # scatter_matrix
    # ------------------------------------------------------------------
    def test_scatter_matrix_kde_diagonal(self):
        fig = braintools.visualize.scatter_matrix(
            self.data[:, :3], labels=self.labels[:3], diagonal='kde'
        )
        self.assertIsNotNone(fig)

    def test_scatter_matrix_single_feature_full(self):
        fig = braintools.visualize.scatter_matrix(self.data[:, :1])
        self.assertIsNotNone(fig)

    def test_scatter_matrix_no_labels(self):
        fig = braintools.visualize.scatter_matrix(self.data[:, :2])
        self.assertIsNotNone(fig)

    def test_scatter_matrix_single_feature_on_ax(self):
        fig, ax = plt.subplots()
        out = braintools.visualize.scatter_matrix(self.data[:, :1], ax=ax)
        self.assertIsNotNone(out)

    def test_scatter_matrix_on_ax_two_features(self):
        fig, ax = plt.subplots()
        out = braintools.visualize.scatter_matrix(
            self.data[:, :3], labels=self.labels[:3], ax=ax
        )
        self.assertIsNotNone(out)

    def test_scatter_matrix_on_ax_two_features_no_labels(self):
        fig, ax = plt.subplots()
        out = braintools.visualize.scatter_matrix(self.data[:, :2], ax=ax)
        self.assertIsNotNone(out)

    # ------------------------------------------------------------------
    # regression_plot
    # ------------------------------------------------------------------
    def test_regression_plot_no_fit(self):
        x = self.data[:, 0]
        y = self.data[:, 1]
        ax = braintools.visualize.regression_plot(
            x, y, fit_line=False, title='Reg'
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_regression_plot_fit_no_ci(self):
        x = self.data[:, 0]
        y = 3 * x + 1
        fig, ax = plt.subplots()
        out = braintools.visualize.regression_plot(
            x, y, fit_line=True, confidence_interval=False, ax=ax
        )
        self.assertIs(out, ax)

    def test_regression_plot_fit_with_ci(self):
        x = self.data[:, 0]
        y = 2 * x + np.random.randn(len(x)) * 0.3
        ax = braintools.visualize.regression_plot(
            x, y, fit_line=True, confidence_interval=True
        )
        self.assertIsInstance(ax, plt.Axes)

    # ------------------------------------------------------------------
    # residual_plot
    # ------------------------------------------------------------------
    def test_residual_plot(self):
        y_true = self.data[:, 0]
        y_pred = y_true + np.random.randn(len(y_true)) * 0.1
        ax = braintools.visualize.residual_plot(y_true, y_pred)
        self.assertIsInstance(ax, plt.Axes)

    def test_residual_plot_on_ax(self):
        fig, ax = plt.subplots()
        y_true = self.data[:, 0]
        y_pred = self.data[:, 1]
        out = braintools.visualize.residual_plot(y_true, y_pred, ax=ax)
        self.assertIs(out, ax)

    # ------------------------------------------------------------------
    # confusion_matrix
    # ------------------------------------------------------------------
    def test_confusion_matrix_unnormalized_show_values(self):
        y_true = np.random.randint(0, 3, 60)
        y_pred = np.random.randint(0, 3, 60)
        ax = braintools.visualize.confusion_matrix(
            y_true, y_pred, normalize=None, show_values=True
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_confusion_matrix_normalize_true(self):
        y_true = np.random.randint(0, 3, 60)
        y_pred = np.random.randint(0, 3, 60)
        ax = braintools.visualize.confusion_matrix(
            y_true, y_pred, normalize='true'
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_confusion_matrix_normalize_pred(self):
        y_true = np.random.randint(0, 2, 60)
        y_pred = np.random.randint(0, 2, 60)
        ax = braintools.visualize.confusion_matrix(
            y_true, y_pred, normalize='pred', labels=['neg', 'pos']
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_confusion_matrix_normalize_all_no_values(self):
        y_true = np.random.randint(0, 3, 60)
        y_pred = np.random.randint(0, 3, 60)
        fig, ax = plt.subplots()
        out = braintools.visualize.confusion_matrix(
            y_true, y_pred, normalize='all', show_values=False, ax=ax
        )
        self.assertIs(out, ax)

    # ------------------------------------------------------------------
    # roc_curve
    # ------------------------------------------------------------------
    def test_roc_curve(self):
        y_true = np.random.randint(0, 2, 100)
        y_scores = np.random.rand(100)
        ax = braintools.visualize.roc_curve(y_true, y_scores)
        self.assertIsInstance(ax, plt.Axes)

    def test_roc_curve_on_ax(self):
        fig, ax = plt.subplots()
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
        out = braintools.visualize.roc_curve(y_true, y_scores, ax=ax, color='green')
        self.assertIs(out, ax)

    # ------------------------------------------------------------------
    # precision_recall_curve
    # ------------------------------------------------------------------
    def test_precision_recall_curve(self):
        y_true = np.random.randint(0, 2, 100)
        y_scores = np.random.rand(100)
        ax = braintools.visualize.precision_recall_curve(y_true, y_scores)
        self.assertIsInstance(ax, plt.Axes)

    def test_precision_recall_curve_on_ax(self):
        fig, ax = plt.subplots()
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_scores = np.array([0.2, 0.7, 0.6, 0.3, 0.9, 0.1])
        out = braintools.visualize.precision_recall_curve(y_true, y_scores, ax=ax)
        self.assertIs(out, ax)

    # ------------------------------------------------------------------
    # learning_curve
    # ------------------------------------------------------------------
    def test_learning_curve_1d(self):
        train_sizes = np.array([10, 20, 30, 40])
        train_scores = np.array([0.6, 0.7, 0.8, 0.85])
        val_scores = np.array([0.5, 0.6, 0.7, 0.75])
        ax = braintools.visualize.learning_curve(
            train_sizes, train_scores, val_scores
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_learning_curve_2d(self):
        train_sizes = np.array([10, 20, 30])
        train_scores = np.random.rand(3, 5) * 0.2 + 0.7
        val_scores = np.random.rand(3, 5) * 0.2 + 0.6
        fig, ax = plt.subplots()
        out = braintools.visualize.learning_curve(
            train_sizes, train_scores, val_scores, ax=ax, title='LC'
        )
        self.assertIs(out, ax)


def test_precision_recall_average_precision_is_stepwise():
    # AP = sum_k (R_k - R_{k-1}) P_k over descending thresholds: for scores
    # [0.9, 0.8, 0.7, 0.6] with labels [1, 0, 1, 0] the operating points are
    # (R, P) = (0.5, 1), (0.5, 0.5), (1, 2/3), (1, 0.5), so AP = 0.5 + 0.5 * 2/3
    y_true = np.array([1, 0, 1, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6])
    ax = braintools.visualize.precision_recall_curve(y_true, y_scores)
    label = ax.get_legend().get_texts()[0].get_text()
    assert label == f'AP = {0.5 + 0.5 * 2 / 3:.3f}'
    plt.close('all')


def test_qq_plot_uses_standard_plotting_positions():
    data = np.random.default_rng(0).normal(0, 1, 1000)
    ax = braintools.visualize.qq_plot(data)
    x = ax.collections[0].get_offsets()[:, 0]
    from scipy import stats
    assert np.allclose(x, stats.norm.ppf((np.arange(1, 1001) - 0.5) / 1000))
    plt.close('all')


def test_group_plots_accept_2d_array_as_columns():
    X = np.random.default_rng(0).normal(size=(100, 3)) + np.array([0.0, 2.0, 4.0])
    ax = braintools.visualize.box_plot(X)
    assert len(ax.patches) == 3  # one box per column
    plt.close('all')
    ax = braintools.visualize.violin_plot(X)
    assert len([c for c in ax.collections if c.__class__.__name__ in ('PolyCollection', 'FillBetweenPolyCollection')]) == 3
    plt.close('all')
    ax = braintools.visualize.distribution_plot(X, labels=['a', 'b', 'c'])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['a', 'b', 'c']
    plt.close('all')
