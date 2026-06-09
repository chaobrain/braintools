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


if __name__ == '__main__':
    unittest.main()
