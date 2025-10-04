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
