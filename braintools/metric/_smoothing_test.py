# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

import braintools


class SmoothLabelsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.ts = np.array([[0., 1., 0.], [1., 0., 0.]], dtype=np.float32)
        # compute expected outputs in numpy.
        self.exp_alpha_zero = self.ts
        self.exp_alpha_zero_point_one = 0.9 * self.ts + 0.1 / self.ts.shape[-1]
        self.exp_alpha_one = jnp.ones_like(self.ts) / self.ts.shape[-1]

    def test_scalar(self):
        """Tests for a full batch."""
        np.testing.assert_allclose(braintools.metric.smooth_labels(self.ts[0], 0.), self.exp_alpha_zero[0], atol=1e-4)
        np.testing.assert_allclose(braintools.metric.smooth_labels(self.ts[0], 0.1), self.exp_alpha_zero_point_one[0],
                                   atol=1e-4)
        np.testing.assert_allclose(braintools.metric.smooth_labels(self.ts[0], 1.), self.exp_alpha_one[0], atol=1e-4)

    def test_batched(self):
        """Tests for a full batch."""
        np.testing.assert_allclose(braintools.metric.smooth_labels(self.ts, 0.), self.exp_alpha_zero, atol=1e-4)
        np.testing.assert_allclose(braintools.metric.smooth_labels(self.ts, 0.1), self.exp_alpha_zero_point_one,
                                   atol=1e-4)
        np.testing.assert_allclose(braintools.metric.smooth_labels(self.ts, 1.), self.exp_alpha_one, atol=1e-4)

    def test_smooth_labels_assertion_error(self):
        # Integer labels are rejected with an explicit ``TypeError`` (a real
        # exception that survives ``python -O``, unlike a bare ``assert``).
        with self.assertRaises(TypeError):
            braintools.metric.smooth_labels(jnp.array([[1, 0, 0], [0, 1, 0]]), 0.1)

    def test_alpha_very_small_positive(self):
        """Tests for a very small positive alpha close to zero."""
        very_small_alpha = 1e-10
        expected_output = (1.0 - very_small_alpha) * self.ts + very_small_alpha / self.ts.shape[-1]
        np.testing.assert_allclose(
            braintools.metric.smooth_labels(self.ts, very_small_alpha), expected_output, atol=1e-4)

    def test_rows_sum_to_one_for_valid_distribution(self):
        """B22: rows sum to 1 when the input rows are valid distributions."""
        labels = jnp.eye(5)
        for alpha in (0.0, 0.05, 0.1, 0.5, 1.0):
            smoothed = braintools.metric.smooth_labels(labels, alpha=alpha)
            np.testing.assert_allclose(jnp.sum(smoothed, axis=1), jnp.ones(5), atol=1e-5)
            self.assertTrue(bool(jnp.all(smoothed >= 0)))

    def test_rows_do_not_sum_to_one_for_invalid_input(self):
        """B22: precondition - invalid input rows do not yield rows summing to 1."""
        # rows sum to 2.0, not 1.0
        labels = jnp.array([[1.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
        smoothed = braintools.metric.smooth_labels(labels, alpha=0.2)
        row_sums = jnp.sum(smoothed, axis=1)
        # (1-0.2)*2 + 0.2 = 1.8 != 1.0
        np.testing.assert_allclose(row_sums, jnp.array([1.8, 1.8]), atol=1e-5)

    def test_alpha_out_of_range_raises(self):
        """B23: alpha must be validated to lie in [0, 1]."""
        labels = jnp.eye(3)
        with self.assertRaises(ValueError):
            braintools.metric.smooth_labels(labels, alpha=-0.1)
        with self.assertRaises(ValueError):
            braintools.metric.smooth_labels(labels, alpha=1.5)

    def test_alpha_boundary_values_ok(self):
        """B23: alpha exactly at 0 and 1 must be accepted."""
        labels = jnp.eye(3)
        out0 = braintools.metric.smooth_labels(labels, alpha=0.0)
        np.testing.assert_allclose(out0, labels, atol=1e-6)
        out1 = braintools.metric.smooth_labels(labels, alpha=1.0)
        np.testing.assert_allclose(out1, jnp.ones_like(labels) / 3.0, atol=1e-6)

    def test_alpha_one_is_uniform(self):
        """alpha=1.0 yields a uniform distribution over classes."""
        labels = jnp.eye(4)
        out = braintools.metric.smooth_labels(labels, alpha=1.0)
        np.testing.assert_allclose(out, jnp.full((4, 4), 0.25), atol=1e-6)
