# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from braintools.metric import _smoothing


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
        np.testing.assert_allclose(_smoothing.smooth_labels(self.ts[0], 0.), self.exp_alpha_zero[0], atol=1e-4)
        np.testing.assert_allclose(_smoothing.smooth_labels(self.ts[0], 0.1), self.exp_alpha_zero_point_one[0],
                                   atol=1e-4)
        np.testing.assert_allclose(_smoothing.smooth_labels(self.ts[0], 1.), self.exp_alpha_one[0], atol=1e-4)

    def test_batched(self):
        """Tests for a full batch."""
        np.testing.assert_allclose(_smoothing.smooth_labels(self.ts, 0.), self.exp_alpha_zero, atol=1e-4)
        np.testing.assert_allclose(_smoothing.smooth_labels(self.ts, 0.1), self.exp_alpha_zero_point_one, atol=1e-4)
        np.testing.assert_allclose(_smoothing.smooth_labels(self.ts, 1.), self.exp_alpha_one, atol=1e-4)

    def test_smooth_labels_assertion_error(self):
        with self.assertRaises(AssertionError):
            _smoothing.smooth_labels(jnp.array([[1, 0, 0], [0, 1, 0]]), 0.1)

    def test_alpha_very_small_positive(self):
        """Tests for a very small positive alpha close to zero."""
        very_small_alpha = 1e-10
        expected_output = (1.0 - very_small_alpha) * self.ts + very_small_alpha / self.ts.shape[-1]
        np.testing.assert_allclose(
            _smoothing.smooth_labels(self.ts, very_small_alpha), expected_output, atol=1e-4)
