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

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braintools

pcs = braintools.metric.pairwise_cosine_similarity
pcd = braintools.metric.pairwise_cosine_distance


class TestPairwiseCosineSimilarity(unittest.TestCase):
    def test_self_similarity_diagonal_is_one(self):
        X = jnp.array([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        sim = pcs(X)
        self.assertEqual(sim.shape, (3, 3))
        np.testing.assert_allclose(jnp.diag(sim), jnp.ones(3), atol=1e-6)

    def test_known_values(self):
        X = jnp.array([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        sim = pcs(X)
        r = 1.0 / np.sqrt(2.0)
        expected = np.array([[1., 0., r], [0., 1., r], [r, r, 1.]])
        np.testing.assert_allclose(sim, expected, atol=1e-6)

    def test_cross_shape(self):
        X = jnp.array([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        Y = jnp.array([[1., 1., 1.], [0., 0., 1.]])
        self.assertEqual(pcs(X, Y).shape, (3, 2))

    def test_symmetry(self):
        rng = np.random.default_rng(0)
        X = jnp.asarray(rng.standard_normal((5, 4)))
        sim = pcs(X)
        np.testing.assert_allclose(sim, sim.T, atol=1e-6)

    def test_bounded_range(self):
        rng = np.random.default_rng(1)
        X = jnp.asarray(rng.standard_normal((6, 3)))
        Y = jnp.asarray(rng.standard_normal((4, 3)))
        sim = pcs(X, Y)
        self.assertTrue(bool(jnp.all(sim <= 1.0 + 1e-5)))
        self.assertTrue(bool(jnp.all(sim >= -1.0 - 1e-5)))

    def test_opposite_vectors(self):
        X = jnp.array([[1., 0.], [-1., 0.]])
        sim = pcs(X)
        np.testing.assert_allclose(sim[0, 1], -1.0, atol=1e-6)

    def test_zero_vector_returns_zero_not_nan(self):
        X = jnp.array([[0., 0., 0.], [1., 2., 3.]])
        sim = pcs(X)
        self.assertFalse(bool(jnp.any(jnp.isnan(sim))))
        # All pairs touching the zero vector must be exactly 0.
        np.testing.assert_allclose(sim[0, :], jnp.zeros(2), atol=1e-7)
        np.testing.assert_allclose(sim[:, 0], jnp.zeros(2), atol=1e-7)

    def test_gradient_finite_with_zero_vector(self):
        X = jnp.array([[0., 0., 0.], [1., 2., 3.], [4., 5., 6.]])
        g = jax.grad(lambda x: jnp.sum(pcs(x)))(X)
        self.assertFalse(bool(jnp.any(jnp.isnan(g))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

    def test_quantity_inputs_match_magnitude(self):
        X = jnp.array([[1., 2., 3.], [4., 5., 6.], [0., 1., 0.]])
        sim_plain = pcs(X)
        sim_units = pcs(X * u.mV)
        # cosine similarity is scale/unit invariant and dimensionless
        np.testing.assert_allclose(sim_plain, sim_units, atol=1e-6)

    def test_scale_invariance(self):
        rng = np.random.default_rng(2)
        X = jnp.asarray(rng.standard_normal((4, 3)))
        np.testing.assert_allclose(pcs(X), pcs(100.0 * X), atol=1e-5)

    def test_jit_matches_eager(self):
        X = jnp.array([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        np.testing.assert_allclose(jax.jit(pcs)(X), pcs(X), atol=1e-6)

    def test_vmap_over_batch(self):
        rng = np.random.default_rng(3)
        Xb = jnp.asarray(rng.standard_normal((2, 5, 3)))
        out = jax.vmap(pcs)(Xb)
        self.assertEqual(out.shape, (2, 5, 5))


class TestPairwiseCosineDistance(unittest.TestCase):
    def test_is_one_minus_similarity(self):
        rng = np.random.default_rng(10)
        X = jnp.asarray(rng.standard_normal((5, 4)))
        Y = jnp.asarray(rng.standard_normal((3, 4)))
        np.testing.assert_allclose(pcd(X, Y), 1.0 - pcs(X, Y), atol=1e-6)

    def test_self_distance_diagonal_is_zero(self):
        X = jnp.array([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        dist = pcd(X)
        self.assertEqual(dist.shape, (3, 3))
        np.testing.assert_allclose(jnp.diag(dist), jnp.zeros(3), atol=1e-6)

    def test_opposite_vectors_distance_is_two(self):
        X = jnp.array([[1., 0.], [-1., 0.]])
        dist = pcd(X)
        np.testing.assert_allclose(dist[0, 1], 2.0, atol=1e-6)

    def test_zero_vector_distance_is_one(self):
        # A pair touching a zero vector has similarity 0 -> distance 1.
        X = jnp.array([[0., 0., 0.], [1., 2., 3.]])
        dist = pcd(X)
        self.assertFalse(bool(jnp.any(jnp.isnan(dist))))
        np.testing.assert_allclose(dist[0, 1], 1.0, atol=1e-7)

    def test_cross_shape(self):
        X = jnp.asarray(np.random.default_rng(11).standard_normal((3, 4)))
        Y = jnp.asarray(np.random.default_rng(12).standard_normal((2, 4)))
        self.assertEqual(pcd(X, Y).shape, (3, 2))


if __name__ == '__main__':
    unittest.main()
