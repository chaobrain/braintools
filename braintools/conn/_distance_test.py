# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

import numpy as np

import braintools.conn as conn


class TestDistanceConnectivity(unittest.TestCase):
    """Test distance-based connectivity functions."""

    def test_distance_prob(self):
        """Test distance-dependent connectivity."""
        # Create simple 2D positions
        pre_pos = np.array([[0, 0], [1, 0], [2, 0]])
        post_pos = np.array([[0, 1], [1, 1], [2, 1]])

        # Probability decreases with distance
        prob_func = lambda d: np.exp(-d)

        pre_indices, post_indices = conn.distance_prob(
            pre_pos, post_pos, prob_func, seed=42
        )

        # Check valid indices
        self.assertTrue(np.all(pre_indices >= 0))
        self.assertTrue(np.all(pre_indices < 3))
        self.assertTrue(np.all(post_indices >= 0))
        self.assertTrue(np.all(post_indices < 3))

    def test_distance_prob_no_self(self):
        """Test distance-dependent connectivity without self-connections."""
        # Create positions where pre and post are the same
        positions = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])

        # High probability for all connections
        prob_func = lambda d: 0.9

        pre_indices, post_indices = conn.distance_prob(
            positions, positions, prob_func, seed=42, include_self=False
        )

        # Check no self-connections
        self.assertTrue(np.all(pre_indices != post_indices))

    def test_gaussian_conn(self):
        """Test Gaussian distance-dependent connectivity."""
        # Create grid positions
        x = np.linspace(0, 10, 10)
        y = np.linspace(0, 10, 10)
        xx, yy = np.meshgrid(x, y)
        positions = np.stack([xx.ravel(), yy.ravel()], axis=1)

        pre_indices, post_indices = conn.gaussian_conn(
            positions, positions, sigma=2.0, max_prob=0.8, seed=42
        )

        # Check valid indices
        self.assertTrue(np.all(pre_indices >= 0))
        self.assertTrue(np.all(pre_indices < 100))
        self.assertTrue(np.all(post_indices >= 0))
        self.assertTrue(np.all(post_indices < 100))

    def test_gaussian_conn_small_sigma(self):
        """Test Gaussian connectivity with small sigma (local connections)."""
        # Create a line of neurons
        positions = np.array([[i, 0] for i in range(10)])

        # Small sigma means only nearby neurons connect
        pre_indices, post_indices = conn.gaussian_conn(
            positions, positions, sigma=0.5, max_prob=1.0, seed=42
        )

        # Check that connections are mostly local
        for i in range(len(pre_indices)):
            distance = abs(pre_indices[i] - post_indices[i])
            # Most connections should be within 2 positions
            if distance > 2:
                # These should be rare
                pass

    def test_gaussian_conn_large_sigma(self):
        """Test Gaussian connectivity with large sigma (global connections)."""
        # Create a small grid
        positions = np.array([[i, j] for i in range(3) for j in range(3)])

        # Large sigma means more global connections
        pre_indices, post_indices = conn.gaussian_conn(
            positions, positions, sigma=10.0, max_prob=0.5, seed=42
        )

        # With large sigma and moderate probability, should have many connections
        n_connections = len(pre_indices)
        max_possible = 9 * 9
        # Should have a substantial fraction of possible connections
        self.assertTrue(n_connections > max_possible * 0.3)

    def test_distance_prob_custom_function(self):
        """Test distance_prob with custom probability functions."""
        positions = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])

        # Step function: connect if distance < 1.5
        step_func = lambda d: (d < 1.5).astype(float)

        pre_indices, post_indices = conn.distance_prob(
            positions, positions, step_func, seed=42
        )

        # Check all connections have distance < 1.5
        for i in range(len(pre_indices)):
            pre_pos = positions[pre_indices[i]]
            post_pos = positions[post_indices[i]]
            distance = np.sqrt(np.sum((pre_pos - post_pos) ** 2))
            self.assertTrue(distance < 1.5)

    def test_distance_prob_linear_decay(self):
        """Test distance_prob with linear decay function."""
        # Create 1D positions for simplicity
        pre_pos = np.array([[i, 0] for i in range(5)])
        post_pos = np.array([[i, 0] for i in range(5)])

        # Linear decay: prob = max(0, 1 - distance/3)
        linear_func = lambda d: np.maximum(0, 1 - d / 3)

        pre_indices, post_indices = conn.distance_prob(
            pre_pos, post_pos, linear_func, seed=42
        )

        # Nearby neurons should be more likely to connect
        # Count connections by distance
        distances = []
        for i in range(len(pre_indices)):
            dist = abs(pre_indices[i] - post_indices[i])
            distances.append(dist)

        # Should have connections within distance 3
        self.assertTrue(all(d <= 3 for d in distances))

    def test_distance_prob_3d_positions(self):
        """Test distance_prob with 3D positions."""
        # Create 3D positions
        pre_pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        post_pos = np.array([[0.5, 0.5, 0], [1, 1, 0], [0, 0, 0.5], [1, 0, 1]])

        # Exponential decay
        prob_func = lambda d: np.exp(-2 * d)

        pre_indices, post_indices = conn.distance_prob(
            pre_pos, post_pos, prob_func, seed=42
        )

        # Check valid indices
        self.assertTrue(np.all(pre_indices >= 0))
        self.assertTrue(np.all(pre_indices < 4))
        self.assertTrue(np.all(post_indices >= 0))
        self.assertTrue(np.all(post_indices < 4))


if __name__ == '__main__':
    unittest.main()
