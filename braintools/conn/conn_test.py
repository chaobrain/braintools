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


class TestConnectivityFunctions(unittest.TestCase):
    
    def test_random_conn(self):
        """Test random connectivity generation."""
        pre_indices, post_indices = conn.random_conn(100, 100, 0.1, seed=42)
        
        # Check that indices are valid
        self.assertTrue(np.all(pre_indices >= 0))
        self.assertTrue(np.all(pre_indices < 100))
        self.assertTrue(np.all(post_indices >= 0))
        self.assertTrue(np.all(post_indices < 100))
        
        # Check approximate connection probability
        n_connections = len(pre_indices)
        expected_connections = 100 * 100 * 0.1
        self.assertAlmostEqual(n_connections, expected_connections, delta=expected_connections * 0.2)
        
    def test_random_conn_no_self(self):
        """Test random connectivity without self-connections."""
        pre_indices, post_indices = conn.random_conn(100, 100, 0.1, seed=42, include_self=False)
        
        # Check no self-connections
        self.assertTrue(np.all(pre_indices != post_indices))
        
    def test_fixed_in_degree(self):
        """Test fixed in-degree connectivity."""
        in_degree = 10
        pre_indices, post_indices = conn.fixed_in_degree(100, 50, in_degree, seed=42)
        
        # Check that each post neuron has exactly in_degree connections
        for post_idx in range(50):
            mask = post_indices == post_idx
            self.assertEqual(np.sum(mask), in_degree)
            
    def test_fixed_out_degree(self):
        """Test fixed out-degree connectivity."""
        out_degree = 10
        pre_indices, post_indices = conn.fixed_out_degree(50, 100, out_degree, seed=42)
        
        # Check that each pre neuron has exactly out_degree connections
        for pre_idx in range(50):
            mask = pre_indices == pre_idx
            self.assertEqual(np.sum(mask), out_degree)
            
    def test_fixed_total_num(self):
        """Test fixed total number connectivity."""
        total_num = 500
        pre_indices, post_indices = conn.fixed_total_num(100, 100, total_num, seed=42)
        
        # Check exact number of connections
        self.assertEqual(len(pre_indices), total_num)
        self.assertEqual(len(post_indices), total_num)
        
    def test_all_to_all(self):
        """Test all-to-all connectivity."""
        pre_indices, post_indices = conn.all_to_all(10, 10)
        
        # Check total connections
        self.assertEqual(len(pre_indices), 10 * 10)
        
        # Check all pairs exist
        pairs = set(zip(pre_indices.tolist(), post_indices.tolist()))
        expected_pairs = {(i, j) for i in range(10) for j in range(10)}
        self.assertEqual(pairs, expected_pairs)
        
    def test_all_to_all_no_self(self):
        """Test all-to-all connectivity without self-connections."""
        pre_indices, post_indices = conn.all_to_all(10, 10, include_self=False)
        
        # Check total connections
        self.assertEqual(len(pre_indices), 10 * 10 - 10)
        
        # Check no self-connections
        self.assertTrue(np.all(pre_indices != post_indices))
        
    def test_one_to_one(self):
        """Test one-to-one connectivity."""
        pre_indices, post_indices = conn.one_to_one(100)
        
        # Check indices match
        self.assertTrue(np.all(pre_indices == post_indices))
        self.assertEqual(len(pre_indices), 100)
        
    def test_ring(self):
        """Test ring connectivity."""
        n_nodes = 10
        n_neighbors = 2
        pre_indices, post_indices = conn.ring(n_nodes, n_neighbors)
        
        # Each node should connect to n_neighbors on each side
        for i in range(n_nodes):
            mask = pre_indices == i
            self.assertEqual(np.sum(mask), n_neighbors * 2)
            
    def test_grid_2d(self):
        """Test 2D grid connectivity."""
        grid_shape = (5, 5)
        pre_indices, post_indices = conn.grid(grid_shape, n_neighbors=4)
        
        # Check connectivity patterns
        # Corner nodes have 2 neighbors, edge nodes have 3, internal nodes have 4
        n_connections = len(pre_indices)
        self.assertTrue(n_connections > 0)
        
        # Test periodic boundary
        pre_indices_p, post_indices_p = conn.grid(grid_shape, n_neighbors=4, periodic=True)
        # With periodic boundaries, all nodes have 4 neighbors
        for i in range(25):
            mask = pre_indices_p == i
            self.assertEqual(np.sum(mask), 4)
            
    def test_grid_3d(self):
        """Test 3D grid connectivity."""
        grid_shape = (3, 3, 3)
        pre_indices, post_indices = conn.grid(grid_shape, n_neighbors=6)
        
        # Check valid indices
        self.assertTrue(np.all(pre_indices >= 0))
        self.assertTrue(np.all(pre_indices < 27))
        self.assertTrue(np.all(post_indices >= 0))
        self.assertTrue(np.all(post_indices < 27))
        
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
        
    def test_small_world(self):
        """Test small-world connectivity."""
        n_nodes = 20
        n_neighbors = 4
        rewire_prob = 0.1
        
        pre_indices, post_indices = conn.small_world(
            n_nodes, n_neighbors, rewire_prob, seed=42
        )
        
        # Check valid indices
        self.assertTrue(np.all(pre_indices >= 0))
        self.assertTrue(np.all(pre_indices < n_nodes))
        self.assertTrue(np.all(post_indices >= 0))
        self.assertTrue(np.all(post_indices < n_nodes))
        
        # Check approximate number of edges
        n_edges = len(pre_indices)
        expected_edges = n_nodes * n_neighbors
        self.assertEqual(n_edges, expected_edges)
        
    def test_scale_free(self):
        """Test scale-free connectivity."""
        n_nodes = 20
        m_edges = 3
        
        pre_indices, post_indices = conn.scale_free(n_nodes, m_edges, seed=42)
        
        # Check valid indices
        self.assertTrue(np.all(pre_indices >= 0))
        self.assertTrue(np.all(pre_indices < n_nodes))
        self.assertTrue(np.all(post_indices >= 0))
        self.assertTrue(np.all(post_indices < n_nodes))
        
        # Check that we have connections
        self.assertTrue(len(pre_indices) > 0)
        
    def test_multidimensional_sizes(self):
        """Test connectivity with multi-dimensional population sizes."""
        # Test with 2D population sizes
        pre_size = (10, 10)
        post_size = (5, 5)
        
        pre_indices, post_indices = conn.random_conn(pre_size, post_size, 0.1, seed=42)
        
        # Check indices are within bounds
        self.assertTrue(np.all(pre_indices >= 0))
        self.assertTrue(np.all(pre_indices < 100))
        self.assertTrue(np.all(post_indices >= 0))
        self.assertTrue(np.all(post_indices < 25))


if __name__ == '__main__':
    unittest.main()