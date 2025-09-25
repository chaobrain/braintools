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


class TestRegularConnectivity(unittest.TestCase):
    """Test regular/structured connectivity functions."""
    
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
        
    def test_all_to_all_different_sizes(self):
        """Test all-to-all with different pre and post sizes."""
        pre_indices, post_indices = conn.all_to_all(5, 8)
        
        # Check total connections
        self.assertEqual(len(pre_indices), 5 * 8)
        
        # Check indices are in correct range
        self.assertTrue(np.all(pre_indices >= 0))
        self.assertTrue(np.all(pre_indices < 5))
        self.assertTrue(np.all(post_indices >= 0))
        self.assertTrue(np.all(post_indices < 8))
        
    def test_one_to_one(self):
        """Test one-to-one connectivity."""
        pre_indices, post_indices = conn.one_to_one(100)
        
        # Check indices match
        self.assertTrue(np.all(pre_indices == post_indices))
        self.assertEqual(len(pre_indices), 100)
        
    def test_one_to_one_multidimensional(self):
        """Test one-to-one with multi-dimensional size."""
        size = (10, 10)
        pre_indices, post_indices = conn.one_to_one(size)
        
        # Check indices match and correct total
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
            
    def test_ring_single_neighbor(self):
        """Test ring with single neighbor."""
        n_nodes = 8
        pre_indices, post_indices = conn.ring(n_nodes, n_neighbors=1)
        
        # Each node connects to 2 neighbors (one on each side)
        for i in range(n_nodes):
            mask = pre_indices == i
            self.assertEqual(np.sum(mask), 2)
            
            # Check specific connections
            connections = post_indices[mask]
            expected = [(i + 1) % n_nodes, (i - 1) % n_nodes]
            self.assertCountEqual(connections.tolist(), expected)
            
    def test_grid_2d(self):
        """Test 2D grid connectivity."""
        grid_shape = (5, 5)
        pre_indices, post_indices = conn.grid(grid_shape, n_neighbors=4)
        
        # Check connectivity patterns
        # Corner nodes have 2 neighbors, edge nodes have 3, internal nodes have 4
        n_connections = len(pre_indices)
        self.assertTrue(n_connections > 0)
        
        # Count connections per node
        for i in range(25):
            mask = pre_indices == i
            count = np.sum(mask)
            # Corner nodes: 2 neighbors
            # Edge nodes: 3 neighbors  
            # Internal nodes: 4 neighbors
            self.assertTrue(count >= 2 and count <= 4)
            
    def test_grid_2d_periodic(self):
        """Test 2D grid with periodic boundaries."""
        grid_shape = (5, 5)
        pre_indices_p, post_indices_p = conn.grid(grid_shape, n_neighbors=4, periodic=True)
        
        # With periodic boundaries, all nodes have 4 neighbors
        for i in range(25):
            mask = pre_indices_p == i
            self.assertEqual(np.sum(mask), 4)
            
    def test_grid_2d_8neighbors(self):
        """Test 2D grid with 8-connectivity."""
        grid_shape = (3, 3)
        pre_indices, post_indices = conn.grid(grid_shape, n_neighbors=8)
        
        # Check valid indices
        self.assertTrue(np.all(pre_indices >= 0))
        self.assertTrue(np.all(pre_indices < 9))
        self.assertTrue(np.all(post_indices >= 0))
        self.assertTrue(np.all(post_indices < 9))
        
        # Center node should have 8 connections
        center = 4  # Center of 3x3 grid
        mask = pre_indices == center
        self.assertEqual(np.sum(mask), 8)
        
    def test_grid_3d(self):
        """Test 3D grid connectivity."""
        grid_shape = (3, 3, 3)
        pre_indices, post_indices = conn.grid(grid_shape, n_neighbors=6)
        
        # Check valid indices
        self.assertTrue(np.all(pre_indices >= 0))
        self.assertTrue(np.all(pre_indices < 27))
        self.assertTrue(np.all(post_indices >= 0))
        self.assertTrue(np.all(post_indices < 27))
        
        # Center node should have 6 connections in 3D
        center = 13  # Center of 3x3x3 grid
        mask = pre_indices == center
        self.assertEqual(np.sum(mask), 6)
        
    def test_grid_1d(self):
        """Test 1D grid connectivity (equivalent to a line)."""
        grid_shape = (10,)
        pre_indices, post_indices = conn.grid(grid_shape)
        
        # Check connectivity
        # End nodes have 1 neighbor, internal nodes have 2
        for i in range(10):
            mask = pre_indices == i
            count = np.sum(mask)
            if i == 0 or i == 9:
                self.assertEqual(count, 1)  # End nodes
            else:
                self.assertEqual(count, 2)  # Internal nodes


if __name__ == '__main__':
    unittest.main()