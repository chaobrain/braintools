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


class TestRandomConnectivity(unittest.TestCase):
    """Test random connectivity functions."""
    
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
        
    def test_fixed_prob(self):
        """Test fixed probability connectivity (alias for random_conn)."""
        pre_indices1, post_indices1 = conn.random_conn(50, 50, 0.2, seed=123)
        pre_indices2, post_indices2 = conn.fixed_prob(50, 50, 0.2, seed=123)
        
        # Should produce identical results
        np.testing.assert_array_equal(pre_indices1, pre_indices2)
        np.testing.assert_array_equal(post_indices1, post_indices2)
        
    def test_fixed_in_degree(self):
        """Test fixed in-degree connectivity."""
        in_degree = 10
        pre_indices, post_indices = conn.fixed_in_degree(100, 50, in_degree, seed=42)
        
        # Check that each post neuron has exactly in_degree connections
        for post_idx in range(50):
            mask = post_indices == post_idx
            self.assertEqual(np.sum(mask), in_degree)
            
    def test_fixed_in_degree_no_self(self):
        """Test fixed in-degree without self-connections."""
        in_degree = 5
        pre_indices, post_indices = conn.fixed_in_degree(50, 50, in_degree, seed=42, include_self=False)
        
        # Check no self-connections
        for i in range(len(pre_indices)):
            if post_indices[i] < 50:  # Only check when indices could be equal
                self.assertNotEqual(pre_indices[i], post_indices[i])
                
    def test_fixed_out_degree(self):
        """Test fixed out-degree connectivity."""
        out_degree = 10
        pre_indices, post_indices = conn.fixed_out_degree(50, 100, out_degree, seed=42)
        
        # Check that each pre neuron has exactly out_degree connections
        for pre_idx in range(50):
            mask = pre_indices == pre_idx
            self.assertEqual(np.sum(mask), out_degree)
            
    def test_fixed_out_degree_no_self(self):
        """Test fixed out-degree without self-connections."""
        out_degree = 5
        pre_indices, post_indices = conn.fixed_out_degree(50, 50, out_degree, seed=42, include_self=False)
        
        # Check no self-connections  
        for i in range(len(pre_indices)):
            if pre_indices[i] < 50:  # Only check when indices could be equal
                self.assertNotEqual(pre_indices[i], post_indices[i])
                
    def test_fixed_total_num(self):
        """Test fixed total number connectivity."""
        total_num = 500
        pre_indices, post_indices = conn.fixed_total_num(100, 100, total_num, seed=42)
        
        # Check exact number of connections
        self.assertEqual(len(pre_indices), total_num)
        self.assertEqual(len(post_indices), total_num)
        
    def test_fixed_total_num_no_self(self):
        """Test fixed total number without self-connections."""
        total_num = 200
        pre_indices, post_indices = conn.fixed_total_num(50, 50, total_num, seed=42, include_self=False)
        
        # Check no self-connections
        self.assertTrue(np.all(pre_indices != post_indices))
        self.assertEqual(len(pre_indices), total_num)
        
    def test_fixed_total_num_error(self):
        """Test fixed total number with too many requested connections."""
        with self.assertRaises(ValueError):
            # Requesting more connections than possible
            conn.fixed_total_num(5, 5, 30, include_self=False)
            
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