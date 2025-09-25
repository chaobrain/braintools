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


class TestStatisticalConnectivity(unittest.TestCase):
    """Test statistical model-based connectivity patterns."""

    def test_erdos_renyi_fixed_edges(self):
        """Test Erdős-Rényi with fixed number of edges."""
        n_nodes = 20
        n_edges = 50

        # Directed graph
        pre, post = conn.erdos_renyi(
            n_nodes=n_nodes,
            n_edges=n_edges,
            directed=True,
            seed=42
        )

        self.assertEqual(len(pre), n_edges)
        self.assertEqual(len(post), n_edges)
        self.assertTrue(np.all(pre >= 0))
        self.assertTrue(np.all(pre < n_nodes))
        self.assertTrue(np.all(post >= 0))
        self.assertTrue(np.all(post < n_nodes))

        # Undirected graph
        pre_u, post_u = conn.erdos_renyi(
            n_nodes=n_nodes,
            n_edges=n_edges,
            directed=False,
            seed=42
        )

        # Undirected graph has edges in both directions
        self.assertEqual(len(pre_u), n_edges * 2)
        self.assertEqual(len(post_u), n_edges * 2)

    def test_erdos_renyi_edge_prob(self):
        """Test Erdős-Rényi with edge probability."""
        n_nodes = 30
        edge_prob = 0.1

        pre, post = conn.erdos_renyi(
            n_nodes=n_nodes,
            edge_prob=edge_prob,
            directed=True,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # Check approximate edge count
        expected_edges = n_nodes * (n_nodes - 1) * edge_prob
        self.assertGreater(len(pre), expected_edges * 0.5)
        self.assertLess(len(pre), expected_edges * 1.5)

    def test_erdos_renyi_errors(self):
        """Test Erdős-Rényi error cases."""
        # Neither n_edges nor edge_prob specified
        with self.assertRaises(ValueError):
            conn.erdos_renyi(n_nodes=10)

        # Too many edges requested
        with self.assertRaises(ValueError):
            conn.erdos_renyi(n_nodes=10, n_edges=100, directed=False)

    def test_stochastic_block_model(self):
        """Test stochastic block model."""
        block_sizes = [10, 15, 20]
        n_blocks = len(block_sizes)

        # Create probability matrix
        prob_matrix = np.array([
            [0.5, 0.1, 0.05],
            [0.1, 0.4, 0.1],
            [0.05, 0.1, 0.3]
        ])

        pre, post = conn.stochastic_block_model(
            block_sizes=block_sizes,
            prob_matrix=prob_matrix,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        total_nodes = sum(block_sizes)
        self.assertTrue(np.all(pre >= 0))
        self.assertTrue(np.all(pre < total_nodes))
        self.assertTrue(np.all(post >= 0))
        self.assertTrue(np.all(post < total_nodes))

        # Test shape mismatch error
        with self.assertRaises(ValueError):
            wrong_prob = np.array([[0.5, 0.1], [0.1, 0.4]])
            conn.stochastic_block_model(block_sizes, wrong_prob)

    def test_configuration_model_undirected(self):
        """Test configuration model for undirected graph."""
        # Create valid degree sequence (sum must be even)
        degrees = np.array([3, 3, 2, 2, 2, 2])

        pre, post = conn.configuration_model(
            in_degrees=degrees,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # Check no self-loops
        self.assertTrue(np.all(pre != post))

        # Test odd sum error
        odd_degrees = np.array([3, 3, 2, 2, 1])
        with self.assertRaises(ValueError):
            conn.configuration_model(in_degrees=odd_degrees)

    def test_configuration_model_directed(self):
        """Test configuration model for directed graph."""
        in_degrees = np.array([2, 3, 1, 2])
        out_degrees = np.array([3, 2, 2, 1])

        pre, post = conn.configuration_model(
            in_degrees=in_degrees,
            out_degrees=out_degrees,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # Test sum mismatch error
        bad_out = np.array([3, 2, 2, 2])
        with self.assertRaises(ValueError):
            conn.configuration_model(in_degrees, bad_out)

    def test_power_law_degree(self):
        """Test power-law degree distribution."""
        n_nodes = 50

        pre, post = conn.power_law_degree(
            n_nodes=n_nodes,
            gamma=2.5,
            min_degree=1,
            max_degree=10,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # Check all indices valid
        self.assertTrue(np.all(pre >= 0))
        self.assertTrue(np.all(pre < n_nodes))
        self.assertTrue(np.all(post >= 0))
        self.assertTrue(np.all(post < n_nodes))

        # Test with default max_degree
        pre2, post2 = conn.power_law_degree(
            n_nodes=100,
            gamma=3.0,
            min_degree=2,
            seed=42
        )

        self.assertIsInstance(pre2, np.ndarray)
        self.assertIsInstance(post2, np.ndarray)

    def test_lognormal_degree(self):
        """Test log-normal degree distribution."""
        n_nodes = 40

        pre, post = conn.lognormal_degree(
            n_nodes=n_nodes,
            mean=2.0,
            std=1.0,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # Check all indices valid
        self.assertTrue(np.all(pre >= 0))
        self.assertTrue(np.all(pre < n_nodes))
        self.assertTrue(np.all(post >= 0))
        self.assertTrue(np.all(post < n_nodes))

    def test_exponential_random_graph(self):
        """Test exponential random graph model."""
        n_nodes = 15

        pre, post = conn.exponential_random_graph(
            n_nodes=n_nodes,
            edge_weight=1.0,
            triangle_weight=0.5,
            star_weight=0.2,
            n_samples=100,  # Small for testing
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # Should produce some edges
        self.assertGreater(len(pre), 0)

        # Check indices
        self.assertTrue(np.all(pre >= 0))
        self.assertTrue(np.all(pre < n_nodes))
        self.assertTrue(np.all(post >= 0))
        self.assertTrue(np.all(post < n_nodes))

    def test_degree_sequence_configuration(self):
        """Test degree sequence with configuration method."""
        degrees = [4, 3, 3, 2, 2, 2]

        pre, post = conn.degree_sequence(
            degrees=degrees,
            method='configuration',
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

    def test_degree_sequence_havel_hakimi(self):
        """Test degree sequence with Havel-Hakimi method."""
        degrees = [3, 3, 2, 2, 2]

        pre, post = conn.degree_sequence(
            degrees=degrees,
            method='havel-hakimi',
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # Test non-graphical sequence
        bad_degrees = [5, 1, 1, 1, 1]  # Can't connect node with degree 5
        with self.assertRaises(ValueError):
            conn.degree_sequence(bad_degrees, method='havel-hakimi')

        # Test unknown method
        with self.assertRaises(ValueError):
            conn.degree_sequence(degrees, method='unknown')

    def test_expected_degree_model(self):
        """Test expected degree model."""
        expected_degrees = np.array([5.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0])

        pre, post = conn.expected_degree_model(
            expected_degrees=expected_degrees,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        n_nodes = len(expected_degrees)
        self.assertTrue(np.all(pre >= 0))
        self.assertTrue(np.all(pre < n_nodes))
        self.assertTrue(np.all(post >= 0))
        self.assertTrue(np.all(post < n_nodes))

    def test_statistical_edge_cases(self):
        """Test edge cases for statistical models."""
        # Single node
        pre, post = conn.erdos_renyi(
            n_nodes=1,
            edge_prob=0.5,
            directed=True,
            seed=42
        )
        self.assertEqual(len(pre), 0)
        self.assertEqual(len(post), 0)

        # Zero probability
        pre, post = conn.erdos_renyi(
            n_nodes=10,
            edge_prob=0.0,
            directed=True,
            seed=42
        )
        self.assertEqual(len(pre), 0)
        self.assertEqual(len(post), 0)

        # Empty blocks
        block_sizes = []
        prob_matrix = np.array([])
        pre, post = conn.stochastic_block_model(
            block_sizes=block_sizes,
            prob_matrix=prob_matrix.reshape(0, 0),
            seed=42
        )
        self.assertEqual(len(pre), 0)
        self.assertEqual(len(post), 0)

        # Zero expected degrees
        expected_degrees = np.zeros(5)
        pre, post = conn.expected_degree_model(
            expected_degrees=expected_degrees,
            seed=42
        )
        self.assertEqual(len(pre), 0)
        self.assertEqual(len(post), 0)


if __name__ == '__main__':
    unittest.main()
