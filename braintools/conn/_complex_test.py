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


class TestComplexNetworks(unittest.TestCase):
    """Test complex network connectivity functions."""

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

    def test_small_world_no_rewiring(self):
        """Test small-world with no rewiring (regular ring lattice)."""
        n_nodes = 10
        n_neighbors = 2
        rewire_prob = 0.0  # No rewiring

        pre_indices, post_indices = conn.small_world(
            n_nodes, n_neighbors, rewire_prob, seed=42
        )

        # Should be a regular ring lattice
        # Each node connects to n_neighbors/2 on each side
        for i in range(n_nodes):
            mask = pre_indices == i
            connections = post_indices[mask]
            # Should connect to nearest neighbors
            expected = []
            for j in range(1, n_neighbors // 2 + 1):
                expected.append((i + j) % n_nodes)
                expected.append((i - j) % n_nodes)
            self.assertCountEqual(connections.tolist(), expected)

    def test_small_world_full_rewiring(self):
        """Test small-world with full rewiring."""
        n_nodes = 20
        n_neighbors = 4
        rewire_prob = 1.0  # Full rewiring

        pre_indices, post_indices = conn.small_world(
            n_nodes, n_neighbors, rewire_prob, seed=42
        )

        # Should still have same number of edges
        n_edges = len(pre_indices)
        expected_edges = n_nodes * n_neighbors
        self.assertEqual(n_edges, expected_edges)

        # Connections should be more random than regular lattice
        # (difficult to test precisely due to randomness)

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

    def test_scale_free_degree_distribution(self):
        """Test that scale-free has appropriate degree distribution."""
        n_nodes = 100
        m_edges = 3

        pre_indices, post_indices = conn.scale_free(n_nodes, m_edges, seed=42)

        # Calculate degree for each node
        degrees = np.zeros(n_nodes)
        for i in range(len(pre_indices)):
            degrees[pre_indices[i]] += 1
            degrees[post_indices[i]] += 1

        # Remove duplicates (since edges are bidirectional)
        degrees = degrees / 2

        # Check minimum degree (should be at least m_edges for most nodes)
        self.assertTrue(np.min(degrees) >= m_edges - 1)

        # Check for presence of high-degree nodes (hubs)
        # In scale-free networks, some nodes should have much higher degree
        max_degree = np.max(degrees)
        mean_degree = np.mean(degrees)
        self.assertTrue(max_degree > 2 * mean_degree)

    def test_scale_free_initial_network(self):
        """Test scale-free with small initial network."""
        n_nodes = 10
        m_edges = 2

        pre_indices, post_indices = conn.scale_free(n_nodes, m_edges, seed=42)

        # Should create a valid network
        self.assertTrue(len(pre_indices) > 0)

        # All indices should be valid
        self.assertTrue(np.all(pre_indices >= 0))
        self.assertTrue(np.all(pre_indices < n_nodes))
        self.assertTrue(np.all(post_indices >= 0))
        self.assertTrue(np.all(post_indices < n_nodes))

    def test_scale_free_error(self):
        """Test scale-free with invalid parameters."""
        with self.assertRaises(ValueError):
            # m_edges must be less than n_nodes
            conn.scale_free(5, 5, seed=42)

        with self.assertRaises(ValueError):
            # m_edges must be less than n_nodes
            conn.scale_free(5, 10, seed=42)

    def test_small_world_clustering(self):
        """Test that small-world maintains some clustering."""
        n_nodes = 30
        n_neighbors = 6
        rewire_prob = 0.3

        pre_indices, post_indices = conn.small_world(
            n_nodes, n_neighbors, rewire_prob, seed=42
        )

        # Build adjacency list
        adjacency = {i: set() for i in range(n_nodes)}
        for i in range(len(pre_indices)):
            adjacency[pre_indices[i]].add(post_indices[i])
            adjacency[post_indices[i]].add(pre_indices[i])

        # Check that nodes still have some local clustering
        # (neighbors of a node are likely to be connected)
        clustering_coefficients = []
        for node in range(n_nodes):
            neighbors = list(adjacency[node])
            if len(neighbors) < 2:
                continue

            # Count edges between neighbors
            edges_between_neighbors = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in adjacency[neighbors[i]]:
                        edges_between_neighbors += 1

            # Calculate clustering coefficient
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            if possible_edges > 0:
                clustering = edges_between_neighbors / possible_edges
                clustering_coefficients.append(clustering)

        # Average clustering should be non-zero
        avg_clustering = np.mean(clustering_coefficients)
        self.assertTrue(avg_clustering > 0)

    def test_scale_free_preferential_attachment(self):
        """Test that scale-free uses preferential attachment."""
        n_nodes = 50
        m_edges = 2

        pre_indices, post_indices = conn.scale_free(n_nodes, m_edges, seed=42)

        # Early nodes should have higher average degree
        # (they had more time to accumulate connections)
        degrees = np.zeros(n_nodes)
        for i in range(len(pre_indices)):
            degrees[pre_indices[i]] += 1
            degrees[post_indices[i]] += 1
        degrees = degrees / 2

        # Compare average degree of first half vs second half of nodes
        first_half_avg = np.mean(degrees[:n_nodes // 2])
        second_half_avg = np.mean(degrees[n_nodes // 2:])

        # First half should have higher average degree
        self.assertTrue(first_half_avg > second_half_avg)


if __name__ == '__main__':
    unittest.main()
