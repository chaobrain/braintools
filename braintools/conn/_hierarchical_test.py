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


class TestHierarchicalConnectivity(unittest.TestCase):
    """Test hierarchical connectivity patterns."""

    def test_hierarchical(self):
        """Test hierarchical connectivity."""
        sizes = [10, 20, 15]
        pre, post = conn.hierarchical(
            sizes,
            forward_prob=0.3,
            backward_prob=0.1,
            lateral_prob=0.05,
            seed=42
        )

        # Check output types
        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)
        self.assertEqual(len(pre), len(post))

        # Check indices are valid
        total_nodes = sum(sizes)
        self.assertTrue(np.all(pre >= 0))
        self.assertTrue(np.all(pre < total_nodes))
        self.assertTrue(np.all(post >= 0))
        self.assertTrue(np.all(post < total_nodes))

        # Check no self-connections in lateral
        self.assertTrue(np.all(pre != post))

    def test_block_connect(self):
        """Test block-structured connectivity."""
        block_sizes = [15, 20, 10]
        pre, post = conn.block_connect(
            block_sizes,
            within_block_prob=0.3,
            between_block_prob=0.05,
            seed=42
        )

        # Check output
        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)
        self.assertEqual(len(pre), len(post))

        # Check indices are valid
        total_nodes = sum(block_sizes)
        self.assertTrue(np.all(pre >= 0))
        self.assertTrue(np.all(pre < total_nodes))
        self.assertTrue(np.all(post >= 0))
        self.assertTrue(np.all(post < total_nodes))

    def test_layered_network(self):
        """Test layered network connectivity."""
        layer_sizes = [10, 15, 8]

        # Test with default connection probs
        pre, post = conn.layered_network(layer_sizes, seed=42)

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # Test with custom connection matrix
        n_layers = len(layer_sizes)
        connection_probs = np.zeros((n_layers, n_layers))
        connection_probs[0, 1] = 0.5  # Layer 0 -> 1
        connection_probs[1, 2] = 0.3  # Layer 1 -> 2

        pre, post = conn.layered_network(
            layer_sizes,
            connection_probs=connection_probs,
            seed=42
        )

        total_nodes = sum(layer_sizes)
        self.assertTrue(np.all(pre >= 0))
        self.assertTrue(np.all(pre < total_nodes))
        self.assertTrue(np.all(post >= 0))
        self.assertTrue(np.all(post < total_nodes))

    def test_feedforward_layers(self):
        """Test feedforward layer connectivity."""
        layer_sizes = [10, 20, 15, 5]

        # Without skip connections
        pre, post = conn.feedforward_layers(
            layer_sizes,
            connection_prob=0.2,
            skip_connections=False,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # With skip connections
        pre_skip, post_skip = conn.feedforward_layers(
            layer_sizes,
            connection_prob=0.2,
            skip_connections=True,
            seed=42
        )

        # Skip connections should add more edges
        self.assertGreaterEqual(len(pre_skip), len(pre))

    def test_cortical_hierarchy(self):
        """Test cortical hierarchy connectivity."""
        area_sizes = [50, 100, 75, 60]

        # With default hierarchy levels
        pre, post = conn.cortical_hierarchy(
            area_sizes,
            forward_prob=0.2,
            backward_prob=0.1,
            lateral_prob=0.05,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # With custom hierarchy levels
        hierarchy_levels = [0.0, 0.3, 0.7, 1.0]
        pre, post = conn.cortical_hierarchy(
            area_sizes,
            forward_prob=0.2,
            backward_prob=0.1,
            lateral_prob=0.05,
            hierarchy_levels=hierarchy_levels,
            seed=42
        )

        total_nodes = sum(area_sizes)
        self.assertTrue(np.all(pre >= 0))
        self.assertTrue(np.all(pre < total_nodes))
        self.assertTrue(np.all(post >= 0))
        self.assertTrue(np.all(post < total_nodes))

    def test_modular_network(self):
        """Test modular network connectivity."""
        module_sizes = [20, 25, 15]

        # Without hubs
        pre, post = conn.modular_network(
            module_sizes,
            n_hubs=0,
            within_module_prob=0.3,
            between_module_prob=0.05,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        total_nodes = sum(module_sizes)
        self.assertTrue(np.all(pre >= 0))
        self.assertTrue(np.all(pre < total_nodes))

        # With hub nodes
        n_hubs = 3
        pre_hub, post_hub = conn.modular_network(
            module_sizes,
            n_hubs=n_hubs,
            within_module_prob=0.3,
            between_module_prob=0.05,
            hub_prob=0.2,
            seed=42
        )

        # Should have connections involving hub nodes
        total_with_hubs = total_nodes + n_hubs
        self.assertTrue(np.any(pre_hub >= total_nodes) or np.any(post_hub >= total_nodes))
        self.assertTrue(np.all(pre_hub < total_with_hubs))
        self.assertTrue(np.all(post_hub < total_with_hubs))

    def test_hierarchical_edge_cases(self):
        """Test edge cases for hierarchical functions."""
        # Single layer
        sizes = [20]
        pre, post = conn.hierarchical(
            sizes,
            forward_prob=0.3,
            backward_prob=0.1,
            lateral_prob=0.1,
            seed=42
        )

        # Should only have lateral connections
        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # Empty layer sizes
        sizes = []
        pre, post = conn.hierarchical(
            sizes,
            forward_prob=0.3,
            backward_prob=0.1,
            lateral_prob=0.05,
            seed=42
        )

        self.assertEqual(len(pre), 0)
        self.assertEqual(len(post), 0)

        # Zero probabilities
        sizes = [10, 10, 10]
        pre, post = conn.hierarchical(
            sizes,
            forward_prob=0.0,
            backward_prob=0.0,
            lateral_prob=0.0,
            seed=42
        )

        self.assertEqual(len(pre), 0)
        self.assertEqual(len(post), 0)

    def test_block_connect_single_block(self):
        """Test block connect with single block."""
        block_sizes = [30]
        pre, post = conn.block_connect(
            block_sizes,
            within_block_prob=0.2,
            between_block_prob=0.05,
            seed=42
        )

        # Should only have within-block connections
        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)
        self.assertTrue(np.all(pre < 30))
        self.assertTrue(np.all(post < 30))
        # No self-connections
        self.assertTrue(np.all(pre != post))


if __name__ == '__main__':
    unittest.main()
