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
import numpy as np

from braintools.conn import (
    AllToAll,
    OneToOne,
)
from braintools.init import Constant


class TestAllToAll(unittest.TestCase):
    """
    Test AllToAll connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import AllToAll

        # Basic all-to-all connectivity
        conn = AllToAll(
            include_self_connections=False,
            weight=1.0 * u.nS,
            delay=2.0 * u.ms
        )

        result = conn(pre_size=10, post_size=10)

        # Should have 10*10 - 10 = 90 connections (excluding self)
        assert result.n_connections == 90
        assert result.metadata['pattern'] == 'all_to_all'

        # With self-connections
        conn_self = AllToAll(include_self_connections=True)
        result_self = conn_self(pre_size=10, post_size=10)
        assert result_self.n_connections == 100
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_all_to_all(self):
        conn = AllToAll(include_self_connections=False, seed=42)
        result = conn(pre_size=8, post_size=8)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.n_connections, 8 * 8 - 8)  # 56 connections
        self.assertEqual(result.metadata['pattern'], 'all_to_all')
        self.assertFalse(result.metadata['include_self_connections'])

        # Check no self-connections
        self_connections = np.sum(result.pre_indices == result.post_indices)
        self.assertEqual(self_connections, 0)

    def test_all_to_all_with_self_connections(self):
        conn = AllToAll(include_self_connections=True, seed=42)
        result = conn(pre_size=6, post_size=6)

        self.assertEqual(result.n_connections, 6 * 6)  # 36 connections
        self.assertTrue(result.metadata['include_self_connections'])

        # Check that all possible connections exist
        expected_connections = set((i, j) for i in range(6) for j in range(6))
        actual_connections = set(zip(result.pre_indices, result.post_indices))
        self.assertEqual(expected_connections, actual_connections)

    def test_all_to_all_asymmetric(self):
        conn = AllToAll(include_self_connections=False, seed=42)
        result = conn(pre_size=5, post_size=8)

        self.assertEqual(result.n_connections, 5 * 8)  # 40 connections
        self.assertEqual(result.shape, (5, 8))

        # Should connect every pre to every post
        expected_connections = set((i, j) for i in range(5) for j in range(8))
        actual_connections = set(zip(result.pre_indices, result.post_indices))
        self.assertEqual(expected_connections, actual_connections)

    def test_all_to_all_with_weights_and_delays(self):
        weight_init = Constant(0.8 * u.nS)
        delay_init = Constant(1.5 * u.ms)

        conn = AllToAll(
            include_self_connections=True,
            weight=weight_init,
            delay=delay_init,
            seed=42
        )

        result = conn(pre_size=4, post_size=4)

        self.assertEqual(result.n_connections, 16)
        self.assertIsNotNone(result.weights)
        self.assertIsNotNone(result.delays)

        # All weights and delays should be constant
        np.testing.assert_array_almost_equal(u.get_mantissa(result.weights), 0.8)
        np.testing.assert_array_almost_equal(u.get_mantissa(result.delays), 1.5)

    def test_all_to_all_tuple_sizes(self):
        conn = AllToAll(include_self_connections=False, seed=42)
        result = conn(pre_size=(2, 3), post_size=(2, 2))

        self.assertEqual(result.pre_size, (2, 3))
        self.assertEqual(result.post_size, (2, 2))
        self.assertEqual(result.shape, (6, 4))

        # 6 pre neurons, 4 post neurons
        # Since pre_size != post_size, no self-connections to exclude
        self.assertEqual(result.n_connections, 6 * 4)

    def test_all_to_all_single_neuron(self):
        # Test edge case with single neuron
        conn = AllToAll(include_self_connections=True, seed=42)
        result = conn(pre_size=1, post_size=1)

        self.assertEqual(result.n_connections, 1)
        np.testing.assert_array_equal(result.pre_indices, [0])
        np.testing.assert_array_equal(result.post_indices, [0])


class TestOneToOne(unittest.TestCase):
    """
    Test OneToOne connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import OneToOne

        # Basic one-to-one connectivity
        conn = OneToOne(
            weight=3.0 * u.nS,
            delay=0.5 * u.ms,
            circular=False
        )

        result = conn(pre_size=10, post_size=10)

        # Should have 10 connections: 0->0, 1->1, ..., 9->9
        assert result.n_connections == 10
        assert result.metadata['pattern'] == 'one_to_one'

        # Test circular indexing with different sizes
        conn_circular = OneToOne(circular=True)
        result_circular = conn_circular(pre_size=5, post_size=8)
        assert result_circular.n_connections == 8  # max(5, 8)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_one_to_one(self):
        conn = OneToOne(circular=False, seed=42)
        result = conn(pre_size=12, post_size=12)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.n_connections, 12)
        self.assertEqual(result.metadata['pattern'], 'one_to_one')
        self.assertFalse(result.metadata['circular'])

        # Check that connections are exactly i->i
        expected_pre = np.arange(12)
        expected_post = np.arange(12)
        np.testing.assert_array_equal(result.pre_indices, expected_pre)
        np.testing.assert_array_equal(result.post_indices, expected_post)

    def test_one_to_one_different_sizes_non_circular(self):
        # Smaller post population
        conn = OneToOne(circular=False, seed=42)
        result = conn(pre_size=10, post_size=6)

        self.assertEqual(result.n_connections, 6)  # min(10, 6)
        np.testing.assert_array_equal(result.pre_indices, np.arange(6))
        np.testing.assert_array_equal(result.post_indices, np.arange(6))

        # Smaller pre population
        result2 = conn(pre_size=4, post_size=8, recompute=True)
        self.assertEqual(result2.n_connections, 4)  # min(4, 8)
        np.testing.assert_array_equal(result2.pre_indices, np.arange(4))
        np.testing.assert_array_equal(result2.post_indices, np.arange(4))

    def test_one_to_one_circular(self):
        conn = OneToOne(circular=True, seed=42)

        # More post than pre
        result = conn(pre_size=5, post_size=8)
        self.assertEqual(result.n_connections, 8)  # max(5, 8)
        self.assertTrue(result.metadata['circular'])

        # Check circular indexing
        expected_pre = np.array([0, 1, 2, 3, 4, 0, 1, 2])  # Wraps around
        expected_post = np.arange(8)
        np.testing.assert_array_equal(result.pre_indices, expected_pre)
        np.testing.assert_array_equal(result.post_indices, expected_post)

        # More pre than post
        result2 = conn(pre_size=7, post_size=4, recompute=True)
        self.assertEqual(result2.n_connections, 7)  # max(7, 4)

        expected_pre2 = np.arange(7)
        expected_post2 = np.array([0, 1, 2, 3, 0, 1, 2])  # Post wraps around
        np.testing.assert_array_equal(result2.pre_indices, expected_pre2)
        np.testing.assert_array_equal(result2.post_indices, expected_post2)

    def test_one_to_one_with_weights_and_delays(self):
        weight_init = Constant(2.5 * u.nS)
        delay_init = Constant(0.8 * u.ms)

        conn = OneToOne(
            weight=weight_init,
            delay=delay_init,
            circular=False,
            seed=42
        )

        result = conn(pre_size=8, post_size=8)

        self.assertEqual(result.n_connections, 8)
        self.assertIsNotNone(result.weights)
        self.assertIsNotNone(result.delays)

        np.testing.assert_array_almost_equal(u.get_mantissa(result.weights), 2.5)
        np.testing.assert_array_almost_equal(u.get_mantissa(result.delays), 0.8)

    def test_one_to_one_tuple_sizes(self):
        conn = OneToOne(circular=False, seed=42)
        result = conn(pre_size=(2, 4), post_size=(3, 3))

        # pre_size = 8, post_size = 9, min = 8
        self.assertEqual(result.n_connections, 8)
        self.assertEqual(result.pre_size, (2, 4))
        self.assertEqual(result.post_size, (3, 3))

    def test_one_to_one_single_neuron(self):
        conn = OneToOne(circular=False, seed=42)
        result = conn(pre_size=1, post_size=1)

        self.assertEqual(result.n_connections, 1)
        np.testing.assert_array_equal(result.pre_indices, [0])
        np.testing.assert_array_equal(result.post_indices, [0])
