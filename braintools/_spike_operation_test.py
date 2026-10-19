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

import jax.numpy as jnp
import numpy as np

import braintools


class TestSpikeBitwiseOps(unittest.TestCase):
    def setUp(self):
        # Basic truth-table vectors: pairs (x, y) = (0,0), (0,1), (1,0), (1,1)
        self.x = jnp.array([0, 0, 1, 1])
        self.y = jnp.array([0, 1, 0, 1])

    def test_bitwise_or(self):
        expected = jnp.array([0, 1, 1, 1])
        out1 = braintools.spike_bitwise_or(self.x, self.y)
        out2 = braintools.spike_bitwise(self.x, self.y, 'or')
        np.testing.assert_array_equal(np.array(out1), np.array(expected))
        np.testing.assert_array_equal(np.array(out2), np.array(expected))

    def test_bitwise_and(self):
        expected = jnp.array([0, 0, 0, 1])
        out1 = braintools.spike_bitwise_and(self.x, self.y)
        out2 = braintools.spike_bitwise(self.x, self.y, 'and')
        np.testing.assert_array_equal(np.array(out1), np.array(expected))
        np.testing.assert_array_equal(np.array(out2), np.array(expected))

    def test_bitwise_iand(self):
        # (NOT x) AND y -> 0,1,0,0
        expected = jnp.array([0, 1, 0, 0])
        out1 = braintools.spike_bitwise_iand(self.x, self.y)
        out2 = braintools.spike_bitwise(self.x, self.y, 'iand')
        np.testing.assert_array_equal(np.array(out1), np.array(expected))
        np.testing.assert_array_equal(np.array(out2), np.array(expected))

    def test_bitwise_not(self):
        expected = jnp.array([1, 1, 0, 0])
        out = braintools.spike_bitwise_not(self.x)
        np.testing.assert_array_equal(np.array(out), np.array(expected))

    def test_bitwise_xor(self):
        expected = jnp.array([0, 1, 1, 0])
        out1 = braintools.spike_bitwise_xor(self.x, self.y)
        out2 = braintools.spike_bitwise(self.x, self.y, 'xor')
        np.testing.assert_array_equal(np.array(out1), np.array(expected))
        np.testing.assert_array_equal(np.array(out2), np.array(expected))

    def test_bitwise_ixor(self):
        # IXOR as implemented equals XOR on binary inputs
        expected = jnp.array([0, 1, 1, 0])
        out1 = braintools.spike_bitwise_ixor(self.x, self.y)
        out2 = braintools.spike_bitwise(self.x, self.y, 'ixor')
        np.testing.assert_array_equal(np.array(out1), np.array(expected))
        np.testing.assert_array_equal(np.array(out2), np.array(expected))

    def test_bitwise_invalid_op_raises(self):
        with self.assertRaises(NotImplementedError):
            _ = braintools.spike_bitwise(self.x, self.y, 'nand')

    def test_broadcasting(self):
        # Ensure elementwise broadcasting works as expected
        x = jnp.array([[0], [1]])  # shape (2,1)
        y = jnp.array([0, 1])  # shape (2,)

        # OR truth table broadcast -> [[0,1],[1,1]]
        expected_or = jnp.array([[0, 1], [1, 1]])
        out_or = braintools.spike_bitwise_or(x, y)
        np.testing.assert_array_equal(np.array(out_or), np.array(expected_or))

        # AND truth table broadcast -> [[0,0],[0,1]]
        expected_and = jnp.array([[0, 0], [0, 1]])
        out_and = braintools.spike_bitwise_and(x, y)
        np.testing.assert_array_equal(np.array(out_and), np.array(expected_and))


class TestSpikeBitwiseBooleanMasks(unittest.TestCase):
    """Boolean 0/1 masks are a natural spike representation and must work.

    Regression test: ``bool + bool`` is *logical OR* in numpy/jax and ``bool -
    bool`` is rejected outright, so the polynomial identities used by these ops
    were wrong (or raised) for boolean inputs -- ``spike_bitwise_xor`` returned
    ``-1`` for the ``(True, True)`` entry, and ``spike_bitwise_or`` raised
    ``TypeError: sub does not accept dtype bool``.
    """

    def setUp(self):
        # (x, y) = (False, False), (False, True), (True, False), (True, True)
        self.xb = jnp.array([False, False, True, True])
        self.yb = jnp.array([False, True, False, True])

    def test_truth_tables_for_boolean_inputs(self):
        expected = {
            'or':   [0, 1, 1, 1],
            'and':  [0, 0, 0, 1],
            'iand': [0, 1, 0, 0],
            'xor':  [0, 1, 1, 0],
            'ixor': [0, 1, 1, 0],
        }
        for op, exp in expected.items():
            out = braintools.spike_bitwise(self.xb, self.yb, op)
            # boolean masks in, boolean mask out -- for every op
            self.assertEqual(out.dtype, jnp.bool_, op)
            np.testing.assert_array_equal(np.asarray(out), np.array(exp, dtype=bool))

    def test_not_for_boolean_inputs(self):
        out = braintools.spike_bitwise_not(self.xb)
        self.assertEqual(out.dtype, jnp.bool_)
        np.testing.assert_array_equal(np.asarray(out), [True, True, False, False])

    def test_bool_mixed_with_float_takes_float_dtype(self):
        yf = self.yb.astype(jnp.float32)
        for op, exp in {'or': [0, 1, 1, 1], 'and': [0, 0, 0, 1], 'iand': [0, 1, 0, 0],
                        'xor': [0, 1, 1, 0], 'ixor': [0, 1, 1, 0]}.items():
            out = braintools.spike_bitwise(self.xb, yf, op)
            self.assertEqual(out.dtype, jnp.float32, op)
            np.testing.assert_array_equal(np.asarray(out), np.array(exp, dtype=np.float32))

    def test_float_and_int_paths_unchanged(self):
        xf, yf = self.xb.astype(jnp.float32), self.yb.astype(jnp.float32)
        np.testing.assert_array_equal(
            np.asarray(braintools.spike_bitwise_or(xf, yf)), np.array([0.0, 1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(
            np.asarray(braintools.spike_bitwise_xor(xf, yf)), np.array([0.0, 1.0, 1.0, 0.0]))


if __name__ == '__main__':
    unittest.main()
