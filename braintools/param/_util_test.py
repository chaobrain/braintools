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

"""Tests for utility functions."""

import unittest

import brainstate
import jax.numpy as jnp

from braintools.param._util import get_param, get_size


class TestGetParam(unittest.TestCase):
    """Tests for get_param utility function."""

    def test_with_param_state(self):
        """Test get_param with ParamState."""
        state = brainstate.ParamState(jnp.array([1.0, 2.0]))
        result = get_param(state)
        self.assertTrue(jnp.array_equal(result, jnp.array([1.0, 2.0])))

    def test_with_hidden_state(self):
        """Test get_param with HiddenState."""
        state = brainstate.HiddenState(jnp.array([3.0, 4.0]))
        result = get_param(state)
        self.assertTrue(jnp.array_equal(result, jnp.array([3.0, 4.0])))

    def test_with_array(self):
        """Test get_param with plain array."""
        arr = jnp.array([5.0, 6.0])
        result = get_param(arr)
        self.assertTrue(jnp.array_equal(result, arr))

    def test_with_scalar(self):
        """Test get_param with scalar."""
        scalar = 1.5
        result = get_param(scalar)
        self.assertEqual(result, scalar)

    def test_with_none(self):
        """Test get_param with None."""
        result = get_param(None)
        self.assertIsNone(result)


class TestGetSize(unittest.TestCase):
    """Tests for get_size utility function."""

    def test_with_int(self):
        """Test get_size with int."""
        result = get_size(5)
        self.assertEqual(result, (5,))

    def test_with_tuple(self):
        """Test get_size with tuple."""
        result = get_size((3, 4))
        self.assertEqual(result, (3, 4))

    def test_with_list(self):
        """Test get_size with list."""
        result = get_size([2, 3, 4])
        self.assertEqual(result, (2, 3, 4))

    def test_with_empty_tuple(self):
        """Test get_size with empty tuple."""
        result = get_size(())
        self.assertEqual(result, ())

    def test_with_empty_list(self):
        """Test get_size with empty list."""
        result = get_size([])
        self.assertEqual(result, ())

    def test_invalid_type_raises(self):
        """Test get_size raises ValueError for invalid type."""
        with self.assertRaises(ValueError) as context:
            get_size("invalid")
        self.assertIn("str", str(context.exception))

    def test_invalid_float_raises(self):
        """Test get_size raises ValueError for float."""
        with self.assertRaises(ValueError) as context:
            get_size(3.5)
        self.assertIn("float", str(context.exception))

    def test_invalid_dict_raises(self):
        """Test get_size raises ValueError for dict."""
        with self.assertRaises(ValueError) as context:
            get_size({"a": 1})
        self.assertIn("dict", str(context.exception))


if __name__ == '__main__':
    unittest.main()
