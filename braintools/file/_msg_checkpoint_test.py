# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


import importlib.util
import unittest
import warnings
from collections import namedtuple
from tempfile import TemporaryDirectory

import brainstate
import brainunit as u
import pytest

import braintools

spec = importlib.util.find_spec("msgpack")

if spec is None:
    pytest.skip("msgpack not installed", allow_module_level=True)


class TestMsgCheckpoint(unittest.TestCase):
    def test_checkpoint_quantity(self):
        data = {
            "name": brainstate.random.rand(3) * u.ms,
        }

        with TemporaryDirectory('test_checkpoint_quantity', ignore_cleanup_errors=True) as tmpdirname:
            print(tmpdirname)
            filename = tmpdirname + "/test_msg_checkpoint.msg"
            braintools.file.msgpack_save(filename, data)
            data['name'] += 1 * u.ms

            data2 = braintools.file.msgpack_load(filename, target=data)
            self.assertTrue('name' in data2)
            self.assertTrue(isinstance(data2['name'], u.Quantity))
            self.assertTrue(not u.math.allclose(data['name'], data2['name']))

    def test_checkpoint_state(self):
        data = {
            "a": brainstate.State(brainstate.random.rand(1)),
            "b": brainstate.ShortTermState(brainstate.random.rand(2)),
            "c": brainstate.ParamState(brainstate.random.rand(3)),
        }

        with TemporaryDirectory('test_checkpoint_state', ignore_cleanup_errors=True) as tmpdirname:
            filename = tmpdirname + "/test_msg_checkpoint.msg"
            braintools.file.msgpack_save(filename, data)

            data2 = braintools.file.msgpack_load(filename, target=data)
            self.assertTrue('a' in data2)
            self.assertTrue('b' in data2)
            self.assertTrue('c' in data2)
            self.assertTrue(isinstance(data2['a'], brainstate.State))
            self.assertTrue(isinstance(data2['b'], brainstate.ShortTermState))
            self.assertTrue(isinstance(data2['c'], brainstate.ParamState))
            self.assertTrue(u.math.allclose(data['a'].value, data2['a'].value))
            self.assertTrue(u.math.allclose(data['b'].value, data2['b'].value))
            self.assertTrue(u.math.allclose(data['c'].value, data2['c'].value))

    def test_flatteneddict(self):
        net = brainstate.nn.Sequential(
            brainstate.nn.Linear(10, 20),
            brainstate.nn.ReLU(),
            brainstate.nn.Linear(20, 5)
        )
        net = brainstate.nn.init_all_states(net)

        data = {'state': net.states(), 'version': 1.0}

        with TemporaryDirectory('test_flatteneddict', ignore_cleanup_errors=True) as tmpdirname:
            filename = tmpdirname + "/test_msg_checkpoint.msg"
            braintools.file.msgpack_save(filename, data)
            loaded = braintools.file.msgpack_load(filename)
            print(loaded)
            loaded2 = braintools.file.msgpack_load(filename, target=data)


class TestMismatchSettings(unittest.TestCase):
    """Test cases for mismatch setting support in from_state_dict and msgpack_load"""

    def test_dict_mismatch_error(self):
        """Test that mismatch='error' raises ValueError for dictionary mismatches"""
        target = {'a': 1, 'b': 2, 'c': 3}
        state_dict = {'a': 10, 'b': 20}  # Missing 'c'

        with self.assertRaises(ValueError) as cm:
            braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='error')

        self.assertIn('do not match', str(cm.exception))
        self.assertIn('path', str(cm.exception))

    def test_dict_mismatch_warn(self):
        """Test that mismatch='warn' issues warning and preserves missing keys"""
        target = {'a': 1, 'b': 2, 'c': 3}
        state_dict = {'a': 10, 'b': 20}  # Missing 'c'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='warn')

            # Should issue warning
            self.assertEqual(len(w), 1)
            self.assertIn('do not match', str(w[0].message))

            # Should preserve missing key from target
            self.assertEqual(result, {'a': 10, 'b': 20, 'c': 3})

    def test_dict_mismatch_ignore(self):
        """Test that mismatch='ignore' silently preserves missing keys"""
        target = {'a': 1, 'b': 2, 'c': 3}
        state_dict = {'a': 10, 'b': 20}  # Missing 'c'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='ignore')

            # Should not issue warning
            self.assertEqual(len(w), 0)

            # Should preserve missing key from target
            self.assertEqual(result, {'a': 10, 'b': 20, 'c': 3})

    def test_list_mismatch_error(self):
        """Test that mismatch='error' raises ValueError for list size mismatches"""
        target = [1, 2, 3]
        state_dict = {'0': 10, '1': 20}  # Missing index '2'

        with self.assertRaises(ValueError) as cm:
            braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='error')

        self.assertIn('size', str(cm.exception))
        self.assertIn('do not match', str(cm.exception))

    def test_list_mismatch_warn(self):
        """Test that mismatch='warn' issues warning and preserves missing elements"""
        target = [1, 2, 3]
        state_dict = {'0': 10, '1': 20}  # Missing index '2'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='warn')

            # Should issue warning
            self.assertEqual(len(w), 1)
            self.assertIn('size', str(w[0].message))

            # Should preserve missing element from target
            self.assertEqual(result, [10, 20, 3])

    def test_list_mismatch_ignore(self):
        """Test that mismatch='ignore' silently preserves missing elements"""
        target = [1, 2, 3]
        state_dict = {'0': 10, '1': 20}  # Missing index '2'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='ignore')

            # Should not issue warning
            self.assertEqual(len(w), 0)

            # Should preserve missing element from target
            self.assertEqual(result, [10, 20, 3])

    def test_namedtuple_mismatch_error(self):
        """Test that mismatch='error' raises ValueError for namedtuple field mismatches"""
        TestTuple = namedtuple('TestTuple', ['a', 'b', 'c'])
        target = TestTuple(a=1, b=2, c=3)
        state_dict = {'a': 10, 'b': 20}  # Missing 'c'

        with self.assertRaises(ValueError) as cm:
            braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='error')

        self.assertIn('field names', str(cm.exception))
        self.assertIn('do not match', str(cm.exception))

    def test_namedtuple_mismatch_warn(self):
        """Test that mismatch='warn' issues warning and preserves missing fields"""
        TestTuple = namedtuple('TestTuple', ['a', 'b', 'c'])
        target = TestTuple(a=1, b=2, c=3)
        state_dict = {'a': 10, 'b': 20}  # Missing 'c'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='warn')

            # Should issue warning
            self.assertEqual(len(w), 1)
            self.assertIn('field names', str(w[0].message))

            # Should preserve missing field from target
            self.assertEqual(result, TestTuple(a=10, b=20, c=3))

    def test_namedtuple_mismatch_ignore(self):
        """Test that mismatch='ignore' silently preserves missing fields"""
        TestTuple = namedtuple('TestTuple', ['a', 'b', 'c'])
        target = TestTuple(a=1, b=2, c=3)
        state_dict = {'a': 10, 'b': 20}  # Missing 'c'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='ignore')

            # Should not issue warning
            self.assertEqual(len(w), 0)

            # Should preserve missing field from target
            self.assertEqual(result, TestTuple(a=10, b=20, c=3))

    def test_quantity_mismatch_error(self):
        """Test that mismatch='error' raises ValueError for unit mismatches"""
        target = 1.0 * u.ms
        state_dict = {
            'mantissa': 2.0,
            'scale': u.second.scale,
            'base': u.second.base,
            'dim': u.second.dim._dims,
            'factor': u.second.factor
        }  # Different unit (second instead of millisecond)

        with self.assertRaises(ValueError) as cm:
            braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='error')

        self.assertIn('Unit mismatch', str(cm.exception))

    def test_quantity_mismatch_warn(self):
        """Test that mismatch='warn' issues warning for unit mismatches"""
        target = 1.0 * u.ms
        state_dict = {
            'mantissa': 2.0,
            'scale': u.second.scale,
            'base': u.second.base,
            'dim': u.second.dim._dims,
            'factor': u.second.factor
        }  # Different unit (second instead of millisecond)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='warn')

            # Should issue warning
            self.assertEqual(len(w), 1)
            self.assertIn('Unit mismatch', str(w[0].message))

            # Should use the loaded unit
            self.assertEqual(result.mantissa, 2.0)
            self.assertEqual(result.unit, u.second)

    def test_quantity_mismatch_ignore(self):
        """Test that mismatch='ignore' silently handles unit mismatches"""
        target = 1.0 * u.ms
        state_dict = {
            'mantissa': 2.0,
            'scale': u.second.scale,
            'base': u.second.base,
            'dim': u.second.dim._dims,
            'factor': u.second.factor
        }  # Different unit (second instead of millisecond)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='ignore')

            # Should not issue warning
            self.assertEqual(len(w), 0)

            # Should use the loaded unit
            self.assertEqual(result.mantissa, 2.0)
            self.assertEqual(result.unit, u.second)

    def test_mismatch_parameter_validation(self):
        """Test that invalid mismatch parameters are handled correctly"""
        target = {'a': 1, 'b': 2, 'c': 3}
        state_dict = {'a': 10, 'b': 20}  # Missing 'c'

        # Test invalid mismatch parameter - should default to error behavior
        with self.assertRaises(ValueError):
            braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='invalid_mode')

    def test_nested_structure_mismatch(self):
        """Test mismatch handling with nested data structures"""
        target = {
            'model': {'weights': [1, 2, 3], 'bias': 0.1},
            'config': {'learning_rate': 0.01, 'batch_size': 32}
        }
        state_dict = {
            'model': {'weights': {'0': 10, '1': 20}, 'bias': 0.5},  # Missing element in weights
            'config': {'learning_rate': 0.001}  # Missing batch_size
        }

        # Test with warn mode - should handle nested mismatches
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='warn')

            # Should issue warnings for both list and dict mismatches
            self.assertGreaterEqual(len(w), 2)

            # Should preserve original values for missing items
            expected = {
                'model': {'weights': [10, 20, 3], 'bias': 0.5},
                'config': {'learning_rate': 0.001, 'batch_size': 32}
            }
            self.assertEqual(result, expected)
