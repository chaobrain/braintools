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
from dataclasses import is_dataclass

import jax.numpy as jnp
import numpy as np

from braintools.param import Data


class TestAutoDataclass(unittest.TestCase):
    """Test that subclasses of Data automatically become dataclasses."""

    def test_subclass_is_dataclass(self):
        class MyState(Data):
            x: float
            y: float

        self.assertTrue(is_dataclass(MyState))

    def test_subclass_can_instantiate(self):
        class MyState(Data):
            x: float
            y: float

        state = MyState(x=1.0, y=2.0)
        self.assertEqual(state.x, 1.0)
        self.assertEqual(state.y, 2.0)

    def test_nested_inheritance(self):
        class BaseState(Data):
            x: float

        class DerivedState(BaseState):
            y: float

        self.assertTrue(is_dataclass(BaseState))
        self.assertTrue(is_dataclass(DerivedState))


class TestDataMethods(unittest.TestCase):
    """Test Data class methods."""

    def setUp(self):
        class SimpleState(Data):
            x: jnp.ndarray
            y: jnp.ndarray

        self.StateClass = SimpleState
        self.state = SimpleState(
            x=jnp.array([1.0, 2.0]),
            y=jnp.array([3.0, 4.0])
        )

    def test_to_dict(self):
        d = self.state.to_dict()
        self.assertIn('x', d)
        self.assertIn('y', d)
        np.testing.assert_array_equal(d['x'], jnp.array([1.0, 2.0]))
        np.testing.assert_array_equal(d['y'], jnp.array([3.0, 4.0]))

    def test_from_dict(self):
        d = {'x': jnp.array([5.0, 6.0]), 'y': jnp.array([7.0, 8.0])}
        state = self.StateClass.from_dict(d)
        np.testing.assert_array_equal(state.x, jnp.array([5.0, 6.0]))
        np.testing.assert_array_equal(state.y, jnp.array([7.0, 8.0]))

    def test_to_dict_from_dict_roundtrip(self):
        d = self.state.to_dict()
        state2 = self.StateClass.from_dict(d)
        np.testing.assert_array_equal(state2.x, self.state.x)
        np.testing.assert_array_equal(state2.y, self.state.y)

    def test_dtype(self):
        self.assertEqual(self.state.dtype, jnp.float32)

    def test_dtype_no_tensors_raises(self):
        class EmptyState(Data):
            name: str

        state = EmptyState(name="test")
        with self.assertRaises(ValueError):
            _ = state.dtype

    def test_state_size(self):
        self.assertEqual(self.state.state_size, 2)

    def test_replace(self):
        new_x = jnp.array([10.0, 20.0])
        new_state = self.state.replace(x=new_x)
        np.testing.assert_array_equal(new_state.x, new_x)
        np.testing.assert_array_equal(new_state.y, self.state.y)
        # Original unchanged
        np.testing.assert_array_equal(self.state.x, jnp.array([1.0, 2.0]))

    def test_replace_multiple(self):
        new_x = jnp.array([10.0, 20.0])
        new_y = jnp.array([30.0, 40.0])
        new_state = self.state.replace(x=new_x, y=new_y)
        np.testing.assert_array_equal(new_state.x, new_x)
        np.testing.assert_array_equal(new_state.y, new_y)


class TestDataWithNone(unittest.TestCase):
    """Test Data class with None values."""

    def test_to_dict_with_none(self):
        class OptionalState(Data):
            x: jnp.ndarray
            y: jnp.ndarray = None

        state = OptionalState(x=jnp.array([1.0]), y=None)
        d = state.to_dict()
        self.assertIsNone(d['y'])

    def test_dtype_skips_none(self):
        class OptionalState(Data):
            x: jnp.ndarray = None
            y: jnp.ndarray = None

        state = OptionalState(x=None, y=jnp.array([1.0]))
        self.assertEqual(state.dtype, jnp.float32)


class TestComposedData(unittest.TestCase):
    """Test Data class."""

    def test_is_dataclass(self):
        self.assertTrue(is_dataclass(Data))

    def test_empty_init(self):
        composed = Data()
        self.assertEqual(len(composed.children), 0)

    def test_init_with_children(self):
        class ChildState(Data):
            val: jnp.ndarray

        child = ChildState(val=jnp.array([1.0]))
        composed = Data(children={'child1': child})
        self.assertIn('child1', composed)

    def test_getitem(self):
        class ChildState(Data):
            val: jnp.ndarray

        child = ChildState(val=jnp.array([1.0]))
        composed = Data(children={'child1': child})
        np.testing.assert_array_equal(composed['child1'].val, jnp.array([1.0]))

    def test_setitem(self):
        class ChildState(Data):
            val: jnp.ndarray

        composed = Data()
        child = ChildState(val=jnp.array([2.0]))
        composed['child1'] = child
        np.testing.assert_array_equal(composed['child1'].val, jnp.array([2.0]))

    def test_contains(self):
        composed = Data(children={'a': 1, 'b': 2})
        self.assertIn('a', composed)
        self.assertIn('b', composed)
        self.assertNotIn('c', composed)

    def test_keys_items_values(self):
        composed = Data(children={'a': 1, 'b': 2})
        self.assertEqual(set(composed.keys()), {'a', 'b'})
        self.assertEqual(set(composed.values()), {1, 2})
        self.assertEqual(set(composed.items()), {('a', 1), ('b', 2)})

    def test_state_size(self):
        class ChildState(Data):
            x: jnp.ndarray
            y: jnp.ndarray

        child1 = ChildState(x=jnp.array([1.0]), y=jnp.array([2.0]))
        child2 = ChildState(x=jnp.array([3.0]), y=jnp.array([4.0]))
        composed = Data(children={'c1': child1, 'c2': child2})
        self.assertEqual(composed.state_size, 4)  # 2 fields * 2 children

    def test_state_size_with_none(self):
        class ChildState(Data):
            x: jnp.ndarray

        child = ChildState(x=jnp.array([1.0]))
        composed = Data(children={'c1': child, 'c2': None})
        self.assertEqual(composed.state_size, 1)

    def test_dtype(self):
        class ChildState(Data):
            x: jnp.ndarray

        child = ChildState(x=jnp.array([1.0], dtype=jnp.float32))
        composed = Data(children={'c1': child})
        self.assertEqual(composed.dtype, jnp.float32)

    def test_dtype_no_children_raises(self):
        composed = Data()
        with self.assertRaises(ValueError):
            _ = composed.dtype

    def test_replace(self):
        class ChildState(Data):
            x: jnp.ndarray

        child1 = ChildState(x=jnp.array([1.0]))
        child2 = ChildState(x=jnp.array([2.0]))
        composed = Data(children={'c1': child1})

        new_composed = composed.replace(c1=child2)
        np.testing.assert_array_equal(new_composed['c1'].x, jnp.array([2.0]))
        # Original unchanged
        np.testing.assert_array_equal(composed['c1'].x, jnp.array([1.0]))

    def test_clone(self):
        class ChildState(Data):
            x: jnp.ndarray

            def clone(self):
                return ChildState(x=self.x.copy())

        child = ChildState(x=jnp.array([1.0]))
        composed = Data(children={'c1': child, 'c2': None})
        cloned = composed.clone()

        self.assertIsNot(cloned, composed)
        self.assertIsNot(cloned.children, composed.children)
        np.testing.assert_array_equal(cloned['c1'].x, child.x)
        self.assertIsNone(cloned['c2'])

    def test_clone_without_clone_method(self):
        composed = Data(children={'c1': 'simple_value'})
        cloned = composed.clone()
        self.assertEqual(cloned['c1'], 'simple_value')


class TestComposedDataKwargsInit(unittest.TestCase):
    """Test Data kwargs initialization and attribute access."""

    def test_init_with_kwargs(self):
        class ChildState(Data):
            val: jnp.ndarray

        child1 = ChildState(val=jnp.array([1.0]))
        child2 = ChildState(val=jnp.array([2.0]))
        composed = Data(key1=child1, key2=child2)

        self.assertIn('key1', composed)
        self.assertIn('key2', composed)

    def test_getattr_access(self):
        class ChildState(Data):
            val: jnp.ndarray

        child = ChildState(val=jnp.array([1.0]))
        composed = Data(mykey=child)

        np.testing.assert_array_equal(composed.mykey.val, jnp.array([1.0]))

    def test_getattr_missing_raises(self):
        composed = Data()
        with self.assertRaises(AttributeError):
            _ = composed.nonexistent

    def test_mixed_init_children_and_kwargs(self):
        composed = Data(children={'a': 1}, b=2, c=3)
        self.assertEqual(composed.a, 1)
        self.assertEqual(composed.b, 2)
        self.assertEqual(composed.c, 3)

    def test_kwargs_override_children(self):
        composed = Data(children={'a': 1}, a=2)
        self.assertEqual(composed.a, 2)

    def test_attribute_and_item_access_equivalent(self):
        composed = Data(key1='value1', key2='value2')
        self.assertEqual(composed.key1, composed['key1'])
        self.assertEqual(composed.key2, composed['key2'])

    def test_children_attribute_accessible(self):
        composed = Data(a=1, b=2)
        self.assertIsInstance(composed.children, dict)
        self.assertEqual(composed.children, {'a': 1, 'b': 2})


if __name__ == '__main__':
    unittest.main()
