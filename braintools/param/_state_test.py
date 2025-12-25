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

"""Tests for ArrayHidden and ArrayParam state containers."""

import unittest

import brainstate
import jax.numpy as jnp
import numpy as np

from braintools.param import (
    ArrayHidden,
    ArrayParam,
    IdentityT,
    SigmoidT,
    SoftplusT,
)


class TestArrayHidden(unittest.TestCase):
    """Tests for ArrayHidden state container."""

    def test_basic_instantiation(self):
        """Test basic instantiation with default transform."""
        hidden = ArrayHidden(jnp.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(hidden.data, jnp.array([1.0, 2.0, 3.0]))

    def test_with_identity_transform(self):
        """Test with explicit identity transform."""
        hidden = ArrayHidden(jnp.array([1.0, 2.0]), t=IdentityT())
        np.testing.assert_allclose(hidden.data, jnp.array([1.0, 2.0]))

    def test_with_softplus_transform(self):
        """Test with softplus transform for positive constraint."""
        value = jnp.array([1.0, 2.0, 3.0])
        hidden = ArrayHidden(value, t=SoftplusT(0.0))
        # Data should match original value
        np.testing.assert_allclose(hidden.data, value, rtol=1e-5)
        # Internal value should be transformed
        self.assertFalse(np.allclose(hidden.value, value))

    def test_with_sigmoid_transform(self):
        """Test with sigmoid transform for bounded constraint."""
        value = jnp.array([0.3, 0.5, 0.7])
        hidden = ArrayHidden(value, t=SigmoidT(0.0, 1.0))
        # Data should approximately match original value
        np.testing.assert_allclose(hidden.data, value, rtol=1e-4)

    def test_data_setter(self):
        """Test setting data property."""
        hidden = ArrayHidden(jnp.array([1.0, 2.0]), t=SoftplusT(0.0))
        new_value = jnp.array([3.0, 4.0])
        hidden.data = new_value
        np.testing.assert_allclose(hidden.data, new_value, rtol=1e-5)

    def test_repr(self):
        """Test string representation."""
        hidden = ArrayHidden(jnp.array([1.0]), t=IdentityT())
        r = repr(hidden)
        self.assertIn("ArrayHidden", r)
        self.assertIn("IdentityT", r)

    def test_inherits_from_hidden_state(self):
        """Test that ArrayHidden inherits from brainstate.HiddenState."""
        hidden = ArrayHidden(jnp.array([1.0]))
        self.assertIsInstance(hidden, brainstate.HiddenState)

    def test_roundtrip_softplus(self):
        """Test roundtrip with softplus transform."""
        original = jnp.array([0.5, 1.0, 2.0])
        hidden = ArrayHidden(original, t=SoftplusT(0.0))
        # Set and get should roundtrip
        hidden.data = original
        np.testing.assert_allclose(hidden.data, original, rtol=1e-5)


class TestArrayParam(unittest.TestCase):
    """Tests for ArrayParam state container."""

    def test_basic_instantiation(self):
        """Test basic instantiation with default transform."""
        param = ArrayParam(jnp.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(param.data, jnp.array([1.0, 2.0, 3.0]))

    def test_with_identity_transform(self):
        """Test with explicit identity transform."""
        param = ArrayParam(jnp.array([1.0, 2.0]), t=IdentityT())
        np.testing.assert_allclose(param.data, jnp.array([1.0, 2.0]))

    def test_with_softplus_transform(self):
        """Test with softplus transform for positive constraint."""
        value = jnp.array([1.0, 2.0, 3.0])
        param = ArrayParam(value, t=SoftplusT(0.0))
        # Data should match original value
        np.testing.assert_allclose(param.data, value, rtol=1e-5)
        # Internal value should be transformed
        self.assertFalse(np.allclose(param.value, value))

    def test_with_sigmoid_transform(self):
        """Test with sigmoid transform for bounded constraint."""
        value = jnp.array([0.3, 0.5, 0.7])
        param = ArrayParam(value, t=SigmoidT(0.0, 1.0))
        # Data should approximately match original value
        np.testing.assert_allclose(param.data, value, rtol=1e-4)

    def test_data_setter(self):
        """Test setting data property."""
        param = ArrayParam(jnp.array([1.0, 2.0]), t=SoftplusT(0.0))
        new_value = jnp.array([3.0, 4.0])
        param.data = new_value
        np.testing.assert_allclose(param.data, new_value, rtol=1e-5)

    def test_repr(self):
        """Test string representation."""
        param = ArrayParam(jnp.array([1.0]), t=IdentityT())
        r = repr(param)
        self.assertIn("ArrayParam", r)
        self.assertIn("IdentityT", r)

    def test_inherits_from_param_state(self):
        """Test that ArrayParam inherits from brainstate.ParamState."""
        param = ArrayParam(jnp.array([1.0]))
        self.assertIsInstance(param, brainstate.ParamState)

    def test_roundtrip_softplus(self):
        """Test roundtrip with softplus transform."""
        original = jnp.array([0.5, 1.0, 2.0])
        param = ArrayParam(original, t=SoftplusT(0.0))
        # Set and get should roundtrip
        param.data = original
        np.testing.assert_allclose(param.data, original, rtol=1e-5)


class TestArrayStateTypeError(unittest.TestCase):
    """Tests for type error handling in array state containers."""

    def test_array_hidden_invalid_type(self):
        """Test ArrayHidden raises TypeError for invalid value."""
        with self.assertRaises(TypeError):
            ArrayHidden("not an array")

    def test_array_param_invalid_type(self):
        """Test ArrayParam raises TypeError for invalid value."""
        with self.assertRaises(TypeError):
            ArrayParam("not an array")


if __name__ == '__main__':
    unittest.main()
