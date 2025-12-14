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
import jax.numpy as jnp
import numpy as np

from braintools.param import (
    Identity,
    Sigmoid,
    Softplus,
    NegSoftplus,
    Affine,
    Chain,
    Masked,
    Custom,
    Log,
    Exp,
    Tanh,
    Softsign,
    Positive,
    Negative,
    ScaledSigmoid,
    Power,
    Ordered,
    Simplex,
    UnitVector,
)
from braintools.param._transform import save_exp


class TestSaveExp(unittest.TestCase):
    def test_save_exp_clipping(self):
        large = 1000.0
        out = save_exp(large)
        np.testing.assert_allclose(out, np.exp(20.0), rtol=1e-6)

    def test_save_exp_regular(self):
        x = jnp.array([-2.0, 0.0, 2.0])
        out = save_exp(x)
        np.testing.assert_allclose(out, np.exp(np.array(x)), rtol=1e-6)


class TestIdentityTransform(unittest.TestCase):
    def test_roundtrip(self):
        t = Identity()
        x = jnp.array([-3.0, 0.0, 4.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr)


class TestSigmoidTransform(unittest.TestCase):
    def test_forward_inverse_numeric(self):
        t = Sigmoid(0.0, 1.0)
        x = jnp.array([-5.0, 0.0, 5.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_unit_roundtrip(self):
        unit = u.mV
        t = Sigmoid(0.0 * unit, 1.0 * unit)
        x = jnp.array([-2.0, 0.0, 2.0])
        y = t.forward(x)
        self.assertTrue(isinstance(y, u.Quantity))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_range(self):
        t = Sigmoid(-2.0, 3.0)
        y = t.forward(jnp.array([-100.0, 0.0, 100.0]))
        self.assertTrue(np.all(y >= -2.0))
        self.assertTrue(np.all(y <= 3.0))


class TestSoftplusTransforms(unittest.TestCase):
    def test_softplus_roundtrip(self):
        t = Softplus(0.0)
        x = jnp.array([-5.0, -1.0, 0.0, 2.0, 5.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_negsoftplus_roundtrip(self):
        t = NegSoftplus(0.0)
        x = jnp.array([-5.0, -1.0, 0.0, 2.0, 5.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)


class TestAffineTransform(unittest.TestCase):
    def test_forward_inverse(self):
        t = Affine(2.5, -3.0)
        x = jnp.array([-2.0, 0.0, 1.2])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr)

    def test_invalid_scale_raises(self):
        with self.assertRaises(ValueError):
            _ = Affine(0.0, 1.0)


class TestLogExpTransform(unittest.TestCase):
    def test_log_transform_roundtrip_units(self):
        lower = 1.0 * u.mV
        t = Log(lower)
        x = jnp.array([-3.0, 0.0, 3.0])
        y = t.forward(x)
        self.assertTrue(isinstance(y, u.Quantity))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_exp_transform_equivalent(self):
        lower = 0.5 * u.mV
        t1 = Log(lower)
        t2 = Exp(lower)
        x = jnp.array([-2.0, 0.5, 2.0])
        y1 = t1.forward(x)
        y2 = t2.forward(x)
        assert u.math.allclose(y1, y2)
        xr1 = t1.inverse(y1)
        xr2 = t2.inverse(y2)
        np.testing.assert_allclose(xr1, xr2)


class TestTanhSoftsignTransform(unittest.TestCase):
    def test_tanh_roundtrip_and_range(self):
        t = Tanh(-2.0, 5.0)
        x = jnp.array([-4.0, 0.0, 4.0])
        y = t.forward(x)
        self.assertTrue(np.all(y > -2.0))
        self.assertTrue(np.all(y < 5.0))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-2, atol=1e-2)

    def test_softsign_roundtrip_and_range(self):
        t = Softsign(-1.0, 2.0)
        x = jnp.array([-4.0, 0.0, 4.0])
        y = t.forward(x)
        self.assertTrue(np.all(y > -1.0))
        self.assertTrue(np.all(y < 2.0))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)


class TestChainTransform(unittest.TestCase):
    def test_chain_roundtrip(self):
        # Map R -> (0,1) then affine to (-1,1)
        sigmoid = Sigmoid(0.0, 1.0)
        affine = Affine(2.0, -1.0)
        chain = Chain(sigmoid, affine)
        x = jnp.array([-3.0, 0.0, 3.0])
        y = chain.forward(x)
        self.assertTrue(np.all(y > -1.0))
        self.assertTrue(np.all(y < 1.0))
        xr = chain.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)


class TestMaskedTransform(unittest.TestCase):
    def test_masked_forward_inverse(self):
        mask = jnp.array([False, True, False, True])
        base = Softplus(0.0)
        t = Masked(mask, base)
        x = jnp.array([-1.0, -1.0, 2.0, 2.0])
        y = t.forward(x)
        # Unmasked indices unchanged
        np.testing.assert_allclose(y[0], x[0])
        np.testing.assert_allclose(y[2], x[2])
        # Masked indices transformed (softplus(x) >= 0)
        self.assertTrue(y[1] >= 0.0 and y[3] >= 0.0)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)


class TestCustomTransform(unittest.TestCase):
    def test_custom_roundtrip(self):
        def fwd(x):
            return x ** 3

        def inv(y):
            return jnp.cbrt(y)

        t = Custom(fwd, inv)
        x = jnp.array([-8.0, -1.0, 0.0, 1.0, 8.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-6, atol=1e-7)


class TestPositiveTransform(unittest.TestCase):
    def test_positive_roundtrip(self):
        t = Positive()
        x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        y = t.forward(x)
        # Check all outputs are positive
        self.assertTrue(np.all(y > 0))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_positive_repr(self):
        t = Positive()
        self.assertEqual(repr(t), "Positive()")


class TestNegativeTransform(unittest.TestCase):
    def test_negative_roundtrip(self):
        t = Negative()
        x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        y = t.forward(x)
        # Check all outputs are negative
        self.assertTrue(np.all(y < 0))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_negative_repr(self):
        t = Negative()
        self.assertEqual(repr(t), "Negative()")


class TestScaledSigmoidTransform(unittest.TestCase):
    def test_scaled_sigmoid_roundtrip(self):
        t = ScaledSigmoid(0.0, 1.0, beta=2.0)
        x = jnp.array([-3.0, 0.0, 3.0])
        y = t.forward(x)
        # Check range
        self.assertTrue(np.all(y > 0.0))
        self.assertTrue(np.all(y < 1.0))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_scaled_sigmoid_beta_effect(self):
        t_normal = ScaledSigmoid(0.0, 1.0, beta=1.0)
        t_sharp = ScaledSigmoid(0.0, 1.0, beta=5.0)
        x = jnp.array([0.5])
        y_normal = t_normal.forward(x)
        y_sharp = t_sharp.forward(x)
        # Sharper sigmoid should be closer to 1 for positive x
        self.assertTrue(y_sharp[0] > y_normal[0])

    def test_scaled_sigmoid_repr(self):
        t = ScaledSigmoid(0.0, 1.0, beta=2.0)
        self.assertIn("ScaledSigmoid", repr(t))
        self.assertIn("beta=2.0", repr(t))


class TestPowerTransform(unittest.TestCase):
    def test_power_roundtrip(self):
        t = Power(lmbda=0.5)
        x = jnp.array([0.1, 1.0, 4.0, 9.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_power_lambda_one(self):
        # Lambda = 1 should be close to identity (shifted)
        t = Power(lmbda=1.0)
        x = jnp.array([1.0, 2.0, 3.0])
        y = t.forward(x)
        # y = (x^1 - 1) / 1 = x - 1
        np.testing.assert_allclose(y, x - 1, rtol=1e-5)

    def test_power_repr(self):
        t = Power(lmbda=0.5)
        self.assertEqual(repr(t), "Power(lmbda=0.5)")


class TestOrderedTransform(unittest.TestCase):
    def test_ordered_roundtrip(self):
        t = Ordered()
        x = jnp.array([0.0, 1.0, -0.5, 2.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_ordered_monotonic(self):
        t = Ordered()
        x = jnp.array([0.0, 1.0, -0.5, 2.0])
        y = t.forward(x)
        # Check monotonically increasing
        diffs = jnp.diff(y)
        self.assertTrue(np.all(diffs > 0))

    def test_ordered_repr(self):
        t = Ordered()
        self.assertEqual(repr(t), "Ordered()")


class TestSimplexTransform(unittest.TestCase):
    def test_simplex_roundtrip(self):
        t = Simplex()
        x = jnp.array([0.0, 1.0, -1.0])  # 3D input -> 4D simplex
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-4, atol=1e-5)

    def test_simplex_sums_to_one(self):
        t = Simplex()
        x = jnp.array([0.5, -0.5, 1.0])
        y = t.forward(x)
        np.testing.assert_allclose(jnp.sum(y), 1.0, rtol=1e-6)

    def test_simplex_all_positive(self):
        t = Simplex()
        x = jnp.array([2.0, -2.0, 0.0])
        y = t.forward(x)
        self.assertTrue(np.all(y > 0))

    def test_simplex_repr(self):
        t = Simplex()
        self.assertEqual(repr(t), "Simplex()")


class TestUnitVectorTransform(unittest.TestCase):
    def test_unit_vector_norm(self):
        t = UnitVector()
        x = jnp.array([3.0, 4.0])
        y = t.forward(x)
        np.testing.assert_allclose(jnp.linalg.norm(y), 1.0, rtol=1e-6)

    def test_unit_vector_direction(self):
        t = UnitVector()
        x = jnp.array([3.0, 4.0])
        y = t.forward(x)
        # Direction should be preserved
        np.testing.assert_allclose(y, jnp.array([0.6, 0.8]), rtol=1e-6)

    def test_unit_vector_repr(self):
        t = UnitVector()
        self.assertEqual(repr(t), "UnitVector()")


class TestLogAbsDetJacobian(unittest.TestCase):
    def test_identity_jacobian(self):
        t = Identity()
        x = jnp.array([1.0, 2.0, 3.0])
        y = t.forward(x)
        ladj = t.log_abs_det_jacobian(x, y)
        np.testing.assert_allclose(ladj, 0.0, rtol=1e-6)

    def test_exp_jacobian(self):
        t = Exp(0.0)
        x = jnp.array([0.0, 1.0, 2.0])
        y = t.forward(x)
        ladj = t.log_abs_det_jacobian(x, y)
        # d/dx exp(x) = exp(x), log det = sum(x)
        np.testing.assert_allclose(ladj, jnp.sum(x), rtol=1e-5)

    def test_positive_jacobian(self):
        t = Positive()
        x = jnp.array([0.0, 1.0, 2.0])
        y = t.forward(x)
        ladj = t.log_abs_det_jacobian(x, y)
        np.testing.assert_allclose(ladj, jnp.sum(x), rtol=1e-5)

    def test_affine_jacobian(self):
        t = Affine(2.0, 1.0)
        x = jnp.array([1.0, 2.0, 3.0])
        y = t.forward(x)
        ladj = t.log_abs_det_jacobian(x, y)
        # d/dx (2x + 1) = 2, log det = 3 * log(2)
        np.testing.assert_allclose(ladj, 3 * jnp.log(2.0), rtol=1e-5)


class TestTransformRepr(unittest.TestCase):
    def test_sigmoid_repr(self):
        t = Sigmoid(0.0, 1.0)
        self.assertIn("Sigmoid", repr(t))

    def test_softplus_repr(self):
        t = Softplus(0.0)
        self.assertEqual(repr(t), "Softplus(lower=0.0)")

    def test_chain_repr(self):
        t = Chain(Sigmoid(0.0, 1.0), Affine(2.0, 0.0))
        r = repr(t)
        self.assertIn("Chain", r)
        self.assertIn("Sigmoid", r)
        self.assertIn("Affine", r)

    def test_masked_repr(self):
        mask = jnp.array([True, False])
        t = Masked(mask, Softplus(0.0))
        r = repr(t)
        self.assertIn("Masked", r)
        self.assertIn("Softplus", r)
