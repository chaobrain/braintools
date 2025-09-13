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

import brainunit as u
import jax.numpy as jnp
import numpy as np

from braintools._others.transform import (
    save_exp,
    IdentityTransform,
    SigmoidTransform,
    SoftplusTransform,
    NegSoftplusTransform,
    AffineTransform,
    ChainTransform,
    MaskedTransform,
    CustomTransform,
    LogTransform,
    ExpTransform,
    TanhTransform,
    SoftsignTransform,
)


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
        t = IdentityTransform()
        x = jnp.array([-3.0, 0.0, 4.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr)


class TestSigmoidTransform(unittest.TestCase):
    def test_forward_inverse_numeric(self):
        t = SigmoidTransform(0.0, 1.0)
        x = jnp.array([-5.0, 0.0, 5.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_unit_roundtrip(self):
        unit = u.mV
        t = SigmoidTransform(0.0 * unit, 1.0 * unit)
        x = jnp.array([-2.0, 0.0, 2.0])
        y = t.forward(x)
        self.assertTrue(isinstance(y, u.Quantity))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_range(self):
        t = SigmoidTransform(-2.0, 3.0)
        y = t.forward(jnp.array([-100.0, 0.0, 100.0]))
        self.assertTrue(np.all(y >= -2.0))
        self.assertTrue(np.all(y <= 3.0))


class TestSoftplusTransforms(unittest.TestCase):
    def test_softplus_roundtrip(self):
        t = SoftplusTransform(0.0)
        x = jnp.array([-5.0, -1.0, 0.0, 2.0, 5.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_negsoftplus_roundtrip(self):
        t = NegSoftplusTransform(0.0)
        x = jnp.array([-5.0, -1.0, 0.0, 2.0, 5.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)


class TestAffineTransform(unittest.TestCase):
    def test_forward_inverse(self):
        t = AffineTransform(2.5, -3.0)
        x = jnp.array([-2.0, 0.0, 1.2])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr)

    def test_invalid_scale_raises(self):
        with self.assertRaises(ValueError):
            _ = AffineTransform(0.0, 1.0)


class TestLogExpTransform(unittest.TestCase):
    def test_log_transform_roundtrip_units(self):
        lower = 1.0 * u.mV
        t = LogTransform(lower)
        x = jnp.array([-3.0, 0.0, 3.0])
        y = t.forward(x)
        self.assertTrue(isinstance(y, u.Quantity))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)

    def test_exp_transform_equivalent(self):
        lower = 0.5 * u.mV
        t1 = LogTransform(lower)
        t2 = ExpTransform(lower)
        x = jnp.array([-2.0, 0.5, 2.0])
        y1 = t1.forward(x)
        y2 = t2.forward(x)
        assert u.math.allclose(y1, y2)
        xr1 = t1.inverse(y1)
        xr2 = t2.inverse(y2)
        np.testing.assert_allclose(xr1, xr2)


class TestTanhSoftsignTransform(unittest.TestCase):
    def test_tanh_roundtrip_and_range(self):
        t = TanhTransform(-2.0, 5.0)
        x = jnp.array([-4.0, 0.0, 4.0])
        y = t.forward(x)
        self.assertTrue(np.all(y > -2.0))
        self.assertTrue(np.all(y < 5.0))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-2, atol=1e-2)

    def test_softsign_roundtrip_and_range(self):
        t = SoftsignTransform(-1.0, 2.0)
        x = jnp.array([-4.0, 0.0, 4.0])
        y = t.forward(x)
        self.assertTrue(np.all(y > -1.0))
        self.assertTrue(np.all(y < 2.0))
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)


class TestChainTransform(unittest.TestCase):
    def test_chain_roundtrip(self):
        # Map R -> (0,1) then affine to (-1,1)
        sigmoid = SigmoidTransform(0.0, 1.0)
        affine = AffineTransform(2.0, -1.0)
        chain = ChainTransform(sigmoid, affine)
        x = jnp.array([-3.0, 0.0, 3.0])
        y = chain.forward(x)
        self.assertTrue(np.all(y > -1.0))
        self.assertTrue(np.all(y < 1.0))
        xr = chain.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-5, atol=1e-6)


class TestMaskedTransform(unittest.TestCase):
    def test_masked_forward_inverse(self):
        mask = jnp.array([False, True, False, True])
        base = SoftplusTransform(0.0)
        t = MaskedTransform(mask, base)
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

        t = CustomTransform(fwd, inv)
        x = jnp.array([-8.0, -1.0, 0.0, 1.0, 8.0])
        y = t.forward(x)
        xr = t.inverse(y)
        np.testing.assert_allclose(x, xr, rtol=1e-6, atol=1e-7)
