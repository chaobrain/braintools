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

"""Tests for regularization classes."""

import unittest

import brainstate
import jax.numpy as jnp
import numpy as np

from braintools.param import (
    GaussianReg,
    L1Reg,
    L2Reg,
)


class TestL2Reg(unittest.TestCase):
    """Tests for L2 (Ridge) regularization."""

    def test_basic_loss(self):
        """Test L2 loss computation."""
        reg = L2Reg(weight=1.0)
        value = jnp.array([1.0, 2.0, 3.0])
        loss = reg.loss(value)
        # L2 loss = 1.0 * (1^2 + 2^2 + 3^2) = 14.0
        np.testing.assert_allclose(loss, 14.0, rtol=1e-5)

    def test_weighted_loss(self):
        """Test L2 loss with different weight."""
        reg = L2Reg(weight=0.5)
        value = jnp.array([2.0, 2.0])
        loss = reg.loss(value)
        # L2 loss = 0.5 * (4 + 4) = 4.0
        np.testing.assert_allclose(loss, 4.0, rtol=1e-5)

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = L2Reg(weight=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_sample_init_int_shape(self):
        """Test sample_init with int shape."""
        reg = L2Reg(weight=1.0)
        sample = reg.sample_init(5)
        self.assertEqual(sample.shape, (5,))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = L2Reg(weight=1.0)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_false(self):
        """Test that weight is not trainable by default."""
        reg = L2Reg(weight=1.0, fit_hyper=False)
        self.assertFalse(reg.fit_hyper)
        self.assertNotIsInstance(reg.weight, brainstate.State)

    def test_fit_hyper_true(self):
        """Test that weight is trainable when fit_hyper=True."""
        reg = L2Reg(weight=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)
        self.assertIsInstance(reg.weight, brainstate.ParamState)


class TestL1Reg(unittest.TestCase):
    """Tests for L1 (Lasso) regularization."""

    def test_basic_loss(self):
        """Test L1 loss computation."""
        reg = L1Reg(weight=1.0)
        value = jnp.array([1.0, -2.0, 3.0])
        loss = reg.loss(value)
        # L1 loss = 1.0 * (|1| + |-2| + |3|) = 6.0
        np.testing.assert_allclose(loss, 6.0, rtol=1e-5)

    def test_weighted_loss(self):
        """Test L1 loss with different weight."""
        reg = L1Reg(weight=0.5)
        value = jnp.array([2.0, -2.0])
        loss = reg.loss(value)
        # L1 loss = 0.5 * (2 + 2) = 2.0
        np.testing.assert_allclose(loss, 2.0, rtol=1e-5)

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = L1Reg(weight=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_sample_init_int_shape(self):
        """Test sample_init with int shape."""
        reg = L1Reg(weight=1.0)
        sample = reg.sample_init(5)
        self.assertEqual(sample.shape, (5,))

    def test_reset_value(self):
        """Test reset_value returns zero."""
        reg = L1Reg(weight=1.0)
        reset = reg.reset_value()
        self.assertEqual(reset, 0.0)

    def test_fit_hyper_false(self):
        """Test that weight is not trainable by default."""
        reg = L1Reg(weight=1.0, fit_hyper=False)
        self.assertFalse(reg.fit_hyper)
        self.assertNotIsInstance(reg.weight, brainstate.State)

    def test_fit_hyper_true(self):
        """Test that weight is trainable when fit_hyper=True."""
        reg = L1Reg(weight=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)
        self.assertIsInstance(reg.weight, brainstate.ParamState)


class TestGaussianReg(unittest.TestCase):
    """Tests for Gaussian prior regularization."""

    def test_loss_at_mean(self):
        """Test that loss is minimized at mean."""
        reg = GaussianReg(mean=0.0, std=1.0)
        loss_at_mean = reg.loss(jnp.array([0.0]))
        loss_away = reg.loss(jnp.array([1.0]))
        self.assertLess(float(loss_at_mean), float(loss_away))

    def test_loss_increases_with_distance(self):
        """Test that loss increases with distance from mean."""
        reg = GaussianReg(mean=0.0, std=1.0)
        loss_1 = reg.loss(jnp.array([1.0]))
        loss_2 = reg.loss(jnp.array([2.0]))
        self.assertLess(float(loss_1), float(loss_2))

    def test_sample_init_shape(self):
        """Test sample_init returns correct shape."""
        reg = GaussianReg(mean=0.0, std=1.0)
        sample = reg.sample_init((3, 4))
        self.assertEqual(sample.shape, (3, 4))

    def test_sample_init_int_shape(self):
        """Test sample_init with int shape."""
        reg = GaussianReg(mean=0.0, std=1.0)
        sample = reg.sample_init(5)
        self.assertEqual(sample.shape, (5,))

    def test_reset_value(self):
        """Test reset_value returns mean."""
        reg = GaussianReg(mean=2.5, std=1.0)
        reset = reg.reset_value()
        np.testing.assert_allclose(reset, 2.5)

    def test_fit_hyper_false(self):
        """Test that hyperparams are not trainable by default."""
        reg = GaussianReg(mean=0.0, std=1.0, fit_hyper=False)
        self.assertFalse(reg.fit_hyper)
        self.assertNotIsInstance(reg.mean, brainstate.State)
        self.assertNotIsInstance(reg.precision, brainstate.State)

    def test_fit_hyper_true(self):
        """Test that hyperparams are trainable when fit_hyper=True."""
        reg = GaussianReg(mean=0.0, std=1.0, fit_hyper=True)
        self.assertTrue(reg.fit_hyper)
        self.assertIsInstance(reg.mean, brainstate.ParamState)
        self.assertIsInstance(reg.precision, brainstate.ParamState)

    def test_array_mean_std(self):
        """Test with array-valued mean and std."""
        reg = GaussianReg(mean=jnp.array([0.0, 1.0]), std=jnp.array([1.0, 2.0]))
        reset = reg.reset_value()
        np.testing.assert_allclose(reset, jnp.array([0.0, 1.0]))


class TestRegularizationInheritance(unittest.TestCase):
    """Tests for regularization inheritance."""

    def test_l2_inherits_from_module(self):
        """Test L2Reg inherits from brainstate.nn.Module."""
        reg = L2Reg(weight=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)

    def test_l1_inherits_from_module(self):
        """Test L1Reg inherits from brainstate.nn.Module."""
        reg = L1Reg(weight=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)

    def test_gaussian_inherits_from_module(self):
        """Test GaussianReg inherits from brainstate.nn.Module."""
        reg = GaussianReg(mean=0.0, std=1.0)
        self.assertIsInstance(reg, brainstate.nn.Module)


if __name__ == '__main__':
    unittest.main()
