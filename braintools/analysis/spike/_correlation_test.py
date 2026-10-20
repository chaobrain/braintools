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


import unittest
from functools import partial

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
from jax import jit

import braintools

brainstate.environ.set(dt=0.1)


class TestCrossCorrelation(unittest.TestCase):
    def setUp(self):
        self.rng = brainstate.random.RandomState(0)

    def test_c(self):
        spikes = jnp.asarray([[1, 0, 1, 0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]]).T
        cc1 = braintools.analysis.spike.cross_correlation(spikes, 1., dt=1.)
        f_cc = jit(partial(braintools.analysis.spike.cross_correlation, bin=1, dt=1.))
        cc2 = f_cc(spikes)
        print(cc1, cc2)
        self.assertTrue(cc1 == cc2)

    def test_cc(self):
        spikes = jnp.ones((1000, 10))
        cc1 = braintools.analysis.spike.cross_correlation(spikes, 1.)
        self.assertTrue(cc1 == 1.)

        spikes = jnp.zeros((1000, 10))
        cc2 = braintools.analysis.spike.cross_correlation(spikes, 1.)
        self.assertTrue(cc2 == 0.)

    def test_cc2(self):
        spikes = self.rng.randint(0, 2, (1000, 10))
        print(braintools.analysis.spike.cross_correlation(spikes, 1.))
        print(braintools.analysis.spike.cross_correlation(spikes, 0.5))

    def test_cc3(self):
        spikes = self.rng.random((1000, 100)) < 0.8
        print(braintools.analysis.spike.cross_correlation(spikes, 1.))
        print(braintools.analysis.spike.cross_correlation(spikes, 0.5))

    def test_cc4(self):
        spikes = self.rng.random((1000, 100)) < 0.2
        print(braintools.analysis.spike.cross_correlation(spikes, 1.))
        print(braintools.analysis.spike.cross_correlation(spikes, 0.5))

    def test_cc5(self):
        spikes = self.rng.random((1000, 100)) < 0.05
        print(braintools.analysis.spike.cross_correlation(spikes, 1.))
        print(braintools.analysis.spike.cross_correlation(spikes, 0.5))

    def test_cross_correlation_bounds(self):
        """Test that cross-correlation is bounded between 0 and 1."""
        spikes = self.rng.random((100, 20)) < 0.3
        cc = braintools.analysis.spike.cross_correlation(spikes, 1.0)
        self.assertGreaterEqual(float(cc), 0.0)
        self.assertLessEqual(float(cc), 1.0)

    def test_cross_correlation_single_neuron(self):
        """Test cross-correlation with single neuron (should return 0)."""
        spikes = self.rng.random((100, 1)) < 0.3
        cc = braintools.analysis.spike.cross_correlation(spikes, 1.0)
        # Single neuron should give NaN (no pairs), which gets converted to 0
        self.assertTrue(jnp.isnan(cc) or float(cc) == 0.0)

    def test_cross_correlation_different_bins(self):
        """Test that smaller bins generally give lower correlation."""
        # Create synchronized spikes
        spikes = jnp.zeros((100, 10))
        # Add synchronized bursts
        spikes = spikes.at[10:15, :].set(1)
        spikes = spikes.at[50:55, :].set(1)

        cc_small = braintools.analysis.spike.cross_correlation(spikes, 1.0, dt=1.0)
        cc_large = braintools.analysis.spike.cross_correlation(spikes, 10.0, dt=1.0)

        # Larger bins should capture more synchrony
        self.assertGreaterEqual(float(cc_large), float(cc_small))

    def test_cross_correlation_methods(self):
        """Test that loop and vmap methods give same results."""
        spikes = self.rng.random((50, 10)) < 0.4
        cc_loop = braintools.analysis.spike.cross_correlation(spikes, 2.0, method='loop')
        cc_vmap = braintools.analysis.spike.cross_correlation(spikes, 2.0, method='vmap')
        self.assertAlmostEqual(float(cc_loop), float(cc_vmap), places=5)

    def test_cross_correlation_invalid_method(self):
        """Test invalid method raises error."""
        spikes = jnp.ones((10, 5))
        with self.assertRaises(ValueError):
            braintools.analysis.spike.cross_correlation(spikes, 1.0, method='invalid')


class TestCrossCorrelationRegressions(unittest.TestCase):
    """Targeted regression tests for the audit E-series fixes (cross_correlation)."""

    def setUp(self):
        self.rng = brainstate.random.RandomState(0)

    def test_cross_correlation_bin_smaller_than_dt_raises(self):
        """A bin narrower than dt collapses to <1 sample and must error (E7)."""
        spikes = jnp.ones((100, 5))
        with self.assertRaises(ValueError):
            braintools.analysis.spike.cross_correlation(spikes, bin=0.05, dt=0.1)

    def test_cross_correlation_single_neuron_is_zero(self):
        """A single neuron has no pairs -> exactly 0.0, never NaN (E8)."""
        spikes = (self.rng.random((100, 1)) < 0.3).astype(float)
        cc = braintools.analysis.spike.cross_correlation(spikes, 1.0, dt=1.0)
        self.assertEqual(float(cc), 0.0)
        self.assertFalse(jnp.isnan(cc))

    def test_cross_correlation_non_divisible_padding(self):
        """num_time not divisible by bin_size exercises the zero-pad branch."""
        spikes = (self.rng.random((97, 6)) < 0.4).astype(float)
        cc = braintools.analysis.spike.cross_correlation(spikes, bin=10.0, dt=1.0)
        self.assertGreaterEqual(float(cc), 0.0)
        self.assertLessEqual(float(cc), 1.0)

    def test_cross_correlation_loop_equals_vmap(self):
        """Loop and vmap paths must agree bit-for-bit-ish (E22)."""
        spikes = (self.rng.random((200, 12)) < 0.3).astype(float)
        loop = braintools.analysis.spike.cross_correlation(spikes, 5.0, dt=1.0, method='loop')
        vmp = braintools.analysis.spike.cross_correlation(spikes, 5.0, dt=1.0, method='vmap')
        self.assertAlmostEqual(float(loop), float(vmp), places=6)


class TestCrossCorrelationUnits(unittest.TestCase):
    def test_float_bin_with_environment_dt(self):
        rng = np.random.default_rng(0)
        s = (rng.random((20000, 10)) < 0.002).astype(float)
        ref = float(braintools.analysis.spike.cross_correlation(s, 5., 0.1))
        for b, d in [(5 * u.ms, 0.1 * u.ms), (0.005 * u.second, 0.1 * u.ms), (5 * u.ms, 0.1)]:
            self.assertAlmostEqual(float(braintools.analysis.spike.cross_correlation(s, b, d)), ref, places=6)
        with brainstate.environ.context(dt=0.1 * u.ms):
            self.assertAlmostEqual(float(braintools.analysis.spike.cross_correlation(s, 5.)), ref, places=6)


class TestCOR2CrossCorrelationQuantity(unittest.TestCase):
    def test_quantity_input_does_not_crash(self):
        # COR-2: a Quantity spike matrix previously raised TypeError in jnp.sum.
        rng = np.random.RandomState(1)
        sp = jnp.asarray((rng.rand(50, 3) < 0.2).astype(float))
        out_q = braintools.analysis.spike.cross_correlation(sp * u.UNITLESS, 5.0, dt=1.0)
        out_plain = braintools.analysis.spike.cross_correlation(sp, 5.0, dt=1.0)
        np.testing.assert_allclose(float(out_q), float(out_plain), rtol=1e-6)


class TestCOR4CrossCorrelationZeroBranch(unittest.TestCase):
    def test_all_silent_pair_returns_zero(self):
        # Exercises the lax.cond zero branch (COR-4); must be a clean 0.0.
        spikes = jnp.zeros((100, 3))
        out = braintools.analysis.spike.cross_correlation(spikes, bin=10.0, dt=1.0, method='loop')
        np.testing.assert_allclose(float(out), 0.0, atol=1e-7)
        out_v = braintools.analysis.spike.cross_correlation(spikes, bin=10.0, dt=1.0, method='vmap')
        np.testing.assert_allclose(float(out_v), 0.0, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
