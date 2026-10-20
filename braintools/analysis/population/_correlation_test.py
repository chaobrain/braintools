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


import math
import unittest
from functools import partial

import brainstate
import jax.numpy as jnp
from jax import jit

import braintools

brainstate.environ.set(dt=0.1)


class TestVoltageFluctuation(unittest.TestCase):
    def setUp(self):
        self.rng = brainstate.random.RandomState(0)

    def test_vf1(self):
        voltages = self.rng.normal(0, 10, size=(100, 10))
        print(braintools.analysis.population.voltage_fluctuation(voltages))

        with brainstate.environ.context(precision=64):
            voltages = jnp.ones((100, 10))
            r1 = braintools.analysis.population.voltage_fluctuation(voltages)

            jit_f = jit(partial(braintools.analysis.population.voltage_fluctuation))
            jit_f = jit(lambda a: braintools.analysis.population.voltage_fluctuation(a))
            r2 = jit_f(voltages)
            print(r1, r2)  # TODO: JIT results are different?
            # self.assertTrue(r1 == r2)

    def test_voltage_fluctuation_synchronized(self):
        """Test voltage fluctuation for synchronized signals."""
        # Create synchronized oscillations with stronger synchrony
        t = jnp.linspace(0, 10, 100)
        base_signal = 10 * jnp.sin(2 * jnp.pi * t)  # Stronger base signal

        # Add small amounts of noise to each neuron
        voltages = []
        for i in range(10):
            noise = self.rng.normal(0, 0.5, size=100)  # Small noise
            voltages.append(base_signal + noise)
        voltages = jnp.array(voltages).T

        sync_index = braintools.analysis.population.voltage_fluctuation(voltages)
        # Synchronized signals push the index toward its upper bound of 1.
        self.assertGreater(float(sync_index), 0.5)  # More relaxed threshold

    def test_voltage_fluctuation_asynchronous(self):
        """Test voltage fluctuation for asynchronous signals."""
        # Create independent random signals
        voltages = self.rng.normal(0, 1, size=(100, 10))
        sync_index = braintools.analysis.population.voltage_fluctuation(voltages)

        # Asynchronous signals have a small index (~1/N); it stays within [0, 1].
        self.assertGreater(float(sync_index), 0.0)
        self.assertLessEqual(float(sync_index), 1.0 + 1e-4)

    def test_voltage_fluctuation_constant(self):
        """Test voltage fluctuation for constant signals."""
        # All neurons have constant voltage
        voltages = jnp.ones((100, 10))
        sync_index = braintools.analysis.population.voltage_fluctuation(voltages)

        # Constant signals have zero variance; by convention the index is 1.0.
        # This previously returned -0.0 due to float32 cancellation in
        # ``mean(s**2) - mean(s)**2`` slipping past the ``== 0`` guard (audit E18).
        self.assertFalse(math.isnan(float(sync_index)))
        self.assertEqual(float(sync_index), 1.0)

    def test_voltage_fluctuation_methods(self):
        """Test that loop and vmap methods give same results."""
        voltages = self.rng.normal(0, 1, size=(50, 8))

        sync_loop = braintools.analysis.population.voltage_fluctuation(voltages, method='loop')
        sync_vmap = braintools.analysis.population.voltage_fluctuation(voltages, method='vmap')

        self.assertAlmostEqual(float(sync_loop), float(sync_vmap), places=5)

    def test_voltage_fluctuation_invalid_method(self):
        """Test invalid method raises error."""
        voltages = jnp.ones((10, 5))
        with self.assertRaises(ValueError):
            braintools.analysis.population.voltage_fluctuation(voltages, method='invalid')

    def test_voltage_fluctuation_single_neuron(self):
        """Test voltage fluctuation with single neuron."""
        voltages = self.rng.normal(0, 1, size=(100, 1))
        sync_index = braintools.analysis.population.voltage_fluctuation(voltages)

        # Single neuron case should be handled gracefully
        self.assertFalse(math.isnan(float(sync_index)))


class TestFunctionalConnectivity(unittest.TestCase):
    def setUp(self):
        self.rng = brainstate.random.RandomState(0)

    def test_cf1(self):
        act = self.rng.random((10000, 3))
        r1 = braintools.analysis.population.functional_connectivity(act)

        jit_f = jit(partial(braintools.analysis.population.functional_connectivity))
        r2 = jit_f(act)

        self.assertTrue(jnp.allclose(r1, r2))

    def test_functional_connectivity_properties(self):
        """Test basic properties of functional connectivity matrix."""
        activities = self.rng.normal(0, 1, size=(1000, 5))
        fc = braintools.analysis.population.functional_connectivity(activities)

        # Should be square matrix
        self.assertEqual(fc.shape, (5, 5))

        # Should be symmetric
        self.assertTrue(jnp.allclose(fc, fc.T))

        # Diagonal should be 1 (self-correlation)
        self.assertTrue(jnp.allclose(jnp.diag(fc), 1.0))

        # All values should be between -1 and 1
        self.assertTrue(jnp.all(fc >= -1.0))
        self.assertTrue(jnp.all(fc <= 1.0))

    def test_functional_connectivity_perfect_correlation(self):
        """Test perfect correlation case."""
        # Create perfectly correlated signals
        base_signal = self.rng.normal(0, 1, size=1000)
        activities = jnp.column_stack([base_signal, base_signal, base_signal])

        fc = braintools.analysis.population.functional_connectivity(activities)

        # All correlations should be 1
        expected = jnp.ones((3, 3))
        self.assertTrue(jnp.allclose(fc, expected, atol=1e-6))

    def test_functional_connectivity_anticorrelation(self):
        """Test anti-correlation case."""
        # Create anti-correlated signals
        base_signal = self.rng.normal(0, 1, size=1000)
        activities = jnp.column_stack([base_signal, -base_signal])

        fc = braintools.analysis.population.functional_connectivity(activities)

        # Off-diagonal should be -1
        self.assertAlmostEqual(float(fc[0, 1]), -1.0, places=5)
        self.assertAlmostEqual(float(fc[1, 0]), -1.0, places=5)

    def test_functional_connectivity_input_validation(self):
        """Test input validation."""
        # Test non-2D input
        with self.assertRaises(ValueError):
            braintools.analysis.population.functional_connectivity(jnp.array([1, 2, 3]))

        # Test valid 2D input
        activities = self.rng.normal(0, 1, size=(100, 3))
        fc = braintools.analysis.population.functional_connectivity(activities)
        self.assertEqual(fc.shape, (3, 3))

    def test_functional_connectivity_nan_handling(self):
        """Test that NaN values are handled correctly."""
        # Create data with constant signal (would produce NaN in correlation)
        activities = jnp.ones((100, 3))
        fc = braintools.analysis.population.functional_connectivity(activities)

        # Should replace NaN with 0
        self.assertFalse(jnp.any(jnp.isnan(fc)))


class TestFunctionalConnectivityDynamics(unittest.TestCase):
    def setUp(self):
        self.rng = brainstate.random.RandomState(0)

    def test_fcd_basic(self):
        """Test basic functionality of functional connectivity dynamics."""
        activities = self.rng.normal(0, 1, size=(200, 10))
        fcd = braintools.analysis.population.functional_connectivity_dynamics(activities, window_size=30, step_size=5)

        # Check output shape
        expected_windows = (200 - 30) // 5 + 1
        self.assertEqual(fcd.shape, (expected_windows, expected_windows))

        # Should be symmetric
        self.assertTrue(jnp.allclose(fcd, fcd.T))

        # Diagonal should be 1
        self.assertTrue(jnp.allclose(jnp.diag(fcd), 1.0))

    def test_fcd_edge_cases(self):
        """Test edge cases for FCD."""
        # Too short time series
        short_activities = self.rng.normal(0, 1, size=(10, 5))
        fcd = braintools.analysis.population.functional_connectivity_dynamics(short_activities, window_size=20)
        self.assertEqual(fcd.shape, (0, 0))

        # Single window case
        activities = self.rng.normal(0, 1, size=(30, 5))
        fcd = braintools.analysis.population.functional_connectivity_dynamics(activities, window_size=30, step_size=30)
        self.assertEqual(fcd.shape, (1, 1))
        self.assertEqual(float(fcd[0, 0]), 1.0)

    def test_fcd_parameter_validation(self):
        """Test parameter validation for FCD."""
        activities = self.rng.normal(0, 1, size=(100, 5))

        # Invalid window size
        with self.assertRaises(ValueError):
            braintools.analysis.population.functional_connectivity_dynamics(activities, window_size=1)
        with self.assertRaises(ValueError):
            braintools.analysis.population.functional_connectivity_dynamics(activities, window_size=0)

        # Invalid step size
        with self.assertRaises(ValueError):
            braintools.analysis.population.functional_connectivity_dynamics(activities, step_size=0)
        with self.assertRaises(ValueError):
            braintools.analysis.population.functional_connectivity_dynamics(activities, step_size=-1)

        # Invalid input dimensions
        with self.assertRaises(ValueError):
            braintools.analysis.population.functional_connectivity_dynamics(jnp.array([1, 2, 3]))

    def test_fcd_consistent_patterns(self):
        """Test FCD with consistent connectivity patterns."""
        # Create data with consistent correlations
        t = jnp.linspace(0, 20, 200)
        sig1 = jnp.sin(2 * jnp.pi * t)
        sig2 = jnp.sin(2 * jnp.pi * t + 0.1)  # Slightly phase-shifted
        sig3 = self.rng.normal(0, 1, size=200)  # Uncorrelated

        activities = jnp.column_stack([sig1, sig2, sig3])
        fcd = braintools.analysis.population.functional_connectivity_dynamics(activities, window_size=40, step_size=10)

        # FCD should show high correlation between windows (consistent patterns)
        off_diagonal = fcd[jnp.triu_indices_from(fcd, k=1)]
        mean_fcd = jnp.mean(off_diagonal)
        self.assertGreater(float(mean_fcd), 0.5)  # Should be reasonably high

    def test_fcd_bounds(self):
        """Test that FCD values are properly bounded."""
        activities = self.rng.normal(0, 1, size=(150, 8))
        fcd = braintools.analysis.population.functional_connectivity_dynamics(activities, window_size=25, step_size=8)

        # All values should be between -1 and 1
        self.assertTrue(jnp.all(fcd >= -1.0))
        self.assertTrue(jnp.all(fcd <= 1.0))


class TestPopulationRegressions(unittest.TestCase):
    """Targeted regression tests for the audit E-series fixes (population measures)."""

    def setUp(self):
        self.rng = brainstate.random.RandomState(0)

    def test_voltage_fluctuation_strips_units(self):
        """Quantity-valued potentials are accepted via get_magnitude."""
        import brainunit as u
        raw = self.rng.normal(0, 5, size=(100, 8))
        plain = braintools.analysis.population.voltage_fluctuation(raw)
        quantity = braintools.analysis.population.voltage_fluctuation(raw * u.mV)
        self.assertAlmostEqual(float(plain), float(quantity), places=5)

    def test_voltage_fluctuation_loop_equals_vmap_random(self):
        v = self.rng.normal(0, 3, size=(120, 9))
        loop = braintools.analysis.population.voltage_fluctuation(v, method='loop')
        vmp = braintools.analysis.population.voltage_fluctuation(v, method='vmap')
        self.assertAlmostEqual(float(loop), float(vmp), places=5)

    def test_functional_connectivity_constant_column_diagonal_is_one(self):
        """A constant column yields NaN corr; the diagonal must stay 1.0 (E13)."""
        good = self.rng.normal(0, 1, size=(200, 1))
        const = jnp.ones((200, 1))
        activities = jnp.column_stack([good[:, 0], const[:, 0]])
        fc = braintools.analysis.population.functional_connectivity(activities)
        self.assertFalse(jnp.any(jnp.isnan(fc)))
        self.assertTrue(jnp.allclose(jnp.diag(fc), 1.0))

    def test_fcd_requires_two_signals(self):
        """FCD with a single signal is undefined and must error (E15)."""
        activities = self.rng.normal(0, 1, size=(100, 1))
        with self.assertRaises(ValueError):
            braintools.analysis.population.functional_connectivity_dynamics(
                activities, window_size=30, step_size=5)


if __name__ == '__main__':
    unittest.main()
