# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braintools.analysis.field import unitary_LFP, current_source_density


class TestUnitaryLFP(unittest.TestCase):
    def test_invalid_spike_type(self):
        times = jnp.arange(100) * 0.1
        spikes = jnp.ones((100, 10))
        with self.assertRaises(ValueError) as context:
            unitary_LFP(times, spikes, 'invalid_type')
        self.assertIn('"spike_type" should be "exc" or "inh"', str(context.exception))

    def test_basic_functionality(self):
        """Test basic LFP generation from spikes."""
        times = jnp.arange(1000) * 0.1
        spikes = jnp.zeros((1000, 10))
        spikes = spikes.at[100, :].set(1)  # Synchronized spikes

        lfp_exc = unitary_LFP(times, spikes, 'exc', seed=42)
        lfp_inh = unitary_LFP(times, spikes, 'inh', seed=42)

        self.assertEqual(lfp_exc.shape, (1000,))
        self.assertEqual(lfp_inh.shape, (1000,))
        self.assertFalse(jnp.allclose(lfp_exc, 0.0))
        self.assertFalse(jnp.allclose(lfp_inh, 0.0))

    def test_different_locations(self):
        """Test different recording locations."""
        times = jnp.arange(500) * 0.1
        spikes = jnp.zeros((500, 5))
        spikes = spikes.at[50::100, :].set(1)

        locations = ['soma layer', 'deep layer', 'superficial layer', 'surface layer']
        for location in locations:
            lfp = unitary_LFP(times, spikes, 'exc', location=location, seed=42)
            self.assertEqual(lfp.shape, (500,))
            self.assertFalse(jnp.allclose(lfp, 0.0))


class TestCurrentSourceDensity(unittest.TestCase):
    def test_basic_csd(self):
        """Test basic CSD calculation."""
        # Simulate laminar LFP data
        n_time, n_electrodes = 1000, 8
        lfp_data = jnp.ones((n_time, n_electrodes))

        # Add gradient across electrodes
        for i in range(n_electrodes):
            lfp_data = lfp_data.at[:, i].set(i * 0.1)

        csd = current_source_density(lfp_data, electrode_spacing=0.1)

        # CSD should have 2 fewer electrodes due to boundary conditions
        self.assertEqual(csd.shape, (n_time, n_electrodes - 2))

    def test_uniform_field(self):
        """Test CSD with uniform field should be zero."""
        n_time, n_electrodes = 500, 6
        lfp_uniform = jnp.ones((n_time, n_electrodes)) * 5.0

        csd = current_source_density(lfp_uniform, electrode_spacing=0.1)

        # Uniform field should give zero CSD
        self.assertTrue(jnp.allclose(csd, 0.0, atol=1e-10))

    def test_different_spacing(self):
        """Test CSD with different electrode spacings."""
        n_time, n_electrodes = 100, 5
        brainstate.random.seed(42)
        lfp_data = brainstate.random.normal(size=(n_time, n_electrodes))

        csd1 = current_source_density(lfp_data, electrode_spacing=0.1)
        csd2 = current_source_density(lfp_data, electrode_spacing=0.2)

        # Different spacings should give different results
        self.assertFalse(jnp.allclose(csd1, csd2))
        self.assertEqual(csd1.shape, csd2.shape)


class TestFieldRegressionAndFeatures(unittest.TestCase):
    """Stronger assertions targeting the bugs the original suite masked."""

    def setUp(self):
        self.dt = 0.001
        self.t = np.arange(4000) * self.dt
        self.rng = np.random.RandomState(0)

    def test_csd_curved_profile_nonzero(self):
        # A curved (sinusoidal) laminar profile has a non-zero 2nd derivative.
        profile = np.sin(np.linspace(0, np.pi, 16))
        lam = np.tile(profile, (50, 1))  # (time, 16 electrodes)
        csd = current_source_density(lam, electrode_spacing=0.1)
        self.assertEqual(csd.shape, (50, 14))
        self.assertGreater(float(jnp.max(jnp.abs(csd))), 0.0)

    def test_csd_axis_and_conductivity(self):
        profile = np.sin(np.linspace(0, np.pi, 16))
        lam_tf = np.tile(profile, (50, 1))            # (time, elec)
        csd_default = current_source_density(lam_tf, 0.1)
        # channels-first via axis=0 should match the transpose.
        csd_axis0 = current_source_density(lam_tf.T, 0.1, axis=0)
        np.testing.assert_allclose(np.asarray(csd_default), np.asarray(csd_axis0).T, rtol=1e-6)
        # conductivity scales linearly.
        csd_sigma = current_source_density(lam_tf, 0.1, conductivity=2.0)
        np.testing.assert_allclose(np.asarray(csd_sigma), 2.0 * np.asarray(csd_default), rtol=1e-6)

    def test_csd_quantity_spacing(self):
        lam = np.tile(np.sin(np.linspace(0, np.pi, 6)), (20, 1))
        csd_mm = current_source_density(lam, 0.1)
        csd_q = current_source_density(lam, 100.0 * u.um)  # 100 um == 0.1 mm
        np.testing.assert_allclose(np.asarray(csd_mm), np.asarray(csd_q), rtol=1e-5)

    def test_csd_too_few_electrodes_raises(self):
        with self.assertRaises(ValueError):
            current_source_density(np.ones((10, 2)), 0.1)

    def test_surface_location_alias(self):
        times = jnp.arange(200) * 0.1
        spikes = jnp.zeros((200, 5)).at[50, :].set(1.0)
        lfp_a = unitary_LFP(times, spikes, 'exc', location='surface', seed=3)
        lfp_b = unitary_LFP(times, spikes, 'exc', location='surface layer', seed=3)
        np.testing.assert_allclose(np.asarray(lfp_a), np.asarray(lfp_b), rtol=1e-6)

    def test_unitary_lfp_shape_errors(self):
        times = jnp.arange(100) * 0.1
        with self.assertRaises(ValueError):  # not 2-D
            unitary_LFP(times, jnp.ones(100), 'exc')
        with self.assertRaises(ValueError):  # times/spikes mismatch
            unitary_LFP(times, jnp.ones((50, 5)), 'exc')

    def test_unknown_location_raises(self):
        times = jnp.arange(50) * 0.1
        spikes = jnp.ones((50, 3))
        with self.assertRaises(NotImplementedError):
            unitary_LFP(times, spikes, 'exc', location='nowhere')


if __name__ == '__main__':
    unittest.main()
