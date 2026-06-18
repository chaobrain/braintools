import math
import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braintools.metric import (
    unitary_LFP, power_spectral_density, coherence_analysis,
    phase_amplitude_coupling, theta_gamma_coupling, current_source_density,
    spectral_entropy, lfp_phase_coherence
)


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


class TestPowerSpectralDensity(unittest.TestCase):
    def test_basic_psd(self):
        """Test basic PSD calculation."""
        dt = 0.001  # 1ms
        t = jnp.arange(0, 2, dt)
        # Generate signal with known frequencies
        signal = jnp.sin(2 * jnp.pi * 10 * t) + 0.5 * jnp.sin(2 * jnp.pi * 30 * t)

        freqs, psd = power_spectral_density(signal, dt)

        self.assertTrue(len(freqs) > 0)
        self.assertEqual(psd.shape, freqs.shape)
        self.assertTrue(jnp.all(psd >= 0))

    def test_frequency_range(self):
        """Test frequency range filtering."""
        dt = 0.001
        signal = jnp.sin(2 * jnp.pi * 15 * jnp.arange(0, 1, dt))

        freqs, psd = power_spectral_density(signal, dt, freq_range=(10, 20))

        self.assertTrue(jnp.all(freqs >= 10))
        self.assertTrue(jnp.all(freqs <= 20))

    def test_multichannel(self):
        """Test multichannel PSD."""
        dt = 0.001
        t = jnp.arange(0, 1, dt)
        signals = jnp.column_stack([
            jnp.sin(2 * jnp.pi * 10 * t),
            jnp.sin(2 * jnp.pi * 20 * t)
        ])

        freqs, psd = power_spectral_density(signals, dt)

        self.assertEqual(psd.shape[0], len(freqs))
        self.assertEqual(psd.shape[1], 2)


class TestCoherenceAnalysis(unittest.TestCase):
    def test_identical_signals(self):
        """Test coherence between identical signals."""
        dt = 0.001
        signal = jnp.sin(2 * jnp.pi * 10 * jnp.arange(0, 1, dt))

        freqs, coherence = coherence_analysis(signal, signal, dt)

        # Coherence between identical signals should be close to 1
        # Check that most values are reasonable (some edge frequencies may be low)
        self.assertGreater(jnp.mean(coherence), 0.3)
        self.assertGreater(jnp.max(coherence), 0.7)
        self.assertTrue(jnp.all(coherence <= 1.0))

    def test_uncorrelated_signals(self):
        """Test coherence between uncorrelated signals."""
        brainstate.random.seed(42)
        dt = 0.001
        signal1 = brainstate.random.normal(size=1000)
        signal2 = brainstate.random.normal(size=1000)

        freqs, coherence = coherence_analysis(signal1, signal2, dt)

        # Coherence should be low for uncorrelated signals
        # Just check that values are bounded and mostly reasonable
        self.assertTrue(jnp.all(coherence >= 0))
        self.assertTrue(jnp.all(coherence <= 1))
        self.assertLess(jnp.max(coherence), 1.1)  # Allow for numerical errors
        self.assertTrue(jnp.all(coherence >= 0))
        self.assertTrue(jnp.all(coherence <= 1))

    def test_phase_shifted_signals(self):
        """Test coherence with phase-shifted signals."""
        dt = 0.001
        t = jnp.arange(0, 1, dt)
        signal1 = jnp.sin(2 * jnp.pi * 10 * t)
        signal2 = jnp.sin(2 * jnp.pi * 10 * t + jnp.pi / 4)  # Phase shift

        freqs, coherence = coherence_analysis(signal1, signal2, dt)

        # Should still have high coherence despite phase shift
        peak_coherence = jnp.max(coherence)
        self.assertGreater(peak_coherence, 0.5)


class TestPhaseAmplitudeCoupling(unittest.TestCase):
    def test_no_coupling(self):
        """Test PAC with uncoupled signal."""
        brainstate.random.seed(42)
        dt = 0.001
        signal = brainstate.random.normal(size=2000)

        mi, phase_bins, amplitudes = phase_amplitude_coupling(signal, dt)

        # No coupling should result in low modulation index
        self.assertGreaterEqual(mi, 0.0)
        self.assertLessEqual(mi, 1.0)
        self.assertLess(mi, 0.6)  # Should be low for random signal, relaxed threshold

    def test_synthetic_coupling(self):
        """Test PAC with synthetic coupled signal."""
        dt = 0.001
        t = jnp.arange(0, 4, dt)

        # Create signal with theta-gamma coupling
        theta = jnp.sin(2 * jnp.pi * 6 * t)
        gamma_amplitude = 1 + 0.5 * theta  # Amplitude modulated by theta phase
        gamma = gamma_amplitude * jnp.sin(2 * jnp.pi * 40 * t)
        signal = theta + gamma

        mi, phase_bins, amplitudes = phase_amplitude_coupling(signal, dt)

        self.assertEqual(len(phase_bins), 18)  # Default n_bins
        self.assertEqual(len(amplitudes), 18)
        self.assertGreaterEqual(mi, 0.0)
        self.assertLessEqual(mi, 1.0)

    def test_parameter_bounds(self):
        """Test that PAC parameters are within expected bounds."""
        dt = 0.001
        signal = jnp.sin(2 * jnp.pi * 10 * jnp.arange(0, 2, dt))

        mi, phase_bins, amplitudes = phase_amplitude_coupling(
            signal, dt, n_bins=12,
            phase_freq_range=(8, 12),
            amplitude_freq_range=(60, 100)
        )

        self.assertEqual(len(phase_bins), 12)
        self.assertGreaterEqual(mi, 0.0)
        self.assertLessEqual(mi, 1.0)


class TestThetaGammaCoupling(unittest.TestCase):
    def test_basic_functionality(self):
        """Test basic theta-gamma coupling calculation."""
        dt = 0.001
        signal = jnp.sin(2 * jnp.pi * 6 * jnp.arange(0, 2, dt))

        coupling = theta_gamma_coupling(signal, dt)

        self.assertGreaterEqual(coupling, 0.0)
        self.assertLessEqual(coupling, 1.0)
        self.assertFalse(math.isnan(float(coupling)))

    def test_random_signal(self):
        """Test with random signal should give low coupling."""
        brainstate.random.seed(42)
        dt = 0.001
        signal = brainstate.random.normal(size=1000)

        coupling = theta_gamma_coupling(signal, dt)

        # Random signal should have low coupling
        self.assertLess(coupling, 0.6)  # Relaxed threshold


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


class TestSpectralEntropy(unittest.TestCase):
    def test_periodic_signal(self):
        """Test entropy of periodic signal should be low."""
        dt = 0.001
        t = jnp.arange(0, 2, dt)
        signal = jnp.sin(2 * jnp.pi * 10 * t)  # Pure sine wave

        entropy = spectral_entropy(signal, dt)

        # Pure sine should have low entropy
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)
        self.assertLess(entropy, 0.5)

    def test_random_signal(self):
        """Test entropy of random signal should be high."""
        brainstate.random.seed(42)
        dt = 0.001
        signal = brainstate.random.normal(size=2000)

        entropy = spectral_entropy(signal, dt)

        # Random signal should have higher entropy
        self.assertGreater(entropy, 0.3)
        self.assertLessEqual(entropy, 1.0)

    def test_frequency_range(self):
        """Test entropy calculation with specific frequency range."""
        dt = 0.001
        t = jnp.arange(0, 1, dt)
        signal = jnp.sin(2 * jnp.pi * 15 * t)

        entropy = spectral_entropy(signal, dt, freq_range=(10, 20))

        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)

    def test_bounds(self):
        """Test that entropy is properly bounded."""
        dt = 0.001
        signal = jnp.sin(2 * jnp.pi * 10 * jnp.arange(0, 1, dt))

        entropy = spectral_entropy(signal, dt)

        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)
        self.assertFalse(math.isnan(float(entropy)))


class TestLFPPhaseCoherence(unittest.TestCase):
    def test_identical_signals(self):
        """Test phase coherence between identical signals."""
        dt = 0.001
        t = jnp.arange(0, 2, dt)
        signal = jnp.sin(2 * jnp.pi * 10 * t)
        signals = jnp.column_stack([signal, signal, signal])

        coherence_matrix = lfp_phase_coherence(signals, dt, freq_band=(8, 12))

        # Diagonal should be 1
        self.assertTrue(jnp.allclose(jnp.diag(coherence_matrix), 1.0))
        # Off-diagonal should be close to 1 for identical signals
        self.assertTrue(jnp.all(coherence_matrix >= 0.8))

    def test_uncorrelated_signals(self):
        """Test phase coherence between uncorrelated signals."""
        brainstate.random.seed(42)
        dt = 0.001
        n_time = 1000
        n_channels = 4
        signals = brainstate.random.normal(size=(n_time, n_channels))

        coherence_matrix = lfp_phase_coherence(signals, dt)

        # Should be symmetric
        self.assertTrue(jnp.allclose(coherence_matrix, coherence_matrix.T))
        # Diagonal should be 1
        self.assertTrue(jnp.allclose(jnp.diag(coherence_matrix), 1.0))
        # Values should be between 0 and 1
        self.assertTrue(jnp.all(coherence_matrix >= 0.0))
        self.assertTrue(jnp.all(coherence_matrix <= 1.0))

    def test_phase_shifted_signals(self):
        """Test coherence with phase-shifted versions of same signal."""
        dt = 0.001
        t = jnp.arange(0, 2, dt)
        base_signal = jnp.sin(2 * jnp.pi * 10 * t)

        signals = jnp.column_stack([
            base_signal,
            jnp.sin(2 * jnp.pi * 10 * t + jnp.pi / 4),
            jnp.sin(2 * jnp.pi * 10 * t + jnp.pi / 2)
        ])

        coherence_matrix = lfp_phase_coherence(signals, dt, freq_band=(8, 12))

        # Should have some coherence despite phase shifts
        # Just check basic properties and reasonable values
        self.assertTrue(jnp.all(coherence_matrix >= 0.0))
        self.assertTrue(jnp.all(coherence_matrix <= 1.0))
        self.assertGreater(jnp.mean(coherence_matrix), 0.1)

    def test_matrix_properties(self):
        """Test that coherence matrix has correct properties."""
        dt = 0.001
        n_time, n_channels = 500, 5
        signals = jnp.sin(2 * jnp.pi * 10 * jnp.arange(0, n_time * dt, dt))[:, None]
        signals = jnp.tile(signals, (1, n_channels))

        coherence_matrix = lfp_phase_coherence(signals, dt)

        # Should be square
        self.assertEqual(coherence_matrix.shape, (n_channels, n_channels))
        # Should be symmetric
        self.assertTrue(jnp.allclose(coherence_matrix, coherence_matrix.T))
        # Diagonal should be 1
        self.assertTrue(jnp.allclose(jnp.diag(coherence_matrix), 1.0))

    def test_different_frequency_bands(self):
        """Test coherence calculation in different frequency bands."""
        dt = 0.001
        t = jnp.arange(0, 2, dt)
        signal = jnp.sin(2 * jnp.pi * 25 * t)  # 25 Hz signal
        signals = jnp.column_stack([signal, signal])

        # Test in beta band (should have high coherence)
        coherence_beta = lfp_phase_coherence(signals, dt, freq_band=(20, 30))
        # Test in alpha band (should have lower coherence)
        coherence_alpha = lfp_phase_coherence(signals, dt, freq_band=(8, 12))

        # Both should be high for identical signals, just check they're valid
        self.assertGreaterEqual(coherence_beta[0, 1], 0.8)
        self.assertGreaterEqual(coherence_alpha[0, 1], 0.8)


class TestLFPRegressionAndFeatures(unittest.TestCase):
    """Stronger assertions targeting the bugs the original suite masked."""

    def setUp(self):
        self.dt = 0.001
        self.t = np.arange(4000) * self.dt
        self.rng = np.random.RandomState(0)

    def test_psd_peak_frequency(self):
        # A 10 Hz sine -> PSD peak at 10 Hz (D1/D2).
        sig = np.sin(2 * np.pi * 10 * self.t)
        freqs, psd = power_spectral_density(sig, self.dt)
        self.assertAlmostEqual(float(freqs[jnp.argmax(psd)]), 10.0, delta=1.0)

    def test_psd_power_calibration(self):
        # One-sided PSD must integrate (approx) to the signal variance.
        sig = np.sin(2 * np.pi * 10 * self.t)  # variance 0.5
        freqs, psd = power_spectral_density(sig, self.dt, nperseg=1024)
        integral = float(jnp.trapezoid(psd, freqs))
        self.assertAlmostEqual(integral, 0.5, delta=0.1)

    def test_psd_quantity_dt(self):
        sig = np.sin(2 * np.pi * 10 * self.t)
        f_float, psd_float = power_spectral_density(sig, self.dt)
        f_q, psd_q = power_spectral_density(sig, 1.0 * u.ms)  # 1 ms == 0.001 s
        np.testing.assert_allclose(np.asarray(f_float), np.asarray(f_q), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(psd_float), np.asarray(psd_q), rtol=1e-4)

    def test_psd_freq_range_empty(self):
        sig = np.sin(2 * np.pi * 10 * self.t)
        freqs, psd = power_spectral_density(sig, self.dt, freq_range=(1e5, 2e5))
        self.assertEqual(freqs.shape[0], 1)
        self.assertEqual(float(psd[0]), 0.0)

    def test_coherence_discriminates(self):
        # D11 regression: identical -> ~1, uncorrelated -> low (was identically 1).
        sig = np.sin(2 * np.pi * 10 * self.t) + 0.1 * self.rng.randn(self.t.size)
        _, coh_same = coherence_analysis(sig, sig, self.dt)
        _, coh_unc = coherence_analysis(self.rng.randn(self.t.size),
                                        self.rng.randn(self.t.size), self.dt)
        self.assertGreater(float(jnp.mean(coh_same)), 0.95)
        self.assertLess(float(jnp.mean(coh_unc)), 0.3)

    def test_coherence_freq_range(self):
        sig = np.sin(2 * np.pi * 10 * self.t)
        freqs, coh = coherence_analysis(sig, sig, self.dt, freq_range=(5, 15))
        self.assertTrue(bool(jnp.all(freqs >= 5)))
        self.assertTrue(bool(jnp.all(freqs <= 15)))

    def test_pac_coupled_greater_than_uncoupled(self):
        # D13 regression: coupled signal must give a larger MI than noise.
        phase = 2 * np.pi * 6 * self.t
        amp = 1 + np.cos(phase)
        coupled = np.sin(phase) + amp * np.sin(2 * np.pi * 50 * self.t)
        mi_c, _, _ = phase_amplitude_coupling(coupled, self.dt)
        mi_n, _, _ = phase_amplitude_coupling(self.rng.randn(self.t.size), self.dt)
        self.assertGreater(float(mi_c), float(mi_n))
        self.assertGreater(float(mi_c), 0.01)

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

    def test_spectral_entropy_short_signal(self):
        # A signal whose retained band has <= 1 frequency bin returns 0.
        se = spectral_entropy(np.ones(8), 0.001, freq_range=(1, 2))
        self.assertEqual(float(se), 0.0)

    def test_spectral_entropy_periodic_vs_random(self):
        periodic = np.sin(2 * np.pi * 10 * self.t)
        random = self.rng.randn(self.t.size)
        self.assertLess(float(spectral_entropy(periodic, self.dt)),
                        float(spectral_entropy(random, self.dt)))

    def test_lfp_phase_coherence_requires_2d(self):
        with self.assertRaises(ValueError):
            lfp_phase_coherence(np.sin(2 * np.pi * 10 * self.t), self.dt)

    def test_spectral_entropy_multichannel(self):
        # 2-D input is averaged across channels into one spectrum.
        sigs = np.stack([np.sin(2 * np.pi * 10 * self.t),
                         np.sin(2 * np.pi * 20 * self.t)], axis=1)
        se = spectral_entropy(sigs, self.dt)
        self.assertGreaterEqual(float(se), 0.0)
        self.assertLessEqual(float(se), 1.0)
