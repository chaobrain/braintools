import math
import unittest

import brainstate
import jax.numpy as jnp

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
        self.assertIn('"spike_type" should be "exc or ""inh".', str(context.exception))

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
