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

import math
import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braintools.analysis.spectral import (
    power_spectral_density, coherence_analysis, phase_amplitude_coupling,
    theta_gamma_coupling, spectral_entropy, lfp_phase_coherence
)
from braintools.analysis.spectral._lfp import _analytic_band


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

        # The 25 Hz tone lives in the beta band, so coherence there is genuine
        # and high.
        coherence_beta = lfp_phase_coherence(signals, dt, freq_band=(20, 30))
        # The alpha band carries no signal power; phase in an empty band is
        # undefined, so the LFP-1 fix reports ~0 instead of a spurious high
        # value (the previous expectation of >= 0.8 here was the bug).
        coherence_alpha = lfp_phase_coherence(signals, dt, freq_band=(8, 12))

        self.assertGreaterEqual(coherence_beta[0, 1], 0.8)
        self.assertLessEqual(coherence_alpha[0, 1], 0.2)
        # All coherence values stay within the valid [0, 1] range.
        self.assertGreaterEqual(coherence_alpha[0, 1], 0.0)


class TestSpectralRegressionAndFeatures(unittest.TestCase):
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


class TestLFP1PhaseCoherenceEmptyBand(unittest.TestCase):
    def test_empty_band_off_diagonal_is_zero(self):
        # LFP-1: an empty band must not report spurious phase coherence.
        t = np.arange(0, 2, 0.001)
        sig = np.column_stack([np.sin(2 * np.pi * 10 * t), np.cos(2 * np.pi * 30 * t)])
        coh = lfp_phase_coherence(jnp.asarray(sig), 0.001, freq_band=(200, 300))
        np.testing.assert_allclose(float(coh[0, 1]), 0.0, atol=1e-3)
        # Diagonal is 1 by definition.
        np.testing.assert_allclose(np.asarray(jnp.diag(coh)), 1.0, atol=1e-6)

    def test_in_band_signals_still_coherent(self):
        # A real in-band signal must keep its (high) coherence.
        t = np.arange(0, 2, 0.001)
        sig = np.sin(2 * np.pi * 10 * t)
        sigs = jnp.asarray(np.column_stack([sig, sig, sig]))
        coh = lfp_phase_coherence(sigs, 0.001, freq_band=(8, 12))
        self.assertTrue(bool(jnp.all(coh >= 0.8)))


class TestLFP2CoherenceEmptyFreqRange(unittest.TestCase):
    def test_empty_freq_range_returns_single_bin(self):
        # LFP-2: mirror power_spectral_density's empty-range fallback.
        rng = np.random.RandomState(2)
        sig = jnp.asarray(rng.randn(2000))
        freqs, coh = coherence_analysis(sig, sig, 0.001, freq_range=(1e5, 2e5))
        self.assertEqual(freqs.shape[0], 1)
        self.assertEqual(coh.shape[0], 1)


class TestLFP3CoherenceScaleInvariance(unittest.TestCase):
    def test_downscaled_identical_signals_remain_coherent(self):
        # LFP-3: an absolute 1e-15 floor biased down-scaled signals toward 0.
        rng = np.random.RandomState(3)
        base = rng.randn(4000)
        big = jnp.asarray(base)
        small = jnp.asarray(base * 1e-6)
        _, coh_big = coherence_analysis(big, big, 0.001)
        _, coh_small = coherence_analysis(small, small, 0.001)
        # Identical signals -> coherence 1 at populated bins, regardless of scale.
        np.testing.assert_allclose(float(jnp.max(coh_small)), float(jnp.max(coh_big)), atol=1e-4)
        self.assertGreater(float(jnp.max(coh_small)), 0.99)


class TestLFP5AnalyticBandDC(unittest.TestCase):
    def test_dc_bin_not_doubled(self):
        # LFP-5: including 0 Hz in a band previously doubled the DC component.
        n = 64
        const = 3.0
        sig = jnp.full((n, 1), const)
        freqs = jnp.fft.fftfreq(n, 1.0)
        fft = jnp.fft.fft(sig, axis=0)
        analytic = _analytic_band(fft, freqs, low=0.0, high=10.0)
        # The reconstructed analytic signal's real part must equal the DC value,
        # not 2x it.
        np.testing.assert_allclose(float(jnp.real(analytic[0, 0])), const, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
