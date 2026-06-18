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

"""Regression tests for the 2026-06-19 ``braintools.metric`` audit.

Each test reproduces a specific finding from
``docs/braintools-metric-issues-found-20260619.md`` (IDs referenced in the test
names) and pins the corrected behaviour.
"""

import unittest

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braintools.metric as m
from braintools.metric._lfp import _analytic_band
from braintools.metric._firings import victor_purpura_distance


class TestFIR1PhaseLockingQuantityDt(unittest.TestCase):
    def _spikes(self):
        rng = np.random.RandomState(0)
        return jnp.asarray((rng.rand(1000, 5) < 0.1).astype(float))

    def test_quantity_dt_does_not_crash(self):
        # FIR-1: a brainunit.Quantity dt previously raised TypeError in jnp.exp.
        spikes = self._spikes()
        plv = m.phase_locking_value(spikes, 10.0, dt=0.1 * u.ms)
        self.assertEqual(plv.shape, (5,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(plv))))

    def test_quantity_dt_matches_equivalent_seconds(self):
        # FIR-1: a Quantity dt must be interpreted in seconds (reference_freq is Hz),
        # so 0.1 ms must equal the float value 1e-4 s.
        spikes = self._spikes()
        plv_q = m.phase_locking_value(spikes, 10.0, dt=0.1 * u.ms)
        plv_f = m.phase_locking_value(spikes, 10.0, dt=1e-4)
        np.testing.assert_allclose(np.asarray(plv_q), np.asarray(plv_f), rtol=1e-5, atol=1e-6)


class TestCOR2CrossCorrelationQuantity(unittest.TestCase):
    def test_quantity_input_does_not_crash(self):
        # COR-2: a Quantity spike matrix previously raised TypeError in jnp.sum.
        rng = np.random.RandomState(1)
        sp = jnp.asarray((rng.rand(50, 3) < 0.2).astype(float))
        out_q = m.cross_correlation(sp * u.UNITLESS, 5.0, dt=1.0)
        out_plain = m.cross_correlation(sp, 5.0, dt=1.0)
        np.testing.assert_allclose(float(out_q), float(out_plain), rtol=1e-6)


class TestPW1PairwiseCosineSmallVectors(unittest.TestCase):
    def test_small_identical_rows_have_unit_similarity(self):
        # PW-1: flooring the *product* of norms corrupted small (non-zero) vectors.
        X = jnp.array([[3e-5, 0.0], [3e-5, 0.0]])
        sim = m.pairwise_cosine_similarity(X)
        np.testing.assert_allclose(float(sim[0, 0]), 1.0, atol=1e-5)
        np.testing.assert_allclose(float(sim[0, 1]), 1.0, atol=1e-5)

    def test_orthogonal_small_rows(self):
        X = jnp.array([[1e-5, 0.0], [0.0, 1e-5]])
        sim = m.pairwise_cosine_similarity(X)
        np.testing.assert_allclose(float(sim[0, 1]), 0.0, atol=1e-5)

    def test_zero_vector_still_yields_zero(self):
        # The documented zero-vector behaviour must be preserved.
        X = jnp.array([[0.0, 0.0], [1.0, 2.0]])
        sim = m.pairwise_cosine_similarity(X)
        np.testing.assert_allclose(float(sim[0, 1]), 0.0, atol=1e-6)

    def test_zero_vector_gradient_is_finite(self):
        def f(x):
            return jnp.sum(m.pairwise_cosine_similarity(x))

        g = jax.grad(f)(jnp.array([[0.0, 0.0], [1.0, 2.0]]))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))


class TestREG6L1LossScalar(unittest.TestCase):
    def test_scalar_input_does_not_crash(self):
        # REG-6: 0-d inputs previously raised IndexError via logits.shape[0].
        out = m.l1_loss(jnp.array(1.0), jnp.array(1.5))
        np.testing.assert_allclose(float(out), 0.5, atol=1e-6)

    def test_one_d_input_unchanged(self):
        out = m.l1_loss(jnp.array([1.0, 3.0]), jnp.array([1.5, 2.0]))
        # per-sample MAE of [|1-1.5|, |3-2|] = [0.5, 1.0]; mean = 0.75
        np.testing.assert_allclose(float(out), 0.75, atol=1e-6)


class TestINIT1L1LossExport(unittest.TestCase):
    def test_l1loss_importable_from_metric(self):
        # INIT-1: L1Loss is a documented public class that was missing from
        # braintools.metric.__all__ / __init__, so the import raised ImportError.
        from braintools.metric import L1Loss  # noqa: F401
        self.assertTrue(hasattr(m, 'L1Loss'))
        self.assertIn('L1Loss', m.__all__)

    def test_l1loss_update_matches_l1_loss(self):
        from braintools.metric import L1Loss
        logits = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        targets = jnp.array([[1.5, 2.5], [2.0, 4.0]])
        for reduction in ('mean', 'sum', 'none'):
            crit = L1Loss(reduction)
            np.testing.assert_allclose(
                np.asarray(crit.update(logits, targets)),
                np.asarray(m.l1_loss(logits, targets, reduction=reduction)),
                atol=1e-6,
            )


class TestSM1SmoothLabelsValidation(unittest.TestCase):
    def test_integer_labels_raise_type_error(self):
        # SM-1: validation must use a real exception (survives `python -O`).
        with self.assertRaises(TypeError):
            m.smooth_labels(jnp.array([[1, 0, 0], [0, 1, 0]]), 0.1)


class TestLFP1PhaseCoherenceEmptyBand(unittest.TestCase):
    def test_empty_band_off_diagonal_is_zero(self):
        # LFP-1: an empty band must not report spurious phase coherence.
        t = np.arange(0, 2, 0.001)
        sig = np.column_stack([np.sin(2 * np.pi * 10 * t), np.cos(2 * np.pi * 30 * t)])
        coh = m.lfp_phase_coherence(jnp.asarray(sig), 0.001, freq_band=(200, 300))
        np.testing.assert_allclose(float(coh[0, 1]), 0.0, atol=1e-3)
        # Diagonal is 1 by definition.
        np.testing.assert_allclose(np.asarray(jnp.diag(coh)), 1.0, atol=1e-6)

    def test_in_band_signals_still_coherent(self):
        # A real in-band signal must keep its (high) coherence.
        t = np.arange(0, 2, 0.001)
        sig = np.sin(2 * np.pi * 10 * t)
        sigs = jnp.asarray(np.column_stack([sig, sig, sig]))
        coh = m.lfp_phase_coherence(sigs, 0.001, freq_band=(8, 12))
        self.assertTrue(bool(jnp.all(coh >= 0.8)))


class TestLFP2CoherenceEmptyFreqRange(unittest.TestCase):
    def test_empty_freq_range_returns_single_bin(self):
        # LFP-2: mirror power_spectral_density's empty-range fallback.
        rng = np.random.RandomState(2)
        sig = jnp.asarray(rng.randn(2000))
        freqs, coh = m.coherence_analysis(sig, sig, 0.001, freq_range=(1e5, 2e5))
        self.assertEqual(freqs.shape[0], 1)
        self.assertEqual(coh.shape[0], 1)


class TestLFP3CoherenceScaleInvariance(unittest.TestCase):
    def test_downscaled_identical_signals_remain_coherent(self):
        # LFP-3: an absolute 1e-15 floor biased down-scaled signals toward 0.
        rng = np.random.RandomState(3)
        base = rng.randn(4000)
        big = jnp.asarray(base)
        small = jnp.asarray(base * 1e-6)
        _, coh_big = m.coherence_analysis(big, big, 0.001)
        _, coh_small = m.coherence_analysis(small, small, 0.001)
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


class TestCLS2SigmoidBCEListInputs(unittest.TestCase):
    def test_list_inputs_accepted(self):
        # CLS-2: list inputs previously raised AttributeError on .astype/.dtype.
        out = m.sigmoid_binary_cross_entropy([1.0, -1.0], [1.0, 0.0])
        ref = m.sigmoid_binary_cross_entropy(jnp.array([1.0, -1.0]), jnp.array([1.0, 0.0]))
        np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-6)


class TestFIR3BurstSynchronyEndpointTouch(unittest.TestCase):
    def test_endpoint_touching_bursts_count_as_synchronous(self):
        # FIR-3: bursts that touch at a single instant are the strongest synchrony
        # and must not be excluded by a strict `> 0` overlap test.
        spikes = np.zeros((20, 2))
        spikes[0:5, 0] = 1.0   # neuron 0 burst spans t=0..4
        spikes[4:9, 1] = 1.0   # neuron 1 burst spans t=4..8 (touches at t=4)
        idx = m.burst_synchrony_index(jnp.asarray(spikes), burst_threshold=3,
                                      max_isi=2.0, dt=1.0)
        self.assertGreater(float(idx), 0.0)


class TestFIR2VictorPurpuraCorrectness(unittest.TestCase):
    def test_identical_trains_zero_distance(self):
        s = jnp.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(victor_purpura_distance(s, s, cost_factor=1.0), 0.0, places=5)

    def test_single_extra_spike_costs_one(self):
        s1 = jnp.array([1.0, 2.0])
        s2 = jnp.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(victor_purpura_distance(s1, s2, cost_factor=1.0), 1.0, places=5)

    def test_small_shift_costs_q_times_delta(self):
        s1 = jnp.array([1.0])
        s2 = jnp.array([1.1])
        # cheaper to shift (q*0.1=0.5) than delete+insert (=2)
        self.assertAlmostEqual(victor_purpura_distance(s1, s2, cost_factor=5.0), 0.5, places=5)

    def test_far_shift_prefers_delete_insert(self):
        s1 = jnp.array([1.0])
        s2 = jnp.array([100.0])
        # shifting costs q*99 = 99 >> delete+insert = 2
        self.assertAlmostEqual(victor_purpura_distance(s1, s2, cost_factor=1.0), 2.0, places=5)


class TestFIR6CorrelationIndexBounds(unittest.TestCase):
    def test_result_within_pm_one(self):
        rng = np.random.RandomState(4)
        base = (rng.rand(1000, 1) < 0.1).astype(float)
        spikes = jnp.asarray(np.concatenate([base, base, rng.rand(1000, 1) < 0.1], axis=1).astype(float))
        ci = m.correlation_index(spikes, window_size=50.0, dt=1.0)
        self.assertGreaterEqual(float(ci), -1.0)
        self.assertLessEqual(float(ci), 1.0)


class TestCOR4CrossCorrelationZeroBranch(unittest.TestCase):
    def test_all_silent_pair_returns_zero(self):
        # Exercises the lax.cond zero branch (COR-4); must be a clean 0.0.
        spikes = jnp.zeros((100, 3))
        out = m.cross_correlation(spikes, bin=10.0, dt=1.0, method='loop')
        np.testing.assert_allclose(float(out), 0.0, atol=1e-7)
        out_v = m.cross_correlation(spikes, bin=10.0, dt=1.0, method='vmap')
        np.testing.assert_allclose(float(out_v), 0.0, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
