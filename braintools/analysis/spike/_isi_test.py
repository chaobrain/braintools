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

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.analysis.spike._isi import (
    _static_time, _time_value, cv, cv2, fano_factor, isi, lv,
)
from braintools.spike import SpikeTrains


@pytest.fixture
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', prev)


# Six trains covering 5, 3, 0, 0, 1 and 2 intervals.
TRAINS = [[1., 3., 4., 8., 9.5, 15.], [2., 5., 11., 12.], [7.], [], [0.5, 2.0], [1., 2., 4.]]

# Golden values from Elephant 1.2.1 (/tmp/bt/refenv), per train ``t``:
#   I = elephant.statistics.isi(np.array(t))
#   elephant.statistics.cv(I)                 (= scipy.stats.variation, ddof=0)
#   elephant.statistics.cv2(I, with_nan=True)
#   elephant.statistics.lv(I, with_nan=True)
# Convention difference: Elephant's cv of a single interval is 0.0
# (scipy.stats.variation); braintools returns nan for fewer than 2 intervals.
# Elephant cannot take an empty interval vector; those rows are nan here.
NAN = np.nan
GOLD_ISI = [[2.0, 1.0, 4.0, 1.5, 5.5], [3.0, 6.0, 1.0], [], [], [1.5], [1.0, 2.0]]
GOLD_CV = [0.6039833759031344, 0.6164414002968976, NAN, NAN, NAN, 0.3333333333333333]
GOLD_CV2 = [0.9796536796536797, 1.0476190476190477, NAN, NAN, NAN, 0.6666666666666666]
GOLD_LV = [0.7531899702029571, 0.9319727891156462, NAN, NAN, NAN, 0.3333333333333333]


def _trains(unit=None, **kw):
    trains = [np.asarray(t) for t in TRAINS]
    if unit is not None:
        trains = [t * unit for t in trains]
    t0, t1 = (0., 20.) if unit is None else (0. * unit, 20. * unit)
    return SpikeTrains.from_list(trains, t_start=t0, t_stop=t1, **kw)


def _padded(rows, width):
    out = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


class TestReferenceValues:

    def test_isi_matches_elephant(self, x64):
        got = isi(_trains())
        assert got.shape == (6, 5)
        np.testing.assert_allclose(np.asarray(got), _padded(GOLD_ISI, 5), rtol=1e-12)

    def test_cv_matches_elephant(self, x64):
        np.testing.assert_allclose(np.asarray(cv(_trains())), GOLD_CV, rtol=1e-12)

    def test_cv2_matches_elephant(self, x64):
        np.testing.assert_allclose(np.asarray(cv2(_trains())), GOLD_CV2, rtol=1e-12)

    def test_lv_matches_elephant(self, x64):
        np.testing.assert_allclose(np.asarray(lv(_trains())), GOLD_LV, rtol=1e-12)

    def test_cv_matches_scipy_variation(self, x64):
        stats = pytest.importorskip('scipy.stats')
        st = _trains()
        for row, t in zip(np.asarray(cv(st)), TRAINS):
            if len(t) >= 3:
                np.testing.assert_allclose(row, stats.variation(np.diff(t)), rtol=1e-12)

    def test_float32_close_to_float64(self):
        st = _trains()
        assert st.times.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(cv(st)), GOLD_CV, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(cv2(st)), GOLD_CV2, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(lv(st)), GOLD_LV, rtol=1e-5)


class TestAnalyticCases:

    def test_regular_train_is_zero(self):
        st = SpikeTrains.from_dense(jnp.zeros((50, 2)).at[::5, :].set(1), dt=1.)
        for f in (cv, cv2, lv):
            np.testing.assert_allclose(np.asarray(f(st)), 0., atol=1e-6)

    def test_poisson_is_about_one(self, x64):
        key = jax.random.PRNGKey(0)
        intervals = jax.random.exponential(key, (4, 4000))
        times = jnp.cumsum(intervals, axis=1)
        st = SpikeTrains(times, jnp.full(4, 4000), 0., float(times.max()))
        for f in (cv, cv2, lv):
            np.testing.assert_allclose(np.asarray(f(st)), 1., atol=0.06)

    def test_repeated_spike_times(self):
        # three identical times: intervals (0, 0) -> cv 0/0, cv2/lv 0/0 -> nan
        st = SpikeTrains(jnp.array([[2., 2., 2.], [1., 2., 2.]]), jnp.array([3, 3]), 0., 3.)
        assert np.isnan(np.asarray(cv(st))[0])
        assert np.isnan(np.asarray(cv2(st))[0])
        assert np.isnan(np.asarray(lv(st))[0])
        # intervals (1, 0): one zero interval is fine
        np.testing.assert_allclose(np.asarray(cv2(st))[1], 2.)
        np.testing.assert_allclose(np.asarray(lv(st))[1], 3.)
        np.testing.assert_allclose(np.asarray(cv(st))[1], 1.)


class TestEdgeCases:

    def test_capacity_one(self):
        st = SpikeTrains.from_dense(jnp.eye(3), dt=1.)
        assert st.capacity == 1
        assert isi(st).shape == (3, 0)
        for f in (cv, cv2, lv):
            out = f(st)
            assert out.shape == (3,)
            assert np.all(np.isnan(np.asarray(out)))

    def test_all_silent(self):
        st = SpikeTrains.from_dense(jnp.zeros((10, 4)), dt=1.)
        assert np.all(np.isnan(np.asarray(isi(st))))
        for f in (cv, cv2, lv):
            assert np.all(np.isnan(np.asarray(f(st))))

    def test_zero_one_two_spikes(self):
        st = SpikeTrains.from_list([[], [1.], [1., 3.]], t_start=0., t_stop=5.)
        np.testing.assert_allclose(np.asarray(isi(st)), [[NAN], [NAN], [2.]])
        for f in (cv, cv2, lv):
            assert np.all(np.isnan(np.asarray(f(st))))

    def test_rejects_non_spike_trains(self):
        for f in (isi, cv, cv2, lv):
            with pytest.raises(TypeError, match='SpikeTrains'):
                f(jnp.ones((3, 4)))


class TestUnits:

    def test_isi_keeps_unit(self):
        st = _trains(u.ms)
        got = isi(st)
        assert isinstance(got, u.Quantity)
        assert got.unit == u.ms
        np.testing.assert_allclose(np.asarray(got.mantissa), _padded(GOLD_ISI, 5), rtol=1e-6)

    def test_plain_and_quantity_agree(self):
        plain, quant = _trains(), _trains(u.second)
        for f in (cv, cv2, lv):
            a, b = f(plain), f(quant)
            assert not isinstance(b, u.Quantity)
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)


class TestTransforms:

    def test_jit(self):
        st = _trains()
        for f in (isi, cv, cv2, lv):
            np.testing.assert_allclose(np.asarray(jax.jit(f)(st)), np.asarray(f(st)), rtol=1e-6)

    def test_vmap_over_trials(self):
        key = jax.random.PRNGKey(1)
        rasters = jax.random.bernoulli(key, 0.2, (5, 60, 3))
        batched = jax.vmap(lambda r: SpikeTrains.from_dense(r, dt=1., capacity=60))(rasters)
        for f in (isi, cv, cv2, lv):
            out = jax.vmap(f)(batched)
            for trial in range(5):
                single = SpikeTrains.from_dense(rasters[trial], dt=1., capacity=60)
                np.testing.assert_allclose(np.asarray(out[trial]), np.asarray(f(single)), rtol=1e-6)

    def test_grad_matches_finite_difference(self, x64):
        times = jnp.array([[1., 3., 4., 8., 9.5, 15.], [2., 5., 11., 12., jnp.inf, jnp.inf]])
        counts = jnp.array([6, 4])

        for f in (cv, cv2, lv):
            def loss(t):
                return jnp.sum(f(SpikeTrains(t, counts, 0., 20.)))

            g = np.asarray(jax.grad(loss)(times))
            assert np.all(np.isfinite(g))
            np.testing.assert_array_equal(g[1, 4:], 0.)  # padding gets no gradient
            h = 1e-6
            for (i, j) in [(0, 2), (1, 1), (0, 5)]:
                tp = times.at[i, j].add(h)
                tm = times.at[i, j].add(-h)
                fd = (loss(tp) - loss(tm)) / (2 * h)
                np.testing.assert_allclose(g[i, j], float(fd), rtol=1e-5, atol=1e-8)

    def test_grad_isi(self):
        times = jnp.array([[1., 3., 4.]])
        g = jax.grad(lambda t: jnp.nansum(isi(SpikeTrains(t, jnp.array([2]), 0., 5.))))(times)
        np.testing.assert_allclose(np.asarray(g), [[-1., 1., 0.]])


class TestFanoFactor:

    def test_counts_match_elephant(self, x64):
        # Elephant 1.2.1: elephant.statistics.fanofactor([np.zeros(c) for c in
        #   [3, 0, 5, 2, 2, 4]]) == 0.9583333333333335 (counts.var() / counts.mean())
        got = fano_factor(jnp.array([3, 0, 5, 2, 2, 4]))
        np.testing.assert_allclose(float(got), 0.9583333333333335, rtol=1e-12)

    def test_axis_and_zero_mean(self):
        x = np.array([[3., 0., 1.], [5., 0., 1.], [4., 0., 4.]])
        got = np.asarray(fano_factor(x, axis=0))
        with np.errstate(invalid='ignore'):
            ref = x.var(axis=0) / x.mean(axis=0)
        np.testing.assert_allclose(got[[0, 2]], ref[[0, 2]], rtol=1e-6)
        assert np.isnan(got[1])
        np.testing.assert_allclose(np.asarray(fano_factor(x, axis=1)),
                                   x.var(axis=1) / x.mean(axis=1), rtol=1e-6)

    def test_spike_trains_matches_numpy_binning(self):
        key = jax.random.PRNGKey(3)
        raster = jax.random.bernoulli(key, 0.3, (100, 4))
        st = SpikeTrains.from_dense(raster, dt=1.)
        got = np.asarray(fano_factor(st, bin_size=5.))
        counts = np.asarray(raster).reshape(20, 5, 4).sum(axis=1)
        np.testing.assert_allclose(got, counts.var(axis=0) / counts.mean(axis=0), rtol=1e-5)

    def test_spike_trains_silent_neuron_is_nan(self):
        st = SpikeTrains.from_dense(jnp.zeros((10, 2)).at[::2, 0].set(1), dt=1.)
        got = np.asarray(fano_factor(st, bin_size=2.))
        np.testing.assert_allclose(got[0], 0.)
        assert np.isnan(got[1])

    def test_poisson_about_one(self):
        raster = jax.random.bernoulli(jax.random.PRNGKey(4), 0.05, (20000, 3))
        st = SpikeTrains.from_dense(raster, dt=1.)
        np.testing.assert_allclose(np.asarray(fano_factor(st, bin_size=50.)), 1., atol=0.15)

    def test_units_plain_ms_quantity(self):
        raster = jax.random.bernoulli(jax.random.PRNGKey(5), 0.3, (40, 2))
        st_ms = SpikeTrains.from_dense(raster, dt=1. * u.ms)
        st_plain = SpikeTrains.from_dense(raster, dt=1.)
        ref = np.asarray(fano_factor(st_plain, bin_size=4.))
        np.testing.assert_allclose(np.asarray(fano_factor(st_ms, bin_size=4. * u.ms)), ref)
        # a plain number is milliseconds, also for unit-carrying trains
        np.testing.assert_allclose(np.asarray(fano_factor(st_ms, bin_size=4.)), ref)
        # a Quantity is converted for unitless (millisecond) trains
        np.testing.assert_allclose(np.asarray(fano_factor(st_plain, bin_size=0.004 * u.second)), ref,
                                   rtol=1e-6)
        st_s = SpikeTrains.from_dense(raster, dt=0.001 * u.second)
        np.testing.assert_allclose(np.asarray(fano_factor(st_s, bin_size=4.)), ref, rtol=1e-5)

    def test_errors(self):
        st = SpikeTrains.from_dense(jnp.ones((4, 1)), dt=1.)
        with pytest.raises(ValueError, match='required'):
            fano_factor(st)
        with pytest.raises(ValueError, match='positive'):
            fano_factor(st, bin_size=0.)
        with pytest.raises(ValueError, match='only applies'):
            fano_factor(jnp.ones(3), bin_size=1.)
        with pytest.raises(TypeError, match='unitless'):
            fano_factor(jnp.ones(3) * u.mV)
        # a unitless Quantity is accepted as counts
        assert np.isclose(float(fano_factor(u.Quantity(jnp.array([1., 3.])))), 0.5)

    def test_integer_counts(self):
        np.testing.assert_allclose(float(fano_factor(np.array([1, 3]))), 0.5)

    def test_jit_and_vmap(self):
        rasters = jax.random.bernoulli(jax.random.PRNGKey(6), 0.3, (3, 40, 2))
        batched = jax.vmap(lambda r: SpikeTrains.from_dense(r, dt=1., capacity=40))(rasters)
        f = jax.jit(lambda s: fano_factor(s, bin_size=4., n_bins=10))
        out = jax.vmap(f)(batched)
        assert out.shape == (3, 2)
        for trial in range(3):
            single = SpikeTrains.from_dense(rasters[trial], dt=1.)
            np.testing.assert_allclose(np.asarray(out[trial]),
                                       np.asarray(fano_factor(single, bin_size=4.)), rtol=1e-6)
        x = jax.random.poisson(jax.random.PRNGKey(7), 3., (4, 10, 2)).astype(float)
        np.testing.assert_allclose(np.asarray(jax.vmap(jax.jit(fano_factor))(x)),
                                   np.asarray(x).var(axis=1) / np.asarray(x).mean(axis=1), rtol=1e-5)

    def test_grad_counts(self, x64):
        x = jnp.array([1., 2., 4.])
        g = jax.grad(lambda c: fano_factor(c))(x)
        # d/dx_i (var/mean) = (2 (x_i - m) / n) / m - var / (n m^2)
        m, v, n = 7. / 3., np.var([1., 2., 4.]), 3
        np.testing.assert_allclose(np.asarray(g), (2 * (np.asarray(x) - m) / n) / m - v / (n * m * m),
                                   rtol=1e-10)


class TestHelpers:

    def test_time_value(self):
        assert _time_value(2. * u.ms, None, 'x') == 2.
        np.testing.assert_allclose(_time_value(2. * u.second, None, 'x'), 2000.)
        np.testing.assert_allclose(_time_value(2., u.second, 'x'), 0.002)
        assert _time_value(2., u.ms, 'x') == 2.
        assert _time_value(u.Quantity(2.), None, 'x') == 2.
        with pytest.raises(TypeError, match='time'):
            _time_value(2. * u.mV, None, 'x')

    def test_static_time(self):
        with pytest.raises(ValueError, match='scalar'):
            _static_time(np.ones(2), None, 'x')
        with pytest.raises(ValueError, match='finite'):
            _static_time(np.inf, None, 'x')
        with pytest.raises(TypeError, match='static'):
            jax.jit(lambda v: _static_time(v, None, 'x'))(1.)


class TestFanoWholeBins:
    """Regression: the partial last bin of ``SpikeTrains.bin`` biased the Fano factor."""

    def test_regular_train_drops_partial_bin(self):
        # one spike every 10 ms over 1000 ms; 300 ms bins -> whole bins [30, 30, 30]
        # (the partial bin [900, 1000) with 10 spikes used to give F = 3.0)
        st = SpikeTrains.from_dense(jnp.zeros((1000, 1)).at[::10, 0].set(1), dt=1.)
        np.testing.assert_allclose(np.asarray(fano_factor(st, bin_size=300.)), [0.])

    def test_explicit_n_bins_keeps_partial_bin(self):
        st = SpikeTrains.from_dense(jnp.zeros((1000, 1)).at[::10, 0].set(1), dt=1.)
        # counts [30, 30, 30, 10]: mean 25, var 75 -> 3
        np.testing.assert_allclose(np.asarray(fano_factor(st, bin_size=300., n_bins=4)), [3.])

    def test_exact_multiple_keeps_last_bin(self):
        # duration 100 * 0.1 = 10 (float32 round-off) and bin 1.0: 10 whole bins, not 9
        raster = jnp.zeros((100, 1)).at[jnp.array([5, 95]), 0].set(1)
        st = SpikeTrains.from_dense(raster, dt=0.1)
        counts = np.zeros(10)
        counts[[0, 9]] = 1
        np.testing.assert_allclose(np.asarray(fano_factor(st, bin_size=1.)),
                                   [counts.var() / counts.mean()], rtol=1e-6)

    def test_poisson_long_bins_about_one(self):
        raster = jax.random.bernoulli(jax.random.PRNGKey(8), 0.02, (100000, 1))
        st = SpikeTrains.from_dense(raster, dt=1.)
        # 33 whole bins of 3 s; the partial last second is dropped
        np.testing.assert_allclose(np.asarray(fano_factor(st, bin_size=3000.)), [1.], atol=0.5)

    def test_bin_longer_than_recording_is_nan(self):
        st = SpikeTrains.from_dense(jnp.ones((10, 1)), dt=1.)
        assert np.isnan(np.asarray(fano_factor(st, bin_size=20.))[0])

    def test_traced_bounds_need_n_bins(self):
        st = SpikeTrains.from_dense(jnp.ones((10, 1)), dt=1.)
        with pytest.raises(ValueError, match='n_bins'):
            jax.jit(lambda s: fano_factor(s, bin_size=2.))(st)
