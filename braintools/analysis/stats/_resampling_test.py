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

import itertools

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from braintools.analysis.stats._resampling import (
    bootstrap_ci,
    jitter_spikes,
    phase_randomize,
    shuffle_isi,
)
from braintools.spike import SpikeTrains


@pytest.fixture
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', prev)


def _valid(st, strict_stop=True):
    """Layout invariants of a SpikeTrains, plus spikes strictly before t_stop."""
    st.check()
    times = np.asarray(st._times_m)
    mask = np.asarray(st.mask)
    assert np.all(np.isposinf(times[~mask]))
    if strict_stop:
        assert np.all(times[mask] < float(st._t_stop_m))


def _trains(unit=None):
    rng = np.random.default_rng(0)
    trains = [np.sort(rng.uniform(0., 100., size=n)) for n in (0, 1, 2, 7, 15)]
    if unit is not None:
        trains = [t * unit for t in trains]
        return SpikeTrains.from_list(trains, t_start=0. * unit, t_stop=100. * unit, capacity=20)
    return SpikeTrains.from_list(trains, t_start=0., t_stop=100., capacity=20)


# ----------------------------------------------------------------------------
# jitter_spikes
# ----------------------------------------------------------------------------

class TestJitterSpikes:
    def test_layout_preserved(self):
        st = _trains()
        for method in ('uniform', 'gaussian'):
            out = jitter_spikes(jax.random.key(0), st, 5., method=method)
            assert isinstance(out, SpikeTrains)
            assert out.times.shape == st.times.shape
            np.testing.assert_array_equal(out.counts, st.counts)
            assert float(out.t_start) == 0. and float(out.t_stop) == 100.
            _valid(out)
            assert not np.allclose(np.asarray(out.times)[out.mask], np.asarray(st.times)[st.mask])

    def test_uniform_displacement_bounded(self):
        # Well-separated spikes far from the bounds keep their order.
        times = jnp.array([[10., 20., 30., 40.]])
        st = SpikeTrains(times, jnp.array([4]), 0., 50.)
        out = jitter_spikes(jax.random.key(1), st, 3.)
        d = np.asarray(out.times - st.times)
        assert np.all(np.abs(d) <= 3.) and np.any(d != 0)

    def test_offset_distributions(self):
        st = SpikeTrains(jnp.full((1, 4000), 500.), jnp.array([4000]), 0., 1000.)
        offsets_u = np.asarray(jitter_spikes(jax.random.key(2), st, 10.).times[0]) - 500.
        assert abs(offsets_u.mean()) < 0.5
        np.testing.assert_allclose(offsets_u.std(), 10. / np.sqrt(3.), rtol=0.05)
        assert offsets_u.min() >= -10. and offsets_u.max() <= 10.
        offsets_g = np.asarray(jitter_spikes(jax.random.key(3), st, 10., 'gaussian').times[0]) - 500.
        assert abs(offsets_g.mean()) < 0.5
        np.testing.assert_allclose(offsets_g.std(), 10., rtol=0.05)

    def test_reflection_at_bounds(self):
        # All spikes at t_start, jittered by a width larger than the recording.
        times = jnp.zeros((2, 500)).at[1, 3:].set(jnp.inf)
        st = SpikeTrains(times, jnp.array([500, 3]), 0., 10.)
        for method in ('uniform', 'gaussian'):
            out = jitter_spikes(jax.random.key(4), st, 25., method)
            _valid(out)
            assert np.all(np.asarray(out.times)[np.asarray(out.mask)] >= 0.)

    def test_reflection_small_offset(self):
        # Spikes near the bounds with width 1: the reflection mirrors the offset.
        st = SpikeTrains(jnp.full((1, 1000), 0.2), jnp.array([1000]), 0., 10.)
        out = np.asarray(jitter_spikes(jax.random.key(5), st, 1.).times[0])
        assert np.all((out >= 0.) & (out <= 1.2))
        # Offsets in [-1, -0.2) reflect to (0, 0.8] and offsets in [-0.2, 0.6) land
        # in [0, 0.8), so P(out < 0.8) = P(offset < 0.6) = 0.8 (no spike is lost).
        np.testing.assert_allclose(np.mean(out < 0.8), 0.8, atol=0.05)
        st_hi = SpikeTrains(jnp.full((1, 1000), 9.9), jnp.array([1000]), 0., 10.)
        out_hi = np.asarray(jitter_spikes(jax.random.key(6), st_hi, 1.).times[0])
        assert np.all((out_hi >= 8.9) & (out_hi < 10.))

    def test_spike_at_stop_kept_inside(self):
        st = SpikeTrains(jnp.array([[0., 10.]]), jnp.array([2]), 0., 10.)
        out = jitter_spikes(jax.random.key(0), st, 0.)
        _valid(out)
        np.testing.assert_allclose(out.times[0], [0., 10.], atol=1e-5)

    def test_zero_width_identity(self):
        st = _trains()
        out = jitter_spikes(jax.random.key(7), st, 0.)
        np.testing.assert_array_equal(np.asarray(out.times), np.asarray(st.times))

    def test_zero_duration(self):
        st = SpikeTrains(jnp.array([[3., 3.]]), jnp.array([2]), 3., 3.)
        out = jitter_spikes(jax.random.key(0), st, 1.)
        np.testing.assert_array_equal(out.times, [[3., 3.]])

    def test_capacity_one_and_silent(self):
        st = SpikeTrains.from_dense(jnp.zeros((10, 3)).at[4, 1].set(1.), dt=1.)
        assert st.capacity == 1
        out = jitter_spikes(jax.random.key(0), st, 2.)
        _valid(out)
        np.testing.assert_array_equal(out.counts, [0, 1, 0])

    def test_units(self):
        st = _trains(u.ms)
        a = jitter_spikes(jax.random.key(8), st, 2. * u.ms)
        b = jitter_spikes(jax.random.key(8), st, 0.002 * u.second)
        assert a.times.unit == u.ms
        np.testing.assert_allclose(a.times.mantissa, b.times.mantissa, rtol=1e-5)
        plain = jitter_spikes(jax.random.key(8), _trains(), 2.)
        np.testing.assert_allclose(a.times.mantissa, plain.times, rtol=1e-5)
        with pytest.raises(TypeError):
            jitter_spikes(jax.random.key(8), st, 2. * u.mV)

    def test_plain_width_is_milliseconds(self):
        # Spec rule 3: a plain number is in milliseconds when the trains carry units.
        key = jax.random.key(11)
        st_ms = _trains(u.ms)
        np.testing.assert_allclose(jitter_spikes(key, st_ms, 2.).times.mantissa,
                                   jitter_spikes(key, st_ms, 2. * u.ms).times.mantissa)
        st_s = SpikeTrains.from_list([t.to(u.second) for t in st_ms.to_list()],
                                     t_start=0. * u.second, t_stop=0.1 * u.second, capacity=20)
        out = jitter_spikes(key, st_s, 2.)  # 2 ms = 0.002 s
        assert out.times.unit == u.second
        np.testing.assert_allclose(out.times.mantissa,
                                   jitter_spikes(key, st_s, 0.002 * u.second).times.mantissa)
        d = np.abs(np.asarray(out.times.mantissa - st_s.times.mantissa))[np.asarray(st_s.mask)]
        assert np.all(d <= 0.002 + 1e-7) and np.any(d > 0)
        # Unitless trains are taken to be in ms: a plain width stays as is, and a
        # Quantity width is converted to ms.
        st = _trains()
        np.testing.assert_allclose(jitter_spikes(key, st, 0.002 * u.second).times,
                                   jitter_spikes(key, st, 2.).times, rtol=1e-6)
        # traced plain width with unit trains
        g = jax.jit(lambda w: jitter_spikes(key, st_s, w).times.mantissa)
        np.testing.assert_allclose(g(2.), out.times.mantissa, rtol=1e-6)

    def test_jit_and_vmap_over_keys(self):
        st = _trains()
        f = jax.jit(lambda k, s: jitter_spikes(k, s, 4.))
        out = f(jax.random.key(9), st)
        ref = jitter_spikes(jax.random.key(9), st, 4.)
        np.testing.assert_allclose(out.times, ref.times, rtol=1e-6)
        keys = jax.random.split(jax.random.key(10), 6)
        batch = jax.vmap(lambda k: jitter_spikes(k, st, 4., 'gaussian'))(keys)
        assert batch.times.shape == (6, 5, 20)
        for i in range(6):
            one = SpikeTrains._make(batch.times[i], batch.counts[i], batch.t_start[i],
                                    batch.t_stop[i], batch.overflow[i], None)
            _valid(one)
        assert not np.allclose(batch.times[0][st.mask], batch.times[1][st.mask])
        # traced width
        g = jax.jit(lambda w: jitter_spikes(jax.random.key(0), st, w).times)
        np.testing.assert_allclose(g(0.), st.times)

    def test_raw_prng_key(self):
        out = jitter_spikes(jax.random.PRNGKey(0), _trains(), 1.)
        _valid(out)

    def test_errors(self):
        st = _trains()
        with pytest.raises(ValueError):
            jitter_spikes(jax.random.key(0), st, -1.)
        with pytest.raises(ValueError):
            jitter_spikes(jax.random.key(0), st, 1., method='poisson')
        with pytest.raises(TypeError):
            jitter_spikes(jax.random.key(0), jnp.zeros((5, 2)), 1.)


# ----------------------------------------------------------------------------
# shuffle_isi
# ----------------------------------------------------------------------------

def _host_isis(st):
    return [np.diff(np.asarray(t)) for t in st.to_list()]


class TestShuffleIsi:
    def test_isi_multiset_preserved(self):
        st = _trains()
        out = shuffle_isi(jax.random.key(0), st)
        _valid(out, strict_stop=False)
        np.testing.assert_array_equal(out.counts, st.counts)
        for a, b in zip(_host_isis(st), _host_isis(out)):
            np.testing.assert_allclose(np.sort(a), np.sort(b), rtol=1e-4, atol=1e-4)
        for a, b in zip(st.to_list(), out.to_list()):
            if a.size:
                np.testing.assert_allclose(b[0], a[0])
                np.testing.assert_allclose(b[-1], a[-1], rtol=1e-6)
        # order of the long train changes
        assert not np.allclose(_host_isis(st)[4], _host_isis(out)[4])

    def test_uniform_permutations_with_padding(self):
        # 3 distinct ISIs in a row padded to capacity 8; all 6 orders equally likely.
        st = SpikeTrains(jnp.array([[0., 1., 3., 7.] + [jnp.inf] * 4]), jnp.array([4]), 0., 10.)
        keys = jax.random.split(jax.random.key(1), 3000)
        times = jax.vmap(lambda k: shuffle_isi(k, st).times[0, :4])(keys)
        isis = np.round(np.diff(np.asarray(times), axis=1)).astype(int)
        assert np.all(np.isposinf(jax.vmap(lambda k: shuffle_isi(k, st).times[0, 4:])(keys[:5])))
        observed = {p: 0 for p in itertools.permutations((1, 2, 4))}
        for row in isis:
            observed[tuple(row)] += 1
        freqs = np.array(list(observed.values())) / len(isis)
        np.testing.assert_allclose(freqs, 1 / 6, atol=0.03)

    def test_few_spikes_unchanged(self):
        st = SpikeTrains.from_list([[], [2.], [1., 4.]], t_start=0., t_stop=5., capacity=3)
        out = shuffle_isi(jax.random.key(2), st)
        np.testing.assert_allclose(np.asarray(out.times), np.asarray(st.times))

    def test_capacity_one(self):
        st = SpikeTrains.from_dense(jnp.zeros((5, 2)).at[1, 0].set(1.), dt=1.)
        out = shuffle_isi(jax.random.key(0), st)
        np.testing.assert_array_equal(np.asarray(out.times), np.asarray(st.times))

    def test_units(self):
        st = _trains(u.ms)
        out = shuffle_isi(jax.random.key(3), st)
        assert out.times.unit == u.ms
        plain = shuffle_isi(jax.random.key(3), _trains())
        np.testing.assert_allclose(out.times.mantissa, plain.times, rtol=1e-6)

    def test_jit_and_vmap_over_keys(self):
        st = _trains()
        out = jax.jit(shuffle_isi)(jax.random.key(4), st)
        np.testing.assert_allclose(out.times, shuffle_isi(jax.random.key(4), st).times, rtol=1e-6)
        keys = jax.random.split(jax.random.key(5), 4)
        batch = jax.vmap(shuffle_isi, in_axes=(0, None))(keys, st)
        assert batch.times.shape == (4, 5, 20)
        assert not np.allclose(batch.times[0, 4], batch.times[1, 4])
        np.testing.assert_array_equal(batch.counts, np.broadcast_to(st.counts, (4, 5)))

    def test_errors(self):
        with pytest.raises(TypeError):
            shuffle_isi(jax.random.key(0), [np.array([1., 2.])])


# ----------------------------------------------------------------------------
# phase_randomize
# ----------------------------------------------------------------------------

class TestPhaseRandomize:
    @pytest.mark.parametrize('n', [64, 65])
    def test_power_spectrum_preserved(self, n, x64):
        x = jax.random.normal(jax.random.key(0), (n, 3))
        y = phase_randomize(jax.random.key(1), x)
        assert y.shape == x.shape and y.dtype == x.dtype
        fx = np.fft.rfft(np.asarray(x), axis=0)
        fy = np.fft.rfft(np.asarray(y), axis=0)
        np.testing.assert_allclose(np.abs(fy), np.abs(fx), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(fy[0], fx[0], atol=1e-9)  # DC (mean) unchanged
        if n % 2 == 0:
            np.testing.assert_allclose(fy[-1], fx[-1], atol=1e-9)  # Nyquist unchanged
            assert np.allclose(fy[-1].imag, 0., atol=1e-9)
        assert not np.allclose(y, x, atol=1e-3)
        # Different channels get different phases.
        cross = lambda f: f[:, 0] * np.conj(f[:, 1])
        assert not np.allclose(cross(fy), cross(fx), atol=1e-3)

    @pytest.mark.parametrize('n', [32, 33])
    def test_same_phases_preserves_cross_correlation(self, n, x64):
        x = jax.random.normal(jax.random.key(2), (3, n))
        y = phase_randomize(jax.random.key(3), x, axis=1, same_phases=True)
        fx = np.fft.rfft(np.asarray(x), axis=1)
        fy = np.fft.rfft(np.asarray(y), axis=1)
        np.testing.assert_allclose(fy[0] * np.conj(fy[1]), fx[0] * np.conj(fx[1]), atol=1e-9)
        circ = lambda a, b: np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))))
        np.testing.assert_allclose(circ(y[0], y[2]), circ(x[0], x[2]), atol=1e-9)
        assert not np.allclose(y, x, atol=1e-3)

    def test_short_signals(self):
        for n in (1, 2, 3):
            x = jnp.arange(1., n + 1.)
            y = phase_randomize(jax.random.key(0), x)
            np.testing.assert_allclose(np.abs(np.fft.rfft(y)), np.abs(np.fft.rfft(x)), atol=1e-5)
        np.testing.assert_allclose(phase_randomize(jax.random.key(0), jnp.array([1., 3.])), [1., 3.])

    def test_units_and_int(self):
        x = jax.random.normal(jax.random.key(4), (16,))
        q = phase_randomize(jax.random.key(5), x * u.mV)
        assert q.unit == u.mV
        np.testing.assert_allclose(q.mantissa, phase_randomize(jax.random.key(5), x), rtol=1e-6)
        yi = phase_randomize(jax.random.key(5), jnp.arange(8))
        assert jnp.issubdtype(yi.dtype, jnp.floating)

    def test_jit_vmap_grad(self, x64):
        x = jax.random.normal(jax.random.key(6), (20, 2))
        f = jax.jit(phase_randomize, static_argnums=(2, 3))
        np.testing.assert_allclose(f(jax.random.key(7), x, 0, True),
                                   phase_randomize(jax.random.key(7), x, 0, True), rtol=1e-12)
        keys = jax.random.split(jax.random.key(8), 5)
        batch = jax.vmap(phase_randomize, in_axes=(0, None))(keys, x)
        assert batch.shape == (5, 20, 2)
        assert not np.allclose(batch[0], batch[1])
        check_grads(lambda z: jnp.sum(phase_randomize(jax.random.key(9), z) ** 3), (x,),
                    order=1, modes=['rev'])

    def test_errors(self):
        with pytest.raises(TypeError):
            phase_randomize(jax.random.key(0), jnp.ones(4, dtype=jnp.complex64))
        with pytest.raises(ValueError):
            phase_randomize(jax.random.key(0), jnp.array(1.))
        with pytest.raises(ValueError):
            phase_randomize(jax.random.key(0), jnp.ones((4, 2)), axis=2)
        with pytest.raises(TypeError):
            phase_randomize(jax.random.key(0), jnp.ones((4, 2)), axis=0.5)


# ----------------------------------------------------------------------------
# bootstrap_ci
# ----------------------------------------------------------------------------

class TestBootstrapCi:
    @pytest.mark.parametrize('method', ['percentile', 'basic'])
    def test_scipy_reference(self, method, x64):
        stats = pytest.importorskip('scipy.stats')
        data = np.random.default_rng(0).exponential(2.0, size=60)
        ref = stats.bootstrap((data,), np.mean, n_resamples=20000, confidence_level=0.9,
                              method=method, random_state=np.random.default_rng(1)).confidence_interval
        low, high = bootstrap_ci(jax.random.key(0), data, jnp.mean, n_resamples=20000,
                                 confidence=0.9, method=method)
        # Monte-Carlo tolerance (the interval half-width is ~0.42).
        np.testing.assert_allclose([float(low), float(high)], [ref.low, ref.high], atol=0.02)

    def test_basic_vs_percentile(self):
        data = jnp.asarray(np.random.default_rng(2).exponential(1.0, size=50))
        key = jax.random.key(3)
        pl, ph = bootstrap_ci(key, data, jnp.mean)
        bl, bh = bootstrap_ci(key, data, jnp.mean, method='basic')
        theta = jnp.mean(data)
        np.testing.assert_allclose([bl, bh], [2 * theta - ph, 2 * theta - pl], rtol=1e-5)
        wide = bootstrap_ci(key, data, jnp.mean, confidence=0.99)
        assert wide[0] < pl and wide[1] > ph

    def test_axis_and_vector_statistic(self):
        data = jax.random.normal(jax.random.key(4), (3, 80)) + jnp.arange(3.)[:, None]
        low, high = bootstrap_ci(jax.random.key(5), data, lambda d: d.mean(axis=1), axis=1)
        assert low.shape == (3,)
        assert np.all(np.asarray(low) < np.arange(3.) + 0.5)
        assert np.all(np.asarray(high) > np.arange(3.) - 0.5)
        neg = bootstrap_ci(jax.random.key(5), data, lambda d: d.mean(axis=1), axis=-1)
        np.testing.assert_allclose(neg[0], low)

    def test_units(self):
        data = jax.random.normal(jax.random.key(6), (40,)) + 5.
        key = jax.random.key(7)
        plain = bootstrap_ci(key, data, jnp.mean, method='basic')
        low, high = bootstrap_ci(key, data * u.mV, u.math.mean, method='basic')
        assert low.unit == u.mV
        np.testing.assert_allclose([low.mantissa, high.mantissa], plain, rtol=1e-5)
        vl, vh = bootstrap_ci(key, data * u.mV, lambda d: u.math.var(d))
        assert vl.dim == (u.mV ** 2).dim
        assert 0 < vl.mantissa < vh.mantissa

    @pytest.mark.parametrize('method', ['percentile', 'basic'])
    def test_list_input(self, method):
        data = [1., 2., 3., 4., 5.]
        stat = lambda d: d.mean(axis=0)
        low, high = bootstrap_ci(jax.random.key(0), data, stat, n_resamples=500, method=method)
        ref = bootstrap_ci(jax.random.key(0), jnp.asarray(data), stat, n_resamples=500,
                           method=method)
        np.testing.assert_allclose([low, high], ref)
        assert 1. <= float(low) <= 3. <= float(high) <= 5.

    def test_integer_statistic(self):
        data = jnp.arange(20)
        low, high = bootstrap_ci(jax.random.key(8), data, jnp.max, method='basic')
        assert jnp.issubdtype(low.dtype, jnp.floating)
        assert float(high) >= float(low)

    def test_jit_and_vmap_over_keys(self):
        data = jax.random.normal(jax.random.key(9), (30,))
        f = jax.jit(lambda k, d: bootstrap_ci(k, d, jnp.median, n_resamples=200))
        out = f(jax.random.key(1), data)
        ref = bootstrap_ci(jax.random.key(1), data, jnp.median, n_resamples=200)
        np.testing.assert_allclose(out, ref, rtol=1e-6)
        keys = jax.random.split(jax.random.key(2), 4)
        lows, highs = jax.vmap(lambda k: bootstrap_ci(k, data, jnp.mean, n_resamples=200))(keys)
        assert lows.shape == (4,)
        assert np.all(np.asarray(lows) < np.asarray(highs))

    def test_errors(self):
        data = jnp.ones(5)
        key = jax.random.key(0)
        with pytest.raises(TypeError):
            bootstrap_ci(key, data, 'mean')
        with pytest.raises(ValueError):
            bootstrap_ci(key, data, jnp.mean, method='bca')
        with pytest.raises(ValueError):
            bootstrap_ci(key, data, jnp.mean, confidence=1.)
        with pytest.raises(ValueError):
            bootstrap_ci(key, data, jnp.mean, n_resamples=0)
        with pytest.raises(TypeError):
            bootstrap_ci(key, data, jnp.mean, n_resamples=10.)
        with pytest.raises(TypeError):
            bootstrap_ci(key, data, jnp.mean, n_resamples=True)
        with pytest.raises(ValueError):
            bootstrap_ci(key, jnp.array(1.), jnp.mean)
        with pytest.raises(ValueError):
            bootstrap_ci(key, jnp.ones((0, 2)), jnp.mean)
        with pytest.raises(ValueError):
            bootstrap_ci(key, data, jnp.mean, axis=1)
