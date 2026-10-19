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

from braintools.analysis.spike._psth import psth
from braintools.spike import SpikeTrains


@pytest.fixture
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', prev)


def _numpy_psth(trains, events, t_before, t_after, bin_size):
    """Brute-force reference: half-open bins [-t_before + i b, -t_before + (i+1) b)."""
    n_bins = int(np.ceil((t_before + t_after) / bin_size))
    out = np.zeros((n_bins, len(trains)), dtype=int)
    for n, t in enumerate(trains):
        for e in events:
            if not np.isfinite(e):
                continue
            idx = np.floor((np.asarray(t) - e + t_before) / bin_size).astype(int)
            idx = idx[(idx >= 0) & (idx < n_bins)]
            np.add.at(out[:, n], idx, 1)
    return out


def _random_trains(seed, n=3, k=40, t_stop=100.):
    rng = np.random.default_rng(seed)
    return [np.sort(rng.uniform(0., t_stop, rng.integers(0, k))) for _ in range(n)]


class TestReference:

    def test_matches_numpy(self, x64):
        trains = _random_trains(0)
        events = np.array([10.3, 35.7, 61.1, 88.9])
        st = SpikeTrains.from_list(trains, t_start=0., t_stop=100.)
        counts, edges = psth(st, events, window=(5., 7.5), bin_size=1.5, rate=False)
        ref = _numpy_psth(trains, events, 5., 7.5, 1.5)
        np.testing.assert_array_equal(np.asarray(counts), ref)
        np.testing.assert_allclose(np.asarray(edges), -5. + 1.5 * np.arange(10))

    def test_rate_in_hz(self, x64):
        trains = _random_trains(1)
        events = np.array([20., 50., 80.])
        st = SpikeTrains.from_list(trains, t_start=0., t_stop=100.)
        counts, _ = psth(st, events, window=(10., 10.), bin_size=2., rate=False)
        rates, _ = psth(st, events, window=(10., 10.), bin_size=2.)
        # 3 events, 2 ms bins: rate = count / (3 * 0.002 s)
        np.testing.assert_allclose(np.asarray(rates), np.asarray(counts) / (3 * 0.002), rtol=1e-12)

    def test_float32(self):
        trains = _random_trains(2)
        events = np.array([10.3, 35.7, 61.1, 88.9])
        st = SpikeTrains.from_list(trains, t_start=0., t_stop=100.)
        counts, _ = psth(st, events, window=(5., 7.5), bin_size=1.5, rate=False)
        np.testing.assert_array_equal(np.asarray(counts), _numpy_psth(trains, events, 5., 7.5, 1.5))


class TestBinEdges:

    def test_half_open_bins(self):
        # event at 10; bins [-2, 0), [0, 2), [2, 4)
        st = SpikeTrains.from_list([[8., 10., 12., 14.]], t_start=0., t_stop=20.)
        counts, _ = psth(st, [10.], window=(2., 4.), bin_size=2., rate=False)
        # 8 is on the lower window bound (kept), 10 and 12 on inner edges (later bin),
        # 14 on the upper window bound (excluded)
        np.testing.assert_array_equal(np.asarray(counts[:, 0]), [1, 1, 1])

    def test_grid_round_off(self):
        # times t_start + i * 0.1 carry float32 round-off; every spike must still
        # land in its own 0.1 ms bin
        raster = jnp.zeros((100, 1)).at[20:40, 0].set(1)
        st = SpikeTrains.from_dense(raster, dt=0.1)
        counts, _ = psth(st, jnp.array([2.0]), window=(0., 2.), bin_size=0.1, rate=False)
        np.testing.assert_array_equal(np.asarray(counts[:, 0]), np.ones(20, dtype=int))

    def test_window_not_multiple_of_bin(self):
        st = SpikeTrains.from_list([[10., 13.4, 13.6]], t_start=0., t_stop=20.)
        counts, edges = psth(st, [10.], window=(0., 3.), bin_size=2., rate=False)
        # n_bins = ceil(3 / 2) = 2; the last bin is a full bin [2, 4)
        np.testing.assert_allclose(np.asarray(edges), [0., 2., 4.])
        np.testing.assert_array_equal(np.asarray(counts[:, 0]), [1, 2])

    def test_negative_t_before(self):
        # window (-1, 3) spans [1, 3) after the event
        st = SpikeTrains.from_list([[10.5, 11., 12.5, 13.]], t_start=0., t_stop=20.)
        counts, edges = psth(st, [10.], window=(-1., 3.), bin_size=1., rate=False)
        np.testing.assert_allclose(np.asarray(edges), [1., 2., 3.])
        np.testing.assert_array_equal(np.asarray(counts[:, 0]), [1, 1])


class TestEvents:

    def test_inf_padding_ignored(self):
        trains = _random_trains(3)
        st = SpikeTrains.from_list(trains, t_start=0., t_stop=100.)
        a, _ = psth(st, jnp.array([30., 60.]), window=(5., 5.), bin_size=1.)
        b, _ = psth(st, jnp.array([30., jnp.inf, 60., jnp.inf, -jnp.inf, jnp.nan]),
                    window=(5., 5.), bin_size=1.)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))

    def test_window_outside_recording(self):
        st = SpikeTrains.from_list([[0., 1., 2.], [98., 99.]], t_start=0., t_stop=100.)
        counts, _ = psth(st, [1., 99.], window=(5., 5.), bin_size=1., rate=False)
        ref = _numpy_psth([[0., 1., 2.], [98., 99.]], [1., 99.], 5., 5., 1.)
        np.testing.assert_array_equal(np.asarray(counts), ref)
        assert int(counts.sum()) == 5

    def test_no_finite_event(self):
        st = SpikeTrains.from_list([[1., 2.]], t_start=0., t_stop=5.)
        counts, _ = psth(st, jnp.array([jnp.inf]), window=(1., 1.), bin_size=1., rate=False)
        np.testing.assert_array_equal(np.asarray(counts), 0)
        rates, _ = psth(st, jnp.array([jnp.inf]), window=(1., 1.), bin_size=1.)
        assert np.all(np.isnan(np.asarray(rates)))

    def test_silent_and_capacity_one(self):
        st = SpikeTrains.from_dense(jnp.zeros((10, 3)).at[4, 1].set(1), dt=1.)
        assert st.capacity == 1
        counts, _ = psth(st, [4.], window=(2., 2.), bin_size=1., rate=False)
        np.testing.assert_array_equal(np.asarray(counts), [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]])


class TestUnits:

    def test_quantity_and_plain_agree(self):
        trains = _random_trains(4)
        plain = SpikeTrains.from_list(trains, t_start=0., t_stop=100.)
        quant = SpikeTrains.from_list([t * u.ms for t in trains], t_start=0. * u.ms, t_stop=100. * u.ms)
        r0, e0 = psth(plain, [30., 70.], window=(10., 10.), bin_size=2.)
        r1, e1 = psth(quant, jnp.array([30., 70.]) * u.ms, window=(10. * u.ms, 10. * u.ms),
                      bin_size=2. * u.ms)
        np.testing.assert_allclose(np.asarray(r0), np.asarray(r1))
        assert isinstance(e1, u.Quantity) and e1.unit == u.ms
        assert not isinstance(r1, u.Quantity)
        np.testing.assert_allclose(np.asarray(e0), np.asarray(e1.mantissa))
        # plain numbers are milliseconds, also for unit-carrying trains
        r2, _ = psth(quant, [30., 70.], window=(10., 10.), bin_size=2.)
        np.testing.assert_allclose(np.asarray(r0), np.asarray(r2))

    def test_seconds_trains(self):
        trains = _random_trains(5)
        ms = SpikeTrains.from_list(trains, t_start=0., t_stop=100.)
        sec = SpikeTrains.from_list([t / 1000. * u.second for t in trains],
                                    t_start=0. * u.second, t_stop=0.1 * u.second)
        r0, _ = psth(ms, [30.3, 70.7], window=(10., 10.), bin_size=2.)
        r1, e1 = psth(sec, jnp.array([30.3, 70.7]) * u.ms, window=(10. * u.ms, 10. * u.ms),
                      bin_size=2. * u.ms)
        np.testing.assert_allclose(np.asarray(r0), np.asarray(r1), rtol=1e-5)
        assert e1.unit == u.second


class TestErrors:

    def test_errors(self):
        st = SpikeTrains.from_list([[1., 2.]], t_start=0., t_stop=5.)
        with pytest.raises(TypeError, match='SpikeTrains'):
            psth(np.ones((2, 2)), [1.], (1., 1.), 1.)
        with pytest.raises(ValueError, match='pair'):
            psth(st, [1.], (1.,), 1.)
        with pytest.raises(ValueError, match='positive length'):
            psth(st, [1.], (1., -1.), 1.)
        with pytest.raises(ValueError, match='positive'):
            psth(st, [1.], (1., 1.), 0.)
        with pytest.raises(ValueError, match=r'\(E,\)'):
            psth(st, [[1.]], (1., 1.), 1.)
        with pytest.raises(TypeError, match='static'):
            jax.jit(lambda b: psth(st, [1.], (1., 1.), b))(1.)


class TestTransforms:

    def test_jit_with_traced_events_and_trains(self):
        trains = _random_trains(6)
        st = SpikeTrains.from_list(trains, t_start=0., t_stop=100.)
        events = jnp.array([25., 50., 75.])
        f = jax.jit(lambda s, e: psth(s, e, window=(5., 5.), bin_size=1.))
        r_jit, e_jit = f(st, events)
        r, e = psth(st, events, window=(5., 5.), bin_size=1.)
        np.testing.assert_allclose(np.asarray(r_jit), np.asarray(r))
        np.testing.assert_allclose(np.asarray(e_jit), np.asarray(e))

    def test_vmap_over_trials(self):
        rasters = jax.random.bernoulli(jax.random.PRNGKey(0), 0.2, (4, 80, 2))
        batched = jax.vmap(lambda r: SpikeTrains.from_dense(r, dt=1., capacity=80))(rasters)
        events = jnp.array([20., 40., 60.])
        out = jax.vmap(lambda s: psth(s, events, (5., 5.), 1., rate=False)[0])(batched)
        assert out.shape == (4, 10, 2)
        for trial in range(4):
            single = SpikeTrains.from_dense(rasters[trial], dt=1.)
            np.testing.assert_array_equal(np.asarray(out[trial]),
                                          np.asarray(psth(single, events, (5., 5.), 1., rate=False)[0]))

    def test_vmap_over_event_sets(self):
        st = SpikeTrains.from_list(_random_trains(7), t_start=0., t_stop=100.)
        sets = jnp.array([[20., 40., jnp.inf], [30., 50., 70.]])
        out = jax.vmap(lambda e: psth(st, e, (5., 5.), 1.)[0])(sets)
        np.testing.assert_allclose(np.asarray(out[0]), np.asarray(psth(st, sets[0], (5., 5.), 1.)[0]))
        np.testing.assert_allclose(np.asarray(out[1]), np.asarray(psth(st, sets[1], (5., 5.), 1.)[0]))


class TestMemory:

    def test_temp_memory_bounded(self):
        # N = E = 1000 with 100 bins needed ~2.4 GB of temporaries before; only compiled
        compiled = jax.jit(lambda s, e: psth(s, e, (50., 50.), 1.)[0]).lower(
            SpikeTrains(jnp.zeros((1000, 1000)), jnp.zeros(1000, jnp.int32), 0., 1000.),
            jnp.zeros(1000),
        ).compile()
        try:
            analysis = compiled.memory_analysis()
        except Exception:  # pragma: no cover - backend without the API
            analysis = None
        if analysis is None or not hasattr(analysis, 'temp_size_in_bytes'):  # pragma: no cover
            pytest.skip('memory_analysis() is not available on this backend')
        assert analysis.temp_size_in_bytes < 256 * 2 ** 20, analysis.temp_size_in_bytes

    def test_empty_events_and_batched_scan(self):
        st = SpikeTrains.from_list(_random_trains(8), t_start=0., t_stop=100.)
        counts, _ = psth(st, jnp.zeros(0), window=(5., 5.), bin_size=1., rate=False)
        np.testing.assert_array_equal(np.asarray(counts), 0)
        # many events split over several scan steps give the same counts as one step
        import braintools.analysis.spike._psth as mod
        events = jnp.linspace(5., 95., 37)
        full, _ = psth(st, events, window=(5., 5.), bin_size=1., rate=False)
        old = mod._BATCH_ELEMENTS
        mod._BATCH_ELEMENTS = 5 * st.n_neurons * 11  # 5 events per step, 8 steps with padding
        try:
            split, _ = psth(st, events, window=(5., 5.), bin_size=1., rate=False)
        finally:
            mod._BATCH_ELEMENTS = old
        np.testing.assert_array_equal(np.asarray(split), np.asarray(full))
