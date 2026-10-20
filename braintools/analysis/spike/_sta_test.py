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

from braintools.analysis.spike._sta import spike_triggered_average
from braintools.spike import SpikeTrains


@pytest.fixture
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', prev)


T = 40
SIGNAL = np.sin(np.arange(T) * 0.3) + 0.1 * np.arange(T)


def _st(*trains, t_start=0., t_stop=40.):
    return SpikeTrains.from_list([np.asarray(t, dtype=float) for t in trains], t_start=t_start, t_stop=t_stop)


def _numpy_sta(trains, signal, dt, t_before, t_after, t0=0.):
    """Elephant's algorithm: start = floor((t - t_before - t0) / dt), L = ceil(window / dt)."""
    n_lags = int(np.ceil((t_before + t_after) / dt))
    t_end = t0 + signal.shape[0] * dt
    out = []
    for t in trains:
        rows = [signal[int(np.floor((s - t_before - t0) / dt)):][:n_lags]
                for s in t if s - t_before >= t0 and s + t_after <= t_end]
        out.append(np.mean(rows, axis=0) if rows else np.full((n_lags,) + signal.shape[1:], np.nan))
    return np.stack(out)


class TestElephantReference:

    def test_grid_spikes(self, x64):
        # Elephant 1.2.1 (/tmp/bt/refenv):
        #   an = neo.AnalogSignal(np.stack([SIGNAL] * 3, 1) * pq.mV, sampling_rate=1 / pq.ms)
        #   elephant.sta.spike_triggered_average(an, [neo.SpikeTrain(t * pq.ms, t_stop=40 * pq.ms)
        #       for t in ([1, 10, 20, 37], [5, 6, 30], [2, 38])], (-3 * pq.ms, 2 * pq.ms))
        # Elephant's signed window (-3 ms, 2 ms) is window=(3 ms, 2 ms) here.
        # used_spikes = [3, 3, 1]: the spikes at 1 ms and 2 ms are too close to the start.
        gold = [[1.6791733322425333, 1.7076676443411642, 1.7652527025232339, 1.855717307218984,
                 1.979913226892142],
                [1.8392863979558685, 2.023321634560997, 2.1308344342178214, 2.161153700908002,
                 2.1205038070113775],
                [2.62030424002833, 2.6190637699335086, 2.705447411796011, 2.8806714743353243,
                 3.138016416080967]]
        st = _st([1, 10, 20, 37], [5, 6, 30], [2, 38])
        sta, lags = spike_triggered_average(st, SIGNAL, dt=1., window=(3., 2.))
        np.testing.assert_allclose(np.asarray(sta), gold, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(lags), [-3., -2., -1., 0., 1.])

    def test_off_grid_spikes_and_window(self, x64):
        # Elephant 1.2.1: same signal (2 columns), trains [10.4, 20.7, 2.4] ms and
        # [5.5, 36.2] ms, window (-2.5 * pq.ms, 1 * pq.ms) -> L = ceil(3.5) = 4,
        # used_spikes = [2, 2] (2.4 ms is excluded)
        gold = [[1.2952224395464431, 1.4123888189767566, 1.5239821910174522, 1.6289669542721086],
                [1.9628955079260813, 2.016082199186842, 2.0588996133161923, 2.096455700405852]]
        st = _st([2.4, 10.4, 20.7], [5.5, 36.2])
        sta, lags = spike_triggered_average(st, SIGNAL, dt=1., window=(2.5, 1.))
        np.testing.assert_allclose(np.asarray(sta), gold, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(lags), [-2.5, -1.5, -0.5, 0.5])


class TestBehaviour:

    def test_matches_numpy_multichannel(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(size=(200, 2, 3)).astype(np.float32)
        trains = [np.sort(rng.uniform(0, 20, 15)) for _ in range(4)]
        st = SpikeTrains.from_list(trains, t_start=0., t_stop=20.)
        sta, _ = spike_triggered_average(st, signal, dt=0.1, window=(1., 0.5))
        assert sta.shape == (4, 15, 2, 3)
        np.testing.assert_allclose(np.asarray(sta), _numpy_sta(trains, signal, 0.1, 1., 0.5),
                                   rtol=1e-5, atol=1e-6)

    def test_excluded_near_edges_and_nan(self):
        signal = np.arange(10.)
        # neuron 0: spikes at 0 and 9.5 both leave the signal; neuron 1: silent
        st = _st([0., 9.5], [], [5.], t_stop=10.)
        sta, _ = spike_triggered_average(st, signal, dt=1., window=(1., 1.))
        assert np.all(np.isnan(np.asarray(sta[:2])))
        np.testing.assert_allclose(np.asarray(sta[2]), [4., 5.])

    def test_window_bounds_inclusive(self):
        signal = np.arange(10.)
        # window [-2, 1): spike at 2 starts at sample 0, spike at 9 ends at t_stop = 10
        st = _st([2., 9.], t_stop=10.)
        sta, _ = spike_triggered_average(st, signal, dt=1., window=(2., 1.))
        np.testing.assert_allclose(np.asarray(sta[0]), [(0 + 7) / 2, (1 + 8) / 2, (2 + 9) / 2])

    def test_grid_round_off(self):
        # float32 grid times t_start + i * 0.1 must pick the sample they sit on
        signal = jnp.arange(100.)
        raster = jnp.zeros((100, 1)).at[jnp.array([13, 37, 71]), 0].set(1)
        st = SpikeTrains.from_dense(raster, dt=0.1)
        sta, _ = spike_triggered_average(st, signal, dt=0.1, window=(0.3, 0.2))
        np.testing.assert_allclose(np.asarray(sta[0]), (13 + 37 + 71) / 3 + np.arange(-3, 2), rtol=1e-6)

    def test_t_start_offset(self):
        signal = np.arange(10.)
        st = SpikeTrains.from_list([[105., 107.]], t_start=100., t_stop=110.)
        sta, _ = spike_triggered_average(st, signal, dt=1., window=(1., 1.))
        np.testing.assert_allclose(np.asarray(sta[0]), [5., 6.])

    def test_integer_signal_and_capacity_one(self):
        st = SpikeTrains.from_dense(jnp.zeros((10, 2)).at[5, 0].set(1), dt=1.)
        sta, _ = spike_triggered_average(st, jnp.arange(10), dt=1., window=(1., 1.))
        assert jnp.issubdtype(sta.dtype, jnp.floating)
        np.testing.assert_allclose(np.asarray(sta[0]), [4., 5.])
        assert np.all(np.isnan(np.asarray(sta[1])))

    def test_errors(self):
        st = _st([5.])
        with pytest.raises(TypeError, match='SpikeTrains'):
            spike_triggered_average(np.ones(3), SIGNAL, 1., (1., 1.))
        with pytest.raises(ValueError, match='scalar'):
            spike_triggered_average(st, 1., 1., (1., 1.))
        with pytest.raises(ValueError, match='positive'):
            spike_triggered_average(st, SIGNAL, 0., (1., 1.))
        with pytest.raises(ValueError, match='positive length'):
            spike_triggered_average(st, SIGNAL, 1., (0., 0.))
        with pytest.raises(TypeError, match='static'):
            jax.jit(lambda d: spike_triggered_average(st, SIGNAL, d, (1., 1.)))(1.)


class TestUnits:

    def test_quantities(self):
        trains = [[10., 20.], [15.]]
        plain = _st(*trains)
        quant = SpikeTrains.from_list([np.asarray(t) * u.ms for t in trains],
                                      t_start=0. * u.ms, t_stop=40. * u.ms)
        ref, _ = spike_triggered_average(plain, SIGNAL, dt=1., window=(3., 2.))
        sta, lags = spike_triggered_average(quant, SIGNAL * u.mV, dt=1. * u.ms,
                                            window=(3. * u.ms, 2. * u.ms))
        assert isinstance(sta, u.Quantity) and sta.unit == u.mV
        assert isinstance(lags, u.Quantity) and lags.unit == u.ms
        np.testing.assert_allclose(np.asarray(sta.mantissa), np.asarray(ref), rtol=1e-6)
        # plain numbers are milliseconds; seconds are converted
        sta2, _ = spike_triggered_average(quant, SIGNAL, dt=0.001 * u.second, window=(3., 2.))
        np.testing.assert_allclose(np.asarray(sta2), np.asarray(ref), rtol=1e-6)


class TestTransforms:

    def test_jit(self):
        st = _st([1, 10, 20, 37], [5, 6, 30])
        f = jax.jit(lambda s, x: spike_triggered_average(s, x, 1., (3., 2.))[0])
        np.testing.assert_allclose(np.asarray(f(st, SIGNAL)),
                                   np.asarray(spike_triggered_average(st, SIGNAL, 1., (3., 2.))[0]),
                                   rtol=1e-6)

    def test_vmap_over_trials(self):
        rasters = jax.random.bernoulli(jax.random.PRNGKey(0), 0.1, (3, 40, 2))
        signals = jax.random.normal(jax.random.PRNGKey(1), (3, 40))
        batched = jax.vmap(lambda r: SpikeTrains.from_dense(r, dt=1., capacity=40))(rasters)
        out = jax.vmap(lambda s, x: spike_triggered_average(s, x, 1., (3., 2.))[0])(batched, signals)
        for trial in range(3):
            single = SpikeTrains.from_dense(rasters[trial], dt=1.)
            ref = spike_triggered_average(single, signals[trial], 1., (3., 2.))[0]
            np.testing.assert_allclose(np.asarray(out[trial]), np.asarray(ref), rtol=1e-6)

    def test_grad_wrt_signal(self):
        # neuron 1 has no usable spike (nan); its nan must not leak into the gradient
        st = _st([5., 10.], [0.5])
        x = jnp.asarray(SIGNAL, dtype=jnp.float32)
        g = jax.grad(lambda s: jnp.sum(spike_triggered_average(st, s, 1., (1., 1.))[0][0]))(x)
        expected = np.zeros(T)
        expected[[4, 5, 9, 10]] = 0.5
        np.testing.assert_allclose(np.asarray(g), expected)
