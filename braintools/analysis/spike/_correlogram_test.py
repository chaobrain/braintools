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

from braintools.analysis.spike._correlogram import auto_correlogram, cross_correlogram
from braintools.spike import SpikeTrains


@pytest.fixture
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', prev)


A = [1., 4., 5., 9., 12., 13., 17.]
B = [2., 4., 7., 10., 14., 18., 19.]


def _st(*trains, unit=None):
    trains = [np.asarray(t, dtype=float) for t in trains]
    if unit is None:
        return SpikeTrains.from_list(trains, t_start=0., t_stop=20.)
    return SpikeTrains.from_list([t * unit for t in trains], t_start=0. * unit, t_stop=20. * unit)


def _numpy_cch(a, b, bin_size, max_lag, auto=False):
    """Brute force: bin j covers [(j - 1/2) b, (j + 1/2) b) for j = -M..M."""
    m = int(np.ceil(max_lag / bin_size))
    out = np.zeros(2 * m + 1, dtype=int)
    for i, ta in enumerate(a):
        for j, tb in enumerate(b):
            if auto and i == j:
                continue
            k = int(np.floor((tb - ta) / bin_size + 0.5))
            if -m <= k <= m:
                out[k + m] += 1
    return out


class TestElephantReference:
    # Golden values from Elephant 1.2.1 (/tmp/bt/refenv):
    #   ba = BinnedSpikeTrain(neo.SpikeTrain(A * pq.ms, t_start=0 * pq.ms, t_stop=20 * pq.ms),
    #                         bin_size=1 * pq.ms); bb likewise for B
    #   elephant.spike_train_correlation.cross_correlation_histogram(
    #       ba, bb, window=[-5, 5], border_correction=False, binary=False)
    # Elephant counts the difference of bin indices of the binned trains; with
    # spikes on the bin grid this equals binning the exact time lags t_b - t_a
    # (positive lag: b after a), which is what braintools does.

    def test_cross(self):
        counts, lags = cross_correlogram(_st(A), _st(B), bin_size=1., max_lag=5.)
        np.testing.assert_array_equal(np.asarray(counts[0]), [2, 0, 3, 3, 1, 1, 4, 3, 2, 0, 3])
        np.testing.assert_allclose(np.asarray(lags), np.arange(-5., 6.))

    def test_auto(self):
        # Elephant cross_correlation_histogram(ba, ba, window=[-5, 5]) gives
        # [2, 4, 2, 0, 2, 7, 2, 0, 2, 4, 2]; its centre bin holds the 7 self-pairs,
        # which auto_correlogram removes.
        counts, _ = auto_correlogram(_st(A), bin_size=1., max_lag=5.)
        np.testing.assert_array_equal(np.asarray(counts[0]), [2, 4, 2, 0, 2, 0, 2, 0, 2, 4, 2])

    def test_cross_bin_two(self):
        # Elephant: a2 = [0, 4, 6, 10, 16] ms, b2 = [2, 6, 8, 12, 18] ms, bin_size=2 ms,
        #   window=[-3, 3] -> [0, 3, 2, 1, 5, 1, 2], lags [-3..3] bins
        a2 = _st([0., 4., 6., 10., 16.], unit=u.ms)
        b2 = _st([2., 6., 8., 12., 18.], unit=u.ms)
        counts, lags = cross_correlogram(a2, b2, bin_size=2. * u.ms, max_lag=6. * u.ms)
        np.testing.assert_array_equal(np.asarray(counts[0]), [0, 3, 2, 1, 5, 1, 2])
        np.testing.assert_allclose(np.asarray(lags.to_decimal(u.ms)), 2. * np.arange(-3, 4))


class TestNumpyReference:

    def test_off_grid_random(self, x64):
        rng = np.random.default_rng(0)
        a = [np.sort(rng.uniform(0, 20, n)) for n in (15, 0, 8)]
        b = [np.sort(rng.uniform(0, 20, n)) for n in (12, 5, 0)]
        counts, _ = cross_correlogram(_st(*a), _st(*b), bin_size=0.7, max_lag=3.)
        for n in range(3):
            np.testing.assert_array_equal(np.asarray(counts[n]), _numpy_cch(a[n], b[n], 0.7, 3.))
        auto, _ = auto_correlogram(_st(*a), bin_size=0.7, max_lag=3.)
        for n in range(3):
            np.testing.assert_array_equal(np.asarray(auto[n]), _numpy_cch(a[n], a[n], 0.7, 3., auto=True))

    def test_float32_grid(self):
        # float32 grid times carry round-off; lags that are multiples of dt must
        # still land in their own bins
        raster = jax.random.bernoulli(jax.random.PRNGKey(0), 0.3, (60, 2))
        st = SpikeTrains.from_dense(raster, dt=0.1)
        counts, _ = cross_correlogram(st.select([0]), st.select([1]), bin_size=0.1, max_lag=1.)
        idx = np.nonzero(np.asarray(raster))
        ta = idx[0][idx[1] == 0].astype(float)
        tb = idx[0][idx[1] == 1].astype(float)
        np.testing.assert_array_equal(np.asarray(counts[0]), _numpy_cch(ta, tb, 1., 10.))


class TestShapesAndEdges:

    def test_n_bins_odd_and_centred(self):
        st = _st(A)
        for max_lag, n_bins in [(0., 1), (0.4, 3), (1., 3), (2.5, 7)]:
            counts, lags = cross_correlogram(st, st, bin_size=1., max_lag=max_lag)
            assert counts.shape == (1, n_bins)
            assert float(lags[n_bins // 2]) == 0.

    def test_on_edge_lags_go_farther_from_zero(self):
        # lags exactly on bin edges (+-0.5, +-1.5 with b = 1) go to the bin farther
        # from lag 0 on both sides: -0.5 -> bin -1, +0.5 -> bin +1, and +-1.5 fall
        # outside the 3 bins; the centre bin is the open interval (-0.5, 0.5)
        a = _st([10.])
        b = _st([8.5, 9.5, 10., 10.5, 11.5])
        counts, _ = cross_correlogram(a, b, bin_size=1., max_lag=1.)
        np.testing.assert_array_equal(np.asarray(counts[0]), [1, 1, 1])
        counts, _ = cross_correlogram(a, b, bin_size=1., max_lag=2.)
        np.testing.assert_array_equal(np.asarray(counts[0]), [1, 1, 1, 1, 1])
        counts, _ = cross_correlogram(a, _st([9.5, 10.5]), bin_size=1., max_lag=1.)
        np.testing.assert_array_equal(np.asarray(counts[0]), [1, 0, 1])

    def test_empty_and_single_spike_neurons(self):
        st = _st([], [3.], [1., 2.])
        counts, _ = auto_correlogram(st, bin_size=1., max_lag=2.)
        np.testing.assert_array_equal(np.asarray(counts), [[0] * 5, [0] * 5, [0, 1, 0, 1, 0]])
        cross, _ = cross_correlogram(st, _st([], [], []), bin_size=1., max_lag=2.)
        np.testing.assert_array_equal(np.asarray(cross), 0)

    def test_auto_is_symmetric(self):
        rng = np.random.default_rng(1)
        st = _st(*[np.sort(rng.uniform(0, 20, 10)) for _ in range(3)])
        counts, _ = auto_correlogram(st, bin_size=0.5, max_lag=4.)
        # exact symmetry up to spikes on bin edges (none for random times)
        np.testing.assert_array_equal(np.asarray(counts), np.asarray(counts)[:, ::-1])

    def test_capacity_one(self):
        st = SpikeTrains.from_dense(jnp.eye(3), dt=1.)
        counts, _ = auto_correlogram(st, bin_size=1., max_lag=1.)
        np.testing.assert_array_equal(np.asarray(counts), 0)

    def test_errors(self):
        st = _st(A)
        with pytest.raises(TypeError, match='SpikeTrains'):
            cross_correlogram(st, np.ones((1, 2)), 1., 1.)
        with pytest.raises(ValueError, match='same number'):
            cross_correlogram(st, _st(A, B), 1., 1.)
        with pytest.raises(ValueError, match='positive'):
            cross_correlogram(st, st, 0., 1.)
        with pytest.raises(ValueError, match='non-negative'):
            cross_correlogram(st, st, 1., -1.)
        with pytest.raises(TypeError, match='static'):
            jax.jit(lambda m: cross_correlogram(st, st, 1., m))(1.)


class TestUnits:

    def test_mixed_units_converted(self):
        a_ms = _st(A, unit=u.ms)
        b_s = SpikeTrains.from_list([np.asarray(B) / 1000. * u.second],
                                    t_start=0. * u.second, t_stop=0.02 * u.second)
        counts, lags = cross_correlogram(a_ms, b_s, bin_size=1. * u.ms, max_lag=5. * u.ms)
        np.testing.assert_array_equal(np.asarray(counts[0]), [2, 0, 3, 3, 1, 1, 4, 3, 2, 0, 3])
        assert lags.unit == u.ms

    def test_plain_vs_quantity(self):
        plain, _ = cross_correlogram(_st(A), _st(B), bin_size=1., max_lag=5.)
        quant, lags = cross_correlogram(_st(A, unit=u.ms), _st(B, unit=u.ms), bin_size=1., max_lag=5.)
        np.testing.assert_array_equal(np.asarray(plain), np.asarray(quant))
        assert isinstance(lags, u.Quantity)
        # unitless trains are milliseconds, so a Quantity bin size is converted
        conv, _ = cross_correlogram(_st(A), _st(B), bin_size=0.001 * u.second, max_lag=0.005 * u.second)
        np.testing.assert_array_equal(np.asarray(plain), np.asarray(conv))


class TestTransforms:

    def test_jit(self):
        st, sb = _st(A, B), _st(B, A)
        f = jax.jit(lambda x, y: cross_correlogram(x, y, 1., 5.))
        np.testing.assert_array_equal(np.asarray(f(st, sb)[0]),
                                      np.asarray(cross_correlogram(st, sb, 1., 5.)[0]))
        g = jax.jit(lambda x: auto_correlogram(x, 1., 5.))
        np.testing.assert_array_equal(np.asarray(g(st)[0]), np.asarray(auto_correlogram(st, 1., 5.)[0]))

    def test_vmap_over_trials(self):
        rasters = jax.random.bernoulli(jax.random.PRNGKey(2), 0.2, (3, 50, 2))
        batched = jax.vmap(lambda r: SpikeTrains.from_dense(r, dt=1., capacity=50))(rasters)
        out = jax.vmap(lambda s: auto_correlogram(s, 1., 4.)[0])(batched)
        for trial in range(3):
            single = SpikeTrains.from_dense(rasters[trial], dt=1.)
            np.testing.assert_array_equal(np.asarray(out[trial]),
                                          np.asarray(auto_correlogram(single, 1., 4.)[0]))

    def test_all_pairs_via_select(self):
        st = _st(A, B, [3., 8., 15.])
        i, j = jnp.triu_indices(3, k=1)
        pairs = jax.vmap(lambda i, j: cross_correlogram(st.select(i), st.select(j), 1., 3.)[0][0])(i, j)
        for p, (ii, jj) in enumerate(zip(np.asarray(i), np.asarray(j))):
            ref = cross_correlogram(st.select([int(ii)]), st.select([int(jj)]), 1., 3.)[0][0]
            np.testing.assert_array_equal(np.asarray(pairs[p]), np.asarray(ref))


def _compiled_temp_bytes(fn, *args):
    compiled = jax.jit(fn).lower(*args).compile()
    try:
        analysis = compiled.memory_analysis()
    except Exception:  # pragma: no cover - backend without the API
        analysis = None
    if analysis is None or not hasattr(analysis, 'temp_size_in_bytes'):  # pragma: no cover
        pytest.skip('memory_analysis() is not available on this backend')
    return analysis.temp_size_in_bytes


class TestSymmetry:
    """Regression: on-edge lags went to the later bin, so +lag and -lag differed."""

    def test_two_spikes_even_bin(self):
        st = _st([0., 1.])
        counts, _ = auto_correlogram(st, bin_size=2., max_lag=4.)
        # lags +-1 = +-bin_size / 2 both go to the bin farther from 0
        np.testing.assert_array_equal(np.asarray(counts[0]), [0, 1, 0, 1, 0])

    @pytest.mark.parametrize('dt', [1., 0.1])
    @pytest.mark.parametrize('mult', [1, 2, 3, 4])
    def test_grid_auto_and_reversed_cross_are_mirror_symmetric(self, dt, mult):
        raster = jax.random.bernoulli(jax.random.PRNGKey(mult), 0.3, (80, 3))
        st = SpikeTrains.from_dense(raster, dt=dt)
        bin_size, max_lag = mult * dt, 10 * dt
        auto, _ = auto_correlogram(st, bin_size, max_lag)
        np.testing.assert_array_equal(np.asarray(auto), np.asarray(auto)[:, ::-1])
        ab, _ = cross_correlogram(st.select([0, 1]), st.select([1, 2]), bin_size, max_lag)
        ba, _ = cross_correlogram(st.select([1, 2]), st.select([0, 1]), bin_size, max_lag)
        np.testing.assert_array_equal(np.asarray(ab), np.asarray(ba)[:, ::-1])


class TestMemory:

    def test_cross_correlogram_temp_memory_bounded(self):
        # N = K = 1000 with 201 lag bins needed ~4.8 GB of temporaries before;
        # only compiled, never run
        st = SpikeTrains(jnp.zeros((1000, 1000)), jnp.zeros(1000, jnp.int32), 0., 1000.)
        temp = _compiled_temp_bytes(lambda s: cross_correlogram(s, s, 1., 100.)[0], st)
        assert temp < 256 * 2 ** 20, temp

    def test_psth_and_correlogram_agree_off_edges(self):
        # with no lag on a bin edge, the correlogram is the PSTH of b aligned to a's spikes
        from braintools.analysis.spike._psth import psth
        a, b = [1.3, 7.7], [2.1, 4.9, 8.2, 9.6]
        counts, _ = cross_correlogram(_st(a), _st(b), bin_size=1., max_lag=3.)
        ref, _ = psth(_st(b), jnp.array(a), window=(3.5, 3.5), bin_size=1., rate=False)
        np.testing.assert_array_equal(np.asarray(counts[0]), np.asarray(ref[:, 0]))


class TestNoNeurons:

    def test_zero_neurons(self):
        st = SpikeTrains(jnp.zeros((0, 2)), jnp.zeros(0, jnp.int32), 0., 1.)
        counts, lags = auto_correlogram(st, 1., 2.)
        assert counts.shape == (0, 5) and lags.shape == (5,)

    def test_neuron_batches_match_single_batch(self):
        import braintools.analysis.spike._psth as mod
        raster = jax.random.bernoulli(jax.random.PRNGKey(9), 0.3, (40, 7))
        st = SpikeTrains.from_dense(raster, dt=1.)
        full, _ = auto_correlogram(st, 1., 3.)
        old = mod._BATCH_ELEMENTS
        mod._BATCH_ELEMENTS = 3 * st.capacity * 9  # 3 neurons per batch, remainder 1
        try:
            split, _ = auto_correlogram(st, 1., 3.)
        finally:
            mod._BATCH_ELEMENTS = old
        np.testing.assert_array_equal(np.asarray(split), np.asarray(full))
