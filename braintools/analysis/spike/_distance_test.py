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

from braintools.analysis.spike._distance import isi_distance, spike_distance
from braintools.spike import SpikeTrains

# Golden values: PySpike 0.9.0, generated in the isolated reference env with
#   s1 = pyspike.SpikeTrain(a, edges=(t0, t1)); s2 = pyspike.SpikeTrain(b, edges=(t0, t1))
#   pyspike.isi_distance(s1, s2), pyspike.spike_distance(s1, s2)
# (name, a, b, t0, t1, isi, spike)
GOLDEN = [
    ('basic', [12.0, 16.0, 28.0, 32.0, 44.0, 48.0, 60.0, 64.0, 76.0, 80.0],
     [8.0, 20.0, 36.0, 52.0, 68.0, 84.0, 92.0], 0.0, 100.0, 0.36066666666666664, 0.4069469387755102),
    ('identical', [1.0, 5.0, 9.0], [1.0, 5.0, 9.0], 0.0, 10.0, 0.0, 0.0),
    ('empty_vs_one', [], [5.0], 0.0, 10.0, 0.5, 0.4444444444444445),
    ('empty_vs_many', [], [2.0, 4.0, 7.0], 0.0, 10.0, 0.74, 0.3696581196581197),
    ('both_empty', [], [], 0.0, 10.0, 0.0, 0.0),
    ('single_single', [3.0], [6.0], 0.0, 10.0, 0.36428571428571427, 0.5566433566433566),
    ('single_at_start', [0.0], [6.0], 0.0, 10.0, 0.4800000000000001, 0.350765306122449),
    ('single_at_stop', [10.0], [6.0], 0.0, 10.0, 0.4800000000000001, 0.350765306122449),
    ('both_at_start', [0.0, 4.0], [0.0, 7.0], 0.0, 10.0, 0.2571428571428571, 0.3553188070670588),
    ('edges_both', [0.0, 5.0, 10.0], [2.0, 10.0], 0.0, 10.0, 0.375, 0.2603550295857988),
    ('coincident_mid', [2.0, 5.0, 8.0], [3.0, 5.0, 9.0], 0.0, 10.0, 0.19166666666666665, 0.22576643990929704),
    ('start_and_single_start', [0.0], [0.0, 3.0, 6.0], 0.0, 10.0, 0.6599999999999999, 0.3407800990218573),
    ('offset_window', [11.0, 13.5, 17.0], [12.0, 16.0, 18.5], 10.0, 20.0, 0.27678571428571425, 0.358683373250681),
    ('single_at_start_both', [0.0], [0.0], 0.0, 10.0, 0.0, 0.0),
    ('single_start_vs_empty', [0.0], [], 0.0, 10.0, 0.0, 0.0),
    ('dense_vs_sparse', [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
                         8.0, 8.5, 9.0, 9.5], [5.0], 0.0, 10.0, 0.9, 0.4214876033057851),
    ('random0', [1.6, 4.0, 7.4, 17.4, 26.7, 30.5, 50.3, 50.6, 62.9, 64.8, 81.0, 91.1],
     [3.3, 55.2, 72.8, 76.3, 84.6, 85.3], 0.0, 100.0, 0.5785401628473703, 0.38155184176945867),
    ('random1', [29.9, 48.1], [0.5, 0.8, 2.8, 12.3, 52.5, 66.9], 0.0, 100.0,
     0.4699261911487664, 0.3370081078943797),
    ('random2', [37.8, 65.0, 68.4, 80.0, 94.9, 97.6, 99.1], [30.9, 37.3, 42.1, 48.4, 52.2, 71.5, 71.8, 83.9, 88.9],
     0.0, 100.0, 0.411351982503479, 0.27777946685215976),
    ('random3', [22.6, 26.3, 32.6, 33.5, 38.9, 49.9, 58.8, 62.3, 71.3, 75.5, 88.6], [5.8, 67.1], 0.0, 100.0,
     0.7999738988580752, 0.4048849765294865),
]

# Repeated spike times (PySpike drops them with ``np.unique``) and spikes on the
# bounds; same call as above with edges=(0., 20.). (a, b, isi, spike)
GOLDEN_DUPLICATES = [
    ([2.0, 2.0, 6.0, 9.0, 9.0, 16.0], [2.0, 20.0], 0.6944444444444445, 0.2826983482318547),
    ([1.0, 10.0, 10.0, 14.0, 15.0, 20.0], [1.0, 3.0, 11.0, 17.0], 0.31388888888888883, 0.26231410598998955),
    ([5.0, 5.0, 8.0, 15.0, 17.0, 20.0], [4.0, 9.0, 13.0, 14.0, 17.0], 0.26476190476190475, 0.1879026342975207),
    ([0.0, 3.0, 7.0, 10.0, 18.0, 20.0], [0.0, 8.0, 12.0, 12.0, 16.0, 18.0, 18.0], 0.475, 0.236812690916587),
    ([1.0, 3.0, 5.0, 5.0, 15.0, 20.0], [6.0], 0.47595238095238096, 0.2477454120498615),
]


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


def _trains(rows, t0, t1):
    return SpikeTrains.from_list([np.asarray(r, dtype=float) for r in rows], t_start=t0, t_stop=t1)


def _golden_by_window():
    """Group the golden cases by recording window so each group is one call."""
    groups = {}
    for name, a, b, t0, t1, isi, spike in GOLDEN:
        groups.setdefault((t0, t1), []).append((a, b, isi, spike))
    return groups


# ----------------------------------------------------------------------------
# reference values
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('fn, col', [(isi_distance, 2), (spike_distance, 3)])
def test_matches_pyspike_float64(x64, fn, col):
    for (t0, t1), cases in _golden_by_window().items():
        a = _trains([c[0] for c in cases], t0, t1)
        b = _trains([c[1] for c in cases], t0, t1)
        got = fn(a, b)
        assert got.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(got), [c[col] for c in cases], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('fn, col', [(isi_distance, 2), (spike_distance, 3)])
def test_matches_pyspike_float32(fn, col):
    for (t0, t1), cases in _golden_by_window().items():
        a = _trains([c[0] for c in cases], t0, t1)
        b = _trains([c[1] for c in cases], t0, t1)
        got = fn(a, b)
        assert got.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(got), [c[col] for c in cases], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('fn, col', [(isi_distance, 2), (spike_distance, 3)])
def test_duplicates_and_bounds_match_pyspike(x64, fn, col):
    a = _trains([c[0] for c in GOLDEN_DUPLICATES], 0., 20.)
    b = _trains([c[1] for c in GOLDEN_DUPLICATES], 0., 20.)
    np.testing.assert_allclose(np.asarray(fn(a, b)), [c[col] for c in GOLDEN_DUPLICATES],
                               rtol=1e-12, atol=1e-14)


def test_analytic_isi_values(x64):
    # Empty train = spikes at the bounds, ISI T; one spike at T/2: ISIs T/2 on both
    # halves -> I = 1/2 everywhere.
    a = _trains([[], [1., 5., 9.], []], 0., 10.)
    b = _trains([[5.], [1., 5., 9.], []], 0., 10.)
    np.testing.assert_allclose(np.asarray(isi_distance(a, b)), [0.5, 0., 0.], atol=1e-15)
    np.testing.assert_allclose(np.asarray(spike_distance(a, b))[1:], [0., 0.], atol=1e-15)


# ----------------------------------------------------------------------------
# conventions and edge cases
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('fn', [isi_distance, spike_distance])
def test_symmetric(fn):
    rng = np.random.default_rng(1)
    rows_a = [np.sort(rng.uniform(0, 50, rng.integers(0, 8))) for _ in range(6)]
    rows_b = [np.sort(rng.uniform(0, 50, rng.integers(0, 8))) for _ in range(6)]
    a, b = _trains(rows_a, 0., 50.), _trains(rows_b, 0., 50.)
    np.testing.assert_allclose(np.asarray(fn(a, b)), np.asarray(fn(b, a)), rtol=1e-5, atol=1e-7)
    d = np.asarray(fn(a, b))
    assert np.all((d >= 0) & (d <= 1))


@pytest.mark.parametrize('fn', [isi_distance, spike_distance])
def test_single_spike_at_t_stop_in_both_is_zero(fn):
    # PySpike returns nan here (0/0 on a zero-length interval); we return 0.
    a = _trains([[10.]], 0., 10.)
    assert float(fn(a, a)[0]) == 0.0


@pytest.mark.parametrize('fn, col', [(isi_distance, 5), (spike_distance, 6)])
def test_window_override_ignores_spikes_outside(x64, fn, col):
    name, ra, rb, t0, t1, *_ = GOLDEN[12]
    assert name == 'offset_window'
    # Recorded on [0, 30] with extra spikes outside the [10, 20] analysis window.
    a = _trains([[1.] + ra + [25.]], 0., 30.)
    b = _trains([[3., 9.9] + rb + [20.5, 29.]], 0., 30.)
    got = fn(a, b, t_start=t0, t_stop=t1)
    np.testing.assert_allclose(float(got[0]), GOLDEN[12][col], rtol=1e-12)


@pytest.mark.parametrize('fn', [isi_distance, spike_distance])
def test_default_window_spans_both_trains(fn):
    # PySpike's reconcile_spike_trains: [min(t_start), max(t_stop)].
    a = _trains([[2., 6.]], 0., 8.)
    b = _trains([[3., 9.]], 1., 10.)
    np.testing.assert_allclose(np.asarray(fn(a, b)), np.asarray(fn(a, b, t_start=0., t_stop=10.)))


@pytest.mark.parametrize('fn', [isi_distance, spike_distance])
def test_empty_window_is_nan(fn):
    a = _trains([[1.]], 0., 10.)
    assert np.isnan(float(fn(a, a, t_start=5., t_stop=5.)[0]))
    assert np.isnan(float(fn(a, a, t_start=6., t_stop=5.)[0]))


@pytest.mark.parametrize('fn', [isi_distance, spike_distance])
def test_capacity_one_and_all_silent(fn):
    a = SpikeTrains.from_dense(jnp.zeros((10, 3)), dt=1.)
    assert a.capacity == 1
    np.testing.assert_array_equal(np.asarray(fn(a, a)), [0., 0., 0.])
    b = SpikeTrains.from_dense(jnp.zeros((10, 3)).at[5, 1].set(1.), dt=1.)
    assert b.capacity == 1
    d = np.asarray(fn(a, b))
    assert d[0] == 0 and d[2] == 0 and d[1] > 0


@pytest.mark.parametrize('fn', [isi_distance, spike_distance])
def test_padding_values_are_ignored(fn):
    # Different capacities, and garbage beyond ``counts``, give the same result.
    a = _trains([[1., 4.]], 0., 10.)
    b = _trains([[2., 3., 8.]], 0., 10.)
    a_wide = SpikeTrains(jnp.array([[1., 4., 7., 7.5]]), jnp.array([2]), 0., 10.)
    np.testing.assert_allclose(np.asarray(fn(a_wide, b)), np.asarray(fn(a, b)), rtol=1e-6)


# ----------------------------------------------------------------------------
# units
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('fn', [isi_distance, spike_distance])
def test_units(fn):
    rows_a, rows_b = [[1., 4., 7.], [2.]], [[2., 5.], []]
    plain = fn(_trains(rows_a, 0., 10.), _trains(rows_b, 0., 10.))
    a_ms = SpikeTrains.from_list([np.asarray(r) * u.ms for r in rows_a], t_start=0. * u.ms, t_stop=10. * u.ms)
    b_s = SpikeTrains.from_list([np.asarray(r) * 1e-3 * u.second for r in rows_b if len(r)] + [np.array([]) * u.second],
                                t_start=0. * u.second, t_stop=0.01 * u.second)
    with_units = fn(a_ms, b_s)
    assert not isinstance(with_units, u.Quantity)
    np.testing.assert_allclose(np.asarray(with_units), np.asarray(plain), rtol=1e-5)
    # Window given as a Quantity in another compatible unit.
    np.testing.assert_allclose(np.asarray(fn(a_ms, b_s, t_start=0. * u.second, t_stop=0.01 * u.second)),
                               np.asarray(plain), rtol=1e-5)
    with pytest.raises(TypeError):
        fn(a_ms, _trains(rows_b, 0., 10.))


@pytest.mark.parametrize('fn', [isi_distance, spike_distance])
def test_plain_window_is_milliseconds_for_quantity_trains(fn):
    # Spec rule 3: a plain number is a time in milliseconds, converted to the
    # trains' unit (here seconds).
    rows_a, rows_b = [[1., 4., 7., 12.], [2.]], [[2., 5., 11.], [3.]]
    a_s = SpikeTrains.from_list([np.asarray(r) * 1e-3 * u.second for r in rows_a],
                                t_start=0. * u.second, t_stop=0.015 * u.second)
    b_s = SpikeTrains.from_list([np.asarray(r) * 1e-3 * u.second for r in rows_b],
                                t_start=0. * u.second, t_stop=0.015 * u.second)
    expected = fn(a_s, b_s, t_start=1. * u.ms, t_stop=10. * u.ms)
    np.testing.assert_allclose(np.asarray(fn(a_s, b_s, t_start=1., t_stop=10.)), np.asarray(expected), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(fn(a_s, b_s, t_start=0.001 * u.second, t_stop=10.)),
                               np.asarray(expected), rtol=1e-6)
    # Not the same as reading the plain numbers in seconds.
    assert not np.allclose(np.asarray(fn(a_s, b_s, t_start=1., t_stop=10.)),
                           np.asarray(fn(a_s, b_s, t_start=0. * u.second, t_stop=0.015 * u.second)))
    with pytest.raises(TypeError):
        fn(a_s, b_s, t_start=1. * u.mV)


@pytest.mark.parametrize('fn', [isi_distance, spike_distance])
def test_window_for_unitless_trains(fn):
    # Unitless trains are in milliseconds: plain numbers are used as they are,
    # and a Quantity window is converted to milliseconds.
    a = _trains([[1., 4., 7., 12.]], 0., 15.)
    b = _trains([[2., 5., 11.]], 0., 15.)
    plain = fn(a, b, t_start=1., t_stop=10.)
    np.testing.assert_allclose(np.asarray(fn(a, b, t_start=0.001 * u.second, t_stop=10. * u.ms)),
                               np.asarray(plain), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(jax.jit(fn)(a, b, 1., 10.)), np.asarray(plain), rtol=1e-6)


# ----------------------------------------------------------------------------
# errors
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('fn', [isi_distance, spike_distance])
def test_errors(fn):
    a = _trains([[1.], [2.]], 0., 10.)
    with pytest.raises(TypeError):
        fn(jnp.array([[1.]]), a)
    with pytest.raises(ValueError):
        fn(a, _trains([[1.]], 0., 10.))
    batched = jax.vmap(lambda s: SpikeTrains.from_dense(s, dt=1., capacity=3))(jnp.zeros((2, 5, 2)))
    with pytest.raises(ValueError):
        fn(batched, batched)
    with pytest.raises(ValueError):
        fn(a, a, t_start=jnp.zeros(2))


# ----------------------------------------------------------------------------
# transformations
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('fn', [isi_distance, spike_distance])
def test_jit(fn):
    rows_a = [c[1] for c in GOLDEN[1:6]]
    rows_b = [c[2] for c in GOLDEN[1:6]]
    a, b = _trains(rows_a, 0., 10.), _trains(rows_b, 0., 10.)
    np.testing.assert_allclose(np.asarray(jax.jit(fn)(a, b)), np.asarray(fn(a, b)), rtol=1e-6)
    window = jax.jit(lambda x, y, t0, t1: fn(x, y, t0, t1))
    np.testing.assert_allclose(np.asarray(window(a, b, 0., 10.)), np.asarray(fn(a, b, 0., 10.)), rtol=1e-6)


@pytest.mark.parametrize('fn', [isi_distance, spike_distance])
def test_vmap_over_trials(fn):
    key = jax.random.PRNGKey(0)
    rasters = jax.random.bernoulli(key, 0.1, (4, 60, 3))
    other = jax.random.bernoulli(jax.random.PRNGKey(1), 0.1, (4, 60, 3))

    def per_trial(x, y):
        return fn(SpikeTrains.from_dense(x, dt=1., capacity=60), SpikeTrains.from_dense(y, dt=1., capacity=60))

    batched = jax.vmap(per_trial)(rasters, other)
    assert batched.shape == (4, 3)
    for k in range(4):
        expected = fn(SpikeTrains.from_dense(rasters[k], dt=1.), SpikeTrains.from_dense(other[k], dt=1.))
        np.testing.assert_allclose(np.asarray(batched[k]), np.asarray(expected), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize('fn', [isi_distance, spike_distance])
def test_pair_matrix_via_vmap(fn):
    st = _trains([[1., 4., 7.], [2., 5.], [], [1., 4., 7.]], 0., 10.)
    n = st.n_neurons
    mat = np.asarray(jax.vmap(lambda i: fn(st.select(jnp.full((n,), i)), st))(jnp.arange(n)))
    assert mat.shape == (n, n)
    np.testing.assert_allclose(np.diag(mat), 0., atol=1e-7)
    np.testing.assert_allclose(mat, mat.T, rtol=1e-6, atol=1e-7)
    assert mat[0, 3] == 0.
    pair = fn(st.select(1), st.select(2))
    np.testing.assert_allclose(mat[1, 2], float(pair[0]), rtol=1e-6)
