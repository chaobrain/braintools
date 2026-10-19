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

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.spike import SpikeTrains


def _raster(seed=0, shape=(50, 4), p=0.2):
    rng = np.random.RandomState(seed)
    return rng.rand(*shape) < p


def _mantissa(x):
    return np.asarray(x.mantissa if isinstance(x, u.Quantity) else x)


# ----------------------------------------------------------------------------
# from_dense
# ----------------------------------------------------------------------------

class TestFromDense:
    def test_basic_layout(self):
        spikes = jnp.array([[1, 0], [0, 0], [1, 1], [1, 0]])
        st = SpikeTrains.from_dense(spikes, dt=1.)
        assert st.n_neurons == 2 and st.capacity == 3
        np.testing.assert_array_equal(st.counts, [3, 1])
        assert st.counts.dtype == jnp.int32
        np.testing.assert_array_equal(st.times, [[0., 2., 3.], [2., np.inf, np.inf]])
        assert float(st.t_start) == 0. and float(st.t_stop) == 4.
        assert st.overflow.dtype == jnp.bool_ and not bool(st.overflow)
        np.testing.assert_array_equal(st.mask, [[True, True, True], [True, False, False]])
        assert float(st.duration) == 4.

    def test_all_zero_defaults_capacity_one(self):
        st = SpikeTrains.from_dense(jnp.zeros((20, 3)), dt=0.1)
        assert st.capacity == 1
        np.testing.assert_array_equal(st.counts, [0, 0, 0])
        assert np.all(np.isposinf(np.asarray(st.times)))
        assert not bool(st.overflow)
        st.check()

    def test_single_time_step(self):
        st = SpikeTrains.from_dense(jnp.array([[1, 0, 1]]), dt=0.5, t_start=2.)
        assert st.capacity == 1
        np.testing.assert_array_equal(st.counts, [1, 0, 1])
        np.testing.assert_array_equal(st.times[:, 0], [2., np.inf, 2.])
        assert float(st.t_stop) == 2.5
        np.testing.assert_array_equal(st.to_dense(dt=0.5), [[True, False, True]])

    def test_numeric_nonzero_is_one_spike(self):
        st = SpikeTrains.from_dense(jnp.array([[2.], [0.], [-1.]]), dt=1.)
        np.testing.assert_array_equal(st.times, [[0., 2.]])

    def test_explicit_capacity_pads_with_inf(self):
        st = SpikeTrains.from_dense(jnp.array([[1], [1]]), dt=1., capacity=5)
        assert st.capacity == 5
        np.testing.assert_array_equal(st.times, [[0., 1., np.inf, np.inf, np.inf]])

    def test_capacity_smaller_than_max_count_raises_eagerly(self):
        with pytest.raises(ValueError, match='neuron 0 has 3 spikes'):
            SpikeTrains.from_dense(jnp.ones((3, 2)).at[:, 1].set(0), dt=1., capacity=2)

    def test_capacity_smaller_than_max_count_sets_flag_when_traced(self):
        spikes = jnp.ones((5, 2)).at[:, 1].set(0).at[0, 1].set(1)
        st = jax.jit(lambda s: SpikeTrains.from_dense(s, dt=1., capacity=2))(spikes)
        assert bool(st.overflow)
        np.testing.assert_array_equal(st.counts, [2, 1])  # clipped to the capacity
        np.testing.assert_array_equal(st.times, [[0., 1.], [0., np.inf]])  # first K kept
        with pytest.raises(ValueError, match='overflow'):
            st.check()

    def test_traced_without_capacity_raises(self):
        with pytest.raises(ValueError, match='capacity'):
            jax.jit(lambda s: SpikeTrains.from_dense(s, dt=1.))(jnp.ones((3, 2)))

    @pytest.mark.parametrize('bad, err', [(0, ValueError), (2.5, TypeError), (True, TypeError)])
    def test_invalid_capacity(self, bad, err):
        with pytest.raises(err):
            SpikeTrains.from_dense(jnp.ones((3, 2)), dt=1., capacity=bad)

    def test_quantity_dt_matches_float_dt(self):
        spikes = _raster()
        st_q = SpikeTrains.from_dense(spikes, dt=0.1 * u.ms)
        st_f = SpikeTrains.from_dense(spikes, dt=0.1)
        assert isinstance(st_q.times, u.Quantity) and st_q.times.unit == u.ms
        assert isinstance(st_q.t_start, u.Quantity) and isinstance(st_q.t_stop, u.Quantity)
        assert not isinstance(st_f.times, u.Quantity)
        np.testing.assert_array_equal(_mantissa(st_q.times), np.asarray(st_f.times))
        np.testing.assert_array_equal(st_q.counts, st_f.counts)
        assert u.math.allclose(st_q.t_stop, 5. * u.ms)
        assert u.math.allclose(st_q.duration, 5. * u.ms)

    def test_quantity_t_start_converted_to_dt_unit(self):
        st = SpikeTrains.from_dense(jnp.array([[1], [1]]), dt=1. * u.ms, t_start=0.01 * u.second)
        assert st.times.unit == u.ms
        np.testing.assert_allclose(_mantissa(st.times), [[10., 11.]])
        np.testing.assert_allclose(_mantissa(st.t_stop), 12.)

    def test_plain_nonzero_t_start_with_quantity_dt_raises(self):
        with pytest.raises(TypeError, match='t_start'):
            SpikeTrains.from_dense(jnp.ones((2, 1)), dt=1. * u.ms, t_start=5.)

    def test_quantity_t_start_with_plain_dt_raises(self):
        with pytest.raises(TypeError, match='t_start'):
            SpikeTrains.from_dense(jnp.ones((2, 1)), dt=1., t_start=5. * u.ms)

    def test_incompatible_units_raise(self):
        with pytest.raises(u.UnitMismatchError):
            SpikeTrains.from_dense(jnp.ones((2, 1)), dt=1. * u.ms, t_start=1. * u.mV)

    def test_dt_defaults_to_environ(self):
        with brainstate.environ.context(dt=0.25 * u.ms):
            st = SpikeTrains.from_dense(jnp.array([[0], [1]]))
        assert u.math.allclose(st.times[0, 0], 0.25 * u.ms)

    def test_missing_environ_dt_raises(self, monkeypatch):
        def no_dt():
            raise KeyError('dt')

        monkeypatch.setattr(brainstate.environ, 'get_dt', no_dt)
        with pytest.raises(ValueError, match='dt'):
            SpikeTrains.from_dense(jnp.ones((2, 1)))
        st = SpikeTrains.from_dense(jnp.ones((2, 1)), dt=1.)
        with pytest.raises(ValueError, match='dt'):
            st.to_dense()

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match=r'\(T, N\)'):
            SpikeTrains.from_dense(jnp.ones(5), dt=1.)

    def test_empty_shapes(self):
        st = SpikeTrains.from_dense(jnp.zeros((0, 3)), dt=1.)
        assert st.times.shape == (3, 1) and float(st.duration) == 0.
        st = SpikeTrains.from_dense(jnp.zeros((4, 0)), dt=1.)
        assert st.times.shape == (0, 1)
        assert st.to_events()[0].shape == (0,)

    def test_matches_numpy_reference(self):
        spikes = _raster(seed=3, shape=(200, 7), p=0.3)
        st = SpikeTrains.from_dense(spikes, dt=0.1)
        for n, train in enumerate(st.to_list()):
            np.testing.assert_allclose(train, np.nonzero(spikes[:, n])[0] * 0.1, rtol=1e-6)

    def test_vmap_over_trials_with_fixed_capacity(self):
        trials = np.stack([_raster(seed=s, shape=(40, 3), p=0.25) for s in range(6)])
        make = lambda s: SpikeTrains.from_dense(s, dt=0.5 * u.ms, capacity=40)
        batched = jax.vmap(make)(jnp.asarray(trials))
        assert isinstance(batched, SpikeTrains)
        assert batched.times.shape == (6, 3, 40) and batched.counts.shape == (6, 3)
        assert batched.times.unit == u.ms
        for i in range(6):
            single = SpikeTrains.from_dense(trials[i], dt=0.5 * u.ms, capacity=40)
            np.testing.assert_array_equal(_mantissa(batched.times[i]), _mantissa(single.times))
            np.testing.assert_array_equal(batched.counts[i], single.counts)
        assert not bool(jnp.any(batched.overflow))

    def test_vmap_methods_per_trial(self):
        trials = jnp.asarray(np.stack([_raster(seed=s, shape=(40, 3)) for s in range(4)]))

        def per_trial(s):
            st = SpikeTrains.from_dense(s, dt=1., capacity=40)
            return st.bin(10., n_bins=4), st.window(5., 25.).counts, st.to_dense(dt=1., n_steps=40)

        binned, win_counts, dense = jax.vmap(per_trial)(trials)
        assert binned.shape == (4, 4, 3)
        np.testing.assert_array_equal(binned.sum(axis=1), trials.sum(axis=1))
        np.testing.assert_array_equal(win_counts, trials[:, 5:25].sum(axis=1))
        np.testing.assert_array_equal(dense, trials.astype(bool))


# ----------------------------------------------------------------------------
# from_events / from_list
# ----------------------------------------------------------------------------

class TestFromEvents:
    def test_unsorted_events_and_default_bounds(self):
        st = SpikeTrains.from_events(jnp.array([3., 1., 2., 0.5]), jnp.array([0, 0, 1, 2]), n_neurons=4)
        np.testing.assert_array_equal(st.counts, [2, 1, 1, 0])
        np.testing.assert_array_equal(st.times, [[1., 3.], [2., np.inf], [0.5, np.inf], [np.inf, np.inf]])
        assert float(st.t_start) == 0.5 and float(st.t_stop) == 3.
        st.check()

    def test_quantity_times_and_bounds_in_other_unit(self):
        st = SpikeTrains.from_events(jnp.array([1., 2.]) * u.ms, jnp.array([0, 0]), n_neurons=1,
                                     t_start=0. * u.second, t_stop=0.01 * u.second)
        assert st.times.unit == u.ms
        np.testing.assert_allclose(_mantissa(st.t_stop), 10.)

    def test_no_events(self):
        st = SpikeTrains.from_events(jnp.zeros(0), jnp.zeros(0, dtype=int), n_neurons=2)
        assert st.capacity == 1 and float(st.t_start) == 0. and float(st.t_stop) == 0.
        np.testing.assert_array_equal(st.counts, [0, 0])

    def test_out_of_range_ids_raise_eagerly(self):
        with pytest.raises(ValueError, match='neuron_ids'):
            SpikeTrains.from_events(jnp.array([1., 2.]), jnp.array([0, 2]), n_neurons=2)
        with pytest.raises(ValueError, match='neuron_ids'):
            SpikeTrains.from_events(jnp.array([1., 2.]), jnp.array([-1, 0]), n_neurons=2)

    def test_times_outside_bounds_raise_eagerly(self):
        with pytest.raises(ValueError, match='outside'):
            SpikeTrains.from_events(jnp.array([1., 5.]), jnp.array([0, 0]), n_neurons=1,
                                    t_start=0., t_stop=4.)

    def test_traced_invalid_events_are_dropped(self):
        f = jax.jit(lambda t, i: SpikeTrains.from_events(t, i, 2, t_start=0., t_stop=4., capacity=3))
        st = f(jnp.array([1., 5., 2., 3., 0.5]), jnp.array([0, 0, -1, 1, 7]))
        np.testing.assert_array_equal(st.counts, [1, 1])
        np.testing.assert_array_equal(st.times, [[1., np.inf, np.inf], [3., np.inf, np.inf]])
        assert not bool(st.overflow)

    def test_traced_overflow(self):
        f = jax.jit(lambda t, i: SpikeTrains.from_events(t, i, 1, t_start=0., t_stop=9., capacity=2))
        st = f(jnp.array([3., 1., 2.]), jnp.array([0, 0, 0]))
        assert bool(st.overflow)
        np.testing.assert_array_equal(st.times, [[1., 2.]])

    def test_eager_overflow_raises(self):
        with pytest.raises(ValueError, match='capacity=1'):
            SpikeTrains.from_events(jnp.array([1., 2.]), jnp.array([0, 0]), 1, capacity=1)

    def test_traced_requires_bounds_and_capacity(self):
        times, ids = jnp.array([1., 2.]), jnp.array([0, 1])
        with pytest.raises(ValueError, match='t_start'):
            jax.jit(lambda t: SpikeTrains.from_events(t, ids, 2, capacity=2))(times)
        with pytest.raises(ValueError, match='capacity'):
            jax.jit(lambda t: SpikeTrains.from_events(t, ids, 2, t_start=0., t_stop=3.))(times)

    def test_shape_and_dtype_validation(self):
        with pytest.raises(ValueError, match='entries'):
            SpikeTrains.from_events(jnp.array([1., 2.]), jnp.array([0]), n_neurons=1)
        with pytest.raises(TypeError, match='integers'):
            SpikeTrains.from_events(jnp.array([1.]), jnp.array([0.5]), n_neurons=1)
        with pytest.raises(TypeError, match='static'):
            SpikeTrains.from_events(jnp.array([1.]), jnp.array([0]), n_neurons=1.5)

    def test_unit_mismatch_between_times_and_bounds(self):
        with pytest.raises(TypeError, match='t_stop'):
            SpikeTrains.from_events(jnp.array([1.]) * u.ms, jnp.array([0]), 1, t_stop=3.)
        with pytest.raises(TypeError, match='t_start'):
            SpikeTrains.from_events(jnp.array([1.]), jnp.array([0]), 1, t_start=0. * u.ms)


class TestFromList:
    def test_empty_trains(self):
        st = SpikeTrains.from_list([np.array([0.5, 0.1]), np.array([]), [0.3], []], t_start=0., t_stop=1.)
        np.testing.assert_array_equal(st.counts, [2, 0, 1, 0])
        trains = st.to_list()
        assert len(trains) == 4 and len(trains[1]) == 0 and len(trains[3]) == 0
        np.testing.assert_allclose(trains[0], [0.1, 0.5], rtol=1e-6)

    def test_all_empty_and_no_trains(self):
        st = SpikeTrains.from_list([[], []])
        assert st.times.shape == (2, 1) and float(st.duration) == 0.
        st = SpikeTrains.from_list([])
        assert st.n_neurons == 0 and st.to_list() == []

    def test_quantity_trains_in_mixed_compatible_units(self):
        st = SpikeTrains.from_list([jnp.array([1., 2.]) * u.ms, jnp.array([0.003]) * u.second, []])
        assert st.times.unit == u.ms
        np.testing.assert_allclose(_mantissa(st.times), [[1., 2.], [3., np.inf], [np.inf, np.inf]])
        assert all(isinstance(t, u.Quantity) for t in st.to_list())

    def test_mixing_plain_and_quantity_raises(self):
        with pytest.raises(TypeError, match='trains\\[1\\]'):
            SpikeTrains.from_list([jnp.array([1.]) * u.ms, np.array([2.])])

    def test_capacity_argument(self):
        st = SpikeTrains.from_list([[0.1], [0.2, 0.3]], capacity=4)
        assert st.capacity == 4

    def test_scalar_train_is_one_spike(self):
        st = SpikeTrains.from_list([0.5, [0.1, 0.2]])
        np.testing.assert_array_equal(st.counts, [1, 2])


# ----------------------------------------------------------------------------
# converters
# ----------------------------------------------------------------------------

class TestConverters:
    def test_dense_events_dense_round_trip(self):
        spikes = _raster(seed=1, shape=(100, 6), p=0.15)
        st = SpikeTrains.from_dense(spikes, dt=0.1 * u.ms)
        times, ids = st.to_events()
        assert isinstance(times, u.Quantity)
        assert np.all(np.diff(_mantissa(times)) >= 0)
        back = SpikeTrains.from_events(times, ids, st.n_neurons, t_start=st.t_start, t_stop=st.t_stop)
        np.testing.assert_array_equal(back.to_dense(dt=0.1 * u.ms), spikes)
        np.testing.assert_array_equal(st.to_dense(dt=0.1 * u.ms), spikes)

    def test_dense_list_dense_round_trip(self):
        spikes = _raster(seed=2, shape=(30, 5), p=0.3)
        spikes[:, 2] = False  # an empty train
        st = SpikeTrains.from_dense(spikes, dt=1.)
        back = SpikeTrains.from_list(st.to_list(), t_start=st.t_start, t_stop=st.t_stop)
        np.testing.assert_array_equal(back.to_dense(dt=1.), spikes)

    def test_to_events_orders_ties_by_neuron(self):
        st = SpikeTrains.from_dense(jnp.array([[0, 1, 1], [1, 0, 1]]), dt=1.)
        times, ids = st.to_events()
        np.testing.assert_array_equal(times, [0., 0., 1., 1.])
        np.testing.assert_array_equal(ids, [1, 2, 0, 2])

    def test_to_dense_spike_at_t_stop_and_merges(self):
        st = SpikeTrains.from_events(jnp.array([0., 0.9, 2.]), jnp.array([0, 0, 1]), 2)
        # Default T grows by one row to keep the spike at t_stop=2.
        dense = st.to_dense(dt=1.)
        np.testing.assert_array_equal(dense, [[True, False], [True, False], [False, True]])
        # An explicit n_steps is honoured: the spike on row 2 is dropped.
        dense = st.to_dense(dt=1., n_steps=2)
        np.testing.assert_array_equal(dense, [[True, False], [True, False]])

    def test_to_dense_under_jit_needs_n_steps(self):
        st = SpikeTrains.from_dense(jnp.eye(3), dt=1.)
        with pytest.raises(ValueError, match='n_steps'):
            jax.jit(lambda s: s.to_dense(dt=1.))(st)
        out = jax.jit(lambda s: s.to_dense(dt=1., n_steps=3))(st)
        np.testing.assert_array_equal(out, np.eye(3, dtype=bool))

    def test_to_dense_dt_unit_conversion(self):
        st = SpikeTrains.from_dense(jnp.eye(2), dt=1. * u.ms)
        dense = st.to_dense(dt=0.0005 * u.second)
        assert dense.shape == (4, 2)
        np.testing.assert_array_equal(np.nonzero(np.asarray(dense))[0], [0, 2])
        with pytest.raises(TypeError, match='dt'):
            st.to_dense(dt=1.)

    def test_host_only_methods_raise_under_jit(self):
        st = SpikeTrains.from_dense(jnp.eye(2), dt=1.)
        with pytest.raises(TypeError, match='host-only'):
            jax.jit(lambda s: s.to_events()[1])(st)
        with pytest.raises(TypeError, match='host-only'):
            jax.jit(lambda s: s.to_list()[0])(st)


# ----------------------------------------------------------------------------
# bin / window / select
# ----------------------------------------------------------------------------

class TestBin:
    def test_non_divisible_bin_has_partial_last_bin(self):
        st = SpikeTrains.from_dense(jnp.ones((10, 2)).at[:, 1].set(0).at[9, 1].set(1), dt=1.)
        counts = st.bin(3.)
        assert counts.shape == (4, 2) and counts.dtype == jnp.int32
        np.testing.assert_array_equal(counts[:, 0], [3, 3, 3, 1])
        np.testing.assert_array_equal(counts[:, 1], [0, 0, 0, 1])

    def test_grid_times_bin_exactly(self):
        spikes = _raster(seed=4, shape=(1000, 3), p=0.1)
        st = SpikeTrains.from_dense(spikes, dt=0.1)
        counts = st.bin(0.5)
        assert counts.shape == (200, 3)
        np.testing.assert_array_equal(counts, spikes.reshape(200, 5, 3).sum(axis=1))

    def test_spike_at_t_stop_counts_in_last_bin(self):
        st = SpikeTrains.from_events(jnp.array([0., 4., 6.]), jnp.array([0, 0, 0]), 1, t_start=0., t_stop=6.)
        np.testing.assert_array_equal(st.bin(3.)[:, 0], [1, 2])

    def test_quantity_bin_size(self):
        st = SpikeTrains.from_dense(jnp.ones((4, 1)), dt=1. * u.ms)
        np.testing.assert_array_equal(st.bin(0.002 * u.second)[:, 0], [2, 2])
        with pytest.raises(TypeError, match='bin_size'):
            st.bin(2.)
        unitless = SpikeTrains.from_dense(jnp.ones((4, 1)), dt=1.)
        with pytest.raises(TypeError, match='bin_size'):
            unitless.bin(2. * u.ms)
        np.testing.assert_array_equal(unitless.bin(u.Quantity(2.))[:, 0], [2, 2])

    def test_invalid_bin_size(self):
        st = SpikeTrains.from_dense(jnp.ones((4, 1)), dt=1.)
        with pytest.raises(ValueError, match='positive'):
            st.bin(0.)

    def test_bin_under_jit(self):
        st = SpikeTrains.from_dense(jnp.ones((4, 1)), dt=1.)
        with pytest.raises(ValueError, match='n_bins'):
            jax.jit(lambda s: s.bin(2.))(st)
        out = jax.jit(lambda s: s.bin(2., n_bins=2))(st)
        np.testing.assert_array_equal(out[:, 0], [2, 2])
        # concrete (closed-over) trains do not need n_bins under jit
        np.testing.assert_array_equal(jax.jit(lambda: st.bin(2.))()[:, 0], [2, 2])

    def test_zero_duration_has_no_bins(self):
        st = SpikeTrains.from_dense(jnp.zeros((0, 2)), dt=1.)
        assert st.bin(1.).shape == (0, 2)


class TestWindow:
    def test_boundaries_include_t0_exclude_t1(self):
        st = SpikeTrains.from_dense(jnp.ones((6, 2)).at[:, 1].set(0), dt=1.)
        w = st.window(2., 4.)
        assert w.capacity == st.capacity
        np.testing.assert_array_equal(w.counts, [2, 0])
        np.testing.assert_array_equal(w.times[0], [2., 3., np.inf, np.inf, np.inf, np.inf])
        assert float(w.t_start) == 2. and float(w.t_stop) == 4.
        w.check()

    def test_bounds_are_clipped_to_recording(self):
        st = SpikeTrains.from_dense(jnp.ones((6, 1)), dt=1., t_start=1.)
        w = st.window(-5., 100.)
        assert float(w.t_start) == 1. and float(w.t_stop) == 7.
        np.testing.assert_array_equal(w.counts, [6])
        empty = st.window(50., 60.)
        np.testing.assert_array_equal(empty.counts, [0])
        assert float(empty.duration) == 0.
        empty.check()

    def test_quantity_window_under_jit(self):
        st = SpikeTrains.from_dense(jnp.ones((10, 1)), dt=1. * u.ms)
        w = jax.jit(lambda s: s.window(0.002 * u.second, 5. * u.ms))(st)
        assert w.times.unit == u.ms
        np.testing.assert_allclose(_mantissa(w.times)[0, :3], [2., 3., 4.])
        np.testing.assert_array_equal(w.counts, [3])
        with pytest.raises(TypeError, match='t0'):
            st.window(1., 5. * u.ms)

    def test_overflow_flag_is_carried(self):
        st = jax.jit(lambda s: SpikeTrains.from_dense(s, dt=1., capacity=1))(jnp.ones((3, 1)))
        assert bool(st.window(0., 1.).overflow)


class TestSelect:
    def test_select_forms(self):
        st = SpikeTrains.from_dense(jnp.eye(3), dt=1.)
        np.testing.assert_array_equal(st.select([2, 0]).times[:, 0], [2., 0.])
        assert st.select(1).times.shape == (1, 1)
        assert st.select(np.int64(1)).n_neurons == 1
        np.testing.assert_array_equal(st.select(slice(1, None)).counts, [1, 1])
        np.testing.assert_array_equal(st.select(np.array([True, False, True])).times[:, 0], [0., 2.])

    def test_select_keeps_units_and_bounds_under_jit(self):
        st = SpikeTrains.from_dense(jnp.eye(3), dt=1. * u.ms)
        sub = jax.jit(lambda s: s.select(jnp.array([1])))(st)
        assert sub.times.unit == u.ms
        assert u.math.allclose(sub.t_stop, 3. * u.ms)


# ----------------------------------------------------------------------------
# check / init / pytree / repr
# ----------------------------------------------------------------------------

class TestCheckAndInit:
    def test_init_normalizes_fields(self):
        st = SpikeTrains([[1, 2]], [2], 0, 3)
        assert st.times.dtype == jnp.result_type(float)
        assert st.counts.dtype == jnp.int32 and st.overflow.dtype == jnp.bool_
        assert st.check() is st

    def test_init_with_units(self):
        st = SpikeTrains(jnp.array([[1., np.inf]]) * u.ms, [1], 0. * u.second, 0.002 * u.second)
        assert u.math.allclose(st.t_stop, 2. * u.ms)
        with pytest.raises(TypeError, match='t_start'):
            SpikeTrains(jnp.array([[1.]]) * u.ms, [1], 0., 2. * u.ms)

    @pytest.mark.parametrize('kwargs, match', [
        (dict(times=[1., 2.], counts=[2]), 'n_neurons, capacity'),
        (dict(times=[[1., 2.]], counts=[2, 0]), 'counts'),
        (dict(times=[[1., 2.]], counts=[2], t_start=[0., 1.]), 'scalars'),
    ])
    def test_init_shape_validation(self, kwargs, match):
        kwargs = {'t_start': 0., 't_stop': 5., **kwargs}
        with pytest.raises(ValueError, match=match):
            SpikeTrains(**kwargs)

    @pytest.mark.parametrize('times, counts, t0, t1, match', [
        ([[1., 2.]], [3], 0., 5., 'counts'),
        ([[1., 2.]], [2], 6., 5., 't_start'),
        ([[1., 2.]], [1], 0., 5., 'padding'),
        ([[1., 9.]], [2], 0., 5., r'\[t_start, t_stop\]'),
        ([[2., 1.]], [2], 0., 5., 'sorted'),
    ])
    def test_check_invariants(self, times, counts, t0, t1, match):
        with pytest.raises(ValueError, match=match):
            SpikeTrains(times, counts, t0, t1).check()

    def test_check_under_jit_does_not_raise(self):
        st = SpikeTrains([[1., 2.]], [2], 0., 5., overflow=True)
        out = jax.jit(lambda s: s.check().counts)(st)
        np.testing.assert_array_equal(out, [2])


class TestPytree:
    def test_flatten_unflatten(self):
        st = SpikeTrains.from_dense(_raster(), dt=0.1 * u.ms)
        children, aux = st.tree_flatten()
        assert len(children) == 5 and aux is None
        leaves, treedef = jax.tree.flatten(st)
        assert len(leaves) == 5  # Quantities flatten to their mantissas
        back = jax.tree.unflatten(treedef, leaves)
        assert isinstance(back, SpikeTrains) and back.times.unit == u.ms
        np.testing.assert_array_equal(_mantissa(back.times), _mantissa(st.times))
        np.testing.assert_array_equal(back.counts, st.counts)

    def test_unflatten_accepts_arbitrary_leaves(self):
        treedef = jax.tree.structure(SpikeTrains.from_dense(jnp.eye(2), dt=1.))
        obj = jax.tree.unflatten(treedef, [None] * 5)
        assert obj.times is None

    def test_tree_map_and_jit_identity(self):
        st = SpikeTrains.from_dense(_raster(), dt=1.)
        same = jax.jit(lambda s: s)(st)
        np.testing.assert_array_equal(same.times, st.times)
        shapes = jax.tree.map(lambda x: x.shape, st)
        assert shapes.times == st.times.shape and shapes.overflow == ()

    def test_jit_of_properties(self):
        st = SpikeTrains.from_dense(jnp.eye(3), dt=1. * u.ms)
        mask, duration = jax.jit(lambda s: (s.mask, s.duration))(st)
        np.testing.assert_array_equal(mask, np.ones((3, 1), bool))
        assert u.math.allclose(duration, 3. * u.ms)


class TestRepr:
    def test_concrete_repr(self):
        text = repr(SpikeTrains.from_dense(jnp.eye(3), dt=1. * u.ms))
        assert text.startswith('SpikeTrains(n_neurons=3, capacity=1, n_spikes=3')
        assert 'overflow=False' in text and 'ms' in text

    def test_traced_repr(self):
        seen = []

        def f(s):
            seen.append(repr(s))
            return s.counts

        jax.jit(f)(SpikeTrains.from_dense(jnp.eye(2), dt=1.))
        assert seen == ['SpikeTrains(n_neurons=2, capacity=1, traced)']


class TestReviewRegressions:
    """Bugs found in the phase-0.4 review; each test failed before its fix."""

    def test_bin_explicit_n_bins_drops_spikes_past_last_bin(self):
        st = SpikeTrains.from_dense(jnp.ones((5, 1)), dt=1.)
        np.testing.assert_array_equal(st.bin(1., n_bins=2)[:, 0], [1, 1])
        np.testing.assert_array_equal(st.bin(1., n_bins=3)[:, 0], [1, 1, 1])
        jitted = jax.jit(lambda s: s.bin(1., n_bins=2))(st)
        np.testing.assert_array_equal(jitted[:, 0], [1, 1])

    def test_bin_spike_at_t_stop_still_closes_last_bin(self):
        st = SpikeTrains.from_events(jnp.array([0.5, 4.0]), jnp.array([0, 0]), 1,
                                     t_start=0., t_stop=4.)
        np.testing.assert_array_equal(st.bin(1.)[:, 0], [1, 0, 0, 1])
        np.testing.assert_array_equal(st.bin(1., n_bins=4)[:, 0], [1, 0, 0, 1])

    def test_bin_tolerance_does_not_grow_with_time(self):
        st = SpikeTrains.from_events(jnp.array([9999.99]), jnp.array([0]), 1,
                                     t_start=0., t_stop=10001.)
        counts = st.bin(1.)[:, 0]
        assert int(counts[9999]) == 1 and int(counts[10000]) == 0

    def test_bin_on_grid_spikes_stay_in_their_bin(self):
        # Times t0 + i*dt computed in float32 can round just below the edge.
        st = SpikeTrains.from_dense(jnp.ones((1000, 1)), dt=0.1, t_start=3.7)
        np.testing.assert_array_equal(st.bin(0.1)[:, 0], np.ones(1000))

    def test_events_round_trip_keeps_last_spike(self):
        sp = jnp.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=bool)
        t, i = SpikeTrains.from_dense(sp, dt=1.).to_events()
        back = SpikeTrains.from_events(t, i, 2).to_dense(dt=1.)
        np.testing.assert_array_equal(back, sp)

    def test_window_past_t_stop_keeps_spike_at_t_stop(self):
        st = SpikeTrains.from_events(jnp.array([1., 3.]), jnp.array([0, 0]), 1)
        assert int(st.window(0., 10.).counts[0]) == 2
        assert int(SpikeTrains.from_list([[0.5, 2.0]]).window(0, 100).counts[0]) == 2
        # A window ending exactly at t_stop is still half-open.
        assert int(st.window(0., 3.).counts[0]) == 1

    def test_zero_length_recording_with_spike(self):
        st = SpikeTrains.from_list([[0.5]])
        np.testing.assert_array_equal(st.bin(1.), [[1]])
        np.testing.assert_array_equal(st.to_dense(dt=1.), [[True]])

    def test_select_zero_d_array_index(self):
        st = SpikeTrains.from_dense(jnp.eye(3), dt=1.)
        for idx in (jnp.array(1), np.array(1), np.int64(1), 1):
            sel = st.select(idx)
            assert sel.n_neurons == 1
            np.testing.assert_array_equal(sel.times, [[1.]])
