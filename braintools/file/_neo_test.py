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

"""Tests for ``braintools.file._neo``.

``neo`` and ``quantities`` are optional; the conversion tests are skipped when
they are missing (run them in the reference environment), while the
missing-package tests always run.
"""

import sys

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.file import _neo as N
from braintools.file._neo import from_neo, to_neo
from braintools.spike._spike_trains import SpikeTrains


@pytest.fixture
def neo():
    return pytest.importorskip('neo')


@pytest.fixture
def pq():
    return pytest.importorskip('quantities')


def _trains_ms():
    return SpikeTrains.from_list(
        [np.array([1., 2.5, 7.]) * u.ms, np.array([]) * u.ms, np.array([0.5]) * u.ms],
        t_start=-2. * u.ms, t_stop=10. * u.ms,
    )


class TestToNeo:
    def test_values_units_and_bounds(self, neo, pq):
        st = _trains_ms()
        out = to_neo(st)
        assert len(out) == 3
        assert all(isinstance(t, neo.SpikeTrain) for t in out)
        assert out[0].units.dimensionality.string == 'ms'
        np.testing.assert_allclose(out[0].magnitude, [1., 2.5, 7.])
        assert out[1].size == 0  # neuron with zero spikes
        np.testing.assert_allclose(out[2].magnitude, [0.5])
        for t in out:
            assert float(t.t_start.rescale(pq.ms)) == -2.
            assert float(t.t_stop.rescale(pq.ms)) == 10.
            assert N._OVERFLOW_ANNOTATION not in t.annotations

    def test_seconds_and_microseconds(self, neo, pq):
        for unit, name in [(u.second, 's'), (u.us, 'us')]:
            st = SpikeTrains.from_list([np.array([1., 2.]) * unit], t_start=0. * unit,
                                       t_stop=3. * unit)
            out = to_neo(st)
            assert out[0].units.dimensionality.string == name
            np.testing.assert_allclose(out[0].magnitude, [1., 2.])

    def test_unit_without_quantities_name_uses_seconds(self, neo, pq):
        odd = u.Unit(dim=u.second.dim, scale=-3, factor=2.5)  # 2.5 ms, no pq name
        st = SpikeTrains.from_list([np.array([1., 2.]) * odd], t_start=0. * odd,
                                   t_stop=4. * odd)
        out = to_neo(st)
        assert out[0].units.dimensionality.string == 's'
        np.testing.assert_allclose(out[0].magnitude, [0.0025, 0.005], rtol=1e-6)
        np.testing.assert_allclose(float(out[0].t_stop), 0.01, rtol=1e-6)

    def test_unit_name_unknown_to_quantities_uses_seconds(self, neo, pq):
        st = SpikeTrains.from_list([np.array([1.]) * u.week], t_start=0. * u.week,
                                   t_stop=2. * u.week)  # brainunit 'wk' is not a pq name
        out = to_neo(st)
        assert out[0].units.dimensionality.string == 's'
        np.testing.assert_allclose(out[0].magnitude, [7 * 24 * 3600.], rtol=1e-6)

    def test_overflow_annotated(self, neo, pq):
        st = jax.jit(lambda s: SpikeTrains.from_dense(s, dt=1. * u.ms, capacity=1))(jnp.ones((3, 2)))
        assert bool(st.overflow)
        out = to_neo(st)
        assert all(t.annotations[N._OVERFLOW_ANNOTATION] is True for t in out)

    def test_zero_neurons(self, neo, pq):
        st = SpikeTrains(jnp.zeros((0, 1)) * u.ms, jnp.zeros(0), 0. * u.ms, 1. * u.ms)
        assert to_neo(st) == []

    def test_unitless_rejected(self, neo, pq):
        st = SpikeTrains.from_dense(jnp.ones((2, 1)), dt=1.)
        with pytest.raises(TypeError, match='no time unit'):
            to_neo(st)

    def test_non_time_unit_rejected(self, neo, pq):
        st = SpikeTrains(jnp.array([[1.]]) * u.mV, jnp.array([1]), 0. * u.mV, 2. * u.mV)
        with pytest.raises(TypeError, match='not a time'):
            to_neo(st)

    def test_batched_rejected(self, neo, pq):
        st = jax.vmap(lambda s: SpikeTrains.from_dense(s, dt=1. * u.ms, capacity=3))(
            jnp.ones((2, 3, 1)))
        with pytest.raises(ValueError, match='unbatched'):
            to_neo(st)

    def test_not_spike_trains(self, neo, pq):
        with pytest.raises(TypeError, match='expected a SpikeTrains'):
            to_neo([np.array([1.])])

    def test_traced_rejected(self, neo, pq):
        @jax.jit
        def f(times):
            st = SpikeTrains(times * u.ms, jnp.array([1]), 0. * u.ms, 5. * u.ms)
            to_neo(st)
            return times

        with pytest.raises(TypeError, match='host-only'):
            f(jnp.array([[1.]]))


class TestFromNeo:
    def test_roundtrip(self, neo, pq):
        st = _trains_ms()
        back = from_neo(to_neo(st))
        assert back._unit == u.ms
        np.testing.assert_array_equal(np.asarray(back._times_m), np.asarray(st._times_m))
        np.testing.assert_array_equal(np.asarray(back.counts), [3, 0, 1])
        assert u.math.allclose(back.t_start, -2. * u.ms)
        assert u.math.allclose(back.t_stop, 10. * u.ms)
        assert not bool(back.overflow)
        back.check()

    def test_roundtrip_overflow(self, neo, pq):
        st = jax.jit(lambda s: SpikeTrains.from_dense(s, dt=1. * u.ms, capacity=2))(jnp.ones((3, 2)))
        back = from_neo(to_neo(st))
        assert bool(back.overflow)
        np.testing.assert_array_equal(np.asarray(back.counts), np.asarray(st.counts))

    def test_mixed_units_and_bounds(self, neo, pq):
        trains = [
            neo.SpikeTrain([3., 1.] * pq.ms, t_start=0.5 * pq.ms, t_stop=4. * pq.ms),
            neo.SpikeTrain([0.002] * pq.s, t_start=0. * pq.s, t_stop=0.006 * pq.s),
        ]
        st = from_neo(trains)
        assert st._unit == u.ms
        np.testing.assert_allclose(np.asarray(st._times_m), [[1., 3.], [2., np.inf]], rtol=1e-6)
        assert u.math.allclose(st.t_start, 0. * u.ms)
        assert u.math.allclose(st.t_stop, 6. * u.ms)

    def test_unknown_brainunit_name_uses_seconds(self, neo, pq):
        trains = [neo.SpikeTrain([1., 2.] * pq.shake, t_start=0. * pq.shake, t_stop=5. * pq.shake)]
        st = from_neo(trains)
        assert st._unit == u.second
        np.testing.assert_allclose(np.asarray(st._times_m), [[1e-8, 2e-8]], rtol=1e-6)

    def test_all_empty_trains(self, neo, pq):
        trains = [neo.SpikeTrain([] * pq.ms, t_start=1. * pq.ms, t_stop=3. * pq.ms)] * 2
        st = from_neo(trains)
        assert st.n_neurons == 2 and st.capacity == 1
        np.testing.assert_array_equal(np.asarray(st.counts), [0, 0])
        assert u.math.allclose(st.t_start, 1. * u.ms)
        assert u.math.allclose(st.t_stop, 3. * u.ms)

    def test_empty_sequence(self, neo, pq):
        st = from_neo([])
        assert st.n_neurons == 0
        assert st._unit == u.second
        assert float(st._t_start_m) == 0. and float(st._t_stop_m) == 0.

    def test_zero_neurons_with_bounds(self, neo, pq):
        st = SpikeTrains(jnp.zeros((0, 1)) * u.ms, jnp.zeros(0), 1. * u.ms, 5. * u.ms)
        back = from_neo(to_neo(st), t_start=st.t_start, t_stop=st.t_stop)
        assert back.n_neurons == 0
        assert back._unit == u.ms
        assert float(back._t_start_m) == 1. and float(back._t_stop_m) == 5.

    def test_empty_sequence_partial_bounds(self, neo, pq):
        st = from_neo([], t_stop=5. * u.ms)
        assert st._unit == u.ms
        assert float(st._t_start_m) == 0. and float(st._t_stop_m) == 5.
        st = from_neo([], t_start=2. * u.second)
        assert st._unit == u.second
        assert float(st._t_start_m) == 2. and float(st._t_stop_m) == 2.

    def test_empty_sequence_bad_bounds(self, neo, pq):
        with pytest.raises(TypeError, match='Quantity'):
            from_neo([], t_stop=5.)
        with pytest.raises(TypeError, match='not a time'):
            from_neo([], t_stop=5. * u.mV)

    def test_explicit_bounds_override(self, neo, pq):
        trains = [neo.SpikeTrain([2., 3.] * pq.ms, t_start=1. * pq.ms, t_stop=4. * pq.ms)]
        st = from_neo(trains, t_start=0. * u.second, t_stop=0.01 * u.second)
        assert st._unit == u.ms
        assert float(st._t_start_m) == 0.
        np.testing.assert_allclose(float(st._t_stop_m), 10., rtol=1e-6)
        st = from_neo(trains, t_stop=6. * u.ms)  # t_start keeps its default
        assert float(st._t_start_m) == 1. and float(st._t_stop_m) == 6.
        with pytest.raises(ValueError, match='outside'):
            from_neo(trains, t_stop=2.5 * u.ms)
        with pytest.raises(TypeError):
            from_neo(trains, t_start=0.)

    def test_capacity(self, neo, pq):
        trains = [neo.SpikeTrain([1., 2.] * pq.ms, t_stop=5. * pq.ms)]
        assert from_neo(trains, capacity=6).capacity == 6
        with pytest.raises(ValueError, match='capacity'):
            from_neo(trains, capacity=1)

    def test_accepts_generator(self, neo, pq):
        st = from_neo(t for t in [neo.SpikeTrain([1.] * pq.s, t_stop=2. * pq.s)])
        assert st._unit == u.second and st.n_neurons == 1

    def test_not_neo(self, neo, pq):
        with pytest.raises(TypeError, match='not a neo.SpikeTrain'):
            from_neo([np.array([1., 2.])])

    def test_bu_unit_rejects_non_time(self, neo, pq):
        with pytest.raises(TypeError, match='not a time'):
            N._bu_unit(pq, pq.mV)

    def test_bu_unit_maps_known_units(self, neo, pq):
        for q, expected in [(pq.ms, u.ms), (pq.s, u.second), (pq.us, u.us), (pq.min, u.minute)]:
            unit, scale = N._bu_unit(pq, q)
            assert unit == expected and scale == 1.0


@pytest.mark.parametrize('missing', ['neo', 'quantities'])
def test_missing_package_hint(monkeypatch, missing):
    if missing == 'quantities':
        pytest.importorskip('neo')
    monkeypatch.setitem(sys.modules, missing, None)
    st = SpikeTrains.from_list([np.array([1.]) * u.ms], t_start=0. * u.ms, t_stop=2. * u.ms)
    with pytest.raises(ImportError, match=r'pip install braintools\[neo\]'):
        to_neo(st)
    with pytest.raises(ImportError, match=r'pip install braintools\[neo\]'):
        from_neo([])
