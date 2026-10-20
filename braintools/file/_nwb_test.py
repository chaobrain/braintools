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

"""Tests for ``braintools.file._nwb``.

``pynwb`` is optional; the I/O tests are skipped when it is missing (run them
in the reference environment), while the missing-package tests always run.
"""

import datetime
import os
import sys

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.file import _nwb as W
from braintools.file._nwb import (
    load_nwb_timeseries,
    load_nwb_units,
    save_nwb_timeseries,
    save_nwb_units,
)
from braintools.spike._spike_trains import SpikeTrains

_META = dict(session_description='test session', identifier='test-id')
_T0 = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)


@pytest.fixture
def pynwb():
    return pytest.importorskip('pynwb')


def _trains_ms():
    return SpikeTrains.from_list(
        [np.array([1., 2.5, 7.]) * u.ms, np.array([]) * u.ms, np.array([0.5]) * u.ms],
        t_start=-2. * u.ms, t_stop=10. * u.ms,
    )


def _write_raw(pynwb, path, build):
    """Write an NWB file whose content is set up by ``build(nwbfile)``."""
    f = pynwb.NWBFile(session_description='raw', identifier='raw', session_start_time=_T0)
    build(f)
    with pynwb.NWBHDF5IO(str(path), 'w') as io:
        io.write(f)


# ----------------------------------------------------------------------------
# units
# ----------------------------------------------------------------------------

class TestUnits:
    def test_roundtrip(self, tmp_path, pynwb):
        st = _trains_ms()
        path = tmp_path / 'u.nwb'
        save_nwb_units(path, st, **_META)
        back = load_nwb_units(path)
        assert back._unit == u.second
        np.testing.assert_array_equal(np.asarray(back.counts), [3, 0, 1])
        np.testing.assert_allclose(np.asarray(back.times.to_decimal(u.ms)),
                                   np.asarray(st._times_m), rtol=1e-6)
        assert u.math.allclose(back.t_start, -2. * u.ms)
        assert u.math.allclose(back.t_stop, 10. * u.ms)
        assert not bool(back.overflow)
        back.check()

    def test_file_contents_in_seconds(self, tmp_path, pynwb):
        path = tmp_path / 'u.nwb'
        save_nwb_units(path, _trains_ms(), unit_ids=[10, 20, 30], session_start_time=_T0,
                       **_META)
        with pynwb.NWBHDF5IO(str(path), 'r') as io:
            f = io.read()
            assert f.session_description == 'test session' and f.identifier == 'test-id'
            assert f.session_start_time == _T0
            units = f.units
            assert list(units.id[:]) == [10, 20, 30]
            np.testing.assert_allclose(units['spike_times'][0], [0.001, 0.0025, 0.007])
            assert len(units['spike_times'][1]) == 0
            np.testing.assert_allclose(units['obs_intervals'][2], [[-0.002, 0.01]])
            assert W._OVERFLOW_COLUMN not in units.colnames

    def test_seconds_input(self, tmp_path, pynwb):
        st = SpikeTrains.from_list([np.array([0.25, 1.]) * u.second], t_start=0. * u.second,
                                   t_stop=2. * u.second)
        path = tmp_path / 'u.nwb'
        save_nwb_units(path, st, **_META)
        back = load_nwb_units(path)
        np.testing.assert_array_equal(np.asarray(back._times_m), [[0.25, 1.]])

    def test_overflow_roundtrip(self, tmp_path, pynwb):
        st = jax.jit(lambda s: SpikeTrains.from_dense(s, dt=1. * u.ms, capacity=2))(jnp.ones((3, 2)))
        assert bool(st.overflow)
        path = tmp_path / 'u.nwb'
        save_nwb_units(path, st, **_META)
        back = load_nwb_units(path)
        assert bool(back.overflow)
        np.testing.assert_array_equal(np.asarray(back.counts), [2, 2])
        with pytest.raises(ValueError, match='overflow'):
            back.check()

    def test_zero_neurons(self, tmp_path, pynwb):
        st = SpikeTrains(jnp.zeros((0, 1)) * u.ms, jnp.zeros(0), 0. * u.ms, 5. * u.ms)
        path = tmp_path / 'u.nwb'
        save_nwb_units(path, st, **_META)
        back = load_nwb_units(path)
        assert back.n_neurons == 0
        assert back._unit == u.second
        # the recording bounds survive an empty units table
        assert u.math.allclose(back.t_start, 0. * u.ms)
        assert u.math.allclose(back.t_stop, 5. * u.ms)

    def test_zero_neurons_nonzero_start(self, tmp_path, pynwb):
        st = SpikeTrains(jnp.zeros((0, 2)) * u.second, jnp.zeros(0), 1.5 * u.second,
                         4. * u.second)
        path = tmp_path / 'u.nwb'
        save_nwb_units(path, st, **_META)
        back = load_nwb_units(path)
        assert float(back._t_start_m) == 1.5 and float(back._t_stop_m) == 4.

    def test_recording_bounds_stored_as_time_intervals(self, tmp_path, pynwb):
        path = tmp_path / 'u.nwb'
        save_nwb_units(path, _trains_ms(), **_META)
        with pynwb.NWBHDF5IO(str(path), 'r') as io:
            table = io.read().intervals[W._BOUNDS_TABLE]
            assert len(table) == 1
            np.testing.assert_allclose(table['start_time'][:], [-0.002])
            np.testing.assert_allclose(table['stop_time'][:], [0.01])

    def test_bounds_table_widened_to_spikes(self, tmp_path, pynwb):
        path = tmp_path / 'raw.nwb'

        def build(f):
            f.add_unit(spike_times=[0.5, 2.5])
            table = pynwb.epoch.TimeIntervals(name=W._BOUNDS_TABLE, description='bounds')
            table.add_interval(start_time=1., stop_time=2.)
            f.add_time_intervals(table)

        _write_raw(pynwb, path, build)
        st = load_nwb_units(path)
        assert u.math.allclose(st.t_start, 0.5 * u.second)
        assert u.math.allclose(st.t_stop, 2.5 * u.second)

    def test_no_spikes(self, tmp_path, pynwb):
        st = SpikeTrains.from_dense(jnp.zeros((5, 2)), dt=1. * u.ms)
        path = tmp_path / 'u.nwb'
        save_nwb_units(path, st, **_META)
        back = load_nwb_units(path)
        np.testing.assert_array_equal(np.asarray(back.counts), [0, 0])
        assert u.math.allclose(back.t_stop, 5. * u.ms)

    def test_capacity(self, tmp_path, pynwb):
        path = tmp_path / 'u.nwb'
        save_nwb_units(path, _trains_ms(), **_META)
        assert load_nwb_units(path, capacity=8).capacity == 8
        with pytest.raises(ValueError, match='capacity'):
            load_nwb_units(path, capacity=2)

    def test_overwrite(self, tmp_path, pynwb):
        path = tmp_path / 'u.nwb'
        save_nwb_units(path, _trains_ms(), **_META)
        one = SpikeTrains.from_list([np.array([1.]) * u.ms], t_start=0. * u.ms, t_stop=2. * u.ms)
        with pytest.raises(FileExistsError, match='overwrite=True'):
            save_nwb_units(path, one, **_META)
        assert load_nwb_units(path).n_neurons == 3
        save_nwb_units(path, one, overwrite=True, **_META)
        assert load_nwb_units(path).n_neurons == 1
        assert sorted(os.listdir(tmp_path)) == ['u.nwb']

    def test_unit_ids_validation(self, tmp_path, pynwb):
        with pytest.raises(ValueError, match='entries'):
            save_nwb_units(tmp_path / 'a.nwb', _trains_ms(), unit_ids=[1, 2], **_META)
        with pytest.raises(ValueError, match='unique'):
            save_nwb_units(tmp_path / 'b.nwb', _trains_ms(), unit_ids=[1, 1, 2], **_META)
        assert os.listdir(tmp_path) == []

    def test_rejects_unitless_and_batched(self, tmp_path, pynwb):
        with pytest.raises(TypeError, match='no time unit'):
            save_nwb_units(tmp_path / 'a.nwb', SpikeTrains.from_dense(jnp.ones((2, 1)), dt=1.),
                           **_META)
        batched = jax.vmap(lambda s: SpikeTrains.from_dense(s, dt=1. * u.ms, capacity=3))(
            jnp.ones((2, 3, 1)))
        with pytest.raises(ValueError, match='unbatched'):
            save_nwb_units(tmp_path / 'b.nwb', batched, **_META)

    def test_no_units_table(self, tmp_path, pynwb):
        path = tmp_path / 'ts.nwb'
        save_nwb_timeseries(path, np.ones(3), 1., 'x')
        with pytest.raises(ValueError, match='no Units table'):
            load_nwb_units(path)

    def test_missing_file(self, tmp_path, pynwb):
        with pytest.raises(FileNotFoundError):
            load_nwb_units(tmp_path / 'nope.nwb')

    def test_foreign_without_obs_intervals(self, tmp_path, pynwb):
        path = tmp_path / 'raw.nwb'

        def build(f):
            f.add_unit(spike_times=[0.3, 0.1], id=4)
            f.add_unit(spike_times=[0.9], id=8)

        _write_raw(pynwb, path, build)
        st = load_nwb_units(path)
        np.testing.assert_allclose(np.asarray(st._times_m), [[0.1, 0.3], [0.9, np.inf]], rtol=1e-6)
        assert u.math.allclose(st.t_start, 0.1 * u.second)
        assert u.math.allclose(st.t_stop, 0.9 * u.second)

    def test_foreign_obs_intervals_widened(self, tmp_path, pynwb):
        path = tmp_path / 'raw.nwb'

        def build(f):
            f.add_unit(spike_times=[0.5, 2.5], obs_intervals=[[1., 2.]])
            f.add_unit(spike_times=[1.5], obs_intervals=[[0.8, 1.2], [1.4, 1.9]])

        _write_raw(pynwb, path, build)
        st = load_nwb_units(path)
        assert u.math.allclose(st.t_start, 0.5 * u.second)
        assert u.math.allclose(st.t_stop, 2.5 * u.second)
        st.check()

    def test_foreign_empty_units_table(self, tmp_path, pynwb):
        path = tmp_path / 'raw.nwb'
        _write_raw(pynwb, path, lambda f: setattr(
            f, 'units', pynwb.misc.Units(name='units', description='none')))
        st = load_nwb_units(path)
        assert st.n_neurons == 0
        assert float(st._t_start_m) == 0. and float(st._t_stop_m) == 0.


# ----------------------------------------------------------------------------
# time series
# ----------------------------------------------------------------------------

class TestTimeSeries:
    def test_quantity_roundtrip(self, tmp_path, pynwb):
        v = jnp.arange(12.).reshape(6, 2) * u.mV
        path = tmp_path / 'v.nwb'
        save_nwb_timeseries(path, v, 0.1 * u.ms, 'v')
        data, dt = load_nwb_timeseries(path, 'v')
        assert data.unit == u.mV
        np.testing.assert_array_equal(np.asarray(data.mantissa), np.asarray(v.mantissa))
        assert dt.unit == u.second
        np.testing.assert_allclose(dt.to_decimal(u.ms), 0.1, rtol=1e-12)

    def test_compound_unit(self, tmp_path, pynwb):
        x = np.array([1., 2., 3.]) * (u.mV / u.ms)
        path = tmp_path / 'x.nwb'
        save_nwb_timeseries(path, x, 1. * u.ms, 'slope')
        data, _ = load_nwb_timeseries(path, 'slope')
        assert data.unit == u.mV / u.ms and str(data.unit) == 'mV / ms'

    @pytest.mark.parametrize('unit', [u.volt, 'V'])
    def test_converts_to_requested_unit(self, tmp_path, pynwb, unit):
        v = np.array([-65., 20.]) * u.mV
        path = tmp_path / 'v.nwb'
        save_nwb_timeseries(path, v, 1. * u.ms, 'v', unit=unit)
        with pynwb.NWBHDF5IO(str(path), 'r') as io:
            ts = io.read().acquisition['v']
            assert ts.unit == 'V'
            np.testing.assert_allclose(ts.data[:], [-0.065, 0.02])
        data, _ = load_nwb_timeseries(path, 'v')
        assert data.unit == u.volt
        assert u.math.allclose(data, v)

    def test_plain_data(self, tmp_path, pynwb):
        x = np.arange(5, dtype=np.int16)
        path = tmp_path / 'x.nwb'
        save_nwb_timeseries(path, x, 0.5, 'counts', description='spike counts')
        data, dt = load_nwb_timeseries(path, 'counts')
        assert isinstance(data, np.ndarray) and data.dtype == np.int16
        np.testing.assert_array_equal(data, x)
        assert u.math.allclose(dt, 0.5 * u.second)
        with pynwb.NWBHDF5IO(str(path), 'r') as io:
            ts = io.read().acquisition['counts']
            assert ts.unit == 'n/a' and ts.description == 'spike counts'

    def test_plain_data_with_recorded_unit(self, tmp_path, pynwb):
        path = tmp_path / 'x.nwb'
        save_nwb_timeseries(path, jnp.ones(3), 1e-3, 'i', unit=u.nA)
        data, _ = load_nwb_timeseries(path, 'i')
        assert data.unit == u.nA

    def test_options_and_overwrite(self, tmp_path, pynwb):
        path = tmp_path / 'x.nwb'
        save_nwb_timeseries(path, np.ones(2), 1., 'a', identifier='my-id',
                            session_start_time=_T0, session_description='s')
        with pynwb.NWBHDF5IO(str(path), 'r') as io:
            f = io.read()
            assert f.identifier == 'my-id' and f.session_start_time == _T0
        with pytest.raises(FileExistsError):
            save_nwb_timeseries(path, np.ones(2), 1., 'a')
        save_nwb_timeseries(path, np.zeros(2), 1., 'b', overwrite=True)
        data, _ = load_nwb_timeseries(path, 'b')
        np.testing.assert_array_equal(data, [0., 0.])

    def test_invalid_inputs(self, tmp_path, pynwb):
        path = tmp_path / 'x.nwb'
        with pytest.raises(TypeError, match='must be a time'):
            save_nwb_timeseries(path, np.ones(2), 1. * u.mV, 'a')
        with pytest.raises(ValueError, match='positive'):
            save_nwb_timeseries(path, np.ones(2), 0. * u.ms, 'a')
        with pytest.raises(ValueError, match='positive'):
            save_nwb_timeseries(path, np.ones(2), -1., 'a')
        with pytest.raises(ValueError, match='time axis'):
            save_nwb_timeseries(path, np.float64(1.), 1., 'a')
        with pytest.raises(u.UnitMismatchError):
            save_nwb_timeseries(path, np.ones(2) * u.mV, 1., 'a', unit=u.ms)

        @jax.jit
        def f(x):
            save_nwb_timeseries(path, x, 1., 'a')
            return x

        with pytest.raises(TypeError, match='tracer'):
            f(jnp.ones(2))
        assert os.listdir(tmp_path) == []

    def test_missing_name(self, tmp_path, pynwb):
        path = tmp_path / 'x.nwb'
        save_nwb_timeseries(path, np.ones(2), 1., 'present')
        with pytest.raises(KeyError, match="present"):
            load_nwb_timeseries(path, 'absent')

    def test_missing_file(self, tmp_path, pynwb):
        with pytest.raises(FileNotFoundError):
            load_nwb_timeseries(tmp_path / 'nope.nwb', 'x')

    def test_foreign_processing_module_conversion_offset(self, tmp_path, pynwb):
        path = tmp_path / 'raw.nwb'

        def build(f):
            mod = f.create_processing_module('ecephys', 'processed')
            mod.add(pynwb.TimeSeries(name='lfp', data=np.array([1, 2, 3], dtype=np.int16),
                                     unit='volts', rate=250., conversion=1e-3, offset=0.5))

        _write_raw(pynwb, path, build)
        data, dt = load_nwb_timeseries(path, 'lfp')
        assert data.unit == u.volt
        np.testing.assert_allclose(np.asarray(data.mantissa), [0.501, 0.502, 0.503])
        assert u.math.allclose(dt, 4. * u.ms)

    def test_foreign_uniform_timestamps(self, tmp_path, pynwb):
        path = tmp_path / 'raw.nwb'
        _write_raw(pynwb, path, lambda f: f.add_acquisition(pynwb.TimeSeries(
            name='x', data=np.ones(4), unit='n/a', timestamps=np.array([1., 1.5, 2., 2.5]))))
        data, dt = load_nwb_timeseries(path, 'x')
        assert isinstance(data, np.ndarray)
        assert u.math.allclose(dt, 0.5 * u.second)

    @pytest.mark.parametrize('stamps', [[0., 1., 3.], [0.]])
    def test_foreign_irregular_timestamps(self, tmp_path, pynwb, stamps):
        path = tmp_path / 'raw.nwb'
        _write_raw(pynwb, path, lambda f: f.add_acquisition(pynwb.TimeSeries(
            name='x', data=np.ones(len(stamps)), unit='n/a', timestamps=np.array(stamps))))
        with pytest.raises(ValueError, match='irregular'):
            load_nwb_timeseries(path, 'x')

    def test_foreign_unknown_unit_warns(self, tmp_path, pynwb):
        path = tmp_path / 'raw.nwb'
        _write_raw(pynwb, path, lambda f: f.add_acquisition(pynwb.TimeSeries(
            name='x', data=np.ones(3), unit='furlongs per fortnight', rate=1.)))
        with pytest.warns(UserWarning, match='furlongs'):
            data, _ = load_nwb_timeseries(path, 'x')
        assert isinstance(data, np.ndarray)


@pytest.mark.parametrize('name', ['save_nwb_units', 'load_nwb_units',
                                  'save_nwb_timeseries', 'load_nwb_timeseries'])
def test_missing_pynwb_hint(tmp_path, monkeypatch, name):
    monkeypatch.setitem(sys.modules, 'pynwb', None)
    st = SpikeTrains.from_list([np.array([1.]) * u.ms], t_start=0. * u.ms, t_stop=2. * u.ms)
    calls = {
        'save_nwb_units': lambda: save_nwb_units(tmp_path / 'a.nwb', st, **_META),
        'load_nwb_units': lambda: load_nwb_units(tmp_path / 'a.nwb'),
        'save_nwb_timeseries': lambda: save_nwb_timeseries(tmp_path / 'a.nwb', np.ones(2), 1., 'x'),
        'load_nwb_timeseries': lambda: load_nwb_timeseries(tmp_path / 'a.nwb', 'x'),
    }
    with pytest.raises(ImportError, match=r'pip install braintools\[nwb\]'):
        calls[name]()


@pytest.mark.parametrize('text, expected', [
    ('mV', u.mV), ('volts', u.volt), ('Volts', u.volt), ('amperes', u.amp), ('seconds', u.second),
    ('n/a', None), ('N/A', None), ('', None), ('unitless', None), (' mV ', u.mV), ('nA', u.nA),
])
def test_parse_nwb_unit(text, expected):
    assert W._parse_nwb_unit(text) == expected
