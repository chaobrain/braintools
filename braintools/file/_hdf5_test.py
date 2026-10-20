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

"""Tests for ``braintools.file._hdf5``."""

import collections
import os
import pathlib
import sys

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.file import _hdf5 as H
from braintools.file._hdf5 import load_hdf5, save_hdf5
from braintools.spike._spike_trains import SpikeTrains

h5py = pytest.importorskip('h5py')


def _roundtrip(tmp_path, tree, name='data.h5'):
    path = tmp_path / name
    save_hdf5(path, tree)
    return load_hdf5(path)


def _assert_trains_equal(a: SpikeTrains, b: SpikeTrains):
    assert type(b) is SpikeTrains
    assert a._unit == b._unit
    if a._unit is not None:
        assert str(a._unit) == str(b._unit)
    np.testing.assert_array_equal(np.asarray(a._times_m), np.asarray(b._times_m))
    assert np.asarray(b._times_m).dtype == np.asarray(a._times_m).dtype
    np.testing.assert_array_equal(np.asarray(a.counts), np.asarray(b.counts))
    assert b.counts.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(a._t_start_m), np.asarray(b._t_start_m))
    np.testing.assert_array_equal(np.asarray(a._t_stop_m), np.asarray(b._t_stop_m))
    np.testing.assert_array_equal(np.asarray(a.overflow), np.asarray(b.overflow))
    assert b.overflow.dtype == jnp.bool_


# ----------------------------------------------------------------------------
# containers, scalars and arrays
# ----------------------------------------------------------------------------

class TestContainers:
    def test_nested_roundtrip_keeps_types_and_order(self, tmp_path):
        tree = {
            'z': [1, 2.5, True, 'text', None, 1 + 2j],
            'a': ({'inner': np.arange(3)}, ()),
            'm': {},
            'empty_list': [],
        }
        out = _roundtrip(tmp_path, tree)
        assert list(out) == ['z', 'a', 'm', 'empty_list']  # insertion order kept
        z = out['z']
        assert z == [1, 2.5, True, 'text', None, 1 + 2j]
        assert [type(v) for v in z] == [int, float, bool, str, type(None), complex]
        assert isinstance(out['a'], tuple) and len(out['a']) == 2
        np.testing.assert_array_equal(out['a'][0]['inner'], np.arange(3))
        assert out['a'][1] == ()
        assert out['m'] == {}
        assert out['empty_list'] == []

    def test_long_list_order(self, tmp_path):
        tree = list(range(12))  # '10' and '11' must not sort before '2'
        assert _roundtrip(tmp_path, tree) == tree

    @pytest.mark.parametrize('root', [
        [1, 2], (1, 'a'), 3.5, 'hello', None, True,
    ])
    def test_non_dict_roots(self, tmp_path, root):
        out = _roundtrip(tmp_path, root)
        assert out == root
        assert type(out) is type(root)

    def test_namedtuple_loads_as_tuple(self, tmp_path):
        Point = collections.namedtuple('Point', 'x y')
        out = _roundtrip(tmp_path, {'p': Point(1, 2)})
        assert out['p'] == (1, 2)
        assert type(out['p']) is tuple

    def test_unicode_string(self, tmp_path):
        assert _roundtrip(tmp_path, {'s': 'héllo ∑'})['s'] == 'héllo ∑'


class TestArrays:
    @pytest.mark.parametrize('dtype', [
        np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint32,
        np.float16, np.float32, np.float64, np.complex64,
    ])
    def test_numpy_dtypes_and_shapes(self, tmp_path, dtype):
        arrays = {
            'nd': (np.arange(6).reshape(2, 3) % 2).astype(dtype),
            'zero_d': np.array(1, dtype=dtype),
            'empty': np.zeros((0, 4), dtype=dtype),
        }
        out = _roundtrip(tmp_path, arrays)
        for k, v in arrays.items():
            assert isinstance(out[k], np.ndarray)
            assert out[k].dtype == v.dtype
            assert out[k].shape == v.shape
            np.testing.assert_array_equal(out[k], v)

    def test_zero_d_arrays_keep_kind(self, tmp_path):
        tree = {'np0': np.array(2.5), 'jax0': jnp.array(3, dtype=jnp.int32),
                'jbool': jnp.array(True)}
        out = _roundtrip(tmp_path, tree)
        assert isinstance(out['np0'], np.ndarray) and out['np0'].shape == ()
        assert float(out['np0']) == 2.5
        assert isinstance(out['jax0'], jax.Array) and out['jax0'].shape == ()
        assert out['jax0'].dtype == jnp.int32 and int(out['jax0']) == 3
        assert out['jbool'].dtype == jnp.bool_ and bool(out['jbool'])

    def test_jax_arrays_stay_jax(self, tmp_path):
        x = jnp.linspace(0., 1., 5)
        b = jnp.array([True, False])
        out = _roundtrip(tmp_path, {'x': x, 'b': b})
        assert isinstance(out['x'], jax.Array)
        assert out['x'].dtype == x.dtype
        np.testing.assert_array_equal(np.asarray(out['x']), np.asarray(x))
        assert out['b'].dtype == jnp.bool_
        np.testing.assert_array_equal(np.asarray(out['b']), [True, False])

    def test_numpy_scalars(self, tmp_path):
        tree = {'f': np.float32(1.5), 'i': np.int16(-3), 'b': np.bool_(True)}
        out = _roundtrip(tmp_path, tree)
        for k, v in tree.items():
            assert type(out[k]) is type(v)
            assert out[k] == v

    def test_nan_and_inf(self, tmp_path):
        x = np.array([np.nan, np.inf, -np.inf, 0.])
        np.testing.assert_array_equal(_roundtrip(tmp_path, {'x': x})['x'], x)


# ----------------------------------------------------------------------------
# key escaping
# ----------------------------------------------------------------------------

class TestKeyEscaping:
    KEYS = ['a/b', '/', '//x//', '%', '%2F', '%25', '.', '..', '', ' ', 'nul\x00byte',
            'ünï/cødé', 'plain', '__bt_type__']

    def test_special_keys_roundtrip(self, tmp_path):
        tree = {k: i for i, k in enumerate(self.KEYS)}
        out = _roundtrip(tmp_path, tree)
        assert out == tree
        assert list(out) == self.KEYS

    def test_nested_special_keys(self, tmp_path):
        tree = {'a/b': {'c/d': {'': np.ones(2)}}}
        out = _roundtrip(tmp_path, tree)
        np.testing.assert_array_equal(out['a/b']['c/d'][''], np.ones(2))

    def test_slash_key_is_not_a_path(self, tmp_path):
        path = tmp_path / 'k.h5'
        save_hdf5(path, {'a/b': 1})
        with h5py.File(path, 'r') as f:
            assert 'a' not in f  # no intermediate group was created
            assert list(f.keys()) == ['a%2Fb']

    @pytest.mark.parametrize('key', [
        'a/b', '%', '.', '', 'x%2Fy', 'nul\x00',
    ])
    def test_escape_unescape_inverse(self, key):
        name = H._escape(key)
        assert '/' not in name and '\x00' not in name and name not in ('', '.')
        assert H._unescape(name) == key


# ----------------------------------------------------------------------------
# Quantities
# ----------------------------------------------------------------------------

class TestQuantities:
    @pytest.mark.parametrize('unit', [
        u.mV, u.ms, u.second, u.mV / u.ms, u.nS / u.cm ** 2, u.uA / u.cm ** 2,
        u.mV * u.ms, u.Hz, u.kHz, u.mM, u.mV ** 2, u.ohm * u.cm, u.siemens / u.meter ** 2,
        u.Unit(scale=-7), u.Unit(dim=u.ms.dim, scale=-3, factor=2.5),
    ])
    def test_unit_roundtrips_exactly(self, tmp_path, unit):
        q = jnp.array([1., 2., 3.]) * unit
        out = _roundtrip(tmp_path, {'q': q})['q']
        assert isinstance(out, u.Quantity)
        assert out.unit == q.unit
        assert out.unit.dim == q.unit.dim
        assert out.unit.scale == q.unit.scale and out.unit.factor == q.unit.factor
        np.testing.assert_array_equal(np.asarray(out.mantissa), np.asarray(q.mantissa))

    def test_compound_unit_keeps_display_name(self, tmp_path):
        out = _roundtrip(tmp_path, {'q': jnp.ones(2) * (u.mV / u.ms)})['q']
        assert str(out.unit) == 'mV / ms'
        assert repr(out) == repr(jnp.ones(2) * (u.mV / u.ms))

    def test_mantissa_kind_and_dtype(self, tmp_path):
        tree = {
            'jax': jnp.ones(3, dtype=jnp.float32) * u.mV,
            'numpy': u.Quantity(np.arange(3, dtype=np.int16), unit=u.ms),
            'python': u.Quantity(2.5, unit=u.nA),
            'zero_d': u.Quantity(jnp.array(4.), unit=u.ms),
        }
        out = _roundtrip(tmp_path, tree)
        assert isinstance(out['jax'].mantissa, jax.Array)
        assert out['jax'].mantissa.dtype == jnp.float32
        assert isinstance(out['numpy'].mantissa, np.ndarray)
        assert out['numpy'].mantissa.dtype == np.int16
        assert type(out['python'].mantissa) is float and out['python'].mantissa == 2.5
        assert out['zero_d'].shape == ()
        assert u.math.allclose(out['zero_d'], 4. * u.ms)

    def test_unitless_quantity(self, tmp_path):
        q = u.Quantity(jnp.arange(3.))
        out = _roundtrip(tmp_path, {'q': q})['q']
        assert isinstance(out, u.Quantity)
        assert out.unit.is_unitless
        np.testing.assert_array_equal(np.asarray(out.mantissa), [0., 1., 2.])

    def test_quantity_root(self, tmp_path):
        out = _roundtrip(tmp_path, 3. * u.mV)
        assert out.unit == u.mV and float(out.mantissa) == 3.

    def test_unparseable_string_falls_back_to_exact(self, tmp_path):
        path = tmp_path / 'q.h5'
        save_hdf5(path, {'q': jnp.ones(2) * u.mV})
        with h5py.File(path, 'r+') as f:
            f['q'].attrs[H._UNIT_STR] = 'not a unit at all'
        out = load_hdf5(path)['q']
        assert out.unit == u.mV

    def test_mismatching_string_falls_back_to_exact(self, tmp_path):
        path = tmp_path / 'q.h5'
        save_hdf5(path, {'q': jnp.ones(2) * u.mV})
        with h5py.File(path, 'r+') as f:
            f['q'].attrs[H._UNIT_STR] = 'ms'  # parses, but disagrees with the numbers
        out = load_hdf5(path)['q']
        assert out.unit == u.mV


# ----------------------------------------------------------------------------
# SpikeTrains
# ----------------------------------------------------------------------------

class TestSpikeTrains:
    def test_with_units_and_empty_neuron(self, tmp_path):
        st = SpikeTrains.from_list(
            [np.array([1., 2.5]) * u.ms, np.array([]) * u.ms, np.array([0.5]) * u.ms],
            t_start=-1. * u.ms, t_stop=10. * u.ms,
        )
        out = _roundtrip(tmp_path, {'st': st})['st']
        _assert_trains_equal(st, out)
        assert out.t_start.unit == u.ms
        assert u.math.allclose(out.t_start, -1. * u.ms)
        assert u.math.allclose(out.t_stop, 10. * u.ms)
        assert int(out.counts[1]) == 0
        out.check()

    def test_compound_time_unit_display(self, tmp_path):
        st = SpikeTrains.from_list([np.array([1., 2.]) * u.second], t_start=0. * u.second,
                                   t_stop=3. * u.second)
        out = _roundtrip(tmp_path, st)
        assert str(out._unit) == 's'

    def test_unitless(self, tmp_path):
        st = SpikeTrains.from_dense(jnp.array([[1, 0], [0, 1]]), dt=0.5)
        out = _roundtrip(tmp_path, [st])[0]
        assert out._unit is None
        assert not isinstance(out.times, u.Quantity)
        _assert_trains_equal(st, out)

    def test_zero_neurons(self, tmp_path):
        st = SpikeTrains(jnp.zeros((0, 1)) * u.ms, jnp.zeros(0), 0. * u.ms, 5. * u.ms)
        assert st.n_neurons == 0
        out = _roundtrip(tmp_path, {'st': st})['st']
        _assert_trains_equal(st, out)
        assert out.n_neurons == 0 and out.capacity == st.capacity

    def test_no_spikes_at_all(self, tmp_path):
        st = SpikeTrains.from_dense(jnp.zeros((4, 3)), dt=1. * u.ms)
        out = _roundtrip(tmp_path, st)
        _assert_trains_equal(st, out)
        assert np.all(np.isposinf(np.asarray(out._times_m)))

    def test_overflow_preserved(self, tmp_path):
        make = jax.jit(lambda s: SpikeTrains.from_dense(s, dt=1. * u.ms, capacity=2))
        st = make(jnp.ones((3, 2)))
        assert bool(st.overflow)
        out = _roundtrip(tmp_path, {'st': st})['st']
        _assert_trains_equal(st, out)
        assert bool(out.overflow)
        with pytest.raises(ValueError, match='overflow'):
            out.check()

    def test_batched_trains(self, tmp_path):
        trials = jnp.zeros((2, 5, 3)).at[:, ::2, 0].set(1.)
        st = jax.vmap(lambda s: SpikeTrains.from_dense(s, dt=1. * u.ms, capacity=4))(trials)
        out = _roundtrip(tmp_path, st)
        assert out.times.shape == (2, 3, 4)
        _assert_trains_equal(st, out)

    def test_loaded_trains_usable(self, tmp_path):
        st = SpikeTrains.from_dense(jnp.eye(3), dt=1. * u.ms)
        out = _roundtrip(tmp_path, st)
        np.testing.assert_array_equal(np.asarray(out.bin(1. * u.ms)), np.asarray(st.bin(1. * u.ms)))


# ----------------------------------------------------------------------------
# files, errors and foreign files
# ----------------------------------------------------------------------------

class TestFiles:
    def test_overwrite_false_raises_and_keeps_file(self, tmp_path):
        path = tmp_path / 'f.h5'
        save_hdf5(path, {'a': 1})
        with pytest.raises(FileExistsError, match='overwrite=True'):
            save_hdf5(path, {'a': 2})
        assert load_hdf5(path) == {'a': 1}

    def test_overwrite_true_replaces(self, tmp_path):
        path = tmp_path / 'f.h5'
        save_hdf5(path, {'a': 1})
        save_hdf5(path, {'b': 2}, overwrite=True)
        assert load_hdf5(path) == {'b': 2}

    def test_str_path(self, tmp_path):
        path = str(tmp_path / 'f.h5')
        save_hdf5(path, {'a': 1})
        assert load_hdf5(pathlib.Path(path)) == {'a': 1}

    def test_failed_write_leaves_nothing(self, tmp_path):
        path = tmp_path / 'f.h5'
        with pytest.raises(TypeError):
            save_hdf5(path, {'ok': 1, 'bad': {1, 2}})
        assert os.listdir(tmp_path) == []

    def test_failed_overwrite_keeps_old_file(self, tmp_path):
        path = tmp_path / 'f.h5'
        save_hdf5(path, {'a': 1})
        with pytest.raises(TypeError):
            save_hdf5(path, {'bad': object()}, overwrite=True)
        assert load_hdf5(path) == {'a': 1}
        assert os.listdir(tmp_path) == ['f.h5']

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_hdf5(tmp_path / 'nope.h5')

    @pytest.mark.parametrize('tree, match', [
        ({1: 'int key'}, 'keys must be str'),
        ({'a': np.array(['x', 'y'])}, 'dtype'),
        ({'a': np.array([object()], dtype=object)}, 'dtype'),
        ({'a': {1, 2}}, 'set'),
        ({'a': b'bytes'}, 'bytes'),
    ])
    def test_unsupported_values(self, tmp_path, tree, match):
        with pytest.raises(TypeError, match=match):
            save_hdf5(tmp_path / 'f.h5', tree)

    def test_bfloat16_rejected_with_hint(self, tmp_path):
        with pytest.raises(TypeError, match='bfloat16.*float32'):
            save_hdf5(tmp_path / 'f.h5', {'a': jnp.ones(2, dtype=jnp.bfloat16)})
        with pytest.raises(TypeError, match='bfloat16.*float32'):
            save_hdf5(tmp_path / 'f.h5', {'a': jnp.ones(2, dtype=jnp.bfloat16) * u.mV})

    @pytest.mark.parametrize('value', [2 ** 64, -2 ** 63 - 1, 10 ** 30])
    def test_python_int_out_of_range(self, tmp_path, value):
        with pytest.raises(OverflowError, match='64-bit'):
            save_hdf5(tmp_path / 'f.h5', {'a': value})
        assert os.listdir(tmp_path) == []

    @pytest.mark.parametrize('value', [2 ** 63 - 1, 2 ** 63, 2 ** 64 - 1, -2 ** 63])
    def test_python_int_extremes_roundtrip(self, tmp_path, value):
        out = _roundtrip(tmp_path, {'a': value})['a']
        assert out == value and type(out) is int

    def test_tracer_rejected(self, tmp_path):
        path = tmp_path / 'f.h5'

        @jax.jit
        def f(x):
            save_hdf5(path, {'x': x})
            return x

        with pytest.raises(TypeError, match='tracer'):
            f(jnp.ones(2))

    def test_foreign_file(self, tmp_path):
        path = tmp_path / 'foreign.h5'
        with h5py.File(path, 'w') as f:
            f.create_dataset('x', data=np.arange(4, dtype=np.int32))
            g = f.create_group('grp')
            g.create_dataset('s', data='abc', dtype=h5py.string_dtype())
            g.create_dataset('e', data=h5py.Empty('f'))
            g.create_dataset('v', data=2.)
        out = load_hdf5(path)
        np.testing.assert_array_equal(out['x'], np.arange(4))
        assert out['x'].dtype == np.int32
        assert out['grp']['s'] == 'abc'
        assert out['grp']['e'] is None
        assert isinstance(out['grp']['v'], np.ndarray) and float(out['grp']['v']) == 2.

    def test_format_marker(self, tmp_path):
        path = tmp_path / 'f.h5'
        save_hdf5(path, {'a': 1})
        with h5py.File(path, 'r') as f:
            assert f.attrs[H._FORMAT] == 'braintools.hdf5'
            assert int(f.attrs[H._VERSION]) == 1

    @pytest.mark.parametrize('fn', ['save', 'load'])
    def test_missing_h5py_hint(self, tmp_path, monkeypatch, fn):
        monkeypatch.setitem(sys.modules, 'h5py', None)
        with pytest.raises(ImportError, match=r'pip install braintools\[hdf5\]'):
            if fn == 'save':
                save_hdf5(tmp_path / 'f.h5', {'a': 1})
            else:
                load_hdf5(tmp_path / 'f.h5')


def test_import_optional_returns_module():
    assert H._import_optional('json', 'x').__name__ == 'json'
