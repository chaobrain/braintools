# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

"""Extra tests for ``braintools.file._msg_checkpoint``.

These cover gaps not already exercised by ``_msg_checkpoint_test.py`` and
``_msg_checkpoint_async_test.py``: low-level (de)serialization branches
(complex/scalar/object dtypes, bfloat16, chunking), registry behavior, the
file-save/load error and option paths, and ``_rename_fn`` edge cases.  All file
I/O happens inside a ``TemporaryDirectory``.
"""

import importlib.util
import os
import tempfile
import unittest
import warnings
from collections import namedtuple

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

spec = importlib.util.find_spec("msgpack")
if spec is None:
    pytest.skip("msgpack not installed", allow_module_level=True)

from braintools.file import _msg_checkpoint as M


class TestLowLevelSerialization(unittest.TestCase):
    """Round-trip the low-level msgpack (de)serialization helpers."""

    def test_complex_scalar_round_trip(self):
        data = {"c": 3 + 4j}
        out = M._msgpack_restore(M._msgpack_serialize(data, in_place=False))
        self.assertEqual(out["c"], 3 + 4j)

    def test_numpy_scalar_round_trip(self):
        data = {"s": np.float32(2.5)}
        out = M._msgpack_restore(M._msgpack_serialize(data))
        self.assertEqual(out["s"], np.float32(2.5))
        self.assertIsInstance(out["s"], np.floating)

    def test_ndarray_round_trip(self):
        data = {"a": np.arange(6).reshape(2, 3)}
        out = M._msgpack_restore(M._msgpack_serialize(data))
        np.testing.assert_array_equal(out["a"], data["a"])

    def test_jax_array_round_trip(self):
        data = {"a": jnp.arange(4)}
        out = M._msgpack_restore(M._msgpack_serialize(data))
        np.testing.assert_array_equal(out["a"], np.arange(4))

    def test_ndarray_object_dtype_rejected(self):
        with self.assertRaises(ValueError) as cm:
            M._ndarray_to_bytes(np.array([{1}, {2}], dtype=object))
        self.assertIn("Object and structured dtypes", str(cm.exception))

    def test_bfloat16_round_trip(self):
        arr = np.asarray(jnp.array([1.0, 2.0, 3.0], dtype=jnp.bfloat16))
        restored = M._ndarray_from_bytes(M._ndarray_to_bytes(arr))
        self.assertEqual(restored.dtype, jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(restored, np.float32),
                                      np.asarray(arr, np.float32))

    def test_corrupt_data_raises_invalid_checkpoint(self):
        with self.assertRaises(M.InvalidCheckpointPath) as cm:
            M._msgpack_restore(b"\xff\xff\xff not msgpack at all \x00\x01")
        self.assertIn("Corrupt or invalid", str(cm.exception))

    def test_to_bytes_from_bytes_round_trip(self):
        target = {"w": np.ones(3), "n": 5, "items": [1, 2, 3]}
        restored = M._from_bytes(target, M._to_bytes(target))
        self.assertEqual(restored["n"], 5)
        np.testing.assert_array_equal(restored["w"], np.ones(3))
        self.assertEqual(restored["items"], [1, 2, 3])

    def test_np_convert_in_place_top_level_jax_array(self):
        result = M._np_convert_in_place(jnp.arange(3))
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.arange(3))

    def test_np_convert_in_place_nested_dict(self):
        d = {"outer": {"inner": jnp.arange(2)}, "plain": 7}
        M._np_convert_in_place(d)
        self.assertIsInstance(d["outer"]["inner"], np.ndarray)

    def test_ext_pack_passes_through_plain_values(self):
        # Non-array/scalar/complex values are returned unchanged by the encoder.
        self.assertEqual(M._msgpack_ext_pack("plain"), "plain")
        self.assertEqual(M._msgpack_ext_pack(42), 42)

    def test_ext_unpack_unknown_code_returns_exttype(self):
        import msgpack

        result = M._msgpack_ext_unpack(99, b"abc")
        self.assertIsInstance(result, msgpack.ExtType)
        self.assertEqual(result.code, 99)


class TestChunking(unittest.TestCase):
    """Cover the array-chunking serialization branches."""

    def setUp(self):
        self._orig_max = M.MAX_CHUNK_SIZE

    def tearDown(self):
        M.MAX_CHUNK_SIZE = self._orig_max

    def test_chunk_zero_itemsize_rejected(self):
        with self.assertRaises(ValueError) as cm:
            M._chunk(np.empty((0,), dtype=np.dtype([])))
        self.assertIn("zero itemsize", str(cm.exception))

    def test_chunk_unchunk_round_trip(self):
        M.MAX_CHUNK_SIZE = 8  # force several chunks
        arr = np.arange(100, dtype=np.float64)
        chunked = M._chunk(arr)
        self.assertTrue(chunked["__msgpack_chunked_array__"])
        np.testing.assert_array_equal(M._unchunk(chunked), arr)

    def test_serialize_round_trip_with_chunked_leaf(self):
        M.MAX_CHUNK_SIZE = 8  # trigger chunking inside a dict leaf
        data = {"big": np.arange(50, dtype=np.float64), "small": np.int32(1)}
        out = M._msgpack_restore(M._msgpack_serialize(data))
        np.testing.assert_array_equal(out["big"], data["big"])
        self.assertEqual(out["small"], np.int32(1))

    def test_chunk_array_leaves_top_level_array(self):
        M.MAX_CHUNK_SIZE = 8
        arr = np.arange(20, dtype=np.float64)
        result = M._chunk_array_leaves_in_place(arr)
        self.assertIn("__msgpack_chunked_array__", result)

    def test_unchunk_array_leaves_top_level_and_nested(self):
        M.MAX_CHUNK_SIZE = 8
        arr = np.arange(20, dtype=np.float64)
        chunked = M._chunk(arr)

        # Top-level chunked dict is unchunked directly.
        top = M._unchunk_array_leaves_in_place(dict(chunked))
        self.assertIsInstance(top, np.ndarray)
        np.testing.assert_array_equal(top, arr)

        # A chunked array nested two levels deep is unchunked recursively.
        nested = {"lvl1": {"lvl2": dict(chunked)}}
        M._unchunk_array_leaves_in_place(nested)
        self.assertIsInstance(nested["lvl1"]["lvl2"], np.ndarray)
        np.testing.assert_array_equal(nested["lvl1"]["lvl2"], arr)


class TestRegistry(unittest.TestCase):
    """Cover serialization-registry behavior without leaking global state."""

    def setUp(self):
        self._registered = []

    def tearDown(self):
        # Remove any types we registered to keep the global registry hermetic.
        for ty in self._registered:
            M._STATE_DICT_REGISTRY.pop(ty, None)

    def _register(self, ty, to_sd, from_sd, **kw):
        self._registered.append(ty)
        M.msgpack_register_serialization(ty, to_sd, from_sd, **kw)

    def test_duplicate_registration_raises(self):
        class Foo:
            pass

        self._register(Foo, lambda x: {}, lambda x, sd, mismatch='error': x)
        with self.assertRaises(ValueError) as cm:
            M.msgpack_register_serialization(
                Foo, lambda x: {}, lambda x, sd, mismatch='error': x
            )
        self.assertIn("already registered", str(cm.exception))

    def test_duplicate_registration_override(self):
        class Bar:
            pass

        self._register(Bar, lambda x: {"v": 1}, lambda x, sd, mismatch='error': x)
        # override=True must not raise.
        M.msgpack_register_serialization(
            Bar, lambda x: {"v": 2}, lambda x, sd, mismatch='error': x, override=True
        )
        self.assertEqual(M._STATE_DICT_REGISTRY[Bar][0](None), {"v": 2})

    def test_unregistered_type_passthrough(self):
        class Unregistered:
            pass

        obj = Unregistered()
        # No handler -> to_state_dict returns the object unchanged.
        self.assertIs(M.msgpack_to_state_dict(obj), obj)
        # from_state_dict returns the raw state unchanged.
        self.assertEqual(M.msgpack_from_state_dict(obj, {"x": 1}), {"x": 1})


class TestDictAndListBranches(unittest.TestCase):
    """Cover dict/list/namedtuple/FlattedDict serialization branches."""

    def test_dict_state_dict_str_raises_typeerror(self):
        # A key whose __str__ raises TypeError is re-raised as a ValueError
        # ("Dict contains unhashable keys") by the str()-conversion guard.
        class BadStr:
            def __hash__(self):
                return 1

            def __eq__(self, other):
                return self is other

            def __str__(self):
                raise TypeError("cannot stringify")

        with self.assertRaises(ValueError) as cm:
            M._dict_state_dict({BadStr(): 1})
        self.assertIn("unhashable", str(cm.exception).lower())

    def test_dict_state_dict_key_collision(self):
        # Distinct keys collapsing to the same string trigger the collision branch.
        class SameStr:
            def __init__(self, v):
                self.v = v

            def __str__(self):
                return "dup"

        with self.assertRaises(ValueError) as cm:
            M._dict_state_dict({SameStr(1): "a", SameStr(2): "b"})
        self.assertIn("unique string", str(cm.exception).lower())

    def test_flatteddict_state_dict(self):
        fd = brainstate.util.FlattedDict({('a',): np.arange(3), ('b',): np.arange(2)})
        sd = M._dict_state_dict(fd)
        self.assertIn("a", sd)
        self.assertIn("b", sd)

    def test_restore_dict_with_flatteddict_states(self):
        target = {'a': np.zeros(3), 'b': np.zeros(2)}
        states = brainstate.util.FlattedDict(
            {('a',): np.arange(3), ('b',): np.arange(2)}
        )
        result = M._restore_dict(target, states)
        np.testing.assert_array_equal(result['a'], np.arange(3))
        np.testing.assert_array_equal(result['b'], np.arange(2))

    def test_list_state_dict_round_trip(self):
        target = [10, 20, 30]
        sd = M.msgpack_to_state_dict(target)
        self.assertEqual(sd, {'0': 10, '1': 20, '2': 30})
        restored = M.msgpack_from_state_dict(target, sd)
        self.assertEqual(restored, [10, 20, 30])

    def test_tuple_round_trip(self):
        target = (1, 2, 3)
        sd = M.msgpack_to_state_dict(target)
        restored = M.msgpack_from_state_dict(target, sd)
        self.assertEqual(restored, (1, 2, 3))

    def test_list_target_longer_than_state_dict_pads(self):
        target = [1, 2, 3, 4]
        state = {'0': 10, '1': 20, '2': 30, '3': 40}
        restored = M.msgpack_from_state_dict(target, state)
        self.assertEqual(restored, [10, 20, 30, 40])

    def test_namedtuple_round_trip(self):
        NT = namedtuple("NT", ["a", "b"])
        target = NT(1, 2)
        sd = M.msgpack_to_state_dict(target)
        self.assertEqual(sd, {"a": 1, "b": 2})
        restored = M.msgpack_from_state_dict(target, sd)
        self.assertEqual(restored, NT(1, 2))

    def test_namedtuple_restore_from_name_fields_values_form(self):
        NT = namedtuple("NT", ["a", "b"])
        target = NT(0, 0)
        state = {
            "name": "NT",
            "fields": {"0": "a", "1": "b"},
            "values": {"0": 10, "1": 20},
        }
        restored = M._restore_namedtuple(target, state)
        self.assertEqual(restored, NT(10, 20))

    def test_namedtuple_reconstruction_typeerror_wrapped(self):
        NT = namedtuple("NT", ["a", "b"])

        class StrictNT(NT):
            def __new__(cls, *args, **kwargs):
                # Reject the keyword reconstruction used by _restore_namedtuple.
                if kwargs:
                    raise TypeError("StrictNT does not accept keyword arguments")
                return super().__new__(cls, *args, **kwargs)

        target = StrictNT(1, 2)
        with self.assertRaises(TypeError) as cm:
            M._restore_namedtuple(target, {"a": 10, "b": 20})
        self.assertIn("Failed to reconstruct namedtuple", str(cm.exception))


class TestQuantityAndState(unittest.TestCase):
    """Round-trip BrainUnit quantities and BrainState states through files."""

    def test_quantity_value_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "q.msg")
            data = {"x": np.array([1.0, 2.0, 3.0]) * u.mV}
            M.msgpack_save(fn, data, verbose=False)
            loaded = M.msgpack_load(fn, target=data, verbose=False)
            self.assertIsInstance(loaded["x"], u.Quantity)
            self.assertEqual(loaded["x"].unit, u.mV)
            self.assertTrue(u.math.allclose(loaded["x"], data["x"]))

    def test_quantity_no_target_returns_raw_state_dict(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "q2.msg")
            data = {"x": np.array([1.0, 2.0]) * u.ms}
            M.msgpack_save(fn, data, verbose=False)
            loaded = M.msgpack_load(fn, verbose=False)
            # Without a target, the Quantity comes back as its raw state dict.
            self.assertIn("mantissa", loaded["x"])
            np.testing.assert_array_equal(loaded["x"]["mantissa"], np.array([1.0, 2.0]))

    def test_state_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "s.msg")
            data = {"p": brainstate.ParamState(np.arange(4, dtype=np.float64))}
            M.msgpack_save(fn, data, verbose=False)
            data["p"].value = np.zeros(4)
            loaded = M.msgpack_load(fn, target=data, verbose=False)
            self.assertIsInstance(loaded["p"], brainstate.ParamState)
            np.testing.assert_array_equal(loaded["p"].value, np.arange(4))


class TestPartial(unittest.TestCase):
    """Round-trip a ``jax.tree_util.Partial``."""

    def test_partial_round_trip(self):
        def f(x, y, z=0):
            return x + y + z

        p = jax.tree_util.Partial(f, 1, 2, z=3)
        sd = M.msgpack_to_state_dict(p)
        self.assertEqual(set(sd.keys()), {"args", "keywords"})
        restored = M.msgpack_from_state_dict(p, sd)
        self.assertEqual(restored.args, (1, 2))
        self.assertEqual(restored.keywords, {"z": 3})
        self.assertEqual(restored(), 6)


class TestSaveLoadOptions(unittest.TestCase):
    """Cover ``msgpack_save`` / ``msgpack_load`` option and error branches."""

    def test_save_overwrite_false_on_existing_raises(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "ck.msg")
            M.msgpack_save(fn, {"v": np.arange(3)}, verbose=False)
            with self.assertRaises(M.InvalidCheckpointPath):
                M.msgpack_save(fn, {"v": np.arange(3)}, overwrite=False, verbose=False)

    def test_save_creates_nested_directories(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "a", "b", "c", "ck.msg")
            M.msgpack_save(fn, {"v": np.arange(2)}, verbose=False)
            self.assertTrue(os.path.exists(fn))
            loaded = M.msgpack_load(fn, verbose=False)
            np.testing.assert_array_equal(loaded["v"], np.arange(2))

    def test_save_verbose_true(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "verbose.msg")
            M.msgpack_save(fn, {"v": np.arange(2)}, verbose=True)
            self.assertTrue(os.path.exists(fn))

    def test_save_flatteddict_target(self):
        fd = brainstate.util.FlattedDict(
            {('a',): np.arange(3), ('b',): np.arange(2)}
        )
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "fd.msg")
            M.msgpack_save(fn, fd, verbose=False)
            loaded = M.msgpack_load(fn, verbose=False)
            np.testing.assert_array_equal(loaded["a"], np.arange(3))
            np.testing.assert_array_equal(loaded["b"], np.arange(2))

    def test_load_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError) as cm:
                M.msgpack_load(os.path.join(d, "missing.msg"), verbose=False)
            self.assertIn("Checkpoint not found", str(cm.exception))

    def test_load_parallel_true(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "p.msg")
            M.msgpack_save(fn, {"big": np.arange(1000)}, verbose=False)
            loaded = M.msgpack_load(fn, parallel=True, verbose=False)
            np.testing.assert_array_equal(loaded["big"], np.arange(1000))

    def test_load_non_parallel(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "np.msg")
            M.msgpack_save(fn, {"big": np.arange(1000)}, verbose=False)
            loaded = M.msgpack_load(fn, parallel=False, verbose=False)
            np.testing.assert_array_equal(loaded["big"], np.arange(1000))

    def test_load_verbose_true(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "v.msg")
            M.msgpack_save(fn, {"v": np.arange(3)}, verbose=False)
            loaded = M.msgpack_load(fn, verbose=True)
            np.testing.assert_array_equal(loaded["v"], np.arange(3))

    def test_load_with_mismatch_ignore(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "mm.msg")
            M.msgpack_save(fn, {"a": np.arange(2), "b": np.arange(3)}, verbose=False)
            target = {"a": np.zeros(2), "b": np.zeros(3), "c": np.zeros(1)}
            loaded = M.msgpack_load(fn, target=target, mismatch="ignore", verbose=False)
            np.testing.assert_array_equal(loaded["a"], np.arange(2))
            np.testing.assert_array_equal(loaded["c"], np.zeros(1))


class TestRenameAndSaveHelpers(unittest.TestCase):
    """Cover ``_rename_fn`` edge cases and ``_save_main_ckpt_file`` cleanup."""

    def test_rename_missing_source_is_noop(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "missing.txt")
            dst = os.path.join(d, "dst.txt")
            M._rename_fn(src, dst)  # should silently return
            self.assertFalse(os.path.exists(dst))

    def test_rename_overwrite_replaces(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.txt")
            dst = os.path.join(d, "dst.txt")
            with open(src, "w") as f:
                f.write("new")
            with open(dst, "w") as f:
                f.write("old")
            M._rename_fn(src, dst, overwrite=True)
            self.assertFalse(os.path.exists(src))
            with open(dst) as f:
                self.assertEqual(f.read(), "new")

    def test_save_main_ckpt_file_cleans_tmp_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            # Destination directory does not exist -> the rename fails, and the
            # tmp file should be cleaned up before the exception propagates.
            target = os.path.join(d, "nope_subdir", "f.msg")
            with self.assertRaises(Exception):
                M._save_main_ckpt_file(b"data", target, overwrite=True)
            self.assertFalse(os.path.exists(target + ".tmp"))


class TestAsyncManagerExtra(unittest.TestCase):
    """Cover AsyncManager paths not hit by the async test file."""

    def test_del_without_close(self):
        mgr = M.AsyncManager()
        # Deleting without calling close() exercises __del__'s cleanup branch.
        mgr.__del__()
        self.assertTrue(mgr._closed)

    def test_wait_previous_save_warns_when_pending(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "async.msg")
            mgr = M.AsyncManager()
            try:
                # Two rapid saves: the second triggers the "previous save not
                # finished" warning path inside wait_previous_save.
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    M.msgpack_save(fn, {"big": np.arange(100000)},
                                   async_manager=mgr, verbose=False)
                    M.msgpack_save(fn, {"big": np.arange(100000)},
                                   async_manager=mgr, verbose=False)
                mgr.wait_previous_save()
            finally:
                mgr.close()
            self.assertTrue(os.path.exists(fn))


class TestValidation(unittest.TestCase):
    """Cover mismatch-mode validation entry points."""

    def test_from_state_dict_invalid_mismatch(self):
        with self.assertRaises(ValueError):
            M.msgpack_from_state_dict({"a": 1}, {"a": 1}, mismatch="bogus")

    def test_from_bytes_invalid_mismatch(self):
        encoded = M._to_bytes({"a": np.arange(2)})
        with self.assertRaises(ValueError):
            M._from_bytes({"a": np.zeros(2)}, encoded, mismatch="bogus")

    def test_load_invalid_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "v.msg")
            M.msgpack_save(fn, {"a": np.arange(2)}, verbose=False)
            with self.assertRaises(ValueError):
                M.msgpack_load(fn, target={"a": np.zeros(2)},
                               mismatch="bogus", verbose=False)


if __name__ == "__main__":
    unittest.main()
