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

"""Comprehensive tests for ``braintools.file``.

These complement the existing ``_matfile_extra_test.py``,
``_msg_checkpoint_test.py``, ``_msg_checkpoint_async_test.py`` and
``_msg_checkpoint_extra_test.py`` files, focusing on behaviour added or changed
by the file-module fixes: ``save_matfile``, MATLAB v7.3 detection, the
deprecated/inverted ``header_info`` flag, array shape-mismatch validation,
``max_size`` plumbing, most-specific registry dispatch, the namedtuple envelope
guard, ``msgpack_save`` returning its filename, the parallel-read path, and
several error branches.
"""

import builtins
import importlib.util
import os
import tempfile
import unittest
import warnings
from collections import namedtuple, OrderedDict
from unittest import mock

import numpy as np
import pytest

spec = importlib.util.find_spec("msgpack")
if spec is None:
    pytest.skip("msgpack not installed", allow_module_level=True)

import jax
import jax.numpy as jnp
import brainstate
import brainunit as u
from scipy.io import savemat

from braintools.file import _matfile as MF
from braintools.file import _msg_checkpoint as M
from braintools.file import (
    load_matfile,
    save_matfile,
    msgpack_save,
    msgpack_load,
    msgpack_to_state_dict,
    msgpack_from_state_dict,
    msgpack_register_serialization,
    AsyncManager,
    AlreadyExistsError,
    InvalidCheckpointPath,
)


# --------------------------------------------------------------------------- #
# MATLAB I/O
# --------------------------------------------------------------------------- #
class TestSaveMatfile(unittest.TestCase):
    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "out.mat")
            save_matfile(fn, {"x": np.arange(3), "label": "trial-1"})
            self.assertTrue(os.path.exists(fn))
            out = load_matfile(fn, verbose=False)
            np.testing.assert_array_equal(out["x"], np.arange(3))
            self.assertEqual(out["label"], "trial-1")

    def test_save_matfile_with_kwargs(self):
        # ``do_compression`` is a genuine savemat kwarg; exercises **kwargs.
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "c.mat")
            save_matfile(fn, {"x": np.ones(100)}, do_compression=True)
            out = load_matfile(fn, verbose=False)
            np.testing.assert_array_equal(out["x"], np.ones(100))

    def test_save_matfile_verbose(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "v.mat")
            save_matfile(fn, {"x": np.arange(2)}, verbose=True)
            self.assertTrue(os.path.exists(fn))

    def test_save_matfile_bad_filename_type(self):
        with self.assertRaises(TypeError) as cm:
            save_matfile(123, {"x": np.arange(2)})
        self.assertIn("string or path-like", str(cm.exception))

    def test_save_matfile_bad_data_type(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TypeError) as cm:
                save_matfile(os.path.join(d, "x.mat"), [1, 2, 3])
            self.assertIn("dict", str(cm.exception))

    def test_save_matfile_failure_wrapped(self):
        # Saving into a non-existent directory makes scipy raise; it is wrapped.
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "no_such_dir", "x.mat")
            with self.assertRaises(ValueError) as cm:
                save_matfile(bad, {"x": np.arange(2)})
            self.assertIn("Failed to save MATLAB file", str(cm.exception))


class TestLoadMatfileV73AndDeprecation(unittest.TestCase):
    def test_v73_hdf5_detection(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "v73.mat")
            with open(fn, "wb") as f:
                f.write(MF._HDF5_MAGIC + b"\x00" * 64)
            with self.assertRaises(NotImplementedError) as cm:
                load_matfile(fn, verbose=False)
            self.assertIn("v7.3", str(cm.exception))

    def test_deprecated_header_info_true_excludes(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "h.mat")
            savemat(fn, {"a": np.array([1, 2, 3])})
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = load_matfile(fn, header_info=True, verbose=False)
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
            # old header_info=True == include_header=False -> headers excluded
            self.assertNotIn("__header__", out)
            self.assertIn("a", out)

    def test_deprecated_header_info_false_includes(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "h2.mat")
            savemat(fn, {"a": np.array([1, 2, 3])})
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = load_matfile(fn, header_info=False, verbose=False)
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
            self.assertIn("__header__", out)

    def test_include_header_true(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "h3.mat")
            savemat(fn, {"a": np.array([1, 2, 3])})
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = load_matfile(fn, include_header=True, verbose=False)
            # No deprecation warning for the new flag.
            self.assertFalse(any(issubclass(x.category, DeprecationWarning) for x in w))
            self.assertIn("__header__", out)


# --------------------------------------------------------------------------- #
# Checkpoint: array shape-mismatch validation
# --------------------------------------------------------------------------- #
class TestArrayShapeMismatch(unittest.TestCase):
    def _save(self, d, value):
        fn = os.path.join(d, "a.msg")
        msgpack_save(fn, {"w": value}, verbose=False)
        return fn

    def test_shape_mismatch_error(self):
        with tempfile.TemporaryDirectory() as d:
            fn = self._save(d, np.arange(10, dtype=np.float32))
            with self.assertRaises(ValueError) as cm:
                msgpack_load(fn, target={"w": np.zeros(3)}, mismatch="error", verbose=False)
            self.assertIn("shape mismatch", str(cm.exception).lower())

    def test_shape_mismatch_warn_uses_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            fn = self._save(d, np.arange(10, dtype=np.float32))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = msgpack_load(fn, target={"w": np.zeros(3)}, mismatch="warn", verbose=False)
            self.assertTrue(any("shape mismatch" in str(x.message).lower() for x in w))
            self.assertEqual(out["w"].shape, (10,))

    def test_shape_mismatch_ignore(self):
        with tempfile.TemporaryDirectory() as d:
            fn = self._save(d, np.arange(10, dtype=np.float32))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = msgpack_load(fn, target={"w": np.zeros(3)}, mismatch="ignore", verbose=False)
            self.assertEqual(len(w), 0)
            self.assertEqual(out["w"].shape, (10,))

    def test_dtype_difference_allowed(self):
        # Same shape but different dtype must load without error (coercion).
        with tempfile.TemporaryDirectory() as d:
            fn = self._save(d, np.arange(5, dtype=np.int64))
            out = msgpack_load(fn, target={"w": np.zeros(5, dtype=np.float64)},
                               mismatch="error", verbose=False)
            np.testing.assert_array_equal(out["w"], np.arange(5))

    def test_jax_array_target_shape_check(self):
        with tempfile.TemporaryDirectory() as d:
            fn = self._save(d, np.arange(4))
            with self.assertRaises(ValueError):
                msgpack_load(fn, target={"w": jnp.zeros(2)}, mismatch="error", verbose=False)
            out = msgpack_load(fn, target={"w": jnp.zeros(4)}, mismatch="error", verbose=False)
            np.testing.assert_array_equal(out["w"], np.arange(4))


# --------------------------------------------------------------------------- #
# Checkpoint: max_size, return value, registry, namedtuple envelope
# --------------------------------------------------------------------------- #
class TestMaxSize(unittest.TestCase):
    def test_max_size_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "m.msg")
            msgpack_save(fn, {"a": np.arange(1000)}, verbose=False)
            with self.assertRaises(ValueError) as cm:
                msgpack_load(fn, max_size=10, verbose=False)
            self.assertIn("too large", str(cm.exception).lower())

    def test_max_size_none_unlimited(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "m2.msg")
            msgpack_save(fn, {"a": np.arange(1000)}, verbose=False)
            out = msgpack_load(fn, max_size=None, verbose=False)
            np.testing.assert_array_equal(out["a"], np.arange(1000))


class TestSaveReturnsFilename(unittest.TestCase):
    def test_returns_filename(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "r.msg")
            ret = msgpack_save(fn, {"a": np.arange(3)}, verbose=False)
            self.assertEqual(ret, fn)

    def test_returns_filename_for_pathlike(self):
        import pathlib
        with tempfile.TemporaryDirectory() as d:
            fn = pathlib.Path(d) / "r2.msg"
            ret = msgpack_save(fn, {"a": np.arange(3)}, verbose=False)
            self.assertEqual(ret, os.fspath(fn))


class TestRegistryMostSpecific(unittest.TestCase):
    def setUp(self):
        self._registered = []

    def tearDown(self):
        for ty in self._registered:
            M._STATE_DICT_REGISTRY.pop(ty, None)

    def _register(self, ty, to_sd, from_sd):
        self._registered.append(ty)
        msgpack_register_serialization(ty, to_sd, from_sd)

    def test_most_specific_subclass_wins(self):
        class Base:
            def __init__(self, v):
                self.v = v

        class Derived(Base):
            pass

        self._register(Base, lambda x: {"kind": "base"}, lambda x, sd, mm="error": sd)
        self._register(Derived, lambda x: {"kind": "derived"}, lambda x, sd, mm="error": sd)

        self.assertEqual(msgpack_to_state_dict(Base(1)), {"kind": "base"})
        self.assertEqual(msgpack_to_state_dict(Derived(1)), {"kind": "derived"})

    def test_to_state_dict_non_string_key_raises(self):
        class Weird:
            pass

        self._register(Weird, lambda x: {1: "a"}, lambda x, sd, mm="error": sd)
        with self.assertRaises(TypeError) as cm:
            msgpack_to_state_dict(Weird())
        self.assertIn("string keys", str(cm.exception))


class TestNamedtupleEnvelope(unittest.TestCase):
    def test_name_fields_values_namedtuple_roundtrip(self):
        # A namedtuple whose fields are exactly name/fields/values must not be
        # mistaken for the legacy envelope form.
        NT = namedtuple("NT", ["name", "fields", "values"])
        target = NT(name=np.arange(2), fields=np.arange(3), values=np.arange(4))
        sd = msgpack_to_state_dict(target)
        restored = msgpack_from_state_dict(target, sd)
        np.testing.assert_array_equal(restored.name, np.arange(2))
        np.testing.assert_array_equal(restored.values, np.arange(4))

    def test_legacy_envelope_form_still_supported(self):
        NT = namedtuple("NT", ["a", "b"])
        target = NT(0, 0)
        state = {"name": "NT", "fields": {"0": "a", "1": "b"},
                 "values": {"0": 10, "1": 20}}
        restored = M._restore_namedtuple(target, state)
        self.assertEqual(restored, NT(10, 20))


# --------------------------------------------------------------------------- #
# Checkpoint: parallel-read path and error branches
# --------------------------------------------------------------------------- #
class TestParallelReadPath(unittest.TestCase):
    def test_parallel_path_multiple_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "p.msg")
            msgpack_save(fn, {"a": np.arange(5000)}, verbose=False)
            # Force a tiny buffer so the file spans several parallel chunks.
            with mock.patch.object(M, "PARALLEL_READ_BUF_SIZE", 64):
                out = msgpack_load(fn, parallel=True, verbose=False)
            np.testing.assert_array_equal(out["a"], np.arange(5000))

    def test_parallel_memoryerror_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "p2.msg")
            msgpack_save(fn, {"a": np.arange(5000)}, verbose=False)

            real_bytearray = builtins.bytearray

            def fake_bytearray(*args, **kwargs):
                if args and isinstance(args[0], int) and args[0] > 50:
                    raise MemoryError("simulated")
                return real_bytearray(*args, **kwargs)

            with mock.patch.object(M, "PARALLEL_READ_BUF_SIZE", 64), \
                    mock.patch.object(builtins, "bytearray", fake_bytearray), \
                    warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = msgpack_load(fn, parallel=True, verbose=True)
            np.testing.assert_array_equal(out["a"], np.arange(5000))
            self.assertTrue(any("sequential read" in str(x.message).lower() for x in w))


class TestCheckMsgpack(unittest.TestCase):
    def test_check_msgpack_raises_when_missing(self):
        with mock.patch.object(M, "msgpack", None):
            with self.assertRaises(ModuleNotFoundError) as cm:
                M.check_msgpack()
            self.assertIn("install msgpack", str(cm.exception).lower())


class TestExceptionsAndOverwrite(unittest.TestCase):
    def test_exceptions_importable(self):
        self.assertTrue(issubclass(AlreadyExistsError, Exception))
        self.assertTrue(issubclass(InvalidCheckpointPath, Exception))

    def test_overwrite_false_raises(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "ck.msg")
            msgpack_save(fn, {"a": np.arange(2)}, verbose=False)
            with self.assertRaises(InvalidCheckpointPath):
                msgpack_save(fn, {"a": np.arange(2)}, overwrite=False, verbose=False)

    def test_already_exists_error_on_rename(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "s")
            dst = os.path.join(d, "dst")
            with open(src, "w") as f:
                f.write("a")
            with open(dst, "w") as f:
                f.write("b")
            with self.assertRaises(AlreadyExistsError):
                M._rename_fn(src, dst, overwrite=False)


class TestAsyncDelWarn(unittest.TestCase):
    def test_del_warns_when_close_fails(self):
        mgr = AsyncManager()
        try:
            with mock.patch.object(mgr, "close", side_effect=RuntimeError("boom")):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    mgr.__del__()
                self.assertTrue(any("cleanup" in str(x.message).lower() for x in w))
        finally:
            # close was patched only inside the with; perform a real close now.
            mgr._closed = False
            mgr.close()


class TestStateInPlaceRestore(unittest.TestCase):
    def test_state_restored_in_place(self):
        # Documents the in-place restore behaviour of State leaves.
        st = brainstate.ParamState(np.arange(4, dtype=np.float64))
        data = {"p": st}
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "s.msg")
            msgpack_save(fn, data, verbose=False)
            st.value = np.zeros(4)
            out = msgpack_load(fn, target=data, verbose=False)
            self.assertIs(out["p"], st)  # same object, mutated in place
            np.testing.assert_array_equal(st.value, np.arange(4))


class TestDictSubclassRoundTrip(unittest.TestCase):
    def test_ordereddict_target_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "od.msg")
            data = OrderedDict([("a", np.arange(2)), ("b", np.arange(3))])
            msgpack_save(fn, data, verbose=False)
            out = msgpack_load(fn, target=OrderedDict([("a", np.zeros(2)),
                                                       ("b", np.zeros(3))]),
                               verbose=False)
            np.testing.assert_array_equal(out["a"], np.arange(2))
            np.testing.assert_array_equal(out["b"], np.arange(3))


class TestLowLevelExtra(unittest.TestCase):
    def test_ndarray_to_bytes_accepts_jax_array(self):
        b = M._ndarray_to_bytes(jnp.arange(3))
        restored = M._ndarray_from_bytes(b)
        np.testing.assert_array_equal(restored, np.arange(3))

    def test_save_main_ckpt_file_removes_tmp_on_rename_failure(self):
        with tempfile.TemporaryDirectory() as d:
            dst = os.path.join(d, "ck.msg")
            with open(dst, "wb") as f:
                f.write(b"existing")
            # overwrite=False -> rename raises AlreadyExistsError after the tmp
            # file is written; the tmp must be cleaned up before the re-raise.
            with self.assertRaises(AlreadyExistsError):
                M._save_main_ckpt_file(b"data", dst, overwrite=False)
            leftovers = [p for p in os.listdir(d) if p.startswith("ck.msg.tmp")]
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
