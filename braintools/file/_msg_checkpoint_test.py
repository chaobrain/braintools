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


import builtins
import importlib.util
import os
import tempfile
import time
import unittest
import warnings
from collections import namedtuple, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory
from unittest import mock

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintools

spec = importlib.util.find_spec("msgpack")

if spec is None:
    pytest.skip("msgpack not installed", allow_module_level=True)

from braintools.file import _msg_checkpoint as M
from braintools.file import (
    msgpack_save,
    msgpack_load,
    msgpack_to_state_dict,
    msgpack_from_state_dict,
    msgpack_register_serialization,
    AsyncManager,
    AlreadyExistsError,
    InvalidCheckpointPath,
)


class TestMsgCheckpoint(unittest.TestCase):
    def test_checkpoint_quantity(self):
        data = {
            "name": brainstate.random.rand(3) * u.ms,
        }

        with TemporaryDirectory('test_checkpoint_quantity', ignore_cleanup_errors=True) as tmpdirname:
            print(tmpdirname)
            filename = tmpdirname + "/test_msg_checkpoint.msg"
            braintools.file.msgpack_save(filename, data)
            data['name'] += 1 * u.ms

            data2 = braintools.file.msgpack_load(filename, target=data)
            self.assertTrue('name' in data2)
            self.assertTrue(isinstance(data2['name'], u.Quantity))
            self.assertTrue(not u.math.allclose(data['name'], data2['name']))

    def test_checkpoint_state(self):
        data = {
            "a": brainstate.State(brainstate.random.rand(1)),
            "b": brainstate.ShortTermState(brainstate.random.rand(2)),
            "c": brainstate.ParamState(brainstate.random.rand(3)),
        }

        with TemporaryDirectory('test_checkpoint_state', ignore_cleanup_errors=True) as tmpdirname:
            filename = tmpdirname + "/test_msg_checkpoint.msg"
            braintools.file.msgpack_save(filename, data)

            data2 = braintools.file.msgpack_load(filename, target=data)
            self.assertTrue('a' in data2)
            self.assertTrue('b' in data2)
            self.assertTrue('c' in data2)
            self.assertTrue(isinstance(data2['a'], brainstate.State))
            self.assertTrue(isinstance(data2['b'], brainstate.ShortTermState))
            self.assertTrue(isinstance(data2['c'], brainstate.ParamState))
            self.assertTrue(u.math.allclose(data['a'].value, data2['a'].value))
            self.assertTrue(u.math.allclose(data['b'].value, data2['b'].value))
            self.assertTrue(u.math.allclose(data['c'].value, data2['c'].value))

    def test_flatteneddict(self):
        net = brainstate.nn.Sequential(
            brainstate.nn.Linear(10, 20),
            brainstate.nn.ReLU(),
            brainstate.nn.Linear(20, 5)
        )
        net = brainstate.nn.init_all_states(net)

        data = {'state': net.states(), 'version': 1.0}

        with TemporaryDirectory('test_flatteneddict', ignore_cleanup_errors=True) as tmpdirname:
            filename = tmpdirname + "/test_msg_checkpoint.msg"
            braintools.file.msgpack_save(filename, data)
            loaded = braintools.file.msgpack_load(filename)
            print(loaded)
            loaded2 = braintools.file.msgpack_load(filename, target=data)


class TestMismatchSettings(unittest.TestCase):
    """Test cases for mismatch setting support in from_state_dict and msgpack_load"""

    def test_dict_mismatch_error(self):
        """Test that mismatch='error' raises ValueError for dictionary mismatches"""
        target = {'a': 1, 'b': 2, 'c': 3}
        state_dict = {'a': 10, 'b': 20}  # Missing 'c'

        with self.assertRaises(ValueError) as cm:
            braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='error')

        self.assertIn('do not match', str(cm.exception))
        self.assertIn('path', str(cm.exception))

    def test_dict_mismatch_warn(self):
        """Test that mismatch='warn' issues warning and preserves missing keys"""
        target = {'a': 1, 'b': 2, 'c': 3}
        state_dict = {'a': 10, 'b': 20}  # Missing 'c'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='warn')

            # Should issue warning
            self.assertEqual(len(w), 1)
            self.assertIn('do not match', str(w[0].message))

            # Should preserve missing key from target
            self.assertEqual(result, {'a': 10, 'b': 20, 'c': 3})

    def test_dict_mismatch_ignore(self):
        """Test that mismatch='ignore' silently preserves missing keys"""
        target = {'a': 1, 'b': 2, 'c': 3}
        state_dict = {'a': 10, 'b': 20}  # Missing 'c'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='ignore')

            # Should not issue warning
            self.assertEqual(len(w), 0)

            # Should preserve missing key from target
            self.assertEqual(result, {'a': 10, 'b': 20, 'c': 3})

    def test_list_mismatch_error(self):
        """Test that mismatch='error' raises ValueError for list size mismatches"""
        target = [1, 2, 3]
        state_dict = {'0': 10, '1': 20}  # Missing index '2'

        with self.assertRaises(ValueError) as cm:
            braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='error')

        self.assertIn('size', str(cm.exception))
        self.assertIn('do not match', str(cm.exception))

    def test_list_mismatch_warn(self):
        """Test that mismatch='warn' issues warning and preserves missing elements"""
        target = [1, 2, 3]
        state_dict = {'0': 10, '1': 20}  # Missing index '2'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='warn')

            # Should issue warning
            self.assertEqual(len(w), 1)
            self.assertIn('size', str(w[0].message))

            # Should preserve missing element from target
            self.assertEqual(result, [10, 20, 3])

    def test_list_mismatch_ignore(self):
        """Test that mismatch='ignore' silently preserves missing elements"""
        target = [1, 2, 3]
        state_dict = {'0': 10, '1': 20}  # Missing index '2'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='ignore')

            # Should not issue warning
            self.assertEqual(len(w), 0)

            # Should preserve missing element from target
            self.assertEqual(result, [10, 20, 3])

    def test_namedtuple_mismatch_error(self):
        """Test that mismatch='error' raises ValueError for namedtuple field mismatches"""
        TestTuple = namedtuple('TestTuple', ['a', 'b', 'c'])
        target = TestTuple(a=1, b=2, c=3)
        state_dict = {'a': 10, 'b': 20}  # Missing 'c'

        with self.assertRaises(ValueError) as cm:
            braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='error')

        self.assertIn('field names', str(cm.exception))
        self.assertIn('do not match', str(cm.exception))

    def test_namedtuple_mismatch_warn(self):
        """Test that mismatch='warn' issues warning and preserves missing fields"""
        TestTuple = namedtuple('TestTuple', ['a', 'b', 'c'])
        target = TestTuple(a=1, b=2, c=3)
        state_dict = {'a': 10, 'b': 20}  # Missing 'c'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='warn')

            # Should issue warning
            self.assertEqual(len(w), 1)
            self.assertIn('field names', str(w[0].message))

            # Should preserve missing field from target
            self.assertEqual(result, TestTuple(a=10, b=20, c=3))

    def test_namedtuple_mismatch_ignore(self):
        """Test that mismatch='ignore' silently preserves missing fields"""
        TestTuple = namedtuple('TestTuple', ['a', 'b', 'c'])
        target = TestTuple(a=1, b=2, c=3)
        state_dict = {'a': 10, 'b': 20}  # Missing 'c'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='ignore')

            # Should not issue warning
            self.assertEqual(len(w), 0)

            # Should preserve missing field from target
            self.assertEqual(result, TestTuple(a=10, b=20, c=3))

    def test_quantity_mismatch_error(self):
        """Test that mismatch='error' raises ValueError for unit mismatches"""
        target = 1.0 * u.ms
        state_dict = {
            'mantissa': 2.0,
            'scale': u.second.scale,
            'base': u.second.base,
            'dim': u.second.dim._dims,
            'factor': u.second.factor
        }  # Different unit (second instead of millisecond)

        with self.assertRaises(ValueError) as cm:
            braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='error')

        self.assertIn('Unit mismatch', str(cm.exception))

    def test_quantity_mismatch_warn(self):
        """Test that mismatch='warn' issues warning for unit mismatches"""
        target = 1.0 * u.ms
        state_dict = {
            'mantissa': 2.0,
            'scale': u.second.scale,
            'base': u.second.base,
            'dim': u.second.dim._dims,
            'factor': u.second.factor
        }  # Different unit (second instead of millisecond)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='warn')

            # Should issue warning
            self.assertEqual(len(w), 1)
            self.assertIn('Unit mismatch', str(w[0].message))

            # Should use the loaded unit
            self.assertEqual(result.mantissa, 2.0)
            self.assertEqual(result.unit, u.second)

    def test_quantity_mismatch_ignore(self):
        """Test that mismatch='ignore' silently handles unit mismatches"""
        target = 1.0 * u.ms
        state_dict = {
            'mantissa': 2.0,
            'scale': u.second.scale,
            'base': u.second.base,
            'dim': u.second.dim._dims,
            'factor': u.second.factor
        }  # Different unit (second instead of millisecond)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='ignore')

            # Should not issue warning
            self.assertEqual(len(w), 0)

            # Should use the loaded unit
            self.assertEqual(result.mantissa, 2.0)
            self.assertEqual(result.unit, u.second)

    def test_mismatch_parameter_validation(self):
        """Test that invalid mismatch parameters are handled correctly"""
        target = {'a': 1, 'b': 2, 'c': 3}
        state_dict = {'a': 10, 'b': 20}  # Missing 'c'

        # Test invalid mismatch parameter - should default to error behavior
        with self.assertRaises(ValueError):
            braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='invalid_mode')

    def test_nested_structure_mismatch(self):
        """Test mismatch handling with nested data structures"""
        target = {
            'model': {'weights': [1, 2, 3], 'bias': 0.1},
            'config': {'learning_rate': 0.01, 'batch_size': 32}
        }
        state_dict = {
            'model': {'weights': {'0': 10, '1': 20}, 'bias': 0.5},  # Missing element in weights
            'config': {'learning_rate': 0.001}  # Missing batch_size
        }

        # Test with warn mode - should handle nested mismatches
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = braintools.file.msgpack_from_state_dict(target, state_dict, mismatch='warn')

            # Should issue warnings for both list and dict mismatches
            self.assertGreaterEqual(len(w), 2)

            # Should preserve original values for missing items
            expected = {
                'model': {'weights': [10, 20, 3], 'bias': 0.5},
                'config': {'learning_rate': 0.001, 'batch_size': 32}
            }
            self.assertEqual(result, expected)


class TestMemoryAndPlatformHandling(unittest.TestCase):
    """Test memory handling and platform-specific behavior."""

    def test_msgpack_size_validation(self):
        """Test that msgpack_restore validates data size."""
        from braintools.file._msg_checkpoint import _msgpack_restore

        # Only ``len`` is checked, so a small payload over a small limit takes
        # the same path. (This test used to allocate 11 GB, which dominated the
        # suite's peak memory and could crash WSL.)
        large_data = b'x' * 11

        with self.assertRaises(ValueError) as ctx:
            _msgpack_restore(large_data, max_size=10)

        self.assertIn("too large", str(ctx.exception).lower())

    def test_rename_overwrite(self):
        """Test _rename_fn with overwrite=True."""
        from braintools.file._msg_checkpoint import _rename_fn

        with TemporaryDirectory('test_rename', ignore_cleanup_errors=True) as tmpdir:
            src = tmpdir + "/source.txt"
            dst = tmpdir + "/dest.txt"

            # Create source and destination files
            with open(src, 'w') as f:
                f.write("source content")
            with open(dst, 'w') as f:
                f.write("old dest content")

            # Rename with overwrite should succeed
            _rename_fn(src, dst, overwrite=True)

            self.assertFalse(os.path.exists(src))
            self.assertTrue(os.path.exists(dst))

            with open(dst, 'r') as f:
                content = f.read()
            self.assertEqual(content, "source content")

    def test_rename_no_overwrite(self):
        """Test _rename_fn with overwrite=False raises error."""
        from braintools.file._msg_checkpoint import _rename_fn, AlreadyExistsError

        with TemporaryDirectory('test_rename_no_overwrite', ignore_cleanup_errors=True) as tmpdir:
            src = tmpdir + "/source.txt"
            dst = tmpdir + "/dest.txt"

            # Create source and destination files
            with open(src, 'w') as f:
                f.write("source")
            with open(dst, 'w') as f:
                f.write("dest")

            # Should raise AlreadyExistsError
            with self.assertRaises(AlreadyExistsError):
                _rename_fn(src, dst, overwrite=False)

    def test_chunk_size_warning(self):
        """Test that inefficient chunking triggers a warning."""
        import numpy as np
        from braintools.file._msg_checkpoint import _chunk

        # Create array with large dtype that would result in small chunksize
        # This is a bit artificial, but tests the warning logic
        large_array = np.zeros((2000, 2000), dtype=np.float64)

        # Should not warn for normal arrays
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _chunk(large_array)
            # Should not have warnings for normal case
            chunk_warnings = [warning for warning in w if "inefficient" in str(warning.message).lower()]
            self.assertEqual(len(chunk_warnings), 0)

    def test_dict_key_collision(self):
        """Test improved error message for dict key collisions."""
        from braintools.file._msg_checkpoint import _dict_state_dict

        # Create dict with keys that have same string representation
        class CustomKey:
            def __init__(self, value):
                self.value = value
            def __str__(self):
                return "same"  # All instances have same string repr

        bad_dict = {CustomKey(1): "a", CustomKey(2): "b"}

        with self.assertRaises(ValueError) as ctx:
            _dict_state_dict(bad_dict)

        # Should mention collisions
        self.assertIn("collision", str(ctx.exception).lower())

    def test_namedtuple_reconstruction_error(self):
        """Test improved error message for namedtuple reconstruction."""
        from braintools.file._msg_checkpoint import _restore_namedtuple

        Point = namedtuple('Point', ['x', 'y'])
        target = Point(1, 2)

        # Create state dict with wrong field types that will cause TypeError
        # This is tricky - we need to cause a TypeError in the reconstruction
        # For now, just verify the function works correctly with valid data
        state_dict = {'x': 10, 'y': 20}
        result = _restore_namedtuple(target, state_dict)
        self.assertEqual(result.x, 10)
        self.assertEqual(result.y, 20)

    def test_windows_path_length_warning(self):
        """Test Windows path length warning (conditional on platform)."""
        import sys
        if sys.platform != 'win32':
            self.skipTest("Windows-specific test")

        # Create a very long path
        long_path = "x" * 300 + ".msgpack"
        data = {"value": 123}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                braintools.file.msgpack_save(long_path, data, verbose=False)
            except Exception:
                pass  # May fail due to actual path issues

            # Check if warning was issued
            path_warnings = [warning for warning in w if "MAX_PATH" in str(warning.message)]
            if len(path_warnings) > 0:
                self.assertIn("260", str(path_warnings[0].message))


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


# msgpack availability is enforced by the module-level skip above.
class TestAsyncManager(unittest.TestCase):
    """Test suite for AsyncManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.test_path = self.test_dir.name

    def tearDown(self):
        """Clean up test fixtures."""
        self.test_dir.cleanup()

    def test_async_save_basic(self):
        """Test basic async save functionality."""
        filename = os.path.join(self.test_path, "test_async_basic.msgpack")
        data = {"a": brainstate.random.rand(10), "b": 42}

        with AsyncManager() as manager:
            msgpack_save(filename, data, async_manager=manager, verbose=False)
            # Wait for save to complete
            manager.wait_previous_save()

        # Verify file was created and can be loaded
        self.assertTrue(os.path.exists(filename))
        loaded_data = msgpack_load(filename, verbose=False)
        self.assertEqual(loaded_data["b"], 42)

    def test_async_save_wait(self):
        """Test wait_previous_save() blocks until save completes."""
        filename = os.path.join(self.test_path, "test_async_wait.msgpack")
        # Create larger data to ensure save takes some time
        data = {"array": brainstate.random.rand(1000, 1000)}

        manager = AsyncManager()
        start_time = time.time()
        msgpack_save(filename, data, async_manager=manager, verbose=False)

        # Save should be running in background
        self.assertIsNotNone(manager.save_future)

        # Wait should block until complete
        manager.wait_previous_save()
        elapsed = time.time() - start_time

        # Verify save completed
        self.assertTrue(manager.save_future.done())
        self.assertTrue(os.path.exists(filename))
        manager.close()

    def test_async_manager_context_manager(self):
        """Test AsyncManager as context manager."""
        filename = os.path.join(self.test_path, "test_context.msgpack")
        data = {"value": 123}

        with AsyncManager() as manager:
            msgpack_save(filename, data, async_manager=manager, verbose=False)
            # Context manager should wait for completion on exit

        # After exiting context, save should be complete
        self.assertTrue(os.path.exists(filename))
        loaded = msgpack_load(filename, verbose=False)
        self.assertEqual(loaded["value"], 123)

    def test_async_manager_close(self):
        """Test explicit close() of AsyncManager."""
        filename = os.path.join(self.test_path, "test_close.msgpack")
        data = {"value": 456}

        manager = AsyncManager()
        msgpack_save(filename, data, async_manager=manager, verbose=False)

        # Explicit close should wait for save to complete
        manager.close()

        # Should be able to call close multiple times
        manager.close()  # Should not raise

        self.assertTrue(os.path.exists(filename))

    def test_async_manager_reuse_after_close(self):
        """Test that using AsyncManager after close raises error."""
        manager = AsyncManager()
        manager.close()

        with self.assertRaises(RuntimeError) as ctx:
            manager.save_async(lambda: None)

        self.assertIn("closed", str(ctx.exception).lower())

    def test_async_save_concurrent(self):
        """Test thread safety with concurrent saves."""
        # This test ensures the threading.Lock prevents race conditions
        num_saves = 5
        manager = AsyncManager(max_workers=2)

        def save_data(i):
            filename = os.path.join(self.test_path, f"test_concurrent_{i}.msgpack")
            data = {"index": i, "array": brainstate.random.rand(100)}
            msgpack_save(filename, data, async_manager=manager, verbose=False)

        # Launch multiple saves concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(save_data, i) for i in range(num_saves)]
            # Wait for all to complete
            for future in futures:
                future.result()

        manager.close()

        # Verify all files were created correctly
        for i in range(num_saves):
            filename = os.path.join(self.test_path, f"test_concurrent_{i}.msgpack")
            self.assertTrue(os.path.exists(filename))
            loaded = msgpack_load(filename, verbose=False)
            self.assertEqual(loaded["index"], i)

    def test_async_save_race_condition_stress(self):
        """Stress test for race conditions in AsyncManager."""
        # Rapid-fire saves to try to trigger race conditions
        manager = AsyncManager()
        num_rapid_saves = 10

        for i in range(num_rapid_saves):
            filename = os.path.join(self.test_path, f"test_stress_{i}.msgpack")
            data = {"value": i}
            msgpack_save(filename, data, async_manager=manager, verbose=False)

        # Wait for all saves to complete
        manager.wait_previous_save()
        manager.close()

        # Verify all saves completed successfully
        for i in range(num_rapid_saves):
            filename = os.path.join(self.test_path, f"test_stress_{i}.msgpack")
            self.assertTrue(os.path.exists(filename))

    def test_async_manager_sequential_saves(self):
        """Test that sequential saves wait for previous save."""
        manager = AsyncManager()
        files = []

        for i in range(3):
            filename = os.path.join(self.test_path, f"test_seq_{i}.msgpack")
            files.append(filename)
            data = {"iteration": i, "data": brainstate.random.rand(100)}
            msgpack_save(filename, data, async_manager=manager, verbose=False)

        manager.close()

        # All files should exist
        for filename in files:
            self.assertTrue(os.path.exists(filename))


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

    def test_quantity_keeps_unit_display_name(self):
        # The unit rebuilt from (dim, scale, base, factor) is equal but unnamed;
        # loading against a target must give back the named unit.
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "q_name.msg")
            data = {"tau": 10.0 * u.ms, "R": 100.0 * u.Mohm}
            M.msgpack_save(fn, data, verbose=False)
            loaded = M.msgpack_load(fn, target=data, verbose=False)
            self.assertEqual(str(loaded["tau"].unit), str(u.ms))
            self.assertEqual(str(loaded["R"].unit), str(u.Mohm))
            self.assertEqual(str(loaded["tau"]), str(data["tau"]))

    def test_prng_key_round_trip_resumes_random_stream(self):
        # Typed PRNG keys (brainstate.random states) used to fail with
        # "JAX array with PRNGKey dtype cannot be converted to a NumPy array".
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "rng.msg")
            brainstate.random.seed(7)
            M.msgpack_save(fn, {"rng": brainstate.random.DEFAULT, "key": jax.random.key(3)})
            expected = brainstate.random.rand(4)
            brainstate.random.seed(123)  # scramble the global stream
            out = M.msgpack_load(fn, target={"rng": brainstate.random.DEFAULT, "key": jax.random.key(0)})
            np.testing.assert_allclose(brainstate.random.rand(4), expected)
            self.assertTrue(jax.dtypes.issubdtype(out["key"].dtype, jax.dtypes.prng_key))
            np.testing.assert_array_equal(jax.random.key_data(out["key"]),
                                          jax.random.key_data(jax.random.key(3)))
            raw = M.msgpack_load(fn)
            self.assertEqual(raw["key"]["__prng_key_impl__"], "threefry2x32")

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
