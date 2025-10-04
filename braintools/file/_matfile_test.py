# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import types
import unittest
from typing import Any, Dict

import numpy as np
import pytest

from braintools.file import matfile as matfile_mod


class DummyMatStruct:
    """Mimic scipy.io.matlab.mio5_params.mat_struct for testing."""

    def __init__(self, fields: Dict[str, Any]):
        self._fieldnames = list(fields.keys())
        for k, v in fields.items():
            setattr(self, k, v)


class DummySio:
    """A minimal stub of scipy.io with nested matlab.mio5_params.mat_struct."""

    def __init__(self, return_dict: Dict[str, Any]):
        self._return_dict = return_dict

        # Build nested namespaces: sio.matlab.mio5_params.mat_struct
        matlab_ns = types.SimpleNamespace()
        mio5_params_ns = types.SimpleNamespace()
        mio5_params_ns.mat_struct = DummyMatStruct
        matlab_ns.mio5_params = mio5_params_ns
        self.matlab = matlab_ns

    def loadmat(self, *_args, **_kwargs):
        # loadmat returns a dict-like object from a .mat file; we return the stub
        return dict(self._return_dict)


@pytest.mark.skip(reason="not implemented")
class TestLoadMatfile(unittest.TestCase):
    def setUp(self):
        # Backup any existing global sio in the module
        self._orig_sio = getattr(matfile_mod, "sio", None)

    def tearDown(self):
        # Restore original sio to avoid leaking state across tests
        matfile_mod.sio = self._orig_sio

    def test_excludes_header_keys_by_default(self):
        # Arrange: stub loadmat to return header keys + data
        data = {
            "__header__": "header-bytes",
            "__version__": "1.0",
            "__globals__": [],
            "a": 42,
            "b": np.array([1.0, 2.0]),
        }
        matfile_mod.sio = DummySio(return_dict=data)

        # Act
        out = matfile_mod.load_matfile("ignored.mat")

        # Assert
        self.assertIn("a", out)
        self.assertIn("b", out)
        self.assertNotIn("__header__", out)
        self.assertNotIn("__version__", out)
        self.assertNotIn("__globals__", out)
        self.assertEqual(out["a"], 42)
        np.testing.assert_array_equal(out["b"], np.array([1.0, 2.0]))

    def test_includes_header_keys_when_requested(self):
        # Arrange
        data = {
            "__header__": "header-bytes",
            "a": 1,
        }
        matfile_mod.sio = DummySio(return_dict=data)

        # Act
        out = matfile_mod.load_matfile("ignored.mat", header_info=False)

        # Assert
        self.assertIn("__header__", out)
        self.assertIn("a", out)
        self.assertEqual(out["a"], 1)
        self.assertEqual(out["__header__"], "header-bytes")

    def test_parses_object_array_to_list_recursively(self):
        # Arrange: object array with heterogeneous entries
        obj_array = np.empty(3, dtype=object)
        obj_array[0] = 7
        obj_array[1] = np.array([1, 2, 3])  # numeric ndarray should be kept as ndarray
        # Nested object array element
        nested = np.empty(2, dtype=object)
        nested[0] = 9
        nested[1] = 10
        obj_array[2] = nested

        data = {"arr": obj_array}
        matfile_mod.sio = DummySio(return_dict=data)

        # Act
        out = matfile_mod.load_matfile("ignored.mat")

        # Assert: top-level object array becomes list; nested object array becomes nested list
        self.assertIsInstance(out["arr"], list)
        self.assertEqual(out["arr"][0], 7)
        np.testing.assert_array_equal(out["arr"][1], np.array([1, 2, 3]))
        self.assertEqual(out["arr"][2], [9, 10])

    def test_parses_mat_struct_to_dict_recursively(self):
        # Arrange: mat_struct with nested values
        inner_struct = DummyMatStruct({
            "z": np.array([[1, 2], [3, 4]]),
        })
        top_struct = DummyMatStruct({
            "x": 5,
            "y": inner_struct,
        })
        data = {"s": top_struct}

        # Build DummySio whose matlab.mio5_params.mat_struct equals DummyMatStruct
        matfile_mod.sio = DummySio(return_dict=data)

        # Act
        out = matfile_mod.load_matfile("ignored.mat")

        # Assert: mat_struct converted to dict recursively
        self.assertIsInstance(out["s"], dict)
        self.assertEqual(out["s"]["x"], 5)
        np.testing.assert_array_equal(out["s"]["y"]["z"], np.array([[1, 2], [3, 4]]))
