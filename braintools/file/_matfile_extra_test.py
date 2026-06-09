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

"""Extra round-trip tests for ``braintools.file._matfile``.

These tests exercise the real ``load_matfile`` function against ``.mat`` files
produced by ``scipy.io.savemat``, covering numeric arrays, strings, MATLAB
structs, cell (object) arrays, optional arguments, and the error/edge branches.
"""

import os
import pathlib
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from braintools.file._matfile import load_matfile


class TestLoadMatfileRoundTrip(unittest.TestCase):
    """Round-trip real ``.mat`` files written by ``scipy.io.savemat``."""

    def test_numeric_array_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "num.mat")
            vec = np.array([1.0, 2.0, 3.0])
            mat = np.array([[1, 2], [3, 4]])
            savemat(fn, {"vec": vec, "mat": mat})

            out = load_matfile(fn, verbose=False)

            self.assertEqual(set(out.keys()), {"vec", "mat"})
            self.assertIsInstance(out["vec"], np.ndarray)
            np.testing.assert_array_equal(out["vec"], vec)
            np.testing.assert_array_equal(out["mat"], mat)

    def test_string_value_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "str.mat")
            savemat(fn, {"label": "hello world"})

            out = load_matfile(fn, verbose=False)

            self.assertEqual(out["label"], "hello world")

    def test_excludes_header_keys_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "hdr.mat")
            savemat(fn, {"a": np.array([1, 2, 3])})

            out = load_matfile(fn, verbose=False)

            self.assertIn("a", out)
            self.assertNotIn("__header__", out)
            self.assertNotIn("__version__", out)
            self.assertNotIn("__globals__", out)

    def test_includes_header_keys_when_requested(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "hdr2.mat")
            savemat(fn, {"a": np.array([1, 2, 3])})

            out = load_matfile(fn, header_info=False, verbose=False)

            self.assertIn("a", out)
            self.assertIn("__header__", out)
            self.assertIn("__version__", out)
            self.assertIn("__globals__", out)

    def test_struct_converted_to_dict(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "struct.mat")
            savemat(fn, {"st": {"x": np.array([1, 2, 3]), "y": 5.0}})

            out = load_matfile(fn, verbose=False)

            self.assertIsInstance(out["st"], dict)
            np.testing.assert_array_equal(out["st"]["x"], np.array([1, 2, 3]))
            self.assertEqual(out["st"]["y"], 5.0)

    def test_nested_struct_converted_recursively(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "nested.mat")
            savemat(fn, {"top": {"inner": {"z": np.array([[1, 2], [3, 4]])}, "flag": 1}})

            out = load_matfile(fn, verbose=False)

            self.assertIsInstance(out["top"], dict)
            self.assertIsInstance(out["top"]["inner"], dict)
            np.testing.assert_array_equal(
                out["top"]["inner"]["z"], np.array([[1, 2], [3, 4]])
            )

    def test_cell_array_converted_to_list(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "cell.mat")
            cell = np.empty((2,), dtype=object)
            cell[0] = np.array([1, 2, 3])
            cell[1] = np.array([4, 5])
            savemat(fn, {"c": cell})

            out = load_matfile(fn, verbose=False)

            self.assertIsInstance(out["c"], list)
            self.assertEqual(len(out["c"]), 2)
            np.testing.assert_array_equal(out["c"][0], np.array([1, 2, 3]))
            np.testing.assert_array_equal(out["c"][1], np.array([4, 5]))

    def test_squeeze_me_true_collapses_singleton_dims(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "sq.mat")
            savemat(fn, {"v": np.array([[1.0, 2.0, 3.0]])})

            out = load_matfile(fn, squeeze_me=True, verbose=False)

            self.assertEqual(np.shape(out["v"]), (3,))

    def test_squeeze_me_false_keeps_dims(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "nosq.mat")
            savemat(fn, {"v": np.array([[1.0, 2.0, 3.0]])})

            out = load_matfile(fn, squeeze_me=False, verbose=False)

            self.assertEqual(out["v"].shape, (1, 3))

    def test_struct_as_record_true(self):
        # When struct_as_record=True, MATLAB structs come back as numpy record
        # arrays rather than mat_struct objects; parse_mat keeps them as-is.
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "rec.mat")
            savemat(fn, {"st": {"x": np.array([1, 2, 3])}})

            out = load_matfile(
                fn, struct_as_record=True, squeeze_me=False, verbose=False
            )

            self.assertIn("st", out)
            self.assertIsInstance(out["st"], np.ndarray)

    def test_verbose_true_prints(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "verbose.mat")
            savemat(fn, {"a": np.array([1, 2, 3])})

            # verbose=True exercises the print branch; just assert it succeeds.
            out = load_matfile(fn, verbose=True)
            np.testing.assert_array_equal(out["a"], np.array([1, 2, 3]))

    def test_pathlike_filename_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            fn = pathlib.Path(d) / "plike.mat"
            savemat(str(fn), {"q": np.array([7, 8, 9])})

            out = load_matfile(fn, verbose=False)
            np.testing.assert_array_equal(out["q"], np.array([7, 8, 9]))

    def test_extra_kwargs_forwarded_to_loadmat(self):
        # ``mat_dtype`` is a genuine scipy.io.loadmat kwarg; passing it through
        # **kwargs exercises the kwargs-forwarding branch.
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "kw.mat")
            savemat(fn, {"a": np.array([1, 2, 3], dtype=np.int16)})

            out = load_matfile(fn, mat_dtype=True, verbose=False)
            np.testing.assert_array_equal(out["a"], np.array([1, 2, 3]))


class TestLoadMatfileErrors(unittest.TestCase):
    """Error and edge branches of ``load_matfile``."""

    def test_type_error_for_non_path(self):
        with self.assertRaises(TypeError) as cm:
            load_matfile(123)
        self.assertIn("string or path-like", str(cm.exception))

    def test_file_not_found(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "does_not_exist.mat")
            with self.assertRaises(FileNotFoundError) as cm:
                load_matfile(missing)
            self.assertIn("not found", str(cm.exception))

    def test_path_is_directory(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError) as cm:
                load_matfile(d)
            self.assertIn("not a file", str(cm.exception))

    def test_corrupt_file_raises_value_error(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.mat")
            with open(bad, "wb") as f:
                f.write(b"this is definitely not a valid MATLAB file")
            with self.assertRaises(ValueError) as cm:
                load_matfile(bad, verbose=False)
            self.assertIn("Failed to load MATLAB file", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
