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

"""Tests for ``braintools.file._matfile``.

These tests exercise the real ``load_matfile`` function against ``.mat`` files
produced by ``scipy.io.savemat``, covering numeric arrays, strings, MATLAB
structs, cell (object) arrays, optional arguments, and the error/edge branches;
plus ``save_matfile``, MATLAB v7.3 detection and the deprecated/inverted
``header_info`` flag.
"""

import os
import pathlib
import tempfile
import unittest
import warnings

import numpy as np
from scipy.io import savemat

from braintools._deprecation import BraintoolsDeprecationWarning, registered_deprecations
from braintools.file import _matfile as MF
from braintools.file._matfile import load_matfile, save_matfile


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

            out = load_matfile(fn, include_header=True, verbose=False)

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

    def test_save_matfile_appends_mat_extension(self):
        # Matches ``scipy.io.savemat``'s ``appendmat=True`` default.
        with tempfile.TemporaryDirectory() as d:
            save_matfile(os.path.join(d, "noext"), {"x": np.arange(2)})
            self.assertEqual(os.listdir(d), ["noext.mat"])
            save_matfile(os.path.join(d, "raw"), {"x": np.arange(2)}, appendmat=False)
            self.assertIn("raw", os.listdir(d))

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

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_save_matfile_overwrite_keeps_permissions(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "perm.mat")
            save_matfile(fn, {"x": np.arange(2)})
            os.chmod(fn, 0o640)
            save_matfile(fn, {"x": np.arange(5)})
            self.assertEqual(os.stat(fn).st_mode & 0o777, 0o640)
            self.assertEqual(load_matfile(fn)["x"].tolist(), [0, 1, 2, 3, 4])

    def test_save_matfile_unsavable_data_cleans_up_temp_file(self):
        # scipy rejects a non-string key after the temp file was created; the
        # temp file must be removed and the error wrapped.
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "bad.mat")
            with self.assertRaises(ValueError) as cm:
                save_matfile(fn, {"x": np.arange(2), "y": object()})
            self.assertIn("Failed to save MATLAB file", str(cm.exception))
            self.assertEqual(os.listdir(d), [])


class TestLoadMatfileV73AndDeprecation(unittest.TestCase):
    def test_v73_hdf5_detection(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "v73.mat")
            with open(fn, "wb") as f:
                f.write(MF._HDF5_MAGIC + b"\x00" * 64)
            with self.assertRaises(NotImplementedError) as cm:
                load_matfile(fn, verbose=False)
            self.assertIn("v7.3", str(cm.exception))

    def test_v73_matlab_layout_detection(self):
        # Files written by MATLAB put a 128-byte text header in a 512-byte HDF5
        # user block, so the HDF5 signature is at offset 512, not 0.
        header = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, HDF5 schema 1.00 ."
        header = header.ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02IM"
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "v73.mat")
            with open(fn, "wb") as f:
                f.write(header.ljust(512, b"\x00") + MF._HDF5_MAGIC + b"\x00" * 64)
            with self.assertRaises(NotImplementedError) as cm:
                load_matfile(fn)
            self.assertIn("h5py", str(cm.exception))

    def test_deprecated_header_info_true_excludes(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "h.mat")
            savemat(fn, {"a": np.array([1, 2, 3])})
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = load_matfile(fn, header_info=True, verbose=False)
            self.assertTrue(any(issubclass(x.category, BraintoolsDeprecationWarning) for x in w))
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
            self.assertTrue(any(issubclass(x.category, BraintoolsDeprecationWarning) for x in w))
            self.assertIn("__header__", out)

    def test_include_header_true(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "h3.mat")
            savemat(fn, {"a": np.array([1, 2, 3])})
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = load_matfile(fn, include_header=True, verbose=False)
            # No deprecation warning for the new flag.
            self.assertFalse(any(issubclass(x.category, BraintoolsDeprecationWarning) for x in w))
            self.assertIn("__header__", out)

    def test_header_info_record_registered(self):
        records = {r.old_name: r for r in registered_deprecations()}
        record = records["braintools.file.load_matfile(header_info=...)"]
        self.assertEqual(record.kind, "argument")
        self.assertEqual(record.since, "0.2.0")
        self.assertEqual(record.removed_in, "1.0.0")
        self.assertIn("include_header", record.new_name)

    def test_header_info_warns_once_with_registered_message(self):
        records = {r.old_name: r for r in registered_deprecations()}
        record = records["braintools.file.load_matfile(header_info=...)"]
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "h4.mat")
            savemat(fn, {"a": np.array([1, 2, 3])})
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                load_matfile(fn, header_info=True, verbose=False)
        self.assertEqual(len(w), 1)
        self.assertIs(w[0].category, BraintoolsDeprecationWarning)
        self.assertEqual(str(w[0].message), record.message)
        # The message still explains the inverted sense of the old flag.
        self.assertIn("include_header=not header_info", str(w[0].message))
        self.assertEqual(w[0].filename, __file__)

    def test_no_warning_without_header_info(self):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "h5.mat")
            savemat(fn, {"a": np.array([1, 2, 3])})
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                load_matfile(fn, verbose=False)
        self.assertEqual(w, [])


if __name__ == "__main__":
    unittest.main()
