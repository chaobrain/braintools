# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

import os
import tempfile
import unittest

import numpy as np

import braintools.conn as conn


class TestConnectivityIO(unittest.TestCase):
    """Test connectivity IO functions."""

    def setUp(self):
        """Set up test data."""
        # Create sample connectivity data
        self.pre_indices = np.array([0, 1, 2, 3, 4, 5])
        self.post_indices = np.array([5, 4, 3, 2, 1, 0])
        self.properties = {
            'weights': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            'delays': np.array([1, 2, 3, 4, 5, 6])
        }

        # Create temp directory for test files
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp files."""
        # Remove temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dict_io(self):
        """Test dictionary conversion."""
        # Convert to dict
        data = conn.to_dict(self.pre_indices, self.post_indices, self.properties)

        # Check structure
        self.assertIn('pre_indices', data)
        self.assertIn('post_indices', data)
        self.assertIn('weights', data)
        self.assertIn('delays', data)

        # Load from dict
        pre, post, props = conn.from_dict(data)

        # Verify
        np.testing.assert_array_equal(pre, self.pre_indices)
        np.testing.assert_array_equal(post, self.post_indices)
        np.testing.assert_array_equal(props['weights'], self.properties['weights'])

    def test_csv_io(self):
        """Test CSV file IO."""
        filepath = os.path.join(self.temp_dir, 'test.csv')

        # Save to CSV
        conn.to_csv(self.pre_indices, self.post_indices, filepath, self.properties)

        # Load from CSV
        pre, post, props = conn.from_csv(filepath)

        # Verify
        np.testing.assert_array_equal(pre.astype(int), self.pre_indices)
        np.testing.assert_array_equal(post.astype(int), self.post_indices)
        np.testing.assert_array_almost_equal(props['weights'], self.properties['weights'])

    def test_hdf5_io(self):
        """Test HDF5 file IO."""
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py not installed")

        filepath = os.path.join(self.temp_dir, 'test.h5')

        # Save to HDF5
        conn.to_hdf5(self.pre_indices, self.post_indices, filepath,
                     self.properties, overwrite=True)

        # Load from HDF5
        pre, post, props = conn.from_hdf5(filepath)

        # Verify
        np.testing.assert_array_equal(pre, self.pre_indices)
        np.testing.assert_array_equal(post, self.post_indices)
        np.testing.assert_array_equal(props['weights'], self.properties['weights'])

    def test_hdf5_with_group(self):
        """Test HDF5 with group."""
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py not installed")

        filepath = os.path.join(self.temp_dir, 'test_group.h5')

        # Save to HDF5 with group
        conn.to_hdf5(self.pre_indices, self.post_indices, filepath,
                     self.properties, group='connectivity', overwrite=True)

        # Load from HDF5 with group
        pre, post, props = conn.from_hdf5(filepath, group='connectivity')

        # Verify
        np.testing.assert_array_equal(pre, self.pre_indices)
        np.testing.assert_array_equal(post, self.post_indices)

    def test_sonata_io(self):
        """Test SONATA format IO."""
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py not installed")

        filepath = os.path.join(self.temp_dir, 'test_sonata.h5')

        # Save to SONATA
        conn.to_sonata(self.pre_indices, self.post_indices, filepath,
                       self.properties, overwrite=True)

        # Load from SONATA
        pre, post, props = conn.from_sonata(filepath)

        # Verify
        np.testing.assert_array_equal(pre, self.pre_indices)
        np.testing.assert_array_equal(post, self.post_indices)
        if 'weights' in props:
            np.testing.assert_array_equal(props['weights'], self.properties['weights'])

    def test_nwb_io(self):
        """Test NWB format IO."""
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py not installed")

        filepath = os.path.join(self.temp_dir, 'test_nwb.nwb')

        # Save to NWB
        conn.to_nwb(self.pre_indices, self.post_indices, filepath,
                    self.properties, overwrite=True)

        # Load from NWB
        pre, post, props = conn.from_nwb(filepath)

        # Verify basic structure
        np.testing.assert_array_equal(pre, self.pre_indices)
        np.testing.assert_array_equal(post, self.post_indices)

    def test_parquet_io(self):
        """Test Parquet format IO."""
        try:
            import pandas as pd
            import pyarrow
        except ImportError:
            self.skipTest("pandas or pyarrow not installed")

        filepath = os.path.join(self.temp_dir, 'test.parquet')

        # Save to Parquet
        conn.to_parquet(self.pre_indices, self.post_indices, filepath, self.properties)

        # Load from Parquet
        pre, post, props = conn.from_parquet(filepath)

        # Verify
        np.testing.assert_array_equal(pre, self.pre_indices)
        np.testing.assert_array_equal(post, self.post_indices)
        np.testing.assert_array_equal(props['weights'], self.properties['weights'])

    def test_zarr_io(self):
        """Test Zarr format IO."""
        try:
            import zarr
        except ImportError:
            self.skipTest("zarr not installed")

        filepath = os.path.join(self.temp_dir, 'test.zarr')

        # Save to Zarr
        conn.to_zarr(self.pre_indices, self.post_indices, filepath,
                     self.properties, overwrite=True)

        # Load from Zarr
        pre, post, props = conn.from_zarr(filepath)

        # Verify
        np.testing.assert_array_equal(pre, self.pre_indices)
        np.testing.assert_array_equal(post, self.post_indices)
        np.testing.assert_array_equal(props['weights'], self.properties['weights'])

    def test_from_dict_alternative_keys(self):
        """Test loading from dict with alternative key names."""
        # Test with 'pre' and 'post'
        data = {
            'pre': self.pre_indices,
            'post': self.post_indices
        }
        pre, post, _ = conn.from_dict(data)
        np.testing.assert_array_equal(pre, self.pre_indices)
        np.testing.assert_array_equal(post, self.post_indices)

        # Test with 'source' and 'target'
        data = {
            'source': self.pre_indices,
            'target': self.post_indices
        }
        pre, post, _ = conn.from_dict(data)
        np.testing.assert_array_equal(pre, self.pre_indices)
        np.testing.assert_array_equal(post, self.post_indices)


if __name__ == '__main__':
    unittest.main()
