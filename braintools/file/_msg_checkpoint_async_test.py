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

"""Tests for AsyncManager in checkpoint module."""

import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import brainstate

try:
    import msgpack
    HAS_MSGPACK = True
except (ModuleNotFoundError, ImportError):
    HAS_MSGPACK = False

if HAS_MSGPACK:
    from braintools.file._msg_checkpoint import (
        AsyncManager,
        msgpack_save,
        msgpack_load,
    )


@unittest.skipIf(not HAS_MSGPACK, "msgpack not installed")
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


if __name__ == "__main__":
    unittest.main()

