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

import unittest

import numpy as np

import braintools.conn as conn


class TestKernelConnectivity(unittest.TestCase):
    """Test kernel-based connectivity patterns."""

    def test_conv_kernel(self):
        """Test convolutional kernel connectivity."""
        # 1D case
        kernel_1d = np.array([0.25, 0.5, 0.25])
        pre, post = conn.conv_kernel(
            input_shape=(10,),
            kernel=kernel_1d,
            stride=1,
            padding='valid',
            threshold=0.2,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)
        self.assertEqual(len(pre), len(post))

        # 2D case
        kernel_2d = np.array([[0.1, 0.2, 0.1],
                              [0.2, 0.4, 0.2],
                              [0.1, 0.2, 0.1]])
        pre, post = conn.conv_kernel(
            input_shape=(10, 10),
            kernel=kernel_2d,
            stride=1,
            padding='same',
            threshold=0.15,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # With stride
        pre_stride, post_stride = conn.conv_kernel(
            input_shape=(10, 10),
            kernel=kernel_2d,
            stride=2,
            padding='valid',
            threshold=0.15,
            seed=42
        )

        # Stride should reduce output size
        self.assertIsInstance(pre_stride, np.ndarray)
        self.assertIsInstance(post_stride, np.ndarray)

    def test_gaussian_kernel(self):
        """Test Gaussian kernel connectivity."""
        # 1D Gaussian
        pre, post = conn.gaussian_kernel(
            input_shape=(20,),
            sigma=2.0,
            kernel_size=5,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # 2D Gaussian
        pre_2d, post_2d = conn.gaussian_kernel(
            input_shape=(15, 15),
            sigma=1.5,
            kernel_size=7,
            seed=42
        )

        self.assertIsInstance(pre_2d, np.ndarray)
        self.assertIsInstance(post_2d, np.ndarray)

        # Check indices are valid
        n_input = 15 * 15
        n_output = 15 * 15  # same padding by default
        self.assertTrue(np.all(pre_2d >= 0))
        self.assertTrue(np.all(pre_2d < n_input))
        self.assertTrue(np.all(post_2d >= 0))
        self.assertTrue(np.all(post_2d < n_output))

    def test_gabor_kernel(self):
        """Test Gabor kernel connectivity."""
        pre, post = conn.gabor_kernel(
            input_shape=(20, 20),
            kernel_size=9,
            sigma=2.0,
            theta=np.pi / 4,
            frequency=0.5,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)
        self.assertEqual(len(pre), len(post))

        # Test with different parameters
        pre2, post2 = conn.gabor_kernel(
            input_shape=(15, 15),
            kernel_size=7,
            sigma=1.5,
            theta=0,
            frequency=0.3,
            phase=np.pi / 2,
            seed=42
        )

        self.assertIsInstance(pre2, np.ndarray)
        self.assertIsInstance(post2, np.ndarray)

    def test_dog_kernel(self):
        """Test Difference of Gaussians kernel."""
        pre, post = conn.dog_kernel(
            input_shape=(20, 20),
            sigma1=1.0,
            sigma2=2.0,
            kernel_size=9,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # Test with different ratio
        pre2, post2 = conn.dog_kernel(
            input_shape=(15, 15),
            sigma1=0.5,
            sigma2=1.5,
            kernel_size=7,
            seed=42
        )

        self.assertIsInstance(pre2, np.ndarray)
        self.assertIsInstance(post2, np.ndarray)

    def test_mexican_hat(self):
        """Test Mexican hat kernel."""
        # 1D Mexican hat
        pre, post = conn.mexican_hat(
            input_shape=(30,),
            sigma=2.0,
            kernel_size=11,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # 2D Mexican hat
        pre_2d, post_2d = conn.mexican_hat(
            input_shape=(20, 20),
            sigma=1.5,
            kernel_size=9,
            seed=42
        )

        self.assertIsInstance(pre_2d, np.ndarray)
        self.assertIsInstance(post_2d, np.ndarray)

    def test_sobel_kernel(self):
        """Test Sobel edge detection kernel."""
        # Horizontal Sobel
        pre_h, post_h = conn.sobel_kernel(
            input_shape=(20, 20),
            direction='horizontal',
            seed=42
        )

        self.assertIsInstance(pre_h, np.ndarray)
        self.assertIsInstance(post_h, np.ndarray)

        # Vertical Sobel
        pre_v, post_v = conn.sobel_kernel(
            input_shape=(20, 20),
            direction='vertical',
            seed=42
        )

        self.assertIsInstance(pre_v, np.ndarray)
        self.assertIsInstance(post_v, np.ndarray)

        # Both directions
        pre_b, post_b = conn.sobel_kernel(
            input_shape=(20, 20),
            direction='both',
            seed=42
        )

        self.assertIsInstance(pre_b, np.ndarray)
        self.assertIsInstance(post_b, np.ndarray)

    def test_laplacian_kernel(self):
        """Test Laplacian kernel."""
        # 4-connected
        pre_4, post_4 = conn.laplacian_kernel(
            input_shape=(20, 20),
            kernel_type='4-connected',
            seed=42
        )

        self.assertIsInstance(pre_4, np.ndarray)
        self.assertIsInstance(post_4, np.ndarray)

        # 8-connected
        pre_8, post_8 = conn.laplacian_kernel(
            input_shape=(20, 20),
            kernel_type='8-connected',
            seed=42
        )

        self.assertIsInstance(pre_8, np.ndarray)
        self.assertIsInstance(post_8, np.ndarray)

        # Gaussian type (delegates to mexican_hat)
        pre_g, post_g = conn.laplacian_kernel(
            input_shape=(20, 20),
            kernel_type='gaussian',
            seed=42
        )

        self.assertIsInstance(pre_g, np.ndarray)
        self.assertIsInstance(post_g, np.ndarray)

    def test_custom_kernel(self):
        """Test custom kernel connectivity."""

        # Define custom kernel function
        def my_kernel_func(x, y):
            return np.exp(-(x ** 2 + y ** 2) / 2)

        pre, post = conn.custom_kernel(
            input_shape=(15, 15),
            kernel_func=my_kernel_func,
            kernel_size=5,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # Test with different kernel function
        def gaussian_like(x, y, sigma=1.0):
            return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

        pre_n, post_n = conn.custom_kernel(
            input_shape=(10, 10),
            kernel_func=gaussian_like,
            kernel_size=7,
            sigma=1.5,
            seed=42
        )

        self.assertIsInstance(pre_n, np.ndarray)
        self.assertIsInstance(post_n, np.ndarray)

    def test_kernel_edge_cases(self):
        """Test edge cases for kernel functions."""
        # Very small input
        kernel = np.array([[0.5, 0.5], [0.5, 0.5]])
        pre, post = conn.conv_kernel(
            input_shape=(2, 2),
            kernel=kernel,
            stride=1,
            padding='valid',
            threshold=0.3,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # Large stride
        pre_ls, post_ls = conn.conv_kernel(
            input_shape=(10, 10),
            kernel=kernel,
            stride=5,
            padding='valid',
            threshold=0.3,
            seed=42
        )

        self.assertIsInstance(pre_ls, np.ndarray)
        self.assertIsInstance(post_ls, np.ndarray)

        # Very small sigma (few connections)
        pre_ss, post_ss = conn.gaussian_kernel(
            input_shape=(10, 10),
            sigma=0.1,  # Very small sigma
            kernel_size=3,
            seed=42
        )

        # Should have some connections but not many
        self.assertIsInstance(pre_ss, np.ndarray)
        self.assertIsInstance(post_ss, np.ndarray)

    def test_3d_kernel(self):
        """Test 3D kernel connectivity."""
        # 3D Gaussian kernel
        kernel_3d = np.ones((3, 3, 3)) / 27

        pre, post = conn.conv_kernel(
            input_shape=(5, 5, 5),
            kernel=kernel_3d,
            stride=1,
            padding='valid',
            threshold=0.02,
            seed=42
        )

        self.assertIsInstance(pre, np.ndarray)
        self.assertIsInstance(post, np.ndarray)

        # Check indices are valid for 3D
        n_input = 5 * 5 * 5
        n_output = 3 * 3 * 3  # valid padding reduces size
        self.assertTrue(np.all(pre >= 0))
        self.assertTrue(np.all(pre < n_input))
        self.assertTrue(np.all(post >= 0))
        self.assertTrue(np.all(post < n_output))


if __name__ == '__main__':
    unittest.main()
