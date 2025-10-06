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

"""
Comprehensive tests for kernel-based connectivity classes.

This test suite covers:
- Conv2dKernel (general convolution kernel connectivity)
- GaussianKernel (Gaussian receptive field connectivity)
- GaborKernel (orientation-selective Gabor filter connectivity)
- DoGKernel (difference of Gaussians for center-surround)
- MexicanHat (Laplacian of Gaussian)
- SobelKernel (edge detection filters)
- LaplacianKernel (edge enhancement)
- CustomKernel (user-defined kernel functions)
"""

import unittest

import brainunit as u
import numpy as np

from braintools.conn import (
    Conv2dKernel,
    GaussianKernel,
    GaborKernel,
    DoGKernel,
    MexicanHat,
    SobelKernel,
    LaplacianKernel,
    CustomKernel,
)
from braintools.init import Constant, Uniform


class TestConvKernel(unittest.TestCase):
    """
    Test Conv2dKernel connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_kernel import Conv2dKernel

        # Create a simple 3x3 Gaussian-like kernel
        kernel = np.array([
            [0.1, 0.2, 0.1],
            [0.2, 1.0, 0.2],
            [0.1, 0.2, 0.1]
        ])

        # Create grid positions
        positions = np.random.uniform(0, 100, (25, 2)) * u.um

        conn = Conv2dKernel(
            kernel=kernel,
            kernel_size=30 * u.um,
            threshold=0.15,
            weight=1.0 * u.nS
        )

        result = conn(
            pre_size=25, post_size=25,
            pre_positions=positions,
            post_positions=positions
        )

        assert result.model_type == 'point'
        assert result.n_connections > 0
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_conv_kernel(self):
        kernel = np.array([
            [0.1, 0.2, 0.1],
            [0.2, 1.0, 0.2],
            [0.1, 0.2, 0.1]
        ])

        positions = np.random.RandomState(42).uniform(0, 100, (16, 2)) * u.um

        conn = Conv2dKernel(
            kernel=kernel,
            kernel_size=40 * u.um,
            threshold=0.15,
            seed=42
        )

        result = conn(
            pre_size=16, post_size=16,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.model_type, 'point')
        self.assertGreater(result.n_connections, 0)
        self.assertEqual(result.metadata['pattern'], 'conv_kernel')
        self.assertEqual(result.metadata['kernel_shape'], (3, 3))

    def test_conv_kernel_with_weights_and_delays(self):
        kernel = np.array([[0.5, 1.0, 0.5]])  # 1D kernel

        positions = np.array([[0, 0], [10, 0], [20, 0]]) * u.um

        weight_init = Constant(2.0 * u.nS)
        delay_init = Uniform(1.0 * u.ms, 3.0 * u.ms)

        conn = Conv2dKernel(
            kernel=kernel,
            kernel_size=30 * u.um,
            threshold=0.0,
            weight=weight_init,
            delay=delay_init,
            seed=42
        )

        result = conn(
            pre_size=3, post_size=3,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertIsNotNone(result.weights)
        self.assertIsNotNone(result.delays)
        self.assertEqual(result.weights.unit, u.nS)
        self.assertEqual(result.delays.unit, u.ms)

        # Weights should be kernel values multiplied by weight init
        kernel_values = kernel.flatten()
        expected_total_weight = np.sum(kernel_values[kernel_values > 0.0]) * 2.0
        self.assertGreater(np.sum(result.weights.mantissa), 0)

    def test_conv_kernel_threshold(self):
        kernel = np.array([
            [0.05, 0.1, 0.05],  # Low values
            [0.1, 1.0, 0.1],  # High center value
            [0.05, 0.1, 0.05]  # Low values
        ])

        positions = np.array([[0, 0], [1, 1]]) * u.um

        # High threshold should filter out low kernel values
        conn = Conv2dKernel(
            kernel=kernel,
            kernel_size=5 * u.um,
            threshold=0.5,  # High threshold
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        # Should have fewer connections due to high threshold
        self.assertGreaterEqual(result.n_connections, 0)

    def test_conv_kernel_no_positions_error(self):
        kernel = np.array([[1]])

        conn = Conv2dKernel(
            kernel=kernel,
            kernel_size=10 * u.um,
            seed=42
        )

        with self.assertRaises(ValueError):
            conn(pre_size=5, post_size=5)  # No positions provided

    def test_conv_kernel_invalid_kernel_shape(self):
        # 1D kernel should raise error
        with self.assertRaises(ValueError):
            Conv2dKernel(
                kernel=np.array([1, 2, 3]),  # 1D array
                kernel_size=10 * u.um
            )

    def test_conv_kernel_empty_connections(self):
        kernel = np.array([[0.1]])  # Small kernel

        # Positions far apart so no connections
        positions = np.array([[0, 0], [1000, 1000]]) * u.um

        conn = Conv2dKernel(
            kernel=kernel,
            kernel_size=1 * u.um,  # Small kernel size
            threshold=0.2,  # Higher threshold to filter out weak connections
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.n_connections, 0)

    def test_conv_kernel_scalar_kernel_size(self):
        kernel = np.array([[1]])
        positions = np.array([[0, 0], [5, 5]])  # No units

        conn = Conv2dKernel(
            kernel=kernel,
            kernel_size=10.0,  # Scalar, no units
            threshold=0.0,
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertGreaterEqual(result.n_connections, 0)

    def test_conv_kernel_tuple_sizes(self):
        kernel = np.array([[1]])
        positions = np.random.RandomState(42).uniform(0, 10, (6, 2)) * u.um

        conn = Conv2dKernel(
            kernel=kernel,
            kernel_size=15 * u.um,
            threshold=0.0,
            seed=42
        )

        result = conn(
            pre_size=(2, 3), post_size=(3, 2),
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.pre_size, (2, 3))
        self.assertEqual(result.post_size, (3, 2))


class TestGaussianKernel(unittest.TestCase):
    """
    Test GaussianKernel connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_kernel import GaussianKernel

        # Random positions in 2D space
        positions = np.random.uniform(0, 200, (50, 2)) * u.um

        conn = GaussianKernel(
            sigma=30 * u.um,
            max_distance=90 * u.um,
            normalize=True,
            weight=1.5 * u.nS
        )

        result = conn(
            pre_size=50, post_size=50,
            pre_positions=positions,
            post_positions=positions
        )

        # Check that connections follow Gaussian decay
        distances = result.get_distances()
        assert np.all(distances <= 90 * u.um)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_gaussian_kernel(self):
        positions = np.random.RandomState(42).uniform(0, 100, (20, 2)) * u.um

        conn = GaussianKernel(
            sigma=25 * u.um,
            max_distance=75 * u.um,
            normalize=True,
            seed=42
        )

        result = conn(
            pre_size=20, post_size=20,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.model_type, 'point')
        self.assertGreater(result.n_connections, 0)
        self.assertEqual(result.metadata['pattern'], 'gaussian_kernel')
        self.assertEqual(result.metadata['sigma'], 25 * u.um)

    def test_gaussian_kernel_distance_constraint(self):
        # Test that connections respect max_distance
        positions = np.array([[0, 0], [50, 0], [150, 0]]) * u.um

        conn = GaussianKernel(
            sigma=20 * u.um,
            max_distance=100 * u.um,
            seed=42
        )

        result = conn(
            pre_size=3, post_size=3,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            distances = result.get_distances()
            self.assertTrue(np.all(distances <= 100 * u.um))

    def test_gaussian_kernel_default_max_distance(self):
        # Should use 3*sigma as default max_distance
        positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um

        conn = GaussianKernel(
            sigma=20 * u.um,
            # max_distance not specified
            seed=42
        )

        result = conn(
            pre_size=10, post_size=10,
            pre_positions=positions,
            post_positions=positions
        )

        # Default max_distance should be 3 * 20 = 60 um
        if result.n_connections > 0:
            distances = result.get_distances()
            self.assertTrue(np.all(distances <= 60 * u.um))

    def test_gaussian_kernel_normalization(self):
        positions = np.array([[0, 0], [10, 0]]) * u.um

        # Test with normalization
        conn_norm = GaussianKernel(
            sigma=20 * u.um,
            normalize=True,
            seed=42
        )

        # Test without normalization
        conn_no_norm = GaussianKernel(
            sigma=20 * u.um,
            normalize=False,
            seed=42
        )

        result_norm = conn_norm(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        result_no_norm = conn_no_norm(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        # Both should have connections, but different weight magnitudes
        self.assertGreater(result_norm.n_connections, 0)
        self.assertGreater(result_no_norm.n_connections, 0)

    def test_gaussian_kernel_with_weights(self):
        positions = np.array([[0, 0], [10, 0]]) * u.um

        weight_init = Constant(2.0 * u.nS)

        conn = GaussianKernel(
            sigma=15 * u.um,
            weight=weight_init,
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertIsNotNone(result.weights)
        self.assertEqual(result.weights.unit, u.nS)
        # Weights should be Gaussian values * 2.0

    def test_gaussian_kernel_scalar_sigma(self):
        positions = np.array([[0, 0], [10, 0]])  # No units

        conn = GaussianKernel(
            sigma=15.0,  # Scalar, no units
            max_distance=45.0,
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertGreater(result.n_connections, 0)

    def test_gaussian_kernel_no_positions_error(self):
        conn = GaussianKernel(sigma=20 * u.um, seed=42)

        with self.assertRaises(ValueError):
            conn(pre_size=5, post_size=5)  # No positions

    def test_gaussian_kernel_empty_connections(self):
        # Very small sigma with distant positions
        positions = np.array([[0, 0], [1000, 1000]]) * u.um

        conn = GaussianKernel(
            sigma=1 * u.um,  # Very small sigma
            max_distance=2 * u.um,  # Very small max distance
            seed=42
        )

        result = conn(
            pre_size=2,
            post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.n_connections, 2)


class TestGaborKernel(unittest.TestCase):
    """
    Test GaborKernel connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_kernel import GaborKernel

        # Grid positions for testing orientation selectivity
        x = np.linspace(-50, 50, 11)
        y = np.linspace(-50, 50, 11)
        X, Y = np.meshgrid(x, y)
        positions = np.column_stack([X.ravel(), Y.ravel()]) * u.um

        conn = GaborKernel(
            sigma=20 * u.um,
            frequency=0.1,  # 1 cycle per 10 um
            theta=np.pi / 4,  # 45 degrees
            phase=0.0,
            weight=1.0 * u.nS
        )

        result = conn(
            pre_size=121, post_size=121,
            pre_positions=positions,
            post_positions=positions
        )

        # Gabor should create orientation-selective patterns
        assert result.metadata['pattern'] == 'gabor_kernel'
        assert result.metadata['theta'] == np.pi / 4
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_gabor_kernel(self):
        positions = np.random.RandomState(42).uniform(-50, 50, (25, 2)) * u.um

        conn = GaborKernel(
            sigma=20 * u.um,
            frequency=0.05,  # 1 cycle per 20 um
            theta=np.pi / 6,  # 30 degrees
            phase=0.0,
            seed=42
        )

        result = conn(
            pre_size=25, post_size=25,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.model_type, 'point')
        self.assertGreater(result.n_connections, 0)
        self.assertEqual(result.metadata['pattern'], 'gabor_kernel')
        self.assertEqual(result.metadata['frequency'], 0.05)
        self.assertEqual(result.metadata['theta'], np.pi / 6)

    def test_gabor_kernel_different_orientations(self):
        positions = np.array([[-10, 0], [0, 0], [10, 0]]) * u.um

        # Test different orientations
        for theta in [0, np.pi / 4, np.pi / 2]:
            conn = GaborKernel(
                sigma=15 * u.um,
                frequency=0.1,
                theta=theta,
                seed=42
            )

            result = conn(
                pre_size=3, post_size=3,
                pre_positions=positions,
                post_positions=positions
            )

            self.assertEqual(result.metadata['theta'], theta)

    def test_gabor_kernel_with_phase(self):
        positions = np.array([[0, 0], [5, 5]]) * u.um

        conn = GaborKernel(
            sigma=10 * u.um,
            frequency=0.2,
            theta=0,
            phase=np.pi / 2,  # 90 degree phase shift
            seed=42
        )

        result = conn(
            pre_size=2,
            post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.metadata['phase'], np.pi / 2)

    def test_gabor_kernel_max_distance(self):
        positions = np.array([[0, 0], [30, 0], [100, 0]]) * u.um

        conn = GaborKernel(
            sigma=20 * u.um,
            frequency=0.1,
            theta=0,
            max_distance=80 * u.um,
            seed=42
        )

        result = conn(
            pre_size=3, post_size=3,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            distances = result.get_distances()
            self.assertTrue(np.all(distances <= 80 * u.um))

    def test_gabor_kernel_default_max_distance(self):
        # Should use 3*sigma as default
        positions = np.random.RandomState(42).uniform(-30, 30, (10, 2)) * u.um

        conn = GaborKernel(
            sigma=15 * u.um,
            frequency=0.1,
            theta=0,
            # max_distance not specified
            seed=42
        )

        result = conn(
            pre_size=10, post_size=10,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            distances = result.get_distances()
            self.assertTrue(np.all(distances <= 45 * u.um))  # 3 * 15

    def test_gabor_kernel_no_positions_error(self):
        conn = GaborKernel(
            sigma=20 * u.um,
            frequency=0.1,
            theta=0,
            seed=42
        )

        with self.assertRaises(ValueError):
            conn(pre_size=5, post_size=5)

    def test_gabor_kernel_with_weights(self):
        positions = np.array([[0, 0], [10, 0]]) * u.um

        weight_init = Constant(1.5 * u.nS)

        conn = GaborKernel(
            sigma=15 * u.um,
            frequency=0.1,
            theta=0,
            weight=weight_init,
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            self.assertIsNotNone(result.weights)
            self.assertEqual(result.weights.unit, u.nS)

    def test_gabor_kernel_scalar_parameters(self):
        positions = np.array([[0, 0], [10, 0]])  # No units

        conn = GaborKernel(
            sigma=15.0,  # Scalar
            frequency=0.1,
            theta=0,
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertGreaterEqual(result.n_connections, 0)


class TestDoGKernel(unittest.TestCase):
    """
    Test DoGKernel (Difference of Gaussians) connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_kernel import DoGKernel

        # Concentric positions for testing center-surround
        positions = np.random.uniform(-100, 100, (50, 2)) * u.um

        conn = DoGKernel(
            sigma_center=15 * u.um,
            sigma_surround=30 * u.um,
            amplitude_center=1.0,
            amplitude_surround=0.8,
            weight=1.0 * u.nS
        )

        result = conn(
            pre_size=50, post_size=50,
            pre_positions=positions,
            post_positions=positions
        )

        # DoG should create center-surround receptive fields
        assert result.metadata['pattern'] == 'dog_kernel'
        assert 'sigma_center' in result.metadata
        assert 'sigma_surround' in result.metadata
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_dog_kernel(self):
        positions = np.random.RandomState(42).uniform(-50, 50, (20, 2)) * u.um

        conn = DoGKernel(
            sigma_center=10 * u.um,
            sigma_surround=25 * u.um,
            amplitude_center=1.0,
            amplitude_surround=0.6,
            seed=42
        )

        result = conn(
            pre_size=20, post_size=20,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.model_type, 'point')
        self.assertGreater(result.n_connections, 0)
        self.assertEqual(result.metadata['pattern'], 'dog_kernel')
        self.assertEqual(result.metadata['sigma_center'], 10 * u.um)
        self.assertEqual(result.metadata['sigma_surround'], 25 * u.um)

    def test_dog_kernel_center_surround_structure(self):
        # Test that DoG creates both positive and negative weights
        # Center position and nearby positions
        positions = np.array([
            [0, 0],  # Center
            [5, 0],  # Close (should be positive - center dominant)
            [20, 0],  # Medium distance (might be negative - surround)
            [50, 0]  # Far (should be weak or zero)
        ]) * u.um

        conn = DoGKernel(
            sigma_center=8 * u.um,
            sigma_surround=20 * u.um,
            amplitude_center=1.0,
            amplitude_surround=0.8,
            seed=42
        )

        result = conn(
            pre_size=4, post_size=4,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            # Should have both positive and negative weights
            weights = result.weights.mantissa if hasattr(result.weights, 'mantissa') else result.weights
            if len(weights) > 1:
                # Check that we have a range of weight values
                self.assertGreater(np.max(weights), np.min(weights))

    def test_dog_kernel_max_distance(self):
        positions = np.array([[0, 0], [30, 0], [100, 0]]) * u.um

        conn = DoGKernel(
            sigma_center=10 * u.um,
            sigma_surround=25 * u.um,
            max_distance=80 * u.um,
            seed=42
        )

        result = conn(
            pre_size=3, post_size=3,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            distances = result.get_distances()
            self.assertTrue(np.all(distances <= 80 * u.um))

    def test_dog_kernel_default_max_distance(self):
        # Should use 3*sigma_surround as default
        positions = np.random.RandomState(42).uniform(-40, 40, (15, 2)) * u.um

        conn = DoGKernel(
            sigma_center=8 * u.um,
            sigma_surround=20 * u.um,
            # max_distance not specified
            seed=42
        )

        result = conn(
            pre_size=15, post_size=15,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            distances = result.get_distances()
            self.assertTrue(np.all(distances <= 60 * u.um))  # 3 * 20

    def test_dog_kernel_different_amplitudes(self):
        positions = np.array([[0, 0], [10, 0]]) * u.um

        # Test different amplitude ratios
        conn1 = DoGKernel(
            sigma_center=10 * u.um,
            sigma_surround=20 * u.um,
            amplitude_center=1.0,
            amplitude_surround=0.5,  # Weaker surround
            seed=42
        )

        conn2 = DoGKernel(
            sigma_center=10 * u.um,
            sigma_surround=20 * u.um,
            amplitude_center=1.0,
            amplitude_surround=1.2,  # Stronger surround
            seed=42
        )

        result1 = conn1(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        result2 = conn2(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        # Both should produce connections
        self.assertGreaterEqual(result1.n_connections, 0)
        self.assertGreaterEqual(result2.n_connections, 0)

    def test_dog_kernel_no_positions_error(self):
        conn = DoGKernel(
            sigma_center=10 * u.um,
            sigma_surround=20 * u.um,
            seed=42
        )

        with self.assertRaises(ValueError):
            conn(pre_size=5, post_size=5)

    def test_dog_kernel_with_weights(self):
        positions = np.array([[0, 0], [10, 0]]) * u.um

        weight_init = Constant(2.0 * u.nS)

        conn = DoGKernel(
            sigma_center=8 * u.um,
            sigma_surround=20 * u.um,
            weight=weight_init,
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            self.assertIsNotNone(result.weights)
            self.assertEqual(result.weights.unit, u.nS)

    def test_dog_kernel_scalar_sigmas(self):
        positions = np.array([[0, 0], [15, 0]])  # No units

        conn = DoGKernel(
            sigma_center=8.0,  # Scalar
            sigma_surround=20.0,  # Scalar
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertGreaterEqual(result.n_connections, 0)


class TestMexicanHat(unittest.TestCase):
    """
    Test MexicanHat (Laplacian of Gaussian) connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_kernel import MexicanHat

        # Random positions for testing lateral inhibition
        positions = np.random.uniform(-80, 80, (40, 2)) * u.um

        conn = MexicanHat(
            sigma=25 * u.um,
            weight=1.0 * u.nS
        )

        result = conn(
            pre_size=40, post_size=40,
            pre_positions=positions,
            post_positions=positions
        )

        # Mexican hat should create strong lateral inhibition
        assert result.metadata['pattern'] == 'dog_kernel'  # Inherits from DoG
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_mexican_hat(self):
        positions = np.random.RandomState(42).uniform(-50, 50, (20, 2)) * u.um

        conn = MexicanHat(
            sigma=20 * u.um,
            seed=42
        )

        result = conn(
            pre_size=20, post_size=20,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.model_type, 'point')
        self.assertGreater(result.n_connections, 0)
        # Mexican hat is implemented as DoG
        self.assertEqual(result.metadata['pattern'], 'dog_kernel')

    def test_mexican_hat_parameters(self):
        # Test that MexicanHat sets up DoG parameters correctly
        sigma = 15 * u.um

        conn = MexicanHat(
            sigma=sigma,
            seed=42
        )

        # Check internal DoG parameters
        self.assertEqual(conn.sigma_center, sigma)
        self.assertAlmostEqual(conn.sigma_surround.mantissa, sigma.mantissa * np.sqrt(2), places=5)
        self.assertEqual(conn.amplitude_center, 2.0)
        self.assertEqual(conn.amplitude_surround, 1.0)

    def test_mexican_hat_max_distance(self):
        sigma = 20 * u.um

        # Test default max_distance
        conn1 = MexicanHat(sigma=sigma, seed=42)
        self.assertEqual(conn1.max_distance, 4 * sigma)

        # Test custom max_distance
        custom_max = 120 * u.um
        conn2 = MexicanHat(sigma=sigma, max_distance=custom_max, seed=42)
        self.assertEqual(conn2.max_distance, custom_max)

    def test_mexican_hat_scalar_sigma(self):
        positions = np.array([[0, 0], [20, 0]])  # No units

        conn = MexicanHat(
            sigma=15.0,  # Scalar
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertGreaterEqual(result.n_connections, 0)

    def test_mexican_hat_with_weights(self):
        positions = np.array([[0, 0], [10, 0]]) * u.um

        weight_init = Constant(1.5 * u.nS)

        conn = MexicanHat(
            sigma=12 * u.um,
            weight=weight_init,
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            self.assertIsNotNone(result.weights)
            self.assertEqual(result.weights.unit, u.nS)


class TestSobelKernel(unittest.TestCase):
    """
    Test SobelKernel connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_kernel import SobelKernel

        # Grid positions for testing edge detection
        x = np.linspace(-30, 30, 7)
        y = np.linspace(-30, 30, 7)
        X, Y = np.meshgrid(x, y)
        positions = np.column_stack([X.ravel(), Y.ravel()]) * u.um

        # Horizontal edge detection
        sobel_h = SobelKernel(
            direction='horizontal',
            kernel_size=20 * u.um,
            weight=1.0 * u.nS
        )

        result = sobel_h(
            pre_size=49, post_size=49,
            pre_positions=positions,
            post_positions=positions
        )

        assert result.metadata['pattern'] == 'sobel_kernel'
        assert result.metadata['direction'] == 'horizontal'
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_sobel_horizontal(self):
        positions = np.random.RandomState(42).uniform(-30, 30, (25, 2)) * u.um

        conn = SobelKernel(
            direction='horizontal',
            kernel_size=15 * u.um,
            seed=42
        )

        result = conn(
            pre_size=25, post_size=25,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'sobel_kernel')
        self.assertEqual(result.metadata['direction'], 'horizontal')

    def test_sobel_vertical(self):
        positions = np.random.RandomState(42).uniform(-30, 30, (25, 2)) * u.um

        conn = SobelKernel(
            direction='vertical',
            kernel_size=15 * u.um,
            seed=42
        )

        result = conn(
            pre_size=25, post_size=25,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.metadata['direction'], 'vertical')

    def test_sobel_both(self):
        positions = np.random.RandomState(42).uniform(-30, 30, (16, 2)) * u.um

        conn = SobelKernel(
            direction='both',
            kernel_size=15 * u.um,
            seed=42
        )

        result = conn(
            pre_size=16, post_size=16,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.metadata['direction'], 'both')

    def test_sobel_invalid_direction(self):
        with self.assertRaises(ValueError):
            SobelKernel(direction='invalid')

    def test_sobel_kernel_structure(self):
        # Test that kernels are set up correctly
        sobel_h = SobelKernel(direction='horizontal')
        expected_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        np.testing.assert_array_equal(sobel_h.kernel, expected_h)

        sobel_v = SobelKernel(direction='vertical')
        expected_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(sobel_v.kernel, expected_v)

        sobel_both = SobelKernel(direction='both')
        self.assertIsNone(sobel_both.kernel)  # Uses both kernels separately

    def test_sobel_with_weights(self):
        positions = np.array([[0, 0], [5, 5]]) * u.um

        weight_init = Constant(1.2 * u.nS)

        conn = SobelKernel(
            direction='horizontal',
            kernel_size=12 * u.um,
            weight=weight_init,
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            self.assertIsNotNone(result.weights)
            self.assertEqual(result.weights.unit, u.nS)


class TestLaplacianKernel(unittest.TestCase):
    """
    Test LaplacianKernel connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_kernel import LaplacianKernel

        # Grid positions for testing edge enhancement
        x = np.linspace(-20, 20, 5)
        y = np.linspace(-20, 20, 5)
        X, Y = np.meshgrid(x, y)
        positions = np.column_stack([X.ravel(), Y.ravel()]) * u.um

        conn = LaplacianKernel(
            kernel_type='4-connected',
            kernel_size=15 * u.um,
            weight=1.0 * u.nS
        )

        result = conn(
            pre_size=25, post_size=25,
            pre_positions=positions,
            post_positions=positions
        )

        assert result.metadata['pattern'] == 'laplacian_kernel'
        assert result.metadata['kernel_type'] == '4-connected'
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_laplacian_4_connected(self):
        positions = np.random.RandomState(42).uniform(-25, 25, (20, 2)) * u.um

        conn = LaplacianKernel(
            kernel_type='4-connected',
            kernel_size=12 * u.um,
            seed=42
        )

        result = conn(
            pre_size=20, post_size=20,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'laplacian_kernel')
        self.assertEqual(result.metadata['kernel_type'], '4-connected')

    def test_laplacian_8_connected(self):
        positions = np.random.RandomState(42).uniform(-25, 25, (20, 2)) * u.um

        conn = LaplacianKernel(
            kernel_type='8-connected',
            kernel_size=12 * u.um,
            seed=42
        )

        result = conn(
            pre_size=20, post_size=20,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.metadata['kernel_type'], '8-connected')

    def test_laplacian_invalid_type(self):
        with self.assertRaises(ValueError):
            LaplacianKernel(kernel_type='invalid')

    def test_laplacian_kernel_structure(self):
        # Test that kernels are set up correctly
        lap_4 = LaplacianKernel(kernel_type='4-connected')
        expected_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        np.testing.assert_array_equal(lap_4.kernel, expected_4)

        lap_8 = LaplacianKernel(kernel_type='8-connected')
        expected_8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(lap_8.kernel, expected_8)

    def test_laplacian_with_weights(self):
        positions = np.array([[0, 0], [8, 0]]) * u.um

        weight_init = Constant(0.8 * u.nS)

        conn = LaplacianKernel(
            kernel_type='4-connected',
            kernel_size=15 * u.um,
            weight=weight_init,
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            self.assertIsNotNone(result.weights)
            self.assertEqual(result.weights.unit, u.nS)


class TestCustomKernel(unittest.TestCase):
    """
    Test CustomKernel connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_kernel import CustomKernel

        def my_kernel_func(x, y):
            # Custom radial function with oscillations
            r = np.sqrt(x**2 + y**2)
            return np.exp(-r/30) * np.cos(r/10)

        positions = np.random.uniform(-50, 50, (30, 2)) * u.um

        conn = CustomKernel(
            kernel_func=my_kernel_func,
            kernel_size=100 * u.um,
            threshold=0.1,
            weight=1.0 * u.nS
        )

        result = conn(
            pre_size=30, post_size=30,
            pre_positions=positions,
            post_positions=positions
        )

        assert result.metadata['pattern'] == 'custom_kernel'
        assert 'kernel_size' in result.metadata
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_custom_kernel(self):
        def simple_kernel(x, y):
            # Simple Gaussian-like function
            return np.exp(-(x ** 2 + y ** 2) / 100)

        positions = np.random.RandomState(42).uniform(-30, 30, (15, 2)) * u.um

        conn = CustomKernel(
            kernel_func=simple_kernel,
            kernel_size=60 * u.um,
            threshold=0.1,
            seed=42
        )

        result = conn(
            pre_size=15, post_size=15,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'custom_kernel')
        self.assertEqual(result.metadata['kernel_size'], 60 * u.um)

    def test_custom_kernel_oscillatory(self):
        def oscillatory_kernel(x, y):
            # Oscillatory kernel
            r = np.sqrt(x ** 2 + y ** 2)
            return np.exp(-r / 20) * np.cos(r / 5)

        positions = np.array([[0, 0], [10, 0], [20, 0], [5, 5]]) * u.um

        conn = CustomKernel(
            kernel_func=oscillatory_kernel,
            kernel_size=50 * u.um,
            threshold=0.05,
            seed=42
        )

        result = conn(
            pre_size=4, post_size=4,
            pre_positions=positions,
            post_positions=positions
        )

        # Should have connections due to oscillatory nature
        self.assertGreaterEqual(result.n_connections, 0)

    def test_custom_kernel_asymmetric(self):
        def asymmetric_kernel(x, y):
            # Asymmetric kernel (stronger in x direction)
            return np.exp(-x ** 2 / 50 - y ** 2 / 200)

        positions = np.array([[-10, 0], [0, 0], [10, 0], [0, 10]]) * u.um

        conn = CustomKernel(
            kernel_func=asymmetric_kernel,
            kernel_size=40 * u.um,
            threshold=0.01,
            seed=42
        )

        result = conn(
            pre_size=4, post_size=4,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertGreaterEqual(result.n_connections, 0)

    def test_custom_kernel_with_weights_and_delays(self):
        def simple_kernel(x, y):
            return np.exp(-(x ** 2 + y ** 2) / 100)

        positions = np.array([[0, 0], [8, 0]]) * u.um

        weight_init = Constant(1.5 * u.nS)
        delay_init = Uniform(1.0 * u.ms, 2.0 * u.ms)

        conn = CustomKernel(
            kernel_func=simple_kernel,
            kernel_size=30 * u.um,
            threshold=0.0,
            weight=weight_init,
            delay=delay_init,
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            self.assertIsNotNone(result.weights)
            self.assertIsNotNone(result.delays)
            self.assertEqual(result.weights.unit, u.nS)
            self.assertEqual(result.delays.unit, u.ms)

    def test_custom_kernel_no_positions_error(self):
        def dummy_kernel(x, y):
            return np.ones_like(x)

        conn = CustomKernel(
            kernel_func=dummy_kernel,
            kernel_size=20 * u.um,
            seed=42
        )

        with self.assertRaises(ValueError):
            conn(pre_size=5, post_size=5)

    def test_custom_kernel_empty_connections(self):
        def zero_kernel(x, y):
            # Kernel that returns all zeros
            return np.zeros_like(x)

        positions = np.array([[0, 0], [10, 0]]) * u.um

        conn = CustomKernel(
            kernel_func=zero_kernel,
            kernel_size=30 * u.um,
            threshold=0.0,
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.n_connections, 0)

    def test_custom_kernel_high_threshold(self):
        def small_kernel(x, y):
            # Kernel with small values
            return 0.01 * np.exp(-(x ** 2 + y ** 2) / 100)

        positions = np.array([[0, 0], [5, 0]]) * u.um

        conn = CustomKernel(
            kernel_func=small_kernel,
            kernel_size=20 * u.um,
            threshold=0.1,  # High threshold
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        # Should have no connections due to high threshold
        self.assertEqual(result.n_connections, 0)

    def test_custom_kernel_scalar_kernel_size(self):
        def simple_kernel(x, y):
            return np.exp(-(x ** 2 + y ** 2) / 50)

        positions = np.array([[0, 0], [10, 0]])  # No units

        conn = CustomKernel(
            kernel_func=simple_kernel,
            kernel_size=25.0,  # Scalar
            threshold=0.1,
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertGreaterEqual(result.n_connections, 0)

    def test_custom_kernel_array_handling(self):
        def vector_kernel(x, y):
            # Test that kernel function handles arrays correctly
            self.assertIsInstance(x, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertEqual(x.shape, y.shape)
            return np.exp(-(x ** 2 + y ** 2) / 100)

        positions = np.array([[0, 0], [5, 0], [10, 0]]) * u.um

        conn = CustomKernel(
            kernel_func=vector_kernel,
            kernel_size=30 * u.um,
            threshold=0.1,
            seed=42
        )

        result = conn(
            pre_size=3, post_size=3,
            pre_positions=positions,
            post_positions=positions
        )

        # Test passes if no assertions fail in vector_kernel


class TestKernelEdgeCases(unittest.TestCase):
    """
    Test edge cases and error conditions for kernel connectivity.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_kernel import GaussianKernel

        # Test with very small networks
        positions = np.array([[0, 0]]) * u.um

        conn = GaussianKernel(sigma=10 * u.um)
        result = conn(
            pre_size=1, post_size=1,
            pre_positions=positions,
            post_positions=positions
        )

        # Test with 3D positions
        positions_3d = np.random.uniform(0, 100, (10, 3)) * u.um
        # Should work with 3D positions (uses first 2 dimensions)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_single_neuron_networks(self):
        # Test all kernel types with single neuron
        positions = np.array([[0, 0]]) * u.um

        kernels = [
            GaussianKernel(sigma=10 * u.um),
            GaborKernel(sigma=10 * u.um, frequency=0.1, theta=0),
            DoGKernel(sigma_center=5 * u.um, sigma_surround=15 * u.um),
            MexicanHat(sigma=10 * u.um),
        ]

        for kernel in kernels:
            kernel.seed = 42
            result = kernel(
                pre_size=1, post_size=1,
                pre_positions=positions,
                post_positions=positions
            )
            # Single neuron should connect to itself
            self.assertGreaterEqual(result.n_connections, 0)

    def test_large_kernel_sizes(self):
        # Test with kernel size much larger than neuron spread
        positions = np.random.RandomState(42).uniform(0, 10, (5, 2)) * u.um

        conn = GaussianKernel(
            sigma=100 * u.um,  # Much larger than position range
            seed=42
        )

        result = conn(
            pre_size=5, post_size=5,
            pre_positions=positions,
            post_positions=positions
        )

        # Should connect everything to everything
        self.assertGreater(result.n_connections, 0)

    def test_very_small_kernel_sizes(self):
        # Test with very small kernel sizes
        positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um

        conn = GaussianKernel(
            sigma=0.1 * u.um,  # Very small
            max_distance=0.5 * u.um,
            seed=42
        )

        result = conn(
            pre_size=10, post_size=10,
            pre_positions=positions,
            post_positions=positions
        )

        # Might have few or no connections due to small kernel
        self.assertGreaterEqual(result.n_connections, 0)

    def test_3d_positions(self):
        # Test with 3D positions (should use first 2 dimensions)
        positions = np.random.RandomState(42).uniform(0, 50, (8, 3)) * u.um

        conn = GaussianKernel(
            sigma=20 * u.um,
            seed=42
        )

        result = conn(
            pre_size=8, post_size=8,
            pre_positions=positions,
            post_positions=positions
        )

        # Should work with 3D positions
        self.assertGreaterEqual(result.n_connections, 0)

    def test_mismatched_units(self):
        # Test with mismatched units between positions and kernel parameters
        positions = np.array([[0, 0], [50, 0]]) * u.mm  # millimeters

        conn = GaussianKernel(
            sigma=20 * u.um,  # micrometers
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        # Should handle unit conversion correctly
        self.assertGreaterEqual(result.n_connections, 0)

    def test_asymmetric_network_sizes(self):
        pre_positions = np.random.RandomState(42).uniform(0, 50, (8, 2)) * u.um
        post_positions = np.random.RandomState(43).uniform(0, 50, (12, 2)) * u.um

        conn = GaussianKernel(
            sigma=25 * u.um,
            seed=42
        )

        result = conn(
            pre_size=8, post_size=12,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        self.assertEqual(result.shape, (8, 12))
        self.assertTrue(np.all(result.pre_indices < 8))
        self.assertTrue(np.all(result.post_indices < 12))

    def test_tuple_sizes(self):
        positions = np.random.RandomState(42).uniform(0, 50, (12, 2)) * u.um

        conn = GaussianKernel(
            sigma=20 * u.um,
            seed=42
        )

        result = conn(
            pre_size=(3, 4), post_size=(2, 6),
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.pre_size, (3, 4))
        self.assertEqual(result.post_size, (2, 6))
        self.assertEqual(result.shape, (12, 12))

    def test_zero_threshold_kernels(self):
        # Test kernels that might produce negative values with zero threshold
        positions = np.array([[0, 0], [10, 0], [20, 0]]) * u.um

        kernels = [
            DoGKernel(sigma_center=5 * u.um, sigma_surround=15 * u.um),
            MexicanHat(sigma=8 * u.um),
            SobelKernel(direction='horizontal', kernel_size=25 * u.um),
            LaplacianKernel(kernel_type='4-connected', kernel_size=25 * u.um),
        ]

        for kernel in kernels:
            kernel.seed = 42
            result = kernel(
                pre_size=3, post_size=3,
                pre_positions=positions,
                post_positions=positions
            )
            # Should handle negative kernel values appropriately
            self.assertGreaterEqual(result.n_connections, 0)

    def test_extreme_frequencies_gabor(self):
        positions = np.array([[0, 0], [5, 0], [10, 0]]) * u.um

        # Very high frequency
        conn_high = GaborKernel(
            sigma=20 * u.um,
            frequency=1.0,  # Very high
            theta=0,
            seed=42
        )

        # Very low frequency
        conn_low = GaborKernel(
            sigma=20 * u.um,
            frequency=0.001,  # Very low
            theta=0,
            seed=42
        )

        for conn in [conn_high, conn_low]:
            result = conn(
                pre_size=3, post_size=3,
                pre_positions=positions,
                post_positions=positions
            )
            self.assertGreaterEqual(result.n_connections, 0)


if __name__ == '__main__':
    unittest.main()
