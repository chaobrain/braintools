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
Kernel-based connectivity patterns for point neurons with spatial positions.

These patterns apply convolution-like kernels to spatially arranged point neurons,
creating connections based on spatial relationships weighted by kernel functions.
Useful for implementing center-surround receptive fields, orientation selectivity,
and other spatially-structured connectivity patterns in spiking neural networks.
"""

from typing import Optional, Callable

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike
from scipy.spatial.distance import cdist

from braintools.init import param, Initializer
from ._base import PointConnectivity, ConnectionResult

__all__ = [
    'Conv2dKernel',
    'GaussianKernel',
    'GaborKernel',
    'DoGKernel',
    'MexicanHat',
    'SobelKernel',
    'LaplacianKernel',
    'CustomKernel'
]


class Conv2dKernel(PointConnectivity):
    """Convolutional kernel connectivity for spatially arranged point neurons.

    Applies a 2D convolution kernel to neuron positions, creating connections
    where the kernel weight exceeds a threshold. This allows implementing
    receptive field structures in spiking neural networks.

    Parameters
    ----------
    kernel : np.ndarray
        2D convolution kernel array.
    kernel_size : float or Quantity
        Physical size of the kernel in position units.
    threshold : float
        Connection threshold - only kernel values above this create connections.
    weight : Initialization, optional
        Weight initialization (kernel values are multiplied by this).
    delay : Initialization, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> # Create 5x5 Gaussian-like kernel
        >>> kernel = np.array([
        ...     [0.04, 0.12, 0.18, 0.12, 0.04],
        ...     [0.12, 0.37, 0.56, 0.37, 0.12],
        ...     [0.18, 0.56, 1.00, 0.56, 0.18],
        ...     [0.12, 0.37, 0.56, 0.37, 0.12],
        ...     [0.04, 0.12, 0.18, 0.12, 0.04]
        ... ])
        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>> conn = Conv2dKernel(
        ...     kernel=kernel,
        ...     kernel_size=100 * u.um,
        ...     threshold=0.1,
        ...     weight=1.0 * u.nS
        ... )
        >>> result = conn(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        kernel: np.ndarray,
        kernel_size: ArrayLike,
        threshold: float = 0.0,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel = np.asarray(kernel)
        if self.kernel.ndim != 2:
            raise ValueError("Kernel must be 2D array")
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate convolutional kernel connections."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']
        pre_positions = kwargs.get('pre_positions', None)
        post_positions = kwargs.get('post_positions', None)

        if pre_positions is None or post_positions is None:
            raise ValueError("Positions required for kernel connectivity")

        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Extract position values and units
        pre_pos_val, pos_unit = u.split_mantissa_unit(pre_positions)
        post_pos_val = u.Quantity(post_positions).to(pos_unit).mantissa

        # Get kernel size in position units
        if isinstance(self.kernel_size, u.Quantity):
            kernel_size_val = u.Quantity(self.kernel_size).to(pos_unit).mantissa
        else:
            kernel_size_val = self.kernel_size

        # Kernel grid coordinates (centered at 0)
        kh, kw = self.kernel.shape
        kernel_y = np.linspace(-kernel_size_val / 2, kernel_size_val / 2, kh)
        kernel_x = np.linspace(-kernel_size_val / 2, kernel_size_val / 2, kw)
        kernel_grid_y, kernel_grid_x = np.meshgrid(kernel_y, kernel_x, indexing='ij')

        pre_indices = []
        post_indices = []
        kernel_weights = []

        # For each post neuron, check if pre neurons fall within kernel support
        for post_idx in range(post_num):
            post_pos = post_pos_val[post_idx]

            # Calculate relative positions of pre neurons to this post neuron
            rel_positions = pre_pos_val - post_pos  # Shape: (pre_num, 2)

            # Find pre neurons within kernel support
            in_range_x = np.abs(rel_positions[:, 0]) <= kernel_size_val / 2
            in_range_y = np.abs(rel_positions[:, 1]) <= kernel_size_val / 2
            in_range = in_range_x & in_range_y
            candidate_pre = np.where(in_range)[0]

            for pre_idx in candidate_pre:
                rel_x, rel_y = rel_positions[pre_idx]

                # Find nearest kernel position
                ki = np.argmin(np.abs(kernel_grid_y[:, 0] - rel_y))
                kj = np.argmin(np.abs(kernel_grid_x[0, :] - rel_x))

                kernel_val = self.kernel[ki, kj]

                if kernel_val > self.threshold:
                    pre_indices.append(pre_idx)
                    post_indices.append(post_idx)
                    kernel_weights.append(kernel_val)

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                pre_size=pre_size, post_size=post_size,
                pre_positions=pre_positions, post_positions=post_positions,
                model_type='point'
            )

        pre_indices = np.array(pre_indices, dtype=np.int64)
        post_indices = np.array(post_indices, dtype=np.int64)
        kernel_weights = np.array(kernel_weights)
        n_connections = len(pre_indices)

        # Generate base weights
        weights = param(
            self.weight_init,
            n_connections,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            rng=self.rng
        )

        # Multiply by kernel weights
        if weights is not None:
            weights_vals, weights_unit = u.split_mantissa_unit(weights)
            if u.math.isscalar(weights_vals):
                weights_vals = np.full(n_connections, weights_vals)
            else:
                weights_vals = np.asarray(weights_vals)

            final_weights = weights_vals * kernel_weights
            if weights_unit is not None:
                weights = u.maybe_decimal(final_weights * weights_unit)
            else:
                weights = final_weights
        else:
            weights = kernel_weights

        delays = param(
            self.delay_init,
            n_connections,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            rng=self.rng
        )

        return ConnectionResult(
            pre_indices, post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=pre_positions,
            post_positions=post_positions,
            metadata={'pattern': 'conv_kernel', 'kernel_shape': self.kernel.shape, 'kernel_size': self.kernel_size}
        )


class GaussianKernel(PointConnectivity):
    """Gaussian kernel connectivity for center-surround receptive fields.

    Creates connections weighted by a 2D Gaussian function of distance,
    useful for implementing smooth spatial receptive fields.

    Parameters
    ----------
    sigma : float or Quantity
        Standard deviation of Gaussian.
    max_distance : float or Quantity, optional
        Maximum distance for connections (default: 3*sigma).
    normalize : bool
        Whether to normalize the Gaussian (default: True).
    weight : Initialization, optional
        Weight initialization (Gaussian is multiplied by this).
    delay : Initialization, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>> gauss = GaussianKernel(
        ...     sigma=50 * u.um,
        ...     max_distance=150 * u.um,
        ...     weight=2.0 * u.nS
        ... )
        >>> result = gauss(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        sigma: ArrayLike,
        max_distance: Optional[ArrayLike] = None,
        normalize: bool = True,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.max_distance = max_distance
        self.normalize = normalize
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate Gaussian kernel connections."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']
        pre_positions = kwargs.get('pre_positions', None)
        post_positions = kwargs.get('post_positions', None)

        if pre_positions is None or post_positions is None:
            raise ValueError("Positions required for Gaussian kernel connectivity")

        # Calculate distances
        pre_pos_val, pos_unit = u.split_mantissa_unit(pre_positions)
        post_pos_val = u.Quantity(post_positions).to(pos_unit).mantissa
        distances = cdist(pre_pos_val, post_pos_val)

        # Get sigma and max_distance in position units
        if isinstance(self.sigma, u.Quantity):
            sigma_val = u.Quantity(self.sigma).to(pos_unit).mantissa
        else:
            sigma_val = self.sigma

        if self.max_distance is not None:
            if isinstance(self.max_distance, u.Quantity):
                max_dist_val = u.Quantity(self.max_distance).to(pos_unit).mantissa
            else:
                max_dist_val = self.max_distance
        else:
            max_dist_val = 3 * sigma_val

        # Calculate Gaussian weights
        gaussian_weights = np.exp(-distances ** 2 / (2 * sigma_val ** 2))

        if self.normalize:
            gaussian_weights /= (2 * np.pi * sigma_val ** 2)

        # Apply distance threshold
        connection_mask = distances <= max_dist_val
        gaussian_weights = gaussian_weights * connection_mask

        # Get connections above threshold
        pre_indices, post_indices = np.where(gaussian_weights > 1e-6)
        gaussian_weights = gaussian_weights[pre_indices, post_indices]

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                pre_size=pre_size, post_size=post_size,
                pre_positions=pre_positions, post_positions=post_positions,
                model_type='point'
            )

        n_connections = len(pre_indices)

        # Generate base weights
        weights = param(
            self.weight_init,
            n_connections,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            rng=self.rng
        )

        # Multiply by Gaussian weights
        if weights is not None:
            weights_vals, weights_unit = u.split_mantissa_unit(weights)
            if u.math.isscalar(weights_vals):
                weights_vals = np.full(n_connections, weights_vals)
            else:
                weights_vals = np.asarray(weights_vals)

            final_weights = weights_vals * gaussian_weights
            if weights_unit is not None:
                weights = u.maybe_decimal(final_weights * weights_unit)
            else:
                weights = final_weights
        else:
            weights = gaussian_weights

        delays = param(
            self.delay_init,
            n_connections,
            rng=self.rng,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        return ConnectionResult(
            pre_indices, post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=pre_positions,
            post_positions=post_positions,
            metadata={
                'pattern': 'gaussian_kernel',
                'sigma': self.sigma,
                'max_distance': self.max_distance
            }
        )


class GaborKernel(PointConnectivity):
    """Gabor kernel connectivity for orientation-selective receptive fields.

    Implements Gabor filters in spatial connectivity, useful for creating
    orientation-selective neurons similar to V1 simple cells.

    Parameters
    ----------
    sigma : float or Quantity
        Standard deviation of Gaussian envelope.
    frequency : float
        Frequency of sinusoidal component (cycles per unit distance).
    theta : float
        Orientation angle in radians.
    phase : float
        Phase offset in radians (default: 0).
    max_distance : float or Quantity, optional
        Maximum distance for connections (default: 3*sigma).
    weight : Initialization, optional
        Weight initialization (Gabor values are multiplied by this).
    delay : Initialization, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>> gabor = GaborKernel(
        ...     sigma=50 * u.um,
        ...     frequency=0.02,  # 1 cycle per 50 um
        ...     theta=np.pi / 4,  # 45 degrees
        ...     weight=1.0 * u.nS
        ... )
        >>> result = gabor(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        sigma: ArrayLike,
        frequency: float,
        theta: float,
        phase: float = 0.0,
        max_distance: Optional[ArrayLike] = None,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.frequency = frequency
        self.theta = theta
        self.phase = phase
        self.max_distance = max_distance
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate Gabor kernel connections."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']
        pre_positions = kwargs.get('pre_positions', None)
        post_positions = kwargs.get('post_positions', None)

        if pre_positions is None or post_positions is None:
            raise ValueError("Positions required for Gabor kernel connectivity")

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Extract position values
        pre_pos_val, pos_unit = u.split_mantissa_unit(pre_positions)
        post_pos_val = u.Quantity(post_positions).to(pos_unit).mantissa

        # Get sigma and max_distance in position units
        if isinstance(self.sigma, u.Quantity):
            sigma_val = u.Quantity(self.sigma).to(pos_unit).mantissa
        else:
            sigma_val = self.sigma

        if self.max_distance is not None:
            if isinstance(self.max_distance, u.Quantity):
                max_dist_val = u.Quantity(self.max_distance).to(pos_unit).mantissa
            else:
                max_dist_val = self.max_distance
        else:
            max_dist_val = 3 * sigma_val

        pre_indices = []
        post_indices = []
        gabor_weights = []

        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # For each post neuron
        for post_idx in range(post_num):
            post_pos = post_pos_val[post_idx]

            # Relative positions
            rel_pos = pre_pos_val - post_pos
            distances = np.sqrt(np.sum(rel_pos ** 2, axis=1))

            # Filter by max distance
            in_range = distances <= max_dist_val
            candidate_pre = np.where(in_range)[0]

            for pre_idx in candidate_pre:
                dx, dy = rel_pos[pre_idx]

                # Rotate coordinates
                x_rot = dx * cos_theta + dy * sin_theta
                y_rot = -dx * sin_theta + dy * cos_theta

                # Calculate Gabor function
                gaussian = np.exp(-(x_rot ** 2 + y_rot ** 2) / (2 * sigma_val ** 2))
                sinusoid = np.cos(2 * np.pi * self.frequency * x_rot + self.phase)
                gabor_val = gaussian * sinusoid

                if np.abs(gabor_val) > 1e-3:
                    pre_indices.append(pre_idx)
                    post_indices.append(post_idx)
                    gabor_weights.append(gabor_val)

        metadata = {
            'pattern': 'gabor_kernel',
            'sigma': self.sigma,
            'frequency': self.frequency,
            'theta': self.theta,
            'phase': self.phase
        }

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
                model_type='point',
                metadata=metadata,
            )

        pre_indices = np.array(pre_indices, dtype=np.int64)
        post_indices = np.array(post_indices, dtype=np.int64)
        gabor_weights = np.array(gabor_weights)
        n_connections = len(pre_indices)

        # Generate base weights
        weights = param(
            self.weight_init,
            n_connections,
            rng=self.rng,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        # Multiply by Gabor weights
        if weights is not None:
            weights_vals, weights_unit = u.split_mantissa_unit(weights)
            if u.math.isscalar(weights_vals):
                weights_vals = np.full(n_connections, weights_vals)
            else:
                weights_vals = np.asarray(weights_vals)

            final_weights = weights_vals * gabor_weights
            if weights_unit is not None:
                weights = u.maybe_decimal(final_weights * weights_unit)
            else:
                weights = final_weights
        else:
            weights = gabor_weights

        delays = param(
            self.delay_init,
            n_connections,
            rng=self.rng,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        return ConnectionResult(
            pre_indices,
            post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=pre_positions,
            post_positions=post_positions,
            metadata=metadata,
        )


class DoGKernel(PointConnectivity):
    """Difference of Gaussians (DoG) kernel for center-surround receptive fields.

    Implements DoG filters commonly found in retinal ganglion cells and LGN neurons,
    with excitatory center and inhibitory surround (or vice versa).

    Parameters
    ----------
    sigma_center : float or Quantity
        Standard deviation of center Gaussian.
    sigma_surround : float or Quantity
        Standard deviation of surround Gaussian.
    amplitude_center : float
        Amplitude of center Gaussian (default: 1.0).
    amplitude_surround : float
        Amplitude of surround Gaussian (default: 0.8).
    max_distance : float or Quantity, optional
        Maximum distance for connections (default: 3*sigma_surround).
    weight : Initialization, optional
        Weight initialization (DoG values are multiplied by this).
    delay : Initialization, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>> dog = DoGKernel(
        ...     sigma_center=30 * u.um,
        ...     sigma_surround=60 * u.um,
        ...     amplitude_center=1.0,
        ...     amplitude_surround=0.8,
        ...     weight=1.0 * u.nS
        ... )
        >>> result = dog(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        sigma_center: ArrayLike,
        sigma_surround: ArrayLike,
        amplitude_center: float = 1.0,
        amplitude_surround: float = 0.8,
        max_distance: Optional[ArrayLike] = None,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sigma_center = sigma_center
        self.sigma_surround = sigma_surround
        self.amplitude_center = amplitude_center
        self.amplitude_surround = amplitude_surround
        self.max_distance = max_distance
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate DoG kernel connections."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']
        pre_positions = kwargs.get('pre_positions', None)
        post_positions = kwargs.get('post_positions', None)

        if pre_positions is None or post_positions is None:
            raise ValueError("Positions required for DoG kernel connectivity")

        # Calculate distances
        pre_pos_val, pos_unit = u.split_mantissa_unit(pre_positions)
        post_pos_val = u.Quantity(post_positions).to(pos_unit).mantissa
        distances = cdist(pre_pos_val, post_pos_val)

        # Get sigma values in position units
        sigma_c_val = u.Quantity(self.sigma_center).to(pos_unit).mantissa
        sigma_s_val = u.Quantity(self.sigma_surround).to(pos_unit).mantissa
        if self.max_distance is not None:
            max_dist_val = u.Quantity(self.max_distance).to(pos_unit).mantissa
        else:
            max_dist_val = 3 * sigma_s_val

        # Calculate DoG weights
        center_gauss = self.amplitude_center * np.exp(-distances ** 2 / (2 * sigma_c_val ** 2))
        surround_gauss = self.amplitude_surround * np.exp(-distances ** 2 / (2 * sigma_s_val ** 2))
        dog_weights = center_gauss - surround_gauss

        # Apply distance threshold
        connection_mask = distances <= max_dist_val
        dog_weights = dog_weights * connection_mask

        # Get connections above threshold
        pre_indices, post_indices = np.where(np.abs(dog_weights) > 1e-4)
        dog_weights = dog_weights[pre_indices, post_indices]

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
                model_type='point',
            )

        n_connections = len(pre_indices)

        # Generate base weights
        weights = param(
            self.weight_init, n_connections,
            param_type='weight', pre_size=pre_size, post_size=post_size,
            pre_positions=pre_positions, post_positions=post_positions,
            rng=self.rng
        )

        # Multiply by DoG weights
        if weights is not None:
            weights_vals, weights_unit = u.split_mantissa_unit(weights)
            if u.math.isscalar(weights_vals):
                weights_vals = np.full(n_connections, weights_vals)
            else:
                weights_vals = np.asarray(weights_vals)

            final_weights = weights_vals * dog_weights
            if weights_unit is not None:
                weights = u.maybe_decimal(final_weights * weights_unit)
            else:
                weights = final_weights
        else:
            weights = dog_weights

        delays = param(
            self.delay_init,
            n_connections,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            rng=self.rng
        )

        return ConnectionResult(
            pre_indices,
            post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            pre_positions=pre_positions,
            post_positions=post_positions,
            metadata={
                'pattern': 'dog_kernel',
                'sigma_center': self.sigma_center,
                'sigma_surround': self.sigma_surround
            }
        )


class MexicanHat(DoGKernel):
    """Mexican hat (Laplacian of Gaussian) connectivity pattern.

    A special case of DoG with specific amplitude ratios to approximate
    the Laplacian of Gaussian. Creates strong lateral inhibition patterns.

    Parameters
    ----------
    sigma : float or Quantity
        Standard deviation of the Gaussian (surround sigma will be sqrt(2)*sigma).
    max_distance : float or Quantity, optional
        Maximum distance for connections (default: 4*sigma).
    weight : Initialization, optional
        Weight initialization.
    delay : Initialization, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>> mexican = MexicanHat(
        ...     sigma=40 * u.um,
        ...     weight=1.0 * u.nS
        ... )
        >>> result = mexican(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        sigma: ArrayLike,
        max_distance: Optional[ArrayLike] = None,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        # Mexican hat is DoG with sigma_surround = sqrt(2) * sigma_center
        # and amplitude_center = 2, amplitude_surround = 1
        if isinstance(sigma, u.Quantity):
            sigma_surround = sigma * np.sqrt(2)
        else:
            sigma_surround = sigma * np.sqrt(2)

        if max_distance is None:
            if isinstance(sigma, u.Quantity):
                max_distance = 4 * sigma
            else:
                max_distance = 4 * sigma

        super().__init__(
            sigma_center=sigma,
            sigma_surround=sigma_surround,
            amplitude_center=2.0,
            amplitude_surround=1.0,
            max_distance=max_distance,
            weight=weight,
            delay=delay,
            **kwargs
        )


class SobelKernel(PointConnectivity):
    """Sobel edge detection kernel for orientation-selective connectivity.

    Implements Sobel operators for detecting edges at specific orientations,
    useful for implementing orientation-selective connectivity patterns.

    Parameters
    ----------
    direction : str
        Direction of edge detection ('horizontal', 'vertical', 'both').
    kernel_size : float or Quantity
        Physical size of the 3x3 kernel in position units.
    weight : Initialization, optional
        Weight initialization (Sobel values are multiplied by this).
    delay : Initialization, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>> sobel = SobelKernel(
        ...     direction='horizontal',
        ...     kernel_size=60 * u.um,
        ...     weight=1.0 * u.nS
        ... )
        >>> result = sobel(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        direction: str = 'horizontal',
        kernel_size: ArrayLike = 1.0,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.direction = direction
        self.kernel_size = kernel_size
        self.weight_init = weight
        self.delay_init = delay

        # Define Sobel kernels
        if direction == 'horizontal':
            self.kernel = np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ], dtype=np.float32)
        elif direction == 'vertical':
            self.kernel = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=np.float32)
        elif direction == 'both':
            # Use magnitude of both directions
            self.kernel_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
            self.kernel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            self.kernel = None  # Will compute both
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate Sobel kernel connections."""
        # Delegate to Conv2dKernel
        if self.kernel is not None:
            conv = Conv2dKernel(
                kernel=self.kernel,
                kernel_size=self.kernel_size,
                threshold=0.1,
                weight=self.weight_init,
                delay=self.delay_init,
                seed=self.seed
            )
            result = conv.generate(**kwargs)
            result.metadata['pattern'] = 'sobel_kernel'
            result.metadata['direction'] = self.direction
            return result
        else:
            # Both directions - compute magnitude
            conv_h = Conv2dKernel(
                kernel=self.kernel_h,
                kernel_size=self.kernel_size,
                threshold=0.1,
                weight=self.weight_init,
                delay=self.delay_init,
                seed=self.seed
            )
            conv_v = Conv2dKernel(
                kernel=self.kernel_v,
                kernel_size=self.kernel_size,
                threshold=0.1,
                weight=self.weight_init,
                delay=self.delay_init,
                seed=self.seed + 1 if self.seed is not None else None
            )
            # Union of both
            from ._base import CompositeConnectivity
            composite = CompositeConnectivity(conv_h, conv_v, 'union')
            result = composite.generate(**kwargs)
            result.metadata['pattern'] = 'sobel_kernel'
            result.metadata['direction'] = 'both'
            return result


class LaplacianKernel(PointConnectivity):
    """Laplacian kernel for edge detection connectivity.

    Implements Laplacian operators for detecting discontinuities and edges,
    useful for lateral inhibition and edge enhancement.

    Parameters
    ----------
    kernel_type : str
        Type of Laplacian ('4-connected', '8-connected').
    kernel_size : float or Quantity
        Physical size of the kernel in position units.
    weight : Initialization, optional
        Weight initialization (Laplacian values are multiplied by this).
    delay : Initialization, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>> laplacian = LaplacianKernel(
        ...     kernel_type='4-connected',
        ...     kernel_size=60 * u.um,
        ...     weight=1.0 * u.nS
        ... )
        >>> result = laplacian(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        kernel_type: str = '4-connected',
        kernel_size: ArrayLike = 1.0,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        self.weight_init = weight
        self.delay_init = delay

        # Define Laplacian kernels
        if kernel_type == '4-connected':
            self.kernel = np.array([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=np.float32)
        elif kernel_type == '8-connected':
            self.kernel = np.array([
                [1, 1, 1],
                [1, -8, 1],
                [1, 1, 1]
            ], dtype=np.float32)
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate Laplacian kernel connections."""
        # Delegate to Conv2dKernel
        conv = Conv2dKernel(
            kernel=self.kernel,
            kernel_size=self.kernel_size,
            threshold=0.1,
            weight=self.weight_init,
            delay=self.delay_init,
            seed=self.seed
        )
        result = conv.generate(**kwargs)
        result.metadata['pattern'] = 'laplacian_kernel'
        result.metadata['kernel_type'] = self.kernel_type
        return result


class CustomKernel(PointConnectivity):
    """Custom kernel connectivity using user-defined kernel function.

    Allows implementing arbitrary spatial kernel functions for connectivity.

    Parameters
    ----------
    kernel_func : callable
        Function that takes (x, y) coordinates and returns kernel value.
        Should accept arrays and return array of same shape.
    kernel_size : float or Quantity
        Physical size of the kernel support in position units.
    threshold : float
        Connection threshold (default: 0.0).
    weight : Initialization, optional
        Weight initialization (kernel values are multiplied by this).
    delay : Initialization, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> def my_kernel(x, y):
        ...     # Custom kernel function
        ...     r = np.sqrt(x**2 + y**2)
        ...     return np.exp(-r/50) * np.cos(r/10)
        >>>
        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>> custom = CustomKernel(
        ...     kernel_func=my_kernel,
        ...     kernel_size=200 * u.um,
        ...     threshold=0.1,
        ...     weight=1.0 * u.nS
        ... )
        >>> result = custom(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        kernel_func: Callable,
        kernel_size: ArrayLike,
        threshold: float = 0.0,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_func = kernel_func
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate custom kernel connections."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']
        pre_positions = kwargs.get('pre_positions', None)
        post_positions = kwargs.get('post_positions', None)

        if pre_positions is None or post_positions is None:
            raise ValueError("Positions required for custom kernel connectivity")

        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Extract position values
        pre_pos_val, pos_unit = u.split_mantissa_unit(pre_positions)
        post_pos_val = u.Quantity(post_positions).to(pos_unit).mantissa

        # Get kernel size in position units
        if isinstance(self.kernel_size, u.Quantity):
            kernel_size_val = u.Quantity(self.kernel_size).to(pos_unit).mantissa
        else:
            kernel_size_val = self.kernel_size

        pre_indices = []
        post_indices = []
        kernel_weights = []

        # For each post neuron
        for post_idx in range(post_num):
            post_pos = post_pos_val[post_idx]

            # Relative positions
            rel_pos = pre_pos_val - post_pos
            distances = np.sqrt(np.sum(rel_pos ** 2, axis=1))

            # Filter by kernel size
            in_range = distances <= kernel_size_val / 2
            candidate_pre = np.where(in_range)[0]

            if len(candidate_pre) > 0:
                rel_x = rel_pos[candidate_pre, 0]
                rel_y = rel_pos[candidate_pre, 1]

                # Evaluate kernel function
                kernel_vals = self.kernel_func(rel_x, rel_y)

                # Apply threshold
                above_threshold = np.abs(kernel_vals) > self.threshold
                valid_pre = candidate_pre[above_threshold]
                valid_weights = kernel_vals[above_threshold]

                pre_indices.extend(valid_pre)
                post_indices.extend([post_idx] * len(valid_pre))
                kernel_weights.extend(valid_weights)

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
                model_type='point',
            )

        pre_indices = np.array(pre_indices, dtype=np.int64)
        post_indices = np.array(post_indices, dtype=np.int64)
        kernel_weights = np.array(kernel_weights)
        n_connections = len(pre_indices)

        # Generate base weights
        weights = param(
            self.weight_init,
            n_connections,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            rng=self.rng
        )

        # Multiply by kernel weights
        if weights is not None:
            weights_vals, weights_unit = u.split_mantissa_unit(weights)
            if u.math.isscalar(weights_vals):
                weights_vals = np.full(n_connections, weights_vals)
            else:
                weights_vals = np.asarray(weights_vals)

            final_weights = weights_vals * kernel_weights
            if weights_unit is not None:
                weights = u.maybe_decimal(final_weights * weights_unit)
            else:
                weights = final_weights
        else:
            weights = kernel_weights

        delays = param(
            self.delay_init,
            n_connections,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            rng=self.rng
        )

        return ConnectionResult(
            pre_indices,
            post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=pre_positions, post_positions=post_positions,
            metadata={'pattern': 'custom_kernel', 'kernel_size': self.kernel_size}
        )
