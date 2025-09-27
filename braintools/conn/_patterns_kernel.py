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

"""
Kernel-based connectivity patterns for composable system.

Includes patterns based on convolution kernels:
- Convolutional kernels
- Gaussian kernels
- Gabor kernels
- Difference of Gaussians (DoG)
- Mexican hat kernels
- Sobel kernels
- Laplacian kernels
- Custom kernels
"""

from typing import Optional, Tuple, Union, Callable
import numpy as np
import brainunit as u

from ._composable_base import Connectivity, ConnectionResult


__all__ = [
    # Kernel patterns
    'ConvKernel',
    'GaussianKernel',
    'GaborKernel',
    'DoGKernel',
    'MexicanHat',
    'SobelKernel',
    'LaplacianKernel',
    'CustomKernel'
]

class ConvKernel(Connectivity):
    """Convolutional kernel connectivity pattern.

    Creates connections based on convolutional kernels, useful for
    implementing neural circuits with spatial structure.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer (height, width) or (depth, height, width)
    kernel : np.ndarray
        Convolution kernel array
    stride : int or tuple
        Stride for convolution (default: 1)
    padding : str or int
        Padding mode ('valid', 'same') or padding size (default: 'valid')
    dilation : int
        Dilation factor for kernel (default: 1)
    threshold : float
        Connection threshold (default: 0.0)
    weight_scale : float or Quantity
        Scale factor for connection weights
    """

    __module__ = 'braintools.conn'

    def __init__(self, input_shape: Tuple[int, ...], kernel: np.ndarray,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[str, int] = 'valid',
                 dilation: int = 1,
                 threshold: float = 0.0,
                 weight_scale: Union[float, u.Quantity] = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.threshold = threshold
        self.weight_scale = weight_scale

    def _generate(self, pre_size, post_size, pre_positions=None, post_positions=None):
        from ._kernel import conv_kernel

        # Generate kernel connectivity
        pre_indices, post_indices = conv_kernel(
            self.input_shape, self.kernel, self.stride,
            self.padding, self.dilation, self.threshold, self.seed
        )

        if len(pre_indices) == 0:
            return ConnectionResult(np.array([]), np.array([]),
                                  metadata={'pattern': 'conv_kernel'})

        # Create weights based on kernel values
        weights = np.ones(len(pre_indices))
        if isinstance(self.weight_scale, u.Quantity):
            weights = weights * self.weight_scale
        else:
            weights = weights * self.weight_scale

        return ConnectionResult(
            pre_indices, post_indices, weights=weights,
            metadata={'pattern': 'conv_kernel', 'input_shape': self.input_shape}
        )


class GaussianKernel(Connectivity):
    """Gaussian kernel connectivity pattern.

    Creates connections using a Gaussian convolution kernel, useful for
    implementing center-surround receptive fields.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer
    sigma : float or Quantity
        Standard deviation of Gaussian (with units if using brainunit)
    kernel_size : int, optional
        Size of kernel (default: 4*sigma + 1)
    stride : int
        Stride for convolution (default: 1)
    normalize : bool
        Whether to normalize the kernel (default: True)
    weight_scale : float or Quantity
        Scale factor for connection weights
    """

    __module__ = 'braintools.conn'

    def __init__(self, input_shape: Tuple[int, ...],
                 sigma: Union[float, u.Quantity],
                 kernel_size: Optional[int] = None,
                 stride: int = 1,
                 normalize: bool = True,
                 weight_scale: Union[float, u.Quantity] = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.stride = stride
        self.normalize = normalize
        self.weight_scale = weight_scale

    def _generate(self, pre_size, post_size, pre_positions=None, post_positions=None):
        from ._kernel import gaussian_kernel

        # Extract magnitude if using units
        sigma_val = u.get_magnitude(self.sigma) if isinstance(self.sigma, u.Quantity) else self.sigma

        # Generate Gaussian kernel connectivity
        pre_indices, post_indices = gaussian_kernel(
            self.input_shape, sigma_val, self.kernel_size,
            self.stride, self.normalize, self.seed
        )

        if len(pre_indices) == 0:
            return ConnectionResult(np.array([]), np.array([]),
                                  metadata={'pattern': 'gaussian_kernel'})

        # Create weights
        weights = np.ones(len(pre_indices))
        if isinstance(self.weight_scale, u.Quantity):
            weights = weights * self.weight_scale
        else:
            weights = weights * self.weight_scale

        return ConnectionResult(
            pre_indices, post_indices, weights=weights,
            metadata={'pattern': 'gaussian_kernel', 'sigma': self.sigma}
        )


class GaborKernel(Connectivity):
    """Gabor kernel connectivity pattern.

    Creates connections using Gabor filters, useful for implementing
    orientation-selective receptive fields.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer (height, width)
    frequency : float
        Frequency of sinusoidal component (default: 0.1)
    theta : float
        Orientation angle in radians (default: 0)
    sigma : float or Quantity
        Standard deviation of Gaussian envelope
    phase : float
        Phase offset (default: 0)
    kernel_size : int, optional
        Size of kernel (default: 4*sigma + 1)
    stride : int
        Stride for convolution (default: 1)
    weight_scale : float or Quantity
        Scale factor for connection weights
    """

    __module__ = 'braintools.conn'

    def __init__(self, input_shape: Tuple[int, int],
                 frequency: float = 0.1,
                 theta: float = 0,
                 sigma: Union[float, u.Quantity] = 1.0,
                 phase: float = 0,
                 kernel_size: Optional[int] = None,
                 stride: int = 1,
                 weight_scale: Union[float, u.Quantity] = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.frequency = frequency
        self.theta = theta
        self.sigma = sigma
        self.phase = phase
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_scale = weight_scale

    def _generate(self, pre_size, post_size, pre_positions=None, post_positions=None):
        from ._kernel import gabor_kernel

        # Extract magnitude if using units
        sigma_val = u.get_magnitude(self.sigma) if isinstance(self.sigma, u.Quantity) else self.sigma

        # Generate Gabor kernel connectivity
        pre_indices, post_indices = gabor_kernel(
            self.input_shape, self.frequency, self.theta, sigma_val,
            self.phase, self.kernel_size, self.stride, self.seed
        )

        if len(pre_indices) == 0:
            return ConnectionResult(np.array([]), np.array([]),
                                  metadata={'pattern': 'kernel'})

        # Create weights
        weights = np.ones(len(pre_indices))
        if isinstance(self.weight_scale, u.Quantity):
            weights = weights * self.weight_scale
        else:
            weights = weights * self.weight_scale

        return ConnectionResult(
            pre_indices, post_indices, weights=weights,
            metadata={'pattern': 'kernel'}
        )


class DoGKernel(Connectivity):
    """Difference of Gaussians (DoG) kernel connectivity pattern.

    Creates connections using DoG filters, implementing center-surround
    antagonistic receptive fields.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer
    sigma1 : float or Quantity
        Standard deviation of center Gaussian (default: 1.0)
    sigma2 : float or Quantity
        Standard deviation of surround Gaussian (default: 2.0)
    kernel_size : int, optional
        Size of kernel (default: 4*max(sigma1,sigma2) + 1)
    stride : int
        Stride for convolution (default: 1)
    weight_scale : float or Quantity
        Scale factor for connection weights
    """

    __module__ = 'braintools.conn'

    def __init__(self, input_shape: Tuple[int, ...],
                 sigma1: Union[float, u.Quantity] = 1.0,
                 sigma2: Union[float, u.Quantity] = 2.0,
                 kernel_size: Optional[int] = None,
                 stride: int = 1,
                 weight_scale: Union[float, u.Quantity] = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_scale = weight_scale

    def _generate(self, pre_size, post_size, pre_positions=None, post_positions=None):
        from ._kernel import dog_kernel

        # Extract magnitudes if using units
        sigma1_val = u.get_magnitude(self.sigma1) if isinstance(self.sigma1, u.Quantity) else self.sigma1
        sigma2_val = u.get_magnitude(self.sigma2) if isinstance(self.sigma2, u.Quantity) else self.sigma2

        # Generate DoG kernel connectivity
        pre_indices, post_indices = dog_kernel(
            self.input_shape, sigma1_val, sigma2_val,
            self.kernel_size, self.stride, self.seed
        )

        if len(pre_indices) == 0:
            return ConnectionResult(np.array([]), np.array([]),
                                  metadata={'pattern': 'kernel'})

        # Create weights
        weights = np.ones(len(pre_indices))
        if isinstance(self.weight_scale, u.Quantity):
            weights = weights * self.weight_scale
        else:
            weights = weights * self.weight_scale

        return ConnectionResult(
            pre_indices, post_indices, weights=weights,
            metadata={'pattern': 'dog_kernel', 'sigma1': self.sigma1, 'sigma2': self.sigma2}
        )


class MexicanHat(Connectivity):
    """Mexican hat (Laplacian of Gaussian) kernel connectivity pattern.

    Creates connections with center-surround structure using the Mexican hat
    function, useful for lateral inhibition patterns.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer
    sigma : float or Quantity
        Standard deviation of Gaussian
    kernel_size : int, optional
        Size of kernel (default: 6*sigma + 1)
    stride : int
        Stride for convolution (default: 1)
    weight_scale : float or Quantity
        Scale factor for connection weights
    """

    __module__ = 'braintools.conn'

    def __init__(self, input_shape: Tuple[int, ...],
                 sigma: Union[float, u.Quantity] = 1.0,
                 kernel_size: Optional[int] = None,
                 stride: int = 1,
                 weight_scale: Union[float, u.Quantity] = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_scale = weight_scale

    def _generate(self, pre_size, post_size, pre_positions=None, post_positions=None):
        from ._kernel import mexican_hat

        # Extract magnitude if using units
        sigma_val = u.get_magnitude(self.sigma) if isinstance(self.sigma, u.Quantity) else self.sigma

        # Generate Mexican hat connectivity
        pre_indices, post_indices = mexican_hat(
            self.input_shape, sigma_val, self.kernel_size, self.stride, self.seed
        )

        if len(pre_indices) == 0:
            return ConnectionResult(np.array([]), np.array([]),
                                  metadata={'pattern': 'kernel'})

        # Create weights
        weights = np.ones(len(pre_indices))
        if isinstance(self.weight_scale, u.Quantity):
            weights = weights * self.weight_scale
        else:
            weights = weights * self.weight_scale

        return ConnectionResult(
            pre_indices, post_indices, weights=weights,
            metadata={'pattern': 'kernel'}
        )


class SobelKernel(Connectivity):
    """Sobel edge detection kernel connectivity pattern.

    Creates connections using Sobel operators for edge detection,
    useful for implementing orientation-selective connectivity.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer (height, width)
    direction : str
        Direction of edge detection ('horizontal', 'vertical', 'both')
    stride : int
        Stride for convolution (default: 1)
    weight_scale : float or Quantity
        Scale factor for connection weights
    """

    __module__ = 'braintools.conn'

    def __init__(self, input_shape: Tuple[int, int],
                 direction: str = 'both',
                 stride: int = 1,
                 weight_scale: Union[float, u.Quantity] = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.direction = direction
        self.stride = stride
        self.weight_scale = weight_scale

    def _generate(self, pre_size, post_size, pre_positions=None, post_positions=None):
        from ._kernel import sobel_kernel

        # Generate Sobel kernel connectivity
        pre_indices, post_indices = sobel_kernel(
            self.input_shape, self.direction, self.stride, self.seed
        )

        if len(pre_indices) == 0:
            return ConnectionResult(np.array([]), np.array([]),
                                  metadata={'pattern': 'kernel'})

        # Create weights
        weights = np.ones(len(pre_indices))
        if isinstance(self.weight_scale, u.Quantity):
            weights = weights * self.weight_scale
        else:
            weights = weights * self.weight_scale

        return ConnectionResult(
            pre_indices, post_indices, weights=weights,
            metadata={'pattern': 'kernel'}
        )


class LaplacianKernel(Connectivity):
    """Laplacian kernel connectivity pattern.

    Creates connections using Laplacian operators for detecting
    discontinuities and edges.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer
    kernel_type : str
        Type of Laplacian ('4-connected', '8-connected', 'gaussian')
    stride : int
        Stride for convolution (default: 1)
    weight_scale : float or Quantity
        Scale factor for connection weights
    """

    __module__ = 'braintools.conn'

    def __init__(self, input_shape: Tuple[int, ...],
                 kernel_type: str = '4-connected',
                 stride: int = 1,
                 weight_scale: Union[float, u.Quantity] = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.kernel_type = kernel_type
        self.stride = stride
        self.weight_scale = weight_scale

    def _generate(self, pre_size, post_size, pre_positions=None, post_positions=None):
        from ._kernel import laplacian_kernel

        # Generate Laplacian kernel connectivity
        pre_indices, post_indices = laplacian_kernel(
            self.input_shape, self.kernel_type, self.stride, self.seed
        )

        if len(pre_indices) == 0:
            return ConnectionResult(np.array([]), np.array([]),
                                  metadata={'pattern': 'kernel'})

        # Create weights
        weights = np.ones(len(pre_indices))
        if isinstance(self.weight_scale, u.Quantity):
            weights = weights * self.weight_scale
        else:
            weights = weights * self.weight_scale

        return ConnectionResult(
            pre_indices, post_indices, weights=weights,
            metadata={'pattern': 'kernel'}
        )


class CustomKernel(Connectivity):
    """Custom kernel connectivity pattern.

    Creates connections using a user-defined kernel function,
    allowing for arbitrary kernel designs.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer
    kernel_func : callable
        Function that takes coordinates and returns kernel values
    kernel_size : int
        Size of kernel
    stride : int
        Stride for convolution (default: 1)
    weight_scale : float or Quantity
        Scale factor for connection weights
    **kwargs
        Additional arguments for base class and kernel function
    """

    __module__ = 'braintools.conn'

    def __init__(self, input_shape: Tuple[int, ...],
                 kernel_func: Callable,
                 kernel_size: int,
                 stride: int = 1,
                 weight_scale: Union[float, u.Quantity] = 1.0,
                 **kwargs):
        # Separate kernel function kwargs from base class kwargs
        base_kwargs = {}
        kernel_kwargs = {}
        for key, value in kwargs.items():
            if key in ['seed', 'metadata']:
                base_kwargs[key] = value
            else:
                kernel_kwargs[key] = value

        super().__init__(**base_kwargs)
        self.input_shape = input_shape
        self.kernel_func = kernel_func
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_scale = weight_scale
        self.kernel_kwargs = kernel_kwargs

    def _generate(self, pre_size, post_size, pre_positions=None, post_positions=None):
        from ._kernel import custom_kernel

        # Generate custom kernel connectivity
        pre_indices, post_indices = custom_kernel(
            self.input_shape, self.kernel_func, self.kernel_size,
            self.stride, self.seed, **self.kernel_kwargs
        )

        if len(pre_indices) == 0:
            return ConnectionResult(np.array([]), np.array([]),
                                  metadata={'pattern': 'kernel'})

        # Create weights
        weights = np.ones(len(pre_indices))
        if isinstance(self.weight_scale, u.Quantity):
            weights = weights * self.weight_scale
        else:
            weights = weights * self.weight_scale

        return ConnectionResult(
            pre_indices, post_indices, weights=weights,
            metadata={'pattern': 'kernel'}
        )