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
Kernel-based and convolutional connectivity patterns.

Includes:
- Convolutional kernels
- Gabor connectivity
- Difference of Gaussians (DoG)
- Mexican hat connectivity
- Custom kernel connectivity
"""

from typing import Optional, Tuple, Union, Callable

import numpy as np

from braintools._misc import set_module_as

__all__ = [
    'conv_kernel',
    'gaussian_kernel',
    'gabor_kernel',
    'dog_kernel',
    'mexican_hat',
    'sobel_kernel',
    'laplacian_kernel',
    'custom_kernel',
]


@set_module_as('braintools.conn')
def conv_kernel(
    input_shape: Tuple[int, ...],
    kernel: np.ndarray,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[str, int] = 'valid',
    dilation: int = 1,
    threshold: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create convolutional connectivity using a kernel.
    
    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer (height, width) or (depth, height, width).
    kernel : np.ndarray
        Convolution kernel.
    stride : int or tuple
        Stride for convolution.
    padding : str or int
        Padding mode ('valid', 'same') or padding size.
    dilation : int
        Dilation factor for kernel.
    threshold : float
        Connection threshold (only create connections where kernel > threshold).
    seed : int, optional
        Random seed for probabilistic connections.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    """
    rng = np.random if seed is None else np.random.RandomState(seed)

    # Handle different input dimensions
    if len(input_shape) == 1:
        # 1D case
        w = input_shape[0]
        h = 1
        d = 1
        input_shape = (1, 1, w)
        kernel = kernel.reshape(1, 1, -1)
    elif len(input_shape) == 2:
        h, w = input_shape
        d = 1
        input_shape = (1, h, w)
        kernel = kernel.reshape(1, *kernel.shape)
    elif len(input_shape) == 3:
        d, h, w = input_shape
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")

    kernel_shape = kernel.shape

    # Handle stride
    if isinstance(stride, int):
        stride = (stride,) * len(input_shape)

    # Calculate output shape
    if padding == 'same':
        out_shape = input_shape
        pad = tuple((k - 1) // 2 for k in kernel_shape)
    elif padding == 'valid':
        out_shape = tuple((i - k) // s + 1
                          for i, k, s in zip(input_shape, kernel_shape, stride))
        pad = (0, 0, 0)
    else:
        pad = (padding,) * len(input_shape) if isinstance(padding, int) else padding
        out_shape = tuple((i + 2 * p - k) // s + 1
                          for i, k, p, s in zip(input_shape, kernel_shape, pad, stride))

    pre_list = []
    post_list = []

    # Create connections based on kernel
    out_idx = 0
    for out_d in range(out_shape[0]):
        for out_h in range(out_shape[1]):
            for out_w in range(out_shape[2]):
                # Calculate receptive field
                for kd in range(kernel_shape[0]):
                    for kh in range(kernel_shape[1]):
                        for kw in range(kernel_shape[2]):
                            # Input position
                            in_d = out_d * stride[0] + kd * dilation - pad[0]
                            in_h = out_h * stride[1] + kh * dilation - pad[1]
                            in_w = out_w * stride[2] + kw * dilation - pad[2]

                            # Check bounds
                            if (0 <= in_d < d and 0 <= in_h < h and 0 <= in_w < w):
                                weight = kernel[kd, kh, kw]

                                if abs(weight) > threshold:
                                    # Convert to linear indices
                                    in_idx = in_d * h * w + in_h * w + in_w

                                    # Probabilistic connection based on weight
                                    if rng.random() < abs(weight):
                                        pre_list.append(in_idx)
                                        post_list.append(out_idx)

                out_idx += 1

    return np.array(pre_list), np.array(post_list)


@set_module_as('braintools.conn')
def gaussian_kernel(
    input_shape: Tuple[int, ...],
    sigma: float,
    kernel_size: Optional[int] = None,
    stride: int = 1,
    normalize: bool = True,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create connectivity using Gaussian kernel.
    
    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer.
    sigma : float
        Standard deviation of Gaussian.
    kernel_size : int, optional
        Size of kernel (default: 4*sigma + 1).
    stride : int
        Stride for convolution.
    normalize : bool
        Whether to normalize the kernel.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    """
    if kernel_size is None:
        kernel_size = int(4 * sigma) + 1

    # Create Gaussian kernel
    if len(input_shape) == 1:
        # 1D Gaussian
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    elif len(input_shape) == 2:
        # 2D Gaussian
        x = np.arange(kernel_size) - kernel_size // 2
        y = np.arange(kernel_size) - kernel_size // 2
        xx, yy = np.meshgrid(x, y)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    else:
        # 3D Gaussian
        x = np.arange(kernel_size) - kernel_size // 2
        y = np.arange(kernel_size) - kernel_size // 2
        z = np.arange(kernel_size) - kernel_size // 2
        xx, yy, zz = np.meshgrid(x, y, z)
        kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))

    if normalize:
        kernel = kernel / kernel.sum()

    return conv_kernel(input_shape, kernel, stride, 'same', seed=seed)


@set_module_as('braintools.conn')
def gabor_kernel(
    input_shape: Tuple[int, int],
    frequency: float = 0.1,
    theta: float = 0,
    sigma: float = 1.0,
    phase: float = 0,
    kernel_size: Optional[int] = None,
    stride: int = 1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create connectivity using Gabor kernel.
    
    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer (height, width).
    frequency : float
        Frequency of sinusoidal component.
    theta : float
        Orientation angle in radians.
    sigma : float
        Standard deviation of Gaussian envelope.
    phase : float
        Phase offset.
    kernel_size : int, optional
        Size of kernel.
    stride : int
        Stride for convolution.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    """
    if kernel_size is None:
        kernel_size = int(4 * sigma) + 1

    # Create Gabor kernel
    x = np.arange(kernel_size) - kernel_size // 2
    y = np.arange(kernel_size) - kernel_size // 2
    xx, yy = np.meshgrid(x, y)

    # Rotate coordinates
    x_theta = xx * np.cos(theta) + yy * np.sin(theta)
    y_theta = -xx * np.sin(theta) + yy * np.cos(theta)

    # Gabor function
    gaussian = np.exp(-(x_theta ** 2 + y_theta ** 2) / (2 * sigma ** 2))
    sinusoid = np.cos(2 * np.pi * frequency * x_theta + phase)
    kernel = gaussian * sinusoid

    return conv_kernel(input_shape, kernel, stride, 'same', seed=seed)


@set_module_as('braintools.conn')
def dog_kernel(
    input_shape: Tuple[int, ...],
    sigma1: float = 1.0,
    sigma2: float = 2.0,
    kernel_size: Optional[int] = None,
    stride: int = 1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create connectivity using Difference of Gaussians (DoG) kernel.
    
    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer.
    sigma1 : float
        Standard deviation of first (center) Gaussian.
    sigma2 : float
        Standard deviation of second (surround) Gaussian.
    kernel_size : int, optional
        Size of kernel.
    stride : int
        Stride for convolution.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    """
    if kernel_size is None:
        kernel_size = int(4 * max(sigma1, sigma2)) + 1

    if len(input_shape) == 2:
        # 2D DoG
        x = np.arange(kernel_size) - kernel_size // 2
        y = np.arange(kernel_size) - kernel_size // 2
        xx, yy = np.meshgrid(x, y)
        r2 = xx ** 2 + yy ** 2

        g1 = np.exp(-r2 / (2 * sigma1 ** 2)) / (2 * np.pi * sigma1 ** 2)
        g2 = np.exp(-r2 / (2 * sigma2 ** 2)) / (2 * np.pi * sigma2 ** 2)
    else:
        # 3D DoG
        x = np.arange(kernel_size) - kernel_size // 2
        y = np.arange(kernel_size) - kernel_size // 2
        z = np.arange(kernel_size) - kernel_size // 2
        xx, yy, zz = np.meshgrid(x, y, z)
        r2 = xx ** 2 + yy ** 2 + zz ** 2

        g1 = np.exp(-r2 / (2 * sigma1 ** 2)) / ((2 * np.pi) ** (3 / 2) * sigma1 ** 3)
        g2 = np.exp(-r2 / (2 * sigma2 ** 2)) / ((2 * np.pi) ** (3 / 2) * sigma2 ** 3)

    kernel = g1 - g2

    return conv_kernel(input_shape, kernel, stride, 'same', seed=seed)


@set_module_as('braintools.conn')
def mexican_hat(
    input_shape: Tuple[int, ...],
    sigma: float = 1.0,
    kernel_size: Optional[int] = None,
    stride: int = 1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create connectivity using Mexican hat (Laplacian of Gaussian) kernel.
    
    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer.
    sigma : float
        Standard deviation of Gaussian.
    kernel_size : int, optional
        Size of kernel.
    stride : int
        Stride for convolution.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    """
    if kernel_size is None:
        kernel_size = int(6 * sigma) + 1

    if len(input_shape) == 2:
        # 2D Mexican hat
        x = np.arange(kernel_size) - kernel_size // 2
        y = np.arange(kernel_size) - kernel_size // 2
        xx, yy = np.meshgrid(x, y)
        r2 = xx ** 2 + yy ** 2

        kernel = (1 - r2 / (2 * sigma ** 2)) * np.exp(-r2 / (2 * sigma ** 2))
    else:
        # 3D Mexican hat
        x = np.arange(kernel_size) - kernel_size // 2
        y = np.arange(kernel_size) - kernel_size // 2
        z = np.arange(kernel_size) - kernel_size // 2
        xx, yy, zz = np.meshgrid(x, y, z)
        r2 = xx ** 2 + yy ** 2 + zz ** 2

        kernel = (1 - r2 / (2 * sigma ** 2)) * np.exp(-r2 / (2 * sigma ** 2))

    return conv_kernel(input_shape, kernel, stride, 'same', seed=seed)


@set_module_as('braintools.conn')
def sobel_kernel(
    input_shape: Tuple[int, int],
    direction: str = 'both',
    stride: int = 1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create connectivity using Sobel edge detection kernel.
    
    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer (height, width).
    direction : str
        Direction of edge detection ('horizontal', 'vertical', 'both').
    stride : int
        Stride for convolution.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    """
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]) / 8.0

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]]) / 8.0

    if direction == 'horizontal':
        kernel = sobel_x
    elif direction == 'vertical':
        kernel = sobel_y
    else:  # both
        kernel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    return conv_kernel(input_shape, kernel, stride, 'same', seed=seed)


@set_module_as('braintools.conn')
def laplacian_kernel(
    input_shape: Tuple[int, ...],
    kernel_type: str = '4-connected',
    stride: int = 1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create connectivity using Laplacian kernel.
    
    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer.
    kernel_type : str
        Type of Laplacian ('4-connected', '8-connected', 'gaussian').
    stride : int
        Stride for convolution.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    """
    if len(input_shape) == 2:
        if kernel_type == '4-connected':
            kernel = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]]) / 4.0
        elif kernel_type == '8-connected':
            kernel = np.array([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]]) / 8.0
        else:  # gaussian
            return mexican_hat(input_shape, sigma=1.0, stride=stride, seed=seed)
    else:
        # 3D Laplacian
        kernel = np.zeros((3, 3, 3))
        kernel[1, 1, 1] = 6
        kernel[0, 1, 1] = -1
        kernel[2, 1, 1] = -1
        kernel[1, 0, 1] = -1
        kernel[1, 2, 1] = -1
        kernel[1, 1, 0] = -1
        kernel[1, 1, 2] = -1
        kernel = kernel / 6.0

    return conv_kernel(input_shape, kernel, stride, 'same', seed=seed)


@set_module_as('braintools.conn')
def custom_kernel(
    input_shape: Tuple[int, ...],
    kernel_func: Callable,
    kernel_size: int,
    stride: int = 1,
    seed: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Create connectivity using custom kernel function.
    
    Parameters
    ----------
    input_shape : tuple of int
        Shape of input layer.
    kernel_func : callable
        Function that takes coordinates and returns kernel values.
    kernel_size : int
        Size of kernel.
    stride : int
        Stride for convolution.
    seed : int, optional
        Random seed.
    **kwargs
        Additional arguments passed to kernel_func.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    """
    if len(input_shape) == 2:
        x = np.arange(kernel_size) - kernel_size // 2
        y = np.arange(kernel_size) - kernel_size // 2
        xx, yy = np.meshgrid(x, y)
        kernel = kernel_func(xx, yy, **kwargs)
    else:
        x = np.arange(kernel_size) - kernel_size // 2
        y = np.arange(kernel_size) - kernel_size // 2
        z = np.arange(kernel_size) - kernel_size // 2
        xx, yy, zz = np.meshgrid(x, y, z)
        kernel = kernel_func(xx, yy, zz, **kwargs)

    return conv_kernel(input_shape, kernel, stride, 'same', seed=seed)
