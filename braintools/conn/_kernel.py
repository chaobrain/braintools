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

All kernels have a finite support (a cutoff distance or a kernel extent), so the
candidate pairs are found with a k-d tree and processed in chunks of at most
``MAX_CHUNK_PAIRS`` pairs: time and memory scale with the number of pairs inside
the support, never with ``N_pre * N_post``. The kernels are deterministic and
give the same connections, weights and edge order as the per-neuron loops of
earlier versions.
"""

from typing import Optional, Callable, Iterator, Tuple, Union

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike
from scipy.spatial import cKDTree

from braintools.init import param, Initializer, DistanceProfile, GaussianProfile
from ._base import PointConnectivity, ConnectionResult
from ._sampling import MAX_CHUNK_PAIRS, row_chunks

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


def _difference_of_gaussians(
    sigma_center,
    sigma_surround,
    amplitude_center,
    amplitude_surround,
) -> DistanceProfile:
    """Build the signed difference-of-Gaussians profile used by the DoG kernels.

    The profile is ``amplitude_center * G(sigma_center) - amplitude_surround *
    G(sigma_surround)``, composed from two :class:`braintools.init.GaussianProfile`
    instances. :class:`braintools.init.DoGProfile` is not used because it clips
    negative values to zero (it models a connection probability), while the
    kernel weights keep their inhibitory surround. For the same reason
    :class:`MexicanHat` (a DoG with a fixed ratio) does not use
    :class:`braintools.init.MexicanHatProfile`, which is a clipped Ricker wavelet.

    Parameters
    ----------
    sigma_center, sigma_surround : float
        Standard deviations as plain positive numbers in the distance unit.
    amplitude_center, amplitude_surround : float
        Peak amplitudes of the two Gaussians.

    Returns
    -------
    DistanceProfile
        A composed profile whose ``probability(distances)`` returns the signed
        DoG values.
    """
    return (amplitude_center * GaussianProfile(sigma_center)
            - amplitude_surround * GaussianProfile(sigma_surround))


def _same_population(pre_positions, post_positions) -> bool:
    """Whether pre and post refer to the same population.

    Autapses (``pre_idx == post_idx``) are only meaningful to drop when the two
    populations are identical. They are treated as the same population when the
    position arrays are the same object or have identical coordinates.
    """
    if pre_positions is post_positions:
        return True
    pre_val = u.get_mantissa(pre_positions)
    post_val = u.get_mantissa(post_positions)
    pre_val = np.asarray(pre_val)
    post_val = np.asarray(post_val)
    return pre_val.shape == post_val.shape and np.array_equal(pre_val, post_val)


def _xy_values(pre_positions, post_positions, dtype=None):
    """Return the first two position columns as plain numbers in one unit.

    Returns ``(pre_xy, post_xy, unit)``; ``post_positions`` is converted to the
    unit of ``pre_positions``. The native dtype is kept unless ``dtype`` is
    given, so the arithmetic matches the earlier per-neuron loops exactly.
    """
    pre_pos_val, pos_unit = u.split_mantissa_unit(pre_positions)
    post_pos_val = u.Quantity(post_positions).to(pos_unit).mantissa
    pre_xy = np.asarray(pre_pos_val, dtype=dtype)[:, :2]
    post_xy = np.asarray(post_pos_val, dtype=dtype)[:, :2]
    return pre_xy, post_xy, pos_unit


def _length_value(length, pos_unit):
    """Express a kernel length in the position unit (plain numbers pass through)."""
    if isinstance(length, u.Quantity):
        return u.Quantity(length).to(pos_unit).mantissa
    return length


def _candidate_pairs(
    query: np.ndarray,
    data: np.ndarray,
    radius: float,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield chunks of ``(query_index, data_index)`` pairs that may lie within ``radius``.

    The pairs are a superset of those whose Euclidean distance is at most
    ``radius``: the tree search uses a radius inflated by a few machine epsilons
    of the coordinate dtype (at least ``1e-9`` relative), so no pair is lost to
    round-off. Callers apply the exact test themselves.
    Chunks are consecutive ranges of query rows holding at most
    ``MAX_CHUNK_PAIRS`` candidate pairs (at least one row), and the pairs are
    sorted by query index, then data index, so concatenating the chunks gives
    a globally sorted list.

    Parameters
    ----------
    query, data : np.ndarray
        ``(n_query, d)`` and ``(n_data, d)`` coordinates as plain numbers.
    radius : float
        Search radius in the same unit. A non-finite radius makes every pair a
        candidate (chunked by rows); a negative or NaN radius yields nothing.

    Yields
    ------
    query_index, data_index : np.ndarray
        int64 arrays of equal length.
    """
    n_query, n_data = query.shape[0], data.shape[0]
    radius = float(radius)
    if n_query == 0 or n_data == 0 or not radius >= 0:
        return
    max_pairs = int(MAX_CHUNK_PAIRS)
    if not np.isfinite(radius):
        cols = np.arange(n_data, dtype=np.int64)
        for start, stop in row_chunks(n_query, n_data, max_pairs):
            rows: np.ndarray = np.arange(start, stop, dtype=np.int64)
            yield np.repeat(rows, n_data), np.tile(cols, stop - start)
        return
    dtype = np.result_type(query, data)
    slack = max(1e-9, 64 * float(np.finfo(dtype).eps)) if np.issubdtype(dtype, np.floating) else 1e-9
    search = radius * (1.0 + slack) + 1e-300
    data_tree = cKDTree(data)
    counts = np.asarray(data_tree.query_ball_point(query, search, return_length=True), dtype=np.int64)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    start = 0
    while start < n_query:
        stop = int(np.searchsorted(bounds, bounds[start] + max_pairs, side='right')) - 1
        stop = min(max(stop, start + 1), n_query)
        if bounds[stop] > bounds[start]:
            found = cKDTree(query[start:stop]).sparse_distance_matrix(
                data_tree, search, output_type='ndarray'
            )
            q_idx = found['i'].astype(np.int64)
            d_idx = found['j'].astype(np.int64)
            order = np.lexsort((d_idx, q_idx))
            yield q_idx[order] + start, d_idx[order]
        start = stop


def _within(query, data, radius, exclude_self):
    """Yield chunks ``(q, d, rel, dist)`` of pairs with ``dist <= radius``.

    ``rel = data[d] - query[q]`` and ``dist`` is its Euclidean norm, computed as
    ``sqrt(sum(rel ** 2))`` (bitwise equal to ``scipy.spatial.distance.cdist``).
    With ``exclude_self`` the pairs ``q == d`` are dropped.
    """
    for q, d in _candidate_pairs(query, data, radius):
        rel = data[d] - query[q]
        dist = np.sqrt(np.sum(rel ** 2, axis=1))
        keep = dist <= radius
        if exclude_self:
            keep &= q != d
        if keep.any():
            yield q[keep], d[keep], rel[keep], dist[keep]


def _first_rows(xy, n):
    """The first ``n`` rows of ``xy`` (the neurons indexed by ``range(n)``)."""
    n = int(n)
    if n > xy.shape[0]:
        raise ValueError(f'Got {xy.shape[0]} positions for {n} neurons.')
    return xy[:n]


def _nearest_index(grid, values):
    """Index of the nearest ``grid`` point for each value (first one on ties).

    Equivalent to ``np.argmin(np.abs(grid - v))`` per value, evaluated in
    blocks of at most ``MAX_CHUNK_PAIRS`` grid-value pairs.
    """
    out = np.empty(len(values), dtype=np.intp)
    step = max(1, int(MAX_CHUNK_PAIRS) // max(len(grid), 1))
    for start in range(0, len(values), step):
        block = values[start:start + step]
        out[start:start + step] = np.argmin(np.abs(grid[None, :] - block[:, None]), axis=1)
    return out


def _concat(parts, dtype=None):
    """Concatenate a list of arrays (an empty list gives an empty array)."""
    if not parts:
        return np.zeros(0, dtype=np.int64 if dtype is None else dtype)
    return np.concatenate(parts)


def _empty_result(pre_size, post_size, pre_positions, post_positions, metadata=None):
    kwargs = {} if metadata is None else {'metadata': metadata}
    return ConnectionResult(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64),
        pre_size=pre_size, post_size=post_size,
        pre_positions=pre_positions, post_positions=post_positions,
        model_type='point', **kwargs
    )


def _scaled_weights(conn, kernel_values, pre_size, post_size, pre_positions, post_positions):
    """Draw the base weights from ``conn.weight_init`` and multiply them by the kernel values."""
    n_connections = len(kernel_values)
    weights = param(
        conn.weight_init,
        n_connections,
        param_type='weight',
        pre_size=pre_size,
        post_size=post_size,
        pre_positions=pre_positions,
        post_positions=post_positions,
        rng=conn.rng
    )
    if weights is None:
        return kernel_values
    weights_vals, weights_unit = u.split_mantissa_unit(weights)
    if u.math.isscalar(weights_vals):
        weights_vals = np.full(n_connections, weights_vals)
    else:
        weights_vals = np.asarray(weights_vals)
    final_weights = weights_vals * kernel_values
    if weights_unit is not None:
        return u.maybe_decimal(final_weights * weights_unit)
    return final_weights


def _delays(conn, n_connections, pre_size, post_size, pre_positions, post_positions):
    return param(
        conn.delay_init,
        n_connections,
        param_type='delay',
        pre_size=pre_size,
        post_size=post_size,
        pre_positions=pre_positions,
        post_positions=post_positions,
        rng=conn.rng
    )


class Conv2dKernel(PointConnectivity):
    """Convolutional kernel connectivity for spatially arranged point neurons.

    Applies a 2D convolution kernel to neuron positions, creating connections
    where the kernel weight magnitude exceeds a threshold. This allows
    implementing receptive field structures in spiking neural networks.

    Only the first two position dimensions (x, y) are used; any extra columns
    (e.g. z) are ignored. ``kernel_size`` is the physical support of the kernel
    along BOTH axes: the kernel rows are mapped onto the y extent and the kernel
    columns onto the x extent, each bounded independently so non-square kernels
    cover the correct neighbourhood.

    Parameters
    ----------
    kernel : np.ndarray
        2D convolution kernel array.
    kernel_size : float or Quantity
        Physical size of the kernel support (along both axes) in position units.
    threshold : float
        Connection threshold - only kernel values whose magnitude exceeds this
        create connections. Thresholding on magnitude keeps essential negative
        coefficients (e.g. for edge-detection kernels). Default 0.0.
    allow_self_connections : bool
        Whether a neuron may connect to itself when pre and post are the same
        population and coincide with the kernel centre (default: False).
    weight : Initialization, optional
        Weight initialization (kernel values are multiplied by this).
    delay : Initialization, optional
        Delay initialization.

    Notes
    -----
    Candidate pairs are found with a k-d tree over the positions and handled
    in chunks of bounded size, so memory grows with the number of edges
    rather than with ``pre_size * post_size``. The edges, their order and the
    weights are the same as in version 0.6. Generation runs on the host with
    NumPy and cannot be traced by ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import Conv2dKernel
        >>> # Create 5x5 Gaussian-like kernel
        >>> kernel = np.array([
        ...     [0.04, 0.12, 0.18, 0.12, 0.04],
        ...     [0.12, 0.37, 0.56, 0.37, 0.12],
        ...     [0.18, 0.56, 1.00, 0.56, 0.18],
        ...     [0.12, 0.37, 0.56, 0.37, 0.12],
        ...     [0.04, 0.12, 0.18, 0.12, 0.04]
        ... ])
        >>> positions = np.random.RandomState(0).uniform(0, 1000, (500, 2)) * u.um
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
        >>> result.n_connections
        2230
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        kernel: np.ndarray,
        kernel_size: ArrayLike,
        threshold: float = 0.0,
        allow_self_connections: bool = False,
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
        self.allow_self_connections = allow_self_connections
        self.weight_init = weight
        self.delay_init = delay

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate convolutional kernel connections."""

        if pre_positions is None or post_positions is None:
            raise ValueError("Positions required for kernel connectivity")

        post_num = int(np.prod(post_size)) if isinstance(post_size, tuple) else post_size

        # Position values (only the first two dims are used); the post side is
        # limited to the first ``post_num`` rows, as the earlier loop over posts.
        pre_pos_val, post_pos_val, pos_unit = _xy_values(pre_positions, post_positions)
        post_pos_val = _first_rows(post_pos_val, post_num)
        kernel_size_val = _length_value(self.kernel_size, pos_unit)

        # Per-axis physical extents derived from the kernel shape, using the
        # pixel pitch implied by ``kernel_size`` (the larger axis spans
        # ``kernel_size``). Rows map to the y extent, columns to the x extent,
        # so non-square kernels cover the correct neighbourhood.
        kh, kw = self.kernel.shape
        pitch = kernel_size_val / max(kh, kw)
        extent_y = pitch * kh
        extent_x = pitch * kw
        kernel_y = np.linspace(-extent_y / 2, extent_y / 2, kh)
        kernel_x = np.linspace(-extent_x / 2, extent_x / 2, kw)

        same_population = (
            not self.allow_self_connections
            and _same_population(pre_positions, post_positions)
        )

        # Candidates inside the circle circumscribing the kernel box, grouped by
        # post neuron; the exact per-axis box test follows.
        radius = float(np.hypot(extent_x / 2, extent_y / 2))
        pre_parts, post_parts, value_parts = [], [], []
        for post_c, pre_c in _candidate_pairs(post_pos_val, pre_pos_val, radius):
            rel = pre_pos_val[pre_c] - post_pos_val[post_c]
            keep = (np.abs(rel[:, 0]) <= extent_x / 2) & (np.abs(rel[:, 1]) <= extent_y / 2)
            if same_population:
                keep &= pre_c != post_c
            if not keep.any():
                continue
            pre_c, post_c, rel = pre_c[keep], post_c[keep], rel[keep]
            # Nearest kernel cell (first one on ties, as ``np.argmin``).
            ki = _nearest_index(kernel_y, rel[:, 1])
            kj = _nearest_index(kernel_x, rel[:, 0])
            kernel_val = self.kernel[ki, kj]
            # Threshold on magnitude so signed kernels keep negative coeffs.
            keep = np.abs(kernel_val) > self.threshold
            pre_parts.append(pre_c[keep])
            post_parts.append(post_c[keep])
            value_parts.append(kernel_val[keep])

        pre_indices = _concat(pre_parts)
        if len(pre_indices) == 0:
            return _empty_result(pre_size, post_size, pre_positions, post_positions)

        post_indices = _concat(post_parts)
        kernel_weights = np.concatenate(value_parts)
        n_connections = len(pre_indices)

        weights = _scaled_weights(self, kernel_weights, pre_size, post_size, pre_positions, post_positions)
        delays = _delays(self, n_connections, pre_size, post_size, pre_positions, post_positions)

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

    Creates connections weighted by a Gaussian function of distance,
    useful for implementing smooth spatial receptive fields.

    Only the first two position dimensions (x, y) are used; any extra columns
    (e.g. z) are ignored.

    Parameters
    ----------
    sigma : float or Quantity
        Standard deviation of Gaussian.
    max_distance : float or Quantity, optional
        Maximum distance for connections (default: 3*sigma).
    normalize : bool
        Whether to normalize the Gaussian (default: True). Normalization uses
        the 2-D constant ``1 / (2 * pi * sigma**2)`` (positions are treated as
        2-D, see above).
    allow_self_connections : bool
        Whether a neuron may connect to itself when pre and post are the same
        population (default: False). A coincident pre/post would otherwise
        connect at the Gaussian peak.
    weight : Initialization, optional
        Weight initialization (Gaussian is multiplied by this).
    delay : Initialization, optional
        Delay initialization.

    Raises
    ------
    ValueError
        At generation time, if ``sigma`` is not positive.

    See Also
    --------
    braintools.init.GaussianProfile : The distance profile that computes the
        Gaussian values.

    Notes
    -----
    The unnormalized weight between neurons at distance ``d`` is
    ``exp(-d**2 / (2 * sigma**2))``, evaluated by
    :class:`braintools.init.GaussianProfile`. Pairs farther apart than
    ``max_distance`` and weights not above ``1e-6`` are dropped.

    Candidate pairs are found with a k-d tree over the positions and handled
    in chunks of bounded size, so memory grows with the number of edges
    rather than with ``pre_size * post_size``. The edges, their order and the
    weights are the same as in version 0.6. Generation runs on the host with
    NumPy and cannot be traced by ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import GaussianKernel
        >>> positions = np.random.RandomState(0).uniform(0, 1000, (500, 2)) * u.um
        >>> gauss = GaussianKernel(
        ...     sigma=50 * u.um,
        ...     max_distance=150 * u.um,
        ...     weight=2.0 * u.nS
        ... )
        >>> result = gauss(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
        >>> result.n_connections
        14104
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        sigma: ArrayLike,
        max_distance: Optional[ArrayLike] = None,
        normalize: bool = True,
        allow_self_connections: bool = False,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.max_distance = max_distance
        self.normalize = normalize
        self.allow_self_connections = allow_self_connections
        self.weight_init = weight
        self.delay_init = delay

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate Gaussian kernel connections."""

        if pre_positions is None or post_positions is None:
            raise ValueError("Positions required for Gaussian kernel connectivity")

        # Positions as float64 plain numbers (only the first two dims are used)
        pre_pos_val, post_pos_val, pos_unit = _xy_values(pre_positions, post_positions, dtype=float)

        # Sigma and max_distance in position units
        sigma_val = _length_value(self.sigma, pos_unit)
        if self.max_distance is not None:
            max_dist_val = _length_value(self.max_distance, pos_unit)
        else:
            max_dist_val = 3 * sigma_val

        # Gaussian weights from braintools.init.GaussianProfile (raises for sigma <= 0).
        profile = GaussianProfile(sigma_val)
        exclude_self = not self.allow_self_connections and _same_population(pre_positions, post_positions)

        pre_parts, post_parts, value_parts = [], [], []
        for pre_c, post_c, _, dist in _within(pre_pos_val, post_pos_val, max_dist_val, exclude_self):
            values = np.asarray(profile.probability(dist))
            if self.normalize:
                # 2-D normalization constant (positions are treated as 2-D).
                values = values / (2 * np.pi * sigma_val ** 2)
            # Prune cutoff: 1e-6
            keep = values > 1e-6
            pre_parts.append(pre_c[keep])
            post_parts.append(post_c[keep])
            value_parts.append(values[keep])

        pre_indices = _concat(pre_parts)
        if len(pre_indices) == 0:
            return _empty_result(pre_size, post_size, pre_positions, post_positions)

        post_indices = _concat(post_parts)
        gaussian_weights = np.concatenate(value_parts)
        n_connections = len(pre_indices)

        weights = _scaled_weights(self, gaussian_weights, pre_size, post_size, pre_positions, post_positions)
        delays = _delays(self, n_connections, pre_size, post_size, pre_positions, post_positions)

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

    Only the first two position dimensions (x, y) are used; any extra columns
    (e.g. z) are ignored.

    Parameters
    ----------
    sigma : float or Quantity
        Standard deviation of Gaussian envelope.
    frequency : float
        Frequency of the sinusoidal component, expressed in cycles per
        position-unit (i.e. ``1 / pos_unit``). This is NOT unit-converted: the
        rotated coordinates are taken in the position unit, so the wavelength
        changes if the position unit changes. Express ``frequency`` in the same
        length unit as the positions.
    theta : float
        Orientation angle in radians.
    phase : float
        Phase offset in radians (default: 0).
    max_distance : float or Quantity, optional
        Maximum distance for connections (default: 3*sigma).
    allow_self_connections : bool
        Whether a neuron may connect to itself when pre and post are the same
        population (default: False).
    weight : Initialization, optional
        Weight initialization (Gabor values are multiplied by this).
    delay : Initialization, optional
        Delay initialization.

    Raises
    ------
    ValueError
        At generation time, if ``sigma`` is not positive.

    Notes
    -----
    For a pre neuron at offset ``(dx, dy)`` from a post neuron, the weight is
    ``G(r) * cos(2 * pi * frequency * x_rot + phase)``, where ``r`` is the
    distance, ``x_rot = dx * cos(theta) + dy * sin(theta)`` and the Gaussian
    envelope ``G`` is evaluated by :class:`braintools.init.GaussianProfile`.
    Weights whose magnitude is not above ``1e-3`` are dropped.

    Candidate pairs are found with a k-d tree over the positions and handled
    in chunks of bounded size, so memory grows with the number of edges
    rather than with ``pre_size * post_size``. The edges, their order and the
    weights are the same as in version 0.6. Generation runs on the host with
    NumPy and cannot be traced by ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import GaborKernel
        >>> positions = np.random.RandomState(0).uniform(0, 1000, (500, 2)) * u.um
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
        >>> result.n_connections
        15086
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        sigma: ArrayLike,
        frequency: float,
        theta: float,
        phase: float = 0.0,
        max_distance: Optional[ArrayLike] = None,
        allow_self_connections: bool = False,
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
        self.allow_self_connections = allow_self_connections
        self.weight_init = weight
        self.delay_init = delay

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate Gabor kernel connections."""

        if pre_positions is None or post_positions is None:
            raise ValueError("Positions required for Gabor kernel connectivity")

        post_num = int(np.prod(post_size)) if isinstance(post_size, tuple) else post_size

        # Position values (only the first two dims are used)
        pre_pos_val, post_pos_val, pos_unit = _xy_values(pre_positions, post_positions)
        post_pos_val = _first_rows(post_pos_val, post_num)

        # Sigma and max_distance in position units
        sigma_val = _length_value(self.sigma, pos_unit)
        if self.max_distance is not None:
            max_dist_val = _length_value(self.max_distance, pos_unit)
        else:
            max_dist_val = 3 * sigma_val

        same_population = (
            not self.allow_self_connections
            and _same_population(pre_positions, post_positions)
        )

        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # The Gaussian envelope is radially symmetric, so it is evaluated on the
        # (rotation-invariant) distance through braintools.init.GaussianProfile.
        envelope_profile = GaussianProfile(sigma_val)

        # Pairs within ``max_distance``, grouped by post neuron (rel = pre - post).
        pre_parts, post_parts, value_parts = [], [], []
        for post_c, pre_c, rel, dist in _within(post_pos_val, pre_pos_val, max_dist_val, same_population):
            # Rotated coordinate along the carrier wave.
            x_rot = rel[:, 0] * cos_theta + rel[:, 1] * sin_theta

            # Gabor function: Gaussian envelope times sinusoidal carrier.
            envelope = np.asarray(envelope_profile.probability(dist))
            sinusoid = np.cos(2 * np.pi * self.frequency * x_rot + self.phase)
            gabor_vals = envelope * sinusoid

            # Prune cutoff: 1e-3 (magnitude).
            keep = np.abs(gabor_vals) > 1e-3
            pre_parts.append(pre_c[keep])
            post_parts.append(post_c[keep])
            value_parts.append(gabor_vals[keep])

        metadata = {
            'pattern': 'gabor_kernel',
            'sigma': self.sigma,
            'frequency': self.frequency,
            'theta': self.theta,
            'phase': self.phase
        }

        pre_indices = _concat(pre_parts)
        if len(pre_indices) == 0:
            return _empty_result(pre_size, post_size, pre_positions, post_positions, metadata)

        post_indices = _concat(post_parts)
        gabor_weights = np.concatenate(value_parts).astype(float)
        n_connections = len(pre_indices)

        weights = _scaled_weights(self, gabor_weights, pre_size, post_size, pre_positions, post_positions)
        delays = _delays(self, n_connections, pre_size, post_size, pre_positions, post_positions)

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

    Only the first two position dimensions (x, y) are used; any extra columns
    (e.g. z) are ignored.

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
    allow_self_connections : bool
        Whether a neuron may connect to itself when pre and post are the same
        population (default: False).
    weight : Initialization, optional
        Weight initialization (DoG values are multiplied by this).
    delay : Initialization, optional
        Delay initialization.

    Raises
    ------
    ValueError
        At generation time, if either sigma is not positive.

    Notes
    -----
    The weight at distance ``d`` is
    ``amplitude_center * G_c(d) - amplitude_surround * G_s(d)``, where ``G_c``
    and ``G_s`` are unit-peak :class:`braintools.init.GaussianProfile`
    instances. Unlike :class:`braintools.init.DoGProfile`, negative values are
    kept, so the surround is inhibitory. Weights whose magnitude is not above
    ``1e-4`` are dropped.

    Candidate pairs are found with a k-d tree over the positions and handled
    in chunks of bounded size, so memory grows with the number of edges
    rather than with ``pre_size * post_size``. The edges, their order and the
    weights are the same as in version 0.6. Generation runs on the host with
    NumPy and cannot be traced by ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import DoGKernel
        >>> positions = np.random.RandomState(0).uniform(0, 1000, (500, 2)) * u.um
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
        >>> result.n_connections
        21182
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        sigma_center: ArrayLike,
        sigma_surround: ArrayLike,
        amplitude_center: float = 1.0,
        amplitude_surround: float = 0.8,
        max_distance: Optional[ArrayLike] = None,
        allow_self_connections: bool = False,
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
        self.allow_self_connections = allow_self_connections
        self.weight_init = weight
        self.delay_init = delay

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate DoG kernel connections."""

        if pre_positions is None or post_positions is None:
            raise ValueError("Positions required for DoG kernel connectivity")

        # Positions as float64 plain numbers (only the first two dims are used)
        pre_pos_val, post_pos_val, pos_unit = _xy_values(pre_positions, post_positions, dtype=float)

        # Sigma values in position units
        sigma_c_val = u.Quantity(self.sigma_center).to(pos_unit).mantissa
        sigma_s_val = u.Quantity(self.sigma_surround).to(pos_unit).mantissa
        if self.max_distance is not None:
            max_dist_val = u.Quantity(self.max_distance).to(pos_unit).mantissa
        else:
            max_dist_val = 3 * sigma_s_val

        # Signed DoG weights, composed from two braintools.init.GaussianProfile.
        dog_profile = _difference_of_gaussians(
            sigma_c_val, sigma_s_val, self.amplitude_center, self.amplitude_surround
        )
        exclude_self = not self.allow_self_connections and _same_population(pre_positions, post_positions)

        pre_parts, post_parts, value_parts = [], [], []
        for pre_c, post_c, _, dist in _within(pre_pos_val, post_pos_val, max_dist_val, exclude_self):
            values = np.asarray(dog_profile.probability(dist))
            # Prune cutoff: 1e-4 on magnitude
            keep = np.abs(values) > 1e-4
            pre_parts.append(pre_c[keep])
            post_parts.append(post_c[keep])
            value_parts.append(values[keep])

        pre_indices = _concat(pre_parts)
        if len(pre_indices) == 0:
            return _empty_result(pre_size, post_size, pre_positions, post_positions)

        post_indices = _concat(post_parts)
        dog_weights = np.concatenate(value_parts)
        n_connections = len(pre_indices)

        weights = _scaled_weights(self, dog_weights, pre_size, post_size, pre_positions, post_positions)
        delays = _delays(self, n_connections, pre_size, post_size, pre_positions, post_positions)

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

    Notes
    -----
    The weight at distance ``d`` is ``2 * G(d; sigma) - G(d; sqrt(2) * sigma)``
    with unit-peak Gaussians from :class:`braintools.init.GaussianProfile`
    (see :class:`DoGKernel`). This is not
    :class:`braintools.init.MexicanHatProfile`, which is a Ricker wavelet
    clipped at zero.

    Candidate pairs are found with a k-d tree over the positions and handled
    in chunks of bounded size, so memory grows with the number of edges
    rather than with ``pre_size * post_size``. The edges, their order and the
    weights are the same as in version 0.6. Generation runs on the host with
    NumPy and cannot be traced by ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import MexicanHat
        >>> positions = np.random.RandomState(0).uniform(0, 1000, (500, 2)) * u.um
        >>> mexican = MexicanHat(
        ...     sigma=40 * u.um,
        ...     weight=1.0 * u.nS
        ... )
        >>> result = mexican(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
        >>> result.n_connections
        17110
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

    The Sobel kernel contains essential negative coefficients; connections are
    selected on coefficient magnitude so those negative weights are retained.

    Parameters
    ----------
    direction : str
        Direction of edge detection ('horizontal', 'vertical', 'both').
    kernel_size : float or Quantity
        Physical size of the 3x3 kernel in position units.
    allow_self_connections : bool
        Whether a neuron may connect to itself when pre and post are the same
        population (default: False).
    weight : Initialization, optional
        Weight initialization (Sobel values are multiplied by this).
    delay : Initialization, optional
        Delay initialization.

    Notes
    -----
    Candidate pairs are found with a k-d tree over the positions and handled
    in chunks of bounded size, so memory grows with the number of edges
    rather than with ``pre_size * post_size``. The edges, their order and the
    weights are the same as in version 0.6. Generation runs on the host with
    NumPy and cannot be traced by ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import SobelKernel
        >>> positions = np.random.RandomState(0).uniform(0, 1000, (500, 2)) * u.um
        >>> sobel = SobelKernel(
        ...     direction='horizontal',
        ...     kernel_size=60 * u.um,
        ...     weight=1.0 * u.nS
        ... )
        >>> result = sobel(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
        >>> result.n_connections
        418
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        direction: str = 'horizontal',
        kernel_size: ArrayLike = 1.0,
        allow_self_connections: bool = False,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.direction = direction
        self.kernel_size = kernel_size
        self.allow_self_connections = allow_self_connections
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

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate Sobel kernel connections."""
        # Delegate to Conv2dKernel. The Sobel kernel has essential negative
        # coefficients, so threshold=0 (magnitude) retains all non-zero weights.
        if self.kernel is not None:
            conv = Conv2dKernel(
                kernel=self.kernel,
                kernel_size=self.kernel_size,
                threshold=0.0,
                allow_self_connections=self.allow_self_connections,
                weight=self.weight_init,
                delay=self.delay_init,
                seed=self.seed
            )
            result = conv.generate(
                pre_size=pre_size, post_size=post_size,
                pre_positions=pre_positions, post_positions=post_positions, **kwargs
            )
            result.metadata['pattern'] = 'sobel_kernel'
            result.metadata['direction'] = self.direction
            return result
        else:
            # Both directions - compute magnitude
            conv_h = Conv2dKernel(
                kernel=self.kernel_h,
                kernel_size=self.kernel_size,
                threshold=0.0,
                allow_self_connections=self.allow_self_connections,
                weight=self.weight_init,
                delay=self.delay_init,
                seed=self.seed
            )
            conv_v = Conv2dKernel(
                kernel=self.kernel_v,
                kernel_size=self.kernel_size,
                threshold=0.0,
                allow_self_connections=self.allow_self_connections,
                weight=self.weight_init,
                delay=self.delay_init,
                seed=self.seed + 1 if self.seed is not None else None
            )
            # Union of both
            from ._base import CompositeConnectivity
            composite = CompositeConnectivity(conv_h, conv_v, 'union')
            result = composite.generate(
                pre_size=pre_size, post_size=post_size,
                pre_positions=pre_positions, post_positions=post_positions, **kwargs
            )
            result.metadata['pattern'] = 'sobel_kernel'
            result.metadata['direction'] = 'both'
            return result


class LaplacianKernel(PointConnectivity):
    """Laplacian kernel for edge detection connectivity.

    Implements Laplacian operators for detecting discontinuities and edges,
    useful for lateral inhibition and edge enhancement.

    The Laplacian kernel has an essential negative centre coefficient (-4 or
    -8); connections are selected on coefficient magnitude so that centre
    weight is retained.

    Parameters
    ----------
    kernel_type : str
        Type of Laplacian ('4-connected', '8-connected').
    kernel_size : float or Quantity
        Physical size of the kernel in position units.
    allow_self_connections : bool
        Whether a neuron may connect to itself when pre and post are the same
        population (default: False). The Laplacian centre coefficient maps onto
        the self-connection, so enabling this retains the centre weight on the
        diagonal.
    weight : Initialization, optional
        Weight initialization (Laplacian values are multiplied by this).
    delay : Initialization, optional
        Delay initialization.

    Notes
    -----
    Candidate pairs are found with a k-d tree over the positions and handled
    in chunks of bounded size, so memory grows with the number of edges
    rather than with ``pre_size * post_size``. The edges, their order and the
    weights are the same as in version 0.6. Generation runs on the host with
    NumPy and cannot be traced by ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import LaplacianKernel
        >>> positions = np.random.RandomState(0).uniform(0, 1000, (500, 2)) * u.um
        >>> laplacian = LaplacianKernel(
        ...     kernel_type='4-connected',
        ...     kernel_size=60 * u.um,
        ...     weight=1.0 * u.nS
        ... )
        >>> result = laplacian(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
        >>> result.n_connections
        634
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        kernel_type: str = '4-connected',
        kernel_size: ArrayLike = 1.0,
        allow_self_connections: bool = False,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        self.allow_self_connections = allow_self_connections
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

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate Laplacian kernel connections."""
        # Delegate to Conv2dKernel. The Laplacian has an essential negative
        # centre coefficient, so threshold=0 (magnitude) retains all weights.
        conv = Conv2dKernel(
            kernel=self.kernel,
            kernel_size=self.kernel_size,
            threshold=0.0,
            allow_self_connections=self.allow_self_connections,
            weight=self.weight_init,
            delay=self.delay_init,
            seed=self.seed
        )
        result = conv.generate(
            pre_size=pre_size, post_size=post_size,
            pre_positions=pre_positions, post_positions=post_positions, **kwargs
        )
        result.metadata['pattern'] = 'laplacian_kernel'
        result.metadata['kernel_type'] = self.kernel_type
        return result


class CustomKernel(PointConnectivity):
    """Custom kernel connectivity using user-defined kernel function.

    Allows implementing arbitrary spatial kernel functions for connectivity.

    Only the first two position dimensions (x, y) are used; any extra columns
    (e.g. z) are ignored.

    Parameters
    ----------
    kernel_func : callable
        Function that takes (x, y) coordinates and returns kernel value.
        Should accept arrays and return array of same shape.
    kernel_size : float or Quantity
        Physical size of the kernel support in position units.
    threshold : float
        Connection threshold on kernel-value magnitude (default: 0.0).
    allow_self_connections : bool
        Whether a neuron may connect to itself when pre and post are the same
        population (default: False).
    weight : Initialization, optional
        Weight initialization (kernel values are multiplied by this).
    delay : Initialization, optional
        Delay initialization.

    Notes
    -----
    Candidate pairs are found with a k-d tree over the positions and handled
    in chunks of bounded size, so memory grows with the number of edges
    rather than with ``pre_size * post_size``. The edges, their order and the
    weights are the same as in version 0.6. Generation runs on the host with
    NumPy and cannot be traced by ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import CustomKernel
        >>> def my_kernel(x, y):
        ...     # Custom kernel function
        ...     r = np.sqrt(x**2 + y**2)
        ...     return np.exp(-r/50) * np.cos(r/10)
        >>>
        >>> positions = np.random.RandomState(0).uniform(0, 1000, (500, 2)) * u.um
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
        >>> result.n_connections
        5648
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        kernel_func: Callable,
        kernel_size: ArrayLike,
        threshold: float = 0.0,
        allow_self_connections: bool = False,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_func = kernel_func
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.allow_self_connections = allow_self_connections
        self.weight_init = weight
        self.delay_init = delay

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate custom kernel connections."""

        if pre_positions is None or post_positions is None:
            raise ValueError("Positions required for custom kernel connectivity")

        post_num = int(np.prod(post_size)) if isinstance(post_size, tuple) else post_size

        # Position values (only the first two dims are used)
        pre_pos_val, post_pos_val, pos_unit = _xy_values(pre_positions, post_positions)
        post_pos_val = _first_rows(post_pos_val, post_num)
        kernel_size_val = _length_value(self.kernel_size, pos_unit)

        same_population = (
            not self.allow_self_connections
            and _same_population(pre_positions, post_positions)
        )

        # Pairs within the kernel support, grouped by post neuron (rel = pre - post).
        # ``kernel_func`` is called once per chunk on all its offsets.
        pre_parts, post_parts, value_parts = [], [], []
        for post_c, pre_c, rel, _ in _within(post_pos_val, pre_pos_val, kernel_size_val / 2, same_population):
            rel_x = rel[:, 0]
            rel_y = rel[:, 1]
            kernel_vals = np.broadcast_to(np.asarray(self.kernel_func(rel_x, rel_y)), rel_x.shape)

            # Apply threshold on magnitude
            above_threshold = np.abs(kernel_vals) > self.threshold
            pre_parts.append(pre_c[above_threshold])
            post_parts.append(post_c[above_threshold])
            value_parts.append(kernel_vals[above_threshold])

        pre_indices = _concat(pre_parts)
        if len(pre_indices) == 0:
            return _empty_result(pre_size, post_size, pre_positions, post_positions)

        post_indices = _concat(post_parts)
        kernel_weights = np.concatenate(value_parts)
        n_connections = len(pre_indices)

        weights = _scaled_weights(self, kernel_weights, pre_size, post_size, pre_positions, post_positions)
        delays = _delays(self, n_connections, pre_size, post_size, pre_positions, post_positions)

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
