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

from typing import Union

import brainstate
import jax
import numpy as onp
from jax import vmap, lax, numpy as jnp

from braintools._misc import set_module_as

__all__ = [
    'cross_correlation',
    'voltage_fluctuation',
    'matrix_correlation',
    'weighted_correlation',
    'functional_connectivity',
    'functional_connectivity_dynamics',
]


@set_module_as('braintools.metric')
def cross_correlation(
    spikes: brainstate.typing.ArrayLike,
    bin: Union[int, float],
    dt: Union[int, float] = None,
    method: str = 'loop'
):
    r"""Calculate cross-correlation index between neurons.

    The coherence between two neurons i and j is measured by their
    cross-correlation of spike trains at zero time lag within a time bin.
    This function computes the population synchronization index based on
    pairwise cross-correlations.

    The coherence measure for a pair is defined as:

    .. math::

        \kappa_{ij}(\tau) = \frac{\sum_{l=1}^{K} X(l) Y(l)}
        {\sqrt{\sum_{l=1}^{K} X(l) \sum_{l=1}^{K} Y(l)}}

    where the time interval is divided into K bins of size :math:`\Delta t = \tau`,
    and :math:`X(l)`, :math:`Y(l)` are binary spike indicators (0 or 1) for each bin.

    The population coherence measure :math:`\kappa(\tau)` is the average of
    :math:`\kappa_{ij}(\tau)` over all pairs of neurons.

    Parameters
    ----------
    spikes : brainstate.typing.ArrayLike
        Spike history matrix with shape ``(num_time, num_neurons)``.
        Binary values indicating spike occurrences.
    bin : Union[int, float]
        Time bin size for binning spike trains.
    dt : Union[int, float], optional
        Time precision. If None, uses ``brainstate.environ.get_dt()``.
    method : str, default='loop'
        Method for computing cross-correlations:
        
        - ``'loop'``: Memory-efficient iterative approach
        - ``'vmap'``: Vectorized approach (uses more memory)

    Returns
    -------
    float
        Cross-correlation index representing the population synchronization level.
        Values closer to 1 indicate higher synchronization.

    Notes
    -----
    To JIT compile this function, make ``bin``, ``dt``, and ``method`` static.
    For example: ``partial(cross_correlation, bin=10, method='loop')``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> # Generate random spike data
    >>> spikes = jnp.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    >>> sync_index = braintools.metric.cross_correlation(spikes, bin=1.0)
    >>> print(f"Synchronization index: {sync_index:.3f}")
    >>> 
    >>> # For larger datasets, use vectorized method
    >>> large_spikes = jnp.random.binomial(1, 0.1, (1000, 50))
    >>> sync_fast = braintools.metric.cross_correlation(large_spikes, bin=10.0, method='vmap')
    >>> print(f"Population synchronization: {sync_fast:.3f}")

    References
    ----------
    .. [1] Wang, Xiao-Jing, and György Buzsáki. "Gamma oscillation by synaptic
           inhibition in a hippocampal interneuronal network model." Journal of
           Neuroscience 16.20 (1996): 6402-6413.
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    bin_size = int(bin / dt)
    num_hist, num_neu = spikes.shape
    num_bin = int(onp.ceil(num_hist / bin_size))
    if num_bin * bin_size != num_hist:
        spikes = jnp.append(spikes, jnp.zeros((num_bin * bin_size - num_hist, num_neu)), axis=0)
    states = spikes.T.reshape((num_neu, num_bin, bin_size))
    states = jnp.asarray(jnp.sum(states, axis=2) > 0., dtype=jnp.float_)
    indices = jnp.tril_indices(num_neu, k=-1)

    if method == 'loop':
        def _f(i, j):
            sqrt_ij = jnp.sqrt(jnp.sum(states[i]) * jnp.sum(states[j]))
            return lax.cond(sqrt_ij == 0.,
                            lambda _: 0.,
                            lambda _: jnp.sum(states[i] * states[j]) / sqrt_ij,
                            None)

        res = brainstate.compile.for_loop(_f, *indices)

    elif method == 'vmap':
        @vmap
        def _cc(i, j):
            sqrt_ij = jnp.sqrt(jnp.sum(states[i]) * jnp.sum(states[j]))
            return lax.cond(sqrt_ij == 0.,
                            lambda _: 0.,
                            lambda _: jnp.sum(states[i] * states[j]) / sqrt_ij,
                            None)

        res = _cc(*indices)
    else:
        raise ValueError(f'Do not support {method}. We only support "loop" or "vmap".')

    return jnp.mean(jnp.asarray(res))


def _f_signal(signal):
    return jnp.mean(signal * signal) - jnp.mean(signal) ** 2


@set_module_as('braintools.metric')
def voltage_fluctuation(
    potentials,
    method='loop'
):
    r"""Calculate neuronal synchronization via voltage variance analysis.

    This method quantifies synchronization by comparing the variance of the
    population-averaged membrane potential to the average variance of individual
    neurons' membrane potentials.

    The synchronization measure is computed as:

    .. math::

        \chi^2(N) = \frac{\sigma_V^2}{\frac{1}{N} \sum_{i=1}^N \sigma_{V_i}^2}

    where:
    
    - :math:`\sigma_V^2` is the variance of the population average potential
    - :math:`\sigma_{V_i}^2` is the variance of individual neuron potentials
    - :math:`N` is the number of neurons

    The population average potential is:

    .. math::

        V(t) = \frac{1}{N} \sum_{i=1}^{N} V_i(t)

    And its variance is:

    .. math::

        \sigma_V^2 = \left\langle V(t)^2 \right\rangle_t - \left\langle V(t) \right\rangle_t^2

    Parameters
    ----------
    potentials : brainstate.typing.ArrayLike
        Membrane potential matrix with shape ``(num_time, num_neurons)``.
        Contains the voltage traces for each neuron over time.
    method : str, default='loop'
        Computational method:
        
        - ``'loop'``: Memory-efficient iterative computation
        - ``'vmap'``: Vectorized computation (higher memory usage)

    Returns
    -------
    float
        Synchronization index. Values > 1 indicate synchronized activity,
        values ≈ 1 indicate asynchronous activity.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> # Generate correlated voltage traces
    >>> t = jnp.linspace(0, 10, 1000)
    >>> # Synchronous case: common oscillation + noise
    >>> common_signal = jnp.sin(2 * jnp.pi * t)
    >>> potentials_sync = common_signal[:, None] + 0.1 * jnp.random.normal((1000, 10))
    >>> sync_idx = braintools.metric.voltage_fluctuation(potentials_sync)
    >>> print(f"Synchronized case: {sync_idx:.3f}")
    >>> 
    >>> # Asynchronous case: independent noise
    >>> potentials_async = jnp.random.normal((1000, 10))
    >>> async_idx = braintools.metric.voltage_fluctuation(potentials_async)
    >>> print(f"Asynchronous case: {async_idx:.3f}")

    References
    ----------
    .. [1] Golomb, D. and Rinzel, J. (1993). "Dynamics of globally coupled
           inhibitory neurons with heterogeneity." Physical Review E
           48(6): 4810-4814.
    .. [2] Golomb, D. and Rinzel, J. (1994). "Clustering in globally coupled
           inhibitory neurons." Physica D 72(1-2): 259-282.
    .. [3] Golomb, David (2007). "Neuronal synchrony measures."
           Scholarpedia 2(1): 1347.
    """

    avg = jnp.mean(potentials, axis=1)
    avg_var = jnp.mean(avg * avg) - jnp.mean(avg) ** 2

    if method == 'loop':
        _var = brainstate.compile.for_loop(_f_signal, jnp.moveaxis(potentials, 0, 1))
    elif method == 'vmap':
        _var = vmap(_f_signal, in_axes=1)(potentials)
    else:
        raise ValueError(f'Do not support {method}. We only support "loop" or "vmap".')

    var_mean = jnp.mean(_var)
    r = jnp.where(var_mean == 0., 1., avg_var / var_mean)
    return r


@set_module_as('braintools.metric')
def matrix_correlation(x, y):
    r"""Compute Pearson correlation of upper triangular elements of two matrices.

    This function calculates the correlation coefficient between corresponding
    upper triangular elements of two matrices, excluding the diagonal.
    This is useful for comparing connectivity matrices or similarity matrices.

    Parameters
    ----------
    x : brainstate.typing.ArrayLike
        First matrix. Must be 2-dimensional.
    y : brainstate.typing.ArrayLike
        Second matrix. Must have the same shape as `x`.

    Returns
    -------
    float
        Pearson correlation coefficient between the upper triangular elements
        of the two matrices (excluding diagonal).

    Raises
    ------
    ValueError
        If input arrays are not 2-dimensional.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> # Create two correlation matrices with similar structure
    >>> x = jnp.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]])
    >>> y = jnp.array([[1.0, 0.7, 0.4], [0.7, 1.0, 0.6], [0.4, 0.6, 1.0]])
    >>> corr = braintools.metric.matrix_correlation(x, y)
    >>> print(f"Matrix correlation: {corr:.3f}")
    >>> 
    >>> # Compare connectivity matrices from different conditions
    >>> baseline_fc = jnp.random.rand(5, 5)
    >>> baseline_fc = (baseline_fc + baseline_fc.T) / 2  # Make symmetric
    >>> jnp.fill_diagonal(baseline_fc, 1.0)  # Set diagonal to 1
    >>> 
    >>> treatment_fc = baseline_fc + 0.1 * jnp.random.rand(5, 5)
    >>> similarity = braintools.metric.matrix_correlation(baseline_fc, treatment_fc)
    >>> print(f"Condition similarity: {similarity:.3f}")

    Notes
    -----
    The function uses ``jnp.triu_indices_from(x, k=1)`` to extract upper
    triangular elements, where ``k=1`` excludes the diagonal.
    
    This measure is particularly useful for:
    
    - Comparing functional connectivity matrices across conditions
    - Assessing similarity of network structures
    - Validating model predictions against empirical connectivity
    
    For matrices that are not symmetric, only the upper triangle is used,
    which may not capture the full relationship structure.
    
    See Also
    --------
    functional_connectivity : Compute connectivity matrix from time series
    weighted_correlation : Weighted correlation for individual vectors
    """
    if x.ndim != 2:
        raise ValueError(f'Only support 2d array, but we got a array '
                         f'with the shape of {x.shape}')
    if y.ndim != 2:
        raise ValueError(f'Only support 2d array, but we got a array '
                         f'with the shape of {y.shape}')
    x = x[jnp.triu_indices_from(x, k=1)]
    y = y[jnp.triu_indices_from(y, k=1)]
    cc = jnp.corrcoef(x, y)[0, 1]
    return cc


@set_module_as('braintools.metric')
def functional_connectivity(activities):
    r"""Compute functional connectivity matrix from time series data.

    Calculates the pairwise Pearson correlation coefficients between all
    pairs of signals to create a functional connectivity matrix. This is
    commonly used in neuroscience to assess statistical dependencies
    between different brain regions or neurons.

    Parameters
    ----------
    activities : brainstate.typing.ArrayLike
        Time series data with shape ``(num_time, num_signals)`` where
        each column represents a different signal/neuron/region.

    Returns
    -------
    brainstate.typing.ArrayLike
        Functional connectivity matrix with shape ``(num_signals, num_signals)``.
        Element (i,j) represents the correlation between signals i and j.
        Diagonal elements are 1.0. NaN values are replaced with 0.0.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> # Generate correlated time series
    >>> t = jnp.linspace(0, 10, 100)
    >>> sig1 = jnp.sin(t) + 0.1 * jnp.random.normal(size=100)
    >>> sig2 = jnp.sin(t + 0.2) + 0.1 * jnp.random.normal(size=100)
    >>> activities = jnp.column_stack([sig1, sig2])
    >>> fc_matrix = braintools.metric.functional_connectivity(activities)
    >>> print(f"Connectivity shape: {fc_matrix.shape}")
    >>> print(f"Correlation: {fc_matrix[0, 1]:.3f}")

    Notes
    -----
    The function uses ``jnp.corrcoef`` internally and handles NaN values
    by replacing them with 0.0 using ``jnp.nan_to_num``.
    
    For very short time series, correlations may be unreliable due to
    insufficient data points. Consider using longer recordings or smoothing
    techniques for more stable estimates.
    
    See Also
    --------
    functional_connectivity_dynamics : Time-varying connectivity analysis
    matrix_correlation : Correlation between connectivity matrices
    """
    if activities.ndim != 2:
        raise ValueError('Only support 2d array with shape of "(num_time, num_sample)". '
                         f'But we got a array with the shape of {activities.shape}')
    fc = jnp.corrcoef(activities.T)
    return jnp.nan_to_num(fc)


@set_module_as('braintools.metric')
def functional_connectivity_dynamics(
    activities,
    window_size=30,
    step_size=5
):
    r"""Compute functional connectivity dynamics (FCD) matrix.

    Functional Connectivity Dynamics (FCD) captures the temporal evolution
    of functional connectivity by computing connectivity matrices over
    sliding windows and then measuring correlations between these matrices.
    This provides insights into how network connectivity patterns change over time.

    Parameters
    ----------
    activities : brainstate.typing.ArrayLike
        Time series data with shape ``(num_time, num_signals)``.
    window_size : int, default=30
        Size of each sliding window in time steps. Larger windows provide
        more stable connectivity estimates but lower temporal resolution.
    step_size : int, default=5
        Step size between consecutive windows. Smaller steps provide higher
        temporal resolution but more computational cost.

    Returns
    -------
    brainstate.typing.ArrayLike
        FCD matrix of shape ``(num_windows, num_windows)`` measuring correlations
        between connectivity patterns across different time windows.

    Notes
    -----
    FCD computation steps:
    
    1. Compute FC matrices for sliding windows (Pearson correlations)
    2. Vectorize upper triangular elements of each FC matrix (exclude diagonal)
    3. Compute Pearson correlations between these vectors across windows

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> import brainstate as brainstate
    >>> activities = brainstate.random.rand(200, 10)
    >>> fcd = braintools.metric.functional_connectivity_dynamics(activities)
    """

    if activities.ndim != 2:
        raise ValueError('Only support 2d array with shape of "(num_time, num_sample)". '
                         f'But we got a array with the shape of {activities.shape}')

    t_len, n_sig = activities.shape
    if window_size <= 1:
        raise ValueError('window_size must be > 1.')
    if step_size <= 0:
        raise ValueError('step_size must be > 0.')

    # Determine window start indices
    if t_len < window_size:
        return jnp.zeros((0, 0), dtype=activities.dtype)
    starts = jnp.arange(0, t_len - window_size + 1, step_size)
    n_windows = starts.shape[0]

    # Indices for vectorizing FC (upper triangle, excluding diagonal)
    iu = jnp.triu_indices(n_sig, k=1)
    vec_len = iu[0].shape[0]

    def _slice_fc_vec(start):
        seg = lax.dynamic_slice(activities, (start, 0), (window_size, n_sig))
        fc = functional_connectivity(seg)
        return fc[iu]

    # Compute FC vectors for all windows
    fc_vectors = jax.vmap(_slice_fc_vec)(starts)  # shape: (n_windows, vec_len)

    # Center each vector (remove mean across edges)
    centered = fc_vectors - jnp.mean(fc_vectors, axis=1, keepdims=True)
    # Normalize to unit norm to get Pearson correlation via cosine similarity
    norms = jnp.linalg.norm(centered, axis=1)
    norms = jnp.where(norms > 0, norms, 1.0)
    normalized = centered / norms[:, None]

    # Correlation matrix between windows
    fcd = normalized @ normalized.T
    # Ensure exact ones on diagonal
    fcd = fcd - jnp.diag(jnp.diag(fcd)) + jnp.eye(n_windows, dtype=fcd.dtype)
    return fcd


@set_module_as('braintools.metric')
def weighted_correlation(
    x,
    y,
    w,
):
    r"""Compute weighted Pearson correlation between two data series.

    Calculates the Pearson correlation coefficient between two variables
    with weighted observations. This is useful when some data points
    should contribute more to the correlation calculation than others.

    The weighted correlation is computed as:

    .. math::

        r_w = \frac{\mathrm{Cov}_w(X,Y)}{\sqrt{\mathrm{Var}_w(X) \cdot \mathrm{Var}_w(Y)}}

    where :math:`\mathrm{Cov}_w` is the weighted covariance.

    Parameters
    ----------
    x : brainstate.typing.ArrayLike
        First data series. Must be 1-dimensional.
    y : brainstate.typing.ArrayLike
        Second data series. Must be 1-dimensional and same length as `x`.
    w : brainstate.typing.ArrayLike
        Weight vector. Must be 1-dimensional and same length as `x` and `y`.
        Higher weights give more importance to corresponding data points.

    Returns
    -------
    float
        Weighted Pearson correlation coefficient between -1 and 1.

    Raises
    ------
    ValueError
        If any input array is not 1-dimensional or if arrays have different lengths.

    Notes
    -----
    The weighted correlation reduces to the standard Pearson correlation when
    all weights are equal. Weights should be non-negative; zero weights
    effectively exclude those data points from the calculation.

    For numerical stability, avoid using weights with very large differences
    in magnitude, as this can lead to precision issues.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> # Perfect linear relationship
    >>> x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> # Weight middle points more heavily
    >>> w = jnp.array([1.0, 1.0, 2.0, 2.0, 1.0])
    >>> corr = braintools.metric.weighted_correlation(x, y, w)
    >>> print(f"Weighted correlation: {corr:.3f}")
    >>>
    >>> # Compare with unweighted correlation
    >>> unweighted = jnp.corrcoef(x, y)[0, 1]
    >>> print(f"Unweighted correlation: {unweighted:.3f}")
    >>>
    >>> # Example with reliability weights (higher for more reliable measurements)
    >>> reliability = jnp.array([0.5, 0.8, 0.9, 0.7, 0.6])
    >>> corr_reliable = braintools.metric.weighted_correlation(x, y, reliability)
    >>> print(f"Reliability-weighted: {corr_reliable:.3f}")
    """

    def _weighted_mean(x, w):
        """Weighted Mean"""
        return jnp.sum(x * w) / jnp.sum(w)

    def _weighted_cov(x, y, w):
        """Weighted Covariance"""
        return jnp.sum(w * (x - _weighted_mean(x, w)) * (y - _weighted_mean(y, w))) / jnp.sum(w)

    if x.ndim != 1:
        raise ValueError(f'Only support 1d array, but we got a array '
                         f'with the shape of {x.shape}')
    if y.ndim != 1:
        raise ValueError(f'Only support 1d array, but we got a array '
                         f'with the shape of {y.shape}')
    if w.ndim != 1:
        raise ValueError(f'Only support 1d array, but we got a array '
                         f'with the shape of {w.shape}')
    return _weighted_cov(x, y, w) / jnp.sqrt(_weighted_cov(x, x, w) * _weighted_cov(y, y, w))
