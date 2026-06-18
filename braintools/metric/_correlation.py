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
import brainunit as u
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
        {\sqrt{\left(\sum_{l=1}^{K} X(l)\right) \left(\sum_{l=1}^{K} Y(l)\right)}}

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

    This is the coincidence-based coherence of Wang & Buzsáki (1996), not a
    Pearson correlation coefficient.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> spikes = jnp.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        >>> sync_index = braintools.metric.cross_correlation(spikes, bin=1.0, dt=1.0)
        >>> float(sync_index) >= 0.0
        True

        >>> # For larger datasets, the vectorized method is faster.
        >>> import brainstate
        >>> large_spikes = (brainstate.random.rand(1000, 50) < 0.1).astype(float)
        >>> sync_fast = braintools.metric.cross_correlation(
        ...     large_spikes, bin=10.0, dt=1.0, method='vmap')

    References
    ----------
    .. [1] Wang, Xiao-Jing, and György Buzsáki. "Gamma oscillation by synaptic
           inhibition in a hippocampal interneuronal network model." Journal of
           Neuroscience 16.20 (1996): 6402-6413.
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    bin_size = int(round(bin / dt))
    if bin_size < 1:
        raise ValueError(f'`bin` ({bin}) must be at least as large as `dt` ({dt}); '
                         f'got a bin width of {bin_size} samples.')
    # Strip units so ``Quantity`` spike matrices behave like the other metrics.
    spikes = jnp.asarray(u.get_magnitude(spikes))
    num_hist, num_neu = spikes.shape
    if num_neu < 2:
        # Coincidence is only defined for pairs; a single neuron has no pairs.
        return jnp.asarray(0.0)
    num_bin = int(onp.ceil(num_hist / bin_size))
    if num_bin * bin_size != num_hist:
        spikes = jnp.append(spikes, jnp.zeros((num_bin * bin_size - num_hist, num_neu)), axis=0)
    states = spikes.T.reshape((num_neu, num_bin, bin_size))
    states = jnp.asarray(jnp.sum(states, axis=2) > 0., dtype=float)
    indices = jnp.tril_indices(num_neu, k=-1)

    if method == 'loop':
        def _f(i, j):
            sqrt_ij = jnp.sqrt(jnp.sum(states[i]) * jnp.sum(states[j]))
            return lax.cond(sqrt_ij == 0.,
                            lambda _: jnp.zeros((), dtype=sqrt_ij.dtype),
                            lambda _: jnp.sum(states[i] * states[j]) / sqrt_ij,
                            None)

        res = brainstate.transform.for_loop(_f, *indices)

    elif method == 'vmap':
        @vmap
        def _cc(i, j):
            sqrt_ij = jnp.sqrt(jnp.sum(states[i]) * jnp.sum(states[j]))
            return lax.cond(sqrt_ij == 0.,
                            lambda _: jnp.zeros((), dtype=sqrt_ij.dtype),
                            lambda _: jnp.sum(states[i] * states[j]) / sqrt_ij,
                            None)

        res = _cc(*indices)
    else:
        raise ValueError(f'Do not support {method}. We only support "loop" or "vmap".')

    return jnp.mean(jnp.asarray(res))


def _f_signal(signal):
    # ``jnp.var`` (mean-subtracted) is numerically stable; the algebraically
    # equivalent ``mean(s**2) - mean(s)**2`` suffers float32 cancellation and can
    # return a small *negative* value for constant signals, breaking the
    # zero-variance guard in ``voltage_fluctuation``.
    return jnp.var(signal)


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
    jax.Array
        Scalar (0-d array) synchronization index, bounded in approximately
        ``[1/N, 1]``. Values near ``1`` indicate strong synchrony; values near
        ``1/N`` indicate asynchronous activity. (The ratio of the population-mean
        variance to the mean single-neuron variance cannot exceed 1.) By
        convention a constant (zero-variance) population returns ``1.0``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> t = jnp.linspace(0, 10, 1000)
        >>> # Synchronous case: shared oscillation + small independent noise.
        >>> common = jnp.sin(2 * jnp.pi * t)[:, None]
        >>> sync = common + 0.1 * brainstate.random.normal(size=(1000, 10))
        >>> async_ = brainstate.random.normal(size=(1000, 10))
        >>> bool(braintools.metric.voltage_fluctuation(sync)
        ...      > braintools.metric.voltage_fluctuation(async_))
        True

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

    potentials = jnp.asarray(u.get_magnitude(potentials))
    avg = jnp.mean(potentials, axis=1)
    avg_var = jnp.var(avg)

    if method == 'loop':
        _var = brainstate.transform.for_loop(_f_signal, jnp.moveaxis(potentials, 0, 1))
    elif method == 'vmap':
        _var = vmap(_f_signal, in_axes=1)(potentials)
    else:
        raise ValueError(f'Do not support {method}. We only support "loop" or "vmap".')

    var_mean = jnp.mean(_var)
    # Double-``where`` guard: never evaluate ``avg_var / 0`` on the dead branch, which
    # previously produced eager-vs-jit discrepancies and division-by-zero warnings.
    safe_denom = jnp.where(var_mean == 0., 1., var_mean)
    return jnp.where(var_mean == 0., 1., avg_var / safe_denom)


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
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> # Two correlation matrices with similar structure.
        >>> x = jnp.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]])
        >>> y = jnp.array([[1.0, 0.7, 0.4], [0.7, 1.0, 0.6], [0.4, 0.6, 1.0]])
        >>> corr = braintools.metric.matrix_correlation(x, y)
        >>> bool(corr > 0.9)
        True

        >>> # Compare connectivity matrices from different conditions.
        >>> import brainstate
        >>> base = brainstate.random.rand(5, 5)
        >>> base = (base + base.T) / 2
        >>> base = base.at[jnp.diag_indices(5)].set(1.0)
        >>> treat = base + 0.1 * brainstate.random.rand(5, 5)
        >>> similarity = braintools.metric.matrix_correlation(base, treat)

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
    x = jnp.asarray(u.get_magnitude(x))
    y = jnp.asarray(u.get_magnitude(y))
    if x.ndim != 2:
        raise ValueError(f'Only support 2d array, but we got a array '
                         f'with the shape of {x.shape}')
    if y.ndim != 2:
        raise ValueError(f'Only support 2d array, but we got a array '
                         f'with the shape of {y.shape}')
    if x.shape != y.shape:
        raise ValueError(f'`x` and `y` must have the same shape, '
                         f'but got {x.shape} and {y.shape}.')
    x = x[jnp.triu_indices_from(x, k=1)]
    y = y[jnp.triu_indices_from(y, k=1)]
    cc = jnp.corrcoef(x, y)[0, 1]
    # Constant inputs make corrcoef ill-defined (NaN); map to 0 (no linear relation).
    return jnp.nan_to_num(cc)


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
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainstate
        >>> import braintools
        >>> t = jnp.linspace(0, 10, 100)
        >>> sig1 = jnp.sin(t) + 0.1 * brainstate.random.normal(size=100)
        >>> sig2 = jnp.sin(t + 0.2) + 0.1 * brainstate.random.normal(size=100)
        >>> activities = jnp.column_stack([sig1, sig2])
        >>> fc_matrix = braintools.metric.functional_connectivity(activities)
        >>> fc_matrix.shape
        (2, 2)

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
    activities = jnp.asarray(u.get_magnitude(activities))
    if activities.ndim != 2:
        raise ValueError('Only support 2d array with shape of "(num_time, num_sample)". '
                         f'But we got a array with the shape of {activities.shape}')
    n = activities.shape[1]
    fc = jnp.nan_to_num(jnp.atleast_2d(jnp.corrcoef(activities.T)))
    # A constant signal yields NaN correlations; ``nan_to_num`` zeros them, including
    # the diagonal. Restore the documented unit diagonal.
    fc = fc.at[jnp.diag_indices(n)].set(1.0)
    return fc


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
    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>> activities = brainstate.random.rand(200, 10)
        >>> fcd = braintools.metric.functional_connectivity_dynamics(activities)
        >>> fcd.shape
        (35, 35)
    """

    if activities.ndim != 2:
        raise ValueError('Only support 2d array with shape of "(num_time, num_sample)". '
                         f'But we got a array with the shape of {activities.shape}')

    activities = jnp.asarray(u.get_magnitude(activities))
    t_len, n_sig = activities.shape
    if window_size <= 1:
        raise ValueError('window_size must be > 1.')
    if step_size <= 0:
        raise ValueError('step_size must be > 0.')
    if n_sig < 2:
        raise ValueError(f'Functional connectivity needs at least 2 signals, but got {n_sig}.')

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
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> # Perfect linear relationship y = 2x.
        >>> x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])
        >>> # Weight middle points more heavily.
        >>> w = jnp.array([1.0, 1.0, 2.0, 2.0, 1.0])
        >>> corr = braintools.metric.weighted_correlation(x, y, w)
        >>> print(f"Weighted correlation: {corr:.3f}")
        Weighted correlation: 1.000
    """

    x = jnp.asarray(u.get_magnitude(x))
    y = jnp.asarray(u.get_magnitude(y))
    w = jnp.asarray(u.get_magnitude(w))
    if x.ndim != 1:
        raise ValueError(f'Only support 1d array, but we got a array '
                         f'with the shape of {x.shape}')
    if y.ndim != 1:
        raise ValueError(f'Only support 1d array, but we got a array '
                         f'with the shape of {y.shape}')
    if w.ndim != 1:
        raise ValueError(f'Only support 1d array, but we got a array '
                         f'with the shape of {w.shape}')
    if not (x.shape == y.shape == w.shape):
        raise ValueError(f'`x`, `y` and `w` must have the same length, '
                         f'but got {x.shape}, {y.shape} and {w.shape}.')

    # Guard against all-zero weights (0/0) without evaluating the bad branch.
    w_sum = jnp.sum(w)
    safe_w_sum = jnp.where(w_sum == 0., 1., w_sum)

    def _wmean(a):
        return jnp.sum(a * w) / safe_w_sum

    def _wcov(a, b):
        return jnp.sum(w * (a - _wmean(a)) * (b - _wmean(b))) / safe_w_sum

    denom = jnp.sqrt(_wcov(x, x) * _wcov(y, y))
    safe_denom = jnp.where(denom == 0., 1., denom)
    corr = jnp.where(denom == 0., 0., _wcov(x, y) / safe_denom)
    return jnp.clip(corr, -1.0, 1.0)
