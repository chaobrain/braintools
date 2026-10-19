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

import brainunit as u
from jax import numpy as jnp

from braintools._misc import set_module_as

__all__ = [
    'matrix_correlation',
    'weighted_correlation',
]


@set_module_as('braintools.analysis.stats')
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
        >>> corr = braintools.analysis.stats.matrix_correlation(x, y)
        >>> bool(corr > 0.9)
        True

        >>> # Compare connectivity matrices from different conditions.
        >>> import brainstate
        >>> base = brainstate.random.rand(5, 5)
        >>> base = (base + base.T) / 2
        >>> base = base.at[jnp.diag_indices(5)].set(1.0)
        >>> treat = base + 0.1 * brainstate.random.rand(5, 5)
        >>> similarity = braintools.analysis.stats.matrix_correlation(base, treat)

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
    braintools.analysis.population.functional_connectivity : Compute connectivity matrix from time series
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


@set_module_as('braintools.analysis.stats')
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
        >>> corr = braintools.analysis.stats.weighted_correlation(x, y, w)
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
