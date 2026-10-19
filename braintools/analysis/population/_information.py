# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Information-theoretic dependence between discrete (or binned) variables."""

import operator
from typing import Any, Optional, Tuple, Union

import brainunit as u
import jax
import jax.numpy as jnp

from braintools._misc import set_module_as

__all__ = [
    'mutual_information',
]


def _static_bins(bins: Any) -> Tuple[int, int]:
    if isinstance(bins, (tuple, list)):
        if len(bins) != 2:
            raise ValueError(f'`bins` must be an int or a pair of ints; got {bins!r}.')
        pair = tuple(bins)
    else:
        pair = (bins, bins)
    out = []
    for b in pair:
        if isinstance(b, bool):
            raise TypeError(f'`bins` must be static ints; got {bins!r}.')
        try:
            b = operator.index(b)
        except TypeError:
            raise TypeError(f'`bins` must be static ints; got {bins!r}.') from None
        if b < 1:
            raise ValueError(f'`bins` must be >= 1; got {bins!r}.')
        out.append(int(b))
    return out[0], out[1]


def _equal_width_bin(v: jax.Array, n_bins: int) -> jax.Array:
    """Bin index in ``[0, n_bins)``, identical to ``numpy.histogram2d`` with ``bins=n_bins``."""
    if not jnp.issubdtype(v.dtype, jnp.floating):
        v = v.astype(jnp.result_type(float))
    lo = jnp.min(v)
    hi = jnp.max(v)
    same = lo == hi
    lo = jnp.where(same, lo - 0.5, lo)
    hi = jnp.where(same, hi + 0.5, hi)
    edges = jnp.linspace(lo, hi, n_bins + 1)
    idx = jnp.searchsorted(edges, v, side='right') - 1
    return jnp.clip(idx, 0, n_bins - 1)


def _value_counts(v: jax.Array) -> jax.Array:
    """For every element, the number of elements equal to it."""
    s = jnp.sort(v)
    return jnp.searchsorted(s, v, side='right') - jnp.searchsorted(s, v, side='left')


def _pair_counts(a: jax.Array, b: jax.Array) -> jax.Array:
    """For every index ``i``, the number of ``j`` with ``(a[j], b[j]) == (a[i], b[i])``."""
    n = a.shape[0]
    order = jnp.lexsort((b, a))
    sa = a[order]
    sb = b[order]
    new = jnp.concatenate([jnp.ones((1,), dtype=bool), (sa[1:] != sa[:-1]) | (sb[1:] != sb[:-1])])
    seg = jnp.cumsum(new) - 1
    run = jax.ops.segment_sum(jnp.ones((n,), dtype=jnp.int32), seg, num_segments=n)
    return jnp.zeros((n,), dtype=jnp.int32).at[order].set(run[seg])


@set_module_as('braintools.analysis.population')
def mutual_information(x, y, bins: Optional[Union[int, Tuple[int, int]]] = None) -> jax.Array:
    r"""Plug-in mutual information (in nats) between two discrete variables.

    .. math::

        I(X; Y) = \sum_{a, b} p(a, b) \log \frac{p(a, b)}{p(a)\, p(b)}
                = \frac{1}{T} \sum_{t=1}^{T}
                  \log \frac{T\, n(x_t, y_t)}{n(x_t)\, n(y_t)},

    where the probabilities are empirical frequencies over the ``T`` paired
    samples and :math:`n(\cdot)` counts matching samples [1]_.

    Parameters
    ----------
    x, y : array_like or brainunit.Quantity
        Paired samples of shape ``(T,)``. Without ``bins`` every distinct value
        is its own symbol (typically integer labels or spike counts). Units
        are irrelevant and are dropped.
    bins : int or tuple of int, optional
        Static number of equal-width bins for continuous input, for both
        variables or as ``(bins_x, bins_y)``. Each variable is binned over its
        own ``[min, max]`` range exactly as ``numpy.histogram2d`` does (the
        last bin is closed; a constant variable falls in a single bin).

    Returns
    -------
    jax.Array
        0-d mutual information in nats, clipped below at 0. Divide by
        ``log(2)`` for bits.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` are not 1-D arrays of the same non-zero length, or
        ``bins`` is invalid.

    See Also
    --------
    decode : Decoding accuracy as an alternative dependence measure.

    Notes
    -----
    Matches ``sklearn.metrics.mutual_info_score(x, y)`` and, with ``bins``,
    ``mutual_info_score(None, None, contingency=numpy.histogram2d(x, y,
    bins)[0])``. The plug-in estimator is biased upwards for small samples
    (Panzeri-Treves bias). Counts are obtained by sorting (``O(T log T)``, no
    ``(T, T)`` intermediates), so the function is jittable and vmappable (e.g.
    over trials or neurons) for any value range. It is not differentiable.

    References
    ----------
    .. [1] Cover, T. M. and Thomas, J. A. (2006). Elements of Information
           Theory, 2nd edition. Wiley.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis import population
        >>> x = jnp.array([0, 0, 1, 1, 2, 2])
        >>> round(float(population.mutual_information(x, x)), 4)  # = log(3)
        1.0986
        >>> round(float(population.mutual_information(x, jnp.array([0, 1, 0, 1, 0, 1]))), 4)
        0.0
        >>>
        >>> # Continuous signals: equal-width binning.
        >>> t = jnp.linspace(0., 1., 100)
        >>> round(float(population.mutual_information(t, t ** 2, bins=4)), 4)
        0.9527
    """
    if isinstance(x, u.Quantity):
        x = x.mantissa
    if isinstance(y, u.Quantity):
        y = y.mantissa
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(f'`x` and `y` must be 1-D arrays of the same length; got shapes '
                         f'{x.shape} and {y.shape}.')
    n = x.shape[0]
    if n == 0:
        raise ValueError('`x` and `y` must not be empty.')
    if bins is not None:
        bx, by = _static_bins(bins)
        x = _equal_width_bin(x, bx)
        y = _equal_width_bin(y, by)
    ftype = jnp.result_type(float)
    cx = _value_counts(x).astype(ftype)
    cy = _value_counts(y).astype(ftype)
    cxy = _pair_counts(x, y).astype(ftype)
    mi = jnp.mean(jnp.log(n * cxy / (cx * cy)))
    return jnp.maximum(mi, 0.0)
