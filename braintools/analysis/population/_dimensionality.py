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

"""Dimensionality, alignment and representational geometry of population activity.

Every function here is JAX-native: shapes depend only on the input shapes and
on static Python arguments, so all of them run under ``jax.jit`` and
``jax.vmap``.
"""

import operator
from typing import Any, NamedTuple, Optional, Tuple

import brainunit as u
import jax
import jax.numpy as jnp

from braintools._misc import set_module_as

__all__ = [
    'PCAResult',
    'CCAResult',
    'pca',
    'participation_ratio',
    'cca',
    'rdm',
    'rsa',
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _split(x: Any) -> Tuple[Any, Optional[u.Unit]]:
    """Split ``x`` into ``(mantissa, unit)``; ``unit`` is None for unitless input."""
    if isinstance(x, u.Quantity):
        if x.unit.is_unitless:
            return x.mantissa, None
        return x.mantissa, x.unit
    return x, None


def _wrap(mantissa: Any, unit: Optional[u.Unit]) -> Any:
    """Attach ``unit`` to ``mantissa`` (no-op for unitless data)."""
    if unit is None:
        return mantissa
    return u.Quantity(mantissa, unit=unit)


def _as_float(x: Any) -> jax.Array:
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(jnp.result_type(float))
    return x


def _as_2d(x: jax.Array, name: str) -> jax.Array:
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f'`{name}` must have shape (n_samples, n_features); got shape {x.shape}.')
    return x


def _n_components(n_components: Any, maximum: int) -> int:
    if n_components is None:
        return maximum
    if isinstance(n_components, bool):
        raise TypeError(f'`n_components` must be a static int; got {n_components!r}.')
    try:
        n_components = operator.index(n_components)
    except TypeError:
        raise TypeError(f'`n_components` must be a static int; got {n_components!r}.') from None
    if not 1 <= n_components <= maximum:
        raise ValueError(f'`n_components` must be in [1, {maximum}]; got {n_components}.')
    return int(n_components)


def _is_concrete(value: Any) -> bool:
    return not isinstance(value, jax.core.Tracer)


def _inv_sqrt_psd(c: jax.Array) -> jax.Array:
    """Pseudo-inverse square root of a symmetric PSD matrix (eigh based)."""
    w, v = jnp.linalg.eigh(c)
    tol = jnp.max(jnp.abs(w)) * c.shape[0] * jnp.finfo(c.dtype).eps
    ok = w > tol
    inv = jnp.where(ok, 1.0 / jnp.sqrt(jnp.where(ok, w, 1.0)), 0.0)
    return (v * inv) @ v.T


def _sign_of_max_abs(m: jax.Array) -> jax.Array:
    """Sign of the largest-magnitude entry of every row of ``m`` (never 0)."""
    idx = jnp.argmax(jnp.abs(m), axis=1)
    s = jnp.sign(jnp.take_along_axis(m, idx[:, None], axis=1)[:, 0])
    return jnp.where(s == 0, 1.0, s).astype(m.dtype)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

class PCAResult(NamedTuple):
    """Result of :func:`pca` (a pytree, so it passes through ``jax.jit``).

    Attributes
    ----------
    components : jax.Array
        Principal axes, shape ``(k, D)``, unit-norm rows ordered by decreasing
        explained variance (sklearn's ``components_``).
    explained_variance : jax.Array or brainunit.Quantity
        Variance along each axis, shape ``(k,)``, with ``ddof=1``. Carries the
        squared input unit for Quantity input.
    explained_variance_ratio : jax.Array
        Fraction of the total variance explained by each axis, shape ``(k,)``.
    mean : jax.Array or brainunit.Quantity
        Per-feature mean that was subtracted, shape ``(D,)``.
    projection : jax.Array or brainunit.Quantity
        Mean-centred data projected onto the axes, shape ``(T, k)``.
    """
    components: Any
    explained_variance: Any
    explained_variance_ratio: Any
    mean: Any
    projection: Any


@set_module_as('braintools.analysis.population')
def pca(x, n_components: Optional[int] = None) -> PCAResult:
    r"""Principal component analysis by mean-centred SVD.

    The data matrix is centred per feature and decomposed as
    :math:`X_c = U S V^\top` [1]_. The rows of :math:`V^\top` are the principal
    axes; the variance along axis :math:`i` is :math:`s_i^2 / (T - 1)`.
    Signs are made deterministic as in sklearn's ``svd_flip`` with
    ``u_based_decision=False``: the largest-magnitude entry of every component
    is positive.

    Parameters
    ----------
    x : array_like or brainunit.Quantity
        Activity of shape ``(T, D)``: rows are samples (time points or trials),
        columns are features (neurons). A 1-D array is treated as ``(T, 1)``.
    n_components : int, optional
        Number of components ``k`` to keep (static). Defaults to
        ``min(T, D)``, which is also the maximum.

    Returns
    -------
    PCAResult
        Named tuple ``(components (k, D), explained_variance (k,),
        explained_variance_ratio (k,), mean (D,), projection (T, k))``.
        ``explained_variance`` carries ``unit**2`` and ``mean``/``projection``
        carry ``unit`` when ``x`` is a Quantity. The ratio is relative to the
        total variance over *all* ``min(T, D)`` components, as in sklearn; it
        is ``nan`` for constant data (zero total variance).

    Raises
    ------
    ValueError
        If ``x`` is not 1-D/2-D, has fewer than two samples, or
        ``n_components`` is out of range.

    See Also
    --------
    participation_ratio : Effective dimensionality from the same spectrum.
    cca : Alignment between two populations.

    Notes
    -----
    Output shapes depend only on ``x.shape`` and the static ``n_components``,
    so the function is jittable and vmappable (e.g. over trials). It is
    differentiable through ``jnp.linalg.svd`` when the non-zero singular values
    are distinct; repeated singular values make the gradient of the components
    undefined. Matches ``sklearn.decomposition.PCA(n_components=k)`` (with
    ``whiten=False``): ``components_``, ``explained_variance_``,
    ``explained_variance_ratio_``, ``mean_`` and ``transform(x)``.

    References
    ----------
    .. [1] Jolliffe, I. T. and Cadima, J. (2016). "Principal component
           analysis: a review and recent developments." Philosophical
           Transactions of the Royal Society A 374: 20150202.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis import population
        >>> x = jnp.array([[1., 2.], [3., 4.], [5., 7.], [7., 9.]])
        >>> res = population.pca(x)
        >>> res.components.shape, res.projection.shape
        ((2, 2), (4, 2))
        >>> [round(float(v), 3) for v in res.explained_variance_ratio]
        [0.998, 0.002]
        >>>
        >>> # Quantities keep their units.
        >>> import brainunit as u
        >>> res = population.pca(x * u.mV, n_components=1)
        >>> res.explained_variance.unit
        Unit("mV^2")
    """
    xm, unit = _split(x)
    xm = _as_2d(_as_float(xm), 'x')
    n_samples, n_features = xm.shape
    if n_samples < 2:
        raise ValueError(f'`pca` needs at least 2 samples; got {n_samples}.')
    k_max = min(n_samples, n_features)
    k = _n_components(n_components, k_max)

    mean = jnp.mean(xm, axis=0)
    xc = xm - mean
    _, s, vt = jnp.linalg.svd(xc, full_matrices=False)
    vt = vt * _sign_of_max_abs(vt)[:, None]

    ev = s ** 2 / (n_samples - 1)
    ratio = ev / jnp.sum(ev)
    components = vt[:k]
    projection = xc @ components.T
    return PCAResult(
        components=components,
        explained_variance=_wrap(ev[:k], None if unit is None else unit ** 2),
        explained_variance_ratio=ratio[:k],
        mean=_wrap(mean, unit),
        projection=_wrap(projection, unit),
    )


@set_module_as('braintools.analysis.population')
def participation_ratio(x) -> jax.Array:
    r"""Participation ratio (effective dimensionality) of population activity.

    .. math::

        \mathrm{PR} = \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}
                    = \frac{\operatorname{tr}(C)^2}{\operatorname{tr}(C^2)},

    where :math:`\lambda_i` are the eigenvalues of the feature covariance
    matrix :math:`C` [1]_ [2]_. It ranges from 1 (one dominant dimension) to
    ``D`` (isotropic variance).

    Parameters
    ----------
    x : array_like or brainunit.Quantity
        Activity of shape ``(T, D)`` (samples by features). Units cancel.

    Returns
    -------
    jax.Array
        0-d participation ratio. ``nan`` for constant data (zero variance).

    Raises
    ------
    ValueError
        If ``x`` is not 2-D or has fewer than two samples.

    See Also
    --------
    pca : The covariance spectrum itself.

    Notes
    -----
    Computed from traces without an eigendecomposition, using the smaller of
    the ``(D, D)`` covariance and the ``(T, T)`` Gram matrix (both have the same
    non-zero spectrum). The normalisation (``ddof``) cancels. Jittable,
    vmappable and differentiable (except for constant data).

    References
    ----------
    .. [1] Gao, P., Trautmann, E., Yu, B., Santhanam, G., Ryu, S., Shenoy, K.
           and Ganguli, S. (2017). "A theory of multineuronal dimensionality,
           dynamics and measurement." bioRxiv 214262.
    .. [2] Litwin-Kumar, A., Harris, K. D., Axel, R., Sompolinsky, H. and
           Abbott, L. F. (2017). "Optimal degrees of synaptic connectivity."
           Neuron 93(5): 1153-1164.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from braintools.analysis import population
        >>> x = jax.random.normal(jax.random.PRNGKey(0), (2000, 4))
        >>> round(float(population.participation_ratio(x)))
        4
        >>> latent = jax.random.normal(jax.random.PRNGKey(1), (2000, 1))
        >>> round(float(population.participation_ratio(latent @ jnp.ones((1, 4)))), 3)
        1.0
    """
    xm, _ = _split(x)
    xm = _as_float(xm)
    if xm.ndim != 2:
        raise ValueError(f'`x` must have shape (n_samples, n_features); got shape {xm.shape}.')
    n_samples, n_features = xm.shape
    if n_samples < 2:
        raise ValueError(f'`participation_ratio` needs at least 2 samples; got {n_samples}.')
    xc = xm - jnp.mean(xm, axis=0)
    gram = xc.T @ xc if n_features <= n_samples else xc @ xc.T
    return jnp.trace(gram) ** 2 / jnp.sum(gram * gram)


# ---------------------------------------------------------------------------
# CCA
# ---------------------------------------------------------------------------

class CCAResult(NamedTuple):
    """Result of :func:`cca` (a pytree, so it passes through ``jax.jit``).

    Attributes
    ----------
    correlations : jax.Array
        Canonical correlations, shape ``(k,)``, in decreasing order.
    x_weights : jax.Array or brainunit.Quantity
        Canonical weights for ``x``, shape ``(p, k)`` (unit ``1/unit_x``).
    y_weights : jax.Array or brainunit.Quantity
        Canonical weights for ``y``, shape ``(q, k)`` (unit ``1/unit_y``).
    """
    correlations: Any
    x_weights: Any
    y_weights: Any


@set_module_as('braintools.analysis.population')
def cca(x, y, n_components: Optional[int] = None, reg=0.0) -> CCAResult:
    r"""Canonical correlation analysis via SVD of the whitened cross-covariance.

    With centred data, covariances
    :math:`C_{xx} = X_c^\top X_c/(T-1) + \text{reg}\,I`,
    :math:`C_{yy}` likewise and :math:`C_{xy} = X_c^\top Y_c/(T-1)`, the
    whitened cross-covariance
    :math:`M = C_{xx}^{-1/2} C_{xy} C_{yy}^{-1/2} = U S V^\top` gives the
    canonical correlations :math:`S` and weights
    :math:`W_x = C_{xx}^{-1/2} U`, :math:`W_y = C_{yy}^{-1/2} V` [1]_ [2]_.

    Parameters
    ----------
    x : array_like or brainunit.Quantity
        First data set, shape ``(T, p)`` (a 1-D array is treated as ``(T, 1)``).
    y : array_like or brainunit.Quantity
        Second data set, shape ``(T, q)``, same number of samples.
    n_components : int, optional
        Number of canonical pairs ``k`` (static). Defaults to ``min(p, q)``,
        which is also the maximum.
    reg : float, default=0.0
        Ridge added to the diagonal of both covariance matrices (in the squared
        data units). Must be non-negative. Use ``reg > 0`` when ``T`` is not
        much larger than ``p + q``.

    Returns
    -------
    CCAResult
        Named tuple ``(correlations (k,), x_weights (p, k), y_weights (q, k))``.
        With ``reg=0`` the canonical variates ``(x - x.mean(0)) @ x_weights``
        have unit variance (``ddof=1``) and are mutually uncorrelated. Signs are
        deterministic: the largest-magnitude entry of each ``x_weights`` column
        is positive (and the paired ``y_weights`` column follows, so the
        correlations are non-negative).

    Raises
    ------
    ValueError
        If the inputs are not 1-D/2-D, their sample counts differ, there are
        fewer than two samples, ``n_components`` is out of range or a concrete
        ``reg`` is negative.

    See Also
    --------
    pca : Variance-maximising projection of a single data set.

    Notes
    -----
    Rank-deficient covariances (e.g. collinear features with ``reg=0``) are
    whitened with a pseudo-inverse square root, so the corresponding directions
    get zero weight instead of producing ``inf``/``nan``. Jittable and
    vmappable; differentiable for generic data (the ``eigh``-based whitening
    has an undefined derivative when a covariance has repeated eigenvalues).
    ``sklearn.cross_decomposition.CCA`` is an iterative PLS-style algorithm and
    only agrees approximately.

    References
    ----------
    .. [1] Hotelling, H. (1936). "Relations between two sets of variates."
           Biometrika 28(3/4): 321-377.
    .. [2] Uurtio, V., Monteiro, J. M., Kandola, J., Shawe-Taylor, J.,
           Fernandez-Reyes, D. and Rousu, J. (2017). "A tutorial on canonical
           correlation methods." ACM Computing Surveys 50(6): 95.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from braintools.analysis import population
        >>> x = jax.random.normal(jax.random.PRNGKey(0), (500, 3))
        >>> y = x @ jnp.array([[1., 0.], [2., 1.], [0., -1.]])  # linear transform
        >>> res = population.cca(x, y)
        >>> [round(float(c), 4) for c in res.correlations]
        [1.0, 1.0]
        >>> res.x_weights.shape, res.y_weights.shape
        ((3, 2), (2, 2))
    """
    xm, ux = _split(x)
    ym, uy = _split(y)
    xm = _as_2d(_as_float(xm), 'x')
    ym = _as_2d(_as_float(ym), 'y')
    if xm.shape[0] != ym.shape[0]:
        raise ValueError(f'`x` and `y` must have the same number of samples; '
                         f'got {xm.shape[0]} and {ym.shape[0]}.')
    n_samples, p = xm.shape
    q = ym.shape[1]
    if n_samples < 2:
        raise ValueError(f'`cca` needs at least 2 samples; got {n_samples}.')
    k = _n_components(n_components, min(p, q))
    if _is_concrete(reg) and float(jnp.asarray(reg)) < 0:
        raise ValueError(f'`reg` must be non-negative; got {reg}.')
    dtype = jnp.result_type(xm.dtype, ym.dtype)
    xm = xm.astype(dtype)
    ym = ym.astype(dtype)
    reg = jnp.asarray(reg, dtype=dtype)

    xc = xm - jnp.mean(xm, axis=0)
    yc = ym - jnp.mean(ym, axis=0)
    denom = n_samples - 1
    cxx = xc.T @ xc / denom + reg * jnp.eye(p, dtype=dtype)
    cyy = yc.T @ yc / denom + reg * jnp.eye(q, dtype=dtype)
    cxy = xc.T @ yc / denom

    wx = _inv_sqrt_psd(cxx)
    wy = _inv_sqrt_psd(cyy)
    uu, s, vt = jnp.linalg.svd(wx @ cxy @ wy, full_matrices=False)
    xw = wx @ uu[:, :k]
    yw = wy @ vt[:k].T
    signs = _sign_of_max_abs(xw.T)
    xw = xw * signs
    yw = yw * signs
    return CCAResult(
        correlations=jnp.minimum(s[:k], 1.0),
        x_weights=_wrap(xw, None if ux is None else ux ** -1),
        y_weights=_wrap(yw, None if uy is None else uy ** -1),
    )


# ---------------------------------------------------------------------------
# RDM / RSA
# ---------------------------------------------------------------------------

_RDM_METRICS = ('correlation', 'euclidean', 'cosine')


@set_module_as('braintools.analysis.population')
def rdm(x, metric: str = 'correlation'):
    r"""Representational dissimilarity matrix between conditions.

    Entry ``(i, j)`` is the dissimilarity between the activity patterns of
    conditions ``i`` and ``j`` (rows of ``x``) [1]_:

    - ``'correlation'``: :math:`1 - r(x_i, x_j)`, one minus the Pearson
      correlation across features;
    - ``'cosine'``: :math:`1 - x_i \cdot x_j / (\lVert x_i\rVert \lVert x_j\rVert)`;
    - ``'euclidean'``: :math:`\lVert x_i - x_j \rVert_2`.

    Parameters
    ----------
    x : array_like or brainunit.Quantity
        Condition-by-feature patterns, shape ``(C, D)``.
    metric : {'correlation', 'euclidean', 'cosine'}, default='correlation'
        Dissimilarity measure (same definitions as
        ``scipy.spatial.distance.pdist``).

    Returns
    -------
    jax.Array or brainunit.Quantity
        Symmetric ``(C, C)`` matrix with a zero diagonal. ``'euclidean'``
        carries the unit of ``x``; the other metrics are dimensionless.
        Off-diagonal entries involving a zero-norm (``'cosine'``) or constant
        (``'correlation'``) pattern are ``nan``, as in scipy.

    Raises
    ------
    ValueError
        If ``x`` is not 2-D or ``metric`` is unknown.

    See Also
    --------
    rsa : Compare two RDMs.

    Notes
    -----
    Equal to ``scipy.spatial.distance.squareform(pdist(x, metric))``.
    The Euclidean metric is computed from explicit differences (memory
    ``O(C^2 D)``) for accuracy. Jittable, vmappable and differentiable (the
    Euclidean branch has a zero subgradient at zero distance).

    References
    ----------
    .. [1] Kriegeskorte, N., Mur, M. and Bandettini, P. (2008).
           "Representational similarity analysis - connecting the branches of
           systems neuroscience." Frontiers in Systems Neuroscience 2: 4.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis import population
        >>> x = jnp.array([[0., 0.], [3., 4.], [1., 0.]])
        >>> population.rdm(x, metric='euclidean')
        Array([[0.      , 5.      , 1.      ],
               [5.      , 0.      , 4.472136],
               [1.      , 4.472136, 0.      ]], dtype=float32)
    """
    if metric not in _RDM_METRICS:
        raise ValueError(f'`metric` must be one of {_RDM_METRICS}; got {metric!r}.')
    xm, unit = _split(x)
    xm = _as_float(xm)
    if xm.ndim != 2:
        raise ValueError(f'`x` must have shape (n_conditions, n_features); got shape {xm.shape}.')
    n = xm.shape[0]

    if metric == 'euclidean':
        def row(a):
            sq = jnp.sum((a[None, :] - xm) ** 2, axis=1)
            pos = sq > 0
            return jnp.where(pos, jnp.sqrt(jnp.where(pos, sq, 1.0)), 0.0)

        d = jax.vmap(row)(xm)
    else:
        z = xm - jnp.mean(xm, axis=1, keepdims=True) if metric == 'correlation' else xm
        z = z / jnp.linalg.norm(z, axis=1, keepdims=True)
        d = 1.0 - z @ z.T
    d = 0.5 * (d + d.T)
    d = jnp.where(jnp.eye(n, dtype=bool), 0.0, d)
    return _wrap(d, unit if metric == 'euclidean' else None)


def _average_ranks(v: jax.Array) -> jax.Array:
    """1-based ranks with ties replaced by their average (``scipy.stats.rankdata``)."""
    s = jnp.sort(v)
    lo = jnp.searchsorted(s, v, side='left')
    hi = jnp.searchsorted(s, v, side='right')
    return (lo + hi + 1).astype(v.dtype) / 2


def _pearson(a: jax.Array, b: jax.Array) -> jax.Array:
    a = a - jnp.mean(a)
    b = b - jnp.mean(b)
    return jnp.sum(a * b) / jnp.sqrt(jnp.sum(a * a) * jnp.sum(b * b))


def _upper(m: jax.Array, name: str) -> jax.Array:
    if m.ndim == 1:
        return m
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f'`{name}` must be a square (C, C) RDM or a condensed 1-D vector; '
                         f'got shape {m.shape}.')
    iu = jnp.triu_indices(m.shape[0], k=1)
    return m[iu]


@set_module_as('braintools.analysis.population')
def rsa(rdm_a, rdm_b, method: str = 'spearman') -> jax.Array:
    r"""Representational similarity: correlation between two RDMs.

    The strict upper triangles of the two dissimilarity matrices are correlated
    [1]_ with Spearman's rank correlation (ranks with ties averaged) or with
    Pearson's correlation.

    Parameters
    ----------
    rdm_a, rdm_b : array_like or brainunit.Quantity
        Square ``(C, C)`` RDMs (e.g. from :func:`rdm`), or condensed vectors of
        shape ``(C (C - 1) / 2,)`` as returned by ``scipy.spatial.distance.pdist``.
        Both must have the same shape. Units cancel.
    method : {'spearman', 'pearson'}, default='spearman'
        Correlation coefficient.

    Returns
    -------
    jax.Array
        0-d correlation in ``[-1, 1]``. ``nan`` if either upper triangle is
        constant or contains ``nan`` (scipy's ``nan_policy='propagate'``).

    Raises
    ------
    ValueError
        If the shapes differ, are not square/1-D, or ``method`` is unknown.

    See Also
    --------
    rdm : Build an RDM from condition patterns.

    Notes
    -----
    Matches ``scipy.stats.spearmanr(a, b).statistic`` and
    ``scipy.stats.pearsonr(a, b).statistic`` on the condensed vectors. Ranks
    are computed with a sort and two ``searchsorted`` calls, so the function is
    jittable and vmappable (e.g. over a batch of model RDMs). ``'pearson'`` is
    differentiable; ``'spearman'`` has a zero gradient almost everywhere.

    References
    ----------
    .. [1] Kriegeskorte, N., Mur, M. and Bandettini, P. (2008).
           "Representational similarity analysis - connecting the branches of
           systems neuroscience." Frontiers in Systems Neuroscience 2: 4.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis import population
        >>> x = jnp.array([[0., 1.], [1., 3.], [4., 2.], [5., 5.]])
        >>> a = population.rdm(x, metric='euclidean')
        >>> round(float(population.rsa(a, a ** 2)), 4)  # monotone transform
        1.0
        >>> round(float(population.rsa(a, a ** 2, method='pearson')), 4)
        0.9882
    """
    if method not in ('spearman', 'pearson'):
        raise ValueError(f"`method` must be 'spearman' or 'pearson'; got {method!r}.")
    a, _ = _split(rdm_a)
    b, _ = _split(rdm_b)
    a = _as_float(a)
    b = _as_float(b)
    if a.shape != b.shape:
        raise ValueError(f'`rdm_a` and `rdm_b` must have the same shape; got {a.shape} and {b.shape}.')
    dtype = jnp.result_type(a.dtype, b.dtype)
    va = _upper(a, 'rdm_a').astype(dtype)
    vb = _upper(b, 'rdm_b').astype(dtype)
    has_nan = jnp.any(jnp.isnan(va)) | jnp.any(jnp.isnan(vb))
    if method == 'spearman':
        va = _average_ranks(va)
        vb = _average_ranks(vb)
    return jnp.where(has_nan, jnp.nan, _pearson(va, vb))
