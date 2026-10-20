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

"""Loss functions comparing a model prediction with a target.

Every loss has the signature ``loss(prediction, target, **static_options)``
and returns a scalar ``jax.Array`` (a number to *minimise*, ``0`` for
identical inputs). The options are plain keyword arguments, so a configured
loss is ``functools.partial(loss, ...)``; the options that fix array shapes
(window sizes, segment lengths, sampling steps, bands) must be static.

The losses fall in four groups:

- **Time series** (time on axis 0): :func:`mse`, :func:`rmse`, :func:`mae`,
  :func:`correlation_loss`.
- **Functional connectivity**: :func:`fc_correlation_loss`, :func:`fc_rmse`,
  :func:`fcd_ks`, :func:`fcd_wasserstein`, and the 1-D distribution distances
  :func:`ks_distance` and :func:`wasserstein_1d`.
- **Spectra**: :func:`psd_log_distance`.
- **Spikes**: :func:`rate_loss`, :func:`cv_loss`, :func:`van_rossum_loss`.

Time steps given as plain numbers follow the convention of the matching
analysis module: ``dt`` is in **seconds** for :func:`psd_log_distance` (as in
:mod:`braintools.analysis.spectral`) but in **milliseconds** for
:func:`rate_loss` and :func:`van_rossum_loss` (as in
:mod:`braintools.analysis.spike` and ``brainstate``). Passing ``dt`` as a
``brainunit`` Quantity avoids the ambiguity.

The functional-connectivity-dynamics losses and the 1-D distribution
distances are ported from ``brainmass.objectives`` (BrainX, Apache-2.0) with
identical numerics; brainmass builds them as factories, whereas here the
configuration is passed as keyword arguments.
"""

from typing import Any, Optional, Tuple

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braintools._misc import set_module_as
from braintools.analysis.population import (
    functional_connectivity,
    functional_connectivity_dynamics,
)
from braintools.analysis.spectral import power_spectral_density
from braintools.analysis.spike import cv as _cv
from braintools.analysis.stats import matrix_correlation
from braintools.spike import SpikeTrains

__all__ = [
    'mse',
    'rmse',
    'mae',
    'correlation_loss',
    'fc_correlation_loss',
    'fc_rmse',
    'fcd_ks',
    'fcd_wasserstein',
    'ks_distance',
    'wasserstein_1d',
    'psd_log_distance',
    'rate_loss',
    'cv_loss',
    'van_rossum_loss',
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _has_unit(x: Any) -> bool:
    """Whether ``x`` is a Quantity with a physical (non-unitless) unit."""
    return isinstance(x, u.Quantity) and not x.unit.is_unitless


def _plain(x: Any) -> jax.Array:
    """Magnitude of ``x`` as a floating JAX array (no unit check)."""
    x = jnp.asarray(u.get_magnitude(x))
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        x = x.astype(jnp.result_type(float))
    return x


def _pair(prediction: Any, target: Any, where: str) -> Tuple[jax.Array, jax.Array]:
    """Unit-checked magnitudes of ``prediction`` and ``target``, in the prediction's unit.

    Both must be plain arrays, or both Quantities of the same dimension; the
    target is converted to the prediction's unit before the magnitudes are
    taken. Shapes must match.
    """
    hp, ht = _has_unit(prediction), _has_unit(target)
    if hp != ht:
        raise TypeError(
            f'{where}: `prediction` and `target` must both carry compatible units or '
            f'both be plain arrays; got {type(prediction).__name__} and '
            f'{type(target).__name__}.'
        )
    if hp:
        if not prediction.unit.has_same_dim(target.unit):
            raise u.UnitMismatchError(
                f'{where}: `prediction` ({prediction.unit}) and `target` '
                f'({target.unit}) have incompatible units.'
            )
        target = target.to_decimal(prediction.unit)
        prediction = prediction.mantissa
    p, t = _plain(prediction), _plain(target)
    _check_same_shape(p, t, where)
    return p, t


def _check_same_shape(p: jax.Array, t: jax.Array, where: str) -> None:
    if p.shape != t.shape:
        raise ValueError(
            f'{where}: `prediction` and `target` must have the same shape, '
            f'got {p.shape} and {t.shape}.'
        )


def _safe_sqrt(x: jax.Array) -> jax.Array:
    """``sqrt(x)`` with a zero (not infinite) gradient at ``x == 0``."""
    pos = x > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1)), 0)


def _pearson(a: jax.Array, b: jax.Array, axis=None) -> jax.Array:
    """Pearson correlation along ``axis``; 0 where either input is constant.

    Gradient-safe: a constant input gives a finite (zero) gradient instead of
    ``nan``.
    """
    a = a - jnp.mean(a, axis=axis, keepdims=True)
    b = b - jnp.mean(b, axis=axis, keepdims=True)
    cov = jnp.sum(a * b, axis=axis)
    prod = jnp.sum(a * a, axis=axis) * jnp.sum(b * b, axis=axis)
    ok = prod > 0
    return jnp.where(ok, cov / jnp.sqrt(jnp.where(ok, prod, 1)), 0)


def _check_2d(x: jax.Array, name: str, where: str) -> None:
    if x.ndim != 2:
        raise ValueError(
            f'{where}: `{name}` must have shape (n_time, n_signals), got shape {x.shape}.'
        )
    if x.shape[1] < 2:
        raise ValueError(
            f'{where}: functional connectivity needs at least 2 signals, '
            f'got {x.shape[1]}.'
        )


def _time_ms(value: Any, name: str) -> Any:
    """A time as a plain value in milliseconds (plain numbers are already ms)."""
    if isinstance(value, u.Quantity):
        if value.unit.is_unitless:
            return value.mantissa
        try:
            return value.to_decimal(u.ms)
        except Exception as e:  # incompatible dimension
            raise TypeError(f'`{name}` must be a time, got {value!r}.') from e
    return value


def _dt_ms(dt: Any, where: str) -> Any:
    """The raster time step in ms, falling back to ``brainstate.environ``."""
    if dt is None:
        try:
            dt = brainstate.environ.get_dt()
        except KeyError:
            raise ValueError(
                f'{where}: `dt` was not given and `brainstate.environ` has no time step; '
                f'pass `dt=` explicitly or set it with `brainstate.environ.set(dt=...)`.'
            ) from None
    return _time_ms(dt, 'dt')


def _as_raster(x: Any, where: str, name: str) -> jax.Array:
    """A dense ``(T, N)`` raster (a 1-D raster becomes ``(T, 1)``)."""
    if _has_unit(x):
        raise TypeError(f'{where}: the `{name}` raster must be unitless, got unit {x.unit}.')
    x = _plain(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(
            f'{where}: the `{name}` raster must have shape (T,) or (T, N), got shape {x.shape}.'
        )
    return x


# ----------------------------------------------------------------------------
# time-series losses
# ----------------------------------------------------------------------------

@set_module_as('braintools.fit.losses')
def mse(prediction: Any, target: Any) -> jax.Array:
    r"""Compute the mean squared error between two time series.

    .. math::

        \mathrm{MSE} = \frac{1}{M} \sum_{i=1}^{M} (\hat y_i - y_i)^2,

    averaged over all ``M`` elements (time on axis 0, any trailing shape).

    Parameters
    ----------
    prediction : array-like or Quantity
        Model output, e.g. of shape ``(n_time, n_channels)``.
    target : array-like or Quantity
        Data of the same shape. If one input carries units, both must, with
        the same dimension.

    Returns
    -------
    jax.Array
        Scalar MSE. For Quantity inputs it is the plain magnitude in the
        *prediction's* unit squared (the target is converted first).

    Raises
    ------
    ValueError
        If the shapes differ.
    TypeError
        If only one input carries units.
    brainunit.UnitMismatchError
        If the units have different dimensions.

    See Also
    --------
    rmse, mae

    Notes
    -----
    Removing an initial transient is left to the caller (``Fitter`` does it).
    Works under ``jax.jit`` and ``jax.vmap`` and is differentiable.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from braintools.fit import losses
        >>> float(losses.mse(jnp.array([1., 2., 3.]), jnp.array([1., 2., 5.])))
        1.3333333730697632
        >>>
        >>> float(losses.mse(jnp.ones(3) * u.mV, jnp.zeros(3) * u.volt))
        1.0
    """
    p, t = _pair(prediction, target, 'mse')
    return jnp.mean((p - t) ** 2)


@set_module_as('braintools.fit.losses')
def rmse(prediction: Any, target: Any) -> jax.Array:
    r"""Compute the root-mean-square error between two time series.

    .. math::

        \mathrm{RMSE} = \sqrt{\frac{1}{M} \sum_{i=1}^{M} (\hat y_i - y_i)^2}.

    Parameters
    ----------
    prediction : array-like or Quantity
        Model output, e.g. of shape ``(n_time, n_channels)``.
    target : array-like or Quantity
        Data of the same shape and compatible units.

    Returns
    -------
    jax.Array
        Scalar RMSE; for Quantity inputs the plain magnitude in the
        prediction's unit.

    Raises
    ------
    ValueError
        If the shapes differ.
    TypeError
        If only one input carries units.
    brainunit.UnitMismatchError
        If the units have different dimensions.

    See Also
    --------
    mse, mae, fc_rmse

    Notes
    -----
    The same numerics as ``brainmass.objectives.timeseries_rmse``. Works under
    ``jax.jit`` and ``jax.vmap`` and is differentiable; at a perfect fit the
    gradient is 0 rather than ``nan``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.fit import losses
        >>> x = jnp.zeros((10, 3))
        >>> float(losses.rmse(x + 2., x))
        2.0
    """
    p, t = _pair(prediction, target, 'rmse')
    return _safe_sqrt(jnp.mean((p - t) ** 2))


@set_module_as('braintools.fit.losses')
def mae(prediction: Any, target: Any) -> jax.Array:
    r"""Compute the mean absolute error between two time series.

    .. math::

        \mathrm{MAE} = \frac{1}{M} \sum_{i=1}^{M} |\hat y_i - y_i|.

    Parameters
    ----------
    prediction : array-like or Quantity
        Model output, e.g. of shape ``(n_time, n_channels)``.
    target : array-like or Quantity
        Data of the same shape and compatible units.

    Returns
    -------
    jax.Array
        Scalar MAE; for Quantity inputs the plain magnitude in the
        prediction's unit.

    Raises
    ------
    ValueError
        If the shapes differ.
    TypeError
        If only one input carries units.
    brainunit.UnitMismatchError
        If the units have different dimensions.

    See Also
    --------
    mse, rmse

    Notes
    -----
    Works under ``jax.jit`` and ``jax.vmap``; differentiable except where
    ``prediction == target`` (JAX uses the subgradient 0 there).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.fit import losses
        >>> float(losses.mae(jnp.array([1., -1., 4.]), jnp.zeros(3)))
        2.0
    """
    p, t = _pair(prediction, target, 'mae')
    return jnp.mean(jnp.abs(p - t))


@set_module_as('braintools.fit.losses')
def correlation_loss(
    prediction: Any,
    target: Any,
    *,
    per_channel: bool = False,
) -> jax.Array:
    r"""Compute one minus the Pearson correlation between two time series.

    .. math::

        L = 1 - r(\hat y, y), \qquad
        r = \frac{\sum_i (\hat y_i - \bar{\hat y})(y_i - \bar y)}
                 {\sqrt{\sum_i (\hat y_i - \bar{\hat y})^2 \sum_i (y_i - \bar y)^2}}.

    The loss lies in ``[0, 2]`` and is invariant to the scale and offset of
    either input.

    Parameters
    ----------
    prediction : array-like or Quantity
        Model output, time on axis 0.
    target : array-like or Quantity
        Data of the same shape. Units are not compared (the loss is
        scale-invariant); only the magnitudes are used.
    per_channel : bool, default False
        If False, correlate the flattened arrays. If True, correlate each
        column along the time axis 0 and average the correlations over the
        trailing axes.

    Returns
    -------
    jax.Array
        Scalar loss.

    Raises
    ------
    ValueError
        If the shapes differ.

    See Also
    --------
    fc_correlation_loss, mse

    Notes
    -----
    A constant input (zero variance) has an undefined correlation, which is
    taken to be 0 (loss 1), as :func:`~braintools.analysis.population.functional_connectivity`
    does; its gradient is then 0 rather than ``nan``. Works under
    ``jax.jit`` and ``jax.vmap`` and is differentiable.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.fit import losses
        >>> t = jnp.linspace(0., 6., 50)
        >>> round(float(losses.correlation_loss(3. * jnp.sin(t) + 1., jnp.sin(t))), 6)
        0.0
        >>> round(float(losses.correlation_loss(-jnp.sin(t), jnp.sin(t))), 6)
        2.0
    """
    p, t = _plain(prediction), _plain(target)
    _check_same_shape(p, t, 'correlation_loss')
    if per_channel:
        if p.ndim < 1:
            raise ValueError('correlation_loss: `per_channel=True` needs a time axis.')
        return 1. - jnp.mean(_pearson(p, t, axis=0))
    return 1. - _pearson(p.reshape(-1), t.reshape(-1))


# ----------------------------------------------------------------------------
# functional connectivity
# ----------------------------------------------------------------------------

@set_module_as('braintools.fit.losses')
def fc_correlation_loss(prediction: Any, target: Any) -> jax.Array:
    r"""Compute one minus the correlation between two functional-connectivity matrices.

    Both inputs are turned into Pearson functional-connectivity (FC) matrices
    with :func:`~braintools.analysis.population.functional_connectivity`; the
    loss is ``1 - r`` where ``r`` is the Pearson correlation of their upper
    triangles (diagonal excluded), computed by
    :func:`~braintools.analysis.stats.matrix_correlation`.

    Parameters
    ----------
    prediction : array-like or Quantity
        Model output of shape ``(n_time, n_regions)``, ``n_regions >= 2``.
    target : array-like or Quantity
        Data of shape ``(n_time', n_regions)``; the number of time points may
        differ. FC is scale-invariant, so units are not compared.

    Returns
    -------
    jax.Array
        Scalar loss in ``[0, 2]``.

    Raises
    ------
    ValueError
        If an input is not 2-D, has fewer than 2 regions, or the region
        counts differ.

    See Also
    --------
    fc_rmse, fcd_ks, fcd_wasserstein

    Notes
    -----
    Identical to ``brainmass.objectives.fc_corr(as_loss=True)``. With 2
    regions the upper triangle has one element and its correlation is
    undefined; it is taken to be 0 (loss 1). Works under ``jax.jit`` and
    ``jax.vmap`` and is differentiable.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import jax.numpy as jnp
        >>> from braintools.fit import losses
        >>> x = jnp.asarray(np.random.default_rng(0).standard_normal((200, 5)))
        >>> round(float(losses.fc_correlation_loss(x, x)), 6)
        0.0
    """
    p, t = _plain(prediction), _plain(target)
    _check_2d(p, 'prediction', 'fc_correlation_loss')
    _check_2d(t, 'target', 'fc_correlation_loss')
    if p.shape[1] != t.shape[1]:
        raise ValueError(
            f'fc_correlation_loss: `prediction` and `target` must have the same number '
            f'of regions, got {p.shape[1]} and {t.shape[1]}.'
        )
    return 1. - matrix_correlation(functional_connectivity(p), functional_connectivity(t))


@set_module_as('braintools.fit.losses')
def fc_rmse(prediction: Any, target: Any) -> jax.Array:
    r"""Compute the RMSE between the upper triangles of two functional-connectivity matrices.

    .. math::

        L = \sqrt{\frac{2}{n(n-1)} \sum_{i<j} (\hat C_{ij} - C_{ij})^2},

    where :math:`\hat C` and :math:`C` are the Pearson functional-connectivity
    matrices (:func:`~braintools.analysis.population.functional_connectivity`)
    of ``n`` regions.

    Parameters
    ----------
    prediction : array-like or Quantity
        Model output of shape ``(n_time, n_regions)``, ``n_regions >= 2``.
    target : array-like or Quantity
        Data of shape ``(n_time', n_regions)``. FC is scale-invariant, so
        units are not compared.

    Returns
    -------
    jax.Array
        Scalar loss in ``[0, 2]``.

    Raises
    ------
    ValueError
        If an input is not 2-D, has fewer than 2 regions, or the region
        counts differ.

    See Also
    --------
    fc_correlation_loss, rmse

    Notes
    -----
    Only the ``n(n-1)/2`` off-diagonal pairs enter the mean, as the 0.6
    specification requires. This **differs from**
    ``brainmass.objectives.fc_rmse``, which averages over the full ``n x n``
    matrix, including the diagonal (always equal, so it adds zeros to the
    mean). The two are related by

    .. math::

        L_{\mathrm{brainmass}} = L \sqrt{\frac{n - 1}{n}},
        \qquad L = L_{\mathrm{brainmass}} \sqrt{\frac{n}{n - 1}},

    so brainmass values are smaller by the factor :math:`\sqrt{(n-1)/n}`
    (0.926 for :math:`n = 7`, e.g. 0.0869 versus 0.0939). Code ported from
    brainmass must rescale thresholds or weights by that factor. Works under
    ``jax.jit`` and ``jax.vmap`` and is differentiable (gradient 0 at a
    perfect fit).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.fit import losses
        >>> t = jnp.linspace(0., 10., 200)
        >>> a = jnp.stack([jnp.sin(t), jnp.sin(t), jnp.cos(t)], axis=1)
        >>> b = jnp.stack([jnp.sin(t), -jnp.sin(t), jnp.cos(t)], axis=1)
        >>> round(float(losses.fc_rmse(a, b)), 4)
        1.1563
    """
    p, t = _plain(prediction), _plain(target)
    _check_2d(p, 'prediction', 'fc_rmse')
    _check_2d(t, 'target', 'fc_rmse')
    n = p.shape[1]
    if t.shape[1] != n:
        raise ValueError(
            f'fc_rmse: `prediction` and `target` must have the same number of regions, '
            f'got {n} and {t.shape[1]}.'
        )
    iu = np.triu_indices(n, k=1)
    diff = functional_connectivity(p)[iu] - functional_connectivity(t)[iu]
    return _safe_sqrt(jnp.mean(diff ** 2))


# ----------------------------------------------------------------------------
# 1-D distribution distances and FCD (ported from brainmass.objectives)
# ----------------------------------------------------------------------------

def _default_midpoints() -> jax.Array:
    # Same grid as brainmass.objectives._DEFAULT_FCD_MIDPOINTS: 100 points on
    # [-0.99, 0.99] (the tvboptim convention). Built lazily, not at import.
    return jnp.linspace(-0.99, 0.99, 100)


def _distribution_pair(p: Any, q: Any, where: str) -> Tuple[jax.Array, jax.Array]:
    """Magnitudes of two densities on a shared grid (1-D, same length)."""
    p, q = _pair(p, q, where)
    if p.ndim != 1:
        raise ValueError(f'{where}: `p` and `q` must be 1-D, got shape {p.shape}.')
    return p, q


def _cdfs(p: jax.Array, q: jax.Array) -> Tuple[jax.Array, jax.Array]:
    # Ported from brainmass.objectives (Apache-2.0): each cumulative sum is
    # normalised by its last entry so that both CDFs run from ~0 to 1.
    p_cdf = jnp.cumsum(p)
    q_cdf = jnp.cumsum(q)
    return p_cdf / p_cdf[-1], q_cdf / q_cdf[-1]


@set_module_as('braintools.fit.losses')
def ks_distance(p: Any, q: Any) -> jax.Array:
    r"""Compute the Kolmogorov-Smirnov statistic between two densities on a shared grid.

    Each cumulative sum is normalised to a proper CDF before the supremum is
    taken,

    .. math::

        D = \max_k \left| F_p(x_k) - F_q(x_k) \right|,

    so the result lies in ``[0, 1]`` independently of the bin width, and is
    comparable with :func:`scipy.stats.ks_2samp`.

    Parameters
    ----------
    p : array-like or Quantity
        1-D density or (unnormalised) histogram on an ordered grid.
    q : array-like or Quantity
        Density on the same grid (same length, compatible units).

    Returns
    -------
    jax.Array
        Scalar KS statistic in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``p`` is not 1-D or the shapes differ.

    See Also
    --------
    wasserstein_1d : The smooth counterpart, better suited to gradient fitting.
    fcd_ks : KS distance between two FCD distributions.

    Notes
    -----
    Ported from ``brainmass.objectives.ks_distance`` with identical numerics.
    The ``max`` makes the statistic non-smooth: its gradient is non-zero only
    at the arg-max. Prefer :func:`wasserstein_1d` as a fitting loss and use KS
    for evaluation. Works under ``jax.jit`` and ``jax.vmap``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.fit import losses
        >>> float(losses.ks_distance(jnp.array([1., 0., 0., 0.]), jnp.array([0., 0., 0., 1.])))
        1.0
        >>> float(losses.ks_distance(jnp.array([1., 1., 0.]), jnp.array([2., 2., 0.])))
        0.0
    """
    p, q = _distribution_pair(p, q, 'ks_distance')
    p_cdf, q_cdf = _cdfs(p, q)
    return jnp.max(jnp.abs(p_cdf - q_cdf))


@set_module_as('braintools.fit.losses')
def wasserstein_1d(p: Any, q: Any, x: Any) -> Any:
    r"""Compute the Wasserstein-1 distance between two densities on a shared uniform grid.

    .. math::

        W_1(p, q) = \int \left| F_p(x) - F_q(x) \right| \, dx
        \approx \Delta x \sum_k \left| F_p(x_k) - F_q(x_k) \right|,

    where :math:`F_p` and :math:`F_q` are the cumulative sums of ``p`` and
    ``q`` normalised to end at 1, so unnormalised histograms are accepted.

    Parameters
    ----------
    p : array-like or Quantity
        1-D density or histogram on the grid ``x``.
    q : array-like or Quantity
        Density on the same grid (same length, compatible units).
    x : array-like or Quantity
        Uniform evaluation grid of the same length, at least 2 points. Only
        its spacing ``x[1] - x[0]`` is used.

    Returns
    -------
    jax.Array or Quantity
        Scalar distance, in the units of ``x`` (a Quantity if ``x`` is one).

    Raises
    ------
    ValueError
        If ``p`` is not 1-D, the shapes differ, or ``x`` has fewer than 2
        points.

    See Also
    --------
    ks_distance : The non-smooth counterpart.
    fcd_wasserstein : Wasserstein distance between two FCD distributions.

    Notes
    -----
    Ported from ``brainmass.objectives.wasserstein_1d`` with identical
    numerics. It converges to :func:`scipy.stats.wasserstein_distance` as the
    grid is refined. Works under ``jax.jit`` and ``jax.vmap`` and is
    differentiable in ``p`` and ``q``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.fit import losses
        >>> x = jnp.arange(5.)
        >>> p = jnp.array([1., 0., 0., 0., 0.])
        >>> q = jnp.array([0., 0., 1., 0., 0.])
        >>> float(losses.wasserstein_1d(p, q, x))
        2.0
    """
    p, q = _distribution_pair(p, q, 'wasserstein_1d')
    x_shape = np.shape(u.get_magnitude(x))
    if len(x_shape) != 1 or x_shape[0] < 2:
        raise ValueError(f'wasserstein_1d: `x` must be 1-D with at least 2 points, got shape {x_shape}.')
    dx = x[1] - x[0]
    p_cdf, q_cdf = _cdfs(p, q)
    return jnp.sum(jnp.abs(p_cdf - q_cdf)) * dx


#: Largest number of (FCD value, midpoint) kernel terms evaluated at once by
#: the FCD KDE (1 Mi elements = 4 MiB in float32). Above it the midpoints are
#: processed in blocks, which bounds the temporaries of long series.
_KDE_BLOCK_ELEMENTS = 2 ** 20


def _kde_evaluate(kde, midpoints):
    """``kde.evaluate(midpoints)`` for 1-D data, in blocks of midpoints.

    Same arithmetic as ``jax.scipy.stats.gaussian_kde.evaluate`` (whitening by
    the Cholesky factor of the precision, then a weighted sum of Gaussian
    kernels over the data for every midpoint), but the ``(n_data, n_points)``
    kernel matrix is never materialised whole: at most
    ``_KDE_BLOCK_ELEMENTS`` terms exist at a time, and each block is
    rematerialised on the backward pass instead of being stored.
    """
    data, weights = kde.dataset[0], kde.weights
    whitening = jnp.linalg.cholesky(kde.inv_cov)
    x_train = jnp.dot(kde.dataset.T, whitening)[:, 0]
    x_test = jnp.dot(midpoints[:, None].astype(x_train.dtype), whitening)[:, 0]
    log_norm = jnp.sum(jnp.log(jnp.diag(whitening))) - 0.5 * jnp.log(2 * np.pi)

    def at(x):
        return jnp.sum(weights * jnp.exp(log_norm - 0.5 * jnp.square(x_train - x)))

    n_data, n_points = data.shape[0], x_test.shape[0]
    block = max(1, _KDE_BLOCK_ELEMENTS // max(1, n_data))
    if block >= n_points:
        return jax.vmap(at)(x_test)
    return jax.lax.map(jax.checkpoint(at), x_test, batch_size=block)


def _fcd_density(fcd_matrix, midpoints, n_diag, bw_method):
    # Ported from brainmass.objectives.fcd_distribution (Apache-2.0):
    # Gaussian KDE of the FCD upper-triangle values, normalised to integrate
    # to 1 over ``midpoints``. The evaluation is blocked over midpoints to
    # bound memory (see _kde_evaluate); the numerics are unchanged.
    vals = fcd_matrix[jnp.triu_indices(fcd_matrix.shape[0], k=n_diag)]
    kde = jax.scipy.stats.gaussian_kde(vals, bw_method=bw_method)
    density = _kde_evaluate(kde, midpoints)
    dx = midpoints[1] - midpoints[0]
    return density / (jnp.sum(density) * dx)


def _fcd_densities(prediction, target, window_size, step_size, midpoints, bw_method,
                   n_diag, where):
    p, t = _plain(prediction), _plain(target)
    _check_2d(p, 'prediction', where)
    _check_2d(t, 'target', where)
    if midpoints is None:
        midpoints = _default_midpoints()
    midpoints = jnp.asarray(midpoints)
    out = []
    for x, name in ((p, 'prediction'), (t, 'target')):
        fcd = functional_connectivity_dynamics(x, window_size=window_size, step_size=step_size)
        n_win = fcd.shape[0]
        n_vals = (n_win - n_diag) * (n_win - n_diag + 1) // 2 if n_win > n_diag else 0
        if n_vals < 2:
            raise ValueError(
                f'{where}: the FCD of `{name}` has {n_win} windows and so {n_vals} values '
                f'above diagonal offset {n_diag}; the KDE needs at least 2. Use a longer '
                f'series or a smaller `window_size`/`step_size`.'
            )
        out.append(_fcd_density(fcd, midpoints, n_diag, bw_method))
    return out[0], out[1], midpoints


@set_module_as('braintools.fit.losses')
def fcd_ks(
    prediction: Any,
    target: Any,
    *,
    window_size: int = 30,
    step_size: int = 5,
    midpoints: Optional[Any] = None,
    bw_method: Optional[Any] = None,
    n_diag: int = 1,
) -> jax.Array:
    r"""Compute the Kolmogorov-Smirnov distance between two FCD value distributions.

    The functional-connectivity-dynamics (FCD) matrix of each input is
    computed with
    :func:`~braintools.analysis.population.functional_connectivity_dynamics`;
    the values above the ``n_diag``-th diagonal are smoothed into a density
    on ``midpoints`` by a Gaussian kernel density estimate
    (:class:`jax.scipy.stats.gaussian_kde`), and the two densities are
    compared with :func:`ks_distance`. Comparing FCD *distributions* (rather
    than the matrices) is the standard FCD fitting target [1]_.

    Parameters
    ----------
    prediction : array-like or Quantity
        Model output of shape ``(n_time, n_regions)``, ``n_regions >= 2``.
    target : array-like or Quantity
        Data of shape ``(n_time', n_regions')``. FCD is scale-invariant, so
        units are not compared.
    window_size : int, default 30
        Sliding-window length in samples (static).
    step_size : int, default 5
        Sliding-window stride in samples (static).
    midpoints : array-like, optional
        Grid of FCD values on which the densities are evaluated. Defaults to
        100 points on ``[-0.99, 0.99]``.
    bw_method : str, float or callable, optional
        KDE bandwidth, forwarded to :class:`jax.scipy.stats.gaussian_kde`
        (default: Scott's rule).
    n_diag : int, default 1
        Diagonal offset of the upper triangle; 1 excludes the unit diagonal.

    Returns
    -------
    jax.Array
        Scalar KS distance in ``[0, 1]``.

    Raises
    ------
    ValueError
        If an input is not 2-D with at least 2 regions, or its FCD has fewer
        than 2 values above the diagonal offset.

    See Also
    --------
    fcd_wasserstein : The smooth counterpart, preferred for gradient fitting.
    ks_distance

    Notes
    -----
    Ported from ``brainmass.objectives.fcd_ks(...)`` (a factory there) with
    identical numerics. The KS statistic is non-smooth; use it for evaluation
    and :func:`fcd_wasserstein` as a gradient-fitting loss. If all FCD values
    of an input are equal, the KDE is degenerate and the result is ``nan``.
    Works under ``jax.jit`` and ``jax.vmap`` with static window options.

    **Memory.** The FCD matrix of ``W`` windows has ``W**2`` entries and the
    KDE sums ``W(W-1)/2`` kernel terms for each of the ``M`` midpoints. The
    ``W(W-1)/2 x M`` kernel matrix is never formed whole: midpoints are
    processed in blocks of at most about 2**20 terms (4 MiB in float32), and
    each block is recomputed on the backward pass rather than stored, so
    temporaries grow as ``O(W**2)`` rather than ``O(W**2 M)``. For example,
    20 regions, ``T = 5000``, ``window_size=30``, ``step_size=5`` (995
    windows) need about 24 MiB of temporaries forward (previously 192
    MiB). Under ``vmap`` (e.g. over the starts of a fit) the
    temporaries scale with the batch size.

    References
    ----------
    .. [1] Hansen, E. C. A., Battaglia, D., Spiegler, A., Deco, G., & Jirsa,
           V. K. (2015). Functional connectivity dynamics: modeling the
           switching behavior of the resting state. NeuroImage, 105, 525-535.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import jax.numpy as jnp
        >>> from braintools.fit import losses
        >>> x = jnp.asarray(np.random.default_rng(0).standard_normal((200, 6)))
        >>> float(losses.fcd_ks(x, x, window_size=30, step_size=5))
        0.0
    """
    dp, dq, _ = _fcd_densities(prediction, target, window_size, step_size, midpoints,
                               bw_method, n_diag, 'fcd_ks')
    return ks_distance(dp, dq)


@set_module_as('braintools.fit.losses')
def fcd_wasserstein(
    prediction: Any,
    target: Any,
    *,
    window_size: int = 30,
    step_size: int = 5,
    midpoints: Optional[Any] = None,
    bw_method: Optional[Any] = None,
    n_diag: int = 1,
) -> jax.Array:
    r"""Compute the Wasserstein-1 distance between two FCD value distributions.

    Like :func:`fcd_ks`, but the two kernel density estimates of the FCD
    values are compared with :func:`wasserstein_1d` on ``midpoints``, which
    is smooth in the inputs and so the recommended FCD loss for gradient
    fitting [1]_.

    Parameters
    ----------
    prediction : array-like or Quantity
        Model output of shape ``(n_time, n_regions)``, ``n_regions >= 2``.
    target : array-like or Quantity
        Data of shape ``(n_time', n_regions')``. FCD is scale-invariant, so
        units are not compared.
    window_size : int, default 30
        Sliding-window length in samples (static).
    step_size : int, default 5
        Sliding-window stride in samples (static).
    midpoints : array-like, optional
        Uniform grid of FCD values on which the densities are evaluated.
        Defaults to 100 points on ``[-0.99, 0.99]``.
    bw_method : str, float or callable, optional
        KDE bandwidth, forwarded to :class:`jax.scipy.stats.gaussian_kde`
        (default: Scott's rule).
    n_diag : int, default 1
        Diagonal offset of the upper triangle; 1 excludes the unit diagonal.

    Returns
    -------
    jax.Array
        Scalar distance (in FCD-correlation units).

    Raises
    ------
    ValueError
        If an input is not 2-D with at least 2 regions, or its FCD has fewer
        than 2 values above the diagonal offset.

    See Also
    --------
    fcd_ks, wasserstein_1d

    Notes
    -----
    Ported from ``brainmass.objectives.fcd_wasserstein(...)`` (a factory
    there) with identical numerics. If all FCD values of an input are equal,
    the KDE is degenerate and the result is ``nan``. Works under ``jax.jit``
    and ``jax.vmap`` with static window options, and is differentiable.

    **Memory.** The FCD matrix of ``W`` windows has ``W**2`` entries and the
    KDE sums ``W(W-1)/2`` kernel terms for each of the ``M`` midpoints. The
    ``W(W-1)/2 x M`` kernel matrix is never formed whole: midpoints are
    processed in blocks of at most about 2**20 terms (4 MiB in float32), and
    each block is recomputed on the backward pass rather than stored, so
    temporaries grow as ``O(W**2)`` rather than ``O(W**2 M)``. For example,
    20 regions, ``T = 5000``, ``window_size=30``, ``step_size=5`` (995
    windows) need about 24 MiB of temporaries forward and 36 MiB for the gradient (previously 192
    MiB and 588 MiB). Under ``vmap`` (e.g. over the starts of a fit) the
    temporaries scale with the batch size.

    References
    ----------
    .. [1] Hansen, E. C. A., Battaglia, D., Spiegler, A., Deco, G., & Jirsa,
           V. K. (2015). Functional connectivity dynamics: modeling the
           switching behavior of the resting state. NeuroImage, 105, 525-535.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import jax.numpy as jnp
        >>> from braintools.fit import losses
        >>> x = jnp.asarray(np.random.default_rng(0).standard_normal((200, 6)))
        >>> float(losses.fcd_wasserstein(x, x, window_size=30, step_size=5))
        0.0
    """
    dp, dq, mid = _fcd_densities(prediction, target, window_size, step_size, midpoints,
                                 bw_method, n_diag, 'fcd_wasserstein')
    return wasserstein_1d(dp, dq, mid)


# ----------------------------------------------------------------------------
# spectra
# ----------------------------------------------------------------------------

def _seconds(dt: Any) -> float:
    if isinstance(dt, u.Quantity):
        try:
            return float(dt.to_decimal(u.second))
        except Exception as e:
            raise TypeError(f'`dt` must be a time, got {dt!r}.') from e
    return float(dt)


def _hz(f: Any) -> float:
    if isinstance(f, u.Quantity):
        try:
            return float(f.to_decimal(u.Hz))
        except Exception as e:
            raise TypeError(f'`band` must hold frequencies, got {f!r}.') from e
    return float(f)


@set_module_as('braintools.fit.losses')
def psd_log_distance(
    prediction: Any,
    target: Any,
    dt: Any,
    nperseg: Optional[int] = None,
    band: Optional[Tuple[Any, Any]] = None,
    *,
    noverlap: Optional[int] = None,
    eps: Optional[float] = None,
) -> jax.Array:
    r"""Compute the mean squared difference of the log10 power spectra of two signals.

    .. math::

        L = \frac{1}{|F| C} \sum_{f \in F} \sum_{c=1}^{C}
            \left[\log_{10}(\hat P_c(f) + \epsilon) - \log_{10}(P_c(f) + \epsilon)\right]^2,

    where :math:`\hat P` and :math:`P` are Welch power spectral densities from
    :func:`~braintools.analysis.spectral.power_spectral_density` (Hann
    window), :math:`F` is the set of frequencies (optionally restricted to
    ``band``) and :math:`C` the number of channels.

    Parameters
    ----------
    prediction : array-like or Quantity
        Signal of shape ``(n_time,)`` or ``(n_time, n_channels)``.
    target : array-like or Quantity
        Signal of the same shape. If one input carries units, both must, with
        the same dimension; the target is converted to the prediction's unit.
    dt : float or Quantity
        Sampling interval, static. A plain number is in **seconds** (the
        continuous-signal convention of :mod:`braintools.analysis.spectral`),
        *not* milliseconds as in :func:`rate_loss` and
        :func:`van_rossum_loss`; pass a Quantity (e.g. ``0.1 * u.ms``) to
        avoid ambiguity.
    nperseg : int, optional
        Welch segment length (static). Default ``n_time // 8``.
    band : tuple of (float or Quantity), optional
        ``(f_low, f_high)``; only frequencies with ``f_low <= f <= f_high``
        enter the mean. Plain numbers are in Hz.
    noverlap : int, optional
        Overlap between segments (static). Default ``nperseg // 2``.
    eps : float, optional
        Floor added to the PSDs before the logarithm. Defaults to the
        smallest positive normal number of the dtype.

    Returns
    -------
    jax.Array
        Scalar loss in squared decades.

    Raises
    ------
    ValueError
        If the shapes differ, or ``band`` contains no frequency of the Welch
        grid.
    TypeError
        If only one input carries units, or ``dt``/``band`` have the wrong
        dimension.

    See Also
    --------
    braintools.analysis.spectral.power_spectral_density

    Notes
    -----
    The frequency grid ``rfftfreq(nperseg, dt)`` is computed on the host, so
    the band selection is a static index and the loss works under
    ``jax.jit`` and ``jax.vmap``. It is differentiable with respect to the
    signals. An overall gain changes every log-PSD by the same amount, so the
    loss is sensitive to amplitude as well as to spectral shape [1]_.

    References
    ----------
    .. [1] Welch, P. (1967). The use of fast Fourier transform for the
           estimation of power spectra: a method based on time averaging over
           short, modified periodograms. IEEE Transactions on Audio and
           Electroacoustics, 15(2), 70-73.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from braintools.fit import losses
        >>> t = jnp.arange(1000) * 1e-3
        >>> x = jnp.sin(2 * jnp.pi * 10 * t)
        >>> float(losses.psd_log_distance(x, x, dt=1. * u.ms, nperseg=200))
        0.0
        >>> # Doubling the amplitude raises the PSD by log10(4) everywhere.
        >>> round(float(losses.psd_log_distance(2 * x, x, 1e-3, 200, band=(5., 50.))), 4)
        0.3625
    """
    p, t = _pair(prediction, target, 'psd_log_distance')
    if p.ndim not in (1, 2):
        raise ValueError(
            f'psd_log_distance: signals must have shape (n_time,) or (n_time, n_channels), '
            f'got {p.shape}.'
        )
    dt_s = _seconds(dt)
    n_time = p.shape[0]
    # Same segment length rule as power_spectral_density, to build the static grid.
    seg = max(1, n_time // 8) if nperseg is None else int(nperseg)
    seg = int(min(seg, n_time))
    _, psd_p = power_spectral_density(p, dt_s, nperseg=seg, noverlap=noverlap)
    _, psd_t = power_spectral_density(t, dt_s, nperseg=seg, noverlap=noverlap)
    if band is not None:
        lo, hi = _hz(band[0]), _hz(band[1])
        freqs = np.fft.rfftfreq(seg, dt_s)
        idx = np.nonzero((freqs >= lo) & (freqs <= hi))[0]
        if idx.size == 0:
            raise ValueError(
                f'psd_log_distance: band ({lo}, {hi}) Hz contains none of the Welch '
                f'frequencies (resolution {freqs[1] - freqs[0] if freqs.size > 1 else 0.} Hz, '
                f'max {freqs[-1]} Hz).'
            )
        psd_p, psd_t = psd_p[idx], psd_t[idx]
    if eps is None:
        eps = float(jnp.finfo(psd_p.dtype).tiny)
    return jnp.mean((jnp.log10(psd_p + eps) - jnp.log10(psd_t + eps)) ** 2)


# ----------------------------------------------------------------------------
# spikes
# ----------------------------------------------------------------------------

def _rates_hz(x: Any, dt: Any, where: str, name: str) -> jax.Array:
    """Per-neuron mean firing rates in Hz from a raster or a SpikeTrains."""
    if isinstance(x, SpikeTrains):
        unit = x._unit
        sec_per = float(u.Quantity(1., unit=u.ms if unit is None else unit).to_decimal(u.second))
        duration = (x._t_stop_m - x._t_start_m) * sec_per
        return x.counts.astype(duration.dtype) / duration
    raster = _as_raster(x, where, name)
    dt_s = _dt_ms(dt, where) * 1e-3
    return jnp.sum(raster, axis=0) / (raster.shape[0] * dt_s)


@set_module_as('braintools.fit.losses')
def rate_loss(
    prediction: Any,
    target: Any,
    dt: Any = None,
    *,
    per_neuron: bool = False,
) -> jax.Array:
    r"""Compute the squared difference of mean firing rates (in Hz).

    Each input is a dense spike raster of shape ``(T,)`` or ``(T, N)`` (time
    on axis 0) or a :class:`~braintools.spike.SpikeTrains`. The rate of a
    neuron is its spike count divided by the recording length: ``T * dt``
    for a raster, ``t_stop - t_start`` for a ``SpikeTrains``. The loss is

    .. math::

        L = (\bar r_{\mathrm{pred}} - \bar r_{\mathrm{target}})^2,

    the squared difference of the population-mean rates, or with
    ``per_neuron=True``

    .. math::

        L = \frac{1}{N} \sum_{n=1}^{N} (r_{\mathrm{pred},n} - r_{\mathrm{target},n})^2.

    Parameters
    ----------
    prediction : array-like or SpikeTrains
        Model spikes. Raster values may be real (e.g. surrogate-gradient
        spikes); the count is their sum.
    target : array-like or SpikeTrains
        Data spikes. Rasters may have different ``T`` from the prediction.
    dt : float or Quantity, optional
        Raster time step. A plain number is in **milliseconds** (the
        spike-time convention of :mod:`braintools.analysis.spike` and
        ``brainstate``), *not* seconds as in :func:`psd_log_distance`; pass a
        Quantity (e.g. ``0.1 * u.ms``) to avoid ambiguity. Defaults to
        ``brainstate.environ.get_dt()``. Not used for ``SpikeTrains`` inputs.
    per_neuron : bool, default False
        If False, compare the population-mean rates. If True, average the
        squared per-neuron rate differences (both inputs need the same ``N``).

    Returns
    -------
    jax.Array
        Scalar loss in Hz².

    Raises
    ------
    ValueError
        If a raster is not 1-D or 2-D, ``dt`` is missing for a raster, or
        ``per_neuron=True`` with different neuron counts.
    TypeError
        If a raster carries units or ``dt`` is not a time.

    See Also
    --------
    cv_loss, van_rossum_loss, braintools.analysis.spike.firing_rate

    Notes
    -----
    Works under ``jax.jit`` and ``jax.vmap`` (``dt`` may be traced) and is
    differentiable with respect to raster values, so it trains spiking
    networks through surrogate spikes. A ``SpikeTrains`` rate does not
    depend on the spike times, so it carries no gradient.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from braintools.fit import losses
        >>> from braintools.spike import SpikeTrains
        >>> pred = jnp.zeros((1000, 4)).at[::100].set(1.)   # 10 Hz at dt = 1 ms
        >>> target = jnp.zeros((1000, 4)).at[::50].set(1.)  # 20 Hz
        >>> float(losses.rate_loss(pred, target, dt=1. * u.ms))
        100.0
        >>> st = SpikeTrains.from_dense(target, dt=1. * u.ms)
        >>> float(losses.rate_loss(pred, st, dt=1.))
        100.0
    """
    rp = _rates_hz(prediction, dt, 'rate_loss', 'prediction')
    rt = _rates_hz(target, dt, 'rate_loss', 'target')
    if per_neuron:
        if rp.shape != rt.shape:
            raise ValueError(
                f'rate_loss: `per_neuron=True` needs the same number of neurons, got '
                f'{rp.shape[0]} and {rt.shape[0]}.'
            )
        return jnp.mean((rp - rt) ** 2)
    return (jnp.mean(rp) - jnp.mean(rt)) ** 2


def _mean_cv(trains: Any, name: str) -> jax.Array:
    if not isinstance(trains, SpikeTrains):
        raise TypeError(
            f'cv_loss: `{name}` must be a braintools.spike.SpikeTrains, got '
            f'{type(trains).__name__}. Build one from a dense (T, N) raster with '
            f'SpikeTrains.from_dense(spikes, dt=...).'
        )
    c = _cv(trains)
    ok = ~jnp.isnan(c)
    n = jnp.sum(ok)
    total = jnp.sum(jnp.where(ok, c, 0))
    return jnp.where(n > 0, total / jnp.where(n > 0, n, 1), jnp.nan)


@set_module_as('braintools.fit.losses')
def cv_loss(prediction: Any, target: Any) -> jax.Array:
    r"""Compute the squared difference of the mean inter-spike-interval CV.

    .. math::

        L = \left(\langle \mathrm{CV} \rangle_{\mathrm{pred}}
                  - \langle \mathrm{CV} \rangle_{\mathrm{target}}\right)^2,

    where :math:`\langle \mathrm{CV} \rangle` averages
    :func:`~braintools.analysis.spike.cv` over the neurons whose CV is
    defined (at least 3 spikes); neurons with a ``nan`` CV are ignored [1]_.

    Parameters
    ----------
    prediction : SpikeTrains
        Model spikes.
    target : SpikeTrains
        Data spikes; the neuron counts may differ.

    Returns
    -------
    jax.Array
        Scalar loss (dimensionless). ``nan`` if every neuron of either input
        has an undefined CV.

    Raises
    ------
    TypeError
        If an input is not a :class:`~braintools.spike.SpikeTrains`.

    See Also
    --------
    rate_loss, braintools.analysis.spike.cv

    Notes
    -----
    Works under ``jax.jit`` and ``jax.vmap`` and is differentiable with
    respect to the spike times; the ignored neurons contribute neither value
    nor ``nan`` gradient.

    References
    ----------
    .. [1] Softky, W. R., & Koch, C. (1993). The highly irregular firing of
           cortical cells is inconsistent with temporal integration of random
           EPSPs. Journal of Neuroscience, 13(1), 334-350.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.fit import losses
        >>> from braintools.spike import SpikeTrains
        >>> regular = jnp.zeros((100, 3)).at[::10].set(1.)    # CV 0
        >>> bursty = jnp.zeros((100, 3)).at[jnp.array([0, 2, 6]), :2].set(1.)
        >>> pred = SpikeTrains.from_dense(regular, dt=1.)
        >>> target = SpikeTrains.from_dense(bursty, dt=1.)    # CV 1/3; neuron 2 is silent
        >>> round(float(losses.cv_loss(pred, target)), 6)
        0.111111
    """
    return (_mean_cv(prediction, 'prediction') - _mean_cv(target, 'target')) ** 2


@set_module_as('braintools.fit.losses')
def van_rossum_loss(
    prediction: Any,
    target: Any,
    tau: Any,
    dt: Any = None,
    *,
    reduction: str = 'mean',
    squared: bool = False,
) -> jax.Array:
    r"""Compute a differentiable van Rossum distance between two dense spike rasters.

    The difference of the two rasters is filtered by the causal exponential
    kernel :math:`K(t) = \frac{1}{\tau} e^{-t/\tau} H(t)` with a
    ``scan`` over time,

    .. math::

        d_k = e^{-\Delta t/\tau}\, d_{k-1} + \frac{\hat s_k - s_k}{\tau},

    which is the exact filtered difference just after step :math:`k`. Between
    grid points it decays exponentially, so its squared integral over
    :math:`[0, T\Delta t]` is exactly

    .. math::

        D^2 = \frac{\tau}{2}\left(1 - e^{-2\Delta t/\tau}\right) \sum_k d_k^2,

    and the distance of each neuron is :math:`D = \sqrt{D^2}` [1]_.

    Parameters
    ----------
    prediction : array-like
        Raster of shape ``(T,)`` or ``(T, N)``, time on axis 0. Values may be
        real, e.g. spikes from a surrogate-gradient function such as
        :func:`braintools.surrogate.sigmoid`.
    target : array-like
        Raster of the same shape.
    tau : float or Quantity
        Kernel time constant: a Quantity, or a plain number in
        **milliseconds**. May be traced (and differentiated).
    dt : float or Quantity, optional
        Raster time step. A plain number is in **milliseconds** (the
        spike-time convention of :mod:`braintools.analysis.spike` and
        ``brainstate``, same as ``tau``), *not* seconds as in
        :func:`psd_log_distance`; pass a Quantity (e.g. ``0.1 * u.ms``) to
        avoid ambiguity. Defaults to ``brainstate.environ.get_dt()``.
    reduction : {'mean', 'sum', 'none'}, default 'mean'
        How the per-neuron distances are combined: their mean, their sum, or
        no reduction (an ``(N,)`` array; ``()`` for a 1-D raster).
    squared : bool, default False
        Return :math:`D^2` instead of :math:`D` per neuron (smooth at
        :math:`D = 0`).

    Returns
    -------
    jax.Array
        The reduced distance, in :math:`\mathrm{ms}^{-1/2}` (``D²`` in
        :math:`\mathrm{ms}^{-1}`), the convention of
        :func:`~braintools.analysis.spike.van_rossum_distance` with times in ms.

    Raises
    ------
    ValueError
        If a raster is not 1-D or 2-D, the shapes differ, ``dt`` is missing,
        or ``reduction`` is unknown.
    TypeError
        If a raster carries units, or ``tau``/``dt`` are not times.

    See Also
    --------
    braintools.analysis.spike.van_rossum_distance : Event-based closed form.
    rate_loss

    Notes
    -----
    For spikes on the grid (spike ``k`` at time :math:`k\,\Delta t`) the
    result equals ``van_rossum_distance(times_1, times_2, tau, t_max=T*dt)``
    up to float round-off: the only discretisation is that spike times are
    snapped to the grid. Works under ``jax.jit`` and ``jax.vmap`` and is
    differentiable with respect to the raster values and ``tau``; at
    :math:`D = 0` the gradient of :math:`D` is taken to be 0.

    References
    ----------
    .. [1] van Rossum, M. C. W. (2001). A novel spike distance. Neural
           Computation, 13(4), 751-763.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from braintools.analysis.spike import van_rossum_distance
        >>> from braintools.fit import losses
        >>> a = jnp.zeros(100).at[jnp.array([10, 30, 50])].set(1.)
        >>> b = jnp.zeros(100).at[jnp.array([12, 32, 52])].set(1.)
        >>> d = losses.van_rossum_loss(a, b, tau=5. * u.ms, dt=0.1 * u.ms)
        >>> round(float(d), 4)
        0.1512
        >>> ref = van_rossum_distance(jnp.array([1., 3., 5.]), jnp.array([1.2, 3.2, 5.2]),
        ...                           tau=5., t_max=10.)
        >>> round(float(ref), 4)
        0.1512
    """
    if reduction not in ('mean', 'sum', 'none'):
        raise ValueError(
            f"van_rossum_loss: `reduction` must be 'mean', 'sum' or 'none', got {reduction!r}."
        )
    vector = np.ndim(u.get_magnitude(prediction)) == 1
    p = _as_raster(prediction, 'van_rossum_loss', 'prediction')
    t = _as_raster(target, 'van_rossum_loss', 'target')
    _check_same_shape(p, t, 'van_rossum_loss')
    tau_ms = jnp.asarray(_time_ms(tau, 'tau'), dtype=p.dtype)
    dt_ms = jnp.asarray(_dt_ms(dt, 'van_rossum_loss'), dtype=p.dtype)
    decay = jnp.exp(-dt_ms / tau_ms)

    def step(d, s):
        d = decay * d + s / tau_ms
        return d, d * d

    _, d2 = jax.lax.scan(step, jnp.zeros(p.shape[1], p.dtype), p - t)
    weight = 0.5 * tau_ms * (-jnp.expm1(-2. * dt_ms / tau_ms))
    dist = weight * jnp.sum(d2, axis=0)
    if not squared:
        dist = _safe_sqrt(dist)
    if reduction == 'mean':
        return jnp.mean(dist)
    if reduction == 'sum':
        return jnp.sum(dist)
    return dist[0] if vector else dist
