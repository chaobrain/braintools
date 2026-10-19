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

"""Population dynamics: phase synchrony, Granger causality and criticality."""

import operator
from typing import Any, NamedTuple, Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braintools._misc import set_module_as

__all__ = [
    'kuramoto_order_parameter',
    'granger_causality',
    'AvalancheResult',
    'avalanches',
    'branching_ratio',
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _static_int(value: Any, name: str, minimum: int) -> int:
    """Validate a static Python integer that fixes an array shape."""
    if isinstance(value, bool):
        raise TypeError(f'`{name}` must be an integer, got {value!r}.')
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(
            f'`{name}` must be a static Python integer (it fixes an array shape), '
            f'got {value!r}.'
        ) from None
    if value < minimum:
        raise ValueError(f'`{name}` must be >= {minimum}, got {value}.')
    return value


def _dimensionless(x: Any, name: str) -> jax.Array:
    """Return ``x`` as a plain array; Quantities must be dimensionless."""
    if isinstance(x, u.Quantity):
        if not x.dim.is_dimensionless:
            raise TypeError(
                f'`{name}` must be dimensionless (counts or radians), got unit {x.unit}.'
            )
        return jnp.asarray(x.to_decimal(u.UNITLESS))
    return jnp.asarray(x)


def _as_float(x: jax.Array) -> jax.Array:
    if not jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(jnp.result_type(float))
    return x


# ----------------------------------------------------------------------------
# synchrony
# ----------------------------------------------------------------------------

@set_module_as('braintools.analysis.population')
def kuramoto_order_parameter(phases: Any, axis: int = -1, return_phase: bool = False):
    r"""Compute the Kuramoto order parameter of a population of oscillators.

    The complex order parameter is the mean phasor of the oscillators,

    .. math::

        R\, e^{i\psi} = \frac{1}{N} \sum_{k=1}^{N} e^{i\theta_k},

    computed along ``axis`` (the oscillator axis) [1]_. :math:`R \in [0, 1]`
    is 1 for perfectly phase-locked oscillators and close to 0 for
    incoherent ones.

    Parameters
    ----------
    phases : array-like or Quantity
        Phases in radians, for example ``(T, N)`` for ``N`` oscillators over
        ``T`` time steps. A Quantity must be dimensionless (radians or
        degrees; degrees are converted to radians).
    axis : int, default -1
        Axis of the oscillators, which is averaged out.
    return_phase : bool, default False
        If True, also return the mean phase :math:`\psi`.

    Returns
    -------
    R : jax.Array
        Order parameter, ``phases`` with ``axis`` removed (``(T,)`` for
        ``(T, N)`` input).
    psi : jax.Array
        Mean phase in ``(-pi, pi]``, same shape as ``R``. Only returned
        when ``return_phase=True``.

    Raises
    ------
    TypeError
        If ``phases`` is a Quantity with a physical dimension.

    See Also
    --------
    braintools.analysis.spectral.analytic_signal : Instantaneous phases of continuous signals.
    braintools.analysis.spectral.plv : Phase locking between two signals.

    Notes
    -----
    Works under ``jax.jit`` and ``jax.vmap`` and is differentiable with
    respect to ``phases`` wherever :math:`R > 0`.

    References
    ----------
    .. [1] Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and
       Turbulence*. Springer, Berlin.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis.population import kuramoto_order_parameter
        >>> phases = jnp.array([[0.1, 0.1, 0.1],
        ...                     [0.0, 2 * jnp.pi / 3, 4 * jnp.pi / 3]])
        >>> R = kuramoto_order_parameter(phases)
        >>> [round(float(r), 4) for r in R]
        [1.0, 0.0]
        >>>
        >>> R, psi = kuramoto_order_parameter(phases, return_phase=True)
        >>> round(float(psi[0]), 4)
        0.1
    """
    theta = _as_float(_dimensionless(phases, 'phases'))
    z = jnp.mean(jnp.exp(1j * theta), axis=axis)
    r = jnp.abs(z)
    if return_phase:
        return r, jnp.angle(z)
    return r


# ----------------------------------------------------------------------------
# Granger causality
# ----------------------------------------------------------------------------

@set_module_as('braintools.analysis.population')
def granger_causality(x: Any, order: int) -> jax.Array:
    r"""Compute pairwise conditional Granger causality from a VAR model.

    For every target :math:`i` and source :math:`j \ne i`, two vector
    autoregressive (VAR) models of order :math:`p` with an intercept are
    fitted to :math:`x_i(t)` by ordinary least squares [1]_ [2]_:

    - the *full* model regresses :math:`x_i(t)` on the past
      :math:`x_k(t-1), \dots, x_k(t-p)` of **all** :math:`N` variables;
    - the *restricted* model drops the past of the source :math:`j`.

    The conditional Granger causality is the log ratio of the residual
    variances,

    .. math::

        \mathrm{GC}_{j \to i} = \ln
        \frac{\sigma^2_{i \mid \setminus j}}{\sigma^2_{i \mid \mathrm{all}}}
        \ge 0 .

    Parameters
    ----------
    x : array-like or Quantity
        Signals with time on axis 0, shape ``(T, N)``. The unit of a Quantity
        does not affect the (dimensionless) result.
    order : int
        Model order :math:`p` (number of lags), a static Python integer.

    Returns
    -------
    jax.Array
        Matrix of shape ``(N, N)`` with ``GC[i, j]`` the causal influence of
        ``j`` on ``i`` (``j -> i``) and a zero diagonal.

    Raises
    ------
    ValueError
        If ``x`` is not 2-D, or ``T <= order``.
    TypeError
        If ``order`` is not a static integer.

    See Also
    --------
    braintools.analysis.population.functional_connectivity : Undirected (zero-lag) correlations.

    Notes
    -----
    The fits use the normal equations. The data are centred, and the Gram
    matrix :math:`D^\top D` of the lagged design (with
    :math:`P = N p + 1` columns) and :math:`D^\top y` are accumulated once
    from lagged cross-products, without materialising the
    :math:`(T - p) \times P` design matrix. Each restricted model is a static
    sub-block of the same Gram matrix, solved sequentially (``lax.map``),
    and the residual sum of squares is
    :math:`y^\top y - \hat\beta^\top D^\top y`. Time is
    :math:`O(T N^2 p + N P^3)`; memory is a few copies of ``x`` plus
    :math:`O(P^2)`.

    Each small system is Jacobi-scaled and solved with a symmetric
    pseudo-inverse (``jnp.linalg.pinv``, relative cutoff
    :math:`10 P \epsilon`), so collinear or duplicated signals give finite
    values instead of NaN. The Gram matrices are accumulated in float64 when
    ``jax_enable_x64`` is on, otherwise in float32. Residual variances are
    floored at :math:`\epsilon` times the mean square of the raw target plus
    :math:`P \epsilon` times its variance (the round-off of an exact fit),
    so a perfectly predictable or constant target gives 0, and tiny negative
    round-off values are clipped to 0. ``T`` should be much larger than
    :math:`P` for a meaningful estimate.

    With ``N = 2`` this is the bivariate (pairwise) Granger causality of
    Geweke [2]_, as computed by ``elephant.causality.granger.pairwise_granger``
    (which fits the model with Yule-Walker equations instead of least
    squares, so finite-sample values differ slightly).

    Works under ``jax.jit`` and ``jax.vmap`` (for example over trials) and
    is differentiable with respect to ``x`` for well-conditioned data (the
    gradient of the pseudo-inverse is undefined for collinear signals).

    References
    ----------
    .. [1] Granger, C. W. J. (1969). Investigating causal relations by
       econometric models and cross-spectral methods. *Econometrica*, 37(3),
       424-438.
    .. [2] Geweke, J. (1982). Measurement of linear dependence and feedback
       between multiple time series. *Journal of the American Statistical
       Association*, 77(378), 304-313.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from braintools.analysis.population import granger_causality
        >>> # x drives y with one step of delay; y does not drive x.
        >>> noise = jax.random.normal(jax.random.key(0), (2000, 2))
        >>> x = noise[:, 0]
        >>> y = jnp.concatenate([noise[:1, 1], 0.8 * x[:-1] + 0.5 * noise[1:, 1]])
        >>> gc = granger_causality(jnp.stack([x, y], axis=1), order=1)
        >>> bool(gc[1, 0] > 0.5), bool(gc[0, 1] < 0.01)
        (True, True)
        >>> gc.shape
        (2, 2)
    """
    order = _static_int(order, 'order', 1)
    if isinstance(x, u.Quantity):
        x = x.mantissa
    x = _as_float(jnp.asarray(x))
    if x.ndim != 2:
        raise ValueError(f'`x` must have shape (T, N), got shape {x.shape}.')
    n_time, n_var = x.shape
    if n_time <= order:
        raise ValueError(
            f'`x` has T={n_time} time steps, which must exceed `order`={order}.'
        )

    p = order
    n_obs = n_time - p  # number of regression equations M
    # Accumulate in float64 when x64 is enabled, otherwise in (at least) float32.
    cdt = jnp.float64 if jax.config.jax_enable_x64 else jnp.promote_types(x.dtype, jnp.float32)
    hi = jax.lax.Precision.HIGHEST
    # Centring does not change the fits (every model has an intercept) but
    # improves the conditioning of the Gram matrix.
    xc = x.astype(cdt)
    xc = xc - jnp.mean(xc, axis=0)

    # Lagged second moments H[k, l] = sum_{t=p}^{T-1} x[t-k] x[t-l]^T, k, l = 0..p,
    # without materialising the (M, N p + 1) design matrix. First the p + 1
    # products H[0, d] (one (M, N) slice in memory at a time), then the
    # diagonals H[k, k+d] by adding / removing one outer product per step.
    base = xc[p:]

    def _cross(d):
        lagged = jax.lax.dynamic_slice_in_dim(xc, p - d, n_obs, axis=0)
        return jnp.matmul(base.T, lagged, precision=hi)

    head = jax.lax.map(_cross, jnp.arange(p + 1))  # head[d] = H[0, d]
    moments = jnp.zeros((p + 1, p + 1, n_var, n_var), dtype=cdt)
    for d in range(p + 1):  # static loop over lag differences (not over time)
        ks = np.arange(1, p - d + 1)
        add = jnp.einsum('ki,kj->kij', xc[p - ks], xc[p - ks - d], precision=hi)
        sub = jnp.einsum('ki,kj->kij', xc[n_time - ks], xc[n_time - ks - d], precision=hi)
        blocks = jnp.concatenate([head[d][None], head[d][None] + jnp.cumsum(add - sub, axis=0)])
        rows = np.arange(p - d + 1)
        moments = moments.at[rows, rows + d].set(blocks)
        moments = moments.at[rows + d, rows].set(jnp.swapaxes(blocks, 1, 2))
    # Window sums of every lag, for the intercept column.
    steps = xc[p - np.arange(1, p + 1)] - xc[n_time - np.arange(1, p + 1)]
    sums = jnp.concatenate([base.sum(axis=0)[None], base.sum(axis=0) + jnp.cumsum(steps, axis=0)])

    # Augmented Gram matrix over [lag 0 (targets), lags 1..p, intercept].
    q = (p + 1) * n_var
    gram = jnp.zeros((q + 1, q + 1), dtype=cdt)
    gram = gram.at[:q, :q].set(moments.transpose(0, 2, 1, 3).reshape(q, q))
    gram = gram.at[:q, q].set(sums.reshape(q)).at[q, :q].set(sums.reshape(q))
    gram = gram.at[q, q].set(n_obs)
    regs = np.concatenate([np.arange(n_var, q), [q]])  # regressor columns
    g_full = gram[regs][:, regs]  # D^T D, (P, P) with P = N p + 1
    b_full = gram[regs][:, :n_var]  # D^T y, (P, N)
    yy = jnp.diagonal(gram)[:n_var]  # y^T y, (N,)

    # Columns kept by the restricted model of every source j (static indices):
    # all lags of the other variables, plus the intercept (last column).
    owner = np.concatenate([np.tile(np.arange(n_var), order), [-1]])
    keep = np.stack([np.flatnonzero(owner != j) for j in range(n_var)])  # (N, (N-1) p + 1)

    def _residual_var(cols):
        g = g_full[cols][:, cols]
        b = b_full[cols]
        # Jacobi scaling, then a pseudo-inverse with the default relative cutoff
        # (10 * P * eps) so that collinear regressors are dropped, not amplified.
        diag = jnp.diagonal(g)
        scale = jnp.where(diag > 0, 1. / jnp.sqrt(jnp.where(diag > 0, diag, 1.)), 1.)
        gs = g * scale[:, None] * scale[None, :]
        bs = b * scale[:, None]
        coef = jnp.matmul(jnp.linalg.pinv(gs, hermitian=True), bs, precision=hi)
        return (yy - jnp.sum(bs * coef, axis=0)) / n_obs  # (N,) one per target

    var_full = _residual_var(np.arange(regs.size))
    var_restricted = jax.lax.map(_residual_var, jnp.asarray(keep))  # (source j, target i)

    # Floor of the residual variances: an exact fit (or a constant target) leaves
    # round-off of order eps * mean(y**2) of the raw data and of order
    # P * eps * var(y) from the Gram cancellation; both are mapped to GC = 0.
    target = x[p:]
    raw_scale = jnp.mean(target.astype(cdt) ** 2, axis=0)
    floor = (jnp.finfo(x.dtype).eps * raw_scale
             + regs.size * jnp.finfo(cdt).eps * yy / n_obs
             + jnp.finfo(cdt).tiny)
    gc = jnp.log(jnp.maximum(var_restricted, floor) / jnp.maximum(var_full, floor)[None, :])
    gc = jnp.maximum(gc, 0.).T  # GC[i, j] = j -> i
    return jnp.where(jnp.eye(n_var, dtype=bool), 0., gc).astype(x.dtype)


# ----------------------------------------------------------------------------
# criticality
# ----------------------------------------------------------------------------

class AvalancheResult(NamedTuple):
    """Neuronal avalanches detected in a population count series.

    Attributes
    ----------
    sizes : jax.Array
        Size of every avalanche (sum of ``counts - threshold`` over its
        bins), shape ``(max_avalanches,)``; entries from ``n`` on are 0.
    durations : jax.Array
        Number of bins of every avalanche, shape ``(max_avalanches,)``,
        ``int32``; entries from ``n`` on are 0.
    n : jax.Array
        Number of avalanches stored, a scalar ``int32``.
    """
    sizes: jax.Array
    durations: jax.Array
    n: jax.Array


AvalancheResult.__module__ = 'braintools.analysis.population'


@set_module_as('braintools.analysis.population')
def avalanches(
    counts: Any,
    threshold: Any = 0,
    max_avalanches: Optional[int] = None,
) -> AvalancheResult:
    r"""Detect neuronal avalanches in a population count series.

    An avalanche is a maximal run of consecutive bins in which the count
    exceeds ``threshold`` [1]_. Its duration is the number of bins in the run
    and its size is the sum of ``counts - threshold`` over the run (the
    number of events for ``threshold=0``). Runs that touch the start or the
    end of the series count as avalanches.

    Parameters
    ----------
    counts : array-like
        Population event counts per time bin, shape ``(T,)``, for example
        ``SpikeTrains.bin(bin_size).sum(axis=1)``.
    threshold : int or float, default 0
        Activity threshold; bins with ``counts > threshold`` are active.
    max_avalanches : int, optional
        Static length of the padded outputs. Defaults to ``T // 2 + 1``,
        which can hold every possible avalanche. When more avalanches occur,
        only the first ``max_avalanches`` are kept and ``n`` is clipped.

    Returns
    -------
    AvalancheResult
        Named tuple ``(sizes, durations, n)``. ``sizes`` and ``durations``
        have shape ``(max_avalanches,)`` and are padded with zeros after the
        first ``n`` entries. ``sizes`` has the dtype of
        ``counts - threshold``.

    Raises
    ------
    ValueError
        If ``counts`` is not 1-D, or ``max_avalanches < 1``.

    See Also
    --------
    branching_ratio : Ratio of descendants to ancestors.

    Notes
    -----
    Runs are found without a loop over time: the run index of every active
    bin is a cumulative sum of run starts, and the sizes and durations are
    segment sums. Works under ``jax.jit`` and ``jax.vmap`` (for example over
    trials). The output shape depends only on ``T`` and ``max_avalanches``.

    References
    ----------
    .. [1] Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches in neocortical
       circuits. *Journal of Neuroscience*, 23(35), 11167-11177.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis.population import avalanches
        >>> counts = jnp.array([2, 1, 0, 0, 3, 0, 1, 4, 2])
        >>> res = avalanches(counts, max_avalanches=4)
        >>> res.sizes, res.durations, res.n
        (Array([3, 3, 7, 0], dtype=int32), Array([2, 1, 3, 0], dtype=int32), Array(3, dtype=int32))
        >>>
        >>> # Get the counts from a spike raster:
        >>> from braintools.spike import SpikeTrains
        >>> raster = jnp.zeros((10, 4)).at[jnp.array([1, 2, 6]), jnp.array([0, 3, 1])].set(1.)
        >>> st = SpikeTrains.from_dense(raster, dt=1.)
        >>> avalanches(st.bin(1.).sum(axis=1)).n
        Array(2, dtype=int32)
    """
    counts = _dimensionless(counts, 'counts')
    if counts.ndim != 1:
        raise ValueError(f'`counts` must have shape (T,), got shape {counts.shape}.')
    n_time = counts.shape[0]
    if max_avalanches is None:
        max_avalanches = n_time // 2 + 1
    max_avalanches = _static_int(max_avalanches, 'max_avalanches', 1)

    excess = counts - jnp.asarray(_dimensionless(threshold, 'threshold'))
    active = excess > 0
    previous = jnp.concatenate([jnp.zeros((1,), dtype=bool), active[:-1]])
    start = active & ~previous
    run = jnp.cumsum(start, dtype=jnp.int32) - 1
    idx = jnp.where(active, run, max_avalanches)  # inactive bins are dropped

    sizes = jnp.zeros((max_avalanches,), dtype=excess.dtype)
    sizes = sizes.at[idx].add(jnp.where(active, excess, 0), mode='drop')
    durations = jnp.zeros((max_avalanches,), dtype=jnp.int32)
    durations = durations.at[idx].add(1, mode='drop')
    n = jnp.minimum(jnp.sum(start, dtype=jnp.int32), max_avalanches)
    return AvalancheResult(sizes, durations, n)


@set_module_as('braintools.analysis.population')
def branching_ratio(counts: Any, method: str = 'ratio') -> jax.Array:
    r"""Estimate the branching ratio of a population count series.

    The branching ratio :math:`m` is the expected number of events in bin
    :math:`t+1` per event in bin :math:`t`; :math:`m = 1` marks a critical
    branching process.

    - ``'ratio'``: the mean of :math:`n_{t+1} / n_t` over the bins with
      :math:`n_t > 0` [1]_.
    - ``'regression'``: the slope of the least-squares regression of
      :math:`n_{t+1}` on :math:`n_t`, the multistep-regression (MR)
      estimator at lag 1 [2]_, which is robust to subsampling.

    Parameters
    ----------
    counts : array-like
        Population event counts per time bin, shape ``(T,)`` with
        ``T >= 2``.
    method : {'ratio', 'regression'}, default 'ratio'
        Estimator.

    Returns
    -------
    jax.Array
        Scalar estimate. ``nan`` when it is undefined: no bin with
        :math:`n_t > 0` (``'ratio'``) or constant :math:`n_t`
        (``'regression'``).

    Raises
    ------
    ValueError
        If ``counts`` is not 1-D with at least 2 bins, or ``method`` is
        unknown.

    See Also
    --------
    avalanches : Sizes and durations of neuronal avalanches.

    Notes
    -----
    Works under ``jax.jit`` and ``jax.vmap`` and is differentiable with
    respect to (float) ``counts``.

    References
    ----------
    .. [1] Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches in neocortical
       circuits. *Journal of Neuroscience*, 23(35), 11167-11177.
    .. [2] Wilting, J., & Priesemann, V. (2018). Inferring collective dynamical
       states from widely unobserved systems. *Nature Communications*, 9,
       2325.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis.population import branching_ratio
        >>> counts = jnp.array([1., 2., 4., 0., 3., 3.])
        >>> float(branching_ratio(counts))  # (2 + 2 + 0 + 1) / 4
        1.25
        >>> round(float(branching_ratio(2. ** jnp.arange(6.), method='regression')), 4)
        2.0
    """
    if method not in ('ratio', 'regression'):
        raise ValueError(f"`method` must be 'ratio' or 'regression', got {method!r}.")
    counts = _as_float(_dimensionless(counts, 'counts'))
    if counts.ndim != 1 or counts.shape[0] < 2:
        raise ValueError(
            f'`counts` must have shape (T,) with T >= 2, got shape {counts.shape}.'
        )
    ancestors = counts[:-1]
    descendants = counts[1:]
    if method == 'ratio':
        ok = ancestors > 0
        ratio = jnp.where(ok, descendants / jnp.where(ok, ancestors, 1.), 0.)
        n_ok = jnp.sum(ok)
        return jnp.where(n_ok > 0, jnp.sum(ratio) / jnp.maximum(n_ok, 1), jnp.nan)
    da = ancestors - jnp.mean(ancestors)
    dd = descendants - jnp.mean(descendants)
    var = jnp.mean(da * da)
    ok = var > 0
    return jnp.where(ok, jnp.mean(da * dd) / jnp.where(ok, var, 1.), jnp.nan)
