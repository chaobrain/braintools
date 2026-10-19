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

"""Closed-form and proximal linear regressions (readout training).

Pure functions that fit ``Y ~ X @ coef + intercept``:

- :func:`linear_regression`: ordinary least squares (``lstsq``);
- :func:`ridge_regression`: L2-penalised least squares, solved through the SVD;
- :func:`lasso_regression` and :func:`elastic_net_regression`: L1 and L1+L2
  penalised least squares, solved with FISTA (with adaptive restart) inside a
  ``lax.while_loop``.

All of them return a :class:`RegressionResult` and can be wrapped in
``jax.jit`` and ``jax.vmap``. The objectives are scaled exactly as in
scikit-learn's ``LinearRegression``, ``Ridge``, ``Lasso`` and ``ElasticNet``.
"""

import numbers
from typing import Any, NamedTuple

import brainunit as u
import jax.numpy as jnp
from jax import lax

from braintools._misc import set_module_as

__all__ = [
    'RegressionResult',
    'linear_regression',
    'ridge_regression',
    'lasso_regression',
    'elastic_net_regression',
]


class RegressionResult(NamedTuple):
    """Fitted coefficients of a linear regression ``Y ~ X @ coef + intercept``.

    Attributes
    ----------
    coef : jax.Array or brainunit.Quantity
        Coefficients, shape ``(n_features,)`` for a 1-D target and
        ``(n_features, n_targets)`` for a 2-D target (the transpose of
        scikit-learn's ``coef_``, so that ``X @ coef`` is the prediction). In
        the unit of ``Y`` divided by the unit of ``X``.
    intercept : jax.Array or brainunit.Quantity
        Intercept, a scalar for a 1-D target and shape ``(n_targets,)`` for a
        2-D target; zero when ``fit_intercept=False``. In the unit of ``Y``.
    """
    coef: Any
    intercept: Any


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _prepare(X, Y, fit_intercept):
    """Validate shapes, split units and centre the data.

    Returns ``(Xc, Yc, x_mean, y_mean, squeeze, x_unit, y_unit)`` with ``Yc``
    always 2-D.
    """
    X, x_unit = u.split_mantissa_unit(X)
    Y, y_unit = u.split_mantissa_unit(Y)
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)
    if X.ndim != 2:
        raise ValueError(f'X must be a 2-D array (n_samples, n_features), got shape {X.shape}.')
    if Y.ndim not in (1, 2):
        raise ValueError(f'Y must be a 1-D or 2-D array, got shape {Y.shape}.')
    if Y.shape[0] != X.shape[0]:
        raise ValueError(
            f'X and Y must have the same number of samples, got {X.shape[0]} and {Y.shape[0]}.'
        )
    if X.shape[0] == 0:
        raise ValueError('X must contain at least one sample.')
    dtype = jnp.result_type(X.dtype, Y.dtype, jnp.float32)
    X = X.astype(dtype)
    Y = Y.astype(dtype)
    squeeze = Y.ndim == 1
    if squeeze:
        Y = Y[:, None]
    if fit_intercept:
        x_mean = jnp.mean(X, axis=0)
        y_mean = jnp.mean(Y, axis=0)
        X = X - x_mean
        Y = Y - y_mean
    else:
        x_mean = jnp.zeros(X.shape[1], dtype)
        y_mean = jnp.zeros(Y.shape[1], dtype)
    return X, Y, x_mean, y_mean, squeeze, x_unit, y_unit


def _finish(coef, x_mean, y_mean, fit_intercept, squeeze, x_unit, y_unit):
    """Compute the intercept, restore the target shape and attach units."""
    if fit_intercept:
        intercept = y_mean - x_mean @ coef
    else:
        intercept = jnp.zeros_like(y_mean)
    if squeeze:
        coef = coef[:, 0]
        intercept = intercept[0]
    coef = u.maybe_decimal(u.Quantity(coef, unit=y_unit / x_unit))
    intercept = u.maybe_decimal(u.Quantity(intercept, unit=y_unit))
    return RegressionResult(coef, intercept)


def _penalty(value, unit, name):
    """Return the mantissa of a penalty strength expressed in ``unit``.

    A plain number is accepted only when ``unit`` is dimensionless; otherwise
    ``value`` must be a Quantity of the dimension of ``unit``.
    """
    if isinstance(value, u.Quantity):
        if not u.get_dim(value) == u.get_dim(unit):
            raise u.UnitMismatchError(
                f'{name} must have the unit {unit} (the penalty must match the units of '
                f'the data), got {value.unit}.'
            )
        return value.to_decimal(unit)
    if not unit.is_unitless:
        raise u.UnitMismatchError(
            f'{name} must be a Quantity with the unit {unit} when X or Y carry units, '
            f'got the plain number {value!r}.'
        )
    if isinstance(value, numbers.Real) and value < 0:
        raise ValueError(f'{name} must be >= 0, got {value}.')
    return value


def _rcond(X):
    return jnp.finfo(X.dtype).eps * max(X.shape)


def _check_iter(max_iter, tol):
    if not isinstance(max_iter, numbers.Integral) or max_iter < 1:
        raise ValueError(f'max_iter must be a positive integer, got {max_iter!r}.')
    if isinstance(tol, numbers.Real) and tol < 0:
        raise ValueError(f'tol must be >= 0, got {tol}.')


def _default_tol(tol, dtype):
    """Resolve ``tol=None``: 1e-8 in float64, 1e-7 (about the rounding floor) in float32."""
    if tol is not None:
        return tol
    return 1e-8 if dtype == jnp.float64 else 1e-7


def _soft_threshold(x, thresh):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0.)


def _fista(X, Y, l1, l2, max_iter, tol):
    """Minimise ``1/(2n) ||Y - X W||^2 + l1 ||W||_1 + l2/2 ||W||^2`` with FISTA.

    Uses the gradient-based adaptive restart of O'Donoghue and Candes, which
    restores linear convergence on strongly convex problems.

    Stopping rule: with ``L`` the Lipschitz constant of the smooth part and
    ``W_{k+1} = prox(Z_k - grad(Z_k) / L)``, the gradient mapping
    ``G_k = L (Z_k - W_{k+1})`` is an element of ``grad(Z_k) + d(l1 ||.||_1)(W_{k+1})``
    and is zero exactly at a minimiser, so ``max|G_k|`` is a first-order
    optimality (KKT) residual in the units of the gradient. The loop stops when,
    for every target column ``j``, ``max_i |G_k|_ij <= tol * max_i |X^T Y|_ij / n``;
    the right-hand side is the gradient at ``W = 0`` (for the lasso, the
    smallest ``alpha`` with an all-zero solution), so the test is invariant to
    rescaling ``X`` and ``Y``. A negative or NaN penalty returns NaN.
    """
    n = X.shape[0]
    dtype = X.dtype
    l1 = jnp.asarray(l1, dtype)
    l2 = jnp.asarray(l2, dtype)
    # `not (x >= 0)` also catches NaN; traced penalties cannot be checked eagerly.
    invalid = jnp.logical_not(jnp.logical_and(l1 >= 0, l2 >= 0))
    smax = jnp.linalg.norm(X, ord=2)
    lip = smax ** 2 / n + l2
    # All-zero X with l2 == 0: any step works (the minimiser is W = 0).
    lip = jnp.where(lip > 0, lip, jnp.ones_like(lip))
    threshold = jnp.asarray(_default_tol(tol, dtype), dtype) * jnp.max(jnp.abs(X.T @ Y), axis=0) / n

    def body(carry):
        k, W, Z, t, _ = carry
        grad = X.T @ (X @ Z - Y) / n + l2 * Z
        W_new = _soft_threshold(Z - grad / lip, l1 / lip)
        restart = jnp.vdot(Z - W_new, W_new - W) > 0
        t_new = (1. + jnp.sqrt(1. + 4. * t * t)) / 2.
        beta = jnp.where(restart, 0., (t - 1.) / t_new)
        t_new = jnp.where(restart, jnp.ones_like(t_new), t_new)
        Z_new = W_new + beta * (W_new - W)
        residual = lip * jnp.max(jnp.abs(Z - W_new), axis=0)
        done = jnp.all(residual <= threshold)
        return k + 1, W_new, Z_new, t_new, done

    def cond(carry):
        k, _, _, _, done = carry
        return jnp.logical_and(k < max_iter, jnp.logical_not(done))

    W0 = jnp.zeros((X.shape[1], Y.shape[1]), dtype)
    init = (jnp.asarray(0), W0, W0, jnp.asarray(1., dtype), invalid)
    _, W, _, _, _ = lax.while_loop(cond, body, init)
    return jnp.where(invalid, jnp.nan, W)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

@set_module_as('braintools.optim')
def linear_regression(X, Y, *, fit_intercept: bool = True) -> RegressionResult:
    r"""Fit an ordinary least-squares linear regression.

    Minimises :math:`\|Y - X W - b\|_F^2` with :func:`jax.numpy.linalg.lstsq`.
    When ``X`` is rank deficient (for example more features than samples, or a
    constant feature with ``fit_intercept=True``) the minimum-norm solution is
    returned, so constant features get a zero coefficient.

    Parameters
    ----------
    X : array_like or brainunit.Quantity
        Design matrix, shape ``(n_samples, n_features)``.
    Y : array_like or brainunit.Quantity
        Targets, shape ``(n_samples,)`` or ``(n_samples, n_targets)``.
    fit_intercept : bool, default True
        Fit an unpenalised intercept by centring ``X`` and ``Y`` first.

    Returns
    -------
    RegressionResult
        ``coef`` of shape ``(n_features,)`` or ``(n_features, n_targets)`` (in
        the unit of ``Y`` over the unit of ``X``) and ``intercept`` (in the unit
        of ``Y``).

    Raises
    ------
    ValueError
        If ``X`` is not 2-D, ``Y`` is not 1-D or 2-D, or the sample counts differ.

    See Also
    --------
    ridge_regression : L2-penalised least squares.
    RLS : the same solution computed online, one batch at a time.

    Notes
    -----
    Matches ``sklearn.linear_model.LinearRegression``. The function is pure and
    supports ``jax.jit``, ``jax.vmap`` and ``jax.grad`` (gradients with respect
    to ``X`` require distinct non-zero singular values of the centred ``X``;
    gradients with respect to ``Y`` always exist).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> X = jnp.array([[0.], [1.], [2.], [3.]])
        >>> Y = 2. * X[:, 0] + 1.
        >>> res = braintools.optim.linear_regression(X, Y)
        >>> print(jnp.round(res.coef, 4), round(float(res.intercept), 4))
        [2.] 1.0
    """
    Xc, Yc, xm, ym, squeeze, xu, yu = _prepare(X, Y, fit_intercept)
    coef = jnp.linalg.lstsq(Xc, Yc, rcond=_rcond(Xc))[0]
    return _finish(coef, xm, ym, fit_intercept, squeeze, xu, yu)


@set_module_as('braintools.optim')
def ridge_regression(X, Y, alpha=1.0, *, fit_intercept: bool = True) -> RegressionResult:
    r"""Fit a ridge (L2-penalised least-squares) regression in closed form.

    Minimises

    .. math::

        \|Y - X W - b\|_F^2 + \alpha \|W\|_F^2,

    the objective of ``sklearn.linear_model.Ridge``; the intercept :math:`b` is
    not penalised. The solution is computed from the thin SVD
    :math:`X_c = U S V^\top` of the centred design matrix as
    :math:`W = V \operatorname{diag}(s / (s^2 + \alpha)) U^\top Y_c`, which works
    for any shape (including more features than samples) and reduces to the
    minimum-norm least-squares solution at ``alpha=0``.

    Parameters
    ----------
    X : array_like or brainunit.Quantity
        Design matrix, shape ``(n_samples, n_features)``.
    Y : array_like or brainunit.Quantity
        Targets, shape ``(n_samples,)`` or ``(n_samples, n_targets)``.
    alpha : float or brainunit.Quantity, default 1.0
        Non-negative L2 penalty. When ``X`` carries a unit, ``alpha`` must be a
        Quantity in that unit squared.
    fit_intercept : bool, default True
        Fit an unpenalised intercept by centring ``X`` and ``Y`` first.

    Returns
    -------
    RegressionResult
        ``coef`` of shape ``(n_features,)`` or ``(n_features, n_targets)`` and
        ``intercept``.

    Raises
    ------
    ValueError
        If the shapes are invalid or a concrete ``alpha`` is negative.
    brainunit.UnitMismatchError
        If ``alpha`` does not have the unit of ``X`` squared.

    See Also
    --------
    linear_regression : the ``alpha=0`` case.
    lasso_regression, elastic_net_regression : L1 and L1+L2 penalties.
    RLS : online ridge regression with ``P0 = I / alpha``.

    Notes
    -----
    Pure; supports ``jax.jit``, ``jax.vmap`` (for example over a grid of
    ``alpha``) and ``jax.grad``. Gradients with respect to ``Y`` and ``alpha``
    always exist; gradients with respect to ``X`` go through the SVD and need
    distinct non-zero singular values of the centred ``X``. A negative ``alpha``
    that is only known at run time (traced under ``jax.jit`` or ``jax.vmap``)
    cannot raise, so it returns NaN coefficients and intercept instead.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> X = jax.random.normal(jax.random.PRNGKey(0), (50, 3))
        >>> Y = X @ jnp.array([1., -2., 0.5]) + 3.
        >>> res = braintools.optim.ridge_regression(X, Y, alpha=1e-6)
        >>> print(jnp.round(res.coef, 3), round(float(res.intercept), 3))
        [ 1.  -2.   0.5] 3.0
        >>>
        >>> # vmap over a grid of penalties
        >>> alphas = jnp.array([0.1, 1., 10.])
        >>> coefs = jax.vmap(lambda a: braintools.optim.ridge_regression(X, Y, a).coef)(alphas)
        >>> coefs.shape
        (3, 3)
    """
    Xc, Yc, xm, ym, squeeze, xu, yu = _prepare(X, Y, fit_intercept)
    alpha = jnp.asarray(_penalty(alpha, xu * xu, 'alpha'), Xc.dtype)
    U, s, Vt = jnp.linalg.svd(Xc, full_matrices=False)
    keep = s > _rcond(Xc) * s[0]
    s_safe = jnp.where(keep, s, 1.)
    d = jnp.where(keep, s_safe / (s_safe ** 2 + alpha), 0.)
    coef = Vt.T @ (d[:, None] * (U.T @ Yc))
    # A traced alpha cannot be validated eagerly: a negative (or NaN) one gives NaN.
    coef = jnp.where(alpha >= 0, coef, jnp.nan)
    return _finish(coef, xm, ym, fit_intercept, squeeze, xu, yu)


@set_module_as('braintools.optim')
def elastic_net_regression(
    X,
    Y,
    alpha=1.0,
    l1_ratio=0.5,
    *,
    fit_intercept: bool = True,
    max_iter: int = 10000,
    tol: float | None = None,
) -> RegressionResult:
    r"""Fit an elastic-net (L1 + L2 penalised) linear regression with FISTA.

    Minimises the objective of ``sklearn.linear_model.ElasticNet``,

    .. math::

        \frac{1}{2 n} \|Y - X W - b\|_F^2
        + \alpha \rho \|W\|_1 + \frac{\alpha (1 - \rho)}{2} \|W\|_F^2,

    with :math:`\rho` = ``l1_ratio`` and :math:`n` the number of samples, by
    FISTA [1]_ with gradient-based adaptive restart [2]_ in a
    ``lax.while_loop``. The intercept is not penalised. Each target column is an
    independent problem (as in scikit-learn, not a multi-task penalty).

    Parameters
    ----------
    X : array_like
        Design matrix, shape ``(n_samples, n_features)``.
    Y : array_like
        Targets, shape ``(n_samples,)`` or ``(n_samples, n_targets)``.
    alpha : float, default 1.0
        Non-negative overall penalty strength.
    l1_ratio : float, default 0.5
        Mixing parameter :math:`\rho \in [0, 1]`: 1 is the lasso, 0 is ridge
        (with the ``1/(2n)`` data-term scaling).
    fit_intercept : bool, default True
        Fit an unpenalised intercept by centring ``X`` and ``Y`` first.
    max_iter : int, default 10000
        Maximum number of FISTA iterations (static).
    tol : float, optional
        Relative tolerance on the first-order optimality residual (see Notes).
        Default ``1e-8`` for float64 data and ``1e-7`` (close to the rounding
        floor) for float32 data. ``tol=0`` runs all ``max_iter`` iterations
        unless an exact minimiser is reached.

    Returns
    -------
    RegressionResult
        ``coef`` of shape ``(n_features,)`` or ``(n_features, n_targets)`` and
        ``intercept``.

    Raises
    ------
    ValueError
        If the shapes are invalid, a concrete ``alpha`` is negative, a concrete
        ``l1_ratio`` is outside ``[0, 1]``, or ``max_iter``/``tol`` are invalid.
    brainunit.UnitMismatchError
        If ``X`` or ``Y`` carry physical units: the L1 and L2 terms then have
        different units, so the objective is not defined. Pass mantissas.

    See Also
    --------
    lasso_regression : the ``l1_ratio=1`` case.
    ridge_regression : closed-form L2 penalty.

    Notes
    -----
    Pure; supports ``jax.jit`` and ``jax.vmap`` (a batched ``while_loop`` runs
    until every member has converged). The iteration count is data dependent,
    so reverse-mode differentiation (``jax.grad``) is not supported; forward
    mode (``jax.jvp``) works but differentiates the iterations, not the
    solution map.

    Stopping rule. Let :math:`L = \sigma_{\max}(X_c)^2 / n + \alpha (1 - \rho)`
    be the Lipschitz constant of the smooth part, :math:`Z_k` the extrapolated
    point and :math:`W_{k+1}` the proximal-gradient step from it. The gradient
    mapping :math:`G_k = L (Z_k - W_{k+1})` lies in
    :math:`\nabla f(Z_k) + \alpha \rho\, \partial \|W_{k+1}\|_1` and vanishes
    exactly at a minimiser, so it measures the violation of the optimality
    (KKT) conditions in the units of the gradient. The loop stops once, for
    every target column :math:`j`,

    .. math::

        \max_i |G_k|_{ij} \le \mathrm{tol} \cdot \max_i |X_c^\top Y_c|_{ij} / n,

    i.e. the residual is small relative to the gradient at :math:`W = 0`. The
    test is invariant to rescaling ``X`` and ``Y``, and unlike a test on the
    step :math:`|W_{k+1} - Z_k|` it does not stop early when the curvature
    is small (more features than samples, ill-conditioned ``X``). The
    coefficient error is roughly the residual divided by the smallest curvature
    on the active set, so for very ill-conditioned problems lower ``tol`` (and
    raise ``max_iter``).

    A penalty that is only known at run time (traced under ``jax.jit`` or
    ``jax.vmap``) cannot raise: a negative ``alpha`` (or an ``l1_ratio``
    outside ``[0, 1]`` that makes a penalty negative) returns NaN
    coefficients and intercept, without iterating.

    References
    ----------
    .. [1] A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding
       algorithm for linear inverse problems," SIAM J. Imaging Sci. 2(1),
       183-202, 2009.
    .. [2] B. O'Donoghue and E. Candes, "Adaptive restart for accelerated
       gradient schemes," Found. Comput. Math. 15, 715-732, 2015.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> X = jax.random.normal(jax.random.PRNGKey(0), (100, 5))
        >>> Y = X @ jnp.array([3., 0., 0., -2., 0.])
        >>> res = braintools.optim.elastic_net_regression(X, Y, alpha=0.1, l1_ratio=0.7)
        >>> print(jnp.round(res.coef, 2))
        [ 2.79 -0.    0.   -1.88  0.  ]
    """
    _check_iter(max_iter, tol)
    if isinstance(l1_ratio, numbers.Real) and not 0. <= float(l1_ratio) <= 1.:
        raise ValueError(f'l1_ratio must be in [0, 1], got {l1_ratio}.')
    Xc, Yc, xm, ym, squeeze, xu, yu = _prepare(X, Y, fit_intercept)
    if not (xu.is_unitless and yu.is_unitless):
        raise u.UnitMismatchError(
            'elastic_net_regression does not support X or Y with physical units: the L1 '
            'and L2 penalties would have different units. Pass the mantissas instead.'
        )
    alpha = _penalty(alpha, xu, 'alpha')
    coef = _fista(Xc, Yc, alpha * l1_ratio, alpha * (1. - l1_ratio), max_iter, tol)
    return _finish(coef, xm, ym, fit_intercept, squeeze, xu, yu)


@set_module_as('braintools.optim')
def lasso_regression(
    X,
    Y,
    alpha=1.0,
    *,
    fit_intercept: bool = True,
    max_iter: int = 10000,
    tol: float | None = None,
) -> RegressionResult:
    r"""Fit a lasso (L1-penalised) linear regression with FISTA.

    Minimises the objective of ``sklearn.linear_model.Lasso``,

    .. math::

        \frac{1}{2 n} \|Y - X W - b\|_F^2 + \alpha \|W\|_1,

    with :math:`n` the number of samples, by FISTA [1]_ with gradient-based
    adaptive restart [2]_ in a ``lax.while_loop``. The intercept is not
    penalised, and each target column is an independent problem.

    Parameters
    ----------
    X : array_like or brainunit.Quantity
        Design matrix, shape ``(n_samples, n_features)``.
    Y : array_like or brainunit.Quantity
        Targets, shape ``(n_samples,)`` or ``(n_samples, n_targets)``.
    alpha : float or brainunit.Quantity, default 1.0
        Non-negative L1 penalty. When ``X`` or ``Y`` carry units, ``alpha``
        must be a Quantity in the unit of ``X`` times the unit of ``Y``.
    fit_intercept : bool, default True
        Fit an unpenalised intercept by centring ``X`` and ``Y`` first.
    max_iter : int, default 10000
        Maximum number of FISTA iterations (static).
    tol : float, optional
        Relative tolerance on the first-order optimality residual (the stopping
        rule is described in :func:`elastic_net_regression`). Default ``1e-8``
        for float64 data and ``1e-7`` (close to the rounding floor) for float32
        data. ``tol=0`` runs all ``max_iter`` iterations unless an exact
        minimiser is reached.

    Returns
    -------
    RegressionResult
        ``coef`` of shape ``(n_features,)`` or ``(n_features, n_targets)`` and
        ``intercept``.

    Raises
    ------
    ValueError
        If the shapes are invalid, a concrete ``alpha`` is negative, or
        ``max_iter``/``tol`` are invalid.
    brainunit.UnitMismatchError
        If ``alpha`` does not have the unit of ``X`` times the unit of ``Y``.

    See Also
    --------
    elastic_net_regression : mixed L1 and L2 penalties.
    ridge_regression : closed-form L2 penalty.

    Notes
    -----
    Pure; supports ``jax.jit`` and ``jax.vmap``. The iteration count is data
    dependent, so reverse-mode differentiation (``jax.grad``) is not supported.
    For large ``alpha`` (above ``max|X_c^T Y_c| / n``) every coefficient is
    exactly zero.

    The loop stops when, for every target column, the largest entry of the
    proximal gradient mapping :math:`L (Z_k - W_{k+1})` (a first-order
    optimality residual, zero exactly at the minimiser) is at most
    ``tol * max|X_c^T Y_c| / n``; see :func:`elastic_net_regression`. A
    negative ``alpha`` that is only known at run time (traced under
    ``jax.jit`` or ``jax.vmap``) cannot raise, so it returns NaN coefficients
    and intercept.

    References
    ----------
    .. [1] A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding
       algorithm for linear inverse problems," SIAM J. Imaging Sci. 2(1),
       183-202, 2009.
    .. [2] B. O'Donoghue and E. Candes, "Adaptive restart for accelerated
       gradient schemes," Found. Comput. Math. 15, 715-732, 2015.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> X = jax.random.normal(jax.random.PRNGKey(0), (100, 5))
        >>> Y = X @ jnp.array([3., 0., 0., -2., 0.]) + 1.
        >>> res = braintools.optim.lasso_regression(X, Y, alpha=0.1)
        >>> print(jnp.round(res.coef, 2), round(float(res.intercept), 2))
        [ 2.86 -0.    0.   -1.9   0.  ] 0.98
    """
    _check_iter(max_iter, tol)
    Xc, Yc, xm, ym, squeeze, xu, yu = _prepare(X, Y, fit_intercept)
    alpha = _penalty(alpha, xu * yu, 'alpha')
    coef = _fista(Xc, Yc, alpha, 0., max_iter, tol)
    return _finish(coef, xm, ym, fit_intercept, squeeze, xu, yu)
