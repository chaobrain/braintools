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

"""Curvature (Laplace) confidence intervals of fitted parameters."""

from typing import Any, Callable, NamedTuple, Optional

import brainunit as u
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from braintools._misc import set_module_as

__all__ = [
    'HessianCI',
    'hessian_ci',
]


class HessianCI(NamedTuple):
    """Result of :func:`hessian_ci`.

    Every field has the structure (and units) of ``best_params``.

    Attributes
    ----------
    estimate : PyTree
        The parameter values (``best_params``).
    stderr : PyTree
        Standard errors.
    low : PyTree
        Lower confidence limits.
    high : PyTree
        Upper confidence limits.
    """
    estimate: Any
    stderr: Any
    low: Any
    high: Any


def _check_static_number(value: Any, name: str) -> float:
    if isinstance(value, jax.core.Tracer):
        raise TypeError(f'`{name}` must be a concrete Python number, not a traced value.')
    return float(value)


@set_module_as('braintools.fit')
def hessian_ci(
    loss: Callable[[Any], Any],
    best_params: Any,
    *,
    confidence: float = 0.95,
    kind: str = 'sse',
    n_data: Optional[int] = None,
) -> HessianCI:
    r"""Compute confidence intervals from the Hessian of a loss at its minimum.

    The loss is expanded to second order around the optimum
    :math:`\hat\theta` (in the constrained, physical parameter space), with
    Hessian :math:`H = \nabla^2 L(\hat\theta)` taken over all ``p`` scalar
    entries of ``best_params``:

    - ``kind='nll'``: ``loss`` is a negative log-likelihood and the Laplace
      approximation gives :math:`\mathrm{Cov} = H^{-1}`; intervals use the
      normal quantile.
    - ``kind='sse'``: ``loss`` is a sum of squared errors over ``n_data``
      residuals. With :math:`s^2 = L(\hat\theta) / (n - p)`,
      :math:`\mathrm{Cov} = 2 s^2 H^{-1}`, and intervals use the Student-t
      quantile with :math:`n - p` degrees of freedom. For a linear model this
      is exactly the ordinary-least-squares covariance
      :math:`s^2 (X^\top X)^{-1}`.

    The interval of parameter :math:`i` is
    :math:`\hat\theta_i \pm q\,\sqrt{\mathrm{Cov}_{ii}}`.

    Parameters
    ----------
    loss : callable
        ``loss(params) -> scalar``, twice differentiable, taking a pytree with
        the structure of ``best_params``.
    best_params : PyTree
        The optimum (e.g. ``FitResult.best_params``). Leaves may be
        ``brainunit`` Quantities; the loss then receives Quantities and the
        results carry the same units.
    confidence : float, default 0.95
        Two-sided confidence level in ``(0, 1)``.
    kind : {'sse', 'nll'}, default 'sse'
        Meaning of the loss, see above.
    n_data : int, optional
        Number of residuals in the sum of squares; required for
        ``kind='sse'`` and must exceed the number of parameters.

    Returns
    -------
    HessianCI
        ``(estimate, stderr, low, high)``, each with the structure of
        ``best_params``.

    Raises
    ------
    ValueError
        If ``kind`` is unknown, ``confidence`` is outside ``(0, 1)``,
        ``n_data`` is missing or not larger than the number of parameters
        for ``kind='sse'``, or ``best_params`` has no leaves.
    TypeError
        If ``confidence`` or ``n_data`` is traced, or the loss carries a
        physical unit.

    See Also
    --------
    Fitter : Produces ``best_params``.

    Notes
    -----
    The intervals are asymptotic and assume the optimum is an interior
    minimum where the loss is locally quadratic. A singular or indefinite
    Hessian (non-identifiable parameters, a saddle) gives non-finite or
    meaningless standard errors.

    ``confidence`` and ``n_data`` fix the quantile, so they must be concrete;
    with them static, the function works under ``jax.jit`` and ``jax.vmap``
    over ``best_params``. It is not meant to be differentiated.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.fit import hessian_ci
        >>> x = jnp.arange(6.)
        >>> y = jnp.array([0.1, 1.9, 4.2, 5.8, 8.1, 9.9])
        >>> sse = lambda p: jnp.sum((y - p['a'] - p['b'] * x) ** 2)
        >>> best = {'a': jnp.asarray(0.05714286), 'b': jnp.asarray(1.9771429)}
        >>> ci = hessian_ci(sse, best, n_data=6)
        >>> print(round(float(ci.stderr['b']), 4), round(float(ci.low['b']), 3), round(float(ci.high['b']), 3))
        0.0398 1.867 2.088
    """
    from scipy import stats

    if kind not in ('sse', 'nll'):
        raise ValueError(f"`kind` must be 'sse' or 'nll', got {kind!r}.")
    confidence = _check_static_number(confidence, 'confidence')
    if not 0. < confidence < 1.:
        raise ValueError(f'`confidence` must lie in (0, 1), got {confidence}.')

    leaves, treedef = jax.tree.flatten(best_params, is_leaf=u.math.is_quantity)
    if not leaves:
        raise ValueError('`best_params` has no leaves.')
    units = [leaf.unit if isinstance(leaf, u.Quantity) else None for leaf in leaves]
    mantissas = [jnp.asarray(u.get_mantissa(leaf)) for leaf in leaves]
    dtype = jnp.result_type(*[m.dtype for m in mantissas], jnp.float32)
    mantissas = [m.astype(dtype) for m in mantissas]
    theta, unravel = ravel_pytree(mantissas)
    n_params = int(theta.size)

    def wrap(ms):
        return jax.tree.unflatten(
            treedef, [m if unit is None else u.Quantity(m, unit) for m, unit in zip(ms, units)]
        )

    def flat_loss(t):
        value = loss(wrap(unravel(t)))
        if isinstance(value, u.Quantity):
            if not value.dim.is_dimensionless:
                raise TypeError(f'`loss` must return a dimensionless scalar, got unit {value.unit}.')
            value = value.mantissa
        value = jnp.asarray(value)
        if value.ndim != 0:
            raise ValueError(f'`loss` must return a scalar, got shape {value.shape}.')
        return value

    hess = jax.hessian(flat_loss)(theta)
    hess = 0.5 * (hess + hess.T)
    cov = jnp.linalg.inv(hess)
    alpha = 1. - confidence
    if kind == 'sse':
        if n_data is None:
            raise ValueError("`n_data` is required for kind='sse'.")
        n_data = int(_check_static_number(n_data, 'n_data'))
        dof = n_data - n_params
        if dof <= 0:
            raise ValueError(
                f'`n_data` ({n_data}) must exceed the number of parameters ({n_params}).'
            )
        s2 = flat_loss(theta) / dof
        cov = 2. * s2 * cov
        q = float(stats.t.ppf(1. - alpha / 2., dof))
    else:
        q = float(stats.norm.ppf(1. - alpha / 2.))

    se = jnp.sqrt(jnp.diag(cov))
    se_leaves = unravel(se)
    low_leaves = [m - q * s for m, s in zip(mantissas, se_leaves)]
    high_leaves = [m + q * s for m, s in zip(mantissas, se_leaves)]
    return HessianCI(
        estimate=wrap(mantissas),
        stderr=wrap(se_leaves),
        low=wrap(low_leaves),
        high=wrap(high_leaves),
    )

