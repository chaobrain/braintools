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

"""Cross-validated linear decoding of discrete labels from population activity."""

import operator
from typing import Any, Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braintools._misc import set_module_as

__all__ = [
    'decode',
]

_MODELS = ('lda', 'ridge')


def _static_int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f'`{name}` must be a static int; got {value!r}.')
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f'`{name}` must be a static int; got {value!r}.') from None
    if value < minimum:
        raise ValueError(f'`{name}` must be >= {minimum}; got {value}.')
    return int(value)


def _fit_predict_lda(x_train, y_onehot, x_test, reg):
    """Shrinkage LDA scores, identical to sklearn's ``solver='lsqr'``."""
    n, d = x_train.shape
    counts = jnp.sum(y_onehot, axis=0)
    present = counts > 0
    means = (y_onehot.T @ x_train) / jnp.maximum(counts, 1.0)[:, None]
    resid = x_train - y_onehot @ means
    # Pooled within-class covariance = sum_k prior_k * cov_k (ddof=0 per class).
    s = resid.T @ resid / n
    mu = jnp.trace(s) / d
    cov = (1.0 - reg) * s + reg * mu * jnp.eye(d, dtype=s.dtype)
    coef = jnp.linalg.lstsq(cov, means.T)[0].T
    log_prior = jnp.log(jnp.where(present, counts / n, 1.0))
    intercept = -0.5 * jnp.sum(means * coef, axis=1) + log_prior
    scores = x_test @ coef.T + intercept
    return jnp.where(present, scores, -jnp.inf)


def _fit_predict_ridge(x_train, y_onehot, x_test, reg):
    """One-vs-rest ridge scores, identical to sklearn's ``RidgeClassifier``."""
    d = x_train.shape[1]
    present = jnp.sum(y_onehot, axis=0) > 0
    targets = 2.0 * y_onehot - 1.0
    x_mean = jnp.mean(x_train, axis=0)
    t_mean = jnp.mean(targets, axis=0)
    xc = x_train - x_mean
    a = xc.T @ xc + reg * jnp.eye(d, dtype=xc.dtype)
    w = jnp.linalg.lstsq(a, xc.T @ (targets - t_mean))[0]
    scores = x_test @ w + (t_mean - x_mean @ w)
    return jnp.where(present, scores, -jnp.inf)


@set_module_as('braintools.analysis.population')
def decode(
    x,
    y,
    n_classes: Optional[int] = None,
    n_folds: int = 5,
    model: str = 'lda',
    reg=1e-3,
    key: Optional[jax.Array] = None,
) -> jax.Array:
    r"""Cross-validated linear decoding of integer labels from population activity.

    The ``T`` samples are split into ``n_folds`` contiguous folds of equal size
    ``m = T // n_folds``; the trailing ``T mod n_folds`` samples are dropped so
    that every fold has the same static shape. For each fold a linear
    classifier is fit in closed form on the other ``n_folds - 1`` folds and its
    accuracy is measured on the held-out fold. The folds are processed in
    parallel with ``jax.vmap``.

    Two classifiers are available:

    - ``model='lda'``: shrinkage linear discriminant analysis [1]_ [2]_. The
      pooled within-class covariance :math:`S` (class covariances with
      ``ddof=0``, weighted by the class priors) is shrunk towards a scaled
      identity, :math:`\Sigma = (1 - \text{reg})\,S + \text{reg}\,
      \frac{\operatorname{tr} S}{D} I`, and the class scores are
      :math:`x^\top \Sigma^{-1}\mu_k - \tfrac12 \mu_k^\top\Sigma^{-1}\mu_k +
      \log \pi_k` with empirical priors :math:`\pi_k`.
    - ``model='ridge'``: one-vs-rest ridge regression [3]_ on ``+1/-1``
      targets with an unpenalised intercept,
      :math:`W = (X_c^\top X_c + \text{reg}\,I)^{-1} X_c^\top Y_c`;
      the predicted class has the largest score.

    Parameters
    ----------
    x : array_like or brainunit.Quantity
        Activity of shape ``(T, D)`` (samples by features); a 1-D array is
        treated as ``(T, 1)``. Quantities are decoded on their mantissa (the
        accuracy is dimensionless; for ``'ridge'`` the penalty ``reg`` is
        expressed in the squared units of that mantissa).
    y : array_like of int
        Labels of shape ``(T,)`` with values in ``[0, n_classes)``.
    n_classes : int, optional
        Number of classes (static). ``None`` means ``int(y.max()) + 1``, which
        needs a concrete ``y``: under ``jax.jit``/``jax.vmap`` it must be given
        explicitly.
    n_folds : int, default=5
        Number of folds (static, ``>= 2``).
    model : {'lda', 'ridge'}, default='lda'
        Classifier.
    reg : float, default=1e-3
        Regularisation. For ``'lda'`` it is the shrinkage intensity in
        ``[0, 1]``; for ``'ridge'`` it is the ridge penalty ``alpha >= 0``.
    key : jax.Array, optional
        PRNG key. If given, the samples are randomly permuted before folding
        (and before the trailing samples are dropped). If ``None`` the folds
        are contiguous blocks in the given order.

    Returns
    -------
    jax.Array
        Per-fold test accuracy, shape ``(n_folds,)``.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, ``T < n_folds``, ``model`` is unknown,
        a concrete ``reg`` is out of range, or ``n_classes`` is ``None`` while
        ``y`` is traced.
    TypeError
        If ``y`` is not an integer array or a static argument is not an int.

    See Also
    --------
    mutual_information : Model-free dependence between labels and responses.
    pca : Reduce dimensionality before decoding.

    Notes
    -----
    **Correspondence with scikit-learn.** On identical training/test splits
    the per-fold predictions equal those of

    - ``sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
      solver='lsqr', shrinkage=reg)`` for ``model='lda'``: sklearn's fixed
      shrinkage applies ``shrunk_covariance(empirical_covariance(X_k), reg)``
      to every class and sums them weighted by the priors, which by linearity
      is exactly :math:`\Sigma` above;
    - ``sklearn.linear_model.RidgeClassifier(alpha=reg)`` for
      ``model='ridge'`` (for two classes sklearn fits a single ``+1/-1``
      column, whose sign decision is identical to the arg-max over the two
      complementary one-vs-rest columns used here).

    **Absent classes.** A class that does not occur in a training fold cannot
    be predicted for that fold (its score is ``-inf``), as in scikit-learn,
    whose fitted ``classes_`` only contain the training labels; no ``nan`` is
    produced. Labels outside ``[0, n_classes)`` are never predicted and always
    count as errors.

    The function is jittable and vmappable (with explicit ``n_classes``) but
    not differentiable (the prediction is an arg-max).

    References
    ----------
    .. [1] Ledoit, O. and Wolf, M. (2004). "A well-conditioned estimator for
           large-dimensional covariance matrices." Journal of Multivariate
           Analysis 88(2): 365-411.
    .. [2] Blankertz, B., Lemm, S., Treder, M., Haufe, S. and Muller, K.-R.
           (2011). "Single-trial analysis and classification of ERP
           components - a tutorial." NeuroImage 56(2): 814-825.
    .. [3] Hoerl, A. E. and Kennard, R. W. (1970). "Ridge regression: biased
           estimation for nonorthogonal problems." Technometrics 12(1): 55-67.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from braintools.analysis import population
        >>> k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        >>> y = jnp.arange(103) % 3                         # 3 classes
        >>> x = jax.random.normal(k1, (103, 8)) + 3.0 * jax.nn.one_hot(y, 8)
        >>> acc = population.decode(x, y, n_folds=5)        # 3 trailing samples dropped
        >>> acc.shape
        (5,)
        >>> bool(acc.mean() > 0.9)
        True
        >>>
        >>> # Under jit, pass the number of classes explicitly.
        >>> f = jax.jit(lambda x, y, k: population.decode(x, y, n_classes=3,
        ...                                                 model='ridge', key=k))
        >>> bool(f(x, y, k2).mean() > 0.9)
        True
    """
    if model not in _MODELS:
        raise ValueError(f'`model` must be one of {_MODELS}; got {model!r}.')
    if isinstance(x, u.Quantity):
        x = x.mantissa
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(jnp.result_type(float))
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f'`x` must have shape (n_samples, n_features); got shape {x.shape}.')
    y = jnp.asarray(y)
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError(f'`y` must have shape ({x.shape[0]},); got {y.shape}.')
    if not jnp.issubdtype(y.dtype, jnp.integer):
        raise TypeError(f'`y` must hold integer labels; got dtype {y.dtype}.')
    n_folds = _static_int(n_folds, 'n_folds', 2)
    n_samples = x.shape[0]
    if n_samples < n_folds:
        raise ValueError(f'Need at least `n_folds`={n_folds} samples; got {n_samples}.')

    if n_classes is None:
        if isinstance(y, jax.core.Tracer):
            raise ValueError('`n_classes` must be given explicitly when `y` is traced '
                             '(e.g. under jax.jit or jax.vmap); it fixes the output shapes.')
        n_classes = int(jnp.max(y)) + 1
    n_classes = _static_int(n_classes, 'n_classes', 1)

    if not isinstance(reg, jax.core.Tracer):
        r = float(np.asarray(reg))
        if model == 'lda' and not 0.0 <= r <= 1.0:
            raise ValueError(f"For model='lda', `reg` is a shrinkage intensity in [0, 1]; got {r}.")
        if model == 'ridge' and r < 0.0:
            raise ValueError(f"For model='ridge', `reg` must be non-negative; got {r}.")
    reg = jnp.asarray(reg, dtype=x.dtype)

    if key is not None:
        perm = jax.random.permutation(key, n_samples)
        x = x[perm]
        y = y[perm]

    fold = n_samples // n_folds
    n_used = fold * n_folds
    x = x[:n_used]
    y = y[:n_used]

    # Static fold indices (host-side numpy, vectorised).
    idx = np.tile(np.arange(n_used), (n_folds, 1))
    keep = (idx // fold) != np.arange(n_folds)[:, None]
    train_idx = idx[keep].reshape(n_folds, n_used - fold)
    test_idx = np.arange(n_used).reshape(n_folds, fold)

    fit_predict = _fit_predict_lda if model == 'lda' else _fit_predict_ridge
    onehot = jax.nn.one_hot(y, n_classes, dtype=x.dtype)

    def one_fold(tr, te):
        scores = fit_predict(x[tr], onehot[tr], x[te], reg)
        pred = jnp.argmax(scores, axis=1)
        return jnp.mean((pred == y[te]).astype(x.dtype))

    return jax.vmap(one_fold)(jnp.asarray(train_idx), jnp.asarray(test_idx))
