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

"""Regression losses."""

from typing import Optional, Union

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from braintools._misc import set_module_as
from ._util import _reduce

__all__ = [
    'squared_error',
    'absolute_error',
    'l1_loss',
    'l2_loss',
    'l2_norm',
    'huber_loss',
    'log_cosh',
    'cosine_similarity',
    'cosine_distance',
]


@set_module_as('braintools.metric')
def safe_norm(x: brainstate.typing.ArrayLike,
              min_norm,
              ord: Optional[Union[int, float, str]] = None,  # pylint: disable=redefined-builtin
              axis: Union[None, tuple[int, ...], int] = None,
              keepdims: bool = False) -> brainstate.typing.ArrayLike:
    """Compute vector or matrix norm with gradient-safe lower bound.

    Calculates the norm of input arrays while ensuring the result is at least
    ``min_norm``, with proper gradient handling. This prevents gradient issues
    that arise when using ``jax.numpy.maximum(jax.numpy.linalg.norm(x), min_norm)``
    directly, which can produce NaN gradients at zero due to JAX evaluating
    both branches of the maximum operation.

    Parameters
    ----------
    x : brainstate.typing.ArrayLike
        Input array for which to compute the norm. Can be of any shape.
    min_norm : float
        Minimum value for the returned norm. The result will be at least this value.
    ord : {int, float, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm:
        
        - For vectors: any real number, inf, -inf (default: 2)
        - For matrices: 'fro' (Frobenius), 'nuc' (nuclear), inf, -inf, 1, -1, 2, -2
        
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which to compute the norm:
        
        - None: flatten array and compute single norm
        - int: compute norms along specified axis
        - tuple: for matrix norms, specify the two axes defining matrices
        
    keepdims : bool, default=False
        If True, the reduced axes are left as dimensions with size one,
        allowing the result to broadcast against the original array.

    Returns
    -------
    brainstate.typing.ArrayLike
        Array norms with values bounded below by ``min_norm``. Shape depends
        on ``axis`` and ``keepdims`` parameters, following ``jax.numpy.linalg.norm``
        conventions.

    Notes
    -----
    This function addresses a specific issue with automatic differentiation where
    the gradient of ``max(norm(x), min_norm)`` is undefined (NaN) when ``norm(x) = 0``.
    The implementation ensures that:
    
    - Forward pass: Returns ``max(norm(x), min_norm)``
    - Backward pass: Provides well-defined gradients even at zero
    
    The gradient handling works by masking the input when the norm would be below
    the threshold, ensuring the gradient computation path remains valid.

    Examples
    --------
    Basic usage with vector norms:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.metric import safe_norm
        >>> x = jnp.array([0.0, 0.0, 0.0])  # Zero vector
        >>> safe_norm(x, min_norm=1e-8)  # Returns 1e-8 instead of 0.0
        Array(1.e-08, dtype=float32)

    Compare with regular norm:

    .. code-block:: python

        >>> jnp.linalg.norm(x)  # Returns 0.0
        Array(0., dtype=float32)
        >>> safe_norm(x, min_norm=0.1)  # Returns 0.1
        Array(0.1, dtype=float32)

    Matrix norms with axis specification:

    .. code-block:: python

        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> # L2 norm along rows
        >>> safe_norm(X, min_norm=0.1, axis=1)
        Array([2.2360680, 5.       ], dtype=float32)

    See Also
    --------
    jax.numpy.linalg.norm : Standard norm computation without lower bound
    jax.numpy.maximum : Element-wise maximum operation
    """
    norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=True)
    x = jnp.where(norm <= min_norm, jnp.ones_like(x), x)
    # When ``keepdims`` is False we drop the reduced axes. ``axis=None`` reduces
    # over all axes, so squeeze with ``axis=None`` removes every singleton
    # dimension, matching ``jax.numpy.linalg.norm``'s ``keepdims=False`` output.
    norm = jnp.squeeze(norm, axis=axis) if not keepdims else norm
    masked_norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    return jnp.where(norm <= min_norm, min_norm, masked_norm)


@set_module_as('braintools.metric')
def squared_error(
    predictions: brainstate.typing.ArrayLike,
    targets: Optional[brainstate.typing.ArrayLike] = None,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    reduction: str = 'none',
) -> brainstate.typing.ArrayLike:
    r"""Compute element-wise squared error between predictions and targets.

    Calculates the squared differences between predicted and target values,
    which forms the basis for Mean Squared Error (MSE) and related regression
    metrics. This is one of the most fundamental loss functions in machine
    learning, particularly effective for regression tasks.

    The squared error is defined as:

    .. math::

        \text{SE}(y, \hat{y}) = (y - \hat{y})^2

    where :math:`y` are the true values and :math:`\hat{y}` are the predictions.

    Parameters
    ----------
    predictions : brainstate.typing.ArrayLike
        Predicted values with arbitrary shape. Must be floating-point type.
    targets : brainstate.typing.ArrayLike, optional
        Ground truth target values. When provided, ``targets`` must have the
        *exact* same shape as ``predictions`` (the implementation asserts shape
        equality and does not broadcast). If not provided, targets are assumed
        to be zeros, making this equivalent to computing the squared magnitude
        of predictions.
    axis : int or tuple of ints, optional
        Axis or axes along which to reduce the error. If None, no reduction
        is performed and element-wise errors are returned.
    reduction : {'none', 'mean', 'sum'}, default='none'
        Reduction operation to apply:
        
        - ``'none'``: Return element-wise errors without reduction
        - ``'mean'``: Return mean of errors (MSE when no axis specified)
        - ``'sum'``: Return sum of errors

    Returns
    -------
    brainstate.typing.ArrayLike
        Squared errors. Shape depends on ``axis`` and ``reduction`` parameters:
        
        - If ``reduction='none'``: same shape as ``predictions``
        - If reduction is applied: reduced according to ``axis`` parameter

    Notes
    -----
    This function is closely related to L2 loss, with the relationship:
    
    .. math::
    
        \text{L2 loss} = \frac{1}{2} \times \text{squared error}
        
    The factor of 0.5 is conventional in some textbooks (e.g., Bishop's "Pattern
    Recognition and Machine Learning") but not others (e.g., "The Elements of
    Statistical Learning" by Hastie, Tibshirani, and Friedman).

    Mean Squared Error (MSE) is computed as ``squared_error(pred, target, reduction='mean')``.

    For ``brainunit.Quantity`` inputs the result carries **squared** units
    (e.g. ``mV`` inputs produce a ``mV ** 2`` result), since the error is
    squared element-wise.

    Examples
    --------
    Basic element-wise squared error:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> predictions = jnp.array([1.0, 2.0, 3.0])
        >>> targets = jnp.array([1.1, 1.9, 3.2])
        >>> braintools.metric.squared_error(predictions, targets)
        Array([0.01      , 0.01      , 0.04000002], dtype=float32)

    Mean Squared Error:

    .. code-block:: python

        >>> braintools.metric.squared_error(predictions, targets, reduction='mean')
        Array(0.02000001, dtype=float32)

    Squared error with missing targets (assuming zero targets):

    .. code-block:: python

        >>> pred_only = jnp.array([0.5, -0.3, 0.8])
        >>> braintools.metric.squared_error(pred_only)
        Array([0.25      , 0.09      , 0.64000005], dtype=float32)

    Batch processing with axis reduction:

    .. code-block:: python

        >>> batch_pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> batch_targets = jnp.array([[1.1, 1.9], [2.8, 4.2]])
        >>> # MSE per sample
        >>> braintools.metric.squared_error(batch_pred, batch_targets,
        ...                                 axis=1, reduction='mean')
        Array([0.01      , 0.03999997], dtype=float32)

    See Also
    --------
    braintools.metric.absolute_error : L1 loss alternative
    braintools.metric.l2_loss : Squared error scaled by 0.5
    braintools.metric.huber_loss : Robust alternative combining L1 and L2

    References
    ----------
    .. [1] Bishop, Christopher M. "Pattern recognition and machine learning." 
           Springer, 2006.
    .. [2] Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. 
           "The elements of statistical learning: data mining, inference, 
           and prediction." Springer, 2009.
    """
    assert u.math.is_float(predictions), 'predictions must be float.'
    if targets is not None:
        # Avoid broadcasting logic for "-" operator.
        assert predictions.shape == targets.shape, 'predictions and targets must have the same shape.'
    errors = predictions - targets if targets is not None else predictions
    # return errors ** 2
    return _reduce(errors ** 2, reduction, axis=axis)


@set_module_as('braintools.metric')
def absolute_error(
    predictions: brainstate.typing.ArrayLike,
    targets: Optional[brainstate.typing.ArrayLike] = None,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    reduction: str = 'mean',
) -> brainstate.typing.ArrayLike:
    r"""Compute element-wise absolute error between predictions and targets.

    Calculates the absolute differences between predicted and target values,
    forming the basis for Mean Absolute Error (MAE) and L1 loss. This metric
    is more robust to outliers than squared error and provides intuitive
    error measurements in the same units as the original data.

    The absolute error is defined as:

    .. math::

        \text{AE}(y, \hat{y}) = |y - \hat{y}|

    where :math:`y` are the true values and :math:`\hat{y}` are the predictions.

    Parameters
    ----------
    predictions : brainstate.typing.ArrayLike
        Predicted values with arbitrary shape. Must be floating-point type.
    targets : brainstate.typing.ArrayLike, optional
        Ground truth target values. When provided, ``targets`` must have the
        *exact* same shape as ``predictions`` (the implementation asserts shape
        equality and does not broadcast). If not provided, targets are assumed
        to be zeros, making this equivalent to computing the absolute magnitude
        of predictions.
    axis : int or tuple of ints, optional
        Axis or axes along which to reduce the error. If None, no reduction
        is performed unless specified by ``reduction`` parameter.
    reduction : {'none', 'mean', 'sum'}, default='mean'
        Reduction operation to apply:
        
        - ``'none'``: Return element-wise errors without reduction
        - ``'mean'``: Return mean of errors (MAE when no axis specified)
        - ``'sum'``: Return sum of errors

    Returns
    -------
    brainstate.typing.ArrayLike
        Absolute errors. Shape depends on ``axis`` and ``reduction`` parameters:
        
        - If ``reduction='none'``: same shape as ``predictions``
        - If reduction is applied: reduced according to ``axis`` parameter

    Notes
    -----
    Absolute error is equivalent to L1 loss and provides several advantages:
    
    - **Robustness**: Less sensitive to outliers than squared error
    - **Interpretability**: Error magnitude in original units
    - **Gradient properties**: Constant gradient magnitude (except at zero)
    
    The relationship to other metrics:
    
    - Mean Absolute Error (MAE): ``absolute_error(pred, target, reduction='mean')``
    - L1 norm of differences: ``absolute_error(pred, target, reduction='sum')``

    Examples
    --------
    Basic element-wise absolute error:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> predictions = jnp.array([1.0, 2.0, 3.0])
        >>> targets = jnp.array([1.1, 1.9, 3.2])
        >>> braintools.metric.absolute_error(predictions, targets, reduction='none')
        Array([0.10000002, 0.10000002, 0.20000005], dtype=float32)

    Mean Absolute Error (default):

    .. code-block:: python

        >>> braintools.metric.absolute_error(predictions, targets)
        Array(0.13333337, dtype=float32)

    Batch processing with axis reduction:

    .. code-block:: python

        >>> batch_pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> batch_targets = jnp.array([[1.1, 1.9], [2.8, 4.2]])
        >>> # MAE per sample
        >>> braintools.metric.absolute_error(batch_pred, batch_targets, axis=1)
        Array([0.10000002, 0.19999993], dtype=float32)

    See Also
    --------
    braintools.metric.squared_error : L2 loss alternative
    braintools.metric.l1_loss : Alternative L1 implementation
    braintools.metric.huber_loss : Combines L1 and L2 properties

    References
    ----------
    .. [1] Willmott, Cort J., and Kenji Matsuura. "Advantages of the mean 
           absolute error (MAE) over the root mean square error (RMSE) in 
           assessing average model performance." Climate research 30.1 (2005): 79-82.
    """
    assert u.math.is_float(predictions), 'predictions must be float.'
    if targets is not None:
        # Avoid broadcasting logic for "-" operator.
        assert predictions.shape == targets.shape, 'predictions and targets must have the same shape.'
    errors = predictions - targets if targets is not None else predictions
    return _reduce(jnp.abs(errors), reduction, axis=axis)


class L1Loss:
    r"""Creates a criterion that measures the mean absolute error (MAE) between each element in
    the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The mean operation divides by the total number of elements :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Supports real-valued and complex-valued inputs.

    Parameters
    ----------
    reduction : str, optional
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``.

        - ``'none'``: no reduction will be applied,
        - ``'mean'``: the sum of the output will be divided by the number of
          elements in the output,
        - ``'sum'``: the output will be summed.

        Default: ``'mean'``.

    Notes
    -----
    - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
    - Target: :math:`(*)`, same shape as the input.
    - Output: scalar. If :attr:`reduction` is ``'none'``, then a per-sample
      mean absolute error of shape :math:`(N,)`.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> from braintools.metric import L1Loss
        >>> loss = L1Loss()
        >>> input = brainstate.random.randn(3, 5)
        >>> target = brainstate.random.randn(3, 5)
        >>> output = loss.update(input, target)
    """

    def __init__(self, reduction: str = 'mean') -> None:
        self.reduction = reduction

    def update(self,
               input: brainstate.typing.ArrayLike,
               target: brainstate.typing.ArrayLike) -> brainstate.typing.ArrayLike:
        return l1_loss(input, target, reduction=self.reduction)


@set_module_as('braintools.metric')
def l1_loss(logits: brainstate.typing.ArrayLike,
            targets: brainstate.typing.ArrayLike,
            reduction: str = 'mean'):
    r"""Measure the mean absolute error (MAE) between each element in the logits
    :math:`x` and the targets :math:`y`. It is useful in regression problems.

    The per-sample mean absolute error (i.e. with :attr:`reduction` set to
    ``'none'``) can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \frac{1}{D} \sum_{d=1}^{D} \left| x_{n,d} - y_{n,d} \right|,

    where :math:`N` is the batch size and :math:`D` is the number of features
    per sample (i.e. the product of all non-batch dimensions). If
    :attr:`reduction` is not ``'none'`` (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    Supports real-valued and complex-valued inputs.

    Parameters
    ----------
    logits : brainstate.typing.ArrayLike
        :math:`(N, *)` where :math:`*` means any number of additional dimensions.
    targets : brainstate.typing.ArrayLike
        :math:`(N, *)`, same shape as ``logits``.
    reduction : str, optional
        Specifies the reduction to apply to the per-sample losses:
        ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.

        - ``'none'``: no reduction will be applied; returns the per-sample MAE
          of shape :math:`(N,)`,
        - ``'mean'``: the per-sample losses are averaged,
        - ``'sum'``: the per-sample losses are summed.

    Returns
    -------
    output : brainstate.typing.ArrayLike
        Scalar if :attr:`reduction` is ``'mean'`` or ``'sum'``. If
        :attr:`reduction` is ``'none'``, an array of shape :math:`(N,)` holding
        the per-sample mean absolute errors.

    Notes
    -----
    This computes a true **mean** absolute error: the absolute differences are
    averaged over the feature axis (all non-batch dimensions) for each sample,
    rather than summed. The previous implementation returned a per-row L1
    *sum* (``jnp.linalg.norm(diff, ord=1, axis=1)``); this corrected version
    divides by the number of features so the result matches the documented MAE
    semantics. With ``reduction='mean'`` over a single feature this equals
    ``jnp.mean(jnp.abs(logits - targets))``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> logits = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> targets = jnp.array([[1.5, 2.5], [2.0, 5.0]])
        >>> braintools.metric.l1_loss(logits, targets)
        Array(0.75, dtype=float32)
        >>> braintools.metric.l1_loss(logits, targets, reduction='none')
        Array([0.5, 1. ], dtype=float32)
    """

    diff = (logits - targets).reshape((logits.shape[0], -1))
    per_sample_mae = jnp.mean(jnp.abs(diff), axis=-1)
    return _reduce(outputs=per_sample_mae, reduction=reduction)


@set_module_as('braintools.metric')
def l2_loss(
    predictions: brainstate.typing.ArrayLike,
    targets: Optional[brainstate.typing.ArrayLike] = None,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    reduction: str = 'none',
) -> brainstate.typing.ArrayLike:
    r"""Calculate the L2 loss for a set of predictions.

    The L2 loss is half the squared error:

    .. math::

        \text{L2 loss} = \frac{1}{2} (y - \hat{y})^2

    Parameters
    ----------
    predictions : brainstate.typing.ArrayLike
        A vector of arbitrary shape ``[...]``.
    targets : brainstate.typing.ArrayLike, optional
        A vector with the same shape as ``predictions`` (shape equality is
        asserted; no broadcasting). If not provided then it is assumed to be a
        vector of zeros.
    axis : int or tuple of ints, optional
        Axis or axes along which to reduce when ``reduction`` is ``'mean'`` or
        ``'sum'``. If None, reduction (if any) is over all elements.
    reduction : {'none', 'mean', 'sum'}, default='none'
        Reduction operation to apply:

        - ``'none'``: return element-wise losses with the same shape as
          ``predictions`` (backward-compatible default),
        - ``'mean'``: return the mean of the losses,
        - ``'sum'``: return the sum of the losses.

    Returns
    -------
    brainstate.typing.ArrayLike
        Element-wise squared differences scaled by ``0.5`` when
        ``reduction='none'``; otherwise the reduced value.

    Notes
    -----
    The ``0.5`` factor is standard in "Pattern Recognition and Machine Learning"
    by Bishop, but not in "The Elements of Statistical Learning" by Tibshirani.

    For ``brainunit.Quantity`` inputs the result carries **squared** units
    (e.g. ``mV`` inputs produce a ``mV ** 2`` result), since the error is
    squared element-wise.

    References
    ----------
    .. [1] Bishop, Christopher M. "Pattern recognition and machine learning."
           Springer, 2006. https://bit.ly/3eeP0ga

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> predictions = jnp.array([1.0, 2.0, 3.0])
        >>> targets = jnp.array([1.5, 2.5, 2.0])
        >>> braintools.metric.l2_loss(predictions, targets)
        Array([0.125, 0.125, 0.5  ], dtype=float32)
        >>> braintools.metric.l2_loss(predictions, targets, reduction='mean')
        Array(0.25, dtype=float32)
    """
    return _reduce(0.5 * squared_error(predictions, targets), reduction, axis=axis)


@set_module_as('braintools.metric')
def l2_norm(
    predictions: brainstate.typing.ArrayLike,
    targets: Optional[brainstate.typing.ArrayLike] = None,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
) -> brainstate.typing.ArrayLike:
    r"""Compute the L2 norm of the difference between predictions and targets.

    Parameters
    ----------
    predictions : brainstate.typing.ArrayLike
        A vector of arbitrary shape ``[...]``.
    targets : brainstate.typing.ArrayLike, optional
        A vector with the same shape as ``predictions`` (shape equality is
        asserted; no broadcasting). If not provided then it is assumed to be a
        vector of zeros.
    axis : int or tuple of ints, optional
        The dimensions to reduce. If None, the norm is computed over the whole
        (flattened) array and reduced to a scalar.

    Returns
    -------
    brainstate.typing.ArrayLike
        The L2 norm of the differences along ``axis``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> predictions = jnp.array([3.0, 4.0])
        >>> braintools.metric.l2_norm(predictions)
        Array(5., dtype=float32)
    """
    assert u.math.is_float(predictions), 'predictions must be float.'
    if targets is not None:
        # Avoid broadcasting logic for "-" operator.
        assert predictions.shape == targets.shape, 'predictions and targets must have the same shape.'
    errors = predictions - targets if targets is not None else predictions
    return jnp.linalg.norm(errors, axis=axis, ord=2)


@set_module_as('braintools.metric')
def huber_loss(
    predictions: brainstate.typing.ArrayLike,
    targets: Optional[brainstate.typing.ArrayLike] = None,
    delta: Union[float, brainstate.typing.ArrayLike] = 1.,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    reduction: str = 'none',
) -> brainstate.typing.ArrayLike:
    r"""Compute Huber loss combining L1 and L2 properties for robust regression.

    The Huber loss provides a compromise between L1 and L2 losses, being quadratic
    for small errors (like L2) and linear for large errors (like L1). This makes
    it robust to outliers while maintaining smooth gradients near zero, combining
    the best properties of both loss functions.

    The Huber loss is defined as:

    .. math::

        \ell_\delta(a) = \begin{cases}
            \frac{1}{2} a^2 & \text{if } |a| \leq \delta \\
            \delta |a| - \frac{1}{2} \delta^2 & \text{if } |a| > \delta
        \end{cases}

    where :math:`a = y - \hat{y}` is the residual and :math:`\delta` is the threshold.

    Parameters
    ----------
    predictions : brainstate.typing.ArrayLike
        Predicted values with arbitrary shape. Must be floating-point type.
    targets : brainstate.typing.ArrayLike, optional
        Ground truth target values with shape broadcastable to ``predictions``.
        If not provided, targets are assumed to be zeros.
    delta : float or brainstate.typing.ArrayLike, default=1.0
        Threshold parameter that controls the transition between quadratic and
        linear regions. Smaller values make the loss more L1-like (robust but
        less smooth), while larger values make it more L2-like (smooth but
        less robust). May be a ``brainunit.Quantity`` whose units match those
        of the errors (see Notes).
    axis : int or tuple of ints, optional
        Axis or axes along which to reduce when ``reduction`` is ``'mean'`` or
        ``'sum'``. If None, reduction (if any) is over all elements.
    reduction : {'none', 'mean', 'sum'}, default='none'
        Reduction operation to apply:

        - ``'none'``: return element-wise losses with the same shape as
          ``predictions`` (backward-compatible default),
        - ``'mean'``: return the mean of the losses,
        - ``'sum'``: return the sum of the losses.

    Returns
    -------
    brainstate.typing.ArrayLike
        Element-wise Huber losses with the same shape as ``predictions`` when
        ``reduction='none'``; otherwise the reduced value.

    Notes
    -----
    The Huber loss has several important properties:

    - **Robustness**: Linear growth for large errors reduces outlier sensitivity
    - **Smoothness**: Quadratic near zero ensures smooth gradients for optimization
    - **Gradient clipping**: Equivalent to clipping L2 gradients to ``[-delta, delta]``

    The choice of ``delta`` parameter affects the balance:

    - Small ``delta``: More robust, approaches L1 loss
    - Large ``delta``: Less robust, approaches L2 loss
    - ``delta = 1.0``: Common default providing good balance

    This loss is particularly effective for regression with outliers and in
    reinforcement learning for value function approximation.

    **Units:** comparisons are performed with ``brainunit.math`` so that a
    ``brainunit.Quantity`` ``delta`` is handled correctly. When the errors carry
    units, ``delta`` must carry the *same* units as the errors (a dimensionless
    ``delta`` against unit-bearing errors raises a unit-mismatch error). For
    ``Quantity`` inputs the loss carries **squared** units in the quadratic
    region.

    Examples
    --------
    Basic Huber loss computation:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> predictions = jnp.array([1.0, 2.0, 5.0])
        >>> targets = jnp.array([1.1, 1.9, 3.0])  # Last prediction is outlier
        >>> braintools.metric.huber_loss(predictions, targets)
        Array([0.00499997, 0.00499998, 1.5       ], dtype=float32)

    Reduce to a scalar mean:

    .. code-block:: python

        >>> braintools.metric.huber_loss(predictions, targets, reduction='mean')
        Array(0.50333333, dtype=float32)

    See Also
    --------
    braintools.metric.absolute_error : Pure L1 loss
    braintools.metric.squared_error : Pure L2 loss  
    braintools.metric.log_cosh : Smooth alternative to Huber loss

    References
    ----------
    .. [1] Huber, Peter J. "Robust estimation of a location parameter." 
           The annals of mathematical statistics 35.1 (1964): 73-101.
    .. [2] Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. 
           "The elements of statistical learning: data mining, inference, 
           and prediction." Springer, 2009.
    """
    assert u.math.is_float(predictions), 'predictions must be float.'
    errors = (predictions - targets) if (targets is not None) else predictions
    # 0.5 * err^2                  if |err| <= d
    # 0.5 * d^2 + d * (|err| - d)  if |err| > d
    # Use ``brainunit.math`` so Quantity errors / Quantity delta compare safely.
    abs_errors = u.math.abs(errors)
    quadratic = u.math.minimum(abs_errors, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_errors - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return _reduce(loss, reduction, axis=axis)


@set_module_as('braintools.metric')
def log_cosh(
    predictions: brainstate.typing.ArrayLike,
    targets: Optional[brainstate.typing.ArrayLike] = None,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    reduction: str = 'none',
) -> brainstate.typing.ArrayLike:
    r"""Calculate the log-cosh loss for a set of predictions.

    ``log(cosh(x))`` is approximately ``(x ** 2) / 2`` for small ``x`` and
    ``abs(x) - log(2)`` for large ``x``. It is a twice differentiable
    alternative to the Huber loss.

    Parameters
    ----------
    predictions : brainstate.typing.ArrayLike
        A vector of arbitrary shape ``[...]``.
    targets : brainstate.typing.ArrayLike, optional
        A vector with the same shape as ``predictions`` (no broadcasting). If
        not provided then it is assumed to be a vector of zeros.
    axis : int or tuple of ints, optional
        Axis or axes along which to reduce when ``reduction`` is ``'mean'`` or
        ``'sum'``. If None, reduction (if any) is over all elements.
    reduction : {'none', 'mean', 'sum'}, default='none'
        Reduction operation to apply:

        - ``'none'``: return element-wise losses with the same shape as
          ``predictions`` (backward-compatible default),
        - ``'mean'``: return the mean of the losses,
        - ``'sum'``: return the sum of the losses.

    Returns
    -------
    brainstate.typing.ArrayLike
        The log-cosh loss with the same shape as ``predictions`` when
        ``reduction='none'``; otherwise the reduced value.

    Notes
    -----
    The loss is computed with the numerically stable, gradient-safe identity

    .. math::

        \log(\cosh(x)) = |x| + \operatorname{softplus}(-2 |x|) - \log 2,

    which avoids overflow in ``cosh``/``exp`` and yields finite gradients even
    for very large ``|x|`` (e.g. ``|x| = 500``).

    References
    ----------
    .. [1] Chen et al., 2019. https://openreview.net/pdf?id=rkglvsC9Ym

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> predictions = jnp.array([0.0, 1.0, -1.0])
        >>> braintools.metric.log_cosh(predictions)
        Array([0.        , 0.43378806, 0.43378806], dtype=float32)
    """
    assert u.math.is_float(predictions), 'predictions must be float.'
    errors = (predictions - targets) if (targets is not None) else predictions
    # Numerically stable, gradient-safe form of log(cosh(x)):
    #   log(cosh(x)) = |x| + softplus(-2|x|) - log(2)
    abs_errors = jnp.abs(errors)
    loss = abs_errors + jax.nn.softplus(-2.0 * abs_errors) - jnp.log(2.0).astype(errors.dtype)
    return _reduce(loss, reduction, axis=axis)


@set_module_as('braintools.metric')
def cosine_similarity(
    predictions: brainstate.typing.ArrayLike,
    targets: brainstate.typing.ArrayLike,
    epsilon: float = 1e-8,
) -> brainstate.typing.ArrayLike:
    r"""Compute cosine similarity between predicted and target vectors.

    Calculates the cosine of the angle between vectors, providing a measure of
    similarity that is independent of vector magnitude. This metric is particularly
    useful for comparing direction or orientation of high-dimensional vectors,
    commonly used in natural language processing, computer vision, and
    recommendation systems.

    The cosine similarity is defined as:

    .. math::

        \text{cosine\_similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{||\mathbf{u}|| ||\mathbf{v}||}

    where :math:`\mathbf{u}` and :math:`\mathbf{v}` are vectors, and :math:`||\cdot||`
    denotes the L2 (Euclidean) norm.

    Parameters
    ----------
    predictions : brainstate.typing.ArrayLike
        Predicted vectors with shape ``(..., dim)`` where ``dim`` is the vector
        dimension. Must be floating-point type.
    targets : brainstate.typing.ArrayLike
        Ground truth target vectors with shape ``(..., dim)`` matching the
        shape of ``predictions``. Must be floating-point type.
    epsilon : float, default=1e-8
        Minimum norm used as a floor in the denominator to prevent division by
        zero when computing norms (via :func:`safe_norm`). This provides
        numerical stability for zero or near-zero vectors and keeps both the
        value and the gradient finite for zero-vector inputs. The default of
        ``1e-8`` (rather than ``0``) ensures the zero-vector safety actually
        takes effect.

    Returns
    -------
    brainstate.typing.ArrayLike
        Cosine similarity values with shape ``(...,)`` where the last dimension
        has been reduced. Values range from -1 (opposite directions) to 1
        (same direction), with 0 indicating orthogonal vectors.

    Notes
    -----
    Properties of cosine similarity:
    
    - **Scale invariant**: Only depends on vector direction, not magnitude
    - **Bounded**: Values always in [-1, 1] range
    - **Symmetric**: sim(u, v) = sim(v, u)
    - **Geometric interpretation**: Cosine of angle between vectors
    
    Common use cases:
    
    - **Text similarity**: Comparing document embeddings
    - **Image features**: Comparing visual feature vectors  
    - **Recommendation**: Finding similar user/item profiles
    - **Clustering**: Measuring vector similarity in high dimensions

    The function handles zero vectors gracefully using the ``epsilon`` parameter
    to floor the denominator, which keeps both the value and the gradient finite
    for zero-vector inputs.

    Examples
    --------
    Basic cosine similarity:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> # Two 3D vectors, same direction, different magnitude
        >>> pred = jnp.array([1.0, 2.0, 3.0])
        >>> target = jnp.array([2.0, 4.0, 6.0])
        >>> braintools.metric.cosine_similarity(pred, target)
        Array(1., dtype=float32)

    Batch computation (all orthogonal pairs):

    .. code-block:: python

        >>> pred_batch = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        >>> target_batch = jnp.array([[0.0, 1.0], [1.0, 0.0], [1.0, -1.0]])
        >>> braintools.metric.cosine_similarity(pred_batch, target_batch)
        Array([0., 0., 0.], dtype=float32)

    Handling zero vectors (finite result):

    .. code-block:: python

        >>> zero_vec = jnp.array([0.0, 0.0, 0.0])
        >>> normal_vec = jnp.array([1.0, 2.0, 3.0])
        >>> braintools.metric.cosine_similarity(zero_vec, normal_vec)
        Array(0., dtype=float32)

    See Also
    --------
    braintools.metric.cosine_distance : 1 - cosine_similarity
    jax.numpy.dot : Dot product computation
    jax.numpy.linalg.norm : Vector norm computation

    References
    ----------
    .. [1] "Cosine similarity." Wikipedia, The Free Encyclopedia. Accessed 2024.
           https://en.wikipedia.org/wiki/Cosine_similarity
    .. [2] Manning, Christopher D., Prabhakar Raghavan, and Hinrich Schütze. 
           "Introduction to information retrieval." Cambridge university press, 2008.
    """
    assert u.math.is_float(predictions), 'predictions must be float.'
    assert u.math.is_float(targets), 'targets must be float.'
    # vectorize norm fn, to treat all dimensions except the last as batch dims.
    batched_norm_fn = jnp.vectorize(safe_norm, signature='(k)->()', excluded={1})
    # normalise the last dimension of targets and predictions.
    unit_targets = targets / jnp.expand_dims(
        batched_norm_fn(targets, epsilon), axis=-1)
    unit_predictions = predictions / jnp.expand_dims(
        batched_norm_fn(predictions, epsilon), axis=-1)
    # return cosine similarity.
    return jnp.sum(unit_targets * unit_predictions, axis=-1)


@set_module_as('braintools.metric')
def cosine_distance(
    predictions: brainstate.typing.ArrayLike,
    targets: brainstate.typing.ArrayLike,
    epsilon: float = 1e-8,
) -> brainstate.typing.ArrayLike:
    r"""Compute the cosine distance between targets and predictions.

    The cosine **distance**, implemented here, measures the **dissimilarity**
    of two vectors as the opposite of cosine **similarity**: :math:`1 - \cos(\theta)`.

    Parameters
    ----------
    predictions : brainstate.typing.ArrayLike
        The predicted vectors, with shape ``[..., dim]``.
    targets : brainstate.typing.ArrayLike
        Ground truth target vectors, with shape ``[..., dim]``.
    epsilon : float, default=1e-8
        Minimum norm used as a floor in the denominator of the cosine
        similarity. The default of ``1e-8`` (rather than ``0``) ensures
        zero-vector safety actually takes effect.

    Returns
    -------
    brainstate.typing.ArrayLike
        Cosine distances, with shape ``[...]``.

    References
    ----------
    .. [1] "Cosine similarity." Wikipedia, The Free Encyclopedia.
           https://en.wikipedia.org/wiki/Cosine_similarity

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> pred = jnp.array([1.0, 0.0])
        >>> target = jnp.array([0.0, 1.0])
        >>> braintools.metric.cosine_distance(pred, target)
        Array(1., dtype=float32)
    """
    assert u.math.is_float(predictions), 'predictions must be float.'
    assert u.math.is_float(targets), 'targets must be float.'
    # cosine distance = 1 - cosine similarity.
    return 1. - cosine_similarity(predictions, targets, epsilon=epsilon)
