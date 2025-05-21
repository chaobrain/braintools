# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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
import jax.numpy as jnp

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


def safe_norm(x: brainstate.typing.ArrayLike,
              min_norm,
              ord: Optional[Union[int, float, str]] = None,  # pylint: disable=redefined-builtin
              axis: Union[None, tuple[int, ...], int] = None,
              keepdims: bool = False) -> brainstate.typing.ArrayLike:
    """Returns jnp.maximum(jnp.linalg.norm(x), min_norm) with correct gradients.

    The gradients of `jnp.maximum(jnp.linalg.norm(x), min_norm)` at 0.0 is `NaN`,
    because jax will evaluate both branches of the `jnp.maximum`. This function
    will instead return the correct gradient of 0.0 also in such setting.

    Args:
      x: jax array.
      min_norm: lower bound for the returned norm.
      ord: {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional. Order of the norm.
        inf means numpy’s inf object. The default is None.
      axis: {None, int, 2-tuple of ints}, optional. If axis is an integer, it
        specifies the axis of x along which to compute the vector norms. If axis
        is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix
        norms of these matrices are computed. If axis is None then either a vector
        norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned. The
        default is None.
      keepdims: bool, optional. If this is set to True, the axes which are normed
        over are left in the result as dimensions with size one. With this option
        the result will broadcast correctly against the original x.

    Returns:
      The safe norm of the input vector, accounting for correct gradient.
    """
    norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=True)
    x = jnp.where(norm <= min_norm, jnp.ones_like(x), x)
    norm = jnp.squeeze(norm, axis=axis) if not keepdims else norm
    masked_norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    return jnp.where(norm <= min_norm, min_norm, masked_norm)


def squared_error(
    predictions: brainstate.typing.ArrayLike,
    targets: Optional[brainstate.typing.ArrayLike] = None,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    reduction: str = 'none',
) -> brainstate.typing.ArrayLike:
    """Calculates the squared error for a set of predictions.

    Mean Squared Error can be computed as ``squared_error(a, b, reduction='mean')``.

    Note: l2_loss = 0.5 * squared_error, where the 0.5 term is standard in
    "Pattern Recognition and Machine Learning" by Bishop, but not
    "The Elements of Statistical Learning" by Tibshirani.

    References:
      [Chris Bishop, 2006](https://bit.ly/3eeP0ga)

    Args:
      predictions: a vector of arbitrary shape `[...]`.
      targets: a vector with shape broadcastable to that of `predictions`;
        if not provided then it is assumed to be a vector of zeros.
      axis: the dimensions to reduce. If `None`, the loss is reduced to a scalar.
      reduction: the reduction operation to apply to the output. One of
        'none', 'mean', 'sum'. Default is 'none'.

    Returns:
      elementwise squared differences, with same shape as `predictions`.
    """
    assert u.math.is_float(predictions), 'predictions must be float.'
    if targets is not None:
        # Avoid broadcasting logic for "-" operator.
        assert predictions.shape == targets.shape, 'predictions and targets must have the same shape.'
    errors = predictions - targets if targets is not None else predictions
    # return errors ** 2
    return _reduce(errors ** 2, reduction, axis=axis)


def absolute_error(
    predictions: brainstate.typing.ArrayLike,
    targets: Optional[brainstate.typing.ArrayLike] = None,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    reduction: str = 'mean',
) -> brainstate.typing.ArrayLike:
    """Calculates the absolute error for a set of predictions.

    Mean Absolute Error can be computed as absolute_error(a, b).mean().

    Args:
      predictions: a vector of arbitrary shape `[...]`.
      targets: a vector with shape broadcastable to that of `predictions`;
        if not provided then it is assumed to be a vector of zeros.
      axis: the dimensions to reduce. If `None`, the loss is reduced to a scalar.
      reduction: the reduction operation to apply to the output. One of
        'none', 'mean', 'sum'. Default is 'mean'.

    Returns:
      elementwise absolute differences, with same shape as `predictions`.
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

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Supports real-valued and complex-valued inputs.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(*)`, same shape as the input.

    Examples::

        >>> import brainstate as bst
        >>> loss = nn.L1Loss()
        >>> input = bst.random.randn(3, 5)
        >>> target = bst.random.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, reduction: str = 'mean') -> None:
        self.reduction = reduction

    def update(self,
               input: brainstate.typing.ArrayLike,
               target: brainstate.typing.ArrayLike) -> brainstate.typing.ArrayLike:
        return l1_loss(input, target, reduction=self.reduction)


def l1_loss(logits: brainstate.typing.ArrayLike,
            targets: brainstate.typing.ArrayLike,
            reduction: str = 'sum'):
    r"""Creates a criterion that measures the mean absolute error (MAE) between each element in
    the logits :math:`x` and targets :math:`y`. It is useful in regression problems.

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

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Supports real-valued and complex-valued inputs.

    Parameters
    ----------
    logits : bst.typing.ArrayLike
      :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
    targets : bst.typing.ArrayLike
      :math:`(N, *)`, same shape as the input.
    reduction : str
      Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
      Default: ``'mean'``.
      - ``'none'``: no reduction will be applied,
      - ``'mean'``: the sum of the output will be divided by the number of elements in the output,
      - ``'sum'``: the output will be summed. Note: :attr:`size_average`

    Returns
    -------
    output : scalar.
      If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same shape as the input.
    """

    diff = (logits - targets).reshape((logits.shape[0], -1))
    norm = jnp.linalg.norm(diff, ord=1, axis=1, keepdims=False)
    return _reduce(outputs=norm, reduction=reduction)


def l2_loss(
    predictions: brainstate.typing.ArrayLike,
    targets: Optional[brainstate.typing.ArrayLike] = None,
) -> brainstate.typing.ArrayLike:
    """Calculates the L2 loss for a set of predictions.

    Note: the 0.5 term is standard in "Pattern Recognition and Machine Learning"
    by Bishop, but not "The Elements of Statistical Learning" by Tibshirani.

    References:
      [Chris Bishop, 2006](https://bit.ly/3eeP0ga)

    Args:
      predictions: a vector of arbitrary shape `[...]`.
      targets: a vector with shape broadcastable to that of `predictions`;
        if not provided then it is assumed to be a vector of zeros.

    Returns:
      elementwise squared differences, with same shape as `predictions`.
    """
    return 0.5 * squared_error(predictions, targets)


def l2_norm(
    predictions: brainstate.typing.ArrayLike,
    targets: Optional[brainstate.typing.ArrayLike] = None,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
) -> brainstate.typing.ArrayLike:
    """Computes the L2 norm of the difference between predictions and targets.

    Args:
      predictions: a vector of arbitrary shape `[...]`.
      targets: a vector with shape broadcastable to that of `predictions`;
        if not provided then it is assumed to be a vector of zeros.
      axis: the dimensions to reduce. If `None`, the loss is reduced to a scalar.

    Returns:
      elementwise l2 norm of the differences, with same shape as `predictions`.
    """
    assert u.math.is_float(predictions), 'predictions must be float.'
    if targets is not None:
        # Avoid broadcasting logic for "-" operator.
        assert predictions.shape == targets.shape, 'predictions and targets must have the same shape.'
    errors = predictions - targets if targets is not None else predictions
    return jnp.linalg.norm(errors, axis=axis, ord=2)


def huber_loss(
    predictions: brainstate.typing.ArrayLike,
    targets: Optional[brainstate.typing.ArrayLike] = None,
    delta: float = 1.
) -> brainstate.typing.ArrayLike:
    """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.

    If gradient descent is applied to the `huber loss`, it is equivalent to
    clipping gradients of an `l2_loss` to `[-delta, delta]` in the backward pass.

    References:
      [Huber, 1964](www.projecteuclid.org/download/pdf_1/euclid.aoms/1177703732)

    Args:
      predictions: a vector of arbitrary shape `[...]`.
      targets: a vector with shape broadcastable to that of `predictions`;
        if not provided then it is assumed to be a vector of zeros.
      delta: the bounds for the huber loss transformation, defaults at 1.

    Returns:
      elementwise huber losses, with the same shape of `predictions`.
    """
    assert u.math.is_float(predictions), 'predictions must be float.'
    errors = (predictions - targets) if (targets is not None) else predictions
    # 0.5 * err^2                  if |err| <= d
    # 0.5 * d^2 + d * (|err| - d)  if |err| > d
    abs_errors = jnp.abs(errors)
    quadratic = jnp.minimum(abs_errors, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_errors - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def log_cosh(
    predictions: brainstate.typing.ArrayLike,
    targets: Optional[brainstate.typing.ArrayLike] = None,
) -> brainstate.typing.ArrayLike:
    """Calculates the log-cosh loss for a set of predictions.

    log(cosh(x)) is approximately `(x**2) / 2` for small x and `abs(x) - log(2)`
    for large x.  It is a twice differentiable alternative to the Huber loss.

    References:
      [Chen et al, 2019](https://openreview.net/pdf?id=rkglvsC9Ym)

    Args:
      predictions: a vector of arbitrary shape `[...]`.
      targets: a vector with shape broadcastable to that of `predictions`;
        if not provided then it is assumed to be a vector of zeros.

    Returns:
      the log-cosh loss, with same shape as `predictions`.
    """
    assert u.math.is_float(predictions), 'predictions must be float.'
    errors = (predictions - targets) if (targets is not None) else predictions
    # log(cosh(x)) = log((exp(x) + exp(-x))/2) = log(exp(x) + exp(-x)) - log(2)
    return jnp.logaddexp(errors, -errors) - jnp.log(2.0).astype(errors.dtype)


def cosine_similarity(
    predictions: brainstate.typing.ArrayLike,
    targets: brainstate.typing.ArrayLike,
    epsilon: float = 0.,
) -> brainstate.typing.ArrayLike:
    r"""Computes the cosine similarity between targets and predictions.

    The cosine **similarity** is a measure of similarity between vectors defined
    as the cosine of the angle between them, which is also the inner product of
    those vectors normalized to have unit norm.

    References:
      [Wikipedia, 2021](https://en.wikipedia.org/wiki/Cosine_similarity)

    Args:
      predictions: The predicted vectors, with shape `[..., dim]`.
      targets: Ground truth target vectors, with shape `[..., dim]`.
      epsilon: minimum norm for terms in the denominator of the cosine similarity.

    Returns:
      cosine similarity measures, with shape `[...]`.
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


def cosine_distance(
    predictions: brainstate.typing.ArrayLike,
    targets: brainstate.typing.ArrayLike,
    epsilon: float = 0.,
) -> brainstate.typing.ArrayLike:
    r"""Computes the cosine distance between targets and predictions.

    The cosine **distance**, implemented here, measures the **dissimilarity**
    of two vectors as the opposite of cosine **similarity**: `1 - cos(\theta)`.

    References:
      [Wikipedia, 2021](https://en.wikipedia.org/wiki/Cosine_similarity)

    Args:
      predictions: The predicted vectors, with shape `[..., dim]`.
      targets: Ground truth target vectors, with shape `[..., dim]`.
      epsilon: minimum norm for terms in the denominator of the cosine similarity.

    Returns:
      cosine distances, with shape `[...]`.
    """
    assert u.math.is_float(predictions), 'predictions must be float.'
    assert u.math.is_float(targets), 'targets must be float.'
    # cosine distance = 1 - cosine similarity.
    return 1. - cosine_similarity(predictions, targets, epsilon=epsilon)
