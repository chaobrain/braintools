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

"""
Ranking losses.

A ranking loss is a differentiable function that expresses the cost of a ranking
induced by item scores compared to a ranking induced from relevance labels.

Ranking losses are designed to operate on the last dimension of its inputs. The
leading dimensions are considered batch dimensions.

Standalone usage:

>>> import braintools as braintools
>>> scores = jnp.array([2., 1., 3.])
>>> labels = jnp.array([1., 0., 0.])
>>> loss = braintools.metric.ranking_softmax_loss(scores, labels)
>>> print(f"{loss:.3f}")
1.408

Usage with a batch of data and a mask to indicate valid items.

>>> scores = jnp.array([[2., 1., 0.], [1., 0.5, 1.5]])
>>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
>>> where = jnp.array([[True, True, False], [True, True, True]])
>>> loss = braintools.metric.ranking_softmax_loss(scores, labels, where=where)
>>> print(f"{loss:.3f}")
0.497
"""

from typing import Callable, Optional

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from braintools._misc import set_module_as

__all__ = [
    'ranking_softmax_loss',
]


def _safe_reduce(
    a: brainstate.typing.ArrayLike,
    where: Optional[brainstate.typing.ArrayLike] = None,
    reduce_fn: Optional[Callable[..., brainstate.typing.ArrayLike]] = None,
) -> brainstate.typing.ArrayLike:
    """Safely reduce array values while preventing NaN outputs from masking.

    Performs reduction operations on arrays with proper handling of masked elements
    to prevent NaN values in the output. This is particularly important for ranking
    losses where all elements might be masked, leading to undefined reductions.

    Parameters
    ----------
    a : brainstate.typing.ArrayLike
        Input array to be reduced. Can be of any shape compatible with the
        reduction function.
    where : brainstate.typing.ArrayLike, optional
        Boolean mask array with the same shape as ``a`` indicating which elements
        should be included in the reduction. Elements where ``where`` is False
        are excluded from the computation.
    reduce_fn : callable, optional
        Reduction function to apply (e.g., ``jax.numpy.mean``, ``jax.numpy.sum``).
        If None, no reduction is performed and the masked array is returned.

    Returns
    -------
    brainstate.typing.ArrayLike
        Reduced array or masked array if no reduction function is provided.
        For mean reductions, NaN values are replaced with 0.0 when they result
        from empty masks rather than invalid input data.

    Notes
    -----
    Special handling for different scenarios:
    
    - **Mean reduction with empty mask**: Returns 0.0 instead of NaN when all
      elements are masked but input contains no NaN values
    - **No reduction function**: Masked elements are set to 0.0 to enable
      consistent manual reduction operations
    - **NaN preservation**: Original NaN values in input data are preserved
      and not masked

    This function is primarily used internally by ranking loss functions to
    ensure stable behavior when dealing with sparse or heavily masked data.

    Examples
    --------
    Safe mean reduction with partial masking:

    >>> import jax.numpy as jnp
    >>> a = jnp.array([1.0, 2.0, 3.0, 4.0])
    >>> where = jnp.array([True, True, False, False])
    >>> result = _safe_reduce(a, where=where, reduce_fn=jnp.mean)
    >>> # Returns 1.5 (mean of first two elements)

    Safe mean reduction with complete masking:

    >>> where_empty = jnp.array([False, False, False, False])
    >>> result = _safe_reduce(a, where=where_empty, reduce_fn=jnp.mean)
    >>> # Returns 0.0 instead of NaN

    No reduction with masking:

    >>> result = _safe_reduce(a, where=where, reduce_fn=None)
    >>> # Returns [1.0, 2.0, 0.0, 0.0] (masked elements set to 0)
    """
    # Reduce values if there is a reduce_fn, otherwise keep the values as-is.
    output = reduce_fn(a, where=where) if reduce_fn is not None else a

    if reduce_fn is jnp.mean:
        # For mean reduction, we have to check whether the input contains any NaN
        # values, to ensure that masked mean reduction does not hide them (see
        # below).
        is_input_valid = jnp.logical_not(jnp.any(jnp.isnan(a)))

        # The standard jnp.mean implementation returns NaN if `where` is False
        # everywhere. This can happen in our case, e.g. pairwise losses with no
        # valid pairs. Instead, we prefer that the loss returns 0 in these cases.
        # Note that this only hides those NaN values if the input did not contain
        # any NaN values. Otherwise it just returns the output as-is.
        output = jnp.where(jnp.isnan(output) & is_input_valid, 0.0, output)

    if reduce_fn is None and where is not None:
        # When there is no reduce_fn (i.e. we are returning an unreduced
        # loss/metric), set the values of `a` to 0 for invalid (masked) items.
        # This makes sure that manual sum reduction on an unreduced loss works as
        # expected:
        # `jnp.sum(loss_fn(reduce_fn=None)) == loss_fn(reduce_fn=jnp.sum)`
        output = jnp.where(where, output, 0.0)

    return output


@set_module_as('braintools.metric')
def ranking_softmax_loss(
    logits: brainstate.typing.ArrayLike,
    labels: brainstate.typing.ArrayLike,
    *,
    where: Optional[brainstate.typing.ArrayLike] = None,
    weights: Optional[brainstate.typing.ArrayLike] = None,
    reduce_fn: Optional[Callable[..., brainstate.typing.ArrayLike]] = jnp.mean
) -> brainstate.typing.ArrayLike:
    r"""Compute ranking softmax loss for learning-to-rank applications.

    Calculates a differentiable ranking loss that measures the cost of a ranking
    induced by item scores compared to ground truth relevance labels. This loss
    is particularly effective for information retrieval, recommendation systems,
    and other ranking tasks where the goal is to prioritize relevant items.

    The loss is computed as the negative log-likelihood of the softmax distribution
    over items, weighted by their relevance labels:

    .. math::

        \ell(s, y) = -\sum_{i=1}^{n} y_i \log \frac{\exp(s_i)}{\sum_{j=1}^{n} \exp(s_j)}

    where :math:`s_i` are the logit scores, :math:`y_i` are the relevance labels,
    and :math:`n` is the number of items in the list.

    Parameters
    ----------
    logits : brainstate.typing.ArrayLike
        Predicted scores for each item with shape ``(..., list_size)``.
        Higher scores should indicate higher relevance. The function operates
        on the last dimension, treating leading dimensions as batch dimensions.
    labels : brainstate.typing.ArrayLike
        Ground truth relevance labels with shape ``(..., list_size)``.
        Typically non-negative values where higher values indicate greater
        relevance. Labels are automatically normalized by the softmax operation.
    where : brainstate.typing.ArrayLike, optional
        Boolean mask with shape ``(..., list_size)`` indicating valid items.
        Items where ``where`` is False are excluded from loss computation.
        This is useful for handling variable-length lists or missing data.
    weights : brainstate.typing.ArrayLike, optional
        Per-item weights with shape ``(..., list_size)`` for emphasizing
        certain items in the loss calculation. Applied to labels before
        computing the softmax cross-entropy.
    reduce_fn : callable, optional
        Function to reduce loss values across batch dimensions. Common choices:
        
        - ``jax.numpy.mean`` (default): Average loss across batches
        - ``jax.numpy.sum``: Sum loss across batches  
        - ``None``: Return unreduced per-batch losses

    Returns
    -------
    brainstate.typing.ArrayLike
        Ranking softmax loss. Shape depends on ``reduce_fn``:
        
        - If ``reduce_fn`` is not None: scalar loss value
        - If ``reduce_fn`` is None: array with shape ``(batch_dims,)``

    Notes
    -----
    This loss function implements a probabilistic approach to ranking where:
    
    - Items with higher relevance labels should receive higher probability mass
    - The softmax operation ensures valid probability distributions
    - Masked items (where ``where`` is False) are effectively ignored
    - The loss is differentiable w.r.t. logits, enabling gradient-based optimization

    The function handles edge cases gracefully:
    
    - Empty masks (all items invalid) return 0.0 instead of NaN
    - Numerical stability is maintained through log-softmax computation
    - Mixed data types are handled by casting labels to match logit precision

    Examples
    --------
    Basic ranking loss with single query:

    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> # Scores for 3 items
    >>> logits = jnp.array([2.0, 1.0, 3.0])  
    >>> # Relevance: item 3 most relevant, item 1 second, item 2 least
    >>> labels = jnp.array([1.0, 0.0, 2.0])
    >>> loss = braintools.metric.ranking_softmax_loss(logits, labels)
    >>> print(f"Loss: {loss:.4f}")

    Batch processing with masking:

    >>> # Batch of 2 queries with 3 items each
    >>> logits = jnp.array([[2.0, 1.0, 0.0], [1.0, 0.5, 1.5]])
    >>> labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    >>> # Second query only has first 2 items valid
    >>> where = jnp.array([[True, True, False], [True, True, True]])
    >>> loss = braintools.metric.ranking_softmax_loss(logits, labels, where=where)
    >>> print(f"Batch loss: {loss:.4f}")

    Per-item weighting:

    >>> weights = jnp.array([1.0, 2.0, 1.0])  # Emphasize middle item
    >>> loss = braintools.metric.ranking_softmax_loss(logits[0], labels[0], weights=weights)
    >>> print(f"Weighted loss: {loss:.4f}")

    Unreduced losses for analysis:

    >>> batch_losses = braintools.metric.ranking_softmax_loss(
    ...     logits, labels, where=where, reduce_fn=None
    ... )
    >>> print(f"Individual losses: {batch_losses}")

    See Also
    --------
    jax.nn.log_softmax : Underlying log-softmax computation
    jax.nn.softmax_cross_entropy : Related cross-entropy function
    braintools.metric.sigmoid_binary_cross_entropy : Alternative for binary relevance

    References
    ----------
    .. [1] Liu, Tie-Yan. "Learning to rank for information retrieval." 
           Foundations and Trends in Information Retrieval 3.3 (2009): 225-331.
    .. [2] Cao, Zhe, et al. "Learning to rank: from pairwise approach to listwise 
           approach." Proceedings of the 24th international conference on Machine 
           learning. 2007.
    """
    assert u.math.is_float(logits), "logits must be a float type."
    labels = labels.astype(logits.dtype)

    # Applies mask so that masked elements do not count towards the loss.
    if where is not None:
        labels = jnp.where(where, labels, jnp.zeros_like(labels))
        logits = jnp.where(where, logits, -jnp.ones_like(logits) * jnp.inf)

    # Apply weights to labels.
    if weights is not None:
        labels *= weights

    # Scales labels and logits to match the cross entropy loss.
    logits_log_softmax = jax.nn.log_softmax(logits, axis=-1)

    # Computes per-element cross entropy.
    softmax_cross_entropy = labels * logits_log_softmax

    # Reduces softmax cross-entropy loss.
    loss = -jnp.sum(softmax_cross_entropy, axis=-1, where=where)

    # Setup mask to ignore lists with only invalid items in reduce_fn.
    if where is not None:
        where = jnp.any(where, axis=-1)

    return _safe_reduce(loss, where=where, reduce_fn=reduce_fn)
