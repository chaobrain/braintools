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

"""Smoothing functions."""

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from braintools._misc import set_module_as

__all__ = ['smooth_labels']


@set_module_as('braintools.metric')
def smooth_labels(
    labels: brainstate.typing.ArrayLike,
    alpha: float,
) -> jax.Array:
    r"""Apply label smoothing regularization to one-hot encoded labels.

    Label smoothing is a regularization technique that prevents neural networks
    from becoming overconfident in their predictions by introducing controlled
    uncertainty in the training labels. This technique replaces hard targets
    with a weighted mixture of the original one-hot labels and a uniform
    distribution over all classes.

    The smoothing transformation is defined as:

    .. math::

        \tilde{y}_k = (1 - \alpha) y_k + \frac{\alpha}{K}

    where :math:`y_k` is the original label for class :math:`k`, :math:`\alpha`
    is the smoothing parameter, :math:`K` is the number of classes, and
    :math:`\tilde{y}_k` is the smoothed label.

    Parameters
    ----------
    labels : brainstate.typing.ArrayLike
        One-hot encoded labels with shape ``(..., num_classes)`` where the last
        dimension represents class probabilities. Must be floating-point type.
        Each row should contain exactly one 1.0 and zeros elsewhere for proper
        one-hot encoding.
    alpha : float
        Smoothing parameter in the range [0, 1] controlling the degree of smoothing:

        - ``alpha = 0.0``: No smoothing (original hard labels)
        - ``alpha = 0.1``: Light smoothing (common choice)
        - ``alpha = 1.0``: Maximum smoothing (uniform distribution)

        Typical values range from 0.05 to 0.2 depending on the task complexity.
        Values outside ``[0, 1]`` raise a :class:`ValueError`.

    Returns
    -------
    jax.Array
        Smoothed label distribution with the same shape as input. Each row sums
        to 1.0 **provided the corresponding input row is itself a valid
        probability distribution** (i.e. its entries sum to 1.0); see Notes.

    Raises
    ------
    TypeError
        If ``labels`` is not a floating-point array.
    ValueError
        If ``alpha`` is outside the closed interval ``[0, 1]``.

    Notes
    -----
    **Row-sum precondition.** The smoothing is
    :math:`\tilde{y} = (1 - \alpha) y + \alpha / K`. Summing over the :math:`K`
    classes gives :math:`(1 - \alpha) \sum_k y_k + \alpha`. This equals 1.0
    **only when** :math:`\sum_k y_k = 1` for the input row (e.g. proper one-hot
    or probability rows). If the input rows do not sum to 1, the smoothed rows
    will not sum to 1 either; this function does not normalize the input.

    ``alpha`` is validated to lie in ``[0, 1]``. Note that ``alpha`` is treated
    as a static Python float; passing a traced JAX value will raise during the
    bounds check, so keep ``alpha`` concrete (or mark it static under
    ``jax.jit``).

    Label smoothing provides several benefits:
    
    - **Improved calibration**: Reduces overconfident predictions
    - **Better generalization**: Acts as regularization to prevent overfitting
    - **Robustness**: Less sensitive to label noise and annotation errors
    - **Gradient stability**: Provides more stable training dynamics
    
    The technique is particularly effective for:
    
    - Image classification with large numbers of classes
    - Tasks with potential label ambiguity or noise
    - Training very deep networks prone to overconfidence
    - Knowledge distillation scenarios
    
    Common usage patterns:
    
    - Use with cross-entropy loss for classification
    - Combine with other regularization techniques (dropout, weight decay)
    - Tune alpha based on validation performance

    Examples
    --------
    Basic label smoothing for 3-class classification:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> # One-hot labels for 2 samples, 3 classes
        >>> labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> braintools.metric.smooth_labels(labels, alpha=0.1)
        Array([[0.93333334, 0.03333334, 0.03333334],
               [0.03333334, 0.93333334, 0.03333334]], dtype=float32)

    Verify probability distribution properties for valid one-hot inputs:

    .. code-block:: python

        >>> smoothed = braintools.metric.smooth_labels(jnp.eye(4), alpha=0.2)
        >>> bool(jnp.allclose(jnp.sum(smoothed, axis=1), 1.0))
        True
        >>> bool(jnp.all(smoothed >= 0))
        True

    See Also
    --------
    braintools.metric.sigmoid_binary_cross_entropy : Binary classification loss
    jax.nn.softmax_cross_entropy : Standard cross-entropy with smoothed labels
    jax.numpy.eye : Create one-hot encoded labels

    References
    ----------
    .. [1] Müller, Rafael, Simon Kornblith, and Geoffrey E. Hinton. 
           "When does label smoothing help?." Advances in neural information
           processing systems 32 (2019): 2234-2243.
           https://arxiv.org/pdf/1906.02629.pdf
    .. [2] Szegedy, Christian, et al. "Rethinking the inception architecture for 
           computer vision." Proceedings of the IEEE conference on computer vision 
           and pattern recognition. 2016.
    .. [3] Pereyra, Gabriel, et al. "Regularizing neural networks by penalizing 
           confident output distributions." arXiv preprint arXiv:1701.06548 (2017).
    """
    if not u.math.is_float(labels):
        raise TypeError('labels should be of float type.')
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f'alpha must be in the range [0, 1], but got {alpha}.')
    num_categories = labels.shape[-1]
    return (1.0 - alpha) * labels + alpha / num_categories
