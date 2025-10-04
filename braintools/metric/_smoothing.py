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
import jax.numpy as jnp

from braintools._misc import set_module_as

__all__ = ['smooth_labels']


@set_module_as('braintools.metric')
def smooth_labels(
    labels: brainstate.typing.ArrayLike,
    alpha: float,
) -> jnp.ndarray:
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

    Returns
    -------
    jax.Array
        Smoothed label distribution with the same shape as input. Each row sums
        to 1.0 and contains the smoothed probability distribution over classes.

    Notes
    -----
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

    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> # One-hot labels for 2 samples, 3 classes
    >>> labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> smoothed = braintools.metric.smooth_labels(labels, alpha=0.1)
    >>> print("Original:")
    >>> print(labels)
    >>> print("Smoothed:")
    >>> print(smoothed)
    [[0.93333334 0.03333333 0.03333333]
     [0.03333333 0.93333334 0.03333333]]

    Effect of different smoothing parameters:

    >>> single_label = jnp.array([[1.0, 0.0, 0.0]])
    >>> # Light smoothing
    >>> light = braintools.metric.smooth_labels(single_label, alpha=0.05)
    >>> print(f"Light smoothing (α=0.05): {light[0]}")
    >>> # Moderate smoothing  
    >>> moderate = braintools.metric.smooth_labels(single_label, alpha=0.1)
    >>> print(f"Moderate smoothing (α=0.1): {moderate[0]}")
    >>> # Heavy smoothing
    >>> heavy = braintools.metric.smooth_labels(single_label, alpha=0.5)
    >>> print(f"Heavy smoothing (α=0.5): {heavy[0]}")

    Batch processing with different numbers of classes:

    >>> # 5-class problem
    >>> labels_5class = jnp.eye(5)  # Identity matrix as one-hot labels
    >>> smoothed_5class = braintools.metric.smooth_labels(labels_5class, alpha=0.1)
    >>> print(f"5-class smoothed shape: {smoothed_5class.shape}")
    >>> print(f"Row sum (should be ~1.0): {jnp.sum(smoothed_5class[0]):.6f}")

    Integration with cross-entropy loss:

    >>> # Typical usage in training loop
    >>> logits = jnp.array([[2.0, 1.0, 0.5]])  # Model predictions
    >>> hard_labels = jnp.array([[1.0, 0.0, 0.0]])
    >>> smooth_labels_result = braintools.metric.smooth_labels(hard_labels, alpha=0.1)
    >>> # Use smooth_labels_result with cross-entropy loss function

    Verify probability distribution properties:

    >>> smoothed = braintools.metric.smooth_labels(jnp.eye(4), alpha=0.2)
    >>> print(f"All rows sum to 1: {jnp.allclose(jnp.sum(smoothed, axis=1), 1.0)}")
    >>> print(f"All values non-negative: {jnp.all(smoothed >= 0)}")

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
    assert u.math.is_float(labels), f'labels should be of float type.'
    num_categories = labels.shape[-1]
    return (1.0 - alpha) * labels + alpha / num_categories
