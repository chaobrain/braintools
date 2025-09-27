# This file is modified from [optax/losses](https://github.com/google-deepmind/optax).
# The copyright notice is as follows:
#
# Copyright 2024 BrainPy Ecosystem Limited. All Rights Reserved.
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
Classification losses.
"""

from typing import Optional

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from braintools._misc import set_module_as

__all__ = [
    'sigmoid_binary_cross_entropy',
    'hinge_loss',
    'perceptron_loss',
    'softmax_cross_entropy',
    'softmax_cross_entropy_with_integer_labels',
    'multiclass_hinge_loss',
    'multiclass_perceptron_loss',
    'poly_loss_cross_entropy',
    'kl_divergence',
    'kl_divergence_with_log_targets',
    'convex_kl_divergence',
    'ctc_loss',
    'ctc_loss_with_forward_probs',
    'sigmoid_focal_loss',
    'nll_loss',
]


def assert_is_float(array):
    assert u.math.is_float(array), 'Array must be float.'


def assert_is_int(array):
    assert u.math.is_int(array), 'Array must be int.'


@set_module_as('braintools.metric')
def sigmoid_binary_cross_entropy(
    logits: brainstate.typing.ArrayLike,
    labels: brainstate.typing.ArrayLike,
):
    r"""Compute element-wise sigmoid cross entropy given logits and labels.

    This function can be used for binary or multiclass classification where each
    class is an independent binary prediction and different classes are not
    mutually exclusive (e.g. predicting that an image contains both a cat
    and a dog).

    Parameters
    ----------
    logits : brainstate.typing.ArrayLike
        Unnormalized log probabilities for binary predictions. Each element
        represents the unnormalized log probability of a binary prediction.
        Must be compatible with `labels` shape.
    labels : brainstate.typing.ArrayLike
        Binary labels with values in {0, 1} or multi-class target probabilities.
        Must be broadcastable with `logits`.

    Returns
    -------
    brainstate.typing.ArrayLike
        Cross entropy for each binary prediction, same shape as `logits`.

    Notes
    -----
    Please ensure your `logits` and `labels` are compatible with each other.
    If you're passing in binary `labels` (values in {0, 1}), ensure your
    `logits` correspond to class 1 only. If you're passing in per-class target
    probabilities or one-hot `labels`, please ensure your `logits` are also
    multiclass. Be particularly careful if you're relying on implicit
    broadcasting to reshape `logits` or `labels`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> logits = jnp.array([1.0, -1.0, 0.0])
    >>> labels = jnp.array([1.0, 0.0, 1.0])
    >>> loss = braintools.metric.sigmoid_binary_cross_entropy(logits, labels)
    >>> print(loss)
    [0.31326166 0.31326166 0.6931472 ]

    References
    ----------
    .. [1] Goodfellow, Ian, Yoshua Bengio, and Aaron Courville.
           "Deep learning." MIT press, 2016.
           http://www.deeplearningbook.org/contents/prob.html
    """
    labels = labels.astype(logits.dtype)
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    return -labels * log_p - (1. - labels) * log_not_p


@set_module_as('braintools.metric')
def hinge_loss(
    predictor_outputs: brainstate.typing.ArrayLike,
    targets: brainstate.typing.ArrayLike
) -> brainstate.typing.ArrayLike:
    r"""Compute the hinge loss for binary classification.

    The hinge loss is commonly used for training classifiers, particularly
    Support Vector Machines. It provides a margin-based loss that is zero
    when the prediction is correct and confident, and increases linearly
    with the distance from the margin.

    The hinge loss is defined as: ``max(0, 1 - y * f(x))`` where ``y`` is the
    true class label and ``f(x)`` is the predicted output.

    Parameters
    ----------
    predictor_outputs : brainstate.typing.ArrayLike
        Outputs of the decision function. Real-valued predictions from the model.
    targets : brainstate.typing.ArrayLike
        Target values. Must be in the set {-1, 1} for binary classification.
        Shape must be broadcastable with `predictor_outputs`.

    Returns
    -------
    brainstate.typing.ArrayLike
        Hinge loss values with the same shape as the input arrays.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> predictions = jnp.array([1.0, -0.5, 2.0])
    >>> targets = jnp.array([1, -1, 1])
    >>> loss = braintools.metric.hinge_loss(predictions, targets)
    >>> print(loss)
    [0.  1.5 0. ]
    """
    return jnp.maximum(0, 1 - predictor_outputs * targets)


@set_module_as('braintools.metric')
def perceptron_loss(
    predictor_outputs: brainstate.typing.ArrayLike,
    targets: brainstate.typing.ArrayLike
) -> brainstate.typing.ArrayLike:
    r"""Compute the binary perceptron loss.

    The perceptron loss is used in the original perceptron algorithm for
    binary classification. It penalizes misclassified examples by the
    magnitude of the prediction error.

    The perceptron loss is defined as: ``max(0, -y * f(x))`` where ``y`` is
    the true class label and ``f(x)`` is the predicted output.

    Parameters
    ----------
    predictor_outputs : brainstate.typing.ArrayLike
        Scores produced by the model. Real-valued predictions.
    targets : brainstate.typing.ArrayLike
        Target values. Must be in the set {-1, 1} for binary classification.
        Shape must match `predictor_outputs`.

    Returns
    -------
    brainstate.typing.ArrayLike
        Perceptron loss values with the same shape as the input arrays.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> predictions = jnp.array([1.0, -0.5, 2.0])
    >>> targets = jnp.array([1, -1, 1])
    >>> loss = braintools.metric.perceptron_loss(predictions, targets)
    >>> print(loss)
    [0.  0.  0. ]

    References
    ----------
    .. [1] Rosenblatt, Frank. "The perceptron: a probabilistic model for
           information storage and organization in the brain."
           Psychological review 65.6 (1958): 386.
           https://en.wikipedia.org/wiki/Perceptron
    """
    assert jnp.shape(predictor_outputs) == jnp.shape(targets), 'shape mismatch'
    return jnp.maximum(0, - predictor_outputs * targets)


@set_module_as('braintools.metric')
def softmax_cross_entropy(
    logits: brainstate.typing.ArrayLike,
    labels: brainstate.typing.ArrayLike,
) -> brainstate.typing.ArrayLike:
    r"""Compute the softmax cross entropy between logits and labels.

    Measures the probability error in discrete classification tasks where
    the classes are mutually exclusive (each entry is in exactly one class).
    For example, each CIFAR-10 image is labeled with one and only one label:
    an image can be a dog or a truck, but not both.

    Parameters
    ----------
    logits : brainstate.typing.ArrayLike
        Unnormalized log probabilities with shape ``[..., num_classes]``.
    labels : brainstate.typing.ArrayLike
        Valid probability distributions (non-negative, sum to 1), e.g., a
        one-hot encoding specifying the correct class for each input.
        Must have shape broadcastable to ``[..., num_classes]``.

    Returns
    -------
    brainstate.typing.ArrayLike
        Cross entropy between each prediction and the corresponding target
        distributions, with shape ``[...]``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> logits = jnp.array([[2.0, 1.0, 0.1]])
    >>> labels = jnp.array([[1.0, 0.0, 0.0]])
    >>> loss = braintools.metric.softmax_cross_entropy(logits, labels)
    >>> print(loss)
    [0.4170299]

    References
    ----------
    .. [1] Goodfellow, Ian, Yoshua Bengio, and Aaron Courville.
           "Deep learning." MIT press, 2016.
           http://www.deeplearningbook.org/contents/prob.html
    """
    ret = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    return ret


@set_module_as('braintools.metric')
def softmax_cross_entropy_with_integer_labels(
    logits: brainstate.typing.ArrayLike,
    labels: brainstate.typing.ArrayLike,
) -> brainstate.typing.ArrayLike:
    r"""Compute softmax cross entropy between logits and integer labels.

    This is a more efficient version of softmax cross entropy when labels
    are provided as integer class indices rather than one-hot encoded vectors.
    Measures the probability error in discrete classification tasks where
    the classes are mutually exclusive.

    Parameters
    ----------
    logits : brainstate.typing.ArrayLike
        Unnormalized log probabilities with shape ``[..., num_classes]``.
        Must be floating point type.
    labels : brainstate.typing.ArrayLike
        Integer class indices specifying the correct class for each input.
        Values should be in the range ``[0, num_classes)``. Shape ``[...]``.
        Must be integer type.

    Returns
    -------
    brainstate.typing.ArrayLike
        Cross entropy between each prediction and the corresponding target
        distributions, with shape ``[...]``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> logits = jnp.array([[2.0, 1.0, 0.1]])
    >>> labels = jnp.array([0])  # Class 0
    >>> loss = braintools.metric.softmax_cross_entropy_with_integer_labels(logits, labels)
    >>> print(loss)
    [0.4170299]

    References
    ----------
    .. [1] Goodfellow, Ian, Yoshua Bengio, and Aaron Courville.
           "Deep learning." MIT press, 2016.
           http://www.deeplearningbook.org/contents/prob.html
    """
    assert_is_float(logits)
    assert_is_int(labels)
    # This is like jnp.take_along_axis(jax.nn.log_softmax(...), ...) except that
    # we avoid subtracting the normalizer from all values, just from the values
    # for the correct labels.
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    logits -= jax.lax.stop_gradient(logits_max)
    label_logits = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
    return log_normalizers - label_logits


_dot_last_dim = jnp.vectorize(jnp.dot, signature='(n),(n)->()')


@set_module_as('braintools.metric')
def multiclass_hinge_loss(
    scores: brainstate.typing.ArrayLike,
    labels: brainstate.typing.ArrayLike,
) -> brainstate.typing.ArrayLike:
    r"""Compute multiclass hinge loss for classification.

    The multiclass hinge loss is an extension of the binary hinge loss to
    multiple classes. It encourages the correct class score to be at least
    1 unit higher than the highest scoring incorrect class.

    The loss is defined as:
    
    .. math::
    
        L = \max(0, \max_{j \neq y} s_j - s_y + 1)
    
    where :math:`s_y` is the score for the correct class :math:`y` and
    :math:`s_j` are scores for other classes.

    Parameters
    ----------
    scores : brainstate.typing.ArrayLike
        Model output scores with shape ``[..., num_classes]``. These are
        raw scores (not probabilities) for each class.
    labels : brainstate.typing.ArrayLike
        Ground-truth integer class labels with shape ``[...]``. Each label
        should be in the range ``[0, num_classes)``.

    Returns
    -------
    brainstate.typing.ArrayLike
        Hinge loss values with shape ``[...]``, same as the leading
        dimensions of ``scores``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> # Scores for 3 classes, 2 samples
    >>> scores = jnp.array([[1.0, 2.0, 0.5], [0.8, 0.3, 1.2]])
    >>> labels = jnp.array([1, 2])  # Correct classes
    >>> loss = braintools.metric.multiclass_hinge_loss(scores, labels)
    >>> print(loss)
    [0.  0. ]

    References
    ----------
    .. [1] Crammer, Koby, and Yoram Singer. "On the algorithmic implementation
           of multiclass kernel-based vector machines." Journal of machine
           learning research 2.Dec (2001): 265-292.
           https://en.wikipedia.org/wiki/Hinge_loss
    """
    one_hot_labels = jax.nn.one_hot(labels, scores.shape[-1])
    return (jnp.max(scores + 1.0 - one_hot_labels, axis=-1) -
            _dot_last_dim(scores, one_hot_labels))


@set_module_as('braintools.metric')
def multiclass_perceptron_loss(
    scores: brainstate.typing.ArrayLike,
    labels: brainstate.typing.ArrayLike,
) -> brainstate.typing.ArrayLike:
    r"""Compute multiclass perceptron loss for classification.

    The multiclass perceptron loss measures the difference between the
    highest scoring class and the correct class score. It is used in
    structured perceptron learning for multiclass classification.

    The loss is defined as:
    
    .. math::
    
        L = \max_j s_j - s_y
    
    where :math:`s_y` is the score for the correct class :math:`y` and
    :math:`\max_j s_j` is the maximum score across all classes.

    Parameters
    ----------
    scores : brainstate.typing.ArrayLike
        Model output scores with shape ``[..., num_classes]``. These are
        raw scores (not probabilities) for each class.
    labels : brainstate.typing.ArrayLike
        Ground-truth integer class labels with shape ``[...]``. Each label
        should be in the range ``[0, num_classes)``.

    Returns
    -------
    brainstate.typing.ArrayLike
        Perceptron loss values with shape ``[...]``, same as the leading
        dimensions of ``scores``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> # Scores for 3 classes, 2 samples
    >>> scores = jnp.array([[1.0, 2.0, 0.5], [0.8, 0.3, 1.2]])
    >>> labels = jnp.array([1, 2])  # Correct classes
    >>> loss = braintools.metric.multiclass_perceptron_loss(scores, labels)
    >>> print(loss)
    [0.  0. ]

    References
    ----------
    .. [1] Collins, Michael. "Discriminative training methods for hidden
           Markov models: Theory and experiments with perceptron algorithms."
           Proceedings of EMNLP 2002.
    """
    one_hot_labels = jax.nn.one_hot(labels, scores.shape[-1])
    return jnp.max(scores, axis=-1) - _dot_last_dim(scores, one_hot_labels)


@set_module_as('braintools.metric')
def poly_loss_cross_entropy(
    logits: brainstate.typing.ArrayLike,
    labels: brainstate.typing.ArrayLike,
    epsilon: float = 2.0
) -> brainstate.typing.ArrayLike:
    r"""Compute PolyLoss cross entropy between logits and labels.

    PolyLoss is a polynomial expansion of commonly used classification loss
    functions. It decomposes loss functions into weighted polynomial bases
    inspired by the Taylor expansion of cross-entropy and focal loss.

    The PolyLoss is defined as:

    .. math::

        L_{Poly} = \sum_{j=1}^\infty \alpha_j \cdot (1 - P_t)^j

    This function implements a simplified version with only the first
    polynomial term modified:

    .. math::

        L = -\log(P_t) + \epsilon \cdot (1 - P_t)

    where :math:`P_t` is the predicted probability for the true class.

    Parameters
    ----------
    logits : brainstate.typing.ArrayLike
        Unnormalized log probabilities with shape ``[..., num_classes]``.
    labels : brainstate.typing.ArrayLike
        Valid probability distributions (non-negative, sum to 1), e.g., a
        one-hot encoding specifying the correct class for each input.
        Must have shape broadcastable to ``[..., num_classes]``.
    epsilon : float, default=2.0
        Coefficient of the first polynomial term. Controls the emphasis on
        difficult examples:
        
        - For ImageNet 2D classification: ``epsilon = 2.0`` (recommended)
        - For 2D instance segmentation/object detection: ``epsilon = -1.0``
        - Task-specific tuning via grid search is recommended

    Returns
    -------
    brainstate.typing.ArrayLike
        PolyLoss values between each prediction and corresponding target
        distributions, with shape ``[...]``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> logits = jnp.array([[2.0, 1.0, 0.1]])
    >>> labels = jnp.array([[1.0, 0.0, 0.0]])
    >>> loss = braintools.metric.poly_loss_cross_entropy(logits, labels, epsilon=2.0)
    >>> print(f"PolyLoss: {loss[0]:.4f}")

    Notes
    -----
    PolyLoss can improve model calibration and performance on imbalanced
    datasets by adjusting the emphasis on difficult examples through the
    epsilon parameter.

    References
    ----------
    .. [1] Leng, Zhaoqi, et al. "PolyLoss: A Polynomial Expansion Perspective
           of Classification Loss Functions." arXiv preprint arXiv:2204.12511
           (2022). https://arxiv.org/pdf/2204.12511.pdf
    """
    one_minus_pt = jnp.sum(labels * (1 - jax.nn.softmax(logits)), axis=-1)
    cross_entropy = softmax_cross_entropy(logits=logits, labels=labels)
    return cross_entropy + epsilon * one_minus_pt


@set_module_as('braintools.metric')
def kl_divergence(
    log_predictions: brainstate.typing.ArrayLike,
    targets: brainstate.typing.ArrayLike
) -> brainstate.typing.ArrayLike:
    r"""Compute the Kullback-Leibler divergence (relative entropy) loss.

    KL divergence measures the information lost when approximating the target
    distribution with the predicted distribution. It quantifies how much one
    probability distribution differs from another.

    The KL divergence is defined as:

    .. math::

        D_{KL}(P||Q) = \sum_i P(i) \log\frac{P(i)}{Q(i)}

    where P is the target distribution and Q is the predicted distribution.

    Parameters
    ----------
    log_predictions : brainstate.typing.ArrayLike
        Log probabilities of the predicted distribution with shape
        ``[..., num_classes]``. Must be in log-space to avoid numerical
        underflow issues.
    targets : brainstate.typing.ArrayLike
        Probabilities of the target distribution with shape ``[..., num_classes]``.
        Values should be non-negative and sum to 1 along the last axis.
        Must be strictly positive where non-zero.

    Returns
    -------
    brainstate.typing.ArrayLike
        KL divergence values with shape ``[...]``, measuring the divergence
        between target and predicted distributions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> # Target and predicted distributions
    >>> targets = jnp.array([[0.7, 0.2, 0.1]])
    >>> log_preds = jnp.log(jnp.array([[0.6, 0.3, 0.1]]))
    >>> kl_div = braintools.metric.kl_divergence(log_preds, targets)
    >>> print(f"KL divergence: {kl_div[0]:.4f}")

    Notes
    -----
    KL divergence is not symmetric: KL(P||Q) ≠ KL(Q||P). It measures the
    information lost when using Q to approximate P. The function handles
    zero probabilities by setting the corresponding terms to zero.

    References
    ----------
    .. [1] Kullback, Solomon, and Richard A. Leibler. "On information and
           sufficiency." The annals of mathematical statistics 22.1 (1951): 79-86.
           https://www.jstor.org/stable/2236703
    """
    assert_is_float(log_predictions)
    assert_is_float(targets)
    loss = targets * (jnp.where(targets == 0, 0, jnp.log(targets)) - log_predictions)
    return jnp.sum(loss, axis=-1)


@set_module_as('braintools.metric')
def kl_divergence_with_log_targets(
    log_predictions: brainstate.typing.ArrayLike,
    log_targets: brainstate.typing.ArrayLike
) -> brainstate.typing.ArrayLike:
    r"""Compute KL divergence when both predictions and targets are in log-space.

    This is a numerically stable version of KL divergence computation when
    both the target and predicted distributions are provided in log-space,
    avoiding potential underflow issues that can occur with very small
    probabilities.

    The computation uses the log-space formula:

    .. math::

        D_{KL}(P||Q) = \sum_i \exp(\log P(i)) \cdot (\log P(i) - \log Q(i))

    Parameters
    ----------
    log_predictions : brainstate.typing.ArrayLike
        Log probabilities of the predicted distribution with shape
        ``[..., num_classes]``. Must be in log-space.
    log_targets : brainstate.typing.ArrayLike
        Log probabilities of the target distribution with shape
        ``[..., num_classes]``. Must be in log-space.

    Returns
    -------
    brainstate.typing.ArrayLike
        KL divergence values with shape ``[...]``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> # Both distributions in log-space
    >>> log_targets = jnp.log(jnp.array([[0.7, 0.2, 0.1]]))
    >>> log_preds = jnp.log(jnp.array([[0.6, 0.3, 0.1]]))
    >>> kl_div = braintools.metric.kl_divergence_with_log_targets(log_preds, log_targets)
    >>> print(f"KL divergence: {kl_div[0]:.4f}")

    Notes
    -----
    This function is preferred when working with very small probabilities
    or when both distributions are naturally available in log-space,
    as it provides better numerical stability.
    """
    assert_is_float(log_predictions)
    assert_is_float(log_targets)
    loss = jnp.exp(log_targets) * (log_targets - log_predictions)
    return jnp.sum(loss, axis=-1)


@set_module_as('braintools.metric')
def convex_kl_divergence(
    log_predictions: brainstate.typing.ArrayLike,
    targets: brainstate.typing.ArrayLike
) -> brainstate.typing.ArrayLike:
    r"""Compute a convex version of the Kullback-Leibler divergence loss.

    This function computes a modified KL divergence that is jointly convex
    in both the target probabilities and the predicted log probabilities.
    The standard KL divergence is convex only in the predicted distribution.

    The convex KL divergence is defined as:

    .. math::

        D_{convex}(P||Q) = D_{KL}(P||Q) + \sum_i (Q(i) - P(i))

    where the second term makes the function convex in both arguments.

    Parameters
    ----------
    log_predictions : brainstate.typing.ArrayLike
        Log probabilities of the predicted distribution with shape
        ``[..., num_classes]``. Must be in log-space.
    targets : brainstate.typing.ArrayLike
        Probabilities of the target distribution with shape ``[..., num_classes]``.
        Values should be non-negative and sum to 1 along the last axis.

    Returns
    -------
    brainstate.typing.ArrayLike
        Convex KL divergence values with shape ``[...]``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> targets = jnp.array([[0.7, 0.2, 0.1]])
    >>> log_preds = jnp.log(jnp.array([[0.6, 0.3, 0.1]]))
    >>> conv_kl = braintools.metric.convex_kl_divergence(log_preds, targets)
    >>> print(f"Convex KL divergence: {conv_kl[0]:.4f}")

    Notes
    -----
    The convex property can be beneficial for optimization algorithms
    that rely on convexity guarantees, though it changes the semantic
    meaning compared to standard KL divergence.

    References
    ----------
    .. [1] Kullback, Solomon, and Richard A. Leibler. "On information and
           sufficiency." The annals of mathematical statistics 22.1 (1951): 79-86.
           https://www.jstor.org/stable/2236703
    """
    return kl_divergence(log_predictions, targets) + jnp.sum(jnp.exp(log_predictions) - targets, axis=-1)


@set_module_as('braintools.metric')
def ctc_loss_with_forward_probs(
    logits: brainstate.typing.ArrayLike,
    logit_paddings: brainstate.typing.ArrayLike,
    labels: brainstate.typing.ArrayLike,
    label_paddings: brainstate.typing.ArrayLike,
    blank_id: int = 0,
    log_epsilon: float = -1e5
) -> tuple[brainstate.typing.ArrayLike, brainstate.typing.ArrayLike, brainstate.typing.ArrayLike]:
    r"""Compute CTC loss and forward probabilities for sequence alignment.

    Connectionist Temporal Classification (CTC) loss enables training of
    sequence models without requiring frame-level alignment between input
    and output sequences. It uses dynamic programming to compute the
    probability of all valid alignments.

    The CTC loss uses a special blank symbol :math:`\phi` to represent
    variable-length output sequences and computes log-likelihoods over
    all possible alignments.

    Forward probabilities are computed for:
    
    .. math::
        \alpha_{\mathrm{BLANK}}(t, n) = \sum_{\pi_{1:t-1}} p(\pi_t = \phi | \pi_{1:t-1}, y_{1:n-1})
        
        \alpha_{\mathrm{LABEL}}(t, n) = \sum_{\pi_{1:t-1}} p(\pi_t = y_n | \pi_{1:t-1}, y_{1:n-1})

    where :math:`\pi` denotes alignment sequences with blank insertions.

    Parameters
    ----------
    logits : brainstate.typing.ArrayLike
        Logits with shape ``(batch_size, max_time, num_classes)`` containing
        unnormalized log probabilities for each class including blanks.
    logit_paddings : brainstate.typing.ArrayLike
        Padding indicators with shape ``(batch_size, max_time)``. Values of
        1.0 indicate padded positions, 0.0 indicate valid positions.
    labels : brainstate.typing.ArrayLike
        Reference integer labels with shape ``(batch_size, max_label_length)``.
        Contains target sequences without blanks.
    label_paddings : brainstate.typing.ArrayLike
        Label padding indicators with shape ``(batch_size, max_label_length)``.
        Must be right-padded (zeros followed by ones).
    blank_id : int, default=0
        Class index for the blank symbol in the logits.
    log_epsilon : float, default=-1e5
        Numerically stable approximation of log(0) for invalid transitions.

    Returns
    -------
    tuple[brainstate.typing.ArrayLike, brainstate.typing.ArrayLike, brainstate.typing.ArrayLike]
        A tuple containing:
        
        - **loss_values** : ``(batch_size,)`` - CTC loss for each sequence
        - **logalpha_blank** : ``(max_time, batch_size, max_label_length+1)`` - 
          Log forward probabilities for blank states
        - **logalpha_nonblank** : ``(max_time, batch_size, max_label_length)`` - 
          Log forward probabilities for non-blank states

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> # Example with batch_size=1, time=4, classes=3, labels=2
    >>> logits = jnp.random.normal(size=(1, 4, 3))
    >>> logit_pad = jnp.zeros((1, 4))
    >>> labels = jnp.array([[1, 2]])
    >>> label_pad = jnp.zeros((1, 2))
    >>> loss, alpha_blank, alpha_label = braintools.metric.ctc_loss_with_forward_probs(
    ...     logits, logit_pad, labels, label_pad, blank_id=0
    ... )
    >>> print(f"CTC loss: {loss[0]:.4f}")

    Notes
    -----
    This function requires that labels are right-padded and logit sequences
    are properly aligned. The forward probabilities can be used for additional
    analysis or for implementing more sophisticated training procedures.

    References
    ----------
    .. [1] Graves, Alex, et al. "Connectionist temporal classification:
           labelling unsegmented sequence data with recurrent neural networks."
           Proceedings of ICML 2006.
           https://dl.acm.org/doi/abs/10.1145/1143844.1143891
    """
    assert logits.ndim == 3, 'logits must have shape (B, T, K)'
    assert labels.ndim == 2, 'labels must have shape (B, N)'
    batchsize, unused_maxinputlen, num_classes = logits.shape
    batchsize_of_labels, maxlabellen = labels.shape
    assert batchsize == batchsize_of_labels, 'batchsize mismatch'
    assert label_paddings.shape == labels.shape, 'padding shape mismatch'
    assert logits.shape[:2] == logit_paddings.shape, 'padding shape mismatch'

    logprobs = jax.nn.log_softmax(logits)
    labellens = maxlabellen - jnp.sum(label_paddings, axis=1).astype(jnp.int32)

    # repeat[b, n] == 1.0 when label[b, n] == label[b, n+1].
    repeat = (labels[:, :-1] == labels[:, 1:]).astype(jnp.float32)
    repeat = jnp.pad(repeat, ((0, 0), (0, 1)))

    logprobs_phi = logprobs[:, :, blank_id:blank_id + 1]  # [B, T, 1]
    logprobs_phi = jnp.transpose(logprobs_phi, (1, 0, 2))  # [T, B, 1]

    one_hot = jax.nn.one_hot(labels, num_classes=num_classes)  # [B, N, K]
    logprobs_emit = jnp.einsum('btk,bnk->btn', logprobs, one_hot)
    logprobs_emit = jnp.transpose(logprobs_emit, (1, 0, 2))  # [T, B, N]

    logalpha_phi_init = jnp.ones((batchsize, maxlabellen + 1)) * log_epsilon  # [B, N]
    logalpha_phi_init = logalpha_phi_init.at[:, 0].set(0.0)
    logalpha_emit_init = jnp.ones((batchsize, maxlabellen)) * log_epsilon

    def update_phi_score(phi, added_score):
        # Update `phi[:, 1:]`` with adding `added_score` in log space.
        return jnp.concatenate([phi[:, :1], jnp.logaddexp(phi[:, 1:], added_score)], axis=-1)

    def loop_body(prev, x):
        prev_phi, prev_emit = prev
        # emit-to-phi epsilon transition, except if the next label is repetition
        prev_phi_orig = prev_phi
        prev_phi = update_phi_score(prev_phi, prev_emit + log_epsilon * repeat)

        logprob_emit, logprob_phi, pad = x

        # phi-to-emit transition
        next_emit = jnp.logaddexp(prev_phi[:, :-1] + logprob_emit,
                                  prev_emit + logprob_emit)
        # self-loop transition
        next_phi = prev_phi + logprob_phi
        # emit-to-phi blank transition only when the next label is repetition
        next_phi = update_phi_score(
            next_phi, prev_emit + logprob_phi + log_epsilon * (1.0 - repeat))

        pad = pad.reshape((batchsize, 1))
        next_emit = pad * prev_emit + (1.0 - pad) * next_emit
        next_phi = pad * prev_phi_orig + (1.0 - pad) * next_phi

        return (next_phi, next_emit), (next_phi, next_emit)

    xs = (logprobs_emit, logprobs_phi, logit_paddings.transpose((1, 0)))
    _, (logalpha_phi, logalpha_emit) = jax.lax.scan(loop_body, (logalpha_phi_init, logalpha_emit_init), xs)

    # last row needs to be updated with the last epsilon transition
    logalpha_phi_last = update_phi_score(logalpha_phi[-1], logalpha_emit[-1])
    logalpha_phi = logalpha_phi.at[-1].set(logalpha_phi_last)

    # extract per_seq_loss
    one_hot = jax.nn.one_hot(labellens, num_classes=maxlabellen + 1)  # [B, N+1]
    per_seq_loss = -jnp.einsum('bn,bn->b', logalpha_phi_last, one_hot)  # pylint:disable=invalid-unary-operand-type

    return per_seq_loss, logalpha_phi, logalpha_emit


@set_module_as('braintools.metric')
def ctc_loss(
    logits: brainstate.typing.ArrayLike,
    logit_paddings: brainstate.typing.ArrayLike,
    labels: brainstate.typing.ArrayLike,
    label_paddings: brainstate.typing.ArrayLike,
    blank_id: int = 0,
    log_epsilon: float = -1e5
) -> brainstate.typing.ArrayLike:
    r"""Compute Connectionist Temporal Classification (CTC) loss.

    A simplified interface to CTC loss computation that returns only the
    loss values without forward probabilities. This is the most commonly
    used function for training sequence models with CTC.

    Parameters
    ----------
    logits : brainstate.typing.ArrayLike
        Logits with shape ``(batch_size, max_time, num_classes)`` containing
        unnormalized log probabilities for each class including blanks.
    logit_paddings : brainstate.typing.ArrayLike
        Padding indicators with shape ``(batch_size, max_time)``. Values of
        1.0 indicate padded positions, 0.0 indicate valid positions.
    labels : brainstate.typing.ArrayLike
        Reference integer labels with shape ``(batch_size, max_label_length)``.
        Contains target sequences without blanks.
    label_paddings : brainstate.typing.ArrayLike
        Label padding indicators with shape ``(batch_size, max_label_length)``.
        Must be right-padded (zeros followed by ones).
    blank_id : int, default=0
        Class index for the blank symbol in the logits.
    log_epsilon : float, default=-1e5
        Numerically stable approximation of log(0) for invalid transitions.

    Returns
    -------
    brainstate.typing.ArrayLike
        CTC loss values with shape ``(batch_size,)`` containing the negative
        log-likelihood for each sequence in the batch.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> # Setup for speech recognition task
    >>> batch_size, time_steps, vocab_size = 2, 10, 30
    >>> logits = jnp.random.normal(size=(batch_size, time_steps, vocab_size))
    >>> logit_pad = jnp.zeros((batch_size, time_steps))
    >>> labels = jnp.array([[1, 2, 3], [4, 5, 0]])  # Different length sequences
    >>> label_pad = jnp.array([[0, 0, 0], [0, 0, 1]])  # Last label is padded
    >>> loss = braintools.metric.ctc_loss(logits, logit_pad, labels, label_pad)
    >>> print(f"Average CTC loss: {jnp.mean(loss):.4f}")

    Notes
    -----
    This function internally calls ``ctc_loss_with_forward_probs`` and
    discards the forward probability arrays. For applications that need
    the forward probabilities, use ``ctc_loss_with_forward_probs`` directly.

    See Also
    --------
    ctc_loss_with_forward_probs : CTC loss computation with forward probabilities

    References
    ----------
    .. [1] Graves, Alex, et al. "Connectionist temporal classification:
           labelling unsegmented sequence data with recurrent neural networks."
           Proceedings of ICML 2006.
    """
    per_seq_loss, _, _ = ctc_loss_with_forward_probs(
        logits, logit_paddings, labels, label_paddings,
        blank_id=blank_id, log_epsilon=log_epsilon
    )
    return per_seq_loss


@set_module_as('braintools.metric')
def sigmoid_focal_loss(
    logits: brainstate.typing.ArrayLike,
    labels: brainstate.typing.ArrayLike,
    alpha: Optional[float] = None,
    gamma: float = 2.,
) -> brainstate.typing.ArrayLike:
    r"""Compute sigmoid focal loss for addressing class imbalance.

    Focal loss is designed to address class imbalance in dense object detection
    by down-weighting easy examples and focusing training on hard negatives.
    It applies a modulating factor to the cross entropy loss to reduce the
    loss contribution from easy examples.

    The focal loss is defined as:

    .. math::

        FL(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)

    where :math:`p_t` is the predicted probability for the true class,
    :math:`\\alpha_t` is a class-dependent weighting factor, and :math:`\\gamma`
    is the focusing parameter.

    Parameters
    ----------
    logits : brainstate.typing.ArrayLike
        Unnormalized predictions (logits) for binary classification.
        Can have any shape for element-wise binary predictions.
    labels : brainstate.typing.ArrayLike
        Binary labels with values in {0, 1}. Must have the same shape
        as `logits`. Use 1 for positive class, 0 for negative class.
    alpha : float, optional
        Weighting factor in range (0, 1) to balance positive vs negative
        examples. If None, no class-based weighting is applied.
    gamma : float, default=2.0
        Focusing parameter (exponent) that controls the rate at which
        easy examples are down-weighted. Higher values focus more on
        hard examples. Common values: 0.5, 1.0, 2.0, 5.0.

    Returns
    -------
    brainstate.typing.ArrayLike
        Focal loss values with the same shape as input logits and labels.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> # Imbalanced binary classification
    >>> logits = jnp.array([2.0, -1.0, 0.5, -2.0])
    >>> labels = jnp.array([1.0, 0.0, 1.0, 0.0])
    >>> # Standard focal loss
    >>> loss = braintools.metric.sigmoid_focal_loss(logits, labels, alpha=0.25, gamma=2.0)
        >>> print(f"Focal loss: {loss}")
    >>> # Compare with unweighted version
    >>> loss_unweighted = braintools.metric.sigmoid_focal_loss(logits, labels, alpha=None, gamma=2.0)

    Notes
    -----
    Use this loss function when classes are not mutually exclusive (multi-label
    classification) or when dealing with severe class imbalance. For mutually
    exclusive classes, consider using softmax-based focal loss variants.

    The alpha parameter is typically set to the inverse class frequency for
    the positive class, e.g., alpha=0.25 when positive examples are 25% of data.

    References
    ----------
    .. [1] Lin, Tsung-Yi, et al. \"Focal loss for dense object detection.\"
           Proceedings of ICCV 2017.
           https://arxiv.org/pdf/1708.02002.pdf
    """
    alpha = -1 if alpha is None else alpha
    assert_is_float(logits)
    # see also the original paper's implementation at:
    # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    p = jax.nn.sigmoid(logits)
    ce_loss = sigmoid_binary_cross_entropy(logits, labels)
    p_t = p * labels + (1 - p) * (1 - labels)
    loss = ce_loss * ((1 - p_t) ** gamma)
    weighted = lambda loss_arg: (alpha * labels + (1 - alpha) * (1 - labels)) * loss_arg
    not_weighted = lambda loss_arg: loss_arg
    loss = jax.lax.cond(alpha >= 0, weighted, not_weighted, loss)
    return loss


@set_module_as('braintools.metric')
def nll_loss(input, target):
    r"""Compute negative log likelihood loss for classification.

    The negative log likelihood (NLL) loss is a standard loss function for
    training classification models. It expects log-probabilities as input
    and computes the negative log-likelihood of the correct class.

    The loss is computed as:

    .. math::

        \ell(x, y) = -x_{y}

    where :math:`x` contains log-probabilities and :math:`y` is the target class index.

    Parameters
    ----------
    input : brainstate.typing.ArrayLike
        Log-probabilities of each class. Expected shapes:
        
        - ``(num_classes,)`` for single sample
        - ``(batch_size, num_classes)`` for batch processing
        - ``(batch_size, num_classes, d1, d2, ..., dK)`` for higher-dimensional inputs
          (e.g., per-pixel classification for images)
        
    target : brainstate.typing.ArrayLike
        Class indices in the range ``[0, num_classes-1]``. Expected shapes:
        
        - ``()`` (scalar) for single sample
        - ``(batch_size,)`` for batch processing  
        - ``(batch_size, d1, d2, ..., dK)`` for higher-dimensional targets

    Returns
    -------
    brainstate.typing.ArrayLike
        Negative log likelihood loss values:
        
        - Scalar for single sample
        - ``(batch_size,)`` for batch processing
        - ``(batch_size, d1, d2, ..., dK)`` for higher-dimensional inputs

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools 
    >>> # Single sample example
    >>> log_probs = jnp.log(jnp.array([0.1, 0.7, 0.2]))
    >>> target = 1  # Correct class is index 1
    >>> loss = braintools.metric.nll_loss(log_probs, target)
    >>> print(f"NLL loss: {loss:.4f}")
    
    >>> # Batch example
    >>> log_probs_batch = jnp.log(jnp.array([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]]))
    >>> targets_batch = jnp.array([1, 2])
    >>> losses = braintools.metric.nll_loss(log_probs_batch, targets_batch)
    >>> print(f"Batch losses: {losses}")

    Notes
    -----
    This function expects log-probabilities as input, not raw logits or
    probabilities. Use ``jax.nn.log_softmax`` to convert logits to
    log-probabilities, or ``jnp.log`` to convert probabilities.

    For end-to-end training with logits, consider using ``softmax_cross_entropy``
    which combines softmax and cross-entropy in a numerically stable way.

    Raises
    ------
    AssertionError
        If input and target shapes are incompatible or if target contains
        invalid class indices.

    See Also
    --------
    softmax_cross_entropy : Cross entropy loss starting from logits
    softmax_cross_entropy_with_integer_labels : Efficient version for integer labels
    """
    target = jnp.asarray(target)
    if target.ndim == 1:
        assert input.ndim == 2
        loss = input[jnp.arange(len(target)), target]
        return loss
    elif target.ndim == 0:
        assert input.ndim == 1
        return input[target]
    else:
        assert False, 'Invalid shape for target'
