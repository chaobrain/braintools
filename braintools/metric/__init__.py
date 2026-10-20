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
Metrics and Loss Functions for Neural Networks.

This module provides loss functions and metrics for training neural networks:
classification, regression, ranking, pairwise similarity and label smoothing.

The neural-activity measures that used to live here (spike-train metrics, LFP
analysis and correlation/synchrony measures) moved to :mod:`braintools.analysis`
in 0.4; see "Moved to braintools.analysis" below.

**Key Features:**

- **Classification Losses**: Binary and multi-class cross-entropy, hinge loss, focal loss
- **Regression Losses**: MSE, MAE, Huber loss, cosine similarity
- **Ranking Losses**: Softmax ranking loss for learning to rank
- **Pairwise Metrics**: Cosine similarity for pairwise comparisons

**Quick Start - Classification:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.metric import softmax_cross_entropy, sigmoid_focal_loss

    # Multi-class classification
    logits = jnp.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    loss = softmax_cross_entropy(logits, labels)

    # Binary classification with focal loss
    predictions = jnp.array([0.9, 0.1, 0.7])
    targets = jnp.array([1.0, 0.0, 1.0])
    focal_loss = sigmoid_focal_loss(predictions, targets)

**Quick Start - Regression:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.metric import squared_error, huber_loss

    predictions = jnp.array([1.5, 2.3, 3.1])
    targets = jnp.array([1.0, 2.0, 3.0])

    # Mean squared error
    mse = squared_error(predictions, targets)

    # Huber loss (robust to outliers)
    huber = huber_loss(predictions, targets, delta=1.0)

**Classification Losses:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.metric import (
        sigmoid_binary_cross_entropy,
        softmax_cross_entropy_with_integer_labels,
        hinge_loss,
        multiclass_hinge_loss,
        kl_divergence,
        sigmoid_focal_loss
    )

    # Binary cross-entropy
    logits = jnp.array([2.0, -1.0, 0.5])
    labels = jnp.array([1.0, 0.0, 1.0])
    bce = sigmoid_binary_cross_entropy(logits, labels)

    # Multi-class with integer labels
    logits = jnp.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    labels = jnp.array([0, 1])  # Class indices
    ce = softmax_cross_entropy_with_integer_labels(logits, labels)

    # Hinge loss for SVM-style classification
    predictions = jnp.array([0.9, -0.5, 0.3])
    targets = jnp.array([1.0, -1.0, 1.0])
    hinge = hinge_loss(predictions, targets)

    # KL divergence
    p = jnp.array([0.5, 0.3, 0.2])
    q = jnp.array([0.4, 0.4, 0.2])
    kl = kl_divergence(p, q)

    # Focal loss for imbalanced datasets
    predictions = jnp.array([0.9, 0.1, 0.6])
    targets = jnp.array([1.0, 0.0, 1.0])
    focal = sigmoid_focal_loss(predictions, targets, alpha=0.25, gamma=2.0)

**Regression Losses:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.metric import (
        squared_error,
        absolute_error,
        l1_loss,
        l2_loss,
        huber_loss,
        log_cosh,
        cosine_similarity,
        cosine_distance
    )

    predictions = jnp.array([1.5, 2.3, 3.1, 4.2])
    targets = jnp.array([1.0, 2.0, 3.0, 4.0])

    # Various regression losses
    mse = squared_error(predictions, targets)
    mae = absolute_error(predictions, targets)
    l1 = l1_loss(predictions, targets)
    l2 = l2_loss(predictions, targets)
    huber = huber_loss(predictions, targets, delta=1.0)
    log_cosh_loss = log_cosh(predictions, targets)

    # Cosine similarity/distance
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = jnp.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
    similarity = cosine_similarity(x, y)
    distance = cosine_distance(x, y)

**Advanced: Fenchel-Young Losses:**

.. code-block:: python

    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    from braintools.metric import make_fenchel_young_loss

    # ``max_fun`` is applied over the last axis via ``jnp.vectorize`` with
    # signature ``(n)->()``, so it must map a 1-D score vector to a scalar.
    # ``logsumexp`` recovers the softmax cross-entropy Fenchel-Young loss.
    loss_fn = make_fenchel_young_loss(max_fun=logsumexp)
    scores = jnp.array([[2.0, 1.0, 3.0], [1.5, 2.5, 1.0]])
    targets = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    loss = loss_fn(scores, targets)

**Ranking Losses:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.metric import ranking_softmax_loss

    # Ranking loss for learning to rank
    scores = jnp.array([[2.0, 1.0, 3.0], [1.0, 0.5, 1.5]])
    labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    loss = ranking_softmax_loss(scores, labels)

**Utilities:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.metric import smooth_labels

    # Label smoothing for regularization
    labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    smoothed = smooth_labels(labels, alpha=0.1)

**Moved to braintools.analysis:**

The following names still resolve in ``braintools.metric`` but are deprecated
aliases: calling one emits a ``BraintoolsDeprecationWarning``, and they will be
removed in 1.0. Use the new locations instead.

- :mod:`braintools.analysis.spike`: ``firing_rate``, ``victor_purpura_distance``,
  ``van_rossum_distance``, ``spike_train_synchrony``, ``burst_synchrony_index``,
  ``phase_locking_value``, ``spike_time_tiling_coefficient``,
  ``correlation_index``, ``cross_correlation``
- :mod:`braintools.analysis.spectral`: ``power_spectral_density``,
  ``coherence_analysis``, ``phase_amplitude_coupling``,
  ``theta_gamma_coupling``, ``spectral_entropy``, ``lfp_phase_coherence``
- :mod:`braintools.analysis.field`: ``unitary_LFP``, ``current_source_density``
- :mod:`braintools.analysis.population`: ``voltage_fluctuation``,
  ``functional_connectivity``, ``functional_connectivity_dynamics``
- :mod:`braintools.analysis.stats`: ``matrix_correlation``,
  ``weighted_correlation``

``raster_plot`` is deprecated as well; ``braintools.spike.SpikeTrains.from_dense``
replaces it.

.. code-block:: python

    # Before (deprecated)
    from braintools.metric import firing_rate

    # After
    from braintools.analysis.spike import firing_rate

"""

# Classification losses
from ._classification import (
    sigmoid_binary_cross_entropy,
    hinge_loss,
    perceptron_loss,
    softmax_cross_entropy,
    softmax_cross_entropy_with_integer_labels,
    multiclass_hinge_loss,
    multiclass_perceptron_loss,
    poly_loss_cross_entropy,
    kl_divergence,
    kl_divergence_with_log_targets,
    convex_kl_divergence,
    ctc_loss,
    ctc_loss_with_forward_probs,
    sigmoid_focal_loss,
    nll_loss,
)

# Deprecated aliases: spike-train, LFP and correlation measures moved to
# ``braintools.analysis`` (removed in 1.0). They warn only when called.
from ._deprecated import (
    cross_correlation,
    voltage_fluctuation,
    matrix_correlation,
    weighted_correlation,
    functional_connectivity,
    functional_connectivity_dynamics,
    raster_plot,
    firing_rate,
    victor_purpura_distance,
    van_rossum_distance,
    spike_train_synchrony,
    burst_synchrony_index,
    phase_locking_value,
    spike_time_tiling_coefficient,
    correlation_index,
    unitary_LFP,
    power_spectral_density,
    coherence_analysis,
    phase_amplitude_coupling,
    theta_gamma_coupling,
    current_source_density,
    spectral_entropy,
    lfp_phase_coherence,
)

# Fenchel-Young losses
from ._fenchel_young import (
    make_fenchel_young_loss,
)

# Pairwise metrics
from ._pairwise import (
    pairwise_cosine_similarity,
    pairwise_cosine_distance,
)

# Ranking losses
from ._ranking import (
    ranking_softmax_loss,
)

# Regression losses
from ._regression import (
    squared_error,
    absolute_error,
    l1_loss,
    l2_loss,
    l2_norm,
    safe_norm,
    huber_loss,
    log_cosh,
    cosine_similarity,
    cosine_distance,
    L1Loss,
)

# Smoothing utilities
from ._smoothing import (
    smooth_labels,
)

__all__ = [
    # Classification losses
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

    # Correlation analysis (deprecated; moved to braintools.analysis)
    'cross_correlation',
    'voltage_fluctuation',
    'matrix_correlation',
    'weighted_correlation',
    'functional_connectivity',
    'functional_connectivity_dynamics',

    # Fenchel-Young losses
    'make_fenchel_young_loss',

    # Spike train metrics (deprecated; moved to braintools.analysis)
    'raster_plot',
    'firing_rate',
    'victor_purpura_distance',
    'van_rossum_distance',
    'spike_train_synchrony',
    'burst_synchrony_index',
    'phase_locking_value',
    'spike_time_tiling_coefficient',
    'correlation_index',

    # LFP analysis (deprecated; moved to braintools.analysis)
    'unitary_LFP',
    'power_spectral_density',
    'coherence_analysis',
    'phase_amplitude_coupling',
    'theta_gamma_coupling',
    'current_source_density',
    'spectral_entropy',
    'lfp_phase_coherence',

    # Pairwise metrics
    'pairwise_cosine_similarity',
    'pairwise_cosine_distance',

    # Ranking losses
    'ranking_softmax_loss',

    # Regression losses
    'squared_error',
    'absolute_error',
    'l1_loss',
    'l2_loss',
    'l2_norm',
    'safe_norm',
    'huber_loss',
    'log_cosh',
    'cosine_similarity',
    'cosine_distance',
    'L1Loss',

    # Smoothing utilities
    'smooth_labels',
]
