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
Metrics and Loss Functions for Neural Networks and Neuroscience.

This module provides comprehensive metrics and loss functions for both machine learning
and neuroscience applications, including classification, regression, ranking, spike train
analysis, and local field potential (LFP) analysis.

**Key Features:**

- **Classification Losses**: Binary and multi-class cross-entropy, hinge loss, focal loss
- **Regression Losses**: MSE, MAE, Huber loss, cosine similarity
- **Ranking Losses**: Softmax ranking loss for learning to rank
- **Spike Train Metrics**: Firing rate, synchrony, distance measures
- **LFP Analysis**: Power spectral density, coherence, phase-amplitude coupling
- **Correlation Analysis**: Cross-correlation, functional connectivity
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

**Quick Start - Spike Train Analysis:**

.. code-block:: python

    import brainunit as u
    import jax.numpy as jnp
    from braintools.metric import (
        firing_rate, victor_purpura_distance,
        spike_train_synchrony
    )

    # Calculate firing rate
    spike_times = jnp.array([0.1, 0.3, 0.5, 0.7]) * u.second
    rate = firing_rate(spike_times, duration=1.0 * u.second)

    # Victor-Purpura distance between spike trains
    train1 = jnp.array([0.1, 0.3, 0.5]) * u.second
    train2 = jnp.array([0.12, 0.31, 0.52]) * u.second
    distance = victor_purpura_distance(train1, train2, cost=1.0)

    # Spike train synchrony
    spike_matrix = jnp.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]])
    synchrony = spike_train_synchrony(spike_matrix)

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

**Spike Train Analysis:**

.. code-block:: python

    import brainunit as u
    import jax.numpy as jnp
    from braintools.metric import (
        raster_plot,
        firing_rate,
        victor_purpura_distance,
        van_rossum_distance,
        spike_train_synchrony,
        burst_synchrony_index,
        phase_locking_value,
        spike_time_tiling_coefficient,
        correlation_index
    )

    # Raster plot data extraction
    spike_matrix = jnp.array([[1, 0, 1, 0], [0, 1, 1, 0]])
    times = jnp.arange(4) * 0.1 * u.second
    neuron_ids, spike_times = raster_plot(spike_matrix, times)

    # Firing rate calculation
    spikes = jnp.array([1, 0, 1, 1, 0, 0, 1, 0])
    rate = firing_rate(spikes, window_size=100 * u.ms, dt=10 * u.ms)

    # Distance metrics between spike trains
    train1 = jnp.array([0.1, 0.3, 0.5]) * u.second
    train2 = jnp.array([0.12, 0.31, 0.52]) * u.second
    vp_dist = victor_purpura_distance(train1, train2, cost=1.0)
    vr_dist = van_rossum_distance(train1, train2, tau=0.01 * u.second)

    # Synchrony measures
    spike_matrix = jnp.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]])
    synchrony = spike_train_synchrony(spike_matrix)
    burst_sync = burst_synchrony_index(spike_matrix)
    plv = phase_locking_value(spike_matrix)
    sttc = spike_time_tiling_coefficient(train1, train2, dt=0.05 * u.second)
    corr_idx = correlation_index(spike_matrix)

**LFP Analysis:**

.. code-block:: python

    import brainunit as u
    import jax.numpy as jnp
    from braintools.metric import (
        unitary_LFP,
        power_spectral_density,
        coherence_analysis,
        phase_amplitude_coupling,
        theta_gamma_coupling,
        current_source_density,
        spectral_entropy,
        lfp_phase_coherence
    )

    # Unitary LFP from spike trains
    times = jnp.arange(1000) * 0.001 * u.second
    spikes = jnp.random.randint(0, 2, (100, 1000))
    ulfp = unitary_LFP(times, spikes, spike_type='excitatory')

    # Power spectral density
    lfp_signal = jnp.sin(2 * jnp.pi * 10 * times.mantissa)  # 10 Hz signal
    freqs, psd = power_spectral_density(lfp_signal, fs=1000 * u.Hz)

    # Coherence analysis
    signal1 = jnp.sin(2 * jnp.pi * 10 * times.mantissa)
    signal2 = jnp.sin(2 * jnp.pi * 10 * times.mantissa + 0.1)
    freqs, coherence = coherence_analysis(signal1, signal2, fs=1000 * u.Hz)

    # Phase-amplitude coupling
    pac = phase_amplitude_coupling(
        lfp_signal,
        phase_freq=(4, 8),  # Theta band
        amp_freq=(30, 80),  # Gamma band
        fs=1000 * u.Hz
    )

    # Theta-gamma coupling
    tgc = theta_gamma_coupling(lfp_signal, fs=1000 * u.Hz)

    # Current source density
    lfp_channels = jnp.random.randn(16, 1000)  # 16 channels
    csd = current_source_density(lfp_channels, spacing=100 * u.um)

    # Spectral entropy
    entropy = spectral_entropy(lfp_signal, fs=1000 * u.Hz)

    # Phase coherence
    phase_coh = lfp_phase_coherence(signal1, signal2, freq_band=(8, 12))

**Correlation Analysis:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.metric import (
        cross_correlation,
        voltage_fluctuation,
        matrix_correlation,
        weighted_correlation,
        functional_connectivity,
        functional_connectivity_dynamics
    )

    # Cross-correlation between spike trains
    spikes = jnp.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]])
    cc = cross_correlation(spikes, bin=10, dt=1)

    # Voltage fluctuation correlation
    voltages = jnp.random.randn(100, 1000)  # 100 neurons, 1000 time points
    vf_corr = voltage_fluctuation(voltages)

    # Correlation matrix
    data = jnp.random.randn(50, 100)  # 50 samples, 100 features
    corr_matrix = matrix_correlation(data)

    # Weighted correlation
    x = jnp.array([1, 2, 3, 4, 5])
    y = jnp.array([2, 4, 5, 4, 5])
    weights = jnp.array([1, 1, 2, 2, 1])
    w_corr = weighted_correlation(x, y, weights)

    # Functional connectivity
    time_series = jnp.random.randn(10, 1000)  # 10 regions, 1000 time points
    fc = functional_connectivity(time_series, method='pearson')

    # Dynamic functional connectivity
    fc_dynamics = functional_connectivity_dynamics(
        time_series,
        window_size=100,
        step_size=50
    )

**Advanced: Fenchel-Young Losses:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.metric import make_fenchel_young_loss

    # Create custom loss from max function
    def max_fun(scores):
        return jnp.max(scores, axis=-1, keepdims=True)

    loss_fn = make_fenchel_young_loss(max_fun)
    scores = jnp.array([2.0, 1.0, 3.0])
    targets = jnp.array([1.0, 0.0, 0.0])
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

# Correlation analysis
from ._correlation import (
    cross_correlation,
    voltage_fluctuation,
    matrix_correlation,
    weighted_correlation,
    functional_connectivity,
    functional_connectivity_dynamics,
)

# Fenchel-Young losses
from ._fenchel_young import (
    make_fenchel_young_loss,
)

# Spike train metrics
from ._firings import (
    raster_plot,
    firing_rate,
    victor_purpura_distance,
    van_rossum_distance,
    spike_train_synchrony,
    burst_synchrony_index,
    phase_locking_value,
    spike_time_tiling_coefficient,
    correlation_index,
)

# LFP analysis
from ._lfp import (
    unitary_LFP,
    power_spectral_density,
    coherence_analysis,
    phase_amplitude_coupling,
    theta_gamma_coupling,
    current_source_density,
    spectral_entropy,
    lfp_phase_coherence,
)

# Pairwise metrics
from ._pariwise import (
    cosine_similarity as _cosine_similarity_pairwise,
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
    huber_loss,
    log_cosh,
    cosine_similarity,
    cosine_distance,
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

    # Correlation analysis
    'cross_correlation',
    'voltage_fluctuation',
    'matrix_correlation',
    'weighted_correlation',
    'functional_connectivity',
    'functional_connectivity_dynamics',

    # Fenchel-Young losses
    'make_fenchel_young_loss',

    # Spike train metrics
    'raster_plot',
    'firing_rate',
    'victor_purpura_distance',
    'van_rossum_distance',
    'spike_train_synchrony',
    'burst_synchrony_index',
    'phase_locking_value',
    'spike_time_tiling_coefficient',
    'correlation_index',

    # LFP analysis
    'unitary_LFP',
    'power_spectral_density',
    'coherence_analysis',
    'phase_amplitude_coupling',
    'theta_gamma_coupling',
    'current_source_density',
    'spectral_entropy',
    'lfp_phase_coherence',

    # Ranking losses
    'ranking_softmax_loss',

    # Regression losses
    'squared_error',
    'absolute_error',
    'l1_loss',
    'l2_loss',
    'l2_norm',
    'huber_loss',
    'log_cosh',
    'cosine_similarity',
    'cosine_distance',

    # Smoothing utilities
    'smooth_labels',
]
