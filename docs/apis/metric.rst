``braintools.metric`` module
============================

.. currentmodule:: braintools.metric 
.. automodule:: braintools.metric 

Comprehensive metric collection covering spiking activity, statistical
analysis, and supervised learning objectives for neural modeling.

Classification Losses
---------------------

Objective functions for training classifiers on neural or behavioral labels.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   sigmoid_binary_cross_entropy
   hinge_loss
   perceptron_loss
   softmax_cross_entropy
   softmax_cross_entropy_with_integer_labels
   multiclass_hinge_loss
   multiclass_perceptron_loss
   poly_loss_cross_entropy
   kl_divergence
   kl_divergence_with_log_targets
   convex_kl_divergence
   ctc_loss
   ctc_loss_with_forward_probs
   sigmoid_focal_loss
   nll_loss


Correlation
-----------

Tools for measuring synchrony, functional connectivity, and aggregated
correlations between neural signals.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   cross_correlation
   voltage_fluctuation
   matrix_correlation
   weighted_correlation
   functional_connectivity
   functional_connectivity_dynamics


Fenchel-Young Loss
------------------

Generalized convex losses derived from Fenchel-Young duality for structured
prediction problems.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   make_fenchel_young_loss


Spike Firing
------------

Descriptive statistics that summarize firing rates, timing variability, and
spiking reliability.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    raster_plot
    firing_rate
    victor_purpura_distance
    van_rossum_distance
    spike_train_synchrony
    burst_synchrony_index
    phase_locking_value
    spike_time_tiling_coefficient
    correlation_index


Local Field Potential
---------------------

Metrics tailored to local field potential (LFP) analysis such as spectral
characteristics and connectivity.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    unitary_LFP
    power_spectral_density
    coherence_analysis
    phase_amplitude_coupling
    theta_gamma_coupling
    current_source_density
    spectral_entropy
    lfp_phase_coherence


Ranking Losses
--------------

Losses for ordered prediction tasks including pairwise and list-wise ranking
setups.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ranking_softmax_loss


Regression Losses
-----------------

Continuous-valued error metrics for fitting neural or behavioral signals.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   squared_error
   absolute_error
   l1_loss
   l2_loss
   l2_norm
   huber_loss
   log_cosh
   cosine_similarity
   cosine_distance


Smoothing Losses
----------------

Regularizers that promote smooth trajectories or label distributions over time.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   smooth_labels


