``braintools.metric`` module
============================

.. currentmodule:: braintools.metric 
.. automodule:: braintools.metric 

Classification Losses
---------------------

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


Correlation
-----------

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

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   make_fenchel_young_loss


Spike Firing
------------

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

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ranking_softmax_loss


Regression Losses
-----------------

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

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   smooth_labels


