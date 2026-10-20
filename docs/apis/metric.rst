``braintools.metric`` module
============================

.. currentmodule:: braintools.metric
.. automodule:: braintools.metric

Loss functions and metrics for training neural networks: classification,
regression, ranking, pairwise similarity and label smoothing.

.. note::

   **Moved to** :mod:`braintools.analysis`. In 0.4 the spike-train, LFP and
   correlation measures moved out of this module (see :doc:`analysis`):

   - ``firing_rate``, ``victor_purpura_distance``, ``van_rossum_distance``,
     ``spike_train_synchrony``, ``burst_synchrony_index``,
     ``phase_locking_value``, ``spike_time_tiling_coefficient``,
     ``correlation_index`` and ``cross_correlation`` are in
     :mod:`braintools.analysis.spike`;
   - ``power_spectral_density``, ``coherence_analysis``,
     ``phase_amplitude_coupling``, ``theta_gamma_coupling``,
     ``spectral_entropy`` and ``lfp_phase_coherence`` are in
     :mod:`braintools.analysis.spectral`;
   - ``unitary_LFP`` and ``current_source_density`` are in
     :mod:`braintools.analysis.field`;
   - ``voltage_fluctuation``, ``functional_connectivity`` and
     ``functional_connectivity_dynamics`` are in
     :mod:`braintools.analysis.population`;
   - ``matrix_correlation`` and ``weighted_correlation`` are in
     :mod:`braintools.analysis.stats`.

   The old ``braintools.metric.<name>`` names still work but emit a
   ``BraintoolsDeprecationWarning`` when called, and will be removed in 1.0.
   ``braintools.metric.raster_plot`` is deprecated as well; use
   ``braintools.spike.SpikeTrains.from_dense`` instead.

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


Fenchel-Young Loss
------------------

Generalized convex losses derived from Fenchel-Young duality for structured
prediction problems.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   make_fenchel_young_loss


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
   L1Loss
   l2_loss
   l2_norm
   safe_norm
   huber_loss
   log_cosh
   cosine_similarity
   cosine_distance


Pairwise Metrics
----------------

Pairwise distance and similarity matrices between sets of samples.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   pairwise_cosine_similarity
   pairwise_cosine_distance


Smoothing Losses
----------------

Regularizers that promote smooth trajectories or label distributions over time.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   smooth_labels


