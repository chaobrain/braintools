``braintools.analysis`` module
==============================

.. currentmodule:: braintools.analysis
.. automodule:: braintools.analysis

.. seealso::

   For runnable, narrative walkthroughs, see the tutorials:

   - :doc:`../analysis/05_spiking_metrics`
   - :doc:`../analysis/06_spiking_sync`
   - :doc:`../analysis/07_lfp_analysis`
   - :doc:`../analysis/08_analysis_toolbox`


Spike Trains
------------

.. currentmodule:: braintools.analysis.spike
.. automodule:: braintools.analysis.spike

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   firing_rate
   victor_purpura_distance
   van_rossum_distance
   spike_train_synchrony
   burst_synchrony_index
   phase_locking_value
   spike_time_tiling_coefficient
   correlation_index
   cross_correlation

Event-based measures on :class:`braintools.spike.SpikeTrains`: interval
statistics, event-aligned averages, correlograms and spike-train distances:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   isi
   cv
   cv2
   lv
   fano_factor
   psth
   spike_triggered_average
   cross_correlogram
   auto_correlogram
   isi_distance
   spike_distance


Spectral Analysis
-----------------

.. currentmodule:: braintools.analysis.spectral
.. automodule:: braintools.analysis.spectral

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   power_spectral_density
   coherence_analysis
   phase_amplitude_coupling
   theta_gamma_coupling
   spectral_entropy
   lfp_phase_coherence

Multitaper spectra, time-frequency analysis, filtering and phase synchrony
between continuous signals:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   spectrogram
   multitaper_psd
   morlet_transform
   bandpass
   analytic_signal
   band_power
   plv
   pli
   wpli


Field Potentials
----------------

.. currentmodule:: braintools.analysis.field
.. automodule:: braintools.analysis.field

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   unitary_LFP
   current_source_density


Population Activity
-------------------

.. currentmodule:: braintools.analysis.population
.. automodule:: braintools.analysis.population

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   voltage_fluctuation
   functional_connectivity
   functional_connectivity_dynamics

Dimensionality, representational alignment, decoding and information:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pca
   participation_ratio
   cca
   rdm
   rsa
   decode
   mutual_information

Collective dynamics:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   kuramoto_order_parameter
   granger_causality
   avalanches
   branching_ratio

Result containers (named tuples, documented by their ``Attributes``):

.. autosummary::
   :toctree: generated/
   :nosignatures:

   PCAResult
   CCAResult
   AvalancheResult


Statistics
----------

.. currentmodule:: braintools.analysis.stats
.. automodule:: braintools.analysis.stats

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   matrix_correlation
   weighted_correlation

Resampling for significance testing, and bootstrap confidence intervals:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   jitter_spikes
   shuffle_isi
   phase_randomize
   bootstrap_ci


Graph Measures
--------------

.. currentmodule:: braintools.analysis.graph
.. automodule:: braintools.analysis.graph

.. autosummary::
   :toctree: generated/
   :nosignatures:

   degree_distribution
   clustering_coefficient
   characteristic_path_length
   modularity
   motif_counts
   spectral_radius
   small_world_index

Result containers (named tuples, documented by their ``Attributes``):

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DegreeDistribution
   PathLengthResult
   MotifCounts
   SmallWorldResult
