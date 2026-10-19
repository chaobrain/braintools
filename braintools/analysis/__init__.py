# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Neuroscience data analysis.

``braintools.analysis`` groups the measures used to analyse recorded or
simulated neural activity. Functions live in five subpackages and are always
accessed through them (they are not re-exported at this level):

=========================================  ==========================================================================
Subpackage                                 Contents
=========================================  ==========================================================================
:mod:`braintools.analysis.spike`           ISI statistics (CV, CV2, LV, Fano), PSTH, correlograms, spike-triggered
                                           average, firing rates, spike-train distances, synchrony
:mod:`braintools.analysis.spectral`        Welch and multitaper spectra, spectrograms, Morlet wavelets, band-pass
                                           filtering, band power, PLV/PLI/wPLI, coherence, cross-frequency coupling
:mod:`braintools.analysis.field`           LFP forward model, current source density
:mod:`braintools.analysis.population`      PCA, CCA, RSA, decoding, mutual information, Kuramoto order, Granger
                                           causality, avalanches, branching ratio, functional connectivity
:mod:`braintools.analysis.stats`           correlation helpers, spike jitter, ISI shuffling, phase randomisation,
                                           bootstrap confidence intervals
=========================================  ==========================================================================

These functions were previously part of :mod:`braintools.metric`. The old names
(for example ``braintools.metric.firing_rate``) still work in 0.4 but emit a
``BraintoolsDeprecationWarning`` when called and
will be removed in 1.0.

**Time conventions.** Time arguments accept :class:`brainunit.Quantity` values, which are converted
automatically. Plain numbers follow each function's documented convention:
spike-matrix measures take ``dt`` in milliseconds (the ``brainstate``
convention), whereas :func:`~braintools.analysis.spike.firing_rate`,
:func:`~braintools.analysis.spike.phase_locking_value` and every function in
:mod:`~braintools.analysis.spectral` take seconds. Passing quantities avoids
having to remember which.

Examples
--------
Spike-train analysis on a spike matrix of shape ``(n_time, n_neurons)``:

.. code-block:: python

    >>> import brainunit as u
    >>> import jax.numpy as jnp
    >>> from braintools.analysis import spike
    >>> spikes = jnp.zeros((200, 5)).at[50:53, :].set(1.0)
    >>> rate = spike.firing_rate(spikes, width=10 * u.ms, dt=1 * u.ms)
    >>> rate.shape
    (200,)
    >>> sync = spike.spike_train_synchrony(spikes, window_size=5 * u.ms, dt=1 * u.ms)
    >>> round(float(sync), 3)
    1.0

Spectral analysis of an LFP sampled every millisecond:

.. code-block:: python

    >>> from braintools.analysis import spectral
    >>> t = jnp.arange(4000) * 0.001  # seconds
    >>> lfp = jnp.sin(2 * jnp.pi * 10 * t)
    >>> freqs, psd = spectral.power_spectral_density(lfp, dt=1 * u.ms, nperseg=1000)
    >>> float(freqs[jnp.argmax(psd)])
    10.0

Population and statistics helpers:

.. code-block:: python

    >>> from braintools.analysis import population, stats
    >>> activities = jnp.column_stack([jnp.sin(t), jnp.sin(t), jnp.cos(t)])
    >>> population.functional_connectivity(activities).shape
    (3, 3)
    >>> fc = population.functional_connectivity(activities)
    >>> round(float(stats.matrix_correlation(fc, fc)), 3)
    1.0
"""

from . import field
from . import population
from . import spectral
from . import spike
from . import stats

__all__ = [
    'spike',
    'spectral',
    'field',
    'population',
    'stats',
]
