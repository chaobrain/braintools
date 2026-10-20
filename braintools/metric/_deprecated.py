# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Deprecated neural-analysis names kept in :mod:`braintools.metric`.

In 0.4 the spike-train, LFP and correlation measures moved to
:mod:`braintools.analysis`. Each old name below is an alias built with
:func:`braintools._deprecation.deprecated_function`: it forwards to the new
function and emits a ``BraintoolsDeprecationWarning`` when called (never on
import). The aliases are removed in 1.0.

``raster_plot`` is not moved. Its implementation stays here and is deprecated
in favour of :meth:`braintools.spike.SpikeTrains.from_dense`.
"""

import brainstate
import brainunit as u
import numpy as onp

from braintools._deprecation import deprecated_function
from braintools.analysis import field as _field
from braintools.analysis import population as _population
from braintools.analysis import spectral as _spectral
from braintools.analysis import spike as _spike
from braintools.analysis import stats as _stats

__all__ = [
    # analysis.spike
    'firing_rate',
    'victor_purpura_distance',
    'van_rossum_distance',
    'spike_train_synchrony',
    'burst_synchrony_index',
    'phase_locking_value',
    'spike_time_tiling_coefficient',
    'correlation_index',
    'cross_correlation',
    # analysis.spectral
    'power_spectral_density',
    'coherence_analysis',
    'phase_amplitude_coupling',
    'theta_gamma_coupling',
    'spectral_entropy',
    'lfp_phase_coherence',
    # analysis.field
    'unitary_LFP',
    'current_source_density',
    # analysis.population
    'voltage_fluctuation',
    'functional_connectivity',
    'functional_connectivity_dynamics',
    # analysis.stats
    'matrix_correlation',
    'weighted_correlation',
    # not moved; replaced by braintools.spike.SpikeTrains
    'raster_plot',
]


def _moved(subpackage, name: str):
    """Build the deprecated ``braintools.metric.<name>`` alias of a moved function."""
    sub = subpackage.__name__.rsplit('.', 1)[-1]
    return deprecated_function(
        getattr(subpackage, name),
        f'braintools.metric.{name}',
        f'braintools.analysis.{sub}.{name}',
    )


# ---------------------------------------------------------------------------
# Moved to braintools.analysis.spike
# ---------------------------------------------------------------------------
firing_rate = _moved(_spike, 'firing_rate')
victor_purpura_distance = _moved(_spike, 'victor_purpura_distance')
van_rossum_distance = _moved(_spike, 'van_rossum_distance')
spike_train_synchrony = _moved(_spike, 'spike_train_synchrony')
burst_synchrony_index = _moved(_spike, 'burst_synchrony_index')
phase_locking_value = _moved(_spike, 'phase_locking_value')
spike_time_tiling_coefficient = _moved(_spike, 'spike_time_tiling_coefficient')
correlation_index = _moved(_spike, 'correlation_index')
cross_correlation = _moved(_spike, 'cross_correlation')

# ---------------------------------------------------------------------------
# Moved to braintools.analysis.spectral
# ---------------------------------------------------------------------------
power_spectral_density = _moved(_spectral, 'power_spectral_density')
coherence_analysis = _moved(_spectral, 'coherence_analysis')
phase_amplitude_coupling = _moved(_spectral, 'phase_amplitude_coupling')
theta_gamma_coupling = _moved(_spectral, 'theta_gamma_coupling')
spectral_entropy = _moved(_spectral, 'spectral_entropy')
lfp_phase_coherence = _moved(_spectral, 'lfp_phase_coherence')

# ---------------------------------------------------------------------------
# Moved to braintools.analysis.field
# ---------------------------------------------------------------------------
unitary_LFP = _moved(_field, 'unitary_LFP')
current_source_density = _moved(_field, 'current_source_density')

# ---------------------------------------------------------------------------
# Moved to braintools.analysis.population
# ---------------------------------------------------------------------------
voltage_fluctuation = _moved(_population, 'voltage_fluctuation')
functional_connectivity = _moved(_population, 'functional_connectivity')
functional_connectivity_dynamics = _moved(_population, 'functional_connectivity_dynamics')

# ---------------------------------------------------------------------------
# Moved to braintools.analysis.stats
# ---------------------------------------------------------------------------
matrix_correlation = _moved(_stats, 'matrix_correlation')
weighted_correlation = _moved(_stats, 'weighted_correlation')


# ---------------------------------------------------------------------------
# Not moved: deprecated in favour of braintools.spike.SpikeTrains.from_dense
# ---------------------------------------------------------------------------
def _raster_plot(
    sp_matrix: brainstate.typing.ArrayLike,
    times: brainstate.typing.ArrayLike
):
    """Extract spike times and neuron indices for raster plot visualization.

    .. deprecated:: 0.4.0
        ``braintools.metric.raster_plot`` will be removed in 1.0.0. Use
        ``braintools.spike.SpikeTrains.from_dense`` to hold the spikes, or
        ``t, i = numpy.nonzero(sp_matrix > 0)`` and return ``i, times[t]``
        to get the same arrays.

    A raster plot displays the spiking activity of a population of neurons over time,
    where each row represents a neuron and each dot or line indicates a spike occurrence.
    This function extracts the necessary data (neuron indices and corresponding spike
    times) from a spike matrix to create such visualizations.

    Parameters
    ----------
    sp_matrix : brainstate.typing.ArrayLike
        Spike matrix with shape ``(n_time_steps, n_neurons)`` where non-zero values
        indicate spike occurrences. Each element ``sp_matrix[t, i]`` represents the
        spike activity of neuron ``i`` at time step ``t``.
    times : brainstate.typing.ArrayLike
        Time points corresponding to each row of the spike matrix with shape
        ``(n_time_steps,)``. These represent the actual time values for each
        time step in the simulation.

    Returns
    -------
    neuron_indices : numpy.ndarray
        Array of neuron indices where spikes occurred. Each index corresponds
        to a neuron that fired at the corresponding time in ``spike_times``.
    spike_times : numpy.ndarray
        Array of spike times corresponding to each spike event. These are the
        actual time values when spikes occurred, extracted from the ``times`` array.

    See Also
    --------
    braintools.analysis.spike.firing_rate : Calculate population firing rates
    braintools.visualize.raster_plot : Plot spike times as a raster

    Notes
    -----
    The function uses ``numpy.where`` to find non-zero elements in the spike matrix,
    making it efficient for sparse spike data. The returned arrays have the same
    length and can be directly used for scatter plots or other visualizations.

    Examples
    --------
    .. code-block:: python

        >>> import warnings
        >>> import numpy as np
        >>> import braintools
        >>> spikes = np.array([
        ...     [0, 1, 0],  # t=0: neuron 1 spikes
        ...     [1, 0, 0],  # t=1: neuron 0 spikes
        ...     [0, 0, 1],  # t=2: neuron 2 spikes
        ...     [0, 1, 1],  # t=3: neurons 1, 2 spike
        ...     [0, 0, 0],  # t=4: no spikes
        ... ])
        >>> times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter('ignore', FutureWarning)
        ...     neuron_ids, spike_times = braintools.metric.raster_plot(spikes, times)
        >>> neuron_ids.tolist()
        [1, 0, 2, 1, 2]
        >>> spike_times.tolist()
        [0.0, 0.1, 0.2, 0.3, 0.3]
    """
    # Locate spikes from the (unit-stripped) matrix, but index ``times`` directly so
    # that ``brainunit.Quantity`` time axes keep their units in the returned array.
    elements = onp.where(onp.asarray(u.get_magnitude(sp_matrix)) > 0.)
    index = elements[1]
    time = times[elements[0]]
    return index, time


raster_plot = deprecated_function(
    _raster_plot,
    'braintools.metric.raster_plot',
    'braintools.spike.SpikeTrains.from_dense',
)
