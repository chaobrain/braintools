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

"""Spike-train analysis: statistics, PSTHs, correlograms, distances and synchrony.

Two input conventions are used:

- event-based measures take a :class:`braintools.spike.SpikeTrains` (build one
  from a dense raster with :meth:`~braintools.spike.SpikeTrains.from_dense`);
- the rate and population-synchrony measures take a spike matrix of shape
  ``(n_time_steps, n_neurons)`` (time on axis 0) with the sampling step ``dt``.

- **Interval statistics**: :func:`isi`, :func:`cv`, :func:`cv2`, :func:`lv`,
  :func:`fano_factor`.
- **Event-aligned measures**: :func:`psth`, :func:`spike_triggered_average`.
- **Correlograms**: :func:`cross_correlogram`, :func:`auto_correlogram`.
- **Rates**: :func:`firing_rate`.
- **Distances between two trains**: :func:`isi_distance`,
  :func:`spike_distance`, :func:`victor_purpura_distance`,
  :func:`van_rossum_distance`.
- **Population synchrony**: :func:`cross_correlation`,
  :func:`spike_train_synchrony`, :func:`burst_synchrony_index`,
  :func:`spike_time_tiling_coefficient`, :func:`correlation_index`.
- **Locking to a rhythm**: :func:`phase_locking_value`.

Examples
--------
.. code-block:: python

    >>> import brainunit as u
    >>> import jax.numpy as jnp
    >>> from braintools.analysis import spike
    >>> spikes = jnp.zeros((100, 4)).at[20:25, :].set(1.0)
    >>> rate = spike.firing_rate(spikes, width=5 * u.ms, dt=1 * u.ms)
    >>> rate.shape
    (100,)
    >>> float(spike.victor_purpura_distance(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0])))
    0.0

Interval statistics from a :class:`~braintools.spike.SpikeTrains`:

.. code-block:: python

    >>> from braintools.spike import SpikeTrains
    >>> raster = jnp.zeros((100, 2)).at[::10, 0].set(1.0).at[::25, 1].set(1.0)
    >>> trains = SpikeTrains.from_dense(raster, dt=1 * u.ms)
    >>> spike.cv(trains)
    Array([0., 0.], dtype=float32)
"""

from ._correlation import (
    cross_correlation,
)
from ._correlogram import (
    cross_correlogram,
    auto_correlogram,
)
from ._distance import (
    isi_distance,
    spike_distance,
)
from ._firings import (
    firing_rate,
    victor_purpura_distance,
    van_rossum_distance,
    spike_train_synchrony,
    burst_synchrony_index,
    phase_locking_value,
    spike_time_tiling_coefficient,
    correlation_index,
)
from ._isi import (
    isi,
    cv,
    cv2,
    lv,
    fano_factor,
)
from ._psth import psth
from ._sta import spike_triggered_average

__all__ = [
    'isi',
    'cv',
    'cv2',
    'lv',
    'fano_factor',
    'psth',
    'spike_triggered_average',
    'cross_correlogram',
    'auto_correlogram',
    'isi_distance',
    'spike_distance',
    'firing_rate',
    'victor_purpura_distance',
    'van_rossum_distance',
    'spike_train_synchrony',
    'burst_synchrony_index',
    'phase_locking_value',
    'spike_time_tiling_coefficient',
    'correlation_index',
    'cross_correlation',
]
