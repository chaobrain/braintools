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

"""Spike-train analysis: firing rates, spike-train distances and synchrony.

Most functions take a spike matrix of shape ``(n_time_steps, n_neurons)``
(time on axis 0) together with the sampling step ``dt``; the two distance
measures take arrays of spike times.

- **Rates**: :func:`firing_rate`.
- **Distances between two trains**: :func:`victor_purpura_distance`,
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
    >>> spike.victor_purpura_distance(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))
    0.0
"""

from ._correlation import (
    cross_correlation,
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

__all__ = [
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
