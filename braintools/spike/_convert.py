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

"""Host-side (NumPy) helpers that convert between spike-time formats.

These helpers are private: :class:`braintools.spike.SpikeTrains` uses them in
its host-only constructors and converters, and ``braintools.visualize`` uses
them to accept the legacy spike-time formats.
"""

import numpy as np

from braintools.tree import as_numpy

__all__ = [
    'flatten_spikes',
    'inter_spike_intervals',
]


def _as_array(x):
    """Convert an array-like (NumPy, JAX, nested list) to a NumPy array."""
    return np.asarray(as_numpy(x))


def flatten_spikes(spike_times, neuron_ids=None):
    """Flatten spike-time input into ``(times, ids)`` arrays.

    Parameters
    ----------
    spike_times : array-like or list of array-like
        Accepted formats:

        - a list (or tuple) of arrays, one spike train per neuron
          (``neuron_ids`` is then ignored); a list of numbers is a single
          train;
        - a flat array of spike times with matching ``neuron_ids``;
        - a flat array without ``neuron_ids``: a single train (ids all 0).
    neuron_ids : array-like, optional
        Neuron index of every spike in a flat ``spike_times`` array.

    Returns
    -------
    times : numpy.ndarray
        Flat spike times, in input order.
    ids : numpy.ndarray
        Neuron index of every spike.

    Raises
    ------
    ValueError
        If ``neuron_ids`` and a flat ``spike_times`` differ in length.
    """
    if isinstance(spike_times, (list, tuple)) and neuron_ids is None:
        if all(np.ndim(t) == 0 for t in spike_times):
            times = _as_array(spike_times).ravel()
            return times, np.zeros(len(times), dtype=int)
        trains = [_as_array(t).ravel() for t in spike_times]
        times = np.concatenate(trains) if trains else np.zeros(0)
        ids = np.concatenate([np.full(len(t), i, dtype=int) for i, t in enumerate(trains)]) if trains else np.zeros(0, int)
        return times, ids
    times = _as_array(spike_times).ravel()
    if neuron_ids is None:
        return times, np.zeros(len(times), dtype=int)
    ids = _as_array(neuron_ids).ravel()
    if len(ids) != len(times):
        raise ValueError(f"neuron_ids has {len(ids)} entries but there are {len(times)} spike times.")
    return times, ids


def inter_spike_intervals(times, ids):
    """Compute inter-spike intervals within each neuron.

    Parameters
    ----------
    times : numpy.ndarray
        Flat spike times.
    ids : numpy.ndarray
        Neuron index of every spike.

    Returns
    -------
    numpy.ndarray
        The intervals between consecutive spikes of the same neuron (spike
        times are sorted per neuron first), concatenated over neurons.
    """
    isis = [np.diff(np.sort(times[ids == i])) for i in np.unique(ids)]
    return np.concatenate(isis) if isis else np.zeros(0)
