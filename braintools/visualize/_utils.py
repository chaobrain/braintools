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

import brainunit as u
import numpy as np

# The spike-format helpers live in ``braintools.spike``; they are re-exported
# here for the plotting functions.
from braintools.spike._convert import flatten_spikes, inter_spike_intervals  # noqa: F401
from braintools.spike._spike_trains import SpikeTrains
from braintools.tree import as_numpy


def as_array(x):
    """Convert an array-like (NumPy, JAX or ``brainunit`` array, or a nested
    Python list) to a NumPy array.

    ``as_numpy`` maps over Python containers and leaves a list a list; plotting
    functions that expect a single array need ``np.asarray`` on top of it.
    """
    return np.asarray(as_numpy(x))


def is_spike_trains(x):
    """Whether ``x`` is a :class:`braintools.spike.SpikeTrains`."""
    return isinstance(x, SpikeTrains)


def spike_trains_events(spike_trains, neuron_ids=None, unit=None):
    """Flatten a :class:`~braintools.spike.SpikeTrains` for plotting.

    Parameters
    ----------
    spike_trains : SpikeTrains
        The spike trains (concrete, not traced).
    neuron_ids : None
        Must be None: a ``SpikeTrains`` already carries the neuron indices.
    unit : brainunit.Unit, optional
        Unit to express the times in, when the trains carry units. Defaults
        to the trains' own unit.

    Returns
    -------
    times : numpy.ndarray
        Spike times as plain numbers, sorted.
    ids : numpy.ndarray
        Neuron index of every spike.
    unit : brainunit.Unit or None
        The unit of ``times``, or None for unitless trains.

    Raises
    ------
    ValueError
        If ``neuron_ids`` is given.
    """
    if neuron_ids is not None:
        raise ValueError('neuron_ids must be None when spike_times is a SpikeTrains; '
                         'the SpikeTrains already holds the neuron indices.')
    times, ids = spike_trains.to_events()
    if isinstance(times, u.Quantity):
        unit = times.unit if unit is None else unit
        times = times.to_decimal(unit)
    else:
        unit = None
    return np.asarray(times), np.asarray(ids), unit


def in_time_unit(value, unit):
    """Express a time value (or tuple of them) as plain numbers in ``unit``.

    Parameters
    ----------
    value : float, Quantity, tuple, list or None
        A time, or a tuple or list of times such as ``time_range``. Plain numbers are
        returned unchanged (they are taken to be in ``unit`` already).
    unit : brainunit.Unit or None
        The unit of the plotted spike times, or None when they are unitless.

    Returns
    -------
    float, tuple or None
        ``value`` with every Quantity converted to a number in ``unit``.

    Raises
    ------
    ValueError
        If ``value`` holds a Quantity but the spike times are unitless.
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(in_time_unit(v, unit) for v in value)
    if isinstance(value, u.Quantity):
        if unit is None:
            raise ValueError(f'Got the Quantity {value} but the spike times are unitless; '
                             f'pass a plain number instead.')
        return float(value.to_decimal(unit))
    return value
