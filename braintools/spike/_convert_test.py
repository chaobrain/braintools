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

import numpy as np
import pytest

from braintools.spike._convert import flatten_spikes, inter_spike_intervals


def test_flatten_spikes_formats():
    t, i = flatten_spikes([np.array([0.1, 0.3]), np.array([0.2])])
    assert np.allclose(t, [0.1, 0.3, 0.2]) and list(i) == [0, 0, 1]
    t, i = flatten_spikes([0.1, 0.2, 0.4])          # a list of numbers is one train
    assert list(i) == [0, 0, 0]
    t, i = flatten_spikes(np.array([0.1, 0.2]), neuron_ids=[3, 4])
    assert list(i) == [3, 4]
    with pytest.raises(ValueError):
        flatten_spikes(np.array([0.1, 0.2]), neuron_ids=[1])


def test_inter_spike_intervals_stay_within_neurons():
    times = np.array([0.0, 0.05, 1.0, 1.1])
    ids = np.array([0, 1, 0, 1])
    assert np.allclose(sorted(inter_spike_intervals(times, ids)), [1.0, 1.05])


def test_flatten_spikes_empty_and_units_free_arrays():
    t, i = flatten_spikes([])
    assert t.shape == (0,) and i.shape == (0,)
    t, i = flatten_spikes([np.array([]), np.array([0.5])])
    assert t.tolist() == [0.5] and i.tolist() == [1]
    t, i = flatten_spikes(np.array([0.3, 0.1]))  # flat array without ids: one train
    assert i.tolist() == [0, 0]


def test_inter_spike_intervals_sorts_and_handles_empty():
    assert inter_spike_intervals(np.zeros(0), np.zeros(0, int)).shape == (0,)
    assert np.allclose(inter_spike_intervals(np.array([3.0, 1.0]), np.array([0, 0])), [2.0])
