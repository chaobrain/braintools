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

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import braintools
from braintools.visualize._utils import as_array, flatten_spikes, inter_spike_intervals

v = braintools.visualize


def test_as_array_converts_python_lists():
    assert isinstance(as_array([[1, 2], [3, 4]]), np.ndarray)
    assert as_array([[1, 2], [3, 4]]).shape == (2, 2)


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


@pytest.mark.parametrize('call', [
    lambda: v.learning_curve([10, 20, 30], [0.6, 0.7, 0.8], [0.5, 0.6, 0.7]),
    lambda: v.residual_plot([1.0, 2.0, 3.0], [1.1, 1.9, 3.2]),
    lambda: v.regression_plot([1.0, 2.0, 3.0, 4.0], [2.0, 4.1, 5.9, 8.2]),
    lambda: v.tuning_curve([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 2.0, 0.5], bins=2),
    lambda: v.population_activity([0.1, 0.4, 0.3]),
    lambda: v.correlation_matrix([[1, 2], [2, 3.5], [3, 5.0]]),
    lambda: v.network_topology([[0, 1], [1, 0]], layout='circular'),
    lambda: v.firing_rate_map([[1, 2], [3, 4.0]]),
    lambda: v.neural_trajectory([[0, 1], [1, 2], [2, 0.0]]),
    lambda: v.isi_distribution([0.1, 0.25, 0.5]),
    lambda: v.spike_histogram([0.1, 0.25, 0.5]),
    lambda: v.spike_raster([0.1, 0.2], neuron_ids=[0, 1]),
])
def test_plots_accept_python_lists(call):
    call()
    plt.close('all')


def test_dashboard_isis_computed_per_neuron():
    pytest.importorskip('plotly')
    fig = v.dashboard_neural_activity(np.array([0.0, 0.05, 1.0, 1.1]), neuron_ids=np.array([0, 1, 0, 1]))
    hist = [tr for tr in fig.data if tr.type == 'histogram'][0]
    assert np.allclose(sorted(hist.x), [1.0, 1.05])


def test_dashboard_rates_use_recording_span():
    pytest.importorskip('plotly')
    # neuron 0 spans the 2 s recording with 4 spikes; neuron 1 fires twice, 10 ms apart
    times = np.array([0.0, 0.5, 1.0, 2.0, 1.0, 1.01])
    ids = np.array([0, 0, 0, 0, 1, 1])
    fig = v.dashboard_neural_activity(times, neuron_ids=ids)
    rate_hist = [tr for tr in fig.data if tr.type == 'histogram'][1]
    assert np.allclose(sorted(rate_hist.x), [1.0, 2.0])
