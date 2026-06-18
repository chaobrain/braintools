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

"""Comprehensive coverage for braintools.visualize._three_d."""

import unittest
import warnings

import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from braintools.visualize import (
    neural_network_3d,
    brain_surface_3d,
    connectivity_3d,
    trajectory_3d,
    volume_rendering,
    electrode_array_3d,
    dendrite_tree_3d,
    phase_space_3d,
)


def _ax3d():
    fig = plt.figure()
    return fig.add_subplot(111, projection='3d')


class TestNeuralNetwork3D(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_basic(self):
        ax = neural_network_3d([4, 6, 2])
        self.assertIsNotNone(ax)

    def test_single_neuron_layer(self):
        # layer_size == 1 -> single neuron placed at the centre
        ax = neural_network_3d([1, 3, 1])
        self.assertIsNotNone(ax)

    def test_with_activations(self):
        sizes = [3, 4]
        activations = [np.random.rand(3), np.random.rand(4)]
        ax = neural_network_3d(sizes, activations=activations)
        self.assertIsNotNone(ax)

    def test_with_weights(self):
        sizes = [3, 4, 2]
        weights = [np.random.randn(3, 4), np.random.randn(4, 2)]
        ax = neural_network_3d(sizes, weights=weights, title='net')
        self.assertIsNotNone(ax)

    def test_with_existing_axis(self):
        ax = _ax3d()
        out = neural_network_3d([2, 2], ax=ax)
        self.assertIs(out, ax)


class TestBrainSurface3D(unittest.TestCase):
    def setUp(self):
        self.vertices = np.random.randn(20, 3)
        self.faces = np.random.randint(0, 20, (12, 3))

    def tearDown(self):
        plt.close('all')

    def test_no_values(self):
        ax = brain_surface_3d(self.vertices, self.faces)
        self.assertIsNotNone(ax)

    def test_with_values_no_deprecation(self):
        values = np.random.rand(20)
        with warnings.catch_warnings():
            warnings.simplefilter('error', matplotlib.MatplotlibDeprecationWarning)
            ax = brain_surface_3d(self.vertices, self.faces, values=values, title='brain')
        self.assertIsNotNone(ax)

    def test_all_zero_values_no_nan(self):
        # degenerate range must not produce NaN colors / crash
        ax = brain_surface_3d(self.vertices, self.faces, values=np.zeros(20))
        self.assertIsNotNone(ax)

    def test_with_existing_axis(self):
        ax = _ax3d()
        out = brain_surface_3d(self.vertices, self.faces, ax=ax)
        self.assertIs(out, ax)


class TestConnectivity3D(unittest.TestCase):
    def setUp(self):
        self.src = np.random.randn(5, 3)
        self.tgt = np.random.randn(4, 3)

    def tearDown(self):
        plt.close('all')

    def test_adjacency(self):
        conns = (np.random.rand(5, 4) > 0.5)
        ax = connectivity_3d(self.src, self.tgt, conns)
        self.assertIsNotNone(ax)

    def test_adjacency_with_strengths(self):
        conns = (np.random.rand(5, 4) > 0.5)
        strengths = np.random.rand(5, 4)
        ax = connectivity_3d(self.src, self.tgt, conns,
                             connection_strengths=strengths)
        self.assertIsNotNone(ax)

    def test_edge_list_format(self):
        # a non-2D (object) array of (src, tgt) pairs exercises the
        # list-of-connections branch
        conns = np.empty(3, dtype=object)
        conns[0], conns[1], conns[2] = (0, 1), (2, 3), (4, 0)
        ax = connectivity_3d(self.src, self.tgt, conns)
        self.assertIsNotNone(ax)

    def test_node_colors_and_sizes(self):
        conns = (np.random.rand(5, 4) > 0.5)
        node_colors = np.arange(9)
        node_sizes = np.full(9, 80)
        ax = connectivity_3d(self.src, self.tgt, conns,
                             node_colors=node_colors, node_sizes=node_sizes,
                             title='conn')
        self.assertIsNotNone(ax)

    def test_with_existing_axis(self):
        ax = _ax3d()
        out = connectivity_3d(self.src, self.tgt, np.random.rand(5, 4) > 0.5, ax=ax)
        self.assertIs(out, ax)


class TestTrajectory3D(unittest.TestCase):
    def setUp(self):
        self.traj = np.random.randn(40, 3)

    def tearDown(self):
        plt.close('all')

    def test_time_colors(self):
        ax = trajectory_3d(self.traj, time_colors=True, title='t')
        self.assertIsNotNone(ax)

    def test_no_time_colors(self):
        ax = trajectory_3d(self.traj, time_colors=False)
        self.assertIsNotNone(ax)

    def test_with_existing_axis(self):
        ax = _ax3d()
        out = trajectory_3d(self.traj, ax=ax)
        self.assertIs(out, ax)


class TestVolumeRendering(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_default_threshold(self):
        vol = np.random.rand(6, 6, 6)
        ax = volume_rendering(vol)
        self.assertIsNotNone(ax)

    def test_explicit_threshold(self):
        vol = np.random.rand(6, 6, 6)
        ax = volume_rendering(vol, threshold=0.7, alpha=0.2, title='v')
        self.assertIsNotNone(ax)

    def test_with_existing_axis(self):
        ax = _ax3d()
        out = volume_rendering(np.random.rand(5, 5, 5), ax=ax)
        self.assertIs(out, ax)


class TestElectrodeArray3D(unittest.TestCase):
    def setUp(self):
        self.pos = np.random.randn(8, 3)

    def tearDown(self):
        plt.close('all')

    def test_no_signals(self):
        ax = electrode_array_3d(self.pos)
        self.assertIsNotNone(ax)

    def test_signals_1d(self):
        ax = electrode_array_3d(self.pos, signals=np.random.randn(8))
        self.assertIsNotNone(ax)

    def test_signals_2d(self):
        ax = electrode_array_3d(self.pos, signals=np.random.randn(8, 5))
        self.assertIsNotNone(ax)

    def test_labels(self):
        labels = [f'E{i}' for i in range(8)]
        ax = electrode_array_3d(self.pos, electrode_labels=labels, title='arr')
        self.assertIsNotNone(ax)

    def test_with_existing_axis(self):
        ax = _ax3d()
        out = electrode_array_3d(self.pos, ax=ax)
        self.assertIs(out, ax)


class TestDendriteTree3D(unittest.TestCase):
    def setUp(self):
        coords = np.random.randn(10, 3)
        self.segments = [(coords[i], coords[i + 1]) for i in range(9)]

    def tearDown(self):
        plt.close('all')

    def test_basic(self):
        ax = dendrite_tree_3d(self.segments)
        self.assertIsNotNone(ax)

    def test_diameters_and_colors(self):
        diameters = [0.1 * (i + 1) for i in range(9)]
        colors = ['red'] * 9
        ax = dendrite_tree_3d(self.segments, diameters=diameters,
                             colors=colors, title='tree')
        self.assertIsNotNone(ax)

    def test_with_existing_axis(self):
        ax = _ax3d()
        out = dendrite_tree_3d(self.segments, ax=ax)
        self.assertIs(out, ax)


class TestPhaseSpace3D(unittest.TestCase):
    def setUp(self):
        t = np.linspace(0, 10, 100)
        self.x, self.y, self.z = np.sin(t), np.cos(t), t / 10

    def tearDown(self):
        plt.close('all')

    def test_time_colors(self):
        ax = phase_space_3d(self.x, self.y, self.z, time_colors=True, title='ps')
        self.assertIsNotNone(ax)

    def test_no_time_colors(self):
        ax = phase_space_3d(self.x, self.y, self.z, time_colors=False)
        self.assertIsNotNone(ax)

    def test_with_existing_axis(self):
        ax = _ax3d()
        out = phase_space_3d(self.x, self.y, self.z, ax=ax)
        self.assertIs(out, ax)


if __name__ == '__main__':
    unittest.main()
