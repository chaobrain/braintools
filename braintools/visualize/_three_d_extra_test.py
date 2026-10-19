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

    def _drawn_segments(self, ax):
        segments = []
        for line in ax.lines:
            try:
                xs, ys, zs = line.get_data_3d()
            except Exception:
                continue
            if len(xs) == 2:
                segments.append((np.array([xs[0], ys[0], zs[0]]),
                                 np.array([xs[1], ys[1], zs[1]])))
        return segments

    def test_all_documented_input_forms_draw_the_same_edges(self):
        """The documented "list of connections" form must actually be honoured.

        Regression test: the branch was chosen with ``connections.ndim == 2``, but
        a list of ``(source_idx, target_idx)`` pairs is *also* 2-D once coerced to
        an array -- so pair lists were silently read as an adjacency matrix (an
        arbitrary set of edges), and a plain Python list of tuples raised
        ``AttributeError: 'list' object has no attribute 'ndim'``.
        """
        pairs = [(0, 1), (2, 3), (4, 0)]
        expected = [(self.src[i], self.tgt[j]) for i, j in pairs]

        matrix = np.zeros((len(self.src), len(self.tgt)))
        for i, j in pairs:
            matrix[i, j] = 1.0

        object_array = np.empty(3, dtype=object)
        for k, pair in enumerate(pairs):
            object_array[k] = pair

        forms = {
            'pair list': pairs,
            'pair ndarray': np.array(pairs),
            'pair float ndarray': np.array(pairs, dtype=float),
            '1-D object array': object_array,
            'adjacency matrix': matrix,
        }
        for label, connections in forms.items():
            with self.subTest(form=label):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                connectivity_3d(self.src, self.tgt, connections, ax=ax)
                drawn = self._drawn_segments(ax)
                self.assertEqual(len(drawn), len(expected), f'{label}: wrong edge count')
                for start, end in expected:
                    self.assertTrue(
                        any(np.allclose(a, start, atol=1e-9) and np.allclose(b, end, atol=1e-9)
                            for a, b in drawn),
                        f'{label}: edge {start} -> {end} was not drawn',
                    )
                plt.close('all')

    def test_mismatched_shape_raises(self):
        with self.assertRaises(ValueError):
            connectivity_3d(self.src, self.tgt, np.zeros((3, 3, 3)))
        with self.assertRaises(ValueError):
            connectivity_3d(self.src, self.tgt, np.zeros((2, 5)))

    def test_empty_pair_list_draws_no_edges(self):
        ax = connectivity_3d(self.src, self.tgt, [])
        self.assertEqual(len(ax.lines), 0)

    def _edges(self, src, tgt, connections, **kw):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        connectivity_3d(src, tgt, connections, ax=ax, **kw)
        return sorted(
            (int(np.argmin(np.linalg.norm(src - a, axis=1))),
             int(np.argmin(np.linalg.norm(tgt - b, axis=1))))
            for a, b in self._drawn_segments(ax)
        )

    def test_ambiguous_shape_resolution(self):
        # n_target == 2 and two connections: shape (2, 2) fits both forms.
        src, tgt = np.random.randn(2, 3), np.random.randn(2, 3)
        pairs = [(0, 1), (1, 1)]
        as_pairs = [(0, 1), (1, 1)]
        as_matrix = [(0, 1), (1, 0), (1, 1)]   # nonzero entries of [[0, 1], [1, 1]]

        # a Python sequence is read as pairs, silently
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertEqual(self._edges(src, tgt, pairs), as_pairs)
        # an ndarray is read as a matrix, with a warning
        with self.assertWarns(UserWarning):
            self.assertEqual(self._edges(src, tgt, np.array(pairs)), as_matrix)
        # an explicit format overrides both, with no warning
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertEqual(
                self._edges(src, tgt, np.array(pairs), connection_format='pairs'), as_pairs)
            self.assertEqual(
                self._edges(src, tgt, pairs, connection_format='matrix'), as_matrix)

    def test_unambiguous_shapes_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            connectivity_3d(self.src, self.tgt, np.random.rand(5, 4) > 0.5)
            connectivity_3d(self.src, self.tgt, np.array([(0, 1), (2, 3)]))

    def test_explicit_format_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            connectivity_3d(self.src, self.tgt, np.array([(0, 1)]), connection_format='matrix')
        with self.assertRaises(ValueError):
            connectivity_3d(self.src, self.tgt, np.zeros((5, 4)), connection_format='pairs')
        with self.assertRaises(ValueError):
            connectivity_3d(self.src, self.tgt, [], connection_format='bogus')

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


class TestArtistKwargForwarding(unittest.TestCase):
    """``**kwargs`` is documented as "additional arguments passed to <artist>".

    Regression tests: each renderer expanded its own artist arguments and
    ``**kwargs`` separately, so a caller passing a natural matplotlib keyword that
    the function also sets (``alpha``, ``cmap``, ``s``, ``color``, ``linewidth``,
    ``label``, ``c``, ``facecolors``) hit a bare
    ``TypeError: got multiple values for keyword argument``. ``cmap`` was also
    hardcoded in ``neural_network_3d`` and ``electrode_array_3d`` with no parameter
    to override it, so it could not be set at all.
    """

    def tearDown(self):
        plt.close('all')

    def test_documented_artist_kwargs_do_not_collide(self):
        positions = np.random.RandomState(0).randn(9, 3)
        faces = np.arange(9).reshape(3, 3)
        cases = {
            'neural_network_3d alpha': lambda: neural_network_3d([3, 3], alpha=0.2),
            'neural_network_3d s': lambda: neural_network_3d([3, 3], s=10),
            'neural_network_3d cmap': lambda: neural_network_3d(
                [3, 3], activations=[np.zeros(3), np.zeros(3)], cmap='hot'),
            'electrode_array_3d cmap': lambda: electrode_array_3d(
                positions, signals=np.arange(9.0), cmap='hot'),
            'electrode_array_3d s': lambda: electrode_array_3d(
                positions, signals=np.arange(9.0), s=10),
            'dendrite_tree_3d color': lambda: dendrite_tree_3d(
                [((0, 0, 0), (1, 1, 1))], color='red'),
            'dendrite_tree_3d linewidth': lambda: dendrite_tree_3d(
                [((0, 0, 0), (1, 1, 1))], linewidth=5),
            'connectivity_3d label': lambda: connectivity_3d(
                positions[:4], positions[4:], np.zeros((4, 5)), label='x'),
            'trajectory_3d c': lambda: trajectory_3d(np.zeros((5, 3)), c=np.arange(5)),
            'phase_space_3d c': lambda: phase_space_3d(
                np.zeros(5), np.zeros(5), np.zeros(5), c=np.arange(5)),
            'brain_surface_3d facecolors': lambda: brain_surface_3d(
                positions, faces, facecolors='gray'),
        }
        for label, call in cases.items():
            with self.subTest(case=label):
                call()  # must not raise TypeError

    def test_activation_colormap_is_actually_applied(self):
        """``cmap`` must reach the scatter rather than being hardcoded."""
        activations = [np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 4)]
        drawn = {}
        for cmap in ('hot', 'viridis'):
            ax = _ax3d()
            neural_network_3d([5, 4], activations=activations, cmap=cmap, ax=ax)
            drawn[cmap] = np.asarray(ax.collections[0].get_facecolors())
            plt.close('all')
        self.assertFalse(
            np.allclose(drawn['hot'], drawn['viridis']),
            'cmap was ignored: both colormaps produced identical node colors',
        )

    def test_no_colormapping_warning_without_activations(self):
        """A colormap with nothing to map must not warn."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            neural_network_3d([4, 3], cmap='hot')
            electrode_array_3d(np.random.randn(4, 3), cmap='hot')
        self.assertFalse(
            [w for w in caught if 'colormapping' in str(w.message)],
            'a colormap was passed with no values to color by',
        )


class TestColorbarTargetsItsOwnFigure(unittest.TestCase):
    """``plt.colorbar`` acts on the *current* figure, not on ``ax``'s figure.

    Regression test: when ``ax`` belonged to a figure that was not current,
    ``plt.colorbar(scatter, ax=ax)`` emitted ``UserWarning: Adding colorbar to a
    different Figure ...`` while attaching the colorbar to ``ax``'s figure anyway.
    """

    def tearDown(self):
        plt.close('all')

    def test_no_warning_when_ax_figure_is_not_current(self):
        makers = {
            'trajectory_3d': lambda ax: trajectory_3d(
                np.zeros((5, 3)), ax=ax, time_colors=True),
            'phase_space_3d': lambda ax: phase_space_3d(
                np.zeros(5), np.zeros(5), np.zeros(5), ax=ax),
            'electrode_array_3d': lambda ax: electrode_array_3d(
                np.zeros((4, 3)), signals=np.arange(4.0), ax=ax),
        }
        for label, make in makers.items():
            with self.subTest(case=label):
                plt.close('all')
                target = plt.figure()
                ax = target.add_subplot(111, projection='3d')
                plt.figure()  # a decoy current figure
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    make(ax)
                self.assertFalse(
                    [w for w in caught if 'different Figure' in str(w.message)],
                    'the colorbar was added through pyplot global state',
                )
                # and it still landed on the figure that owns ``ax``
                self.assertEqual(len(target.axes), 2)


if __name__ == '__main__':
    unittest.main()


def test_volume_rendering_default_threshold_handles_negative_data():
    import braintools
    vol = -np.ones((4, 4, 4))
    vol[1:3, 1:3, 1:3] = -0.1
    ax = braintools.visualize.volume_rendering(vol)
    assert len(ax.collections) > 0      # the high-valued core is drawn
    plt.close('all')


def test_electrode_array_signal_scale_sets_marker_sizes():
    import braintools
    pos = np.random.default_rng(0).normal(size=(3, 3))
    ax = braintools.visualize.electrode_array_3d(pos, signals=np.array([0.0, 1.0, 2.0]), signal_scale=0.5)
    assert np.allclose(ax.collections[0].get_sizes(), [10.0, 32.5, 55.0])
    plt.close('all')


def test_neural_network_3d_rejects_transposed_weights():
    import pytest
    import braintools
    with pytest.raises(ValueError):
        braintools.visualize.neural_network_3d([3, 2], weights=[np.ones((2, 3))])
    plt.close('all')
