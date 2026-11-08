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


import unittest
import warnings
from unittest import mock

import brainunit as u
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from braintools.visualize.morphology import TopologyGraph, plot_topology_graph
from braintools.visualize.morphology._testing import (
    BaselineMixin,
    make_baseline_cv_graph,
    make_baseline_node_graph,
)
from braintools.visualize.morphology.point_topology import (
    _blend_node_rgba,
    _plot_discrete_topology_graph,
    _resolve_coverage_intensities,
)


class _NodeTreeLike:
    """Stands in for a simulator object implementing ``__topology_data__()``.

    The graph is the solver-node tree of a soma with two one-CV dendrites:
    each CV contributes a proximal boundary, a midpoint and a distal
    boundary node, shared at the junctions.
    """

    def __topology_data__(self) -> TopologyGraph:
        return TopologyGraph(
            node_ids=(0, 1, 2, 3, 4, 5, 6),
            edges=((0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)),
            root_id=0,
        )

    @property
    def nodes(self) -> tuple[int, ...]:
        return self.__topology_data__().node_ids


def _make_node_tree() -> _NodeTreeLike:
    return _NodeTreeLike()


class PointTopologyPlotTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_plot_topology_graph_returns_axes(self) -> None:
        node_tree = _make_node_tree()
        ax = plot_topology_graph(node_tree, layout="kamada_kawai")
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_invalid_layout_scale_is_rejected(self) -> None:
        node_tree = _make_node_tree()
        for bad_value in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(
                    ValueError, "layout_scale must be a finite positive float"
                ):
                    plot_topology_graph(
                        node_tree, layout="kamada_kawai", layout_scale=bad_value
                    )

    def test_rejects_highlight_and_values_together(self) -> None:
        node_tree = _make_node_tree()
        with self.assertRaisesRegex(
            ValueError, "does not support highlight_point_ids together with values"
        ):
            plot_topology_graph(
                node_tree,
                highlight_point_ids=[1],
                values=np.linspace(0.0, 1.0, len(node_tree.nodes)),
            )

    def test_the_shared_renderer_attributes_errors_to_its_caller(self) -> None:
        # _plot_discrete_topology_graph can sit behind more than one public
        # entry point, so its messages must name whichever one the user
        # actually called rather than a hardcoded favourite.
        with self.assertRaisesRegex(
            ValueError, r"^plot_cell_topology\(level='cv', \.\.\.\) does not support"
        ):
            _plot_discrete_topology_graph(
                node_ids=(0, 1),
                edges=((0, 1),),
                root_id=0,
                highlight_point_ids=[1],
                values=np.asarray([0.0, 1.0]),
                caller="plot_cell_topology(level='cv', ...)",
            )

    def test_values_shape_must_match_n_points(self) -> None:
        node_tree = _make_node_tree()
        with self.assertRaisesRegex(ValueError, "values must have shape"):
            plot_topology_graph(
                node_tree,
                layout="kamada_kawai",
                color_mode="values",
                values=np.ones(len(node_tree.nodes) - 1),
            )

    def test_graphviz_layout_falls_back_to_kamada_kawai(self) -> None:
        node_tree = _make_node_tree()
        with mock.patch(
            "braintools.visualize.morphology.point_topology._graphviz_layout_positions",
            side_effect=RuntimeError("graphviz unavailable"),
        ):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                ax = plot_topology_graph(node_tree, layout="twopi")
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertTrue(
            any(
                "falling back to 'kamada_kawai'" in str(item.message)
                for item in captured
            )
        )

    def test_depth_color_mode_draws_all_points(self) -> None:
        node_tree = _make_node_tree()
        ax = plot_topology_graph(node_tree, preset="depth", layout="kamada_kawai")
        offsets = ax.collections[-1].get_offsets()
        self.assertEqual(offsets.shape[0], len(node_tree.nodes))

    def test_kamada_kawai_layout_scale_changes_point_spacing(self) -> None:
        node_tree = _make_node_tree()
        ax_small = plot_topology_graph(
            node_tree, layout="kamada_kawai", layout_scale=0.5
        )
        ax_large = plot_topology_graph(
            node_tree, layout="kamada_kawai", layout_scale=2.0
        )

        coords_small = np.asarray(ax_small.collections[-1].get_offsets(), dtype=float)
        coords_large = np.asarray(ax_large.collections[-1].get_offsets(), dtype=float)
        radius_small = float(np.max(np.linalg.norm(coords_small, axis=1)))
        radius_large = float(np.max(np.linalg.norm(coords_large, axis=1)))
        self.assertGreater(radius_large, radius_small)

    def test_graphviz_layout_scale_scales_returned_positions(self) -> None:
        node_tree = _make_node_tree()
        node_ids = tuple(point for point in node_tree.nodes)
        base_positions = {
            node_id: np.asarray((index + 1.0, (index + 1.0) * 2.0), dtype=float)
            for index, node_id in enumerate(node_ids)
        }
        with mock.patch(
            "braintools.visualize.morphology.point_topology._graphviz_layout_positions",
            return_value=base_positions,
        ):
            ax = plot_topology_graph(node_tree, layout="twopi", layout_scale=2.0)
        coords = np.asarray(ax.collections[-1].get_offsets(), dtype=float)
        expected = np.asarray(
            [base_positions[node_id] * 2.0 for node_id in node_ids], dtype=float
        )
        np.testing.assert_allclose(coords, expected)

    def test_graphviz_fallback_preserves_layout_scale(self) -> None:
        node_tree = _make_node_tree()
        with mock.patch(
            "braintools.visualize.morphology.point_topology._graphviz_layout_positions",
            side_effect=RuntimeError("graphviz unavailable"),
        ):
            with mock.patch(
                "braintools.visualize.morphology.point_topology._kamada_kawai_positions",
                return_value={
                    point: np.zeros(2, dtype=float) for point in node_tree.nodes
                },
            ) as mocked:
                plot_topology_graph(node_tree, layout="twopi", layout_scale=2.5)
        mocked.assert_called_once()
        self.assertEqual(mocked.call_args.kwargs["layout_scale"], 2.5)

    def test_highlight_mode_draws_selected_overlay(self) -> None:
        node_tree = _make_node_tree()
        ax = plot_topology_graph(
            node_tree, highlight_point_ids=[3], layout="kamada_kawai"
        )
        highlighted = np.asarray(ax.collections[-1].get_offsets(), dtype=float)
        self.assertEqual(highlighted.shape[0], 1)

    def test_values_mode_can_draw_colorbar(self) -> None:
        node_tree = _make_node_tree()
        ax = plot_topology_graph(
            node_tree,
            values=np.linspace(-70.0, -50.0, len(node_tree.nodes)),
            value_label="V",
            show_colorbar=True,
        )
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(len(ax.figure.axes), 2)

    def test_fraction_coverage_mode_preserves_fraction(self) -> None:
        intensities = _resolve_coverage_intensities(
            node_ids=(0, 1, 2),
            highlight_fractions={1: 0.5},
            coverage_mode="fraction",
        )
        np.testing.assert_allclose(intensities, np.asarray([0.0, 0.5, 0.0]))

    def test_any_coverage_mode_promotes_partial_overlap(self) -> None:
        intensities = _resolve_coverage_intensities(
            node_ids=(0, 1, 2),
            highlight_fractions={1: 0.2},
            coverage_mode="any",
        )
        np.testing.assert_allclose(intensities, np.asarray([0.0, 1.0, 0.0]))

    def test_all_coverage_mode_requires_full_overlap(self) -> None:
        intensities = _resolve_coverage_intensities(
            node_ids=(0, 1, 2),
            highlight_fractions={1: 0.999, 2: 1.0},
            coverage_mode="all",
        )
        np.testing.assert_allclose(intensities, np.asarray([0.0, 0.0, 1.0]))

    def test_invalid_coverage_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported coverage_mode"):
            _resolve_coverage_intensities(
                node_ids=(0, 1),
                highlight_fractions={0: 1.0},
                coverage_mode="bad",  # type: ignore[arg-type]
            )

    def test_blend_node_rgba_matches_expected_mix(self) -> None:
        base = np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=float)
        blended = _blend_node_rgba(
            base,
            highlight_color="#ff0000",
            intensities=np.asarray([0.5], dtype=float),
        )
        np.testing.assert_allclose(blended[0, :3], np.asarray([0.5, 0.0, 0.5]))


class TopologyGraphInputTest(unittest.TestCase):
    """What :func:`plot_topology_graph` reads from the graph itself."""

    def tearDown(self) -> None:
        plt.close("all")

    def _graph(self, **kwargs) -> TopologyGraph:
        return TopologyGraph(
            node_ids=(0, 1, 2), edges=((0, 1), (0, 2)), root_id=0, **kwargs
        )

    def _colours(self, ax) -> np.ndarray:
        return np.asarray(ax.collections[-1].get_facecolors(), dtype=float)

    def test_graph_values_colour_the_nodes_and_frame_selects_a_row(self) -> None:
        graph = self._graph(
            values=np.array([[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]]) * u.mV, value_label="V"
        )
        ax = plot_topology_graph(graph, frame=1, layout="kamada_kawai")
        direct = plot_topology_graph(
            self._graph(),
            values=np.array([5.0, 6.0, 7.0]) * u.mV,
            layout="kamada_kawai",
        )
        np.testing.assert_array_equal(self._colours(ax), self._colours(direct))
        self.assertEqual(ax.figure.axes[-1].get_ylabel(), "V [mV]")
        with self.assertRaisesRegex(ValueError, "pass frame=<index>"):
            plot_topology_graph(graph, layout="kamada_kawai")

    def test_values_or_fractions_in_both_places_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "values both in the graph and as an argument"
        ):
            plot_topology_graph(
                self._graph(values=np.zeros(3)),
                values=np.ones(3),
                layout="kamada_kawai",
            )
        with self.assertRaisesRegex(ValueError, "highlight_fractions both"):
            plot_topology_graph(
                self._graph(highlight_fractions={1: 1.0}),
                highlight_fractions={2: 1.0},
                layout="kamada_kawai",
            )

    def test_graph_highlight_fractions_are_drawn(self) -> None:
        graph = self._graph(highlight_fractions={1: 1.0})
        a = plot_topology_graph(graph, layout="kamada_kawai")
        b = plot_topology_graph(
            self._graph(), highlight_fractions={1: 1.0}, layout="kamada_kawai"
        )
        np.testing.assert_array_equal(
            a.collections[-1].get_facecolors(), b.collections[-1].get_facecolors()
        )

    def test_branch_level_graphs_are_topology_only(self) -> None:
        graph = self._graph(level="branch")
        plot_topology_graph(graph, layout="kamada_kawai")
        for kwargs in (
            dict(cmap="viridis"),
            dict(vmin=0.0),
            dict(show_colorbar=False),
            dict(values=np.ones(3)),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "level='branch' graph"):
                    plot_topology_graph(graph, layout="kamada_kawai", **kwargs)

    def test_non_graph_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "__topology_data__"):
            plot_topology_graph([(0, 1)])

    def test_missing_networkx_names_the_extra(self) -> None:
        with mock.patch.dict("sys.modules", {"networkx": None}):
            with self.assertRaisesRegex(
                ImportError, r"pip install braintools\[morphology\]"
            ):
                plot_topology_graph(self._graph(), layout="kamada_kawai")


class TopologyGraphPixelBaselineTest(BaselineMixin, unittest.TestCase):
    """Pixel baselines for topology graphs; regenerate with ``BRAINTOOLS_UPDATE_BASELINES=1``.

    The graphs are the ones braincell builds for the baseline tree, so these
    images are the same files as braincell's CV-level cell topology and
    point topology baselines.
    """

    def test_cv_level_with_highlight(self) -> None:
        fig, ax = self.new_axes()
        plot_topology_graph(make_baseline_cv_graph(), layout="kamada_kawai", ax=ax)
        self.assert_baseline(fig, "cell_topology_cv")

    def test_node_level(self) -> None:
        fig, ax = self.new_axes()
        plot_topology_graph(make_baseline_node_graph(), layout="kamada_kawai", ax=ax)
        self.assert_baseline(fig, "point_topology")


if __name__ == "__main__":
    unittest.main()
