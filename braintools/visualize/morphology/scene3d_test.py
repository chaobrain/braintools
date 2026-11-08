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

"""Tests for :mod:`braintools.visualize.morphology.scene3d`."""

import unittest

import brainunit as u
import numpy as np

from braintools.visualize.morphology._testing import Branch, Morphology
from braintools.visualize.morphology import _testing as _sel
from braintools.visualize.morphology import plot3d
from braintools.visualize.morphology._testing import (
    FakeBackend,
    VisDefaultsResetMixin,
    make_projected_node_tree,
)
from braintools.visualize.morphology.backend import BackendChooser
from braintools.visualize.morphology.scene3d import build_render_scene_3d


class BuildRenderScene3dTest(unittest.TestCase):
    def test_build_render_scene_3d_groups_branches_by_type(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
            radii=[5.0, 5.0] * u.um,
            type="soma",
        )
        axon = Branch.from_points(
            points=[[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]] * u.um,
            radii=[1.0, 0.8, 0.6] * u.um,
            type="axon",
        )
        dend = Branch.from_points(
            points=[[10.0, 0.0, 0.0], [10.0, 10.0, 0.0]] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="apical_dendrite",
        )
        tree = Morphology.from_root(soma, name="soma")
        tree.attach(
            parent="soma", child_branch=axon, child_name="axon_slot", parent_x=1.0
        )
        tree.attach(
            parent="soma", child_branch=dend, child_name="dend_slot", parent_x=1.0
        )

        scene = build_render_scene_3d(tree)

        self.assertEqual(len(scene.branches), 3)
        self.assertEqual(
            {batch.branch_type for batch in scene.batches},
            {"soma", "axon", "apical_dendrite"},
        )
        soma_branch = scene.branches[0]
        self.assertEqual(soma_branch.branch_name, "soma")
        self.assertEqual(soma_branch.points_um.shape, (2, 3))
        self.assertTrue(np.allclose(soma_branch.radii_um, np.array([5.0, 5.0])))
        self.assertEqual(scene.mode, "geometry")

    def test_build_render_scene_3d_carries_skeleton_mode(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
            radii=[5.0, 5.0] * u.um,
            type="soma",
        )
        tree = Morphology.from_root(soma, name="soma")

        scene = build_render_scene_3d(tree, mode="skeleton")

        self.assertEqual(scene.mode, "skeleton")

    def test_build_render_scene_3d_requires_point_geometry(self) -> None:
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma"
        )
        tree = Morphology.from_root(soma, name="soma")

        with self.assertRaises(ValueError):
            build_render_scene_3d(tree)


class Scene3dOverlayTest(VisDefaultsResetMixin, unittest.TestCase):
    def test_plot3d_with_region_overlay_emits_highlight_strokes(self) -> None:
        tree = make_projected_node_tree()
        region = _sel.all_region(tree)
        backend = FakeBackend()

        rendered = plot3d(
            tree, region=region, chooser=BackendChooser(backends=(backend,))
        )

        self.assertIs(rendered.overlay.region, region)
        self.assertEqual(len(rendered.scene.highlight_strokes), len(tree.branches))
        for stroke in rendered.scene.highlight_strokes:
            self.assertEqual(stroke.points_um.shape[1], 3)

    def test_plot3d_with_locset_overlay_emits_markers(self) -> None:
        tree = make_projected_node_tree()
        locset = _sel.LocsetData(
            _sel.branch_points(tree).points + _sel.terminals(tree).points
        )
        backend = FakeBackend()

        rendered = plot3d(
            tree, locset=locset, chooser=BackendChooser(backends=(backend,))
        )

        self.assertIs(rendered.overlay.locset, locset)
        self.assertEqual(len(rendered.scene.markers), len(locset.points))
        for marker in rendered.scene.markers:
            self.assertEqual(marker.position_um.shape, (3,))

    def test_plot3d_per_branch_values_emit_value_batches(self) -> None:
        tree = make_projected_node_tree()
        backend = FakeBackend()

        rendered = plot3d(
            tree,
            values=np.array([0.25, 0.75]),
            chooser=BackendChooser(backends=(backend,)),
        )

        self.assertGreaterEqual(len(rendered.scene.value_batches), 1)
        self.assertIsNotNone(rendered.scene.value_spec)


if __name__ == "__main__":
    unittest.main()
