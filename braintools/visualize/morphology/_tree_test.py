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

"""Tests for the tree view in :mod:`braintools.visualize.morphology._tree`."""

import unittest

import brainunit as u

from braintools.visualize.morphology import as_morphology_data
from braintools.visualize.morphology._testing import (
    make_four_type_tree,
    make_projected_node_tree,
)
from braintools.visualize.morphology._tree import MorphologyTree, as_tree


class MorphologyTreeTest(unittest.TestCase):
    def test_tree_is_built_once_per_data_instance(self) -> None:
        data = as_morphology_data(make_four_type_tree())
        tree = as_tree(data)
        self.assertIs(as_tree(data), tree)
        self.assertIs(as_tree(tree), tree)
        self.assertIs(tree.__morphology_data__(), data)

    def test_nodes_navigate_like_the_data(self) -> None:
        tree = as_tree(make_four_type_tree())
        root = tree.root
        self.assertIsNone(root.parent)
        self.assertEqual(
            [child.name for child in root.children], ["apical", "basal", "axon"]
        )
        self.assertEqual(root.n_children, 3)
        apical = root.children[0]
        self.assertIs(apical.parent, root)
        self.assertEqual(
            (apical.index, apical.type, apical.parent_x, apical.child_x),
            (1, "apical_dendrite", 1.0, 0.0),
        )
        self.assertEqual(apical.n_segments, 2)
        self.assertAlmostEqual(float(apical.length.to_decimal(u.um)), 90.0)
        self.assertIs(apical.branch, tree.data.branches[1])
        self.assertIs(apical.lengths, apical.branch.lengths)
        self.assertIs(apical.radii_proximal, apical.branch.radii_proximal)
        self.assertIs(apical.radii_distal, apical.branch.radii_distal)
        self.assertIsNone(apical.points_proximal)
        self.assertIsNone(apical.points_distal)
        self.assertIn("apical", repr(apical))
        self.assertIsNone(tree.name)
        self.assertFalse(tree.has_full_point_geometry)
        self.assertTrue(as_tree(make_projected_node_tree()).has_full_point_geometry)
        self.assertIsInstance(tree, MorphologyTree)

    def test_rejection_names_the_caller(self) -> None:
        with self.assertRaisesRegex(
            TypeError, r"^plot_x\(\.\.\.\) expected MorphologyData"
        ):
            as_tree(1, caller="plot_x(...)")
        with self.assertRaisesRegex(TypeError, r"^expected MorphologyData"):
            as_tree(1)


if __name__ == "__main__":
    unittest.main()
