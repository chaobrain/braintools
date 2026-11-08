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

"""Tests for the neutral data types in :mod:`braintools.visualize.morphology._data`."""

import dataclasses
import unittest

import brainunit as u
import numpy as np

import braintools.visualize.morphology as bvm
from braintools.visualize.morphology._data import _as_locset_data, _as_region_data


def _branch(index, name="b", parent=None, *, n_seg=1, points=False, **overrides):
    kwargs = dict(
        index=index,
        name=name,
        type="soma" if parent is None else "basal_dendrite",
        parent_index=parent,
        parent_x=None if parent is None else 1.0,
        child_x=None if parent is None else 0.0,
        lengths=np.full(n_seg, 10.0) * u.um,
        radii_proximal=np.full(n_seg, 2.0) * u.um,
        radii_distal=np.full(n_seg, 1.0) * u.um,
    )
    if points:
        xs = np.arange(n_seg + 1, dtype=float) * 10.0
        pts = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
        kwargs.update(points_proximal=pts[:-1] * u.um, points_distal=pts[1:] * u.um)
    kwargs.update(overrides)
    return bvm.BranchData(**kwargs)


def _tree(*parents, **kwargs):
    """Branch ``i`` has parent ``parents[i]``; branch 0 is the root."""
    branches = [_branch(0, "soma")] + [
        _branch(i, f"d{i}", p) for i, p in enumerate(parents, start=1)
    ]
    return bvm.MorphologyData(branches=tuple(branches), **kwargs)


class BranchDataTest(unittest.TestCase):
    def test_geometry_is_stored_as_read_only_float64_micrometres(self) -> None:
        branch = _branch(
            0,
            lengths=[0.01, 0.02] * u.mmeter,
            radii_proximal=[2.0, 1.5] * u.um,
            radii_distal=[1.5, 1.0] * u.um,
            n_seg=2,
        )
        np.testing.assert_allclose(branch.lengths.to_decimal(u.um), [10.0, 20.0])
        self.assertEqual(branch.lengths.mantissa.dtype, np.float64)
        self.assertFalse(branch.lengths.mantissa.flags.writeable)
        self.assertEqual(branch.n_segments, 2)
        self.assertAlmostEqual(float(branch.length.to_decimal(u.um)), 30.0)

    def test_is_frozen(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            _branch(0).name = "other"  # type: ignore[misc]

    def test_rejects_mismatched_segment_arrays(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"BranchData\('b'\)\.radii_distal has 2 entries, but lengths has 1",
        ):
            _branch(0, radii_distal=[1.0, 1.0] * u.um)

    def test_rejects_non_length_units(self) -> None:
        with self.assertRaisesRegex(
            TypeError, r"BranchData\('b'\)\.lengths must have a length unit"
        ):
            _branch(0, lengths=[10.0] * u.mV)

    def test_rejects_bare_arrays(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a brainunit Quantity"):
            _branch(0, lengths=np.array([10.0]))

    def test_rejects_empty_branch(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one segment"):
            _branch(
                0,
                lengths=np.zeros(0) * u.um,
                radii_proximal=np.zeros(0) * u.um,
                radii_distal=np.zeros(0) * u.um,
            )

    def test_points_come_in_pairs_with_shape_n_by_3(self) -> None:
        self.assertTrue(_branch(0, points=True).has_points)
        with self.assertRaisesRegex(
            ValueError, "both points_proximal and points_distal"
        ):
            _branch(0, points_proximal=np.zeros((1, 3)) * u.um)
        with self.assertRaisesRegex(ValueError, r"must have shape \(1, 3\)"):
            _branch(
                0,
                points_proximal=np.zeros((1, 2)) * u.um,
                points_distal=np.zeros((1, 2)) * u.um,
            )


class MorphologyDataTest(unittest.TestCase):
    def test_children_are_derived_in_index_order(self) -> None:
        data = _tree(0, 0, 1)
        self.assertEqual(data.children, ((1, 2), (3,), (), ()))
        self.assertEqual(data.n_branches, 4)
        self.assertFalse(data.has_full_point_geometry)

    def test_explicit_children_keep_their_order(self) -> None:
        data = _tree(0, 0, children=((2, 1), (), ()))
        self.assertEqual(data.children, ((2, 1), (), ()))

    def test_explicit_children_must_describe_the_same_tree(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "children does not describe the same tree"
        ):
            _tree(0, 0, children=((1,), (2,), ()))

    def test_index_must_match_position(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Branch 'd2' has index 2, but it is at position 1"
        ):
            bvm.MorphologyData(branches=(_branch(0, "soma"), _branch(2, "d2", 0)))

    def test_parent_must_exist(self) -> None:
        with self.assertRaisesRegex(ValueError, "Branch 'd1' has parent_index 5"):
            bvm.MorphologyData(branches=(_branch(0, "soma"), _branch(1, "d1", 5)))

    def test_cycle_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "single tree"):
            bvm.MorphologyData(
                branches=(_branch(0, "soma"), _branch(1, "d1", 2), _branch(2, "d2", 1))
            )

    def test_only_the_root_may_lack_a_parent(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Branch 'd1' has no parent but is not the root"
        ):
            bvm.MorphologyData(branches=(_branch(0, "soma"), _branch(1, "d1")))
        with self.assertRaisesRegex(
            ValueError, "Root branch 'd1' must have parent_index=None"
        ):
            bvm.MorphologyData(
                branches=(_branch(0, "soma"), _branch(1, "d1", 0)), root_index=1
            )
        with self.assertRaisesRegex(ValueError, "out of range"):
            bvm.MorphologyData(branches=(_branch(0, "soma"),), root_index=3)

    def test_needs_branches_of_the_right_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one branch"):
            bvm.MorphologyData(branches=())
        with self.assertRaisesRegex(TypeError, r"branches\[0\] must be BranchData"):
            bvm.MorphologyData(branches=("soma",))  # type: ignore[arg-type]

    def test_structure_key_is_cached_and_tracks_child_order(self) -> None:
        a = _tree(0, 0)
        self.assertIs(a.structure_key, a.structure_key)
        self.assertEqual(a.structure_key, _tree(0, 0).structure_key)
        self.assertNotEqual(
            a.structure_key, _tree(0, 0, children=((2, 1), (), ())).structure_key
        )

    def test_full_point_geometry(self) -> None:
        data = bvm.MorphologyData(
            branches=(_branch(0, "soma", points=True), _branch(1, "d", 0, points=True))
        )
        self.assertTrue(data.has_full_point_geometry)


class ProtocolTest(unittest.TestCase):
    def test_as_morphology_data_accepts_the_data_and_the_protocol(self) -> None:
        data = _tree(0)

        class Source:
            def __morphology_data__(self):
                return data

        self.assertIs(bvm.as_morphology_data(data), data)
        self.assertIs(bvm.as_morphology_data(Source()), data)

    def test_as_morphology_data_rejects_other_objects(self) -> None:
        class Wrong:
            def __morphology_data__(self):
                return "not data"

        with self.assertRaisesRegex(TypeError, "__morphology_data__"):
            bvm.as_morphology_data(object())
        with self.assertRaisesRegex(
            TypeError,
            r"Wrong\.__morphology_data__\(\) must return MorphologyData, got str",
        ):
            bvm.as_morphology_data(Wrong())

    def test_as_topology_graph(self) -> None:
        graph = bvm.TopologyGraph(node_ids=(0, 1), edges=((0, 1),), root_id=0)

        class Source:
            def __topology_data__(self):
                return graph

        class Wrong:
            def __topology_data__(self):
                return 1

        self.assertIs(bvm.as_topology_graph(Source()), graph)
        with self.assertRaisesRegex(TypeError, "__topology_data__"):
            bvm.as_topology_graph(object())
        with self.assertRaisesRegex(TypeError, "must return TopologyGraph, got int"):
            bvm.as_topology_graph(Wrong())

    def test_plots_reject_objects_without_the_protocol(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            r"^plot2d\(\.\.\.\) expected MorphologyData or an object with __morphology_data__",
        ):
            bvm.plot2d(object())


class SelectionDataTest(unittest.TestCase):
    def test_region_and_locset_accept_masks_exposing_intervals_and_points(self) -> None:
        class Mask:
            intervals = [(1, 0, 0.5)]
            points = [np.array([0, 1.0])]

        self.assertEqual(_as_region_data(Mask()).intervals, ((1, 0.0, 0.5),))
        self.assertEqual(_as_locset_data(Mask()).points, ((0, 1.0),))
        self.assertIsNone(_as_region_data(None))

    def test_unevaluated_selections_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            TypeError, "region must be RegionData or expose .intervals"
        ):
            _as_region_data("soma")
        with self.assertRaisesRegex(
            TypeError, "locset must be LocsetData or expose .points"
        ):
            _as_locset_data(3)

    def test_fractions_are_range_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 <= prox <= dist <= 1"):
            bvm.RegionData(((0, 0.6, 0.4),))
        with self.assertRaisesRegex(ValueError, "0 <= x <= 1"):
            bvm.LocsetData(((0, 1.5),))


class TopologyGraphTest(unittest.TestCase):
    def test_frames(self) -> None:
        one = bvm.TopologyGraph(
            node_ids=(0, 1), edges=((0, 1),), root_id=0, values=np.array([1.0, 2.0])
        )
        self.assertEqual(one.n_frames, 1)
        np.testing.assert_array_equal(one.frame(), [1.0, 2.0])
        two = bvm.TopologyGraph(
            node_ids=(0, 1),
            edges=((0, 1),),
            root_id=0,
            values=np.array([[1.0, 2.0], [3.0, 4.0]]) * u.mV,
            times=np.array([0.0, 0.1]) * u.ms,
        )
        self.assertEqual(two.n_frames, 2)
        np.testing.assert_array_equal(two.frame(1).to_decimal(u.mV), [3.0, 4.0])
        with self.assertRaisesRegex(ValueError, "pass frame=<index>"):
            two.frame()
        self.assertEqual(
            bvm.TopologyGraph(node_ids=(0,), edges=(), root_id=0).n_frames, 0
        )

    def test_rejects_inconsistent_graphs(self) -> None:
        cases = {
            "unique": dict(node_ids=(0, 0), edges=(), root_id=0),
            "names a node": dict(node_ids=(0, 1), edges=((0, 2),), root_id=0),
            "root_id 5": dict(node_ids=(0, 1), edges=(), root_id=5),
            r"shape \(2,\) or \(T, 2\)": dict(
                node_ids=(0, 1), edges=(), root_id=0, values=np.ones(3)
            ),
            "not both": dict(
                node_ids=(0, 1),
                edges=(),
                root_id=0,
                values=np.ones(2),
                highlight_fractions={0: 1.0},
            ),
            "unknown node 4": dict(
                node_ids=(0, 1), edges=(), root_id=0, highlight_fractions={4: 1.0}
            ),
            r"must be in \[0, 1\]": dict(
                node_ids=(0, 1), edges=(), root_id=0, highlight_fractions={0: 2.0}
            ),
            "one entry per frame": dict(
                node_ids=(0, 1),
                edges=(),
                root_id=0,
                values=np.ones((2, 2)),
                times=np.ones(3),
            ),
            "times needs": dict(node_ids=(0, 1), edges=(), root_id=0, times=np.ones(2)),
            "at least one node": dict(node_ids=(), edges=(), root_id=0),
        }
        for message, kwargs in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    bvm.TopologyGraph(**kwargs)


if __name__ == "__main__":
    unittest.main()
