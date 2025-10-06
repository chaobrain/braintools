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
import unittest

import brainunit as u
import numpy as np
import pytest

from braintools.conn import (
    SmallWorld,
    ScaleFree,
    Regular,
    ModularRandom,
    ModularGeneral,
    HierarchicalRandom,
    CorePeripheryRandom,
    Random,
)
from braintools.init import Normal


class TestTopologicalPatterns(unittest.TestCase):
    """
    Test topological connectivity patterns (SmallWorld, ScaleFree, Regular, Modular).

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import SmallWorld, ScaleFree, Regular, Modular

        # Small-world network
        sw = SmallWorld(k=6, p=0.3, weight=1.0 * u.nS)
        result_sw = sw(pre_size=100, post_size=100)
        assert result_sw.metadata['pattern'] == 'small_world'

        # Scale-free network
        sf = ScaleFree(m=3, weight=0.8 * u.nS)
        result_sf = sf(pre_size=200, post_size=200)
        assert result_sf.metadata['pattern'] == 'scale_free'

        # Regular network
        reg = Regular(degree=8, weight=1.2 * u.nS)
        result_reg = reg(pre_size=150, post_size=150)
        assert result_reg.metadata['pattern'] == 'regular'

        # Modular network
        mod = Modular(n_modules=5, intra_prob=0.3, inter_prob=0.01)
        result_mod = mod(pre_size=100, post_size=100)
        assert result_mod.metadata['pattern'] == 'modular'
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_small_world_basic(self):
        conn = SmallWorld(k=4, p=0.2, seed=42)
        result = conn(pre_size=20, post_size=20)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'small_world')
        self.assertEqual(result.metadata['k'], 4)
        self.assertEqual(result.metadata['p'], 0.2)

        # Each neuron connects to k neighbors, total = n * k
        self.assertEqual(result.n_connections, 20 * 4)

    def test_small_world_different_sizes_error(self):
        conn = SmallWorld(k=2, seed=42)

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=8)  # Different sizes not allowed

    def test_small_world_rewiring(self):
        # Test that rewiring works (p=1.0 means all edges are rewired)
        conn = SmallWorld(k=2, p=1.0, seed=42)
        result = conn(pre_size=10, post_size=10)

        # Should still have same number of connections
        self.assertEqual(result.n_connections, 10 * 2)

        # But topology should be different from regular ring
        # (Hard to test directly, but at least check no errors)

    def test_scale_free_basic(self):
        conn = ScaleFree(m=2, seed=42)
        result = conn(pre_size=15, post_size=15)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'scale_free')
        self.assertEqual(result.metadata['m'], 2)

        # Should have connections (exact number depends on algorithm)
        self.assertGreater(result.n_connections, 0)

    def test_scale_free_different_sizes_error(self):
        conn = ScaleFree(m=2, seed=42)

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=8)  # Different sizes not allowed

    def test_regular_basic(self):
        conn = Regular(degree=5, seed=42)
        result = conn(pre_size=12, post_size=12)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'regular')
        self.assertEqual(result.metadata['degree'], 5)

        # Each neuron has exactly 'degree' connections
        self.assertEqual(result.n_connections, 12 * 5)

    def test_regular_different_sizes_error(self):
        conn = Regular(degree=3, seed=42)

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=8)  # Different sizes not allowed

    def test_modular_basic(self):
        conn = ModularRandom(
            n_modules=3,
            intra_prob=0.4,
            inter_prob=0.05,
            seed=42
        )
        result = conn(pre_size=12, post_size=12)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'modular')
        self.assertEqual(result.metadata['n_modules'], 3)
        self.assertEqual(result.metadata['intra_prob'], 0.4)
        self.assertEqual(result.metadata['inter_prob'], 0.05)

        # Should have more intra-module than inter-module connections
        self.assertGreater(result.n_connections, 0)

    def test_modular_different_sizes_error(self):
        conn = ModularRandom(n_modules=2, seed=42)

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=8)  # Different sizes not allowed

    def test_modular_uneven_module_assignment(self):
        # Test when population size doesn't divide evenly by number of modules
        conn = ModularRandom(
            n_modules=3,
            intra_prob=0.3,
            inter_prob=0.01,
            seed=42
        )
        result = conn(pre_size=10, post_size=10)  # 10 doesn't divide evenly by 3

        # Should still work (extra neurons assigned to last module)
        self.assertGreater(result.n_connections, 0)


class TestModularGeneral(unittest.TestCase):
    """Comprehensive tests for ModularGeneral connectivity."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_modular_with_uniform_inter_conn(self):
        """Test basic modular network with same connectivity for all modules."""
        intra = [
            Random(prob=0.3, seed=42),
            Random(prob=0.3, seed=43),
            Random(prob=0.3, seed=44)
        ]
        inter = Random(prob=0.05, seed=45)

        conn = ModularGeneral(intra_conn=intra, inter_conn=inter)
        result = conn(pre_size=90, post_size=90)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'modular_general')
        self.assertEqual(result.metadata['n_modules'], 3)
        self.assertEqual(len(result.metadata['module_sizes']), 3)
        self.assertEqual(sum(result.metadata['module_sizes']), 90)
        self.assertIsNotNone(result.metadata['inter_conn'])
        self.assertEqual(len(result.metadata['intra_conn']), 3)

        # Should have connections
        self.assertGreater(result.n_connections, 0)

        # Verify indices are within bounds
        self.assertTrue(np.all(result.pre_indices >= 0))
        self.assertTrue(np.all(result.pre_indices < 90))
        self.assertTrue(np.all(result.post_indices >= 0))
        self.assertTrue(np.all(result.post_indices < 90))

    def test_no_inter_conn_specified(self):
        """Test when no inter-module connectivity is specified."""
        intra = [
            Random(prob=0.4, seed=42),
            Random(prob=0.4, seed=43)
        ]

        conn = ModularGeneral(intra_conn=intra, inter_conn=None)
        result = conn(pre_size=100, post_size=100)

        # Should only have intra-module connections
        self.assertGreater(result.n_connections, 0)
        self.assertEqual(result.metadata['n_modules'], 2)
        self.assertIsNone(result.metadata['inter_conn'])

    def test_inter_conn_pair_overrides(self):
        """Test that inter_conn_pair overrides default inter_conn."""
        intra = [
            Random(prob=0.3, seed=42),
            Random(prob=0.3, seed=43),
            Random(prob=0.3, seed=44)
        ]
        default_inter = Random(prob=0.01, seed=45)
        specific_inter = {
            (0, 1): Random(prob=0.1, seed=46),  # Stronger connection
            (1, 2): Random(prob=0.15, seed=47),
        }

        conn = ModularGeneral(
            intra_conn=intra,
            inter_conn=default_inter,
            inter_conn_pair=specific_inter
        )
        result = conn(pre_size=90, post_size=90)

        self.assertEqual(result.metadata['n_modules'], 3)
        self.assertEqual(len(result.metadata['inter_conn_pair']), 2)
        self.assertIn((0, 1), result.metadata['inter_conn_pair'])
        self.assertIn((1, 2), result.metadata['inter_conn_pair'])
        self.assertGreater(result.n_connections, 0)

    def test_only_inter_conn_pair_no_default(self):
        """Test using only inter_conn_pair without default inter_conn."""
        intra = [
            Random(prob=0.4, seed=42),
            Random(prob=0.4, seed=43),
            Random(prob=0.4, seed=44)
        ]
        specific_inter = {
            (0, 1): Random(prob=0.1, seed=45),
            (2, 0): Random(prob=0.08, seed=46),
        }

        conn = ModularGeneral(
            intra_conn=intra,
            inter_conn=None,
            inter_conn_pair=specific_inter
        )
        result = conn(pre_size=120, post_size=120)

        # Should have intra-module connections and only specified inter-module connections
        self.assertGreater(result.n_connections, 0)
        self.assertIsNone(result.metadata['inter_conn'])
        self.assertEqual(len(result.metadata['inter_conn_pair']), 2)

    def test_module_ratios_fixed_sizes(self):
        """Test module_ratios with fixed integer sizes."""
        intra = [
            Random(prob=0.3, seed=42),
            Random(prob=0.3, seed=43),
            Random(prob=0.3, seed=44)
        ]
        inter = Random(prob=0.05, seed=45)

        conn = ModularGeneral(
            intra_conn=intra,
            inter_conn=inter,
            module_ratios=[20, 30]  # Last module gets remaining 50
        )
        result = conn(pre_size=100, post_size=100)

        self.assertEqual(result.metadata['module_sizes'], [20, 30, 50])
        self.assertEqual(sum(result.metadata['module_sizes']), 100)
        self.assertGreater(result.n_connections, 0)

    def test_module_ratios_proportional(self):
        """Test module_ratios with proportional float values."""
        intra = [
            Random(prob=0.3, seed=42),
            Random(prob=0.3, seed=43),
            Random(prob=0.3, seed=44)
        ]
        inter = Random(prob=0.05, seed=45)

        conn = ModularGeneral(
            intra_conn=intra,
            inter_conn=inter,
            module_ratios=[0.2, 0.3]  # 20%, 30%, remaining 50%
        )
        result = conn(pre_size=100, post_size=100)

        # 0.2*100=20, 0.3*100=30, remaining=50
        self.assertEqual(result.metadata['module_sizes'], [20, 30, 50])
        self.assertGreater(result.n_connections, 0)

    def test_module_ratios_mixed(self):
        """Test module_ratios with mixed int and float values."""
        intra = [
            Random(prob=0.3, seed=42),
            Random(prob=0.3, seed=43),
            Random(prob=0.3, seed=44)
        ]
        inter = Random(prob=0.05, seed=45)

        conn = ModularGeneral(
            intra_conn=intra,
            inter_conn=inter,
            module_ratios=[25, 0.25]  # 25 fixed, 25% of 100 = 25, remaining = 50
        )
        result = conn(pre_size=100, post_size=100)

        self.assertEqual(result.metadata['module_sizes'], [25, 25, 50])
        self.assertGreater(result.n_connections, 0)

    def test_module_ratios_wrong_length_error(self):
        """Test that wrong length module_ratios raises error."""
        intra = [
            Random(prob=0.3, seed=42),
            Random(prob=0.3, seed=43),
            Random(prob=0.3, seed=44)
        ]

        with self.assertRaises(ValueError) as ctx:
            ModularGeneral(
                intra_conn=intra,
                inter_conn=None,
                module_ratios=[20, 30, 50]  # Should be n_modules-1 = 2
            )
        self.assertIn("n_modules-1", str(ctx.exception))

    def test_module_ratios_exceeds_total_error(self):
        """Test that module sizes exceeding total raises error."""
        intra = [
            Random(prob=0.3, seed=42),
            Random(prob=0.3, seed=43)
        ]

        conn = ModularGeneral(
            intra_conn=intra,
            inter_conn=None,
            module_ratios=[80]  # Leaves only 20 for second module
        )

        # Should work
        result = conn(pre_size=100, post_size=100)
        self.assertEqual(result.metadata['module_sizes'], [80, 20])

        # But this should fail
        conn2 = ModularGeneral(
            intra_conn=intra,
            inter_conn=None,
            module_ratios=[120]  # Exceeds total
        )

        with self.assertRaises(ValueError) as ctx:
            conn2(pre_size=100, post_size=100)
        self.assertIn("exceeds remaining", str(ctx.exception))

    def test_uneven_module_sizes_default(self):
        """Test default module size assignment when size doesn't divide evenly."""
        intra = [
            Random(prob=0.3, seed=42),
            Random(prob=0.3, seed=43),
            Random(prob=0.3, seed=44)
        ]
        inter = Random(prob=0.05, seed=45)

        conn = ModularGeneral(intra_conn=intra, inter_conn=inter)
        result = conn(pre_size=100, post_size=100)

        # 100 / 3 = 33 with remainder 1
        # Should get [34, 33, 33] or similar distribution
        sizes = result.metadata['module_sizes']
        self.assertEqual(sum(sizes), 100)
        self.assertEqual(len(sizes), 3)
        # Check they're approximately equal
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_different_connectivity_per_module(self):
        """Test that different modules can have different connectivity patterns."""
        intra = [
            Random(prob=0.5, seed=42),  # Dense
            Random(prob=0.1, seed=43),  # Sparse
            SmallWorld(k=4, p=0.3, seed=44)  # Small-world
        ]
        inter = Random(prob=0.02, seed=45)

        conn = ModularGeneral(intra_conn=intra, inter_conn=inter)
        result = conn(pre_size=90, post_size=90)

        self.assertEqual(result.metadata['n_modules'], 3)
        self.assertEqual(result.metadata['intra_conn'][0], 'Random')
        self.assertEqual(result.metadata['intra_conn'][1], 'Random')
        self.assertEqual(result.metadata['intra_conn'][2], 'SmallWorld')
        self.assertGreater(result.n_connections, 0)

    def test_intra_conn_not_sequence_error(self):
        """Test that non-sequence intra_conn raises TypeError."""
        with self.assertRaises(TypeError) as ctx:
            ModularGeneral(
                intra_conn=Random(prob=0.3, seed=42),  # Not a sequence
                inter_conn=None
            )
        self.assertIn("list/tuple", str(ctx.exception))

    def test_inter_conn_pair_not_dict_error(self):
        """Test that non-dict inter_conn_pair raises TypeError."""
        intra = [Random(prob=0.3, seed=42), Random(prob=0.3, seed=43)]

        with self.assertRaises(TypeError) as ctx:
            ModularGeneral(
                intra_conn=intra,
                inter_conn=None,
                inter_conn_pair=[(0, 1)]  # Not a dict
            )
        self.assertIn("dict", str(ctx.exception))

    def test_different_sizes_error(self):
        """Test that different pre_size and post_size raises error."""
        intra = [Random(prob=0.3, seed=42), Random(prob=0.3, seed=43)]
        conn = ModularGeneral(intra_conn=intra, inter_conn=None)

        with self.assertRaises(ValueError) as ctx:
            conn(pre_size=100, post_size=80)
        self.assertIn("require pre_size == post_size", str(ctx.exception))

    def test_empty_modules(self):
        """Test behavior with very small population and many modules."""
        intra = [
            Random(prob=0.3, seed=42),
            Random(prob=0.3, seed=43),
            Random(prob=0.3, seed=44)
        ]

        conn = ModularGeneral(
            intra_conn=intra,
            inter_conn=None,
            module_ratios=[2, 1]  # Leaves 0 for last module
        )

        result = conn(pre_size=3, post_size=3)

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        intra = [
            Random(prob=0.3, seed=42),
            Random(prob=0.3, seed=43)
        ]
        inter = Random(prob=0.05, seed=44)

        conn1 = ModularGeneral(intra_conn=intra, inter_conn=inter, seed=100)
        result1 = conn1(pre_size=100, post_size=100)

        # Recreate with same seeds
        intra2 = [
            Random(prob=0.3, seed=42),
            Random(prob=0.3, seed=43)
        ]
        inter2 = Random(prob=0.05, seed=44)

        conn2 = ModularGeneral(intra_conn=intra2, inter_conn=inter2, seed=100)
        result2 = conn2(pre_size=100, post_size=100)

        # Should produce identical results
        self.assertEqual(result1.n_connections, result2.n_connections)
        np.testing.assert_array_equal(result1.pre_indices, result2.pre_indices)
        np.testing.assert_array_equal(result1.post_indices, result2.post_indices)

    def test_metadata_completeness(self):
        """Test that all expected metadata is present."""
        intra = [Random(prob=0.3, seed=42), Random(prob=0.3, seed=43)]
        inter = Random(prob=0.05, seed=44)
        inter_pair = {(0, 1): Random(prob=0.1, seed=45)}

        conn = ModularGeneral(
            intra_conn=intra,
            inter_conn=inter,
            inter_conn_pair=inter_pair,
            module_ratios=[40]
        )
        result = conn(pre_size=100, post_size=100)

        metadata = result.metadata
        self.assertIn('pattern', metadata)
        self.assertIn('n_modules', metadata)
        self.assertIn('module_sizes', metadata)
        self.assertIn('module_ratios', metadata)
        self.assertIn('inter_conn', metadata)
        self.assertIn('inter_conn_pair', metadata)
        self.assertIn('intra_conn', metadata)

        self.assertEqual(metadata['pattern'], 'modular_general')
        self.assertEqual(metadata['n_modules'], 2)
        self.assertEqual(metadata['module_sizes'], [40, 60])
        self.assertEqual(metadata['module_ratios'], [40])
        self.assertEqual(metadata['inter_conn'], 'Random')
        self.assertEqual(len(metadata['inter_conn_pair']), 1)

    def test_large_network(self):
        """Test with a larger network to ensure scalability."""
        intra = [
            Random(prob=0.1, seed=42),
            Random(prob=0.1, seed=43),
            Random(prob=0.1, seed=44),
            Random(prob=0.1, seed=45),
            Random(prob=0.1, seed=46)
        ]
        inter = Random(prob=0.005, seed=47)

        conn = ModularGeneral(intra_conn=intra, inter_conn=inter)
        result = conn(pre_size=5000, post_size=5000)

        self.assertEqual(result.metadata['n_modules'], 5)
        self.assertEqual(sum(result.metadata['module_sizes']), 5000)
        self.assertGreater(result.n_connections, 0)

        # Verify no invalid indices
        self.assertTrue(np.all(result.pre_indices >= 0))
        self.assertTrue(np.all(result.pre_indices < 5000))
        self.assertTrue(np.all(result.post_indices >= 0))
        self.assertTrue(np.all(result.post_indices < 5000))


class TestHierarchicalRandom(unittest.TestCase):
    """Comprehensive tests for HierarchicalRandom connectivity."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_hierarchical(self):
        """Test basic hierarchical network with default parameters."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.3,
            feedback_prob=0.1,
            recurrent_prob=0.2,
            seed=42
        )
        result = conn(pre_size=90, post_size=90)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'hierarchical')
        self.assertEqual(result.metadata['n_levels'], 3)
        self.assertEqual(result.metadata['feedforward_prob'], 0.3)
        self.assertEqual(result.metadata['feedback_prob'], 0.1)
        self.assertEqual(result.metadata['recurrent_prob'], 0.2)
        self.assertEqual(result.metadata['skip_prob'], 0.0)
        self.assertEqual(len(result.metadata['level_sizes']), 3)
        self.assertEqual(sum(result.metadata['level_sizes']), 90)

        # Should have connections
        self.assertGreater(result.n_connections, 0)

        # Verify indices are within bounds
        self.assertTrue(np.all(result.pre_indices >= 0))
        self.assertTrue(np.all(result.pre_indices < 90))
        self.assertTrue(np.all(result.post_indices >= 0))
        self.assertTrue(np.all(result.post_indices < 90))

    def test_two_level_hierarchy(self):
        """Test minimal 2-level hierarchy."""
        conn = HierarchicalRandom(
            n_levels=2,
            feedforward_prob=0.4,
            feedback_prob=0.2,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        self.assertEqual(result.metadata['n_levels'], 2)
        self.assertEqual(len(result.metadata['level_sizes']), 2)
        self.assertEqual(sum(result.metadata['level_sizes']), 100)
        self.assertGreater(result.n_connections, 0)

    def test_n_levels_too_small_error(self):
        """Test that n_levels < 2 raises error."""
        with self.assertRaises(ValueError) as ctx:
            HierarchicalRandom(n_levels=1)
        self.assertIn("at least 2", str(ctx.exception))

    def test_different_sizes_error(self):
        """Test that different pre_size and post_size raises error."""
        conn = HierarchicalRandom(n_levels=3, seed=42)

        with self.assertRaises(ValueError) as ctx:
            conn(pre_size=100, post_size=80)
        self.assertIn("require pre_size == post_size", str(ctx.exception))

    def test_skip_connections(self):
        """Test hierarchical network with skip connections."""
        conn = HierarchicalRandom(
            n_levels=4,
            feedforward_prob=0.3,
            feedback_prob=0.1,
            recurrent_prob=0.2,
            skip_prob=0.05,
            seed=42
        )
        result = conn(pre_size=120, post_size=120)

        self.assertEqual(result.metadata['skip_prob'], 0.05)
        self.assertEqual(result.metadata['n_levels'], 4)
        self.assertGreater(result.n_connections, 0)

    def test_level_ratios_fixed_sizes(self):
        """Test level_ratios with fixed integer sizes."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.3,
            feedback_prob=0.1,
            level_ratios=[30, 40],  # Last level gets remaining 30
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        self.assertEqual(result.metadata['level_sizes'], [30, 40, 30])
        self.assertEqual(sum(result.metadata['level_sizes']), 100)
        self.assertGreater(result.n_connections, 0)

    def test_level_ratios_proportional(self):
        """Test level_ratios with proportional float values."""
        conn = HierarchicalRandom(
            n_levels=4,
            feedforward_prob=0.3,
            feedback_prob=0.1,
            level_ratios=[0.4, 0.3, 0.2],  # 40%, 30%, 20%, remaining 10%
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        # 0.4*100=40, 0.3*100=30, 0.2*100=20, remaining=10
        self.assertEqual(result.metadata['level_sizes'], [40, 30, 20, 10])
        self.assertGreater(result.n_connections, 0)

    def test_level_ratios_mixed(self):
        """Test level_ratios with mixed int and float values."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.3,
            feedback_prob=0.1,
            level_ratios=[25, 0.5],  # 25 fixed, 50% of 100 = 50, remaining = 25
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        self.assertEqual(result.metadata['level_sizes'], [25, 50, 25])
        self.assertGreater(result.n_connections, 0)

    def test_level_ratios_wrong_length_error(self):
        """Test that wrong length level_ratios raises error."""
        with self.assertRaises(ValueError) as ctx:
            HierarchicalRandom(
                n_levels=3,
                feedforward_prob=0.3,
                level_ratios=[20, 30, 50]  # Should be n_levels-1 = 2
            )
        self.assertIn("n_levels-1", str(ctx.exception))

    def test_level_ratios_exceeds_total_error(self):
        """Test that level sizes exceeding total raises error."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.3,
            level_ratios=[80, 30]  # Exceeds 100
        )

        with self.assertRaises(ValueError) as ctx:
            conn(pre_size=100, post_size=100)
        self.assertIn("exceeds remaining", str(ctx.exception))

    def test_level_ratios_negative_error(self):
        """Test that negative level sizes raise error."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.3,
            level_ratios=[-10, 60]
        )

        with self.assertRaises(ValueError) as ctx:
            conn(pre_size=100, post_size=100)
        self.assertIn("cannot be negative", str(ctx.exception))

    def test_uneven_level_sizes_default(self):
        """Test default level size assignment when size doesn't divide evenly."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.3,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        # 100 / 3 = 33 with remainder 1
        # Should get [34, 33, 33] or similar distribution
        sizes = result.metadata['level_sizes']
        self.assertEqual(sum(sizes), 100)
        self.assertEqual(len(sizes), 3)
        # Check they're approximately equal
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_feedforward_only(self):
        """Test hierarchy with only feedforward connections."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.4,
            feedback_prob=0.0,
            recurrent_prob=0.0,
            skip_prob=0.0,
            seed=42
        )
        result = conn(pre_size=90, post_size=90)

        self.assertGreater(result.n_connections, 0)

        # Verify connections are only feedforward
        level_sizes = result.metadata['level_sizes']
        level_boundaries = [0]
        for size in level_sizes:
            level_boundaries.append(level_boundaries[-1] + size)

        for pre_idx, post_idx in zip(result.pre_indices, result.post_indices):
            # Find which levels pre and post belong to
            pre_level = next(i for i in range(3) if level_boundaries[i] <= pre_idx < level_boundaries[i + 1])
            post_level = next(i for i in range(3) if level_boundaries[i] <= post_idx < level_boundaries[i + 1])

            # Should only be feedforward (pre_level < post_level)
            self.assertLess(pre_level, post_level, "Should only have feedforward connections")

    def test_recurrent_only(self):
        """Test hierarchy with only recurrent connections."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.0,
            feedback_prob=0.0,
            recurrent_prob=0.3,
            skip_prob=0.0,
            seed=42
        )
        result = conn(pre_size=90, post_size=90)

        self.assertGreater(result.n_connections, 0)

        # Verify connections are only within same level
        level_sizes = result.metadata['level_sizes']
        level_boundaries = [0]
        for size in level_sizes:
            level_boundaries.append(level_boundaries[-1] + size)

        for pre_idx, post_idx in zip(result.pre_indices, result.post_indices):
            # Find which levels pre and post belong to
            pre_level = next(i for i in range(3) if level_boundaries[i] <= pre_idx < level_boundaries[i + 1])
            post_level = next(i for i in range(3) if level_boundaries[i] <= post_idx < level_boundaries[i + 1])

            # Should only be within same level
            self.assertEqual(pre_level, post_level, "Should only have recurrent connections")

    def test_no_self_connections(self):
        """Test that self-connections are excluded."""
        conn = HierarchicalRandom(
            n_levels=2,
            feedforward_prob=0.3,
            feedback_prob=0.3,
            recurrent_prob=0.5,
            seed=42
        )
        result = conn(pre_size=50, post_size=50)

        # Check no self-connections
        self.assertTrue(np.all(result.pre_indices != result.post_indices))

    def test_deep_hierarchy(self):
        """Test deep hierarchy with many levels."""
        conn = HierarchicalRandom(
            n_levels=10,
            feedforward_prob=0.2,
            feedback_prob=0.05,
            recurrent_prob=0.15,
            seed=42
        )
        result = conn(pre_size=1000, post_size=1000)

        self.assertEqual(result.metadata['n_levels'], 10)
        self.assertEqual(len(result.metadata['level_sizes']), 10)
        self.assertEqual(sum(result.metadata['level_sizes']), 1000)
        self.assertGreater(result.n_connections, 0)

    def test_weights_and_delays(self):
        """Test hierarchical network with weights and delays."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.3,
            feedback_prob=0.1,
            recurrent_prob=0.2,
            weight=1.5 * u.nS,
            delay=2.0 * u.ms,
            seed=42
        )
        result = conn(pre_size=90, post_size=90)

        self.assertIsNotNone(result.weights)
        self.assertIsNotNone(result.delays)
        self.assertTrue(u.math.isscalar(result.weights))
        self.assertTrue(u.math.isscalar(result.delays))

        # Check weight values
        self.assertTrue(u.math.allclose(result.weights, 1.5 * u.nS))
        self.assertTrue(u.math.allclose(result.delays, 2.0 * u.ms))

    def test_custom_weight_initializer(self):
        """Test with custom weight initializer."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.3,
            feedback_prob=0.1,
            weight=Normal(mean=1.0 * u.nS, std=0.2 * u.nS),
            seed=42
        )
        result = conn(pre_size=90, post_size=90)

        self.assertIsNotNone(result.weights)
        self.assertEqual(len(result.weights), result.n_connections)
        # Check that weights vary (not all the same)
        self.assertGreater(u.math.std(result.weights).mantissa, 0)

    def test_zero_probabilities(self):
        """Test with all probabilities set to zero."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.0,
            feedback_prob=0.0,
            recurrent_prob=0.0,
            skip_prob=0.0,
            seed=42
        )
        result = conn(pre_size=90, post_size=90)

        # Should have no connections
        self.assertEqual(result.n_connections, 0)
        self.assertEqual(len(result.pre_indices), 0)
        self.assertEqual(len(result.post_indices), 0)

    def test_high_probabilities(self):
        """Test with high connection probabilities."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.9,
            feedback_prob=0.9,
            recurrent_prob=0.9,
            seed=42
        )
        result = conn(pre_size=90, post_size=90)

        # Should have many connections
        self.assertGreater(result.n_connections, 1000)

    def test_bottom_heavy_hierarchy(self):
        """Test bottom-heavy hierarchy (sensory-like)."""
        conn = HierarchicalRandom(
            n_levels=4,
            feedforward_prob=0.3,
            feedback_prob=0.1,
            level_ratios=[0.5, 0.3, 0.15],  # 50%, 30%, 15%, 5%
            seed=42
        )
        result = conn(pre_size=1000, post_size=1000)

        sizes = result.metadata['level_sizes']
        self.assertEqual(sizes, [500, 300, 150, 50])
        # Verify decreasing sizes (bottom-heavy)
        for i in range(len(sizes) - 1):
            self.assertGreater(sizes[i], sizes[i + 1])

    def test_top_heavy_hierarchy(self):
        """Test top-heavy hierarchy (motor-like)."""
        conn = HierarchicalRandom(
            n_levels=4,
            feedforward_prob=0.3,
            feedback_prob=0.1,
            level_ratios=[0.1, 0.2, 0.3],  # 10%, 20%, 30%, 40%
            seed=42
        )
        result = conn(pre_size=1000, post_size=1000)

        sizes = result.metadata['level_sizes']
        self.assertEqual(sizes, [100, 200, 300, 400])
        # Verify increasing sizes (top-heavy)
        for i in range(len(sizes) - 1):
            self.assertLess(sizes[i], sizes[i + 1])

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        conn1 = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.3,
            feedback_prob=0.1,
            recurrent_prob=0.2,
            seed=100
        )
        result1 = conn1(pre_size=100, post_size=100)

        conn2 = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.3,
            feedback_prob=0.1,
            recurrent_prob=0.2,
            seed=100
        )
        result2 = conn2(pre_size=100, post_size=100)

        # Should produce identical results
        self.assertEqual(result1.n_connections, result2.n_connections)
        np.testing.assert_array_equal(result1.pre_indices, result2.pre_indices)
        np.testing.assert_array_equal(result1.post_indices, result2.post_indices)

    def test_metadata_completeness(self):
        """Test that all expected metadata is present."""
        conn = HierarchicalRandom(
            n_levels=4,
            feedforward_prob=0.35,
            feedback_prob=0.12,
            recurrent_prob=0.25,
            skip_prob=0.03,
            level_ratios=[30, 40, 20],
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        metadata = result.metadata
        self.assertIn('pattern', metadata)
        self.assertIn('n_levels', metadata)
        self.assertIn('level_sizes', metadata)
        self.assertIn('level_ratios', metadata)
        self.assertIn('feedforward_prob', metadata)
        self.assertIn('feedback_prob', metadata)
        self.assertIn('recurrent_prob', metadata)
        self.assertIn('skip_prob', metadata)

        self.assertEqual(metadata['pattern'], 'hierarchical')
        self.assertEqual(metadata['n_levels'], 4)
        self.assertEqual(metadata['level_sizes'], [30, 40, 20, 10])
        self.assertEqual(metadata['level_ratios'], [30, 40, 20])
        self.assertEqual(metadata['feedforward_prob'], 0.35)
        self.assertEqual(metadata['feedback_prob'], 0.12)
        self.assertEqual(metadata['recurrent_prob'], 0.25)
        self.assertEqual(metadata['skip_prob'], 0.03)

    @pytest.mark.skip(reason="too slow for regular test runs")
    def test_large_network(self):
        """Test with a larger network to ensure scalability."""
        conn = HierarchicalRandom(
            n_levels=5,
            feedforward_prob=0.1,
            feedback_prob=0.05,
            recurrent_prob=0.1,
            seed=42
        )
        result = conn(pre_size=5000, post_size=5000)

        self.assertEqual(result.metadata['n_levels'], 5)
        self.assertEqual(sum(result.metadata['level_sizes']), 5000)
        self.assertGreater(result.n_connections, 0)

        # Verify no invalid indices
        self.assertTrue(np.all(result.pre_indices >= 0))
        self.assertTrue(np.all(result.pre_indices < 5000))
        self.assertTrue(np.all(result.post_indices >= 0))
        self.assertTrue(np.all(result.post_indices < 5000))

    def test_empty_levels(self):
        """Test behavior with very small population and level ratios."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.3,
            level_ratios=[2, 1],  # Leaves 0 for last level
            seed=42
        )
        result = conn(pre_size=3, post_size=3)

        self.assertEqual(result.metadata['level_sizes'], [2, 1, 0])
        # May or may not have connections depending on random draws

    def test_tuple_size_input(self):
        """Test that tuple pre_size/post_size is handled correctly."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.3,
            feedback_prob=0.1,
            seed=42
        )
        result = conn(pre_size=(10, 10), post_size=(10, 10))

        # Should treat as 100 neurons
        self.assertEqual(sum(result.metadata['level_sizes']), 100)
        self.assertGreater(result.n_connections, 0)

    def test_asymmetric_feedforward_feedback(self):
        """Test that feedforward probability is typically higher than feedback."""
        conn = HierarchicalRandom(
            n_levels=3,
            feedforward_prob=0.5,
            feedback_prob=0.1,
            recurrent_prob=0.0,
            seed=42
        )
        result = conn(pre_size=300, post_size=300)

        # Count feedforward vs feedback connections
        level_sizes = result.metadata['level_sizes']
        level_boundaries = [0]
        for size in level_sizes:
            level_boundaries.append(level_boundaries[-1] + size)

        feedforward_count = 0
        feedback_count = 0

        for pre_idx, post_idx in zip(result.pre_indices, result.post_indices):
            pre_level = next(i for i in range(3) if level_boundaries[i] <= pre_idx < level_boundaries[i + 1])
            post_level = next(i for i in range(3) if level_boundaries[i] <= post_idx < level_boundaries[i + 1])

            if post_level == pre_level + 1:
                feedforward_count += 1
            elif post_level == pre_level - 1:
                feedback_count += 1

        # Should have more feedforward than feedback connections
        self.assertGreater(feedforward_count, feedback_count)


class TestCorePeripheryRandom(unittest.TestCase):
    """Comprehensive tests for CorePeripheryRandom connectivity."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_core_periphery(self):
        """Test basic core-periphery network with default parameters."""
        conn = CorePeripheryRandom(
            core_size=20,
            core_core_prob=0.5,
            core_periphery_prob=0.2,
            periphery_core_prob=0.2,
            periphery_periphery_prob=0.05,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'core_periphery')
        self.assertEqual(result.metadata['core_size'], 20)
        self.assertEqual(result.metadata['core_core_prob'], 0.5)
        self.assertEqual(result.metadata['core_periphery_prob'], 0.2)
        self.assertEqual(result.metadata['periphery_core_prob'], 0.2)
        self.assertEqual(result.metadata['periphery_periphery_prob'], 0.05)

        # Should have connections
        self.assertGreater(result.n_connections, 0)

        # Verify indices are within bounds
        self.assertTrue(np.all(result.pre_indices >= 0))
        self.assertTrue(np.all(result.pre_indices < 100))
        self.assertTrue(np.all(result.post_indices >= 0))
        self.assertTrue(np.all(result.post_indices < 100))

    def test_core_size_as_float(self):
        """Test core_size specified as a proportion."""
        conn = CorePeripheryRandom(
            core_size=0.2,  # 20% of network
            core_core_prob=0.5,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        # 0.2 * 100 = 20 neurons in core
        self.assertEqual(result.metadata['core_size'], 20)
        self.assertGreater(result.n_connections, 0)

    def test_core_size_as_int(self):
        """Test core_size specified as absolute number."""
        conn = CorePeripheryRandom(
            core_size=30,  # Exactly 30 neurons
            core_core_prob=0.5,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        self.assertEqual(result.metadata['core_size'], 30)
        self.assertGreater(result.n_connections, 0)

    def test_core_size_float_invalid_error(self):
        """Test that core_size as float outside (0,1) raises error."""
        conn = CorePeripheryRandom(
            core_size=1.5,  # Invalid: > 1
            seed=42
        )

        with self.assertRaises(ValueError) as ctx:
            conn(pre_size=100, post_size=100)
        self.assertIn("(0, 1)", str(ctx.exception))

    def test_core_size_exceeds_network_error(self):
        """Test that core_size >= network size raises error."""
        conn = CorePeripheryRandom(
            core_size=100,  # Equal to network size
            seed=42
        )

        with self.assertRaises(ValueError) as ctx:
            conn(pre_size=100, post_size=100)
        self.assertIn("less than network size", str(ctx.exception))

    def test_different_sizes_error(self):
        """Test that different pre_size and post_size raises error."""
        conn = CorePeripheryRandom(core_size=20, seed=42)

        with self.assertRaises(ValueError) as ctx:
            conn(pre_size=100, post_size=80)
        self.assertIn("require pre_size == post_size", str(ctx.exception))

    def test_symmetric_core_periphery(self):
        """Test symmetric core-periphery connections."""
        conn = CorePeripheryRandom(
            core_size=20,
            core_core_prob=0.5,
            core_periphery_prob=0.3,
            periphery_core_prob=0.3,  # Same as core_periphery_prob
            periphery_periphery_prob=0.05,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        self.assertGreater(result.n_connections, 0)

        # Count core->periphery and periphery->core connections
        core_size = result.metadata['core_size']
        core_to_periphery = 0
        periphery_to_core = 0

        for pre_idx, post_idx in zip(result.pre_indices, result.post_indices):
            is_pre_core = pre_idx < core_size
            is_post_core = post_idx < core_size

            if is_pre_core and not is_post_core:
                core_to_periphery += 1
            elif not is_pre_core and is_post_core:
                periphery_to_core += 1

        # Both directions should exist and be roughly equal
        self.assertGreater(core_to_periphery, 0)
        self.assertGreater(periphery_to_core, 0)
        # Ratio should be close to 1.0 for symmetric probabilities
        ratio = core_to_periphery / periphery_to_core
        self.assertGreater(ratio, 0.7)
        self.assertLess(ratio, 1.3)

    def test_asymmetric_core_periphery_feedforward(self):
        """Test asymmetric core-periphery with strong feedforward."""
        conn = CorePeripheryRandom(
            core_size=20,
            core_core_prob=0.5,
            core_periphery_prob=0.4,  # Strong feedforward
            periphery_core_prob=0.1,  # Weak feedback
            periphery_periphery_prob=0.05,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        self.assertGreater(result.n_connections, 0)

        # Count core->periphery and periphery->core connections
        core_size = result.metadata['core_size']
        core_to_periphery = 0
        periphery_to_core = 0

        for pre_idx, post_idx in zip(result.pre_indices, result.post_indices):
            is_pre_core = pre_idx < core_size
            is_post_core = post_idx < core_size

            if is_pre_core and not is_post_core:
                core_to_periphery += 1
            elif not is_pre_core and is_post_core:
                periphery_to_core += 1

        # Core->periphery should be much greater than periphery->core
        self.assertGreater(core_to_periphery, periphery_to_core)

    def test_asymmetric_core_periphery_feedback(self):
        """Test asymmetric core-periphery with strong feedback."""
        conn = CorePeripheryRandom(
            core_size=20,
            core_core_prob=0.5,
            core_periphery_prob=0.1,  # Weak feedforward
            periphery_core_prob=0.4,  # Strong feedback
            periphery_periphery_prob=0.05,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        self.assertGreater(result.n_connections, 0)

        # Count core->periphery and periphery->core connections
        core_size = result.metadata['core_size']
        core_to_periphery = 0
        periphery_to_core = 0

        for pre_idx, post_idx in zip(result.pre_indices, result.post_indices):
            is_pre_core = pre_idx < core_size
            is_post_core = post_idx < core_size

            if is_pre_core and not is_post_core:
                core_to_periphery += 1
            elif not is_pre_core and is_post_core:
                periphery_to_core += 1

        # Periphery->core should be much greater than core->periphery
        self.assertGreater(periphery_to_core, core_to_periphery)

    def test_no_self_connections(self):
        """Test that self-connections are excluded."""
        conn = CorePeripheryRandom(
            core_size=10,
            core_core_prob=0.8,
            core_periphery_prob=0.5,
            periphery_core_prob=0.5,
            periphery_periphery_prob=0.3,
            seed=42
        )
        result = conn(pre_size=50, post_size=50)

        # Check no self-connections
        self.assertTrue(np.all(result.pre_indices != result.post_indices))

    def test_core_connectivity_higher_than_periphery(self):
        """Test that core has denser connectivity than periphery."""
        conn = CorePeripheryRandom(
            core_size=20,
            core_core_prob=0.6,
            core_periphery_prob=0.2,
            periphery_core_prob=0.2,
            periphery_periphery_prob=0.05,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        core_size = result.metadata['core_size']

        # Count core-core and periphery-periphery connections
        core_core = 0
        periphery_periphery = 0

        for pre_idx, post_idx in zip(result.pre_indices, result.post_indices):
            is_pre_core = pre_idx < core_size
            is_post_core = post_idx < core_size

            if is_pre_core and is_post_core:
                core_core += 1
            elif not is_pre_core and not is_post_core:
                periphery_periphery += 1

        # Core should have connections
        self.assertGreater(core_core, 0)

        # Core density should be higher than periphery density
        core_possible = core_size * (core_size - 1)
        periphery_size = 100 - core_size
        periphery_possible = periphery_size * (periphery_size - 1)

        core_density = core_core / core_possible if core_possible > 0 else 0
        periphery_density = periphery_periphery / periphery_possible if periphery_possible > 0 else 0

        self.assertGreater(core_density, periphery_density)

    def test_small_core_large_periphery(self):
        """Test network with small core and large periphery."""
        conn = CorePeripheryRandom(
            core_size=0.1,  # 10% core
            core_core_prob=0.7,
            core_periphery_prob=0.15,
            periphery_periphery_prob=0.03,
            seed=42
        )
        result = conn(pre_size=200, post_size=200)

        self.assertEqual(result.metadata['core_size'], 20)  # 10% of 200
        self.assertGreater(result.n_connections, 0)

    def test_large_core_small_periphery(self):
        """Test network with large core and small periphery."""
        conn = CorePeripheryRandom(
            core_size=0.8,  # 80% core
            core_core_prob=0.5,
            core_periphery_prob=0.2,
            periphery_periphery_prob=0.05,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        self.assertEqual(result.metadata['core_size'], 80)  # 80% of 100
        self.assertGreater(result.n_connections, 0)

    def test_zero_probabilities(self):
        """Test with all probabilities set to zero."""
        conn = CorePeripheryRandom(
            core_size=20,
            core_core_prob=0.0,
            core_periphery_prob=0.0,
            periphery_core_prob=0.0,
            periphery_periphery_prob=0.0,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        # Should have no connections
        self.assertEqual(result.n_connections, 0)
        self.assertEqual(len(result.pre_indices), 0)
        self.assertEqual(len(result.post_indices), 0)

    def test_high_probabilities(self):
        """Test with high connection probabilities."""
        conn = CorePeripheryRandom(
            core_size=20,
            core_core_prob=0.9,
            core_periphery_prob=0.8,
            periphery_core_prob=0.8,
            periphery_periphery_prob=0.7,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        # Should have many connections
        self.assertGreater(result.n_connections, 5000)

    def test_core_only_connections(self):
        """Test network with only core-core connections."""
        conn = CorePeripheryRandom(
            core_size=20,
            core_core_prob=0.5,
            core_periphery_prob=0.0,
            periphery_core_prob=0.0,
            periphery_periphery_prob=0.0,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        core_size = result.metadata['core_size']

        # All connections should be within core
        for pre_idx, post_idx in zip(result.pre_indices, result.post_indices):
            self.assertLess(pre_idx, core_size)
            self.assertLess(post_idx, core_size)

    def test_periphery_only_connections(self):
        """Test network with only periphery-periphery connections."""
        conn = CorePeripheryRandom(
            core_size=20,
            core_core_prob=0.0,
            core_periphery_prob=0.0,
            periphery_core_prob=0.0,
            periphery_periphery_prob=0.3,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        core_size = result.metadata['core_size']

        # All connections should be within periphery
        for pre_idx, post_idx in zip(result.pre_indices, result.post_indices):
            self.assertGreaterEqual(pre_idx, core_size)
            self.assertGreaterEqual(post_idx, core_size)

    def test_core_periphery_only_connections(self):
        """Test network with only core-periphery connections."""
        conn = CorePeripheryRandom(
            core_size=20,
            core_core_prob=0.0,
            core_periphery_prob=0.3,
            periphery_core_prob=0.3,
            periphery_periphery_prob=0.0,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        core_size = result.metadata['core_size']

        # All connections should be between core and periphery
        for pre_idx, post_idx in zip(result.pre_indices, result.post_indices):
            is_pre_core = pre_idx < core_size
            is_post_core = post_idx < core_size

            # Should not be both in core or both in periphery
            self.assertNotEqual(is_pre_core, is_post_core)

    def test_weights_and_delays(self):
        """Test core-periphery network with weights and delays."""
        conn = CorePeripheryRandom(
            core_size=20,
            core_core_prob=0.5,
            core_periphery_prob=0.2,
            periphery_periphery_prob=0.05,
            weight=1.5 * u.nS,
            delay=2.0 * u.ms,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        self.assertIsNotNone(result.weights)
        self.assertIsNotNone(result.delays)
        self.assertTrue(u.math.isscalar(result.weights))
        self.assertTrue(u.math.isscalar(result.delays))

        # Check weight values
        self.assertTrue(u.math.allclose(result.weights, 1.5 * u.nS))
        self.assertTrue(u.math.allclose(result.delays, 2.0 * u.ms))

    def test_custom_weight_initializer(self):
        """Test with custom weight initializer."""
        conn = CorePeripheryRandom(
            core_size=20,
            core_core_prob=0.5,
            core_periphery_prob=0.2,
            periphery_periphery_prob=0.05,
            weight=Normal(mean=1.0 * u.nS, std=0.2 * u.nS),
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        self.assertIsNotNone(result.weights)
        self.assertEqual(len(result.weights), result.n_connections)
        # Check that weights vary (not all the same)
        self.assertGreater(u.math.std(result.weights).mantissa, 0)

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        conn1 = CorePeripheryRandom(
            core_size=20,
            core_core_prob=0.5,
            core_periphery_prob=0.2,
            periphery_periphery_prob=0.05,
            seed=100
        )
        result1 = conn1(pre_size=100, post_size=100)

        conn2 = CorePeripheryRandom(
            core_size=20,
            core_core_prob=0.5,
            core_periphery_prob=0.2,
            periphery_periphery_prob=0.05,
            seed=100
        )
        result2 = conn2(pre_size=100, post_size=100)

        # Should produce identical results
        self.assertEqual(result1.n_connections, result2.n_connections)
        np.testing.assert_array_equal(result1.pre_indices, result2.pre_indices)
        np.testing.assert_array_equal(result1.post_indices, result2.post_indices)

    def test_metadata_completeness(self):
        """Test that all expected metadata is present."""
        conn = CorePeripheryRandom(
            core_size=25,
            core_core_prob=0.6,
            core_periphery_prob=0.25,
            periphery_core_prob=0.15,
            periphery_periphery_prob=0.08,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        metadata = result.metadata
        self.assertIn('pattern', metadata)
        self.assertIn('core_size', metadata)
        self.assertIn('core_core_prob', metadata)
        self.assertIn('core_periphery_prob', metadata)
        self.assertIn('periphery_core_prob', metadata)
        self.assertIn('periphery_periphery_prob', metadata)

        self.assertEqual(metadata['pattern'], 'core_periphery')
        self.assertEqual(metadata['core_size'], 25)
        self.assertEqual(metadata['core_core_prob'], 0.6)
        self.assertEqual(metadata['core_periphery_prob'], 0.25)
        self.assertEqual(metadata['periphery_core_prob'], 0.15)
        self.assertEqual(metadata['periphery_periphery_prob'], 0.08)

    @pytest.mark.skip(reason="too slow for regular test runs")
    def test_large_network(self):
        """Test with a larger network to ensure scalability."""
        conn = CorePeripheryRandom(
            core_size=0.15,  # 15% core
            core_core_prob=0.3,
            core_periphery_prob=0.1,
            periphery_periphery_prob=0.02,
            seed=42
        )
        result = conn(pre_size=5000, post_size=5000)

        self.assertEqual(result.metadata['core_size'], 750)  # 15% of 5000
        self.assertGreater(result.n_connections, 0)

        # Verify no invalid indices
        self.assertTrue(np.all(result.pre_indices >= 0))
        self.assertTrue(np.all(result.pre_indices < 5000))
        self.assertTrue(np.all(result.post_indices >= 0))
        self.assertTrue(np.all(result.post_indices < 5000))

    def test_minimal_core(self):
        """Test with minimal core size (1 neuron)."""
        conn = CorePeripheryRandom(
            core_size=1,
            core_core_prob=0.5,
            core_periphery_prob=0.2,
            periphery_periphery_prob=0.05,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        self.assertEqual(result.metadata['core_size'], 1)
        # Core neuron should have connections to periphery
        self.assertGreater(result.n_connections, 0)

    def test_tuple_size_input(self):
        """Test that tuple pre_size/post_size is handled correctly."""
        conn = CorePeripheryRandom(
            core_size=0.2,
            core_core_prob=0.5,
            core_periphery_prob=0.2,
            periphery_periphery_prob=0.05,
            seed=42
        )
        result = conn(pre_size=(10, 10), post_size=(10, 10))

        # Should treat as 100 neurons, 20% = 20 neurons in core
        self.assertEqual(result.metadata['core_size'], 20)
        self.assertGreater(result.n_connections, 0)

    def test_connection_type_distribution(self):
        """Test distribution of different connection types."""
        conn = CorePeripheryRandom(
            core_size=30,
            core_core_prob=0.8,
            core_periphery_prob=0.2,
            periphery_periphery_prob=0.1,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        core_size = result.metadata['core_size']

        # Count each connection type
        core_core = 0
        core_periphery = 0
        periphery_core = 0
        periphery_periphery = 0

        for pre_idx, post_idx in zip(result.pre_indices, result.post_indices):
            is_pre_core = pre_idx < core_size
            is_post_core = post_idx < core_size

            if is_pre_core and is_post_core:
                core_core += 1
            elif is_pre_core and not is_post_core:
                core_periphery += 1
            elif not is_pre_core and is_post_core:
                periphery_core += 1
            else:
                periphery_periphery += 1

        # All types should exist with these parameters
        self.assertGreater(core_core, 0)
        self.assertGreater(core_periphery, 0)
        self.assertGreater(periphery_core, 0)
        self.assertGreater(periphery_periphery, 0)

        # Core-core should be most dense
        self.assertGreater(core_core, periphery_periphery)

    def test_very_small_network(self):
        """Test with very small network."""
        conn = CorePeripheryRandom(
            core_size=2,
            core_core_prob=0.8,
            core_periphery_prob=0.5,
            periphery_periphery_prob=0.3,
            seed=42
        )
        result = conn(pre_size=5, post_size=5)

        self.assertEqual(result.metadata['core_size'], 2)
        # Should work without errors

    def test_default_parameters(self):
        """Test with default parameters."""
        conn = CorePeripheryRandom(
            core_size=20,
            seed=42
        )
        result = conn(pre_size=100, post_size=100)

        # Check default values
        self.assertEqual(result.metadata['core_core_prob'], 0.5)
        self.assertEqual(result.metadata['core_periphery_prob'], 0.2)
        self.assertEqual(result.metadata['periphery_core_prob'], 0.2)
        self.assertEqual(result.metadata['periphery_periphery_prob'], 0.05)
        self.assertGreater(result.n_connections, 0)
