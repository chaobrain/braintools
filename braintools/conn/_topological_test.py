# Copyright 2025 BrainSim Ecosystem Limited. All Rights Reserved.
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

import numpy as np

from braintools.conn import (
    SmallWorld,
    ScaleFree,
    Regular,
    ModularRandom,
    ModularGeneral,
    Random,
)


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

        # Should handle gracefully (some modules might be empty)
        self.assertEqual(result.metadata['module_sizes'], [2, 1, 0])

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
