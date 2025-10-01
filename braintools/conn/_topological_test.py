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
    RandomModular,
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
        conn = RandomModular(
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
        conn = RandomModular(n_modules=2, seed=42)

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=8)  # Different sizes not allowed

    def test_modular_uneven_module_assignment(self):
        # Test when population size doesn't divide evenly by number of modules
        conn = RandomModular(
            n_modules=3,
            intra_prob=0.3,
            inter_prob=0.01,
            seed=42
        )
        result = conn(pre_size=10, post_size=10)  # 10 doesn't divide evenly by 3

        # Should still work (extra neurons assigned to last module)
        self.assertGreater(result.n_connections, 0)
