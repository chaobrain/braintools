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

from braintools.conn import (
    ExcitatoryInhibitory,
)
from braintools.init import Constant, Uniform, Normal


class TestExcitatoryInhibitory(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_excitatory_inhibitory(self):
        exc_weight_init = Constant(1.2 * u.nS)
        inh_weight_init = Constant(-0.8 * u.nS)

        conn = ExcitatoryInhibitory(
            exc_ratio=0.75,
            exc_prob=0.15,
            inh_prob=0.25,
            exc_weight=exc_weight_init,
            inh_weight=inh_weight_init,
            seed=42
        )

        result = conn(pre_size=40, post_size=40)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'excitatory_inhibitory')
        self.assertEqual(result.metadata['exc_ratio'], 0.75)
        self.assertEqual(result.metadata['n_excitatory'], 30)  # 40 * 0.75
        self.assertEqual(result.metadata['n_inhibitory'], 10)  # 40 - 30

        if result.n_connections > 0:
            # Should have both positive and negative weights
            weights = u.get_mantissa(result.weights)
            self.assertTrue(np.any(weights > 0))  # Excitatory
            self.assertTrue(np.any(weights < 0))  # Inhibitory

            # Check that excitatory weights are 1.2 and inhibitory are -0.8
            exc_weights = weights[weights > 0]
            inh_weights = weights[weights < 0]

            if len(exc_weights) > 0:
                np.testing.assert_array_almost_equal(exc_weights, 1.2)
            if len(inh_weights) > 0:
                np.testing.assert_array_almost_equal(inh_weights, -0.8)

    def test_excitatory_inhibitory_only_excitatory(self):
        exc_weight_init = Constant(1.0 * u.nS)
        inh_weight_init = Constant(-1.0 * u.nS)

        conn = ExcitatoryInhibitory(
            exc_ratio=1.0,  # All excitatory
            exc_prob=0.2,
            inh_prob=0.3,  # Won't be used
            exc_weight=exc_weight_init,
            inh_weight=inh_weight_init,
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        self.assertEqual(result.metadata['n_excitatory'], 20)
        self.assertEqual(result.metadata['n_inhibitory'], 0)

        if result.n_connections > 0:
            # All weights should be positive
            weights = u.get_mantissa(result.weights)
            self.assertTrue(np.all(weights > 0))

    def test_excitatory_inhibitory_only_inhibitory(self):
        exc_weight_init = Constant(1.0 * u.nS)
        inh_weight_init = Constant(-1.5 * u.nS)

        conn = ExcitatoryInhibitory(
            exc_ratio=0.0,  # All inhibitory
            exc_prob=0.2,  # Won't be used
            inh_prob=0.3,
            exc_weight=exc_weight_init,
            inh_weight=inh_weight_init,
            seed=42
        )

        result = conn(pre_size=15, post_size=15)

        self.assertEqual(result.metadata['n_excitatory'], 0)
        self.assertEqual(result.metadata['n_inhibitory'], 15)

        if result.n_connections > 0:
            # All weights should be negative
            weights = u.get_mantissa(result.weights)
            self.assertTrue(np.all(weights < 0))

    def test_excitatory_inhibitory_with_delays(self):
        exc_delay_init = Constant(1.5 * u.ms)
        inh_delay_init = Constant(0.8 * u.ms)

        conn = ExcitatoryInhibitory(
            exc_ratio=0.6,
            exc_prob=0.1,
            inh_prob=0.2,
            exc_weight=0.5 * u.nS,
            inh_weight=-0.3 * u.nS,
            exc_delay=exc_delay_init,
            inh_delay=inh_delay_init,
            seed=42
        )

        result = conn(pre_size=25, post_size=25)

        if result.n_connections > 0:
            self.assertIsNotNone(result.delays)
            delays = u.get_mantissa(result.delays)

            # Check that we have both types of delays
            self.assertTrue(np.any(delays == 1.5) or np.any(delays == 0.8))

    def test_excitatory_inhibitory_asymmetric_sizes(self):
        conn = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.1,
            inh_prob=0.15,
            seed=42
        )

        result = conn(pre_size=25, post_size=30)

        self.assertEqual(result.shape, (25, 30))
        # Pre population split: 20 excitatory, 5 inhibitory
        self.assertEqual(result.metadata['n_excitatory'], 20)
        self.assertEqual(result.metadata['n_inhibitory'], 5)

    def test_excitatory_inhibitory_zero_probabilities(self):
        conn = ExcitatoryInhibitory(
            exc_ratio=0.7,
            exc_prob=0.0,  # No excitatory connections
            inh_prob=0.0,  # No inhibitory connections
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        self.assertEqual(result.n_connections, 0)

    def test_excitatory_inhibitory_tuple_sizes(self):
        conn = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.1,
            inh_prob=0.2,
            seed=42
        )

        result = conn(pre_size=(4, 5), post_size=(2, 10))

        self.assertEqual(result.pre_size, (4, 5))
        self.assertEqual(result.post_size, (2, 10))
        # Pre size = 20, 80% excitatory = 16, 20% inhibitory = 4
        self.assertEqual(result.metadata['n_excitatory'], 16)
        self.assertEqual(result.metadata['n_inhibitory'], 4)

    def test_no_weights_or_delays(self):
        """Test connectivity without weights or delays."""
        conn = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.15,
            inh_prob=0.2,
            seed=42
        )

        result = conn(pre_size=30, post_size=30)

        # Should have connections but no weights or delays
        self.assertGreater(result.n_connections, 0)
        self.assertIsNone(result.weights)
        self.assertIsNone(result.delays)

    def test_mismatched_weight_specifications(self):
        """Test that having only one of exc_weight or inh_weight raises error."""
        with self.assertRaises(ValueError) as context:
            conn = ExcitatoryInhibitory(
                exc_ratio=0.8,
                exc_prob=0.1,
                inh_prob=0.2,
                exc_weight=1.0 * u.nS,
                inh_weight=None,  # Missing!
                seed=42
            )
            conn(pre_size=20, post_size=20)

        self.assertIn("must be both None or both specified", str(context.exception))

        with self.assertRaises(ValueError) as context:
            conn = ExcitatoryInhibitory(
                exc_ratio=0.8,
                exc_prob=0.1,
                inh_prob=0.2,
                exc_weight=None,  # Missing!
                inh_weight=-1.0 * u.nS,
                seed=42
            )
            conn(pre_size=20, post_size=20)

        self.assertIn("must be both None or both specified", str(context.exception))

    def test_mismatched_delay_specifications(self):
        """Test that having only one of exc_delay or inh_delay raises error."""
        with self.assertRaises(ValueError) as context:
            conn = ExcitatoryInhibitory(
                exc_ratio=0.8,
                exc_prob=0.1,
                inh_prob=0.2,
                exc_weight=1.0 * u.nS,
                inh_weight=-1.0 * u.nS,
                exc_delay=1.0 * u.ms,
                inh_delay=None,  # Missing!
                seed=42
            )
            conn(pre_size=20, post_size=20)

        self.assertIn("must be both None or both specified", str(context.exception))

    def test_different_exc_inh_delays(self):
        """Test that excitatory and inhibitory neurons can have different delays."""
        conn = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.2,
            inh_prob=0.2,
            exc_weight=1.0 * u.nS,
            inh_weight=-1.0 * u.nS,
            exc_delay=Constant(2.0 * u.ms),
            inh_delay=Constant(0.5 * u.ms),
            seed=42
        )

        result = conn(pre_size=100, post_size=100)

        if result.n_connections > 0:
            delays = u.get_mantissa(result.delays)
            weights = u.get_mantissa(result.weights)

            # Get delays for excitatory (positive weight) and inhibitory (negative weight)
            exc_delays = delays[weights > 0]
            inh_delays = delays[weights < 0]

            if len(exc_delays) > 0:
                np.testing.assert_array_almost_equal(exc_delays, 2.0)
            if len(inh_delays) > 0:
                np.testing.assert_array_almost_equal(inh_delays, 0.5)

    def test_variable_weights_uniform(self):
        """Test with variable weights using Uniform initializer."""
        conn = ExcitatoryInhibitory(
            exc_ratio=0.75,
            exc_prob=0.15,
            inh_prob=0.2,
            exc_weight=Uniform(0.5 * u.nS, 1.5 * u.nS),
            inh_weight=Uniform(-1.5 * u.nS, -0.5 * u.nS),
            seed=42
        )

        result = conn(pre_size=50, post_size=50)

        if result.n_connections > 0:
            weights = u.get_mantissa(result.weights)

            # Check that excitatory weights are in range
            exc_weights = weights[weights > 0]
            if len(exc_weights) > 0:
                self.assertTrue(np.all(exc_weights >= 0.5))
                self.assertTrue(np.all(exc_weights <= 1.5))

            # Check that inhibitory weights are in range
            inh_weights = weights[weights < 0]
            if len(inh_weights) > 0:
                self.assertTrue(np.all(inh_weights >= -1.5))
                self.assertTrue(np.all(inh_weights <= -0.5))

    def test_variable_weights_normal(self):
        """Test with variable weights using Normal initializer."""
        conn = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.15,
            inh_prob=0.2,
            exc_weight=Normal(1.0 * u.nS, 0.1 * u.nS),
            inh_weight=Normal(-0.8 * u.nS, 0.1 * u.nS),
            seed=42
        )

        result = conn(pre_size=100, post_size=100)

        if result.n_connections > 0:
            weights = u.get_mantissa(result.weights)

            # Check mean values are approximately correct
            exc_weights = weights[weights > 0]
            inh_weights = weights[weights < 0]

            if len(exc_weights) > 10:  # Need enough samples
                self.assertAlmostEqual(np.mean(exc_weights), 1.0, delta=0.2)

            if len(inh_weights) > 10:  # Need enough samples
                self.assertAlmostEqual(np.mean(inh_weights), -0.8, delta=0.2)

    def test_high_probability_connections(self):
        """Test with very high connection probabilities."""
        conn = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.9,
            inh_prob=0.9,
            exc_weight=1.0 * u.nS,
            inh_weight=-1.0 * u.nS,
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        # With prob=0.9, should have many connections
        # Expected: ~(20*0.8*20*0.9 + 20*0.2*20*0.9) = ~288 + ~72 = ~360
        self.assertGreater(result.n_connections, 200)

    def test_connection_index_validity(self):
        """Test that all connection indices are valid."""
        conn = ExcitatoryInhibitory(
            exc_ratio=0.7,
            exc_prob=0.15,
            inh_prob=0.2,
            seed=42
        )

        result = conn(pre_size=30, post_size=40)

        # All pre indices should be < 30
        self.assertTrue(np.all(result.pre_indices >= 0))
        self.assertTrue(np.all(result.pre_indices < 30))

        # All post indices should be < 40
        self.assertTrue(np.all(result.post_indices >= 0))
        self.assertTrue(np.all(result.post_indices < 40))

    def test_excitatory_inhibitory_ratio_validation(self):
        """Test that excitatory/inhibitory neuron counts match exc_ratio."""
        for exc_ratio in [0.2, 0.5, 0.75, 0.8, 0.9]:
            with self.subTest(exc_ratio=exc_ratio):
                conn = ExcitatoryInhibitory(
                    exc_ratio=exc_ratio,
                    exc_prob=0.1,
                    inh_prob=0.1,
                    seed=42
                )

                result = conn(pre_size=100, post_size=50)

                expected_exc = int(100 * exc_ratio)
                expected_inh = 100 - expected_exc

                self.assertEqual(result.metadata['n_excitatory'], expected_exc)
                self.assertEqual(result.metadata['n_inhibitory'], expected_inh)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with the same seed."""
        conn1 = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.15,
            inh_prob=0.2,
            exc_weight=Uniform(0.5 * u.nS, 1.5 * u.nS),
            inh_weight=Uniform(-1.5 * u.nS, -0.5 * u.nS),
            seed=12345
        )

        conn2 = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.15,
            inh_prob=0.2,
            exc_weight=Uniform(0.5 * u.nS, 1.5 * u.nS),
            inh_weight=Uniform(-1.5 * u.nS, -0.5 * u.nS),
            seed=12345
        )

        result1 = conn1(pre_size=50, post_size=50)
        result2 = conn2(pre_size=50, post_size=50)

        # Should have identical connectivity
        np.testing.assert_array_equal(result1.pre_indices, result2.pre_indices)
        np.testing.assert_array_equal(result1.post_indices, result2.post_indices)
        np.testing.assert_array_almost_equal(
            u.get_mantissa(result1.weights),
            u.get_mantissa(result2.weights)
        )

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        conn1 = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.15,
            inh_prob=0.2,
            seed=111
        )

        conn2 = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.15,
            inh_prob=0.2,
            seed=222
        )

        result1 = conn1(pre_size=50, post_size=50)
        result2 = conn2(pre_size=50, post_size=50)

        # Should have different connectivity
        self.assertFalse(np.array_equal(result1.pre_indices, result2.pre_indices))

    def test_metadata_completeness(self):
        """Test that metadata contains all expected fields."""
        conn = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.1,
            inh_prob=0.15,
            seed=42
        )

        result = conn(pre_size=50, post_size=40)

        # Check all expected metadata fields
        self.assertIn('pattern', result.metadata)
        self.assertIn('exc_ratio', result.metadata)
        self.assertIn('n_excitatory', result.metadata)
        self.assertIn('n_inhibitory', result.metadata)

        self.assertEqual(result.metadata['pattern'], 'excitatory_inhibitory')
        self.assertEqual(result.metadata['exc_ratio'], 0.8)

    def test_large_network(self):
        """Test with a larger network."""
        conn = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.05,
            inh_prob=0.1,
            exc_weight=1.0 * u.nS,
            inh_weight=-1.0 * u.nS,
            seed=42
        )

        result = conn(pre_size=500, post_size=500)

        # Should have connections
        self.assertGreater(result.n_connections, 0)

        # Check weight distribution
        if result.n_connections > 0:
            weights = u.get_mantissa(result.weights)
            n_exc = np.sum(weights > 0)
            n_inh = np.sum(weights < 0)

            # With 80% excitatory neurons and exc_prob=0.05, inh_prob=0.1
            # Expect roughly: 500*0.8*500*0.05 = 10000 exc connections
            #                 500*0.2*500*0.1 = 5000 inh connections
            # So ratio should be roughly 2:1
            if n_inh > 0:
                ratio = n_exc / n_inh
                self.assertGreater(ratio, 1.0)  # More excitatory than inhibitory
