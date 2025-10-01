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

import brainunit as u
import numpy as np

from braintools.conn import (
    ExcitatoryInhibitory,
)
from braintools.init import Constant


class TestExcitatoryInhibitory(unittest.TestCase):
    """
    Test ExcitatoryInhibitory connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import ExcitatoryInhibitory

        # Standard E-I network with Dale's principle
        ei_net = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.1,
            inh_prob=0.2,
            exc_weight=1.0 * u.nS,
            inh_weight=-0.8 * u.nS,
            delay=1.5 * u.ms
        )

        result = ei_net(pre_size=100, post_size=100)

        # Check E-I structure
        assert result.metadata['pattern'] == 'excitatory_inhibitory'
        assert result.metadata['exc_ratio'] == 0.8
        assert result.metadata['n_excitatory'] == 80
        assert result.metadata['n_inhibitory'] == 20

        # Should have both positive and negative weights
        if result.n_connections > 0:
            weights = u.get_mantissa(result.weights)
            assert np.any(weights > 0)  # Excitatory weights
            assert np.any(weights < 0)  # Inhibitory weights
    """

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

        conn = ExcitatoryInhibitory(
            exc_ratio=1.0,  # All excitatory
            exc_prob=0.2,
            inh_prob=0.3,  # Won't be used
            exc_weight=exc_weight_init,
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
        inh_weight_init = Constant(-1.5 * u.nS)

        conn = ExcitatoryInhibitory(
            exc_ratio=0.0,  # All inhibitory
            exc_prob=0.2,  # Won't be used
            inh_prob=0.3,
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
        delay_init = Constant(2.0 * u.ms)

        conn = ExcitatoryInhibitory(
            exc_ratio=0.6,
            exc_prob=0.1,
            inh_prob=0.2,
            exc_weight=0.5 * u.nS,
            inh_weight=-0.3 * u.nS,
            delay=delay_init,
            seed=42
        )

        result = conn(pre_size=25, post_size=25)

        if result.n_connections > 0:
            self.assertIsNotNone(result.delays)
            np.testing.assert_array_almost_equal(u.get_mantissa(result.delays), 2.0)

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
