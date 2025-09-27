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

"""
Comprehensive tests for population rate connectivity classes.

This test suite covers:
- Basic population patterns (PopulationCoupling, MeanField, AllToAllPopulations, RandomPopulations)
- Population-specific patterns (ExcitatoryInhibitory, FeedforwardInhibition, RecurrentAmplification, CompetitiveNetwork)
- Hierarchical patterns (HierarchicalPopulations, FeedforwardHierarchy, RecurrentHierarchy, LayeredNetwork)
- Specialized patterns (PopulationDistance, RateDependent, WilsonCowanNetwork, FiringRateNetworks)
- Custom patterns (CustomPopulation)
"""

import unittest

import brainunit as u
import numpy as np

from braintools.conn._conn_population import (
    PopulationCoupling,
    MeanField,
    AllToAllPopulations,
    RandomPopulations,
    ExcitatoryInhibitory,
    FeedforwardInhibition,
    RecurrentAmplification,
    CompetitiveNetwork,
    HierarchicalPopulations,
    FeedforwardHierarchy,
    RecurrentHierarchy,
    LayeredNetwork,
    PopulationDistance,
    RateDependent,
    WilsonCowanNetwork,
    FiringRateNetworks,
    CustomPopulation,
)
from braintools.conn import ConstantWeight


class TestPopulationCoupling(unittest.TestCase):
    """
    Test PopulationCoupling connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_population import PopulationCoupling

        # Basic excitatory-inhibitory coupling matrix
        coupling_matrix = np.array([
            [0.5, 0.8],    # E -> [E, I]
            [-1.2, -0.3]   # I -> [E, I]
        ])

        conn = PopulationCoupling(
            coupling_matrix=coupling_matrix,
            coupling_type='additive'
        )

        result = conn(pre_size=2, post_size=2)

        assert result.model_type == 'population_rate'
        assert result.metadata['pattern'] == 'population_coupling'
        assert result.n_connections == 4  # All entries are non-zero

        # Dictionary-based coupling
        coupling_dict = {
            (0, 0): 0.5,    # E -> E
            (0, 1): 0.8,    # E -> I
            (1, 0): -1.2,   # I -> E
            (1, 1): -0.3    # I -> I
        }

        dict_conn = PopulationCoupling(coupling_matrix=coupling_dict)
        result_dict = dict_conn(pre_size=2, post_size=2)
        assert result_dict.n_connections == 4
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_coupling_matrix(self):
        coupling_matrix = np.array([
            [0.5, 0.8],
            [-1.2, -0.3]
        ])

        conn = PopulationCoupling(coupling_matrix=coupling_matrix, seed=42)
        result = conn(pre_size=2, post_size=2)

        self.assertEqual(result.model_type, 'population_rate')
        self.assertEqual(result.metadata['pattern'], 'population_coupling')
        self.assertEqual(result.metadata['coupling_type'], 'additive')
        self.assertEqual(result.n_connections, 4)  # All entries non-zero

        # Check that weights match coupling matrix
        expected_weights = [0.5, 0.8, -1.2, -0.3]
        actual_weights = sorted(result.weights)
        np.testing.assert_array_almost_equal(actual_weights, sorted(expected_weights))

    def test_coupling_with_zeros(self):
        # Matrix with some zero entries
        coupling_matrix = np.array([
            [0.5, 0.0],
            [0.0, -0.3]
        ])

        conn = PopulationCoupling(coupling_matrix=coupling_matrix, seed=42)
        result = conn(pre_size=2, post_size=2)

        self.assertEqual(result.n_connections, 2)  # Only non-zero entries
        expected_weights = [0.5, -0.3]
        actual_weights = sorted(result.weights)
        np.testing.assert_array_almost_equal(actual_weights, sorted(expected_weights))

    def test_dictionary_coupling(self):
        coupling_dict = {
            (0, 0): 0.5,
            (0, 1): 0.8,
            (1, 0): -1.2,
            (1, 1): -0.3
        }

        conn = PopulationCoupling(coupling_matrix=coupling_dict, seed=42)
        result = conn(pre_size=2, post_size=2)

        self.assertEqual(result.n_connections, 4)
        expected_weights = [0.5, 0.8, -1.2, -0.3]
        actual_weights = sorted(result.weights)
        np.testing.assert_array_almost_equal(actual_weights, sorted(expected_weights))

    def test_coupling_with_time_constants(self):
        coupling_matrix = np.array([
            [0.5, 0.8]
        ])
        time_constants = [10 * u.ms, 20 * u.ms]

        conn = PopulationCoupling(
            coupling_matrix=coupling_matrix,
            time_constants=time_constants,
            seed=42
        )

        result = conn(pre_size=1, post_size=2)

        self.assertIsNotNone(result.delays)
        self.assertEqual(result.delays.unit, u.ms)
        # Delays should be time constants of target populations
        np.testing.assert_array_almost_equal(u.get_mantissa(result.delays), [10, 20])

    def test_coupling_time_constants_no_units(self):
        coupling_matrix = np.array([[0.5, 0.8]])
        time_constants = [10.0, 20.0]  # No units

        conn = PopulationCoupling(
            coupling_matrix=coupling_matrix,
            time_constants=time_constants,
            seed=42
        )

        result = conn(pre_size=1, post_size=2)

        self.assertIsNotNone(result.delays)
        self.assertEqual(result.delays.unit, u.ms)  # Should get default units
        np.testing.assert_array_almost_equal(u.get_mantissa(result.delays), [10, 20])

    def test_coupling_asymmetric_sizes(self):
        # 3x2 coupling matrix
        coupling_matrix = np.array([
            [0.5, 0.8],
            [-1.2, -0.3],
            [0.1, 0.4]
        ])

        conn = PopulationCoupling(coupling_matrix=coupling_matrix, seed=42)
        result = conn(pre_size=3, post_size=2)

        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.n_connections, 6)  # All entries non-zero

    def test_coupling_size_mismatch(self):
        # Matrix larger than network sizes
        coupling_matrix = np.array([
            [0.5, 0.8, 0.1],
            [-1.2, -0.3, 0.2],
            [0.1, 0.4, 0.6]
        ])

        conn = PopulationCoupling(coupling_matrix=coupling_matrix, seed=42)
        result = conn(pre_size=2, post_size=2)

        # Should use submatrix
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.n_connections, 4)

    def test_coupling_empty_connections(self):
        # All-zero coupling matrix
        coupling_matrix = np.zeros((2, 2))

        conn = PopulationCoupling(coupling_matrix=coupling_matrix, seed=42)
        result = conn(pre_size=2, post_size=2)

        self.assertEqual(result.n_connections, 0)

    def test_coupling_tuple_sizes(self):
        coupling_matrix = np.array([[0.5, 0.8]])

        conn = PopulationCoupling(coupling_matrix=coupling_matrix, seed=42)
        result = conn(pre_size=(1, 1), post_size=(2, 1))

        self.assertEqual(result.pre_size, (1, 1))
        self.assertEqual(result.post_size, (2, 1))
        self.assertEqual(result.shape, (1, 2))

    def test_coupling_metadata(self):
        coupling_matrix = np.array([[0.5]])
        population_sizes = [100]

        conn = PopulationCoupling(
            coupling_matrix=coupling_matrix,
            population_sizes=population_sizes,
            coupling_type='multiplicative',
            seed=42
        )

        result = conn(pre_size=1, post_size=1)

        self.assertEqual(result.metadata['coupling_type'], 'multiplicative')
        self.assertEqual(result.metadata['population_sizes'], population_sizes)
        self.assertEqual(result.metadata['coupling_matrix'], [[0.5]])


class TestMeanField(unittest.TestCase):
    """
    Test MeanField connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_population import MeanField

        # Basic mean-field coupling
        mf = MeanField(
            field_strength=0.1,
            normalization='sqrt',
            connectivity_fraction=1.0
        )

        result = mf(pre_size=5, post_size=3)

        assert result.model_type == 'population_rate'
        assert result.metadata['pattern'] == 'mean_field'
        assert result.n_connections == 15  # 5 * 3 (fully connected)

        # Sparse mean-field
        sparse_mf = MeanField(
            field_strength=0.05,
            connectivity_fraction=0.3
        )

        result_sparse = sparse_mf(pre_size=10, post_size=10)
        # Should have approximately 30% of total possible connections
        expected_connections = int(0.3 * 10 * 10)
        assert abs(result_sparse.n_connections - expected_connections) <= 5
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_mean_field(self):
        conn = MeanField(
            field_strength=0.1,
            normalization='sqrt',
            connectivity_fraction=1.0,
            seed=42
        )

        result = conn(pre_size=4, post_size=3)

        self.assertEqual(result.model_type, 'population_rate')
        self.assertEqual(result.metadata['pattern'], 'mean_field')
        self.assertEqual(result.metadata['field_strength'], 0.1)
        self.assertEqual(result.metadata['normalization'], 'sqrt')
        self.assertEqual(result.n_connections, 12)  # 4 * 3

        # Check weight normalization
        expected_weight = 0.1 / np.sqrt(4 * 3)
        np.testing.assert_array_almost_equal(result.weights, expected_weight)

    def test_mean_field_normalization_types(self):
        normalizations = ['none', 'source', 'target', 'sqrt', 'both']

        for norm in normalizations:
            conn = MeanField(
                field_strength=0.2,
                normalization=norm,
                seed=42
            )

            result = conn(pre_size=3, post_size=4)

            if norm == 'none':
                expected_weight = 0.2
            elif norm == 'source':
                expected_weight = 0.2 / 3
            elif norm == 'target':
                expected_weight = 0.2 / 4
            elif norm == 'sqrt':
                expected_weight = 0.2 / np.sqrt(3 * 4)
            elif norm == 'both':
                expected_weight = 0.2 / (3 * 4)

            np.testing.assert_array_almost_equal(result.weights, expected_weight)

    def test_mean_field_sparse_connectivity(self):
        conn = MeanField(
            field_strength=0.1,
            connectivity_fraction=0.4,
            seed=42
        )

        result = conn(pre_size=10, post_size=10)

        total_possible = 10 * 10
        expected_connections = int(total_possible * 0.4)

        # Allow some tolerance due to random selection
        self.assertAlmostEqual(result.n_connections, expected_connections, delta=8)

    def test_mean_field_with_distance_dependence(self):
        def distance_decay(distances):
            return np.exp(-distances / 50.0)

        positions = np.random.RandomState(42).uniform(0, 100, (5, 2))

        conn = MeanField(
            field_strength=0.2,
            distance_dependence=distance_decay,
            seed=42
        )

        result = conn(
            pre_size=5, post_size=5,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.n_connections, 25)  # Still all-to-all
        # Weights should be modulated by distance
        # (Hard to test exact values due to distance calculation complexity)
        self.assertGreater(np.max(result.weights), np.min(result.weights))

    def test_mean_field_field_strength_with_units(self):
        conn = MeanField(
            field_strength=0.15 * u.nS,
            normalization='none',
            seed=42
        )

        result = conn(pre_size=2, post_size=2)

        # Should preserve units
        self.assertIsInstance(result.weights, u.Quantity)
        self.assertEqual(result.weights.unit, u.nS)
        np.testing.assert_array_almost_equal(u.get_magnitude(result.weights), 0.15)

    def test_mean_field_empty_connections(self):
        conn = MeanField(
            field_strength=0.1,
            connectivity_fraction=0.0,  # No connections
            seed=42
        )

        result = conn(pre_size=5, post_size=5)

        print(result)
        self.assertEqual(result.n_connections, 0)

    def test_mean_field_asymmetric_sizes(self):
        conn = MeanField(
            field_strength=0.1,
            normalization='source',
            seed=42
        )

        result = conn(pre_size=6, post_size=4)

        self.assertEqual(result.shape, (6, 4))
        self.assertEqual(result.n_connections, 24)  # 6 * 4

        expected_weight = 0.1 / 6  # source normalization
        np.testing.assert_array_almost_equal(result.weights, expected_weight)

    def test_mean_field_tuple_sizes(self):
        conn = MeanField(field_strength=0.1, seed=42)

        result = conn(pre_size=(2, 3), post_size=(1, 4))

        self.assertEqual(result.pre_size, (2, 3))
        self.assertEqual(result.post_size, (1, 4))
        self.assertEqual(result.shape, (6, 4))


class TestExcitatoryInhibitory(unittest.TestCase):
    """
    Test ExcitatoryInhibitory population connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_population import ExcitatoryInhibitory

        # Standard E-I network
        ei = ExcitatoryInhibitory(
            exc_self_coupling=0.5,
            exc_to_inh_coupling=0.8,
            inh_to_exc_coupling=-1.2,
            inh_self_coupling=-0.3,
            exc_time_constant=20 * u.ms,
            inh_time_constant=10 * u.ms
        )

        result = ei(pre_size=2, post_size=2)

        assert result.model_type == 'population_rate'
        assert result.metadata['pattern'] == 'excitatory_inhibitory'
        assert result.n_connections == 4  # Full 2x2 coupling

        # Check that coupling matrix has expected structure
        # E->E: positive, E->I: positive, I->E: negative, I->I: negative
        weights = result.weights
        assert np.any(weights > 0)  # Excitatory connections
        assert np.any(weights < 0)  # Inhibitory connections
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_excitatory_inhibitory(self):
        conn = ExcitatoryInhibitory(
            exc_self_coupling=0.5,
            exc_to_inh_coupling=0.8,
            inh_to_exc_coupling=-1.2,
            inh_self_coupling=-0.3,
            seed=42
        )

        result = conn(pre_size=2, post_size=2)

        self.assertEqual(result.model_type, 'population_rate')
        self.assertEqual(result.metadata['pattern'], 'excitatory_inhibitory')
        self.assertEqual(result.n_connections, 4)

        # Check coupling structure
        expected_weights = [0.5, 0.8, -1.2, -0.3]
        actual_weights = sorted(result.weights)
        np.testing.assert_array_almost_equal(actual_weights, sorted(expected_weights))

    def test_excitatory_inhibitory_with_time_constants(self):
        exc_tau = 20 * u.ms
        inh_tau = 10 * u.ms

        conn = ExcitatoryInhibitory(
            exc_self_coupling=0.4,
            exc_to_inh_coupling=0.6,
            inh_to_exc_coupling=-1.0,
            inh_self_coupling=-0.2,
            exc_time_constant=exc_tau,
            inh_time_constant=inh_tau,
            seed=42
        )

        result = conn(pre_size=2, post_size=2)

        self.assertIsNotNone(result.delays)
        self.assertEqual(result.delays.unit, u.ms)

        # Delays should be time constants of target populations
        # Connections: (0,0), (0,1), (1,0), (1,1)
        # Target time constants: 20, 10, 20, 10
        expected_delays = [20, 10, 20, 10]
        actual_delays = sorted(u.get_mantissa(result.delays))
        np.testing.assert_array_almost_equal(actual_delays, sorted(expected_delays))

    def test_excitatory_inhibitory_scalar_time_constants(self):
        conn = ExcitatoryInhibitory(
            exc_time_constant=15.0,  # Scalar
            inh_time_constant=8.0,   # Scalar
            seed=42
        )

        result = conn(pre_size=2, post_size=2)

        self.assertIsNotNone(result.delays)
        self.assertEqual(result.delays.unit, u.ms)  # Should get default units

    def test_excitatory_inhibitory_default_parameters(self):
        conn = ExcitatoryInhibitory(seed=42)

        result = conn(pre_size=2, post_size=2)

        # Should use default values
        expected_weights = [0.5, 0.8, -1.2, -0.3]  # Default coupling values
        actual_weights = sorted(result.weights)
        np.testing.assert_array_almost_equal(actual_weights, sorted(expected_weights))

    def test_excitatory_inhibitory_different_sizes_error(self):
        # E-I should work with 2x2 structure
        conn = ExcitatoryInhibitory(seed=42)

        # Should work with 2x2
        result = conn(pre_size=2, post_size=2)
        self.assertEqual(result.n_connections, 4)

        # Should also work with larger sizes (uses submatrix)
        result2 = conn(pre_size=3, post_size=3)
        self.assertEqual(result2.n_connections, 4)  # Still uses 2x2 submatrix


class TestHierarchicalPopulations(unittest.TestCase):
    """
    Test HierarchicalPopulations connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from braintools.conn._conn_population import HierarchicalPopulations

        # 3-level hierarchy: 4 -> 2 -> 1 populations
        hierarchy = HierarchicalPopulations(
            hierarchy_levels=[4, 2, 1],
            feedforward_strength=[0.6, 0.8],  # Level i to i+1
            feedback_strength=[0.1, 0.2],     # Level i+1 to i
            lateral_strength=0.05,
            skip_connections=False
        )

        result = hierarchy(pre_size=7, post_size=7)

        assert result.model_type == 'population_rate'
        assert result.metadata['pattern'] == 'hierarchical_populations'

        # Should have feedforward, feedback, and lateral connections
        # Feedforward: 4*2 + 2*1 = 10 connections
        # Feedback: 2*4 + 1*2 = 10 connections
        # Lateral: (4*3) + (2*1) + 0 = 14 connections (within levels)
        # Total: 10 + 10 + 14 = 34 connections
        assert result.n_connections == 34
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_hierarchical_populations(self):
        hierarchy_levels = [3, 2, 1]

        conn = HierarchicalPopulations(
            hierarchy_levels=hierarchy_levels,
            feedforward_strength=0.5,
            feedback_strength=0.1,
            lateral_strength=0.05,
            seed=42
        )

        result = conn(pre_size=6, post_size=6)

        self.assertEqual(result.model_type, 'population_rate')
        self.assertEqual(result.metadata['pattern'], 'hierarchical_populations')
        self.assertEqual(result.metadata['hierarchy_levels'], hierarchy_levels)

        # Calculate expected connections:
        # Feedforward: 3*2 + 2*1 = 8
        # Feedback: 2*3 + 1*2 = 8
        # Lateral: (3*2) + (2*1) + 0 = 8
        # Total: 8 + 8 + 8 = 24
        self.assertEqual(result.n_connections, 24)

    def test_hierarchical_different_strengths(self):
        hierarchy_levels = [2, 2, 1]
        feedforward_strengths = [0.6, 0.8]
        feedback_strengths = [0.1, 0.2]

        conn = HierarchicalPopulations(
            hierarchy_levels=hierarchy_levels,
            feedforward_strength=feedforward_strengths,
            feedback_strength=feedback_strengths,
            lateral_strength=0.0,  # No lateral connections
            seed=42
        )

        result = conn(pre_size=5, post_size=5)

        # Should have different weights for different levels
        weights = result.weights
        unique_weights = np.unique(np.round(weights, 3))
        self.assertGreaterEqual(len(unique_weights), 2)  # At least feedforward and feedback

    def test_hierarchical_with_skip_connections(self):
        hierarchy_levels = [3, 2, 1]

        conn = HierarchicalPopulations(
            hierarchy_levels=hierarchy_levels,
            feedforward_strength=0.5,
            feedback_strength=0.1,
            lateral_strength=0.0,
            skip_connections=True,
            seed=42
        )

        result = conn(pre_size=6, post_size=6)

        # Should have more connections due to skip connections
        # Feedforward: 3*2 + 2*1 = 8
        # Feedback: 2*3 + 1*2 = 8
        # Skip: 3*1 = 3 (level 0 to level 2)
        # Total: 8 + 8 + 3 = 19
        self.assertEqual(result.n_connections, 19)

    def test_hierarchical_no_lateral_connections(self):
        hierarchy_levels = [2, 1]

        conn = HierarchicalPopulations(
            hierarchy_levels=hierarchy_levels,
            feedforward_strength=0.5,
            feedback_strength=0.1,
            lateral_strength=0.0,  # No lateral
            seed=42
        )

        result = conn(pre_size=3, post_size=3)

        # Only feedforward and feedback
        # Feedforward: 2*1 = 2
        # Feedback: 1*2 = 2
        # Total: 4
        self.assertEqual(result.n_connections, 4)

    def test_hierarchical_single_level(self):
        # Single level should only have lateral connections
        hierarchy_levels = [3]

        conn = HierarchicalPopulations(
            hierarchy_levels=hierarchy_levels,
            feedforward_strength=0.5,
            feedback_strength=0.1,
            lateral_strength=0.2,
            seed=42
        )

        result = conn(pre_size=3, post_size=3)

        # Only lateral connections: 3*2 = 6 (no self-connections)
        self.assertEqual(result.n_connections, 6)

    def test_hierarchical_empty_levels(self):
        # Empty hierarchy should return empty result
        hierarchy_levels = []

        conn = HierarchicalPopulations(
            hierarchy_levels=hierarchy_levels,
            seed=42
        )

        result = conn(pre_size=0, post_size=0)

        self.assertEqual(result.n_connections, 0)

    def test_hierarchical_uniform_vs_list_strengths(self):
        hierarchy_levels = [2, 2]

        # Test with uniform strength
        conn_uniform = HierarchicalPopulations(
            hierarchy_levels=hierarchy_levels,
            feedforward_strength=0.5,
            feedback_strength=0.1,
            lateral_strength=0.0,
            seed=42
        )

        # Test with list strengths
        conn_list = HierarchicalPopulations(
            hierarchy_levels=hierarchy_levels,
            feedforward_strength=[0.5],  # Single value in list
            feedback_strength=[0.1],
            lateral_strength=0.0,
            seed=42
        )

        result_uniform = conn_uniform(pre_size=4, post_size=4)
        result_list = conn_list(pre_size=4, post_size=4)

        # Should produce same result
        self.assertEqual(result_uniform.n_connections, result_list.n_connections)
        np.testing.assert_array_almost_equal(result_uniform.weights, result_list.weights)

    def test_hierarchical_metadata(self):
        hierarchy_levels = [3, 2]
        ff_strength = [0.6]
        fb_strength = [0.15]

        conn = HierarchicalPopulations(
            hierarchy_levels=hierarchy_levels,
            feedforward_strength=ff_strength,
            feedback_strength=fb_strength,
            lateral_strength=0.05,
            skip_connections=True,
            seed=42
        )

        result = conn(pre_size=5, post_size=5)

        metadata = result.metadata
        self.assertEqual(metadata['hierarchy_levels'], hierarchy_levels)
        self.assertEqual(metadata['feedforward_strength'], ff_strength)
        self.assertEqual(metadata['feedback_strength'], fb_strength)
        self.assertEqual(metadata['lateral_strength'], 0.05)
        self.assertTrue(metadata['skip_connections'])


class TestWilsonCowanNetwork(unittest.TestCase):
    """
    Test WilsonCowanNetwork connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_population import WilsonCowanNetwork

        # Classic Wilson-Cowan parameters
        wc = WilsonCowanNetwork(
            w_ee=1.25,
            w_ei=1.0,
            w_ie=-1.0,
            w_ii=-0.75,
            tau_e=10 * u.ms,
            tau_i=20 * u.ms
        )

        result = wc(pre_size=2, post_size=2)

        assert result.model_type == 'population_rate'
        assert result.metadata['pattern'] == 'wilson_cowan'
        assert result.n_connections == 4  # 2x2 coupling matrix

        # Check Wilson-Cowan structure: E self-excitation, I->E inhibition
        weights = result.weights
        assert 1.25 in weights  # w_ee
        assert -1.0 in weights  # w_ie
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_wilson_cowan(self):
        conn = WilsonCowanNetwork(
            w_ee=1.25,
            w_ei=1.0,
            w_ie=-1.0,
            w_ii=-0.75,
            tau_e=10 * u.ms,
            tau_i=20 * u.ms,
            seed=42
        )

        result = conn(pre_size=2, post_size=2)

        self.assertEqual(result.model_type, 'population_rate')
        self.assertEqual(result.metadata['pattern'], 'wilson_cowan')
        self.assertEqual(result.n_connections, 4)

        # Check weights match Wilson-Cowan parameters
        expected_weights = [1.25, 1.0, -1.0, -0.75]
        actual_weights = sorted(result.weights)
        np.testing.assert_array_almost_equal(actual_weights, sorted(expected_weights))

    def test_wilson_cowan_with_time_constants(self):
        tau_e = 15 * u.ms
        tau_i = 25 * u.ms

        conn = WilsonCowanNetwork(
            w_ee=1.0,
            w_ei=0.8,
            w_ie=-0.9,
            w_ii=-0.6,
            tau_e=tau_e,
            tau_i=tau_i,
            seed=42
        )

        result = conn(pre_size=2, post_size=2)

        self.assertIsNotNone(result.delays)
        self.assertEqual(result.delays.unit, u.ms)

        # Check that delays match time constants of target populations
        expected_delays = [15, 25, 15, 25]  # tau_e, tau_i, tau_e, tau_i
        actual_delays = sorted(u.get_magnitude(result.delays))
        np.testing.assert_array_almost_equal(actual_delays, sorted(expected_delays))

    def test_wilson_cowan_default_parameters(self):
        conn = WilsonCowanNetwork(seed=42)

        result = conn(pre_size=2, post_size=2)

        # Should use default Wilson-Cowan parameters
        expected_weights = [1.25, 1.0, -1.0, -0.75]
        actual_weights = sorted(result.weights)
        np.testing.assert_array_almost_equal(actual_weights, sorted(expected_weights))

    def test_wilson_cowan_scalar_time_constants(self):
        conn = WilsonCowanNetwork(
            tau_e=12.0,  # Scalar
            tau_i=18.0,  # Scalar
            seed=42
        )

        result = conn(pre_size=2, post_size=2)

        self.assertIsNotNone(result.delays)
        self.assertEqual(result.delays.unit, u.ms)


class TestSpecializedPatterns(unittest.TestCase):
    """
    Test specialized population patterns.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_population import (
            FeedforwardInhibition, RecurrentAmplification, CompetitiveNetwork,
            PopulationDistance, RateDependent, FiringRateNetworks
        )

        # Feedforward inhibition
        ffi = FeedforwardInhibition(
            exc_to_exc=0.8,
            exc_to_inh=1.2,
            inh_to_exc=-1.5
        )

        # Competitive network
        comp = CompetitiveNetwork(
            self_excitation=1.0,
            lateral_inhibition=-0.8
        )

        # Distance-dependent populations
        positions = np.random.uniform(0, 100, (5, 2)) * u.um
        dist_conn = PopulationDistance(
            sigma=50 * u.um,
            decay_function='gaussian'
        )
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_feedforward_inhibition(self):
        conn = FeedforwardInhibition(
            exc_to_exc=0.8,
            exc_to_inh=1.2,
            inh_to_exc=-1.5,
            seed=42
        )

        result = conn(pre_size=2, post_size=2)

        self.assertEqual(result.model_type, 'population_rate')
        # Should have 3 connections: E->E, E->I, I->E (no I->I)
        self.assertEqual(result.n_connections, 2)

        expected_weights = [0.8, 1.2, -1.5]
        actual_weights = sorted(result.weights)
        np.testing.assert_array_almost_equal(actual_weights, sorted(expected_weights))

    def test_recurrent_amplification(self):
        conn = RecurrentAmplification(
            self_coupling=1.5,
            cross_coupling=0.3,
            seed=42
        )

        result = conn(pre_size=3, post_size=3)

        self.assertEqual(result.model_type, 'population_rate')
        self.assertEqual(result.n_connections, 9)  # 3x3 full matrix

        # Check that diagonal elements have self_coupling strength
        # and off-diagonal have cross_coupling strength
        weights = result.weights
        self.assertTrue(np.any(np.isclose(weights, 1.5)))  # Self-coupling
        self.assertTrue(np.any(np.isclose(weights, 0.3)))  # Cross-coupling

    def test_competitive_network(self):
        conn = CompetitiveNetwork(
            self_excitation=1.0,
            lateral_inhibition=-0.8,
            seed=42
        )

        result = conn(pre_size=4, post_size=4)

        self.assertEqual(result.model_type, 'population_rate')
        self.assertEqual(result.n_connections, 16)  # 4x4 full matrix

        # Should have self-excitation on diagonal, lateral inhibition off-diagonal
        weights = result.weights
        self.assertTrue(np.any(np.isclose(weights, 1.0)))   # Self-excitation
        self.assertTrue(np.any(np.isclose(weights, -0.8)))  # Lateral inhibition

    def test_population_distance_gaussian(self):
        positions = np.array([[0, 0], [50, 0], [100, 0]]) * u.um

        conn = PopulationDistance(
            sigma=30 * u.um,
            decay_function='gaussian',
            seed=42
        )

        result = conn(
            pre_size=3, post_size=3,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.model_type, 'population_rate')
        self.assertEqual(result.metadata['pattern'], 'population_distance')
        self.assertGreater(result.n_connections, 0)

        # Closer populations should have stronger connections
        # (Exact test depends on threshold and distance calculation)

    def test_population_distance_exponential(self):
        positions = np.array([[0, 0], [25, 25]]) * u.um

        conn = PopulationDistance(
            sigma=40 * u.um,
            decay_function='exponential',
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.metadata['sigma'], 40 * u.um)
        self.assertGreater(result.n_connections, 0)

    def test_population_distance_no_positions_error(self):
        conn = PopulationDistance(sigma=50 * u.um, seed=42)

        with self.assertRaises(ValueError):
            conn(pre_size=3, post_size=3)  # No positions provided

    def test_population_distance_invalid_decay_function(self):
        with self.assertRaises(ValueError):
            conn = PopulationDistance(sigma=50 * u.um, decay_function='invalid')
            positions = np.array([[0, 0]]) * u.um
            conn(pre_size=1, post_size=1, pre_positions=positions, post_positions=positions)

    def test_population_distance_scalar_sigma(self):
        positions = np.array([[0, 0], [30, 0]])  # No units

        conn = PopulationDistance(
            sigma=20.0,  # Scalar, no units
            decay_function='gaussian',
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertGreater(result.n_connections, 0)

    def test_rate_dependent(self):
        def rate_function(rate):
            return rate ** 2

        base_pattern = AllToAllPopulations(weight=ConstantWeight(0.5))

        conn = RateDependent(
            base_pattern=base_pattern,
            rate_function=rate_function,
            seed=42
        )

        result = conn(pre_size=2, post_size=2)

        self.assertTrue(result.metadata['rate_dependent'])
        self.assertEqual(result.metadata['rate_function'], rate_function)
        # Should have same structure as base pattern
        self.assertEqual(result.n_connections, 4)

    def test_firing_rate_networks(self):
        conn = FiringRateNetworks(field_strength=0.15, seed=42)

        result = conn(pre_size=3, post_size=4)

        self.assertEqual(result.model_type, 'population_rate')
        # Should delegate to MeanField
        self.assertEqual(result.n_connections, 12)  # 3 * 4


class TestSimplePatterns(unittest.TestCase):
    """
    Test simple population patterns.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from braintools.conn._conn_population import AllToAllPopulations, RandomPopulations

        # All-to-all populations
        all_to_all = AllToAllPopulations(weight=ConstantWeight(0.5))
        result = all_to_all(pre_size=3, post_size=4)
        assert result.n_connections == 12  # 3 * 4

        # Random populations
        random_pop = RandomPopulations(prob=0.6, weight=ConstantWeight(0.3))
        result_random = random_pop(pre_size=5, post_size=5)
        # Should have approximately 60% of total connections
        expected = int(0.6 * 5 * 5)
        assert abs(result_random.n_connections - expected) <= 3
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_all_to_all_populations(self):
        weight_init = ConstantWeight(0.7)

        conn = AllToAllPopulations(weight=weight_init, seed=42)
        result = conn(pre_size=3, post_size=4)

        self.assertEqual(result.model_type, 'population_rate')
        self.assertEqual(result.metadata['pattern'], 'all_to_all_populations')
        self.assertEqual(result.n_connections, 12)  # 3 * 4

        # All weights should be 0.7
        np.testing.assert_array_almost_equal(u.get_magnitude(result.weights), 0.7)

    def test_all_to_all_populations_no_weights(self):
        conn = AllToAllPopulations(seed=42)
        result = conn(pre_size=2, post_size=3)

        self.assertEqual(result.n_connections, 6)
        self.assertIsNone(result.weights)

    def test_random_populations(self):
        weight_init = ConstantWeight(0.4)

        conn = RandomPopulations(prob=0.5, weight=weight_init, seed=42)
        result = conn(pre_size=6, post_size=6)

        self.assertEqual(result.model_type, 'population_rate')
        self.assertEqual(result.metadata['pattern'], 'random_populations')
        self.assertEqual(result.metadata['prob'], 0.5)

        # Should have approximately 50% of total connections
        total_possible = 6 * 6
        expected_connections = int(total_possible * 0.5)
        self.assertAlmostEqual(result.n_connections, expected_connections, delta=3)

        if result.n_connections > 0:
            np.testing.assert_array_almost_equal(u.get_magnitude(result.weights), 0.4)

    def test_random_populations_zero_probability(self):
        conn = RandomPopulations(prob=0.0, seed=42)
        result = conn(pre_size=5, post_size=5)

        self.assertEqual(result.n_connections, 0)

    def test_random_populations_high_probability(self):
        conn = RandomPopulations(prob=0.9, seed=42)
        result = conn(pre_size=4, post_size=4)

        # Should have most connections
        total_possible = 4 * 4
        expected_connections = int(total_possible * 0.9)
        self.assertAlmostEqual(result.n_connections, expected_connections, delta=2)


class TestHierarchyAliases(unittest.TestCase):
    """
    Test hierarchy alias patterns.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from braintools.conn._conn_population import (
            FeedforwardHierarchy, RecurrentHierarchy, LayeredNetwork
        )

        # Feedforward hierarchy (no feedback)
        ff_hierarchy = FeedforwardHierarchy(
            hierarchy_levels=[4, 2, 1],
            feedforward_strength=0.6
        )

        # Recurrent hierarchy (with feedback)
        rec_hierarchy = RecurrentHierarchy(
            hierarchy_levels=[4, 2, 1],
            feedforward_strength=0.5,
            feedback_strength=0.2
        )

        # Layered network (uniform layer sizes)
        layered = LayeredNetwork(
            n_layers=3,
            populations_per_layer=2,
            feedforward_strength=0.4
        )
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_feedforward_hierarchy(self):
        conn = FeedforwardHierarchy(
            hierarchy_levels=[3, 2, 1],
            feedforward_strength=0.6,
            seed=42
        )

        result = conn(pre_size=6, post_size=6)

        self.assertEqual(result.model_type, 'population_rate')
        # Should only have feedforward connections (no feedback or lateral)
        # Feedforward: 3*2 + 2*1 = 8
        self.assertEqual(result.n_connections, 8)

        # All weights should be feedforward strength
        np.testing.assert_array_almost_equal(result.weights, 0.6)

    def test_recurrent_hierarchy(self):
        conn = RecurrentHierarchy(
            hierarchy_levels=[2, 2, 1],
            feedforward_strength=0.5,
            feedback_strength=0.2,
            seed=42
        )

        result = conn(pre_size=5, post_size=5)

        self.assertEqual(result.model_type, 'population_rate')
        # Should have feedforward, feedback, and lateral connections
        # Feedforward: 2*2 + 2*1 = 6
        # Feedback: 2*2 + 1*2 = 6
        # Lateral: (2*1) + (2*1) + 0 = 4
        # Total: 6 + 6 + 4 = 16
        self.assertEqual(result.n_connections, 16)

        # Should have different weight values
        unique_weights = np.unique(np.round(result.weights, 3))
        self.assertGreaterEqual(len(unique_weights), 2)

    def test_layered_network(self):
        conn = LayeredNetwork(
            n_layers=3,
            populations_per_layer=2,
            feedforward_strength=0.4,
            seed=42
        )

        result = conn(pre_size=6, post_size=6)

        self.assertEqual(result.model_type, 'population_rate')
        # Should create hierarchy [2, 2, 2] with feedforward and lateral connections
        # Feedforward: 2*2 + 2*2 = 8
        # Lateral: (2*1) + (2*1) + (2*1) = 6
        # Total: 8 + 6 = 14
        self.assertEqual(result.n_connections, 14)


class TestCustomPopulation(unittest.TestCase):
    """
    Test CustomPopulation connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from braintools.conn._conn_population import CustomPopulation

        def my_population_func(pre_size, post_size, **kwargs):
            # Custom population connectivity logic
            pre_indices = [0, 1, 0]
            post_indices = [1, 0, 0]
            weights = [0.5, 0.3, 0.8]
            return pre_indices, post_indices, weights

        conn = CustomPopulation(connection_func=my_population_func)
        result = conn(pre_size=3, post_size=2)

        assert result.model_type == 'population_rate'
        assert result.metadata['pattern'] == 'custom_population'
        assert result.n_connections == 3
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_custom_population_basic(self):
        def simple_func(pre_size, post_size, **kwargs):
            pre_indices = [0, 1]
            post_indices = [1, 0]
            weights = [0.5, 0.8]
            return pre_indices, post_indices, weights

        conn = CustomPopulation(connection_func=simple_func, seed=42)
        result = conn(pre_size=3, post_size=2)

        self.assertEqual(result.model_type, 'population_rate')
        self.assertEqual(result.metadata['pattern'], 'custom_population')
        self.assertEqual(result.n_connections, 2)

        np.testing.assert_array_equal(result.pre_indices, [0, 1])
        np.testing.assert_array_equal(result.post_indices, [1, 0])
        np.testing.assert_array_almost_equal(result.weights, [0.5, 0.8])

    def test_custom_population_with_kwargs(self):
        def func_with_kwargs(pre_size, post_size, custom_param=0.1, **kwargs):
            n_connections = min(pre_size, post_size)
            pre_indices = list(range(n_connections))
            post_indices = list(range(n_connections))
            weights = [custom_param] * n_connections
            return pre_indices, post_indices, weights

        conn = CustomPopulation(connection_func=func_with_kwargs, seed=42)
        result = conn(pre_size=4, post_size=3, custom_param=0.7)

        self.assertEqual(result.n_connections, 3)  # min(4, 3)
        np.testing.assert_array_almost_equal(result.weights, [0.7, 0.7, 0.7])

    def test_custom_population_empty_connections(self):
        def empty_func(pre_size, post_size, **kwargs):
            return [], [], []

        conn = CustomPopulation(connection_func=empty_func, seed=42)
        result = conn(pre_size=5, post_size=5)

        self.assertEqual(result.n_connections, 0)

    def test_custom_population_with_positions(self):
        def func_with_positions(pre_size, post_size, pre_positions=None, post_positions=None, **kwargs):
            pre_indices = [0]
            post_indices = [0]
            weights = [1.0]
            return pre_indices, post_indices, weights

        positions = np.array([[0, 0], [10, 10]]) * u.um

        conn = CustomPopulation(connection_func=func_with_positions, seed=42)
        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertIsNotNone(result.pre_positions)
        self.assertIsNotNone(result.post_positions)


class TestEdgeCases(unittest.TestCase):
    """
    Test edge cases and error conditions.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from braintools.conn._conn_population import PopulationCoupling, MeanField

        # Empty coupling matrix
        empty_coupling = np.zeros((2, 2))
        conn = PopulationCoupling(coupling_matrix=empty_coupling)
        result = conn(pre_size=2, post_size=2)
        assert result.n_connections == 0

        # Single population
        single_coupling = np.array([[0.5]])
        conn_single = PopulationCoupling(coupling_matrix=single_coupling)
        result_single = conn_single(pre_size=1, post_size=1)
        assert result_single.n_connections == 1

        # Very sparse mean-field
        sparse_mf = MeanField(connectivity_fraction=0.01)
        result_sparse = sparse_mf(pre_size=100, post_size=100)
        # Should have approximately 1% of connections
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_empty_coupling_matrix(self):
        # All-zero coupling matrix
        coupling_matrix = np.zeros((3, 3))

        conn = PopulationCoupling(coupling_matrix=coupling_matrix, seed=42)
        result = conn(pre_size=3, post_size=3)

        self.assertEqual(result.n_connections, 0)

    def test_single_population_networks(self):
        patterns = [
            PopulationCoupling(coupling_matrix=np.array([[0.5]])),
            MeanField(field_strength=0.1),
            AllToAllPopulations(),
            RandomPopulations(prob=1.0),
        ]

        for pattern in patterns:
            pattern.seed = 42
            result = pattern(pre_size=1, post_size=1)
            if hasattr(pattern, 'prob') and pattern.prob == 1.0:
                self.assertEqual(result.n_connections, 1)
            elif isinstance(pattern, PopulationCoupling):
                self.assertEqual(result.n_connections, 1)
            elif isinstance(pattern, (MeanField, AllToAllPopulations)):
                self.assertEqual(result.n_connections, 1)

    def test_large_population_networks(self):
        # Test with reasonably large population networks
        conn = MeanField(
            field_strength=0.01,
            connectivity_fraction=0.05,  # Sparse
            seed=42
        )

        result = conn(pre_size=100, post_size=100)

        self.assertEqual(result.shape, (100, 100))
        # Should have about 5% of total connections
        expected_connections = int(0.05 * 100 * 100)
        self.assertAlmostEqual(result.n_connections, expected_connections, delta=50)

    def test_asymmetric_population_sizes(self):
        patterns = [
            PopulationCoupling(coupling_matrix=np.array([[0.5, 0.3], [0.2, 0.8]])),
            MeanField(field_strength=0.1),
            AllToAllPopulations(),
            RandomPopulations(prob=0.3),
        ]

        for pattern in patterns:
            pattern.seed = 42
            result = pattern(pre_size=3, post_size=5)

            self.assertEqual(result.shape, (3, 5))
            self.assertTrue(np.all(result.pre_indices < 3))
            self.assertTrue(np.all(result.post_indices < 5))

    def test_tuple_sizes_consistency(self):
        conn = MeanField(field_strength=0.1, seed=42)

        result = conn(pre_size=(2, 3), post_size=(1, 5))

        self.assertEqual(result.pre_size, (2, 3))
        self.assertEqual(result.post_size, (1, 5))
        self.assertEqual(result.shape, (6, 5))

    def test_reproducibility_with_seeds(self):
        # Test that same seed produces same results
        conn1 = RandomPopulations(prob=0.4, seed=42)
        result1 = conn1(pre_size=20, post_size=20)

        conn2 = RandomPopulations(prob=0.4, seed=42)
        result2 = conn2(pre_size=20, post_size=20)

        # Should have identical connections
        np.testing.assert_array_equal(result1.pre_indices, result2.pre_indices)
        np.testing.assert_array_equal(result1.post_indices, result2.post_indices)

    def test_different_seeds_produce_different_results(self):
        conn1 = RandomPopulations(prob=0.5, seed=42)
        result1 = conn1(pre_size=30, post_size=30)

        conn2 = RandomPopulations(prob=0.5, seed=123)
        result2 = conn2(pre_size=30, post_size=30)

        # Should have different connections (very unlikely to be identical)
        self.assertFalse(np.array_equal(result1.pre_indices, result2.pre_indices) and
                         np.array_equal(result1.post_indices, result2.post_indices))

    def test_unit_handling_consistency(self):
        # Test that units are handled consistently
        time_constants = [15 * u.ms, 25 * u.ms]
        coupling_matrix = np.array([[0.5, 0.8], [0.3, 0.2]])

        conn = PopulationCoupling(
            coupling_matrix=coupling_matrix,
            time_constants=time_constants,
            seed=42
        )

        result = conn(pre_size=2, post_size=2)

        if result.delays is not None:
            self.assertEqual(result.delays.unit, u.ms)

    def test_metadata_preservation(self):
        # Test that metadata is properly preserved across different patterns
        patterns = [
            PopulationCoupling(coupling_matrix=np.array([[0.5]])),
            MeanField(field_strength=0.1, normalization='sqrt'),
            ExcitatoryInhibitory(),
            HierarchicalPopulations(hierarchy_levels=[2, 1]),
        ]

        for pattern in patterns:
            pattern.seed = 42
            result = pattern(pre_size=2, post_size=2)

            self.assertIn('pattern', result.metadata)
            self.assertEqual(result.model_type, 'population_rate')

    def test_weight_initialization_edge_cases(self):
        # Test with different weight initialization scenarios
        weight_init = ConstantWeight(0.0)  # Zero weights

        conn = AllToAllPopulations(weight=weight_init, seed=42)
        result = conn(pre_size=3, post_size=3)

        if result.n_connections > 0:
            np.testing.assert_array_almost_equal(u.get_magnitude(result.weights), 0.0)


if __name__ == '__main__':
    unittest.main()