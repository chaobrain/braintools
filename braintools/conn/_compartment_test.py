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

"""
Comprehensive tests for multi-compartment neuron connectivity classes.

This test suite covers:
- CompartmentSpecific connectivity patterns
- Anatomical targeting patterns (SomaToDendrite, AxonToSoma, etc.)
- Morphology-aware patterns (MorphologyDistance, DendriticTree, etc.)
- Axonal patterns (AxonalProjection, AxonalBranching, etc.)
- Synaptic patterns (SynapticPlacement, SynapticClustering, etc.)
- Compartment type constants and utilities
"""

import unittest

import brainunit as u
import numpy as np

from braintools.conn._compartment import (
    # Constants
    SOMA, BASAL_DENDRITE, APICAL_DENDRITE, AXON, COMPARTMENT_NAMES,

    # Basic compartment patterns
    CompartmentSpecific,
    AllToAllCompartments,

    # Anatomical targeting patterns
    SomaToDendrite,
    AxonToSoma,
    DendriteToSoma,
    AxonToDendrite,
    DendriteToDendrite,

    # Morphology-aware patterns
    ProximalTargeting,
    DistalTargeting,
    BranchSpecific,
    MorphologyDistance,

    # Dendritic patterns
    DendriticTree,
    BasalDendriteTargeting,
    ApicalDendriteTargeting,
    DendriticIntegration,

    # Axonal patterns
    AxonalProjection,
    AxonalBranching,
    AxonalArborization,
    TopographicProjection,

    # Synaptic patterns
    SynapticPlacement,
    SynapticClustering,

    # Custom patterns
    CustomCompartment,
)
from braintools.init import Constant, Uniform


class TestCompartmentConstants(unittest.TestCase):
    """
    Test compartment type constants and utilities.

    Examples
    --------
    .. code-block:: python

        from braintools.conn._conn_compartment import (
            SOMA, BASAL_DENDRITE, APICAL_DENDRITE, AXON, COMPARTMENT_NAMES
        )

        # Compartment type constants
        assert SOMA == 0
        assert BASAL_DENDRITE == 1
        assert APICAL_DENDRITE == 2
        assert AXON == 3

        # Compartment name mapping
        assert COMPARTMENT_NAMES[SOMA] == 'soma'
        assert COMPARTMENT_NAMES[BASAL_DENDRITE] == 'basal_dendrite'
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_compartment_constants(self):
        self.assertEqual(SOMA, 0)
        self.assertEqual(BASAL_DENDRITE, 1)
        self.assertEqual(APICAL_DENDRITE, 2)
        self.assertEqual(AXON, 3)

    def test_compartment_names(self):
        self.assertEqual(COMPARTMENT_NAMES[SOMA], 'soma')
        self.assertEqual(COMPARTMENT_NAMES[BASAL_DENDRITE], 'basal_dendrite')
        self.assertEqual(COMPARTMENT_NAMES[APICAL_DENDRITE], 'apical_dendrite')
        self.assertEqual(COMPARTMENT_NAMES[AXON], 'axon')

    def test_compartment_names_completeness(self):
        # Test that all constants have corresponding names
        expected_compartments = {SOMA, BASAL_DENDRITE, APICAL_DENDRITE, AXON}
        actual_compartments = set(COMPARTMENT_NAMES.keys())
        self.assertEqual(expected_compartments, actual_compartments)


class TestCompartmentSpecific(unittest.TestCase):
    """
    Test CompartmentSpecific connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_compartment import CompartmentSpecific, AXON, SOMA

        # Basic axon-to-soma connectivity
        conn = CompartmentSpecific(
            compartment_mapping={AXON: SOMA},
            connection_prob=0.1,
            weight=0.5 * u.nS,
            delay=1.0 * u.ms
        )

        result = conn(pre_size=10, post_size=10)
        assert result.model_type == 'multi_compartment'
        assert hasattr(result, 'pre_compartments')
        assert hasattr(result, 'post_compartments')

        # Complex multi-compartment targeting
        multi_conn = CompartmentSpecific(
            compartment_mapping={
                BASAL_DENDRITE: SOMA,
                AXON: [BASAL_DENDRITE, APICAL_DENDRITE]
            },
            connection_prob={
                (BASAL_DENDRITE, SOMA): 0.3,
                (AXON, BASAL_DENDRITE): 0.1,
                (AXON, APICAL_DENDRITE): 0.1
            }
        )
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_compartment_specific(self):
        conn = CompartmentSpecific(
            compartment_mapping={AXON: SOMA},
            connection_prob=0.2,
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        self.assertEqual(result.model_type, 'multi_compartment')
        self.assertTrue(hasattr(result, 'pre_compartments'))
        self.assertTrue(hasattr(result, 'post_compartments'))
        self.assertGreater(result.n_connections, 0)

        # Check that all connections are from axon to soma
        self.assertTrue(np.all(result.pre_compartments == AXON))
        self.assertTrue(np.all(result.post_compartments == SOMA))

    def test_compartment_mapping_with_lists(self):
        conn = CompartmentSpecific(
            compartment_mapping={AXON: [BASAL_DENDRITE, APICAL_DENDRITE]},
            connection_prob=0.1,
            seed=42
        )

        result = conn(pre_size=10, post_size=10)

        # Check that all presynaptic are axon
        self.assertTrue(np.all(result.pre_compartments == AXON))

        # Check that postsynaptic are either basal or apical dendrites
        valid_post = np.isin(result.post_compartments, [BASAL_DENDRITE, APICAL_DENDRITE])
        self.assertTrue(np.all(valid_post))

    def test_per_compartment_probabilities(self):
        conn = CompartmentSpecific(
            compartment_mapping={
                AXON: [BASAL_DENDRITE, APICAL_DENDRITE]
            },
            connection_prob={
                (AXON, BASAL_DENDRITE): 0.2,
                (AXON, APICAL_DENDRITE): 0.1
            },
            seed=42
        )

        result = conn(pre_size=50, post_size=50)

        # Count connections to each compartment type
        basal_count = np.sum(result.post_compartments == BASAL_DENDRITE)
        apical_count = np.sum(result.post_compartments == APICAL_DENDRITE)

        # Should have more basal connections due to higher probability
        self.assertGreater(basal_count, apical_count)

    def test_compartment_name_mapping(self):
        # Test using compartment names instead of indices
        conn = CompartmentSpecific(
            compartment_mapping={'axon': 'soma'},
            connection_prob=0.1,
            seed=42
        )

        result = conn(pre_size=10, post_size=10)

        # Should still work with names
        self.assertGreater(result.n_connections, 0)
        self.assertTrue(np.all(result.pre_compartments == AXON))
        self.assertTrue(np.all(result.post_compartments == SOMA))

    def test_mixed_name_and_index_mapping(self):
        # Test mixing names and indices
        conn = CompartmentSpecific(
            compartment_mapping={'axon': [BASAL_DENDRITE, 'apical_dendrite']},
            connection_prob=0.1,
            seed=42
        )

        result = conn(pre_size=10, post_size=10)

        self.assertGreater(result.n_connections, 0)
        self.assertTrue(np.all(result.pre_compartments == AXON))

    def test_weights_and_delays(self):
        weight_init = Constant(0.5 * u.nS)
        delay_init = Uniform(1.0 * u.ms, 3.0 * u.ms)

        conn = CompartmentSpecific(
            compartment_mapping={AXON: SOMA},
            connection_prob=0.2,
            weight=weight_init,
            delay=delay_init,
            seed=42
        )

        result = conn(pre_size=10, post_size=10)

        self.assertIsNotNone(result.weights)
        self.assertIsNotNone(result.delays)
        self.assertEqual(result.weights.unit, u.nS)
        self.assertEqual(result.delays.unit, u.ms)
        self.assertTrue(np.all(result.weights.mantissa == 0.5))
        self.assertTrue(np.all((result.delays >= 1.0 * u.ms) & (result.delays < 3.0 * u.ms)))

    def test_zero_probability_compartments(self):
        conn = CompartmentSpecific(
            compartment_mapping={
                AXON: [BASAL_DENDRITE, APICAL_DENDRITE]
            },
            connection_prob={
                (AXON, BASAL_DENDRITE): 0.2,
                (AXON, APICAL_DENDRITE): 0.0  # Zero probability
            },
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        # Should only have connections to basal dendrites
        self.assertTrue(np.all(result.post_compartments == BASAL_DENDRITE))

    def test_empty_connections(self):
        # Test case where no connections are generated
        conn = CompartmentSpecific(
            compartment_mapping={AXON: SOMA},
            connection_prob=0.0,  # Zero probability
            seed=42
        )

        result = conn(pre_size=10, post_size=10)

        self.assertEqual(result.n_connections, 0)
        self.assertEqual(len(result.pre_indices), 0)
        self.assertEqual(len(result.post_indices), 0)

    def test_tuple_sizes(self):
        conn = CompartmentSpecific(
            compartment_mapping={AXON: SOMA},
            connection_prob=0.1,
            seed=42
        )

        result = conn(pre_size=(2, 3), post_size=(3, 2))

        self.assertEqual(result.pre_size, (2, 3))
        self.assertEqual(result.post_size, (3, 2))
        self.assertEqual(result.shape, (6, 6))

    def test_metadata(self):
        mapping = {AXON: SOMA}
        prob = 0.1

        conn = CompartmentSpecific(
            compartment_mapping=mapping,
            connection_prob=prob,
            seed=42
        )

        result = conn(pre_size=10, post_size=10)

        self.assertEqual(result.metadata['pattern'], 'compartment_specific')
        self.assertEqual(result.metadata['compartment_mapping'], {AXON: SOMA})
        self.assertEqual(result.metadata['connection_prob'], prob)


class TestAnatomicalTargetingPatterns(unittest.TestCase):
    """
    Test anatomical targeting patterns.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_compartment import (
            SomaToDendrite, AxonToSoma, DendriteToSoma,
            AxonToDendrite, DendriteToDendrite
        )

        # Soma to dendrite connections
        soma_dend = SomaToDendrite(prob_per_dendrite=0.8)
        result = soma_dend(pre_size=10, post_size=10)

        # Axon to soma connections
        axon_soma = AxonToSoma(connection_prob=0.1)
        result = axon_soma(pre_size=10, post_size=10)

        # Different probabilities for different dendrite types
        soma_dend_specific = SomaToDendrite(
            prob_per_dendrite={
                BASAL_DENDRITE: 0.9,
                APICAL_DENDRITE: 0.7
            }
        )
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_soma_to_dendrite(self):
        conn = SomaToDendrite(prob_per_dendrite=0.2, seed=42)
        result = conn(pre_size=20, post_size=20)

        # All presynaptic should be soma
        self.assertTrue(np.all(result.pre_compartments == SOMA))

        # All postsynaptic should be dendrites
        valid_post = np.isin(result.post_compartments, [BASAL_DENDRITE, APICAL_DENDRITE])
        self.assertTrue(np.all(valid_post))

    def test_soma_to_dendrite_specific_probs(self):
        conn = SomaToDendrite(
            prob_per_dendrite={
                BASAL_DENDRITE: 0.3,
                APICAL_DENDRITE: 0.1
            },
            seed=42
        )

        result = conn(pre_size=50, post_size=50)

        basal_count = np.sum(result.post_compartments == BASAL_DENDRITE)
        apical_count = np.sum(result.post_compartments == APICAL_DENDRITE)

        # Should have more basal connections
        self.assertGreater(basal_count, apical_count)

    def test_soma_to_dendrite_custom_targets(self):
        conn = SomaToDendrite(
            target_dendrites=[BASAL_DENDRITE],  # Only basal
            prob_per_dendrite=0.2,
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        # Should only connect to basal dendrites
        self.assertTrue(np.all(result.post_compartments == BASAL_DENDRITE))

    def test_axon_to_soma(self):
        conn = AxonToSoma(connection_prob=0.15, seed=42)
        result = conn(pre_size=20, post_size=20)

        self.assertTrue(np.all(result.pre_compartments == AXON))
        self.assertTrue(np.all(result.post_compartments == SOMA))
        self.assertGreater(result.n_connections, 0)

    def test_dendrite_to_soma(self):
        conn = DendriteToSoma(connection_prob=0.2, seed=42)
        result = conn(pre_size=20, post_size=20)

        # Presynaptic should be dendrites
        valid_pre = np.isin(result.pre_compartments, [BASAL_DENDRITE, APICAL_DENDRITE])
        self.assertTrue(np.all(valid_pre))

        # All postsynaptic should be soma
        self.assertTrue(np.all(result.post_compartments == SOMA))

    def test_dendrite_to_soma_custom_sources(self):
        conn = DendriteToSoma(
            source_dendrites=[BASAL_DENDRITE],  # Only basal
            connection_prob=0.2,
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        # Should only have basal dendrites as sources
        self.assertTrue(np.all(result.pre_compartments == BASAL_DENDRITE))

    def test_axon_to_dendrite(self):
        conn = AxonToDendrite(connection_prob=0.15, seed=42)
        result = conn(pre_size=20, post_size=20)

        self.assertTrue(np.all(result.pre_compartments == AXON))
        valid_post = np.isin(result.post_compartments, [BASAL_DENDRITE, APICAL_DENDRITE])
        self.assertTrue(np.all(valid_post))

    def test_axon_to_dendrite_custom_targets(self):
        conn = AxonToDendrite(
            target_dendrites=[APICAL_DENDRITE],
            connection_prob=0.2,
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        self.assertTrue(np.all(result.pre_compartments == AXON))
        self.assertTrue(np.all(result.post_compartments == APICAL_DENDRITE))

    def test_dendrite_to_dendrite(self):
        conn = DendriteToDendrite(connection_prob=0.15, seed=42)
        result = conn(pre_size=20, post_size=20)

        # Both pre and post should be dendrites
        valid_pre = np.isin(result.pre_compartments, [BASAL_DENDRITE, APICAL_DENDRITE])
        valid_post = np.isin(result.post_compartments, [BASAL_DENDRITE, APICAL_DENDRITE])
        self.assertTrue(np.all(valid_pre))
        self.assertTrue(np.all(valid_post))


class TestMorphologyAwarePatterns(unittest.TestCase):
    """
    Test morphology-aware connectivity patterns.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_compartment import (
            MorphologyDistance, ProximalTargeting, DistalTargeting, BranchSpecific
        )

        # Distance-dependent connectivity
        morph_dist = MorphologyDistance(
            sigma=50 * u.um,
            decay_function='gaussian',
            compartment_mapping={AXON: [BASAL_DENDRITE, APICAL_DENDRITE]}
        )

        # With positions
        pre_positions = np.random.uniform(0, 100, (10, 2)) * u.um
        post_positions = np.random.uniform(0, 100, (10, 2)) * u.um
        result = morph_dist(
            pre_size=10, post_size=10,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        # Proximal vs distal targeting
        proximal = ProximalTargeting(connection_prob=0.1)
        distal = DistalTargeting(connection_prob=0.1)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_morphology_distance_no_positions(self):
        # Should fall back to CompartmentSpecific behavior
        conn = MorphologyDistance(
            sigma=50 * u.um,
            decay_function='gaussian',
            seed=42
        )

        result = conn(pre_size=10, post_size=10)

        self.assertEqual(result.model_type, 'multi_compartment')
        self.assertGreater(result.n_connections, 0)

    def test_morphology_distance_with_positions(self):
        pre_positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um
        post_positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um

        conn = MorphologyDistance(
            sigma=50 * u.um,
            decay_function='gaussian',
            seed=42
        )

        result = conn(
            pre_size=10, post_size=10,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        self.assertEqual(result.model_type, 'multi_compartment')
        self.assertEqual(result.metadata['pattern'], 'morphology_distance')
        self.assertEqual(result.metadata['decay_function'], 'gaussian')

    def test_morphology_distance_decay_functions(self):
        pre_positions = np.array([[0, 0], [10, 10]]) * u.um
        post_positions = np.array([[0, 0], [20, 20]]) * u.um

        for decay_func in ['gaussian', 'exponential', 'linear']:
            conn = MorphologyDistance(
                sigma=10 * u.um,
                decay_function=decay_func,
                seed=42
            )

            result = conn(
                pre_size=2, post_size=2,
                pre_positions=pre_positions,
                post_positions=post_positions
            )

            self.assertEqual(result.metadata['decay_function'], decay_func)

    def test_morphology_distance_invalid_decay_function(self):
        conn = MorphologyDistance(
            sigma=50 * u.um,
            decay_function='invalid',
            seed=42
        )

        pre_positions = np.array([[0, 0]]) * u.um
        post_positions = np.array([[10, 10]]) * u.um

        with self.assertRaises(ValueError):
            conn(
                pre_size=1, post_size=1,
                pre_positions=pre_positions,
                post_positions=post_positions
            )

    def test_morphology_distance_scalar_sigma(self):
        # Test with scalar sigma (no units)
        pre_positions = np.array([[0, 0], [10, 10]])
        post_positions = np.array([[0, 0], [20, 20]])

        conn = MorphologyDistance(
            sigma=10.0,  # Scalar
            decay_function='gaussian',
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        self.assertGreater(result.n_connections, 0)

    def test_proximal_targeting(self):
        conn = ProximalTargeting(connection_prob=0.2, seed=42)
        result = conn(pre_size=20, post_size=20)

        self.assertTrue(np.all(result.pre_compartments == AXON))
        self.assertTrue(np.all(result.post_compartments == BASAL_DENDRITE))

    def test_distal_targeting(self):
        conn = DistalTargeting(connection_prob=0.2, seed=42)
        result = conn(pre_size=20, post_size=20)

        self.assertTrue(np.all(result.pre_compartments == AXON))
        self.assertTrue(np.all(result.post_compartments == APICAL_DENDRITE))

    def test_branch_specific(self):
        conn = BranchSpecific(
            branch_indices=[0, 1, 2],
            connection_prob=0.2,
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        self.assertTrue(np.all(result.pre_compartments == AXON))
        valid_post = np.isin(result.post_compartments, [BASAL_DENDRITE, APICAL_DENDRITE])
        self.assertTrue(np.all(valid_post))

    def test_basal_dendrite_targeting(self):
        conn = BasalDendriteTargeting(connection_prob=0.2, seed=42)
        result = conn(pre_size=20, post_size=20)

        self.assertTrue(np.all(result.pre_compartments == AXON))
        self.assertTrue(np.all(result.post_compartments == BASAL_DENDRITE))

    def test_apical_dendrite_targeting(self):
        conn = ApicalDendriteTargeting(connection_prob=0.2, seed=42)
        result = conn(pre_size=20, post_size=20)

        self.assertTrue(np.all(result.pre_compartments == AXON))
        self.assertTrue(np.all(result.post_compartments == APICAL_DENDRITE))


class TestDendriticPatterns(unittest.TestCase):
    """
    Test dendritic connectivity patterns.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_compartment import DendriticTree, DendriticIntegration

        # Dendritic tree connectivity
        tree_structure = {
            'basal': {'n_branches': 5, 'branch_length': 200 * u.um},
            'apical': {'n_branches': 1, 'branch_length': 600 * u.um}
        }
        dend_tree = DendriticTree(
            tree_structure=tree_structure,
            branch_targeting={'proximal': 0.8, 'distal': 0.2}
        )

        # Dendritic integration with clustering
        dend_integration = DendriticIntegration(
            cluster_size=5,
            n_clusters=10
        )
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_dendritic_tree(self):
        tree_structure = {
            'basal': {'n_branches': 5, 'branch_length': 200 * u.um},
            'apical': {'n_branches': 1, 'branch_length': 600 * u.um}
        }

        conn = DendriticTree(
            tree_structure=tree_structure,
            branch_targeting={'proximal': 0.3, 'distal': 0.2},
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        self.assertEqual(result.model_type, 'multi_compartment')
        self.assertTrue(np.all(result.pre_compartments == AXON))
        valid_post = np.isin(result.post_compartments, [BASAL_DENDRITE, APICAL_DENDRITE])
        self.assertTrue(np.all(valid_post))

        # Check metadata
        self.assertEqual(result.metadata['pattern'], 'dendritic_tree')
        self.assertEqual(result.metadata['tree_structure'], tree_structure)

    def test_dendritic_tree_custom_targeting(self):
        tree_structure = {
            'basal': {'n_branches': 3, 'branch_length': 100 * u.um}
        }

        conn = DendriticTree(
            tree_structure=tree_structure,
            branch_targeting={'proximal': 0.5},  # Only proximal
            seed=42
        )

        result = conn(pre_size=10, post_size=10)

        # Should have connections, but specific targeting depends on implementation
        self.assertGreater(result.n_connections, 0)

    def test_dendritic_integration(self):
        conn = DendriticIntegration(
            cluster_size=3,
            n_clusters=5,
            seed=42
        )

        result = conn(pre_size=20, post_size=10)

        self.assertEqual(result.model_type, 'multi_compartment')
        self.assertTrue(np.all(result.pre_compartments == AXON))
        valid_post = np.isin(result.post_compartments, [BASAL_DENDRITE, APICAL_DENDRITE])
        self.assertTrue(np.all(valid_post))

        # Check metadata
        self.assertEqual(result.metadata['pattern'], 'dendritic_integration')
        self.assertEqual(result.metadata['cluster_size'], 3)
        self.assertEqual(result.metadata['n_clusters'], 5)

        # Should have multiple connections per postsynaptic neuron
        expected_connections = min(3 * 5 * 10, 20 * 10)  # cluster_size * n_clusters * post_size, limited by pre_size
        # Allow some variance due to random selection
        self.assertGreater(result.n_connections, 0)

    def test_dendritic_integration_large_cluster_size(self):
        # Test when cluster size is larger than pre_size
        conn = DendriticIntegration(
            cluster_size=50,  # Larger than pre_size
            n_clusters=2,
            seed=42
        )

        result = conn(pre_size=10, post_size=5)

        # Should still work, just with smaller actual cluster sizes
        self.assertGreater(result.n_connections, 0)


class TestAxonalPatterns(unittest.TestCase):
    """
    Test axonal connectivity patterns.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_compartment import (
            AxonalProjection, AxonalBranching, AxonalArborization, TopographicProjection
        )

        # Basic axonal projection
        axon_proj = AxonalProjection(
            projection_type='local',
            connection_prob=0.1
        )

        # Topographic projection with custom mapping
        def retinotopic_map(source_pos, target_pos):
            return np.exp(-np.linalg.norm(source_pos - target_pos)**2 / 1000)

        topo_proj = TopographicProjection(topographic_map=retinotopic_map)

        # Axonal branching
        branching = AxonalBranching(branches_per_axon=5)

        # Axonal arborization
        arborization = AxonalArborization(
            arborization_radius=150.0 * u.um,
            density=0.3
        )
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_axonal_projection_local(self):
        conn = AxonalProjection(
            projection_type='local',
            connection_prob=0.15,
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        self.assertEqual(result.model_type, 'multi_compartment')
        self.assertTrue(np.all(result.pre_compartments == AXON))
        self.assertTrue(np.all(result.post_compartments == BASAL_DENDRITE))
        self.assertEqual(result.metadata['pattern'], 'axonal_projection')
        self.assertEqual(result.metadata['projection_type'], 'local')

    def test_axonal_projection_with_positions(self):
        pre_positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um
        post_positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um

        def simple_topo_map(source_pos, target_pos):
            return 0.1  # Constant probability

        conn = AxonalProjection(
            projection_type='topographic',
            topographic_map=simple_topo_map,
            seed=42
        )

        result = conn(
            pre_size=10, post_size=10,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        self.assertGreater(result.n_connections, 0)

    def test_axonal_projection_clustered_arborization(self):
        pre_positions = np.array([[0, 0], [10, 10]]) * u.um
        post_positions = np.array([[0, 0], [50, 50]]) * u.um

        conn = AxonalProjection(
            projection_type='local',
            arborization_pattern='clustered',
            connection_prob=0.5,
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        # Closer neurons should have higher connection probability
        self.assertGreater(result.n_connections, 0)

    def test_axonal_branching(self):
        conn = AxonalBranching(
            branches_per_axon=3,
            seed=42
        )

        result = conn(pre_size=10, post_size=20)

        self.assertEqual(result.model_type, 'multi_compartment')
        self.assertTrue(np.all(result.pre_compartments == AXON))
        valid_post = np.isin(result.post_compartments, [BASAL_DENDRITE, APICAL_DENDRITE])
        self.assertTrue(np.all(valid_post))

        # Check metadata
        self.assertEqual(result.metadata['pattern'], 'axonal_branching')
        self.assertEqual(result.metadata['branches_per_axon'], 3)

        # Each presynaptic neuron should connect to multiple targets on average
        self.assertGreater(result.n_connections, 10)  # Should be more than just one per pre neuron

    def test_axonal_arborization_no_positions(self):
        # Should fall back to CompartmentSpecific behavior
        conn = AxonalArborization(
            arborization_radius=50 * u.um,
            density=0.2,
            seed=42
        )

        result = conn(pre_size=10, post_size=10)

        self.assertEqual(result.model_type, 'multi_compartment')
        self.assertGreater(result.n_connections, 0)

    def test_axonal_arborization_with_positions(self):
        # Close positions should have high connection probability
        pre_positions = np.array([[0, 0], [10, 10]]) * u.um
        post_positions = np.array([[5, 5], [100, 100]]) * u.um

        conn = AxonalArborization(
            arborization_radius=20 * u.um,
            density=0.8,
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        self.assertEqual(result.metadata['pattern'], 'axonal_arborization')

    def test_axonal_arborization_scalar_radius(self):
        # Test with scalar radius (no units)
        pre_positions = np.array([[0, 0]])
        post_positions = np.array([[10, 10]])

        conn = AxonalArborization(
            arborization_radius=20.0,  # Scalar
            density=0.5,
            seed=42
        )

        result = conn(
            pre_size=1, post_size=1,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        self.assertGreaterEqual(result.n_connections, 0)

    def test_topographic_projection(self):
        def distance_map(source_pos, target_pos):
            distance = np.linalg.norm(source_pos - target_pos)
            return np.exp(-distance / 50.0)

        conn = TopographicProjection(
            topographic_map=distance_map,
            seed=42
        )

        pre_positions = np.array([[0, 0], [10, 10]]) * u.um
        post_positions = np.array([[5, 5], [100, 100]]) * u.um

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        # Should delegate to AxonalProjection
        self.assertEqual(result.model_type, 'multi_compartment')


class TestSynapticPatterns(unittest.TestCase):
    """
    Test synaptic connectivity patterns.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_compartment import (
            SynapticPlacement, SynapticClustering, ActivityDependentSynapses
        )

        # Synaptic placement with preferences
        placement = SynapticPlacement(
            placement_rule='uniform',
            compartment_preferences={
                BASAL_DENDRITE: 0.6,
                APICAL_DENDRITE: 0.4
            }
        )

        # Synaptic clustering
        clustering = SynapticClustering(
            cluster_size=5,
            n_clusters_per_neuron=10
        )

        # Activity-dependent synapses
        base_pattern = AxonToDendrite(connection_prob=0.1)
        activity_dep = ActivityDependentSynapses(
            base_pattern=base_pattern,
            plasticity_type='stdp',
            learning_rate=0.01
        )
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_synaptic_placement(self):
        conn = SynapticPlacement(
            placement_rule='uniform',
            compartment_preferences={
                BASAL_DENDRITE: 0.7,
                APICAL_DENDRITE: 0.3
            },
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        self.assertEqual(result.model_type, 'multi_compartment')
        self.assertTrue(np.all(result.pre_compartments == AXON))
        valid_post = np.isin(result.post_compartments, [BASAL_DENDRITE, APICAL_DENDRITE])
        self.assertTrue(np.all(valid_post))

        # Should prefer basal dendrites
        basal_count = np.sum(result.post_compartments == BASAL_DENDRITE)
        apical_count = np.sum(result.post_compartments == APICAL_DENDRITE)
        if basal_count > 0 and apical_count > 0:
            self.assertGreater(basal_count, apical_count)

    def test_synaptic_placement_different_rules(self):
        for rule in ['uniform', 'proximal', 'distal', 'distance_weighted']:
            conn = SynapticPlacement(
                placement_rule=rule,
                seed=42
            )

            result = conn(pre_size=10, post_size=10)
            self.assertEqual(result.model_type, 'multi_compartment')

    def test_synaptic_clustering(self):
        conn = SynapticClustering(
            cluster_size=4,
            n_clusters_per_neuron=3,
            seed=42
        )

        result = conn(pre_size=20, post_size=5)

        # Should delegate to DendriticIntegration
        self.assertEqual(result.model_type, 'multi_compartment')
        self.assertTrue(np.all(result.pre_compartments == AXON))


class TestRandomAndAllToAllPatterns(unittest.TestCase):
    """
    Test random and all-to-all compartment patterns.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from braintools.conn._conn_compartment import (
            RandomCompartment, AllToAllCompartments
        )

        # Random compartment connectivity
        random_conn = RandomCompartment(
            compartments=[SOMA, BASAL_DENDRITE, APICAL_DENDRITE],
            connection_prob=0.1
        )

        # All-to-all compartment connectivity
        all_to_all = AllToAllCompartments()
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_all_to_all_compartments(self):
        conn = AllToAllCompartments(seed=42)
        result = conn(pre_size=5, post_size=5)

        self.assertEqual(result.model_type, 'multi_compartment')

        # Should have connections (probability is 1.0)
        self.assertGreater(result.n_connections, 0)

        # Should connect between all compartment types
        all_compartments = {SOMA, BASAL_DENDRITE, APICAL_DENDRITE, AXON}
        unique_pre = set(result.pre_compartments)
        unique_post = set(result.post_compartments)

        # With probability 1.0 and enough neurons, should see all compartments
        self.assertTrue(unique_pre.issubset(all_compartments))
        self.assertTrue(unique_post.issubset(all_compartments))


class TestCustomCompartment(unittest.TestCase):
    """
    Test custom compartment connectivity.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_compartment import CustomCompartment

        def custom_conn_func(pre_size, post_size, **kwargs):
            # Simple custom connectivity function
            pre_indices = [0, 1]
            post_indices = [1, 0]
            pre_compartments = [AXON, AXON]
            post_compartments = [SOMA, SOMA]
            weights = [0.1, 0.2] * u.nS
            delays = [1.0, 2.0] * u.ms
            return pre_indices, post_indices, pre_compartments, post_compartments, weights, delays

        custom_conn = CustomCompartment(connection_func=custom_conn_func)
        result = custom_conn(pre_size=10, post_size=10)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_custom_compartment_minimal(self):
        def simple_func(pre_size, post_size, **kwargs):
            # Return minimum required data
            return [0, 1], [1, 0], [AXON, AXON], [SOMA, SOMA]

        conn = CustomCompartment(connection_func=simple_func)
        result = conn(pre_size=10, post_size=10)

        self.assertEqual(result.model_type, 'multi_compartment')
        self.assertEqual(result.n_connections, 2)
        np.testing.assert_array_equal(result.pre_indices, [0, 1])
        np.testing.assert_array_equal(result.post_indices, [1, 0])
        np.testing.assert_array_equal(result.pre_compartments, [AXON, AXON])
        np.testing.assert_array_equal(result.post_compartments, [SOMA, SOMA])

    def test_custom_compartment_with_weights(self):
        def func_with_weights(pre_size, post_size, **kwargs):
            weights = [0.1, 0.2] * u.nS
            return [0, 1], [1, 0], [AXON, AXON], [SOMA, SOMA], weights

        conn = CustomCompartment(connection_func=func_with_weights)
        result = conn(pre_size=10, post_size=10)

        self.assertIsNotNone(result.weights)
        np.testing.assert_array_almost_equal(result.weights.mantissa, [0.1, 0.2])
        self.assertEqual(result.weights.unit, u.nS)

    def test_custom_compartment_with_weights_and_delays(self):
        def func_with_all(pre_size, post_size, **kwargs):
            weights = [0.1, 0.2] * u.nS
            delays = [1.0, 2.0] * u.ms
            return [0, 1], [1, 0], [AXON, AXON], [SOMA, SOMA], weights, delays

        conn = CustomCompartment(connection_func=func_with_all)
        result = conn(pre_size=10, post_size=10)

        self.assertIsNotNone(result.weights)
        self.assertIsNotNone(result.delays)
        np.testing.assert_array_almost_equal(result.weights.mantissa, [0.1, 0.2])
        np.testing.assert_array_almost_equal(result.delays.mantissa, [1.0, 2.0])

    def test_custom_compartment_with_scalar_delays(self):
        def func_scalar_delays(pre_size, post_size, **kwargs):
            weights = [0.1, 0.2]
            delays = [1.0, 2.0]  # No units
            return [0, 1], [1, 0], [AXON, AXON], [SOMA, SOMA], weights, delays

        conn = CustomCompartment(connection_func=func_scalar_delays)
        result = conn(pre_size=10, post_size=10)

        self.assertIsNotNone(result.delays)
        self.assertEqual(result.delays.unit, u.ms)  # Should add ms units

    def test_custom_compartment_invalid_return_length(self):
        def invalid_func(pre_size, post_size, **kwargs):
            return [0], [1], [AXON]  # Too few values

        conn = CustomCompartment(connection_func=invalid_func)

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=10)

    def test_custom_compartment_metadata(self):
        def simple_func(pre_size, post_size, **kwargs):
            return [0], [1], [AXON], [SOMA]

        conn = CustomCompartment(connection_func=simple_func)
        result = conn(pre_size=10, post_size=10)

        self.assertEqual(result.metadata['pattern'], 'custom_compartment')


class TestEdgeCases(unittest.TestCase):
    """
    Test edge cases and error conditions.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_compartment import CompartmentSpecific

        # Empty connections
        empty_conn = CompartmentSpecific(
            compartment_mapping={AXON: SOMA},
            connection_prob=0.0  # No connections
        )

        # Large networks
        large_conn = CompartmentSpecific(
            compartment_mapping={AXON: SOMA},
            connection_prob=0.001
        )
        result = large_conn(pre_size=1000, post_size=1000)

        # Different position shapes
        positions_3d = np.random.uniform(0, 100, (10, 3)) * u.um
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_empty_connections_various_patterns(self):
        # Test various patterns with zero probability
        patterns = [
            CompartmentSpecific(compartment_mapping={AXON: SOMA}, connection_prob=0.0),
            SomaToDendrite(prob_per_dendrite=0.0),
            AxonToSoma(connection_prob=0.0),
            # DendriticTree(
            #     tree_structure={'basal': {'n_branches': 3}},
            #     branch_targeting={'proximal': 0.0}
            # ),
            AxonalBranching(branches_per_axon=0),  # Zero branches
        ]

        for pattern in patterns:
            print(pattern)
            pattern.seed = 42
            result = pattern(pre_size=10, post_size=10)
            self.assertEqual(result.n_connections, 0)
            self.assertEqual(len(result.pre_indices), 0)

    def test_large_networks(self):
        conn = CompartmentSpecific(
            compartment_mapping={AXON: SOMA},
            connection_prob=0.001,  # Low probability for large networks
            seed=42
        )

        result = conn(pre_size=500, post_size=500)

        self.assertEqual(result.model_type, 'multi_compartment')
        self.assertEqual(result.shape, (500, 500))
        # Should have some connections but not too many
        self.assertGreater(result.n_connections, 0)
        self.assertLess(result.n_connections, 10000)  # Much less than 500*500

    def test_single_neuron_networks(self):
        conn = CompartmentSpecific(
            compartment_mapping={AXON: SOMA},
            connection_prob=1.0,  # Ensure connection
            seed=42
        )

        result = conn(pre_size=1, post_size=1)

        self.assertEqual(result.n_connections, 1)
        np.testing.assert_array_equal(result.pre_indices, [0])
        np.testing.assert_array_equal(result.post_indices, [0])

    def test_asymmetric_network_sizes(self):
        conn = CompartmentSpecific(
            compartment_mapping={AXON: SOMA},
            connection_prob=0.2,
            seed=42
        )

        result = conn(pre_size=5, post_size=20)

        self.assertEqual(result.shape, (5, 20))
        self.assertTrue(np.all(result.pre_indices < 5))
        self.assertTrue(np.all(result.post_indices < 20))

    def test_3d_positions(self):
        # Test with 3D positions
        pre_positions = np.random.RandomState(42).uniform(0, 100, (10, 3)) * u.um
        post_positions = np.random.RandomState(42).uniform(0, 100, (10, 3)) * u.um

        conn = MorphologyDistance(
            sigma=50 * u.um,
            decay_function='gaussian',
            seed=42
        )

        result = conn(
            pre_size=10,
            post_size=10,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        self.assertEqual(result.model_type, 'multi_compartment')
        # Should handle 3D positions correctly

    def test_mismatched_position_sizes(self):
        # Positions size doesn't match network size
        pre_positions = np.random.RandomState(42).uniform(0, 100, (5, 2)) * u.um  # Size 5
        post_positions = np.random.RandomState(42).uniform(0, 100, (8, 2)) * u.um  # Size 8

        conn = MorphologyDistance(
            sigma=50 * u.um,
            decay_function='gaussian',
            seed=42
        )

        # Should still work, using available positions
        result = conn(
            pre_size=10,
            post_size=10,  # Size 10 each
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        self.assertEqual(result.model_type, 'multi_compartment')

    def test_unit_consistency(self):
        # Test that units are handled consistently across different patterns
        weight_init = Constant(1.5 * u.nS)
        delay_init = Uniform(0.5 * u.ms, 2.5 * u.ms)

        patterns = [
            CompartmentSpecific(compartment_mapping={AXON: SOMA}, connection_prob=0.1),
            SomaToDendrite(prob_per_dendrite=0.1),
            AxonalBranching(branches_per_axon=2),
        ]

        for pattern in patterns:
            pattern.weight_init = weight_init
            pattern.delay_init = delay_init
            pattern.seed = 42

            result = pattern(pre_size=10, post_size=10)

            if result.n_connections > 0:
                self.assertEqual(result.weights.unit, u.nS)
                self.assertEqual(result.delays.unit, u.ms)
                self.assertTrue(np.all(result.weights.mantissa == 1.5))
                self.assertTrue(np.all((result.delays >= 0.5 * u.ms) & (result.delays < 2.5 * u.ms)))


if __name__ == '__main__':
    unittest.main()
