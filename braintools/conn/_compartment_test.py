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
import pytest
from scipy.spatial.distance import cdist

from braintools._deprecation import BraintoolsDeprecationWarning
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
from braintools.init import (
    Constant,
    Uniform,
    ExponentialProfile,
    GaussianProfile,
    LinearProfile,
)


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
        # CMP-2: distance-dependent connectivity is undefined without positions.
        # Previously it silently fell back to a hardcoded prob=0.1; now it raises.
        conn = MorphologyDistance(
            sigma=50 * u.um,
            seed=42
        )

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=10)

    def test_morphology_distance_with_positions(self):
        pre_positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um
        post_positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um

        # Legacy string decay function (deprecated in 0.4.0).
        with pytest.warns(BraintoolsDeprecationWarning):
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
            # allow_self_connections so the zero-distance diagonal (i==j) is kept;
            # otherwise the sparse off-diagonal pairs can yield an empty result with
            # no metadata for the 'linear' decay on this toy data.
            # Legacy string decay functions (deprecated in 0.4.0).
            with pytest.warns(BraintoolsDeprecationWarning):
                conn = MorphologyDistance(
                    sigma=10 * u.um,
                    decay_function=decay_func,
                    allow_self_connections=True,
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
        axon_proj = AxonalProjection(connection_prob=0.1)

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
            connection_prob=0.15,
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        self.assertEqual(result.model_type, 'multi_compartment')
        self.assertTrue(np.all(result.pre_compartments == AXON))
        # CMP-3: AxonalProjection now targets both basal and apical dendrites
        # (previously hardcoded to BASAL_DENDRITE only).
        valid_post = np.isin(result.post_compartments, [BASAL_DENDRITE, APICAL_DENDRITE])
        self.assertTrue(np.all(valid_post))
        self.assertEqual(result.metadata['pattern'], 'axonal_projection')
        self.assertFalse(result.metadata['topographic'])
        self.assertIsNone(result.metadata['arborization_sigma'])

    def test_axonal_projection_with_positions(self):
        pre_positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um
        post_positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um

        def simple_topo_map(source_pos, target_pos):
            return 0.1  # Constant probability

        conn = AxonalProjection(
            topographic_map=simple_topo_map,
            seed=42
        )

        result = conn(
            pre_size=10, post_size=10,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        self.assertGreater(result.n_connections, 0)

    def test_axonal_projection_arborization_sigma(self):
        # Pairs far beyond arborization_sigma never connect; near pairs keep
        # (almost) the full connection_prob.
        pre_positions = np.array([[0., 0.], [1000., 1000.]]) * u.um
        post_positions = np.array([[0., 0.], [1000., 1000.]]) * u.um
        conn = AxonalProjection(
            connection_prob=1.0,
            arborization_sigma=10 * u.um,
            allow_self_connections=True,
            seed=42
        )
        result = conn(
            pre_size=2, post_size=2,
            pre_positions=pre_positions,
            post_positions=post_positions
        )
        self.assertEqual(sorted(zip(result.pre_indices.tolist(), result.post_indices.tolist())),
                         [(0, 0), (1, 1)])

    def test_axonal_projection_arborization_sigma_statistics(self):
        # The realized fraction of pairs matches connection_prob * exp(-d^2 / 2 sigma^2).
        rs = np.random.RandomState(0)
        pre = rs.uniform(0, 400, (200, 2)) * u.um
        post = rs.uniform(0, 400, (200, 2)) * u.um
        res = AxonalProjection(connection_prob=0.5, arborization_sigma=0.05 * u.mm, seed=1)(
            pre_size=200, post_size=200, pre_positions=pre, post_positions=post)
        from scipy.spatial.distance import cdist
        d = cdist(pre.mantissa, post.mantissa)
        expected = (0.5 * np.exp(-d ** 2 / (2 * 50.0 ** 2)))
        np.fill_diagonal(expected, 0.0)  # equal sizes: autapse index pairs dropped
        self.assertAlmostEqual(res.n_connections / expected.sum(), 1.0, delta=0.06)

    def test_axonal_projection_spatial_terms_require_positions(self):
        with self.assertRaises(ValueError):
            AxonalProjection(arborization_sigma=50 * u.um, seed=0)(pre_size=5, post_size=5)
        with self.assertRaises(ValueError):
            AxonalProjection(topographic_map=lambda a, b: 0.1, seed=0)(pre_size=5, post_size=5)

    def test_axonal_projection_removed_parameters_rejected(self):
        for kwargs in [dict(projection_type='local'), dict(arborization_pattern='clustered'),
                       dict(spatial_scale=100.0)]:
            with self.assertRaises(TypeError):
                AxonalProjection(**kwargs)

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

    def test_axonal_branching_spread_keeps_branches_local(self):
        rs = np.random.RandomState(0)
        positions = rs.uniform(0, 1000, (150, 2)) * u.um
        call = dict(pre_size=150, post_size=150, pre_positions=positions, post_positions=positions)
        uniform = AxonalBranching(branches_per_axon=6, seed=1)(**call)
        local = AxonalBranching(branches_per_axon=6, branch_spread=40 * u.um, seed=1)(**call)
        d_uniform = np.asarray(u.get_mantissa(uniform.get_distances())).mean()
        d_local = np.asarray(u.get_mantissa(local.get_distances())).mean()
        self.assertLess(d_local, 100.0)
        self.assertGreater(d_uniform, 300.0)
        # Autapses are excluded from the candidates, not dropped afterwards, so the
        # mean out-degree still matches branches_per_axon.
        for result in (uniform, local):
            self.assertEqual(int(np.sum(result.pre_indices == result.post_indices)), 0)
            self.assertAlmostEqual(result.n_connections / 150, 6.0, delta=0.5)
        self.assertEqual(local.metadata['branch_spread'], 40 * u.um)
        # a plain number is in the positions' unit
        plain = AxonalBranching(branches_per_axon=6, branch_spread=40.0, seed=1)(**call)
        np.testing.assert_array_equal(plain.post_indices, local.post_indices)

    def test_axonal_branching_spread_requires_positions(self):
        with self.assertRaises(ValueError):
            AxonalBranching(branch_spread=50 * u.um, seed=0)(pre_size=5, post_size=5)

    def test_plain_length_uses_position_unit(self):
        # A plain-number radius/sigma used to crash on Quantity positions.
        positions = np.random.RandomState(0).uniform(0, 300, (40, 2)) * u.um
        call = dict(pre_size=40, post_size=40, pre_positions=positions, post_positions=positions)
        for make in [
            lambda r: AxonalArborization(arborization_radius=r, density=0.5, seed=0),
            lambda r: MorphologyDistance(sigma=r, seed=0),
        ]:
            plain = make(60.0)(**call)
            quantity = make(60.0 * u.um)(**call)
            self.assertGreater(plain.n_connections, 0)
            np.testing.assert_array_equal(plain.pre_indices, quantity.pre_indices)
            np.testing.assert_array_equal(plain.post_indices, quantity.post_indices)

    def test_axonal_arborization_no_positions(self):
        # CMP-2: spatial arborization is undefined without positions.
        # Previously it silently fell back to a dense pattern; now it raises.
        conn = AxonalArborization(
            arborization_radius=50 * u.um,
            density=0.2,
            seed=42
        )

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=10)

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
            connection_prob=0.2,
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

    def test_synaptic_placement_rates(self):
        # connection_prob sets how many pairs connect; preferences set where.
        result = SynapticPlacement(
            connection_prob=0.1,
            compartment_preferences={SOMA: 1.0, APICAL_DENDRITE: 3.0},
            seed=0,
        )(pre_size=200, post_size=200)
        n_pairs = 200 * 199
        self.assertAlmostEqual(result.n_connections / n_pairs, 0.1, delta=0.005)
        # one synapse per connected pair
        pairs = list(zip(result.pre_indices.tolist(), result.post_indices.tolist()))
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(int(np.sum(result.pre_indices == result.post_indices)), 0)
        frac_apical = np.mean(result.post_compartments == APICAL_DENDRITE)
        self.assertAlmostEqual(frac_apical, 0.75, delta=0.02)
        self.assertTrue(set(result.post_compartments.tolist()) <= {SOMA, APICAL_DENDRITE})
        self.assertEqual(result.metadata['pattern'], 'synaptic_placement')

    def test_synaptic_placement_single_target(self):
        result = SynapticPlacement(compartment_preferences={BASAL_DENDRITE: 1.0}, seed=0)(
            pre_size=30, post_size=30)
        self.assertTrue(np.all(result.post_compartments == BASAL_DENDRITE))

    def test_synaptic_placement_validation(self):
        with self.assertRaises(TypeError):
            SynapticPlacement(placement_rule='uniform')
        with self.assertRaises(ValueError):
            SynapticPlacement(connection_prob=1.5)
        with self.assertRaises(ValueError):
            SynapticPlacement(compartment_preferences={BASAL_DENDRITE: 0.0})
        with self.assertRaises(ValueError):
            SynapticPlacement(compartment_preferences={7: 1.0})

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
        # Test various patterns with zero probability. Seeds are passed at
        # construction (reassigning ``pattern.seed`` afterwards has no effect
        # on the already-created ``self.rng``).
        patterns = [
            CompartmentSpecific(compartment_mapping={AXON: SOMA}, connection_prob=0.0, seed=42),
            SomaToDendrite(prob_per_dendrite=0.0, seed=42),
            AxonToSoma(connection_prob=0.0, seed=42),
            DendriticTree(branch_targeting={'proximal': 0.0, 'distal': 0.0}, seed=42),
            AxonalBranching(branches_per_axon=0, seed=42),  # Zero branches
        ]

        for pattern in patterns:
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
        # allow_self_connections=True so the single 0->0 connection is kept;
        # by default same-population self-connections (autapses) are dropped.
        conn = CompartmentSpecific(
            compartment_mapping={AXON: SOMA},
            connection_prob=1.0,  # Ensure connection
            allow_self_connections=True,
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


class TestConnFixesRegression(unittest.TestCase):
    """Regression tests for the conn._compartment bug fixes."""

    @staticmethod
    def _quads(result):
        return set(zip(
            result.pre_indices.tolist(),
            result.post_indices.tolist(),
            result.pre_compartments.tolist(),
            result.post_compartments.tolist(),
        ))

    def test_dendritic_integration_no_duplicate_quads(self):
        # CMP-1: clusters sampled across many clusters used to repeat the same
        # (pre, post, pre_comp, post_comp) quad, inflating weights after conversion.
        conn = DendriticIntegration(cluster_size=5, n_clusters=20, seed=42)
        result = conn(pre_size=6, post_size=4)
        quads = self._quads(result)
        self.assertEqual(len(quads), result.n_connections)

    def test_synaptic_clustering_no_duplicate_quads(self):
        # CMP-1: SynapticClustering delegates to DendriticIntegration.
        conn = SynapticClustering(cluster_size=5, n_clusters_per_neuron=20, seed=42)
        result = conn(pre_size=6, post_size=4)
        quads = self._quads(result)
        self.assertEqual(len(quads), result.n_connections)

    def test_morphology_distance_requires_positions(self):
        # CMP-2: must raise rather than silently use a hardcoded prob.
        conn = MorphologyDistance(sigma=50 * u.um, seed=42)
        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=10)

    def test_axonal_arborization_requires_positions(self):
        # CMP-2: must raise rather than silently produce a dense pattern.
        conn = AxonalArborization(arborization_radius=50 * u.um, density=0.2, seed=42)
        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=10)

    def test_axonal_projection_targets_basal_and_apical(self):
        # CMP-3: previously hardcoded to BASAL_DENDRITE only.
        conn = AxonalProjection(connection_prob=0.3, seed=42)
        result = conn(pre_size=40, post_size=40)
        post = set(result.post_compartments.tolist())
        self.assertIn(BASAL_DENDRITE, post)
        self.assertIn(APICAL_DENDRITE, post)
        self.assertTrue(post.issubset({BASAL_DENDRITE, APICAL_DENDRITE}))

    def test_1d_positions_accepted(self):
        # CMP-7: cdist used to crash on 1-D (N,) position arrays.
        pre_positions = np.array([0.0, 10.0, 20.0]) * u.um
        post_positions = np.array([0.0, 10.0, 20.0]) * u.um

        morph = MorphologyDistance(sigma=15 * u.um, seed=42)
        result = morph(
            pre_size=3, post_size=3,
            pre_positions=pre_positions, post_positions=post_positions
        )
        self.assertEqual(result.model_type, 'multi_compartment')

        arbor = AxonalArborization(arborization_radius=15 * u.um, density=0.9, seed=42)
        result = arbor(
            pre_size=3, post_size=3,
            pre_positions=pre_positions, post_positions=post_positions
        )
        self.assertEqual(result.model_type, 'multi_compartment')

    def test_out_of_range_probability_raises(self):
        # CMP-9 / VALIDATION
        with self.assertRaises(ValueError):
            CompartmentSpecific(compartment_mapping={AXON: SOMA}, connection_prob=1.5)
        with self.assertRaises(ValueError):
            AxonalProjection(connection_prob=-0.1)
        with self.assertRaises(ValueError):
            AxonalArborization(density=2.0)
        with self.assertRaises(ValueError):
            BranchSpecific(connection_prob=1.1)
        with self.assertRaises(ValueError):
            DendriticTree(branch_targeting={'proximal': 1.2})

    def test_negative_count_raises(self):
        # CMP-9 / VALIDATION
        with self.assertRaises(ValueError):
            DendriticIntegration(cluster_size=-1)
        with self.assertRaises(ValueError):
            DendriticIntegration(n_clusters=-2)
        with self.assertRaises(ValueError):
            AxonalBranching(branches_per_axon=-1)
        with self.assertRaises(ValueError):
            SynapticClustering(cluster_size=-3)

    def test_no_self_connections_by_default(self):
        # CMP-6: same-neuron self-loops must be dropped by default.
        conn = CompartmentSpecific(
            compartment_mapping={SOMA: SOMA}, connection_prob=1.0, seed=42
        )
        result = conn(pre_size=10, post_size=10)
        self.assertTrue(np.all(result.pre_indices != result.post_indices))

    def test_all_to_all_compartments_no_autapses(self):
        # CMP-6: AllToAllCompartments emitted SOMA->SOMA self-loops on neuron i.
        conn = AllToAllCompartments(seed=42)
        result = conn(pre_size=5, post_size=5)
        self.assertTrue(np.all(result.pre_indices != result.post_indices))

    def test_self_connections_allowed_when_requested(self):
        conn = CompartmentSpecific(
            compartment_mapping={SOMA: SOMA},
            connection_prob=1.0,
            allow_self_connections=True,
            seed=42,
        )
        result = conn(pre_size=10, post_size=10)
        self.assertTrue(np.any(result.pre_indices == result.post_indices))

    def test_dict_prob_missing_pair_warns(self):
        # CMP-12: a mapped pair with no probability entry should warn.
        conn = CompartmentSpecific(
            compartment_mapping={AXON: [BASAL_DENDRITE, APICAL_DENDRITE]},
            connection_prob={(AXON, BASAL_DENDRITE): 0.2},  # APICAL missing
            seed=42,
        )
        with self.assertWarns(UserWarning):
            conn(pre_size=10, post_size=10)

    def test_custom_compartment_non_tuple_return_raises(self):
        # CMP-11: fragile return parsing should give a clear error.
        def bad_func(pre_size, post_size, **kwargs):
            return 42  # not a tuple/list

        conn = CustomCompartment(connection_func=bad_func)
        with self.assertRaises(ValueError):
            conn(pre_size=5, post_size=5)


# ---------------------------------------------------------------------------
# C4 (v0.4): MorphologyDistance takes a ``braintools.init`` DistanceProfile.
# The legacy decay strings map onto profiles; ``_ref_morphology`` re-implements
# the pre-refactor closed forms verbatim so the outputs stay pinned.
# ---------------------------------------------------------------------------

def _ref_morphology(pre, post, sigma, kind, seed, allow_self=False):
    """Old ``MorphologyDistance.generate`` (plain numbers, default mapping)."""
    distances = cdist(pre, post)
    if kind == 'gaussian':
        probs = np.exp(-distances ** 2 / (2 * sigma ** 2))
    elif kind == 'exponential':
        probs = np.exp(-distances / sigma)
    elif kind == 'linear':
        probs = np.maximum(0, 1 - distances / sigma)
    else:
        raise AssertionError(kind)
    rng = np.random.RandomState(seed)
    quads = []
    for target in (BASAL_DENDRITE, APICAL_DENDRITE):
        i, j = np.where(rng.random((len(pre), len(post))) < probs)
        quads.extend(zip(i.tolist(), j.tolist(), [AXON] * len(i), [target] * len(i)))
    if not allow_self and len(pre) == len(post):
        quads = [q for q in quads if q[0] != q[1]]
    return quads


def _quads(result):
    return list(zip(
        np.asarray(result.pre_indices).tolist(),
        np.asarray(result.post_indices).tolist(),
        np.asarray(result.pre_compartments).tolist(),
        np.asarray(result.post_compartments).tolist(),
    ))


_LEGACY_DECAYS = ['gaussian', 'exponential', 'linear']


class TestMorphologyDistanceProfiles:
    """``decay_function`` accepts DistanceProfile instances; strings are deprecated."""

    pre = np.random.RandomState(0).uniform(0, 120, (25, 2))
    post = np.random.RandomState(1).uniform(0, 120, (25, 2))

    def _call(self, conn, unit=u.um):
        scale = 1 if unit is None else unit
        return conn(pre_size=25, post_size=25,
                    pre_positions=self.pre * scale, post_positions=self.post * scale)

    @pytest.mark.parametrize('kind', _LEGACY_DECAYS)
    def test_legacy_string_warns_and_matches_old_output(self, kind):
        with pytest.warns(BraintoolsDeprecationWarning, match='decay_function') as record:
            conn = MorphologyDistance(sigma=30 * u.um, decay_function=kind, seed=7)
        assert len(record) == 1
        assert '1.0.0' in str(record[0].message)
        assert 'braintools.init' in str(record[0].message)
        result = self._call(conn)
        assert result.n_connections > 0
        assert _quads(result) == _ref_morphology(self.pre, self.post, 30.0, kind, seed=7)
        assert result.metadata['decay_function'] == kind

    @pytest.mark.parametrize('kind', _LEGACY_DECAYS)
    def test_legacy_string_unitless(self, kind):
        with pytest.warns(BraintoolsDeprecationWarning):
            conn = MorphologyDistance(sigma=30.0, decay_function=kind, seed=3)
        result = self._call(conn, unit=None)
        assert _quads(result) == _ref_morphology(self.pre, self.post, 30.0, kind, seed=3)

    def test_default_is_gaussian_and_does_not_warn(self):
        # pytest turns BraintoolsDeprecationWarning into an error, so merely
        # constructing and calling proves the default path is silent.
        conn = MorphologyDistance(sigma=30 * u.um, seed=7)
        assert conn.decay_function is None
        result = self._call(conn)
        assert _quads(result) == _ref_morphology(self.pre, self.post, 30.0, 'gaussian', seed=7)
        # Same metadata as before 0.4, when the default was the string 'gaussian'.
        assert result.metadata['decay_function'] == 'gaussian'

    @pytest.mark.parametrize('profile, kind', [
        (GaussianProfile(30 * u.um), 'gaussian'),
        (ExponentialProfile(30 * u.um), 'exponential'),
        (LinearProfile(30 * u.um), 'linear'),
    ])
    def test_profile_instance_matches_legacy_string(self, profile, kind):
        conn = MorphologyDistance(decay_function=profile, seed=11)
        result = self._call(conn)
        assert _quads(result) == _ref_morphology(self.pre, self.post, 30.0, kind, seed=11)
        assert result.metadata['decay_function'] is profile

    def test_profile_unitless_positions(self):
        conn = MorphologyDistance(decay_function=GaussianProfile(30.0), seed=5)
        result = self._call(conn, unit=None)
        assert _quads(result) == _ref_morphology(self.pre, self.post, 30.0, 'gaussian', seed=5)

    def test_profile_unit_converted_to_positions(self):
        # A profile length in mm is applied to micrometre positions.
        conn = MorphologyDistance(decay_function=ExponentialProfile(0.03 * u.mm), seed=5)
        result = self._call(conn)
        assert _quads(result) == _ref_morphology(self.pre, self.post, 30.0, 'exponential', seed=5)

    def test_profile_max_distance_is_respected(self):
        conn = MorphologyDistance(
            decay_function=GaussianProfile(30 * u.um, max_distance=20 * u.um),
            allow_self_connections=True, seed=5,
        )
        result = self._call(conn)
        d = cdist(self.pre, self.post)
        assert result.n_connections > 0
        assert np.all(d[result.pre_indices, result.post_indices] <= 20.0)

    def test_composed_profile_connects_subset(self):
        full = self._call(MorphologyDistance(decay_function=GaussianProfile(30 * u.um), seed=9))
        half = self._call(MorphologyDistance(decay_function=GaussianProfile(30 * u.um) * 0.5, seed=9))
        assert 0 < half.n_connections < full.n_connections
        assert set(_quads(half)) <= set(_quads(full))

    def test_profile_incompatible_with_unitless_positions_raises(self):
        conn = MorphologyDistance(decay_function=GaussianProfile(30 * u.um), seed=5)
        with pytest.raises(Exception):
            self._call(conn, unit=None)

    def test_sigma_and_profile_together_raises(self):
        with pytest.raises(ValueError, match='sigma'):
            MorphologyDistance(sigma=30 * u.um, decay_function=GaussianProfile(30 * u.um))

    def test_missing_sigma_raises(self):
        with pytest.raises(ValueError, match='sigma'):
            MorphologyDistance()

    def test_unknown_string_raises_at_generate_without_deprecation(self):
        conn = MorphologyDistance(sigma=30 * u.um, decay_function='cosine', seed=1)
        with pytest.raises(ValueError, match='Unknown decay function'):
            self._call(conn)

    def test_invalid_type_raises_at_generate(self):
        conn = MorphologyDistance(sigma=30 * u.um, decay_function=42, seed=1)
        with pytest.raises(ValueError, match='Unknown decay function'):
            self._call(conn)

    @pytest.mark.parametrize('sigma', [0.0, -30.0])
    def test_non_positive_sigma_raises(self, sigma):
        conn = MorphologyDistance(sigma=sigma * u.um, seed=1)
        with pytest.raises(ValueError, match='sigma'):
            self._call(conn)


if __name__ == '__main__':
    unittest.main()
