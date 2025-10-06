#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive test cases for UniqueStateManager class.
Tests all methods and edge cases for managing unique State objects in PyTrees.
"""

import brainstate
import jax.numpy as jnp
from brainstate import ParamState, State

from braintools.optim._state_uniquifier import UniqueStateManager


# ============================================================================
# Initialization Tests
# ============================================================================

class TestInitialization:
    """Test UniqueStateManager initialization."""

    def test_empty_initialization(self):
        """Test creating an empty manager."""
        manager = UniqueStateManager()
        assert manager.num_unique_states == 0
        assert len(manager.unique_states) == 0
        assert len(manager.unique_paths) == 0
        assert len(manager.seen_ids) == 0
        assert len(manager.flattened_states) == 0
        assert manager.pytree_structure is None

    def test_initialization_with_pytree(self):
        """Test creating a manager with initial pytree."""
        state1 = ParamState(jnp.ones((2, 3)))
        state2 = ParamState(jnp.zeros((3, 4)))
        pytree = {'a': state1, 'b': state2}

        manager = UniqueStateManager(pytree)
        assert manager.num_unique_states == 2
        assert len(manager.unique_states) == 2
        assert manager.pytree_structure is not None

    def test_initialization_with_duplicate_states(self):
        """Test initialization with duplicate state references."""
        state = ParamState(jnp.ones((2, 3)))
        pytree = {'a': state, 'b': state, 'c': state}

        manager = UniqueStateManager(pytree)
        assert manager.num_unique_states == 1
        assert id(state) in manager.seen_ids

    def test_len_method(self):
        """Test __len__ method."""
        state1 = ParamState(jnp.ones((2, 3)))
        state2 = ParamState(jnp.zeros((3, 4)))
        pytree = {'a': state1, 'b': state2}

        manager = UniqueStateManager(pytree)
        assert len(manager) == 2


# ============================================================================
# make_unique Tests
# ============================================================================

class TestMakeUnique:
    """Test make_unique method with various PyTree structures."""

    def test_simple_dict(self):
        """Test with a simple dictionary."""
        state1 = ParamState(jnp.ones((2, 3)))
        state2 = ParamState(jnp.zeros((3, 4)))
        pytree = {'weight': state1, 'bias': state2}

        manager = UniqueStateManager()
        result = manager.make_unique(pytree)

        assert manager.num_unique_states == 2
        assert isinstance(result, dict)

    def test_nested_dict(self):
        """Test with nested dictionaries."""
        state1 = ParamState(jnp.ones((2, 3)))
        state2 = ParamState(jnp.zeros((3, 4)))
        pytree = {
            'layer1': {'weight': state1, 'bias': state2},
            'layer2': {'weight': ParamState(jnp.eye(3))}
        }

        manager = UniqueStateManager()
        result = manager.make_unique(pytree)

        assert manager.num_unique_states == 3

    def test_deeply_nested_structure(self):
        """Test with deeply nested PyTree."""
        state = ParamState(jnp.ones((5, 5)))
        pytree = {
            'a': {
                'b': {
                    'c': {
                        'd': {'state': state}
                    }
                }
            }
        }

        manager = UniqueStateManager()
        result = manager.make_unique(pytree)

        assert manager.num_unique_states == 1

    def test_duplicate_removal(self):
        """Test that duplicates are correctly removed."""
        shared = ParamState(jnp.ones((3, 3)))
        pytree = {
            'a': shared,
            'b': shared,
            'c': shared,
            'd': ParamState(jnp.zeros((2, 2)))
        }

        manager = UniqueStateManager()
        result = manager.make_unique(pytree)

        assert manager.num_unique_states == 2
        assert id(shared) in manager.seen_ids

    def test_tied_parameters(self):
        """Test handling of tied parameters (common in transformers)."""
        embedding = ParamState(jnp.ones((1000, 128)))
        pytree = {
            'embedding': {'weight': embedding},
            'output': {'weight': embedding},  # Tied
            'hidden': {'weight': ParamState(jnp.ones((128, 256)))}
        }

        manager = UniqueStateManager()
        result = manager.make_unique(pytree)

        # Should have 2 unique states (embedding used twice, hidden once)
        assert manager.num_unique_states == 2

    def test_empty_pytree(self):
        """Test with empty pytree."""
        manager = UniqueStateManager()
        result = manager.make_unique({})

        assert manager.num_unique_states == 0
        assert result == {}

    def test_single_state(self):
        """Test with single state."""
        state = ParamState(jnp.ones((3, 3)))
        pytree = {'state': state}

        manager = UniqueStateManager()
        result = manager.make_unique(pytree)

        assert manager.num_unique_states == 1
        assert id(state) in manager.seen_ids

    def test_multiple_calls_reset(self):
        """Test that calling make_unique multiple times resets state."""
        state1 = ParamState(jnp.ones((2, 2)))
        state2 = ParamState(jnp.zeros((3, 3)))

        manager = UniqueStateManager()

        # First call
        manager.make_unique({'a': state1})
        assert manager.num_unique_states == 1

        # Second call should reset
        manager.make_unique({'b': state2})
        assert manager.num_unique_states == 1
        assert id(state1) not in manager.seen_ids
        assert id(state2) in manager.seen_ids


# ============================================================================
# PyTree Reconstruction Tests
# ============================================================================

class TestPyTreeReconstruction:
    """Test pytree reconstruction and recovery methods."""

    def test_to_pytree_simple(self):
        """Test to_pytree with simple structure."""
        state1 = ParamState(jnp.ones((2, 3)))
        state2 = ParamState(jnp.zeros((3, 4)))
        pytree = {'a': state1, 'b': state2}

        manager = UniqueStateManager(pytree)
        recovered = manager.to_pytree()

        assert isinstance(recovered, dict)
        assert 'a' in recovered
        assert 'b' in recovered

    def test_to_pytree_nested(self):
        """Test to_pytree with nested structure."""
        state = ParamState(jnp.ones((3, 3)))
        pytree = {
            'layer1': {'weight': state},
            'layer2': {'bias': ParamState(jnp.zeros((3,)))}
        }

        manager = UniqueStateManager(pytree)
        recovered = manager.to_pytree()

        assert 'layer1' in recovered
        assert 'layer2' in recovered
        assert 'weight' in recovered['layer1']
        assert 'bias' in recovered['layer2']

    def test_to_pytree_empty(self):
        """Test to_pytree with empty manager."""
        manager = UniqueStateManager()
        result = manager.to_pytree()

        assert result == {}

    def test_to_pytree_value(self):
        """Test to_pytree_value extracts State.value."""
        state1 = ParamState(jnp.ones((2, 3)))
        state2 = ParamState(jnp.zeros((3, 4)))
        pytree = {'a': state1, 'b': state2}

        manager = UniqueStateManager(pytree)
        value_tree = manager.to_pytree_value()

        assert isinstance(value_tree, dict)
        assert 'a' in value_tree
        assert 'b' in value_tree
        # Values should be arrays, not State objects
        assert isinstance(value_tree['a'], jnp.ndarray)
        assert isinstance(value_tree['b'], jnp.ndarray)
        assert jnp.array_equal(value_tree['a'], state1.value)
        assert jnp.array_equal(value_tree['b'], state2.value)

    def test_to_pytree_value_empty(self):
        """Test to_pytree_value with empty manager."""
        manager = UniqueStateManager()
        result = manager.to_pytree_value()

        assert result == {}


# ============================================================================
# State Management Tests
# ============================================================================

class TestStateManagement:
    """Test state management methods (update, get, merge)."""

    def test_get_state_by_path(self):
        """Test retrieving state by path."""
        state1 = ParamState(jnp.ones((2, 3)))
        state2 = ParamState(jnp.zeros((3, 4)))
        pytree = {'a': state1, 'b': state2}

        manager = UniqueStateManager(pytree)

        # Get first state's path
        path = manager.unique_paths[0]
        retrieved = manager.get_state_by_path(path)

        assert retrieved is not None
        assert isinstance(retrieved, State)

    def test_get_state_by_path_not_found(self):
        """Test getting state with non-existent path."""
        state = ParamState(jnp.ones((2, 3)))
        pytree = {'a': state}

        manager = UniqueStateManager(pytree)

        # Use a fake path
        fake_path = (('nonexistent',),)
        result = manager.get_state_by_path(fake_path)

        assert result is None

    def test_update_state(self):
        """Test updating a state."""
        old_state = ParamState(jnp.ones((3, 3)))
        pytree = {'weight': old_state}

        manager = UniqueStateManager(pytree)
        path = manager.unique_paths[0]

        # Update with new state
        new_state = ParamState(jnp.ones((3, 3)) * 2)
        success = manager.update_state(path, new_state)

        assert success is True
        retrieved = manager.get_state_by_path(path)
        assert id(retrieved) == id(new_state)
        assert jnp.allclose(retrieved.value, jnp.ones((3, 3)) * 2)

    def test_update_state_changes_seen_ids(self):
        """Test that update_state correctly manages seen_ids."""
        old_state = ParamState(jnp.ones((3, 3)))
        pytree = {'weight': old_state}

        manager = UniqueStateManager(pytree)
        old_id = id(old_state)
        path = manager.unique_paths[0]

        assert old_id in manager.seen_ids

        # Update with new state
        new_state = ParamState(jnp.zeros((3, 3)))
        new_id = id(new_state)
        manager.update_state(path, new_state)

        assert new_id in manager.seen_ids
        assert old_id not in manager.seen_ids

    def test_update_state_not_found(self):
        """Test updating with non-existent path."""
        state = ParamState(jnp.ones((2, 3)))
        pytree = {'a': state}

        manager = UniqueStateManager(pytree)

        fake_path = (('nonexistent',),)
        new_state = ParamState(jnp.zeros((2, 3)))
        success = manager.update_state(fake_path, new_state)

        assert success is False

    def test_merge_with_new_states(self):
        """Test merging with completely new states."""
        state1 = ParamState(jnp.ones((2, 2)))
        state2 = ParamState(jnp.zeros((3, 3)))

        pytree1 = {'a': state1}
        pytree2 = {'b': state2}

        manager = UniqueStateManager(pytree1)
        assert manager.num_unique_states == 1

        manager.merge_with(pytree2)
        assert manager.num_unique_states == 2

    def test_merge_with_shared_states(self):
        """Test merging with some shared states."""
        shared = ParamState(jnp.ones((3, 3)))
        unique1 = ParamState(jnp.zeros((2, 2)))
        unique2 = ParamState(jnp.eye(4))

        pytree1 = {'a': shared, 'b': unique1}
        pytree2 = {'c': shared, 'd': unique2}

        manager = UniqueStateManager(pytree1)
        assert manager.num_unique_states == 2

        manager.merge_with(pytree2)
        # Should have 3 unique (shared, unique1, unique2)
        assert manager.num_unique_states == 3

    def test_merge_with_empty_pytree(self):
        """Test merging with empty pytree."""
        state = ParamState(jnp.ones((2, 2)))
        pytree = {'a': state}

        manager = UniqueStateManager(pytree)
        original_count = manager.num_unique_states

        manager.merge_with({})
        assert manager.num_unique_states == original_count

    def test_merge_returns_self(self):
        """Test that merge_with returns self for chaining."""
        state1 = ParamState(jnp.ones((2, 2)))
        state2 = ParamState(jnp.zeros((3, 3)))

        manager = UniqueStateManager({'a': state1})
        result = manager.merge_with({'b': state2})

        assert result is manager

    def test_get_flattened(self):
        """Test get_flattened returns correct structure."""
        state1 = ParamState(jnp.ones((2, 3)))
        state2 = ParamState(jnp.zeros((3, 4)))
        pytree = {'a': state1, 'b': state2}

        manager = UniqueStateManager(pytree)
        flattened = manager.get_flattened()

        assert len(flattened) == 2
        assert all(isinstance(item, tuple) for item in flattened)
        assert all(len(item) == 2 for item in flattened)


# ============================================================================
# Conversion Methods Tests
# ============================================================================

class TestConversionMethods:
    """Test conversion methods (to_dict, to_dict_value)."""

    def test_to_dict_simple(self):
        """Test to_dict with simple structure."""
        state1 = ParamState(jnp.ones((2, 3)))
        state2 = ParamState(jnp.zeros((3, 4)))
        pytree = {'a': state1, 'b': state2}

        manager = UniqueStateManager(pytree)
        result = manager.to_dict()

        assert isinstance(result, dict)
        assert len(result) == 2
        # Values should be State objects
        assert all(isinstance(v, State) for v in result.values())

    def test_to_dict_nested(self):
        """Test to_dict with nested structure."""
        state = ParamState(jnp.ones((3, 3)))
        pytree = {
            'layer1': {'weight': state},
            'layer2': {'bias': ParamState(jnp.zeros((3,)))}
        }

        manager = UniqueStateManager(pytree)
        result = manager.to_dict()

        assert isinstance(result, dict)
        assert len(result) == 2
        # Keys should be path strings
        assert all(isinstance(k, str) for k in result.keys())

    def test_to_dict_value(self):
        """Test to_dict_value extracts State.value."""
        state1 = ParamState(jnp.ones((2, 3)))
        state2 = ParamState(jnp.zeros((3, 4)))
        pytree = {'a': state1, 'b': state2}

        manager = UniqueStateManager(pytree)
        result = manager.to_dict_value()

        assert isinstance(result, dict)
        assert len(result) == 2
        # Values should be arrays, not State objects
        assert all(isinstance(v, jnp.ndarray) for v in result.values())

    def test_to_dict_empty(self):
        """Test to_dict with empty manager."""
        manager = UniqueStateManager()
        result = manager.to_dict()

        assert result == {}

    def test_to_dict_value_empty(self):
        """Test to_dict_value with empty manager."""
        manager = UniqueStateManager()
        result = manager.to_dict_value()

        assert result == {}

    def test_path_to_string_simple(self):
        """Test _path_to_string with simple paths."""
        state1 = ParamState(jnp.ones((2, 3)))
        pytree = {'layer': {'weight': state1}}

        manager = UniqueStateManager(pytree)
        path_dict = manager.to_dict()

        # Should have a path like "layer.weight"
        assert any('layer' in key for key in path_dict.keys())

    def test_path_to_string_nested(self):
        """Test _path_to_string with deeply nested paths."""
        state = ParamState(jnp.ones((2, 3)))
        pytree = {'a': {'b': {'c': state}}}

        manager = UniqueStateManager(pytree)
        path_dict = manager.to_dict()

        # Should have nested path
        assert len(path_dict) == 1


# ============================================================================
# Edge Cases and Error Conditions
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_clear(self):
        """Test clear method."""
        state = ParamState(jnp.ones((2, 3)))
        pytree = {'a': state}

        manager = UniqueStateManager(pytree)
        assert manager.num_unique_states == 1

        manager.clear()

        assert manager.num_unique_states == 0
        assert len(manager.unique_states) == 0
        assert len(manager.unique_paths) == 0
        assert len(manager.seen_ids) == 0
        assert len(manager.flattened_states) == 0
        assert manager.pytree_structure is None

    def test_clear_and_reuse(self):
        """Test clearing and reusing manager."""
        state1 = ParamState(jnp.ones((2, 3)))
        state2 = ParamState(jnp.zeros((3, 4)))

        manager = UniqueStateManager()

        # First use
        manager.make_unique({'a': state1})
        assert manager.num_unique_states == 1

        # Clear
        manager.clear()
        assert manager.num_unique_states == 0

        # Reuse
        manager.make_unique({'b': state2})
        assert manager.num_unique_states == 1
        assert id(state2) in manager.seen_ids
        assert id(state1) not in manager.seen_ids

    def test_many_duplicates(self):
        """Test handling many duplicate references."""
        state = ParamState(jnp.ones((5, 5)))

        # Create pytree with 100 references to same state
        pytree = {f'ref_{i}': state for i in range(100)}

        manager = UniqueStateManager(pytree)

        # Should still only have 1 unique state
        assert manager.num_unique_states == 1
        assert id(state) in manager.seen_ids

    def test_large_pytree(self):
        """Test with a large pytree structure."""
        # Create 50 unique states
        states = [ParamState(jnp.ones((2, 2))) for _ in range(50)]
        pytree = {f'state_{i}': states[i] for i in range(50)}

        manager = UniqueStateManager(pytree)

        assert manager.num_unique_states == 50
        assert len(manager.seen_ids) == 50

    def test_num_unique_states_property(self):
        """Test num_unique_states property."""
        manager = UniqueStateManager()
        assert manager.num_unique_states == 0

        state = ParamState(jnp.ones((2, 2)))
        manager.make_unique({'a': state})
        assert manager.num_unique_states == 1

    def test_with_nn_module(self):
        """Test integration with brainstate.nn.Module."""

        class SimpleModel(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = brainstate.nn.Linear(10, 20)
                self.l2 = brainstate.nn.Linear(20, 10)

        model = SimpleModel()
        states = model.states()

        manager = UniqueStateManager()
        manager.make_unique(states)

        # Should have unique states from the model
        assert manager.num_unique_states > 0

    def test_with_shared_module_weights(self):
        """Test with module that has shared weights."""

        class ModelWithSharedWeights(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_linear = brainstate.nn.Linear(10, 20)
                self.l1 = self.shared_linear
                self.l2 = self.shared_linear

        model = ModelWithSharedWeights()
        states = model.states()

        manager = UniqueStateManager()
        manager.make_unique(states)

        # Should deduplicate shared weights
        assert manager.num_unique_states > 0

    def test_state_values_preserved(self):
        """Test that state values are preserved through operations."""
        original_value = jnp.array([[1., 2.], [3., 4.]])
        state = ParamState(original_value)
        pytree = {'weight': state}

        manager = UniqueStateManager(pytree)
        recovered = manager.to_pytree()

        # Check that values are preserved
        assert jnp.array_equal(recovered['weight'].value, original_value)

    def test_merge_multiple_times(self):
        """Test merging multiple pytrees sequentially."""
        states = [ParamState(jnp.ones((i + 1, i + 1))) for i in range(5)]

        manager = UniqueStateManager()

        for i, state in enumerate(states):
            manager.merge_with({f'state_{i}': state})

        assert manager.num_unique_states == 5

    def test_empty_nested_dict(self):
        """Test with nested dict containing empty dicts."""
        state = ParamState(jnp.ones((2, 2)))
        pytree = {
            'layer1': {},
            'layer2': {'weight': state},
            'layer3': {}
        }

        manager = UniqueStateManager()
        manager.make_unique(pytree)

        assert manager.num_unique_states == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with realistic use cases."""

    def test_optimizer_workflow(self):
        """Test typical optimizer workflow with tied parameters."""
        # Simulate transformer with tied embeddings
        embedding_weight = ParamState(jnp.ones((1000, 128)))

        params = {
            'embedding': {'weight': embedding_weight},
            'encoder': {
                'layer1': {
                    'weight': ParamState(jnp.ones((128, 256))),
                    'bias': ParamState(jnp.zeros((256,)))
                }
            },
            'decoder': {
                'layer1': {
                    'weight': ParamState(jnp.ones((256, 128))),
                    'bias': ParamState(jnp.zeros((128,)))
                }
            },
            'output': {'weight': embedding_weight}  # Tied
        }

        manager = UniqueStateManager(params)

        # Should have 5 unique states (tied embedding counted once)
        assert manager.num_unique_states == 5

        # Convert to dict for gradient updates
        state_dict = manager.to_dict_value()
        assert len(state_dict) == 5

    def test_model_states_extraction(self):
        """Test extracting states from a neural network model."""

        class TestModel(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = brainstate.nn.Linear(10, 20)
                self.l2 = [
                    brainstate.nn.Linear(20, 20),
                    brainstate.nn.Linear(20, 20),
                ]
                self.l3 = brainstate.nn.Linear(20, 10)

        model = TestModel()
        manager = UniqueStateManager()
        states = model.states()
        manager.make_unique(states)

        # Model should have unique states
        assert manager.num_unique_states > 0

        # Should be able to convert to pytree
        pytree = manager.to_pytree()
        assert isinstance(pytree, dict)

    def test_gradient_accumulation_scenario(self):
        """Test scenario for gradient accumulation with unique states."""
        # Create model params
        state1 = ParamState(jnp.ones((10, 10)))
        state2 = ParamState(jnp.zeros((10,)))

        params = {
            'layer1': {'w': state1, 'b': state2},
            'layer2': {'w': state1}  # Shared weight
        }

        manager = UniqueStateManager(params)

        # Simulate gradient computation only on unique params
        unique_grads = {}
        for path, state in manager.get_flattened():
            # Compute "gradient"
            grad = jnp.ones_like(state.value) * 0.01
            path_str = manager._path_to_string(path)
            unique_grads[path_str] = grad

        # Should have 2 unique gradients (w and b)
        assert len(unique_grads) == 2
