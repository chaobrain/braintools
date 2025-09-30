#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive test cases for UniqueStateManager class.
Demonstrates various use cases for managing unique State objects in PyTrees.
"""

import brainstate
import jax.numpy as jnp
from brainstate import ParamState
from braintools.optim._state_uniquifier import UniqueStateManager


def test_1_basic_uniquification():
    """Test Case 1: Basic deduplication of State objects"""
    print("\n" + "="*60)
    print("Test Case 1: Basic State Deduplication")
    print("="*60)

    # Create some State objects
    state1 = ParamState(jnp.ones((2, 3)))
    state2 = ParamState(jnp.zeros((3, 4)))
    state3 = ParamState(jnp.ones((5,)))

    # Create a pytree with duplicate references
    pytree = {
        'layer1': {
            'weight': state1,
            'bias': state2
        },
        'layer2': {
            'weight': state1,  # Duplicate reference to state1
            'bias': state3
        },
        'shared': state1  # Another duplicate reference
    }

    print(f"Original PyTree has 5 leaf nodes (3 unique State objects)")
    print(f"  state1 id: {id(state1)}")
    print(f"  state2 id: {id(state2)}")
    print(f"  state3 id: {id(state3)}")

    # Create manager and process
    manager = UniqueStateManager()
    unique_pytree = manager.make_unique(pytree)

    print(f"\nAfter deduplication:")
    print(f"  Number of unique states: {manager.num_unique_states}")
    print(f"  Unique state IDs: {manager.seen_ids}")

    # Verify the flattened structure
    for path, state in manager.get_flattened():
        print(f"  Path: {path}, State shape: {state.value.shape}")

    return manager


def test_2_nested_structure():
    """Test Case 2: Deeply nested PyTree structures"""
    print("\n" + "="*60)
    print("Test Case 2: Deeply Nested PyTree")
    print("="*60)

    # Create States
    shared_weight = ParamState(jnp.ones((10, 10)))
    unique_bias = ParamState(jnp.zeros((10,)))

    # Create deeply nested structure
    pytree = {
        'module1': {
            'submodule1': {
                'layer1': {'w': shared_weight, 'b': unique_bias},
                'layer2': {'w': shared_weight}  # Shared
            }
        },
        'module2': {
            'submodule2': {
                'layer3': {'w': shared_weight}  # Shared again
            }
        }
    }

    print("Original nested structure with shared weights")

    manager = UniqueStateManager()
    unique_pytree = manager.make_unique(pytree)

    print(f"Unique states found: {manager.num_unique_states}")
    print("Paths to unique states:")
    for path, state in manager.get_flattened():
        print(f"  {path}")

    return manager


def test_3_state_updates():
    """Test Case 3: Updating states within the manager"""
    print("\n" + "="*60)
    print("Test Case 3: State Updates")
    print("="*60)

    # Initial pytree
    state1 = ParamState(jnp.ones((3, 3)))
    state2 = ParamState(jnp.zeros((3, 3)))

    pytree = {
        'layer1': {'weight': state1},
        'layer2': {'weight': state2}
    }

    manager = UniqueStateManager()
    manager.make_unique(pytree)

    print(f"Initial states: {manager.num_unique_states}")

    # Get the path to layer1's weight
    target_path = manager.unique_paths[0]

    # Create new state and update
    new_state = ParamState(jnp.ones((3, 3)) * 2)
    success = manager.update_state(target_path, new_state)

    print(f"Update successful: {success}")
    print(f"Number of states after update: {manager.num_unique_states}")

    # Verify the update
    retrieved = manager.get_state_by_path(target_path)
    print(f"Updated state value sum: {jnp.sum(retrieved.value)}")

    return manager


def test_4_pytree_recovery():
    """Test Case 4: Recovering PyTree from unique states"""
    print("\n" + "="*60)
    print("Test Case 4: PyTree Recovery")
    print("="*60)

    # Create original pytree
    states = {
        'a': ParamState(jnp.ones((2, 2))),
        'b': ParamState(jnp.zeros((3, 3))),
        'c': ParamState(jnp.eye(4))
    }

    pytree = {
        'model': {
            'encoder': {'weight': states['a'], 'bias': states['b']},
            'decoder': {'weight': states['c'], 'bias': states['a']}  # Reuse states['a']
        }
    }

    print("Original PyTree structure created")

    manager = UniqueStateManager()
    unique_pytree = manager.make_unique(pytree)

    print(f"Unique states: {manager.num_unique_states}")

    # Recover to pytree
    recovered = manager.to_pytree()

    print("Recovered PyTree structure:")
    def print_structure(tree, indent=0):
        for key, value in tree.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                print_structure(value, indent + 2)
            else:
                print(" " * indent + f"{key}: State with shape {value.value.shape}")

    print_structure(recovered)

    print(recovered)
    return manager


def test_5_merging_pytrees():
    """Test Case 5: Merging multiple PyTrees"""
    print("\n" + "="*60)
    print("Test Case 5: Merging PyTrees")
    print("="*60)

    # First pytree
    state1 = ParamState(jnp.ones((5, 5)))
    state2 = ParamState(jnp.zeros((5, 5)))

    pytree1 = {
        'component1': {'param': state1},
        'component2': {'param': state2}
    }

    # Second pytree with some shared states
    state3 = ParamState(jnp.eye(5))

    pytree2 = {
        'component3': {'param': state1},  # Shared with pytree1
        'component4': {'param': state3}   # New state
    }

    print("Creating manager with first PyTree")
    manager = UniqueStateManager()
    manager.make_unique(pytree1)
    print(f"States from first PyTree: {manager.num_unique_states}")

    print("\nMerging with second PyTree")
    merged = manager.merge_with(pytree2)
    print(f"States after merging: {manager.num_unique_states}")

    print("All unique state paths after merge:")
    for path, state in manager.get_flattened():
        print(f"  {path}")

    return manager


def test_6_optimizer_integration():
    """Test Case 6: Integration with optimizer workflow"""
    print("\n" + "="*60)
    print("Test Case 6: Optimizer Integration (Tied Parameters)")
    print("="*60)

    # Simulate a model with tied parameters (e.g., embedding and output weights)
    embedding_weight = ParamState(brainstate.random.normal(size=(1000, 128)))

    # Model parameters with tied weights
    params = {
        'embedding': {
            'weight': embedding_weight
        },
        'encoder': {
            'weight': ParamState(brainstate.random.normal(size=(128, 256))),
            'bias': ParamState(jnp.zeros((256,)))
        },
        'decoder': {
            'weight': ParamState(brainstate.random.normal(size=(256, 128))),
            'bias': ParamState(jnp.zeros((128,)))
        },
        'output': {
            'weight': embedding_weight  # Tied with embedding
        }
    }

    print(f"Model has tied parameters (embedding and output weights)")
    print(f"Total parameter references: 6")

    # Use UniqueStateManager to handle tied parameters
    manager = UniqueStateManager()
    unique_params = manager.make_unique(params)

    print(f"Unique parameters: {manager.num_unique_states}")

    # Simulate gradient computation on unique parameters
    print("\nSimulating gradient updates on unique parameters:")
    for i, (path, param) in enumerate(manager.get_flattened()):
        # Simulate gradient
        grad = jnp.ones_like(param.value) * 0.01
        # Update would be: param.value = param.value - learning_rate * grad
        print(f"  Param {i}: shape {param.value.shape}, gradient norm: {jnp.linalg.norm(grad):.4f}")

    print("\nTied parameters will receive the same gradient update")

    return manager


def test_7_clear_and_reuse():
    """Test Case 7: Clearing and reusing the manager"""
    print("\n" + "="*60)
    print("Test Case 7: Clear and Reuse")
    print("="*60)

    manager = UniqueStateManager()

    # First use
    pytree1 = {
        'layer1': {'w': ParamState(jnp.ones((3, 3)))}
    }
    manager.make_unique(pytree1)
    print(f"First use - states: {manager.num_unique_states}")

    # Clear
    manager.clear()
    print(f"After clear - states: {manager.num_unique_states}")

    # Reuse
    pytree2 = {
        'layer2': {'w': ParamState(jnp.zeros((4, 4)))},
        'layer3': {'w': ParamState(jnp.eye(4))}
    }
    manager.make_unique(pytree2)
    print(f"After reuse - states: {manager.num_unique_states}")

    return manager


def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*60)
    print("COMPREHENSIVE UNIQUESTATEMANAGER TEST SUITE")
    print("="*60)

    results = []

    # Run all tests
    try:
        results.append(("Basic Uniquification", test_1_basic_uniquification()))
        results.append(("Nested Structure", test_2_nested_structure()))
        results.append(("State Updates", test_3_state_updates()))
        results.append(("PyTree Recovery", test_4_pytree_recovery()))
        results.append(("Merging PyTrees", test_5_merging_pytrees()))
        results.append(("Optimizer Integration", test_6_optimizer_integration()))
        results.append(("Clear and Reuse", test_7_clear_and_reuse()))

        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"All {len(results)} test cases completed successfully!")

        for name, manager in results:
            print(f"  ✓ {name}: {manager.num_unique_states} unique states")

    except Exception as e:
        print(f"\nError during tests: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()