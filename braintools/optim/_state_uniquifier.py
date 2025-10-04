# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from typing import Set, Tuple, Any, List, Dict

import jax.tree
from brainstate import State
from brainstate.typing import PyTree

__all__ = [
    'UniqueStateManager',
]


class UniqueStateManager:
    """
    A class to manage unique State objects in a PyTree structure.

    This class:
    1. Flattens a PyTree with State leaves to (path, leaf) pairs
    2. Removes duplicate State objects based on their id()
    3. Supports recovering the unique states back to a PyTree structure

    Example usage:
        >>> import jax.numpy as jnp
        >>> from brainstate import ParamState
        >>>
        >>> # Create a pytree with some duplicate State objects
        >>> state1 = ParamState(jnp.ones((2, 3)))
        >>> state2 = ParamState(jnp.zeros((3, 4)))
        >>>
        >>> pytree = {
        ...     'layer1': {'weight': state1, 'bias': state2},
        ...     'layer2': {'weight': state1, 'bias': ParamState(jnp.ones((4,)))}  # state1 is duplicate
        ... }
        >>>
        >>> # Create manager and process the pytree
        >>> manager = UniqueStateManager()
        >>> unique_pytree = manager.make_unique(pytree)
        >>>
        >>> # The duplicate state1 in layer2.weight will be removed
        >>> print(len(manager.flattened_states))  # Will be 3 instead of 4
        >>>
        >>> # Recover to pytree structure
        >>> recovered = manager.to_pytree()
    """
    __module__ = 'braintools.optim'

    def __init__(self, pytree: PyTree[State] = None):
        """Initialize the UniqueStateManager."""
        self.flattened_states: List[Tuple[Any, State]] = []
        self.seen_ids: Set[int] = set()
        self.pytree_structure = None
        self.unique_paths: List[Any] = []
        self.unique_states: List[State] = []
        if pytree is not None:
            self.make_unique(pytree)

    def make_unique(self, pytree: PyTree[State]) -> PyTree[State]:
        """
        Process a PyTree with State leaves and remove duplicates.

        Args:
            pytree: A PyTree where leaves are State objects

        Returns:
            A PyTree with only unique State objects (duplicates removed)
        """
        # Flatten the pytree to get (leaf, path) pairs
        leaves, treedef = jax.tree.flatten_with_path(
            pytree, is_leaf=lambda x: isinstance(x, State)
        )

        # Store the tree structure for later recovery
        self.pytree_structure = treedef

        # Reset containers
        self.flattened_states = []
        self.seen_ids = set()
        self.unique_paths = []
        self.unique_states = []

        # Process each (path, leaf) pair
        for path, leaf in leaves:
            if isinstance(leaf, State):
                leaf_id = id(leaf)
                if leaf_id not in self.seen_ids:
                    # This is a unique State object
                    self.seen_ids.add(leaf_id)
                    self.flattened_states.append((path, leaf))
                    self.unique_paths.append(path)
                    self.unique_states.append(leaf)

        # Create a new pytree with only unique states
        if self.unique_states:
            return self._reconstruct_pytree(self.unique_paths, self.unique_states)
        else:
            return {}

    def _reconstruct_pytree(
        self,
        paths: List[Any],
        leaves: List[State]
    ) -> PyTree[State]:
        """
        Reconstruct a PyTree from paths and leaves.

        Args:
            paths: List of paths to leaves
            leaves: List of State objects

        Returns:
            Reconstructed PyTree
        """
        # Build a dictionary from paths
        result = {}

        for path, leaf in zip(paths, leaves):
            # Convert path to nested dictionary keys
            current = result
            path_keys = []

            # Extract keys from the path
            for key in path:
                if hasattr(key, 'key'):
                    # Handle jax.tree_util.DictKey or similar
                    path_keys.append(key.key)
                elif hasattr(key, 'idx'):
                    # Handle jax.tree_util.SequenceKey or similar
                    path_keys.append(key.idx)
                else:
                    # Direct key
                    path_keys.append(key)

            # Navigate/create nested structure
            for key in path_keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the leaf at the final position
            if path_keys:
                current[path_keys[-1]] = leaf

        return result

    def to_pytree(self) -> PyTree[State]:
        """
        Convert the stored unique states back to a PyTree structure.

        Returns:
            PyTree with unique State objects
        """
        if not self.unique_states:
            return {}

        return self._reconstruct_pytree(self.unique_paths, self.unique_states)

    def to_pytree_value(self) -> PyTree:
        """
        Convert the stored unique states to a PyTree with State.value as leaves.

        Returns:
            PyTree where leaves are the values (State.value) of the State objects
        """
        if not self.unique_states:
            return {}

        # Extract values from State objects
        state_values = [state.value for state in self.unique_states]

        return self._reconstruct_pytree(self.unique_paths, state_values)

    def to_dict(self) -> Dict[str, State]:
        """
        Convert the stored unique states to a dictionary with path strings as keys.

        Returns:
            Dictionary where keys are string representations of paths and values are State objects
        """
        result = {}
        for path, state in zip(self.unique_paths, self.unique_states):
            # Convert path to a string key
            path_str = self._path_to_string(path)
            result[path_str] = state
        return result

    def to_dict_value(self) -> Dict[str, Any]:
        """
        Convert the stored unique states to a dictionary with path strings as keys and State.value as values.

        Returns:
            Dictionary where keys are string representations of paths and values are State.value
        """
        result = {}
        for path, state in zip(self.unique_paths, self.unique_states):
            # Convert path to a string key
            path_str = self._path_to_string(path)
            result[path_str] = state.value
        return result

    def _path_to_string(self, path: Tuple) -> str:
        """
        Convert a path tuple to a string representation.

        Args:
            path: Tuple representing the path to a leaf

        Returns:
            String representation of the path (e.g., "layer1.weight")
        """
        path_parts = []
        for key in path:
            if hasattr(key, 'key'):
                # Handle jax.tree_util.DictKey or similar
                path_parts.append(str(key.key))
            elif hasattr(key, 'idx'):
                # Handle jax.tree_util.SequenceKey or similar
                path_parts.append(f"[{key.idx}]")
            else:
                # Direct key
                path_parts.append(str(key))

        # Join with dots, but handle list indices specially
        result = ""
        for i, part in enumerate(path_parts):
            if part.startswith("["):
                result += part  # Add list index directly
            elif i == 0:
                result = part  # First element
            else:
                result += "." + part  # Add with dot separator

        return result

    def get_flattened(self) -> List[Tuple[Any, State]]:
        """
        Get the flattened list of (path, state) pairs.

        Returns:
            List of tuples containing (path, State) for unique states
        """
        return self.flattened_states

    def get_state_by_path(self, target_path: Any) -> State:
        """
        Retrieve a State object by its path.

        Args:
            target_path: The path to the desired State

        Returns:
            The State object at the given path, or None if not found
        """
        for path, state in self.flattened_states:
            if path == target_path:
                return state
        return None

    def update_state(
        self,
        target_path: Any,
        new_state: State
    ) -> bool:
        """
        Update a State object at a specific path.

        Args:
            target_path: The path to the State to update
            new_state: The new State object

        Returns:
            True if update was successful, False if path not found
        """
        for i, (path, state) in enumerate(self.flattened_states):
            if path == target_path:
                # Check if new_state is truly new (different id)
                new_id = id(new_state)
                old_id = id(state)

                # Remove old id from seen set
                self.seen_ids.discard(old_id)

                # Add new id and update lists
                self.seen_ids.add(new_id)
                self.flattened_states[i] = (path, new_state)
                self.unique_states[i] = new_state
                return True
        return False

    def merge_with(self, other_pytree: PyTree[State]) -> 'UniqueStateManager':
        """
        Merge another PyTree with the current unique states, maintaining uniqueness.

        Args:
            other_pytree: Another PyTree with State leaves to merge

        Returns:
            Merged PyTree with all unique State objects
        """
        # Flatten the other pytree
        other_leaves, _ = jax.tree.flatten_with_path(
            other_pytree,
            is_leaf=lambda x: isinstance(x, State)
        )

        # Add new unique states
        for path, leaf in other_leaves:
            if isinstance(leaf, State):
                leaf_id = id(leaf)
                if leaf_id not in self.seen_ids:
                    self.seen_ids.add(leaf_id)
                    self.flattened_states.append((path, leaf))
                    self.unique_paths.append(path)
                    self.unique_states.append(leaf)

        # Return the merged pytree
        return self

    @property
    def num_unique_states(self) -> int:
        """Get the number of unique State objects."""
        return len(self.unique_states)

    def clear(self):
        """Clear all stored states and reset the manager."""
        self.flattened_states.clear()
        self.seen_ids.clear()
        self.unique_paths.clear()
        self.unique_states.clear()
        self.pytree_structure = None

    def __len__(self) -> int:
        """Return the number of unique states."""
        return len(self.unique_states)
