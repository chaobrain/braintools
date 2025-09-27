# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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
Topological connectivity patterns for composable system.

Includes patterns based on network topology:
- Small-world networks
- Scale-free networks
- Modular networks
- Hierarchical networks
"""

from typing import Optional, Tuple, Union, List
import numpy as np
import brainunit as u

from ._composable_base import Connectivity, ConnectionResult

__all__ = [
    # Topological patterns
    'SmallWorld',
    'ScaleFree',
    'Modular',
    'Hierarchical',
]


class SmallWorld(Connectivity):
    """Small-world connectivity (Watts-Strogatz model).

    Parameters
    ----------
    k : int
        Mean degree (each node is connected to k nearest neighbors).
    p : float
        Rewiring probability (0 = regular, 1 = random).
    """

    __module__ = 'braintools.conn'

    def __init__(self, k: int, p: float, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.p = p

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate small-world connectivity."""
        if not isinstance(pre_size, int) or pre_size != post_size:
            raise ValueError("Small-world connectivity requires symmetric integer sizes")

        n = pre_size
        rng = np.random.RandomState(self.seed)

        # Start with regular ring lattice
        pre_indices = []
        post_indices = []

        # Connect each node to k/2 neighbors on each side
        for i in range(n):
            for j in range(1, self.k // 2 + 1):
                # Connect to neighbor
                neighbor = (i + j) % n
                pre_indices.append(i)
                post_indices.append(neighbor)

                # Connect to previous neighbor (for undirected)
                neighbor = (i - j) % n
                pre_indices.append(i)
                post_indices.append(neighbor)

        # Rewire with probability p
        final_pre = []
        final_post = []

        for pre, post in zip(pre_indices, post_indices):
            if rng.random() < self.p:
                # Rewire to random node
                new_post = rng.randint(0, n)
                # Avoid self-connections
                while new_post == pre:
                    new_post = rng.randint(0, n)
                final_pre.append(pre)
                final_post.append(new_post)
            else:
                # Keep original connection
                final_pre.append(pre)
                final_post.append(post)

        return ConnectionResult(
            np.array(final_pre, dtype=np.int64),
            np.array(final_post, dtype=np.int64),
            metadata={'pattern': 'small_world', 'k': self.k, 'p': self.p}
        )


class ScaleFree(Connectivity):
    """Scale-free connectivity (Barabási-Albert model).

    Parameters
    ----------
    m : int
        Number of edges to attach from a new node to existing nodes.
    """

    __module__ = 'braintools.conn'

    def __init__(self, m: int, **kwargs):
        super().__init__(**kwargs)
        self.m = m

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate scale-free connectivity using preferential attachment."""
        if not isinstance(pre_size, int) or pre_size != post_size:
            raise ValueError("Scale-free connectivity requires symmetric integer sizes")

        n = pre_size
        rng = np.random.RandomState(self.seed)

        # Start with m+1 nodes fully connected
        pre_indices = []
        post_indices = []

        # Initial complete graph
        for i in range(self.m + 1):
            for j in range(i + 1, self.m + 1):
                pre_indices.extend([i, j])
                post_indices.extend([j, i])

        # Track degree for preferential attachment
        degrees = np.ones(n, dtype=int) * self.m

        # Add remaining nodes with preferential attachment
        for new_node in range(self.m + 1, n):
            # Select m nodes to connect to based on their degree
            existing_nodes = list(range(new_node))
            existing_degrees = degrees[:new_node]

            # Preferential attachment: probability proportional to degree
            probs = existing_degrees / existing_degrees.sum()

            # Choose m unique nodes
            targets = rng.choice(existing_nodes, size=self.m, replace=False, p=probs)

            for target in targets:
                pre_indices.extend([new_node, target])
                post_indices.extend([target, new_node])
                degrees[new_node] += 1
                degrees[target] += 1

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            metadata={'pattern': 'scale_free', 'm': self.m}
        )


class Modular(Connectivity):
    """Modular connectivity pattern.

    Parameters
    ----------
    module_sizes : list of int
        Size of each module.
    within_module : Connectivity
        Connectivity pattern within modules.
    between_module : Connectivity
        Connectivity pattern between modules.
    """

    __module__ = 'braintools.conn'

    def __init__(self,
                 module_sizes: List[int],
                 within_module: Connectivity,
                 between_module: Connectivity,
                 **kwargs):
        super().__init__(**kwargs)
        self.module_sizes = module_sizes
        self.within_module = within_module
        self.between_module = between_module

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate modular connectivity."""
        total_size = sum(self.module_sizes)
        if isinstance(pre_size, int):
            expected_size = pre_size
        else:
            expected_size = int(np.prod(np.asarray(pre_size)))

        if total_size != expected_size:
            raise ValueError(f"Module sizes sum ({total_size}) doesn't match expected size ({expected_size})")

        all_pre_indices = []
        all_post_indices = []
        all_weights = []

        # Track module boundaries
        module_starts = [0]
        for size in self.module_sizes[:-1]:
            module_starts.append(module_starts[-1] + size)

        # Within-module connections
        for i, (start, size) in enumerate(zip(module_starts, self.module_sizes)):
            # Generate within-module connectivity
            within_result = self.within_module(size, size)

            # Offset indices to correct position
            pre_offset = within_result.pre_indices + start
            post_offset = within_result.post_indices + start

            all_pre_indices.extend(pre_offset)
            all_post_indices.extend(post_offset)

            if within_result.weights is not None:
                all_weights.extend(within_result.weights)

        # Between-module connections
        for i in range(len(self.module_sizes)):
            for j in range(i + 1, len(self.module_sizes)):
                # Generate between-module connectivity
                between_result = self.between_module(
                    self.module_sizes[i], self.module_sizes[j]
                )

                # Offset indices
                pre_offset = between_result.pre_indices + module_starts[i]
                post_offset = between_result.post_indices + module_starts[j]

                all_pre_indices.extend(pre_offset)
                all_post_indices.extend(post_offset)

                # Reverse direction
                all_pre_indices.extend(post_offset)
                all_post_indices.extend(pre_offset)

                if between_result.weights is not None:
                    all_weights.extend(between_result.weights)
                    all_weights.extend(between_result.weights)  # Reverse direction

        # Handle weights
        weights = None
        if all_weights:
            if isinstance(all_weights[0], u.Quantity):
                weights = u.math.stack(all_weights)
            else:
                weights = np.array(all_weights)

        return ConnectionResult(
            np.array(all_pre_indices, dtype=np.int64),
            np.array(all_post_indices, dtype=np.int64),
            weights=weights,
            metadata={'pattern': 'modular', 'n_modules': len(self.module_sizes), 'module_sizes': self.module_sizes}
        )


class Hierarchical(Connectivity):
    """Hierarchical connectivity pattern.

    Parameters
    ----------
    levels : list of Connectivity
        List of connectivity patterns for each hierarchical level.
    """

    __module__ = 'braintools.conn'

    def __init__(self, levels: List[Connectivity], **kwargs):
        super().__init__(**kwargs)
        self.levels = levels

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate hierarchical connectivity."""
        # This is a simplified implementation
        # A full implementation would need more sophisticated hierarchical logic
        if len(self.levels) > 0:
            return self.levels[0](pre_size, post_size, pre_positions, post_positions)
        else:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                metadata={'pattern': 'hierarchical', 'levels': len(self.levels)}
            )