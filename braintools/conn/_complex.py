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

from typing import Optional, Tuple

import numpy as np

from braintools._misc import set_module_as

__all__ = [
    'small_world',
    'scale_free',
]


@set_module_as('braintools.conn')
def small_world(
    n_nodes: int,
    n_neighbors: int,
    rewire_prob: float,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create small-world connectivity (Watts-Strogatz model).
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network.
    n_neighbors : int
        Number of nearest neighbors to connect in the ring lattice.
    rewire_prob : float
        Probability of rewiring each edge.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source node indices.
    post_indices : np.ndarray
        Target node indices.
    """
    rng = np.random if seed is None else np.random.RandomState(seed)

    # Start with ring lattice
    pre_list = []
    post_list = []

    for i in range(n_nodes):
        for j in range(1, n_neighbors // 2 + 1):
            pre_list.append(i)
            post_list.append((i + j) % n_nodes)
            pre_list.append(i)
            post_list.append((i - j) % n_nodes)

    pre_indices = np.array(pre_list)
    post_indices = np.array(post_list)

    # Rewiring
    n_edges = len(pre_indices)
    rewire_mask = rng.rand(n_edges) < rewire_prob

    for i in range(n_edges):
        if rewire_mask[i]:
            # Choose new target avoiding self-loops and duplicates
            available = np.arange(n_nodes)
            available = np.delete(available, pre_indices[i])
            new_target = rng.choice(available)
            post_indices[i] = new_target

    return pre_indices, post_indices


@set_module_as('braintools.conn')
def scale_free(
    n_nodes: int,
    m_edges: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create scale-free connectivity (Barabási-Albert model).
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network.
    m_edges : int
        Number of edges to attach from a new node to existing nodes.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source node indices.
    post_indices : np.ndarray
        Target node indices.
    """
    if m_edges >= n_nodes:
        raise ValueError("m_edges must be less than n_nodes")

    rng = np.random if seed is None else np.random.RandomState(seed)

    # Start with a complete graph on m_edges+1 nodes
    pre_list = []
    post_list = []

    for i in range(m_edges + 1):
        for j in range(i + 1, m_edges + 1):
            pre_list.append(i)
            post_list.append(j)
            pre_list.append(j)
            post_list.append(i)

    # Add remaining nodes using preferential attachment
    degrees = np.zeros(n_nodes)
    for i in range(m_edges + 1):
        degrees[i] = m_edges

    for new_node in range(m_edges + 1, n_nodes):
        # Preferential attachment: probability proportional to degree
        probs = degrees[:new_node] / np.sum(degrees[:new_node])
        targets = rng.choice(new_node, size=m_edges, p=probs, replace=False)

        for target in targets:
            pre_list.append(new_node)
            post_list.append(target)
            pre_list.append(target)
            post_list.append(new_node)
            degrees[target] += 1

        degrees[new_node] = m_edges

    return np.array(pre_list), np.array(post_list)
