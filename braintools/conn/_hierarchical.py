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
Hierarchical and structured connectivity patterns.

Includes:
- Hierarchical modular networks
- Block connectivity patterns
- Layer-wise connectivity
- Feedforward/feedback patterns
"""

from typing import List, Optional, Tuple

import numpy as np

from braintools._misc import set_module_as

__all__ = [
    'hierarchical',
    'block_connect',
    'layered_network',
    'feedforward_layers',
    'cortical_hierarchy',
    'modular_network',
]


@set_module_as('braintools.conn')
def hierarchical(
    sizes: List[int],
    forward_prob: float = 0.3,
    backward_prob: float = 0.1,
    lateral_prob: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create hierarchical connectivity between multiple layers.
    
    Parameters
    ----------
    sizes : list of int
        Size of each hierarchical layer.
    forward_prob : float
        Connection probability from layer i to layer i+1.
    backward_prob : float
        Connection probability from layer i+1 to layer i.
    lateral_prob : float
        Connection probability within the same layer.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    """
    rng = np.random if seed is None else np.random.RandomState(seed)

    pre_list = []
    post_list = []

    # Calculate cumulative indices
    cum_sizes = np.cumsum([0] + sizes)

    for i in range(len(sizes)):
        start_i = cum_sizes[i]
        end_i = cum_sizes[i + 1]

        # Lateral connections within layer
        if lateral_prob > 0:
            n_lateral = int(sizes[i] * sizes[i] * lateral_prob)
            if n_lateral > 0:
                pre_lateral = rng.randint(start_i, end_i, n_lateral)
                post_lateral = rng.randint(start_i, end_i, n_lateral)
                # Remove self-connections
                mask = pre_lateral != post_lateral
                pre_list.extend(pre_lateral[mask])
                post_list.extend(post_lateral[mask])

        # Forward connections to next layer
        if i < len(sizes) - 1 and forward_prob > 0:
            start_j = cum_sizes[i + 1]
            end_j = cum_sizes[i + 2]
            n_forward = int(sizes[i] * sizes[i + 1] * forward_prob)
            if n_forward > 0:
                pre_forward = rng.randint(start_i, end_i, n_forward)
                post_forward = rng.randint(start_j, end_j, n_forward)
                pre_list.extend(pre_forward)
                post_list.extend(post_forward)

        # Backward connections from next layer
        if i < len(sizes) - 1 and backward_prob > 0:
            start_j = cum_sizes[i + 1]
            end_j = cum_sizes[i + 2]
            n_backward = int(sizes[i + 1] * sizes[i] * backward_prob)
            if n_backward > 0:
                pre_backward = rng.randint(start_j, end_j, n_backward)
                post_backward = rng.randint(start_i, end_i, n_backward)
                pre_list.extend(pre_backward)
                post_list.extend(post_backward)

    return np.array(pre_list), np.array(post_list)


@set_module_as('braintools.conn')
def block_connect(
    block_sizes: List[int],
    within_block_prob: float = 0.3,
    between_block_prob: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create block-structured connectivity.
    
    Neurons are organized into blocks with dense within-block connectivity
    and sparse between-block connectivity.
    
    Parameters
    ----------
    block_sizes : list of int
        Size of each block.
    within_block_prob : float
        Connection probability within blocks.
    between_block_prob : float
        Connection probability between blocks.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    """
    rng = np.random if seed is None else np.random.RandomState(seed)

    pre_list = []
    post_list = []

    # Calculate cumulative indices
    cum_sizes = np.cumsum([0] + block_sizes)
    n_blocks = len(block_sizes)

    for i in range(n_blocks):
        start_i = cum_sizes[i]
        end_i = cum_sizes[i + 1]

        for j in range(n_blocks):
            start_j = cum_sizes[j]
            end_j = cum_sizes[j + 1]

            if i == j:
                # Within-block connections
                prob = within_block_prob
            else:
                # Between-block connections
                prob = between_block_prob

            n_connections = int(block_sizes[i] * block_sizes[j] * prob)

            if n_connections > 0:
                pre = rng.randint(start_i, end_i, n_connections)
                post = rng.randint(start_j, end_j, n_connections)

                if i == j:
                    # Remove self-connections within blocks
                    mask = pre != post
                    pre = pre[mask]
                    post = post[mask]

                pre_list.extend(pre)
                post_list.extend(post)

    return np.array(pre_list), np.array(post_list)


@set_module_as('braintools.conn')
def layered_network(
    layer_sizes: List[int],
    connection_probs: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create layered network with custom connectivity matrix.
    
    Parameters
    ----------
    layer_sizes : list of int
        Size of each layer.
    connection_probs : np.ndarray, optional
        Matrix of connection probabilities between layers.
        Shape should be (n_layers, n_layers).
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    """
    n_layers = len(layer_sizes)

    if connection_probs is None:
        # Default: feedforward with some recurrence
        connection_probs = np.zeros((n_layers, n_layers))
        for i in range(n_layers - 1):
            connection_probs[i, i + 1] = 0.3  # Forward
            connection_probs[i, i] = 0.1  # Recurrent
        connection_probs[-1, -1] = 0.1  # Output recurrence

    rng = np.random if seed is None else np.random.RandomState(seed)

    pre_list = []
    post_list = []

    # Calculate cumulative indices
    cum_sizes = np.cumsum([0] + layer_sizes)

    for i in range(n_layers):
        for j in range(n_layers):
            prob = connection_probs[i, j]

            if prob > 0:
                start_i = cum_sizes[i]
                end_i = cum_sizes[i + 1]
                start_j = cum_sizes[j]
                end_j = cum_sizes[j + 1]

                n_connections = int(layer_sizes[i] * layer_sizes[j] * prob)

                if n_connections > 0:
                    pre = rng.randint(start_i, end_i, n_connections)
                    post = rng.randint(start_j, end_j, n_connections)

                    if i == j:
                        # Remove self-connections
                        mask = pre != post
                        pre = pre[mask]
                        post = post[mask]

                    pre_list.extend(pre)
                    post_list.extend(post)

    return np.array(pre_list), np.array(post_list)


@set_module_as('braintools.conn')
def feedforward_layers(
    layer_sizes: List[int],
    connection_prob: float = 0.2,
    skip_connections: bool = False,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create feedforward connectivity between layers.
    
    Parameters
    ----------
    layer_sizes : list of int
        Size of each layer.
    connection_prob : float
        Base connection probability between adjacent layers.
    skip_connections : bool
        If True, add skip connections between non-adjacent layers.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    """
    n_layers = len(layer_sizes)
    connection_matrix = np.zeros((n_layers, n_layers))

    # Adjacent layer connections
    for i in range(n_layers - 1):
        connection_matrix[i, i + 1] = connection_prob

    # Skip connections
    if skip_connections:
        for i in range(n_layers - 2):
            for j in range(i + 2, n_layers):
                # Decreasing probability with distance
                distance = j - i
                skip_prob = connection_prob / (2 ** (distance - 1))
                connection_matrix[i, j] = skip_prob

    return layered_network(layer_sizes, connection_matrix, seed)


@set_module_as('braintools.conn')
def cortical_hierarchy(
    area_sizes: List[int],
    forward_prob: float = 0.2,
    backward_prob: float = 0.1,
    lateral_prob: float = 0.05,
    hierarchy_levels: Optional[List[float]] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create cortical-like hierarchical connectivity.
    
    Models connectivity between cortical areas with hierarchy-dependent
    connection strengths.
    
    Parameters
    ----------
    area_sizes : list of int
        Size of each cortical area.
    forward_prob : float
        Base forward connection probability.
    backward_prob : float
        Base backward connection probability.
    lateral_prob : float
        Base lateral connection probability.
    hierarchy_levels : list of float, optional
        Hierarchical level of each area (0=lowest, 1=highest).
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    """
    n_areas = len(area_sizes)

    if hierarchy_levels is None:
        # Default: linear hierarchy
        hierarchy_levels = np.linspace(0, 1, n_areas)

    # Build connection probability matrix based on hierarchy
    connection_matrix = np.zeros((n_areas, n_areas))

    for i in range(n_areas):
        for j in range(n_areas):
            if i == j:
                # Within-area connections (not handled here)
                continue

            h_diff = hierarchy_levels[j] - hierarchy_levels[i]

            if abs(h_diff) < 0.1:
                # Similar hierarchical level - lateral
                connection_matrix[i, j] = lateral_prob
            elif h_diff > 0:
                # Forward connection (lower to higher)
                connection_matrix[i, j] = forward_prob * (1 + h_diff)
            else:
                # Backward connection (higher to lower)
                connection_matrix[i, j] = backward_prob * (1 - h_diff)

    # Normalize probabilities
    connection_matrix = np.clip(connection_matrix, 0, 1)

    return layered_network(area_sizes, connection_matrix, seed)


@set_module_as('braintools.conn')
def modular_network(
    module_sizes: List[int],
    n_hubs: int = 0,
    within_module_prob: float = 0.3,
    between_module_prob: float = 0.05,
    hub_prob: float = 0.2,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create modular network with optional hub nodes.
    
    Parameters
    ----------
    module_sizes : list of int
        Size of each module.
    n_hubs : int
        Number of hub nodes (highly connected nodes).
    within_module_prob : float
        Connection probability within modules.
    between_module_prob : float
        Connection probability between modules.
    hub_prob : float
        Connection probability for hub nodes.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    """
    rng = np.random if seed is None else np.random.RandomState(seed)

    # First create block connectivity
    pre_indices, post_indices = block_connect(
        module_sizes, within_module_prob, between_module_prob, seed
    )

    if n_hubs > 0:
        # Add hub nodes
        total_nodes = sum(module_sizes)
        hub_start = total_nodes
        hub_end = total_nodes + n_hubs

        pre_list = list(pre_indices)
        post_list = list(post_indices)

        # Hubs connect broadly
        for hub_idx in range(hub_start, hub_end):
            # Outgoing connections from hub
            n_out = int(total_nodes * hub_prob)
            targets = rng.choice(total_nodes, n_out, replace=False)
            pre_list.extend([hub_idx] * n_out)
            post_list.extend(targets)

            # Incoming connections to hub
            n_in = int(total_nodes * hub_prob)
            sources = rng.choice(total_nodes, n_in, replace=False)
            pre_list.extend(sources)
            post_list.extend([hub_idx] * n_in)

        # Hub-to-hub connections
        if n_hubs > 1:
            n_hub_conn = int(n_hubs * n_hubs * hub_prob)
            for _ in range(n_hub_conn):
                h1 = rng.randint(hub_start, hub_end)
                h2 = rng.randint(hub_start, hub_end)
                if h1 != h2:
                    pre_list.append(h1)
                    post_list.append(h2)

        pre_indices = np.array(pre_list)
        post_indices = np.array(post_list)

    return pre_indices, post_indices
