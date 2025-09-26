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
Statistical model-based connectivity patterns.

Includes:
- Erdős-Rényi random graphs
- Stochastic block models
- Configuration models
- Power-law distributions
- Log-normal connectivity
- Exponential random graphs
"""

from typing import List, Optional, Tuple, Union

import numpy as np

from braintools._misc import set_module_as

__all__ = [
    'erdos_renyi',
    'stochastic_block_model',
    'configuration_model',
    'power_law_degree',
    'lognormal_degree',
    'exponential_random_graph',
    'degree_sequence',
    'expected_degree_model',
]


@set_module_as('braintools.conn')
def erdos_renyi(
    n_nodes: int,
    n_edges: Optional[int] = None,
    edge_prob: Optional[float] = None,
    directed: bool = False,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create Erdős-Rényi random graph connectivity.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    n_edges : int, optional
        Fixed number of edges (G(n,m) model).
    edge_prob : float, optional
        Probability of each edge (G(n,p) model).
    directed : bool
        Whether the graph is directed.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source node indices.
    post_indices : np.ndarray
        Target node indices.
    """
    if n_edges is None and edge_prob is None:
        raise ValueError("Either n_edges or edge_prob must be specified")

    rng = np.random if seed is None else np.random.RandomState(seed)

    if edge_prob is not None:
        # G(n,p) model
        if directed:
            conn_mat = rng.rand(n_nodes, n_nodes) < edge_prob
            np.fill_diagonal(conn_mat, False)
        else:
            # Symmetric for undirected
            conn_mat = np.triu(rng.rand(n_nodes, n_nodes) < edge_prob, k=1)
            conn_mat = conn_mat + conn_mat.T

        pre_indices, post_indices = np.where(conn_mat)

    else:
        # G(n,m) model
        max_edges = n_nodes * (n_nodes - 1)
        if not directed:
            max_edges //= 2

        if n_edges > max_edges:
            raise ValueError(f"Too many edges requested: {n_edges} > {max_edges}")

        # Generate all possible edges
        if directed:
            all_edges = [(i, j) for i in range(n_nodes)
                         for j in range(n_nodes) if i != j]
        else:
            all_edges = [(i, j) for i in range(n_nodes)
                         for j in range(i + 1, n_nodes)]

        # Randomly select edges
        selected = rng.choice(len(all_edges), n_edges, replace=False)
        edges = [all_edges[i] for i in selected]

        if directed:
            pre_indices = np.array([e[0] for e in edges])
            post_indices = np.array([e[1] for e in edges])
        else:
            # Add both directions for undirected
            pre_indices = np.array([e[0] for e in edges] + [e[1] for e in edges])
            post_indices = np.array([e[1] for e in edges] + [e[0] for e in edges])

    return pre_indices, post_indices


@set_module_as('braintools.conn')
def stochastic_block_model(
    block_sizes: List[int],
    prob_matrix: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create stochastic block model connectivity.
    
    Parameters
    ----------
    block_sizes : list of int
        Size of each block.
    prob_matrix : np.ndarray
        Matrix of connection probabilities between blocks.
        Shape should be (n_blocks, n_blocks).
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source node indices.
    post_indices : np.ndarray
        Target node indices.
    """
    n_blocks = len(block_sizes)
    if prob_matrix.shape != (n_blocks, n_blocks):
        raise ValueError(f"prob_matrix shape {prob_matrix.shape} doesn't match "
                         f"number of blocks {n_blocks}")

    rng = np.random if seed is None else np.random.RandomState(seed)

    pre_list = []
    post_list = []

    # Calculate cumulative indices
    cum_sizes = np.cumsum([0] + block_sizes)

    for i in range(n_blocks):
        for j in range(n_blocks):
            prob = prob_matrix[i, j]

            if prob > 0:
                start_i = cum_sizes[i]
                end_i = cum_sizes[i + 1]
                start_j = cum_sizes[j]
                end_j = cum_sizes[j + 1]

                # Generate connections
                conn_mat = rng.rand(end_i - start_i, end_j - start_j) < prob

                if i == j:
                    # Remove self-connections within blocks
                    np.fill_diagonal(conn_mat, False)

                pre, post = np.where(conn_mat)
                pre_list.extend(pre + start_i)
                post_list.extend(post + start_j)

    return np.array(pre_list), np.array(post_list)


@set_module_as('braintools.conn')
def configuration_model(
    in_degrees: np.ndarray,
    out_degrees: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create connectivity using configuration model.
    
    Generates random graph with specified degree sequence.
    
    Parameters
    ----------
    in_degrees : np.ndarray
        In-degree sequence for each node.
    out_degrees : np.ndarray, optional
        Out-degree sequence (for directed graphs).
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source node indices.
    post_indices : np.ndarray
        Target node indices.
    """
    rng = np.random if seed is None else np.random.RandomState(seed)

    n_nodes = len(in_degrees)

    if out_degrees is None:
        # Undirected graph - use in_degrees as degree sequence
        degrees = in_degrees
        if np.sum(degrees) % 2 != 0:
            raise ValueError("Sum of degrees must be even for undirected graph")

        # Create stub list
        stubs = []
        for node, degree in enumerate(degrees):
            stubs.extend([node] * int(degree))

        # Randomly pair stubs
        rng.shuffle(stubs)

        pre_indices = []
        post_indices = []

        for i in range(0, len(stubs), 2):
            if i + 1 < len(stubs):
                # Avoid self-loops
                if stubs[i] != stubs[i + 1]:
                    pre_indices.append(stubs[i])
                    post_indices.append(stubs[i + 1])
                    pre_indices.append(stubs[i + 1])
                    post_indices.append(stubs[i])

    else:
        # Directed graph
        if np.sum(in_degrees) != np.sum(out_degrees):
            raise ValueError("Sum of in-degrees must equal sum of out-degrees")

        # Create in and out stub lists
        in_stubs = []
        out_stubs = []

        for node in range(n_nodes):
            in_stubs.extend([node] * int(in_degrees[node]))
            out_stubs.extend([node] * int(out_degrees[node]))

        # Randomly pair in and out stubs
        rng.shuffle(in_stubs)
        rng.shuffle(out_stubs)

        pre_indices = []
        post_indices = []

        for out_node, in_node in zip(out_stubs, in_stubs):
            # Avoid self-loops
            if out_node != in_node:
                pre_indices.append(out_node)
                post_indices.append(in_node)

    return np.array(pre_indices), np.array(post_indices)


@set_module_as('braintools.conn')
def power_law_degree(
    n_nodes: int,
    gamma: float = 2.5,
    min_degree: int = 1,
    max_degree: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create connectivity with power-law degree distribution.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    gamma : float
        Power-law exponent (typical: 2-3).
    min_degree : int
        Minimum degree.
    max_degree : int, optional
        Maximum degree cutoff.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source node indices.
    post_indices : np.ndarray
        Target node indices.
    """
    rng = np.random if seed is None else np.random.RandomState(seed)

    if max_degree is None:
        max_degree = int(np.sqrt(n_nodes))

    # Generate power-law degree sequence
    degrees = []
    for _ in range(n_nodes):
        # Use inverse transform sampling
        u = rng.random()
        if gamma != 1:
            k = ((max_degree ** (1 - gamma) - min_degree ** (1 - gamma)) * u +
                 min_degree ** (1 - gamma)) ** (1 / (1 - gamma))
        else:
            k = min_degree * (max_degree / min_degree) ** u
        degrees.append(int(k))

    degrees = np.array(degrees)

    # Ensure sum is even for undirected graph
    if np.sum(degrees) % 2 != 0:
        degrees[rng.randint(n_nodes)] += 1

    return configuration_model(degrees, seed=seed)


@set_module_as('braintools.conn')
def lognormal_degree(
    n_nodes: int,
    mean: float = 2.0,
    std: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create connectivity with log-normal degree distribution.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    mean : float
        Mean of log-normal distribution (in log space).
    std : float
        Standard deviation of log-normal distribution (in log space).
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source node indices.
    post_indices : np.ndarray
        Target node indices.
    """
    rng = np.random if seed is None else np.random.RandomState(seed)

    # Generate log-normal degree sequence
    degrees = rng.lognormal(mean, std, n_nodes).astype(int)
    degrees = np.maximum(degrees, 1)  # Ensure at least degree 1

    # Ensure sum is even for undirected graph
    if np.sum(degrees) % 2 != 0:
        degrees[rng.randint(n_nodes)] += 1

    return configuration_model(degrees, seed=seed)


@set_module_as('braintools.conn')
def exponential_random_graph(
    n_nodes: int,
    edge_weight: float = 1.0,
    triangle_weight: float = 0.0,
    star_weight: float = 0.0,
    n_samples: int = 1000,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create connectivity using exponential random graph model (ERGM).
    
    Simplified ERGM with edge, triangle, and star statistics.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    edge_weight : float
        Weight for edge count statistic.
    triangle_weight : float
        Weight for triangle count statistic.
    star_weight : float
        Weight for star (hub) statistic.
    n_samples : int
        Number of MCMC samples.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source node indices.
    post_indices : np.ndarray
        Target node indices.
    """
    rng = np.random if seed is None else np.random.RandomState(seed)

    # Initialize with random graph
    adj_mat = rng.rand(n_nodes, n_nodes) < 0.1
    adj_mat = np.triu(adj_mat, k=1)
    adj_mat = adj_mat + adj_mat.T

    # MCMC sampling
    for _ in range(n_samples):
        # Propose edge toggle
        i = rng.randint(n_nodes)
        j = rng.randint(n_nodes)

        if i != j:
            # Calculate change in statistics
            old_val = adj_mat[i, j]
            new_val = not old_val

            # Change in edge count
            delta_edges = 1 if new_val else -1

            # Change in triangle count
            common_neighbors = np.sum(adj_mat[i, :] & adj_mat[j, :])
            delta_triangles = common_neighbors * delta_edges

            # Change in star count (simplified as degree)
            delta_stars = (np.sum(adj_mat[i, :]) + np.sum(adj_mat[j, :])) * delta_edges

            # Calculate acceptance probability
            delta_score = (edge_weight * delta_edges +
                           triangle_weight * delta_triangles +
                           star_weight * delta_stars)

            if delta_score > 0 or rng.random() < np.exp(delta_score):
                adj_mat[i, j] = new_val
                adj_mat[j, i] = new_val

    pre_indices, post_indices = np.where(adj_mat)

    return pre_indices, post_indices


@set_module_as('braintools.conn')
def degree_sequence(
    degrees: Union[np.ndarray, List[int]],
    method: str = 'configuration',
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create graph with specified degree sequence.
    
    Parameters
    ----------
    degrees : array-like
        Desired degree sequence.
    method : str
        Method to use ('configuration', 'havel-hakimi').
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source node indices.
    post_indices : np.ndarray
        Target node indices.
    """
    degrees = np.asarray(degrees)

    if method == 'configuration':
        return configuration_model(degrees, seed=seed)

    elif method == 'havel-hakimi':
        # Havel-Hakimi algorithm for simple graphs
        n_nodes = len(degrees)
        adj_mat = np.zeros((n_nodes, n_nodes), dtype=bool)

        # Sort nodes by degree
        node_degrees = [(i, d) for i, d in enumerate(degrees)]

        while True:
            # Sort by degree (descending)
            node_degrees.sort(key=lambda x: x[1], reverse=True)

            # Check if done
            if node_degrees[0][1] == 0:
                break

            # Connect highest degree node
            v, d = node_degrees[0]

            if d > len(node_degrees) - 1:
                raise ValueError("Degree sequence is not graphical")

            # Connect to next d highest degree nodes
            for i in range(1, d + 1):
                u, du = node_degrees[i]
                adj_mat[v, u] = True
                adj_mat[u, v] = True
                node_degrees[i] = (u, du - 1)

            node_degrees[0] = (v, 0)

        pre_indices, post_indices = np.where(adj_mat)
        return pre_indices, post_indices

    else:
        raise ValueError(f"Unknown method: {method}")


@set_module_as('braintools.conn')
def expected_degree_model(
    expected_degrees: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create connectivity using expected degree model.
    
    Each edge (i,j) exists with probability proportional to 
    expected_degree[i] * expected_degree[j].
    
    Parameters
    ----------
    expected_degrees : np.ndarray
        Expected degree for each node.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source node indices.
    post_indices : np.ndarray
        Target node indices.
    """
    rng = np.random if seed is None else np.random.RandomState(seed)

    n_nodes = len(expected_degrees)

    # Normalize to get probabilities
    sum_degrees = np.sum(expected_degrees)

    # Calculate edge probabilities
    prob_matrix = np.outer(expected_degrees, expected_degrees) / sum_degrees

    # Ensure probabilities are valid
    prob_matrix = np.minimum(prob_matrix, 1.0)

    # Generate edges
    conn_mat = rng.rand(n_nodes, n_nodes) < prob_matrix

    # Make undirected and remove self-loops
    conn_mat = np.triu(conn_mat, k=1)
    conn_mat = conn_mat + conn_mat.T

    pre_indices, post_indices = np.where(conn_mat)

    return pre_indices, post_indices
