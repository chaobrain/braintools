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

from typing import Optional, Tuple, Union

import numpy as np

from braintools._misc import set_module_as

__all__ = [
    'random_conn',
    'fixed_prob',
    'fixed_in_degree',
    'fixed_out_degree',
    'fixed_total_num',
    'distance_prob',
    'small_world',
    'scale_free',
    'all_to_all',
    'one_to_one',
    'ring',
    'grid',
    'gaussian_conn',
]


@set_module_as('braintools.conn')
def random_conn(
    pre_size: Union[int, Tuple[int, ...]],
    post_size: Union[int, Tuple[int, ...]],
    prob: float,
    seed: Optional[int] = None,
    include_self: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Create random synaptic connectivity between populations.
    
    Parameters
    ----------
    pre_size : int or tuple of ints
        Size of presynaptic population.
    post_size : int or tuple of ints  
        Size of postsynaptic population.
    prob : float
        Connection probability between 0 and 1.
    seed : int, optional
        Random seed for reproducibility.
    include_self : bool
        Whether to allow self-connections.
        
    Returns
    -------
    pre_indices : np.ndarray
        Indices of presynaptic neurons.
    post_indices : np.ndarray
        Indices of postsynaptic neurons.
    """
    if isinstance(pre_size, int):
        pre_num = pre_size
    else:
        pre_num = int(np.prod(np.asarray(pre_size)))

    if isinstance(post_size, int):
        post_num = post_size
    else:
        post_num = int(np.prod(np.asarray(post_size)))

    rng = np.random if seed is None else np.random.RandomState(seed)

    if not include_self and pre_num == post_num:
        mask = np.ones((pre_num, post_num), dtype=bool)
        np.fill_diagonal(mask, False)
        conn_mat = (rng.rand(pre_num, post_num) < prob) & mask
    else:
        conn_mat = rng.rand(pre_num, post_num) < prob

    pre_indices, post_indices = np.where(conn_mat)
    return pre_indices, post_indices


@set_module_as('braintools.conn')
def fixed_prob(
    pre_size: Union[int, Tuple[int, ...]],
    post_size: Union[int, Tuple[int, ...]],
    prob: float,
    seed: Optional[int] = None,
    include_self: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed probability connection (alias for random_conn).
    
    Parameters
    ----------
    pre_size : int or tuple of ints
        Size of presynaptic population.
    post_size : int or tuple of ints
        Size of postsynaptic population.
    prob : float
        Connection probability between 0 and 1.
    seed : int, optional
        Random seed for reproducibility.
    include_self : bool
        Whether to allow self-connections.
        
    Returns
    -------
    pre_indices : np.ndarray
        Indices of presynaptic neurons.
    post_indices : np.ndarray
        Indices of postsynaptic neurons.
    """
    return random_conn(pre_size, post_size, prob, seed, include_self)


@set_module_as('braintools.conn')
def fixed_in_degree(
    pre_size: Union[int, Tuple[int, ...]],
    post_size: Union[int, Tuple[int, ...]],
    in_degree: int,
    seed: Optional[int] = None,
    include_self: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed in-degree connectivity where each post neuron receives exactly k inputs.
    
    Parameters
    ----------
    pre_size : int or tuple of ints
        Size of presynaptic population.
    post_size : int or tuple of ints
        Size of postsynaptic population.
    in_degree : int
        Number of incoming connections per postsynaptic neuron.
    seed : int, optional
        Random seed for reproducibility.
    include_self : bool
        Whether to allow self-connections.
        
    Returns
    -------
    pre_indices : np.ndarray
        Indices of presynaptic neurons.
    post_indices : np.ndarray
        Indices of postsynaptic neurons.
    """
    if isinstance(pre_size, int):
        pre_num = pre_size
    else:
        pre_num = int(np.prod(np.asarray(pre_size)))

    if isinstance(post_size, int):
        post_num = post_size
    else:
        post_num = int(np.prod(np.asarray(post_size)))

    rng = np.random if seed is None else np.random.RandomState(seed)

    pre_indices_list = []
    post_indices_list = []

    for post_idx in range(post_num):
        if not include_self and pre_num == post_num:
            available = np.arange(pre_num)
            available = np.delete(available, post_idx)
            selected = rng.choice(available, size=in_degree, replace=False)
        else:
            selected = rng.choice(pre_num, size=in_degree, replace=False)

        pre_indices_list.append(selected)
        post_indices_list.append(np.full(in_degree, post_idx))

    pre_indices = np.concatenate(pre_indices_list)
    post_indices = np.concatenate(post_indices_list)

    return pre_indices, post_indices


@set_module_as('braintools.conn')
def fixed_out_degree(
    pre_size: Union[int, Tuple[int, ...]],
    post_size: Union[int, Tuple[int, ...]],
    out_degree: int,
    seed: Optional[int] = None,
    include_self: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed out-degree connectivity where each pre neuron projects to exactly k targets.
    
    Parameters
    ----------
    pre_size : int or tuple of ints
        Size of presynaptic population.
    post_size : int or tuple of ints
        Size of postsynaptic population.
    out_degree : int
        Number of outgoing connections per presynaptic neuron.
    seed : int, optional
        Random seed for reproducibility.
    include_self : bool
        Whether to allow self-connections.
        
    Returns
    -------
    pre_indices : np.ndarray
        Indices of presynaptic neurons.
    post_indices : np.ndarray
        Indices of postsynaptic neurons.
    """
    if isinstance(pre_size, int):
        pre_num = pre_size
    else:
        pre_num = int(np.prod(np.asarray(pre_size)))

    if isinstance(post_size, int):
        post_num = post_size
    else:
        post_num = int(np.prod(np.asarray(post_size)))

    rng = np.random if seed is None else np.random.RandomState(seed)

    pre_indices_list = []
    post_indices_list = []

    for pre_idx in range(pre_num):
        if not include_self and pre_num == post_num:
            available = np.arange(post_num)
            available = np.delete(available, pre_idx)
            selected = rng.choice(available, size=out_degree, replace=False)
        else:
            selected = rng.choice(post_num, size=out_degree, replace=False)

        pre_indices_list.append(np.full(out_degree, pre_idx))
        post_indices_list.append(selected)

    pre_indices = np.concatenate(pre_indices_list)
    post_indices = np.concatenate(post_indices_list)

    return pre_indices, post_indices


@set_module_as('braintools.conn')
def fixed_total_num(
    pre_size: Union[int, Tuple[int, ...]],
    post_size: Union[int, Tuple[int, ...]],
    total_num: int,
    seed: Optional[int] = None,
    include_self: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Create connectivity with a fixed total number of synapses.
    
    Parameters
    ----------
    pre_size : int or tuple of ints
        Size of presynaptic population.
    post_size : int or tuple of ints
        Size of postsynaptic population.
    total_num : int
        Total number of connections to create.
    seed : int, optional
        Random seed for reproducibility.
    include_self : bool
        Whether to allow self-connections.
        
    Returns
    -------
    pre_indices : np.ndarray
        Indices of presynaptic neurons.
    post_indices : np.ndarray
        Indices of postsynaptic neurons.
    """
    if isinstance(pre_size, int):
        pre_num = pre_size
    else:
        pre_num = int(np.prod(np.asarray(pre_size)))

    if isinstance(post_size, int):
        post_num = post_size
    else:
        post_num = int(np.prod(np.asarray(post_size)))

    rng = np.random if seed is None else np.random.RandomState(seed)

    if not include_self and pre_num == post_num:
        # Create all possible connections excluding diagonal
        pre_all = np.repeat(np.arange(pre_num), post_num)
        post_all = np.tile(np.arange(post_num), pre_num)
        mask = pre_all != post_all
        pre_all = pre_all[mask]
        post_all = post_all[mask]
    else:
        # Create all possible connections
        pre_all = np.repeat(np.arange(pre_num), post_num)
        post_all = np.tile(np.arange(post_num), pre_num)

    # Randomly select connections
    n_possible = len(pre_all)
    if total_num > n_possible:
        raise ValueError(f"Requested {total_num} connections but only {n_possible} are possible")

    indices = rng.choice(n_possible, size=total_num, replace=False)

    return pre_all[indices], post_all[indices]


@set_module_as('braintools.conn')
def distance_prob(
    pre_pos: np.ndarray,
    post_pos: np.ndarray,
    prob_func,
    seed: Optional[int] = None,
    include_self: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Distance-dependent probabilistic connectivity.
    
    Parameters
    ----------
    pre_pos : np.ndarray
        Positions of presynaptic neurons, shape (n_pre, n_dims).
    post_pos : np.ndarray
        Positions of postsynaptic neurons, shape (n_post, n_dims).
    prob_func : callable
        Function that maps distance to connection probability.
    seed : int, optional
        Random seed for reproducibility.
    include_self : bool
        Whether to allow self-connections.
        
    Returns
    -------
    pre_indices : np.ndarray
        Indices of presynaptic neurons.
    post_indices : np.ndarray
        Indices of postsynaptic neurons.
    """
    pre_num = pre_pos.shape[0]
    post_num = post_pos.shape[0]

    rng = np.random if seed is None else np.random.RandomState(seed)

    # Compute pairwise distances
    dist_matrix = np.sqrt(np.sum((pre_pos[:, None, :] - post_pos[None, :, :]) ** 2, axis=-1))

    # Apply probability function
    prob_matrix = prob_func(dist_matrix)

    # Generate connections based on probabilities
    conn_matrix = rng.rand(pre_num, post_num) < prob_matrix

    if not include_self and pre_num == post_num:
        np.fill_diagonal(conn_matrix, False)

    pre_indices, post_indices = np.where(conn_matrix)

    return pre_indices, post_indices


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


@set_module_as('braintools.conn')
def all_to_all(
    pre_size: Union[int, Tuple[int, ...]],
    post_size: Union[int, Tuple[int, ...]],
    include_self: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """All-to-all connectivity.
    
    Parameters
    ----------
    pre_size : int or tuple of ints
        Size of presynaptic population.
    post_size : int or tuple of ints
        Size of postsynaptic population.
    include_self : bool
        Whether to allow self-connections.
        
    Returns
    -------
    pre_indices : np.ndarray
        Indices of presynaptic neurons.
    post_indices : np.ndarray
        Indices of postsynaptic neurons.
    """
    if isinstance(pre_size, int):
        pre_num = pre_size
    else:
        pre_num = int(np.prod(np.asarray(pre_size)))

    if isinstance(post_size, int):
        post_num = post_size
    else:
        post_num = int(np.prod(np.asarray(post_size)))

    pre_indices = np.repeat(np.arange(pre_num), post_num)
    post_indices = np.tile(np.arange(post_num), pre_num)

    if not include_self and pre_num == post_num:
        mask = pre_indices != post_indices
        pre_indices = pre_indices[mask]
        post_indices = post_indices[mask]

    return pre_indices, post_indices


@set_module_as('braintools.conn')
def one_to_one(
    size: Union[int, Tuple[int, ...]]
) -> Tuple[np.ndarray, np.ndarray]:
    """One-to-one connectivity.
    
    Parameters
    ----------
    size : int or tuple of ints
        Size of the population.
        
    Returns
    -------
    pre_indices : np.ndarray
        Indices of presynaptic neurons.
    post_indices : np.ndarray
        Indices of postsynaptic neurons.
    """
    if isinstance(size, int):
        num = size
    else:
        num = int(np.prod(np.asarray(size)))

    indices = np.arange(num)
    return indices, indices


@set_module_as('braintools.conn')
def ring(
    n_nodes: int,
    n_neighbors: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Ring connectivity where each node connects to k nearest neighbors.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the ring.
    n_neighbors : int
        Number of neighbors on each side to connect to.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source node indices.
    post_indices : np.ndarray
        Target node indices.
    """
    pre_list = []
    post_list = []

    for i in range(n_nodes):
        for j in range(1, n_neighbors + 1):
            pre_list.append(i)
            post_list.append((i + j) % n_nodes)
            pre_list.append(i)
            post_list.append((i - j) % n_nodes)

    return np.array(pre_list), np.array(post_list)


@set_module_as('braintools.conn')
def grid(
    grid_shape: Tuple[int, ...],
    n_neighbors: int = 4,
    periodic: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Grid connectivity in n dimensions.
    
    Parameters
    ----------
    grid_shape : tuple of ints
        Shape of the grid (e.g., (10, 10) for 2D grid).
    n_neighbors : int
        Number of neighbors to connect (4 or 8 for 2D).
    periodic : bool
        Whether to use periodic boundary conditions.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source node indices.
    post_indices : np.ndarray
        Target node indices.
    """
    n_dims = len(grid_shape)
    n_nodes = int(np.prod(np.array(grid_shape)))

    pre_list = []
    post_list = []

    # Convert linear index to multi-index and vice versa
    def linear_to_multi(idx):
        multi = []
        for dim in reversed(grid_shape):
            multi.append(idx % dim)
            idx //= dim
        return list(reversed(multi))

    def multi_to_linear(multi):
        idx = 0
        for i, coord in enumerate(multi):
            idx = idx * grid_shape[i] + coord
        return idx

    # Generate connections
    for node_idx in range(n_nodes):
        coords = linear_to_multi(node_idx)

        # Connect to neighbors
        if n_dims == 2:
            # 4-connectivity
            offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if n_neighbors == 8:
                # 8-connectivity  
                offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        elif n_dims == 3:
            # 6-connectivity
            offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                       (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        else:
            # General case: connect along each dimension
            offsets = []
            for dim in range(n_dims):
                offset_pos = [0] * n_dims
                offset_pos[dim] = 1
                offsets.append(tuple(offset_pos))
                offset_neg = [0] * n_dims
                offset_neg[dim] = -1
                offsets.append(tuple(offset_neg))

        for offset in offsets:
            neighbor_coords = [coords[i] + offset[i] for i in range(n_dims)]

            # Check boundaries
            valid = True
            for i in range(n_dims):
                if periodic:
                    neighbor_coords[i] = neighbor_coords[i] % grid_shape[i]
                else:
                    if neighbor_coords[i] < 0 or neighbor_coords[i] >= grid_shape[i]:
                        valid = False
                        break

            if valid:
                neighbor_idx = multi_to_linear(neighbor_coords)
                pre_list.append(node_idx)
                post_list.append(neighbor_idx)

    return np.array(pre_list), np.array(post_list)


@set_module_as('braintools.conn')
def gaussian_conn(
    pre_pos: np.ndarray,
    post_pos: np.ndarray,
    sigma: float,
    max_prob: float = 1.0,
    seed: Optional[int] = None,
    include_self: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian distance-dependent connectivity.
    
    Connection probability decreases as a Gaussian function of distance.
    
    Parameters
    ----------
    pre_pos : np.ndarray
        Positions of presynaptic neurons, shape (n_pre, n_dims).
    post_pos : np.ndarray
        Positions of postsynaptic neurons, shape (n_post, n_dims).
    sigma : float
        Standard deviation of the Gaussian.
    max_prob : float
        Maximum connection probability (at zero distance).
    seed : int, optional
        Random seed for reproducibility.
    include_self : bool
        Whether to allow self-connections.
        
    Returns
    -------
    pre_indices : np.ndarray
        Indices of presynaptic neurons.
    post_indices : np.ndarray
        Indices of postsynaptic neurons.
    """
    prob_func = lambda d: max_prob * np.exp(-(d ** 2) / (2 * sigma ** 2))
    return distance_prob(pre_pos, post_pos, prob_func, seed, include_self)
