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

from typing import Tuple, Union

import numpy as np

from braintools._misc import set_module_as

__all__ = [
    'all_to_all',
    'one_to_one',
    'ring',
    'grid',
]


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
