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