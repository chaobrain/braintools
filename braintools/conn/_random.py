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

    pre_indices_list = []
    post_indices_list = []

    for pre_idx in range(pre_num):
        samples = rng.random_sample(post_num)
        if not include_self and pre_num == post_num and post_num > 0:
            samples[pre_idx] = 1.0
        hits = np.nonzero(samples < prob)[0]
        if hits.size:
            pre_indices_list.append(np.full(hits.size, pre_idx, dtype=np.int64))
            post_indices_list.append(hits.astype(np.int64))

    if pre_indices_list:
        pre_indices = np.concatenate(pre_indices_list)
        post_indices = np.concatenate(post_indices_list)
    else:
        pre_indices = np.empty(0, dtype=np.int64)
        post_indices = np.empty(0, dtype=np.int64)

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

    if total_num < 0:
        raise ValueError("`total_num` must be non-negative")

    if pre_num == 0 or post_num == 0 or total_num == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

    rng = np.random if seed is None else np.random.RandomState(seed)

    square_no_self = (not include_self) and (pre_num == post_num)

    if square_no_self:
        if post_num <= 1:
            if total_num > 0:
                raise ValueError("No valid connections available when population size is 1 and `include_self` is False")
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

        n_per_row = post_num - 1
        n_possible = pre_num * n_per_row
        if total_num > n_possible:
            raise ValueError(f"Requested {total_num} connections but only {n_possible} are possible")

        indices = rng.choice(n_possible, size=total_num, replace=False)
        pre_indices = indices // n_per_row
        col_rank = indices % n_per_row
        post_indices = col_rank + (col_rank >= pre_indices)
    else:
        n_possible = pre_num * post_num
        if total_num > n_possible:
            raise ValueError(f"Requested {total_num} connections but only {n_possible} are possible")

        indices = rng.choice(n_possible, size=total_num, replace=False)
        pre_indices = indices // post_num
        post_indices = indices % post_num

    return pre_indices.astype(np.int64, copy=False), post_indices.astype(np.int64, copy=False)
