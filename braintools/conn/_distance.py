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
    'distance_prob',
    'gaussian_conn',
]


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