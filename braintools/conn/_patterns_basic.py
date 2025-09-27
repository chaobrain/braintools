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
Basic connectivity patterns for composable system.

Includes fundamental patterns:
- Random connectivity
- All-to-all connectivity
- One-to-one connectivity
- Custom connectivity
"""

from typing import Optional, Tuple, Union, Callable
import numpy as np
import brainunit as u

from ._composable_base import Connectivity, ConnectionResult


__all__ = [
    # Basic patterns
    'Random',
    'AllToAll',
    'OneToOne',
    'Custom',
]

class Random(Connectivity):
    """Random connectivity with fixed connection probability.

    Parameters
    ----------
    prob : float
        Connection probability (0 to 1).
    include_self : bool
        Whether to allow self-connections.
    directed : bool
        Whether connections are directed.

    Examples
    --------
    Basic random connectivity:

    .. code-block:: python

        >>> random_conn = Random(prob=0.1, seed=42)
        >>> result = random_conn(pre_size=100, post_size=100)

    Combine with distance-dependent:

    .. code-block:: python

        >>> local = Random(prob=0.3) * DistanceDependent(sigma=50.0)
        >>> long_range = Random(prob=0.01).filter_distance(min_dist=200.0)
        >>> mixed = local + long_range
    """

    __module__ = 'braintools.conn'

    def __init__(self,
                 prob: float,
                 include_self: bool = True,
                 directed: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.prob = prob
        self.include_self = include_self
        self.directed = directed

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate random connectivity."""
        if isinstance(pre_size, int):
            pre_num = pre_size
        else:
            pre_num = int(np.prod(np.asarray(pre_size)))

        if isinstance(post_size, int):
            post_num = post_size
        else:
            post_num = int(np.prod(np.asarray(post_size)))

        rng = np.random.RandomState(self.seed)

        pre_indices_list = []
        post_indices_list = []

        for pre_idx in range(pre_num):
            samples = rng.random_sample(post_num)

            # Handle self-connections
            if not self.include_self and pre_num == post_num:
                samples[pre_idx] = 1.0  # Ensure no self-connection

            hits = np.nonzero(samples < self.prob)[0]
            if hits.size:
                pre_indices_list.append(np.full(hits.size, pre_idx, dtype=np.int64))
                post_indices_list.append(hits.astype(np.int64))

        if pre_indices_list:
            pre_indices = np.concatenate(pre_indices_list)
            post_indices = np.concatenate(post_indices_list)
        else:
            pre_indices = np.array([], dtype=np.int64)
            post_indices = np.array([], dtype=np.int64)

        return ConnectionResult(
            pre_indices,
            post_indices,
            metadata={'pattern': 'random', 'prob': self.prob, 'directed': self.directed}
        )


class AllToAll(Connectivity):
    """All-to-all connectivity pattern.

    Parameters
    ----------
    include_self : bool
        Whether to include self-connections (default: True).
    """

    __module__ = 'braintools.conn'

    def __init__(self, include_self: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.include_self = include_self

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate all-to-all connectivity."""
        if isinstance(pre_size, int):
            pre_num = pre_size
        else:
            pre_num = int(np.prod(np.asarray(pre_size)))

        if isinstance(post_size, int):
            post_num = post_size
        else:
            post_num = int(np.prod(np.asarray(post_size)))

        # Create all possible connections
        pre_indices = []
        post_indices = []

        for i in range(pre_num):
            for j in range(post_num):
                if self.include_self or i != j or pre_num != post_num:
                    pre_indices.append(i)
                    post_indices.append(j)

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            metadata={'pattern': 'all_to_all', 'include_self': self.include_self}
        )


class OneToOne(Connectivity):
    """One-to-one connectivity pattern.

    Parameters
    ----------
    offset : int
        Offset for connections (default: 0).
    """

    __module__ = 'braintools.conn'

    def __init__(self, offset: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.offset = offset

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate one-to-one connectivity."""
        if isinstance(pre_size, int):
            pre_num = pre_size
        else:
            pre_num = int(np.prod(np.asarray(pre_size)))

        if isinstance(post_size, int):
            post_num = post_size
        else:
            post_num = int(np.prod(np.asarray(post_size)))

        n_connections = min(pre_num, post_num - abs(self.offset))

        if n_connections <= 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                metadata={'pattern': 'one_to_one', 'offset': self.offset}
            )

        pre_indices = np.arange(n_connections, dtype=np.int64)
        post_indices = (pre_indices + self.offset) % post_num

        return ConnectionResult(
            pre_indices,
            post_indices,
            metadata={'pattern': 'one_to_one', 'offset': self.offset}
        )


class Custom(Connectivity):
    """Custom connectivity pattern using user-defined function.

    Parameters
    ----------
    func : callable
        Function that generates connectivity. Should accept (pre_size, post_size,
        pre_positions, post_positions) and return ConnectionResult.
    """

    __module__ = 'braintools.conn'

    def __init__(self, func: Callable, **kwargs):
        super().__init__(**kwargs)
        self.func = func

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate custom connectivity."""
        return self.func(pre_size, post_size, pre_positions, post_positions)