# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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


from typing import Optional

import brainunit as u
import numpy as np

from braintools.init._init_base import param, Initializer
from ._base import PointConnectivity, ConnectionResult

__all__ = [
    'AllToAll',
    'OneToOne',
]


class AllToAll(PointConnectivity):
    """Fully connected network where every neuron connects to every other neuron.

    Parameters
    ----------
    include_self_connections : bool
        Whether neurons connect to themselves.
    weight : Initialization, optional
        Weight initialization for all connections.
        If None, no weights are generated.
    delay : Initialization, optional
        Delay initialization for all connections.
        If None, no delays are generated.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.init import Constant
        >>> all_to_all = AllToAll(
        ...     weight=Constant(0.5 * u.nS),
        ...     delay=Constant(1.0 * u.ms)
        ... )
        >>> result = all_to_all(pre_size=50, post_size=50)
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        include_self_connections: bool = False,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.include_self_connections = include_self_connections
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate all-to-all connections."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']

        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Vectorized generation using meshgrid
        pre_grid, post_grid = np.meshgrid(np.arange(pre_num), np.arange(post_num), indexing='ij')
        pre_indices = pre_grid.flatten()
        post_indices = post_grid.flatten()

        # Remove self-connections if needed
        if not self.include_self_connections and pre_num == post_num:
            mask = pre_indices != post_indices
            pre_indices = pre_indices[mask]
            post_indices = post_indices[mask]

        n_connections = len(pre_indices)

        # Generate weights and delays using initialization classes
        weights = param(
            self.weight_init,
            n_connections,
            rng=self.rng,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None)
        )
        delays = param(
            self.delay_init,
            n_connections,
            rng=self.rng,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None)
        )

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None),
            model_type='point',
            metadata={
                'pattern': 'all_to_all',
                'include_self_connections': self.include_self_connections,
                'weight_initialization': self.weight_init,
                'delay_initialization': self.delay_init
            }
        )


class OneToOne(PointConnectivity):
    """One-to-one connectivity where neuron i connects to neuron i.

    Parameters
    ----------
    weight : Initialization, optional
        Weight initialization for each connection.
        If None, no weights are generated.
    delay : Initialization, optional
        Delay initialization for each connection.
        If None, no delays are generated.
    circular : bool
        If True and sizes differ, use circular indexing.

    Examples
    --------
    .. code-block:: python

        >>> one_to_one = OneToOne(weight=1.5 * u.nS)
        >>> result = one_to_one(pre_size=100, post_size=100)
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        circular: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.weight_init = weight
        self.delay_init = delay
        self.circular = circular

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate one-to-one connections."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']

        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        if self.circular:
            n_connections = max(pre_num, post_num)
            pre_indices = np.arange(n_connections) % pre_num
            post_indices = np.arange(n_connections) % post_num
        else:
            n_connections = min(pre_num, post_num)
            pre_indices = np.arange(n_connections)
            post_indices = np.arange(n_connections)

        # Generate weights and delays using initialization classes
        weights = param(
            self.weight_init,
            n_connections,
            rng=self.rng,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None)
        )
        delays = param(
            self.delay_init,
            n_connections,
            rng=self.rng,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None)
        )

        return ConnectionResult(
            pre_indices.astype(np.int64),
            post_indices.astype(np.int64),
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None),
            model_type='point',
            metadata={'pattern': 'one_to_one', 'circular': self.circular}
        )
