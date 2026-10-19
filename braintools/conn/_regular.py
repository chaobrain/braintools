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


from typing import Optional, Tuple, Union

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
        Weight initialization for all connections. Bare scalars stay
        dimensionless; pass a ``brainunit`` Quantity (e.g. ``0.5 * u.nS``)
        for physical units. If None, no weights are generated.
    delay : Initialization, optional
        Delay initialization for all connections. Bare scalars stay
        dimensionless; pass a ``brainunit`` Quantity (e.g. ``1.0 * u.ms``)
        for physical units. If None, no delays are generated.

    Notes
    -----
    The output is dense by nature (``N_pre * N_post`` edges, minus the diagonal
    when self-connections are excluded on a square block). Edges are built
    directly in row-major ``(pre, post)`` order without intermediate index
    grids. This is host-side NumPy build-time code; it is not traceable by
    ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import AllToAll
        >>> from braintools.init import Constant
        >>> all_to_all = AllToAll(
        ...     weight=Constant(0.5 * u.nS),
        ...     delay=Constant(1.0 * u.ms)
        ... )
        >>> result = all_to_all(pre_size=50, post_size=50)
        >>> result.n_connections
        2450
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

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate all-to-all connections."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Row-major (pre, post) order, built directly without meshgrid copies. For a
        # square block without self-connections, row i lists 0..n-2 with the
        # entries >= i shifted up by one, which skips the diagonal.
        drop_self = not self.include_self_connections and pre_num == post_num
        n_cols = post_num - 1 if drop_self else post_num
        n_cols = max(n_cols, 0)
        pre_indices = np.repeat(np.arange(pre_num, dtype=np.int64), n_cols)
        post_indices = np.tile(np.arange(n_cols, dtype=np.int64), pre_num)
        if drop_self:
            post_indices += post_indices >= pre_indices

        n_connections = len(pre_indices)

        # Generate weights and delays using initialization classes
        weights = param(
            self.weight_init,
            n_connections,
            rng=self.rng,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions
        )
        delays = param(
            self.delay_init,
            n_connections,
            rng=self.rng,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        return ConnectionResult(
            pre_indices,
            post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            pre_positions=pre_positions,
            post_positions=post_positions,
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
        Weight initialization for each connection. Bare scalars stay
        dimensionless; pass a ``brainunit`` Quantity (e.g. ``1.5 * u.nS``)
        for physical units. If None, no weights are generated.
    delay : Initialization, optional
        Delay initialization for each connection. Bare scalars stay
        dimensionless; pass a ``brainunit`` Quantity (e.g. ``1.0 * u.ms``)
        for physical units. If None, no delays are generated.
    circular : bool
        If True and sizes differ, use circular indexing.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import OneToOne
        >>> one_to_one = OneToOne(weight=1.5 * u.nS)
        >>> result = one_to_one(pre_size=100, post_size=100)
        >>> result.n_connections
        100
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

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate one-to-one connections."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Empty population: no connections possible (avoids division by zero
        # in circular indexing).
        if pre_num == 0 or post_num == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
                model_type='point',
                metadata={'pattern': 'one_to_one', 'circular': self.circular}
            )

        if self.circular:
            n_connections = max(pre_num, post_num)
            pre_indices = np.arange(n_connections, dtype=np.int64) % pre_num
            post_indices = np.arange(n_connections, dtype=np.int64) % post_num
        else:
            n_connections = min(pre_num, post_num)
            pre_indices = np.arange(n_connections, dtype=np.int64)
            post_indices = pre_indices

        # Generate weights and delays using initialization classes
        weights = param(
            self.weight_init,
            n_connections,
            rng=self.rng,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions
        )
        delays = param(
            self.delay_init,
            n_connections,
            rng=self.rng,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        return ConnectionResult(
            pre_indices,
            post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            pre_positions=pre_positions,
            post_positions=post_positions,
            model_type='point',
            metadata={'pattern': 'one_to_one', 'circular': self.circular}
        )
