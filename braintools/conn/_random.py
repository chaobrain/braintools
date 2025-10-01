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
Point neuron connectivity patterns.

This module provides connectivity patterns specifically designed for point
neuron models (single-compartment integrate-and-fire neurons). These patterns
focus on synaptic connections between individual neurons with realistic
biological constraints and dynamics.

Key Features:
- Synaptic weight and delay modeling
- Spatial connectivity patterns
- Topological network structures
- Dale's principle enforcement
- Degree constraints and pruning
"""

from typing import Optional, Tuple, Union

import brainunit as u
import numpy as np

from braintools.init._init_base import init_call, Initializer
from ._base import PointNeuronConnectivity, ConnectionResult

__all__ = [
    # Basic connectivity patterns
    'Random',
    'FixedProbability',
]


class Random(PointNeuronConnectivity):
    """Random connectivity with fixed connection probability.

    This is the fundamental random connectivity pattern for point neurons,
    where each potential connection is made with a fixed probability.

    Parameters
    ----------
    prob : float
        Connection probability between 0 and 1.
    allow_self_connections : bool
        Whether to allow neurons to connect to themselves.
    weight : Initializer, optional
        Weight initialization. Can be:
        - Initialization class (e.g., Normal, LogNormal, Constant)
        - Scalar value (float/int, will use nS units)
        - Quantity scalar or array
        - Array-like values
        If None, no weights are generated.
    delay : Initializer, optional
        Delay initialization. Can be:
        - Initialization class (e.g., ConstantDelay, UniformDelay)
        - Scalar value (float/int, will use ms units)
        - Quantity scalar or array
        - Array-like values
        If None, no delays are generated.
    seed : int, optional
        Random seed for reproducible results.

    Examples
    --------
    Basic random connectivity:

    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import Random
        >>> from braintools.init import Constant
        >>>
        >>> # With weights and delays
        >>> conn = Random(
        ...     prob=0.1,
        ...     weight=Constant(2.0 * u.nS),
        ...     delay=Constant(1.0 * u.ms),
        ...     seed=42
        ... )
        >>> result = conn(pre_size=1000, post_size=1000)
        >>>
        >>> # Topology only (no weights or delays)
        >>> topology_only = Random(prob=0.1, seed=42)
        >>> result = topology_only(pre_size=1000, post_size=1000)
        >>>
        >>> # Using scalar values (automatic units)
        >>> simple_conn = Random(prob=0.1, weight=2.5, delay=1.0, seed=42)
        >>> result = simple_conn(pre_size=1000, post_size=1000)

    Random with realistic synaptic weights:

    .. code-block:: python

        >>> from braintools.init import LogNormal, Normal
        >>>
        >>> # AMPA-like excitatory synapses
        >>> ampa_conn = Random(
        ...     prob=0.05,
        ...     weight=LogNormal(mean=1.0 * u.nS, std=0.5 * u.nS),
        ...     delay=Normal(mean=1.5 * u.ms, std=0.3 * u.ms)
        ... )

    Inhibitory connections with Dale's principle:

    .. code-block:: python

        >>> from braintools.init import Normal, Constant
        >>>
        >>> # GABA-like inhibitory synapses
        >>> gaba_conn = Random(
        ...     prob=0.08,
        ...     weight=Normal(mean=-0.8 * u.nS, std=0.2 * u.nS),
        ...     delay=Constant(0.8 * u.ms)
        ... )
    """

    __module__ = 'braintools.conn'

    def __init__(
        self,
        prob: float,
        allow_self_connections: bool = False,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.prob = prob
        self.allow_self_connections = allow_self_connections
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
        """Generate random point neuron connections."""
        if isinstance(pre_size, (tuple, list)):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, (tuple, list)):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Generate all potential connections
        pre_indices = []
        post_indices = []
        for i in range(pre_num):
            for j in range(post_num):
                if not self.allow_self_connections and i == j:
                    continue
                if self.rng.random() < self.prob:
                    pre_indices.append(i)
                    post_indices.append(j)

        n_connections = len(pre_indices)
        if n_connections == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
                model_type='point'
            )

        # Generate weights using the initialization class
        weights = init_call(
            self.weight_init,
            n_connections,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            rng=self.rng
        )

        # Generate delays using the initialization class
        delays = init_call(
            self.delay_init,
            n_connections,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            rng=self.rng
        )

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            pre_positions=pre_positions,
            post_positions=post_positions,
            model_type='point',
            metadata={
                'pattern': 'random',
                'probability': self.prob,
                'allow_self_connections': self.allow_self_connections,
                'weight_initialization': self.weight_init,
                'delay_initialization': self.delay_init,
            }
        )


# Convenience aliases for common patterns
class FixedProbability(Random):
    """Alias for Random connectivity with fixed probability."""

    __module__ = 'braintools.conn'
    pass
