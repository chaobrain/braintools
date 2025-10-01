# Copyright 2025 BrainSim Ecosystem Limited. All Rights Reserved.
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

from braintools.init._init_base import init_call, Initializer
from ._base import PointNeuronConnectivity, ConnectionResult

__all__ = [

    # Biological patterns
    'ExcitatoryInhibitory',

]


class ExcitatoryInhibitory(PointNeuronConnectivity):
    """Standard excitatory-inhibitory network following Dale's principle.

    Parameters
    ----------
    exc_ratio : float
        Fraction of neurons that are excitatory.
    exc_prob : float
        Connection probability from excitatory neurons.
    inh_prob : float
        Connection probability from inhibitory neurons.
    exc_weight : Initialization, optional
        Weight initialization for excitatory connections.
        If None, no excitatory weights are generated.
    inh_weight : Initialization, optional
        Weight initialization for inhibitory connections.
        If None, no inhibitory weights are generated.
    delay : Initialization, optional
        Delay initialization for all connections.
        If None, no delays are generated.

    Examples
    --------
    .. code-block:: python

        >>> ei_net = ExcitatoryInhibitory(
        ...     exc_ratio=0.8,
        ...     exc_prob=0.1,
        ...     inh_prob=0.2,
        ...     exc_weight=1.0 * u.nS,
        ...     inh_weight=-0.8 * u.nS
        ... )
        >>> result = ei_net(pre_size=1000, post_size=1000)
    """

    def __init__(
        self,
        exc_ratio: float = 0.8,
        exc_prob: float = 0.1,
        inh_prob: float = 0.2,
        exc_weight: Optional[Initializer] = None,
        inh_weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.exc_ratio = exc_ratio
        self.exc_prob = exc_prob
        self.inh_prob = inh_prob
        self.exc_weight_init = exc_weight
        self.inh_weight_init = inh_weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate excitatory-inhibitory network."""
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

        # Determine which neurons are excitatory vs inhibitory
        n_exc = int(pre_num * self.exc_ratio)

        # Vectorized generation for excitatory connections
        exc_random = self.rng.random((n_exc, post_num))
        exc_mask = exc_random < self.exc_prob
        exc_pre, exc_post = np.where(exc_mask)

        # Vectorized generation for inhibitory connections
        n_inh = pre_num - n_exc
        inh_random = self.rng.random((n_inh, post_num))
        inh_mask = inh_random < self.inh_prob
        inh_pre, inh_post = np.where(inh_mask)
        inh_pre = inh_pre + n_exc  # Offset to correct neuron indices

        # Combine excitatory and inhibitory connections
        pre_indices = np.concatenate([exc_pre, inh_pre])
        post_indices = np.concatenate([exc_post, inh_post])
        is_excitatory = np.concatenate([np.ones(len(exc_pre), dtype=bool), np.zeros(len(inh_pre), dtype=bool)])

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                model_type='point',
                pre_positions=kwargs.get('pre_positions', None),
                post_positions=kwargs.get('post_positions', None),
            )

        n_connections = len(pre_indices)
        n_exc_conn = len(exc_pre)
        n_inh_conn = len(inh_pre)

        # Generate weights separately for excitatory and inhibitory
        exc_weights = init_call(
            self.exc_weight_init,
            n_exc_conn,
            rng=self.rng,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None)
        ) if n_exc_conn > 0 else None

        inh_weights = init_call(
            self.inh_weight_init,
            n_inh_conn,
            rng=self.rng,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None)
        ) if n_inh_conn > 0 else None

        # Combine weights in correct order
        weights = None
        if exc_weights is not None or inh_weights is not None:
            # Handle scalar and array weights
            if exc_weights is not None:
                if u.math.isscalar(exc_weights):
                    exc_weights_array = np.full(
                        n_exc_conn,
                        u.get_mantissa(exc_weights) if isinstance(exc_weights, u.Quantity) else exc_weights
                    )
                    exc_unit = u.get_unit(exc_weights) if isinstance(exc_weights, u.Quantity) else None
                else:
                    exc_weights_array = (
                        u.get_mantissa(exc_weights)
                        if isinstance(exc_weights, u.Quantity) else
                        np.asarray(exc_weights)
                    )
                    exc_unit = u.get_unit(exc_weights) if isinstance(exc_weights, u.Quantity) else None
            else:
                exc_weights_array = np.zeros(n_exc_conn)
                exc_unit = None

            if inh_weights is not None:
                if u.math.isscalar(inh_weights):
                    inh_weights_array = np.full(
                        n_inh_conn,
                        u.get_mantissa(inh_weights) if isinstance(inh_weights, u.Quantity) else inh_weights
                    )
                    inh_unit = u.get_unit(inh_weights) if isinstance(inh_weights, u.Quantity) else None
                else:
                    inh_weights_array = (
                        u.get_mantissa(inh_weights)
                        if isinstance(inh_weights, u.Quantity) else np.asarray(inh_weights)
                    )
                    inh_unit = u.get_unit(inh_weights) if isinstance(inh_weights, u.Quantity) else None
            else:
                inh_weights_array = np.zeros(n_inh_conn)
                inh_unit = None

            # Concatenate weights
            weights_array = np.concatenate([exc_weights_array, inh_weights_array])
            common_unit = exc_unit or inh_unit

            if common_unit is not None:
                weights = u.maybe_decimal(weights_array * common_unit)
            else:
                weights = weights_array

        # Generate delays
        delays = init_call(
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
            weights=weights,
            delays=delays,
            pre_size=pre_size,
            post_size=post_size,
            model_type='point',
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None),
            metadata={
                'pattern': 'excitatory_inhibitory',
                'exc_ratio': self.exc_ratio,
                'n_excitatory': n_exc,
                'n_inhibitory': n_inh,
            }
        )
