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
    'ExcitatoryInhibitory',
]


class ExcitatoryInhibitory(PointConnectivity):
    """Standard excitatory-inhibitory network following Dale's principle.

    This connectivity pattern implements a biologically-inspired network where neurons
    are divided into excitatory and inhibitory populations. Each population has its own
    connection probability and can have distinct weights and delays. This follows Dale's
    principle that a neuron releases the same neurotransmitter(s) at all of its synapses.

    The connectivity is generated probabilistically: for each potential connection from
    a presynaptic neuron to a postsynaptic neuron, a connection is formed with probability
    ``exc_prob`` (for excitatory neurons) or ``inh_prob`` (for inhibitory neurons).

    Parameters
    ----------
    exc_ratio : float, default=0.8
        Fraction of presynaptic neurons that are excitatory. Must be between 0 and 1.
        The first ``int(pre_size * exc_ratio)`` neurons are treated as excitatory,
        and the remaining neurons are inhibitory.
    exc_prob : float, default=0.1
        Connection probability for excitatory-to-postsynaptic connections.
        Must be between 0 and 1.
    inh_prob : float, default=0.2
        Connection probability for inhibitory-to-postsynaptic connections.
        Must be between 0 and 1.
    exc_weight : Initializer, optional
        Weight initialization for excitatory connections. Can be a scalar, array,
        or Initializer object. Must be specified together with ``inh_weight``
        (both None or both specified).
    inh_weight : Initializer, optional
        Weight initialization for inhibitory connections. Can be a scalar, array,
        or Initializer object. Must be specified together with ``exc_weight``
        (both None or both specified).
    exc_delay : Initializer, optional
        Delay initialization for excitatory connections. Can be a scalar, array,
        or Initializer object. Must be specified together with ``inh_delay``
        (both None or both specified).
    inh_delay : Initializer, optional
        Delay initialization for inhibitory connections. Can be a scalar, array,
        or Initializer object. Must be specified together with ``exc_delay``
        (both None or both specified).

    Notes
    -----
    - Both ``exc_weight`` and ``inh_weight`` must be either both None or both specified.
      If only one is provided, a ValueError will be raised.
    - Similarly, both ``exc_delay`` and ``inh_delay`` must be either both None or both specified.
    - Typical cortical networks have an exc_ratio of ~0.8 (80% excitatory, 20% inhibitory).
    - Inhibitory connections often have higher connection probabilities than excitatory ones.

    Examples
    --------
    Create a basic E-I network with 80% excitatory neurons:

    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import ExcitatoryInhibitory
        >>>
        >>> ei_net = ExcitatoryInhibitory(
        ...     exc_ratio=0.8,
        ...     exc_prob=0.1,
        ...     inh_prob=0.2,
        ...     exc_weight=1.0 * u.nS,
        ...     inh_weight=-0.8 * u.nS
        ... )
        >>> result = ei_net(pre_size=1000, post_size=1000)

    Create an E-I network with delays:

    .. code-block:: python

        >>> ei_net = ExcitatoryInhibitory(
        ...     exc_ratio=0.8,
        ...     exc_prob=0.1,
        ...     inh_prob=0.2,
        ...     exc_weight=1.0 * u.nS,
        ...     inh_weight=-0.8 * u.nS,
        ...     exc_delay=1.5 * u.ms,
        ...     inh_delay=0.8 * u.ms
        ... )
        >>> result = ei_net(pre_size=1000, post_size=1000)

    Create an E-I network with only connectivity (no weights or delays):

    .. code-block:: python

        >>> ei_net = ExcitatoryInhibitory(
        ...     exc_ratio=0.8,
        ...     exc_prob=0.1,
        ...     inh_prob=0.2
        ... )
        >>> result = ei_net(pre_size=1000, post_size=1000)
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        exc_ratio: float = 0.8,
        exc_prob: float = 0.1,
        inh_prob: float = 0.2,
        exc_weight: Optional[Initializer] = None,
        inh_weight: Optional[Initializer] = None,
        exc_delay: Optional[Initializer] = None,
        inh_delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.exc_ratio = exc_ratio
        self.exc_prob = exc_prob
        self.inh_prob = inh_prob
        self.exc_weight = exc_weight
        self.inh_weight = inh_weight
        self.exc_delay = exc_delay
        self.inh_delay = inh_delay

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
        # is_excitatory = np.concatenate([np.ones(len(exc_pre), dtype=bool), np.zeros(len(inh_pre), dtype=bool)])
        n_connections = len(pre_indices)

        if n_connections == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                model_type='point',
                pre_positions=kwargs.get('pre_positions', None),
                post_positions=kwargs.get('post_positions', None),
            )

        n_exc_conn = len(exc_pre)
        n_inh_conn = len(inh_pre)

        # Generate weights separately for excitatory and inhibitory
        if self.exc_weight is None and self.inh_weight is None:
            weights = None
        elif self.exc_weight is None or self.inh_weight is None:
            raise ValueError("exc_weight and inh_weight must be both None or both specified")
        else:
            exc_weights = param(
                self.exc_weight,
                n_exc_conn,
                rng=self.rng,
                param_type='weight',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=kwargs.get('pre_positions', None),
                post_positions=kwargs.get('post_positions', None)
            )
            exc_weights, weight_unit = u.split_mantissa_unit(exc_weights)

            inh_weights = param(
                self.inh_weight,
                n_inh_conn,
                rng=self.rng,
                param_type='weight',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=kwargs.get('pre_positions', None),
                post_positions=kwargs.get('post_positions', None)
            )
            if weight_unit is None:
                inh_weights, weight_unit = u.split_mantissa_unit(inh_weights)
            else:
                inh_weights = u.Quantity(inh_weights).to(weight_unit).mantissa

            # Combine weights in correct order
            if u.math.isscalar(exc_weights):
                exc_weights = np.full(n_exc_conn, exc_weights)
            if u.math.isscalar(inh_weights):
                inh_weights = np.full(n_inh_conn, inh_weights)

            # Concatenate weights
            weights_array = np.concatenate([exc_weights, inh_weights])
            weights = u.maybe_decimal(weights_array * weight_unit)

        # Generate delays separately for excitatory and inhibitory
        if self.exc_delay is None and self.inh_delay is None:
            delays = None
        elif self.exc_delay is None or self.inh_delay is None:
            raise ValueError("exc_delay and inh_delay must be both None or both specified")
        else:
            exc_delays = param(
                self.exc_delay,
                n_exc_conn,
                rng=self.rng,
                param_type='delay',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=kwargs.get('pre_positions', None),
                post_positions=kwargs.get('post_positions', None)
            )
            exc_delays, delay_unit = u.split_mantissa_unit(exc_delays)

            inh_delays = param(
                self.inh_delay,
                n_inh_conn,
                rng=self.rng,
                param_type='delay',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=kwargs.get('pre_positions', None),
                post_positions=kwargs.get('post_positions', None)
            )
            if delay_unit is None:
                inh_delays, delay_unit = u.split_mantissa_unit(inh_delays)
            else:
                inh_delays = u.Quantity(inh_delays).to(delay_unit).mantissa

            # Combine delays in correct order
            if u.math.isscalar(exc_delays):
                exc_delays = np.full(n_exc_conn, exc_delays)
            if u.math.isscalar(inh_delays):
                inh_delays = np.full(n_inh_conn, inh_delays)

            # Concatenate delays
            delays_array = np.concatenate([exc_delays, inh_delays])
            delays = u.maybe_decimal(delays_array * delay_unit)

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
