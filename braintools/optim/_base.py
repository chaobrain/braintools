# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


from typing import Dict, Hashable

import brainstate
from brainstate import State
from brainstate.graph import Node
from brainstate.typing import PyTree

__all__ = [
    'Optimizer',
    'OptimState',
]


class Optimizer(Node):
    """Base class for all optimizers.

    Subclasses must implement :meth:`register_trainable_weights` and :meth:`update`
    to register the parameters to optimize and to apply a single optimization step,
    respectively.
    """
    __module__ = 'braintools.optim'

    def register_trainable_weights(self, param_states: Dict[Hashable, State]):
        """Register the trainable weights with the optimizer.

        Parameters
        ----------
        param_states : dict of {hashable : brainstate.State}
            The trainable weights to optimize, as a pytree whose leaves are
            ``brainstate.State`` objects.

        Raises
        ------
        NotImplementedError
            Always, in the base class. Subclasses must override this method.
        """
        raise NotImplementedError

    def update(self, grads: Dict[Hashable, PyTree]):
        """Update the trainable weights from their gradients.

        Parameters
        ----------
        grads : dict of {hashable : PyTree}
            The gradients of the loss with respect to each registered weight, with
            the same structure as the registered parameters.

        Raises
        ------
        NotImplementedError
            Always, in the base class. Subclasses must override this method.
        """
        raise NotImplementedError


class OptimState(brainstate.LongTermState):
    """A :class:`brainstate.LongTermState` holding optimizer state.

    Used for quantities that persist across optimization steps but are not trainable
    parameters, such as ``optax`` optimizer state, step counters and learning rates.
    """
    __module__ = 'braintools.optim'
