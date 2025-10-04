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
    """
    Base Optimizer Class.
    """
    __module__ = 'braintools.optim'

    def register_trainable_weights(self, param_states: Dict[Hashable, State]):
        """
        Register the trainable weights.

        Parameters:
        -----------
        param_states: Dict[Hashable, State]
            The trainable weights.
        """
        raise NotImplementedError

    def update(self, grads: Dict[Hashable, PyTree]):
        """
        Update the trainable weights according to weight gradients.

        Parameters:
        -----------
        grads: Dict[Hashable, PyTree]
            The weight gradients.
        """
        raise NotImplementedError


class OptimState(brainstate.LongTermState):
    pass
