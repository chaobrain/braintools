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
from brainstate.typing import ArrayLike

from ._distance_base import DistanceProfile
from ._init_base import Initialization

__all__ = [
    'DistanceModulated',
]


class DistanceModulated(Initialization):
    """
    Weight distribution modulated by distance.

    Generates weights from a base distribution and then modulates them based on
    distance using a specified function (e.g., exponential decay, gaussian).

    Parameters
    ----------
    base_dist : Initialization
        Base weight distribution.
    distance_profile : DistanceProfile
        Distance modulation function.
    min_weight : Quantity, optional
        Minimum weight floor (default: 0).

    Examples
    --------

    .. code-block:: python

        >>> from braintools.init import GaussianProfile, Normal
        >>>
        >>> profile = GaussianProfile(sigma=100.0 * u.um)
        >>> init = DistanceModulated(
        ...     base_dist=Normal(1.0 * u.nS, 0.2 * u.nS),
        ...     distance_profile=profile,
        ...     min_weight=0.01 * u.nS
        ... )
        >>> weights = init(100, distances=distances, rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        base_dist: Initialization,
        distance_profile: DistanceProfile,
        min_weight: Optional[ArrayLike] = None
    ):
        self.base_dist = base_dist
        self.min_weight = min_weight
        self.distance_profile = distance_profile

    def __call__(self, size, distances: Optional[ArrayLike] = None, **kwargs):
        base_weights = self.base_dist(size, **kwargs)

        if distances is None:
            return base_weights

        modulation = self.distance_profile.weight_scaling(distances)
        weight_vals, weight_unit = u.split_mantissa_unit(base_weights)
        modulated = weight_vals * modulation
        if self.min_weight is not None:
            min_val = u.Quantity(self.min_weight).to(weight_unit).mantissa
            modulated = np.maximum(modulated, min_val)
        return u.maybe_decimal(modulated * weight_unit)

    def __repr__(self):
        return (f'DistanceModulated(base_dist={self.base_dist}, '
                f'distance_profile={self.distance_profile}, '
                f'min_weight={self.min_weight})')
