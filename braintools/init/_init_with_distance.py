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

import brainunit as u
from scipy.spatial.distance import cdist

from ._distance_base import DistanceProfile
from ._init_base import Initialization

__all__ = [
    'DistanceModulated',
]


class DistanceModulated(Initialization):
    """
    Initialization modulated by distance.

    Generates weights from a base distribution and then modulates them based on
    distance using a specified function (e.g., exponential decay, gaussian).

    Parameters
    ----------
    base_dist : Initialization
        Base weight distribution.
    distance_profile : DistanceProfile
        Distance modulation function.

    Examples
    --------

    .. code-block:: python

        >>> from braintools.init import GaussianProfile, Normal
        >>>
        >>> profile = GaussianProfile(sigma=100.0 * u.um)
        >>> init = DistanceModulated(
        ...     base_dist=Normal(1.0 * u.nS, 0.2 * u.nS),
        ...     distance_profile=profile,
        ... )
        >>> weights = init(100, distances=distances, rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        base_dist: Initialization,
        distance_profile: DistanceProfile,
    ):
        self.base_dist = base_dist
        self.distance_profile = distance_profile

    def __call__(self, size, **kwargs):
        base_weights = self.base_dist(size, **kwargs)

        if 'distances' in kwargs:
            distances = kwargs['distances']
        else:
            if 'pre_positions' not in kwargs or 'post_positions' not in kwargs:
                raise ValueError("Must provide 'distances' or both 'pre_positions' and 'post_positions'.")
            pre_positions = kwargs['pre_positions']
            post_positions = kwargs['post_positions']
            pre_positions, pos_unit = u.split_mantissa_unit(pre_positions)
            post_positions = u.Quantity(post_positions).to(pos_unit).mantissa
            distances = u.maybe_decimal(cdist(pre_positions, post_positions) * pos_unit)

        return base_weights * self.distance_profile(distances)

    def __repr__(self):
        return (f'DistanceModulated(base_dist={self.base_dist}, '
                f'distance_profile={self.distance_profile})')
