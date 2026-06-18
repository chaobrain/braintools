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
import numpy as np
from scipy.spatial.distance import cdist

from ._distance_base import DistanceProfile
from ._init_base import Initialization

__all__ = [
    'DistanceModulated',
]


class DistanceModulated(Initialization):
    """
    Initialization modulated by distance.

    Generates weights from a base distribution and then multiplies them by a
    distance-dependent gain taken from a :class:`DistanceProfile` (e.g.
    exponential decay, Gaussian). The result is ``base_weights *
    distance_profile(distances)`` evaluated element-wise.

    Parameters
    ----------
    base_dist : Initialization
        Base weight distribution.
    distance_profile : DistanceProfile
        Distance modulation function. Its :meth:`~DistanceProfile.weight_scaling`
        output multiplies the base weights.

    Notes
    -----
    This initializer is a *deterministic weight modulator*, not a stochastic
    connectivity mask. The distance profile scales the magnitude of every
    weight; it never performs Bernoulli sampling to drop connections. A profile
    value of ``0`` produces a weight of exactly ``0`` (a present-but-silent
    synapse), not an absent connection. Distance-dependent *connectivity*
    (deciding which pairs are connected at all) belongs to the ``braintools.conn``
    module, where the profile's ``probability`` is consumed by a sampler.

    The connection distances can be supplied directly via the ``distances``
    keyword, or computed from neuron coordinates via ``pre_positions`` and
    ``post_positions``. Coordinates may be given either as a 2-D ``(N, d)`` array
    (``N`` neurons in ``d``-dimensional space) or, for one-dimensional layouts,
    as a flat ``(N,)`` vector that is promoted to ``(N, 1)`` automatically.

    Examples
    --------

    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import DistanceModulated, GaussianProfile, Normal
        >>>
        >>> rng = np.random.default_rng(0)
        >>> profile = GaussianProfile(sigma=100.0 * u.um)
        >>> init = DistanceModulated(
        ...     base_dist=Normal(1.0 * u.nS, 0.2 * u.nS),
        ...     distance_profile=profile,
        ... )
        >>>
        >>> # Modulate using pre-computed pairwise distances.
        >>> distances = np.array([0.0, 50.0, 100.0, 150.0]) * u.um
        >>> weights = init(4, distances=distances, rng=rng)
        >>>
        >>> # ... or compute distances from neuron coordinates (flat 1-D layout).
        >>> pre = np.array([0.0, 100.0, 200.0]) * u.um
        >>> post = np.array([0.0, 100.0, 200.0]) * u.um
        >>> matrix = init((3, 3), pre_positions=pre, post_positions=post, rng=rng)
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
            pre_mantissa, pos_unit = u.split_mantissa_unit(pre_positions)
            post_mantissa = u.Quantity(post_positions).to(pos_unit).mantissa
            # ``cdist`` requires 2-D ``(N, d)`` inputs; promote flat ``(N,)``
            # coordinate vectors (1-D spatial layouts) to a single column.
            pre_mantissa = np.atleast_1d(np.asarray(pre_mantissa))
            post_mantissa = np.atleast_1d(np.asarray(post_mantissa))
            if pre_mantissa.ndim == 1:
                pre_mantissa = pre_mantissa.reshape(-1, 1)
            if post_mantissa.ndim == 1:
                post_mantissa = post_mantissa.reshape(-1, 1)
            distances = u.maybe_decimal(cdist(pre_mantissa, post_mantissa) * pos_unit)

        return base_weights * self.distance_profile(distances)

    def __repr__(self):
        return (f'DistanceModulated(base_dist={self.base_dist}, '
                f'distance_profile={self.distance_profile})')
