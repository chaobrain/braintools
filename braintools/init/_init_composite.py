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

"""
Composite weight initialization distributions.

This module provides composite weight initialization strategies that combine
or modify other distributions including:
- Mixture distributions
- Conditional distributions
- Scaled distributions
- Clipped distributions
- Distance-modulated distributions
"""

from typing import Optional

import brainstate
import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike

from ._init_base import Initialization

__all__ = [
    'Mixture',
    'Conditional',
    'Scaled',
    'Clipped',
]


class Mixture(Initialization):
    """
    Mixture of multiple weight distributions.

    Randomly selects from multiple distributions for each connection according to specified weights.

    Parameters
    ----------
    distributions : list of Initialization
        List of initialization distributions to mix.
    weights : list of float, optional
        Probability weights for each distribution (must sum to 1).
        If None, uses equal weights.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Mixture, Normal, Uniform
        >>>
        >>> init = Mixture(
        ...     distributions=[
        ...         Normal(0.5 * u.siemens, 0.1 * u.siemens),
        ...         Uniform(0.8 * u.siemens, 1.2 * u.siemens)
        ...     ],
        ...     weights=[0.7, 0.3]
        ... )
        >>> rng = np.random.default_rng(0)
        >>> weights = init(1000, rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(self, distributions: list, weights: Optional[list] = None):
        self.distributions = distributions
        self.weights = weights if weights is not None else [1.0 / len(distributions)] * len(distributions)

    def __call__(self, size, **kwargs):
        rng = kwargs.get('rng', brainstate.random)
        choices = rng.choice(len(self.distributions), size=size, p=self.weights)

        if isinstance(size, int):
            samples = np.zeros(size)
            unit = None
        else:
            samples = np.zeros(size)
            unit = None

        for i, dist in enumerate(self.distributions):
            mask = (choices == i)
            if np.any(mask):
                dist_samples = dist(np.sum(mask), **kwargs)
                if unit is None:
                    unit = dist_samples.unit
                samples[mask] = dist_samples.to(unit).mantissa

        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'Mixture(distributions={self.distributions}, weights={self.weights})'


class Conditional(Initialization):
    """
    Conditional weight distribution based on neuron properties.

    Uses different distributions based on a condition function applied to neuron indices.

    Parameters
    ----------
    condition_fn : callable
        Function that takes neuron indices and returns boolean array.
    true_dist : Initialization
        Distribution to use when condition is True.
    false_dist : Initialization
        Distribution to use when condition is False.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Conditional, Constant, Normal
        >>>
        >>> def is_excitatory(indices):
        ...     return indices < 800
        >>>
        >>> init = Conditional(
        ...     condition_fn=is_excitatory,
        ...     true_dist=Normal(0.5 * u.siemens, 0.1 * u.siemens),
        ...     false_dist=Normal(-0.3 * u.siemens, 0.05 * u.siemens)
        ... )
        >>> rng = np.random.default_rng(0)
        >>> weights = init(1000, neuron_indices=np.arange(1000), rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        condition_fn,
        true_dist: Initialization,
        false_dist: Initialization
    ):
        self.condition_fn = condition_fn
        self.true_dist = true_dist
        self.false_dist = false_dist

    def __call__(self, size, neuron_indices: Optional[np.ndarray] = None, **kwargs):
        if neuron_indices is None:
            neuron_indices = np.arange(size if isinstance(size, int) else np.prod(size))

        conditions = self.condition_fn(neuron_indices)

        true_samples = self.true_dist(np.sum(conditions), **kwargs)
        false_samples = self.false_dist(np.sum(~conditions), **kwargs)

        if isinstance(size, int):
            samples = np.zeros(size)
        else:
            samples = np.zeros(size)

        unit = true_samples.unit
        samples[conditions] = true_samples.to(unit).mantissa
        samples[~conditions] = false_samples.to(unit).mantissa

        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'Conditional(condition_fn={self.condition_fn}, true_dist={self.true_dist}, false_dist={self.false_dist})'


class Scaled(Initialization):
    """
    Scaled version of another distribution.

    Multiplies the output of another distribution by a constant factor.

    Parameters
    ----------
    base_dist : Initialization
        Base distribution to scale.
    scale_factor : float or Quantity
        Factor to multiply the base distribution by.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Scaled, Normal
        >>>
        >>> base = Normal(1.0 * u.siemens, 0.2 * u.siemens)
        >>> init = Scaled(base, scale_factor=0.5)
        >>> rng = np.random.default_rng(0)
        >>> weights = init(1000, rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        base_dist: Initialization,
        scale_factor: ArrayLike
    ):
        self.base_dist = base_dist
        self.scale_factor = scale_factor

    def __call__(self, size, **kwargs):
        base_samples = self.base_dist(size, **kwargs)
        return base_samples * self.scale_factor

    def __repr__(self):
        return f'Scaled(base_dist={self.base_dist}, scale_factor={self.scale_factor})'


class Clipped(Initialization):
    """
    Clipped version of another distribution.

    Clips the output of another distribution to specified minimum and maximum values.

    Parameters
    ----------
    base_dist : Initialization
        Base distribution to clip.
    min_val : Quantity, optional
        Minimum value (default: no lower bound).
    max_val : Quantity, optional
        Maximum value (default: no upper bound).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Clipped, Normal
        >>>
        >>> base = Normal(0.5 * u.siemens, 0.3 * u.siemens)
        >>> init = Clipped(base, min_val=0.0 * u.siemens, max_val=1.0 * u.siemens)
        >>> rng = np.random.default_rng(0)
        >>> weights = init(1000, rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        base_dist: Initialization,
        min_val: Optional[ArrayLike] = None,
        max_val: Optional[ArrayLike] = None
    ):
        self.base_dist = base_dist
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, size, **kwargs):
        samples = self.base_dist(size, **kwargs)

        if self.min_val is not None:
            min_val = u.Quantity(self.min_val).to(samples.unit).mantissa
            samples = u.math.maximum(samples, min_val * samples.unit)

        if self.max_val is not None:
            max_val = u.Quantity(self.max_val).to(samples.unit).mantissa
            samples = u.math.minimum(samples, max_val * samples.unit)

        return samples

    def __repr__(self):
        return f'Clipped(base_dist={self.base_dist}, min_val={self.min_val}, max_val={self.max_val})'
