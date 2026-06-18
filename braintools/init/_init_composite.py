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
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike

from ._init_base import Initialization, ClipInit

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
        if len(distributions) == 0:
            raise ValueError('Mixture requires at least one distribution.')
        if weights is None:
            weights = [1.0 / len(distributions)] * len(distributions)
        else:
            if len(weights) != len(distributions):
                raise ValueError(
                    f'`weights` must have the same length as `distributions` '
                    f'({len(distributions)}), got {len(weights)}.'
                )
            weights_arr = np.asarray(weights, dtype=float)
            if np.any(weights_arr < 0):
                raise ValueError('`weights` must be non-negative.')
            total = float(weights_arr.sum())
            if not np.isclose(total, 1.0):
                raise ValueError(f'`weights` must sum to 1, got {total}.')
        self.distributions = distributions
        self.weights = weights

    def __call__(self, size, **kwargs):
        rng = kwargs.get('rng', brainstate.random)
        shape = (size,) if isinstance(size, int) else tuple(size)
        n = int(np.prod(shape)) if len(shape) > 0 else 1

        choices = np.asarray(rng.choice(len(self.distributions), size=n, p=self.weights))

        unit = None
        flat = None
        for i, dist in enumerate(self.distributions):
            idx = np.where(choices == i)[0]
            if idx.size == 0:
                continue
            sample = dist(int(idx.size), **kwargs)
            if unit is None:
                mantissa, unit = u.split_mantissa_unit(sample)
                flat = jnp.zeros(n, dtype=jnp.asarray(mantissa).dtype)
            else:
                # Reconcile units across components (raises on incompatibility).
                mantissa = u.Quantity(sample).to(unit).mantissa
            flat = flat.at[idx].set(mantissa)

        if flat is None:
            # Empty draw (size 0): use the first distribution to fix the unit/dtype.
            mantissa, unit = u.split_mantissa_unit(self.distributions[0](0, **kwargs))
            flat = jnp.zeros(n, dtype=jnp.asarray(mantissa).dtype)

        result = flat.reshape(shape)
        return u.maybe_decimal(result * unit)

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
        shape = (size,) if isinstance(size, int) else tuple(size)
        n = int(np.prod(shape)) if len(shape) > 0 else 1

        if neuron_indices is None:
            neuron_indices = np.arange(n)

        conditions = np.asarray(self.condition_fn(neuron_indices)).reshape(-1).astype(bool)
        if conditions.shape[0] != n:
            raise ValueError(
                f'`condition_fn` must return a boolean array with one entry per '
                f'element ({n}), got {conditions.shape[0]}.'
            )

        true_idx = np.where(conditions)[0]
        false_idx = np.where(~conditions)[0]

        true_samples = self.true_dist(int(true_idx.size), **kwargs)
        false_samples = self.false_dist(int(false_idx.size), **kwargs)

        # Use the "true" branch to fix the unit; convert the "false" branch to match.
        true_mantissa, unit = u.split_mantissa_unit(true_samples)
        false_mantissa = u.Quantity(false_samples).to(unit).mantissa

        flat = jnp.zeros(n, dtype=jnp.result_type(jnp.asarray(true_mantissa),
                                                  jnp.asarray(false_mantissa)))
        flat = flat.at[true_idx].set(true_mantissa)
        flat = flat.at[false_idx].set(false_mantissa)

        return u.maybe_decimal(flat.reshape(shape) * unit)

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
        # Delegate to the base ``ClipInit`` so the unit-aware / unitless branching
        # lives in one place (bug H2/M8: the previous code accessed ``.unit``
        # unconditionally and crashed on unitless distributions).
        return ClipInit(self.base_dist, self.min_val, self.max_val)(size, **kwargs)

    def __repr__(self):
        return f'Clipped(base_dist={self.base_dist}, min_val={self.min_val}, max_val={self.max_val})'
