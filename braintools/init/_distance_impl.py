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
Distance profile classes for connectivity generation.

This module provides distance-dependent connectivity profiles that define how
connection probability and weight strength vary with spatial distance.
All classes inherit from the DistanceProfile base class.
"""

from typing import Optional

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike
from jax.scipy.special import expit

from ._distance_base import DistanceProfile


def _check_positive_length(value: ArrayLike, name: str) -> None:
    """Raise if a length-like parameter is non-positive (ignores None)."""
    if value is None:
        return
    if np.any(np.asarray(u.get_mantissa(value)) <= 0):
        raise ValueError(f'`{name}` must be positive, got {value!r}.')


__all__ = [
    # Distance profile classes
    'GaussianProfile',
    'ExponentialProfile',
    'PowerLawProfile',
    'LinearProfile',
    'StepProfile',
    'SigmoidProfile',
    'DoGProfile',
    'LogisticProfile',
    'BimodalProfile',
    'MexicanHatProfile',
]


# =============================================================================
# Distance Profiles
# =============================================================================

class GaussianProfile(DistanceProfile):
    """
    Gaussian distance profile.

    Connection probability and weight scaling follow a Gaussian (bell curve) profile.

    Parameters
    ----------
    sigma : Quantity
        Standard deviation of the Gaussian profile.
    max_distance : Quantity, optional
        Maximum connection distance (connections beyond this are set to 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import GaussianProfile
        >>>
        >>> profile = GaussianProfile(sigma=50.0 * u.um, max_distance=200.0 * u.um)
        >>> distances = np.array([0, 25, 50, 100, 200]) * u.um
        >>> probs = profile.probability(distances)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        sigma: ArrayLike,
        max_distance: Optional[ArrayLike] = None
    ):
        _check_positive_length(sigma, 'sigma')
        _check_positive_length(max_distance, 'max_distance')
        self.sigma = sigma
        self.max_distance = max_distance

    def probability(self, distances: ArrayLike) -> np.ndarray:
        sigma, unit = u.split_mantissa_unit(self.sigma)
        dist_vals = u.Quantity(distances).to(unit).mantissa

        prob = u.math.exp(-0.5 * (dist_vals / sigma) ** 2)

        if self.max_distance is not None:
            max_val = u.Quantity(self.max_distance).to(unit).mantissa
            prob = u.math.where(dist_vals > max_val, 0.0, prob)

        return prob

    def __repr__(self):
        return f'GaussianProfile(sigma={self.sigma}, max_distance={self.max_distance})'


class ExponentialProfile(DistanceProfile):
    """
    Exponential distance profile.

    Connection probability and weight scaling decay exponentially with distance.

    Parameters
    ----------
    decay_constant : Quantity
        Distance constant for exponential decay.
    max_distance : Quantity, optional
        Maximum connection distance (connections beyond this are set to 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import ExponentialProfile
        >>>
        >>> profile = ExponentialProfile(
        ...     decay_constant=100.0 * u.um,
        ...     max_distance=500.0 * u.um
        ... )
        >>> distances = np.array([0, 50, 100, 200, 500]) * u.um
        >>> probs = profile.probability(distances)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        decay_constant: ArrayLike,
        max_distance: Optional[ArrayLike] = None,
    ):
        _check_positive_length(decay_constant, 'decay_constant')
        _check_positive_length(max_distance, 'max_distance')
        self.decay_constant = decay_constant
        self.max_distance = max_distance

    def probability(self, distances: ArrayLike) -> np.ndarray:
        decay, unit = u.split_mantissa_unit(self.decay_constant)
        dist_vals = u.Quantity(distances).to(unit).mantissa

        prob = u.math.exp(-dist_vals / decay)

        if self.max_distance is not None:
            max_val = u.Quantity(self.max_distance).to(unit).mantissa
            prob = u.math.where(dist_vals > max_val, 0.0, prob)

        return prob

    def __repr__(self):
        return f'ExponentialProfile(decay_constant={self.decay_constant}, max_distance={self.max_distance})'


class PowerLawProfile(DistanceProfile):
    """
    Power-law distance profile.

    Connection probability follows a scale-invariant power-law decay:
    ``p(d) = (max(d, d0) / d0) ** (-exponent)``, where ``d0`` is the reference
    length ``min_distance``. Expressed as a dimensionless ratio, the profile is
    invariant to the unit in which distances are measured (100 um and 0.1 mm give
    the same probability), peaks at 1 for ``d <= d0``, and is finite at ``d = 0``.

    Parameters
    ----------
    exponent : float
        Power-law exponent (positive values cause decay with distance).
    min_distance : Quantity, optional
        Reference length ``d0`` below which the probability saturates at 1.0,
        also avoiding division by zero (default: ``1.0 * u.um``, or ``1.0`` for
        unitless distances).
    max_distance : Quantity, optional
        Maximum connection distance (connections beyond this are set to 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import PowerLawProfile
        >>>
        >>> profile = PowerLawProfile(
        ...     exponent=2.0,
        ...     min_distance=1.0 * u.um,
        ...     max_distance=1000.0 * u.um
        ... )
        >>> distances = np.array([1, 10, 100, 1000]) * u.um
        >>> probs = profile.probability(distances)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        exponent: float,
        min_distance: Optional[ArrayLike] = None,
        max_distance: Optional[ArrayLike] = None
    ):
        _check_positive_length(min_distance, 'min_distance')
        _check_positive_length(max_distance, 'max_distance')
        self.exponent = exponent
        self.min_distance = min_distance
        self.max_distance = max_distance

    def probability(self, distances: ArrayLike) -> np.ndarray:
        dist_vals, dist_unit = u.split_mantissa_unit(distances)

        # Reference length d0 in the distance's own unit. Using a fixed physical
        # length (1 um default) keeps the dimensionless ratio scale-invariant.
        if self.min_distance is None:
            d0 = 1.0 if dist_unit == u.UNITLESS else float(u.Quantity(1.0 * u.um).to(dist_unit).mantissa)
        elif dist_unit == u.UNITLESS:
            d0 = float(u.get_mantissa(self.min_distance))
        else:
            d0 = float(u.Quantity(self.min_distance).to(dist_unit).mantissa)

        ratio = u.math.maximum(dist_vals, d0) / d0
        prob = ratio ** (-self.exponent)

        if self.max_distance is not None:
            max_val = u.Quantity(self.max_distance).to(dist_unit).mantissa
            prob = u.math.where(dist_vals > max_val, 0.0, prob)

        return prob

    def __repr__(self):
        return f'PowerLawProfile(exponent={self.exponent}, min_distance={self.min_distance}, max_distance={self.max_distance})'


class LinearProfile(DistanceProfile):
    """
    Linear distance profile.

    Connection probability decreases linearly from 1 at distance 0 to 0 at max_distance.

    Parameters
    ----------
    max_distance : Quantity
        Maximum connection distance.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import LinearProfile
        >>>
        >>> profile = LinearProfile(max_distance=200.0 * u.um)
        >>> distances = np.array([0, 50, 100, 150, 200]) * u.um
        >>> probs = profile.probability(distances)
    """
    __module__ = 'braintools.init'

    def __init__(self, max_distance: ArrayLike):
        _check_positive_length(max_distance, 'max_distance')
        self.max_distance = max_distance

    def probability(self, distances: ArrayLike) -> np.ndarray:
        max_val, unit = u.split_mantissa_unit(self.max_distance)
        dist_vals = u.Quantity(distances).to(unit).mantissa

        prob = u.math.maximum(0.0, 1 - dist_vals / max_val)
        return prob

    def __repr__(self):
        return f'LinearProfile(max_distance={self.max_distance})'


class StepProfile(DistanceProfile):
    """
    Step function distance profile.

    Connection probability has two distinct values: one inside the threshold distance
    and another outside.

    Parameters
    ----------
    threshold : Quantity
        Distance threshold.
    inside_prob : float, optional
        Probability for distances <= threshold (default: 1.0).
    outside_prob : float, optional
        Probability for distances > threshold (default: 0.0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import StepProfile
        >>>
        >>> profile = StepProfile(
        ...     threshold=100.0 * u.um,
        ...     inside_prob=0.8,
        ...     outside_prob=0.1
        ... )
        >>> distances = np.array([50, 100, 150]) * u.um
        >>> probs = profile.probability(distances)
    """
    __module__ = 'braintools.init'

    def __init__(self,
                 threshold: ArrayLike,
                 inside_prob: float = 1.0,
                 outside_prob: float = 0.0):
        _check_positive_length(threshold, 'threshold')
        self.threshold = threshold
        self.inside_prob = inside_prob
        self.outside_prob = outside_prob

    def probability(self, distances: ArrayLike) -> np.ndarray:
        threshold, unit = u.split_mantissa_unit(self.threshold)
        dist_vals = u.Quantity(distances).to(unit).mantissa

        prob = u.math.where(dist_vals <= threshold, self.inside_prob, self.outside_prob)
        return prob

    def __repr__(self):
        return f'StepProfile(threshold={self.threshold}, inside_prob={self.inside_prob}, outside_prob={self.outside_prob})'


class SigmoidProfile(DistanceProfile):
    """
    Sigmoid distance profile.

    Connection probability follows a sigmoid function that smoothly transitions
    from high to low probability around a midpoint distance.

    Parameters
    ----------
    midpoint : Quantity
        Distance at which probability is 0.5.
    slope : float
        Steepness of the sigmoid transition (higher values = steeper).
    max_distance : Quantity, optional
        Maximum connection distance (connections beyond this are set to 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import SigmoidProfile
        >>>
        >>> profile = SigmoidProfile(
        ...     midpoint=100.0 * u.um,
        ...     slope=0.05,
        ...     max_distance=300.0 * u.um
        ... )
        >>> distances = np.array([0, 50, 100, 150, 200, 300]) * u.um
        >>> probs = profile.probability(distances)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        midpoint: ArrayLike,
        slope: float = 0.05,
        max_distance: Optional[ArrayLike] = None
    ):
        _check_positive_length(midpoint, 'midpoint')
        _check_positive_length(max_distance, 'max_distance')
        self.midpoint = midpoint
        self.slope = slope
        self.max_distance = max_distance

    def probability(self, distances: ArrayLike) -> np.ndarray:
        midpoint, unit = u.split_mantissa_unit(self.midpoint)
        dist_vals = u.Quantity(distances).to(unit).mantissa

        # expit is the numerically stable logistic sigmoid (no exp overflow for
        # large slope * distance): 1 / (1 + exp(slope * (d - midpoint))).
        prob = expit(-self.slope * (dist_vals - midpoint))

        if self.max_distance is not None:
            max_val = u.Quantity(self.max_distance).to(unit).mantissa
            prob = u.math.where(dist_vals > max_val, 0.0, prob)

        return prob

    def __repr__(self):
        return f'SigmoidProfile(midpoint={self.midpoint}, slope={self.slope}, max_distance={self.max_distance})'


class DoGProfile(DistanceProfile):
    """
    Difference of Gaussians (DoG) distance profile.

    Connection probability follows a center-surround pattern with excitation at
    close range and inhibition at intermediate range.

    Parameters
    ----------
    sigma_center : Quantity
        Standard deviation of the center Gaussian.
    sigma_surround : Quantity
        Standard deviation of the surround Gaussian (should be > sigma_center).
    amplitude_center : float, optional
        Amplitude of center Gaussian (default: 1.0).
    amplitude_surround : float, optional
        Amplitude of surround Gaussian (default: 0.5).
    max_distance : Quantity, optional
        Maximum connection distance (connections beyond this are set to 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import DoGProfile
        >>>
        >>> profile = DoGProfile(
        ...     sigma_center=30.0 * u.um,
        ...     sigma_surround=90.0 * u.um,
        ...     amplitude_center=1.0,
        ...     amplitude_surround=0.5
        ... )
        >>> distances = np.array([0, 30, 60, 90, 120]) * u.um
        >>> probs = profile.probability(distances)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        sigma_center: ArrayLike,
        sigma_surround: ArrayLike,
        amplitude_center: float = 1.0,
        amplitude_surround: float = 0.5,
        max_distance: Optional[ArrayLike] = None
    ):
        _check_positive_length(sigma_center, 'sigma_center')
        _check_positive_length(sigma_surround, 'sigma_surround')
        _check_positive_length(max_distance, 'max_distance')
        self.sigma_center = sigma_center
        self.sigma_surround = sigma_surround
        self.amplitude_center = amplitude_center
        self.amplitude_surround = amplitude_surround
        self.max_distance = max_distance

    def probability(self, distances: ArrayLike) -> np.ndarray:
        sigma_c, unit = u.split_mantissa_unit(self.sigma_center)
        sigma_s = u.Quantity(self.sigma_surround).to(unit).mantissa
        dist_vals = u.Quantity(distances).to(unit).mantissa

        center = self.amplitude_center * u.math.exp(-0.5 * (dist_vals / sigma_c) ** 2)
        surround = self.amplitude_surround * u.math.exp(-0.5 * (dist_vals / sigma_s) ** 2)
        prob = center - surround

        # Clip negative values to 0
        prob = u.math.maximum(prob, 0.0)

        if self.max_distance is not None:
            max_val = u.Quantity(self.max_distance).to(unit).mantissa
            prob = u.math.where(dist_vals > max_val, 0.0, prob)

        return prob

    def __repr__(self):
        return (f'DoGProfile(sigma_center={self.sigma_center}, sigma_surround={self.sigma_surround}, '
                f'amplitude_center={self.amplitude_center}, amplitude_surround={self.amplitude_surround}, '
                f'max_distance={self.max_distance})')


class LogisticProfile(DistanceProfile):
    """
    Logistic distance profile.

    Connection probability follows a logistic decay function, similar to sigmoid
    but normalized for distance-dependent connectivity.

    Parameters
    ----------
    growth_rate : float
        Growth rate parameter controlling decay speed.
    midpoint : Quantity
        Distance at which decay is at its midpoint.
    max_distance : Quantity, optional
        Maximum connection distance (connections beyond this are set to 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import LogisticProfile
        >>>
        >>> profile = LogisticProfile(
        ...     growth_rate=0.05,
        ...     midpoint=100.0 * u.um,
        ...     max_distance=500.0 * u.um
        ... )
        >>> distances = np.array([0, 50, 100, 200, 500]) * u.um
        >>> probs = profile.probability(distances)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        growth_rate: float,
        midpoint: ArrayLike,
        max_distance: Optional[ArrayLike] = None
    ):
        _check_positive_length(midpoint, 'midpoint')
        _check_positive_length(max_distance, 'max_distance')
        self.growth_rate = growth_rate
        self.midpoint = midpoint
        self.max_distance = max_distance

    def probability(self, distances: ArrayLike) -> np.ndarray:
        midpoint, unit = u.split_mantissa_unit(self.midpoint)
        dist_vals = u.Quantity(distances).to(unit).mantissa

        # Stable logistic sigmoid: 1 / (1 + exp(growth_rate * (d - midpoint))).
        prob = expit(-self.growth_rate * (dist_vals - midpoint))

        if self.max_distance is not None:
            max_val = u.Quantity(self.max_distance).to(unit).mantissa
            prob = u.math.where(dist_vals > max_val, 0.0, prob)

        return prob

    def __repr__(self):
        return f'LogisticProfile(growth_rate={self.growth_rate}, midpoint={self.midpoint}, max_distance={self.max_distance})'


class BimodalProfile(DistanceProfile):
    """
    Bimodal distance profile.

    Connection probability has two peaks at different distances, useful for
    modeling connections with both local and long-range components.

    Parameters
    ----------
    sigma1 : Quantity
        Standard deviation of first Gaussian peak.
    sigma2 : Quantity
        Standard deviation of second Gaussian peak.
    center1 : Quantity, optional
        Center position of first peak (default: ``0 * u.um``).
    center2 : Quantity, optional
        Center position of second peak (default: ``0 * u.um``).
    amplitude1 : float, optional
        Amplitude of first peak (default: 1.0).
    amplitude2 : float, optional
        Amplitude of second peak (default: 1.0).
    max_distance : Quantity, optional
        Maximum connection distance (connections beyond this are set to 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import BimodalProfile
        >>>
        >>> profile = BimodalProfile(
        ...     sigma1=30.0 * u.um,
        ...     sigma2=50.0 * u.um,
        ...     center1=0.0 * u.um,
        ...     center2=200.0 * u.um,
        ...     amplitude1=1.0,
        ...     amplitude2=0.8
        ... )
        >>> distances = np.array([0, 50, 100, 200, 300]) * u.um
        >>> probs = profile.probability(distances)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        sigma1: ArrayLike,
        sigma2: ArrayLike,
        center1: ArrayLike = 0. * u.um,
        center2: ArrayLike = 0. * u.um,
        amplitude1: float = 1.0,
        amplitude2: float = 1.0,
        max_distance: Optional[ArrayLike] = None
    ):
        _check_positive_length(sigma1, 'sigma1')
        _check_positive_length(sigma2, 'sigma2')
        _check_positive_length(max_distance, 'max_distance')
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.center1 = center1
        self.center2 = center2
        self.amplitude1 = amplitude1
        self.amplitude2 = amplitude2
        self.max_distance = max_distance

    def probability(self, distances: ArrayLike) -> np.ndarray:
        sigma1, unit = u.split_mantissa_unit(self.sigma1)
        sigma2 = u.Quantity(self.sigma2).to(unit).mantissa
        center1 = u.Quantity(self.center1).to(unit).mantissa
        center2 = u.Quantity(self.center2).to(unit).mantissa
        dist_vals = u.Quantity(distances).to(unit).mantissa

        peak1 = self.amplitude1 * u.math.exp(-0.5 * ((dist_vals - center1) / sigma1) ** 2)
        peak2 = self.amplitude2 * u.math.exp(-0.5 * ((dist_vals - center2) / sigma2) ** 2)
        prob = peak1 + peak2

        if self.max_distance is not None:
            max_val = u.Quantity(self.max_distance).to(unit).mantissa
            prob = u.math.where(dist_vals > max_val, 0.0, prob)

        return prob

    def __repr__(self):
        return (f'BimodalProfile(sigma1={self.sigma1}, sigma2={self.sigma2}, '
                f'center1={self.center1}, center2={self.center2}, '
                f'amplitude1={self.amplitude1}, amplitude2={self.amplitude2}, '
                f'max_distance={self.max_distance})')


class MexicanHatProfile(DistanceProfile):
    """
    Mexican Hat (Ricker wavelet) distance profile.

    Connection probability follows a Mexican hat shape, which is the second derivative
    of a Gaussian function. This creates a center-surround pattern with positive values
    at the center, negative values in the surround, and approaching zero at far distances.
    The negative values are clipped to zero for probability interpretation.

    The (peak-normalized) Mexican hat function used here is:
        f(d) = amplitude * (1 - (d/sigma)^2) * exp(-(d/sigma)^2 / 2)

    This form peaks at ``amplitude`` when ``d = 0`` and is invariant to the unit
    in which ``sigma``/``distances`` are expressed (it depends only on the
    dimensionless ratio ``d/sigma``). The classical wavelet L2-normalization
    constant ``2 / (sqrt(3*sigma) * pi^(1/4))`` is intentionally dropped because
    it is unit-dependent and would make the peak height vary with the chosen
    length unit.

    Parameters
    ----------
    sigma : Quantity
        Standard deviation controlling the width of the profile.
    amplitude : float, optional
        Amplitude scaling factor; equals the peak value at ``d = 0`` (default: 1.0).
    max_distance : Quantity, optional
        Maximum connection distance (connections beyond this are set to 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import MexicanHatProfile
        >>>
        >>> profile = MexicanHatProfile(
        ...     sigma=50.0 * u.um,
        ...     amplitude=1.0,
        ...     max_distance=300.0 * u.um
        ... )
        >>> distances = np.array([0, 25, 50, 100, 200]) * u.um
        >>> probs = profile.probability(distances)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        sigma: ArrayLike,
        amplitude: float = 1.0,
        max_distance: Optional[ArrayLike] = None
    ):
        _check_positive_length(sigma, 'sigma')
        _check_positive_length(max_distance, 'max_distance')
        self.sigma = sigma
        self.amplitude = amplitude
        self.max_distance = max_distance

    def probability(self, distances: ArrayLike) -> np.ndarray:
        sigma, unit = u.split_mantissa_unit(self.sigma)
        dist_vals = u.Quantity(distances).to(unit).mantissa

        # Normalized (dimensionless) distance.
        d_norm = dist_vals / sigma

        # Peak-normalized Mexican hat: amplitude * (1 - d_norm^2) * exp(-d_norm^2 / 2).
        # Peaks at `amplitude` when d == 0; scale-invariant (no unit-dependent
        # normalization constant).
        prob = self.amplitude * (1 - d_norm ** 2) * u.math.exp(-d_norm ** 2 / 2)

        # Clip negative values to 0 for probability interpretation
        prob = u.math.maximum(prob, 0.0)

        if self.max_distance is not None:
            max_val = u.Quantity(self.max_distance).to(unit).mantissa
            prob = u.math.where(dist_vals > max_val, 0.0, prob)

        return prob

    def __repr__(self):
        return f'MexicanHatProfile(sigma={self.sigma}, amplitude={self.amplitude}, max_distance={self.max_distance})'
