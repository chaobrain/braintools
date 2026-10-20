# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Inter-spike-interval statistics of a :class:`~braintools.spike.SpikeTrains`.

This module also holds the small unit and validation helpers shared by the
event-based spike measures (``_psth``, ``_correlogram``, ``_sta``).
"""

from typing import Any, Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braintools._misc import set_module_as
from braintools.spike._spike_trains import (
    SpikeTrains, _ceil_ratio, _is_traced, _split, _static_int, _wrap,
)

__all__ = [
    'isi',
    'cv',
    'cv2',
    'lv',
    'fano_factor',
]

# Same tolerance factor as ``SpikeTrains.bin``: values within a few machine
# epsilons below a bin edge count as lying on the edge.
_EPS_FACTOR = 4


# ----------------------------------------------------------------------------
# shared helpers (also used by _psth, _correlogram and _sta)
# ----------------------------------------------------------------------------

def _check_trains(trains: Any, name: str = 'trains') -> SpikeTrains:
    """Raise ``TypeError`` unless ``trains`` is a :class:`SpikeTrains`."""
    if not isinstance(trains, SpikeTrains):
        raise TypeError(
            f'`{name}` must be a braintools.spike.SpikeTrains, got {type(trains).__name__}. '
            f'Build one from a dense (T, N) raster with SpikeTrains.from_dense(spikes, dt=...).'
        )
    return trains


def _ms_in(unit: u.Unit) -> float:
    """How many ``unit`` make one millisecond."""
    return float(u.Quantity(1., unit=u.ms).to_decimal(unit))


def _time_value(value: Any, unit: Optional[u.Unit], name: str) -> Any:
    """Express a time (scalar or array, possibly traced) as a plain value in ``unit``.

    ``unit=None`` means the spike times are plain numbers, which are taken to
    be milliseconds. A Quantity is converted; a plain number is taken to be in
    milliseconds (the ``brainstate`` convention), and converted to ``unit``.
    """
    if isinstance(value, u.Quantity) and not value.unit.is_unitless:
        target = u.ms if unit is None else unit
        try:
            return value.to_decimal(target)
        except Exception as e:  # incompatible dimension
            raise TypeError(f'`{name}` must be a time, got {value!r}.') from e
    value = _split(value)[0]
    if unit is None or unit == u.ms:
        return value
    return value * _ms_in(unit)


def _static_time(value: Any, unit: Optional[u.Unit], name: str) -> float:
    """A concrete (static) time as a Python float in ``unit``."""
    if _is_traced(value):
        raise TypeError(
            f'`{name}` fixes an output shape, so it must be a static (concrete) value, '
            f'not a traced array; close over it or mark it static in jax.jit.'
        )
    v = np.asarray(_time_value(value, unit, name), dtype=float)
    if v.ndim != 0:
        raise ValueError(f'`{name}` must be a scalar time, got shape {v.shape}.')
    v = float(v)
    if not np.isfinite(v):
        raise ValueError(f'`{name}` must be finite, got {value!r}.')
    return v


def _seconds_per(unit: Optional[u.Unit]) -> float:
    """Length in seconds of one time unit of the spike times (``None`` = ms)."""
    return float(u.Quantity(1., unit=u.ms if unit is None else unit).to_decimal(u.second))


def _bins_for(length: float, bin_size: float, dtype: Any, name: str) -> int:
    """Static number of bins ``ceil(length / bin_size)`` that ignores round-off."""
    return _static_int(_ceil_ratio(length, bin_size, dtype), name, 0)


def _floor_ratio(numerator: float, denominator: float, dtype: Any) -> int:
    """``floor(numerator / denominator)`` that ignores round-off of ``dtype`` values.

    Mirrors ``SpikeTrains``' ``_ceil_ratio``: a ratio within a few machine
    epsilons below an integer counts as that integer.
    """
    ratio = float(numerator) / float(denominator)
    eps = float(jnp.finfo(dtype).eps)
    return max(0, int(np.floor(ratio + _EPS_FACTOR * eps * max(1.0, abs(ratio)))))


def _safe_div(num: jax.Array, den: jax.Array, ok: jax.Array) -> jax.Array:
    """``num / den`` where ``ok``, ``nan`` elsewhere, with finite gradients."""
    return jnp.where(ok, num / jnp.where(ok, den, 1), jnp.nan)


def _intervals(trains: SpikeTrains):
    """Padded ISIs (zeros in the padding, gradient-safe) and their validity mask."""
    times = trains._times_m
    mask = trains.mask
    safe = jnp.where(mask, times, 0)
    intervals = safe[..., 1:] - safe[..., :-1]
    valid = mask[..., 1:]
    return jnp.where(valid, intervals, 0), valid


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------

@set_module_as('braintools.analysis.spike')
def isi(trains: SpikeTrains) -> Any:
    r"""Compute the inter-spike intervals of every neuron.

    For neuron :math:`n` with spike times :math:`t_1 < \dots < t_{c_n}`, the
    intervals are :math:`I_k = t_{k+1} - t_k`, :math:`k = 1, \dots, c_n - 1`.

    Parameters
    ----------
    trains : SpikeTrains
        Spike trains with ``N`` neurons and capacity ``K``.

    Returns
    -------
    jax.Array or Quantity
        Intervals of shape ``(N, K - 1)``, padded with ``nan`` beyond entry
        ``counts - 1`` of every row. A Quantity in the unit of the spike
        times when they carry one.

    Raises
    ------
    TypeError
        If ``trains`` is not a :class:`~braintools.spike.SpikeTrains`.

    See Also
    --------
    cv, cv2, lv : Variability statistics of the intervals.
    braintools.spike.SpikeTrains.from_dense : Build the input from a raster.

    Notes
    -----
    The output shape depends only on the capacity, so the function works
    under ``jax.jit`` and ``jax.vmap`` and is differentiable with respect to
    the spike times (the padding receives zero gradient).

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from braintools.spike import SpikeTrains
        >>> from braintools.analysis.spike import isi
        >>> raster = jnp.zeros((10, 2)).at[jnp.array([0, 2, 7]), 0].set(1).at[4, 1].set(1)
        >>> st = SpikeTrains.from_dense(raster, dt=1. * u.ms)
        >>> isi(st)
        Quantity([[ 2.  5.]
                  [nan nan]], "ms")
    """
    _check_trains(trains)
    intervals, valid = _intervals(trains)
    return _wrap(jnp.where(valid, intervals, jnp.nan), trains._unit)


@set_module_as('braintools.analysis.spike')
def cv(trains: SpikeTrains) -> jax.Array:
    r"""Compute the coefficient of variation of the inter-spike intervals.

    .. math::

        \mathrm{CV} = \frac{\sigma_I}{\mu_I},

    with the mean :math:`\mu_I` and the population standard deviation
    :math:`\sigma_I` (``ddof=0``) of the intervals of each neuron.

    Parameters
    ----------
    trains : SpikeTrains
        Spike trains with ``N`` neurons.

    Returns
    -------
    jax.Array
        CV of every neuron, shape ``(N,)``. ``nan`` for a neuron with fewer
        than 2 intervals (fewer than 3 spikes), or whose intervals are all 0.

    Raises
    ------
    TypeError
        If ``trains`` is not a :class:`~braintools.spike.SpikeTrains`.

    See Also
    --------
    isi, cv2, lv

    Notes
    -----
    A Poisson process has :math:`\mathrm{CV} = 1`, a perfectly regular train
    :math:`\mathrm{CV} = 0` [1]_. The value matches
    ``scipy.stats.variation`` (which ``elephant.statistics.cv`` is) applied
    to the intervals, except that a single interval gives ``nan`` here,
    whereas ``scipy.stats.variation`` returns 0.

    Works under ``jax.jit`` and ``jax.vmap``, and is differentiable with
    respect to the spike times.

    References
    ----------
    .. [1] Softky, W. R., & Koch, C. (1993). The highly irregular firing of
           cortical cells is inconsistent with temporal integration of random
           EPSPs. Journal of Neuroscience, 13(1), 334-350.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.spike import SpikeTrains
        >>> from braintools.analysis.spike import cv
        >>> raster = jnp.zeros((10, 2)).at[jnp.array([0, 2, 6]), 0].set(1).at[::3, 1].set(1)
        >>> st = SpikeTrains.from_dense(raster, dt=1.)
        >>> cv(st)
        Array([0.33333334, 0.        ], dtype=float32)
    """
    _check_trains(trains)
    intervals, valid = _intervals(trains)
    n = jnp.sum(valid, axis=-1).astype(intervals.dtype)
    ok = n >= 2
    mean = jnp.sum(intervals, axis=-1) / jnp.where(ok, n, 1)
    dev = jnp.where(valid, intervals - mean[..., None], 0)
    var = jnp.sum(dev * dev, axis=-1) / jnp.where(ok, n, 1)
    pos = var > 0
    std = jnp.where(pos, jnp.sqrt(jnp.where(pos, var, 1)), 0)
    return _safe_div(std, mean, ok & (mean != 0))


def _adjacent_ratio(trains: SpikeTrains):
    """``(I_{k+1} - I_k) / (I_{k+1} + I_k)`` for consecutive intervals, and its mask."""
    intervals, valid = _intervals(trains)
    first, second = intervals[..., :-1], intervals[..., 1:]
    pair = valid[..., 1:]
    total = first + second
    ok = pair & (total != 0)
    ratio = (second - first) / jnp.where(ok, total, 1)
    # A valid pair whose intervals sum to 0 (repeated spike times) is undefined.
    ratio = jnp.where(pair & ~ok, jnp.nan, jnp.where(ok, ratio, 0))
    n_pairs = jnp.sum(pair, axis=-1).astype(intervals.dtype)
    return ratio, n_pairs


@set_module_as('braintools.analysis.spike')
def cv2(trains: SpikeTrains) -> jax.Array:
    r"""Compute the local coefficient of variation CV2 of Holt et al. (1996).

    .. math::

        \mathrm{CV2} = \frac{1}{n-1} \sum_{k=1}^{n-1}
        \frac{2 |I_{k+1} - I_k|}{I_{k+1} + I_k},

    for the :math:`n` intervals :math:`I_k` of a neuron [1]_. Comparing only
    neighbouring intervals makes CV2 insensitive to slow rate changes.

    Parameters
    ----------
    trains : SpikeTrains
        Spike trains with ``N`` neurons.

    Returns
    -------
    jax.Array
        CV2 of every neuron, shape ``(N,)``. ``nan`` for a neuron with fewer
        than 2 intervals, or with two consecutive zero intervals.

    Raises
    ------
    TypeError
        If ``trains`` is not a :class:`~braintools.spike.SpikeTrains`.

    See Also
    --------
    isi, cv, lv

    Notes
    -----
    Matches ``elephant.statistics.cv2(isi, with_nan=True)``. CV2 is about 1
    for a Poisson process and 0 for a regular train. Works under ``jax.jit``
    and ``jax.vmap``, and is differentiable with respect to the spike times
    (except where two neighbouring intervals are equal, where ``|x|`` has a
    kink).

    References
    ----------
    .. [1] Holt, G. R., Softky, W. R., Koch, C., & Douglas, R. J. (1996).
           Comparison of discharge variability in vitro and in vivo in cat
           visual cortex neurons. Journal of Neurophysiology, 75(5),
           1806-1814.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.spike import SpikeTrains
        >>> from braintools.analysis.spike import cv2
        >>> raster = jnp.zeros((10, 2)).at[jnp.array([0, 1, 3]), 0].set(1).at[0, 1].set(1)
        >>> st = SpikeTrains.from_dense(raster, dt=1.)
        >>> cv2(st)
        Array([0.6666667,       nan], dtype=float32)
    """
    _check_trains(trains)
    ratio, n_pairs = _adjacent_ratio(trains)
    return _safe_div(2. * jnp.sum(jnp.abs(ratio), axis=-1), n_pairs, n_pairs >= 1)


@set_module_as('braintools.analysis.spike')
def lv(trains: SpikeTrains) -> jax.Array:
    r"""Compute the local variation LV of Shinomoto et al. (2003).

    .. math::

        \mathrm{LV} = \frac{3}{n-1} \sum_{k=1}^{n-1}
        \left(\frac{I_k - I_{k+1}}{I_k + I_{k+1}}\right)^2,

    for the :math:`n` intervals :math:`I_k` of a neuron [1]_.

    Parameters
    ----------
    trains : SpikeTrains
        Spike trains with ``N`` neurons.

    Returns
    -------
    jax.Array
        LV of every neuron, shape ``(N,)``. ``nan`` for a neuron with fewer
        than 2 intervals, or with two consecutive zero intervals.

    Raises
    ------
    TypeError
        If ``trains`` is not a :class:`~braintools.spike.SpikeTrains`.

    See Also
    --------
    isi, cv, cv2

    Notes
    -----
    Matches ``elephant.statistics.lv(isi, with_nan=True)``. LV is 1 for a
    Poisson process and 0 for a regular train. Works under ``jax.jit`` and
    ``jax.vmap``, and is differentiable with respect to the spike times.

    References
    ----------
    .. [1] Shinomoto, S., Shima, K., & Tanji, J. (2003). Differences in
           spiking patterns among cortical neurons. Neural Computation,
           15(12), 2823-2842.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.spike import SpikeTrains
        >>> from braintools.analysis.spike import lv
        >>> raster = jnp.zeros((10, 1)).at[jnp.array([0, 1, 4]), 0].set(1)
        >>> st = SpikeTrains.from_dense(raster, dt=1.)
        >>> lv(st)
        Array([0.75], dtype=float32)
    """
    _check_trains(trains)
    ratio, n_pairs = _adjacent_ratio(trains)
    return _safe_div(3. * jnp.sum(ratio * ratio, axis=-1), n_pairs, n_pairs >= 1)


@set_module_as('braintools.analysis.spike')
def fano_factor(
    x: Any,
    bin_size: Any = None,
    axis: int = 0,
    *,
    n_bins: Optional[int] = None,
) -> jax.Array:
    r"""Compute the Fano factor, the variance-to-mean ratio of spike counts.

    .. math::

        F = \frac{\operatorname{Var}[n]}{\operatorname{E}[n]},

    with the population variance (``ddof=0``) [1]_.

    - If ``x`` is a :class:`~braintools.spike.SpikeTrains`, it is binned
      with :meth:`~braintools.spike.SpikeTrains.bin` into whole bins of
      width ``bin_size`` starting at ``t_start``, and the ratio is taken
      across the bins, separately for every neuron. A partial bin at the end
      of the recording is dropped, because its lower count would inflate the
      variance.
    - Otherwise ``x`` is an array of spike counts, and the ratio is taken
      along ``axis`` (typically trials).

    Parameters
    ----------
    x : SpikeTrains or array-like
        Spike trains, or spike counts with the trials on ``axis``.
    bin_size : float or Quantity, optional
        Bin width; required for a ``SpikeTrains``, and must be None for a
        count array. A plain number is in milliseconds.
    axis : int, default 0
        Axis of the count array along which the ratio is computed. Ignored
        for a ``SpikeTrains``.
    n_bins : int, optional
        Number of bins for a ``SpikeTrains``. Defaults to the number of whole
        bins, ``floor(duration / bin_size)`` (a duration that is a multiple of
        ``bin_size`` up to float round-off keeps its last bin). The default
        needs concrete ``t_start`` and ``t_stop``; pass ``n_bins`` explicitly
        when the trains are traced (e.g. an argument of a
        ``jax.jit``-compiled function). An explicit value is used as given,
        so bins past ``t_stop`` (partial or empty) are included.

    Returns
    -------
    jax.Array
        For a ``SpikeTrains``, shape ``(N,)``; for a count array, the shape
        of ``x`` without ``axis``. ``nan`` where the mean count is 0, and for
        a ``SpikeTrains`` shorter than one bin.

    Raises
    ------
    ValueError
        If ``bin_size`` is missing for a ``SpikeTrains`` or given for a count
        array, if ``bin_size`` is not positive, or if ``n_bins`` is missing
        for traced ``t_start``/``t_stop``.
    TypeError
        If the count array is a Quantity with a physical unit.

    See Also
    --------
    braintools.spike.SpikeTrains.bin : The binning used for spike trains.
    cv : The analogous statistic of the intervals.

    Notes
    -----
    For a Poisson process the Fano factor is 1 for every bin width. The
    count-array path matches ``elephant.statistics.fanofactor`` applied to
    one spike train per trial (which computes ``counts.var() /
    counts.mean()``). Works under ``jax.jit`` and ``jax.vmap``; the
    count-array path is differentiable with respect to the counts.

    References
    ----------
    .. [1] Fano, U. (1947). Ionization yield of radiations. II. The
           fluctuations of the number of ions. Physical Review, 72(1), 26.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from braintools.spike import SpikeTrains
        >>> from braintools.analysis.spike import fano_factor
        >>> raster = jnp.zeros((8, 2)).at[jnp.array([0, 1, 2, 4]), 0].set(1).at[::2, 1].set(1)
        >>> st = SpikeTrains.from_dense(raster, dt=1. * u.ms)
        >>> fano_factor(st, bin_size=2. * u.ms)
        Array([0.5, 0. ], dtype=float32)
        >>>
        >>> fano_factor(jnp.array([[3., 1.], [5., 1.], [4., 1.]]), axis=0)
        Array([0.16666667, 0.        ], dtype=float32)
    """
    if isinstance(x, SpikeTrains):
        if bin_size is None:
            raise ValueError('`bin_size` is required when `x` is a SpikeTrains.')
        unit = x._unit
        b = _static_time(bin_size, unit, 'bin_size')
        if b <= 0:
            raise ValueError(f'`bin_size` must be positive, got {bin_size!r}.')
        if n_bins is None:
            if _is_traced(x._t_start_m, x._t_stop_m):
                raise ValueError(
                    'fano_factor: the number of whole bins depends on t_start/t_stop, '
                    'which are traced here; pass `n_bins=` explicitly.'
                )
            n_bins = _floor_ratio(float(x._t_stop_m) - float(x._t_start_m), b, x._times_m.dtype)
        counts = x.bin(_wrap(b, unit), n_bins=n_bins).astype(x._times_m.dtype)
        axis = 0
    else:
        if bin_size is not None:
            raise ValueError('`bin_size` only applies when `x` is a SpikeTrains; '
                             'for a count array, pass `axis` instead.')
        if isinstance(x, u.Quantity) and not x.unit.is_unitless:
            raise TypeError(f'spike counts must be unitless, got unit {x.unit}.')
        counts = jnp.asarray(_split(x)[0])
        if not jnp.issubdtype(counts.dtype, jnp.floating):
            counts = counts.astype(jnp.result_type(float))
    mean = jnp.mean(counts, axis=axis)
    var = jnp.var(counts, axis=axis)
    return _safe_div(var, mean, mean != 0)
