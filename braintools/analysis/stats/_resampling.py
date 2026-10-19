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

"""Resampling: randomised data for significance tests, and bootstrap intervals.

The functions that randomise spike trains or signals generate what the
literature calls *surrogate data*; the names avoid the word to prevent
confusion with the :mod:`braintools.surrogate` (surrogate gradient) package.
"""

import operator
from typing import Any, Callable, Tuple

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braintools._misc import set_module_as
from braintools.spike._spike_trains import SpikeTrains, _is_traced

__all__ = [
    'jitter_spikes',
    'shuffle_isi',
    'phase_randomize',
    'bootstrap_ci',
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _check_trains(trains: Any) -> None:
    if not isinstance(trains, SpikeTrains):
        raise TypeError(
            f'`trains` must be a braintools.spike.SpikeTrains, got {type(trains).__name__}. '
            f'Build one with SpikeTrains.from_dense(raster, dt=...).'
        )


def _time_value(value: Any, unit: Any, name: str) -> Any:
    """Express a time as a plain value in ``unit`` (``None``: unitless trains, in ms).

    A Quantity is converted; a plain number is taken to be in milliseconds
    (the ``brainstate`` convention) and converted to ``unit``.
    """
    if isinstance(value, u.Quantity) and not value.unit.is_unitless:
        target = u.ms if unit is None else unit
        try:
            return value.to_decimal(target)
        except Exception as e:  # incompatible dimension
            raise TypeError(f'`{name}` must be a time, got {value!r}.') from e
    value = _split_any(value)[0]
    if unit is None or unit == u.ms:
        return value
    return value * float(u.Quantity(1., unit=u.ms).to_decimal(unit))


def _split_any(x: Any):
    """Split ``x`` into ``(mantissa, unit)``; ``unit`` is None for plain input."""
    if isinstance(x, u.Quantity):
        return x.mantissa, x.unit
    return x, None


def _wrap(mantissa: Any, unit: Any) -> Any:
    return mantissa if unit is None else u.Quantity(mantissa, unit=unit)


def _normalize_axis(axis: Any, ndim: int) -> int:
    try:
        axis = operator.index(axis)
    except TypeError:
        raise TypeError(f'`axis` must be an integer, got {axis!r}.') from None
    if not -ndim <= axis < ndim:
        raise ValueError(f'`axis`={axis} is out of bounds for an array with {ndim} dimensions.')
    return axis % ndim


# ----------------------------------------------------------------------------
# spike trains
# ----------------------------------------------------------------------------

@set_module_as('braintools.analysis.stats')
def jitter_spikes(key: jax.Array, trains: SpikeTrains, width: Any, method: str = 'uniform') -> SpikeTrains:
    r"""Randomly displace every spike by an independent offset.

    Spike jittering destroys temporal structure finer than ``width`` while
    keeping each neuron's spike count and its slow rate profile [1]_ [2]_.
    Statistics recomputed on many jittered copies give the null
    distribution of a significance test (the copies are called *surrogate
    data* in the literature).

    - ``'uniform'``: offsets drawn from :math:`U(-w, w)`;
    - ``'gaussian'``: offsets drawn from :math:`\mathcal{N}(0, w^2)`.

    Spikes moved outside ``[t_start, t_stop)`` are reflected back at the
    bounds, and the times are re-sorted per neuron.

    Parameters
    ----------
    key : jax.Array
        PRNG key, from ``jax.random.key`` or ``jax.random.PRNGKey``.
    trains : SpikeTrains
        Spike trains to jitter.
    width : float or Quantity
        Jitter half-width (``'uniform'``) or standard deviation
        (``'gaussian'``). Must be non-negative. A Quantity is converted to
        the unit of the spike times; a plain number is in milliseconds (the
        ``brainstate`` convention), and unitless spike times are taken to be
        in milliseconds as well.
    method : {'uniform', 'gaussian'}, default 'uniform'
        Offset distribution.

    Returns
    -------
    SpikeTrains
        New trains with the same ``counts``, capacity, ``t_start``,
        ``t_stop`` and ``overflow``. Valid times are sorted per neuron and lie
        in ``[t_start, t_stop)``; padding stays ``+inf`` at the end.

    Raises
    ------
    TypeError
        If ``trains`` is not a ``SpikeTrains``, or ``width`` is a Quantity
        that is not a time.
    ValueError
        If ``method`` is unknown or a concrete ``width`` is negative.

    See Also
    --------
    shuffle_isi : Randomise spike times while keeping the ISI distribution.
    phase_randomize : Randomised copies of continuous signals.

    Notes
    -----
    Works under ``jax.jit`` and ``jax.vmap``; vmapping over a batch of keys
    produces a batch of independent jittered copies. The reflection is
    periodic, so offsets larger than the recording are folded back as well.
    Not intended for differentiation.

    References
    ----------
    .. [1] Date, A., Bienenstock, E., & Geman, S. (1998). On the temporal
       resolution of neural activity. Technical report, Division of Applied
       Mathematics, Brown University.
    .. [2] Grün, S. (2009). Data-driven significance estimation for precise
       spike correlation. *Journal of Neurophysiology*, 101(3), 1126-1140.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from braintools.spike import SpikeTrains
        >>> from braintools.analysis.stats import jitter_spikes
        >>> raster = jnp.zeros((100, 2)).at[::10, 0].set(1.).at[5, 1].set(1.)
        >>> st = SpikeTrains.from_dense(raster, dt=1. * u.ms)
        >>> jittered = jitter_spikes(jax.random.key(0), st, 2. * u.ms)
        >>> jittered.counts
        Array([10,  1], dtype=int32)
        >>> bool(u.math.all(u.math.abs(jittered.times[0] - st.times[0]) <= 2. * u.ms))
        True
        >>>
        >>> # 100 independent copies for a significance test:
        >>> keys = jax.random.split(jax.random.key(1), 100)
        >>> copies = jax.vmap(lambda k: jitter_spikes(k, st, 2. * u.ms))(keys)
        >>> copies.times.shape
        (100, 2, 10)
    """
    _check_trains(trains)
    if method not in ('uniform', 'gaussian'):
        raise ValueError(f"`method` must be 'uniform' or 'gaussian', got {method!r}.")
    unit = trains._unit
    w = _time_value(width, unit, 'width')
    if not _is_traced(w) and float(np.asarray(w)) < 0:
        raise ValueError(f'`width` must be non-negative, got {width!r}.')
    times = trains._times_m
    dtype = times.dtype
    w = jnp.asarray(w, dtype=dtype)
    mask = trains.mask
    t0 = trains._t_start_m
    t1 = trains._t_stop_m

    if method == 'uniform':
        offsets = jax.random.uniform(key, times.shape, dtype, minval=-1., maxval=1.) * w
    else:
        offsets = jax.random.normal(key, times.shape, dtype) * w

    # Fold into [0, L] by reflection at both bounds (periodic with period 2L).
    length = t1 - t0
    ok = length > 0
    period = jnp.where(ok, 2 * length, 1.)
    y = jnp.mod(jnp.where(mask, times, t0) + offsets - t0, period)
    y = jnp.where(y > length, period - y, y)
    upper = jnp.nextafter(t1, jnp.asarray(-jnp.inf, dtype))
    new = jnp.where(ok, jnp.clip(t0 + y, t0, upper), t0)
    new = jnp.sort(jnp.where(mask, new, jnp.inf), axis=-1)
    return SpikeTrains._make(new, trains.counts, t0, t1, trains.overflow, unit)


@set_module_as('braintools.analysis.stats')
def shuffle_isi(key: jax.Array, trains: SpikeTrains) -> SpikeTrains:
    r"""Randomly permute the inter-spike intervals of every neuron.

    The intervals of each neuron are shuffled and the spike times are
    rebuilt from its first spike, so the first and last spikes, the spike
    count and the ISI distribution are preserved while the serial order of
    the intervals (and the alignment between neurons) is destroyed [1]_.
    Statistics recomputed on many shuffled copies give the null
    distribution of a significance test (the copies are called *surrogate
    data* in the literature).

    Parameters
    ----------
    key : jax.Array
        PRNG key, from ``jax.random.key`` or ``jax.random.PRNGKey``.
    trains : SpikeTrains
        Spike trains to shuffle.

    Returns
    -------
    SpikeTrains
        New trains with the same ``counts``, capacity, ``t_start``,
        ``t_stop`` and ``overflow``; valid times are sorted per neuron and
        padding stays ``+inf`` at the end. Neurons with fewer than 3 spikes
        are unchanged.

    Raises
    ------
    TypeError
        If ``trains`` is not a ``SpikeTrains``.

    See Also
    --------
    jitter_spikes : Randomise spike times locally.

    Notes
    -----
    Each neuron gets its own key (``jax.random.split``) and a
    ``jax.random.permutation`` of its ``K - 1`` interval slots; a stable sort
    then moves the valid intervals to the front, which leaves the padding at
    the end. The times are rebuilt with a cumulative sum, so they carry
    float round-off of the order of ``eps * t``; they never pass the
    original last spike. Works under ``jax.jit`` and ``jax.vmap`` (vmapping
    over keys gives independent copies). Not intended for differentiation.

    References
    ----------
    .. [1] Grün, S. (2009). Data-driven significance estimation for precise
       spike correlation. *Journal of Neurophysiology*, 101(3), 1126-1140.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from braintools.spike import SpikeTrains
        >>> from braintools.analysis.stats import shuffle_isi
        >>> raster = jnp.zeros((20, 1)).at[jnp.array([0, 1, 3, 7, 15]), 0].set(1.)
        >>> st = SpikeTrains.from_dense(raster, dt=1.)
        >>> shuffled = shuffle_isi(jax.random.key(0), st)
        >>> sorted(jnp.diff(shuffled.times[0]).tolist())
        [1.0, 2.0, 4.0, 8.0]
        >>> float(shuffled.times[0, 0]), float(shuffled.times[0, -1])
        (0.0, 15.0)
    """
    _check_trains(trains)
    unit = trains._unit
    times = trains._times_m
    counts = trains.counts
    t0 = trains._t_start_m
    t1 = trains._t_stop_m
    n_neurons, capacity = times.shape
    if capacity < 2 or n_neurons == 0:
        return SpikeTrains._make(times, counts, t0, t1, trains.overflow, unit)

    mask = trains.mask
    slots = jnp.arange(capacity - 1, dtype=jnp.int32)
    valid = slots < (counts - 1)[:, None]
    isi = jnp.where(valid, times[:, 1:] - times[:, :-1], 0.)

    def _order(k, v):
        perm = jax.random.permutation(k, capacity - 1)
        return perm[jnp.argsort(~v[perm], stable=True)]  # valid slots first

    order = jax.vmap(_order)(jax.random.split(key, n_neurons), valid)
    shuffled = jnp.take_along_axis(isi, order, axis=1)
    first = times[:, :1]
    last = jnp.take_along_axis(times, jnp.maximum(counts - 1, 0)[:, None], axis=1)
    rest = jnp.minimum(first + jnp.cumsum(shuffled, axis=1), last)
    new = jnp.where(mask, jnp.concatenate([first, rest], axis=1), jnp.inf)
    return SpikeTrains._make(new, counts, t0, t1, trains.overflow, unit)


# ----------------------------------------------------------------------------
# continuous signals
# ----------------------------------------------------------------------------

@set_module_as('braintools.analysis.stats')
def phase_randomize(key: jax.Array, x: Any, axis: int = 0, same_phases: bool = False) -> Any:
    r"""Randomise the Fourier phases of a real signal, keeping its power spectrum.

    The signal is transformed with a real FFT along ``axis``, every Fourier
    coefficient is multiplied by :math:`e^{i\phi_f}` with
    :math:`\phi_f \sim U(0, 2\pi)`, and the result is transformed back
    [1]_. Because only the non-negative frequencies are randomised and the
    inverse real FFT restores Hermitian symmetry, the output is real and has
    exactly the amplitude spectrum :math:`|X_f|` (hence the power spectrum
    and autocorrelation) of the input, while any nonlinear or phase
    structure is destroyed. The DC term and, for even lengths, the Nyquist
    term keep phase 0, so they stay real and unchanged.

    With ``same_phases=True`` all channels share one set of random phases,
    which also preserves the cross-spectra and hence the (circular)
    cross-correlations between channels [2]_. Statistics recomputed on many
    randomised copies give the null distribution of a significance test
    (the copies are called *surrogate data* in the literature).

    Parameters
    ----------
    key : jax.Array
        PRNG key, from ``jax.random.key`` or ``jax.random.PRNGKey``.
    x : array-like or Quantity
        Real signal; time runs along ``axis``. Any unit is kept.
    axis : int, default 0
        Time axis.
    same_phases : bool, default False
        If True, draw one phase per frequency and share it across all other
        axes (channels).

    Returns
    -------
    jax.Array or Quantity
        Randomised signal with the shape, dtype and unit of ``x``.

    Raises
    ------
    TypeError
        If ``x`` is complex or ``axis`` is not an integer.
    ValueError
        If ``x`` is 0-d or ``axis`` is out of bounds.

    See Also
    --------
    jitter_spikes : Randomised copies of spike trains.
    bootstrap_ci : Bootstrap confidence intervals.

    Notes
    -----
    Works under ``jax.jit`` and ``jax.vmap`` (vmap over keys for many
    copies) and is differentiable with respect to ``x`` (the map is linear
    in ``x`` for a fixed key).

    References
    ----------
    .. [1] Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Farmer,
       J. D. (1992). Testing for nonlinearity in time series: the method of
       surrogate data. *Physica D*, 58(1-4), 77-94.
    .. [2] Prichard, D., & Theiler, J. (1994). Generating surrogate data for
       time series with several simultaneously measured variables.
       *Physical Review Letters*, 73(7), 951-954.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from braintools.analysis.stats import phase_randomize
        >>> x = jax.random.normal(jax.random.key(0), (256, 3))
        >>> y = phase_randomize(jax.random.key(1), x)
        >>> y.shape
        (256, 3)
        >>> power = lambda s: jnp.abs(jnp.fft.rfft(s, axis=0))
        >>> bool(jnp.allclose(power(x), power(y), atol=1e-3))
        True
        >>> bool(jnp.allclose(x, y))
        False
    """
    x_m, unit = _split_any(x)
    x_m = jnp.asarray(x_m)
    if jnp.issubdtype(x_m.dtype, jnp.complexfloating):
        raise TypeError('`x` must be real; got a complex array.')
    if x_m.ndim == 0:
        raise ValueError('`x` must have at least one dimension.')
    if not jnp.issubdtype(x_m.dtype, jnp.floating):
        x_m = x_m.astype(jnp.result_type(float))
    axis = _normalize_axis(axis, x_m.ndim)
    n = x_m.shape[axis]
    spec = jnp.fft.rfft(x_m, axis=axis)
    n_freq = spec.shape[axis]

    bshape = [1] * x_m.ndim
    bshape[axis] = n_freq
    phase_shape = tuple(bshape) if same_phases else spec.shape
    phases = jax.random.uniform(key, phase_shape, x_m.dtype, minval=0., maxval=2 * jnp.pi)
    freq = jnp.arange(n_freq).reshape(bshape)
    fixed = freq == 0
    if n % 2 == 0:
        fixed = fixed | (freq == n_freq - 1)
    phases = jnp.where(fixed, 0., phases)
    y = jnp.fft.irfft(spec * jnp.exp(1j * phases), n=n, axis=axis).astype(x_m.dtype)
    return _wrap(y, unit)


# ----------------------------------------------------------------------------
# bootstrap
# ----------------------------------------------------------------------------

@set_module_as('braintools.analysis.stats')
def bootstrap_ci(
    key: jax.Array,
    data: Any,
    statistic: Callable[[Any], Any],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    method: str = 'percentile',
    axis: int = 0,
) -> Tuple[Any, Any]:
    r"""Bootstrap confidence interval of a statistic.

    ``n_resamples`` bootstrap samples are drawn by resampling the entries of
    ``data`` along ``axis`` with replacement, and ``statistic`` is evaluated
    on each [1]_. With :math:`\alpha = (1 - \mathrm{confidence}) / 2` and
    :math:`q_p` the :math:`p`-quantile of the bootstrap distribution
    :math:`\hat\theta^*`:

    - ``'percentile'``: :math:`(q_\alpha,\ q_{1-\alpha})`;
    - ``'basic'``: :math:`(2\hat\theta - q_{1-\alpha},\ 2\hat\theta - q_\alpha)`,
      with :math:`\hat\theta` the statistic of the original data.

    Resampling is one of the building blocks of significance testing, next
    to the randomised copies (*surrogate data*) produced by
    :func:`jitter_spikes`, :func:`shuffle_isi` and :func:`phase_randomize`.

    Parameters
    ----------
    key : jax.Array
        PRNG key, from ``jax.random.key`` or ``jax.random.PRNGKey``.
    data : array-like or Quantity
        Observations along ``axis``; other axes are carried along (for
        example ``(n_trials, n_neurons)``).
    statistic : callable
        Function of one resampled array (same shape and unit as ``data``)
        returning an array or Quantity, e.g. ``lambda d: jnp.mean(d, axis=0)``.
        It is traced once and vmapped over the resamples, so it must be
        JAX-traceable.
    n_resamples : int, default 1000
        Number of bootstrap samples (static).
    confidence : float, default 0.95
        Confidence level, in ``(0, 1)``.
    method : {'percentile', 'basic'}, default 'percentile'
        Interval type.
    axis : int, default 0
        Axis of the observations that is resampled.

    Returns
    -------
    low, high : jax.Array or Quantity
        Interval bounds with the shape (and unit) of the statistic.

    Raises
    ------
    TypeError
        If ``statistic`` is not callable or ``n_resamples`` is not an integer.
    ValueError
        If ``confidence`` is outside ``(0, 1)``, ``method`` is unknown,
        ``n_resamples < 1``, or ``data`` is 0-d or empty along ``axis``.

    See Also
    --------
    phase_randomize : Randomised copies of continuous signals.
    jitter_spikes : Randomised copies of spike trains.

    Notes
    -----
    The result matches ``scipy.stats.bootstrap(..., method='percentile')``
    or ``method='basic'`` within Monte-Carlo error. Quantiles use linear
    interpolation, as ``numpy.percentile``. The vmapped resamples hold
    ``n_resamples`` copies of ``data`` in memory at once. Works under
    ``jax.jit`` (with ``statistic`` closed over) and ``jax.vmap`` (for
    example over keys).

    References
    ----------
    .. [1] Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the
       Bootstrap*. Chapman & Hall/CRC.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from braintools.analysis.stats import bootstrap_ci
        >>> data = 2. + jax.random.normal(jax.random.key(0), (200,))
        >>> low, high = bootstrap_ci(jax.random.key(1), data, jnp.mean)
        >>> bool(low < jnp.mean(data) < high), bool(high - low < 0.4)
        (True, True)
        >>>
        >>> # Per-neuron intervals of the mean rate over trials:
        >>> rates = jax.random.poisson(jax.random.key(2), 5., (40, 3)).astype(float)
        >>> low, high = bootstrap_ci(jax.random.key(3), rates, lambda r: r.mean(axis=0),
        ...                          method='basic')
        >>> low.shape
        (3,)
    """
    if not callable(statistic):
        raise TypeError(f'`statistic` must be callable, got {type(statistic).__name__}.')
    if method not in ('percentile', 'basic'):
        raise ValueError(f"`method` must be 'percentile' or 'basic', got {method!r}.")
    if isinstance(n_resamples, bool):
        raise TypeError(f'`n_resamples` must be an integer, got {n_resamples!r}.')
    try:
        n_resamples = operator.index(n_resamples)
    except TypeError:
        raise TypeError(
            f'`n_resamples` must be a static Python integer, got {n_resamples!r}.'
        ) from None
    if n_resamples < 1:
        raise ValueError(f'`n_resamples` must be >= 1, got {n_resamples}.')
    confidence = float(confidence)
    if not 0. < confidence < 1.:
        raise ValueError(f'`confidence` must lie in (0, 1), got {confidence}.')

    data_m, unit = _split_any(data)
    data_m = jnp.asarray(data_m)
    if data_m.ndim == 0:
        raise ValueError('`data` must have at least one dimension.')
    axis = _normalize_axis(axis, data_m.ndim)
    n = data_m.shape[axis]
    if n == 0:
        raise ValueError('`data` is empty along `axis`.')

    idx = jax.random.randint(key, (n_resamples, n), 0, n)
    boot = jax.vmap(lambda i: statistic(_wrap(jnp.take(data_m, i, axis=axis), unit)))(idx)
    boot_m, boot_unit = _split_any(boot)
    boot_m = jnp.asarray(boot_m)
    if not jnp.issubdtype(boot_m.dtype, jnp.floating):
        boot_m = boot_m.astype(jnp.result_type(float))
    alpha = (1. - confidence) / 2.
    q_low = jnp.quantile(boot_m, alpha, axis=0)
    q_high = jnp.quantile(boot_m, 1. - alpha, axis=0)
    if method == 'basic':
        theta = statistic(_wrap(data_m, unit))
        if boot_unit is not None:
            theta = u.Quantity(theta).to_decimal(boot_unit)
        theta = jnp.asarray(theta, dtype=q_low.dtype)
        q_low, q_high = 2 * theta - q_high, 2 * theta - q_low
    return _wrap(q_low, boot_unit), _wrap(q_high, boot_unit)
