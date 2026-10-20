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


from typing import Optional

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braintools._deprecation import register_deprecation, warn_deprecated
from ._tuning import gaussian_tuning

__all__ = [
    'LatencyEncoder',
    'RateEncoder',
    'PoissonEncoder',
    'PopulationEncoder',
    'BernoulliEncoder',
    'DeltaEncoder',
    'StepCurrentEncoder',
    'SpikeCountEncoder',
    'TemporalEncoder',
    'RankOrderEncoder',
]


def _in_ms(t) -> float:
    """A time in milliseconds: Quantities are converted, plain numbers are ms."""
    if isinstance(t, u.Quantity):
        return float(t.to_decimal(u.ms))
    return float(t)


def _dt_seconds() -> float:
    """The current ``brainstate.environ`` time step in seconds."""
    return _in_ms(brainstate.environ.get_dt()) / 1000.0


class LatencyEncoder:
    r"""
    Encode input intensities as the time of a single spike (latency coding).

    Each input feature fires at most once; a larger value fires earlier.
    Inputs are expected in ``[0, 1]``; with ``min_val`` and ``max_val`` they are
    first rescaled as :math:`x = (x - \text{min_val}) / (\text{max_val} - \text{min_val})`.

    Parameters
    ----------
    min_val : float, optional
        The minimal value of the data, used for rescaling. Rescaling happens
        only when both ``min_val`` and ``max_val`` are given.
    max_val : float, optional
        The maximal value of the data, used for rescaling.
    method : {'log', 'linear'}, default 'log'
        The latency code (see Notes).
    threshold : float, default 0.01
        Value in ``[0, 1]``. For ``method='log'``, inputs at or below it fire
        at the last step when ``normalize=True``; with ``clip=True`` inputs
        below it do not fire at all.
    clip : bool, default False
        Whether features below ``threshold`` should not fire.
    tau : Quantity or float, default 1 ms
        Time scale of the code when ``normalize=False``, and the default
        window when ``n_time`` is None. Plain numbers are in milliseconds.
    normalize : bool, default False
        Whether to scale the codes to the window (see Notes).
    first_spk_time : Quantity or float, default 0 ms
        Time of the earliest possible spike. Plain numbers are in
        milliseconds.
    epsilon : float, default 1e-7
        Small value that keeps the ``'log'`` code finite.

    Raises
    ------
    ValueError
        If ``method`` is not ``'linear'`` or ``'log'``, or ``threshold`` is
        outside ``[0, 1]``.

    See Also
    --------
    RankOrderEncoder : One spike per feature, ordered by value or rank.
    SpikeCountEncoder : A fixed number of spikes per feature.

    Notes
    -----
    The output has ``n_time`` time steps (the window). Spike times are computed
    as follows, with :math:`t_0` = ``first_spk_time``:

    - ``method='linear'``: :math:`t(x) = t_0 + T\,(1 - x)`.
    - ``method='log'`` (RC charging time of a leaky integrator driven by
      :math:`x`): :math:`t(x) = t_0 + T \log\frac{x}{x - \theta}`, where
      :math:`\theta` is ``threshold``; inputs at or below :math:`\theta` are
      raised to :math:`\theta + \epsilon`.

    With ``normalize=True`` the scale :math:`T` is chosen so the slowest input
    (``x = 0`` for ``'linear'``, ``x = threshold`` for ``'log'``) fires at the
    last step of the window. For ``'linear'``, ``x = 1`` then fires at
    :math:`t_0`; for ``'log'`` the code is strongly compressed toward
    :math:`t_0` for all inputs well above ``threshold``. With
    ``normalize=False``, :math:`T` is the physical time constant ``tau``, and
    spikes that would fall after the window are dropped.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> brainstate.environ.set(dt=1. * u.ms)
        >>> encoder = braintools.spike.LatencyEncoder(method='linear', normalize=True)
        >>> encoder(jnp.array([0.0, 0.5, 1.0]), n_time=5)
        Array([[0., 0., 1.],
               [0., 0., 0.],
               [0., 1., 0.],
               [0., 0., 0.],
               [1., 0., 0.]], dtype=float32)
        >>>
        >>> # The log code fires inputs well above threshold early.
        >>> log_encoder = braintools.spike.LatencyEncoder(method='log', normalize=True, threshold=0.1)
        >>> spikes = log_encoder(jnp.array([0.1, 0.5, 1.0]), n_time=100)
        >>> jnp.argmax(spikes, axis=0)
        Array([99,  2,  1], dtype=int32)
    """
    __module__ = 'braintools.spike'

    def __init__(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        method: str = 'log',
        threshold: float = 0.01,
        clip: bool = False,
        tau: brainstate.typing.ArrayLike = 1. * u.ms,
        normalize: bool = False,
        first_spk_time: brainstate.typing.ArrayLike = 0. * u.ms,
        epsilon: float = 1e-7,
    ):
        super().__init__()
        if method not in ['linear', 'log']:
            raise ValueError('The conversion method can only be "linear" and "log".')
        self.method = method
        self.min_val = min_val
        self.max_val = max_val
        if threshold < 0 or threshold > 1:
            raise ValueError(f"``threshold`` [{threshold}] must be between [0, 1]")
        self.threshold = threshold
        self.clip = clip
        self.tau = tau
        self.normalize = normalize
        self.first_spk_time = first_spk_time
        self.epsilon = epsilon

    def __call__(
        self,
        data: brainstate.typing.ArrayLike,
        n_time: Optional[brainstate.typing.ArrayLike] = None,
    ) -> jax.Array:
        """Generate latency spikes for ``data``.

        Parameters
        ----------
        data : array_like
            The intensities to encode, nominally in ``[0, 1]``.
        n_time : int or Quantity, optional
            Length of the window: a number of time steps, or a duration
            (Quantity) that is converted to steps with the current
            ``brainstate.environ`` ``dt``. Defaults to the duration ``tau``.

        Returns
        -------
        jax.Array
            Spike trains with shape ``(n_steps, *data.shape)``; every feature
            has at most one spike.

        Raises
        ------
        ValueError
            If the window contains less than one time step.
        """
        dt_ms = _in_ms(brainstate.environ.get_dt())
        if n_time is None:
            n_steps = int(np.ceil(_in_ms(self.tau) / dt_ms - 1e-9))
        elif isinstance(n_time, u.Quantity):
            n_steps = int(np.ceil(_in_ms(n_time) / dt_ms - 1e-9))
        else:
            n_steps = int(n_time)
        if n_steps < 1:
            raise ValueError(f'The window must contain at least one time step, got {n_steps}.')
        first_step = int(round(_in_ms(self.first_spk_time) / dt_ms))
        last_step = n_steps - 1

        x = jnp.asarray(data)
        if self.min_val is not None and self.max_val is not None:
            x = (x - self.min_val) / (self.max_val - self.min_val)

        if self.method == 'linear':
            code = 1.0 - x  # 0 for x = 1, 1 for x = 0
            code_max = 1.0
        else:
            x_safe = jnp.maximum(x, self.threshold + self.epsilon)
            code = jnp.log(x_safe / (x_safe - self.threshold))
            code_max = float(np.log((self.threshold + self.epsilon) / self.epsilon))

        if self.normalize:
            steps = first_step + (last_step - first_step) * code / code_max
        else:
            steps = first_step + code * _in_ms(self.tau) / dt_ms
        steps = jnp.round(steps).astype(int)
        fire = (steps >= 0) & (steps <= last_step)
        if self.clip:
            fire = fire & (x >= self.threshold)
        # An index of ``n_steps`` is out of range for one_hot and yields no spike.
        steps = jnp.where(fire, steps, n_steps)
        return jax.nn.one_hot(steps, n_steps, axis=0, dtype=jnp.result_type(float))


class RateEncoder:
    r"""
    Encode analog values into spike rates using various rate encoding methods.

    The rate encoder converts continuous input values into spike trains where
    the firing rate grows with the input intensity. An input :math:`x` is
    mapped to a rate between ``min_rate`` and ``max_rate`` (in Hz), and every
    time step fires independently with probability :math:`\text{rate} \cdot dt`
    (clipped to ``[0, 1]``), where :math:`dt` is the ``brainstate.environ``
    time step.

    Parameters
    ----------
    gain : float, default 100.0
        Firing rate (Hz) for an input of 1; used as ``max_rate`` when that is
        not given.
    method : {'linear', 'exponential', 'sqrt'}, default 'linear'
        How the input is mapped to a rate (see Notes).
    min_rate : float, default 0.0
        Minimum firing rate in Hz.
    max_rate : float, optional
        Maximum firing rate in Hz. Defaults to ``gain``.
    normalize : bool, default False
        Whether to rescale inputs to ``[0, 1]`` with ``min_val`` and
        ``max_val`` (both must be given).
    min_val : float, optional
        Minimum value for normalization.
    max_val : float, optional
        Maximum value for normalization.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``'linear'``, ``'exponential'`` or
        ``'sqrt'``.

    See Also
    --------
    PoissonEncoder : Takes rates in Hz directly.
    BernoulliEncoder : Sets the spike probability per time step directly.

    Notes
    -----
    With :math:`r_0` = ``min_rate`` and :math:`r_1` = ``max_rate``:

    - ``'linear'``: :math:`r = r_0 + x\,(r_1 - r_0)`.
    - ``'exponential'``: :math:`r = r_0 + \frac{e^x - 1}{e - 1}\,(r_1 - r_0)`.
    - ``'sqrt'``: :math:`r = r_0 + \sqrt{x}\,(r_1 - r_0)`.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> brainstate.environ.set(dt=1. * u.ms)
        >>> brainstate.random.seed(0)
        >>> encoder = braintools.spike.RateEncoder(gain=100., method='linear')
        >>> spikes = encoder(jnp.array([0.1, 0.5, 0.9]), n_time=10000)
        >>> spikes.shape
        (10000, 3)
        >>> # The mean is a probability per step; divide by dt (in s) for Hz.
        >>> rates = spikes.mean(axis=0) / 1e-3  # approximately [10, 50, 90] Hz
    """
    __module__ = 'braintools.spike'

    def __init__(
        self,
        gain: float = 100.0,
        method: str = 'linear',
        min_rate: float = 0.0,
        max_rate: Optional[float] = None,
        normalize: bool = False,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ):
        if method not in ['linear', 'exponential', 'sqrt']:
            raise ValueError('Method must be "linear", "exponential", or "sqrt"')

        self.gain = gain
        self.method = method
        self.min_rate = min_rate
        self.max_rate = max_rate if max_rate is not None else gain
        self.normalize = normalize
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data: brainstate.typing.ArrayLike, n_time: int) -> jax.Array:
        """Generate rate-encoded spikes.

        Parameters
        ----------
        data : jax.Array
            Input array to encode.
        n_time : int
            Number of time steps.

        Returns
        -------
        jax.Array
            Spike trains with shape ``(n_time, *data.shape)`` and the dtype
            of ``data``.
        """
        x = data
        if self.normalize and self.min_val is not None and self.max_val is not None:
            x = (x - self.min_val) / (self.max_val - self.min_val)

        # Convert to rates based on method
        if self.method == 'linear':
            rates = self.min_rate + x * (self.max_rate - self.min_rate)
        elif self.method == 'exponential':
            rates = self.min_rate + (u.math.exp(x) - 1) / (u.math.exp(1) - 1) * (self.max_rate - self.min_rate)
        elif self.method == 'sqrt':
            rates = self.min_rate + u.math.sqrt(x) * (self.max_rate - self.min_rate)
        else:
            raise ValueError(f'Unsupported method: {self.method}.')

        # Convert rates (Hz) to probabilities per time step
        probs = rates * _dt_seconds()
        probs = u.math.clip(probs, 0, 1)

        # Generate spikes using Bernoulli process
        spikes = brainstate.random.bernoulli(probs, size=(n_time,) + data.shape)
        return spikes.astype(data.dtype)


_TIME_WINDOW_DEPRECATION = register_deprecation(
    'braintools.spike.PoissonEncoder(time_window=...)',
    'braintools.spike.PoissonEncoder() without time_window (it has no effect)',
    since='0.4.0',
    removed_in='1.0.0',
    kind='argument',
)


class PoissonEncoder:
    r"""
    Encode inputs as Poisson spike trains.

    The input sets the rate of a Poisson process. In each time step the number
    of events is drawn from a Poisson distribution with mean
    :math:`\lambda = \text{rate} \cdot dt` (with :math:`dt` the
    ``brainstate.environ`` time step), and the step spikes when that number is
    positive, i.e. with probability :math:`1 - e^{-\lambda}`.

    Parameters
    ----------
    time_window : float, optional
        Deprecated since 0.4.0 and unused (removed in 1.0.0); passing it emits
        a :class:`~braintools._deprecation.BraintoolsDeprecationWarning`. The
        rate is set by the input (Hz) and the ``brainstate`` time step.
    normalize : bool, default False
        If False, inputs are rates in Hz. If True, inputs are fractions of
        ``max_rate``.
    max_rate : float, default 100.0
        Rate in Hz for an input of 1 when ``normalize=True``.

    See Also
    --------
    RateEncoder : Maps inputs to rates and fires with probability ``rate * dt``.
    BernoulliEncoder : Sets the spike probability per time step directly.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> brainstate.environ.set(dt=1. * u.ms)
        >>> brainstate.random.seed(0)
        >>> rates = jnp.array([10.0, 50.0, 100.0])  # Hz
        >>> spikes = braintools.spike.PoissonEncoder()(rates, n_time=1000)  # 1 s
        >>> spikes.shape
        (1000, 3)
        >>> counts = spikes.sum(axis=0)  # approximately [10, 50, 100] spikes
    """
    __module__ = 'braintools.spike'

    def __init__(
        self,
        time_window: Optional[float] = None,
        normalize: bool = False,
        max_rate: float = 100.0,
    ):
        if time_window is not None:
            # stacklevel=2 attributes the warning to the caller of __init__.
            warn_deprecated(_TIME_WINDOW_DEPRECATION)
        self.time_window = time_window
        self.normalize = normalize
        self.max_rate = max_rate

    def __call__(self, data: brainstate.typing.ArrayLike, n_time: int) -> jax.Array:
        """Generate Poisson spike trains.

        Parameters
        ----------
        data : jax.Array
            Input rates in Hz, or fractions of ``max_rate`` when
            ``normalize=True``.
        n_time : int
            Number of time steps.

        Returns
        -------
        jax.Array
            Binary spike trains with shape ``(n_time, *data.shape)`` and the
            dtype of ``data``.
        """
        rates = data
        if self.normalize:
            rates = data * self.max_rate

        # Expected number of spikes per time step (rates in Hz)
        lam = rates * _dt_seconds()

        # Generate Poisson spikes
        spikes = brainstate.random.poisson(lam, size=(n_time,) + data.shape)
        # Convert to binary spikes (1 if >= 1 spike, 0 otherwise)
        return (spikes > 0).astype(data.dtype)


class PopulationEncoder:
    r"""
    Encode scalar values using population coding.

    Each input value is encoded by a population of neurons with overlapping
    Gaussian receptive fields whose preferred values (``centers``) are evenly
    spaced over ``[min_val, max_val]``. Neuron :math:`i` fires with rate
    :math:`\text{max_rate} \cdot e^{-(x - c_i)^2 / 2\sigma^2}` (Hz), i.e. with
    probability :math:`\text{rate} \cdot dt` per time step, so the population
    response forms a bell curve centered on the input value.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the population.
    min_val : float, default 0.0
        Minimum input value (preferred value of the first neuron).
    max_val : float, default 1.0
        Maximum input value (preferred value of the last neuron).
    sigma : float, optional
        Width of the receptive fields (standard deviation). Defaults to the
        spacing between neighbouring centers,
        ``(max_val - min_val) / (n_neurons - 1)``.
    max_rate : float, default 100.0
        Peak firing rate of a neuron in Hz.

    See Also
    --------
    RateEncoder : One neuron per feature, rate proportional to the input.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import brainunit as u
        >>> import braintools
        >>> brainstate.environ.set(dt=1. * u.ms)
        >>> brainstate.random.seed(0)
        >>> encoder = braintools.spike.PopulationEncoder(n_neurons=11, min_val=0., max_val=1.)
        >>> spikes = encoder(0.5, n_time=1000)  # neuron 5 (center 0.5) fires most
        >>> spikes.shape
        (1000, 11)
    """
    __module__ = 'braintools.spike'

    def __init__(
        self,
        n_neurons: int,
        min_val: float = 0.0,
        max_val: float = 1.0,
        sigma: Optional[float] = None,
        max_rate: float = 100.0,
    ):
        self.n_neurons = n_neurons
        self.min_val = min_val
        self.max_val = max_val
        self.sigma = sigma if sigma is not None else (max_val - min_val) / (n_neurons - 1)
        self.max_rate = max_rate

        # Create neuron preferred values (centers of receptive fields)
        self.centers = u.math.linspace(min_val, max_val, n_neurons)

    def __call__(self, data: brainstate.typing.ArrayLike, n_time: int) -> jax.Array:
        """Generate population-encoded spikes.

        Parameters
        ----------
        data : float or array_like
            Input scalar or array to encode.
        n_time : int
            Number of time steps.

        Returns
        -------
        jax.Array
            Float32 population spike trains with shape
            ``(n_time, n_neurons, *data.shape)`` (``(n_time, n_neurons)`` for
            a scalar input).
        """
        data = u.math.asarray(data)
        # Gaussian receptive fields, one per neuron's preferred value
        activations = gaussian_tuning(data[..., None], self.centers, self.sigma)
        rates = activations * self.max_rate

        # Convert to spike probabilities
        probs = rates * _dt_seconds()
        probs = u.math.clip(probs, 0, 1)

        # Generate spikes
        if data.ndim == 0:  # scalar input
            shape = (n_time, self.n_neurons)
            spikes = brainstate.random.bernoulli(probs, size=shape)
        else:
            # For array inputs, we need to handle the shape correctly
            # probs has shape (*data.shape, n_neurons)
            # We want output shape (n_time, n_neurons, *data.shape)
            shape = (n_time,) + probs.shape
            spikes = brainstate.random.bernoulli(probs, size=shape)
            # Transpose to get (n_time, n_neurons, *data.shape)
            axes = (0, data.ndim + 1) + tuple(range(1, data.ndim + 1))
            spikes = u.math.transpose(spikes, axes)

        return spikes.astype(u.math.float32)


class BernoulliEncoder:
    r"""
    Encode inputs using independent Bernoulli processes.

    Each input value is converted to a probability
    :math:`p = \text{clip}(x \cdot \text{scale}, 0, 1)`, and spikes are
    generated independently at each time step with that probability. The
    probability is per step, independent of the time step ``dt``: the same
    input gives a higher rate in Hz when ``dt`` is smaller.

    Parameters
    ----------
    scale : float, default 1.0
        Scaling factor for the input-to-probability conversion.
    normalize : bool, default True
        Whether to rescale inputs to ``[0, 1]`` with ``min_val`` and
        ``max_val``; it has no effect unless both are given.
    min_val : float, optional
        Minimum value for normalization.
    max_val : float, optional
        Maximum value for normalization.

    See Also
    --------
    RateEncoder : Fires with probability ``rate * dt`` for a rate in Hz.
    PoissonEncoder : Poisson spike trains from rates in Hz.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> brainstate.random.seed(0)
        >>> encoder = braintools.spike.BernoulliEncoder()
        >>> spikes = encoder(jnp.array([0.1, 0.5, 0.9]), n_time=1000)
        >>> spikes.shape
        (1000, 3)
        >>> p = spikes.mean(axis=0)  # approximately [0.1, 0.5, 0.9] per step
    """
    __module__ = 'braintools.spike'

    def __init__(
        self,
        scale: float = 1.0,
        normalize: bool = True,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ):
        self.scale = scale
        self.normalize = normalize
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data: brainstate.typing.ArrayLike, n_time: int) -> jax.Array:
        """Generate Bernoulli-encoded spikes.

        Parameters
        ----------
        data : jax.Array
            Input probabilities, or values to normalize.
        n_time : int
            Number of time steps.

        Returns
        -------
        jax.Array
            Bernoulli spike trains with shape ``(n_time, *data.shape)`` and
            the dtype of ``data``.
        """
        x = data
        if self.normalize and self.min_val is not None and self.max_val is not None:
            x = (x - self.min_val) / (self.max_val - self.min_val)

        probs = x * self.scale
        probs = u.math.clip(probs, 0, 1)

        spikes = brainstate.random.bernoulli(probs, size=(n_time,) + data.shape)
        return spikes.astype(data.dtype)


class DeltaEncoder:
    r"""
    Encode a time series by send-on-delta (level-crossing) modulation.

    The encoder keeps a reference level per feature, starting at the first
    sample. Whenever the signal has moved at least ``threshold`` away from the
    reference, a spike is emitted and the reference is reset to the current
    value. Slow drifts are therefore encoded as well as sudden jumps: a ramp of
    total height :math:`h` produces about :math:`h / \text{threshold}` spikes.

    Parameters
    ----------
    threshold : float, default 0.1
        Change relative to the reference that triggers a spike.
    positive_only : bool, default False
        Emit spikes only for increases (output 0/1). The reference still
        follows decreases.
    absolute : bool, default False
        If False, increases give ``+1`` and decreases ``-1`` (ON/OFF events).
        If True, every crossing gives ``1``.
    normalize : bool, default True
        Rescale each feature to ``[0, 1]`` over time first, so ``threshold``
        is a fraction of the feature's range.

    See Also
    --------
    StepCurrentEncoder : Encodes inputs as constant currents.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> encoder = braintools.spike.DeltaEncoder(threshold=0.25, normalize=False)
        >>> encoder(jnp.array([0.0, 0.1, 0.2, 0.3, 0.3, 0.0]))
        Array([ 0.,  0.,  0.,  1.,  0., -1.], dtype=float32)
        >>>
        >>> absolute = braintools.spike.DeltaEncoder(threshold=0.25, normalize=False, absolute=True)
        >>> absolute(jnp.array([0.0, 0.1, 0.2, 0.3, 0.3, 0.0]))
        Array([0., 0., 0., 1., 0., 1.], dtype=float32)
    """
    __module__ = 'braintools.spike'

    def __init__(
        self,
        threshold: float = 0.1,
        positive_only: bool = False,
        absolute: bool = False,
        normalize: bool = True,
    ):
        self.threshold = threshold
        self.positive_only = positive_only
        self.absolute = absolute
        self.normalize = normalize

    def __call__(self, data: brainstate.typing.ArrayLike) -> jax.Array:
        """Generate delta-encoded spikes.

        Parameters
        ----------
        data : array_like
            Time series with shape ``(n_time, *features)``.

        Returns
        -------
        jax.Array
            Spike trains with the same shape as ``data``: ``+1``/``-1``
            events, or ``0``/``1`` with ``absolute=True`` or
            ``positive_only=True``. The first time step never spikes.
        """
        data = jnp.asarray(data)
        squeeze = data.ndim == 1
        x = data[:, None] if squeeze else data
        x = x.astype(jnp.result_type(float))
        if self.normalize:
            x_min = jnp.min(x, axis=0, keepdims=True)
            x_max = jnp.max(x, axis=0, keepdims=True)
            x = (x - x_min) / (x_max - x_min + 1e-8)

        threshold = self.threshold

        def step(ref, xt):
            d = xt - ref
            crossed = jnp.abs(d) >= threshold
            ref = jnp.where(crossed, xt, ref)
            return ref, jnp.where(crossed, jnp.sign(d), 0.0)

        _, events = jax.lax.scan(step, x[0], x[1:])
        events = jnp.concatenate([jnp.zeros_like(x[:1]), events], axis=0)
        if self.positive_only:
            events = (events > 0).astype(events.dtype)
        elif self.absolute:
            events = jnp.abs(events)
        return events[:, 0] if squeeze else events


class StepCurrentEncoder:
    r"""
    Encode inputs as step current injections for LIF neurons.

    Converts input values to constant current levels
    :math:`I = x \cdot \text{current_scale} + \text{offset}` that can be
    injected into integrate-and-fire neurons; the neuron model then produces
    the spikes, with a firing rate set by the current amplitude. The values
    are plain numbers: multiply by a unit (e.g. ``u.nA``) when the model
    expects a Quantity.

    Parameters
    ----------
    current_scale : float, default 10.0
        Scaling factor that converts inputs to currents.
    offset : float, default 0.0
        Baseline current offset.
    normalize : bool, default True
        Whether to rescale inputs to ``[0, 1]`` with ``min_val`` and
        ``max_val``; it has no effect unless both are given.
    min_val : float, optional
        Minimum value for normalization.
    max_val : float, optional
        Maximum value for normalization.

    See Also
    --------
    DeltaEncoder : Spikes on changes of a time series.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> encoder = braintools.spike.StepCurrentEncoder(current_scale=10.0)
        >>> currents = encoder(jnp.array([0.1, 0.5, 1.0]), n_time=100)
        >>> currents.shape
        (100, 3)
        >>> currents[0]
        Array([ 1.,  5., 10.], dtype=float32)
    """
    __module__ = 'braintools.spike'

    def __init__(
        self,
        current_scale: float = 10.0,  # nA
        offset: float = 0.0,
        normalize: bool = True,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ):
        self.current_scale = current_scale
        self.offset = offset
        self.normalize = normalize
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data: brainstate.typing.ArrayLike, n_time: int) -> jax.Array:
        """Generate step current signals.

        Parameters
        ----------
        data : jax.Array
            Input values to convert to currents.
        n_time : int
            Number of time steps.

        Returns
        -------
        jax.Array
            Step current signals with shape ``(n_time, *data.shape)``; every
            time step holds the same current.
        """
        x = data
        if self.normalize and self.min_val is not None and self.max_val is not None:
            x = (x - self.min_val) / (self.max_val - self.min_val)

        currents = x * self.current_scale + self.offset

        # Repeat for all time steps
        return u.math.tile(currents, (n_time,) + (1,) * currents.ndim)


class SpikeCountEncoder:
    r"""
    Encode inputs as exact spike counts over time windows.

    Each feature fires exactly :math:`\lfloor x \cdot \text{max_spikes} \rfloor`
    times in the window (at most ``n_time``, since a step holds at most one
    spike), spread evenly or at random distinct time steps.

    Parameters
    ----------
    max_spikes : int, default 10
        Number of spikes for an input value of 1.0.
    distribution : {'random', 'uniform'}, default 'random'
        How to distribute the spikes: at random distinct steps, or evenly
        over the window.
    normalize : bool, default True
        Whether to clip inputs to ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``distribution`` is not ``'uniform'`` or ``'random'``.

    See Also
    --------
    LatencyEncoder : One spike per feature, timed by its value.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> brainstate.random.seed(0)
        >>> encoder = braintools.spike.SpikeCountEncoder(max_spikes=10, distribution='uniform')
        >>> spikes = encoder(jnp.array([0.2, 0.5, 1.0]), n_time=100)
        >>> spikes.sum(axis=0)
        Array([ 2.,  5., 10.], dtype=float32)
    """
    __module__ = 'braintools.spike'

    def __init__(
        self,
        max_spikes: int = 10,
        distribution: str = 'random',
        normalize: bool = True,
    ):
        if distribution not in ['uniform', 'random']:
            raise ValueError('Distribution must be "uniform" or "random"')

        self.max_spikes = max_spikes
        self.distribution = distribution
        self.normalize = normalize

    def __call__(self, data: brainstate.typing.ArrayLike, n_time: int) -> jax.Array:
        """Generate spike count-encoded trains.

        Parameters
        ----------
        data : jax.Array
            Input values determining the spike counts.
        n_time : int
            Number of time steps.

        Returns
        -------
        jax.Array
            Spike trains with shape ``(n_time, *data.shape)`` and the dtype of
            ``data``, holding the exact spike counts.
        """
        x = data
        if self.normalize:
            x = u.math.clip(x, 0, 1)

        # At most one spike per time step, so a window holds at most n_time spikes.
        spike_counts = u.math.clip((x * self.max_spikes).astype(int), 0, n_time)
        spikes = u.math.zeros((n_time,) + data.shape, dtype=data.dtype)

        if self.distribution == 'uniform':
            # Distribute spikes uniformly
            spike_counts_flat = spike_counts.flatten()
            for i in range(data.size):
                idx = np.unravel_index(i, data.shape)
                count = spike_counts_flat[i]
                if count > 0:
                    spike_times = u.math.linspace(0, n_time - 1, count).astype(int)
                    spikes = spikes.at[(spike_times,) + idx].set(1)

        elif self.distribution == 'random':
            # Distribute spikes randomly using a simpler approach
            spike_counts_flat = spike_counts.flatten()
            for i in range(data.size):
                idx = jnp.unravel_index(i, data.shape)
                count = int(spike_counts_flat[i])
                if count > 0 and count <= n_time:
                    spike_times = brainstate.random.choice(n_time, (count,), replace=False)
                    spikes = spikes.at[(spike_times,) + idx].set(1)

        return spikes


class TemporalEncoder:
    r"""
    Encode temporal patterns using synchronized spike timing.

    Encodes a sequence of integer symbols. Each symbol occupies a slot of
    ``pattern_length`` time steps; during that slot the neuron with the
    symbol's index fires a fixed pattern of spikes (about 30 % of the slot,
    shifted by twice the symbol index), with Gaussian jitter. Symbols outside
    ``[0, n_patterns)`` leave their slot silent.

    Parameters
    ----------
    n_patterns : int
        Number of distinct symbols (and output neurons).
    pattern_length : int, default 10
        Length of each temporal pattern in time steps.
    jitter : float, default 0.1
        Standard deviation of the temporal jitter added to spike times, as a
        fraction of ``pattern_length``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> brainstate.random.seed(0)
        >>> encoder = braintools.spike.TemporalEncoder(n_patterns=3, pattern_length=10, jitter=0.0)
        >>> sequence = jnp.array([0, 1, 2, 1, 0])
        >>> spikes = encoder(sequence)
        >>> spikes.shape
        (50, 3)
        >>> # The active neuron in each slot decodes the symbol.
        >>> jnp.argmax(spikes.reshape(5, 10, 3).sum(axis=1), axis=1)
        Array([0, 1, 2, 1, 0], dtype=int32)
    """
    __module__ = 'braintools.spike'

    def __init__(
        self,
        n_patterns: int,
        pattern_length: int = 10,
        jitter: float = 0.1,
    ):
        self.n_patterns = n_patterns
        self.pattern_length = pattern_length
        self.jitter = jitter

        # Pre-define temporal patterns
        self.patterns = self._create_patterns()

    def _create_patterns(self) -> dict:
        """Create distinct temporal patterns for each input value."""
        patterns = {}
        for i in range(self.n_patterns):
            # Create a unique temporal pattern
            pattern_times = u.math.linspace(0, self.pattern_length - 1,
                                            max(1, int(self.pattern_length * 0.3))).astype(int)
            # Shift pattern based on index
            pattern_times = (pattern_times + i * 2) % self.pattern_length
            patterns[i] = pattern_times
        return patterns

    def __call__(self, data: brainstate.typing.ArrayLike) -> jax.Array:
        """Generate temporally-encoded spikes.

        Parameters
        ----------
        data : array_like
            Sequence of integer symbols to encode.

        Returns
        -------
        jax.Array
            Temporal spike patterns with shape
            ``(len(data) * pattern_length, n_patterns)``.
        """
        sequence_length = len(data)
        total_time = sequence_length * self.pattern_length
        spikes = u.math.zeros((total_time, self.n_patterns))

        for t, value in enumerate(data):
            if 0 <= value < self.n_patterns:
                pattern_times = self.patterns[int(value)]

                # Add temporal jitter
                jitter_offset = brainstate.random.randn(*pattern_times.shape) * self.jitter * self.pattern_length
                jittered_times = pattern_times + jitter_offset
                jittered_times = u.math.clip(jittered_times, 0, self.pattern_length - 1).astype(int)

                # Place spikes in global time
                global_times = t * self.pattern_length + jittered_times
                global_times = u.math.clip(global_times, 0, total_time - 1).astype(int)

                spikes = spikes.at[global_times, value].set(1)

        return spikes


class RankOrderEncoder:
    r"""
    Encode inputs using rank order coding.

    Orders input features by their values and generates one spike per feature
    in rank order. The most active features spike first, followed by less
    active ones, which preserves the relative ordering of the input features.

    Parameters
    ----------
    use_values : bool, default True
        If True, the spike step is proportional to :math:`1 - x` (equal
        values fire together). If False, only the rank matters and spikes are
        evenly spaced over the window.
    normalize : bool, default True
        Whether to rescale the input vector to ``[0, 1]`` first.
    invert : bool, default False
        Whether to invert the order (smallest first).

    See Also
    --------
    LatencyEncoder : One spike per feature, timed by a linear or log code.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> encoder = braintools.spike.RankOrderEncoder()
        >>> spikes = encoder(jnp.array([0.1, 0.8, 0.3, 0.9, 0.2]), n_time=10)
        >>> # Feature 3 (0.9) spikes first, then feature 1 (0.8), etc.
        >>> jnp.argmax(spikes, axis=0)
        Array([9, 1, 6, 0, 7], dtype=int32)
    """
    __module__ = 'braintools.spike'

    def __init__(
        self,
        use_values: bool = True,
        normalize: bool = True,
        invert: bool = False,
    ):
        self.use_values = use_values
        self.normalize = normalize
        self.invert = invert

    def __call__(self, data: brainstate.typing.ArrayLike, n_time: int) -> jax.Array:
        """Generate rank-order encoded spikes.

        Parameters
        ----------
        data : jax.Array
            Input feature vector to encode.
        n_time : int
            Number of time steps available for encoding.

        Returns
        -------
        jax.Array
            Rank-order spike trains with shape ``(n_time, len(data))``; every
            feature spikes exactly once.
        """
        x = data
        if self.normalize:
            x_min, x_max = u.math.min(x), u.math.max(x)
            if x_max > x_min:
                x = (x - x_min) / (x_max - x_min)

        if self.invert:
            x = 1 - x

        if self.use_values:
            # Use actual values to determine spike timing
            spike_times = ((1 - x) * (n_time - 1)).astype(int)
        else:
            # Use only ranks
            ranks = u.math.argsort(u.math.argsort(-x))  # Descending rank order
            spike_times = (ranks * (n_time - 1) // len(data)).astype(int)

        spike_times = u.math.clip(spike_times, 0, n_time - 1)

        # Create spike trains
        spikes = u.math.zeros((n_time, len(data)), dtype=data.dtype)
        spikes = spikes.at[spike_times, u.math.arange(len(data))].set(1)

        return spikes
