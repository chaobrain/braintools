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


import warnings
from typing import Optional

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

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

    Each input feature fires exactly once; a larger value fires earlier.
    Inputs are expected in ``[0, 1]``; with ``min_val`` and ``max_val`` they are
    first rescaled as :math:`x = (x - \text{min_val}) / (\text{max_val} - \text{min_val})`.

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

    Example::

      >>> import brainstate, brainunit as u, jax.numpy as jnp
      >>> brainstate.environ.set(dt=1. * u.ms)
      >>> encoder = LatencyEncoder(method='linear', normalize=True)
      >>> encoder(jnp.array([0.0, 0.5, 1.0]), n_time=5)
      Array([[0., 0., 1.],
             [0., 0., 0.],
             [0., 1., 0.],
             [0., 0., 0.],
             [1., 0., 0.]], dtype=float32)

    Parameters
    ----------
    min_val : float, optional
        The minimal value of the data, used for rescaling.
    max_val : float, optional
        The maximal value of the data, used for rescaling.
    method : str, default='log'
        ``'linear'`` or ``'log'`` (see above).
    threshold : float, default=0.01
        Value in ``[0, 1]``. Inputs below it fire at the last step
        (``normalize=True``) or not at all when ``clip=True``.
    clip : bool, default=False
        Whether features below ``threshold`` should not fire.
    tau : Quantity or float, default=1 ms
        Time scale of the code (in ms) when ``normalize=False``, and the
        default window when ``n_time`` is None.
    normalize : bool, default=False
        Whether to scale the codes to the window (see above).
    first_spk_time : Quantity or float, default=0 ms
        Time of the earliest possible spike (in ms).
    epsilon : float, default=1e-7
        Small value that keeps the ``'log'`` code finite.
    """
    __module__ = 'braintools'

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

        Args:
          data: The intensities to encode, nominally in ``[0, 1]``.
          n_time: int or Quantity. Length of the window: a number of time steps,
            or a duration (Quantity) that is converted to steps with the
            current ``dt``. Defaults to the duration ``tau``.

        Returns:
          Spike trains with shape ``(n_steps, *data.shape)``; every feature has
          at most one spike.
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
    
    The rate encoder converts continuous input values into spike trains where the 
    firing rate is proportional to the input intensity. Higher input values result 
    in higher firing rates.
    
    Example::
    
      >>> import jax.numpy as jnp
      >>> data = jnp.array([0.1, 0.5, 0.9])
      >>> encoder = RateEncoder(gain=100, method='linear')
      >>> spikes = encoder(data, n_time=100)
      >>> firing_rates = jnp.mean(spikes, axis=0)  # Should be ~[10, 50, 90] Hz
    
    Args:
      gain: float. Scaling factor to convert normalized input to firing rate (Hz).
      method: str. Rate encoding method ('linear', 'exponential', 'sqrt').
      min_rate: float. Minimum firing rate in Hz.
      max_rate: float. Maximum firing rate in Hz.
      normalize: bool. Whether to normalize inputs to [0, 1] range.
      min_val: float. Minimum value for normalization.
      max_val: float. Maximum value for normalization.
    """
    __module__ = 'braintools'

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
        
        Args:
          data: Input array to encode.
          n_time: Number of time steps.
        
        Returns:
          Spike trains with shape (n_time, *data.shape).
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


class PoissonEncoder:
    r"""
    Encode inputs as Poisson spike trains.
    
    Generates spike trains where inter-spike intervals follow a Poisson distribution.
    The input intensity determines the rate parameter of the Poisson process.
    
    Example::
    
      >>> data = jnp.array([10.0, 50.0, 100.0])  # Hz
      >>> encoder = PoissonEncoder()
      >>> spikes = encoder(data, n_time=1000)  # 1 second at 1ms resolution
      >>> # Mean spike counts should be ~[10, 50, 100]
    
    Args:
      time_window: Deprecated and unused; passing it emits a ``DeprecationWarning``.
      normalize: bool. Whether to treat inputs as rates (False) or normalize to rates (True).
      max_rate: float. Maximum rate when normalizing.
    """
    __module__ = 'braintools'

    def __init__(
        self,
        time_window: Optional[float] = None,
        normalize: bool = False,
        max_rate: float = 100.0,
    ):
        if time_window is not None:
            warnings.warn(
                '`PoissonEncoder(time_window=...)` has no effect and will be removed; '
                'the rate is set by the input (Hz) and the brainstate time step.',
                DeprecationWarning,
                stacklevel=2,
            )
        self.time_window = time_window
        self.normalize = normalize
        self.max_rate = max_rate

    def __call__(self, data: brainstate.typing.ArrayLike, n_time: int) -> jax.Array:
        """Generate Poisson spike trains.
        
        Args:
          data: Input rates in Hz, or values to normalize to rates.
          n_time: Number of time steps.
          key: JAX random key.
        
        Returns:
          Poisson spike trains with shape (n_time, *data.shape).
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
    receptive fields. The population response forms a bell curve centered 
    on the input value.
    
    Example::
    
      >>> encoder = PopulationEncoder(n_neurons=10, min_val=0, max_val=1)
      >>> spikes = encoder(0.5, n_time=100)  # Should peak at neuron 5
    
    Args:
      n_neurons: int. Number of neurons in the population.
      min_val: float. Minimum input value.
      max_val: float. Maximum input value.
      sigma: float. Width of receptive fields (standard deviation).
      max_rate: float. Maximum firing rate of neurons.
    """
    __module__ = 'braintools'

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
        
        Args:
          data: Input scalar or array to encode.
          n_time: Number of time steps.
          key: JAX random key.
        
        Returns:
          Population spike trains with shape (n_time, n_neurons, *data.shape).
        """
        data = u.math.asarray(data)
        # Calculate distances from each neuron's preferred value
        distances = u.math.abs(data[..., None] - self.centers)

        # Gaussian activation
        activations = u.math.exp(-0.5 * (distances / self.sigma) ** 2)
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
    
    Each input value is converted to a probability, and spikes are generated
    independently at each time step according to this probability.
    
    Example::
    
      >>> encoder = BernoulliEncoder()
      >>> data = jnp.array([0.1, 0.5, 0.9])
      >>> spikes = encoder(data, n_time=1000)
      >>> # Spike rates should be ~[100, 500, 900] Hz
    
    Args:
      scale: float. Scaling factor for input-to-probability conversion.
      normalize: bool. Whether to normalize inputs to [0, 1].
      min_val: float. Minimum value for normalization.
      max_val: float. Maximum value for normalization.
    """
    __module__ = 'braintools'

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
        
        Args:
          data: Input probabilities or values to normalize.
          n_time: Number of time steps.
          key: JAX random key.
        
        Returns:
          Bernoulli spike trains with shape (n_time, *data.shape).
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

    Example::

      >>> import jax.numpy as jnp
      >>> encoder = DeltaEncoder(threshold=0.25, normalize=False)
      >>> encoder(jnp.array([0.0, 0.1, 0.2, 0.3, 0.3, 0.0]))
      Array([ 0.,  0.,  0.,  1.,  0., -1.], dtype=float32)

    Args:
      threshold: float. Change relative to the reference that triggers a spike.
      positive_only: bool. Emit spikes only for increases (output 0/1). The
        reference still follows decreases.
      absolute: bool. If False (default), increases give ``+1`` and decreases
        ``-1`` (ON/OFF events). If True, every crossing gives ``1``.
      normalize: bool. Rescale each feature to ``[0, 1]`` over time first, so
        ``threshold`` is a fraction of the feature's range.
    """
    __module__ = 'braintools'

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

        Args:
          data: Time series with shape ``(n_time, *features)``.

        Returns:
          Spike trains with the same shape as ``data``: ``+1``/``-1`` events, or
          ``0``/``1`` with ``absolute=True`` or ``positive_only=True``. The first
          time step never spikes.
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
    
    Converts input values to constant current levels that can be injected
    into integrate-and-fire neurons. The current amplitude determines the
    firing rate of the neuron.
    
    Example::
    
      >>> encoder = StepCurrentEncoder(current_scale=10.0)
      >>> currents = encoder(jnp.array([0.1, 0.5, 1.0]), n_time=100)
      >>> # Returns current levels [1.0, 5.0, 10.0] nA
    
    Args:
      current_scale: float. Scaling factor to convert inputs to current (nA).
      offset: float. Baseline current offset.
      normalize: bool. Whether to normalize inputs.
      min_val: float. Minimum value for normalization.
      max_val: float. Maximum value for normalization.
    """
    __module__ = 'braintools'

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
        
        Args:
          data: Input values to convert to currents.
          n_time: Number of time steps.
        
        Returns:
          Step current signals with shape (n_time, *data.shape).
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
    
    Distributes a specific number of spikes (determined by input value)
    randomly or regularly over the encoding time window.
    
    Example::
    
      >>> encoder = SpikeCountEncoder(max_spikes=10)
      >>> data = jnp.array([0.2, 0.5, 1.0])
      >>> spikes = encoder(data, n_time=100)
      >>> # Should generate [2, 5, 10] total spikes respectively
    
    Args:
      max_spikes: int. Maximum number of spikes for input value of 1.0.
      distribution: str. How to distribute spikes ('uniform', 'random').
      normalize: bool. Whether to normalize inputs to [0, 1].
    """
    __module__ = 'braintools'

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
        
        Args:
          data: Input values determining spike counts.
          n_time: Number of time steps.
          key: JAX random key.
        
        Returns:
          Spike trains with exact spike counts.
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
    
    Encodes input sequences by mapping values to precise spike times,
    creating temporal patterns that preserve sequence information.
    
    Example::
    
      >>> encoder = TemporalEncoder(n_patterns=3)
      >>> sequence = jnp.array([0, 1, 2, 1, 0])
      >>> spikes = encoder(sequence)  # Creates temporal spike pattern
    
    Args:
      n_patterns: int. Number of distinct temporal patterns.
      pattern_length: int. Length of each temporal pattern in time steps.
      jitter: float. Temporal jitter to add to spike times (fraction of pattern_length).
    """
    __module__ = 'braintools'

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
        
        Args:
          data: Sequence of integer values to encode.
          key: JAX random key for jitter.
        
        Returns:
          Temporal spike patterns with shape (len(data) * pattern_length, n_patterns).
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
    
    Orders input features by their values and generates spikes in rank order.
    The most active features spike first, followed by less active ones.
    This preserves the relative ordering of input features.
    
    Example::
    
      >>> encoder = RankOrderEncoder()
      >>> data = jnp.array([0.1, 0.8, 0.3, 0.9, 0.2])
      >>> spikes = encoder(data, n_time=10)
      >>> # Feature 3 spikes first (0.9), then feature 1 (0.8), etc.
    
    Args:
      use_values: bool. Whether to use actual values for timing or just ranks.
      normalize: bool. Whether to normalize input values.
      invert: bool. Whether to invert the order (smallest first).
    """
    __module__ = 'braintools'

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
        
        Args:
          data: Input feature vector to encode.
          n_time: Number of time steps available for encoding.
        
        Returns:
          Rank-order spike trains with shape (n_time, len(data)).
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
