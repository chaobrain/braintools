# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


class LatencyEncoder:
    r"""
    Encode the rate input as the spike train using latency encoding.

    Uses input features to determine time-to-first spike. Expected inputs should
    be between 0 and 1. If not, the latency encoder will normalize ``x`` into
    the range ``[0, 1]`` according to:

    .. math::
        x_{\text{normalize}} = \frac{x-\text{min_val}}{\text{max_val} - \text{min_val}}

    to spikes whose firing time is :math:`0 \le t_f \le \text{num_period}-1`.
    A larger ``x`` will cause the earlier firing time.

    Parameters
    ----------
    min_val : float, optional
        The minimal value in the given data `x`, used for data normalization.
    max_val : float, optional
        The maximum value in the given data `x`, used for data normalization.
    method : {'linear', 'log'}, default='log'
        How to convert intensity to firing time.

        - If ``method='linear'``, the firing rate is calculated as
          :math:`t_f(x) = (\text{num_period} - 1)(1 - x)`.
        - If ``method='log'``, the firing rate is calculated as
          :math:`t_f(x) = (\text{num_period} - 1) - ln(\alpha * x + 1)`,
          where :math:`\alpha` satisfies :math:`t_f(1) = \text{num_period} - 1`.
    threshold : float, default=0.01
        Input features below the threshold will fire at the final time step
        unless ``clip=True`` in which case they will not fire at all.
    clip : bool, default=False
        Option to remove spikes from features that fall below the threshold.
    tau : float, default=1.0 * u.ms
        RC Time constant for LIF model used to calculate firing time.
    normalize : bool, default=False
        Option to normalize the latency code such that the final spike(s)
        occur within num_steps.
    first_spk_time : float, default=0.0 * u.ms
        Time of the first possible spike.
    epsilon : float, default=1e-7
        A tiny positive value to avoid rounding errors.

    Examples
    --------
    .. code-block:: python

        import jax
        import jax.numpy as jnp

        # Create encoder with linear method
        encoder = LatencyEncoder(method='linear', normalize=True)

        # Encode input data
        data = jnp.array([0.02, 0.5, 1.0])
        spikes = encoder(data, n_time=5)

        # Output shape: (n_time, n_features)
        print(spikes.shape)  # (5, 3)
    """
    __module__ = 'braintools'

    def __init__(
        self,
        min_val: float = None,
        max_val: float = None,
        method: str = 'log',
        threshold: float = 0.01,
        clip: bool = False,
        tau: float = 1. * u.ms,
        normalize: bool = False,
        first_spk_time: float = 0. * u.ms,
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
        self.first_spk_step = int(u.get_mantissa(first_spk_time) / u.get_mantissa(brainstate.environ.get_dt()))
        self.epsilon = epsilon

    def __call__(self, data, n_time: Optional[brainstate.typing.ArrayLike] = None):
        """Generate latency spikes according to the given input data.

        Parameters
        ----------
        data : array_like
            The rate-based input data to encode.
        n_time : float, optional
            The total time to generate data. If None, use ``tau`` instead.

        Returns
        -------
        array_like
            The output spiking trains with shape (n_time, *data.shape).

        Notes
        -----
        Input data is expected to be in the range [0, 1]. Values outside this
        range will be normalized if min_val and max_val are provided.
        """
        with jax.ensure_compile_time_eval():
            if n_time is None:
                n_time = self.tau
            tau = n_time if self.normalize else self.tau
            x = data
            if self.min_val is not None and self.max_val is not None:
                x = (x - self.min_val) / (self.max_val - self.min_val)

            # Calculate the spike time
            dt = brainstate.environ.get_dt()
            dt_val = u.get_mantissa(dt)
            tau_val = u.get_mantissa(tau) if hasattr(tau, 'mantissa') else tau
            first_spk_val = u.get_mantissa(self.first_spk_time)

            if self.method == 'linear':
                spike_time = (tau_val - first_spk_val - dt_val) * (1 - x) + first_spk_val

            elif self.method == 'log':
                x = u.math.maximum(x, self.threshold + self.epsilon)  # saturates all values below threshold.
                spike_time = (tau_val - first_spk_val - dt_val) * u.math.log(
                    x / (x - self.threshold)) + first_spk_val

            else:
                raise ValueError(f'Unsupported method: {self.method}. Only support "log" and "linear".')

            # Clip the spike time
            if self.clip:
                spike_time = u.math.where(data < self.threshold, jnp.inf, spike_time)
            spike_steps = u.math.round(spike_time / dt_val).astype(int)
            return brainstate.functional.one_hot(spike_steps, num_classes=int(tau_val / dt_val), axis=0, dtype=x.dtype)


class RateEncoder:
    r"""
    Encode analog values into spike rates using various rate encoding methods.

    The rate encoder converts continuous input values into spike trains where the
    firing rate is proportional to the input intensity. Higher input values result
    in higher firing rates.

    Parameters
    ----------
    gain : float, default=100.0
        Scaling factor to convert normalized input to firing rate (Hz).
    method : {'linear', 'exponential', 'sqrt'}, default='linear'
        Rate encoding method to use.
    min_rate : float, default=0.0
        Minimum firing rate in Hz.
    max_rate : float, optional
        Maximum firing rate in Hz. If None, uses gain value.
    normalize : bool, default=False
        Whether to normalize inputs to [0, 1] range.
    min_val : float, optional
        Minimum value for normalization.
    max_val : float, optional
        Maximum value for normalization.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp

        # Create rate encoder
        encoder = RateEncoder(gain=100, method='linear')

        # Encode input data
        data = jnp.array([0.1, 0.5, 0.9])
        spikes = encoder(data, n_time=100)

        # Calculate firing rates
        firing_rates = jnp.mean(spikes, axis=0)  # Should be ~[10, 50, 90] Hz
    """
    __module__ = 'braintools'

    def __init__(
        self,
        gain: float = 100.0,
        method: str = 'linear',
        min_rate: float = 0.0,
        max_rate: float = None,
        normalize: bool = False,
        min_val: float = None,
        max_val: float = None,
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

    def __call__(self, data, n_time: int):
        """Generate rate-encoded spikes.

        Parameters
        ----------
        data : array_like
            Input array to encode.
        n_time : int
            Number of time steps.

        Returns
        -------
        array_like
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

        # Convert rates to probabilities per time step
        dt = brainstate.environ.get_dt()
        dt_val = u.get_mantissa(dt) if hasattr(dt, 'mantissa') else dt
        probs = rates * dt_val / 1000.0  # Convert Hz to probability per ms
        probs = u.math.clip(probs, 0, 1)

        # Generate spikes using Bernoulli process
        spikes = brainstate.random.bernoulli(probs, size=(n_time,) + data.shape)
        return spikes.astype(data.dtype)


class PoissonEncoder:
    r"""
    Encode inputs as Poisson spike trains.

    Generates spike trains where inter-spike intervals follow a Poisson distribution.
    The input intensity determines the rate parameter of the Poisson process.

    Parameters
    ----------
    time_window : float, default=1000.0
        Time window in ms for rate calculation.
    normalize : bool, default=False
        Whether to treat inputs as rates (False) or normalize to rates (True).
    max_rate : float, default=100.0
        Maximum rate when normalizing.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp

        # Create Poisson encoder
        encoder = PoissonEncoder()

        # Encode input rates
        data = jnp.array([10.0, 50.0, 100.0])  # Hz
        spikes = encoder(data, n_time=1000)  # 1 second at 1ms resolution

        # Mean spike counts should be approximately [10, 50, 100]
    """
    __module__ = 'braintools'

    def __init__(
        self,
        time_window: float = 1000.0,  # ms
        normalize: bool = False,
        max_rate: float = 100.0,
    ):
        self.time_window = time_window
        self.normalize = normalize
        self.max_rate = max_rate

    def __call__(self, data, n_time: int):
        """Generate Poisson spike trains.

        Parameters
        ----------
        data : array_like
            Input rates in Hz, or values to normalize to rates.
        n_time : int
            Number of time steps.

        Returns
        -------
        array_like
            Poisson spike trains with shape (n_time, *data.shape).
        """
        rates = data
        if self.normalize:
            rates = data * self.max_rate

        dt = brainstate.environ.get_dt()  # ms
        dt_val = u.get_mantissa(dt) if hasattr(dt, 'mantissa') else dt
        # Rate parameter for each time step
        lam = rates * dt_val / 1000.0  # Convert Hz to expected spikes per time step

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

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the population.
    min_val : float, default=0.0
        Minimum input value.
    max_val : float, default=1.0
        Maximum input value.
    sigma : float, optional
        Width of receptive fields (standard deviation). If None, computed
        as (max_val - min_val) / (n_neurons - 1).
    max_rate : float, default=100.0
        Maximum firing rate of neurons.

    Examples
    --------
    .. code-block:: python

        # Create population encoder
        encoder = PopulationEncoder(n_neurons=10, min_val=0, max_val=1)

        # Encode scalar input
        spikes = encoder(0.5, n_time=100)  # Should peak at neuron 5

        # Output shape: (n_time, n_neurons) for scalar input
    """
    __module__ = 'braintools'

    def __init__(
        self,
        n_neurons: int,
        min_val: float = 0.0,
        max_val: float = 1.0,
        sigma: float = None,
        max_rate: float = 100.0,
    ):
        self.n_neurons = n_neurons
        self.min_val = min_val
        self.max_val = max_val
        self.sigma = sigma if sigma is not None else (max_val - min_val) / (n_neurons - 1)
        self.max_rate = max_rate

        # Create neuron preferred values (centers of receptive fields)
        self.centers = u.math.linspace(min_val, max_val, n_neurons)

    def __call__(self, data, n_time: int):
        """Generate population-encoded spikes.

        Parameters
        ----------
        data : array_like
            Input scalar or array to encode.
        n_time : int
            Number of time steps.

        Returns
        -------
        array_like
            Population spike trains with shape (n_time, n_neurons, *data.shape).
        """
        data = u.math.asarray(data)
        # Calculate distances from each neuron's preferred value
        distances = u.math.abs(data[..., None] - self.centers)

        # Gaussian activation
        activations = u.math.exp(-0.5 * (distances / self.sigma) ** 2)
        rates = activations * self.max_rate

        # Convert to spike probabilities
        dt = brainstate.environ.get_dt()
        dt_val = u.get_mantissa(dt) if hasattr(dt, 'mantissa') else dt
        probs = rates * dt_val / 1000.0
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

    Parameters
    ----------
    scale : float, default=1.0
        Scaling factor for input-to-probability conversion.
    normalize : bool, default=True
        Whether to normalize inputs to [0, 1].
    min_val : float, optional
        Minimum value for normalization.
    max_val : float, optional
        Maximum value for normalization.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp

        # Create Bernoulli encoder
        encoder = BernoulliEncoder()

        # Encode input probabilities
        data = jnp.array([0.1, 0.5, 0.9])
        spikes = encoder(data, n_time=1000)

        # Spike rates should be approximately [100, 500, 900] Hz
    """
    __module__ = 'braintools'

    def __init__(
        self,
        scale: float = 1.0,
        normalize: bool = True,
        min_val: float = None,
        max_val: float = None,
    ):
        self.scale = scale
        self.normalize = normalize
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data, n_time: int):
        """Generate Bernoulli-encoded spikes.

        Parameters
        ----------
        data : array_like
            Input probabilities or values to normalize.
        n_time : int
            Number of time steps.

        Returns
        -------
        array_like
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
    Encode temporal differences (delta changes) in input signals.

    Generates spikes when the input signal changes significantly between
    time steps. Useful for encoding dynamic signals and temporal patterns.

    Parameters
    ----------
    threshold : float, default=0.1
        Minimum change required to generate a spike.
    positive_only : bool, default=False
        Whether to encode only positive changes.
    absolute : bool, default=False
        Whether to use absolute value of changes.
    normalize : bool, default=True
        Whether to normalize the input signal.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp

        # Encode a changing signal
        time_series = jnp.array([0, 0.1, 0.8, 0.7, 0.2])
        encoder = DeltaEncoder(threshold=0.1)
        spikes = encoder(time_series)  # Spikes at significant changes
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

    def __call__(self, data):
        """Generate delta-encoded spikes.

        Parameters
        ----------
        data : array_like
            Time series data with shape (n_time, *features).

        Returns
        -------
        array_like
            Delta spike trains with same shape as input.
        """
        if data.ndim == 1:
            data = data[..., None]  # Add feature dimension

        x = data
        if self.normalize:
            x_min, x_max = u.math.min(x, axis=0, keepdims=True), u.math.max(x, axis=0, keepdims=True)
            x = (x - x_min) / (x_max - x_min + 1e-8)

        # Calculate differences
        diffs = u.math.diff(x, axis=0, prepend=x[0:1])

        if self.absolute:
            diffs = u.math.abs(diffs)
        elif self.positive_only:
            diffs = u.math.maximum(diffs, 0)

        # Generate spikes where changes exceed threshold
        spikes = (u.math.abs(diffs) >= self.threshold).astype(data.dtype)

        return spikes.squeeze() if data.shape[-1] == 1 else spikes


class StepCurrentEncoder:
    r"""
    Encode inputs as step current injections for LIF neurons.

    Converts input values to constant current levels that can be injected
    into integrate-and-fire neurons. The current amplitude determines the
    firing rate of the neuron.

    Parameters
    ----------
    current_scale : float, default=10.0
        Scaling factor to convert inputs to current (nA).
    offset : float, default=0.0
        Baseline current offset.
    normalize : bool, default=True
        Whether to normalize inputs.
    min_val : float, optional
        Minimum value for normalization.
    max_val : float, optional
        Maximum value for normalization.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp

        # Create step current encoder
        encoder = StepCurrentEncoder(current_scale=10.0)

        # Encode input values to currents
        currents = encoder(jnp.array([0.1, 0.5, 1.0]), n_time=100)
        # Returns current levels [1.0, 5.0, 10.0] nA
    """
    __module__ = 'braintools'

    def __init__(
        self,
        current_scale: float = 10.0,  # nA
        offset: float = 0.0,
        normalize: bool = True,
        min_val: float = None,
        max_val: float = None,
    ):
        self.current_scale = current_scale
        self.offset = offset
        self.normalize = normalize
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data, n_time: int):
        """Generate step current signals.

        Parameters
        ----------
        data : array_like
            Input values to convert to currents.
        n_time : int
            Number of time steps.

        Returns
        -------
        array_like
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

    Parameters
    ----------
    max_spikes : int, default=10
        Maximum number of spikes for input value of 1.0.
    distribution : {'uniform', 'random'}, default='random'
        How to distribute spikes in time.
    normalize : bool, default=True
        Whether to normalize inputs to [0, 1].

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp

        # Create spike count encoder
        encoder = SpikeCountEncoder(max_spikes=10)

        # Encode input values
        data = jnp.array([0.2, 0.5, 1.0])
        spikes = encoder(data, n_time=100)
        # Should generate [2, 5, 10] total spikes respectively
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

    def __call__(self, data, n_time: int):
        """Generate spike count-encoded trains.

        Parameters
        ----------
        data : array_like
            Input values determining spike counts.
        n_time : int
            Number of time steps.

        Returns
        -------
        array_like
            Spike trains with exact spike counts.
        """
        x = data
        if self.normalize:
            x = u.math.clip(x, 0, 1)

        spike_counts = (x * self.max_spikes).astype(int)
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

    Parameters
    ----------
    n_patterns : int
        Number of distinct temporal patterns.
    pattern_length : int, default=10
        Length of each temporal pattern in time steps.
    jitter : float, default=0.1
        Temporal jitter to add to spike times (fraction of pattern_length).

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp

        # Create temporal encoder
        encoder = TemporalEncoder(n_patterns=3)

        # Encode sequence
        sequence = jnp.array([0, 1, 2, 1, 0])
        spikes = encoder(sequence)  # Creates temporal spike pattern
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

    def _create_patterns(self):
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

    def __call__(self, data):
        """Generate temporally-encoded spikes.

        Parameters
        ----------
        data : array_like
            Sequence of integer values to encode.

        Returns
        -------
        array_like
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

    Parameters
    ----------
    use_values : bool, default=True
        Whether to use actual values for timing or just ranks.
    normalize : bool, default=True
        Whether to normalize input values.
    invert : bool, default=False
        Whether to invert the order (smallest first).

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp

        # Create rank order encoder
        encoder = RankOrderEncoder()

        # Encode input features
        data = jnp.array([0.1, 0.8, 0.3, 0.9, 0.2])
        spikes = encoder(data, n_time=10)
        # Feature 3 spikes first (0.9), then feature 1 (0.8), etc.
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

    def __call__(self, data, n_time: int):
        """Generate rank-order encoded spikes.

        Parameters
        ----------
        data : array_like
            Input feature vector to encode.
        n_time : int
            Number of time steps available for encoding.

        Returns
        -------
        array_like
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
