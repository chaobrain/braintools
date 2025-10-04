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

# -*- coding: utf-8 -*-

"""
Composable pulse input generators.

This module provides composable versions of pulse-shaped input generators
including spikes, Gaussian pulses, exponential decays, and bursts.
"""

from typing import Sequence, Optional, Union

import brainstate
import brainunit as u

from ._composable_base import Input
from ._deprecation import create_deprecated_class
from ._functional_pulses import (
    spike,
    gaussian_pulse,
    exponential_decay,
    double_exponential,
    burst
)

ArrayLike = brainstate.typing.ArrayLike

__all__ = [
    'Spike',
    'GaussianPulse',
    'ExponentialDecay',
    'DoubleExponential',
    'Burst',
]


class Spike(Input):
    """Generate spike input at specified times.
    
    Creates a series of rectangular current pulses (spikes) at specified
    times with customizable durations and amplitudes. This is useful for
    simulating synaptic inputs, direct current injections, or any protocol
    requiring precise timing of current pulses.
    
    Parameters
    ----------
    sp_times : Sequence[ArrayLike]
        The spike time points. Can include units (e.g., ms).
    duration : ArrayLike
        The total duration of the signal.
    sp_lens : Union[float, Sequence[float]], optional
        The duration of each spike. If scalar, same duration for all spikes.
        If sequence, must match length of sp_times. Default is 1.
    sp_sizes : Union[float, Sequence[float]], optional
        The amplitude of each spike. If scalar, same amplitude for all.
        If sequence, must match length of sp_times. Default is 1.
    
    Attributes
    ----------
    sp_times : Sequence
        The stored spike time points.
    sp_lens : Sequence
        The spike durations (expanded to list if scalar was provided).
    sp_sizes : Sequence
        The spike amplitudes (expanded to list if scalar was provided).
    duration : Quantity or float
        The total duration.
    
    See Also
    --------
    Burst : For generating bursts of activity.
    GaussianPulse : For smooth pulse shapes.
    
    Notes
    -----
    The spikes are rectangular pulses. For more realistic synaptic currents,
    consider using DoubleExponential or combining with smoothing operations.
    This class uses the functional spike API internally.
    
    Examples
    --------
    Simple spike train with uniform properties:
    
    .. code-block:: python

        >>> spikes = Spike(
        ...     sp_times=[10, 20, 30, 200, 300] * u.ms,
        ...     duration=400 * u.ms,
        ...     sp_lens=1 * u.ms,  # All spikes 1ms long
        ...     sp_sizes=0.5 * u.nA  # All spikes 0.5nA amplitude
        ... )
        >>> array = spikes()
    
    Variable spike properties:
    
    .. code-block:: python

        >>> spikes = Spike(
        ...     sp_times=[10, 50, 100] * u.ms,
        ...     duration=150 * u.ms,
        ...     sp_lens=[1, 2, 0.5] * u.ms,  # Different durations
        ...     sp_sizes=[0.5, 1.0, 0.3] * u.nA  # Different amplitudes
        ... )
    
    Add to background activity:
    
    .. code-block:: python

        >>> from braintools.input import Constant
        >>> spikes = Spike([10, 50, 100, 150] * u.ms, 200 * u.ms, sp_sizes=1.0)
        >>> background = Constant([(0.1, 200 * u.ms)])
        >>> combined = spikes + background
    
    High-frequency burst simulation:
    
    .. code-block:: python

        >>> import numpy as np
        >>> times = np.arange(0, 50, 2) * u.ms  # Every 2ms
        >>> spikes = Spike(
        ...     sp_times=times,
        ...     duration=100 * u.ms,
        ...     sp_lens=0.5 * u.ms,
        ...     sp_sizes=2.0 * u.pA
        ... )
    
    Combine with noise for realistic inputs:
    
    .. code-block:: python

        >>> from braintools.input import WienerProcess
        >>> spikes = Spike([20, 40, 60] * u.ms, 100 * u.ms)
        >>> noise = WienerProcess(100 * u.ms, sigma=0.05)
        >>> noisy_spikes = spikes + noise
    
    Pattern of increasing amplitudes:
    
    .. code-block:: python

        >>> amplitudes = np.linspace(0.5, 2.0, 10)
        >>> times = np.linspace(10, 190, 10) * u.ms
        >>> increasing_spikes = Spike(
        ...     sp_times=times,
        ...     duration=200 * u.ms,
        ...     sp_sizes=amplitudes * u.nA
        ... )
    
    Paired-pulse facilitation protocol:
    
    .. code-block:: python

        >>> # Two spikes with short interval
        >>> paired = Spike(
        ...     sp_times=[50, 70] * u.ms,  # 20ms interval
        ...     duration=150 * u.ms,
        ...     sp_lens=2 * u.ms,
        ...     sp_sizes=[1.0, 1.5] * u.nA  # Second spike larger
        ... )
        >>> # Repeat the pattern
        >>> repeated_pairs = paired.repeat(5)
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 sp_times: Sequence[ArrayLike],
                 duration: ArrayLike,
                 sp_lens: Union[float, Sequence[float]] = 1. * u.ms,
                 sp_sizes: Union[float, Sequence[float]] = 1.):
        """Initialize spike input.
        
        Parameters
        ----------
        sp_times : Sequence
            The spike time points.
        duration : ArrayLike
            The total duration.
        sp_lens : Union[float, Sequence[float]], optional
            The spike duration(s). Default is 1 * u.ms.
        sp_sizes : Union[float, Sequence[float]], optional
            The spike amplitude(s). Default is 1.
        """
        super().__init__(duration)

        self.sp_times = sp_times
        # Expand scalars to lists if needed
        if not isinstance(sp_lens, (tuple, list)) and u.math.size(sp_lens) == 1:
            sp_lens = [sp_lens] * len(sp_times)
        if not isinstance(sp_sizes, (tuple, list)) and u.math.size(sp_sizes) == 1:
            sp_sizes = [sp_sizes] * len(sp_times)
        self.sp_lens = sp_lens
        self.sp_sizes = sp_sizes

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the spike input array using functional API."""
        # Use the functional API
        return spike(
            sp_times=self.sp_times,
            sp_lens=self.sp_lens,
            sp_sizes=self.sp_sizes,
            duration=self.duration
        )


class GaussianPulse(Input):
    """Generate Gaussian-shaped pulse input.
    
    Creates a smooth, bell-shaped current pulse centered at a specific time.
    This is useful for modeling smooth synaptic inputs, sensory stimuli,
    or any input requiring a gradual rise and fall.
    
    Parameters
    ----------
    amplitude : float
        Peak amplitude of the pulse. Can include units.
    center : ArrayLike
        Center time of the pulse peak.
    sigma : ArrayLike
        Standard deviation (width) of the pulse. Larger values create wider pulses.
    duration : ArrayLike
        Total duration of the input signal.
    
    Attributes
    ----------
    amplitude : float
        The peak amplitude.
    center : Quantity or float
        The center time.
    sigma : Quantity or float
        The standard deviation.
    duration : Quantity or float
        The total duration.
    
    See Also
    --------
    DoubleExponential : For asymmetric pulse shapes.
    ExponentialDecay : For one-sided decay.
    
    Notes
    -----
    The pulse follows the Gaussian formula:
    amplitude * exp(-0.5 * ((t - center) / sigma)^2)
    
    Approximately 99.7% of the pulse is contained within center ± 3*sigma.
    This class uses the functional gaussian_pulse API internally.
    
    Examples
    --------
    Single Gaussian pulse:
    
    .. code-block:: python

        >>> pulse = GaussianPulse(
        ...     amplitude=1.0 * u.nA,
        ...     center=100 * u.ms,
        ...     sigma=20 * u.ms,
        ...     duration=200 * u.ms
        ... )
        >>> array = pulse()
    
    Multiple overlapping pulses:
    
    .. code-block:: python

        >>> pulse1 = GaussianPulse(1.0, 100 * u.ms, 20 * u.ms, 500 * u.ms)
        >>> pulse2 = GaussianPulse(0.8, 300 * u.ms, 30 * u.ms, 500 * u.ms)
        >>> double_pulse = pulse1 + pulse2
    
    Train of Gaussian pulses:
    
    .. code-block:: python

        >>> # Create evenly spaced pulses
        >>> centers = [50, 150, 250, 350] * u.ms
        >>> pulses = []
        >>> for center in centers:
        ...     pulses.append(GaussianPulse(0.5, center, 10 * u.ms, 400 * u.ms))
        >>> pulse_train = sum(pulses[1:], pulses[0])  # Sum all pulses
    
    Amplitude modulation with Gaussian envelope:
    
    .. code-block:: python

        >>> from braintools.input import Sinusoidal
        >>> envelope = GaussianPulse(1.0, 250 * u.ms, 50 * u.ms, 500 * u.ms)
        >>> carrier = Sinusoidal(1.0, 50 * u.Hz, 500 * u.ms)
        >>> modulated = envelope * carrier
    
    Noisy Gaussian pulse:
    
    .. code-block:: python

        >>> from braintools.input import WienerProcess
        >>> pulse = GaussianPulse(2.0, 100 * u.ms, 15 * u.ms, 200 * u.ms)
        >>> noise = WienerProcess(200 * u.ms, sigma=0.1)
        >>> noisy_pulse = pulse + noise
    
    Wide and narrow pulses comparison:
    
    .. code-block:: python

        >>> narrow = GaussianPulse(1.0, 100 * u.ms, 5 * u.ms, 200 * u.ms)
        >>> wide = GaussianPulse(1.0, 100 * u.ms, 30 * u.ms, 200 * u.ms)
        >>> # Combine with different weights
        >>> mixed = 0.7 * narrow + 0.3 * wide
    
    Inverted (inhibitory) pulse:
    
    .. code-block:: python

        >>> inhibitory = GaussianPulse(-0.5, 150 * u.ms, 25 * u.ms, 300 * u.ms)
        >>> # Or use negation operator
        >>> excitatory = GaussianPulse(0.5, 150 * u.ms, 25 * u.ms, 300 * u.ms)
        >>> inhibitory = -excitatory
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 amplitude: float,
                 center: brainstate.typing.ArrayLike,
                 sigma: brainstate.typing.ArrayLike,
                 duration: brainstate.typing.ArrayLike):
        """Initialize Gaussian pulse.
        
        Parameters
        ----------
        amplitude : float
            Peak amplitude of the pulse.
        center : ArrayLike
            Center time of the pulse.
        sigma : ArrayLike
            Standard deviation (width).
        duration : ArrayLike
            Total duration of the input.
        """
        super().__init__(duration)

        self.amplitude = amplitude
        self.center = center
        self.sigma = sigma

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the Gaussian pulse array using functional API."""
        # Use the functional API - note it has an 'n' parameter for channels
        return gaussian_pulse(
            amplitude=self.amplitude,
            center=self.center,
            sigma=self.sigma,
            duration=self.duration,
            n=1  # Single channel for composable version
        )


class ExponentialDecay(Input):
    """Generate exponentially decaying input.
    
    Creates an input that decays exponentially from an initial amplitude.
    This is useful for modeling synaptic currents, adaptation processes,
    or any phenomenon with exponential relaxation.
    
    Parameters
    ----------
    amplitude : float
        Initial amplitude at the start of decay. Can include units.
    tau : ArrayLike
        Decay time constant. The amplitude decreases to ~37% (1/e) after tau.
    duration : ArrayLike
        Total duration of the input signal.
    t_start : ArrayLike, optional
        Start time of the decay. Default is 0.
    t_end : ArrayLike, optional
        End time of the decay. Default is duration.
    
    Attributes
    ----------
    amplitude : float
        The initial amplitude.
    tau : Quantity or float
        The decay time constant.
    duration : Quantity or float
        The total duration.
    t_start : Quantity or float or None
        The decay start time.
    t_end : Quantity or float or None
        The decay end time.
    
    See Also
    --------
    DoubleExponential : For rise and decay dynamics.
    GaussianPulse : For symmetric pulse shapes.
    
    Notes
    -----
    The decay follows: amplitude * exp(-t/tau) for t >= 0.
    After time tau, the amplitude is reduced to amplitude/e ≈ 0.368*amplitude.
    After 3*tau, it's reduced to ~5% of initial value.
    This class uses the functional exponential_decay API internally.
    
    Examples
    --------
    Simple exponential decay:
    
    .. code-block:: python

        >>> decay = ExponentialDecay(
        ...     amplitude=2.0 * u.nA,
        ...     tau=30 * u.ms,
        ...     duration=200 * u.ms
        ... )
        >>> array = decay()
    
    Delayed decay (starts at t=50ms):
    
    .. code-block:: python

        >>> decay = ExponentialDecay(
        ...     amplitude=1.5,
        ...     tau=20 * u.ms,
        ...     duration=150 * u.ms,
        ...     t_start=50 * u.ms
        ... )
    
    Gated decay with step function:
    
    .. code-block:: python

        >>> from braintools.input import Step
        >>> decay = ExponentialDecay(2.0, 30 * u.ms, 500 * u.ms, t_start=100 * u.ms)
        >>> step = Step([0, 1], [0, 100] * u.ms, 500 * u.ms)
        >>> gated_decay = decay * step  # Gate the decay
    
    Multiple decay phases:
    
    .. code-block:: python

        >>> fast_decay = ExponentialDecay(1.0, 10 * u.ms, 200 * u.ms)
        >>> slow_decay = ExponentialDecay(0.5, 50 * u.ms, 200 * u.ms)
        >>> combined = fast_decay + slow_decay  # Bi-exponential
    
    Adaptation current simulation:
    
    .. code-block:: python

        >>> # Triggered by step input
        >>> from braintools.input import Step
        >>> trigger = Step([0, 1, 0], [0, 50, 150] * u.ms, 300 * u.ms)
        >>> adaptation = ExponentialDecay(0.3, 40 * u.ms, 300 * u.ms, t_start=50 * u.ms)
        >>> net_current = trigger - adaptation  # Adaptation reduces input
    
    Repeated decay pattern:
    
    .. code-block:: python

        >>> single_decay = ExponentialDecay(1.0, 15 * u.ms, 50 * u.ms)
        >>> decay_train = single_decay.repeat(10)  # 500ms total
    
    Synaptic depression model:
    
    .. code-block:: python

        >>> # Each spike triggers smaller response
        >>> amplitudes = [1.0, 0.8, 0.6, 0.4]  # Decreasing amplitudes
        >>> times = [0, 50, 100, 150]  # Spike times
        >>> decays = []
        >>> for amp, t in zip(amplitudes, times):
        ...     decays.append(ExponentialDecay(amp, 20 * u.ms, 200 * u.ms, t_start=t * u.ms))
        >>> depressing_response = sum(decays[1:], decays[0])
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 amplitude: float,
                 tau: ArrayLike,
                 duration: ArrayLike,
                 t_start: Optional[ArrayLike] = None,
                 t_end: Optional[ArrayLike] = None):
        """Initialize exponential decay.
        
        Parameters
        ----------
        amplitude : float
            Initial amplitude of the decay.
        tau : ArrayLike
            Decay time constant.
        duration : ArrayLike
            Total duration of the input.
        t_start : ArrayLike, optional
            Start time of the decay. Default is 0.
        t_end : ArrayLike, optional
            End time of the decay. Default is duration.
        """
        super().__init__(duration)

        self.amplitude = amplitude
        self.tau = tau
        self.t_start = t_start
        self.t_end = t_end

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the exponential decay array using functional API."""
        # Use the functional API
        return exponential_decay(
            amplitude=self.amplitude,
            tau=self.tau,
            duration=self.duration,
            t_start=self.t_start,
            t_end=self.t_end
        )


class DoubleExponential(Input):
    """Generate double exponential (alpha function) input.
    
    Creates an input with distinct rise and decay phases, commonly used to
    model synaptic currents. The shape is characterized by a rapid rise
    followed by a slower decay, creating an asymmetric pulse.
    
    Parameters
    ----------
    amplitude : float
        Peak amplitude of the pulse. Can include units.
    tau_rise : ArrayLike
        Rise time constant. Smaller values give faster rise.
    tau_decay : ArrayLike
        Decay time constant. Should be larger than tau_rise.
    duration : ArrayLike
        Total duration of the input signal.
    t_start : ArrayLike, optional
        Start time of the pulse. Default is 0.
    t_end : ArrayLike, optional
        End time of the pulse. Default is duration.
    
    Attributes
    ----------
    amplitude : float
        The peak amplitude.
    tau_rise : Quantity or float
        The rise time constant.
    tau_decay : Quantity or float
        The decay time constant.
    duration : Quantity or float
        The total duration.
    t_start : Quantity or float or None
        The pulse start time.
    t_end : Quantity or float or None
        The pulse end time.
    
    See Also
    --------
    ExponentialDecay : For single exponential dynamics.
    GaussianPulse : For symmetric pulse shapes.
    
    Notes
    -----
    The alpha function follows:
    amplitude * (exp(-t/tau_decay) - exp(-t/tau_rise)) for t >= 0
    
    The function is normalized so the peak equals the specified amplitude.
    Time to peak: t_peak = (tau_rise * tau_decay)/(tau_decay - tau_rise) * ln(tau_decay/tau_rise)
    This class uses the functional double_exponential API internally.
    
    Examples
    --------
    AMPA-like synaptic current:
    
    .. code-block:: python

        >>> ampa = DoubleExponential(
        ...     amplitude=1.0 * u.nA,
        ...     tau_rise=0.5 * u.ms,
        ...     tau_decay=5 * u.ms,
        ...     duration=50 * u.ms
        ... )
        >>> array = ampa()
    
    NMDA-like synaptic current (slower dynamics):
    
    .. code-block:: python

        >>> nmda = DoubleExponential(
        ...     amplitude=0.5 * u.nA,
        ...     tau_rise=2 * u.ms,
        ...     tau_decay=50 * u.ms,
        ...     duration=200 * u.ms
        ... )
    
    Synaptic current with noise:
    
    .. code-block:: python

        >>> from braintools.input import WienerProcess
        >>> alpha = DoubleExponential(1.0, 5 * u.ms, 20 * u.ms, 200 * u.ms)
        >>> noise = WienerProcess(200 * u.ms, sigma=0.05)
        >>> synaptic = alpha + noise
    
    Train of synaptic inputs:
    
    .. code-block:: python

        >>> # Multiple EPSPs at different times
        >>> times = [10, 30, 55, 80] * u.ms
        >>> epsps = []
        >>> for t in times:
        ...     epsps.append(DoubleExponential(
        ...         1.0, 1 * u.ms, 10 * u.ms, 150 * u.ms, t_start=t
        ...     ))
        >>> epsp_train = sum(epsps[1:], epsps[0])
    
    Paired-pulse facilitation:
    
    .. code-block:: python

        >>> # Second pulse larger than first
        >>> pulse1 = DoubleExponential(1.0, 2 * u.ms, 15 * u.ms, 100 * u.ms, t_start=20 * u.ms)
        >>> pulse2 = DoubleExponential(1.5, 2 * u.ms, 15 * u.ms, 100 * u.ms, t_start=40 * u.ms)
        >>> ppf = pulse1 + pulse2
    
    Combined fast and slow components:
    
    .. code-block:: python

        >>> fast = DoubleExponential(0.7, 0.5 * u.ms, 5 * u.ms, 100 * u.ms)
        >>> slow = DoubleExponential(0.3, 5 * u.ms, 50 * u.ms, 100 * u.ms)
        >>> mixed = fast + slow
    
    Inhibitory synaptic current:
    
    .. code-block:: python

        >>> ipsc = DoubleExponential(
        ...     amplitude=-0.8 * u.nA,  # Negative for inhibition
        ...     tau_rise=1 * u.ms,
        ...     tau_decay=20 * u.ms,
        ...     duration=100 * u.ms
        ... )
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 amplitude: float,
                 tau_rise: ArrayLike,
                 tau_decay: ArrayLike,
                 duration: ArrayLike,
                 t_start: Optional[ArrayLike] = None,
                 t_end: Optional[ArrayLike] = None):
        """Initialize double exponential.
        
        Parameters
        ----------
        amplitude : float
            Peak amplitude.
        tau_rise : ArrayLike
            Rise time constant.
        tau_decay : ArrayLike
            Decay time constant.
        duration : ArrayLike
            Total duration of the input.
        t_start : ArrayLike, optional
            Start time of the pulse. Default is 0.
        t_end : ArrayLike, optional
            End time of the pulse. Default is duration.
        """
        super().__init__(duration)

        self.amplitude = amplitude
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.t_start = t_start
        self.t_end = t_end

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the double exponential array using functional API."""
        # Use the functional API
        return double_exponential(
            amplitude=self.amplitude,
            tau_rise=self.tau_rise,
            tau_decay=self.tau_decay,
            duration=self.duration,
            t_start=self.t_start,
            t_end=self.t_end
        )


class Burst(Input):
    """Generate burst pattern input.
    
    Creates a pattern of rectangular bursts separated by quiet periods.
    This is useful for simulating rhythmic activity, theta-burst stimulation,
    or any protocol requiring repeated stimulation periods.
    
    Parameters
    ----------
    n_bursts : int
        Number of bursts to generate.
    burst_amp : float
        Amplitude during bursts. Can include units.
    burst_freq : ArrayLike
        Frequency of oscillation within each burst.
    burst_duration : ArrayLike
        Duration of each individual burst.
    inter_burst_interval : ArrayLike
        Time between the start of consecutive bursts.
    duration : ArrayLike
        Total duration of the input signal.
    
    Attributes
    ----------
    n_bursts : int
        The number of bursts.
    burst_amp : float
        The burst amplitude.
    burst_freq : Quantity or float
        The oscillation frequency within bursts.
    burst_duration : Quantity or float
        The duration of each burst.
    inter_burst_interval : Quantity or float
        The interval between burst starts.
    duration : Quantity or float
        The total duration.
    
    See Also
    --------
    Spike : For individual spikes at specific times.
    DoubleExponential : For more realistic burst shapes.
    
    Notes
    -----
    The bursts are oscillatory (sinusoidal) at the specified frequency.
    The functional burst API generates sin(2*pi*freq*t) oscillations.
    For DC (rectangular) bursts, this class is not suitable - use repeated
    Step or Spike instead.
    
    Examples
    --------
    Oscillatory bursts at 50Hz:
    
    .. code-block:: python

        >>> bursts = Burst(
        ...     n_bursts=5,
        ...     burst_amp=1.0 * u.nA,
        ...     burst_freq=50 * u.Hz,  # 50Hz oscillation
        ...     burst_duration=30 * u.ms,
        ...     inter_burst_interval=100 * u.ms,
        ...     duration=500 * u.ms
        ... )
        >>> array = bursts()
    Oscillatory bursts (theta-burst stimulation):
    
    .. code-block:: python

        >>> theta_bursts = Burst(
        ...     n_bursts=10,
        ...     burst_amp=2.0,
        ...     burst_freq=100 * u.Hz,  # 100Hz oscillation within bursts
        ...     burst_duration=40 * u.ms,
        ...     inter_burst_interval=200 * u.ms,  # 5Hz burst rate
        ...     duration=2000 * u.ms
        ... )
    Bursts with ramped amplitude:
    
    .. code-block:: python

        >>> from braintools.input import Ramp
        >>> burst = Burst(5, 1.0, 50 * u.Hz, 30 * u.ms, 100 * u.ms, 500 * u.ms)
        >>> ramp = Ramp(0.5, 1.5, 500 * u.ms)
        >>> modulated_burst = burst * ramp
    Burst pattern with noise:
    
    .. code-block:: python

        >>> from braintools.input import WienerProcess
        >>> bursts = Burst(4, 1.5, 30 * u.Hz, 50 * u.ms, 150 * u.ms, 600 * u.ms)
        >>> noise = WienerProcess(600 * u.ms, sigma=0.1)
        >>> noisy_bursts = bursts + noise
    Gamma bursts in theta rhythm:
    
    .. code-block:: python

        >>> # 40Hz gamma bursts at 5Hz theta rhythm
        >>> gamma_in_theta = Burst(
        ...     n_bursts=15,
        ...     burst_amp=1.0 * u.nA,
        ...     burst_freq=40 * u.Hz,  # Gamma frequency
        ...     burst_duration=100 * u.ms,
        ...     inter_burst_interval=200 * u.ms,  # Theta rhythm (5Hz)
        ...     duration=3000 * u.ms
        ... )
    Paired bursts protocol:
    
    .. code-block:: python

        >>> # Two bursts with short interval, then long gap
        >>> burst1 = Burst(1, 1.0, 100 * u.Hz, 20 * u.ms, 100 * u.ms, 100 * u.ms)
        >>> burst2 = Burst(1, 1.2, 100 * u.Hz, 20 * u.ms, 100 * u.ms, 100 * u.ms).shift(30 * u.ms)
        >>> paired = burst1 + burst2
        >>> protocol = paired.repeat(5)  # Repeat paired bursts
    Burst with exponential decay envelope:
    
    .. code-block:: python

        >>> from braintools.input import ExponentialDecay
        >>> bursts = Burst(8, 1.0, 50 * u.Hz, 25 * u.ms, 75 * u.ms, 600 * u.ms)
        >>> envelope = ExponentialDecay(1.0, 150 * u.ms, 600 * u.ms)
        >>> decaying_bursts = bursts * envelope
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 n_bursts: int,
                 burst_amp: float,
                 burst_freq: ArrayLike,
                 burst_duration: ArrayLike,
                 inter_burst_interval: ArrayLike,
                 duration: ArrayLike):
        """Initialize burst input.
        
        Parameters
        ----------
        n_bursts : int
            Number of bursts.
        burst_amp : float
            Amplitude during bursts.
        burst_freq : ArrayLike
            Frequency within bursts (0 for DC).
        burst_duration : ArrayLike
            Duration of each burst.
        inter_burst_interval : ArrayLike
            Interval between burst starts.
        duration : ArrayLike
            Total duration of the input.
        """
        super().__init__(duration)

        self.n_bursts = n_bursts
        self.burst_amp = burst_amp
        self.burst_freq = burst_freq
        self.burst_duration = burst_duration
        self.inter_burst_interval = inter_burst_interval

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the burst input array using functional API."""
        # Use the functional API
        return burst(
            burst_amp=self.burst_amp,
            burst_freq=self.burst_freq,
            burst_duration=self.burst_duration,
            inter_burst_interval=self.inter_burst_interval,
            n_bursts=self.n_bursts,
            duration=self.duration
        )


SpikeInput = create_deprecated_class(Spike, 'SpikeInput', 'Spike')
BurstInput = create_deprecated_class(Burst, 'BurstInput', 'Burst')

__all__.extend(['SpikeInput', 'BurstInput'])
