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

# -*- coding: utf-8 -*-

"""
Composable waveform input generators.
"""

from typing import Optional, Union
import brainstate
import brainunit as u
import numpy as np
from ._composable_base import Input

__all__ = [
    'SinusoidalInput',
    'SquareInput',
    'TriangularInput',
    'SawtoothInput',
    'ChirpInput',
    'NoisySinusoidalInput',
]


def _square(t, duty=0.5):
    """Helper function to generate square wave."""
    t, w = np.asarray(t), np.asarray(duty)
    w = np.asarray(w + (t - t))
    t = np.asarray(t + (w - w))
    if t.dtype.char in 'fFdD':
        ytype = t.dtype.char
    else:
        ytype = 'd'

    y = np.zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    np.place(y, mask1, np.nan)

    # on the interval 0 to duty*2*pi function is 1
    tmod = np.mod(t, 2 * np.pi)
    mask2 = (1 - mask1) & (tmod < w * 2 * np.pi)
    np.place(y, mask2, 1)

    # on the interval duty*2*pi to 2*pi function is -1
    mask3 = (1 - mask1) & (1 - mask2)
    np.place(y, mask3, -1)
    return y


class SinusoidalInput(Input):
    """Sinusoidal input generator.
    
    Examples
    --------
    >>> # Create a sinusoidal input and modulate it
    >>> sine = SinusoidalInput(1.0, 10 * u.Hz, 1000 * u.ms)
    >>> envelope = RampInput(0, 1, 1000 * u.ms)
    >>> modulated = sine * envelope  # Amplitude modulation
    >>> 
    >>> # Combine multiple frequencies
    >>> sine1 = SinusoidalInput(1.0, 5 * u.Hz, 500 * u.ms)
    >>> sine2 = SinusoidalInput(0.5, 15 * u.Hz, 500 * u.ms)
    >>> complex_wave = sine1 + sine2
    """
    
    def __init__(self,
                 amplitude: float,
                 frequency: u.Quantity,
                 duration: Union[float, u.Quantity],
                 dt: Optional[Union[float, u.Quantity]] = None,
                 t_start: Optional[Union[float, u.Quantity]] = None,
                 t_end: Optional[Union[float, u.Quantity]] = None,
                 bias: bool = False):
        """Initialize sinusoidal input.
        
        Parameters
        ----------
        amplitude : float
            Amplitude of the sinusoid.
        frequency : Quantity
            Frequency of the sinus oscillation, in Hz.
        duration : float or Quantity
            The input duration.
        dt : float or Quantity, optional
            The numerical precision.
        t_start : float or Quantity, optional
            The start time. Default is 0.
        t_end : float or Quantity, optional
            The end time. Default is duration.
        bias : bool
            Whether the sinusoid oscillates around 0 (False), or
            has a positive DC bias, thus non-negative (True).
        """
        super().__init__(duration, dt)
        assert frequency.unit.dim == u.Hz.dim, f'The frequency must be in Hz. But got {frequency.unit}.'
        
        self.amplitude = amplitude
        self.frequency = frequency
        self.t_start = t_start
        self.t_end = t_end
        self.bias = bias
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the sinusoidal input array."""
        t_start = 0. * u.get_unit(self.dt) if self.t_start is None else self.t_start
        t_end = self.duration if self.t_end is None else self.t_end
        
        times = u.math.arange(0. * u.ms, t_end - t_start, self.dt)
        start_i = int(t_start / self.dt)
        end_i = int(t_end / self.dt)
        
        sin_inputs = self.amplitude * u.math.sin(2 * u.math.pi * times * self.frequency)
        if self.bias:
            sin_inputs += self.amplitude
            
        currents = u.math.zeros(self.n_steps,
                                dtype=brainstate.environ.dftype(),
                                unit=u.get_unit(sin_inputs))
        currents = currents.at[start_i:end_i].set(sin_inputs)
        return currents


class SquareInput(Input):
    """Square wave input generator.
    
    Examples
    --------
    >>> # Create a square wave with smoothed transitions
    >>> square = SquareInput(1.0, 5 * u.Hz, 500 * u.ms)
    >>> smoothed = square.smooth(tau=5 * u.ms)
    """
    
    def __init__(self,
                 amplitude: float,
                 frequency: u.Quantity,
                 duration: Union[float, u.Quantity],
                 dt: Optional[Union[float, u.Quantity]] = None,
                 bias: bool = False,
                 t_start: Optional[Union[float, u.Quantity]] = None,
                 t_end: Optional[Union[float, u.Quantity]] = None):
        """Initialize square wave input.
        
        Parameters
        ----------
        amplitude : float
            Amplitude of the square oscillation.
        frequency : Quantity
            Frequency of the square oscillation, in Hz.
        duration : float or Quantity
            The input duration.
        dt : float or Quantity, optional
            The numerical precision.
        bias : bool
            Whether the wave oscillates around 0 (False), or
            has a positive DC bias, thus non-negative (True).
        t_start : float or Quantity, optional
            The start time.
        t_end : float or Quantity, optional
            The end time.
        """
        super().__init__(duration, dt)
        assert frequency.unit.dim == u.Hz.dim, f'The frequency must be in Hz. But got {frequency.unit}.'
        
        self.amplitude = amplitude
        self.frequency = frequency
        self.bias = bias
        self.t_start = t_start
        self.t_end = t_end
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the square wave input array."""
        t_start = 0. * u.ms if self.t_start is None else self.t_start
        t_end = self.duration if self.t_end is None else self.t_end
        
        times = u.math.arange(0. * u.ms, t_end - t_start, self.dt)
        sin_inputs = self.amplitude * _square(2 * np.pi * times * self.frequency)
        
        if self.bias:
            sin_inputs += self.amplitude
            
        currents = u.math.zeros(self.n_steps, dtype=brainstate.environ.dftype())
        start_i = int(t_start / self.dt)
        end_i = int(t_end / self.dt)
        currents = currents.at[start_i:end_i].set(sin_inputs)
        return currents


class TriangularInput(Input):
    """Triangular wave input generator.
    
    Examples
    --------
    >>> # Create triangular wave and clip it
    >>> tri = TriangularInput(2.0, 3 * u.Hz, 600 * u.ms)
    >>> clipped = tri.clip(-1.5, 1.5)
    """
    
    def __init__(self,
                 amplitude: float,
                 frequency: u.Quantity,
                 duration: Union[float, u.Quantity],
                 dt: Optional[Union[float, u.Quantity]] = None,
                 t_start: Optional[Union[float, u.Quantity]] = None,
                 t_end: Optional[Union[float, u.Quantity]] = None,
                 bias: bool = False):
        """Initialize triangular wave input.
        
        Parameters
        ----------
        amplitude : float
            Amplitude of the triangular wave.
        frequency : Quantity
            Frequency of the triangular oscillation, in Hz.
        duration : float or Quantity
            The input duration.
        dt : float or Quantity, optional
            The numerical precision.
        t_start : float or Quantity, optional
            The start time.
        t_end : float or Quantity, optional
            The end time.
        bias : bool
            Whether the wave oscillates around 0 (False), or
            has a positive DC bias, thus non-negative (True).
        """
        super().__init__(duration, dt)
        assert frequency.unit.dim == u.Hz.dim, f'The frequency must be in Hz. But got {frequency.unit}.'
        
        self.amplitude = amplitude
        self.frequency = frequency
        self.t_start = t_start
        self.t_end = t_end
        self.bias = bias
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the triangular wave input array."""
        t_start = 0. * u.ms if self.t_start is None else self.t_start
        t_end = self.duration if self.t_end is None else self.t_end
        
        times = u.math.arange(0. * u.ms, t_end - t_start, self.dt)
        
        # Generate triangular wave using arcsin
        phase = 2 * u.math.pi * times * self.frequency
        triangular = (2.0 * self.amplitude / u.math.pi) * u.math.arcsin(u.math.sin(phase))
        
        if self.bias:
            triangular += self.amplitude
        
        currents = u.math.zeros(self.n_steps,
                                dtype=brainstate.environ.dftype(),
                                unit=u.get_unit(triangular))
        start_i = int(t_start / self.dt)
        end_i = int(t_end / self.dt)
        currents = currents.at[start_i:end_i].set(triangular)
        return currents


class SawtoothInput(Input):
    """Sawtooth wave input generator.
    
    Examples
    --------
    >>> # Create sawtooth and combine with DC offset
    >>> saw = SawtoothInput(1.0, 2 * u.Hz, 800 * u.ms)
    >>> offset = ConstantInput([(0.5, 800)], dt=0.1 * u.ms)
    >>> shifted_saw = saw + offset
    """
    
    def __init__(self,
                 amplitude: float,
                 frequency: u.Quantity,
                 duration: Union[float, u.Quantity],
                 dt: Optional[Union[float, u.Quantity]] = None,
                 t_start: Optional[Union[float, u.Quantity]] = None,
                 t_end: Optional[Union[float, u.Quantity]] = None,
                 bias: bool = False):
        """Initialize sawtooth wave input.
        
        Parameters
        ----------
        amplitude : float
            Amplitude of the sawtooth wave.
        frequency : Quantity
            Frequency of the sawtooth oscillation, in Hz.
        duration : float or Quantity
            The input duration.
        dt : float or Quantity, optional
            The numerical precision.
        t_start : float or Quantity, optional
            The start time.
        t_end : float or Quantity, optional
            The end time.
        bias : bool
            Whether the wave oscillates around 0 (False), or
            has a positive DC bias, thus non-negative (True).
        """
        super().__init__(duration, dt)
        assert frequency.unit.dim == u.Hz.dim, f'The frequency must be in Hz. But got {frequency.unit}.'
        
        self.amplitude = amplitude
        self.frequency = frequency
        self.t_start = t_start
        self.t_end = t_end
        self.bias = bias
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the sawtooth wave input array."""
        t_start = 0. * u.ms if self.t_start is None else self.t_start
        t_end = self.duration if self.t_end is None else self.t_end
        
        times = u.math.arange(0. * u.ms, t_end - t_start, self.dt)
        
        # Generate sawtooth wave
        phase = times * self.frequency
        sawtooth = 2 * self.amplitude * (phase - u.math.floor(phase) - 0.5)
        
        if self.bias:
            sawtooth += self.amplitude
        
        currents = u.math.zeros(self.n_steps,
                                dtype=brainstate.environ.dftype(),
                                unit=u.get_unit(sawtooth))
        start_i = int(t_start / self.dt)
        end_i = int(t_end / self.dt)
        currents = currents.at[start_i:end_i].set(sawtooth)
        return currents


class ChirpInput(Input):
    """Chirp signal (frequency sweep) input generator.
    
    Examples
    --------
    >>> # Create a chirp and repeat it
    >>> chirp = ChirpInput(1.0, 1 * u.Hz, 10 * u.Hz, 500 * u.ms)
    >>> repeated = chirp.repeat(3)  # Repeat 3 times
    """
    
    def __init__(self,
                 amplitude: float,
                 f_start: u.Quantity,
                 f_end: u.Quantity,
                 duration: Union[float, u.Quantity],
                 dt: Optional[Union[float, u.Quantity]] = None,
                 t_start: Optional[Union[float, u.Quantity]] = None,
                 t_end: Optional[Union[float, u.Quantity]] = None,
                 method: str = 'linear',
                 bias: bool = False):
        """Initialize chirp input.
        
        Parameters
        ----------
        amplitude : float
            Amplitude of the chirp signal.
        f_start : Quantity
            Starting frequency in Hz.
        f_end : Quantity
            Ending frequency in Hz.
        duration : float or Quantity
            Total duration of the input.
        dt : float or Quantity, optional
            The numerical precision.
        t_start : float or Quantity, optional
            The start time.
        t_end : float or Quantity, optional
            The end time.
        method : str
            'linear' for linear frequency sweep, 'logarithmic' for logarithmic sweep.
        bias : bool
            Whether to add positive DC bias.
        """
        super().__init__(duration, dt)
        assert f_start.unit.dim == u.Hz.dim, f'Start frequency must be in Hz. Got {f_start.unit}.'
        assert f_end.unit.dim == u.Hz.dim, f'End frequency must be in Hz. Got {f_end.unit}.'
        
        self.amplitude = amplitude
        self.f_start = f_start
        self.f_end = f_end
        self.t_start = t_start
        self.t_end = t_end
        self.method = method
        self.bias = bias
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the chirp input array."""
        t_start = 0. * u.ms if self.t_start is None else self.t_start
        t_end = self.duration if self.t_end is None else self.t_end
        
        times = u.math.arange(0. * u.ms, t_end - t_start, self.dt)
        
        f0 = self.f_start / u.Hz  # Convert to dimensionless Hz value
        f1 = self.f_end / u.Hz    # Convert to dimensionless Hz value
        T = (t_end - t_start) / u.ms
        times_ms = times / u.ms
        
        if self.method == 'linear':
            # Linear chirp: f(t) = f0 + (f1-f0)*t/T
            phase = 2 * u.math.pi * (f0 * times_ms / 1000 + 0.5 * (f1 - f0) * times_ms**2 / (T * 1000))
        elif self.method == 'logarithmic':
            # Logarithmic chirp
            k = (f1 / f0) ** (1 / T)
            phase = 2 * u.math.pi * f0 * (k**times_ms - 1) / (u.math.log(k) * 1000)
        else:
            raise ValueError(f"method must be 'linear' or 'logarithmic', got {self.method}")
        
        chirp = self.amplitude * u.math.sin(phase)
        
        if self.bias:
            chirp += self.amplitude
        
        currents = u.math.zeros(self.n_steps,
                                dtype=brainstate.environ.dftype(),
                                unit=u.get_unit(chirp))
        start_i = int(t_start / self.dt)
        end_i = int(t_end / self.dt)
        currents = currents.at[start_i:end_i].set(chirp)
        return currents


class NoisySinusoidalInput(Input):
    """Sinusoidal input with additive noise.
    
    Examples
    --------
    >>> # Create noisy sinusoid and filter it
    >>> noisy = NoisySinusoidalInput(1.0, 10 * u.Hz, 0.2, 500 * u.ms)
    >>> filtered = noisy.smooth(tau=10 * u.ms)  # Low-pass filter
    """
    
    def __init__(self,
                 amplitude: float,
                 frequency: u.Quantity,
                 noise_amplitude: float,
                 duration: Union[float, u.Quantity],
                 dt: Optional[Union[float, u.Quantity]] = None,
                 t_start: Optional[Union[float, u.Quantity]] = None,
                 t_end: Optional[Union[float, u.Quantity]] = None,
                 seed: Optional[int] = None):
        """Initialize noisy sinusoidal input.
        
        Parameters
        ----------
        amplitude : float
            Amplitude of the sinusoid.
        frequency : Quantity
            Frequency of the sinus oscillation, in Hz.
        noise_amplitude : float
            Amplitude of the additive Gaussian noise.
        duration : float or Quantity
            The input duration.
        dt : float or Quantity, optional
            The numerical precision.
        t_start : float or Quantity, optional
            The start time.
        t_end : float or Quantity, optional
            The end time.
        seed : int, optional
            Random seed for noise generation.
        """
        super().__init__(duration, dt)
        assert frequency.unit.dim == u.Hz.dim, f'Frequency must be in Hz. Got {frequency.unit}.'
        
        self.amplitude = amplitude
        self.frequency = frequency
        self.noise_amplitude = noise_amplitude
        self.t_start = t_start
        self.t_end = t_end
        self.seed = seed
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the noisy sinusoidal input array."""
        if self.seed is None:
            rng = brainstate.random.DEFAULT
        else:
            rng = brainstate.random.RandomState(self.seed)
        
        t_start = 0. * u.ms if self.t_start is None else self.t_start
        t_end = self.duration if self.t_end is None else self.t_end
        
        times = u.math.arange(0. * u.ms, t_end - t_start, self.dt)
        
        # Generate sinusoidal component
        sin_component = self.amplitude * u.math.sin(2 * u.math.pi * times * self.frequency)
        
        # Add noise
        noise = self.noise_amplitude * rng.standard_normal(len(times))
        noisy_signal = sin_component + noise
        
        currents = u.math.zeros(self.n_steps,
                                dtype=brainstate.environ.dftype(),
                                unit=u.get_unit(noisy_signal))
        start_i = int(t_start / self.dt)
        end_i = int(t_end / self.dt)
        currents = currents.at[start_i:end_i].set(noisy_signal)
        return currents