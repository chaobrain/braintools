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
Input Current Generation for Neural Simulations.

This module provides comprehensive tools for generating input currents for neural
simulations, supporting both composable object-oriented API and functional API.

**Key Features:**

- **Composable API**: Object-oriented interface with operator overloading
- **Functional API**: Traditional function-based interface for backward compatibility
- **Basic Inputs**: Constant, step, ramp, section-based currents
- **Pulse Patterns**: Spikes, Gaussian pulses, exponential shapes, bursts
- **Waveforms**: Sinusoidal, square, triangular, sawtooth, chirp
- **Stochastic Processes**: Wiener process, Ornstein-Uhlenbeck process, Poisson
- **Unit-Aware**: Full integration with BrainUnit for physical quantities

**Quick Start - Composable API (Recommended):**

.. code-block:: python

    import brainunit as u
    import brainstate
    from braintools.input import Ramp, Sinusoidal, Constant

    # Set time step
    brainstate.environ.set(dt=0.1 * u.ms)

    # Create basic inputs
    ramp = Ramp(0, 1, 500 * u.ms)
    sine = Sinusoidal(0.5, 10 * u.Hz, 500 * u.ms)
    baseline = Constant(0.2, 500 * u.ms)

    # Combine using operators
    combined = baseline + ramp + sine * 0.5

    # Apply transformations
    clipped = combined.clip(0, 1)
    smoothed = clipped.smooth(10 * u.ms)

    # Generate the array
    current = smoothed()

**Quick Start - Functional API:**

.. code-block:: python

    import brainunit as u
    import brainstate
    from braintools.input import ramp, sinusoidal, constant

    # Set time step
    brainstate.environ.set(dt=0.1 * u.ms)

    # Generate inputs
    ramp_signal = ramp(0, 1, 500 * u.ms)
    sine_signal = sinusoidal(0.5, 10 * u.Hz, 500 * u.ms)
    baseline_signal = constant(0.2, 500 * u.ms)

    # Combine using array operations
    combined = baseline_signal + ramp_signal + sine_signal * 0.5

**Basic Inputs:**

.. code-block:: python

    import brainunit as u
    import brainstate
    from braintools.input import Section, Constant, Step, Ramp

    brainstate.environ.set(dt=0.1 * u.ms)

    # Section-based input (different phases)
    section = Section(
        values=[0, 1, 0.5, 0] * u.nA,
        durations=[100, 200, 200, 100] * u.ms
    )

    # Constant current
    const = Constant(1.5 * u.nA, 500 * u.ms)

    # Step function
    step = Step(
        values=[0, 1, 0.5],
        times=[0, 100, 400],
        duration=500 * u.ms
    )

    # Linear ramp
    ramp = Ramp(0, 2 * u.nA, 500 * u.ms)

**Pulse Patterns:**

.. code-block:: python

    import brainunit as u
    import brainstate
    from braintools.input import (
        Spike, GaussianPulse, ExponentialDecay,
        DoubleExponential, Burst
    )

    brainstate.environ.set(dt=0.1 * u.ms)

    # Spike train
    spikes = Spike(
        sp_times=[100, 200, 300] * u.ms,
        duration=500 * u.ms,
        sp_lens=2 * u.ms,
        sp_sizes=[1, 1.5, 1.2] * u.nA
    )

    # Gaussian pulse
    pulse = GaussianPulse(
        amplitude=2 * u.nA,
        center=250 * u.ms,
        sigma=50 * u.ms,
        duration=500 * u.ms
    )

    # Exponential decay (e.g., EPSC)
    exp_decay = ExponentialDecay(
        amplitude=1 * u.nA,
        tau=20 * u.ms,
        t_start=100 * u.ms,
        duration=500 * u.ms
    )

    # Double exponential (realistic synapse)
    synapse = DoubleExponential(
        amplitude=1 * u.nA,
        tau_rise=2 * u.ms,
        tau_decay=20 * u.ms,
        t_start=100 * u.ms,
        duration=500 * u.ms
    )

    # Burst of activity
    burst = Burst(
        amplitude=1 * u.nA,
        n_pulses=5,
        pulse_duration=10 * u.ms,
        interval=50 * u.ms,
        t_start=100 * u.ms,
        duration=500 * u.ms
    )

**Waveforms:**

.. code-block:: python

    import brainunit as u
    import brainstate
    from braintools.input import (
        Sinusoidal, Square, Triangular,
        Sawtooth, Chirp, NoisySinusoidal
    )

    brainstate.environ.set(dt=0.1 * u.ms)

    # Sinusoidal
    sine = Sinusoidal(
        amplitude=1 * u.nA,
        frequency=10 * u.Hz,
        duration=1000 * u.ms
    )

    # Square wave
    square = Square(
        amplitude=1 * u.nA,
        frequency=5 * u.Hz,
        duration=1000 * u.ms,
        duty_cycle=0.5
    )

    # Triangular wave
    triangle = Triangular(
        amplitude=1 * u.nA,
        frequency=5 * u.Hz,
        duration=1000 * u.ms
    )

    # Sawtooth wave
    sawtooth = Sawtooth(
        amplitude=1 * u.nA,
        frequency=5 * u.Hz,
        duration=1000 * u.ms
    )

    # Chirp (frequency sweep)
    chirp = Chirp(
        amplitude=1 * u.nA,
        f_start=1 * u.Hz,
        f_end=50 * u.Hz,
        duration=1000 * u.ms
    )

    # Noisy sinusoidal
    noisy_sine = NoisySinusoidal(
        amplitude=1 * u.nA,
        frequency=10 * u.Hz,
        duration=1000 * u.ms,
        noise_amplitude=0.1 * u.nA
    )

**Stochastic Processes:**

.. code-block:: python

    import brainunit as u
    import brainstate
    from braintools.input import WienerProcess, OUProcess, Poisson

    brainstate.environ.set(dt=0.1 * u.ms)

    # Wiener process (Brownian motion)
    wiener = WienerProcess(
        duration=1000 * u.ms,
        sigma=0.1 * u.nA,
        seed=42
    )

    # Ornstein-Uhlenbeck process
    ou = OUProcess(
        duration=1000 * u.ms,
        mean=0.5 * u.nA,
        sigma=0.2 * u.nA,
        tau=20 * u.ms,
        seed=42
    )

    # Poisson spike train
    poisson = Poisson(
        duration=1000 * u.ms,
        rate=20 * u.Hz,
        amplitude=1 * u.nA,
        seed=42
    )

**Composability:**

.. code-block:: python

    import brainunit as u
    import brainstate
    from braintools.input import Constant, Sinusoidal, WienerProcess

    brainstate.environ.set(dt=0.1 * u.ms)

    # Build complex stimulation protocols
    baseline = Constant(0.5 * u.nA, 1000 * u.ms)
    modulation = Sinusoidal(0.3 * u.nA, 5 * u.Hz, 1000 * u.ms)
    noise = WienerProcess(1000 * u.ms, sigma=0.05 * u.nA)

    # Combine with operators
    signal = baseline + modulation + noise

    # Apply transformations
    clipped = signal.clip(0 * u.nA, 1.5 * u.nA)
    smoothed = clipped.smooth(5 * u.ms)
    delayed = smoothed.shift(50 * u.ms)

    # Sequential concatenation
    protocol = baseline & modulation & baseline

    # Generate
    current = protocol()

"""

# Composable API - Base classes and transformations
from ._composable_base import (
    Input,
    Composite,
    ConstantValue,
    Sequential,
    TimeShifted,
    Clipped,
    Smoothed,
    Repeated,
    Transformed,
)

# Composable API - Basic inputs
from ._composable_basic import (
    Section,
    Constant,
    Step,
    Ramp,
)

# Composable API - Pulse patterns
from ._composable_pulses import (
    Spike,
    GaussianPulse,
    ExponentialDecay,
    DoubleExponential,
    Burst,
)

# Composable API - Waveforms
from ._composable_waveforms import (
    Sinusoidal,
    Square,
    Triangular,
    Sawtooth,
    Chirp,
    NoisySinusoidal,
)

# Composable API - Stochastic processes
from ._composable_stochastic import (
    WienerProcess,
    OUProcess,
    Poisson,
)

# Functional API - Basic inputs
from ._functional_basic import (
    section,
    constant,
    step,
    ramp,
)

# Functional API - Pulse patterns
from ._functional_pulses import (
    spike,
    gaussian_pulse,
    exponential_decay,
    double_exponential,
    burst,
)

# Functional API - Waveforms
from ._functional_waveforms import (
    sinusoidal,
    square,
    triangular,
    sawtooth,
    chirp,
    noisy_sinusoidal,
)

# Functional API - Stochastic processes
from ._functional_stochastic import (
    wiener_process,
    ou_process,
    poisson,
)

__all__ = [
    # Composable API - Base classes
    'Input',
    'Composite',
    'ConstantValue',
    'Sequential',
    'TimeShifted',
    'Clipped',
    'Smoothed',
    'Repeated',
    'Transformed',

    # Composable API - Basic inputs
    'Section',
    'Constant',
    'Step',
    'Ramp',

    # Composable API - Pulse patterns
    'Spike',
    'GaussianPulse',
    'ExponentialDecay',
    'DoubleExponential',
    'Burst',

    # Composable API - Waveforms
    'Sinusoidal',
    'Square',
    'Triangular',
    'Sawtooth',
    'Chirp',
    'NoisySinusoidal',

    # Composable API - Stochastic processes
    'WienerProcess',
    'OUProcess',
    'Poisson',

    # Functional API - Basic inputs
    'section',
    'constant',
    'step',
    'ramp',

    # Functional API - Pulse patterns
    'spike',
    'gaussian_pulse',
    'exponential_decay',
    'double_exponential',
    'burst',

    # Functional API - Waveforms
    'sinusoidal',
    'square',
    'triangular',
    'sawtooth',
    'chirp',
    'noisy_sinusoidal',

    # Functional API - Stochastic processes
    'wiener_process',
    'ou_process',
    'poisson',
]
