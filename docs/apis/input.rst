``braintools.input`` module
===========================

.. currentmodule:: braintools.input

The input toolkit offers both functional helpers and composable classes for
constructing stimulation protocols. The tables below group the public API by
category. Legacy names ending with ``Input``/``_input`` remain available as
deprecated aliases and emit ``DeprecationWarning`` when used.

Functional API
--------------

Basic currents
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   section
   constant
   step
   ramp

Pulse generators
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   spike
   gaussian_pulse
   exponential_decay
   double_exponential
   burst

Stochastic processes
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   wiener_process
   ou_process
   poisson

Waveforms
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   sinusoidal
   square
   triangular
   sawtooth
   chirp
   noisy_sinusoidal

Composable API
--------------

Base classes and utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Input
   Composite
   ConstantValue
   Sequential
   TimeShifted
   Clipped
   Smoothed
   Repeated
   Transformed

Basic currents
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Section
   Constant
   Step
   Ramp

Pulse generators
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Spike
   GaussianPulse
   ExponentialDecay
   DoubleExponential
   Burst

Stochastic processes
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   WienerProcess
   OUProcess
   Poisson

Waveforms
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Sinusoidal
   Square
   Triangular
   Sawtooth
   Chirp
   NoisySinusoidal
