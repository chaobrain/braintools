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

"""Spectral analysis of continuous signals such as local field potentials.

All functions take the sampling interval ``dt`` (a plain number is in
**seconds**; a :class:`brainunit.Quantity` is converted) and return
frequencies in Hz.

- **Spectra**: :func:`power_spectral_density` (Welch), :func:`multitaper_psd`,
  :func:`spectrogram`, :func:`band_power`, :func:`spectral_entropy`.
- **Time-frequency and filtering**: :func:`morlet_transform`, :func:`bandpass`,
  :func:`analytic_signal`.
- **Coupling between signals**: :func:`plv`, :func:`pli`, :func:`wpli`,
  :func:`coherence_analysis`, :func:`lfp_phase_coherence`.
- **Cross-frequency coupling**: :func:`phase_amplitude_coupling`,
  :func:`theta_gamma_coupling`.

Examples
--------
.. code-block:: python

    >>> import brainunit as u
    >>> import jax.numpy as jnp
    >>> from braintools.analysis import spectral
    >>> t = jnp.arange(4000) * 0.001
    >>> lfp = jnp.sin(2 * jnp.pi * 10 * t)
    >>> freqs, psd = spectral.power_spectral_density(lfp, dt=1 * u.ms, nperseg=1000)
    >>> float(freqs[jnp.argmax(psd)])
    10.0
"""

from ._lfp import (
    power_spectral_density,
    coherence_analysis,
    phase_amplitude_coupling,
    theta_gamma_coupling,
    spectral_entropy,
    lfp_phase_coherence,
)
from ._spectral import (
    spectrogram,
    multitaper_psd,
    morlet_transform,
    bandpass,
    analytic_signal,
    band_power,
    plv,
    pli,
    wpli,
)

__all__ = [
    'power_spectral_density',
    'coherence_analysis',
    'phase_amplitude_coupling',
    'theta_gamma_coupling',
    'spectral_entropy',
    'lfp_phase_coherence',
    'spectrogram',
    'multitaper_psd',
    'morlet_transform',
    'bandpass',
    'analytic_signal',
    'band_power',
    'plv',
    'pli',
    'wpli',
]
