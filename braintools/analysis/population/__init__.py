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

"""Population-level analysis of continuous multichannel activity.

The functions take activity of shape ``(n_time, n_signals)``, such as membrane
potentials, firing rates or regional signals.

- :func:`voltage_fluctuation`: Golomb's :math:`\\chi^2` synchrony index.
- :func:`functional_connectivity`: pairwise Pearson correlation matrix.
- :func:`functional_connectivity_dynamics`: correlation between
  sliding-window connectivity matrices.
- **Dimensionality and alignment**: :func:`pca`, :func:`participation_ratio`,
  :func:`cca`, :func:`rdm`, :func:`rsa`.
- **Decoding and information**: :func:`decode`, :func:`mutual_information`.
- **Collective dynamics**: :func:`kuramoto_order_parameter`,
  :func:`granger_causality`, :func:`avalanches`, :func:`branching_ratio`.

Examples
--------
.. code-block:: python

    >>> import jax.numpy as jnp
    >>> from braintools.analysis import population
    >>> t = jnp.linspace(0, 10, 200)
    >>> activities = jnp.column_stack([jnp.sin(t), jnp.sin(t), -jnp.sin(t)])
    >>> fc = population.functional_connectivity(activities)
    >>> fc.shape
    (3, 3)
    >>> round(float(fc[0, 2]), 3)
    -1.0
"""

from ._correlation import (
    voltage_fluctuation,
    functional_connectivity,
    functional_connectivity_dynamics,
)
from ._decoding import decode
from ._dimensionality import (
    PCAResult,
    CCAResult,
    pca,
    participation_ratio,
    cca,
    rdm,
    rsa,
)
from ._dynamics import (
    AvalancheResult,
    kuramoto_order_parameter,
    granger_causality,
    avalanches,
    branching_ratio,
)
from ._information import mutual_information

__all__ = [
    'voltage_fluctuation',
    'functional_connectivity',
    'functional_connectivity_dynamics',
    'PCAResult',
    'CCAResult',
    'pca',
    'participation_ratio',
    'cca',
    'rdm',
    'rsa',
    'decode',
    'mutual_information',
    'AvalancheResult',
    'kuramoto_order_parameter',
    'granger_causality',
    'avalanches',
    'branching_ratio',
]
