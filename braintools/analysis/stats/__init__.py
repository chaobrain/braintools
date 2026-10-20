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

"""General statistics used across neural data analysis.

- :func:`matrix_correlation`: Pearson correlation between the upper triangles
  of two square matrices (for example, two connectivity matrices).
- :func:`weighted_correlation`: weighted Pearson correlation of two series.
- **Resampling** (building blocks for significance tests, often called
  "surrogate data" in the literature): :func:`jitter_spikes`,
  :func:`shuffle_isi`, :func:`phase_randomize`.
- :func:`bootstrap_ci`: bootstrap confidence intervals of any statistic.

Examples
--------
.. code-block:: python

    >>> import jax.numpy as jnp
    >>> from braintools.analysis import stats
    >>> x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> w = jnp.array([1.0, 1.0, 2.0, 2.0, 1.0])
    >>> round(float(stats.weighted_correlation(x, 2 * x, w)), 3)
    1.0
"""

from ._correlation import (
    matrix_correlation,
    weighted_correlation,
)
from ._resampling import (
    jitter_spikes,
    shuffle_isi,
    phase_randomize,
    bootstrap_ci,
)

__all__ = [
    'matrix_correlation',
    'weighted_correlation',
    'jitter_spikes',
    'shuffle_isi',
    'phase_randomize',
    'bootstrap_ci',
]
