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

"""Losses for fitting models to data.

Every loss has the signature ``loss(prediction, target, **options) -> scalar``
and can be wrapped in a :class:`braintools.fit.Term`. Each docstring states
whether the loss is differentiable.

- **Pointwise**: :func:`mse`, :func:`rmse`, :func:`mae`,
  :func:`correlation_loss`.
- **Functional connectivity**: :func:`fc_correlation_loss`, :func:`fc_rmse`,
  :func:`fcd_ks`, :func:`fcd_wasserstein`.
- **Distributions**: :func:`ks_distance`, :func:`wasserstein_1d`.
- **Summary statistics**: :func:`psd_log_distance`, :func:`rate_loss`,
  :func:`cv_loss`.
- **Spike trains**: :func:`van_rossum_loss` (differentiable on dense rasters).

Time steps given as plain numbers follow the convention of the matching
analysis module: ``dt`` is in **seconds** for :func:`psd_log_distance` (as in
:mod:`braintools.analysis.spectral`) but in **milliseconds** for
:func:`rate_loss` and :func:`van_rossum_loss` (as in
:mod:`braintools.analysis.spike` and ``brainstate``). Passing ``dt`` as a
``brainunit`` Quantity avoids the ambiguity.

Examples
--------
.. code-block:: python

    >>> import jax.numpy as jnp
    >>> from braintools.fit import losses
    >>> float(losses.mse(jnp.array([1.0, 2.0]), jnp.array([1.0, 4.0])))
    2.0
"""

from ._losses import (
    mse,
    rmse,
    mae,
    correlation_loss,
    fc_correlation_loss,
    fc_rmse,
    fcd_ks,
    fcd_wasserstein,
    ks_distance,
    wasserstein_1d,
    psd_log_distance,
    rate_loss,
    cv_loss,
    van_rossum_loss,
)

__all__ = [
    'mse',
    'rmse',
    'mae',
    'correlation_loss',
    'fc_correlation_loss',
    'fc_rmse',
    'fcd_ks',
    'fcd_wasserstein',
    'ks_distance',
    'wasserstein_1d',
    'psd_log_distance',
    'rate_loss',
    'cv_loss',
    'van_rossum_loss',
]
