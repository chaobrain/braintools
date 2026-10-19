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

from typing import Optional, Union

import brainstate
import brainunit as u
import numpy as onp
from jax import vmap, lax, numpy as jnp

from braintools._misc import set_module_as

__all__ = [
    'cross_correlation',
]


@set_module_as('braintools.analysis.spike')
def cross_correlation(
    spikes: brainstate.typing.ArrayLike,
    bin: Union[int, float],
    dt: Optional[Union[int, float]] = None,
    method: str = 'loop'
):
    r"""Calculate cross-correlation index between neurons.

    The coherence between two neurons i and j is measured by their
    cross-correlation of spike trains at zero time lag within a time bin.
    This function computes the population synchronization index based on
    pairwise cross-correlations [1]_.

    The coherence measure for a pair is defined as:

    .. math::

        \kappa_{ij}(\tau) = \frac{\sum_{l=1}^{K} X(l) Y(l)}
        {\sqrt{\left(\sum_{l=1}^{K} X(l)\right) \left(\sum_{l=1}^{K} Y(l)\right)}}

    where the time interval is divided into K bins of size :math:`\Delta t = \tau`,
    and :math:`X(l)`, :math:`Y(l)` are binary spike indicators (0 or 1) for each bin.

    The population coherence measure :math:`\kappa(\tau)` is the average of
    :math:`\kappa_{ij}(\tau)` over all pairs of neurons.

    Parameters
    ----------
    spikes : brainstate.typing.ArrayLike
        Spike history matrix with shape ``(num_time, num_neurons)``.
        Binary values indicating spike occurrences.
    bin : float or brainunit.Quantity
        Time bin size for binning spike trains. A plain number is in
        milliseconds; a Quantity is converted.
    dt : float or brainunit.Quantity, optional
        Time step of ``spikes``. If None, uses ``brainstate.environ.get_dt()``.
        A plain number is in milliseconds; a Quantity is converted.
    method : str, default='loop'
        Method for computing cross-correlations:
        
        - ``'loop'``: Memory-efficient iterative approach
        - ``'vmap'``: Vectorized approach (uses more memory)

    Returns
    -------
    float
        Cross-correlation index representing the population synchronization level.
        Values closer to 1 indicate higher synchronization.

    Notes
    -----
    To JIT compile this function, make ``bin``, ``dt``, and ``method`` static.
    For example: ``partial(cross_correlation, bin=10, method='loop')``.

    This is the coincidence-based coherence of Wang & Buzsáki (1996), not a
    Pearson correlation coefficient.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> spikes = jnp.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        >>> sync_index = braintools.analysis.spike.cross_correlation(spikes, bin=1.0, dt=1.0)
        >>> float(sync_index) >= 0.0
        True

        >>> # For larger datasets, the vectorized method is faster.
        >>> import brainstate
        >>> large_spikes = (brainstate.random.rand(1000, 50) < 0.1).astype(float)
        >>> sync_fast = braintools.analysis.spike.cross_correlation(
        ...     large_spikes, bin=10.0, dt=1.0, method='vmap')

    References
    ----------
    .. [1] Wang, Xiao-Jing, and György Buzsáki. "Gamma oscillation by synaptic
           inhibition in a hippocampal interneuronal network model." Journal of
           Neuroscience 16.20 (1996): 6402-6413.
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    # Quantities are converted; a plain number is taken to be in milliseconds
    # (the brainstate convention), so a float ``bin`` works with the default
    # Quantity ``dt``.
    to_ms = lambda x: float(x.to_decimal(u.ms)) if isinstance(x, u.Quantity) else float(x)
    bin_size = int(round(to_ms(bin) / to_ms(dt)))
    if bin_size < 1:
        raise ValueError(f'`bin` ({bin}) must be at least as large as `dt` ({dt}); '
                         f'got a bin width of {bin_size} samples.')
    # Strip units so ``Quantity`` spike matrices behave like the other metrics.
    spikes = jnp.asarray(u.get_magnitude(spikes))
    num_hist, num_neu = spikes.shape
    if num_neu < 2:
        # Coincidence is only defined for pairs; a single neuron has no pairs.
        return jnp.asarray(0.0)
    num_bin = int(onp.ceil(num_hist / bin_size))
    if num_bin * bin_size != num_hist:
        spikes = jnp.append(spikes, jnp.zeros((num_bin * bin_size - num_hist, num_neu)), axis=0)
    states = spikes.T.reshape((num_neu, num_bin, bin_size))
    states = jnp.asarray(jnp.sum(states, axis=2) > 0., dtype=float)
    indices = jnp.tril_indices(num_neu, k=-1)

    if method == 'loop':
        def _f(i, j):
            sqrt_ij = jnp.sqrt(jnp.sum(states[i]) * jnp.sum(states[j]))
            return lax.cond(sqrt_ij == 0.,
                            lambda _: jnp.zeros((), dtype=sqrt_ij.dtype),
                            lambda _: jnp.sum(states[i] * states[j]) / sqrt_ij,
                            None)

        res = brainstate.transform.for_loop(_f, *indices)

    elif method == 'vmap':
        @vmap
        def _cc(i, j):
            sqrt_ij = jnp.sqrt(jnp.sum(states[i]) * jnp.sum(states[j]))
            return lax.cond(sqrt_ij == 0.,
                            lambda _: jnp.zeros((), dtype=sqrt_ij.dtype),
                            lambda _: jnp.sum(states[i] * states[j]) / sqrt_ij,
                            None)

        res = _cc(*indices)
    else:
        raise ValueError(f'Do not support {method}. We only support "loop" or "vmap".')

    return jnp.mean(jnp.asarray(res))
