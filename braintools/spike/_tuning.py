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

"""Tuning curves shared by the population encoders.

This module is private: :func:`gaussian_tuning` is used by
:class:`braintools.spike.PopulationEncoder` and by
:func:`braintools.cogtask.gaussian`, so both compute their receptive fields
with exactly the same arithmetic.
"""

import brainunit as u

__all__ = [
    'gaussian_tuning',
]


def gaussian_tuning(x, centers, sigma):
    r"""Evaluate Gaussian tuning curves.

    Computes

    .. math::

        r = \exp\left(-\frac{1}{2}\left(\frac{x - c}{\sigma}\right)^2\right),

    a bell curve with peak value 1 at :math:`x = c`. ``x``, ``centers`` and
    ``sigma`` follow NumPy broadcasting; no axis is inserted. To evaluate a
    population of tuning curves at every element of ``x``, pass
    ``x[..., None]`` so the result has shape ``(*x.shape, n_centers)``.

    Parameters
    ----------
    x : float, array_like or Quantity
        Stimulus values.
    centers : float, array_like or Quantity
        Preferred values of the tuning curves. Must have the same physical
        dimension as ``x``.
    sigma : float, array_like or Quantity
        Width (standard deviation) of the tuning curves. Must have the same
        physical dimension as ``x``. It must be nonzero: ``sigma == 0``
        gives ``nan`` where ``x == centers`` and ``0`` elsewhere.

    Returns
    -------
    jax.Array
        Dimensionless responses in ``[0, 1]``, with the broadcast shape of
        ``x``, ``centers`` and ``sigma``. Quantities are converted to a
        plain array because ``(x - centers) / sigma`` is dimensionless.

    Raises
    ------
    brainunit.UnitMismatchError
        If ``x`` and ``centers`` have different physical dimensions.
    TypeError
        If ``(x - centers) / sigma`` is not dimensionless.

    See Also
    --------
    braintools.spike.PopulationEncoder : Spiking population code built on these curves.
    braintools.cogtask.gaussian : Cognitive-task input encoder built on these curves.

    Notes
    -----
    The expression is evaluated as ``exp(-0.5 * ((x - centers) / sigma) ** 2)``
    in this exact order, which the two call sites relied on before they were
    unified; changing the order would change results in the last bit.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from braintools.spike._tuning import gaussian_tuning
        >>> centers = jnp.linspace(0., 1., 5)
        >>> r = gaussian_tuning(0.5, centers, 0.25)
        >>> r.shape
        (5,)
        >>> float(r[2])
        1.0

        >>> x = jnp.array([0.0, 1.0])
        >>> gaussian_tuning(x[..., None], centers, 0.25).shape
        (2, 5)

        >>> r = gaussian_tuning(0.5 * u.mV, centers * u.mV, 250. * u.uvolt)
        >>> float(r[2])
        1.0
    """
    return u.math.exp(-0.5 * ((x - centers) / sigma) ** 2)
