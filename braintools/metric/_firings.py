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

from typing import Union

import brainstate as bst
import brainunit as bu
import jax.numpy as jnp
import numpy as onp

__all__ = [
  'raster_plot',
  'firing_rate',
]


def raster_plot(
    sp_matrix: bst.typing.ArrayLike,
    times: bst.typing.ArrayLike
):
  """Get spike raster plot which displays the spiking activity
  of a group of neurons over time.

  Parameters
  ----------
  sp_matrix : bnp.ndarray
      The matrix which record spiking activities.
  times : bnp.ndarray
      The time steps.

  Returns
  -------
  raster_plot : tuple
      Include (neuron index, spike time).
  """
  times = onp.asarray(times)
  elements = onp.where(sp_matrix > 0.)
  index = elements[1]
  time = times[elements[0]]
  return index, time


def firing_rate(
    spikes: bst.typing.ArrayLike,
    width: Union[float, bu.Quantity],
    dt: Union[float, bu.Quantity] = None
):
  r"""Calculate the mean firing rate over in a neuron group.

  This method is adopted from Brian2.

  The firing rate in trial :math:`k` is the spike count :math:`n_{k}^{sp}`
  in an interval of duration :math:`T` divided by :math:`T`:

  .. math::

      v_k = {n_k^{sp} \over T}

  Parameters
  ----------
  spikes : ndarray
    The spike matrix which record spiking activities.
  width : int, float, Quantity
    The width of the ``window`` in millisecond.
  dt : float, Quantity, optional
    The sample rate.

  Returns
  -------
  rate : ndarray
      The population rate in Hz, smoothed with the given window.
  """
  dt = bst.environ.get_dt() if (dt is None) else dt
  width1 = int(width / 2 / dt) * 2 + 1
  window = jnp.ones(width1) * 1000 / width
  return jnp.convolve(jnp.mean(spikes, axis=1), window, mode='same')
