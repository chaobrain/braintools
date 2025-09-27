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

from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.artist import Artist
from matplotlib.figure import Figure

from braintools.tree import as_numpy
from braintools._misc import set_module_as

__all__ = [
    'animator',
]


@set_module_as('braintools.visualize')
def animator(
    data,
    fig,
    ax,
    num_steps=False,
    interval=40,
    cmap="plasma"
):
    """Generate an animation by looping through the first dimension of spiking data.

    Time must be the first dimension of ``data``. This function creates an
    animation by capturing snapshots of each time step and combining them
    into a video or GIF format.

    Parameters
    ----------
    data : array_like
        Data tensor for a single sample across time steps with shape
        ``[num_steps, *spatial_dims]``. Time must be the first dimension.
    fig : matplotlib.figure.Figure
        Top level container for all plot elements.
    ax : matplotlib.axes.Axes
        Contains additional figure elements and sets the coordinate system.
    num_steps : int or False, default=False
        Number of time steps to plot. If ``False``, the number of entries
        in the first dimension of ``data`` will automatically be used.
    interval : int, default=40
        Delay between frames in milliseconds.
    cmap : str, default="plasma"
        Matplotlib colormap name for rendering the data.

    Returns
    -------
    matplotlib.animation.ArtistAnimation
        Animation object that can be displayed using ``matplotlib.pyplot.show()``
        or saved to file.

    Examples
    --------
    .. code-block:: python

        >>> import matplotlib.pyplot as plt
        >>> import braintools
        >>> import jax.numpy as jnp
        >>> from IPython.display import HTML
        >>>
        >>> # Index into a single sample from a minibatch
        >>> spike_data_sample = jnp.random.rand(100, 28, 28)
        >>> print(spike_data_sample.shape)
        (100, 28, 28)
        >>>
        >>> # Plot
        >>> fig, ax = plt.subplots()
        >>> anim = braintools.visualize.animator(spike_data_sample, fig, ax)
        >>> HTML(anim.to_html5_video())
        >>>
        >>> # Save as a gif
        >>> anim.save("spike_mnist.gif")

    Notes
    -----
    This function uses the Camera class internally to capture snapshots
    of the matplotlib figure at each time step and create the animation.
    """

    data = as_numpy(data)
    if not num_steps:
        num_steps = data.shape[0]
    camera = Camera(fig)
    plt.axis("off")
    # iterate over time and take a snapshot with celluloid
    for step in range(
        num_steps
    ):  # im appears unused but is required by camera.snap()
        im = ax.imshow(data[step], cmap=cmap)  # noqa: F841
        camera.snap()
    anim = camera.animate(interval=interval)
    return anim


class Camera:
    """Make animations easier by capturing figure snapshots.

    This class provides a simple interface for creating animations from
    matplotlib figures by capturing the current state at each frame.
    """
    __module__ = 'braintools.visualize'

    def __init__(self, figure: Figure) -> None:
        """Create camera from matplotlib figure.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            The matplotlib figure to capture snapshots from.
        """
        self._figure = figure
        # need to keep track off artists for each axis
        self._offsets: Dict[str, Dict[int, int]] = {
            k: defaultdict(int)
            for k in [
                "collections",
                "patches",
                "lines",
                "texts",
                "artists",
                "images",
            ]
        }
        self._photos: List[List[Artist]] = []

    def snap(self) -> List[Artist]:
        """Capture current state of the figure.

        Returns
        -------
        List[matplotlib.artist.Artist]
            List of artists captured in this frame.
        """
        frame_artists: List[Artist] = []
        for i, axis in enumerate(self._figure.axes):
            if axis.legend_ is not None:
                axis.add_artist(axis.legend_)
            for name in self._offsets:
                new_artists = getattr(axis, name)[self._offsets[name][i]:]
                frame_artists += new_artists
                self._offsets[name][i] += len(new_artists)
        self._photos.append(frame_artists)
        return frame_artists

    def animate(self, *args, **kwargs) -> ArtistAnimation:
        """Animate the snapshots taken.

        Uses matplotlib.animation.ArtistAnimation to create the animation
        from all captured snapshots.

        Parameters
        ----------
        *args
            Positional arguments passed to ArtistAnimation.
        **kwargs
            Keyword arguments passed to ArtistAnimation.

        Returns
        -------
        matplotlib.animation.ArtistAnimation
            The created animation object.
        """
        return ArtistAnimation(self._figure, self._photos, *args, **kwargs)
