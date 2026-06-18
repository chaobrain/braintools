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

from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.artist import Artist
from matplotlib.figure import Figure

from braintools._misc import set_module_as
from braintools.tree import as_numpy

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
    """Generate an animation by looping through the first dimension of a
    sample of image-like data. Time must be the first dimension of ``data``.

    Example::

        import numpy as np
        import matplotlib.pyplot as plt
        from IPython.display import HTML
        from braintools.visualize import animator

        #  A single sample across 100 time steps of 28x28 frames
        spike_data_sample = np.random.rand(100, 28, 28)

        fig, ax = plt.subplots()
        anim = animator(spike_data_sample, fig, ax)
        HTML(anim.to_html5_video())

        #  Save as a gif
        anim.save("spike_mnist.gif", writer="pillow")

    Parameters
    ----------
    data : array-like
        Data array of shape ``(num_steps, height, width)``. Accepts NumPy,
        JAX, or ``brainunit`` arrays (converted internally).
    fig : matplotlib.figure.Figure
        Top level container for all plot elements.
    ax : matplotlib.axes.Axes
        Axes that set the coordinate system, e.g.
        ``fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))``.
    num_steps : int, optional
        Number of time steps to plot. When falsy (the default ``False`` or
        ``0``), every entry along the first dimension of ``data`` is used.
    interval : int, optional
        Delay between frames in milliseconds, defaults to ``40``.
    cmap : str, optional
        Color map, defaults to ``"plasma"``.

    Returns
    -------
    anim : matplotlib.animation.ArtistAnimation
        Animation to be displayed (e.g. via ``anim.to_html5_video()``) or saved
        with ``anim.save(...)``.
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
    __module__ = 'braintools.visualize'
    """Make animations easier."""

    def __init__(self, figure: Figure) -> None:
        """Create camera from matplotlib figure."""
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
        """Capture current state of the figure."""
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
        Uses matplotlib.animation.ArtistAnimation
        Returns
        -------
        ArtistAnimation
        """
        return ArtistAnimation(self._figure, self._photos, *args, **kwargs)
