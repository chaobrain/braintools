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

import logging

import brainstate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.gridspec import GridSpec

from braintools._misc import set_module_as

__all__ = [
    'line_plot',
    'raster_plot',
    'animate_2D',
    'animate_1D',
    'remove_axis',
]


@set_module_as('braintools.visualize')
def line_plot(
    ts,
    val_matrix,
    plot_ids=None,
    ax=None,
    xlim=None,
    ylim=None,
    xlabel='Time (ms)',
    ylabel=None,
    legend=None,
    title=None,
    show=False,
    colors=None,
    alpha=1.0,
    linewidth=1.0,
    **kwargs
):
    """Show the specified value in the given object (Neurons or Synapses.)

    Parameters
    ----------
    ts : np.ndarray
        The time steps.
    val_matrix : np.ndarray
        The value matrix which record the history trajectory.
        It can be easily accessed by specifying the ``monitors``
        of NeuGroup/SynConn by:
        ``neu/syn = NeuGroup/SynConn(..., monitors=[k1, k2])``
    plot_ids : None, int, tuple, a_list
        The index of the value to plot.
    ax : None, Axes
        The figure to plot.
    xlim : list, tuple
        The xlim.
    ylim : list, tuple
        The ylim.
    xlabel : str
        The xlabel.
    ylabel : str
        The ylabel.
    legend : str
        The prefix of legend for plot.
    show : bool
        Whether show the figure.
    colors : list, optional
        Colors for each line.
    alpha : float
        Alpha transparency value.
    linewidth : float
        Width of lines.
    **kwargs
        Additional keyword arguments passed to plot().
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    
    Raises
    ------
    TypeError
        If plot_ids is not the correct type.
    ValueError
        If val_matrix is empty or incompatible dimensions.
    """
    # Input validation
    if val_matrix is None or len(val_matrix) == 0:
        raise ValueError("val_matrix cannot be empty")

    # get plot_ids
    if plot_ids is None:
        plot_ids = [0]
    elif isinstance(plot_ids, int):
        plot_ids = [plot_ids]
    if not isinstance(plot_ids, (list, tuple)) and \
        not (isinstance(plot_ids, np.ndarray) and np.ndim(plot_ids) == 1):
        raise TypeError(f'"plot_ids" specifies the value index to plot, it must '
                        f'be a list/tuple/1D numpy.ndarray, not {type(plot_ids)}.')

    # get ax
    if ax is None:
        ax = plt

    val_matrix = val_matrix.reshape((val_matrix.shape[0], -1))
    # change data
    val_matrix = np.asarray(val_matrix)
    ts = np.asarray(ts)

    # Validate dimensions
    if len(ts) != val_matrix.shape[0]:
        raise ValueError(f"Time array length ({len(ts)}) must match val_matrix first dimension ({val_matrix.shape[0]})")

    # Check plot_ids are valid
    max_idx = val_matrix.shape[1] - 1
    invalid_ids = [idx for idx in plot_ids if idx < 0 or idx > max_idx]
    if invalid_ids:
        raise ValueError(f"Invalid plot_ids {invalid_ids}. Must be between 0 and {max_idx}")

    # Set up colors
    if colors is not None:
        if len(colors) < len(plot_ids):
            # Cycle colors if not enough provided
            colors = (colors * (len(plot_ids) // len(colors) + 1))[:len(plot_ids)]

    # plot
    for i, idx in enumerate(plot_ids):
        plot_kwargs = kwargs.copy()
        plot_kwargs['alpha'] = alpha
        plot_kwargs['linewidth'] = linewidth

        if colors is not None:
            plot_kwargs['color'] = colors[i]

        if legend:
            label = legend if len(plot_ids) == 1 else f'{legend}-{idx}'
            plot_kwargs['label'] = label

        ax.plot(ts, val_matrix[:, idx], **plot_kwargs)

    # legend
    if legend:
        ax.legend()

    # xlim
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    # ylim
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    # xlabel
    if xlabel:
        plt.xlabel(xlabel)

    # ylabel
    if ylabel:
        plt.ylabel(ylabel)

    # title
    if title:
        plt.title(title)

    # show
    if show:
        plt.show()

    return ax


@set_module_as('braintools.visualize')
def raster_plot(
    ts,
    sp_matrix,
    ax=None,
    marker='.',
    markersize=2,
    color='k',
    xlabel='Time (ms)',
    ylabel='Neuron index',
    xlim=None,
    ylim=None,
    title=None,
    show=False,
    alpha=1.0,
    **kwargs
):
    """Show the raster plot of the spikes.

    Parameters
    ----------
    ts : np.ndarray
        The run times.
    sp_matrix : np.ndarray
        The spike matrix which records the spike information.
        It can be easily accessed by specifying the ``monitors``
        of NeuGroup by: ``neu = NeuGroup(..., monitors=['spike'])``
    ax : Axes, optional
        The figure axes. If None, uses plt.
    marker : str
        The marker style.
    markersize : int
        The size of the marker.
    color : str
        The color of the marker.
    xlim : list, tuple, optional
        The xlim.
    ylim : list, tuple, optional
        The ylim.
    xlabel : str
        The xlabel.
    ylabel : str
        The ylabel.
    title : str, optional
        The plot title.
    show : bool
        Show the figure.
    alpha : float
        Alpha transparency value.
    **kwargs
        Additional keyword arguments passed to scatter().
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
        
    Raises
    ------
    ValueError
        If ts is None or sp_matrix is empty.
    """

    # Input validation
    if ts is None:
        raise ValueError('Must provide "ts".')
    if sp_matrix is None or len(sp_matrix) == 0:
        raise ValueError('sp_matrix cannot be empty.')

    sp_matrix = np.asarray(sp_matrix)
    ts = np.asarray(ts)

    # Validate dimensions
    if len(ts) != sp_matrix.shape[0]:
        raise ValueError(f"Time array length ({len(ts)}) must match sp_matrix first dimension ({sp_matrix.shape[0]})")

    # get index and time
    elements = np.where(sp_matrix > 0.)
    index = elements[1]
    time = ts[elements[0]]

    # plot raster
    if ax is None:
        ax = plt

    scatter_kwargs = kwargs.copy()
    scatter_kwargs['alpha'] = alpha
    ax.scatter(time, index, marker=marker, c=color, s=markersize, **scatter_kwargs)

    # xlable
    if xlabel:
        plt.xlabel(xlabel)

    # ylabel
    if ylabel:
        plt.ylabel(ylabel)

    if xlim:
        plt.xlim(xlim[0], xlim[1])

    if ylim:
        plt.ylim(ylim[0], ylim[1])

    if title:
        plt.title(title)

    if show:
        plt.show()

    return ax


@set_module_as('braintools.visualize')
def animate_2D(
    values,
    net_size,
    dt=None,
    val_min=None,
    val_max=None,
    cmap=None,
    frame_delay=10,
    frame_step=1,
    title_size=10,
    figsize=None,
    gif_dpi=None,
    video_fps=None,
    save_path=None,
    show=True,
    repeat=True,
    **kwargs
):
    """Animate the potentials of the neuron group.

    Parameters
    ----------
    values : np.ndarray
        The membrane potentials of the neuron group.
    net_size : tuple
        The size of the neuron group.
    dt : float
        The time duration of each step.
    val_min : float, int
        The minimum of the potential.
    val_max : float, int
        The maximum of the potential.
    cmap : str
        The colormap.
    frame_delay : int, float
        The delay to show each frame.
    frame_step : int
        The step to show the potential. If `frame_step=3`, then each
        frame shows one of the every three steps.
    title_size : int
        The size of the title.
    figsize : None, tuple
        The size of the figure.
    gif_dpi : int
        Controls the dots per inch for the movie frames. This combined with
        the figure's size in inches controls the size of the movie. If
        ``None``, use defaults in matplotlib.
    video_fps : int
        Frames per second in the movie. Defaults to ``None``, which will use
        the animation's specified interval to set the frames per second.
    save_path : None, str
        The save path of the animation.
    show : bool
        Whether show the animation.

    Returns
    -------
    anim : animation.FuncAnimation
        The created animation function.
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    num_step, num_neuron = values.shape
    height, width = net_size

    values = np.asarray(values)
    val_min = values.min() if val_min is None else val_min
    val_max = values.max() if val_max is None else val_max

    figsize = figsize or (6, 6)

    fig = plt.figure(figsize=(figsize[0], figsize[1]), constrained_layout=True)
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    img = values[0]
    mesh = ax.pcolor(img, cmap=cmap, vmin=val_min, vmax=val_max)
    cbar = fig.colorbar(mesh, ax=ax)
    ax.axis('off')
    title = fig.suptitle("Time: {:.2f} ms".format(1 * dt),
                         fontsize=title_size,
                         fontweight='bold')

    def frame(t):
        img = values[t]
        mesh.set_array(img.ravel())
        mesh.set_clim(vmin=val_min, vmax=val_max)
        title.set_text("Time: {:.2f} ms".format((t + 1) * dt))
        return [mesh, title]

    values = values.reshape((num_step, height, width))
    anim = animation.FuncAnimation(fig=fig,
                                   func=frame,
                                   frames=list(range(1, num_step, frame_step)),
                                   init_func=None,
                                   interval=frame_delay,
                                   repeat_delay=3000)
    if save_path is None:
        if show:
            plt.show()
    else:
        logging.warning(f'Saving the animation into {save_path} ...')
        if save_path[-3:] == 'gif':
            anim.save(save_path, dpi=gif_dpi, writer='imagemagick')
        elif save_path[-3:] == 'mp4':
            anim.save(save_path, writer='ffmpeg', fps=video_fps, bitrate=3000)
        else:
            anim.save(save_path + '.mp4', writer='ffmpeg', fps=video_fps, bitrate=3000)
    return anim


@set_module_as('braintools.visualize')
def animate_1D(
    dynamical_vars,
    static_vars=(),
    dt=None,
    xlim=None,
    ylim=None,
    xlabel=None,
    ylabel=None,
    frame_delay=50.,
    frame_step=1,
    title_size=10,
    figsize=None,
    gif_dpi=None,
    video_fps=None,
    save_path=None,
    show=True,
    **kwargs
):
    """Animation of one-dimensional data.

    Parameters
    ----------
    dynamical_vars : dict, np.ndarray, list of np.ndarray, list of dict
        The dynamical variables which will be animated.
    static_vars : dict, np.ndarray, list of np.ndarray, list of dict
        The static variables.
    xticks : list, np.ndarray
        The xticks.
    dt : float
        The numerical integration step.
    xlim : tuple
        The xlim.
    ylim : tuple
        The ylim.
    xlabel : str
        The xlabel.
    ylabel : str
        The ylabel.
    frame_delay : int, float
        The delay to show each frame.
    frame_step : int
        The step to show the potential. If `frame_step=3`, then each
        frame shows one of the every three steps.
    title_size : int
        The size of the title.
    figsize : None, tuple
        The size of the figure.
    gif_dpi : int
        Controls the dots per inch for the movie frames. This combined with
        the figure's size in inches controls the size of the movie. If
        ``None``, use defaults in matplotlib.
    video_fps : int
        Frames per second in the movie. Defaults to ``None``, which will use
        the animation's specified interval to set the frames per second.
    save_path : None, str
        The save path of the animation.
    show : bool
        Whether show the animation.

    Returns
    -------
    figure : plt.figure
        The created figure instance.
    """

    # check dt
    dt = brainstate.environ.get_dt() if dt is None else dt

    # check figure
    fig = plt.figure(figsize=(figsize or (6, 6)), constrained_layout=True)
    gs = GridSpec(1, 1, figure=fig)
    fig.add_subplot(gs[0, 0])

    # check dynamical variables
    final_dynamic_vars = []
    lengths = []
    has_legend = False
    if isinstance(dynamical_vars, (tuple, list)):
        for var in dynamical_vars:
            if isinstance(var, dict):
                assert 'ys' in var, 'Must provide "ys" item.'
                if 'legend' not in var:
                    var['legend'] = None
                else:
                    has_legend = True
                var['ys'] = np.asarray(var['ys'])
                if 'xs' not in var:
                    var['xs'] = np.arange(var['ys'].shape[1])
            elif isinstance(var, (np.ndarray, brainstate.State)):
                var = np.asarray(var)
                var = {'ys': var,
                       'xs': np.arange(var.shape[1]),
                       'legend': None}
            else:
                raise ValueError(f'Unknown data type: {type(var)}')
            assert np.ndim(var['ys']) == 2, "Dynamic variable must be 2D data."
            lengths.append(var['ys'].shape[0])
            final_dynamic_vars.append(var)
    elif isinstance(dynamical_vars, dict):
        assert 'ys' in dynamical_vars, 'Must provide "ys" item.'
        if 'legend' not in dynamical_vars:
            dynamical_vars['legend'] = None
        else:
            has_legend = True
        dynamical_vars['ys'] = np.asarray(dynamical_vars['ys'])
        if 'xs' not in dynamical_vars:
            dynamical_vars['xs'] = np.arange(dynamical_vars['ys'].shape[1])
        lengths.append(dynamical_vars['ys'].shape[0])
        final_dynamic_vars.append(dynamical_vars)
    else:
        assert np.ndim(dynamical_vars) == 2, "Dynamic variable must be 2D data."
        dynamical_vars = np.asarray(dynamical_vars)
        lengths.append(dynamical_vars.shape[0])
        final_dynamic_vars.append({'ys': dynamical_vars,
                                   'xs': np.arange(dynamical_vars.shape[1]),
                                   'legend': None})
    lengths = np.array(lengths)
    assert np.all(lengths == lengths[0]), 'Dynamic variables must have equal length.'

    # check static variables
    final_static_vars = []
    if isinstance(static_vars, (tuple, list)):
        for var in static_vars:
            if isinstance(var, dict):
                assert 'data' in var, 'Must provide "ys" item.'
                if 'legend' not in var:
                    var['legend'] = None
                else:
                    has_legend = True
            elif isinstance(var, np.ndarray):
                var = {'data': var, 'legend': None}
            else:
                raise ValueError(f'Unknown data type: {type(var)}')
            assert np.ndim(var['data']) == 1, "Static variable must be 1D data."
            final_static_vars.append(var)
    elif isinstance(static_vars, np.ndarray):
        final_static_vars.append({'data': static_vars,
                                  'xs': np.arange(static_vars.shape[0]),
                                  'legend': None})
    elif isinstance(static_vars, dict):
        assert 'ys' in static_vars, 'Must provide "ys" item.'
        if 'legend' not in static_vars:
            static_vars['legend'] = None
        else:
            has_legend = True
        if 'xs' not in static_vars:
            static_vars['xs'] = np.arange(static_vars['ys'].shape[0])
        final_static_vars.append(static_vars)

    else:
        raise ValueError(f'Unknown static data type: {type(static_vars)}')

    # ylim
    if ylim is None:
        ylim_min = np.inf
        ylim_max = -np.inf
        for var in final_dynamic_vars + final_static_vars:
            if var['ys'].max() > ylim_max:
                ylim_max = var['ys'].max()
            if var['ys'].min() < ylim_min:
                ylim_min = var['ys'].min()
        if ylim_min > 0:
            ylim_min = ylim_min * 0.98
        else:
            ylim_min = ylim_min * 1.02
        if ylim_max > 0:
            ylim_max = ylim_max * 1.02
        else:
            ylim_max = ylim_max * 0.98
        ylim = (ylim_min, ylim_max)

    def frame(t):
        fig.clf()
        for dvar in final_dynamic_vars:
            plt.plot(dvar['xs'], dvar['ys'][t], label=dvar['legend'], **kwargs)
        for svar in final_static_vars:
            plt.plot(svar['xs'], svar['ys'], label=svar['legend'], **kwargs)
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        if has_legend:
            plt.legend()
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.ylim(ylim[0], ylim[1])
        fig.suptitle(t="Time: {:.2f} ms".format((t + 1) * dt),
                     fontsize=title_size,
                     fontweight='bold')
        return [fig.gca()]

    anim_result = animation.FuncAnimation(fig=fig,
                                          func=frame,
                                          frames=range(1, lengths[0], frame_step),
                                          init_func=None,
                                          interval=frame_delay,
                                          repeat_delay=3000)

    # save or show
    if save_path is None:
        if show: plt.show()
    else:
        logging.warning(f'Saving the animation into {save_path} ...')
        if save_path[-3:] == 'gif':
            anim_result.save(save_path, dpi=gif_dpi, writer='imagemagick')
        elif save_path[-3:] == 'mp4':
            anim_result.save(save_path, writer='ffmpeg', fps=video_fps, bitrate=3000)
        else:
            anim_result.save(save_path + '.mp4', writer='ffmpeg', fps=video_fps, bitrate=3000)
    return fig


@set_module_as('braintools.visualize')
def remove_axis(ax, *pos):
    for p in pos:
        if p not in ['left', 'right', 'top', 'bottom']:
            raise ValueError
        ax.spine[p].set_visible(False)
