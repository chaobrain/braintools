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

import warnings
from typing import Optional, Union, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from braintools._misc import set_module_as
from braintools.tree import as_numpy

__all__ = [
    'spike_raster',
    'population_activity',
    'connectivity_matrix',
    'neural_trajectory',
    'spike_histogram',
    'isi_distribution',
    'firing_rate_map',
    'phase_portrait',
    'network_topology',
    'tuning_curve',
]


@set_module_as('braintools.visualize')
def spike_raster(
    spike_times: Union[np.ndarray, List],
    neuron_ids: Optional[Union[np.ndarray, List]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    neuron_range: Optional[Tuple[int, int]] = None,
    color: Union[str, np.ndarray] = 'black',
    marker: str = '|',
    markersize: float = 1.0,
    alpha: float = 1.0,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    xlabel: str = 'Time',
    ylabel: str = 'Neuron ID',
    title: Optional[str] = None,
    show_stats: bool = False,
    **kwargs
) -> plt.Axes:
    """
    Create a spike raster plot for neural spike data.
    
    Parameters
    ----------
    spike_times : array-like
        Array of spike times or list of spike time arrays for each neuron.
    neuron_ids : array-like, optional
        Array of neuron IDs corresponding to spike_times. If None, assumes
        spike_times is a list with one array per neuron.
    time_range : tuple, optional
        (start, end) time range to display.
    neuron_range : tuple, optional
        (start, end) neuron ID range to display.
    color : str or array-like
        Color for spikes. Can be a single color or array of colors.
    marker : str
        Marker style for spikes.
    markersize : float
        Size of spike markers.
    alpha : float
        Alpha transparency value.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple
        Figure size if creating new figure.
    xlabel, ylabel, title : str
        Axis labels and title.
    show_stats : bool
        Whether to show firing rate statistics.
    **kwargs
        Additional arguments passed to scatter plot.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy arrays
    spike_times = as_numpy(spike_times)
    if neuron_ids is not None:
        neuron_ids = as_numpy(neuron_ids)

    # Handle different input formats
    if isinstance(spike_times, list):
        # List of arrays, one per neuron
        all_times = []
        all_ids = []
        for i, times in enumerate(spike_times):
            times = as_numpy(times)
            if len(times) > 0:
                all_times.extend(times)
                all_ids.extend([i] * len(times))
        spike_times = np.array(all_times)
        neuron_ids = np.array(all_ids)
    elif neuron_ids is None:
        raise ValueError("neuron_ids must be provided when spike_times is not a list")

    # Apply time filtering
    if time_range is not None:
        mask = (spike_times >= time_range[0]) & (spike_times <= time_range[1])
        spike_times = spike_times[mask]
        neuron_ids = neuron_ids[mask]

    # Apply neuron filtering
    if neuron_range is not None:
        mask = (neuron_ids >= neuron_range[0]) & (neuron_ids <= neuron_range[1])
        spike_times = spike_times[mask]
        neuron_ids = neuron_ids[mask]

    # Plot spikes
    if len(spike_times) > 0:
        ax.scatter(spike_times, neuron_ids, c=color, marker=marker,
                   s=markersize, alpha=alpha, **kwargs)

    # Labels and formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Show statistics
    if show_stats and len(spike_times) > 0:
        n_neurons = len(np.unique(neuron_ids))
        time_span = np.max(spike_times) - np.min(spike_times)
        firing_rate = len(spike_times) / (n_neurons * time_span) if time_span > 0 else 0

        stats_text = f'Neurons: {n_neurons}\nSpikes: {len(spike_times)}\nRate: {firing_rate:.2f} Hz'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.grid(True, alpha=0.3)
    return ax


@set_module_as('braintools.visualize')
def population_activity(
    data: np.ndarray,
    time: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
    method: str = 'mean',
    window_size: Optional[int] = None,
    neuron_ids: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    color: str = 'blue',
    alpha: float = 0.7,
    fill: bool = True,
    xlabel: str = 'Time',
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot population activity over time.
    
    Parameters
    ----------
    data : np.ndarray
        Neural activity data of shape (time, neurons) or (time,).
    time : np.ndarray, optional
        Time array. If None, uses indices.
    dt : float, optional
        Time step. Used if time is None.
    method : str
        Aggregation method: 'mean', 'sum', 'std', 'var', 'median'.
    window_size : int, optional
        Size of sliding window for smoothing.
    neuron_ids : array-like, optional
        Specific neuron IDs to include.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    color : str
        Line color.
    alpha : float
        Alpha transparency.
    fill : bool
        Whether to fill area under curve.
    xlabel, ylabel, title : str
        Axis labels and title.
    **kwargs
        Additional arguments passed to plot.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    data = as_numpy(data)

    # Handle 1D data
    if data.ndim == 1:
        activity = data
    else:
        # Select specific neurons if specified
        if neuron_ids is not None:
            neuron_ids = as_numpy(neuron_ids)
            data = data[:, neuron_ids]

        # Aggregate across neurons
        if method == 'mean':
            activity = np.mean(data, axis=1)
        elif method == 'sum':
            activity = np.sum(data, axis=1)
        elif method == 'std':
            activity = np.std(data, axis=1)
        elif method == 'var':
            activity = np.var(data, axis=1)
        elif method == 'median':
            activity = np.median(data, axis=1)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Create time array
    if time is None:
        if dt is not None:
            time = np.arange(len(activity)) * dt
        else:
            time = np.arange(len(activity))
    else:
        time = as_numpy(time)

    # Apply smoothing if specified
    if window_size is not None and window_size > 1:
        kernel = np.ones(window_size) / window_size
        activity = np.convolve(activity, kernel, mode='same')

    # Plot
    if fill:
        ax.fill_between(time, activity, alpha=alpha, color=color, **kwargs)
    else:
        ax.plot(time, activity, color=color, alpha=alpha, **kwargs)

    # Labels
    ax.set_xlabel(xlabel)
    if ylabel is None:
        ylabel = f'Population {method.capitalize()}'
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)
    return ax


@set_module_as('braintools.visualize')
def connectivity_matrix(
    weights: np.ndarray,
    pre_labels: Optional[List[str]] = None,
    post_labels: Optional[List[str]] = None,
    cmap: str = 'RdBu_r',
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    center_zero: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    show_colorbar: bool = True,
    show_values: bool = False,
    value_threshold: Optional[float] = None,
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Visualize connectivity matrix between neural populations.
    
    Parameters
    ----------
    weights : np.ndarray
        Connectivity weight matrix of shape (pre, post).
    pre_labels, post_labels : list, optional
        Labels for pre and post-synaptic populations.
    cmap : str
        Colormap for the matrix.
    center_zero : bool
        Whether to center colormap at zero.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    show_colorbar : bool
        Whether to show colorbar.
    show_values : bool
        Whether to show values in cells.
    value_threshold : float, optional
        Only show values above this threshold.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to imshow.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    weights = as_numpy(weights)

    # Set up colormap normalization
    if center_zero:
        vmax = np.abs(weights).max()
        vmin = -vmax
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    # Plot matrix
    im = ax.imshow(weights, cmap=cmap, norm=norm, aspect='auto', **kwargs)

    # Add colorbar
    if show_colorbar:
        plt.colorbar(im, ax=ax, label='Weight')

    # Add value labels
    if show_values:
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                value = weights[i, j]
                if value_threshold is None or abs(value) >= value_threshold:
                    text_color = 'white' if abs(value) > 0.5 * np.abs(weights).max() else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                            color=text_color, fontsize=8)

    # Labels
    if pre_labels is not None:
        ax.set_yticks(range(len(pre_labels)))
        ax.set_yticklabels(pre_labels)
    if post_labels is not None:
        ax.set_xticks(range(len(post_labels)))
        ax.set_xticklabels(post_labels)

    ax.set_xlabel('Post-synaptic' if xlabel is None else xlabel)
    ax.set_ylabel('Pre-synaptic' if ylabel is None else ylabel)
    if title:
        ax.set_title(title)

    return ax


@set_module_as('braintools.visualize')
def neural_trajectory(
    data: np.ndarray,
    dims: Optional[Tuple[int, int, int]] = None,
    time_color: bool = True,
    start_marker: str = 'o',
    end_marker: str = 's',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'viridis',
    alpha: float = 0.7,
    linewidth: float = 2.0,
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot neural state trajectory in 2D or 3D space.
    
    Parameters
    ----------
    data : np.ndarray
        Neural state data of shape (time, features).
    dims : tuple, optional
        Dimensions to plot (x, y) or (x, y, z). If None, uses first 2-3 dims.
    time_color : bool
        Whether to color trajectory by time.
    start_marker, end_marker : str
        Markers for start and end points.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    cmap : str
        Colormap for time coloring.
    alpha : float
        Alpha transparency.
    linewidth : float
        Width of trajectory line.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to plot.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    data = as_numpy(data)

    # Determine dimensions
    if dims is None:
        if data.shape[1] >= 3:
            dims = (0, 1, 2)
        else:
            dims = (0, 1)

    is_3d = len(dims) == 3

    if ax is None:
        if is_3d:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': '3d'})
        else:
            fig, ax = plt.subplots(figsize=figsize)

    # Extract coordinates
    if is_3d:
        x, y, z = data[:, dims[0]], data[:, dims[1]], data[:, dims[2]]
        if time_color:
            time_points = np.arange(len(data))
            ax.scatter(x, y, z, c=time_points, cmap=cmap, alpha=alpha, **kwargs)
        else:
            ax.plot(x, y, z, alpha=alpha, linewidth=linewidth, **kwargs)

        # Mark start and end
        ax.scatter(x[0], y[0], z[0], marker=start_marker, s=100, c='green', label='Start')
        ax.scatter(x[-1], y[-1], z[-1], marker=end_marker, s=100, c='red', label='End')

        ax.set_xlabel(f'Dimension {dims[0]}')
        ax.set_ylabel(f'Dimension {dims[1]}')
        ax.set_zlabel(f'Dimension {dims[2]}')
    else:
        x, y = data[:, dims[0]], data[:, dims[1]]
        if time_color:
            time_points = np.arange(len(data))
            scatter = ax.scatter(x, y, c=time_points, cmap=cmap, alpha=alpha, **kwargs)
            plt.colorbar(scatter, ax=ax, label='Time')
        else:
            ax.plot(x, y, alpha=alpha, linewidth=linewidth, **kwargs)

        # Mark start and end
        ax.scatter(x[0], y[0], marker=start_marker, s=100, c='green', label='Start')
        ax.scatter(x[-1], y[-1], marker=end_marker, s=100, c='red', label='End')

        ax.set_xlabel(f'Dimension {dims[0]}')
        ax.set_ylabel(f'Dimension {dims[1]}')

    if title:
        ax.set_title(title)
    ax.legend()

    return ax


@set_module_as('braintools.visualize')
def spike_histogram(
    spike_times: Union[np.ndarray, List],
    bins: Union[int, np.ndarray] = 50,
    time_range: Optional[Tuple[float, float]] = None,
    bin_size: Optional[float] = None,
    density: bool = False,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    color: str = 'blue',
    alpha: float = 0.7,
    xlabel: str = 'Time',
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Create histogram of spike times (PSTH - Peristimulus Time Histogram).
    
    Parameters
    ----------
    spike_times : array-like or list
        Spike times or list of spike time arrays.
    bins : int or array-like
        Number of bins or bin edges.
    time_range : tuple, optional
        (start, end) time range for histogram.
    bin_size : float, optional
        Size of each bin. Alternative to bins parameter.
    density : bool
        Whether to normalize to get density.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    color : str
        Bar color.
    alpha : float
        Alpha transparency.
    xlabel, ylabel, title : str
        Axis labels and title.
    **kwargs
        Additional arguments passed to hist.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Flatten spike times if list of arrays
    if isinstance(spike_times, list):
        all_times = []
        for times in spike_times:
            all_times.extend(as_numpy(times))
        spike_times = np.array(all_times)
    else:
        spike_times = as_numpy(spike_times)

    # Apply time filtering
    if time_range is not None:
        mask = (spike_times >= time_range[0]) & (spike_times <= time_range[1])
        spike_times = spike_times[mask]

    # Set up bins
    if bin_size is not None:
        if time_range is not None:
            bins = np.arange(time_range[0], time_range[1] + bin_size, bin_size)
        else:
            bins = np.arange(spike_times.min(), spike_times.max() + bin_size, bin_size)

    # Create histogram
    counts, bin_edges, patches = ax.hist(spike_times, bins=bins, density=density,
                                         color=color, alpha=alpha, **kwargs)

    # Labels
    ax.set_xlabel(xlabel)
    if ylabel is None:
        ylabel = 'Spike Density' if density else 'Spike Count'
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)
    return ax


@set_module_as('braintools.visualize')
def isi_distribution(
    spike_times: Union[np.ndarray, List],
    bins: Union[int, np.ndarray] = 50,
    max_isi: Optional[float] = None,
    log_scale: bool = False,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    color: str = 'blue',
    alpha: float = 0.7,
    xlabel: str = 'Inter-Spike Interval',
    ylabel: str = 'Count',
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot inter-spike interval (ISI) distribution.
    
    Parameters
    ----------
    spike_times : array-like or list
        Spike times or list of spike time arrays.
    bins : int or array-like
        Number of bins or bin edges.
    max_isi : float, optional
        Maximum ISI to include.
    log_scale : bool
        Whether to use log scale for y-axis.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    color : str
        Bar color.
    alpha : float
        Alpha transparency.
    xlabel, ylabel, title : str
        Axis labels and title.
    **kwargs
        Additional arguments passed to hist.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Calculate ISIs
    all_isis = []
    if isinstance(spike_times, list):
        for times in spike_times:
            times = as_numpy(times)
            if len(times) > 1:
                isis = np.diff(times)
                all_isis.extend(isis)
    else:
        spike_times = as_numpy(spike_times)
        if len(spike_times) > 1:
            all_isis = np.diff(spike_times)

    all_isis = np.array(all_isis)

    # Filter ISIs
    if max_isi is not None:
        all_isis = all_isis[all_isis <= max_isi]

    if len(all_isis) == 0:
        warnings.warn("No inter-spike intervals found")
        return ax

    # Create histogram
    counts, bin_edges, patches = ax.hist(all_isis, bins=bins, color=color,
                                         alpha=alpha, **kwargs)

    # Set log scale if requested
    if log_scale:
        ax.set_yscale('log')

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Add statistics
    mean_isi = np.mean(all_isis)
    cv_isi = np.std(all_isis) / mean_isi if mean_isi > 0 else 0
    stats_text = f'Mean ISI: {mean_isi:.3f}\nCV: {cv_isi:.3f}'
    ax.text(0.7, 0.8, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.grid(True, alpha=0.3)
    return ax


@set_module_as('braintools.visualize')
def firing_rate_map(
    rates: np.ndarray,
    positions: Optional[np.ndarray] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    cmap: str = 'hot',
    interpolation: str = 'bilinear',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    show_colorbar: bool = True,
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Create a 2D firing rate map.
    
    Parameters
    ----------
    rates : np.ndarray
        Firing rates. Can be 2D array or 1D array with positions.
    positions : np.ndarray, optional
        Positions corresponding to rates (N, 2) for 1D rates.
    grid_size : tuple, optional
        Size of the grid for 1D rates with positions.
    cmap : str
        Colormap for the rate map.
    interpolation : str
        Interpolation method for display.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    show_colorbar : bool
        Whether to show colorbar.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to imshow.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    rates = as_numpy(rates)

    if rates.ndim == 2:
        # Already a 2D map
        rate_map = rates
    elif positions is not None:
        # Create 2D map from 1D rates and positions
        positions = as_numpy(positions)
        if grid_size is None:
            grid_size = (50, 50)

        # Create grid
        x_edges = np.linspace(positions[:, 0].min(), positions[:, 0].max(), grid_size[0] + 1)
        y_edges = np.linspace(positions[:, 1].min(), positions[:, 1].max(), grid_size[1] + 1)

        # Bin the data
        rate_map = np.zeros(grid_size)
        count_map = np.zeros(grid_size)

        for i, (x, y) in enumerate(positions):
            xi = np.digitize(x, x_edges) - 1
            yi = np.digitize(y, y_edges) - 1
            if 0 <= xi < grid_size[0] and 0 <= yi < grid_size[1]:
                rate_map[yi, xi] += rates[i]
                count_map[yi, xi] += 1

        # Average rates in each bin
        mask = count_map > 0
        rate_map[mask] /= count_map[mask]
    else:
        raise ValueError("For 1D rates, positions must be provided")

    # Plot rate map
    im = ax.imshow(rate_map, cmap=cmap, interpolation=interpolation,
                   origin='lower', **kwargs)

    # Add colorbar
    if show_colorbar:
        plt.colorbar(im, ax=ax, label='Firing Rate (Hz)')

    # Labels
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    if title:
        ax.set_title(title)

    return ax


@set_module_as('braintools.visualize')
def phase_portrait(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    dx: Optional[np.ndarray] = None,
    dy: Optional[np.ndarray] = None,
    trajectory: bool = True,
    vector_field: bool = False,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    cmap: str = 'viridis',
    alpha: float = 0.7,
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Create a phase portrait plot for dynamical systems.
    
    Parameters
    ----------
    x, y : np.ndarray
        State variables. If y is None, assumes x is 2D with (x, y) columns.
    dx, dy : np.ndarray, optional
        Derivatives for vector field.
    trajectory : bool
        Whether to show trajectory.
    vector_field : bool
        Whether to show vector field.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    cmap : str
        Colormap for trajectory.
    alpha : float
        Alpha transparency.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to plot.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x = as_numpy(x)

    # Handle input format
    if y is None:
        if x.ndim == 2 and x.shape[1] >= 2:
            y = x[:, 1]
            x = x[:, 0]
        else:
            raise ValueError("If y is None, x must be 2D with at least 2 columns")
    else:
        y = as_numpy(y)

    # Plot trajectory
    if trajectory:
        if len(x) > 1:
            time_points = np.arange(len(x))
            scatter = ax.scatter(x, y, c=time_points, cmap=cmap, alpha=alpha, **kwargs)
            plt.colorbar(scatter, ax=ax, label='Time')
        else:
            ax.scatter(x, y, alpha=alpha, **kwargs)

    # Plot vector field
    if vector_field and dx is not None and dy is not None:
        dx = as_numpy(dx)
        dy = as_numpy(dy)
        ax.quiver(x, y, dx, dy, alpha=0.6, scale_units='xy', scale=1)

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)
    return ax


@set_module_as('braintools.visualize')
def network_topology(
    adjacency: np.ndarray,
    positions: Optional[np.ndarray] = None,
    node_colors: Optional[np.ndarray] = None,
    edge_colors: Optional[np.ndarray] = None,
    node_sizes: Optional[np.ndarray] = None,
    edge_widths: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 10),
    layout: str = 'spring',
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Visualize network topology from adjacency matrix.
    
    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix of the network.
    positions : np.ndarray, optional
        Node positions (N, 2). If None, uses layout algorithm.
    node_colors : array-like, optional
        Colors for nodes.
    edge_colors : array-like, optional
        Colors for edges.
    node_sizes : array-like, optional
        Sizes for nodes.
    edge_widths : array-like, optional
        Widths for edges.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    layout : str
        Layout algorithm: 'spring', 'circular', 'random'.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to scatter (for nodes).
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    adjacency = as_numpy(adjacency)
    n_nodes = adjacency.shape[0]

    # Generate positions if not provided
    if positions is None:
        if layout == 'spring':
            # Simple spring layout
            positions = np.random.randn(n_nodes, 2)
            for _ in range(50):  # Simple optimization
                forces = np.zeros_like(positions)
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if i != j:
                            diff = positions[i] - positions[j]
                            dist = np.linalg.norm(diff)
                            if dist > 0:
                                if adjacency[i, j] > 0:
                                    # Attractive force
                                    forces[i] -= 0.01 * diff
                                # Repulsive force
                                forces[i] += 0.1 * diff / (dist ** 3)
                positions += 0.1 * forces
        elif layout == 'circular':
            angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
            positions = np.column_stack([np.cos(angles), np.sin(angles)])
        elif layout == 'random':
            positions = np.random.randn(n_nodes, 2)
        else:
            raise ValueError(f"Unknown layout: {layout}")
    else:
        positions = as_numpy(positions)

    # Set default values
    if node_colors is None:
        node_colors = 'blue'
    if edge_colors is None:
        edge_colors = 'gray'
    if node_sizes is None:
        node_sizes = 100
    if edge_widths is None:
        edge_widths = 1.0

    # Draw edges
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adjacency[i, j] > 0:
                x_coords = [positions[i, 0], positions[j, 0]]
                y_coords = [positions[i, 1], positions[j, 1]]

                # Determine edge properties
                if hasattr(edge_colors, '__len__') and not isinstance(edge_colors, str):
                    edge_color = edge_colors[i * n_nodes + j] if len(edge_colors) > i * n_nodes + j else edge_colors[0]
                else:
                    edge_color = edge_colors

                if hasattr(edge_widths, '__len__'):
                    edge_width = edge_widths[i * n_nodes + j] if len(edge_widths) > i * n_nodes + j else edge_widths[0]
                else:
                    edge_width = edge_widths

                ax.plot(x_coords, y_coords, color=edge_color, linewidth=edge_width, alpha=0.6)

    # Draw nodes
    ax.scatter(positions[:, 0], positions[:, 1], c=node_colors, s=node_sizes,
               zorder=10, **kwargs)

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if title:
        ax.set_title(title)

    ax.axis('equal')
    return ax


@set_module_as('braintools.visualize')
def tuning_curve(
    stimulus: np.ndarray,
    response: np.ndarray,
    bins: Union[int, np.ndarray] = 20,
    error_bars: bool = True,
    fit_curve: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    color: str = 'blue',
    alpha: float = 0.7,
    xlabel: str = 'Stimulus',
    ylabel: str = 'Response',
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot tuning curve showing response vs stimulus.
    
    Parameters
    ----------
    stimulus : np.ndarray
        Stimulus values.
    response : np.ndarray
        Neural response values.
    bins : int or array-like
        Number of bins or bin edges for averaging.
    error_bars : bool
        Whether to show error bars (SEM).
    fit_curve : str, optional
        Type of curve to fit: 'gaussian', 'polynomial'.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    color : str
        Plot color.
    alpha : float
        Alpha transparency.
    xlabel, ylabel, title : str
        Axis labels and title.
    **kwargs
        Additional arguments passed to plot.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    stimulus = as_numpy(stimulus)
    response = as_numpy(response)

    # Create bins
    if isinstance(bins, int):
        bin_edges = np.linspace(stimulus.min(), stimulus.max(), bins + 1)
    else:
        bin_edges = as_numpy(bins)

    # Calculate binned statistics
    bin_centers = []
    mean_responses = []
    sem_responses = []

    for i in range(len(bin_edges) - 1):
        mask = (stimulus >= bin_edges[i]) & (stimulus < bin_edges[i + 1])
        if np.any(mask):
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            responses_in_bin = response[mask]
            mean_responses.append(np.mean(responses_in_bin))
            sem_responses.append(np.std(responses_in_bin) / np.sqrt(len(responses_in_bin)))

    bin_centers = np.array(bin_centers)
    mean_responses = np.array(mean_responses)
    sem_responses = np.array(sem_responses)

    # Plot tuning curve
    if error_bars:
        ax.errorbar(bin_centers, mean_responses, yerr=sem_responses,
                    color=color, alpha=alpha, capsize=5, **kwargs)
    else:
        ax.plot(bin_centers, mean_responses, color=color, alpha=alpha, **kwargs)

    # Fit curve if requested
    if fit_curve == 'gaussian' and len(bin_centers) > 3:
        from scipy.optimize import curve_fit

        def gaussian(x, amp, mu, sigma, baseline):
            return baseline + amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        try:
            popt, _ = curve_fit(gaussian, bin_centers, mean_responses)
            x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 100)
            y_fit = gaussian(x_fit, *popt)
            ax.plot(x_fit, y_fit, '--', color='red', alpha=0.8, label='Gaussian fit')
            ax.legend()
        except:
            warnings.warn("Could not fit Gaussian curve")

    elif fit_curve == 'polynomial' and len(bin_centers) > 2:
        try:
            coeffs = np.polyfit(bin_centers, mean_responses, 3)
            x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 100)
            y_fit = np.polyval(coeffs, x_fit)
            ax.plot(x_fit, y_fit, '--', color='red', alpha=0.8, label='Polynomial fit')
            ax.legend()
        except:
            warnings.warn("Could not fit polynomial curve")

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)
    return ax
