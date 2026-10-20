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

from typing import Any, Optional, Union, Tuple, List, cast

import numpy as np

from braintools._misc import set_module_as
from ._neural import _edge_attribute, NEGATIVE_EDGE_COLOR
from ..tree import as_numpy
from ._utils import as_array, flatten_spikes, inter_spike_intervals, is_spike_trains, spike_trains_events

# Try to import plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None

__all__ = [
    'interactive_spike_raster',
    'interactive_line_plot',
    'interactive_heatmap',
    'interactive_3d_scatter',
    'interactive_network',
    'interactive_histogram',
    'interactive_surface',
    'interactive_correlation_matrix',
    'dashboard_neural_activity',
]


@set_module_as('braintools.visualize')
def _check_plotly():
    """Check if plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive plots. "
            "Install with: pip install braintools[viz] (or pip install plotly)"
        )


@set_module_as('braintools.visualize')
def interactive_spike_raster(
    spike_times: Union[np.ndarray, List],
    neuron_ids: Optional[Union[np.ndarray, List]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    neuron_range: Optional[Tuple[int, int]] = None,
    color_by: Optional[str] = None,
    title: str = "Interactive Spike Raster",
    width: int = 800,
    height: int = 600,
    **kwargs
):
    """
    Create interactive spike raster plot using Plotly.
    
    Parameters
    ----------
    spike_times : array-like, list or SpikeTrains
        Array of spike times, list of spike time arrays for each neuron, or a
        :class:`braintools.spike.SpikeTrains` (times are plotted in its unit).
    neuron_ids : array-like, optional
        Array of neuron IDs corresponding to spike_times. Must be None for a
        ``SpikeTrains``.
    time_range : tuple, optional
        (start, end) time range to display.
    neuron_range : tuple, optional
        (start, end) neuron ID range to display.
    color_by : str, optional
        Color spikes by: 'neuron', 'time', or None.
    title : str
        Plot title.
    width, height : int
        Figure dimensions.
    **kwargs
        Additional arguments passed to Plotly scatter.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure.
    """
    _check_plotly()

    xaxis_title = "Time"
    if is_spike_trains(spike_times):
        spike_times, neuron_ids, unit = spike_trains_events(spike_times, neuron_ids)
        if unit is not None:
            xaxis_title = f"Time ({unit})"
    else:
        # A list holds one spike train per neuron; a flat array needs neuron_ids.
        if neuron_ids is None and not isinstance(spike_times, (list, tuple)):
            raise ValueError("neuron_ids must be provided when spike_times is not a list")
        spike_times, neuron_ids = flatten_spikes(spike_times, neuron_ids)
    spike_times = cast(np.ndarray, spike_times)
    neuron_ids = cast(np.ndarray, neuron_ids)

    # Apply filtering
    if time_range is not None:
        mask = (spike_times >= time_range[0]) & (spike_times <= time_range[1])
        spike_times = cast(np.ndarray, spike_times[mask])
        neuron_ids = cast(np.ndarray, neuron_ids[mask])

    if neuron_range is not None:
        mask = (neuron_ids >= neuron_range[0]) & (neuron_ids <= neuron_range[1])
        spike_times = cast(np.ndarray, spike_times[mask])
        neuron_ids = cast(np.ndarray, neuron_ids[mask])

    # Set up colors
    color = None
    if color_by == 'neuron':
        color = neuron_ids
        colorscale = 'Viridis'
    elif color_by == 'time':
        color = spike_times
        colorscale = 'Plasma'
    else:
        colorscale = None

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=spike_times,
        y=neuron_ids,
        mode='markers',
        marker=dict(
            color=color,
            colorscale=colorscale,
            size=3,
            symbol='line-ns-open'
        ),
        name='Spikes',
        hovertemplate='Time: %{x}<br>Neuron: %{y}<extra></extra>',
        **kwargs
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Neuron ID",
        width=width,
        height=height,
        hovermode='closest'
    )

    return fig


@set_module_as('braintools.visualize')
def interactive_line_plot(
    x: np.ndarray,
    y: Union[np.ndarray, List[np.ndarray]],
    labels: Optional[List[str]] = None,
    title: str = "Interactive Line Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    width: int = 800,
    height: int = 600,
    **kwargs
):
    """
    Create interactive line plot using Plotly.
    
    Parameters
    ----------
    x : np.ndarray
        X-axis data.
    y : array-like or list of arrays
        Y-axis data. Can be multiple traces.
    labels : list, optional
        Labels for each trace.
    title : str
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    width, height : int
        Figure dimensions.
    **kwargs
        Additional arguments passed to Plotly scatter.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure.
    """
    _check_plotly()

    x = as_array(x)

    # Ensure y is a list
    if not isinstance(y, list):
        y = [y]

    # Convert to numpy arrays
    y = [as_numpy(trace) for trace in y]

    # Set up labels
    if labels is None:
        labels = [f'Trace {i + 1}' for i in range(len(y))]

    # Create figure
    fig = go.Figure()

    for i, (trace, label) in enumerate(zip(y, labels)):
        fig.add_trace(go.Scatter(
            x=x,
            y=trace,
            mode='lines',
            name=label,
            hovertemplate=f'{label}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
            **kwargs
        ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height,
        hovermode='x unified'
    )

    return fig


@set_module_as('braintools.visualize')
def interactive_heatmap(
    data: np.ndarray,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Interactive Heatmap",
    colorscale: str = 'Viridis',
    width: int = 800,
    height: int = 600,
    **kwargs
):
    """
    Create interactive heatmap using Plotly.
    
    Parameters
    ----------
    data : np.ndarray
        2D data array.
    x_labels, y_labels : list, optional
        Axis labels.
    title : str
        Plot title.
    colorscale : str
        Plotly colorscale name.
    width, height : int
        Figure dimensions.
    **kwargs
        Additional arguments passed to Plotly heatmap.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure.
    """
    _check_plotly()

    data = as_array(data)

    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        hoverongaps=False,
        **kwargs
    ))

    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height
    )

    return fig


@set_module_as('braintools.visualize')
def interactive_3d_scatter(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    color: Optional[np.ndarray] = None,
    size: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    title: str = "Interactive 3D Scatter",
    width: int = 800,
    height: int = 600,
    **kwargs
):
    """
    Create interactive 3D scatter plot using Plotly.
    
    Parameters
    ----------
    x, y, z : np.ndarray
        3D coordinates.
    color : np.ndarray, optional
        Color values for points.
    size : np.ndarray, optional
        Size values for points.
    labels : list, optional
        Labels for hover text.
    title : str
        Plot title.
    width, height : int
        Figure dimensions.
    **kwargs
        Additional arguments passed to Plotly scatter3d.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure.
    """
    _check_plotly()

    x = as_array(x)
    y = as_array(y)
    z = as_array(z)

    if color is not None:
        color = as_array(color)
    if size is not None:
        size = as_array(size)

    # Create figure
    fig = go.Figure(data=go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            color=color,
            size=size if size is not None else 5,
            colorscale='Viridis' if color is not None else None,
            showscale=color is not None
        ),
        text=labels,
        hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>',
        **kwargs
    ))

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        width=width,
        height=height
    )

    return fig


@set_module_as('braintools.visualize')
def interactive_network(
    adjacency: np.ndarray,
    positions: Optional[np.ndarray] = None,
    node_labels: Optional[List[str]] = None,
    node_colors: Optional[np.ndarray] = None,
    edge_weights: Optional[np.ndarray] = None,
    title: str = "Interactive Network",
    width: int = 800,
    height: int = 600,
    **kwargs
):
    """
    Create interactive network visualization using Plotly.
    
    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency or signed weight matrix; every non-zero entry is drawn as an
        edge, positive ones gray and negative ones blue.
    positions : np.ndarray, optional
        Node positions (N, 2).
    node_labels : list, optional
        Node labels.
    node_colors : np.ndarray, optional
        Node colors.
    edge_weights : np.ndarray, optional
        Edge weights, drawn as line widths proportional to ``|weight|`` (up to
        6 px) and shown on hover: an ``(N, N)`` array indexed like
        ``adjacency``, a flattened ``N*N`` array, or one value per edge in
        row-major order. Without it, all edges have width 1.
    title : str
        Plot title.
    width, height : int
        Figure dimensions.
    **kwargs
        Additional arguments passed to Plotly scatter.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure.
    """
    _check_plotly()

    adjacency = as_array(adjacency)
    n_nodes = adjacency.shape[0]

    # Generate positions if not provided
    if positions is None:
        # Simple circular layout
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        positions = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        positions = as_array(positions)

    # Create figure
    fig = go.Figure()

    # Add edges
    edges = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if adjacency[i, j] != 0]
    edge_color = lambda i, j: NEGATIVE_EDGE_COLOR if adjacency[i, j] < 0 else 'gray'
    if edge_weights is None:
        # one trace per sign
        for negative in (False, True):
            edge_x, edge_y = [], []
            for i, j in edges:
                if (adjacency[i, j] < 0) == negative:
                    edge_x.extend([positions[i, 0], positions[j, 0], None])
                    edge_y.extend([positions[i, 1], positions[j, 1], None])
            if edge_x or not negative:
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color=NEGATIVE_EDGE_COLOR if negative else 'gray'),
                    hoverinfo='none',
                    mode='lines',
                    showlegend=False
                ))
    else:
        # One trace per edge: Plotly cannot vary the width within a trace.
        weights = [float(_edge_attribute(edge_weights, i, j, k, n_nodes, len(edges), 'edge_weights'))
                   for k, (i, j) in enumerate(edges)]
        w_max = max((abs(w) for w in weights), default=0.0) or 1.0
        for (i, j), w in zip(edges, weights):
            fig.add_trace(go.Scatter(
                x=[positions[i, 0], positions[j, 0]],
                y=[positions[i, 1], positions[j, 1]],
                line=dict(width=0.5 + 5.5 * abs(w) / w_max, color=edge_color(i, j)),
                mode='lines',
                hovertext=f"Edge: {i} -> {j}<br>Weight: {w:.3f}",
                hoverinfo='text',
                showlegend=False
            ))

    # Add nodes
    node_text = node_labels if node_labels else [f'Node {i}' for i in range(n_nodes)]

    fig.add_trace(go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode='markers+text',
        marker=dict(
            size=20,
            color=node_colors if node_colors is not None else 'lightblue',
            colorscale='Viridis' if node_colors is not None else None,
            showscale=node_colors is not None,
            line=dict(width=2, color='black')
        ),
        text=node_text,
        textposition="middle center",
        hovertemplate='%{text}<extra></extra>',
        **kwargs
    ))

    # Update layout
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=width,
        height=height
    )

    return fig


@set_module_as('braintools.visualize')
def interactive_histogram(
    data: Union[np.ndarray, List[np.ndarray]],
    labels: Optional[List[str]] = None,
    bins: int = 30,
    opacity: float = 0.7,
    title: str = "Interactive Histogram",
    xlabel: str = "Value",
    ylabel: str = "Count",
    width: int = 800,
    height: int = 600,
    **kwargs
):
    """
    Create interactive histogram using Plotly.
    
    Parameters
    ----------
    data : array-like or list of arrays
        Data to plot.
    labels : list, optional
        Labels for each dataset.
    bins : int
        Number of bins, shared by all datasets (common bin edges over the
        pooled data).
    opacity : float
        Histogram opacity.
    title : str
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    width, height : int
        Figure dimensions.
    **kwargs
        Additional arguments passed to Plotly histogram.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure.
    """
    _check_plotly()

    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]

    # Convert to numpy arrays
    data = [np.ravel(as_array(d)) for d in data]

    # Set up labels
    if labels is None:
        labels = [f'Dataset {i + 1}' for i in range(len(data))]

    # Common bins for all datasets: Plotly's ``nbinsx`` is only an upper bound
    # chosen per trace, so overlaid histograms would get different bin widths.
    edges = np.histogram_bin_edges(np.concatenate(data), bins=bins)
    xbins = dict(start=float(edges[0]), end=float(edges[-1]), size=float(edges[1] - edges[0]))

    # Create figure
    fig = go.Figure()

    for i, (d, label) in enumerate(zip(data, labels)):
        fig.add_trace(go.Histogram(
            x=d,
            name=label,
            **{'xbins': xbins, **kwargs},
            opacity=opacity,
        ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        barmode='overlay',
        width=width,
        height=height
    )

    return fig


@set_module_as('braintools.visualize')
def interactive_surface(
    z: np.ndarray,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    colorscale: str = 'Viridis',
    title: str = "Interactive Surface",
    width: int = 800,
    height: int = 600,
    **kwargs
):
    """
    Create interactive 3D surface plot using Plotly.
    
    Parameters
    ----------
    z : np.ndarray
        2D array of surface heights.
    x, y : np.ndarray, optional
        X and Y coordinates.
    colorscale : str
        Plotly colorscale name.
    title : str
        Plot title.
    width, height : int
        Figure dimensions.
    **kwargs
        Additional arguments passed to Plotly surface.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure.
    """
    _check_plotly()

    z = as_array(z)

    if x is not None:
        x = as_array(x)
    if y is not None:
        y = as_array(y)

    # Create figure
    fig = go.Figure(data=go.Surface(
        z=z,
        x=x,
        y=y,
        colorscale=colorscale,
        **kwargs
    ))

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        width=width,
        height=height
    )

    return fig


@set_module_as('braintools.visualize')
def interactive_correlation_matrix(
    data: np.ndarray,
    labels: Optional[List[str]] = None,
    method: str = 'pearson',
    title: str = "Interactive Correlation Matrix",
    width: int = 800,
    height: int = 600,
    **kwargs
):
    """
    Create interactive correlation matrix heatmap using Plotly.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (samples, features).
    labels : list, optional
        Feature labels.
    method : str
        Correlation method: 'pearson', 'spearman'.
    title : str
        Plot title.
    width, height : int
        Figure dimensions.
    **kwargs
        Additional arguments passed to Plotly heatmap.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure.
    """
    _check_plotly()

    data = as_array(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Calculate correlation matrix
    if data.shape[1] == 1:
        # A single feature has the trivial 1x1 correlation matrix; the methods
        # below would return a 0-d scalar that breaks the heatmap.
        corr_matrix = np.array([[1.0]])
    elif method == 'pearson':
        corr_matrix = np.atleast_2d(np.corrcoef(data.T))
    elif method == 'spearman':
        from scipy.stats import spearmanr
        # spearmanr returns a scalar for exactly two columns; build the 2x2
        # matrix explicitly so the heatmap always receives a 2-D array.
        if data.shape[1] == 2:
            rho = spearmanr(data[:, 0], data[:, 1])[0]
            corr_matrix = np.array([[1.0, rho], [rho, 1.0]])
        else:
            corr_matrix = np.atleast_2d(spearmanr(data)[0])
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        text=np.around(corr_matrix, decimals=2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        **kwargs
    ))

    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height
    )

    return fig


@set_module_as('braintools.visualize')
def dashboard_neural_activity(
    spike_times: Union[np.ndarray, List],
    neuron_ids: Optional[Union[np.ndarray, List]] = None,
    population_activity: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    title: str = "Neural Activity Dashboard",
    width: int = 1200,
    height: int = 800
):
    """
    Create comprehensive dashboard for neural activity visualization.
    
    Parameters
    ----------
    spike_times : array-like or list
        Spike times data.
    neuron_ids : array-like, optional
        Neuron IDs corresponding to spike times.
    population_activity : np.ndarray, optional
        Population activity over time.
    time : np.ndarray, optional
        Time array for population activity.
    title : str
        Dashboard title.
    width, height : int
        Figure dimensions.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive dashboard figure.
    """
    _check_plotly()

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Spike Raster', 'ISI Distribution',
                        'Population Activity', 'Firing Rate Histogram',
                        'Spike Count Over Time', 'Statistics'),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "table"}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    # Process spike data
    spike_times_flat, neuron_ids_flat = flatten_spikes(spike_times, neuron_ids)

    # 1. Spike Raster
    fig.add_trace(
        go.Scatter(
            x=spike_times_flat,
            y=neuron_ids_flat,
            mode='markers',
            marker=dict(size=2, symbol='line-ns-open'),
            name='Spikes'
        ),
        row=1, col=1
    )

    # 2. ISI Distribution
    if len(spike_times_flat) > 1:
        # ISIs within each neuron, never across neurons
        all_isis = inter_spike_intervals(spike_times_flat, neuron_ids_flat)

        if len(all_isis) > 0:
            fig.add_trace(
                go.Histogram(x=all_isis, name='ISI', nbinsx=30),
                row=1, col=2
            )

    # 3. Population Activity
    if population_activity is not None:
        population_activity = as_array(population_activity)
        if time is None:
            time = np.arange(len(population_activity))
        else:
            time = as_array(time)

        fig.add_trace(
            go.Scatter(
                x=time,
                y=population_activity,
                mode='lines',
                name='Population Activity'
            ),
            row=2, col=1
        )

    # 4. Firing Rate Histogram. Every neuron's rate is its spike count divided
    # by the span of the whole recording (first to last spike of any neuron);
    # dividing by the neuron's own first-to-last interval inflates the rates of
    # sparsely firing neurons (two spikes 10 ms apart would read as 200 Hz).
    unique_neurons = np.unique(neuron_ids_flat)
    recording_span = (np.max(spike_times_flat) - np.min(spike_times_flat)) if len(spike_times_flat) else 0.0
    firing_rates = [np.sum(neuron_ids_flat == neuron) / recording_span if recording_span > 0 else 0.0
                    for neuron in unique_neurons]

    fig.add_trace(
        go.Histogram(x=firing_rates, name='Firing Rates', nbinsx=20),
        row=2, col=2
    )

    # 5. Spike Count Over Time
    if len(spike_times_flat) > 0:
        bin_size = (np.max(spike_times_flat) - np.min(spike_times_flat)) / 50
        bins = np.arange(np.min(spike_times_flat), np.max(spike_times_flat) + bin_size, bin_size)
        spike_counts, _ = np.histogram(spike_times_flat, bins=bins)
        bin_centers = bins[:-1] + bin_size / 2

        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=spike_counts,
                mode='lines',
                name='Spike Count'
            ),
            row=3, col=1
        )

    # 6. Statistics Table
    n_neurons = len(unique_neurons)
    total_spikes = len(spike_times_flat)
    if len(spike_times_flat) > 0:
        time_span: Any = np.max(spike_times_flat) - np.min(spike_times_flat)
        mean_rate = total_spikes / (n_neurons * time_span) if time_span > 0 else 0
    else:
        mean_rate = 0

    fig.add_trace(
        go.Table(
            header=dict(values=['Statistic', 'Value']),
            cells=dict(values=[
                ['Number of Neurons', 'Total Spikes', 'Mean Firing Rate (Hz)', 'Time Span'],
                [n_neurons, total_spikes, f'{mean_rate:.2f}',
                 f'{time_span:.2f}' if len(spike_times_flat) > 0 else 'N/A']
            ])
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        title=title,
        showlegend=False,
        width=width,
        height=height
    )

    # Update axis labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Neuron ID", row=1, col=1)
    fig.update_xaxes(title_text="ISI", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Activity", row=2, col=1)
    fig.update_xaxes(title_text="Firing Rate (Hz)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Spike Count", row=3, col=1)

    return fig
