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

from typing import Optional, Union, Tuple, List

import numpy as np

from braintools._misc import set_module_as
from ..tree import as_numpy

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
        raise ImportError("Plotly is required for interactive plots. Install with: pip install plotly")


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
    spike_times : array-like or list
        Array of spike times or list of spike time arrays for each neuron.
    neuron_ids : array-like, optional
        Array of neuron IDs corresponding to spike_times.
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

    # Apply filtering
    if time_range is not None:
        mask = (spike_times >= time_range[0]) & (spike_times <= time_range[1])
        spike_times = spike_times[mask]
        neuron_ids = neuron_ids[mask]

    if neuron_range is not None:
        mask = (neuron_ids >= neuron_range[0]) & (neuron_ids <= neuron_range[1])
        spike_times = spike_times[mask]
        neuron_ids = neuron_ids[mask]

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
        xaxis_title="Time",
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

    x = as_numpy(x)

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

    data = as_numpy(data)

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

    x = as_numpy(x)
    y = as_numpy(y)
    z = as_numpy(z)

    if color is not None:
        color = as_numpy(color)
    if size is not None:
        size = as_numpy(size)

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
        Adjacency matrix.
    positions : np.ndarray, optional
        Node positions (N, 2).
    node_labels : list, optional
        Node labels.
    node_colors : np.ndarray, optional
        Node colors.
    edge_weights : np.ndarray, optional
        Edge weights for thickness.
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

    adjacency = as_numpy(adjacency)
    n_nodes = adjacency.shape[0]

    # Generate positions if not provided
    if positions is None:
        # Simple circular layout
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        positions = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        positions = as_numpy(positions)

    # Create figure
    fig = go.Figure()

    # Add edges
    edge_x = []
    edge_y = []
    edge_info = []

    for i in range(n_nodes):
        for j in range(n_nodes):
            if adjacency[i, j] > 0:
                x0, y0 = positions[i]
                x1, y1 = positions[j]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_info.append(f"Edge: {i} -> {j}<br>Weight: {adjacency[i, j]:.3f}")

    # Add edges trace
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines',
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
        Number of bins.
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
    data = [as_numpy(d) for d in data]

    # Set up labels
    if labels is None:
        labels = [f'Dataset {i + 1}' for i in range(len(data))]

    # Create figure
    fig = go.Figure()

    for i, (d, label) in enumerate(zip(data, labels)):
        fig.add_trace(go.Histogram(
            x=d,
            name=label,
            nbinsx=bins,
            opacity=opacity,
            **kwargs
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

    z = as_numpy(z)

    if x is not None:
        x = as_numpy(x)
    if y is not None:
        y = as_numpy(y)

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

    data = as_numpy(data)

    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = np.corrcoef(data.T)
    elif method == 'spearman':
        from scipy.stats import spearmanr
        corr_matrix = spearmanr(data)[0]
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
    if isinstance(spike_times, list):
        all_times = []
        all_ids = []
        for i, times in enumerate(spike_times):
            times = as_numpy(times)
            if len(times) > 0:
                all_times.extend(times)
                all_ids.extend([i] * len(times))
        spike_times_flat = np.array(all_times)
        neuron_ids_flat = np.array(all_ids)
    else:
        spike_times_flat = as_numpy(spike_times)
        neuron_ids_flat = as_numpy(neuron_ids) if neuron_ids is not None else np.zeros_like(spike_times_flat)

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
        if isinstance(spike_times, list):
            all_isis = []
            for times in spike_times:
                times = as_numpy(times)
                if len(times) > 1:
                    all_isis.extend(np.diff(times))
        else:
            all_isis = np.diff(spike_times_flat)

        if len(all_isis) > 0:
            fig.add_trace(
                go.Histogram(x=all_isis, name='ISI', nbinsx=30),
                row=1, col=2
            )

    # 3. Population Activity
    if population_activity is not None:
        population_activity = as_numpy(population_activity)
        if time is None:
            time = np.arange(len(population_activity))
        else:
            time = as_numpy(time)

        fig.add_trace(
            go.Scatter(
                x=time,
                y=population_activity,
                mode='lines',
                name='Population Activity'
            ),
            row=2, col=1
        )

    # 4. Firing Rate Histogram
    unique_neurons = np.unique(neuron_ids_flat)
    firing_rates = []
    for neuron in unique_neurons:
        neuron_spikes = spike_times_flat[neuron_ids_flat == neuron]
        if len(neuron_spikes) > 1:
            time_span = np.max(neuron_spikes) - np.min(neuron_spikes)
            rate = len(neuron_spikes) / time_span if time_span > 0 else 0
        else:
            rate = 0
        firing_rates.append(rate)

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
        time_span = np.max(spike_times_flat) - np.min(spike_times_flat)
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
