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

from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from braintools._misc import set_module_as
from braintools.tree import as_numpy

__all__ = [
    'neural_network_3d',
    'brain_surface_3d',
    'connectivity_3d',
    'trajectory_3d',
    'volume_rendering',
    'electrode_array_3d',
    'dendrite_tree_3d',
    'phase_space_3d',
]


@set_module_as('braintools.visualize')
def neural_network_3d(
    layer_sizes: List[int],
    weights: Optional[List[np.ndarray]] = None,
    activations: Optional[List[np.ndarray]] = None,
    layer_spacing: float = 2.0,
    neuron_spacing: float = 1.0,
    node_size: float = 100,
    edge_alpha: float = 0.3,
    ax: Optional[Axes3D] = None,
    figsize: Tuple[float, float] = (12, 10),
    title: Optional[str] = None,
    **kwargs
) -> Axes3D:
    """
    Visualize neural network architecture in 3D.
    
    Parameters
    ----------
    layer_sizes : list
        Number of neurons in each layer.
    weights : list of arrays, optional
        Weight matrices between layers.
    activations : list of arrays, optional
        Neuron activations for coloring.
    layer_spacing : float
        Spacing between layers.
    neuron_spacing : float
        Spacing between neurons in a layer.
    node_size : float
        Size of neuron nodes.
    edge_alpha : float
        Alpha transparency for connections.
    ax : Axes3D, optional
        3D axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to scatter.
        
    Returns
    -------
    ax : mpl_toolkits.mplot3d.Axes3D
        The 3D axes object.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    n_layers = len(layer_sizes)

    # Generate neuron positions
    neuron_positions = []
    layer_positions = np.linspace(0, (n_layers - 1) * layer_spacing, n_layers)

    for i, (layer_size, layer_z) in enumerate(zip(layer_sizes, layer_positions)):
        if layer_size == 1:
            # Single neuron at center
            positions = [(0, 0, layer_z)]
        else:
            # Arrange neurons in a circle
            angles = np.linspace(0, 2 * np.pi, layer_size, endpoint=False)
            radius = max(1, layer_size * neuron_spacing / (2 * np.pi))
            positions = [(radius * np.cos(angle), radius * np.sin(angle), layer_z)
                         for angle in angles]
        neuron_positions.append(positions)

    # Plot neurons
    for i, (positions, layer_size) in enumerate(zip(neuron_positions, layer_sizes)):
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        z_coords = [pos[2] for pos in positions]

        # Color by activation if provided
        if activations is not None and i < len(activations):
            colors = activations[i]
            scatter = ax.scatter(x_coords, y_coords, z_coords,
                                 c=colors, s=node_size, alpha=0.8,
                                 cmap='viridis', **kwargs)
        else:
            ax.scatter(x_coords, y_coords, z_coords,
                       s=node_size, alpha=0.8, **kwargs)

    # Plot connections
    if weights is not None:
        for i in range(len(weights)):
            if i + 1 < len(neuron_positions):
                weight_matrix = as_numpy(weights[i])
                pre_positions = neuron_positions[i]
                post_positions = neuron_positions[i + 1]

                for j, pre_pos in enumerate(pre_positions):
                    for k, post_pos in enumerate(post_positions):
                        if j < weight_matrix.shape[0] and k < weight_matrix.shape[1]:
                            weight = weight_matrix[j, k]
                            if abs(weight) > 0.01:  # Only draw significant connections
                                line_width = abs(weight) * 5
                                color = 'red' if weight > 0 else 'blue'
                                ax.plot([pre_pos[0], post_pos[0]],
                                        [pre_pos[1], post_pos[1]],
                                        [pre_pos[2], post_pos[2]],
                                        color=color, alpha=edge_alpha,
                                        linewidth=line_width)

    # Labels and formatting
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Layer')
    if title:
        ax.set_title(title)

    return ax


@set_module_as('braintools.visualize')
def brain_surface_3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    values: Optional[np.ndarray] = None,
    cmap: str = 'viridis',
    alpha: float = 0.8,
    ax: Optional[Axes3D] = None,
    figsize: Tuple[float, float] = (12, 10),
    title: Optional[str] = None,
    **kwargs
) -> Axes3D:
    """
    Visualize brain surface mesh in 3D.
    
    Parameters
    ----------
    vertices : np.ndarray
        Vertex coordinates of shape (n_vertices, 3).
    faces : np.ndarray
        Face indices of shape (n_faces, 3).
    values : np.ndarray, optional
        Values to color the surface.
    cmap : str
        Colormap for surface coloring.
    alpha : float
        Surface transparency.
    ax : Axes3D, optional
        3D axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to Poly3DCollection.
        
    Returns
    -------
    ax : mpl_toolkits.mplot3d.Axes3D
        The 3D axes object.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    vertices = as_numpy(vertices)
    faces = as_numpy(faces)

    # Create triangular faces
    triangles = vertices[faces]

    # Set up colors
    if values is not None:
        values = as_numpy(values)
        # Average values for each face
        face_values = np.mean(values[faces], axis=1)
        facecolors = plt.cm.get_cmap(cmap)(face_values / np.max(face_values))
    else:
        facecolors = None

    # Create 3D surface
    mesh = Poly3DCollection(triangles, alpha=alpha, facecolors=facecolors, **kwargs)
    ax.add_collection3d(mesh)

    # Set axis limits
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)

    return ax


@set_module_as('braintools.visualize')
def connectivity_3d(
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    connections: np.ndarray,
    connection_strengths: Optional[np.ndarray] = None,
    node_colors: Optional[np.ndarray] = None,
    node_sizes: Optional[np.ndarray] = None,
    edge_alpha: float = 0.6,
    ax: Optional[Axes3D] = None,
    figsize: Tuple[float, float] = (12, 10),
    title: Optional[str] = None,
    **kwargs
) -> Axes3D:
    """
    Visualize 3D connectivity between neural populations.
    
    Parameters
    ----------
    source_positions : np.ndarray
        3D positions of source nodes.
    target_positions : np.ndarray
        3D positions of target nodes.
    connections : np.ndarray
        Connectivity matrix or list of connections.
    connection_strengths : np.ndarray, optional
        Strength of each connection for line thickness.
    node_colors : np.ndarray, optional
        Colors for nodes.
    node_sizes : np.ndarray, optional
        Sizes for nodes.
    edge_alpha : float
        Alpha transparency for connections.
    ax : Axes3D, optional
        3D axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to scatter.
        
    Returns
    -------
    ax : mpl_toolkits.mplot3d.Axes3D
        The 3D axes object.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    source_positions = as_numpy(source_positions)
    target_positions = as_numpy(target_positions)
    connections = as_numpy(connections)

    # Plot source nodes
    if node_colors is not None:
        source_colors = node_colors[:len(source_positions)]
        target_colors = node_colors[len(source_positions):len(source_positions) + len(target_positions)]
    else:
        source_colors = 'red'
        target_colors = 'blue'

    if node_sizes is not None:
        source_sizes = node_sizes[:len(source_positions)]
        target_sizes = node_sizes[len(source_positions):len(source_positions) + len(target_positions)]
    else:
        source_sizes = 100
        target_sizes = 100

    ax.scatter(source_positions[:, 0], source_positions[:, 1], source_positions[:, 2],
               c=source_colors, s=source_sizes, alpha=0.8, label='Source', **kwargs)
    ax.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2],
               c=target_colors, s=target_sizes, alpha=0.8, label='Target', **kwargs)

    # Plot connections
    if connections.ndim == 2:
        # Adjacency matrix format
        for i in range(connections.shape[0]):
            for j in range(connections.shape[1]):
                if connections[i, j] > 0 and i < len(source_positions) and j < len(target_positions):
                    start = source_positions[i]
                    end = target_positions[j]

                    if connection_strengths is not None:
                        line_width = connection_strengths[i, j] * 5
                    else:
                        line_width = 1

                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                            'k-', alpha=edge_alpha, linewidth=line_width)
    else:
        # List of connections format [(source_idx, target_idx), ...]
        for conn in connections:
            i, j = conn[0], conn[1]
            if i < len(source_positions) and j < len(target_positions):
                start = source_positions[i]
                end = target_positions[j]
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                        'k-', alpha=edge_alpha)

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    if title:
        ax.set_title(title)

    return ax


@set_module_as('braintools.visualize')
def trajectory_3d(
    trajectory: np.ndarray,
    time_colors: bool = True,
    start_marker: str = 'o',
    end_marker: str = 's',
    cmap: str = 'viridis',
    ax: Optional[Axes3D] = None,
    figsize: Tuple[float, float] = (12, 10),
    title: Optional[str] = None,
    **kwargs
) -> Axes3D:
    """
    Visualize 3D trajectory of neural state evolution.
    
    Parameters
    ----------
    trajectory : np.ndarray
        3D trajectory of shape (time_steps, 3).
    time_colors : bool
        Whether to color trajectory by time.
    start_marker, end_marker : str
        Markers for start and end points.
    cmap : str
        Colormap for time coloring.
    ax : Axes3D, optional
        3D axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to plot.
        
    Returns
    -------
    ax : mpl_toolkits.mplot3d.Axes3D
        The 3D axes object.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    trajectory = as_numpy(trajectory)

    if time_colors:
        # Plot with time-based coloring
        time_points = np.arange(len(trajectory))
        scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                             c=time_points, cmap=cmap, **kwargs)
        plt.colorbar(scatter, ax=ax, label='Time')
    else:
        # Plot as simple line
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], **kwargs)

    # Mark start and end points
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
               marker=start_marker, s=200, c='green', label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
               marker=end_marker, s=200, c='red', label='End')

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    if title:
        ax.set_title(title)

    return ax


@set_module_as('braintools.visualize')
def volume_rendering(
    volume: np.ndarray,
    threshold: Optional[float] = None,
    alpha: float = 0.3,
    cmap: str = 'viridis',
    ax: Optional[Axes3D] = None,
    figsize: Tuple[float, float] = (12, 10),
    title: Optional[str] = None,
    **kwargs
) -> Axes3D:
    """
    Simple volume rendering using isosurfaces.
    
    Parameters
    ----------
    volume : np.ndarray
        3D volume data.
    threshold : float, optional
        Threshold for isosurface. If None, uses half of max value.
    alpha : float
        Surface transparency.
    cmap : str
        Colormap for surface.
    ax : Axes3D, optional
        3D axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to voxels.
        
    Returns
    -------
    ax : mpl_toolkits.mplot3d.Axes3D
        The 3D axes object.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    volume = as_numpy(volume)

    if threshold is None:
        threshold = volume.max() / 2

    # Create binary mask
    binary_volume = volume > threshold

    # Plot voxels
    ax.voxels(binary_volume, alpha=alpha, **kwargs)

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)

    return ax


@set_module_as('braintools.visualize')
def electrode_array_3d(
    electrode_positions: np.ndarray,
    signals: Optional[np.ndarray] = None,
    electrode_labels: Optional[List[str]] = None,
    signal_scale: float = 1.0,
    ax: Optional[Axes3D] = None,
    figsize: Tuple[float, float] = (12, 10),
    title: Optional[str] = None,
    **kwargs
) -> Axes3D:
    """
    Visualize 3D electrode array with optional signal data.
    
    Parameters
    ----------
    electrode_positions : np.ndarray
        3D positions of electrodes.
    signals : np.ndarray, optional
        Signal data for each electrode.
    electrode_labels : list, optional
        Labels for each electrode.
    signal_scale : float
        Scaling factor for signal visualization.
    ax : Axes3D, optional
        3D axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to scatter.
        
    Returns
    -------
    ax : mpl_toolkits.mplot3d.Axes3D
        The 3D axes object.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    electrode_positions = as_numpy(electrode_positions)

    # Plot electrodes
    if signals is not None:
        signals = as_numpy(signals)
        # Use signal magnitude for coloring
        if signals.ndim == 2:
            signal_magnitude = np.linalg.norm(signals, axis=1)
        else:
            signal_magnitude = np.abs(signals)

        scatter = ax.scatter(electrode_positions[:, 0],
                             electrode_positions[:, 1],
                             electrode_positions[:, 2],
                             c=signal_magnitude, s=100, cmap='hot', **kwargs)
        plt.colorbar(scatter, ax=ax, label='Signal Magnitude')
    else:
        ax.scatter(electrode_positions[:, 0],
                   electrode_positions[:, 1],
                   electrode_positions[:, 2],
                   s=100, **kwargs)

    # Add electrode labels
    if electrode_labels is not None:
        for i, (pos, label) in enumerate(zip(electrode_positions, electrode_labels)):
            ax.text(pos[0], pos[1], pos[2], label, fontsize=8)

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)

    return ax


@set_module_as('braintools.visualize')
def dendrite_tree_3d(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    diameters: Optional[List[float]] = None,
    colors: Optional[List[str]] = None,
    alpha: float = 0.8,
    ax: Optional[Axes3D] = None,
    figsize: Tuple[float, float] = (12, 10),
    title: Optional[str] = None,
    **kwargs
) -> Axes3D:
    """
    Visualize dendritic tree structure in 3D.
    
    Parameters
    ----------
    segments : list of tuples
        List of (start_point, end_point) for each dendritic segment.
    diameters : list, optional
        Diameter of each segment for line thickness.
    colors : list, optional
        Colors for each segment.
    alpha : float
        Line transparency.
    ax : Axes3D, optional
        3D axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to plot.
        
    Returns
    -------
    ax : mpl_toolkits.mplot3d.Axes3D
        The 3D axes object.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    # Plot each segment
    for i, (start, end) in enumerate(segments):
        start = as_numpy(start)
        end = as_numpy(end)

        # Determine line properties
        if diameters is not None:
            linewidth = diameters[i] * 10
        else:
            linewidth = 2

        if colors is not None:
            color = colors[i]
        else:
            color = 'brown'

        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color=color, linewidth=linewidth, alpha=alpha, **kwargs)

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)

    return ax


@set_module_as('braintools.visualize')
def phase_space_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    time_colors: bool = True,
    cmap: str = 'viridis',
    ax: Optional[Axes3D] = None,
    figsize: Tuple[float, float] = (12, 10),
    title: Optional[str] = None,
    **kwargs
) -> Axes3D:
    """
    Visualize 3D phase space trajectory.
    
    Parameters
    ----------
    x, y, z : np.ndarray
        State variables.
    time_colors : bool
        Whether to color by time.
    cmap : str
        Colormap for time coloring.
    ax : Axes3D, optional
        3D axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to plot.
        
    Returns
    -------
    ax : mpl_toolkits.mplot3d.Axes3D
        The 3D axes object.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    x = as_numpy(x)
    y = as_numpy(y)
    z = as_numpy(z)

    if time_colors:
        # Color by time
        time_points = np.arange(len(x))
        scatter = ax.scatter(x, y, z, c=time_points, cmap=cmap, **kwargs)
        plt.colorbar(scatter, ax=ax, label='Time')
    else:
        # Simple line plot
        ax.plot(x, y, z, **kwargs)

    # Mark start and end
    ax.scatter(x[0], y[0], z[0], marker='o', s=100, c='green', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], marker='s', s=100, c='red', label='End')

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    if title:
        ax.set_title(title)

    return ax
