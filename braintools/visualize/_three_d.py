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

import warnings

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from braintools._misc import set_module_as
from braintools.tree import as_numpy
from ._utils import as_array

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


def _artist_kwargs(kwargs, **defaults):
    """Merge ``**kwargs`` with the artist arguments a renderer sets itself.

    Every function here documents ``**kwargs`` as "additional arguments passed to
    <artist>", so a caller can reasonably pass e.g. ``cmap=``, ``alpha=`` or
    ``linewidth=``. Expanding the function's own arguments and ``**kwargs``
    separately (``ax.scatter(..., alpha=0.8, **kwargs)``) raises a bare
    ``TypeError: got multiple values for keyword argument 'alpha'`` as soon as the
    two name the same one, so merge them instead. The caller's ``kwargs`` win:
    they were written explicitly, while the values here are just the defaults.
    """
    merged = dict(defaults)
    merged.update(kwargs)
    return merged


@set_module_as('braintools.visualize')
def neural_network_3d(
    layer_sizes: List[int],
    weights: Optional[List[np.ndarray]] = None,
    activations: Optional[List[np.ndarray]] = None,
    layer_spacing: float = 2.0,
    neuron_spacing: float = 1.0,
    node_size: float = 100,
    edge_alpha: float = 0.3,
    cmap: str = 'viridis',
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
        Weight matrices between consecutive layers; ``weights[i]`` has shape
        ``(layer_sizes[i], layer_sizes[i + 1])``. Connections with
        ``|w| > 0.01`` are drawn, positive in red and negative in blue, with
        line width ``5 * |w|``.
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
    cmap : str, optional
        Colormap used to color nodes by ``activations``. Ignored when
        ``activations`` is not given, since there is nothing to map.
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
            ax.scatter(x_coords, y_coords, z_coords,
                       **_artist_kwargs(kwargs, c=activations[i], s=node_size,
                                        alpha=0.8, cmap=cmap))
        else:
            # Nothing to map, so passing ``cmap`` here would only make matplotlib
            # warn that it is being ignored.
            ax.scatter(x_coords, y_coords, z_coords,
                       **{k: v for k, v in
                          _artist_kwargs(kwargs, s=node_size, alpha=0.8).items()
                          if k != 'cmap'})

    # Plot connections
    if weights is not None:
        for i in range(len(weights)):
            if i + 1 < len(neuron_positions):
                weight_matrix = as_array(weights[i])
                pre_positions = neuron_positions[i]
                post_positions = neuron_positions[i + 1]
                expected = (len(pre_positions), len(post_positions))
                if weight_matrix.shape != expected:
                    raise ValueError(
                        f"weights[{i}] must have shape (layer_sizes[{i}], layer_sizes[{i + 1}]) = "
                        f"{expected}, got {weight_matrix.shape}."
                    )

                for j, pre_pos in enumerate(pre_positions):
                    for k, post_pos in enumerate(post_positions):
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

    vertices = as_array(vertices)
    faces = as_array(faces)

    # Create triangular faces
    triangles = vertices[faces]

    # Set up colors
    if values is not None:
        values = as_array(values)
        # Average values for each face
        face_values = np.mean(values[faces], axis=1)
        # Min-max normalize to [0, 1] for the colormap so signed values map
        # correctly (dividing by ``max`` alone pushed negatives outside [0, 1],
        # where the colormap clamps them). Guard against a degenerate (constant)
        # range that would produce NaN colors.
        min_val = np.min(face_values)
        max_val = np.max(face_values)
        span = max_val - min_val
        scaled = (face_values - min_val) / span if span != 0 else np.zeros_like(face_values)
        # ``plt.cm.get_cmap`` was deprecated in matplotlib 3.7 and is removed in
        # 3.11; use the colormap registry instead.
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        facecolors = cmap_obj(scaled)
    else:
        facecolors = None

    # Create 3D surface
    mesh = Poly3DCollection(triangles,
                            **_artist_kwargs(kwargs, alpha=alpha, facecolors=facecolors))
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
    connection_format: str = 'auto',
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
        Connectivity matrix of shape ``(n_source, n_target)`` or a sequence of
        ``(source_idx, target_idx)`` pairs of shape ``(n, 2)``.
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
    connection_format : {'auto', 'matrix', 'pairs'}, default 'auto'
        How to read ``connections``. ``'auto'`` dispatches on the shape. The one
        ambiguous case -- ``n_target == 2`` and exactly ``n_source`` connections,
        so the shape ``(n_source, 2)`` fits both forms -- is resolved by container
        type: a list/tuple (or 1-D object array) is read as pairs, an ndarray as a
        matrix with a warning. Pass ``'matrix'`` or ``'pairs'`` to be explicit.
    **kwargs
        Additional arguments passed to scatter.

    Returns
    -------
    ax : mpl_toolkits.mplot3d.Axes3D
        The 3D axes object.
    """
    if connection_format not in ('auto', 'matrix', 'pairs'):
        raise ValueError(
            f"connection_format must be 'auto', 'matrix' or 'pairs'; "
            f"got {connection_format!r}."
        )
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    source_positions = as_array(source_positions)
    target_positions = as_array(target_positions)
    # ``as_numpy`` is a PyTree utility: it *maps over* a sequence rather than
    # coercing it, so a list of ``(source_idx, target_idx)`` pairs comes back as a
    # nested list of per-pair arrays. Normalise both documented forms to a single
    # numeric array: a sequence (or 1-D object array) of pairs becomes shape
    # ``(n, 2)``; an adjacency matrix is an array already and is left untouched.
    connections = as_numpy(connections)
    # Remember whether the caller handed over a Python sequence: it is the
    # natural container for a pair list, so it decides the ambiguous case below.
    from_sequence = isinstance(connections, (list, tuple)) or (
        isinstance(connections, np.ndarray) and connections.dtype == object)
    if from_sequence:
        connections = np.asarray([list(c) for c in connections], dtype=float)
        if connections.size == 0:
            # An empty pair list is valid (nothing to draw), not a shape error.
            connections = connections.reshape(0, 2)

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
               **_artist_kwargs(kwargs, c=source_colors, s=source_sizes, alpha=0.8,
                                label='Source'))
    ax.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2],
               **_artist_kwargs(kwargs, c=target_colors, s=target_sizes, alpha=0.8,
                                label='Target'))

    # Plot connections
    # Dispatch on the *shape*, not on ``ndim``: a list of connection pairs is also
    # 2-D once coerced to an array, so an ``ndim == 2`` test silently sent every
    # pair list down the adjacency-matrix path (drawing an arbitrary set of edges).
    n_source, n_target = len(source_positions), len(target_positions)
    fits_matrix = connections.ndim == 2 and connections.shape == (n_source, n_target)
    fits_pairs = connections.ndim == 2 and connections.shape[1] == 2
    if connection_format == 'matrix':
        is_matrix = True
        if not fits_matrix:
            raise ValueError(
                f"connection_format='matrix' needs shape ({n_source}, {n_target}); "
                f"got {connections.shape}."
            )
    elif connection_format == 'pairs':
        is_matrix = False
        if not fits_pairs:
            raise ValueError(
                f"connection_format='pairs' needs shape (n, 2); got {connections.shape}."
            )
    else:
        if not (fits_matrix or fits_pairs):
            raise ValueError(
                f"connections must be an adjacency matrix of shape "
                f"({n_source}, {n_target}) or an array of (source_idx, target_idx) "
                f"pairs of shape (n, 2); got shape {connections.shape}."
            )
        if fits_matrix and fits_pairs:
            is_matrix = not from_sequence
            if is_matrix:
                warnings.warn(
                    f"connections of shape {connections.shape} fit both an adjacency "
                    f"matrix and a list of (source_idx, target_idx) pairs; reading it "
                    f"as a matrix. Pass connection_format='matrix' or 'pairs' to "
                    f"silence this warning.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            is_matrix = fits_matrix
    if is_matrix:
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
            i, j = int(conn[0]), int(conn[1])
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

    trajectory = as_array(trajectory)

    if time_colors:
        # Plot with time-based coloring
        time_points = np.arange(len(trajectory))
        scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                             **_artist_kwargs(kwargs, c=time_points, cmap=cmap))
        ax.figure.colorbar(scatter, ax=ax, label='Time')
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
        Voxels with values above the threshold are drawn. If None, uses the
        midpoint of the data range, ``min + (max - min) / 2`` (half of the
        maximum for data with minimum 0).
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

    volume = as_array(volume)

    if threshold is None:
        # Midpoint of the data range: ``max / 2`` would select nothing (or
        # everything) for data that is negative or offset from zero.
        threshold = volume.min() + (volume.max() - volume.min()) / 2

    # Create binary mask
    binary_volume = volume > threshold

    # Color the filled voxels by their normalized intensity using ``cmap``
    # (previously ``cmap`` was documented but never applied).
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    voxel_kwargs = dict(kwargs)
    if 'facecolors' not in voxel_kwargs and bool(binary_volume.any()):
        filled = volume[binary_volume]
        vmin = float(filled.min())
        vmax = float(filled.max())
        span = vmax - vmin
        norm_vol = (volume - vmin) / span if span != 0 else np.zeros_like(volume, dtype=float)
        voxel_kwargs['facecolors'] = cmap_obj(np.clip(norm_vol, 0.0, 1.0))

    # Plot voxels
    ax.voxels(binary_volume, alpha=alpha, **voxel_kwargs)

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
    cmap: str = 'hot',
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
        Scale of the signal-dependent marker size. With ``signals``, the marker
        area is ``10 + 90 * signal_scale * |signal| / max|signal|`` (pass ``s``
        to override it); without ``signals`` it is unused.
    cmap : str, optional
        Colormap used to color electrodes by signal magnitude. Only used when
        ``signals`` is given.
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

    electrode_positions = as_array(electrode_positions)

    # Plot electrodes
    if signals is not None:
        signals = as_array(signals)
        # Use signal magnitude for coloring
        if signals.ndim == 2:
            signal_magnitude = np.linalg.norm(signals, axis=1)
        else:
            signal_magnitude = np.abs(signals)

        max_magnitude = np.max(signal_magnitude)
        relative = signal_magnitude / max_magnitude if max_magnitude > 0 else np.zeros_like(signal_magnitude)
        sizes = 10.0 + 90.0 * signal_scale * relative
        scatter = ax.scatter(electrode_positions[:, 0],
                             electrode_positions[:, 1],
                             electrode_positions[:, 2],
                             **_artist_kwargs(kwargs, c=signal_magnitude, s=sizes,
                                              cmap=cmap))
        # ``ax.figure.colorbar`` rather than ``plt.colorbar``: the latter acts on
        # the *current* figure, so it warns ("Adding colorbar to a different
        # Figure") whenever ``ax`` belongs to a figure that is not current.
        ax.figure.colorbar(scatter, ax=ax, label='Signal Magnitude')
    else:
        # Nothing to color by, so dropping ``cmap`` avoids matplotlib's
        # "No data for colormapping provided via 'c'" warning.
        ax.scatter(electrode_positions[:, 0],
                   electrode_positions[:, 1],
                   electrode_positions[:, 2],
                   **{k: v for k, v in _artist_kwargs(kwargs, s=100).items()
                      if k != 'cmap'})

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
        Diameter of each segment, drawn as a line width of ``10 * diameter``
        points (default width 2); rescale physical diameters accordingly.
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
        start = as_array(start)
        end = as_array(end)

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
                **_artist_kwargs(kwargs, color=color, linewidth=linewidth,
                                 alpha=alpha))

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

    x = as_array(x)
    y = as_array(y)
    z = as_array(z)

    if time_colors:
        # Color by time
        time_points = np.arange(len(x))
        scatter = ax.scatter(x, y, z, **_artist_kwargs(kwargs, c=time_points, cmap=cmap))
        ax.figure.colorbar(scatter, ax=ax, label='Time')
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
