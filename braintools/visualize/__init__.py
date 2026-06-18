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

"""
Visualization Tools for Neural Networks and Scientific Data.

This module provides a comprehensive collection of visualization functions for
neural network analysis, scientific data exploration, and publication-quality
figures. It includes specialized tools for spiking neural networks, statistical
analysis, 3D visualization, and interactive dashboards.

**Key Features:**

- **Neural Visualizations**: Spike rasters, population activity, connectivity matrices
- **Statistical Plots**: Correlation, distribution, regression, and model evaluation
- **3D Graphics**: Network topology, brain surfaces, trajectories, volume rendering
- **Interactive Dashboards**: Plotly-based interactive visualizations
- **Publication Styles**: Pre-configured styles for papers and presentations
- **Animation Support**: Temporal dynamics and neural activity animations
- **Colorblind-Friendly**: Accessible color palettes and schemes

**Quick Start - Spike Raster Plot:**

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from braintools.visualize import spike_raster

    # Simulate spike data
    spike_times = np.random.rand(500) * 1000  # 500 spikes over 1000ms
    neuron_ids = np.random.randint(0, 50, 500)  # 50 neurons

    # Create raster plot
    ax = spike_raster(
        spike_times=spike_times,
        neuron_ids=neuron_ids,
        color='black',
        markersize=2.0,
        show_stats=True
    )
    plt.show()

**Quick Start - Neural Network 3D:**

.. code-block:: python

    import matplotlib.pyplot as plt
    from braintools.visualize import neural_network_3d

    # Define network architecture
    layer_sizes = [784, 256, 128, 10]

    # Visualize network structure (returns the 3D Axes)
    ax = neural_network_3d(
        layer_sizes=layer_sizes,
        layer_spacing=2.0,
        node_size=100,
        edge_alpha=0.3
    )
    plt.show()

**Neural Visualizations:**

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from braintools.visualize import (
        spike_raster, population_activity,
        connectivity_matrix, neural_trajectory,
        spike_histogram, isi_distribution,
        firing_rate_map, phase_portrait,
        network_topology, tuning_curve
    )

    # Spike raster plot
    spike_times = np.random.rand(1000) * 1000
    neuron_ids = np.random.randint(0, 100, 1000)
    spike_raster(spike_times, neuron_ids, show_stats=True)

    # Population activity over time (data is (time, neurons))
    activity = np.random.rand(1000, 100)  # 1000 time steps, 100 neurons
    time = np.arange(1000) * 0.1
    population_activity(activity, time, window_size=10)

    # Connectivity matrix heatmap
    connectivity = np.random.rand(50, 50)
    connectivity_matrix(connectivity, cmap='viridis')

    # Neural trajectory in state space
    trajectory = np.random.randn(1000, 3)  # 3D trajectory
    neural_trajectory(trajectory, time_color=True)

    # Spike histogram (PSTH) from spike times
    psth_times = np.random.rand(500) * 1000
    spike_histogram(psth_times, bin_size=10.0)

    # Inter-spike interval distribution
    isi_values = np.random.exponential(20, 1000)
    isi_distribution(isi_values, bins=50)

    # Firing rate map (spatial)
    rate_map = np.random.rand(20, 20) * 50
    firing_rate_map(rate_map, interpolation='bilinear')

    # Phase portrait of a 2D state trajectory (x and y evolve over time)
    t = np.linspace(0, 20, 500)
    v = np.exp(-0.1 * t) * np.cos(t)
    w = np.exp(-0.1 * t) * np.sin(t)
    phase_portrait(v, w, trajectory=True)

    # Network topology visualization
    adjacency = np.random.rand(30, 30) > 0.8
    network_topology(adjacency, layout='spring')

    # Tuning curve (stimulus response)
    stimuli = np.linspace(0, 360, 100)
    responses = np.cos(np.deg2rad(stimuli - 90)) * 50 + 50
    tuning_curve(stimuli, responses, xlabel='Orientation (deg)')

**Statistical Plots:**

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from braintools.visualize import (
        correlation_matrix, distribution_plot,
        qq_plot, box_plot, violin_plot,
        scatter_matrix, regression_plot,
        residual_plot, confusion_matrix,
        roc_curve, precision_recall_curve,
        learning_curve
    )

    # Correlation matrix
    data = np.random.randn(100, 5)
    correlation_matrix(data, labels=['V1', 'V2', 'V3', 'V4', 'V5'])

    # Distribution plot with histogram and KDE
    samples = np.random.normal(0, 1, 1000)
    distribution_plot(samples, plot_type='both')

    # Q-Q plot for normality test
    qq_plot(samples, distribution='norm')

    # Box plot for multiple groups
    groups = [np.random.normal(i, 1, 100) for i in range(5)]
    box_plot(groups, labels=['G1', 'G2', 'G3', 'G4', 'G5'])

    # Violin plot (box + KDE)
    violin_plot(groups, labels=['G1', 'G2', 'G3', 'G4', 'G5'])

    # Scatter matrix (pairwise relationships)
    scatter_matrix(data, labels=['V1', 'V2', 'V3', 'V4', 'V5'])

    # Regression plot with confidence interval
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.randn(100)
    regression_plot(x, y, fit_line=True, confidence_interval=True)

    # Residual plot for regression diagnostics
    predictions = 2 * x + 1
    residual_plot(y, predictions)

    # Confusion matrix
    y_true = np.random.randint(0, 3, 100)
    y_pred = np.random.randint(0, 3, 100)
    confusion_matrix(y_true, y_pred, labels=['Class A', 'Class B', 'Class C'])

    # ROC curve for binary classification
    y_true_binary = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)
    roc_curve(y_true_binary, y_scores)

    # Precision-Recall curve
    precision_recall_curve(y_true_binary, y_scores)

    # Learning curve (training vs validation)
    train_sizes = [10, 50, 100, 500, 1000]
    train_scores = [0.6, 0.75, 0.8, 0.85, 0.88]
    val_scores = [0.55, 0.7, 0.78, 0.82, 0.83]
    learning_curve(train_sizes, train_scores, val_scores)

**3D Visualizations:**

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from braintools.visualize import (
        neural_network_3d, brain_surface_3d,
        connectivity_3d, trajectory_3d,
        volume_rendering, electrode_array_3d,
        dendrite_tree_3d, phase_space_3d
    )

    # 3D neural network architecture
    layer_sizes = [784, 512, 256, 10]
    neural_network_3d(layer_sizes, layer_spacing=3.0)

    # 3D brain surface rendering
    vertices = np.random.randn(1000, 3)
    faces = np.random.randint(0, 1000, (500, 3))
    brain_surface_3d(vertices, faces, alpha=0.7)

    # 3D connectivity between two sets of nodes (source -> target)
    source_positions = np.random.randn(10, 3)
    target_positions = np.random.randn(10, 3)
    connections = np.random.rand(10, 10) > 0.8
    connectivity_3d(source_positions, target_positions, connections)

    # 3D trajectory in state space
    trajectory = np.random.randn(1000, 3)
    trajectory_3d(trajectory, time_colors=True)

    # Volume rendering (3D activity maps)
    volume = np.random.rand(50, 50, 50)
    volume_rendering(volume, threshold=0.5, alpha=0.3)

    # Electrode array positions with per-electrode signal magnitude
    electrode_positions = np.random.randn(64, 3)
    signals = np.random.randn(64)
    electrode_array_3d(electrode_positions, signals=signals)

    # Dendritic tree as a list of (start_point, end_point) segments
    tree_coords = np.random.randn(100, 3)
    segments = [(tree_coords[i], tree_coords[i + 1]) for i in range(99)]
    dendrite_tree_3d(segments)

    # 3D phase-space trajectory
    t = np.linspace(0, 20, 1000)
    phase_space_3d(np.sin(t), np.cos(t), t / 20, time_colors=True)

**Interactive Visualizations:**

.. code-block:: python

    import numpy as np
    from braintools.visualize import (
        interactive_spike_raster,
        interactive_line_plot,
        interactive_heatmap,
        interactive_3d_scatter,
        interactive_network,
        interactive_histogram,
        interactive_surface,
        interactive_correlation_matrix,
        dashboard_neural_activity
    )

    # Interactive spike raster (zoom, pan, hover)
    spike_times = np.random.rand(1000) * 1000
    neuron_ids = np.random.randint(0, 100, 1000)
    fig = interactive_spike_raster(spike_times, neuron_ids)
    fig.show()

    # Interactive line plot with multiple traces
    time = np.linspace(0, 10, 1000)
    traces = [np.sin(2 * np.pi * f * time) for f in [1, 2, 3]]
    interactive_line_plot(time, traces, labels=['1 Hz', '2 Hz', '3 Hz'])

    # Interactive heatmap
    data = np.random.randn(50, 50)
    interactive_heatmap(data, colorscale='Viridis')

    # Interactive 3D scatter plot
    points = np.random.randn(500, 3)
    colors = np.random.rand(500)
    interactive_3d_scatter(points[:, 0], points[:, 1], points[:, 2], color=colors)

    # Interactive network graph
    adjacency = np.random.rand(30, 30) > 0.8
    interactive_network(adjacency)

    # Interactive histogram with sliders
    samples = np.random.normal(0, 1, 10000)
    interactive_histogram(samples, bins=50)

    # Interactive 3D surface (z is the height grid; x, y are the coordinates)
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    interactive_surface(Z, x=x, y=y)

    # Interactive correlation matrix
    data = np.random.randn(100, 10)
    interactive_correlation_matrix(data)

    # Complete neural activity dashboard (flat spike times + neuron ids)
    spike_times = np.random.rand(2000) * 1000
    neuron_ids = np.random.randint(0, 100, 2000)
    dashboard_neural_activity(spike_times, neuron_ids)

**Styling and Colormaps:**

.. code-block:: python

    import matplotlib.pyplot as plt
    from braintools.visualize import (
        neural_style, publication_style,
        dark_style, colorblind_friendly_style,
        create_neural_colormap, brain_colormaps,
        apply_style, get_color_palette,
        set_default_colors
    )

    # Apply neural-specific style
    neural_style(
        spike_color='#FF6B6B',
        membrane_color='#96CEB4',
        grid=True
    )

    # Publication-ready style (high DPI, professional fonts)
    publication_style(
        fontsize=10,
        figsize=(6, 4),
        dpi=300
    )

    # Dark mode for presentations
    dark_style(
        background_color='#1e1e1e',
        text_color='white'
    )

    # Colorblind-friendly palette
    colorblind_friendly_style()

    # Create custom neural colormap
    cmap = create_neural_colormap(
        colors=['blue', 'white', 'red'],
        name='neural_diverging'
    )

    # Access brain-specific colormaps
    cmaps = brain_colormaps()
    # Available: 'spike', 'membrane', 'calcium', 'inhibitory', 'excitatory'

    # Apply style context manager
    with apply_style('publication'):
        plt.plot([1, 2, 3], [1, 4, 9])
        plt.show()

    # Get color palette for categorical data
    colors = get_color_palette('neural', n_colors=5)

    # Override default neural element colors (takes a single mapping)
    set_default_colors({'excitatory': '#45B7D1', 'inhibitory': '#FF6B6B'})

**Animation:**

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from braintools.visualize import animator, animate_2D, animate_1D
    from IPython.display import HTML

    # Animate spiking activity over time
    spike_data = np.random.rand(100, 28, 28)  # 100 frames of 28x28 activity
    fig, ax = plt.subplots()
    anim = animator(spike_data, fig, ax, interval=50, cmap='plasma')

    # Save as GIF
    anim.save('spike_activity.gif', writer='pillow')

    # Display in Jupyter
    HTML(anim.to_html5_video())

    # 2D animation: values are (num_steps, num_neurons), reshaped to net_size
    data_2d = np.random.rand(100, 2500)  # 100 steps, 2500 neurons
    animate_2D(data_2d, net_size=(50, 50), frame_delay=40,
               val_min=0, val_max=1, show=False)

    # 1D animation (line plot evolving over time)
    data_1d = np.random.randn(100, 200)
    animate_1D(data_1d, frame_delay=30, xlim=(0, 200), ylim=(-3, 3), show=False)

**Figure Utilities:**

.. code-block:: python

    import matplotlib.pyplot as plt
    from braintools.visualize import get_figure, remove_axis

    # Create multi-panel figure
    fig, gs = get_figure(row_num=2, col_num=3, row_len=4, col_len=6)

    # Add subplots
    ax1 = fig.add_subplot(gs[0, :])  # Top row, all columns
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom left
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom middle
    ax4 = fig.add_subplot(gs[1, 2])  # Bottom right

    # Plot data
    ax1.plot([1, 2, 3], [1, 4, 9])
    ax2.scatter([1, 2, 3], [1, 2, 3])

    # Remove axis from decorative panels
    remove_axis(ax3)
    remove_axis(ax4)

    plt.show()

**Basic Plots:**

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from braintools.visualize import line_plot, raster_plot

    # Line plot with multiple traces
    time = np.linspace(0, 100, 1000)
    voltages = np.random.randn(1000, 5) + np.arange(5)
    ax = line_plot(
        ts=time,
        val_matrix=voltages,
        plot_ids=[0, 1, 2],  # Plot first 3 neurons
        xlabel='Time (ms)',
        ylabel='Voltage (mV)',
        legend='Neuron',
        colors=['r', 'g', 'b']
    )

    # Raster plot from a (time, neuron) spike matrix
    ts = np.arange(1000) * 0.1                      # 1000 time points
    sp_matrix = np.random.rand(1000, 50) < 0.01     # sparse spikes, 50 neurons
    raster_plot(
        ts,
        sp_matrix,
        xlim=(0, 100),
        xlabel='Time (ms)',
        ylabel='Neuron'
    )

**Complete Example - SNN Analysis:**

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from braintools.visualize import (
        get_figure, spike_raster, population_activity,
        isi_distribution, firing_rate_map,
        neural_style, apply_style
    )

    # Generate simulated SNN data
    n_neurons = 100
    n_timesteps = 1000
    dt = 0.1  # ms

    spike_times = []
    neuron_ids = []
    for i in range(n_neurons):
        # Poisson spiking
        n_spikes = np.random.poisson(20)
        times = np.sort(np.random.rand(n_spikes) * n_timesteps * dt)
        spike_times.extend(times)
        neuron_ids.extend([i] * n_spikes)

    spike_times = np.array(spike_times)
    neuron_ids = np.array(neuron_ids)

    # Create figure with neural style
    with apply_style('neural'):
        fig, gs = get_figure(row_num=2, col_num=2, row_len=4, col_len=6)

        # Spike raster
        ax1 = fig.add_subplot(gs[0, :])
        spike_raster(spike_times, neuron_ids, ax=ax1, show_stats=True)
        ax1.set_title('Spike Raster Plot')

        # Population activity
        ax2 = fig.add_subplot(gs[1, 0])
        # Convert spikes to rate
        time_bins = np.arange(0, n_timesteps * dt, 1.0)
        rates, _ = np.histogram(spike_times, bins=time_bins)
        # ``rates`` is already a 1D population rate per time bin
        population_activity(rates, time_bins[:-1], ax=ax2)
        ax2.set_title('Population Rate')

        # ISI distribution
        ax3 = fig.add_subplot(gs[1, 1])
        isis = np.diff(np.sort(spike_times))
        isi_distribution(isis, bins=50, ax=ax3)
        ax3.set_title('ISI Distribution')

        plt.tight_layout()
        plt.savefig('snn_analysis.png', dpi=300)
        plt.show()

**Integration with BrainTools:**

.. code-block:: python

    import brainstate as bst
    import numpy as np
    from braintools.visualize import spike_raster, line_plot

    # Simulate LIF network
    class LIFNetwork(bst.nn.Module):
        def __init__(self, n_neurons):
            super().__init__()
            self.n_neurons = n_neurons
            self.v = bst.State(np.zeros(n_neurons) - 65)
            self.spikes = []
            self.times = []

        def __call__(self, I_ext, t):
            dv = (-self.v.value + I_ext) / 20.0
            self.v.value = self.v.value + dv * 0.1

            # Spike detection
            spike_mask = self.v.value > -50
            self.v.value = np.where(spike_mask, -65, self.v.value)

            # Record spikes
            if np.any(spike_mask):
                spike_ids = np.where(spike_mask)[0]
                self.spikes.extend(spike_ids)
                self.times.extend([t] * len(spike_ids))

            return spike_mask

    # Run simulation
    net = LIFNetwork(100)
    for t in range(1000):
        I_ext = np.random.randn(100) * 10
        net(I_ext, t * 0.1)

    # Visualize results
    spike_raster(
        spike_times=np.array(net.times),
        neuron_ids=np.array(net.spikes),
        xlabel='Time (ms)',
        show_stats=True
    )

"""

# Animation
from ._animation import (
    animator,
)

# Figures
from ._figures import (
    get_figure,
)

# Colormaps and styles
from ._colormaps import (
    neural_style,
    publication_style,
    dark_style,
    colorblind_friendly_style,
    create_neural_colormap,
    brain_colormaps,
    apply_style,
    get_color_palette,
    set_default_colors,
)

# Basic plots
from ._plots import (
    line_plot,
    raster_plot,
    animate_2D,
    animate_1D,
    remove_axis,
)

# Neural visualizations
from ._neural import (
    spike_raster,
    population_activity,
    connectivity_matrix,
    neural_trajectory,
    spike_histogram,
    isi_distribution,
    firing_rate_map,
    phase_portrait,
    network_topology,
    tuning_curve,
)

# Statistical plots
from ._statistical import (
    correlation_matrix,
    distribution_plot,
    qq_plot,
    box_plot,
    violin_plot,
    scatter_matrix,
    regression_plot,
    residual_plot,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    learning_curve,
)

# 3D visualizations
from ._three_d import (
    neural_network_3d,
    brain_surface_3d,
    connectivity_3d,
    trajectory_3d,
    volume_rendering,
    electrode_array_3d,
    dendrite_tree_3d,
    phase_space_3d,
)

# Interactive visualizations
from ._interactive import (
    interactive_spike_raster,
    interactive_line_plot,
    interactive_heatmap,
    interactive_3d_scatter,
    interactive_network,
    interactive_histogram,
    interactive_surface,
    interactive_correlation_matrix,
    dashboard_neural_activity,
)

__all__ = [
    # Animation
    'animator',

    # Figures
    'get_figure',

    # Colormaps and styles
    'neural_style',
    'publication_style',
    'dark_style',
    'colorblind_friendly_style',
    'create_neural_colormap',
    'brain_colormaps',
    'apply_style',
    'get_color_palette',
    'set_default_colors',

    # Basic plots
    'line_plot',
    'raster_plot',
    'animate_2D',
    'animate_1D',
    'remove_axis',

    # Neural visualizations
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

    # Statistical plots
    'correlation_matrix',
    'distribution_plot',
    'qq_plot',
    'box_plot',
    'violin_plot',
    'scatter_matrix',
    'regression_plot',
    'residual_plot',
    'confusion_matrix',
    'roc_curve',
    'precision_recall_curve',
    'learning_curve',

    # 3D visualizations
    'neural_network_3d',
    'brain_surface_3d',
    'connectivity_3d',
    'trajectory_3d',
    'volume_rendering',
    'electrode_array_3d',
    'dendrite_tree_3d',
    'phase_space_3d',

    # Interactive visualizations
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
