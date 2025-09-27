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

from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from braintools._misc import set_module_as

__all__ = [
    'neural_style',
    'publication_style',
    'dark_style',
    'colorblind_friendly_style',
    'create_neural_colormap',
    'brain_colormaps',
    'apply_style',
    'get_color_palette',
    'set_default_colors',
]

# Neural-specific color palettes
NEURAL_COLORS = {
    'spike': '#FF6B6B',
    'inhibitory': '#4ECDC4',
    'excitatory': '#45B7D1',
    'background': '#F7F7F7',
    'membrane': '#96CEB4',
    'synapse': '#FFEAA7',
    'dendrite': '#DDA0DD',
    'axon': '#98D8C8'
}

COLORBLIND_PALETTE = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf'  # cyan
]


@set_module_as('braintools.visualize')
def neural_style(
    spike_color: str = '#FF6B6B',
    membrane_color: str = '#96CEB4',
    background_color: str = '#F7F7F7',
    fontsize: int = 12,
    grid: bool = True
):
    """Apply neural-specific plotting style.

    This function configures matplotlib with colors and styles optimized for
    neural data visualization. It sets up a cohesive visual theme that enhances
    readability and interpretation of neuroscience plots.

    Parameters
    ----------
    spike_color : str, default='#FF6B6B'
        Color for spike representations and event markers.
    membrane_color : str, default='#96CEB4'
        Color for membrane potential plots and neural activity.
    background_color : str, default='#F7F7F7'
        Background color for figures and axes.
    fontsize : int, default=12
        Base font size for labels and text.
    grid : bool, default=True
        Whether to show grid lines for better readability.

    Notes
    -----
    This style is designed specifically for neural data visualization with:

    - High contrast colors for clear spike visualization
    - Soft background colors to reduce eye strain
    - Appropriate font sizes for scientific figures
    - Grid lines for easier data interpretation

    The style affects all subsequent matplotlib plots until reset or changed.

    Examples
    --------
    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import braintools as bt

        # Apply neural style
        bt.visualize.neural_style()

        # Create example neural data plot
        time = np.linspace(0, 100, 1000)
        membrane_potential = -65 + 10 * np.sin(time * 0.1) + np.random.randn(1000) * 2

        plt.figure(figsize=(10, 6))
        plt.plot(time, membrane_potential)
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.title('Neural Membrane Potential')
        plt.show()

        # Customize colors
        bt.visualize.neural_style(
            spike_color='#FF3333',
            membrane_color='#33AA33',
            background_color='#FFFFFF',
            fontsize=14
        )

        # Create spike raster with custom style
        spike_times = np.random.uniform(0, 100, 50)
        neuron_ids = np.random.randint(0, 10, 50)

        plt.figure(figsize=(10, 6))
        plt.scatter(spike_times, neuron_ids, c='#FF3333', marker='|', s=50)
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron ID')
        plt.title('Spike Raster Plot')
        plt.show()
    """
    params = {
        'figure.facecolor': background_color,
        'axes.facecolor': background_color,
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 1.2,
        'axes.grid': grid,
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        'xtick.labelsize': fontsize - 1,
        'ytick.labelsize': fontsize - 1,
        'axes.labelsize': fontsize + 1,
        'axes.titlesize': fontsize + 2,
        'legend.fontsize': fontsize,
        'font.size': fontsize,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'patch.linewidth': 0.5,
        'patch.facecolor': membrane_color,
        'patch.edgecolor': '#EEEEEE',
        'patch.antialiased': True,
        'text.color': '#333333',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333'
    }
    rcParams.update(params)

    # Set default color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', [spike_color, membrane_color, NEURAL_COLORS['excitatory'],
                  NEURAL_COLORS['inhibitory'], NEURAL_COLORS['synapse']]
    )


@set_module_as('braintools.visualize')
def publication_style(
    fontsize: int = 10,
    figsize: Tuple[float, float] = (6, 4),
    dpi: int = 300,
    usetex: bool = False
):
    """Apply publication-ready style.

    This function configures matplotlib with settings optimized for high-quality
    publication figures. It ensures consistent, professional appearance suitable
    for scientific journals and presentations.

    Parameters
    ----------
    fontsize : int, default=10
        Base font size in points. Typically 8-12 for journal figures.
    figsize : tuple of float, default=(6, 4)
        Default figure size (width, height) in inches.
    dpi : int, default=300
        Dots per inch for figure resolution. 300+ recommended for print.
    usetex : bool, default=False
        Whether to use LaTeX for text rendering. Requires LaTeX installation.

    Notes
    -----
    Publication style features:

    - High DPI for crisp figures
    - Conservative font sizes
    - Clean, minimal design
    - PDF output format for vector graphics
    - Tight bounding boxes to minimize whitespace
    - Serif fonts for professional appearance

    This style is ideal for:

    - Journal manuscripts
    - Conference presentations
    - Thesis figures
    - Grant applications

    Examples
    --------
    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import braintools as bt

        # Apply publication style
        bt.visualize.publication_style()

        # Create publication-quality figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

        # Panel A: Membrane potential
        time = np.linspace(0, 50, 500)
        v_mem = -65 + 15 * np.exp(-time/10) * np.sin(time)
        ax1.plot(time, v_mem, 'k-', linewidth=1.5)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Membrane Potential (mV)')
        ax1.set_title('A', fontweight='bold', loc='left')

        # Panel B: Firing rate
        rates = np.random.exponential(2, 100)
        ax2.hist(rates, bins=20, color='gray', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Firing Rate (Hz)')
        ax2.set_ylabel('Count')
        ax2.set_title('B', fontweight='bold', loc='left')

        plt.tight_layout()
        plt.savefig('figure.pdf', dpi=300, bbox_inches='tight')
        plt.show()

        # High-resolution style for large displays
        bt.visualize.publication_style(
            fontsize=12,
            figsize=(10, 6),
            dpi=300,
            usetex=False  # Set True if LaTeX available
        )

        # Create detailed multi-panel figure
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        for i, ax in enumerate(axes.flat):
            # Simulate different neural data types
            if i % 3 == 0:
                # Spike trains
                spikes = np.random.poisson(5, 100)
                ax.plot(spikes, 'k.', markersize=2)
                ax.set_ylabel('Spikes')
            elif i % 3 == 1:
                # Oscillations
                t = np.linspace(0, 10, 1000)
                signal = np.sin(2*np.pi*t) + 0.3*np.sin(8*np.pi*t)
                ax.plot(t, signal, 'k-', linewidth=1)
                ax.set_ylabel('Amplitude')
            else:
                # Distribution
                data = np.random.gamma(2, 2, 1000)
                ax.hist(data, bins=30, color='lightgray', edgecolor='black')
                ax.set_ylabel('Frequency')

            ax.set_xlabel('Time/Value')
            ax.set_title(f'Panel {chr(65+i)}', fontweight='bold', loc='left')

        plt.tight_layout()
        plt.show()
    """
    params = {
        'figure.figsize': figsize,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'font.size': fontsize,
        'font.family': 'serif' if not usetex else 'serif',
        'text.usetex': usetex,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize + 1,
        'xtick.labelsize': fontsize - 1,
        'ytick.labelsize': fontsize - 1,
        'legend.fontsize': fontsize - 1,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'axes.linewidth': 1,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,
        'xtick.major.width': 1,
        'ytick.major.width': 1,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'legend.frameon': False,
        'legend.numpoints': 1,
        'legend.scatterpoints': 1
    }
    rcParams.update(params)


@set_module_as('braintools.visualize')
def dark_style(
    background_color: str = '#2E2E2E',
    text_color: str = '#FFFFFF',
    grid_color: str = '#404040',
    accent_color: str = '#00D4AA'
):
    """Apply dark theme style for low-light environments.

    This function configures matplotlib with a dark theme optimized for
    comfortable viewing in low-light conditions while maintaining good
    contrast for data visualization.

    Parameters
    ----------
    background_color : str, default='#2E2E2E'
        Background color for figures and axes. Dark gray provides
        comfortable viewing without being pure black.
    text_color : str, default='#FFFFFF'
        Color for all text elements including labels and titles.
    grid_color : str, default='#404040'
        Color for grid lines and axis edges. Lighter than background
        for subtle visual structure.
    accent_color : str, default='#00D4AA'
        Primary accent color for data highlights and emphasis.

    Notes
    -----
    Dark style benefits:

    - Reduced eye strain in low-light environments
    - Better contrast for bright data elements
    - Professional appearance for presentations
    - Easier on battery life for OLED displays

    The style includes a carefully selected color cycle that maintains
    good contrast against the dark background while being visually
    appealing and distinguishable.

    Examples
    --------
    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import braintools as bt

        # Apply dark style
        bt.visualize.dark_style()

        # Create dark-themed neural activity plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: Membrane potential traces
        time = np.linspace(0, 100, 1000)
        n_neurons = 5

        for i in range(n_neurons):
            baseline = -65 + i * 2
            noise = np.random.randn(1000) * 3
            spikes = np.random.poisson(0.05, 1000) * 40
            v_mem = baseline + noise + spikes

            ax1.plot(time, v_mem, linewidth=1.5, alpha=0.8,
                    label=f'Neuron {i+1}')

        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Membrane Potential (mV)')
        ax1.set_title('Neural Activity - Dark Theme')
        ax1.legend()

        # Panel 2: Spike frequency over time
        time_bins = np.linspace(0, 100, 50)
        spike_rates = []

        for i in range(len(time_bins)-1):
            # Simulate varying spike rate
            base_rate = 20 + 15 * np.sin(time_bins[i] * 0.1)
            rate = max(0, base_rate + np.random.randn() * 5)
            spike_rates.append(rate)

        ax2.bar(time_bins[:-1], spike_rates, width=2,
                alpha=0.7, edgecolor='white', linewidth=0.5)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Spike Rate (Hz)')
        ax2.set_title('Population Firing Rate')

        plt.tight_layout()
        plt.show()

        # Customize dark style colors
        bt.visualize.dark_style(
            background_color='#1E1E1E',  # Darker background
            text_color='#E0E0E0',        # Softer white
            grid_color='#333333',        # Subtle grid
            accent_color='#FF6B9D'       # Pink accent
        )

        # Create heatmap with custom dark style
        correlation_matrix = np.random.randn(10, 10)
        correlation_matrix = np.corrcoef(correlation_matrix)

        plt.figure(figsize=(8, 6))
        im = plt.imshow(correlation_matrix, cmap='RdBu_r',
                       aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im, label='Correlation Coefficient')
        plt.title('Neural Connectivity Correlation Matrix')
        plt.xlabel('Neuron Index')
        plt.ylabel('Neuron Index')
        plt.show()

        # Dark style for presentations
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Different plot types showcasing dark theme
        plot_types = [
            ('Line Plot', 'plot'),
            ('Scatter Plot', 'scatter'),
            ('Histogram', 'hist'),
            ('Box Plot', 'boxplot')
        ]

        for i, (title, plot_type) in enumerate(plot_types):
            ax = axes.flat[i]

            if plot_type == 'plot':
                x = np.linspace(0, 10, 100)
                y = np.sin(x) + np.random.randn(100) * 0.1
                ax.plot(x, y, linewidth=2)
            elif plot_type == 'scatter':
                x = np.random.randn(100)
                y = x + np.random.randn(100) * 0.5
                ax.scatter(x, y, alpha=0.6, s=50)
            elif plot_type == 'hist':
                data = np.random.gamma(2, 2, 1000)
                ax.hist(data, bins=30, alpha=0.7, edgecolor='white')
            elif plot_type == 'boxplot':
                data = [np.random.randn(100) + i for i in range(4)]
                ax.boxplot(data, patch_artist=True)

            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    """
    params = {
        'figure.facecolor': background_color,
        'axes.facecolor': background_color,
        'savefig.facecolor': background_color,
        'axes.edgecolor': grid_color,
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.color': grid_color,
        'grid.linewidth': 0.8,
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'legend.facecolor': background_color,
        'legend.edgecolor': grid_color
    }
    rcParams.update(params)

    # Set dark-friendly color cycle
    dark_colors = ['#00D4AA', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7', '#DDA0DD']
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', dark_colors)


@set_module_as('braintools.visualize')
def colorblind_friendly_style():
    """Apply colorblind-friendly color palette for inclusive visualization.

    This function configures matplotlib to use a color palette that is
    accessible to viewers with various forms of color vision deficiency.
    The palette maintains good contrast and distinctiveness across different
    types of colorblindness.

    Notes
    -----
    Colorblind accessibility features:

    - Colors selected to be distinguishable for protanopia (red-blind)
    - Colors selected to be distinguishable for deuteranopia (green-blind)
    - Colors selected to be distinguishable for tritanopia (blue-blind)
    - High contrast ratios for general accessibility
    - Based on colorbrewer2.org recommendations

    The palette includes 10 distinct colors that maintain good separation
    in both normal and colorblind vision. This ensures that scientific
    figures are accessible to the broadest possible audience.

    Examples
    --------
    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import braintools as bt

        # Apply colorblind-friendly style
        bt.visualize.colorblind_friendly_style()

        # Create multi-line plot with distinguishable colors
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: Neural population responses
        time = np.linspace(0, 50, 500)
        n_populations = 6

        for i in range(n_populations):
            # Different response patterns for each population
            if i % 3 == 0:
                response = np.exp(-time/10) * np.sin(time * 0.5)
            elif i % 3 == 1:
                response = np.exp(-time/15) * np.cos(time * 0.3)
            else:
                response = 0.5 * np.exp(-time/20) * (1 + np.sin(time * 0.2))

            # Add noise
            response += np.random.randn(len(time)) * 0.05

            ax1.plot(time, response + i * 0.3, linewidth=2,
                    label=f'Population {i+1}')

        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Response Amplitude')
        ax1.set_title('Neural Population Responses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Grouped bar chart
        categories = ['Exc', 'Inh', 'Mix']
        conditions = ['Control', 'Drug A', 'Drug B', 'Drug C']

        # Generate sample data
        data = np.random.gamma(2, 0.5, (len(conditions), len(categories)))
        data = data + np.array([[0.5, 1.2, 0.8]]) * np.arange(len(conditions)).reshape(-1, 1)

        x = np.arange(len(categories))
        width = 0.2

        for i, condition in enumerate(conditions):
            offset = (i - len(conditions)/2 + 0.5) * width
            bars = ax2.bar(x + offset, data[i], width,
                          label=condition, alpha=0.8)

        ax2.set_xlabel('Neuron Type')
        ax2.set_ylabel('Firing Rate (Hz)')
        ax2.set_title('Drug Effects on Neural Activity')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

        # Test colorblind palette with scatter plot
        plt.figure(figsize=(10, 8))

        # Create clusters of data points
        n_clusters = 8
        n_points_per_cluster = 50

        for i in range(n_clusters):
            # Generate cluster center
            center_x = np.random.uniform(-5, 5)
            center_y = np.random.uniform(-5, 5)

            # Generate points around center
            x = np.random.normal(center_x, 0.8, n_points_per_cluster)
            y = np.random.normal(center_y, 0.8, n_points_per_cluster)

            plt.scatter(x, y, s=60, alpha=0.7,
                       label=f'Cluster {i+1}', edgecolors='white', linewidth=0.5)

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Colorblind-Friendly Clustering Visualization')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Demonstrate accessibility with line styles
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use both color and line style for maximum accessibility
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
        markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '+', 'x']

        time = np.linspace(0, 10, 100)

        for i in range(6):  # Use first 6 colors
            signal = np.sin(time + i * np.pi/3) * np.exp(-time/10)
            ax.plot(time, signal, linestyle=linestyles[i],
                   marker=markers[i], markersize=4, markevery=10,
                   linewidth=2, label=f'Signal {i+1}')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Accessible Visualization with Color + Style')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
    """
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', COLORBLIND_PALETTE)


@set_module_as('braintools.visualize')
def create_neural_colormap(
    name: str,
    colors: List[str],
    n_bins: int = 256
) -> LinearSegmentedColormap:
    """Create custom colormap for neural data visualization.

    This function creates a custom LinearSegmentedColormap from a list of colors
    and registers it with matplotlib for use in plots. The colormap is designed
    to provide smooth transitions between colors optimized for neural data.

    Parameters
    ----------
    name : str
        Name of the colormap. This will be used to reference the colormap
        in matplotlib plotting functions (e.g., plt.imshow(data, cmap=name)).
    colors : list of str
        List of color specifications (hex codes, named colors, or RGB tuples).
        Colors will be evenly distributed across the colormap range.
        Minimum of 2 colors required.
    n_bins : int, default=256
        Number of discrete color levels in the colormap. Higher values
        provide smoother gradients but use more memory.

    Returns
    -------
    cmap : LinearSegmentedColormap
        The created colormap object that can be used directly or referenced by name.

    Notes
    -----
    Color specifications can be:
    - Hex codes: '#FF0000', '#00FF00'
    - Named colors: 'red', 'blue', 'green'
    - RGB tuples: (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)

    The colormap is automatically registered with matplotlib's colormap registry,
    making it available for use in all plotting functions that accept a colormap.

    Examples
    --------
    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import braintools as bt

        # Create a simple blue-to-red colormap
        cmap = bt.visualize.create_neural_colormap(
            name='blue_red',
            colors=['#0000FF', '#FFFFFF', '#FF0000']
        )

        # Use the colormap in a heatmap
        data = np.random.randn(20, 20)
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap='blue_red', aspect='auto')
        plt.colorbar(label='Activity Level')
        plt.title('Neural Activity Heatmap')
        plt.xlabel('Time Bins')
        plt.ylabel('Neurons')
        plt.show()

        # Create membrane potential colormap
        membrane_cmap = bt.visualize.create_neural_colormap(
            name='membrane_potential',
            colors=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
            n_bins=128
        )

        # Simulate membrane potential data
        time = np.linspace(0, 100, 200)
        neurons = np.arange(50)
        Time, Neurons = np.meshgrid(time, neurons)

        # Create varying membrane potentials
        membrane_data = -65 + 20 * np.sin(Time * 0.1 + Neurons * 0.2) * np.exp(-Time/50)
        membrane_data += np.random.randn(*membrane_data.shape) * 2

        plt.figure(figsize=(12, 6))
        im = plt.imshow(membrane_data, cmap='membrane_potential',
                       aspect='auto', extent=[0, 100, 0, 50])
        plt.colorbar(im, label='Membrane Potential (mV)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron ID')
        plt.title('Membrane Potential Dynamics')
        plt.show()

        # Create spike activity colormap with transparency
        spike_cmap = bt.visualize.create_neural_colormap(
            name='spike_activity',
            colors=['#000033', '#0066CC', '#00CCFF', '#FFFF00', '#FF6600'],
            n_bins=64
        )

        # Generate spike raster data
        n_neurons = 100
        n_time_bins = 200
        spike_prob = 0.05

        spike_matrix = np.random.random((n_neurons, n_time_bins)) < spike_prob
        spike_matrix = spike_matrix.astype(float)

        # Add some structure - bursts
        for i in range(0, n_time_bins, 40):
            burst_neurons = np.random.choice(n_neurons, 20, replace=False)
            spike_matrix[burst_neurons, i:i+5] += np.random.random((20, 5)) * 2

        plt.figure(figsize=(12, 8))
        plt.imshow(spike_matrix, cmap='spike_activity', aspect='auto',
                  interpolation='nearest')
        plt.colorbar(label='Spike Activity')
        plt.xlabel('Time Bins')
        plt.ylabel('Neuron ID')
        plt.title('Population Spike Activity')
        plt.show()

        # Create connectivity strength colormap
        connectivity_cmap = bt.visualize.create_neural_colormap(
            name='connectivity',
            colors=['#440154', '#31688E', '#35B779', '#FDE725']
        )

        # Generate connectivity matrix
        n_nodes = 50
        connectivity = np.random.exponential(0.1, (n_nodes, n_nodes))
        connectivity[np.eye(n_nodes, dtype=bool)] = 0  # No self-connections

        # Add some structure
        for i in range(0, n_nodes, 10):
            for j in range(i, min(i+10, n_nodes)):
                connectivity[i:i+10, j:j+10] *= 3  # Stronger within-group connections

        plt.figure(figsize=(10, 8))
        im = plt.imshow(connectivity, cmap='connectivity', aspect='auto')
        plt.colorbar(im, label='Connection Strength')
        plt.xlabel('Target Neuron')
        plt.ylabel('Source Neuron')
        plt.title('Neural Network Connectivity Matrix')
        plt.show()

        # Use multiple custom colormaps in subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Different types of neural data with appropriate colormaps
        data_types = [
            ('Membrane Potential', 'membrane_potential'),
            ('Spike Activity', 'spike_activity'),
            ('Connectivity', 'connectivity'),
            ('General Activity', 'blue_red')
        ]

        for i, (title, cmap_name) in enumerate(data_types):
            ax = axes.flat[i]
            data = np.random.randn(30, 50)

            if 'membrane' in cmap_name:
                data = -65 + data * 10
            elif 'spike' in cmap_name:
                data = np.abs(data)
            elif 'connectivity' in cmap_name:
                data = np.abs(data) * 0.5

            im = ax.imshow(data, cmap=cmap_name, aspect='auto')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
    """
    cmap = LinearSegmentedColormap.from_list(name, colors, N=n_bins)
    plt.colormaps.register(cmap, name=name)
    return cmap


@set_module_as('braintools.visualize')
def brain_colormaps():
    """Create and register brain-specific colormaps."""
    # Membrane potential colormap
    membrane_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    create_neural_colormap('membrane', membrane_colors)

    # Spike activity colormap
    spike_colors = ['#0F0F23', '#FF6B6B', '#FFE66D', '#FFFFFF']
    create_neural_colormap('spikes', spike_colors)

    # Connectivity colormap
    connectivity_colors = ['#440154', '#31688E', '#35B779', '#FDE725']
    create_neural_colormap('connectivity', connectivity_colors)

    # Brain activation colormap
    brain_colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000']
    create_neural_colormap('brain_activation', brain_colors)


@set_module_as('braintools.visualize')
def apply_style(style_name: str, **kwargs):
    """Apply predefined style by name.
    
    Parameters
    ----------
    style_name : str
        Name of style: 'neural', 'publication', 'dark', 'colorblind'.
    **kwargs
        Additional style parameters.
    """
    if style_name == 'neural':
        neural_style(**kwargs)
    elif style_name == 'publication':
        publication_style(**kwargs)
    elif style_name == 'dark':
        dark_style(**kwargs)
    elif style_name == 'colorblind':
        colorblind_friendly_style()
    else:
        raise ValueError(f"Unknown style: {style_name}")


@set_module_as('braintools.visualize')
def get_color_palette(palette_name: str, n_colors: Optional[int] = None) -> List[str]:
    """Get predefined color palette.
    
    Parameters
    ----------
    palette_name : str
        Name of palette: 'neural', 'colorblind', 'dark'.
    n_colors : int, optional
        Number of colors to return.
        
    Returns
    -------
    colors : list
        List of color hex codes.
    """
    if palette_name == 'neural':
        colors = list(NEURAL_COLORS.values())
    elif palette_name == 'colorblind':
        colors = COLORBLIND_PALETTE.copy()
    elif palette_name == 'dark':
        colors = ['#00D4AA', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7', '#DDA0DD']
    else:
        raise ValueError(f"Unknown palette: {palette_name}")

    if n_colors is not None:
        if n_colors <= len(colors):
            colors = colors[:n_colors]
        else:
            # Repeat colors if needed
            colors = (colors * (n_colors // len(colors) + 1))[:n_colors]

    return colors


@set_module_as('braintools.visualize')
def set_default_colors(color_dict: Dict[str, str]):
    """Set default colors for neural elements.
    
    Parameters
    ----------
    color_dict : dict
        Dictionary mapping element names to colors.
    """
    global NEURAL_COLORS
    NEURAL_COLORS.update(color_dict)
