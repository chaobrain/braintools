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
    
    Parameters
    ----------
    spike_color : str
        Color for spike representations.
    membrane_color : str
        Color for membrane potential plots.
    background_color : str
        Background color.
    fontsize : int
        Base font size.
    grid : bool
        Whether to show grid.
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
    
    Parameters
    ----------
    fontsize : int
        Base font size.
    figsize : tuple
        Default figure size.
    dpi : int
        Figure DPI.
    usetex : bool
        Whether to use LaTeX for text rendering.
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
    """Apply dark theme style.
    
    Parameters
    ----------
    background_color : str
        Background color.
    text_color : str
        Text color.
    grid_color : str
        Grid color.
    accent_color : str
        Accent color for highlights.
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
    """Apply colorblind-friendly color palette."""
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', COLORBLIND_PALETTE)


@set_module_as('braintools.visualize')
def create_neural_colormap(
    name: str,
    colors: List[str],
    n_bins: int = 256
) -> LinearSegmentedColormap:
    """Create custom colormap for neural data.
    
    Parameters
    ----------
    name : str
        Name of the colormap.
    colors : list
        List of colors for the colormap.
    n_bins : int
        Number of color bins.
        
    Returns
    -------
    cmap : LinearSegmentedColormap
        Custom colormap.
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
