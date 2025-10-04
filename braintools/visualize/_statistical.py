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
import scipy.stats as stats

from braintools._misc import set_module_as
from braintools.tree import as_numpy

__all__ = [
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
]


@set_module_as('braintools.visualize')
def correlation_matrix(
    data: np.ndarray,
    labels: Optional[List[str]] = None,
    method: str = 'pearson',
    cmap: str = 'RdBu_r',
    mask_diagonal: bool = False,
    show_values: bool = True,
    value_format: str = '.2f',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 8),
    show_colorbar: bool = True,
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot correlation matrix heatmap.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (samples, features).
    labels : list, optional
        Feature labels.
    method : str
        Correlation method: 'pearson', 'spearman', 'kendall'.
    cmap : str
        Colormap for the matrix.
    mask_diagonal : bool
        Whether to mask the diagonal.
    show_values : bool
        Whether to show correlation values.
    value_format : str
        Format string for values.
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

    data = as_numpy(data)

    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = np.corrcoef(data.T)
    elif method == 'spearman':
        corr_matrix = stats.spearmanr(data)[0]
    elif method == 'kendall':
        corr_matrix = stats.kendalltau(data)[0]
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    # Mask diagonal if requested
    if mask_diagonal:
        mask = np.eye(corr_matrix.shape[0], dtype=bool)
        corr_matrix = np.ma.masked_array(corr_matrix, mask=mask)

    # Plot matrix
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, **kwargs)

    # Add colorbar
    if show_colorbar:
        plt.colorbar(im, ax=ax, label='Correlation')

    # Add values
    if show_values:
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                if not (mask_diagonal and i == j):
                    value = corr_matrix[i, j]
                    text_color = 'white' if abs(value) > 0.5 else 'black'
                    ax.text(j, i, format(value, value_format), ha='center', va='center',
                            color=text_color, fontweight='bold')

    # Labels
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

    if title:
        ax.set_title(title)

    return ax


@set_module_as('braintools.visualize')
def distribution_plot(
    data: Union[np.ndarray, List[np.ndarray]],
    labels: Optional[List[str]] = None,
    plot_type: str = 'hist',
    bins: Union[int, np.ndarray] = 30,
    density: bool = True,
    fit_normal: bool = False,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    colors: Optional[List[str]] = None,
    alpha: float = 0.7,
    xlabel: str = 'Value',
    ylabel: str = 'Density',
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot distribution of data with various options.
    
    Parameters
    ----------
    data : array-like or list of arrays
        Data to plot distribution for.
    labels : list, optional
        Labels for each dataset.
    plot_type : str
        Type of plot: 'hist', 'kde', 'both'.
    bins : int or array-like
        Number of bins or bin edges for histogram.
    density : bool
        Whether to normalize histogram.
    fit_normal : bool
        Whether to overlay normal distribution fit.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    colors : list, optional
        Colors for each dataset.
    alpha : float
        Alpha transparency.
    xlabel, ylabel, title : str
        Axis labels and title.
    **kwargs
        Additional arguments passed to plotting functions.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]

    # Convert to numpy arrays
    data = [as_numpy(d) for d in data]

    # Set up colors
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    # Set up labels
    if labels is None:
        labels = [f'Dataset {i + 1}' for i in range(len(data))]

    for i, (d, label, color) in enumerate(zip(data, labels, colors)):
        if plot_type in ['hist', 'both']:
            ax.hist(d, bins=bins, density=density, alpha=alpha, color=color,
                    label=label, **kwargs)

        if plot_type in ['kde', 'both']:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(d)
                x_range = np.linspace(d.min(), d.max(), 200)
                ax.plot(x_range, kde(x_range), color=color, linewidth=2,
                        label=f'{label} (KDE)' if plot_type == 'both' else label)
            except ImportError:
                warnings.warn("SciPy not available for KDE plotting")

        if fit_normal:
            mu, sigma = stats.norm.fit(d)
            x_range = np.linspace(d.min(), d.max(), 200)
            normal_fit = stats.norm.pdf(x_range, mu, sigma)
            ax.plot(x_range, normal_fit, '--', color=color, linewidth=2,
                    label=f'{label} Normal Fit')

    # Labels and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if len(data) > 1 or fit_normal:
        ax.legend()

    ax.grid(True, alpha=0.3)
    return ax


@set_module_as('braintools.visualize')
def qq_plot(
    data: np.ndarray,
    distribution: str = 'norm',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    color: str = 'blue',
    alpha: float = 0.7,
    line_color: str = 'red',
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Create Q-Q plot to compare data against theoretical distribution.
    
    Parameters
    ----------
    data : np.ndarray
        Data to compare.
    distribution : str
        Theoretical distribution: 'norm', 'uniform', 'expon'.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    color : str
        Point color.
    alpha : float
        Alpha transparency.
    line_color : str
        Reference line color.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to scatter.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    data = as_numpy(data)

    # Get theoretical quantiles
    if distribution == 'norm':
        theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
        dist_name = 'Normal'
    elif distribution == 'uniform':
        theoretical_q = stats.uniform.ppf(np.linspace(0.01, 0.99, len(data)))
        dist_name = 'Uniform'
    elif distribution == 'expon':
        theoretical_q = stats.expon.ppf(np.linspace(0.01, 0.99, len(data)))
        dist_name = 'Exponential'
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Get sample quantiles
    sample_q = np.sort(data)

    # Plot Q-Q
    ax.scatter(theoretical_q, sample_q, color=color, alpha=alpha, **kwargs)

    # Add reference line
    min_val = min(theoretical_q.min(), sample_q.min())
    max_val = max(theoretical_q.max(), sample_q.max())
    ax.plot([min_val, max_val], [min_val, max_val], color=line_color, linestyle='--')

    # Labels
    ax.set_xlabel(f'Theoretical Quantiles ({dist_name})')
    ax.set_ylabel('Sample Quantiles')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Q-Q Plot vs {dist_name} Distribution')

    ax.grid(True, alpha=0.3)
    return ax


@set_module_as('braintools.visualize')
def box_plot(
    data: Union[np.ndarray, List[np.ndarray]],
    labels: Optional[List[str]] = None,
    positions: Optional[List[float]] = None,
    notch: bool = False,
    showmeans: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    colors: Optional[List[str]] = None,
    xlabel: str = 'Groups',
    ylabel: str = 'Values',
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Create box plot for comparing distributions.
    
    Parameters
    ----------
    data : array-like or list of arrays
        Data for each group.
    labels : list, optional
        Group labels.
    positions : list, optional
        Positions for each box.
    notch : bool
        Whether to show notches.
    showmeans : bool
        Whether to show means.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    colors : list, optional
        Colors for each box.
    xlabel, ylabel, title : str
        Axis labels and title.
    **kwargs
        Additional arguments passed to boxplot.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]

    # Convert to numpy arrays
    data = [as_numpy(d) for d in data]

    # Create box plot
    bp = ax.boxplot(data, positions=positions, notch=notch,
                    showmeans=showmeans, patch_artist=True, **kwargs)

    # Color boxes
    if colors is not None:
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    # Labels
    if labels is not None:
        ax.set_xticklabels(labels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)
    return ax


@set_module_as('braintools.visualize')
def violin_plot(
    data: Union[np.ndarray, List[np.ndarray]],
    labels: Optional[List[str]] = None,
    positions: Optional[List[float]] = None,
    showmeans: bool = True,
    showmedians: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    colors: Optional[List[str]] = None,
    xlabel: str = 'Groups',
    ylabel: str = 'Values',
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Create violin plot for comparing distributions.
    
    Parameters
    ----------
    data : array-like or list of arrays
        Data for each group.
    labels : list, optional
        Group labels.
    positions : list, optional
        Positions for each violin.
    showmeans : bool
        Whether to show means.
    showmedians : bool
        Whether to show medians.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    colors : list, optional
        Colors for each violin.
    xlabel, ylabel, title : str
        Axis labels and title.
    **kwargs
        Additional arguments passed to violinplot.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]

    # Convert to numpy arrays
    data = [as_numpy(d) for d in data]

    # Create violin plot
    vp = ax.violinplot(data, positions=positions, showmeans=showmeans,
                       showmedians=showmedians, **kwargs)

    # Color violins
    if colors is not None:
        for patch, color in zip(vp['bodies'], colors):
            patch.set_facecolor(color)

    # Labels
    if labels is not None:
        ax.set_xticks(positions or range(1, len(labels) + 1))
        ax.set_xticklabels(labels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)
    return ax


@set_module_as('braintools.visualize')
def scatter_matrix(
    data: np.ndarray,
    labels: Optional[List[str]] = None,
    diagonal: str = 'hist',
    alpha: float = 0.7,
    figsize: Tuple[float, float] = (12, 12),
    color: str = 'blue',
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Figure:
    """
    Create scatter plot matrix for multivariate data.
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (samples, features).
    labels : list, optional
        Feature labels.
    diagonal : str
        What to plot on diagonal: 'hist', 'kde'.
    alpha : float
        Alpha transparency.
    figsize : tuple
        Figure size (only used if ax is None).
    color : str
        Plot color.
    ax : matplotlib.axes.Axes, optional
        Single axes to plot a simplified version (2x2 subset).
        If None, creates full scatter matrix.
    **kwargs
        Additional arguments passed to scatter plots.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing all subplots.
    """
    data = as_numpy(data)
    n_features = data.shape[1]

    if ax is not None:
        # Simplified version for single axis - show 2x2 subset of most important features
        # Select first 2 features for simplicity
        n_subset = min(2, n_features)

        # Create 2x2 subplot within the given axis
        # We'll create a simple scatter plot of first two features
        if n_features >= 2:
            ax.scatter(data[:, 0], data[:, 1], alpha=alpha, color=color, **kwargs)
            ax.set_xlabel(labels[0] if labels else 'Feature 0')
            ax.set_ylabel(labels[1] if labels else 'Feature 1')
            ax.set_title('Scatter Plot (Features 0 vs 1)')
            ax.grid(True, alpha=0.3)
        else:
            # Only one feature - show histogram
            ax.hist(data[:, 0], alpha=alpha, color=color, bins=20)
            ax.set_xlabel(labels[0] if labels else 'Feature 0')
            ax.set_ylabel('Frequency')
            ax.set_title('Feature Distribution')
            ax.grid(True, alpha=0.3)

        return ax.figure

    else:
        # Full scatter matrix
        fig, axes = plt.subplots(n_features, n_features, figsize=figsize)

        # Handle single feature case
        if n_features == 1:
            axes.hist(data[:, 0], alpha=alpha, color=color, bins=20)
            axes.set_xlabel(labels[0] if labels else 'Feature 0')
            axes.set_ylabel('Frequency')
            plt.tight_layout()
            return fig

        for i in range(n_features):
            for j in range(n_features):
                current_ax = axes[i, j] if n_features > 1 else axes

                if i == j:
                    # Diagonal: histogram or KDE
                    if diagonal == 'hist':
                        current_ax.hist(data[:, i], alpha=alpha, color=color, bins=20)
                    elif diagonal == 'kde':
                        try:
                            from scipy.stats import gaussian_kde
                            kde = gaussian_kde(data[:, i])
                            x_range = np.linspace(data[:, i].min(), data[:, i].max(), 100)
                            current_ax.plot(x_range, kde(x_range), color=color)
                        except ImportError:
                            current_ax.hist(data[:, i], alpha=alpha, color=color, bins=20)
                else:
                    # Off-diagonal: scatter plot
                    current_ax.scatter(data[:, j], data[:, i], alpha=alpha, color=color, **kwargs)

                # Labels
                if i == n_features - 1:
                    if labels is not None:
                        current_ax.set_xlabel(labels[j])
                    else:
                        current_ax.set_xlabel(f'Feature {j}')
                if j == 0:
                    if labels is not None:
                        current_ax.set_ylabel(labels[i])
                    else:
                        current_ax.set_ylabel(f'Feature {i}')

        plt.tight_layout()
        return fig


@set_module_as('braintools.visualize')
def regression_plot(
    x: np.ndarray,
    y: np.ndarray,
    fit_line: bool = True,
    confidence_interval: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    color: str = 'blue',
    alpha: float = 0.7,
    xlabel: str = 'X',
    ylabel: str = 'Y',
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Create regression plot with fitted line and confidence interval.
    
    Parameters
    ----------
    x, y : np.ndarray
        Data arrays.
    fit_line : bool
        Whether to fit regression line.
    confidence_interval : bool
        Whether to show confidence interval.
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
        Additional arguments passed to scatter.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x = as_numpy(x)
    y = as_numpy(y)

    # Scatter plot
    ax.scatter(x, y, alpha=alpha, color=color, **kwargs)

    if fit_line:
        # Fit regression line
        coeffs = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)

        ax.plot(x_fit, y_fit, color='red', linewidth=2, label='Fit')

        # Calculate R-squared
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Add R-squared to plot
        ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if confidence_interval:
            # Calculate confidence interval
            n = len(x)
            t_val = stats.t.ppf(0.975, n - 2)  # 95% confidence

            # Standard error of prediction
            y_pred_all = np.polyval(coeffs, x_fit)
            residuals = y - y_pred
            mse = np.sum(residuals ** 2) / (n - 2)
            se = np.sqrt(mse * (1 / n + (x_fit - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2)))

            ci = t_val * se
            ax.fill_between(x_fit, y_pred_all - ci, y_pred_all + ci,
                            alpha=0.2, color='red', label='95% CI')

        ax.legend()

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)
    return ax


@set_module_as('braintools.visualize')
def residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    color: str = 'blue',
    alpha: float = 0.7,
    xlabel: str = 'Predicted Values',
    ylabel: str = 'Residuals',
    title: str = 'Residual Plot',
    **kwargs
) -> plt.Axes:
    """
    Create residual plot for regression diagnostics.
    
    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted values.
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
        Additional arguments passed to scatter.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    y_true = as_numpy(y_true)
    y_pred = as_numpy(y_pred)

    residuals = y_true - y_pred

    # Scatter plot
    ax.scatter(y_pred, residuals, alpha=alpha, color=color, **kwargs)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    return ax


@set_module_as('braintools.visualize')
def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: Optional[str] = None,
    cmap: str = 'Blues',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    show_values: bool = True,
    title: str = 'Confusion Matrix',
    **kwargs
) -> plt.Axes:
    """
    Plot confusion matrix for classification results.
    
    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted class labels.
    labels : list, optional
        Class labels.
    normalize : str, optional
        Normalization method: 'true', 'pred', 'all'.
    cmap : str
        Colormap for the matrix.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    show_values : bool
        Whether to show values in cells.
    title : str
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

    y_true = as_numpy(y_true)
    y_pred = as_numpy(y_pred)

    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    # Create confusion matrix
    cm = np.zeros((n_classes, n_classes))
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

    # Normalize if requested
    if normalize == 'true':
        cm = cm / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm / cm.sum()

    # Plot matrix
    im = ax.imshow(cm, cmap=cmap, **kwargs)
    plt.colorbar(im, ax=ax)

    # Add values
    if show_values:
        for i in range(n_classes):
            for j in range(n_classes):
                value = cm[i, j]
                if normalize:
                    text = f'{value:.2f}'
                else:
                    text = f'{int(value)}'
                text_color = 'white' if value > cm.max() / 2 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=text_color)

    # Labels
    if labels is None:
        labels = [str(c) for c in classes]

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    return ax


@set_module_as('braintools.visualize')
def roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    color: str = 'blue',
    title: str = 'ROC Curve',
    **kwargs
) -> plt.Axes:
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Prediction scores.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    color : str
        Line color.
    title : str
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

    y_true = as_numpy(y_true)
    y_scores = as_numpy(y_scores)

    # Calculate ROC curve
    thresholds = np.unique(y_scores)
    thresholds = np.sort(thresholds)[::-1]  # Descending order

    tpr = []  # True positive rate
    fpr = []  # False positive rate

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    # Calculate AUC
    auc = np.trapz(tpr, fpr)

    # Plot ROC curve
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'AUC = {auc:.3f}', **kwargs)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random')

    # Labels
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


@set_module_as('braintools.visualize')
def precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    color: str = 'blue',
    title: str = 'Precision-Recall Curve',
    **kwargs
) -> plt.Axes:
    """
    Plot Precision-Recall curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Prediction scores.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    color : str
        Line color.
    title : str
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

    y_true = as_numpy(y_true)
    y_scores = as_numpy(y_scores)

    # Calculate Precision-Recall curve
    thresholds = np.unique(y_scores)
    thresholds = np.sort(thresholds)[::-1]  # Descending order

    precision = []
    recall = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision.append(prec)
        recall.append(rec)

    precision = np.array(precision)
    recall = np.array(recall)

    # Calculate average precision
    ap = np.trapz(precision, recall)

    # Plot PR curve
    ax.plot(recall, precision, color=color, linewidth=2,
            label=f'AP = {ap:.3f}', **kwargs)

    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.6,
               label=f'Random (AP = {baseline:.3f})')

    # Labels
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


@set_module_as('braintools.visualize')
def learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    validation_scores: np.ndarray,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = 'Learning Curve',
    **kwargs
) -> plt.Axes:
    """
    Plot learning curve showing training and validation performance.
    
    Parameters
    ----------
    train_sizes : np.ndarray
        Training set sizes.
    train_scores : np.ndarray
        Training scores (can be 2D for multiple runs).
    validation_scores : np.ndarray
        Validation scores (can be 2D for multiple runs).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    title : str
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

    train_sizes = as_numpy(train_sizes)
    train_scores = as_numpy(train_scores)
    validation_scores = as_numpy(validation_scores)

    # Handle 1D or 2D score arrays
    if train_scores.ndim == 1:
        train_mean = train_scores
        train_std = np.zeros_like(train_scores)
    else:
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

    if validation_scores.ndim == 1:
        val_mean = validation_scores
        val_std = np.zeros_like(validation_scores)
    else:
        val_mean = np.mean(validation_scores, axis=1)
        val_std = np.std(validation_scores, axis=1)

    # Plot training scores
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color='blue')

    # Plot validation scores
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color='red')

    # Labels
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
