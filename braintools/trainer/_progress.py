# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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
Progress bar utilities for training visualization.

This module provides various progress bar implementations for displaying
training progress, including support for tqdm and rich.
"""

import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, Union

__all__ = [
    'ProgressBar',
    'SimpleProgressBar',
    'TQDMProgressBarWrapper',
    'RichProgressBarWrapper',
    'get_progress_bar',
]


class ProgressBar(ABC):
    """
    Abstract base class for progress bars.

    Subclass this to implement custom progress bar displays.
    """
    __module__ = 'braintools.trainer'

    @abstractmethod
    def start(
        self,
        total: Optional[int] = None,
        desc: str = '',
        unit: str = 'it',
    ):
        """
        Start the progress bar.

        Parameters
        ----------
        total : int, optional
            Total number of iterations.
        desc : str
            Description to display.
        unit : str
            Unit name for iterations.
        """
        pass

    @abstractmethod
    def update(self, n: int = 1, **kwargs):
        """
        Update the progress bar.

        Parameters
        ----------
        n : int
            Number of iterations to advance.
        **kwargs
            Additional metrics to display.
        """
        pass

    @abstractmethod
    def set_postfix(self, metrics: Dict[str, Any]):
        """
        Set postfix text with metrics.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Metrics to display.
        """
        pass

    @abstractmethod
    def close(self):
        """Close the progress bar."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SimpleProgressBar(ProgressBar):
    """
    Simple text-based progress bar with no dependencies.

    Provides a basic progress bar that works in any terminal.

    Parameters
    ----------
    width : int, default=50
        Width of the progress bar in characters.
    refresh_rate : float, default=0.1
        Minimum time between updates in seconds.

    Examples
    --------
    >>> pbar = SimpleProgressBar()
    >>> pbar.start(total=100, desc='Training')
    >>> for i in range(100):
    ...     pbar.update(1, loss=0.5)
    >>> pbar.close()
    """
    __module__ = 'braintools.trainer'

    def __init__(self, width: int = 50, refresh_rate: float = 0.1):
        self.width = width
        self.refresh_rate = refresh_rate

        self._total: Optional[int] = None
        self._n: int = 0
        self._desc: str = ''
        self._unit: str = 'it'
        self._postfix: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._last_update_time: float = 0

    def start(
        self,
        total: Optional[int] = None,
        desc: str = '',
        unit: str = 'it',
    ):
        self._total = total
        self._n = 0
        self._desc = desc
        self._unit = unit
        self._postfix = {}
        self._start_time = time.time()
        self._last_update_time = 0
        self._print_bar()

    def update(self, n: int = 1, **kwargs):
        self._n += n
        if kwargs:
            self._postfix.update(kwargs)

        # Rate limit updates
        current_time = time.time()
        if current_time - self._last_update_time >= self.refresh_rate:
            self._print_bar()
            self._last_update_time = current_time

    def set_postfix(self, metrics: Dict[str, Any]):
        self._postfix = metrics
        self._print_bar()

    def _format_time(self, seconds: float) -> str:
        """Format seconds into a readable string."""
        if seconds < 60:
            return f'{seconds:.1f}s'
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f'{minutes}:{secs:02d}'
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f'{hours}:{minutes:02d}'

    def _print_bar(self):
        """Print the progress bar."""
        # Calculate progress
        if self._total and self._total > 0:
            fraction = self._n / self._total
            percent = fraction * 100
            filled = int(self.width * fraction)
            bar = '=' * filled + '>' + '-' * (self.width - filled - 1)
        else:
            percent = 0
            bar = '-' * self.width
            fraction = 0

        # Calculate timing
        elapsed = time.time() - self._start_time if self._start_time else 0
        if self._n > 0 and self._total:
            rate = self._n / elapsed if elapsed > 0 else 0
            remaining = (self._total - self._n) / rate if rate > 0 else 0
            time_str = f'{self._format_time(elapsed)}<{self._format_time(remaining)}'
        else:
            time_str = self._format_time(elapsed)

        # Format postfix
        if self._postfix:
            postfix_items = []
            for key, value in self._postfix.items():
                if isinstance(value, float):
                    postfix_items.append(f'{key}={value:.4f}')
                else:
                    postfix_items.append(f'{key}={value}')
            postfix_str = ', '.join(postfix_items)
        else:
            postfix_str = ''

        # Build output
        if self._total:
            output = f'\r{self._desc}: {percent:5.1f}%|{bar}| {self._n}/{self._total} [{time_str}]'
        else:
            output = f'\r{self._desc}: {self._n} {self._unit} [{time_str}]'

        if postfix_str:
            output += f' {postfix_str}'

        # Print
        sys.stdout.write(output)
        sys.stdout.flush()

    def close(self):
        """Close the progress bar."""
        self._print_bar()
        sys.stdout.write('\n')
        sys.stdout.flush()


class TQDMProgressBarWrapper(ProgressBar):
    """
    Progress bar wrapper using tqdm.

    Provides rich progress bar features with tqdm.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to tqdm.

    Examples
    --------
    >>> pbar = TQDMProgressBarWrapper(leave=True)
    >>> pbar.start(total=100, desc='Training')
    """
    __module__ = 'braintools.trainer'

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._pbar = None

    def start(
        self,
        total: Optional[int] = None,
        desc: str = '',
        unit: str = 'it',
    ):
        try:
            from tqdm import tqdm
            self._pbar = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                **self._kwargs
            )
        except ImportError:
            # Fall back to simple progress bar
            self._pbar = SimpleProgressBar()
            self._pbar.start(total, desc, unit)

    def update(self, n: int = 1, **kwargs):
        if self._pbar is not None:
            if hasattr(self._pbar, 'update'):
                self._pbar.update(n)
            if kwargs and hasattr(self._pbar, 'set_postfix'):
                self._pbar.set_postfix(kwargs)

    def set_postfix(self, metrics: Dict[str, Any]):
        if self._pbar is not None and hasattr(self._pbar, 'set_postfix'):
            self._pbar.set_postfix(metrics)

    def close(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


class RichProgressBarWrapper(ProgressBar):
    """
    Progress bar wrapper using rich.

    Provides beautiful progress bars with rich library features.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to rich Progress.

    Examples
    --------
    >>> pbar = RichProgressBarWrapper()
    >>> pbar.start(total=100, desc='Training')
    """
    __module__ = 'braintools.trainer'

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._progress = None
        self._task_id = None

    def start(
        self,
        total: Optional[int] = None,
        desc: str = '',
        unit: str = 'it',
    ):
        try:
            from rich.progress import (
                Progress,
                SpinnerColumn,
                TextColumn,
                BarColumn,
                TaskProgressColumn,
                TimeRemainingColumn,
                TimeElapsedColumn,
            )

            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                **self._kwargs
            )
            self._progress.start()
            self._task_id = self._progress.add_task(desc, total=total)
        except ImportError:
            # Fall back to simple progress bar
            self._progress = SimpleProgressBar()
            self._progress.start(total, desc, unit)
            self._task_id = None

    def update(self, n: int = 1, **kwargs):
        if self._progress is not None:
            if self._task_id is not None:
                self._progress.update(self._task_id, advance=n)
            else:
                self._progress.update(n, **kwargs)

    def set_postfix(self, metrics: Dict[str, Any]):
        # Rich doesn't have postfix, but we can update description
        if self._progress is not None and self._task_id is not None:
            metrics_str = ', '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                                    for k, v in metrics.items())
            self._progress.update(self._task_id, description=metrics_str)

    def close(self):
        if self._progress is not None:
            if hasattr(self._progress, 'stop'):
                self._progress.stop()
            elif hasattr(self._progress, 'close'):
                self._progress.close()
            self._progress = None
            self._task_id = None


def get_progress_bar(
    backend: str = 'auto',
    **kwargs,
) -> ProgressBar:
    """
    Get a progress bar instance.

    Parameters
    ----------
    backend : str, default='auto'
        Progress bar backend to use.
        Options: 'auto', 'simple', 'tqdm', 'rich'
    **kwargs
        Additional arguments passed to the progress bar.

    Returns
    -------
    ProgressBar
        Progress bar instance.

    Examples
    --------
    >>> pbar = get_progress_bar('tqdm')
    >>> pbar.start(total=100, desc='Training')
    """
    if backend == 'auto':
        # Try rich first, then tqdm, then simple
        try:
            import rich
            return RichProgressBarWrapper(**kwargs)
        except ImportError:
            pass

        try:
            import tqdm
            return TQDMProgressBarWrapper(**kwargs)
        except ImportError:
            pass

        return SimpleProgressBar(**kwargs)

    elif backend == 'simple':
        return SimpleProgressBar(**kwargs)

    elif backend == 'tqdm':
        return TQDMProgressBarWrapper(**kwargs)

    elif backend == 'rich':
        return RichProgressBarWrapper(**kwargs)

    else:
        raise ValueError(f"Unknown progress bar backend: {backend}")


class ProgressBarPool:
    """
    Pool of progress bars for nested loops.

    Manages multiple progress bars for training, validation, and testing.

    Parameters
    ----------
    backend : str, default='auto'
        Progress bar backend to use.

    Examples
    --------
    >>> pool = ProgressBarPool()
    >>> with pool.training(total=100) as pbar:
    ...     for i in range(100):
    ...         pbar.update(1)
    >>> with pool.validation(total=50) as pbar:
    ...     for i in range(50):
    ...         pbar.update(1)
    """
    __module__ = 'braintools.trainer'

    def __init__(self, backend: str = 'auto'):
        self.backend = backend
        self._bars: Dict[str, ProgressBar] = {}

    def _get_bar(self, name: str) -> ProgressBar:
        """Get or create a progress bar."""
        if name not in self._bars:
            self._bars[name] = get_progress_bar(self.backend)
        return self._bars[name]

    def training(
        self,
        total: Optional[int] = None,
        desc: str = 'Training',
    ) -> ProgressBar:
        """Get training progress bar."""
        pbar = self._get_bar('training')
        pbar.start(total, desc)
        return pbar

    def validation(
        self,
        total: Optional[int] = None,
        desc: str = 'Validation',
    ) -> ProgressBar:
        """Get validation progress bar."""
        pbar = self._get_bar('validation')
        pbar.start(total, desc)
        return pbar

    def testing(
        self,
        total: Optional[int] = None,
        desc: str = 'Testing',
    ) -> ProgressBar:
        """Get testing progress bar."""
        pbar = self._get_bar('testing')
        pbar.start(total, desc)
        return pbar

    def epoch(
        self,
        total: Optional[int] = None,
        desc: str = 'Epoch',
    ) -> ProgressBar:
        """Get epoch progress bar."""
        pbar = self._get_bar('epoch')
        pbar.start(total, desc)
        return pbar

    def close_all(self):
        """Close all progress bars."""
        for pbar in self._bars.values():
            pbar.close()
        self._bars.clear()


class MetricsDisplay:
    """
    Display training metrics in a formatted way.

    Parameters
    ----------
    format_spec : str, default='.4f'
        Format specification for floating point numbers.
    max_width : int, default=80
        Maximum display width.

    Examples
    --------
    >>> display = MetricsDisplay()
    >>> display.print_epoch_summary(10, {'loss': 0.5, 'acc': 0.95})
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        format_spec: str = '.4f',
        max_width: int = 80,
    ):
        self.format_spec = format_spec
        self.max_width = max_width

    def format_metric(self, name: str, value: Any) -> str:
        """Format a single metric."""
        if isinstance(value, float):
            return f'{name}: {value:{self.format_spec}}'
        return f'{name}: {value}'

    def format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format multiple metrics."""
        items = [self.format_metric(k, v) for k, v in metrics.items()]
        return ', '.join(items)

    def print_epoch_summary(
        self,
        epoch: int,
        train_metrics: Optional[Dict[str, Any]] = None,
        val_metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Print a summary of the epoch.

        Parameters
        ----------
        epoch : int
            Epoch number.
        train_metrics : Dict[str, Any], optional
            Training metrics.
        val_metrics : Dict[str, Any], optional
            Validation metrics.
        """
        print(f"\n{'='*self.max_width}")
        print(f"Epoch {epoch} Summary")
        print(f"{'='*self.max_width}")

        if train_metrics:
            print(f"  Train: {self.format_metrics(train_metrics)}")

        if val_metrics:
            print(f"  Val:   {self.format_metrics(val_metrics)}")

        print(f"{'='*self.max_width}\n")

    def print_training_start(
        self,
        model_name: str = 'Model',
        num_params: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ):
        """Print training start message."""
        print(f"\n{'='*self.max_width}")
        print(f"Starting Training: {model_name}")
        if num_params:
            print(f"  Parameters: {num_params:,}")
        if max_epochs:
            print(f"  Max Epochs: {max_epochs}")
        print(f"{'='*self.max_width}\n")

    def print_training_end(
        self,
        best_metrics: Optional[Dict[str, Any]] = None,
        total_time: Optional[float] = None,
    ):
        """Print training end message."""
        print(f"\n{'='*self.max_width}")
        print("Training Complete!")
        if best_metrics:
            print(f"  Best: {self.format_metrics(best_metrics)}")
        if total_time:
            print(f"  Total Time: {total_time:.1f}s")
        print(f"{'='*self.max_width}\n")
