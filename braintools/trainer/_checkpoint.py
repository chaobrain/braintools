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
Checkpointing utilities for model and training state.

This module provides utilities for saving and loading model checkpoints,
including support for best model tracking and automatic cleanup.
"""

import glob
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

__all__ = [
    'CheckpointManager',
    'save_checkpoint',
    'load_checkpoint',
    'find_checkpoint',
    'list_checkpoints',
]


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    overwrite: bool = True,
) -> str:
    """
    Save a checkpoint to disk.

    Parameters
    ----------
    state : Dict[str, Any]
        State dictionary to save.
    filepath : str
        Path to save the checkpoint.
    overwrite : bool, default=True
        Whether to overwrite existing checkpoint.

    Returns
    -------
    str
        Path where the checkpoint was saved.

    Examples
    --------
    >>> state = {'model': model.state_dict(), 'epoch': 10}
    >>> save_checkpoint(state, 'checkpoint.ckpt')
    """
    from braintools.file import msgpack_save

    # Create directory if needed
    dirpath = os.path.dirname(filepath)
    if dirpath:
        Path(dirpath).mkdir(parents=True, exist_ok=True)

    # Check if file exists
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"Checkpoint already exists: {filepath}")

    # Convert JAX arrays to numpy for serialization
    def to_numpy(x):
        if isinstance(x, jnp.ndarray):
            return np.array(x)
        return x

    state = jax.tree_util.tree_map(to_numpy, state)

    # Save using msgpack
    msgpack_save(filepath, state, overwrite=overwrite, verbose=False)

    return filepath


def load_checkpoint(
    filepath: str,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint from disk.

    Parameters
    ----------
    filepath : str
        Path to the checkpoint file.
    map_location : str, optional
        Device to map tensors to (not used in JAX, included for API compatibility).

    Returns
    -------
    Dict[str, Any]
        Loaded state dictionary.

    Examples
    --------
    >>> state = load_checkpoint('checkpoint.ckpt')
    >>> model.load_state_dict(state['model'])
    """
    from braintools.file import msgpack_load

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # Load using msgpack - target=None returns raw dict
    state = msgpack_load(filepath, target=None, verbose=False)

    # Convert numpy arrays to JAX arrays
    def to_jax(x):
        if isinstance(x, np.ndarray):
            return jnp.array(x)
        return x

    state = jax.tree_util.tree_map(to_jax, state)

    return state


def find_checkpoint(
    dirpath: str,
    pattern: str = '*.ckpt',
    best: bool = False,
    last: bool = False,
) -> Optional[str]:
    """
    Find a checkpoint file in a directory.

    Parameters
    ----------
    dirpath : str
        Directory to search in.
    pattern : str, default='*.ckpt'
        Glob pattern for checkpoint files.
    best : bool, default=False
        Return the best checkpoint (by name, looking for 'best' in filename).
    last : bool, default=False
        Return the last/most recent checkpoint.

    Returns
    -------
    str or None
        Path to the found checkpoint, or None if not found.

    Examples
    --------
    >>> checkpoint = find_checkpoint('checkpoints/', best=True)
    >>> checkpoint = find_checkpoint('checkpoints/', last=True)
    """
    if not os.path.exists(dirpath):
        return None

    checkpoints = list_checkpoints(dirpath, pattern)

    if not checkpoints:
        return None

    if best:
        # Look for checkpoint with 'best' in name
        for ckpt in checkpoints:
            if 'best' in os.path.basename(ckpt).lower():
                return ckpt
        # Fall back to most recent
        return checkpoints[-1]

    if last:
        # Look for 'last.ckpt' or most recent by modification time
        for ckpt in checkpoints:
            if os.path.basename(ckpt).lower() == 'last.ckpt':
                return ckpt
        return checkpoints[-1]

    return checkpoints[0] if checkpoints else None


def list_checkpoints(
    dirpath: str,
    pattern: str = '*.ckpt',
    sort_by: str = 'time',
) -> List[str]:
    """
    List all checkpoints in a directory.

    Parameters
    ----------
    dirpath : str
        Directory to search in.
    pattern : str, default='*.ckpt'
        Glob pattern for checkpoint files.
    sort_by : str, default='time'
        How to sort checkpoints ('time', 'name', 'epoch').

    Returns
    -------
    List[str]
        List of checkpoint file paths.

    Examples
    --------
    >>> checkpoints = list_checkpoints('checkpoints/')
    >>> for ckpt in checkpoints:
    ...     print(ckpt)
    """
    if not os.path.exists(dirpath):
        return []

    search_pattern = os.path.join(dirpath, '**', pattern)
    checkpoints = glob.glob(search_pattern, recursive=True)

    if sort_by == 'time':
        checkpoints.sort(key=lambda x: os.path.getmtime(x))
    elif sort_by == 'name':
        checkpoints.sort()
    elif sort_by == 'epoch':
        def extract_epoch(path):
            basename = os.path.basename(path)
            match = re.search(r'epoch[=_]?(\d+)', basename, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return 0
        checkpoints.sort(key=extract_epoch)

    return checkpoints


class CheckpointManager:
    """
    Manager for saving and loading checkpoints.

    Provides functionality for:
    - Saving checkpoints with automatic naming
    - Tracking and keeping only the best N checkpoints
    - Loading checkpoints by epoch, step, or best metric
    - Automatic cleanup of old checkpoints

    Parameters
    ----------
    dirpath : str
        Directory to save checkpoints.
    max_to_keep : int, default=5
        Maximum number of checkpoints to keep. Set to -1 to keep all.
    filename_template : str, optional
        Template for checkpoint filenames.
        Available variables: {epoch}, {step}, {metric}, {timestamp}
    save_best_only : bool, default=False
        Only save when the monitored metric improves.
    monitor : str, optional
        Metric to monitor for best checkpoint selection.
    mode : str, default='min'
        One of 'min' or 'max'. In 'min' mode, lower is better.
    verbose : bool, default=True
        Whether to print checkpoint operations.

    Examples
    --------
    >>> manager = CheckpointManager(
    ...     'checkpoints/',
    ...     max_to_keep=3,
    ...     monitor='val_loss',
    ...     mode='min',
    ... )
    >>> manager.save(model, optimizer, epoch=10, metrics={'val_loss': 0.5})
    >>> state = manager.load_best()
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        dirpath: str,
        max_to_keep: int = 5,
        filename_template: Optional[str] = None,
        save_best_only: bool = False,
        monitor: Optional[str] = None,
        mode: str = 'min',
        verbose: bool = True,
    ):
        self.dirpath = dirpath
        self.max_to_keep = max_to_keep
        self.filename_template = filename_template or 'checkpoint-epoch={epoch:04d}'
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

        # State tracking
        self._checkpoints: List[Dict[str, Any]] = []
        self._best_score: Optional[float] = None
        self._best_checkpoint: Optional[str] = None

        # Create directory
        Path(dirpath).mkdir(parents=True, exist_ok=True)

        # Validate mode
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == 'min':
            return current < best
        return current > best

    def _format_filename(
        self,
        epoch: int,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Format checkpoint filename from template."""
        format_dict = {
            'epoch': epoch,
            'step': step or 0,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        }

        if metrics:
            for key, value in metrics.items():
                # Clean metric name for filename
                clean_key = key.replace('/', '_').replace(' ', '_')
                format_dict[clean_key] = value
                format_dict['metric'] = value  # Generic metric

        try:
            filename = self.filename_template.format(**format_dict)
        except KeyError:
            # Fall back to simple format
            filename = f'checkpoint-epoch={epoch:04d}'

        return filename

    def save(
        self,
        model: Any,
        optimizer: Optional[Any] = None,
        epoch: int = 0,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Save a checkpoint.

        Parameters
        ----------
        model : Any
            Model to save.
        optimizer : Any, optional
            Optimizer to save.
        epoch : int
            Current epoch.
        step : int, optional
            Current step.
        metrics : Dict[str, float], optional
            Current metrics.
        extra_state : Dict[str, Any], optional
            Additional state to save.

        Returns
        -------
        str or None
            Path to saved checkpoint, or None if not saved (save_best_only).

        Examples
        --------
        >>> path = manager.save(model, optimizer, epoch=10)
        """
        # Check if we should save
        if self.save_best_only and self.monitor and metrics:
            current_score = metrics.get(self.monitor)
            if current_score is None:
                if self.verbose:
                    print(f"Warning: Monitor metric '{self.monitor}' not found in metrics")
            elif self._best_score is not None and not self._is_better(current_score, self._best_score):
                if self.verbose:
                    print(f"Skipping checkpoint: {self.monitor}={current_score:.4f} "
                          f"(best={self._best_score:.4f})")
                return None

        # Build checkpoint state
        state = {
            'epoch': epoch,
            'step': step,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
        }

        # Save model state
        if hasattr(model, 'state_dict'):
            state['model_state_dict'] = model.state_dict()
        else:
            # Try to get states directly
            import brainstate
            param_states = model.states(brainstate.ParamState)
            state['model_state_dict'] = {
                str(k): v.value for k, v in param_states.items()
            }

        # Save optimizer state
        if optimizer is not None and hasattr(optimizer, 'state_dict'):
            state['optimizer_state_dict'] = optimizer.state_dict()

        # Add extra state
        if extra_state:
            state.update(extra_state)

        # Generate filename
        filename = self._format_filename(epoch, step, metrics)
        filepath = os.path.join(self.dirpath, f'{filename}.ckpt')

        # Save checkpoint
        save_checkpoint(state, filepath)

        # Track checkpoint
        checkpoint_info = {
            'path': filepath,
            'epoch': epoch,
            'step': step,
            'metrics': metrics or {},
            'timestamp': state['timestamp'],
        }
        self._checkpoints.append(checkpoint_info)

        # Update best tracking
        if self.monitor and metrics:
            current_score = metrics.get(self.monitor)
            if current_score is not None:
                if self._best_score is None or self._is_better(current_score, self._best_score):
                    self._best_score = current_score
                    self._best_checkpoint = filepath
                    if self.verbose:
                        print(f"New best checkpoint: {self.monitor}={current_score:.4f}")

        # Cleanup old checkpoints
        self._cleanup()

        if self.verbose:
            print(f"Saved checkpoint: {filepath}")

        return filepath

    def _cleanup(self):
        """Remove old checkpoints if max_to_keep is exceeded."""
        if self.max_to_keep <= 0:
            return

        while len(self._checkpoints) > self.max_to_keep:
            # Don't remove best checkpoint
            oldest = self._checkpoints[0]
            if oldest['path'] == self._best_checkpoint:
                if len(self._checkpoints) > 1:
                    oldest = self._checkpoints[1]
                    self._checkpoints.pop(1)
                else:
                    break
            else:
                self._checkpoints.pop(0)

            # Remove file
            if os.path.exists(oldest['path']):
                os.remove(oldest['path'])
                if self.verbose:
                    print(f"Removed old checkpoint: {oldest['path']}")

    def load(
        self,
        filepath: Optional[str] = None,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Parameters
        ----------
        filepath : str, optional
            Path to checkpoint. If None, loads the latest.
        model : Any, optional
            Model to load state into.
        optimizer : Any, optional
            Optimizer to load state into.

        Returns
        -------
        Dict[str, Any]
            Loaded checkpoint state.

        Examples
        --------
        >>> state = manager.load('checkpoint.ckpt', model=model)
        """
        if filepath is None:
            filepath = self.latest

        if filepath is None:
            raise FileNotFoundError("No checkpoint found")

        state = load_checkpoint(filepath)

        # Load into model if provided
        if model is not None and 'model_state_dict' in state:
            if hasattr(model, 'load_state_dict'):
                model.load_state_dict(state['model_state_dict'])
            else:
                import brainstate
                param_states = model.states(brainstate.ParamState)
                for k, v in param_states.items():
                    key = str(k)
                    if key in state['model_state_dict']:
                        v.value = state['model_state_dict'][key]

        # Load into optimizer if provided
        if optimizer is not None and 'optimizer_state_dict' in state:
            if hasattr(optimizer, 'load_state_dict'):
                optimizer.load_state_dict(state['optimizer_state_dict'])

        if self.verbose:
            print(f"Loaded checkpoint: {filepath} (epoch {state.get('epoch', '?')})")

        return state

    def load_best(
        self,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load the best checkpoint based on monitored metric.

        Parameters
        ----------
        model : Any, optional
            Model to load state into.
        optimizer : Any, optional
            Optimizer to load state into.

        Returns
        -------
        Dict[str, Any]
            Loaded checkpoint state.
        """
        if self._best_checkpoint is None:
            # Try to find best checkpoint file
            best_path = find_checkpoint(self.dirpath, best=True)
            if best_path is None:
                raise FileNotFoundError("No best checkpoint found")
            self._best_checkpoint = best_path

        return self.load(self._best_checkpoint, model, optimizer)

    def load_latest(
        self,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load the most recent checkpoint.

        Parameters
        ----------
        model : Any, optional
            Model to load state into.
        optimizer : Any, optional
            Optimizer to load state into.

        Returns
        -------
        Dict[str, Any]
            Loaded checkpoint state.
        """
        return self.load(self.latest, model, optimizer)

    @property
    def latest(self) -> Optional[str]:
        """Path to the latest checkpoint."""
        if self._checkpoints:
            return self._checkpoints[-1]['path']

        # Search directory
        checkpoints = list_checkpoints(self.dirpath, sort_by='time')
        return checkpoints[-1] if checkpoints else None

    @property
    def best(self) -> Optional[str]:
        """Path to the best checkpoint."""
        return self._best_checkpoint

    @property
    def best_score(self) -> Optional[float]:
        """Best score achieved."""
        return self._best_score

    @property
    def checkpoints(self) -> List[Dict[str, Any]]:
        """List of tracked checkpoints."""
        return self._checkpoints.copy()

    def save_best_as(self, filepath: str):
        """
        Copy the best checkpoint to a new location.

        Parameters
        ----------
        filepath : str
            Destination path.
        """
        if self._best_checkpoint is None:
            raise FileNotFoundError("No best checkpoint to copy")

        shutil.copy2(self._best_checkpoint, filepath)
        if self.verbose:
            print(f"Copied best checkpoint to: {filepath}")

    def clear(self):
        """Remove all checkpoints."""
        for ckpt_info in self._checkpoints:
            if os.path.exists(ckpt_info['path']):
                os.remove(ckpt_info['path'])

        self._checkpoints.clear()
        self._best_checkpoint = None
        self._best_score = None

        if self.verbose:
            print("Cleared all checkpoints")


# Import JAX here to avoid circular imports
import jax
import jax.tree_util
