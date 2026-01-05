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
Callback system for training hooks.

This module provides a callback interface similar to PyTorch Lightning,
allowing users to hook into various stages of the training process.
"""

import os
import warnings
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jax.numpy as jnp

__all__ = [
    # Base class
    'Callback',
    'CallbackList',
    # Built-in callbacks
    'ModelCheckpoint',
    'EarlyStopping',
    'LearningRateMonitor',
    'GradientClipCallback',
    'Timer',
    'RichProgressBar',
    'TQDMProgressBar',
    'LambdaCallback',
    'PrintCallback',
]


class Callback(ABC):
    """
    Base class for callbacks.

    Callbacks allow you to hook into various stages of the training process.
    Override the methods you need in your custom callback subclass.

    Examples
    --------
    >>> class MyCallback(Callback):
    ...     def on_train_epoch_end(self, trainer, module):
    ...         print(f"Epoch {module.current_epoch} finished!")
    ...
    >>> trainer = Trainer(callbacks=[MyCallback()])
    """
    __module__ = 'braintools.trainer'

    @property
    def state_key(self) -> str:
        """Identifier for this callback in state dict."""
        return self.__class__.__qualname__

    # -------------------------------------------------------------------------
    # Fit hooks
    # -------------------------------------------------------------------------

    def on_fit_start(self, trainer: Any, module: Any):
        """Called at the very beginning of fit."""
        pass

    def on_fit_end(self, trainer: Any, module: Any):
        """Called at the very end of fit."""
        pass

    # -------------------------------------------------------------------------
    # Training hooks
    # -------------------------------------------------------------------------

    def on_train_start(self, trainer: Any, module: Any):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: Any, module: Any):
        """Called at the end of training."""
        pass

    def on_train_epoch_start(self, trainer: Any, module: Any):
        """Called at the beginning of each training epoch."""
        pass

    def on_train_epoch_end(self, trainer: Any, module: Any):
        """Called at the end of each training epoch."""
        pass

    def on_train_batch_start(
        self,
        trainer: Any,
        module: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Called at the beginning of each training batch."""
        pass

    def on_train_batch_end(
        self,
        trainer: Any,
        module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Called at the end of each training batch."""
        pass

    # -------------------------------------------------------------------------
    # Validation hooks
    # -------------------------------------------------------------------------

    def on_validation_start(self, trainer: Any, module: Any):
        """Called at the beginning of validation."""
        pass

    def on_validation_end(self, trainer: Any, module: Any):
        """Called at the end of validation."""
        pass

    def on_validation_epoch_start(self, trainer: Any, module: Any):
        """Called at the beginning of each validation epoch."""
        pass

    def on_validation_epoch_end(self, trainer: Any, module: Any):
        """Called at the end of each validation epoch."""
        pass

    def on_validation_batch_start(
        self,
        trainer: Any,
        module: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Called at the beginning of each validation batch."""
        pass

    def on_validation_batch_end(
        self,
        trainer: Any,
        module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Called at the end of each validation batch."""
        pass

    # -------------------------------------------------------------------------
    # Test hooks
    # -------------------------------------------------------------------------

    def on_test_start(self, trainer: Any, module: Any):
        """Called at the beginning of testing."""
        pass

    def on_test_end(self, trainer: Any, module: Any):
        """Called at the end of testing."""
        pass

    def on_test_epoch_start(self, trainer: Any, module: Any):
        """Called at the beginning of each test epoch."""
        pass

    def on_test_epoch_end(self, trainer: Any, module: Any):
        """Called at the end of each test epoch."""
        pass

    def on_test_batch_start(
        self,
        trainer: Any,
        module: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Called at the beginning of each test batch."""
        pass

    def on_test_batch_end(
        self,
        trainer: Any,
        module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Called at the end of each test batch."""
        pass

    # -------------------------------------------------------------------------
    # Predict hooks
    # -------------------------------------------------------------------------

    def on_predict_start(self, trainer: Any, module: Any):
        """Called at the beginning of prediction."""
        pass

    def on_predict_end(self, trainer: Any, module: Any):
        """Called at the end of prediction."""
        pass

    def on_predict_batch_start(
        self,
        trainer: Any,
        module: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Called at the beginning of each predict batch."""
        pass

    def on_predict_batch_end(
        self,
        trainer: Any,
        module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Called at the end of each predict batch."""
        pass

    # -------------------------------------------------------------------------
    # Optimization hooks
    # -------------------------------------------------------------------------

    def on_before_optimizer_step(
        self,
        trainer: Any,
        module: Any,
        optimizer: Any,
    ):
        """Called before each optimizer step."""
        pass

    def on_after_optimizer_step(
        self,
        trainer: Any,
        module: Any,
        optimizer: Any,
    ):
        """Called after each optimizer step."""
        pass

    def on_before_backward(self, trainer: Any, module: Any, loss: Any):
        """Called before backward pass (gradient computation)."""
        pass

    def on_after_backward(self, trainer: Any, module: Any):
        """Called after backward pass (gradient computation)."""
        pass

    # -------------------------------------------------------------------------
    # Checkpointing hooks
    # -------------------------------------------------------------------------

    def on_save_checkpoint(self, trainer: Any, module: Any, checkpoint: Dict):
        """Called when saving a checkpoint."""
        pass

    def on_load_checkpoint(self, trainer: Any, module: Any, checkpoint: Dict):
        """Called when loading a checkpoint."""
        pass

    # -------------------------------------------------------------------------
    # State dict methods
    # -------------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from state dict."""
        pass


class CallbackList:
    """
    Container for multiple callbacks.

    This class manages a list of callbacks and provides methods to invoke
    all callbacks for a given hook.

    Parameters
    ----------
    callbacks : List[Callback], optional
        List of callbacks to manage.
    """
    __module__ = 'braintools.trainer'

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def append(self, callback: Callback):
        """Add a callback."""
        self.callbacks.append(callback)

    def __iter__(self):
        return iter(self.callbacks)

    def __len__(self):
        return len(self.callbacks)

    def _call_hook(self, hook_name: str, *args, **kwargs):
        """Call a hook on all callbacks."""
        for callback in self.callbacks:
            hook = getattr(callback, hook_name, None)
            if hook is not None:
                hook(*args, **kwargs)

    # Convenience methods for common hooks
    def on_fit_start(self, trainer, module):
        self._call_hook('on_fit_start', trainer, module)

    def on_fit_end(self, trainer, module):
        self._call_hook('on_fit_end', trainer, module)

    def on_train_epoch_start(self, trainer, module):
        self._call_hook('on_train_epoch_start', trainer, module)

    def on_train_epoch_end(self, trainer, module):
        self._call_hook('on_train_epoch_end', trainer, module)

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        self._call_hook('on_train_batch_start', trainer, module, batch, batch_idx)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self._call_hook('on_train_batch_end', trainer, module, outputs, batch, batch_idx)

    def on_validation_epoch_start(self, trainer, module):
        self._call_hook('on_validation_epoch_start', trainer, module)

    def on_validation_epoch_end(self, trainer, module):
        self._call_hook('on_validation_epoch_end', trainer, module)

    def on_validation_batch_start(self, trainer, module, batch, batch_idx):
        self._call_hook('on_validation_batch_start', trainer, module, batch, batch_idx)

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self._call_hook('on_validation_batch_end', trainer, module, outputs, batch, batch_idx)


class ModelCheckpoint(Callback):
    """
    Save model checkpoints based on monitored metric.

    Parameters
    ----------
    dirpath : str, optional
        Directory to save checkpoints. Default: current directory.
    filename : str, optional
        Checkpoint filename template. Can include {epoch}, {step}, and metric names.
        Default: 'checkpoint-{epoch:02d}-{val_loss:.4f}'
    monitor : str, default='val_loss'
        Metric to monitor for best model selection.
    mode : str, default='min'
        One of 'min' or 'max'. In 'min' mode, the lowest metric value is best.
    save_top_k : int, default=3
        Number of best models to keep. -1 means keep all.
    save_last : bool, default=True
        Whether to save the last checkpoint regardless of metric.
    every_n_epochs : int, default=1
        Save checkpoint every n epochs.
    every_n_train_steps : int, optional
        Save checkpoint every n training steps.
    save_on_train_epoch_end : bool, default=True
        Whether to run checkpointing at end of training epoch.
    verbose : bool, default=False
        Whether to print checkpoint saving messages.

    Examples
    --------
    >>> checkpoint_callback = ModelCheckpoint(
    ...     dirpath='checkpoints/',
    ...     filename='model-{epoch:02d}-{val_loss:.4f}',
    ...     monitor='val_loss',
    ...     mode='min',
    ...     save_top_k=3,
    ... )
    >>> trainer = Trainer(callbacks=[checkpoint_callback])
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_top_k: int = 3,
        save_last: bool = True,
        every_n_epochs: int = 1,
        every_n_train_steps: Optional[int] = None,
        save_on_train_epoch_end: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.dirpath = dirpath or os.getcwd()
        self.filename = filename or 'checkpoint-{epoch:02d}'
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.every_n_epochs = every_n_epochs
        self.every_n_train_steps = every_n_train_steps
        self.save_on_train_epoch_end = save_on_train_epoch_end
        self.verbose = verbose

        # Validation
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        # State
        self.best_score: Optional[float] = None
        self.best_model_path: Optional[str] = None
        self.best_k_models: Dict[str, float] = {}  # path -> score
        self._last_global_step_saved = -1

        # Create directory
        Path(self.dirpath).mkdir(parents=True, exist_ok=True)

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == 'min':
            return current < best
        return current > best

    def _format_checkpoint_name(
        self,
        epoch: int,
        step: int,
        metrics: Dict[str, Any],
    ) -> str:
        """Format the checkpoint filename."""
        format_dict = {'epoch': epoch, 'step': step}
        format_dict.update(metrics)

        try:
            filename = self.filename.format(**format_dict)
        except KeyError:
            # Fall back to simple format if metrics not available
            filename = f'checkpoint-epoch={epoch:02d}-step={step}'

        return filename

    def _save_checkpoint(self, trainer: Any, module: Any, filepath: str):
        """Save a checkpoint."""
        from braintools.file import msgpack_from_state_dict

        checkpoint = {
            'epoch': module.current_epoch,
            'global_step': module.global_step,
            'model_state_dict': module.state_dict(),
            'callbacks': {},
        }

        # Save optimizer state if available
        if hasattr(trainer, 'optimizers') and trainer.optimizers:
            opt_states = []
            for opt in trainer.optimizers:
                if hasattr(opt, 'state_dict'):
                    opt_states.append(opt.state_dict())
            checkpoint['optimizer_state_dict'] = opt_states

        # Save logged metrics
        if hasattr(trainer, 'logged_metrics'):
            checkpoint['metrics'] = trainer.logged_metrics

        # Call save hooks
        for callback in trainer.callbacks:
            callback.on_save_checkpoint(trainer, module, checkpoint)
            cb_state = callback.state_dict()
            if cb_state:
                checkpoint['callbacks'][callback.state_key] = cb_state

        # Serialize and save
        msgpack_from_state_dict(checkpoint, filepath)

        if self.verbose:
            print(f"Saved checkpoint to {filepath}")

    def _remove_checkpoint(self, filepath: str):
        """Remove a checkpoint file."""
        if os.path.exists(filepath):
            os.remove(filepath)
            if self.verbose:
                print(f"Removed checkpoint {filepath}")

    def _update_best_k(self, score: float, filepath: str):
        """Update the best k models."""
        self.best_k_models[filepath] = score

        if self.save_top_k > 0 and len(self.best_k_models) > self.save_top_k:
            # Find the worst model to remove
            if self.mode == 'min':
                worst_path = max(self.best_k_models, key=self.best_k_models.get)
            else:
                worst_path = min(self.best_k_models, key=self.best_k_models.get)

            self._remove_checkpoint(worst_path)
            del self.best_k_models[worst_path]

    def on_train_epoch_end(self, trainer: Any, module: Any):
        """Check if we should save a checkpoint at epoch end."""
        if not self.save_on_train_epoch_end:
            return

        epoch = module.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        # Get monitored metric
        metrics = {}
        if hasattr(trainer, 'callback_metrics'):
            metrics = trainer.callback_metrics
        elif hasattr(trainer, 'logged_metrics'):
            metrics = trainer.logged_metrics

        current_score = metrics.get(self.monitor)

        # Build filepath
        filename = self._format_checkpoint_name(epoch, module.global_step, metrics)
        filepath = os.path.join(self.dirpath, f'{filename}.ckpt')

        # Check if this is the best model
        if current_score is not None:
            if self.best_score is None or self._is_better(current_score, self.best_score):
                self.best_score = current_score
                self.best_model_path = filepath

            self._update_best_k(current_score, filepath)
            self._save_checkpoint(trainer, module, filepath)
        elif self.save_top_k == -1 or self.save_top_k > 0:
            # No metric to monitor, just save
            self._save_checkpoint(trainer, module, filepath)

    def on_train_batch_end(
        self,
        trainer: Any,
        module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Check if we should save a checkpoint at step."""
        if self.every_n_train_steps is None:
            return

        step = module.global_step
        if step == self._last_global_step_saved:
            return

        if step % self.every_n_train_steps == 0:
            metrics = {}
            if hasattr(trainer, 'logged_metrics'):
                metrics = trainer.logged_metrics

            filename = self._format_checkpoint_name(
                module.current_epoch, step, metrics
            )
            filepath = os.path.join(self.dirpath, f'{filename}.ckpt')

            self._save_checkpoint(trainer, module, filepath)
            self._last_global_step_saved = step

    def on_fit_end(self, trainer: Any, module: Any):
        """Save last checkpoint if configured."""
        if self.save_last:
            filepath = os.path.join(self.dirpath, 'last.ckpt')
            self._save_checkpoint(trainer, module, filepath)

    def state_dict(self) -> Dict[str, Any]:
        return {
            'best_score': self.best_score,
            'best_model_path': self.best_model_path,
            'best_k_models': self.best_k_models.copy(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.best_score = state_dict.get('best_score')
        self.best_model_path = state_dict.get('best_model_path')
        self.best_k_models = state_dict.get('best_k_models', {})


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.

    Parameters
    ----------
    monitor : str, default='val_loss'
        Metric to monitor.
    mode : str, default='min'
        One of 'min' or 'max'. In 'min' mode, training stops when the
        quantity monitored has stopped decreasing.
    patience : int, default=3
        Number of epochs with no improvement after which training will be stopped.
    min_delta : float, default=0.0
        Minimum change to qualify as an improvement.
    verbose : bool, default=False
        Whether to print early stopping messages.
    strict : bool, default=True
        If True, raise error if monitor metric not found.
    check_finite : bool, default=True
        Stop training if the metric becomes NaN or infinite.

    Examples
    --------
    >>> early_stop = EarlyStopping(
    ...     monitor='val_loss',
    ...     patience=5,
    ...     mode='min',
    ... )
    >>> trainer = Trainer(callbacks=[early_stop])
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        monitor: str = 'val_loss',
        mode: str = 'min',
        patience: int = 3,
        min_delta: float = 0.0,
        verbose: bool = False,
        strict: bool = True,
        check_finite: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.strict = strict
        self.check_finite = check_finite

        # Validation
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        # State
        self.best_score: Optional[float] = None
        self.wait_count: int = 0
        self.stopped_epoch: int = 0
        self._should_stop: bool = False

        # Adjust min_delta sign based on mode
        if self.mode == 'min':
            self.min_delta *= -1

    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current is an improvement over best."""
        if self.mode == 'min':
            return current < best - self.min_delta
        return current > best + self.min_delta

    def on_validation_epoch_end(self, trainer: Any, module: Any):
        """Check if training should stop."""
        # Get monitored metric
        metrics = {}
        if hasattr(trainer, 'callback_metrics'):
            metrics = trainer.callback_metrics
        elif hasattr(trainer, 'logged_metrics'):
            metrics = trainer.logged_metrics

        current = metrics.get(self.monitor)

        if current is None:
            if self.strict:
                raise RuntimeError(
                    f"EarlyStopping conditioned on metric '{self.monitor}' "
                    f"which is not available. Available metrics: {list(metrics.keys())}"
                )
            return

        # Check for NaN/Inf
        if self.check_finite and not jnp.isfinite(current):
            self._should_stop = True
            self.stopped_epoch = module.current_epoch
            if self.verbose:
                print(f"EarlyStopping: Metric '{self.monitor}' became non-finite.")
            return

        # Check for improvement
        if self.best_score is None:
            self.best_score = current
            return

        if self._is_improvement(current, self.best_score):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.verbose:
                print(
                    f"EarlyStopping: {self.monitor} did not improve "
                    f"({self.wait_count}/{self.patience})"
                )

            if self.wait_count >= self.patience:
                self._should_stop = True
                self.stopped_epoch = module.current_epoch
                if self.verbose:
                    print(
                        f"EarlyStopping: Stopping training at epoch {module.current_epoch}. "
                        f"Best {self.monitor}: {self.best_score:.4f}"
                    )

    @property
    def should_stop(self) -> bool:
        """Whether training should stop."""
        return self._should_stop

    def state_dict(self) -> Dict[str, Any]:
        return {
            'best_score': self.best_score,
            'wait_count': self.wait_count,
            'stopped_epoch': self.stopped_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.best_score = state_dict.get('best_score')
        self.wait_count = state_dict.get('wait_count', 0)
        self.stopped_epoch = state_dict.get('stopped_epoch', 0)


class LearningRateMonitor(Callback):
    """
    Monitor and log learning rate during training.

    Parameters
    ----------
    logging_interval : str, default='step'
        When to log the learning rate. One of 'step' or 'epoch'.
    log_momentum : bool, default=False
        Whether to also log momentum values.

    Examples
    --------
    >>> lr_monitor = LearningRateMonitor(logging_interval='epoch')
    >>> trainer = Trainer(callbacks=[lr_monitor])
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        logging_interval: str = 'step',
        log_momentum: bool = False,
    ):
        super().__init__()
        self.logging_interval = logging_interval
        self.log_momentum = log_momentum
        self._lr_history: List[Dict[str, float]] = []

    def _get_learning_rates(self, trainer: Any) -> Dict[str, float]:
        """Extract learning rates from optimizers."""
        lrs = {}
        if hasattr(trainer, 'optimizers'):
            for i, opt in enumerate(trainer.optimizers):
                if hasattr(opt, 'lr'):
                    lr = opt.lr
                    if hasattr(lr, 'value'):
                        lr = lr.value
                    elif callable(lr):
                        lr = lr()
                    lrs[f'lr-opt{i}'] = float(lr)
                elif hasattr(opt, 'learning_rate'):
                    lrs[f'lr-opt{i}'] = float(opt.learning_rate)
        return lrs

    def on_train_batch_start(
        self,
        trainer: Any,
        module: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Log learning rate at step start."""
        if self.logging_interval != 'step':
            return

        lrs = self._get_learning_rates(trainer)
        self._lr_history.append(lrs)

        # Log to module
        for name, lr in lrs.items():
            module.log(name, lr, prog_bar=False, logger=True)

    def on_train_epoch_start(self, trainer: Any, module: Any):
        """Log learning rate at epoch start."""
        if self.logging_interval != 'epoch':
            return

        lrs = self._get_learning_rates(trainer)
        self._lr_history.append(lrs)

        # Log to module
        for name, lr in lrs.items():
            module.log(name, lr, prog_bar=True, logger=True)

    @property
    def lr_history(self) -> List[Dict[str, float]]:
        """History of logged learning rates."""
        return self._lr_history


class GradientClipCallback(Callback):
    """
    Clip gradients during training.

    Note: This is usually handled by the Trainer's gradient_clip_val parameter,
    but this callback provides more control and logging.

    Parameters
    ----------
    clip_val : float, optional
        Maximum gradient norm or value.
    clip_algorithm : str, default='norm'
        Clipping algorithm. One of 'norm' (global norm) or 'value' (element-wise).
    log_grad_norm : bool, default=False
        Whether to log the gradient norm before clipping.

    Examples
    --------
    >>> grad_clip = GradientClipCallback(clip_val=1.0, log_grad_norm=True)
    >>> trainer = Trainer(callbacks=[grad_clip])
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        clip_val: Optional[float] = None,
        clip_algorithm: str = 'norm',
        log_grad_norm: bool = False,
    ):
        super().__init__()
        self.clip_val = clip_val
        self.clip_algorithm = clip_algorithm
        self.log_grad_norm = log_grad_norm

        if clip_algorithm not in ('norm', 'value'):
            raise ValueError(f"clip_algorithm must be 'norm' or 'value', got '{clip_algorithm}'")

    def on_before_optimizer_step(
        self,
        trainer: Any,
        module: Any,
        optimizer: Any,
    ):
        """Clip gradients before optimizer step."""
        if self.clip_val is None:
            return

        # Note: Gradient clipping is typically done in the Trainer or Optimizer
        # This callback is for logging and custom clipping scenarios
        pass


class Timer(Callback):
    """
    Track training time.

    Parameters
    ----------
    duration : Optional[Dict[str, float]]
        Maximum training duration. Keys can be 'seconds', 'minutes', 'hours', 'days'.
    verbose : bool, default=True
        Whether to print timing information.

    Examples
    --------
    >>> timer = Timer(duration={'hours': 2})  # Stop after 2 hours
    >>> trainer = Trainer(callbacks=[timer])
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        duration: Optional[Dict[str, float]] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.duration = duration
        self.verbose = verbose

        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._epoch_start_time: Optional[float] = None
        self._epoch_times: List[float] = []
        self._should_stop: bool = False

        # Calculate max duration in seconds
        self._max_seconds: Optional[float] = None
        if duration:
            self._max_seconds = (
                duration.get('seconds', 0) +
                duration.get('minutes', 0) * 60 +
                duration.get('hours', 0) * 3600 +
                duration.get('days', 0) * 86400
            )

    def on_fit_start(self, trainer: Any, module: Any):
        """Record start time."""
        import time
        self._start_time = time.time()

    def on_fit_end(self, trainer: Any, module: Any):
        """Record end time and print summary."""
        import time
        self._end_time = time.time()

        if self.verbose and self._start_time:
            duration = self._end_time - self._start_time
            print(f"\nTotal training time: {self._format_time(duration)}")
            if self._epoch_times:
                avg_epoch = sum(self._epoch_times) / len(self._epoch_times)
                print(f"Average epoch time: {self._format_time(avg_epoch)}")

    def on_train_epoch_start(self, trainer: Any, module: Any):
        """Record epoch start time."""
        import time
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer: Any, module: Any):
        """Record epoch time and check duration limit."""
        import time
        if self._epoch_start_time:
            epoch_time = time.time() - self._epoch_start_time
            self._epoch_times.append(epoch_time)

            if self.verbose:
                print(f"Epoch {module.current_epoch} time: {self._format_time(epoch_time)}")

        # Check if we should stop
        if self._max_seconds and self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed >= self._max_seconds:
                self._should_stop = True
                if self.verbose:
                    print(f"Timer: Stopping training after {self._format_time(elapsed)}")

    @property
    def should_stop(self) -> bool:
        """Whether training should stop due to time limit."""
        return self._should_stop

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"

    @property
    def time_elapsed(self) -> Optional[float]:
        """Time elapsed since training started (seconds)."""
        import time
        if self._start_time:
            return time.time() - self._start_time
        return None

    @property
    def time_remaining(self) -> Optional[float]:
        """Estimated time remaining (seconds)."""
        if self._max_seconds and self.time_elapsed:
            return max(0, self._max_seconds - self.time_elapsed)
        return None


class RichProgressBar(Callback):
    """
    Rich progress bar for training visualization.

    Requires the 'rich' package to be installed.

    Parameters
    ----------
    refresh_rate : int, default=1
        Number of batches between progress bar updates.
    leave : bool, default=False
        Whether to leave the progress bar after completion.

    Examples
    --------
    >>> progress = RichProgressBar()
    >>> trainer = Trainer(callbacks=[progress], enable_progress_bar=False)
    """
    __module__ = 'braintools.trainer'

    def __init__(self, refresh_rate: int = 1, leave: bool = False):
        super().__init__()
        self.refresh_rate = refresh_rate
        self.leave = leave
        self._progress = None
        self._task_id = None

    def on_train_epoch_start(self, trainer: Any, module: Any):
        """Create progress bar for epoch."""
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            )
            self._progress.start()

            total = len(trainer.train_dataloader) if hasattr(trainer, 'train_dataloader') else None
            self._task_id = self._progress.add_task(
                f"Epoch {module.current_epoch}", total=total
            )
        except ImportError:
            warnings.warn("rich package not installed. Using simple progress.")

    def on_train_batch_end(
        self,
        trainer: Any,
        module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Update progress bar."""
        if self._progress and self._task_id is not None:
            if batch_idx % self.refresh_rate == 0:
                self._progress.update(self._task_id, advance=self.refresh_rate)

    def on_train_epoch_end(self, trainer: Any, module: Any):
        """Complete progress bar."""
        if self._progress:
            self._progress.stop()
            self._progress = None
            self._task_id = None


class TQDMProgressBar(Callback):
    """
    TQDM progress bar for training visualization.

    Requires the 'tqdm' package to be installed.

    Parameters
    ----------
    refresh_rate : int, default=1
        Number of batches between progress bar updates.
    process_position : int, default=0
        Position of the progress bar.
    leave : bool, default=True
        Whether to leave the progress bar after completion.

    Examples
    --------
    >>> progress = TQDMProgressBar()
    >>> trainer = Trainer(callbacks=[progress], enable_progress_bar=False)
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        refresh_rate: int = 1,
        process_position: int = 0,
        leave: bool = True,
    ):
        super().__init__()
        self.refresh_rate = refresh_rate
        self.process_position = process_position
        self.leave = leave
        self._pbar = None

    def on_train_epoch_start(self, trainer: Any, module: Any):
        """Create progress bar for epoch."""
        try:
            from tqdm import tqdm
            total = len(trainer.train_dataloader) if hasattr(trainer, 'train_dataloader') else None
            self._pbar = tqdm(
                total=total,
                desc=f"Epoch {module.current_epoch}",
                position=self.process_position,
                leave=self.leave,
            )
        except ImportError:
            warnings.warn("tqdm package not installed. Progress bar disabled.")

    def on_train_batch_end(
        self,
        trainer: Any,
        module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Update progress bar."""
        if self._pbar:
            if batch_idx % self.refresh_rate == 0:
                # Update postfix with metrics
                metrics = module._get_prog_bar_metrics()
                if metrics:
                    self._pbar.set_postfix(metrics)
                self._pbar.update(self.refresh_rate)

    def on_train_epoch_end(self, trainer: Any, module: Any):
        """Close progress bar."""
        if self._pbar:
            self._pbar.close()
            self._pbar = None


class LambdaCallback(Callback):
    """
    Create a callback from lambda functions.

    Parameters
    ----------
    **kwargs
        Hook names mapped to callable functions.

    Examples
    --------
    >>> callback = LambdaCallback(
    ...     on_train_epoch_end=lambda trainer, module: print(f"Epoch {module.current_epoch} done!")
    ... )
    >>> trainer = Trainer(callbacks=[callback])
    """
    __module__ = 'braintools.trainer'

    def __init__(self, **kwargs):
        super().__init__()
        for name, fn in kwargs.items():
            if not callable(fn):
                raise ValueError(f"Expected callable for {name}, got {type(fn)}")
            setattr(self, name, fn)


class PrintCallback(Callback):
    """
    Simple callback that prints training progress.

    Parameters
    ----------
    print_freq : int, default=100
        Print every n batches.

    Examples
    --------
    >>> callback = PrintCallback(print_freq=50)
    >>> trainer = Trainer(callbacks=[callback])
    """
    __module__ = 'braintools.trainer'

    def __init__(self, print_freq: int = 100):
        super().__init__()
        self.print_freq = print_freq

    def on_train_epoch_start(self, trainer: Any, module: Any):
        print(f"\n--- Epoch {module.current_epoch} ---")

    def on_train_batch_end(
        self,
        trainer: Any,
        module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        if batch_idx % self.print_freq == 0:
            metrics = module._get_prog_bar_metrics()
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"  Step {batch_idx}: {metrics_str}")

    def on_train_epoch_end(self, trainer: Any, module: Any):
        print(f"Epoch {module.current_epoch} completed.")

    def on_validation_epoch_end(self, trainer: Any, module: Any):
        metrics = {}
        if hasattr(trainer, 'callback_metrics'):
            metrics = trainer.callback_metrics
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"  Validation: {metrics_str}")
