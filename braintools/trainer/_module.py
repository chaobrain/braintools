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
LightningModule - Base class for training modules.

This module provides a PyTorch Lightning-like interface for defining models
that can be trained with the Trainer class.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import brainstate
from brainstate.typing import PyTree

from braintools.optim import Optimizer, LRScheduler

__all__ = [
    'LightningModule',
    'TrainOutput',
    'EvalOutput',
]


def _to_scalar(value: Any) -> Any:
    """
    Convert a value to a Python scalar if it's a JAX array.

    This is safe to call outside of JIT-traced code.
    """
    import jax.numpy as jnp

    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            return float(value)
    elif isinstance(value, jnp.ndarray) and value.ndim == 0:
        return float(value)
    return value


class TrainOutput:
    """
    Output container for training step.

    Parameters
    ----------
    loss : Any
        The loss value to be used for gradient computation.
    metrics : Dict[str, Any], optional
        Additional metrics to log.
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        loss: Any,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        self.loss = loss
        self.metrics = metrics or {}

    def __getitem__(self, key: str) -> Any:
        if key == 'loss':
            return self.loss
        return self.metrics.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == 'loss':
            return self.loss
        return self.metrics.get(key, default)


class EvalOutput:
    """
    Output container for evaluation steps (validation/test).

    Parameters
    ----------
    metrics : Dict[str, Any]
        Metrics computed during evaluation.
    """
    __module__ = 'braintools.trainer'

    def __init__(self, metrics: Optional[Dict[str, Any]] = None):
        self.metrics = metrics or {}

    def __getitem__(self, key: str) -> Any:
        return self.metrics.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metrics.get(key, default)


class LightningModule(brainstate.nn.Module):
    """
    Base class for all training modules.

    This class provides a PyTorch Lightning-like interface for defining models
    that work with the Trainer class. Users should subclass this and implement
    at minimum ``training_step`` and ``configure_optimizers``.

    Parameters
    ----------
    None

    Attributes
    ----------
    trainer : Trainer
        Reference to the Trainer instance (set during training).
    current_epoch : int
        Current training epoch (0-indexed).
    global_step : int
        Total number of training steps across all epochs.
    logged_metrics : Dict[str, Any]
        Metrics logged during the current step.

    Examples
    --------
    Basic usage:

    >>> import braintools
    >>> import brainstate
    >>> import jax.numpy as jnp
    >>>
    >>> class MyModel(braintools.trainer.LightningModule):
    ...     def __init__(self, input_size, hidden_size, output_size):
    ...         super().__init__()
    ...         self.linear1 = brainstate.nn.Linear(input_size, hidden_size)
    ...         self.linear2 = brainstate.nn.Linear(hidden_size, output_size)
    ...
    ...     def __call__(self, x):
    ...         x = jax.nn.relu(self.linear1(x))
    ...         return self.linear2(x)
    ...
    ...     def training_step(self, batch, batch_idx):
    ...         x, y = batch['x'], batch['y']
    ...         logits = self(x)
    ...         loss = jnp.mean((logits - y) ** 2)
    ...         self.log('train_loss', loss)
    ...         return {'loss': loss}
    ...
    ...     def configure_optimizers(self):
    ...         return braintools.optim.Adam(lr=1e-3)

    See Also
    --------
    Trainer : The training orchestration class.
    """
    __module__ = 'braintools.trainer'

    def __init__(self):
        super().__init__()
        # Training state
        self._trainer: Optional[Any] = None
        self._current_epoch: int = 0
        self._global_step: int = 0

        # Logging state (reset each step)
        self._logged_metrics: Dict[str, Any] = {}
        self._prog_bar_metrics: Dict[str, Any] = {}
        self._logger_metrics: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def trainer(self) -> Optional[Any]:
        """Reference to the Trainer instance."""
        return self._trainer

    @trainer.setter
    def trainer(self, value: Any):
        self._trainer = value

    @property
    def current_epoch(self) -> int:
        """Current training epoch (0-indexed)."""
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value: int):
        self._current_epoch = value

    @property
    def global_step(self) -> int:
        """Total number of training steps across all epochs."""
        return self._global_step

    @global_step.setter
    def global_step(self, value: int):
        self._global_step = value

    @property
    def logged_metrics(self) -> Dict[str, Any]:
        """Metrics logged during the current step."""
        return self._logged_metrics.copy()

    @property
    def device(self) -> Any:
        """The device this module is on (inferred from parameters)."""
        params = self.states(brainstate.ParamState)
        if params:
            # Get first parameter and check its device
            import jax
            for state in params.values():
                if hasattr(state, 'value') and hasattr(state.value, 'devices'):
                    devices = state.value.devices()
                    if devices:
                        return devices[0]
        return None

    # -------------------------------------------------------------------------
    # Methods to Override
    # -------------------------------------------------------------------------

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> Union[Dict[str, Any], TrainOutput]:
        """
        Compute the training loss for a single batch.

        Override this method to define your training logic.

        Parameters
        ----------
        batch : Any
            A batch of data from the train dataloader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        Dict[str, Any] or TrainOutput
            A dictionary with at least a 'loss' key, or a TrainOutput instance.
            The loss will be used for gradient computation.

        Examples
        --------
        >>> def training_step(self, batch, batch_idx):
        ...     x, y = batch['x'], batch['y']
        ...     logits = self(x)
        ...     loss = cross_entropy(logits, y)
        ...     self.log('train_loss', loss, prog_bar=True)
        ...     return {'loss': loss}
        """
        raise NotImplementedError(
            "You must implement `training_step` in your LightningModule subclass."
        )

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> Optional[Union[Dict[str, Any], EvalOutput]]:
        """
        Compute validation metrics for a single batch.

        Override this method to define your validation logic.

        Parameters
        ----------
        batch : Any
            A batch of data from the validation dataloader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        Dict[str, Any] or EvalOutput or None
            A dictionary of metrics, an EvalOutput instance, or None.

        Examples
        --------
        >>> def validation_step(self, batch, batch_idx):
        ...     x, y = batch['x'], batch['y']
        ...     logits = self(x)
        ...     loss = cross_entropy(logits, y)
        ...     acc = (logits.argmax(-1) == y).mean()
        ...     self.log_dict({'val_loss': loss, 'val_acc': acc})
        ...     return {'val_loss': loss, 'val_acc': acc}
        """
        pass

    def test_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> Optional[Union[Dict[str, Any], EvalOutput]]:
        """
        Compute test metrics for a single batch.

        Override this method to define your test logic.

        Parameters
        ----------
        batch : Any
            A batch of data from the test dataloader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        Dict[str, Any] or EvalOutput or None
            A dictionary of metrics, an EvalOutput instance, or None.
        """
        pass

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> Any:
        """
        Compute predictions for a single batch.

        Override this method to define your prediction logic.

        Parameters
        ----------
        batch : Any
            A batch of data from the predict dataloader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        Any
            The predictions for this batch.
        """
        pass

    def configure_optimizers(
        self,
    ) -> Union[
        Optimizer,
        Tuple[Optimizer, LRScheduler],
        Tuple[List[Optimizer], List[LRScheduler]],
        Dict[str, Any],
    ]:
        """
        Configure optimizer(s) and learning rate scheduler(s).

        Override this method to define your optimization setup.

        Returns
        -------
        Optimizer
            A single optimizer.
        Tuple[Optimizer, LRScheduler]
            An optimizer and a learning rate scheduler.
        Tuple[List[Optimizer], List[LRScheduler]]
            Multiple optimizers and schedulers.
        Dict[str, Any]
            A dictionary with 'optimizer' and optionally 'lr_scheduler' keys.

        Examples
        --------
        Simple optimizer:

        >>> def configure_optimizers(self):
        ...     return braintools.optim.Adam(lr=1e-3)

        Optimizer with scheduler:

        >>> def configure_optimizers(self):
        ...     optimizer = braintools.optim.Adam(lr=1e-2)
        ...     scheduler = braintools.optim.StepLR(base_lr=1e-2, step_size=10)
        ...     return optimizer, scheduler

        Multiple optimizers (e.g., for GANs):

        >>> def configure_optimizers(self):
        ...     opt_g = braintools.optim.Adam(lr=1e-4)
        ...     opt_d = braintools.optim.Adam(lr=4e-4)
        ...     return [opt_g, opt_d], []
        """
        raise NotImplementedError(
            "You must implement `configure_optimizers` in your LightningModule subclass."
        )

    # -------------------------------------------------------------------------
    # Hooks (Optional Overrides)
    # -------------------------------------------------------------------------

    def on_fit_start(self):
        """Called at the very beginning of fit."""
        pass

    def on_fit_end(self):
        """Called at the very end of fit."""
        pass

    def on_train_start(self):
        """Called at the beginning of training."""
        pass

    def on_train_end(self):
        """Called at the end of training."""
        pass

    def on_train_epoch_start(self):
        """Called at the beginning of each training epoch."""
        pass

    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        pass

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        """Called at the beginning of each training batch."""
        pass

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int):
        """Called at the end of each training batch."""
        pass

    def on_validation_start(self):
        """Called at the beginning of validation."""
        pass

    def on_validation_end(self):
        """Called at the end of validation."""
        pass

    def on_validation_epoch_start(self):
        """Called at the beginning of each validation epoch."""
        pass

    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch."""
        pass

    def on_validation_batch_start(self, batch: Any, batch_idx: int):
        """Called at the beginning of each validation batch."""
        pass

    def on_validation_batch_end(self, outputs: Any, batch: Any, batch_idx: int):
        """Called at the end of each validation batch."""
        pass

    def on_test_start(self):
        """Called at the beginning of testing."""
        pass

    def on_test_end(self):
        """Called at the end of testing."""
        pass

    def on_test_epoch_start(self):
        """Called at the beginning of each test epoch."""
        pass

    def on_test_epoch_end(self):
        """Called at the end of each test epoch."""
        pass

    def on_test_batch_start(self, batch: Any, batch_idx: int):
        """Called at the beginning of each test batch."""
        pass

    def on_test_batch_end(self, outputs: Any, batch: Any, batch_idx: int):
        """Called at the end of each test batch."""
        pass

    def on_predict_start(self):
        """Called at the beginning of prediction."""
        pass

    def on_predict_end(self):
        """Called at the end of prediction."""
        pass

    def on_predict_batch_start(self, batch: Any, batch_idx: int):
        """Called at the beginning of each predict batch."""
        pass

    def on_predict_batch_end(self, outputs: Any, batch: Any, batch_idx: int):
        """Called at the end of each predict batch."""
        pass

    def on_before_optimizer_step(self, optimizer: Optimizer):
        """Called before each optimizer step."""
        pass

    def on_after_optimizer_step(self, optimizer: Optimizer):
        """Called after each optimizer step."""
        pass

    def on_before_backward(self, loss: Any):
        """Called before backward pass (gradient computation)."""
        pass

    def on_after_backward(self):
        """Called after backward pass (gradient computation)."""
        pass

    # -------------------------------------------------------------------------
    # Logging Methods
    # -------------------------------------------------------------------------

    def log(
        self,
        name: str,
        value: Any,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: bool = True,
        on_epoch: bool = False,
        reduce_fx: str = 'mean',
        sync_dist: bool = False,
    ):
        """
        Log a single metric.

        Parameters
        ----------
        name : str
            Name of the metric.
        value : Any
            Value to log. Should be a scalar or JAX array.
        prog_bar : bool, default=False
            Whether to show this metric in the progress bar.
        logger : bool, default=True
            Whether to log to the logger(s).
        on_step : bool, default=True
            Whether to log at each step.
        on_epoch : bool, default=False
            Whether to log at epoch end (accumulated).
        reduce_fx : str, default='mean'
            Reduction function for epoch-level metrics ('mean', 'sum', 'min', 'max').
        sync_dist : bool, default=False
            Whether to synchronize across devices in distributed training.

        Examples
        --------
        >>> self.log('train_loss', loss, prog_bar=True)
        >>> self.log('val_acc', accuracy, on_step=False, on_epoch=True)
        """
        # Note: We do NOT convert to scalar here because this method may be called
        # inside JIT-traced code. The conversion to scalar happens later when
        # metrics are retrieved for display/logging (outside of JIT).

        # Store the metric (value may be a JAX array or tracer)
        self._logged_metrics[name] = {
            'value': value,
            'prog_bar': prog_bar,
            'logger': logger,
            'on_step': on_step,
            'on_epoch': on_epoch,
            'reduce_fx': reduce_fx,
            'sync_dist': sync_dist,
        }

        if prog_bar:
            self._prog_bar_metrics[name] = value

        if logger:
            self._logger_metrics[name] = value

    def log_dict(
        self,
        metrics: Dict[str, Any],
        prog_bar: bool = False,
        logger: bool = True,
        on_step: bool = True,
        on_epoch: bool = False,
        reduce_fx: str = 'mean',
        sync_dist: bool = False,
    ):
        """
        Log multiple metrics at once.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Dictionary of metric names to values.
        prog_bar : bool, default=False
            Whether to show these metrics in the progress bar.
        logger : bool, default=True
            Whether to log to the logger(s).
        on_step : bool, default=True
            Whether to log at each step.
        on_epoch : bool, default=False
            Whether to log at epoch end (accumulated).
        reduce_fx : str, default='mean'
            Reduction function for epoch-level metrics.
        sync_dist : bool, default=False
            Whether to synchronize across devices in distributed training.

        Examples
        --------
        >>> self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True)
        """
        for name, value in metrics.items():
            self.log(
                name=name,
                value=value,
                prog_bar=prog_bar,
                logger=logger,
                on_step=on_step,
                on_epoch=on_epoch,
                reduce_fx=reduce_fx,
                sync_dist=sync_dist,
            )

    def _reset_logged_metrics(self):
        """Reset logged metrics for a new step. Called internally by Trainer."""
        self._logged_metrics.clear()
        self._prog_bar_metrics.clear()
        self._logger_metrics.clear()

    def _get_prog_bar_metrics(self) -> Dict[str, Any]:
        """Get metrics for progress bar display."""
        return {k: _to_scalar(v) for k, v in self._prog_bar_metrics.items()}

    def _get_logger_metrics(self) -> Dict[str, Any]:
        """Get metrics for logger."""
        return {k: _to_scalar(v) for k, v in self._logger_metrics.items()}

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state dictionary for checkpointing.

        Returns
        -------
        Dict[str, Any]
            State dictionary containing all parameter states.
        """
        param_states = self.states(brainstate.ParamState)
        state_dict = {}
        for name, state in param_states.items():
            if hasattr(state, 'value'):
                state_dict[str(name)] = state.value
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state from a state dictionary.

        Parameters
        ----------
        state_dict : Dict[str, Any]
            State dictionary to load.
        """
        param_states = self.states(brainstate.ParamState)
        for name, state in param_states.items():
            name_str = str(name)
            if name_str in state_dict and hasattr(state, 'value'):
                state.value = state_dict[name_str]

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def freeze(self):
        """Freeze all parameters (make non-trainable)."""
        param_states = self.states(brainstate.ParamState)
        for state in param_states.values():
            if hasattr(state, 'requires_grad'):
                state.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters (make trainable)."""
        param_states = self.states(brainstate.ParamState)
        for state in param_states.values():
            if hasattr(state, 'requires_grad'):
                state.requires_grad = True

    def print_summary(self, input_shape: Optional[Tuple[int, ...]] = None):
        """
        Print a summary of the model architecture.

        Parameters
        ----------
        input_shape : Tuple[int, ...], optional
            Shape of input tensor for shape inference.
        """
        print(f"\n{'='*60}")
        print(f"Model: {self.__class__.__name__}")
        print(f"{'='*60}")

        param_states = self.states(brainstate.ParamState)
        total_params = 0
        trainable_params = 0

        for name, state in param_states.items():
            if hasattr(state, 'value'):
                import jax.numpy as jnp
                params = jnp.size(state.value)
                total_params += params
                trainable_params += params  # In brainstate, all ParamState are trainable
                print(f"  {name}: {state.value.shape}")

        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"{'='*60}\n")
