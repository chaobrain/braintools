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
    Convert a value to a Python scalar if it is a 0-d/size-1 array.

    Multi-element arrays are returned unchanged rather than raising, so callers
    can safely pass through metrics that happen to be vectors. This is safe to
    call outside of JIT-traced code.
    """
    # Plain Python scalars pass through unchanged.
    if isinstance(value, (bool, int, float)):
        return value

    # Only scalar (size-1) arrays can be converted; anything larger is returned
    # as-is. ``getattr`` guards against objects that lack a ``size`` attribute.
    size = getattr(value, 'size', None)
    if size is not None and size != 1:
        return value

    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass

    # Last resort: objects that define ``__float__`` (but not strings/bytes,
    # which would coerce surprisingly or raise).
    if not isinstance(value, (str, bytes)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
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
    Define a model, then train it with :class:`Trainer`:

    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import brainstate
        >>> import braintools
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
        ...         loss = jnp.mean((self(x) - y) ** 2)
        ...         self.log('mse', loss)
        ...         return {'loss': loss}
        ...
        ...     def configure_optimizers(self):
        ...         return braintools.optim.Adam(lr=1e-3)
        >>>
        >>> brainstate.random.seed(0)
        >>> model = MyModel(4, 8, 1)
        >>> data = {'x': np.ones((16, 4), np.float32), 'y': np.zeros((16, 1), np.float32)}
        >>> loader = braintools.trainer.DataLoader(data, batch_size=8)
        >>> trainer = braintools.trainer.Trainer(
        ...     max_epochs=2, logger=False, enable_progress_bar=False,
        ...     enable_checkpointing=False)
        >>> trainer.fit(model, loader)
        >>> model.current_epoch, trainer.global_step
        (1, 4)
        >>> sorted(trainer.callback_metrics)  # epoch means, prefixed with 'train_'
        ['train_loss', 'train_mse']

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

        # Names of parameters that should not be trained. The Trainer excludes
        # these from gradient computation and optimizer updates.
        self._frozen_params: set = set()

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
                        # ``jax.Array.devices()`` returns a set, which is not
                        # subscriptable; take an arbitrary element instead.
                        return next(iter(devices))
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

    def on_before_optimizer_step(self, optimizer: Optional[Optimizer]):
        """Called before each optimizer step (``optimizer`` is None when none is configured)."""
        pass

    def on_after_optimizer_step(self, optimizer: Optional[Optimizer]):
        """Called after each optimizer step (``optimizer`` is None when none is configured)."""
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
        Call it from a step method (``self.log('train_loss', loss)``); here it is
        called directly to show what gets recorded:

        .. code-block:: python

            >>> import braintools
            >>> model = braintools.trainer.LightningModule()
            >>> model.log('train_loss', 0.25, prog_bar=True)
            >>> model.log('val_acc', 0.75, on_step=False, on_epoch=True)
            >>> sorted(model.logged_metrics)
            ['train_loss', 'val_acc']
            >>> model.logged_metrics['val_acc']['on_epoch']
            True
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
        .. code-block:: python

            >>> import braintools
            >>> model = braintools.trainer.LightningModule()
            >>> model.log_dict({'val_loss': 0.5, 'val_acc': 0.75}, prog_bar=True)
            >>> {k: v['value'] for k, v in model.logged_metrics.items()}
            {'val_loss': 0.5, 'val_acc': 0.75}
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

    def freeze(self, names: Optional[Union[str, List[str]]] = None):
        """Freeze parameters so they are not updated during training.

        ``brainstate.ParamState`` has no ``requires_grad`` flag, so freezing is
        recorded by name on the module. The :class:`Trainer` reads these names
        when it sets up training and excludes the corresponding states from both
        gradient computation and optimizer updates. Call this *before*
        ``trainer.fit(model)``.

        Parameters
        ----------
        names : str or list of str, optional
            Parameter name(s) to freeze. If ``None`` (default), all parameters
            are frozen. Names correspond to the keys of
            ``model.states(brainstate.ParamState)``.

        See Also
        --------
        unfreeze : Reverse the effect of this method.

        Examples
        --------
        Freeze everything, then re-enable the decoder:

        .. code-block:: python

            >>> import brainstate
            >>> import braintools
            >>> class AutoEncoder(braintools.trainer.LightningModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.encoder = brainstate.nn.Linear(4, 2)
            ...         self.decoder = brainstate.nn.Linear(2, 4)
            >>> model = AutoEncoder()
            >>> sorted(model.states(brainstate.ParamState).keys())
            [('decoder', 'weight'), ('encoder', 'weight')]
            >>> model.freeze()
            >>> model.unfreeze(('decoder', 'weight'))
            >>> sorted(model.frozen_parameters)
            [('encoder', 'weight')]
            >>> model.is_frozen(('decoder', 'weight'))
            False
        """
        param_states = self.states(brainstate.ParamState)
        if names is None:
            self._frozen_params = set(param_states.keys())
        else:
            self._frozen_params.update(self._coerce_param_names(names))

    def unfreeze(self, names: Optional[Union[str, List[str]]] = None):
        """Unfreeze parameters so they are trained again.

        Parameters
        ----------
        names : str or list of str, optional
            Parameter name(s) to unfreeze. If ``None`` (default), all
            parameters are unfrozen.

        See Also
        --------
        freeze : Mark parameters as non-trainable.
        """
        if names is None:
            self._frozen_params.clear()
        else:
            self._frozen_params.difference_update(self._coerce_param_names(names))

    def _coerce_param_names(self, names: Any) -> List[Any]:
        """Normalize ``names`` into a list of parameter keys.

        Parameter keys may be plain strings or tuples (e.g. ``('lin',
        'weight')``). A single key is wrapped in a list; an iterable of keys is
        returned as a list. The ambiguity between a single tuple-key and a list
        of keys is resolved by checking membership in the model's actual
        parameter keys.
        """
        if isinstance(names, str):
            return [names]
        param_keys = set(self.states(brainstate.ParamState).keys())
        try:
            if names in param_keys:
                return [names]
        except TypeError:
            # Unhashable (e.g. a list) -> it is a collection of names.
            pass
        return list(names)

    def is_frozen(self, name: Any) -> bool:
        """Return whether the parameter ``name`` is currently frozen."""
        return name in self._frozen_params

    @property
    def frozen_parameters(self) -> set:
        """Set of parameter names currently frozen."""
        return set(self._frozen_params)

    def print_summary(self, input_shape: Optional[Tuple[int, ...]] = None):
        """
        Print a summary of the model architecture.

        Parameters
        ----------
        input_shape : Tuple[int, ...], optional
            Reserved for future shape-inference support. Currently unused; the
            summary reports parameter counts from the model's existing states
            regardless of this argument.
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
