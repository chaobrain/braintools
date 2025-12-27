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
Trainer - Main orchestration class for training loops.

This module provides the Trainer class which handles the full training loop,
including validation, testing, and prediction.
"""

import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp

import brainstate
from brainstate import ParamState
from brainstate.typing import PyTree

from braintools.optim import Optimizer, UniqueStateManager

from ._callbacks import Callback, CallbackList, EarlyStopping, ModelCheckpoint
from ._checkpoint import CheckpointManager
from ._dataloader import DataLoader
from ._distributed import Strategy, get_strategy
from ._loggers import Logger, CSVLogger, CompositeLogger
from ._module import LightningModule
from ._progress import get_progress_bar, MetricsDisplay

__all__ = [
    'Trainer',
    'TrainerState',
]


class TrainerState:
    """
    Container for trainer state during training.

    Attributes
    ----------
    epoch : int
        Current epoch.
    global_step : int
        Total number of training steps.
    stage : str
        Current stage ('train', 'validate', 'test', 'predict').
    """
    __module__ = 'braintools.trainer'

    def __init__(self):
        self.epoch: int = 0
        self.global_step: int = 0
        self.stage: str = 'train'
        self.batch_idx: int = 0
        self.should_stop: bool = False


class Trainer:
    """
    Orchestrates the training process.

    The Trainer handles the training loop, validation, testing, and prediction,
    integrating callbacks, logging, checkpointing, and distributed training.

    Parameters
    ----------
    max_epochs : int, default=1000
        Maximum number of training epochs.
    min_epochs : int, default=1
        Minimum number of training epochs.
    max_steps : int, default=-1
        Maximum number of training steps. -1 means no limit.
    val_check_interval : int or float, default=1.0
        How often to run validation within a training epoch.
        Integer = every N batches, float = fraction of epoch.
    check_val_every_n_epoch : int, default=1
        Run validation every N epochs.
    callbacks : List[Callback], optional
        List of callbacks to use.
    logger : Logger or List[Logger] or bool, default=True
        Logger(s) to use. True = CSVLogger, False = no logging.
    enable_progress_bar : bool, default=True
        Whether to show progress bars.
    enable_checkpointing : bool, default=True
        Whether to enable automatic checkpointing.
    default_root_dir : str, optional
        Default root directory for logs and checkpoints.
    gradient_clip_val : float, optional
        Value for gradient clipping.
    gradient_clip_algorithm : str, default='norm'
        Gradient clipping algorithm ('norm' or 'value').
    accumulate_grad_batches : int, default=1
        Number of batches to accumulate gradients over.
    devices : int or List[int] or str, default='auto'
        Devices to use for training.
    strategy : str or Strategy, default='auto'
        Distributed training strategy.
    precision : str, default='32'
        Training precision ('32', '16', 'bf16').
    deterministic : bool, default=False
        Whether to use deterministic algorithms.
    benchmark : bool, default=False
        Whether to enable benchmarking mode.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    Basic usage:

    >>> trainer = Trainer(max_epochs=10)
    >>> trainer.fit(model, train_loader, val_loader)

    With callbacks and logging:

    >>> trainer = Trainer(
    ...     max_epochs=100,
    ...     callbacks=[
    ...         ModelCheckpoint(dirpath='checkpoints/', monitor='val_loss'),
    ...         EarlyStopping(monitor='val_loss', patience=5),
    ...     ],
    ...     logger=TensorBoardLogger('logs/'),
    ... )
    >>> trainer.fit(model, train_loader, val_loader)
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        max_epochs: int = 1000,
        min_epochs: int = 1,
        max_steps: int = -1,
        val_check_interval: Union[int, float] = 1.0,
        check_val_every_n_epoch: int = 1,
        callbacks: Optional[List[Callback]] = None,
        logger: Union[Logger, List[Logger], bool] = True,
        enable_progress_bar: bool = True,
        enable_checkpointing: bool = True,
        default_root_dir: Optional[str] = None,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: str = 'norm',
        accumulate_grad_batches: int = 1,
        devices: Union[int, List[int], str] = 'auto',
        strategy: Union[str, Strategy] = 'auto',
        precision: str = '32',
        deterministic: bool = False,
        benchmark: bool = False,
        seed: Optional[int] = None,
    ):
        # Training config
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.max_steps = max_steps
        self.val_check_interval = val_check_interval
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.accumulate_grad_batches = accumulate_grad_batches
        self.precision = precision
        self.deterministic = deterministic
        self.benchmark = benchmark
        self.seed = seed

        # Setup directories
        self.default_root_dir = default_root_dir or os.getcwd()
        Path(self.default_root_dir).mkdir(parents=True, exist_ok=True)

        # Setup callbacks
        self._callbacks = CallbackList(callbacks or [])
        self.enable_checkpointing = enable_checkpointing

        # Setup logging
        self.enable_progress_bar = enable_progress_bar
        self._setup_logger(logger)

        # Setup distributed
        self._setup_devices(devices)
        self.strategy = get_strategy(strategy)

        # Training state
        self.state = TrainerState()
        self.model: Optional[LightningModule] = None
        self.optimizers: List[Optimizer] = []
        self.schedulers: List[Any] = []
        self.param_states: Optional[PyTree] = None

        # Data loaders
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloaders: Optional[List[DataLoader]] = None
        self.test_dataloaders: Optional[List[DataLoader]] = None
        self.predict_dataloaders: Optional[List[DataLoader]] = None

        # Metrics tracking
        self.callback_metrics: Dict[str, Any] = {}
        self.logged_metrics: Dict[str, Any] = {}
        self._epoch_metrics: Dict[str, List[Any]] = {}

        # Checkpoint manager
        self._checkpoint_manager: Optional[CheckpointManager] = None

        # Set random seed
        if seed is not None:
            import numpy as np
            np.random.seed(seed)

    def _setup_logger(self, logger: Union[Logger, List[Logger], bool]):
        """Setup logging."""
        if logger is True:
            log_dir = os.path.join(self.default_root_dir, 'logs')
            self.loggers = [CSVLogger(log_dir)]
        elif logger is False:
            self.loggers = []
        elif isinstance(logger, Logger):
            self.loggers = [logger]
        elif isinstance(logger, list):
            self.loggers = logger
        else:
            self.loggers = []

    def _setup_devices(self, devices: Union[int, List[int], str]):
        """Setup devices for training."""
        if devices == 'auto':
            self.devices = jax.devices()
        elif isinstance(devices, int):
            self.devices = jax.devices()[:devices]
        elif isinstance(devices, list):
            all_devices = jax.devices()
            self.devices = [all_devices[i] for i in devices]
        else:
            self.devices = jax.devices()

        self.num_devices = len(self.devices)

    @property
    def callbacks(self) -> List[Callback]:
        """List of callbacks."""
        return list(self._callbacks.callbacks)

    @property
    def current_epoch(self) -> int:
        """Current epoch."""
        return self.state.epoch

    @property
    def global_step(self) -> int:
        """Global step count."""
        return self.state.global_step

    @property
    def is_training(self) -> bool:
        """Whether currently in training stage."""
        return self.state.stage == 'train'

    # =========================================================================
    # Setup Methods
    # =========================================================================

    def _setup_model(self, model: LightningModule):
        """Setup model for training."""
        self.model = model
        model.trainer = self

        # Get parameter states
        param_states = model.states(ParamState)
        self.param_states = UniqueStateManager(param_states).to_pytree()

    def _setup_optimizers(self):
        """Setup optimizers from model.configure_optimizers()."""
        if self.model is None:
            raise RuntimeError("Model not set up")

        opt_config = self.model.configure_optimizers()

        # Parse optimizer configuration
        if isinstance(opt_config, Optimizer):
            self.optimizers = [opt_config]
            self.schedulers = []
        elif isinstance(opt_config, tuple) and len(opt_config) == 2:
            opts, scheds = opt_config
            if isinstance(opts, Optimizer):
                self.optimizers = [opts]
            else:
                self.optimizers = list(opts)
            if isinstance(scheds, (list, tuple)):
                self.schedulers = list(scheds)
            else:
                self.schedulers = [scheds] if scheds else []
        elif isinstance(opt_config, dict):
            self.optimizers = [opt_config['optimizer']]
            self.schedulers = [opt_config.get('lr_scheduler')]
        else:
            raise ValueError(f"Invalid optimizer configuration: {type(opt_config)}")

        # Register parameters with optimizers
        for opt in self.optimizers:
            opt.register_trainable_weights(self.param_states)

    def _setup_checkpoint_manager(self):
        """Setup checkpoint manager."""
        if self.enable_checkpointing:
            ckpt_dir = os.path.join(self.default_root_dir, 'checkpoints')
            self._checkpoint_manager = CheckpointManager(
                dirpath=ckpt_dir,
                max_to_keep=5,
            )

    def _setup_strategy(self):
        """Setup distributed strategy."""
        if self.model is not None and self.optimizers:
            self.model, self.optimizers[0] = self.strategy.setup(
                self.model, self.optimizers[0]
            )

    # =========================================================================
    # Training Methods
    # =========================================================================

    def fit(
        self,
        model: LightningModule,
        train_dataloaders: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        ckpt_path: Optional[str] = None,
    ):
        """
        Run the full training loop.

        Parameters
        ----------
        model : LightningModule
            Model to train.
        train_dataloaders : DataLoader, optional
            Training data loader.
        val_dataloaders : DataLoader or List[DataLoader], optional
            Validation data loader(s).
        ckpt_path : str, optional
            Path to checkpoint to resume from.

        Examples
        --------
        >>> trainer.fit(model, train_loader, val_loader)
        """
        # Setup
        self._setup_model(model)
        self._setup_optimizers()
        self._setup_checkpoint_manager()
        self._setup_strategy()

        # Store data loaders
        self.train_dataloader = train_dataloaders
        if val_dataloaders is not None:
            if isinstance(val_dataloaders, DataLoader):
                self.val_dataloaders = [val_dataloaders]
            else:
                self.val_dataloaders = list(val_dataloaders)

        # Load checkpoint if provided
        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)

        # Create JIT-compiled training step
        train_step_fn = self._create_train_step()

        # Run training
        try:
            self._run_fit(train_step_fn)
        finally:
            self._cleanup()

    def _create_train_step(self) -> Callable:
        """Create JIT-compiled training step function."""
        model = self.model
        optimizer = self.optimizers[0] if self.optimizers else None
        param_states = self.param_states
        gradient_clip_val = self.gradient_clip_val
        gradient_clip_algorithm = self.gradient_clip_algorithm

        @brainstate.transform.jit
        def train_step(batch, batch_idx):
            """Single training step."""
            # Compute loss
            def loss_fn():
                outputs = model.training_step(batch, batch_idx)
                if isinstance(outputs, dict):
                    loss = outputs['loss']
                else:
                    loss = outputs.loss
                return loss

            loss = loss_fn()

            # Compute gradients
            grads = brainstate.transform.grad(loss_fn, grad_states=param_states)()

            # Clip gradients if needed
            if gradient_clip_val is not None:
                if gradient_clip_algorithm == 'norm':
                    grads = _clip_grad_norm(grads, gradient_clip_val)
                else:
                    grads = _clip_grad_value(grads, gradient_clip_val)

            # Update parameters
            if optimizer is not None:
                optimizer.step(grads)

            return loss

        return train_step

    def _run_fit(self, train_step_fn: Callable):
        """Run the fit loop."""
        model = self.model
        state = self.state

        # Callbacks: fit start
        self._callbacks.on_fit_start(self, model)
        model.on_fit_start()

        # Log hyperparameters
        if self.loggers:
            hparams = {
                'max_epochs': self.max_epochs,
                'gradient_clip_val': self.gradient_clip_val,
                'num_devices': self.num_devices,
            }
            for logger in self.loggers:
                logger.log_hyperparams(hparams)

        # Progress display
        display = MetricsDisplay() if self.enable_progress_bar else None
        if display:
            display.print_training_start(
                model.__class__.__name__,
                max_epochs=self.max_epochs,
            )

        start_time = time.time()

        # Training loop
        for epoch in range(self.max_epochs):
            state.epoch = epoch
            model.current_epoch = epoch

            # Check early stopping
            if self._should_stop():
                break

            # Run training epoch
            self._run_train_epoch(train_step_fn, epoch)

            # Run validation
            if self._should_validate(epoch):
                self._run_validation_epoch(epoch)

            # Update schedulers
            for scheduler in self.schedulers:
                if scheduler is not None and hasattr(scheduler, 'step'):
                    scheduler.step()

            # Print epoch summary
            if display:
                train_metrics = {k: v for k, v in self.callback_metrics.items()
                                 if k.startswith('train_')}
                val_metrics = {k: v for k, v in self.callback_metrics.items()
                               if k.startswith('val_')}
                display.print_epoch_summary(epoch, train_metrics, val_metrics)

            # Check max steps
            if self.max_steps > 0 and state.global_step >= self.max_steps:
                break

            # Min epochs check
            if epoch < self.min_epochs - 1:
                continue

        # Callbacks: fit end
        self._callbacks.on_fit_end(self, model)
        model.on_fit_end()

        # Final summary
        if display:
            display.print_training_end(
                best_metrics=self.callback_metrics,
                total_time=time.time() - start_time,
            )

        # Finalize loggers
        for logger in self.loggers:
            logger.finalize()

    def _run_train_epoch(self, train_step_fn: Callable, epoch: int):
        """Run a single training epoch."""
        model = self.model
        state = self.state
        state.stage = 'train'

        # Callbacks: epoch start
        self._callbacks.on_train_epoch_start(self, model)
        model.on_train_epoch_start()

        # Reset epoch metrics
        self._epoch_metrics.clear()

        # Progress bar
        pbar = None
        if self.enable_progress_bar and self.train_dataloader is not None:
            pbar = get_progress_bar()
            pbar.start(
                total=len(self.train_dataloader),
                desc=f'Epoch {epoch}',
            )

        # Training loop
        for batch_idx, batch in enumerate(self.train_dataloader):
            state.batch_idx = batch_idx

            # Callbacks: batch start
            self._callbacks.on_train_batch_start(self, model, batch, batch_idx)
            model.on_train_batch_start(batch, batch_idx)

            # Reset logged metrics
            model._reset_logged_metrics()

            # Training step (JIT-compiled)
            loss = train_step_fn(batch, batch_idx)

            # Get outputs - only use direct output from JIT function
            # Note: We don't use _get_logger_metrics() here because the values
            # stored during JIT tracing are tracers, not concrete values.
            # Metrics should be returned as part of the training_step output.
            outputs = {'loss': loss}

            # Callbacks: batch end
            self._callbacks.on_train_batch_end(self, model, outputs, batch, batch_idx)
            model.on_train_batch_end(outputs, batch, batch_idx)

            # Accumulate metrics - convert to scalars outside JIT
            for key, value in outputs.items():
                if key not in self._epoch_metrics:
                    self._epoch_metrics[key] = []
                # Safe conversion to scalar (outside JIT)
                if hasattr(value, 'item'):
                    value = value.item()
                elif hasattr(value, '__float__'):
                    try:
                        value = float(value)
                    except Exception:
                        continue
                self._epoch_metrics[key].append(float(value))

            # Update progress bar with accumulated loss
            if pbar is not None:
                pbar.update(1)
                # Show loss in progress bar (use the concrete value we just computed)
                pbar_metrics = {'loss': float(loss) if hasattr(loss, '__float__') else loss}
                pbar.set_postfix(pbar_metrics)

            # Log metrics
            self._log_metrics(outputs, state.global_step)

            state.global_step += 1

            # Check max steps
            if self.max_steps > 0 and state.global_step >= self.max_steps:
                break

            # Validation check interval
            if self._should_validate_batch(batch_idx):
                self._run_validation_epoch(epoch)

        # Close progress bar
        if pbar is not None:
            pbar.close()

        # Aggregate epoch metrics
        for key, values in self._epoch_metrics.items():
            if values:
                self.callback_metrics[f'train_{key}'] = sum(values) / len(values)

        # Callbacks: epoch end
        self._callbacks.on_train_epoch_end(self, model)
        model.on_train_epoch_end()

    def _run_validation_epoch(self, epoch: int):
        """Run validation epoch."""
        if self.val_dataloaders is None:
            return

        model = self.model
        state = self.state
        state.stage = 'validate'

        # Callbacks: validation start
        self._callbacks.on_validation_epoch_start(self, model)
        model.on_validation_epoch_start()

        all_metrics: Dict[str, List[Any]] = {}

        for dataloader in self.val_dataloaders:
            # Progress bar
            pbar = None
            if self.enable_progress_bar:
                pbar = get_progress_bar()
                pbar.start(total=len(dataloader), desc='Validation')

            for batch_idx, batch in enumerate(dataloader):
                state.batch_idx = batch_idx

                # Callbacks: batch start
                self._callbacks.on_validation_batch_start(self, model, batch, batch_idx)
                model.on_validation_batch_start(batch, batch_idx)

                # Reset logged metrics
                model._reset_logged_metrics()

                # Validation step
                outputs = model.validation_step(batch, batch_idx)

                if outputs is not None:
                    # Get logged metrics
                    logged = model._get_logger_metrics()
                    if isinstance(outputs, dict):
                        logged.update(outputs)
                    else:
                        logged.update(outputs.metrics)

                    # Accumulate
                    for key, value in logged.items():
                        if key not in all_metrics:
                            all_metrics[key] = []
                        if hasattr(value, 'item'):
                            value = value.item()
                        all_metrics[key].append(float(value))

                # Callbacks: batch end
                self._callbacks.on_validation_batch_end(self, model, outputs, batch, batch_idx)
                model.on_validation_batch_end(outputs, batch, batch_idx)

                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(model._get_prog_bar_metrics())

            # Close progress bar
            if pbar is not None:
                pbar.close()

        # Aggregate metrics
        for key, values in all_metrics.items():
            if values:
                self.callback_metrics[f'val_{key}'] = sum(values) / len(values)
                self.logged_metrics[f'val_{key}'] = self.callback_metrics[f'val_{key}']

        # Log validation metrics
        self._log_metrics(
            {k: v for k, v in self.callback_metrics.items() if k.startswith('val_')},
            state.global_step
        )

        # Callbacks: validation end
        self._callbacks.on_validation_epoch_end(self, model)
        model.on_validation_epoch_end()

    def _should_stop(self) -> bool:
        """Check if training should stop."""
        # Check callbacks for early stopping
        for callback in self._callbacks:
            if hasattr(callback, 'should_stop') and callback.should_stop:
                return True
        return self.state.should_stop

    def _should_validate(self, epoch: int) -> bool:
        """Check if validation should run this epoch."""
        if self.val_dataloaders is None:
            return False
        return (epoch + 1) % self.check_val_every_n_epoch == 0

    def _should_validate_batch(self, batch_idx: int) -> bool:
        """Check if validation should run after this batch."""
        if self.val_dataloaders is None:
            return False
        if isinstance(self.val_check_interval, float):
            return False  # Only check at epoch end for float
        return (batch_idx + 1) % self.val_check_interval == 0

    def _log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to all loggers."""
        if not self.loggers:
            return

        # Convert to float
        log_metrics = {}
        for key, value in metrics.items():
            if hasattr(value, 'item'):
                log_metrics[key] = value.item()
            elif isinstance(value, (int, float)):
                log_metrics[key] = float(value)

        for logger in self.loggers:
            logger.log_metrics(log_metrics, step)

    def _load_checkpoint(self, ckpt_path: str):
        """Load checkpoint."""
        from ._checkpoint import load_checkpoint
        state = load_checkpoint(ckpt_path)

        if self.model is not None and 'model_state_dict' in state:
            self.model.load_state_dict(state['model_state_dict'])

        if self.optimizers and 'optimizer_state_dict' in state:
            for i, opt_state in enumerate(state['optimizer_state_dict']):
                if i < len(self.optimizers):
                    self.optimizers[i].load_state_dict(opt_state)

        self.state.epoch = state.get('epoch', 0)
        self.state.global_step = state.get('step', 0)

    def _cleanup(self):
        """Cleanup after training."""
        self.model = None
        self.optimizers = []
        self.schedulers = []
        self.param_states = None

    # =========================================================================
    # Validation, Test, and Predict Methods
    # =========================================================================

    def validate(
        self,
        model: Optional[LightningModule] = None,
        dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run validation.

        Parameters
        ----------
        model : LightningModule, optional
            Model to validate. Uses self.model if not provided.
        dataloaders : DataLoader or List[DataLoader], optional
            Validation data loader(s).
        ckpt_path : str, optional
            Path to checkpoint to load.
        verbose : bool, default=True
            Whether to print results.

        Returns
        -------
        Dict[str, Any]
            Validation metrics.
        """
        if model is not None:
            self._setup_model(model)
        elif self.model is None:
            raise RuntimeError("No model provided")

        if dataloaders is not None:
            if isinstance(dataloaders, DataLoader):
                self.val_dataloaders = [dataloaders]
            else:
                self.val_dataloaders = list(dataloaders)

        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)

        self._run_validation_epoch(0)

        if verbose:
            print("\nValidation Results:")
            for key, value in self.callback_metrics.items():
                if key.startswith('val_'):
                    print(f"  {key}: {value:.4f}")

        return {k: v for k, v in self.callback_metrics.items() if k.startswith('val_')}

    def test(
        self,
        model: Optional[LightningModule] = None,
        dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run testing.

        Parameters
        ----------
        model : LightningModule, optional
            Model to test.
        dataloaders : DataLoader or List[DataLoader], optional
            Test data loader(s).
        ckpt_path : str, optional
            Path to checkpoint to load.
        verbose : bool, default=True
            Whether to print results.

        Returns
        -------
        Dict[str, Any]
            Test metrics.
        """
        if model is not None:
            self._setup_model(model)
        elif self.model is None:
            raise RuntimeError("No model provided")

        if dataloaders is not None:
            if isinstance(dataloaders, DataLoader):
                self.test_dataloaders = [dataloaders]
            else:
                self.test_dataloaders = list(dataloaders)

        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)

        # Run test loop (similar to validation)
        model = self.model
        state = self.state
        state.stage = 'test'

        model.on_test_start()

        all_metrics: Dict[str, List[Any]] = {}

        for dataloader in self.test_dataloaders:
            pbar = None
            if self.enable_progress_bar:
                pbar = get_progress_bar()
                pbar.start(total=len(dataloader), desc='Testing')

            for batch_idx, batch in enumerate(dataloader):
                model.on_test_batch_start(batch, batch_idx)
                model._reset_logged_metrics()

                outputs = model.test_step(batch, batch_idx)

                if outputs is not None:
                    logged = model._get_logger_metrics()
                    if isinstance(outputs, dict):
                        logged.update(outputs)
                    else:
                        logged.update(outputs.metrics)

                    for key, value in logged.items():
                        if key not in all_metrics:
                            all_metrics[key] = []
                        if hasattr(value, 'item'):
                            value = value.item()
                        all_metrics[key].append(float(value))

                model.on_test_batch_end(outputs, batch, batch_idx)

                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

        model.on_test_end()

        # Aggregate metrics
        test_metrics = {}
        for key, values in all_metrics.items():
            if values:
                test_metrics[f'test_{key}'] = sum(values) / len(values)

        if verbose:
            print("\nTest Results:")
            for key, value in test_metrics.items():
                print(f"  {key}: {value:.4f}")

        return test_metrics

    def predict(
        self,
        model: Optional[LightningModule] = None,
        dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        ckpt_path: Optional[str] = None,
    ) -> List[Any]:
        """
        Run prediction.

        Parameters
        ----------
        model : LightningModule, optional
            Model to use for prediction.
        dataloaders : DataLoader or List[DataLoader], optional
            Prediction data loader(s).
        ckpt_path : str, optional
            Path to checkpoint to load.

        Returns
        -------
        List[Any]
            Predictions for each batch.
        """
        if model is not None:
            self._setup_model(model)
        elif self.model is None:
            raise RuntimeError("No model provided")

        if dataloaders is not None:
            if isinstance(dataloaders, DataLoader):
                self.predict_dataloaders = [dataloaders]
            else:
                self.predict_dataloaders = list(dataloaders)

        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)

        model = self.model
        state = self.state
        state.stage = 'predict'

        model.on_predict_start()

        all_predictions = []

        for dataloader in self.predict_dataloaders:
            pbar = None
            if self.enable_progress_bar:
                pbar = get_progress_bar()
                pbar.start(total=len(dataloader), desc='Predicting')

            for batch_idx, batch in enumerate(dataloader):
                model.on_predict_batch_start(batch, batch_idx)

                outputs = model.predict_step(batch, batch_idx)
                all_predictions.append(outputs)

                model.on_predict_batch_end(outputs, batch, batch_idx)

                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

        model.on_predict_end()

        return all_predictions


# =============================================================================
# Utility Functions
# =============================================================================

def _clip_grad_norm(grads: PyTree, max_norm: float) -> PyTree:
    """Clip gradients by global norm."""
    import optax
    return optax.clip_by_global_norm(max_norm).update(grads, None)[0]


def _clip_grad_value(grads: PyTree, max_value: float) -> PyTree:
    """Clip gradients by value."""
    return jax.tree.map(
        lambda g: jnp.clip(g, -max_value, max_value),
        grads
    )
