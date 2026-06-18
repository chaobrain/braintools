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
from ._module import LightningModule, _to_scalar
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
        Number of batches to accumulate gradients over before each optimizer
        step. Must be >= 1.
    devices : int or List[int] or str, default='auto'
        Devices to use for training. ``'auto'`` uses all visible JAX devices;
        an ``int`` selects the first N; a list selects by index.
    strategy : str or Strategy, default='auto'
        Distributed training strategy. The active training loop drives a single
        optimizer on the selected device(s); multi-host strategies are provided
        but not exercised by :meth:`fit`.
    precision : str, default='32'
        Requested compute precision (``'32'``, ``'16'``, ``'bf16'``, optionally
        with a ``'-mixed'``/``'-true'`` suffix). Recorded for API compatibility;
        the loop currently computes in float32 and emits a warning for any
        non-32 value. Cast your model/data explicitly for reduced precision.
    deterministic : bool, default=False
        Recorded for API compatibility. Determinism in JAX is governed by the
        PRNG keys you use and XLA flags; this flag does not itself change
        execution.
    benchmark : bool, default=False
        Recorded for API compatibility; has no effect (XLA autotunes and there
        is no cuDNN benchmark switch in JAX). A warning is emitted if set.
    seed : int, optional
        Random seed for reproducibility. Seeds both NumPy and ``brainstate``.

    Notes
    -----
    Only the first optimizer returned by
    :meth:`LightningModule.configure_optimizers` is stepped by the training
    loop; configuring several emits a warning. Freeze parameters with
    :meth:`LightningModule.freeze` *before* calling :meth:`fit` to exclude them
    from gradients and optimizer updates.

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
        if gradient_clip_algorithm not in ('norm', 'value'):
            raise ValueError(
                f"gradient_clip_algorithm must be 'norm' or 'value', "
                f"got {gradient_clip_algorithm!r}"
            )
        self.gradient_clip_algorithm = gradient_clip_algorithm
        if accumulate_grad_batches < 1:
            raise ValueError(
                f"accumulate_grad_batches must be >= 1, got {accumulate_grad_batches}"
            )
        self.accumulate_grad_batches = accumulate_grad_batches

        # Precision is accepted for API compatibility but mixed precision is not
        # yet wired into the JAX compute path; warn rather than silently mislead.
        self.precision = str(precision)
        if self.precision not in ('32', '32-true', '16', '16-mixed', 'bf16', 'bf16-mixed'):
            raise ValueError(
                f"Unsupported precision {precision!r}. Expected one of "
                f"'32', '16', 'bf16' (optionally with a '-mixed'/'-true' suffix)."
            )
        if self.precision not in ('32', '32-true'):
            warnings.warn(
                f"precision={precision!r} is recorded but not yet applied: the "
                f"trainer currently computes in float32. Cast your model/data "
                f"explicitly if you need reduced precision.",
                stacklevel=2,
            )

        self.deterministic = deterministic
        self.benchmark = benchmark
        if benchmark:
            warnings.warn(
                "benchmark=True has no effect: XLA already performs autotuning "
                "and there is no cuDNN benchmark flag to toggle in JAX.",
                stacklevel=2,
            )
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

        # Set random seed across NumPy and brainstate so runs are reproducible.
        if seed is not None:
            import numpy as np
            np.random.seed(seed)
            brainstate.random.seed(seed)

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

        # Get parameter states, excluding any the user has frozen so they
        # receive neither gradients nor optimizer updates.
        param_states = model.states(ParamState)
        frozen = getattr(model, '_frozen_params', None) or set()
        if frozen:
            trainable = {name: st for name, st in param_states.items()
                         if name not in frozen}
            if not trainable:
                warnings.warn(
                    "All parameters are frozen; the optimizer will have nothing "
                    "to update.",
                    stacklevel=2,
                )
        else:
            trainable = param_states
        self.param_states = UniqueStateManager(trainable).to_pytree()

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

        # Only the first optimizer drives the training step today; warn so users
        # configuring several (e.g. GAN-style) do not assume all are stepped.
        if len(self.optimizers) > 1:
            warnings.warn(
                f"{len(self.optimizers)} optimizers were configured but only the "
                f"first is used by the training loop. Multi-optimizer schedules "
                f"are not yet supported.",
                stacklevel=2,
            )

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
        if train_dataloaders is None:
            raise ValueError(
                "fit() requires train_dataloaders; received None."
            )

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

        # Create JIT-compiled gradient computation / application steps
        compute_grads_fn, apply_grads_fn = self._create_train_step()

        # Run training
        try:
            self._run_fit(compute_grads_fn, apply_grads_fn)
        finally:
            self._cleanup()

    def _create_train_step(self) -> Tuple[Callable, Callable]:
        """Create JIT-compiled gradient computation and application steps.

        Two functions are returned so gradients can be accumulated across
        several micro-batches before a single optimizer update:

        ``compute_grads(batch, batch_idx)``
            Runs exactly one forward pass and returns ``(grads, loss, aux)``
            where ``aux`` holds the concrete values logged via ``self.log``
            (captured as gradient auxiliaries so the surrounding Python loop
            never reads stale JIT tracers).
        ``apply_grads(grads)``
            Clips (if configured) and applies the gradients via the optimizer.

        Returns
        -------
        tuple of Callable
            ``(compute_grads, apply_grads)``.
        """
        model = self.model
        optimizer = self.optimizers[0] if self.optimizers else None
        param_states = self.param_states
        gradient_clip_val = self.gradient_clip_val
        gradient_clip_algorithm = self.gradient_clip_algorithm

        def loss_fn(batch, batch_idx):
            # Fresh metric slate for this traced step.
            model._reset_logged_metrics()
            outputs = model.training_step(batch, batch_idx)
            if isinstance(outputs, dict):
                loss = outputs['loss']
            else:
                loss = outputs.loss
            # Return the logged metrics as auxiliaries so they come back as
            # concrete arrays (not tracers) after differentiation.
            aux = {
                'logger': dict(model._logger_metrics),
                'prog_bar': dict(model._prog_bar_metrics),
            }
            return loss, aux

        @brainstate.transform.jit
        def compute_grads(batch, batch_idx):
            grads, loss, aux = brainstate.transform.grad(
                loss_fn,
                grad_states=param_states,
                return_value=True,
                has_aux=True,
            )(batch, batch_idx)
            return grads, loss, aux

        @brainstate.transform.jit
        def apply_grads(grads):
            if gradient_clip_val is not None:
                if gradient_clip_algorithm == 'norm':
                    grads = _clip_grad_norm(grads, gradient_clip_val)
                else:
                    grads = _clip_grad_value(grads, gradient_clip_val)
            if optimizer is not None:
                optimizer.step(grads)

        return compute_grads, apply_grads

    def _run_fit(self, compute_grads_fn: Callable, apply_grads_fn: Callable):
        """Run the fit loop."""
        model = self.model
        state = self.state

        # Callbacks: fit start
        self._callbacks.on_fit_start(self, model)
        model.on_fit_start()

        # Callbacks: train start
        self._callbacks.on_train_start(self, model)
        model.on_train_start()

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

            # Run training epoch
            self._run_train_epoch(compute_grads_fn, apply_grads_fn, epoch)

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

            # Early stopping, but never before ``min_epochs`` have completed.
            if (epoch + 1) >= self.min_epochs and self._should_stop():
                break

        # Callbacks: train end
        self._callbacks.on_train_end(self, model)
        model.on_train_end()

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

    def _run_train_epoch(self, compute_grads_fn: Callable, apply_grads_fn: Callable, epoch: int):
        """Run a single training epoch."""
        model = self.model
        state = self.state
        optimizer = self.optimizers[0] if self.optimizers else None
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

        accumulate = max(1, self.accumulate_grad_batches)
        accum_grads = None
        accum_count = 0

        # Training loop
        for batch_idx, batch in enumerate(self.train_dataloader):
            state.batch_idx = batch_idx

            # Reset logged metrics *before* the batch-start hooks so callbacks
            # never observe the previous batch's stale metrics. (T-11)
            model._reset_logged_metrics()

            # Callbacks: batch start
            self._callbacks.on_train_batch_start(self, model, batch, batch_idx)
            model.on_train_batch_start(batch, batch_idx)

            # Single forward + backward pass; metrics come back concrete.
            grads, loss, aux = compute_grads_fn(batch, batch_idx)

            # Callbacks: after backward
            self._callbacks.on_after_backward(self, model)
            model.on_after_backward()

            # Accumulate gradients across micro-batches (T-17).
            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = jax.tree.map(lambda a, b: a + b, accum_grads, grads)
            accum_count += 1

            if accum_count >= accumulate:
                applied = (jax.tree.map(lambda g: g / accum_count, accum_grads)
                           if accum_count > 1 else accum_grads)
                self._callbacks.on_before_optimizer_step(self, model, optimizer)
                model.on_before_optimizer_step(optimizer)
                apply_grads_fn(applied)
                self._callbacks.on_after_optimizer_step(self, model, optimizer)
                model.on_after_optimizer_step(optimizer)
                accum_grads = None
                accum_count = 0

            # Build concrete outputs from the loss and logged metrics (T-9).
            loss_val = _to_scalar(loss)
            logger_metrics = aux.get('logger', {}) if isinstance(aux, dict) else {}
            prog_bar_metrics = aux.get('prog_bar', {}) if isinstance(aux, dict) else {}
            outputs = {'loss': loss_val}
            for key, value in logger_metrics.items():
                outputs[key] = _to_scalar(value)

            # Callbacks: batch end
            self._callbacks.on_train_batch_end(self, model, outputs, batch, batch_idx)
            model.on_train_batch_end(outputs, batch, batch_idx)

            # Accumulate epoch metrics (skip non-scalars).
            for key, value in outputs.items():
                scalar = _to_scalar(value)
                if isinstance(scalar, (int, float)):
                    self._epoch_metrics.setdefault(key, []).append(float(scalar))

            # Update progress bar.
            if pbar is not None:
                pbar.update(1)
                pbar_show = {'loss': loss_val}
                pbar_show.update({k: _to_scalar(v) for k, v in prog_bar_metrics.items()})
                pbar.set_postfix(pbar_show)

            # Log metrics.
            self._log_metrics(outputs, state.global_step)

            state.global_step += 1

            # Check max steps
            if self.max_steps > 0 and state.global_step >= self.max_steps:
                break

            # Validation check interval
            if self._should_validate_batch(batch_idx):
                self._run_validation_epoch(epoch)

        # Flush any gradients left in a partial accumulation window.
        if accum_count > 0 and accum_grads is not None:
            applied = (jax.tree.map(lambda g: g / accum_count, accum_grads)
                       if accum_count > 1 else accum_grads)
            self._callbacks.on_before_optimizer_step(self, model, optimizer)
            model.on_before_optimizer_step(optimizer)
            apply_grads_fn(applied)
            self._callbacks.on_after_optimizer_step(self, model, optimizer)
            model.on_after_optimizer_step(optimizer)

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
        prev_stage = state.stage
        state.stage = 'validate'

        # Callbacks: validation start
        self._callbacks.on_validation_start(self, model)
        model.on_validation_start()
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

                    # Accumulate (skip non-scalar entries).
                    for key, value in logged.items():
                        scalar = _to_scalar(value)
                        if isinstance(scalar, (int, float)):
                            all_metrics.setdefault(key, []).append(float(scalar))

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

        # Aggregate metrics, avoiding a double ``val_`` prefix when the model
        # already logged metrics under a ``val_`` name (T-2).
        for key, values in all_metrics.items():
            if values:
                name = _prefixed('val', key)
                avg = sum(values) / len(values)
                self.callback_metrics[name] = avg
                self.logged_metrics[name] = avg

        # Log validation metrics
        self._log_metrics(
            {k: v for k, v in self.callback_metrics.items() if k.startswith('val_')},
            state.global_step
        )

        # Callbacks: validation end
        self._callbacks.on_validation_epoch_end(self, model)
        model.on_validation_epoch_end()
        self._callbacks.on_validation_end(self, model)
        model.on_validation_end()

        # Restore the prior stage so validation invoked mid-training does not
        # leave the trainer stuck in the 'validate' stage.
        state.stage = prev_stage

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
        """Check if validation should run after this batch.

        Integer ``val_check_interval`` validates every N batches. A float in
        ``(0, 1)`` validates that fraction of the way through each epoch (e.g.
        ``0.25`` -> four times per epoch); ``>= 1.0`` validates only at epoch
        end. (T-23)
        """
        if self.val_dataloaders is None:
            return False

        interval = self.val_check_interval
        if isinstance(interval, float):
            if interval >= 1.0:
                return False
            try:
                n_batches = len(self.train_dataloader)
            except (TypeError, AttributeError):
                return False
            k = max(1, int(n_batches * interval))
        else:
            if interval <= 0:
                return False
            k = int(interval)
        return (batch_idx + 1) % k == 0

    def _log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to all loggers."""
        if not self.loggers:
            return

        # Convert to float, dropping any non-scalar entries.
        log_metrics = {}
        for key, value in metrics.items():
            scalar = _to_scalar(value)
            if isinstance(scalar, (int, float)):
                log_metrics[key] = float(scalar)

        for logger in self.loggers:
            logger.log_metrics(log_metrics, step)

    def _load_checkpoint(self, ckpt_path: str):
        """Load checkpoint.

        Handles both checkpoint layouts produced in this package: the
        ``ModelCheckpoint`` callback stores ``global_step`` and a *list* of
        optimizer states, while ``CheckpointManager`` stores ``step`` and a
        *single* optimizer-state dict. (T-6, T-29)
        """
        from ._checkpoint import load_checkpoint
        state = load_checkpoint(ckpt_path)

        if self.model is not None and 'model_state_dict' in state:
            self.model.load_state_dict(state['model_state_dict'])

        if self.optimizers and 'optimizer_state_dict' in state:
            for i, single in enumerate(_normalize_optimizer_states(state['optimizer_state_dict'])):
                if i < len(self.optimizers):
                    self.optimizers[i].load_state_dict(single)

        self.state.epoch = state.get('epoch', 0)
        # Prefer 'global_step'; fall back to the legacy 'step' key.
        self.state.global_step = state.get('global_step', state.get('step', 0))
        if self.model is not None:
            self.model.current_epoch = self.state.epoch
            self.model.global_step = self.state.global_step

    def _cleanup(self):
        """Release per-fit transient caches.

        The model, optimizers, schedulers, and parameter states are retained so
        the user can call :meth:`validate`, :meth:`test`, or :meth:`predict`
        after :meth:`fit` without re-supplying the model. (T-24)
        """
        self._epoch_metrics = {}

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

        if self.val_dataloaders is None:
            raise ValueError(
                "validate() requires dataloaders; none were provided and none "
                "are stored on the trainer."
            )

        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)

        self.state.stage = 'validate'
        self._run_validation_epoch(0)

        if verbose:
            print("\nValidation Results:")
            for key, value in self.callback_metrics.items():
                if key.startswith('val_'):
                    print(f"  {key}: {_format_metric(value)}")

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

        if self.test_dataloaders is None:
            raise ValueError(
                "test() requires dataloaders; none were provided and none are "
                "stored on the trainer."
            )

        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)

        # Run test loop (similar to validation)
        model = self.model
        state = self.state
        state.stage = 'test'

        self._callbacks.on_test_start(self, model)
        model.on_test_start()
        self._callbacks.on_test_epoch_start(self, model)
        model.on_test_epoch_start()

        all_metrics: Dict[str, List[Any]] = {}

        for dataloader in self.test_dataloaders:
            pbar = None
            if self.enable_progress_bar:
                pbar = get_progress_bar()
                pbar.start(total=len(dataloader), desc='Testing')

            for batch_idx, batch in enumerate(dataloader):
                model._reset_logged_metrics()
                self._callbacks.on_test_batch_start(self, model, batch, batch_idx)
                model.on_test_batch_start(batch, batch_idx)

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
                        scalar = _to_scalar(value)
                        if isinstance(scalar, (int, float)):
                            all_metrics[key].append(float(scalar))

                self._callbacks.on_test_batch_end(self, model, outputs, batch, batch_idx)
                model.on_test_batch_end(outputs, batch, batch_idx)

                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

        # Aggregate metrics, avoiding a double ``test_`` prefix (T-2).
        test_metrics = {}
        for key, values in all_metrics.items():
            if values:
                test_metrics[_prefixed('test', key)] = sum(values) / len(values)

        # Expose results on the trainer for callbacks consuming epoch-end state.
        self.callback_metrics.update(test_metrics)

        self._callbacks.on_test_epoch_end(self, model)
        model.on_test_epoch_end()
        self._callbacks.on_test_end(self, model)
        model.on_test_end()

        if verbose:
            print("\nTest Results:")
            for key, value in test_metrics.items():
                print(f"  {key}: {_format_metric(value)}")

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

        if self.predict_dataloaders is None:
            raise ValueError(
                "predict() requires dataloaders; none were provided and none "
                "are stored on the trainer."
            )

        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)

        model = self.model
        state = self.state
        state.stage = 'predict'

        self._callbacks.on_predict_start(self, model)
        model.on_predict_start()

        all_predictions = []

        for dataloader in self.predict_dataloaders:
            pbar = None
            if self.enable_progress_bar:
                pbar = get_progress_bar()
                pbar.start(total=len(dataloader), desc='Predicting')

            for batch_idx, batch in enumerate(dataloader):
                self._callbacks.on_predict_batch_start(self, model, batch, batch_idx)
                model.on_predict_batch_start(batch, batch_idx)

                outputs = model.predict_step(batch, batch_idx)
                all_predictions.append(outputs)

                self._callbacks.on_predict_batch_end(self, model, outputs, batch, batch_idx)
                model.on_predict_batch_end(outputs, batch, batch_idx)

                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

        self._callbacks.on_predict_end(self, model)
        model.on_predict_end()

        return all_predictions


# =============================================================================
# Utility Functions
# =============================================================================

def _normalize_optimizer_states(opt_state: Any) -> List[Any]:
    """Normalize a saved optimizer state into a list of per-optimizer states.

    Handles the several shapes a checkpoint may carry (T-29):

    * a genuine ``list``/``tuple`` of per-optimizer state dicts;
    * a single optimizer-state dict (``CheckpointManager`` style), detected by
      well-known optimizer keys;
    * a digit-keyed dict such as ``{'0': ..., '1': ...}``, which is how a saved
      list round-trips through msgpack serialization.
    """
    if isinstance(opt_state, (list, tuple)):
        return list(opt_state)
    if isinstance(opt_state, dict):
        single_markers = {'step_count', 'opt_state', 'param_groups', 'lr'}
        if single_markers & set(opt_state.keys()):
            return [opt_state]
        if opt_state and all(str(k).isdigit() for k in opt_state.keys()):
            return [opt_state[k] for k in sorted(opt_state.keys(), key=lambda x: int(x))]
        return [opt_state]
    return []


def _prefixed(stage: str, key: str) -> str:
    """Prefix ``key`` with ``f'{stage}_'`` unless it already carries it.

    Prevents doubled prefixes such as ``val_val_loss`` when a model logs a
    metric under a name that already includes the stage prefix.
    """
    prefix = f'{stage}_'
    return key if key.startswith(prefix) else f'{prefix}{key}'


def _format_metric(value: Any) -> str:
    """Format a metric for console display, tolerating non-float values."""
    try:
        return f'{float(value):.4f}'
    except (TypeError, ValueError):
        return str(value)


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
