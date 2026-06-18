``braintools.trainer`` module
=============================

.. currentmodule:: braintools.trainer
.. automodule:: braintools.trainer

A PyTorch-Lightning-style training framework for JAX / ``brainstate`` models.
It bundles the training loop, validation/testing/prediction, a callback and
logging system, checkpointing, data loading, and multi-device strategies into a
single high-level API.

Overview
--------

The ``braintools.trainer`` module provides:

- **LightningModule** -- base class for defining models, training/validation
  steps, and optimizers.
- **Trainer** -- orchestrates the full fit/validate/test/predict loops.
- **Callbacks** -- hooks for checkpointing, early stopping, LR monitoring, etc.
- **Loggers** -- pluggable logging backends (CSV, TensorBoard, WandB, ...).
- **DataLoader** -- JAX-compatible batching, shuffling, and sampling.
- **Distributed strategies** -- single-device, data-parallel, sharded, and FSDP.
- **Checkpointing** -- save/restore model and optimizer state.

.. note::

   Metrics logged via :meth:`LightningModule.log` are referenced by the trainer
   and callbacks under the name you pass. The trainer prefixes aggregated
   validation/test metrics with ``val_``/``test_`` only when the name does not
   already start with that prefix, so ``self.log('val_loss', ...)`` is surfaced
   as ``val_loss`` (not ``val_val_loss``).

Core Classes
------------

The two classes you interact with most: subclass :class:`LightningModule` to
define your model and call :meth:`Trainer.fit` to train it.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   LightningModule
   Trainer
   TrainerState
   TrainOutput
   EvalOutput

Callbacks
---------

Callbacks hook into the training loop to add behavior without modifying the
model. ``Callback`` is the base class; ``CallbackList`` dispatches to a list of
callbacks.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Callback
   CallbackList
   ModelCheckpoint
   EarlyStopping
   LearningRateMonitor
   GradientClipCallback
   Timer
   RichProgressBar
   TQDMProgressBar
   LambdaCallback
   PrintCallback

Loggers
-------

Pluggable logging backends. Pass an instance (or list) to the ``logger``
argument of :class:`Trainer`.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Logger
   CSVLogger
   TensorBoardLogger
   WandBLogger
   NeptuneLogger
   MLFlowLogger
   CompositeLogger

Data Loading
------------

JAX-compatible datasets, samplers, and loaders.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DataLoader
   DistributedDataLoader
   Dataset
   ArrayDataset
   DictDataset
   IterableDataset
   Sampler
   RandomSampler
   SequentialSampler
   BatchSampler
   DistributedSampler

.. autosummary::
   :toctree: generated/
   :nosignatures:

   create_distributed_batches

Distributed Strategies
----------------------

Strategies that control how the model and data are distributed across devices.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Strategy
   SingleDeviceStrategy
   DataParallelStrategy
   ShardedDataParallelStrategy
   FullyShardedDataParallelStrategy
   AutoStrategy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_strategy
   all_reduce
   broadcast

Checkpointing
-------------

Utilities to save and restore training state.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   CheckpointManager

.. autosummary::
   :toctree: generated/
   :nosignatures:

   save_checkpoint
   load_checkpoint
   find_checkpoint
   list_checkpoints

Progress Bars
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ProgressBar
   SimpleProgressBar
   TQDMProgressBarWrapper
   RichProgressBarWrapper

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_progress_bar
