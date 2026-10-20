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
Trainer module for PyTorch Lightning-like training in JAX.

This module provides a comprehensive training framework including:
- LightningModule: Base class for defining training models
- Trainer: Orchestration class for training loops
- Callbacks: Hook system for customizing training behavior
- Loggers: Pluggable logging backends (TensorBoard, WandB, CSV, etc.)
- DataLoader: JAX-compatible data loading with distributed support
- Distributed: Strategies for multi-device and multi-host training

Examples
--------
.. code-block:: python

    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> import brainstate
    >>> import braintools
    >>>
    >>> class MyModel(braintools.trainer.LightningModule):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.linear = brainstate.nn.Linear(8, 2)
    ...
    ...     def __call__(self, x):
    ...         return self.linear(x)
    ...
    ...     def training_step(self, batch, batch_idx):
    ...         x, y = batch['x'], batch['y']
    ...         loss = jnp.mean((self(x) - y) ** 2)
    ...         self.log('mse', loss)
    ...         return {'loss': loss}
    ...
    ...     def configure_optimizers(self):
    ...         return braintools.optim.Adam(lr=1e-3)
    ...
    >>> brainstate.random.seed(0)
    >>> model = MyModel()
    >>> data = {'x': np.ones((64, 8), np.float32), 'y': np.zeros((64, 2), np.float32)}
    >>> train_loader = braintools.trainer.DataLoader(data, batch_size=32)
    >>> trainer = braintools.trainer.Trainer(
    ...     max_epochs=10, logger=False, enable_progress_bar=False,
    ...     enable_checkpointing=False)
    >>> trainer.fit(model, train_loader)
    >>> trainer.global_step
    20
"""

# Module
from ._module import (
    LightningModule,
    TrainOutput,
    EvalOutput,
)

# Trainer
from ._trainer import (
    Trainer,
    TrainerState,
)

# Callbacks
from ._callbacks import (
    Callback,
    CallbackList,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    GradientClipCallback,
    Timer,
    RichProgressBar,
    TQDMProgressBar,
    LambdaCallback,
    PrintCallback,
)

# Loggers
from ._loggers import (
    Logger,
    TensorBoardLogger,
    WandBLogger,
    CSVLogger,
    CompositeLogger,
    NeptuneLogger,
    MLFlowLogger,
)

# DataLoader
from ._dataloader import (
    DataLoader,
    DistributedDataLoader,
    Dataset,
    ArrayDataset,
    DictDataset,
    IterableDataset,
    Sampler,
    RandomSampler,
    SequentialSampler,
    BatchSampler,
    DistributedSampler,
    create_distributed_batches,
)

# Distributed
from ._distributed import (
    Strategy,
    SingleDeviceStrategy,
    DataParallelStrategy,
    ShardedDataParallelStrategy,
    FullyShardedDataParallelStrategy,
    AutoStrategy,
    get_strategy,
    all_reduce,
    broadcast,
)

# Checkpointing
from ._checkpoint import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    find_checkpoint,
    list_checkpoints,
)

# Progress
from ._progress import (
    ProgressBar,
    SimpleProgressBar,
    TQDMProgressBarWrapper,
    RichProgressBarWrapper,
    get_progress_bar,
)

__all__ = [
    # Module
    'LightningModule',
    'TrainOutput',
    'EvalOutput',

    # Trainer
    'Trainer',
    'TrainerState',

    # Callbacks
    'Callback',
    'CallbackList',
    'ModelCheckpoint',
    'EarlyStopping',
    'LearningRateMonitor',
    'GradientClipCallback',
    'Timer',
    'RichProgressBar',
    'TQDMProgressBar',
    'LambdaCallback',
    'PrintCallback',

    # Loggers
    'Logger',
    'TensorBoardLogger',
    'WandBLogger',
    'CSVLogger',
    'CompositeLogger',
    'NeptuneLogger',
    'MLFlowLogger',

    # DataLoader
    'DataLoader',
    'DistributedDataLoader',
    'Dataset',
    'ArrayDataset',
    'DictDataset',
    'IterableDataset',
    'Sampler',
    'RandomSampler',
    'SequentialSampler',
    'BatchSampler',
    'DistributedSampler',
    'create_distributed_batches',

    # Distributed
    'Strategy',
    'SingleDeviceStrategy',
    'DataParallelStrategy',
    'ShardedDataParallelStrategy',
    'FullyShardedDataParallelStrategy',
    'AutoStrategy',
    'get_strategy',
    'all_reduce',
    'broadcast',

    # Checkpointing
    'CheckpointManager',
    'save_checkpoint',
    'load_checkpoint',
    'find_checkpoint',
    'list_checkpoints',

    # Progress
    'ProgressBar',
    'SimpleProgressBar',
    'TQDMProgressBarWrapper',
    'RichProgressBarWrapper',
    'get_progress_bar',
]
