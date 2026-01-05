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

Example
-------
>>> import braintools
>>> import brainstate
>>>
>>> class MyModel(braintools.trainer.LightningModule):
...     def __init__(self):
...         super().__init__()
...         self.linear = brainstate.nn.Linear(784, 10)
...
...     def __call__(self, x):
...         return self.linear(x)
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
...
>>> model = MyModel()
>>> trainer = braintools.trainer.Trainer(max_epochs=10)
>>> train_loader = braintools.trainer.DataLoader(data, batch_size=32)
>>> trainer.fit(model, train_loader)
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
