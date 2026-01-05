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
Logging backends for training.

This module provides pluggable logging backends including TensorBoard,
Weights & Biases, CSV, and more.
"""

import csv
import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = [
    'Logger',
    'TensorBoardLogger',
    'WandBLogger',
    'CSVLogger',
    'CompositeLogger',
    'NeptuneLogger',
    'MLFlowLogger',
]


class Logger(ABC):
    """
    Abstract base class for all loggers.

    Subclass this to implement custom logging backends.

    Attributes
    ----------
    name : str
        Name of the experiment/run.
    version : str
        Version or run ID.
    """
    __module__ = 'braintools.trainer'

    def __init__(self):
        self._name: str = 'default'
        self._version: Optional[str] = None

    @property
    def name(self) -> str:
        """Experiment name."""
        return self._name

    @property
    def version(self) -> Optional[str]:
        """Experiment version/run ID."""
        return self._version

    @property
    def root_dir(self) -> Optional[str]:
        """Root directory for logs."""
        return None

    @property
    def log_dir(self) -> Optional[str]:
        """Directory for this specific run's logs."""
        return None

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """
        Log metrics.

        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary of metric names to values.
        step : int, optional
            Global step number.
        """
        pass

    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any]):
        """
        Log hyperparameters.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of hyperparameter names to values.
        """
        pass

    def log_graph(self, model: Any, input_array: Any = None):
        """
        Log model graph/architecture.

        Parameters
        ----------
        model : Any
            The model to log.
        input_array : Any, optional
            Sample input for tracing.
        """
        pass

    def log_image(
        self,
        key: str,
        images: Any,
        step: Optional[int] = None,
    ):
        """
        Log images.

        Parameters
        ----------
        key : str
            Image key/tag.
        images : Any
            Images to log (numpy array or list).
        step : int, optional
            Global step number.
        """
        pass

    def log_text(
        self,
        key: str,
        text: str,
        step: Optional[int] = None,
    ):
        """
        Log text.

        Parameters
        ----------
        key : str
            Text key/tag.
        text : str
            Text to log.
        step : int, optional
            Global step number.
        """
        pass

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ):
        """
        Log an artifact (file).

        Parameters
        ----------
        local_path : str
            Path to the local file.
        artifact_path : str, optional
            Path in artifact storage.
        """
        pass

    def save(self):
        """Save/flush any buffered data."""
        pass

    def finalize(self, status: str = 'success'):
        """
        Finalize logging (called at end of training).

        Parameters
        ----------
        status : str
            Final status ('success', 'failed', 'interrupted').
        """
        pass


class TensorBoardLogger(Logger):
    """
    TensorBoard logging backend.

    Parameters
    ----------
    save_dir : str
        Directory to save TensorBoard logs.
    name : str, default='default'
        Experiment name (subdirectory).
    version : str, optional
        Version string. If None, auto-generates based on timestamp.
    log_graph : bool, default=False
        Whether to log the model graph.
    default_hp_metric : bool, default=True
        Whether to log hyperparameters with a default metric.
    prefix : str, default=''
        Prefix for all metric names.

    Examples
    --------
    >>> logger = TensorBoardLogger('logs/', name='my_experiment')
    >>> trainer = Trainer(logger=logger)
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        save_dir: str,
        name: str = 'default',
        version: Optional[str] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = '',
        **kwargs,
    ):
        super().__init__()
        self._save_dir = save_dir
        self._name = name
        self._version = version or datetime.now().strftime('%Y%m%d_%H%M%S')
        self._log_graph = log_graph
        self._default_hp_metric = default_hp_metric
        self._prefix = prefix

        self._writer = None
        self._initialized = False

    def _init_writer(self):
        """Initialize TensorBoard writer."""
        if self._initialized:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError(
                    "TensorBoardLogger requires either 'tensorboard' or 'tensorboardX'. "
                    "Install with: pip install tensorboard"
                )

        log_dir = self.log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=log_dir)
        self._initialized = True

    @property
    def root_dir(self) -> str:
        return self._save_dir

    @property
    def log_dir(self) -> str:
        return os.path.join(self._save_dir, self._name, self._version)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        self._init_writer()

        for key, value in metrics.items():
            tag = f"{self._prefix}{key}" if self._prefix else key
            self._writer.add_scalar(tag, value, step)

    def log_hyperparams(self, params: Dict[str, Any]):
        self._init_writer()

        # Convert complex types to strings
        processed_params = {}
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)):
                processed_params[key] = value
            else:
                processed_params[key] = str(value)

        if self._default_hp_metric:
            self._writer.add_hparams(processed_params, {'hp_metric': 0})
        else:
            self._writer.add_hparams(processed_params, {})

    def log_graph(self, model: Any, input_array: Any = None):
        if not self._log_graph:
            return
        # Note: Model graph logging is complex in JAX, skip for now
        warnings.warn("Model graph logging not yet implemented for JAX models")

    def log_image(
        self,
        key: str,
        images: Any,
        step: Optional[int] = None,
    ):
        self._init_writer()
        import numpy as np

        if isinstance(images, np.ndarray):
            if images.ndim == 3:  # Single image (H, W, C)
                self._writer.add_image(key, images, step, dataformats='HWC')
            elif images.ndim == 4:  # Batch of images (N, H, W, C)
                self._writer.add_images(key, images, step, dataformats='NHWC')
        else:
            self._writer.add_image(key, images, step)

    def log_text(
        self,
        key: str,
        text: str,
        step: Optional[int] = None,
    ):
        self._init_writer()
        self._writer.add_text(key, text, step)

    def save(self):
        if self._writer:
            self._writer.flush()

    def finalize(self, status: str = 'success'):
        if self._writer:
            self._writer.close()
            self._writer = None
            self._initialized = False


class WandBLogger(Logger):
    """
    Weights & Biases logging backend.

    Parameters
    ----------
    project : str
        W&B project name.
    name : str, optional
        Run name. If None, W&B will auto-generate.
    entity : str, optional
        W&B entity (username or team name).
    config : Dict[str, Any], optional
        Hyperparameters to log.
    tags : List[str], optional
        Tags for the run.
    notes : str, optional
        Notes for the run.
    save_dir : str, optional
        Directory for local W&B files.
    offline : bool, default=False
        Whether to run in offline mode.
    log_model : bool, default=False
        Whether to log model checkpoints.

    Examples
    --------
    >>> logger = WandBLogger(project='my_project', name='run_1')
    >>> trainer = Trainer(logger=logger)
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: bool = False,
        log_model: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._project = project
        self._name = name
        self._entity = entity
        self._config = config or {}
        self._tags = tags
        self._notes = notes
        self._save_dir = save_dir
        self._offline = offline
        self._log_model = log_model

        self._run = None
        self._initialized = False

    def _init_wandb(self):
        """Initialize W&B run."""
        if self._initialized:
            return

        try:
            import wandb
        except ImportError:
            raise ImportError(
                "WandBLogger requires 'wandb'. Install with: pip install wandb"
            )

        mode = 'offline' if self._offline else 'online'
        self._run = wandb.init(
            project=self._project,
            name=self._name,
            entity=self._entity,
            config=self._config,
            tags=self._tags,
            notes=self._notes,
            dir=self._save_dir,
            mode=mode,
        )
        self._version = self._run.id
        self._initialized = True

    @property
    def name(self) -> str:
        return self._name or (self._run.name if self._run else 'default')

    @property
    def version(self) -> Optional[str]:
        return self._version or (self._run.id if self._run else None)

    @property
    def log_dir(self) -> Optional[str]:
        if self._run:
            return self._run.dir
        return self._save_dir

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        self._init_wandb()
        import wandb

        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def log_hyperparams(self, params: Dict[str, Any]):
        self._init_wandb()
        import wandb

        wandb.config.update(params, allow_val_change=True)

    def log_image(
        self,
        key: str,
        images: Any,
        step: Optional[int] = None,
    ):
        self._init_wandb()
        import wandb

        if isinstance(images, list):
            wandb_images = [wandb.Image(img) for img in images]
        else:
            wandb_images = wandb.Image(images)

        wandb.log({key: wandb_images}, step=step)

    def log_text(
        self,
        key: str,
        text: str,
        step: Optional[int] = None,
    ):
        self._init_wandb()
        import wandb

        wandb.log({key: text}, step=step)

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ):
        self._init_wandb()
        import wandb

        artifact_name = artifact_path or os.path.basename(local_path)
        artifact = wandb.Artifact(artifact_name, type='model')
        artifact.add_file(local_path)
        self._run.log_artifact(artifact)

    def save(self):
        pass  # W&B auto-saves

    def finalize(self, status: str = 'success'):
        if self._run:
            import wandb
            wandb.finish()
            self._run = None
            self._initialized = False


class CSVLogger(Logger):
    """
    CSV file logging backend.

    Logs metrics to a CSV file for simple analysis.

    Parameters
    ----------
    save_dir : str
        Directory to save CSV logs.
    name : str, default='logs'
        Experiment name (subdirectory).
    version : str, optional
        Version string.
    flush_logs_every_n_steps : int, default=100
        How often to flush logs to disk.

    Examples
    --------
    >>> logger = CSVLogger('logs/', name='my_experiment')
    >>> trainer = Trainer(logger=logger)
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        save_dir: str,
        name: str = 'logs',
        version: Optional[str] = None,
        flush_logs_every_n_steps: int = 100,
    ):
        super().__init__()
        self._save_dir = save_dir
        self._name = name
        self._version = version or datetime.now().strftime('%Y%m%d_%H%M%S')
        self._flush_interval = flush_logs_every_n_steps

        self._metrics_buffer: List[Dict[str, Any]] = []
        self._hyperparams: Dict[str, Any] = {}
        self._step_count = 0
        self._file = None
        self._writer = None
        self._fieldnames: List[str] = ['step']

    @property
    def root_dir(self) -> str:
        return self._save_dir

    @property
    def log_dir(self) -> str:
        return os.path.join(self._save_dir, self._name, self._version)

    @property
    def metrics_file_path(self) -> str:
        return os.path.join(self.log_dir, 'metrics.csv')

    @property
    def hparams_file_path(self) -> str:
        return os.path.join(self.log_dir, 'hparams.yaml')

    def _init_csv(self):
        """Initialize CSV file."""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        self._init_csv()

        row = {'step': step or self._step_count}
        row.update(metrics)
        self._metrics_buffer.append(row)

        # Update fieldnames
        for key in metrics.keys():
            if key not in self._fieldnames:
                self._fieldnames.append(key)

        self._step_count += 1

        # Flush if needed
        if len(self._metrics_buffer) >= self._flush_interval:
            self.save()

    def log_hyperparams(self, params: Dict[str, Any]):
        self._init_csv()
        self._hyperparams.update(params)

        # Save hyperparams to YAML file
        try:
            import yaml
            with open(self.hparams_file_path, 'w') as f:
                yaml.dump(self._hyperparams, f, default_flow_style=False)
        except ImportError:
            # Fall back to simple text format
            with open(self.hparams_file_path, 'w') as f:
                for key, value in self._hyperparams.items():
                    f.write(f"{key}: {value}\n")

    def save(self):
        """Flush buffered metrics to CSV."""
        if not self._metrics_buffer:
            return

        file_exists = os.path.exists(self.metrics_file_path)

        with open(self.metrics_file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction='ignore')

            if not file_exists:
                writer.writeheader()

            for row in self._metrics_buffer:
                writer.writerow(row)

        self._metrics_buffer.clear()

    def finalize(self, status: str = 'success'):
        self.save()


class CompositeLogger(Logger):
    """
    Combine multiple loggers.

    Logs to all provided loggers simultaneously.

    Parameters
    ----------
    loggers : List[Logger]
        List of loggers to use.

    Examples
    --------
    >>> logger = CompositeLogger([
    ...     TensorBoardLogger('logs/'),
    ...     CSVLogger('logs/'),
    ... ])
    >>> trainer = Trainer(logger=logger)
    """
    __module__ = 'braintools.trainer'

    def __init__(self, loggers: List[Logger]):
        super().__init__()
        self._loggers = loggers

    @property
    def name(self) -> str:
        return self._loggers[0].name if self._loggers else 'default'

    @property
    def version(self) -> Optional[str]:
        return self._loggers[0].version if self._loggers else None

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        for logger in self._loggers:
            logger.log_metrics(metrics, step)

    def log_hyperparams(self, params: Dict[str, Any]):
        for logger in self._loggers:
            logger.log_hyperparams(params)

    def log_graph(self, model: Any, input_array: Any = None):
        for logger in self._loggers:
            logger.log_graph(model, input_array)

    def log_image(
        self,
        key: str,
        images: Any,
        step: Optional[int] = None,
    ):
        for logger in self._loggers:
            logger.log_image(key, images, step)

    def log_text(
        self,
        key: str,
        text: str,
        step: Optional[int] = None,
    ):
        for logger in self._loggers:
            logger.log_text(key, text, step)

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ):
        for logger in self._loggers:
            logger.log_artifact(local_path, artifact_path)

    def save(self):
        for logger in self._loggers:
            logger.save()

    def finalize(self, status: str = 'success'):
        for logger in self._loggers:
            logger.finalize(status)


class NeptuneLogger(Logger):
    """
    Neptune.ai logging backend.

    Parameters
    ----------
    project : str
        Neptune project name (workspace/project).
    api_token : str, optional
        Neptune API token. If None, uses NEPTUNE_API_TOKEN env variable.
    name : str, optional
        Run name.
    tags : List[str], optional
        Tags for the run.
    description : str, optional
        Run description.

    Examples
    --------
    >>> logger = NeptuneLogger(project='workspace/project')
    >>> trainer = Trainer(logger=logger)
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        project: str,
        api_token: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self._project = project
        self._api_token = api_token
        self._name = name
        self._tags = tags
        self._description = description

        self._run = None
        self._initialized = False

    def _init_neptune(self):
        """Initialize Neptune run."""
        if self._initialized:
            return

        try:
            import neptune
        except ImportError:
            raise ImportError(
                "NeptuneLogger requires 'neptune'. Install with: pip install neptune"
            )

        self._run = neptune.init_run(
            project=self._project,
            api_token=self._api_token,
            name=self._name,
            tags=self._tags,
            description=self._description,
        )
        self._version = self._run['sys/id'].fetch()
        self._initialized = True

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        self._init_neptune()

        for key, value in metrics.items():
            if step is not None:
                self._run[key].append(value, step=step)
            else:
                self._run[key].append(value)

    def log_hyperparams(self, params: Dict[str, Any]):
        self._init_neptune()
        self._run['parameters'] = params

    def save(self):
        if self._run:
            self._run.sync()

    def finalize(self, status: str = 'success'):
        if self._run:
            self._run.stop()
            self._run = None
            self._initialized = False


class MLFlowLogger(Logger):
    """
    MLflow logging backend.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name.
    tracking_uri : str, optional
        MLflow tracking server URI.
    run_name : str, optional
        Run name.
    tags : Dict[str, str], optional
        Tags for the run.
    save_dir : str, optional
        Local directory for MLflow artifacts.

    Examples
    --------
    >>> logger = MLFlowLogger(experiment_name='my_experiment')
    >>> trainer = Trainer(logger=logger)
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        save_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._run_name = run_name
        self._tags = tags
        self._save_dir = save_dir

        self._run = None
        self._initialized = False

    def _init_mlflow(self):
        """Initialize MLflow run."""
        if self._initialized:
            return

        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "MLFlowLogger requires 'mlflow'. Install with: pip install mlflow"
            )

        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)

        mlflow.set_experiment(self._experiment_name)
        self._run = mlflow.start_run(run_name=self._run_name, tags=self._tags)
        self._version = self._run.info.run_id
        self._initialized = True

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        self._init_mlflow()
        import mlflow

        mlflow.log_metrics(metrics, step=step)

    def log_hyperparams(self, params: Dict[str, Any]):
        self._init_mlflow()
        import mlflow

        mlflow.log_params(params)

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ):
        self._init_mlflow()
        import mlflow

        mlflow.log_artifact(local_path, artifact_path)

    def save(self):
        pass  # MLflow auto-saves

    def finalize(self, status: str = 'success'):
        if self._run:
            import mlflow
            mlflow.end_run()
            self._run = None
            self._initialized = False
