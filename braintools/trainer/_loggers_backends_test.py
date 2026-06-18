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

"""Happy-path tests for the optional logger backends.

The backend libraries (tensorboard, wandb, neptune, mlflow) are not installed
in the standard CI environment, so the methods that call them are otherwise
never exercised. Here we inject lightweight fake modules into ``sys.modules``
so the logger code runs against a recording double, covering the
``log_metrics``/``log_hyperparams``/``log_image``/``log_text``/``finalize``
paths.
"""

import sys
import types
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest

from braintools.trainer import (
    TensorBoardLogger,
    WandBLogger,
    NeptuneLogger,
    MLFlowLogger,
)


# ---------------------------------------------------------------------------
# Fake backend doubles
# ---------------------------------------------------------------------------

class _FakeSummaryWriter:
    instances = []

    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.scalars = []
        self.hparams = []
        self.images = []
        self.texts = []
        self.flushed = False
        self.closed = False
        _FakeSummaryWriter.instances.append(self)

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_hparams(self, params, metrics):
        self.hparams.append((params, metrics))

    def add_image(self, key, image, step, dataformats='CHW'):
        self.images.append((key, 'single', dataformats))

    def add_images(self, key, images, step, dataformats='NCHW'):
        self.images.append((key, 'batch', dataformats))

    def add_text(self, key, text, step):
        self.texts.append((key, text, step))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


@contextmanager
def fake_module(name, module):
    """Temporarily install ``module`` (and parent packages) in sys.modules."""
    parts = name.split('.')
    to_add = {}
    for i in range(1, len(parts)):
        parent = '.'.join(parts[:i])
        if parent not in sys.modules:
            to_add[parent] = types.ModuleType(parent)
    to_add[name] = module
    with patch.dict(sys.modules, to_add):
        # Wire child onto parent so attribute access resolves too.
        for i in range(1, len(parts)):
            parent = '.'.join(parts[:i])
            child = '.'.join(parts[:i + 1])
            setattr(sys.modules[parent], parts[i], sys.modules[child])
        yield module


# ---------------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------------

class TestTensorBoardHappyPath:
    def _tb_module(self):
        mod = types.ModuleType('torch.utils.tensorboard')
        mod.SummaryWriter = _FakeSummaryWriter
        return mod

    def test_full_logging_cycle(self, tmp_path):
        _FakeSummaryWriter.instances.clear()
        with fake_module('torch.utils.tensorboard', self._tb_module()):
            logger = TensorBoardLogger(str(tmp_path), name='tb', version='v1',
                                       prefix='p/')
            logger.log_metrics({'loss': 0.5, 'acc': 0.9}, step=1)
            logger.log_hyperparams({'lr': 0.1, 'arch': object()})
            logger.log_image('img', np.zeros((4, 4, 3)), step=1)
            logger.log_image('imgs', np.zeros((2, 4, 4, 3)), step=1)
            logger.log_text('note', 'hello', step=1)
            logger.save()
            logger.finalize()

        writer = _FakeSummaryWriter.instances[-1]
        assert ('p/loss', 0.5, 1) in writer.scalars
        assert writer.hparams  # hyperparameters recorded
        assert ('img', 'single', 'HWC') in writer.images
        assert ('imgs', 'batch', 'NHWC') in writer.images
        assert ('note', 'hello', 1) in writer.texts
        assert writer.closed

    def test_default_hp_metric_false(self, tmp_path):
        _FakeSummaryWriter.instances.clear()
        with fake_module('torch.utils.tensorboard', self._tb_module()):
            logger = TensorBoardLogger(str(tmp_path), name='tb', version='v1',
                                       default_hp_metric=False)
            logger.log_hyperparams({'lr': 0.1})
        writer = _FakeSummaryWriter.instances[-1]
        # When default_hp_metric is False, the metric dict is empty.
        assert writer.hparams[0][1] == {}


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------

class _FakeWandbRun:
    def __init__(self):
        self.id = 'run123'
        self.name = 'auto-name'
        self.dir = '/tmp/wandb'
        self.artifacts = []

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)


class _FakeArtifact:
    def __init__(self, name, type=None):
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, path):
        self.files.append(path)


def _wandb_module():
    mod = types.ModuleType('wandb')
    state = {'logs': [], 'config_updates': []}

    def init(**kwargs):
        mod._run = _FakeWandbRun()
        return mod._run

    def log(data, step=None):
        state['logs'].append((data, step))

    config = types.SimpleNamespace(
        update=lambda params, allow_val_change=False: state['config_updates'].append(params))

    mod.init = init
    mod.log = log
    mod.config = config
    mod.Image = lambda img: ('Image', id(img))
    mod.Artifact = _FakeArtifact
    mod.finish = lambda: state.setdefault('finished', True)
    mod._state = state
    return mod


class TestWandBHappyPath:
    def test_metrics_hparams_text(self, tmp_path):
        mod = _wandb_module()
        with fake_module('wandb', mod):
            logger = WandBLogger(project='proj', name='run', save_dir=str(tmp_path))
            logger.log_metrics({'loss': 0.5}, step=2)
            logger.log_metrics({'loss': 0.4})  # step=None branch
            logger.log_hyperparams({'lr': 0.1})
            logger.log_text('note', 'hi', step=1)
            assert logger.version == 'run123'
            assert logger.log_dir == '/tmp/wandb'
            logger.finalize()
        assert ({'loss': 0.5}, 2) in mod._state['logs']
        assert {'lr': 0.1} in mod._state['config_updates']

    def test_image_single_and_list(self):
        mod = _wandb_module()
        with fake_module('wandb', mod):
            logger = WandBLogger(project='proj', name='run')
            logger.log_image('img', np.zeros((4, 4, 3)), step=1)
            logger.log_image('imgs', [np.zeros((2, 2, 3)), np.zeros((2, 2, 3))], step=1)
        keys = [d for d, _ in mod._state['logs']]
        assert any('img' in d for d in keys)

    def test_log_artifact(self, tmp_path):
        f = tmp_path / 'model.bin'
        f.write_text('x')
        mod = _wandb_module()
        with fake_module('wandb', mod):
            logger = WandBLogger(project='proj', name='run')
            logger.log_artifact(str(f))
            assert logger._run.artifacts  # artifact logged


# ---------------------------------------------------------------------------
# Neptune
# ---------------------------------------------------------------------------

class _FakeNeptuneField:
    def __init__(self, store, key):
        self._store = store
        self._key = key

    def append(self, value, step=None):
        self._store.setdefault(self._key, []).append((value, step))

    def fetch(self):
        return 'NEP-1'


class _FakeNeptuneRun:
    def __init__(self):
        self.data = {}
        self.assigned = {}
        self.synced = False
        self.stopped = False

    def __getitem__(self, key):
        return _FakeNeptuneField(self.data, key)

    def __setitem__(self, key, value):
        self.assigned[key] = value

    def sync(self):
        self.synced = True

    def stop(self):
        self.stopped = True


def _neptune_module():
    mod = types.ModuleType('neptune')
    mod._last_run = None

    def init_run(**kwargs):
        mod._last_run = _FakeNeptuneRun()
        return mod._last_run

    mod.init_run = init_run
    return mod


class TestNeptuneHappyPath:
    def test_full_cycle(self):
        mod = _neptune_module()
        with fake_module('neptune', mod):
            logger = NeptuneLogger(project='ws/proj', name='run', tags=['a'])
            logger.log_metrics({'loss': 0.5}, step=3)
            logger.log_metrics({'loss': 0.4})  # step=None branch
            logger.log_hyperparams({'lr': 0.1})
            logger.save()
            run = mod._last_run
            assert run.synced
            logger.finalize()
            assert run.stopped
        assert run.data['loss'] == [(0.5, 3), (0.4, None)]
        assert run.assigned['parameters'] == {'lr': 0.1}


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

def _mlflow_module():
    mod = types.ModuleType('mlflow')
    state = {'metrics': [], 'params': [], 'artifacts': [], 'uri': None,
             'experiment': None, 'ended': False}

    class _Info:
        run_id = 'MLF-1'

    class _Run:
        info = _Info()

    mod.set_tracking_uri = lambda uri: state.update(uri=uri)
    mod.set_experiment = lambda name: state.update(experiment=name)
    mod.start_run = lambda run_name=None, tags=None: _Run()
    mod.log_metrics = lambda metrics, step=None: state['metrics'].append((metrics, step))
    mod.log_params = lambda params: state['params'].append(params)
    mod.log_artifact = lambda path, artifact_path=None: state['artifacts'].append((path, artifact_path))
    mod.end_run = lambda: state.update(ended=True)
    mod._state = state
    return mod


class TestMLFlowHappyPath:
    def test_full_cycle(self, tmp_path):
        f = tmp_path / 'a.txt'
        f.write_text('x')
        mod = _mlflow_module()
        with fake_module('mlflow', mod):
            logger = MLFlowLogger(experiment_name='exp', tracking_uri='file:///tmp',
                                  run_name='run', tags={'k': 'v'})
            logger.log_metrics({'loss': 0.5}, step=1)
            logger.log_hyperparams({'lr': 0.1})
            logger.log_artifact(str(f), 'sub')
            logger.save()
            assert logger.version == 'MLF-1'
            logger.finalize()
        assert mod._state['uri'] == 'file:///tmp'
        assert mod._state['experiment'] == 'exp'
        assert ({'loss': 0.5}, 1) in mod._state['metrics']
        assert {'lr': 0.1} in mod._state['params']
        assert mod._state['ended']

    def test_init_only_once(self):
        mod = _mlflow_module()
        with fake_module('mlflow', mod):
            logger = MLFlowLogger(experiment_name='exp')
            logger.log_metrics({'a': 1.0})
            run_first = logger._run
            logger.log_metrics({'a': 2.0})
            # _initialized guard means the run is not recreated.
            assert logger._run is run_first
