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

"""Tests for the logging backends (``braintools.trainer._loggers``).

The optional backend libraries (tensorboard, wandb, neptune, mlflow) are not
installed in the standard CI environment. The happy-path tests at the end of
this file (formerly ``_loggers_backends_test.py``) inject lightweight fake
modules into ``sys.modules`` so the logger code runs against a recording
double, covering ``log_metrics``/``log_hyperparams``/``log_image``/``log_text``/
``finalize``.
"""

import csv
import os
import sys
import tempfile
import types
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest

import braintools
from braintools.trainer._loggers import (
    Logger,
    CSVLogger,
    CompositeLogger,
    TensorBoardLogger,
    WandBLogger,
    NeptuneLogger,
    MLFlowLogger,
)


class _RecordingLogger(Logger):
    """Minimal concrete Logger that records every call (for base-class tests)."""

    def __init__(self):
        super().__init__()
        self.metrics_calls = []
        self.hparams_calls = []
        self.graph_calls = []
        self.image_calls = []
        self.text_calls = []
        self.artifact_calls = []
        self.saved = 0
        self.finalized = []

    def log_metrics(self, metrics, step=None):
        self.metrics_calls.append((metrics, step))

    def log_hyperparams(self, params):
        self.hparams_calls.append(params)

    def log_graph(self, model, input_array=None):
        self.graph_calls.append((model, input_array))

    def log_image(self, key, images, step=None):
        self.image_calls.append((key, images, step))

    def log_text(self, key, text, step=None):
        self.text_calls.append((key, text, step))

    def log_artifact(self, local_path, artifact_path=None):
        self.artifact_calls.append((local_path, artifact_path))

    def save(self):
        self.saved += 1

    def finalize(self, status='success'):
        self.finalized.append(status)


class TestLoggerBase:
    """Tests for the abstract ``Logger`` base class behaviour."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Logger()

    def test_default_properties(self):
        logger = _RecordingLogger()
        assert logger.name == 'default'
        assert logger.version is None
        assert logger.root_dir is None
        assert logger.log_dir is None

    def test_default_optional_hooks_are_noops(self):
        """The base class provides no-op default implementations."""
        logger = _RecordingLogger()
        # These are defined on the base class (not overridden as no-ops here),
        # exercise them through a plain subclass that does NOT override them.

        class _BareLogger(Logger):
            def log_metrics(self, metrics, step=None):
                pass

            def log_hyperparams(self, params):
                pass

        bare = _BareLogger()
        # All of these are base-class no-ops returning None.
        assert bare.log_graph(object()) is None
        assert bare.log_image('k', np.zeros((2, 2))) is None
        assert bare.log_text('k', 'hello') is None
        assert bare.log_artifact('/tmp/none') is None
        assert bare.save() is None
        assert bare.finalize('failed') is None

    def test_recording_subclass_dispatch(self):
        logger = _RecordingLogger()
        logger.log_metrics({'a': 1.0}, step=3)
        logger.log_hyperparams({'lr': 0.1})
        logger.log_graph('model', 'inp')
        logger.log_image('img', np.zeros((2, 2)), step=1)
        logger.log_text('t', 'note', step=2)
        logger.log_artifact('/p', 'ap')
        logger.save()
        logger.finalize('interrupted')

        assert logger.metrics_calls == [({'a': 1.0}, 3)]
        assert logger.hparams_calls == [{'lr': 0.1}]
        assert logger.graph_calls == [('model', 'inp')]
        assert logger.image_calls == [('img', logger.image_calls[0][1], 1)]
        assert logger.text_calls == [('t', 'note', 2)]
        assert logger.artifact_calls == [('/p', 'ap')]
        assert logger.saved == 1
        assert logger.finalized == ['interrupted']


def _read_csv_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


class TestCSVLogger:
    """Thorough tests for ``CSVLogger``."""

    def test_paths_and_properties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(tmpdir, name='exp', version='v1')
            assert logger.name == 'exp'
            assert logger.version == 'v1'
            assert logger.root_dir == tmpdir
            assert logger.log_dir == os.path.join(tmpdir, 'exp', 'v1')
            assert logger.metrics_file_path.endswith('metrics.csv')
            assert logger.hparams_file_path.endswith('hparams.yaml')

    def test_auto_version_generated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(tmpdir, name='exp')
            # auto-generated timestamp version string
            assert logger.version is not None
            assert len(logger.version) > 0

    def test_log_metrics_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(tmpdir, name='exp', version='v1')
            logger.log_metrics({'loss': 0.5, 'acc': 0.9}, step=0)
            logger.log_metrics({'loss': 0.4, 'acc': 0.95}, step=1)
            logger.save()

            rows = _read_csv_rows(logger.metrics_file_path)
            assert len(rows) == 2
            assert rows[0]['step'] == '0'
            assert rows[0]['loss'] == '0.5'
            assert rows[1]['acc'] == '0.95'

    def test_log_metrics_step_defaults_to_counter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(tmpdir, name='exp', version='v1')
            # No explicit step -> uses internal counter (0, then 1).
            logger.log_metrics({'loss': 1.0})
            logger.log_metrics({'loss': 2.0})
            logger.save()

            rows = _read_csv_rows(logger.metrics_file_path)
            assert [r['step'] for r in rows] == ['0', '1']

    def test_auto_flush_on_interval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(
                tmpdir, name='exp', version='v1', flush_logs_every_n_steps=2
            )
            logger.log_metrics({'loss': 0.1})  # buffered
            assert not os.path.exists(logger.metrics_file_path)
            logger.log_metrics({'loss': 0.2})  # triggers flush at interval
            assert os.path.exists(logger.metrics_file_path)
            rows = _read_csv_rows(logger.metrics_file_path)
            assert len(rows) == 2

    def test_append_across_saves_keeps_single_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(tmpdir, name='exp', version='v1')
            logger.log_metrics({'loss': 0.1}, step=0)
            logger.save()
            logger.log_metrics({'loss': 0.2}, step=1)
            logger.save()

            with open(logger.metrics_file_path, newline='') as f:
                content = f.read()
            # Header 'step,loss' should appear only once.
            assert content.count('loss') == 1
            rows = _read_csv_rows(logger.metrics_file_path)
            assert len(rows) == 2

    def test_save_with_empty_buffer_is_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(tmpdir, name='exp', version='v1')
            # Nothing buffered: save returns early, no file is created.
            logger.save()
            assert not os.path.exists(logger.metrics_file_path)

    def test_log_hyperparams_yaml(self):
        pytest.importorskip('yaml')
        import yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(tmpdir, name='exp', version='v1')
            logger.log_hyperparams({'lr': 0.001, 'batch_size': 32})
            assert os.path.exists(logger.hparams_file_path)

            with open(logger.hparams_file_path) as f:
                loaded = yaml.safe_load(f)
            assert loaded['lr'] == 0.001
            assert loaded['batch_size'] == 32

    def test_log_hyperparams_accumulates(self):
        pytest.importorskip('yaml')
        import yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(tmpdir, name='exp', version='v1')
            logger.log_hyperparams({'lr': 0.1})
            logger.log_hyperparams({'momentum': 0.9})
            with open(logger.hparams_file_path) as f:
                loaded = yaml.safe_load(f)
            assert loaded == {'lr': 0.1, 'momentum': 0.9}

    def test_finalize_flushes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(tmpdir, name='exp', version='v1')
            logger.log_metrics({'loss': 0.5}, step=0)
            # Not yet flushed (interval default is large).
            assert not os.path.exists(logger.metrics_file_path)
            logger.finalize()
            assert os.path.exists(logger.metrics_file_path)
            assert len(_read_csv_rows(logger.metrics_file_path)) == 1

    def test_extra_action_ignore_for_new_keys(self):
        """Keys introduced after the first row still get a header column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(tmpdir, name='exp', version='v1')
            logger.log_metrics({'loss': 0.5}, step=0)
            logger.log_metrics({'loss': 0.4, 'acc': 0.9}, step=1)
            logger.save()
            rows = _read_csv_rows(logger.metrics_file_path)
            # 'acc' should be a recognized fieldname now.
            assert 'acc' in rows[1]
            assert rows[1]['acc'] == '0.9'


class TestCompositeLogger:
    """Tests for ``CompositeLogger`` fan-out semantics."""

    def test_name_version_delegate_to_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            l1 = CSVLogger(tmpdir, name='first', version='va')
            l2 = CSVLogger(tmpdir, name='second', version='vb')
            comp = CompositeLogger([l1, l2])
            assert comp.name == 'first'
            assert comp.version == 'va'

    def test_empty_composite_defaults(self):
        comp = CompositeLogger([])
        assert comp.name == 'default'
        assert comp.version is None

    def test_fan_out_metrics_to_both(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            l1 = CSVLogger(tmpdir, name='c1', version='v1')
            l2 = CSVLogger(tmpdir, name='c2', version='v1')
            comp = CompositeLogger([l1, l2])

            comp.log_metrics({'loss': 0.3}, step=0)
            comp.log_metrics({'loss': 0.2}, step=1)
            comp.save()
            comp.finalize()

            assert os.path.exists(l1.metrics_file_path)
            assert os.path.exists(l2.metrics_file_path)
            assert len(_read_csv_rows(l1.metrics_file_path)) == 2
            assert len(_read_csv_rows(l2.metrics_file_path)) == 2

    def test_fan_out_hyperparams(self):
        pytest.importorskip('yaml')
        with tempfile.TemporaryDirectory() as tmpdir:
            l1 = CSVLogger(tmpdir, name='c1', version='v1')
            l2 = CSVLogger(tmpdir, name='c2', version='v1')
            comp = CompositeLogger([l1, l2])
            comp.log_hyperparams({'lr': 0.01})
            assert os.path.exists(l1.hparams_file_path)
            assert os.path.exists(l2.hparams_file_path)

    def test_fan_out_optional_hooks(self):
        """log_graph/log_image/log_text/log_artifact fan out to all loggers."""
        recorders = [_RecordingLogger(), _RecordingLogger()]
        comp = CompositeLogger(recorders)

        comp.log_graph('m', 'in')
        comp.log_image('img', np.zeros((2, 2)), step=4)
        comp.log_text('t', 'txt', step=5)
        comp.log_artifact('/local', 'remote')

        for r in recorders:
            assert r.graph_calls == [('m', 'in')]
            assert r.image_calls[0][0] == 'img'
            assert r.image_calls[0][2] == 4
            assert r.text_calls == [('t', 'txt', 5)]
            assert r.artifact_calls == [('/local', 'remote')]


class TestTensorBoardLogger:
    """Tests for ``TensorBoardLogger`` (paths + import-guard error path)."""

    def test_paths_and_default_version(self):
        logger = TensorBoardLogger('logs', name='tb', version='v9', prefix='p/')
        assert logger.root_dir == 'logs'
        assert logger.log_dir == os.path.join('logs', 'tb', 'v9')

    def test_auto_version(self):
        logger = TensorBoardLogger('logs', name='tb')
        assert logger.version is not None and len(logger.version) > 0

    def test_log_graph_disabled_is_noop(self):
        logger = TensorBoardLogger('logs', name='tb', log_graph=False)
        # log_graph disabled -> returns immediately, no warning, no writer needed.
        assert logger.log_graph(object()) is None

    def test_log_graph_enabled_warns(self):
        logger = TensorBoardLogger('logs', name='tb', log_graph=True)
        with pytest.warns(UserWarning):
            logger.log_graph(object())

    def test_init_writer_requires_backend(self):
        """If neither tensorboard nor tensorboardX is installed, log raises."""
        try:
            import torch.utils.tensorboard  # noqa: F401
            has_tb = True
        except ImportError:
            try:
                import tensorboardX  # noqa: F401
                has_tb = True
            except ImportError:
                has_tb = False

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(tmpdir, name='tb', version='v1')
            if has_tb:
                logger.log_metrics({'loss': 0.5}, step=0)
                logger.save()
                logger.finalize()
                assert os.path.exists(logger.log_dir)
            else:
                with pytest.raises(ImportError, match='tensorboard'):
                    logger.log_metrics({'loss': 0.5}, step=0)


class TestWandBLogger:
    """Tests for ``WandBLogger`` (construction + import-guard error path)."""

    def test_construction_defaults(self):
        logger = WandBLogger(project='proj')
        # Before init, name falls back to 'default', version is None.
        assert logger.name == 'default'
        assert logger.version is None
        assert logger.log_dir is None

    def test_log_metrics_without_wandb_raises(self):
        wandb_installed = True
        try:
            import wandb  # noqa: F401
        except ImportError:
            wandb_installed = False

        logger = WandBLogger(project='proj', name='run')
        # name comes back as the supplied value regardless of init.
        assert logger.name == 'run'
        if not wandb_installed:
            with pytest.raises(ImportError, match='wandb'):
                logger.log_metrics({'loss': 0.5}, step=0)
        else:
            pytest.skip('wandb installed; skipping import-guard test')

    def test_save_is_noop(self):
        logger = WandBLogger(project='proj')
        assert logger.save() is None

    def test_finalize_without_run_is_noop(self):
        logger = WandBLogger(project='proj')
        # _run is None -> finalize is a no-op (no import needed).
        assert logger.finalize() is None


class TestNeptuneLogger:
    """Tests for ``NeptuneLogger`` (construction + import-guard error path)."""

    def test_construction(self):
        logger = NeptuneLogger(project='ws/proj', name='run', tags=['a'])
        assert logger.name == 'run'

    def test_log_metrics_without_neptune_raises(self):
        try:
            import neptune  # noqa: F401
            pytest.skip('neptune installed; skipping import-guard test')
        except ImportError:
            pass
        logger = NeptuneLogger(project='ws/proj')
        with pytest.raises(ImportError, match='neptune'):
            logger.log_metrics({'loss': 0.5}, step=0)

    def test_log_hyperparams_without_neptune_raises(self):
        try:
            import neptune  # noqa: F401
            pytest.skip('neptune installed; skipping import-guard test')
        except ImportError:
            pass
        logger = NeptuneLogger(project='ws/proj')
        with pytest.raises(ImportError, match='neptune'):
            logger.log_hyperparams({'lr': 0.1})

    def test_save_and_finalize_without_run_are_noops(self):
        logger = NeptuneLogger(project='ws/proj')
        assert logger.save() is None
        assert logger.finalize() is None


class TestMLFlowLogger:
    """Tests for ``MLFlowLogger`` (construction + import-guard error path)."""

    def test_construction(self):
        logger = MLFlowLogger(experiment_name='exp', run_name='r')
        assert logger.save() is None  # MLflow auto-saves -> no-op

    def test_log_metrics_without_mlflow_raises(self):
        try:
            import mlflow  # noqa: F401
            pytest.skip('mlflow installed; skipping import-guard test')
        except ImportError:
            pass
        logger = MLFlowLogger(experiment_name='exp')
        with pytest.raises(ImportError, match='mlflow'):
            logger.log_metrics({'loss': 0.5}, step=0)

    def test_log_params_without_mlflow_raises(self):
        try:
            import mlflow  # noqa: F401
            pytest.skip('mlflow installed; skipping import-guard test')
        except ImportError:
            pass
        logger = MLFlowLogger(experiment_name='exp')
        with pytest.raises(ImportError, match='mlflow'):
            logger.log_hyperparams({'lr': 0.1})

    def test_log_artifact_without_mlflow_raises(self):
        try:
            import mlflow  # noqa: F401
            pytest.skip('mlflow installed; skipping import-guard test')
        except ImportError:
            pass
        logger = MLFlowLogger(experiment_name='exp')
        with pytest.raises(ImportError, match='mlflow'):
            logger.log_artifact('/local/path')

    def test_finalize_without_run_is_noop(self):
        logger = MLFlowLogger(experiment_name='exp')
        assert logger.finalize() is None


class TestPublicExports:
    """The loggers should be importable from the public namespace."""

    def test_all_exports_present(self):
        for name in (
            'Logger',
            'TensorBoardLogger',
            'WandBLogger',
            'CSVLogger',
            'CompositeLogger',
            'NeptuneLogger',
            'MLFlowLogger',
        ):
            assert hasattr(braintools.trainer, name)


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
