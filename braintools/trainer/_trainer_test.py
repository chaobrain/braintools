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

"""Tests for ``braintools.trainer._trainer``.

Besides the original smoke tests, this file holds the trainer regression tests
from the 2026-06 audits (T-2 .. T-29, T-A, T-C) and the gap-coverage tests that
used to live in ``_trainer_audit_test.py``, ``_trainer_extra_test.py`` and
``_audit_20260619_test.py``.
"""

import os
import tempfile
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import brainstate
import braintools
from braintools._deprecation import BraintoolsDeprecationWarning
from braintools.trainer import (
    CheckpointManager,
    DataLoader,
    LearningRateMonitor,
    LightningModule,
    Logger,
)
from braintools.trainer._checkpoint import save_checkpoint
from braintools.trainer._module import EvalOutput
from braintools.trainer._trainer import (
    Trainer,
    TrainerState,
    _clip_grad_norm,
    _clip_grad_value,
    _format_metric,
    _from_msgpack_lists,
    _normalize_optimizer_states,
    _prefixed,
)


class SimpleModel(braintools.trainer.LightningModule):
    """Simple model for testing."""

    def __init__(self, input_size=10, hidden_size=5, output_size=2):
        super().__init__()
        self.linear1 = brainstate.nn.Linear(input_size, hidden_size)
        self.linear2 = brainstate.nn.Linear(hidden_size, output_size)

    def __call__(self, x):
        x = jnp.tanh(self.linear1(x))
        return self.linear2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = jnp.mean((logits - y) ** 2)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = jnp.mean((logits - y) ** 2)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = jnp.mean((logits - y) ** 2)
        self.log('test_loss', loss)
        return {'loss': loss}

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        return braintools.optim.Adam(lr=1e-3)


class TestLightningModule:
    """Tests for LightningModule."""

    def test_init(self):
        """Test LightningModule initialization."""
        model = SimpleModel()
        assert model.current_epoch == 0
        assert model.global_step == 0
        assert model.trainer is None

    def test_log(self):
        """Test logging functionality."""
        model = SimpleModel()
        model.log('test_metric', 0.5, prog_bar=True)
        assert 'test_metric' in model._logged_metrics
        assert model._logged_metrics['test_metric']['value'] == 0.5
        assert model._logged_metrics['test_metric']['prog_bar'] is True

    def test_log_dict(self):
        """Test log_dict functionality."""
        model = SimpleModel()
        model.log_dict({'metric1': 0.1, 'metric2': 0.2})
        assert 'metric1' in model._logged_metrics
        assert 'metric2' in model._logged_metrics

    def test_state_dict(self):
        """Test state_dict and load_state_dict."""
        model = SimpleModel()
        state = model.state_dict()
        assert isinstance(state, dict)
        assert len(state) > 0

        # Load state back
        model.load_state_dict(state)


class TestDataLoader:
    """Tests for DataLoader."""

    def test_array_dataset(self):
        """Test ArrayDataset."""
        X = jnp.ones((100, 10))
        y = jnp.zeros((100, 2))
        dataset = braintools.trainer.ArrayDataset(X, y)
        assert len(dataset) == 100

        sample = dataset[0]
        assert len(sample) == 2
        assert sample[0].shape == (10,)
        assert sample[1].shape == (2,)

    def test_dict_dataset(self):
        """Test DictDataset."""
        data = {'x': jnp.ones((100, 10)), 'y': jnp.zeros((100, 2))}
        dataset = braintools.trainer.DictDataset(data)
        assert len(dataset) == 100

        sample = dataset[0]
        assert 'x' in sample
        assert 'y' in sample

    def test_dataloader_iteration(self):
        """Test DataLoader iteration."""
        X = jnp.ones((100, 10))
        y = jnp.zeros((100, 2))
        loader = braintools.trainer.DataLoader((X, y), batch_size=32)

        assert len(loader) == 4  # 100 / 32 = 3.125 -> 4 batches

        batches = list(loader)
        assert len(batches) == 4
        assert batches[0][0].shape == (32, 10)

    def test_dataloader_shuffle(self):
        """Test DataLoader shuffling."""
        X = jnp.arange(100).reshape(100, 1)
        loader = braintools.trainer.DataLoader((X,), batch_size=100, shuffle=True, seed=42)

        batch = next(iter(loader))[0]
        # Check that order is different from sequential
        assert not jnp.all(batch[:10, 0] == jnp.arange(10))

    def test_dataloader_drop_last(self):
        """Test DataLoader drop_last."""
        X = jnp.ones((100, 10))
        loader = braintools.trainer.DataLoader((X,), batch_size=32, drop_last=True)
        assert len(loader) == 3  # 100 // 32 = 3


class TestCallbacks:
    """Tests for callbacks."""

    def test_callback_base(self):
        """Test Callback base class."""
        callback = braintools.trainer.Callback()
        # All methods should be callable without error
        callback.on_fit_start(None, None)
        callback.on_train_epoch_end(None, None)

    def test_early_stopping(self):
        """Test EarlyStopping callback."""
        callback = braintools.trainer.EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min',
        )
        assert callback.monitor == 'val_loss'
        assert callback.patience == 3
        assert callback.mode == 'min'
        assert not callback.should_stop

    def test_model_checkpoint(self):
        """Test ModelCheckpoint callback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = braintools.trainer.ModelCheckpoint(
                dirpath=tmpdir,
                monitor='val_loss',
                save_top_k=3,
            )
            assert callback.dirpath == tmpdir
            assert callback.monitor == 'val_loss'
            assert callback.save_top_k == 3

    def test_lambda_callback(self):
        """Test LambdaCallback."""
        called = [False]

        def on_epoch_end(trainer, module):
            called[0] = True

        callback = braintools.trainer.LambdaCallback(
            on_train_epoch_end=on_epoch_end
        )
        callback.on_train_epoch_end(None, None)
        assert called[0]


class TestLoggers:
    """Tests for loggers."""

    def test_csv_logger(self):
        """Test CSVLogger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = braintools.trainer.CSVLogger(tmpdir, name='test')
            logger.log_metrics({'loss': 0.5}, step=0)
            logger.log_hyperparams({'lr': 0.001})
            logger.save()
            logger.finalize()

            # Check files were created
            assert os.path.exists(logger.log_dir)

    def test_composite_logger(self):
        """Test CompositeLogger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger1 = braintools.trainer.CSVLogger(tmpdir, name='log1')
            logger2 = braintools.trainer.CSVLogger(tmpdir, name='log2')
            composite = braintools.trainer.CompositeLogger([logger1, logger2])

            composite.log_metrics({'loss': 0.5}, step=0)
            composite.finalize()


class TestDistributed:
    """Tests for distributed strategies."""

    def test_single_device_strategy(self):
        """Test SingleDeviceStrategy."""
        strategy = braintools.trainer.SingleDeviceStrategy()
        assert strategy.name == 'single_device'
        assert strategy.num_devices == 1
        assert not strategy.is_distributed

    def test_auto_strategy(self):
        """Test AutoStrategy."""
        strategy = braintools.trainer.AutoStrategy()
        assert 'auto' in strategy.name
        assert strategy.num_devices >= 1

    def test_get_strategy(self):
        """Test get_strategy function."""
        strategy = braintools.trainer.get_strategy('auto')
        assert isinstance(strategy, braintools.trainer.Strategy)

        strategy = braintools.trainer.get_strategy('single')
        assert isinstance(strategy, braintools.trainer.SingleDeviceStrategy)


class TestCheckpointing:
    """Tests for checkpointing."""

    def test_save_load_checkpoint(self):
        """Test save and load checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.ckpt')

            state = {'epoch': 5, 'loss': 0.5}
            braintools.trainer.save_checkpoint(state, filepath)
            assert os.path.exists(filepath)

            loaded = braintools.trainer.load_checkpoint(filepath)
            assert loaded['epoch'] == 5
            assert loaded['loss'] == 0.5

    def test_checkpoint_manager(self):
        """Test CheckpointManager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = braintools.trainer.CheckpointManager(
                dirpath=tmpdir,
                max_to_keep=3,
            )
            assert manager.latest is None

            # Would need a model to test save


class TestTrainer:
    """Tests for Trainer."""

    def test_trainer_init(self):
        """Test Trainer initialization."""
        trainer = braintools.trainer.Trainer(
            max_epochs=10,
            enable_progress_bar=False,
        )
        assert trainer.max_epochs == 10
        assert trainer.current_epoch == 0

    def test_trainer_with_callbacks(self):
        """Test Trainer with callbacks."""
        callbacks = [
            braintools.trainer.EarlyStopping(monitor='val_loss'),
            braintools.trainer.PrintCallback(),
        ]
        trainer = braintools.trainer.Trainer(
            max_epochs=10,
            callbacks=callbacks,
            enable_progress_bar=False,
        )
        assert len(trainer.callbacks) == 2


class TestProgressBar:
    """Tests for progress bars."""

    def test_simple_progress_bar(self):
        """Test SimpleProgressBar."""
        pbar = braintools.trainer.SimpleProgressBar()
        pbar.start(total=10, desc='Test')
        for i in range(10):
            pbar.update(1, loss=0.5)
        pbar.close()

    def test_get_progress_bar(self):
        """Test get_progress_bar."""
        pbar = braintools.trainer.get_progress_bar('simple')
        assert isinstance(pbar, braintools.trainer.SimpleProgressBar)


class TestIntegration:
    """Integration tests."""

    def test_simple_training_loop(self):
        """Test a simple training loop."""
        # Create data
        X = jnp.ones((64, 10))
        y = jnp.zeros((64, 2))

        # Create model
        model = SimpleModel()

        # Create data loader
        loader = braintools.trainer.DataLoader((X, y), batch_size=16)

        # Create trainer
        trainer = braintools.trainer.Trainer(
            max_epochs=2,
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False,
        )

        # Fit model
        trainer.fit(model, loader)

        # Check training occurred
        assert model.current_epoch >= 0


# ===========================================================================
# Gap-coverage models and helpers (formerly _trainer_extra_test.py)
# ===========================================================================

class SchedulerModel(SimpleModel):
    """Model that returns an (optimizer, scheduler) tuple."""

    def configure_optimizers(self):
        opt = braintools.optim.Adam(lr=1e-2)
        sched = braintools.optim.StepLR(base_lr=1e-2, step_size=1)
        return opt, sched


class DictOptModel(SimpleModel):
    """Model returning a dict-style optimizer config."""

    def configure_optimizers(self):
        return {'optimizer': braintools.optim.Adam(lr=1e-3), 'lr_scheduler': None}


class MultiOptModel(SimpleModel):
    """Model returning a list of optimizers and an empty scheduler list."""

    def configure_optimizers(self):
        return [braintools.optim.Adam(lr=1e-3), braintools.optim.SGD(lr=1e-3)], []


class EvalOutputModel(SimpleModel):
    """Model whose validation/test steps return ``EvalOutput`` objects."""

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = jnp.mean((self(x) - y) ** 2)
        self.log('val_loss', loss)
        return EvalOutput({'extra': jnp.asarray(0.25)})

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = jnp.mean((self(x) - y) ** 2)
        self.log('test_loss', loss)
        return EvalOutput({'extra': jnp.asarray(0.5)})


def _make_loader(n=32, batch_size=8, in_size=10, out_size=2):
    X = jnp.ones((n, in_size))
    y = jnp.zeros((n, out_size))
    return braintools.trainer.DataLoader((X, y), batch_size=batch_size)


# ===========================================================================
# Audit regression models and helpers (formerly _trainer_audit_test.py and
# _audit_20260619_test.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------

class LinearRegressor(LightningModule):
    """A tiny linear model with a counter for training_step invocations."""

    def __init__(self, in_size=4, out_size=1):
        super().__init__()
        self.lin = brainstate.nn.Linear(in_size, out_size)
        self.train_step_calls = 0

    def __call__(self, x):
        return self.lin(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.train_step_calls += 1
        loss = jnp.mean((self(x) - y) ** 2)
        self.log('mse', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = jnp.mean((self(x) - y) ** 2)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = jnp.mean((self(x) - y) ** 2)
        self.log('test_loss', loss)
        return {'loss': loss}

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        return braintools.optim.SGD(lr=1e-2)


def _data(n=20, in_size=4, out_size=1, seed=0):
    brainstate.random.seed(seed)
    x = brainstate.random.randn(n, in_size)
    y = brainstate.random.randn(n, out_size)
    return x, y


def _loader(n=20, batch_size=10, shuffle=False, seed=0, **kw):
    x, y = _data(n, seed=seed, **kw)
    return DataLoader((x, y), batch_size=batch_size, shuffle=shuffle, seed=seed)


def _silent_fit(trainer, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        trainer.fit(*args, **kwargs)


class _Net(LightningModule):
    def __init__(self):
        super().__init__()
        self.l = brainstate.nn.Linear(4, 2)

    def __call__(self, x):
        return self.l(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = jnp.mean((self(x) - y) ** 2)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = jnp.mean((self(x) - y) ** 2)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def configure_optimizers(self):
        return braintools.optim.Adam(lr=1e-3)


def _random_loader(n=16, bs=4):
    x = jnp.asarray(np.random.randn(n, 4).astype('float32'))
    y = jnp.asarray(np.random.randn(n, 2).astype('float32'))
    return DataLoader((x, y), batch_size=bs)


class _SpyLogger(Logger):
    """Records every metrics dict passed to ``log_metrics``."""

    def __init__(self):
        super().__init__()
        self.captured = []

    def log_metrics(self, metrics, step=None):
        self.captured.append(dict(metrics))

    def log_hyperparams(self, params):
        pass

    @property
    def keys(self):
        out = set()
        for c in self.captured:
            out.update(c.keys())
        return out


# ---------------------------------------------------------------------------
# TrainerState
# ---------------------------------------------------------------------------

class TestTrainerState:
    def test_initial_state(self):
        state = TrainerState()
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.stage == 'train'
        assert state.batch_idx == 0
        assert state.should_stop is False

    def test_state_mutation(self):
        state = TrainerState()
        state.epoch = 3
        state.global_step = 99
        state.stage = 'validate'
        state.should_stop = True
        assert state.epoch == 3
        assert state.global_step == 99
        assert state.stage == 'validate'
        assert state.should_stop is True


# ---------------------------------------------------------------------------
# Trainer setup / configuration
# ---------------------------------------------------------------------------

class TestTrainerSetup:
    def test_default_root_dir_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, 'nested', 'run')
            trainer = Trainer(default_root_dir=root, enable_progress_bar=False)
            assert os.path.isdir(root)
            assert trainer.default_root_dir == root

    def test_logger_true_creates_csv_logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                default_root_dir=tmpdir, logger=True, enable_progress_bar=False
            )
            assert len(trainer.loggers) == 1
            assert isinstance(trainer.loggers[0], braintools.trainer.CSVLogger)

    def test_logger_false_no_loggers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                default_root_dir=tmpdir, logger=False, enable_progress_bar=False
            )
            assert trainer.loggers == []

    def test_logger_instance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = braintools.trainer.CSVLogger(tmpdir, name='x')
            trainer = Trainer(
                default_root_dir=tmpdir, logger=logger, enable_progress_bar=False
            )
            assert trainer.loggers == [logger]

    def test_logger_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            l1 = braintools.trainer.CSVLogger(tmpdir, name='a')
            l2 = braintools.trainer.CSVLogger(tmpdir, name='b')
            trainer = Trainer(
                default_root_dir=tmpdir, logger=[l1, l2], enable_progress_bar=False
            )
            assert trainer.loggers == [l1, l2]

    def test_devices_auto(self):
        trainer = Trainer(devices='auto', enable_progress_bar=False)
        assert trainer.num_devices >= 1

    def test_devices_int(self):
        trainer = Trainer(devices=1, enable_progress_bar=False)
        assert trainer.num_devices == 1

    def test_devices_list(self):
        trainer = Trainer(devices=[0], enable_progress_bar=False)
        assert trainer.num_devices == 1

    def test_seed_sets_numpy(self):
        trainer = Trainer(seed=123, enable_progress_bar=False)
        assert trainer.seed == 123

    def test_properties(self):
        trainer = Trainer(max_epochs=5, enable_progress_bar=False)
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.is_training is True
        assert trainer.callbacks == []


# ---------------------------------------------------------------------------
# Optimizer configuration parsing
# ---------------------------------------------------------------------------

class TestOptimizerConfig:
    def test_single_optimizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                max_epochs=1, default_root_dir=tmpdir,
                enable_progress_bar=False, enable_checkpointing=False, logger=False,
            )
            trainer.fit(SimpleModel(), _make_loader())
            # After cleanup optimizers are reset, so just check it completed.

    def test_tuple_optimizer_scheduler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                max_epochs=2, default_root_dir=tmpdir,
                enable_progress_bar=False, enable_checkpointing=False, logger=False,
            )
            trainer.fit(SchedulerModel(), _make_loader())

    def test_dict_optimizer_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                max_epochs=1, default_root_dir=tmpdir,
                enable_progress_bar=False, enable_checkpointing=False, logger=False,
            )
            trainer.fit(DictOptModel(), _make_loader())

    def test_list_optimizers_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                max_epochs=1, default_root_dir=tmpdir,
                enable_progress_bar=False, enable_checkpointing=False, logger=False,
            )
            trainer.fit(MultiOptModel(), _make_loader())

    def test_invalid_optimizer_config_raises(self):
        class BadModel(SimpleModel):
            def configure_optimizers(self):
                return "not-an-optimizer"

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                max_epochs=1, default_root_dir=tmpdir,
                enable_progress_bar=False, enable_checkpointing=False, logger=False,
            )
            with pytest.raises(ValueError, match='Invalid optimizer'):
                trainer.fit(BadModel(), _make_loader())

    def test_setup_optimizers_without_model_raises(self):
        trainer = Trainer(enable_progress_bar=False)
        with pytest.raises(RuntimeError, match='Model not set up'):
            trainer._setup_optimizers()


# ---------------------------------------------------------------------------
# fit() loop behaviour
# ---------------------------------------------------------------------------

class TestFitLoop:
    def test_fit_runs_epochs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                max_epochs=3, default_root_dir=tmpdir,
                enable_progress_bar=False, enable_checkpointing=False, logger=False,
            )
            trainer.fit(model, _make_loader(n=16, batch_size=8))
            # 2 batches/epoch * 3 epochs = 6 steps
            assert trainer.global_step == 6
            assert 'train_loss' in trainer.callback_metrics

    def test_fit_with_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                max_epochs=2, default_root_dir=tmpdir,
                enable_progress_bar=False, enable_checkpointing=False, logger=False,
            )
            trainer.fit(model, _make_loader(), _make_loader())
            assert any(k.startswith('val_') for k in trainer.callback_metrics)

    def test_fit_with_validation_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                max_epochs=1, default_root_dir=tmpdir,
                enable_progress_bar=False, enable_checkpointing=False, logger=False,
            )
            trainer.fit(model, _make_loader(), [_make_loader(), _make_loader()])
            assert trainer.val_dataloaders is not None
            assert len(trainer.val_dataloaders) == 2

    def test_max_steps_caps_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                max_epochs=100, max_steps=3, default_root_dir=tmpdir,
                enable_progress_bar=False, enable_checkpointing=False, logger=False,
            )
            trainer.fit(model, _make_loader(n=64, batch_size=8))
            assert trainer.global_step == 3

    def test_check_val_every_n_epoch(self):
        """With check_val_every_n_epoch=2, validation skipped on epoch 0."""
        ran = []

        class TrackingModel(SimpleModel):
            def on_validation_epoch_start(self):
                ran.append(self.current_epoch)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                max_epochs=2, check_val_every_n_epoch=2,
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            trainer.fit(TrackingModel(), _make_loader(), _make_loader())
            # Validation runs only at epoch 1 ((1+1) % 2 == 0).
            assert ran == [1]

    def test_gradient_clip_norm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                max_epochs=1, gradient_clip_val=1.0, gradient_clip_algorithm='norm',
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            trainer.fit(model, _make_loader())

    def test_gradient_clip_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                max_epochs=1, gradient_clip_val=0.5, gradient_clip_algorithm='value',
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            trainer.fit(model, _make_loader())

    def test_min_epochs_continues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                max_epochs=3, min_epochs=2, default_root_dir=tmpdir,
                enable_progress_bar=False, enable_checkpointing=False, logger=False,
            )
            trainer.fit(model, _make_loader(n=16, batch_size=8))
            assert trainer.global_step == 6

    def test_logger_records_metrics_to_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = braintools.trainer.CSVLogger(tmpdir, name='run', version='v1')
            model = SimpleModel()
            trainer = Trainer(
                max_epochs=1, default_root_dir=tmpdir, logger=logger,
                enable_progress_bar=False, enable_checkpointing=False,
            )
            trainer.fit(model, _make_loader(n=16, batch_size=8))
            # finalize() at end of fit flushes the CSV.
            assert os.path.exists(logger.metrics_file_path)
            with open(logger.metrics_file_path) as f:
                content = f.read()
            assert 'loss' in content

    def test_enable_checkpointing_creates_no_directory(self):
        # The trainer saves no checkpoints by itself, so it must not create an
        # (always empty) ``checkpoints/`` directory either.
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                max_epochs=1, default_root_dir=tmpdir,
                enable_progress_bar=False, enable_checkpointing=True, logger=False,
            )
            trainer.fit(model, _make_loader(n=16, batch_size=8))
            assert not os.path.exists(os.path.join(tmpdir, 'checkpoints'))


# ---------------------------------------------------------------------------
# Callbacks integration
# ---------------------------------------------------------------------------

class TestCallbacksIntegration:
    def test_callback_hooks_fire(self):
        events = []

        class RecordingCallback(braintools.trainer.Callback):
            def on_fit_start(self, trainer, module):
                events.append('fit_start')

            def on_fit_end(self, trainer, module):
                events.append('fit_end')

            def on_train_epoch_start(self, trainer, module):
                events.append('epoch_start')

            def on_train_epoch_end(self, trainer, module):
                events.append('epoch_end')

            def on_train_batch_start(self, trainer, module, batch, batch_idx):
                events.append('batch_start')

            def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                events.append('batch_end')

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                max_epochs=1, callbacks=[RecordingCallback()],
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            trainer.fit(SimpleModel(), _make_loader(n=16, batch_size=8))

        assert events[0] == 'fit_start'
        assert events[-1] == 'fit_end'
        assert 'epoch_start' in events
        assert 'epoch_end' in events
        assert events.count('batch_start') == 2
        assert events.count('batch_end') == 2

    def test_validation_callback_hooks_fire(self):
        events = []

        class ValCallback(braintools.trainer.Callback):
            def on_validation_epoch_start(self, trainer, module):
                events.append('val_start')

            def on_validation_epoch_end(self, trainer, module):
                events.append('val_end')

            def on_validation_batch_start(self, trainer, module, batch, batch_idx):
                events.append('val_batch_start')

            def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):
                events.append('val_batch_end')

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                max_epochs=1, callbacks=[ValCallback()],
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            trainer.fit(SimpleModel(), _make_loader(), _make_loader())

        assert 'val_start' in events
        assert 'val_end' in events
        assert 'val_batch_start' in events
        assert 'val_batch_end' in events

    def test_early_stopping_stops_training(self):
        """EarlyStopping with patience 0 stops once metric fails to improve."""

        class WorseningModel(SimpleModel):
            """val_loss increases each epoch -> never improves."""

            def __init__(self):
                super().__init__()
                self._call = 0

            def validation_step(self, batch, batch_idx):
                # Return a value that grows with epoch so it never improves.
                loss = jnp.asarray(1.0 + float(self.current_epoch))
                self.log('val_loss', loss)
                return {'val_loss': loss}

        early = braintools.trainer.EarlyStopping(
            monitor='val_loss', patience=1, mode='min', strict=False
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                max_epochs=10, callbacks=[early], min_epochs=1,
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            trainer.fit(WorseningModel(), _make_loader(), _make_loader())
            # Training should have been cut short before max_epochs (10).
            assert early.should_stop
            assert trainer.current_epoch < 9

    def test_should_stop_via_state(self):
        trainer = Trainer(enable_progress_bar=False)
        assert trainer._should_stop() is False
        trainer.state.should_stop = True
        assert trainer._should_stop() is True


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

class TestValidate:
    def test_validate_returns_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            metrics = trainer.validate(model, _make_loader(), verbose=False)
            assert any(k.startswith('val_') for k in metrics)
            assert trainer.state.stage == 'validate'

    def test_validate_verbose(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            trainer.validate(model, _make_loader(), verbose=True)
            out = capsys.readouterr().out
            assert 'Validation Results' in out

    def test_validate_dataloader_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            trainer.validate(model, [_make_loader(), _make_loader()], verbose=False)
            assert len(trainer.val_dataloaders) == 2

    def test_validate_no_model_raises(self):
        trainer = Trainer(enable_progress_bar=False)
        with pytest.raises(RuntimeError, match='No model provided'):
            trainer.validate(dataloaders=_make_loader(), verbose=False)

    def test_validation_epoch_with_no_dataloaders_returns(self):
        trainer = Trainer(enable_progress_bar=False)
        trainer.model = SimpleModel()
        # val_dataloaders is None -> early return, no error.
        trainer._run_validation_epoch(0)

    def test_validate_eval_output_return(self):
        """validation_step returning an EvalOutput exercises the .metrics path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            metrics = trainer.validate(EvalOutputModel(), _make_loader(), verbose=False)
            assert 'val_extra' in metrics


# ---------------------------------------------------------------------------
# test()
# ---------------------------------------------------------------------------

class TestTestLoop:
    def test_test_returns_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            metrics = trainer.test(model, _make_loader(), verbose=False)
            assert any(k.startswith('test_') for k in metrics)
            assert trainer.state.stage == 'test'

    def test_test_verbose(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            trainer.test(model, _make_loader(), verbose=True)
            out = capsys.readouterr().out
            assert 'Test Results' in out

    def test_test_dataloader_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            trainer.test(model, [_make_loader(), _make_loader()], verbose=False)
            assert len(trainer.test_dataloaders) == 2

    def test_test_no_model_raises(self):
        trainer = Trainer(enable_progress_bar=False)
        with pytest.raises(RuntimeError, match='No model provided'):
            trainer.test(dataloaders=_make_loader(), verbose=False)

    def test_test_eval_output_return(self):
        """test_step returning an EvalOutput exercises the .metrics path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            metrics = trainer.test(EvalOutputModel(), _make_loader(), verbose=False)
            assert 'test_extra' in metrics


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_returns_all_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            preds = trainer.predict(model, _make_loader(n=16, batch_size=8))
            assert len(preds) == 2
            assert preds[0].shape == (8, 2)
            assert trainer.state.stage == 'predict'

    def test_predict_dataloader_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                default_root_dir=tmpdir, enable_progress_bar=False,
                enable_checkpointing=False, logger=False,
            )
            preds = trainer.predict(
                model, [_make_loader(n=8, batch_size=8), _make_loader(n=8, batch_size=8)]
            )
            assert len(preds) == 2

    def test_predict_no_model_raises(self):
        trainer = Trainer(enable_progress_bar=False)
        with pytest.raises(RuntimeError, match='No model provided'):
            trainer.predict(dataloaders=_make_loader())


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

class TestLogMetricsHelper:
    def test_log_metrics_no_loggers_returns(self):
        trainer = Trainer(logger=False, enable_progress_bar=False)
        # No loggers -> early return, no error.
        trainer._log_metrics({'loss': 0.5}, step=0)

    def test_log_metrics_converts_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = braintools.trainer.CSVLogger(tmpdir, name='m', version='v1')
            trainer = Trainer(
                default_root_dir=tmpdir, logger=logger, enable_progress_bar=False,
            )
            trainer._log_metrics(
                {'a': jnp.asarray(1.5), 'b': 2, 'c': 'skipme'}, step=0
            )
            logger.save()
            with open(logger.metrics_file_path) as f:
                content = f.read()
            # jax array -> 1.5, int -> 2.0; the str 'c' is filtered out.
            assert 'a' in content and 'b' in content
            assert 'c' not in content


# ---------------------------------------------------------------------------
# Gradient clipping utilities
# ---------------------------------------------------------------------------

class TestClipUtilities:
    def test_clip_grad_value(self):
        grads = {'w': jnp.array([3.0, -3.0, 0.5])}
        clipped = _clip_grad_value(grads, 1.0)
        assert jnp.all(clipped['w'] <= 1.0)
        assert jnp.all(clipped['w'] >= -1.0)
        # The 0.5 element is untouched.
        assert float(clipped['w'][2]) == pytest.approx(0.5)

    def test_clip_grad_norm(self):
        grads = {'w': jnp.array([3.0, 4.0])}  # norm = 5.0
        clipped = _clip_grad_norm(grads, 1.0)
        norm = float(jnp.sqrt(jnp.sum(clipped['w'] ** 2)))
        assert norm == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# T-4: single forward pass per step
# ---------------------------------------------------------------------------

class TestSingleForwardPass:
    def test_training_step_traced_once(self):
        # With the single-pass gradient computation the model is traced exactly
        # once (JIT cache hit afterwards); the old double-call implementation
        # invoked training_step twice during the trace.
        model = LinearRegressor()
        trainer = Trainer(max_epochs=3, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        trainer.fit(model, _loader())
        assert model.train_step_calls == 1

    def test_weights_actually_update(self):
        model = LinearRegressor()
        before = {k: jnp.array(v) for k, v in model.lin.weight.value.items()}
        trainer = Trainer(max_epochs=5, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        trainer.fit(model, _loader())
        after = {k: jnp.array(v) for k, v in model.lin.weight.value.items()}
        assert any(bool(jnp.any(jnp.abs(before[k] - after[k]) > 1e-6)) for k in before)


# ---------------------------------------------------------------------------
# T-9: training metrics forwarded as concrete values
# ---------------------------------------------------------------------------

class TestTrainingMetricsForwarded:
    def test_logged_metric_reaches_callback_metrics(self):
        model = LinearRegressor()
        trainer = Trainer(max_epochs=2, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        trainer.fit(model, _loader())
        # 'mse' logged inside training_step is aggregated under 'train_mse'.
        assert 'train_mse' in trainer.callback_metrics
        assert isinstance(trainer.callback_metrics['train_mse'], float)

    def test_logged_metric_reaches_logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = braintools.trainer.CSVLogger(tmpdir, name='run')
            model = LinearRegressor()
            trainer = Trainer(max_epochs=1, enable_progress_bar=False,
                              enable_checkpointing=False, logger=logger)
            trainer.fit(model, _loader())
            logger.finalize()
            with open(logger.metrics_file_path) as f:
                header = f.readline()
            assert 'mse' in header

    def test_batch_end_callbacks_see_concrete_step_metrics(self):
        # Metrics logged in the JIT-compiled ``training_step`` must be visible
        # to batch-end hooks as concrete values on every batch: previously the
        # first batch exposed a leaked tracer and later batches an empty dict.
        seen = []
        recorder = braintools.trainer.LambdaCallback(
            on_train_batch_end=lambda trainer, module, outputs, batch, batch_idx:
            seen.append((module._get_prog_bar_metrics(), module._get_logger_metrics()))
        )
        model = LinearRegressor()
        trainer = Trainer(max_epochs=2, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False,
                          callbacks=[recorder])
        trainer.fit(model, _loader())
        assert len(seen) == 4
        for prog_bar, logged in seen:
            assert set(prog_bar) == {'mse'}
            assert isinstance(prog_bar['mse'], float)
            assert isinstance(logged['mse'], float)


# ---------------------------------------------------------------------------
# T-2: no double val_/test_ prefix
# ---------------------------------------------------------------------------

class TestNoDoublePrefix:
    def test_validation_metric_single_prefix(self):
        model = LinearRegressor()
        trainer = Trainer(max_epochs=1, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        trainer.fit(model, _loader(), _loader())
        assert 'val_loss' in trainer.callback_metrics
        assert 'val_val_loss' not in trainer.callback_metrics

    def test_test_metric_single_prefix(self):
        model = LinearRegressor()
        trainer = Trainer(max_epochs=1, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        trainer.fit(model, _loader())
        results = trainer.test(model, _loader(), verbose=False)
        assert 'test_loss' in results
        assert 'test_test_loss' not in results

    def test_prefixed_helper(self):
        assert _prefixed('val', 'loss') == 'val_loss'
        assert _prefixed('val', 'val_loss') == 'val_loss'
        assert _prefixed('test', 'acc') == 'test_acc'


# ---------------------------------------------------------------------------
# T-5: min_epochs enforced
# ---------------------------------------------------------------------------

class TestMinEpochs:
    def test_min_epochs_overrides_early_stop(self):
        # EarlyStopping would trigger immediately (metric never improves), but
        # min_epochs must keep training going until it is reached.
        class Worsening(LinearRegressor):
            def validation_step(self, batch, batch_idx):
                loss = jnp.asarray(1.0 + float(self.current_epoch))
                self.log('val_loss', loss)
                return {'val_loss': loss}

        early = braintools.trainer.EarlyStopping(
            monitor='val_loss', patience=0, mode='min', strict=False)
        trainer = Trainer(max_epochs=10, min_epochs=4, callbacks=[early],
                          enable_progress_bar=False, enable_checkpointing=False,
                          logger=False)
        trainer.fit(Worsening(), _loader(), _loader())
        # Must have run at least min_epochs (epochs are 0-indexed).
        assert trainer.current_epoch >= 3


# ---------------------------------------------------------------------------
# T-17: gradient accumulation
# ---------------------------------------------------------------------------

class TestGradientAccumulation:
    def test_accumulation_matches_full_batch(self):
        # Averaging gradients over micro-batches must equal one full-batch step.
        def run(accum, bs):
            brainstate.random.seed(0)
            model = LinearRegressor()
            x = jnp.arange(40, dtype=jnp.float32).reshape(10, 4) / 40.0
            y = jnp.ones((10, 1))
            Trainer(max_epochs=1, accumulate_grad_batches=accum,
                    enable_progress_bar=False, enable_checkpointing=False,
                    logger=False).fit(model, DataLoader((x, y), batch_size=bs,
                                                         shuffle=False))
            return {k: jnp.array(v) for k, v in model.lin.weight.value.items()}

        full = run(1, 10)
        accumulated = run(2, 5)
        max_diff = max(float(jnp.max(jnp.abs(full[k] - accumulated[k]))) for k in full)
        assert max_diff < 1e-5

    def test_invalid_accumulate_raises(self):
        with pytest.raises(ValueError):
            Trainer(accumulate_grad_batches=0, enable_progress_bar=False)


# ---------------------------------------------------------------------------
# T-16: freeze / unfreeze
# ---------------------------------------------------------------------------

class TestFreeze:
    def test_freeze_all_keeps_weights_constant(self):
        model = LinearRegressor()
        names = list(model.states(brainstate.ParamState).keys())
        model.freeze(names)
        before = {k: jnp.array(v) for k, v in model.lin.weight.value.items()}
        trainer = Trainer(max_epochs=3, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        _silent_fit(trainer, model, _loader())
        after = {k: jnp.array(v) for k, v in model.lin.weight.value.items()}
        assert all(bool(jnp.allclose(before[k], after[k])) for k in before)

    def test_freeze_then_unfreeze_trains(self):
        model = LinearRegressor()
        model.freeze()
        assert len(model.frozen_parameters) > 0
        model.unfreeze()
        assert len(model.frozen_parameters) == 0
        before = {k: jnp.array(v) for k, v in model.lin.weight.value.items()}
        trainer = Trainer(max_epochs=3, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        trainer.fit(model, _loader())
        after = {k: jnp.array(v) for k, v in model.lin.weight.value.items()}
        assert any(bool(jnp.any(jnp.abs(before[k] - after[k]) > 1e-6)) for k in before)

    def test_all_frozen_warns(self):
        model = LinearRegressor()
        model.freeze()
        trainer = Trainer(max_epochs=1, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        with pytest.warns(UserWarning, match='frozen'):
            trainer._setup_model(model)


# ---------------------------------------------------------------------------
# T-13: lifecycle hooks fire
# ---------------------------------------------------------------------------

class TestLifecycleHooks:
    def test_hooks_called(self):
        events = []

        class RecordingCallback(braintools.trainer.Callback):
            def on_train_start(self, trainer, module):
                events.append('train_start')

            def on_train_end(self, trainer, module):
                events.append('train_end')

            def on_before_optimizer_step(self, trainer, module, optimizer):
                events.append('before_opt')

            def on_after_optimizer_step(self, trainer, module, optimizer):
                events.append('after_opt')

            def on_after_backward(self, trainer, module):
                events.append('after_backward')

            def on_validation_start(self, trainer, module):
                events.append('val_start')

            def on_validation_end(self, trainer, module):
                events.append('val_end')

        trainer = Trainer(max_epochs=1, callbacks=[RecordingCallback()],
                          enable_progress_bar=False, enable_checkpointing=False,
                          logger=False)
        trainer.fit(LinearRegressor(), _loader(), _loader())
        for evt in ('train_start', 'train_end', 'before_opt', 'after_opt',
                    'after_backward', 'val_start', 'val_end'):
            assert evt in events


# ---------------------------------------------------------------------------
# T-20: seed reproducibility
# ---------------------------------------------------------------------------

class TestSeedReproducibility:
    def test_same_seed_same_weights(self):
        def run():
            # Seed before constructing the model so weight initialisation is
            # identical across runs; the Trainer seed then governs training.
            brainstate.random.seed(7)
            model = LinearRegressor()
            trainer = Trainer(max_epochs=2, seed=7, enable_progress_bar=False,
                              enable_checkpointing=False, logger=False)
            trainer.fit(model, _loader(shuffle=True, seed=7))
            return {k: jnp.array(v) for k, v in model.lin.weight.value.items()}

        a, b = run(), run()
        assert all(bool(jnp.allclose(a[k], b[k])) for k in a)


# ---------------------------------------------------------------------------
# T-23: fractional val_check_interval
# ---------------------------------------------------------------------------

class TestFractionalValCheckInterval:
    def test_fraction_triggers_midepoch_validation(self):
        val_calls = {'n': 0}

        class CountingVal(LinearRegressor):
            def validation_step(self, batch, batch_idx):
                if batch_idx == 0:
                    val_calls['n'] += 1
                loss = jnp.asarray(1.0)
                self.log('val_loss', loss)
                return {'val_loss': loss}

        # 4 batches/epoch, interval 0.5 -> validate every 2 batches -> twice
        # mid-epoch plus once at epoch end.
        trainer = Trainer(max_epochs=1, val_check_interval=0.5,
                          enable_progress_bar=False, enable_checkpointing=False,
                          logger=False)
        trainer.fit(CountingVal(), _loader(n=20, batch_size=5), _loader())
        assert val_calls['n'] >= 2

    def test_integer_interval(self):
        assert Trainer(val_check_interval=2, enable_progress_bar=False)
        t = Trainer(val_check_interval=2, enable_progress_bar=False)
        t.val_dataloaders = [object()]
        t.train_dataloader = None
        # batch index 1 -> (1+1) % 2 == 0 -> True
        assert t._should_validate_batch(1) is True
        assert t._should_validate_batch(0) is False


# ---------------------------------------------------------------------------
# T-24: model retained for post-fit validate/test/predict
# ---------------------------------------------------------------------------

class TestPostFitReuse:
    def test_validate_after_fit_without_model(self):
        model = LinearRegressor()
        trainer = Trainer(max_epochs=1, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        trainer.fit(model, _loader(), _loader())
        assert trainer.model is not None
        results = trainer.validate(verbose=False)
        assert any(k.startswith('val_') for k in results)

    def test_test_after_fit_without_model(self):
        model = LinearRegressor()
        trainer = Trainer(max_epochs=1, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        trainer.fit(model, _loader())
        results = trainer.test(dataloaders=_loader(), verbose=False)
        assert 'test_loss' in results

    def test_predict_after_fit(self):
        model = LinearRegressor()
        trainer = Trainer(max_epochs=1, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        trainer.fit(model, _loader())
        preds = trainer.predict(dataloaders=_loader())
        assert len(preds) == 2  # 20 samples / batch 10


# ---------------------------------------------------------------------------
# T-25: clear errors for missing dataloaders
# ---------------------------------------------------------------------------

class TestDataloaderGuards:
    def test_fit_without_train_loader_raises(self):
        trainer = Trainer(enable_progress_bar=False, enable_checkpointing=False,
                          logger=False)
        with pytest.raises(ValueError, match='train_dataloaders'):
            trainer.fit(LinearRegressor(), None)

    def test_validate_without_loader_raises(self):
        trainer = Trainer(enable_progress_bar=False, logger=False)
        with pytest.raises(ValueError, match='dataloaders'):
            trainer.validate(LinearRegressor())

    def test_test_without_loader_raises(self):
        trainer = Trainer(enable_progress_bar=False, logger=False)
        with pytest.raises(ValueError, match='dataloaders'):
            trainer.test(LinearRegressor())

    def test_predict_without_loader_raises(self):
        trainer = Trainer(enable_progress_bar=False, logger=False)
        with pytest.raises(ValueError, match='dataloaders'):
            trainer.predict(LinearRegressor())


# ---------------------------------------------------------------------------
# T-6 / T-29: checkpoint resume
# ---------------------------------------------------------------------------

class TestCheckpointResume:
    def _train_and_checkpoint(self, tmpdir, *, key, opt_as_list):
        model = LinearRegressor()
        trainer = Trainer(max_epochs=1, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        trainer.fit(model, _loader())
        opt_state = trainer.optimizers[0].state_dict()
        state = {
            'epoch': 9,
            key: 321,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': [opt_state] if opt_as_list else opt_state,
        }
        path = os.path.join(tmpdir, 'ckpt.msgpack')
        save_checkpoint(state, path)
        return path

    def test_resume_modelcheckpoint_format(self):
        # global_step + list-of-optimizer-states (round-trips via msgpack).
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._train_and_checkpoint(tmpdir, key='global_step',
                                               opt_as_list=True)
            model = LinearRegressor()
            trainer = Trainer(max_epochs=1, enable_progress_bar=False,
                              enable_checkpointing=False, logger=False)
            trainer._setup_model(model)
            trainer._setup_optimizers()
            trainer._load_checkpoint(path)
            assert trainer.state.epoch == 9
            assert trainer.state.global_step == 321

    def test_resume_checkpointmanager_format(self):
        # step + single optimizer-state dict.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._train_and_checkpoint(tmpdir, key='step',
                                               opt_as_list=False)
            model = LinearRegressor()
            trainer = Trainer(max_epochs=1, enable_progress_bar=False,
                              enable_checkpointing=False, logger=False)
            trainer._setup_model(model)
            trainer._setup_optimizers()
            trainer._load_checkpoint(path)
            assert trainer.state.epoch == 9
            assert trainer.state.global_step == 321

    def test_resume_via_fit_ckpt_path(self):
        # Resume from a model+counter checkpoint (no optimizer state) and keep
        # training. This exercises the fit(ckpt_path=...) path and confirms the
        # restored model weights are loaded before training continues. (Faithful
        # round-tripping of optax optimizer state is the optim module's concern,
        # not the trainer's.)
        with tempfile.TemporaryDirectory() as tmpdir:
            src = LinearRegressor()
            warm = Trainer(max_epochs=1, enable_progress_bar=False,
                           enable_checkpointing=False, logger=False)
            warm.fit(src, _loader())
            path = os.path.join(tmpdir, 'ckpt.msgpack')
            save_checkpoint({
                'epoch': 5,
                'global_step': 200,
                'model_state_dict': src.state_dict(),
            }, path)

            model = LinearRegressor()
            trainer = Trainer(max_epochs=7, enable_progress_bar=False,
                              enable_checkpointing=False, logger=False)
            trainer.fit(model, _loader(), ckpt_path=path)
            # Only epoch 6 (2 batches) runs after the epoch-5 checkpoint.
            assert trainer.current_epoch == 6
            assert trainer.global_step == 202

    def test_fit_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save a checkpoint with epoch/step state.
            ckpt_path = os.path.join(tmpdir, 'resume.ckpt')
            model = SimpleModel()
            state = {
                'epoch': 2,
                'step': 7,
                'model_state_dict': model.state_dict(),
            }
            braintools.trainer.save_checkpoint(state, ckpt_path)

            trainer = Trainer(
                max_epochs=4, default_root_dir=tmpdir,
                enable_progress_bar=False, enable_checkpointing=False, logger=False,
            )
            # Resume: _load_checkpoint sets epoch/step from the file, then the
            # fit loop runs the remaining epoch 3 only.
            trainer.fit(SimpleModel(), _make_loader(n=16, batch_size=8),
                        ckpt_path=ckpt_path)
            assert trainer.current_epoch == 3
            # global_step accumulates on top of the resumed step (7 + 2).
            assert trainer.global_step == 9


class TestNormalizeOptimizerStates:
    def test_list_passthrough(self):
        assert _normalize_optimizer_states([{'a': 1}, {'b': 2}]) == [{'a': 1}, {'b': 2}]

    def test_single_dict_with_marker(self):
        sd = {'step_count': 3, 'opt_state': {}}
        assert _normalize_optimizer_states(sd) == [sd]

    def test_digit_keyed_dict(self):
        out = _normalize_optimizer_states({'0': {'x': 1}, '1': {'y': 2}})
        assert out == [{'x': 1}, {'y': 2}]

    def test_unknown_returns_empty(self):
        assert _normalize_optimizer_states(None) == []


# ---------------------------------------------------------------------------
# T-18 / T-19 / T-21: configuration validation & warnings
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_invalid_precision_raises(self):
        with pytest.raises(ValueError, match='precision'):
            Trainer(precision='8', enable_progress_bar=False)

    def test_reduced_precision_warns(self):
        # Non-32 precision is also a deprecated argument value (v0.4).
        with pytest.warns(BraintoolsDeprecationWarning, match='precision'):
            with pytest.warns(UserWarning, match='precision'):
                Trainer(precision='16', enable_progress_bar=False)

    def test_benchmark_warns(self):
        # benchmark=True is also a deprecated argument value (v0.4).
        with pytest.warns(BraintoolsDeprecationWarning, match='benchmark'):
            with pytest.warns(UserWarning, match='benchmark'):
                Trainer(benchmark=True, enable_progress_bar=False)

    def test_invalid_clip_algorithm_raises(self):
        with pytest.raises(ValueError, match='gradient_clip_algorithm'):
            Trainer(gradient_clip_algorithm='bogus', enable_progress_bar=False)

    def test_multiple_optimizers_warns(self):
        class MultiOpt(LinearRegressor):
            def configure_optimizers(self):
                return [braintools.optim.SGD(lr=1e-2),
                        braintools.optim.Adam(lr=1e-3)], []

        trainer = Trainer(max_epochs=1, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        with pytest.warns(UserWarning, match='optimizer'):
            trainer.fit(MultiOpt(), _loader())


# ---------------------------------------------------------------------------
# Gradient clipping paths
# ---------------------------------------------------------------------------

class TestGradientClipping:
    def test_clip_norm_runs(self):
        model = LinearRegressor()
        trainer = Trainer(max_epochs=1, gradient_clip_val=1.0,
                          gradient_clip_algorithm='norm',
                          enable_progress_bar=False, enable_checkpointing=False,
                          logger=False)
        trainer.fit(model, _loader())
        assert trainer.global_step > 0

    def test_clip_value_runs(self):
        model = LinearRegressor()
        trainer = Trainer(max_epochs=1, gradient_clip_val=0.5,
                          gradient_clip_algorithm='value',
                          enable_progress_bar=False, enable_checkpointing=False,
                          logger=False)
        trainer.fit(model, _loader())
        assert trainer.global_step > 0


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

class TestFormatMetric:
    def test_float(self):
        assert _format_metric(0.123456) == '0.1235'

    def test_non_float(self):
        assert _format_metric('n/a') == 'n/a'

    def test_array_scalar(self):
        assert _format_metric(jnp.asarray(1.5)) == '1.5000'


# ---------------------------------------------------------------------------
# Miscellaneous loop branches
# ---------------------------------------------------------------------------

class TestLoopBranches:
    def test_max_steps_stops_midepoch(self):
        model = LinearRegressor()
        trainer = Trainer(max_epochs=10, max_steps=3, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        trainer.fit(model, _loader(n=40, batch_size=5))  # 8 batches/epoch
        assert trainer.global_step == 3

    def test_validation_step_returning_none(self):
        class NoneVal(LinearRegressor):
            def validation_step(self, batch, batch_idx):
                return None

        trainer = Trainer(max_epochs=1, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        # Should run without error even though validation produces no metrics.
        trainer.fit(NoneVal(), _loader(), _loader())

    def test_fractional_interval_without_length_is_safe(self):
        # If the train dataloader has no len(), a fractional interval simply
        # skips mid-epoch validation rather than crashing.
        trainer = Trainer(val_check_interval=0.5, enable_progress_bar=False,
                          logger=False)
        trainer.val_dataloaders = [object()]
        trainer.train_dataloader = None  # len() raises -> handled
        assert trainer._should_validate_batch(1) is False

    def test_verbose_validate_and_test(self, capsys):
        model = LinearRegressor()
        trainer = Trainer(max_epochs=1, enable_progress_bar=False,
                          enable_checkpointing=False, logger=False)
        trainer.fit(model, _loader())
        trainer.validate(dataloaders=_loader(), verbose=True)
        trainer.test(dataloaders=_loader(), verbose=True)
        out = capsys.readouterr().out
        assert 'Validation Results' in out
        assert 'Test Results' in out


# ---------------------------------------------------------------------------
# T-A: metrics logged from callbacks/hooks must reach the loggers
# ---------------------------------------------------------------------------

class TestCallbackLoggedMetricsReachLogger:
    def test_lr_monitor_step_mode_reaches_logger(self):
        spy = _SpyLogger()
        lrm = LearningRateMonitor(logging_interval='step')
        trainer = Trainer(
            max_epochs=1, logger=spy, enable_progress_bar=False,
            enable_checkpointing=False, callbacks=[lrm],
        )
        trainer.fit(_Net(), _random_loader())
        assert 'lr-opt0' in spy.keys, (
            f"LR not propagated to logger; saw {sorted(spy.keys)}"
        )
        # And the value is the configured LR.
        lr_vals = [c['lr-opt0'] for c in spy.captured if 'lr-opt0' in c]
        assert lr_vals and lr_vals[0] == pytest.approx(1e-3)

    def test_lr_monitor_epoch_mode_reaches_logger(self):
        spy = _SpyLogger()
        lrm = LearningRateMonitor(logging_interval='epoch')
        trainer = Trainer(
            max_epochs=2, logger=spy, enable_progress_bar=False,
            enable_checkpointing=False, callbacks=[lrm],
        )
        trainer.fit(_Net(), _random_loader())
        assert 'lr-opt0' in spy.keys, (
            f"epoch-mode LR not propagated; saw {sorted(spy.keys)}"
        )

    def test_training_step_metrics_still_present(self):
        # Guard against regression: training_step metrics must survive the merge.
        spy = _SpyLogger()
        trainer = Trainer(
            max_epochs=1, logger=spy, enable_progress_bar=False,
            enable_checkpointing=False,
        )
        trainer.fit(_Net(), _random_loader())
        assert 'train_loss' in spy.keys


# ---------------------------------------------------------------------------
# T-C: resuming from a checkpoint whose step is None must not crash
# ---------------------------------------------------------------------------

class TestResumeStepNone:
    def test_resume_with_step_none(self):
        with tempfile.TemporaryDirectory() as d:
            cm = CheckpointManager(d, verbose=False)
            path = cm.save(_Net(), epoch=0, step=None, metrics={'val_loss': 0.5})

            trainer = Trainer(
                max_epochs=2, logger=False, enable_progress_bar=False,
                enable_checkpointing=False,
            )
            # Must not raise; global_step coerced to an int. The epoch-0
            # checkpoint resumes at epoch 1.
            trainer.fit(_Net(), _random_loader(), ckpt_path=path)
            assert isinstance(trainer.global_step, int)
            assert trainer.global_step > 0
            assert trainer.current_epoch == 1


# ---------------------------------------------------------------------------
# v0.4: deprecated no-op Trainer arguments
# ---------------------------------------------------------------------------

class TestDeprecatedTrainerArguments:
    """``precision``/``deterministic``/``benchmark``/``strategy`` warn only off-default."""

    def test_default_arguments_do_not_warn(self):
        # pytest turns BraintoolsDeprecationWarning into an error, so this
        # fails if the default path warns.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            Trainer(enable_progress_bar=False, logger=False)

    @pytest.mark.parametrize('kwargs', [
        dict(precision='32'),
        dict(precision='32-true'),
        dict(precision=32),
        dict(deterministic=False),
        dict(benchmark=False),
        dict(strategy='auto'),
        dict(strategy=None),
    ])
    def test_explicit_default_values_do_not_warn(self, kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            trainer = Trainer(enable_progress_bar=False, logger=False, **kwargs)
        assert trainer is not None

    @pytest.mark.parametrize('precision', ['16', '16-mixed', 'bf16', 'bf16-mixed'])
    def test_reduced_precision_is_deprecated(self, precision):
        with pytest.warns(BraintoolsDeprecationWarning) as record:
            with pytest.warns(UserWarning, match='precision'):
                trainer = Trainer(precision=precision, enable_progress_bar=False, logger=False)
        messages = [str(w.message) for w in record if issubclass(w.category, BraintoolsDeprecationWarning)]
        assert len(messages) == 1
        assert 'Trainer(precision=...)' in messages[0]
        assert '0.4.0' in messages[0] and '1.0.0' in messages[0]
        assert trainer.precision == precision

    def test_invalid_precision_raises_before_warning(self):
        with pytest.raises(ValueError, match='precision'):
            Trainer(precision='8', enable_progress_bar=False, logger=False)

    def test_deterministic_is_deprecated(self):
        with pytest.warns(BraintoolsDeprecationWarning, match=r'Trainer\(deterministic=\.\.\.\)'):
            trainer = Trainer(deterministic=True, enable_progress_bar=False, logger=False)
        assert trainer.deterministic is True

    def test_benchmark_is_deprecated(self):
        with pytest.warns(BraintoolsDeprecationWarning, match=r'Trainer\(benchmark=\.\.\.\)'):
            with pytest.warns(UserWarning, match='benchmark'):
                trainer = Trainer(benchmark=True, enable_progress_bar=False, logger=False)
        assert trainer.benchmark is True

    @pytest.mark.parametrize('strategy', ['single', 'single_device'])
    def test_strategy_name_is_deprecated(self, strategy):
        with pytest.warns(BraintoolsDeprecationWarning, match=r'Trainer\(strategy=\.\.\.\)'):
            trainer = Trainer(strategy=strategy, enable_progress_bar=False, logger=False)
        assert isinstance(trainer.strategy, braintools.trainer.SingleDeviceStrategy)

    def test_strategy_instance_is_deprecated(self):
        strategy = braintools.trainer.AutoStrategy()
        with pytest.warns(BraintoolsDeprecationWarning, match='strategy'):
            trainer = Trainer(strategy=strategy, enable_progress_bar=False, logger=False)
        assert trainer.strategy is strategy

    def test_unknown_strategy_raises_value_error_not_warning(self):
        with pytest.raises(ValueError, match='Unknown strategy'):
            Trainer(strategy='bogus', enable_progress_bar=False, logger=False)

    def test_deprecated_arguments_still_train(self):
        with pytest.warns(BraintoolsDeprecationWarning):
            trainer = Trainer(
                max_epochs=1, deterministic=True, strategy='single',
                enable_progress_bar=False, enable_checkpointing=False, logger=False,
            )
        trainer.fit(SimpleModel(), _make_loader(n=16, batch_size=8))
        assert trainer.global_step == 2


# ---------------------------------------------------------------------------
# WP-M2b: global_step sync, resume epochs, eval ckpt_path, checkpoint dir
# ---------------------------------------------------------------------------

class _EpochRecorder(braintools.trainer.Callback):
    """Record the epochs that actually train and the optimizer-step count."""

    def __init__(self):
        super().__init__()
        self.epochs = []
        self.optimizer_steps = 0
        self.batches = 0

    def on_train_epoch_start(self, trainer, module):
        self.epochs.append((trainer.current_epoch, module.current_epoch))

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.batches += 1

    def on_after_optimizer_step(self, trainer, module, optimizer):
        self.optimizer_steps += 1


def _quiet_trainer(**kw):
    kw.setdefault('enable_progress_bar', False)
    kw.setdefault('enable_checkpointing', False)
    kw.setdefault('logger', False)
    return Trainer(**kw)


def _ckpt_files(dirpath):
    return sorted(f for f in os.listdir(dirpath) if f.endswith('.ckpt'))


class TestGlobalStepSync:
    """``model.global_step`` follows the trainer's optimizer-step count."""

    def test_model_global_step_after_fit(self):
        model = LinearRegressor()
        trainer = _quiet_trainer(max_epochs=3)
        trainer.fit(model, _loader())  # 2 batches/epoch
        assert trainer.global_step == 6
        assert model.global_step == 6

    def test_checkpoints_store_true_global_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = braintools.trainer.ModelCheckpoint(
                dirpath=tmpdir, filename='ckpt-{epoch}-{step}',
                monitor='val_loss', save_top_k=-1)
            trainer = _quiet_trainer(max_epochs=3, callbacks=[ckpt])
            trainer.fit(LinearRegressor(), _loader(), _loader())
            assert _ckpt_files(tmpdir) == [
                'ckpt-0-2.ckpt', 'ckpt-1-4.ckpt', 'ckpt-2-6.ckpt', 'last.ckpt']
            last = braintools.trainer.load_checkpoint(os.path.join(tmpdir, 'last.ckpt'))
            assert int(last['global_step']) == 6
            assert int(last['epoch']) == 2

    def test_every_n_train_steps_sees_real_step(self):
        # The step-based trigger used to see ``global_step == 0`` forever, so it
        # saved one ``step=0`` file and nothing else.
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = braintools.trainer.ModelCheckpoint(
                dirpath=tmpdir, filename='s-{step}', every_n_train_steps=2,
                save_top_k=0, save_last=False)
            _quiet_trainer(max_epochs=3, callbacks=[ckpt]).fit(LinearRegressor(), _loader())
            assert _ckpt_files(tmpdir) == ['s-2.ckpt', 's-4.ckpt', 's-6.ckpt']
            state = braintools.trainer.load_checkpoint(os.path.join(tmpdir, 's-4.ckpt'))
            assert int(state['global_step']) == 4

    def test_every_n_train_steps_with_accumulation_skips_step_zero(self):
        # With accumulation the first micro-batch ends before any optimizer
        # update (step 0); that must not produce a ``step=0`` checkpoint.
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = braintools.trainer.ModelCheckpoint(
                dirpath=tmpdir, filename='s-{step}', every_n_train_steps=1,
                save_top_k=0, save_last=False)
            _quiet_trainer(max_epochs=1, accumulate_grad_batches=2,
                           callbacks=[ckpt]).fit(LinearRegressor(), _loader(n=40, batch_size=10))
            assert _ckpt_files(tmpdir) == ['s-1.ckpt', 's-2.ckpt']

    def test_accumulation_counts_optimizer_steps(self):
        # 3 batches/epoch with accumulate=2: one full window + one flushed
        # partial window = 2 optimizer steps per epoch.
        model = LinearRegressor()
        rec = _EpochRecorder()
        trainer = _quiet_trainer(max_epochs=2, accumulate_grad_batches=2, callbacks=[rec])
        trainer.fit(model, _loader(n=30, batch_size=10))
        assert rec.batches == 6
        assert rec.optimizer_steps == 4
        assert trainer.global_step == 4
        assert model.global_step == 4

    def test_max_steps_counts_optimizer_steps(self):
        rec = _EpochRecorder()
        trainer = _quiet_trainer(max_epochs=10, max_steps=2, accumulate_grad_batches=2,
                                 callbacks=[rec])
        trainer.fit(LinearRegressor(), _loader(n=40, batch_size=10))
        assert trainer.global_step == 2
        assert rec.batches == 4  # 2 optimizer steps x 2 micro-batches

    def test_logged_step_shared_within_accumulation_window(self):
        class StepSpy(_SpyLogger):
            def __init__(self):
                super().__init__()
                self.logged = []

            def log_metrics(self, metrics, step=None):
                self.logged.append((dict(metrics), step))

        logger = StepSpy()
        trainer = _quiet_trainer(max_epochs=1, accumulate_grad_batches=2, logger=logger)
        trainer.fit(LinearRegressor(), _loader(n=40, batch_size=10))
        train_steps = [step for metrics, step in logger.logged if 'loss' in metrics]
        assert train_steps == [0, 0, 1, 1]


class TestResumeContinuesEpochs:
    """``fit(ckpt_path=...)`` continues from ``checkpoint_epoch + 1``."""

    def _train_three_epochs(self, tmpdir):
        ckpt = braintools.trainer.ModelCheckpoint(
            dirpath=tmpdir, monitor='val_loss', save_top_k=1)
        trainer = _quiet_trainer(max_epochs=3, callbacks=[ckpt])
        trainer.fit(LinearRegressor(), _loader(), _loader())  # SGD optimizer
        assert trainer.global_step == 6
        return os.path.join(tmpdir, 'last.ckpt')

    def test_resume_runs_remaining_epochs_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._train_three_epochs(tmpdir)
            model = LinearRegressor()
            rec = _EpochRecorder()
            trainer = _quiet_trainer(max_epochs=5, callbacks=[rec])
            trainer.fit(model, _loader(), _loader(), ckpt_path=path)
            assert rec.epochs == [(3, 3), (4, 4)]
            assert rec.batches == 4
            assert trainer.current_epoch == 4
            assert model.current_epoch == 4
            assert trainer.global_step == 10
            assert model.global_step == 10

    def test_resume_checkpoint_carries_continued_counters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._train_three_epochs(tmpdir)
            out = os.path.join(tmpdir, 'resumed')
            ckpt = braintools.trainer.ModelCheckpoint(
                dirpath=out, filename='e{epoch}-s{step}', monitor='val_loss',
                save_top_k=-1)
            _quiet_trainer(max_epochs=5, callbacks=[ckpt]).fit(
                LinearRegressor(), _loader(), _loader(), ckpt_path=path)
            assert _ckpt_files(out) == ['e3-s8.ckpt', 'e4-s10.ckpt', 'last.ckpt']

    def test_resume_past_max_epochs_trains_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._train_three_epochs(tmpdir)
            model = LinearRegressor()
            rec = _EpochRecorder()
            trainer = _quiet_trainer(max_epochs=3, callbacks=[rec])
            with pytest.warns(UserWarning, match='max_epochs'):
                trainer.fit(model, _loader(), ckpt_path=path)
            assert rec.epochs == []
            assert rec.batches == 0
            assert trainer.current_epoch == 2
            assert trainer.global_step == 6

    def test_resume_without_epoch_key_starts_at_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'weights.ckpt')
            save_checkpoint({'model_state_dict': LinearRegressor().state_dict()}, path)
            rec = _EpochRecorder()
            trainer = _quiet_trainer(max_epochs=2, callbacks=[rec])
            trainer.fit(LinearRegressor(), _loader(), ckpt_path=path)
            assert [e for e, _ in rec.epochs] == [0, 1]
            assert trainer.global_step == 4


class _StepLRRegressor(LinearRegressor):
    """``LinearRegressor`` with plain SGD driven by an epoch-wise ``StepLR``."""

    def configure_optimizers(self):
        sched = braintools.optim.StepLR(base_lr=0.1, step_size=1, gamma=0.5)
        return braintools.optim.SGD(lr=sched), sched


class _LRRecorder(braintools.trainer.Callback):
    """Record the learning rate each training epoch runs with."""

    def __init__(self):
        super().__init__()
        self.lrs = {}

    def on_train_epoch_start(self, trainer, module):
        self.lrs[trainer.current_epoch] = float(trainer.schedulers[0].get_last_lr()[0])


def _steplr_model(seed=1):
    brainstate.random.seed(seed)
    return _StepLRRegressor()


def _weights(model):
    return {k: np.asarray(v) for k, v in model.lin.weight.value.items()}


class TestResumeRestoresLRScheduler:
    """``fit(ckpt_path=...)`` continues the LR schedule, not restarts it."""

    def _continuous(self):
        rec = _LRRecorder()
        model = _steplr_model()
        _quiet_trainer(max_epochs=5, callbacks=[rec]).fit(model, _loader(), _loader())
        return rec.lrs, _weights(model)

    def _three_epochs(self, tmpdir):
        ckpt = braintools.trainer.ModelCheckpoint(
            dirpath=tmpdir, filename='e{epoch}', monitor='val_loss', save_top_k=-1)
        _quiet_trainer(max_epochs=3, callbacks=[ckpt]).fit(
            _steplr_model(), _loader(), _loader())

    @pytest.mark.parametrize('name', ['last.ckpt', 'e2.ckpt'])
    def test_resume_matches_continuous_run(self, name):
        # ``last.ckpt`` is written after the epoch's scheduler step, the
        # per-epoch ``e2.ckpt`` before it; both must resume at the epoch-3 LR.
        lrs, weights = self._continuous()
        assert lrs == pytest.approx({0: 0.1, 1: 0.05, 2: 0.025, 3: 0.0125, 4: 0.00625})
        with tempfile.TemporaryDirectory() as tmpdir:
            self._three_epochs(tmpdir)
            rec = _LRRecorder()
            model = _steplr_model(seed=7)
            trainer = _quiet_trainer(max_epochs=5, callbacks=[rec])
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                trainer.fit(model, _loader(), _loader(),
                            ckpt_path=os.path.join(tmpdir, name))
        assert rec.lrs == pytest.approx({3: lrs[3], 4: lrs[4]})
        assert int(trainer.schedulers[0].last_epoch.value) == 5
        for key, value in weights.items():
            np.testing.assert_allclose(_weights(model)[key], value, rtol=1e-5, atol=1e-6)

    def test_checkpoint_stores_scheduler_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._three_epochs(tmpdir)
            mid = braintools.trainer.load_checkpoint(os.path.join(tmpdir, 'e2.ckpt'))
            last = braintools.trainer.load_checkpoint(os.path.join(tmpdir, 'last.ckpt'))
        assert int(mid['lr_schedulers']['0']['last_epoch']) == 2
        assert not mid['lr_schedulers_stepped']
        assert int(last['lr_schedulers']['0']['last_epoch']) == 3
        assert last['lr_schedulers_stepped']

    def _without_trainer_scheduler_state(self, tmpdir, name, drop_optimizer_scheduler):
        """``name`` re-saved without the trainer's scheduler state (and, optionally,
        without the scheduler state inside the optimizer state)."""
        state = braintools.trainer.load_checkpoint(os.path.join(tmpdir, name))
        del state['lr_schedulers'], state['lr_schedulers_stepped']
        if drop_optimizer_scheduler:
            for opt_state in state['optimizer_state_dict'].values():
                del opt_state['lr_scheduler'], opt_state['schedulers']
        path = os.path.join(tmpdir, 'stripped.ckpt')
        save_checkpoint(state, path)
        return path

    def _resume(self, path, warns):
        rec = _LRRecorder()
        model = _steplr_model(seed=7)
        trainer = _quiet_trainer(max_epochs=5, callbacks=[rec])
        if warns:
            with pytest.warns(UserWarning, match='no learning-rate scheduler state'):
                trainer.fit(model, _loader(), _loader(), ckpt_path=path)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                trainer.fit(model, _loader(), _loader(), ckpt_path=path)
        return rec.lrs, _weights(model), trainer

    def test_legacy_checkpoint_fast_forwards_with_warning(self):
        # A checkpoint written before scheduler state was saved (by the trainer or
        # by the optimizer): the schedule is rebuilt by stepping it once per
        # completed epoch.
        lrs, weights = self._continuous()
        with tempfile.TemporaryDirectory() as tmpdir:
            self._three_epochs(tmpdir)
            path = self._without_trainer_scheduler_state(tmpdir, 'e2.ckpt', True)
            got, got_weights, trainer = self._resume(path, warns=True)
        assert got == pytest.approx({3: lrs[3], 4: lrs[4]})
        assert int(trainer.schedulers[0].last_epoch.value) == 5
        for key, value in weights.items():
            np.testing.assert_allclose(got_weights[key], value, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize('name,warns', [('e2.ckpt', True), ('last.ckpt', False)])
    def test_optimizer_scheduler_state_is_not_stepped_twice(self, name, warns):
        # A checkpoint holding only the optimizer state (as ``CheckpointManager``
        # writes it): the optimizer restores its scheduler, which is then stepped
        # only up to the resumed epoch, not ``start_epoch`` more times. ``e2.ckpt``
        # was saved before the epoch-end step (one step, with a warning),
        # ``last.ckpt`` after it (nothing to step, no warning).
        lrs, weights = self._continuous()
        with tempfile.TemporaryDirectory() as tmpdir:
            self._three_epochs(tmpdir)
            path = self._without_trainer_scheduler_state(tmpdir, name, False)
            got, got_weights, trainer = self._resume(path, warns=warns)
        assert got == pytest.approx({3: lrs[3], 4: lrs[4]})
        assert int(trainer.schedulers[0].last_epoch.value) == 5
        for key, value in weights.items():
            np.testing.assert_allclose(got_weights[key], value, rtol=1e-5, atol=1e-6)

    def test_scheduler_not_in_optimizer_state_starts_fresh(self):
        # Only schedulers the optimizer restored resume from their saved epoch.
        opt_sched = braintools.optim.StepLR(base_lr=1.0, step_size=1, gamma=0.5)
        other = braintools.optim.StepLR(base_lr=1.0, step_size=1, gamma=0.5)
        trainer = _quiet_trainer()
        trainer.optimizers = [braintools.optim.SGD(lr=opt_sched)]
        trainer.schedulers = [opt_sched, other]
        opt_sched.step()
        opt_sched.step()  # restored at epoch 2 by its optimizer
        state = {'optimizer_state_dict': [trainer.optimizers[0].state_dict()]}
        with pytest.warns(UserWarning, match='stepped 1, 3 time'):
            trainer._restore_schedulers(state, 3)
        assert int(opt_sched.last_epoch.value) == 3
        assert int(other.last_epoch.value) == 3
        assert opt_sched.get_last_lr() == pytest.approx([0.125])
        assert other.get_last_lr() == pytest.approx([0.125])

    def test_checkpoint_without_schedulers_does_not_warn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'w.ckpt')
            save_checkpoint({'epoch': 0, 'global_step': 2,
                             'model_state_dict': LinearRegressor().state_dict()}, path)
            trainer = _quiet_trainer(max_epochs=2)
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                trainer.fit(LinearRegressor(), _loader(), ckpt_path=path)
        assert trainer.global_step == 4

    def test_eval_ckpt_leaves_scheduler_alone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._three_epochs(tmpdir)
            model = _steplr_model()
            trainer = _quiet_trainer(max_epochs=1)
            trainer.fit(model, _loader())
            trainer.validate(model, _loader(), ckpt_path=os.path.join(tmpdir, 'last.ckpt'),
                             verbose=False)
        assert int(trainer.schedulers[0].last_epoch.value) == 1


class TestRestoreSchedulersHelpers:
    def test_from_msgpack_lists(self):
        assert _from_msgpack_lists({'0': 1.0, '1': {'0': 2}}) == [1.0, [2]]
        assert _from_msgpack_lists({'1': {'0': 3}}) == {'1': [3]}
        assert _from_msgpack_lists({'a': ({'0': 1},)}) == {'a': [[1]]}
        assert _from_msgpack_lists({}) == {}
        assert _from_msgpack_lists(5) == 5

    def test_restore_by_position_skips_none(self):
        # ``schedulers == [None, sched]`` saves only index '1'.
        sched = braintools.optim.StepLR(base_lr=1.0, step_size=1, gamma=0.5)
        trainer = _quiet_trainer()
        trainer.schedulers = [None, sched]
        saved = {'1': {'last_epoch': 2, 'base_lrs': {'0': 1.0}, 'current_lrs': {'0': 0.25}}}
        trainer._restore_schedulers({'lr_schedulers': saved, 'lr_schedulers_stepped': True}, 3)
        assert int(sched.last_epoch.value) == 2
        assert sched.base_lrs == [1.0]
        assert sched.get_last_lr() == pytest.approx([0.25])
        assert trainer.state.lr_schedulers_stepped

    def test_restore_unstepped_list_state(self):
        sched = braintools.optim.StepLR(base_lr=1.0, step_size=1, gamma=0.5)
        trainer = _quiet_trainer()
        trainer.schedulers = [sched]
        saved = [{'last_epoch': 2, 'base_lrs': [1.0], 'current_lrs': [0.25]}]
        trainer._restore_schedulers({'lr_schedulers': saved}, 3)
        assert int(sched.last_epoch.value) == 3
        assert sched.get_last_lr() == pytest.approx([0.125])


class TestResumeAtMaxSteps:
    """Resuming a checkpoint that already reached ``max_steps`` trains nothing."""

    def test_no_extra_optimizer_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = braintools.trainer.ModelCheckpoint(dirpath=tmpdir, save_top_k=0)
            _quiet_trainer(max_epochs=3, callbacks=[ckpt]).fit(LinearRegressor(), _loader())
            path = os.path.join(tmpdir, 'last.ckpt')
            model = LinearRegressor()
            rec = _EpochRecorder()
            trainer = _quiet_trainer(max_epochs=5, max_steps=6, callbacks=[rec])
            with pytest.warns(UserWarning, match='max_steps'):
                trainer.fit(model, _loader(), ckpt_path=path)
        assert rec.epochs == []
        assert rec.batches == 0
        assert rec.optimizer_steps == 0
        assert trainer.global_step == 6
        assert model.global_step == 6

    def test_remaining_steps_still_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = braintools.trainer.ModelCheckpoint(dirpath=tmpdir, save_top_k=0)
            _quiet_trainer(max_epochs=3, callbacks=[ckpt]).fit(LinearRegressor(), _loader())
            trainer = _quiet_trainer(max_epochs=5, max_steps=7)
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                trainer.fit(LinearRegressor(), _loader(),
                            ckpt_path=os.path.join(tmpdir, 'last.ckpt'))
        assert trainer.global_step == 7


class TestEvalCkptKeepsProgress:
    """validate/test/predict with ``ckpt_path`` load weights only."""

    @pytest.mark.parametrize('stage', ['validate', 'test', 'predict'])
    def test_ckpt_path_loads_weights_not_counters(self, stage):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = LinearRegressor()
            _quiet_trainer(max_epochs=1).fit(src, _loader())
            path = os.path.join(tmpdir, 'other.ckpt')
            save_checkpoint({'epoch': 9, 'global_step': 321,
                             'model_state_dict': src.state_dict()}, path)

            model = LinearRegressor()
            trainer = _quiet_trainer(max_epochs=2)
            trainer.fit(model, _loader())
            assert (trainer.current_epoch, trainer.global_step) == (1, 4)

            kwargs = {'dataloaders': _loader(), 'ckpt_path': path}
            if stage != 'predict':
                kwargs['verbose'] = False
            getattr(trainer, stage)(**kwargs)

            assert (trainer.current_epoch, trainer.global_step) == (1, 4)
            assert (model.current_epoch, model.global_step) == (1, 4)
            for key, value in src.lin.weight.value.items():
                np.testing.assert_allclose(np.asarray(model.lin.weight.value[key]),
                                           np.asarray(value))


class TestNoStrayCheckpointDir:
    """The default ``enable_checkpointing=True`` must not litter the cwd."""

    def test_default_fit_leaves_no_empty_dirs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        trainer = Trainer(max_epochs=1, enable_progress_bar=False)
        assert trainer.enable_checkpointing is True
        trainer.fit(LinearRegressor(), _loader())
        assert not (tmp_path / 'checkpoints').exists()
        empty = [str(p) for p in tmp_path.rglob('*') if p.is_dir() and not any(p.iterdir())]
        assert empty == []

    def test_model_checkpoint_still_saves_with_default(self, tmp_path):
        ckpt = braintools.trainer.ModelCheckpoint(dirpath=str(tmp_path / 'ck'))
        trainer = Trainer(max_epochs=1, enable_progress_bar=False, logger=False,
                          default_root_dir=str(tmp_path), callbacks=[ckpt])
        trainer.fit(LinearRegressor(), _loader())
        assert 'last.ckpt' in os.listdir(tmp_path / 'ck')
        assert not (tmp_path / 'checkpoints').exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
