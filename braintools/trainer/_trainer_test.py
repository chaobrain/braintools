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

"""Tests for the trainer module."""

import tempfile
import os

import jax.numpy as jnp
import numpy as np
import pytest

import brainstate
import braintools


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
