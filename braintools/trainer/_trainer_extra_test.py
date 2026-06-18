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

"""Complementary tests for ``braintools.trainer._trainer`` (gap coverage).

These tests are deliberately separate from ``_trainer_test.py`` and exercise
the validation / test / predict loops, callbacks integration, gradient
accumulation knobs, max_steps vs max_epochs, gradient clipping, scheduler
stepping, logging, checkpoint resume, and ``TrainerState`` transitions.
"""

import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

import brainstate
import braintools
from braintools.trainer._module import EvalOutput
from braintools.trainer._trainer import (
    Trainer,
    TrainerState,
    _clip_grad_norm,
    _clip_grad_value,
)


class SimpleModel(braintools.trainer.LightningModule):
    """Simple model mirroring the one in ``_trainer_test.py``."""

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

    def test_enable_checkpointing_creates_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            trainer = Trainer(
                max_epochs=1, default_root_dir=tmpdir,
                enable_progress_bar=False, enable_checkpointing=True, logger=False,
            )
            trainer.fit(model, _make_loader(n=16, batch_size=8))
            assert os.path.isdir(os.path.join(tmpdir, 'checkpoints'))


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
# Checkpoint resume
# ---------------------------------------------------------------------------

class TestCheckpointResume:
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
                max_epochs=1, default_root_dir=tmpdir,
                enable_progress_bar=False, enable_checkpointing=False, logger=False,
            )
            # Resume: _load_checkpoint sets epoch/step from the file, then the
            # fit loop runs (max_epochs=1) and overwrites epoch with 0.
            trainer.fit(SimpleModel(), _make_loader(n=16, batch_size=8),
                        ckpt_path=ckpt_path)
            # global_step accumulates on top of the resumed step (7 + 2).
            assert trainer.global_step == 9


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
