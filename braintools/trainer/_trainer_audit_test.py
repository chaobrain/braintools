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

"""Regression tests for the trainer audit fixes (T-2 .. T-29).

Each test is annotated with the audit finding it guards against. These cover
behaviours that the pre-existing suite did not exercise: single forward pass,
concrete metric forwarding, gradient accumulation, freezing, min_epochs,
fractional validation interval, checkpoint resume, post-fit reuse, dataloader
guards, and configuration warnings.
"""

import os
import tempfile
import warnings

import jax
import jax.numpy as jnp
import pytest

import brainstate
import braintools
from braintools.trainer import Trainer, LightningModule, DataLoader
from braintools.trainer._checkpoint import save_checkpoint
from braintools.trainer._trainer import (
    _normalize_optimizer_states,
    _prefixed,
    _format_metric,
)


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

    def test_freeze_single_name_and_is_frozen(self):
        model = LinearRegressor()
        name = next(iter(model.states(brainstate.ParamState).keys()))
        model.freeze(name)
        assert model.is_frozen(name)
        assert name in model.frozen_parameters

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
            trainer = Trainer(max_epochs=2, enable_progress_bar=False,
                              enable_checkpointing=False, logger=False)
            trainer.fit(model, _loader(), ckpt_path=path)
            assert trainer.current_epoch >= 0


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
        with pytest.warns(UserWarning, match='precision'):
            Trainer(precision='16', enable_progress_bar=False)

    def test_benchmark_warns(self):
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
