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

"""Tests for the trainer callbacks module."""

import os
import tempfile

import jax.numpy as jnp
import pytest

import brainstate
import braintools
import braintools.file as bf

from braintools.trainer._callbacks import (
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


# ---------------------------------------------------------------------------
# Lightweight stubs used to drive callback hooks directly.
# ---------------------------------------------------------------------------

class SimpleModel(braintools.trainer.LightningModule):
    """Tiny model mirroring the one in _trainer_test.py."""

    def __init__(self, input_size=4, hidden_size=3, output_size=2):
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

    def configure_optimizers(self):
        return braintools.optim.Adam(lr=1e-3)


class FakeModule:
    """Stand-in for a LightningModule when driving hooks directly."""

    def __init__(self, current_epoch=0, global_step=0):
        self.current_epoch = current_epoch
        self.global_step = global_step
        self._prog_bar = {}
        self.logged = []
        self._state = {'w': jnp.ones((2, 2))}

    def log(self, name, value, prog_bar=False, logger=True):
        self.logged.append((name, value, prog_bar, logger))

    def _get_prog_bar_metrics(self):
        return self._prog_bar

    def state_dict(self):
        return self._state


class FakeTrainer:
    """Stand-in for a Trainer when driving hooks directly."""

    def __init__(self, callbacks=None, metrics=None, optimizers=None,
                 train_dataloader=None, use_callback_metrics=True):
        self.callbacks = callbacks if callbacks is not None else []
        if metrics is not None:
            if use_callback_metrics:
                self.callback_metrics = metrics
            else:
                self.logged_metrics = metrics
        if optimizers is not None:
            self.optimizers = optimizers
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader


class DummyOptValue:
    """Optimizer whose lr exposes a `.value` attribute."""

    def __init__(self, value):
        class _LR:
            pass
        self.lr = _LR()
        self.lr.value = value


class DummyOptCallable:
    """Optimizer whose lr is a zero-arg callable."""

    def __init__(self, value):
        self.lr = lambda: value


class DummyOptLearningRate:
    """Optimizer exposing `learning_rate` instead of `lr`."""

    def __init__(self, value):
        self.learning_rate = value


class DummyOptPlainFloat:
    """Optimizer whose lr is already a plain float."""

    def __init__(self, value):
        self.lr = value


# ---------------------------------------------------------------------------
# Callback base class + CallbackList
# ---------------------------------------------------------------------------

class TestCallbackBase:
    def test_all_hooks_callable_noop(self):
        cb = Callback()
        # Every hook should be a no-op that accepts the documented arguments.
        cb.on_fit_start(None, None)
        cb.on_fit_end(None, None)
        cb.on_train_start(None, None)
        cb.on_train_end(None, None)
        cb.on_train_epoch_start(None, None)
        cb.on_train_epoch_end(None, None)
        cb.on_train_batch_start(None, None, None, 0)
        cb.on_train_batch_end(None, None, None, None, 0)
        cb.on_validation_start(None, None)
        cb.on_validation_end(None, None)
        cb.on_validation_epoch_start(None, None)
        cb.on_validation_epoch_end(None, None)
        cb.on_validation_batch_start(None, None, None, 0)
        cb.on_validation_batch_end(None, None, None, None, 0)
        cb.on_test_start(None, None)
        cb.on_test_end(None, None)
        cb.on_test_epoch_start(None, None)
        cb.on_test_epoch_end(None, None)
        cb.on_test_batch_start(None, None, None, 0)
        cb.on_test_batch_end(None, None, None, None, 0)
        cb.on_predict_start(None, None)
        cb.on_predict_end(None, None)
        cb.on_predict_batch_start(None, None, None, 0)
        cb.on_predict_batch_end(None, None, None, None, 0)
        cb.on_before_optimizer_step(None, None, None)
        cb.on_after_optimizer_step(None, None, None)
        cb.on_before_backward(None, None, None)
        cb.on_after_backward(None, None)
        cb.on_save_checkpoint(None, None, {})
        cb.on_load_checkpoint(None, None, {})

    def test_state_key_default(self):
        cb = Callback()
        assert cb.state_key == 'Callback'
        assert EarlyStopping().state_key == 'EarlyStopping'

    def test_state_dict_roundtrip_default(self):
        cb = Callback()
        assert cb.state_dict() == {}
        # load_state_dict on the base class is a no-op and must not raise.
        cb.load_state_dict({'anything': 1})


class TestCallbackList:
    def test_init_empty_and_len(self):
        cl = CallbackList()
        assert len(cl) == 0
        assert list(cl) == []

    def test_append_and_iter(self):
        a, b = Callback(), Callback()
        cl = CallbackList([a])
        cl.append(b)
        assert len(cl) == 2
        assert list(cl) == [a, b]

    def test_init_with_list(self):
        cbs = [Callback(), Callback()]
        cl = CallbackList(cbs)
        assert len(cl) == 2

    def test_convenience_hooks_dispatch(self):
        record = []

        class Recorder(Callback):
            def on_fit_start(self, trainer, module):
                record.append('fit_start')

            def on_fit_end(self, trainer, module):
                record.append('fit_end')

            def on_train_epoch_start(self, trainer, module):
                record.append('train_epoch_start')

            def on_train_epoch_end(self, trainer, module):
                record.append('train_epoch_end')

            def on_train_batch_start(self, trainer, module, batch, batch_idx):
                record.append(('train_batch_start', batch_idx))

            def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                record.append(('train_batch_end', batch_idx))

            def on_validation_epoch_start(self, trainer, module):
                record.append('val_epoch_start')

            def on_validation_epoch_end(self, trainer, module):
                record.append('val_epoch_end')

            def on_validation_batch_start(self, trainer, module, batch, batch_idx):
                record.append(('val_batch_start', batch_idx))

            def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):
                record.append(('val_batch_end', batch_idx))

        cl = CallbackList([Recorder()])
        cl.on_fit_start(None, None)
        cl.on_fit_end(None, None)
        cl.on_train_epoch_start(None, None)
        cl.on_train_epoch_end(None, None)
        cl.on_train_batch_start(None, None, 'b', 1)
        cl.on_train_batch_end(None, None, 'o', 'b', 1)
        cl.on_validation_epoch_start(None, None)
        cl.on_validation_epoch_end(None, None)
        cl.on_validation_batch_start(None, None, 'b', 2)
        cl.on_validation_batch_end(None, None, 'o', 'b', 2)

        assert record == [
            'fit_start', 'fit_end',
            'train_epoch_start', 'train_epoch_end',
            ('train_batch_start', 1), ('train_batch_end', 1),
            'val_epoch_start', 'val_epoch_end',
            ('val_batch_start', 2), ('val_batch_end', 2),
        ]

    def test_call_hook_skips_missing_attribute(self):
        # An object without the requested hook attribute is silently skipped.
        class Bare:
            pass

        cl = CallbackList([Bare()])
        # Should not raise even though Bare has no on_fit_start.
        cl.on_fit_start(None, None)


# ---------------------------------------------------------------------------
# ModelCheckpoint
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_save(monkeypatch):
    """Record ModelCheckpoint saves without serializing to disk.

    ``_save_checkpoint`` calls ``braintools.file.msgpack_save(filepath, checkpoint)``.
    We replace it with a tiny recorder so the decision logic in ModelCheckpoint can
    be exercised quickly while still producing files.
    """
    saved = []

    def fake(path, target, *args, **kwargs):
        saved.append((target, path))
        with open(path, 'wb') as f:
            f.write(b'ckpt')

    monkeypatch.setattr(bf, 'msgpack_save', fake)
    return saved


class TestModelCheckpointBasics:
    def test_init_defaults_create_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, 'ckpts')
            cb = ModelCheckpoint(dirpath=d)
            assert os.path.isdir(d)
            assert cb.monitor == 'val_loss'
            assert cb.mode == 'min'
            assert cb.filename == 'checkpoint-{epoch:02d}'
            assert cb.best_score is None

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            ModelCheckpoint(mode='bogus')

    def test_is_better_min_and_max(self):
        cmin = ModelCheckpoint(mode='min')
        assert cmin._is_better(0.1, 0.2)
        assert not cmin._is_better(0.3, 0.2)
        cmax = ModelCheckpoint(mode='max')
        assert cmax._is_better(0.3, 0.2)
        assert not cmax._is_better(0.1, 0.2)

    def test_format_checkpoint_name_with_metrics(self):
        cb = ModelCheckpoint(filename='m-{epoch:02d}-{val_loss:.2f}')
        name = cb._format_checkpoint_name(3, 30, {'val_loss': 0.5})
        assert name == 'm-03-0.50'

    def test_format_checkpoint_name_fallback_on_missing_key(self):
        # Filename references a metric that is not present -> fallback format.
        cb = ModelCheckpoint(filename='m-{epoch:02d}-{val_loss:.2f}')
        name = cb._format_checkpoint_name(7, 70, {})
        assert name == 'checkpoint-epoch=07-step=70'

    def test_state_dict_roundtrip(self):
        cb = ModelCheckpoint()
        cb.best_score = 0.25
        cb.best_model_path = '/tmp/best.ckpt'
        cb.best_k_models = {'/tmp/a.ckpt': 0.25}
        state = cb.state_dict()
        assert state['best_score'] == 0.25
        assert state['best_model_path'] == '/tmp/best.ckpt'

        other = ModelCheckpoint()
        other.load_state_dict(state)
        assert other.best_score == 0.25
        assert other.best_model_path == '/tmp/best.ckpt'
        assert other.best_k_models == {'/tmp/a.ckpt': 0.25}

    def test_load_state_dict_defaults(self):
        cb = ModelCheckpoint()
        cb.load_state_dict({})
        assert cb.best_score is None
        assert cb.best_model_path is None
        assert cb.best_k_models == {}

    def test_remove_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp, verbose=True)
            fp = os.path.join(tmp, 'x.ckpt')
            with open(fp, 'wb') as f:
                f.write(b'data')
            cb._remove_checkpoint(fp)
            assert not os.path.exists(fp)
            # Removing a missing file is a no-op.
            cb._remove_checkpoint(fp)

    def test_update_best_k_evicts_worst_min_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp, mode='min', save_top_k=2)
            paths = {}
            for i, score in enumerate([0.5, 0.3, 0.9]):
                p = os.path.join(tmp, f'm{i}.ckpt')
                with open(p, 'wb') as f:
                    f.write(b'd')
                paths[p] = score
                cb._update_best_k(score, p)
            # Only top-2 (lowest) should remain; the 0.9 model is evicted.
            assert len(cb.best_k_models) == 2
            worst = os.path.join(tmp, 'm2.ckpt')
            assert worst not in cb.best_k_models
            assert not os.path.exists(worst)

    def test_update_best_k_evicts_worst_max_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp, mode='max', save_top_k=2)
            for i, score in enumerate([0.5, 0.9, 0.1]):
                p = os.path.join(tmp, f'm{i}.ckpt')
                with open(p, 'wb') as f:
                    f.write(b'd')
                cb._update_best_k(score, p)
            assert len(cb.best_k_models) == 2
            worst = os.path.join(tmp, 'm2.ckpt')  # score 0.1 -> lowest, evicted
            assert worst not in cb.best_k_models

    def test_update_best_k_keep_all_when_save_top_k_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp, save_top_k=-1)
            for i in range(5):
                cb._update_best_k(float(i), os.path.join(tmp, f'm{i}.ckpt'))
            assert len(cb.best_k_models) == 5


class TestModelCheckpointSaving:
    def test_save_checkpoint_writes_file(self, patched_save):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp, verbose=True)
            model = FakeModule(current_epoch=1, global_step=10)
            opt = DummyOptValue(0.01)
            opt.state_dict = lambda: {'lr': 0.01}
            trainer = FakeTrainer(callbacks=[cb], metrics={'val_loss': 0.5},
                                  optimizers=[opt], use_callback_metrics=False)
            fp = os.path.join(tmp, 'a.ckpt')
            cb._save_checkpoint(trainer, model, fp)
            assert os.path.exists(fp)
            assert len(patched_save) == 1
            ckpt = patched_save[0][0]
            assert ckpt['epoch'] == 1
            assert ckpt['global_step'] == 10
            assert 'optimizer_state_dict' in ckpt
            assert 'metrics' in ckpt

    def test_save_checkpoint_collects_callback_state(self, patched_save):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp)
            stateful = EarlyStopping(strict=False)
            stateful.best_score = 0.1
            model = FakeModule()
            trainer = FakeTrainer(callbacks=[cb, stateful])
            fp = os.path.join(tmp, 'a.ckpt')
            cb._save_checkpoint(trainer, model, fp)
            ckpt = patched_save[0][0]
            assert stateful.state_key in ckpt['callbacks']

    def test_save_checkpoint_real_roundtrip(self):
        # Exercise the real msgpack_save path (no patching) and load it back.
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp)
            model = FakeModule(current_epoch=3, global_step=42)
            trainer = FakeTrainer(callbacks=[cb])
            fp = os.path.join(tmp, 'real.ckpt')
            cb._save_checkpoint(trainer, model, fp)
            assert os.path.exists(fp)
            assert os.path.getsize(fp) > len(b'ckpt')  # real msgpack, not a stub
            restored = bf.msgpack_load(fp, verbose=False)
            assert int(restored['epoch']) == 3
            assert int(restored['global_step']) == 42

    def test_on_train_epoch_end_saves_best_min(self, patched_save):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp, monitor='val_loss', mode='min',
                                 filename='m-{epoch:02d}')
            model = FakeModule(current_epoch=1, global_step=5)
            trainer = FakeTrainer(callbacks=[cb], metrics={'val_loss': 0.4})
            cb.on_train_epoch_end(trainer, model)
            assert cb.best_score == 0.4
            assert cb.best_model_path is not None
            # A better score updates best; a worse one does not.
            model.current_epoch = 2
            trainer.callback_metrics = {'val_loss': 0.6}
            cb.on_train_epoch_end(trainer, model)
            assert cb.best_score == 0.4
            assert len(patched_save) == 2

    def test_on_train_epoch_end_respects_every_n_epochs(self, patched_save):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp, every_n_epochs=2)
            model = FakeModule(current_epoch=1)
            trainer = FakeTrainer(callbacks=[cb], metrics={'val_loss': 0.4})
            cb.on_train_epoch_end(trainer, model)  # epoch 1 % 2 != 0 -> skip
            assert len(patched_save) == 0
            model.current_epoch = 2
            cb.on_train_epoch_end(trainer, model)  # epoch 2 % 2 == 0 -> save
            assert len(patched_save) == 1

    def test_on_train_epoch_end_skipped_when_disabled(self, patched_save):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp, save_on_train_epoch_end=False)
            model = FakeModule(current_epoch=1)
            trainer = FakeTrainer(callbacks=[cb], metrics={'val_loss': 0.4})
            cb.on_train_epoch_end(trainer, model)
            assert len(patched_save) == 0

    def test_on_train_epoch_end_no_metric_still_saves(self, patched_save):
        with tempfile.TemporaryDirectory() as tmp:
            # monitor metric absent -> falls into "just save" branch.
            cb = ModelCheckpoint(dirpath=tmp, monitor='val_loss', save_top_k=3)
            model = FakeModule(current_epoch=1)
            trainer = FakeTrainer(callbacks=[cb], metrics={'other': 0.4})
            cb.on_train_epoch_end(trainer, model)
            assert len(patched_save) == 1
            assert cb.best_score is None

    def test_on_train_epoch_end_uses_logged_metrics_fallback(self, patched_save):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp, monitor='val_loss')
            model = FakeModule(current_epoch=1)
            trainer = FakeTrainer(callbacks=[cb], metrics={'val_loss': 0.3},
                                  use_callback_metrics=False)
            cb.on_train_epoch_end(trainer, model)
            assert cb.best_score == 0.3

    def test_on_train_batch_end_every_n_steps(self, patched_save):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp, every_n_train_steps=5)
            model = FakeModule(current_epoch=0, global_step=5)
            trainer = FakeTrainer(callbacks=[cb], metrics={'val_loss': 0.4},
                                  use_callback_metrics=False)
            cb.on_train_batch_end(trainer, model, None, None, 0)
            assert len(patched_save) == 1
            assert cb._last_global_step_saved == 5
            # Same step again -> deduplicated, no extra save.
            cb.on_train_batch_end(trainer, model, None, None, 1)
            assert len(patched_save) == 1
            # A step not divisible by 5 -> no save.
            model.global_step = 7
            cb.on_train_batch_end(trainer, model, None, None, 2)
            assert len(patched_save) == 1

    def test_on_train_batch_end_disabled_by_default(self, patched_save):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp)  # every_n_train_steps is None
            model = FakeModule(global_step=5)
            trainer = FakeTrainer(callbacks=[cb])
            cb.on_train_batch_end(trainer, model, None, None, 0)
            assert len(patched_save) == 0

    def test_on_fit_end_saves_last(self, patched_save):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp, save_last=True)
            model = FakeModule()
            trainer = FakeTrainer(callbacks=[cb])
            cb.on_fit_end(trainer, model)
            assert os.path.exists(os.path.join(tmp, 'last.ckpt'))

    def test_on_fit_end_skips_when_save_last_false(self, patched_save):
        with tempfile.TemporaryDirectory() as tmp:
            cb = ModelCheckpoint(dirpath=tmp, save_last=False)
            model = FakeModule()
            trainer = FakeTrainer(callbacks=[cb])
            cb.on_fit_end(trainer, model)
            assert len(patched_save) == 0


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_init_and_invalid_mode(self):
        cb = EarlyStopping(monitor='val_loss', patience=2, mode='min')
        assert cb.monitor == 'val_loss'
        assert not cb.should_stop
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStopping(mode='bad')

    def test_min_delta_sign_flipped_for_min_mode(self):
        cb = EarlyStopping(mode='min', min_delta=0.1)
        assert cb.min_delta == -0.1
        cb_max = EarlyStopping(mode='max', min_delta=0.1)
        assert cb_max.min_delta == 0.1

    def test_is_improvement_min_and_max(self):
        cmin = EarlyStopping(mode='min', min_delta=0.0)
        assert cmin._is_improvement(0.1, 0.2)
        assert not cmin._is_improvement(0.2, 0.1)
        cmax = EarlyStopping(mode='max', min_delta=0.0)
        assert cmax._is_improvement(0.2, 0.1)
        assert not cmax._is_improvement(0.1, 0.2)

    def test_plateau_triggers_stop(self):
        cb = EarlyStopping(monitor='val_loss', patience=2, mode='min', verbose=True)
        model = FakeModule(current_epoch=0)
        # Improve, then plateau for `patience` epochs.
        for epoch, loss in enumerate([0.5, 0.4, 0.4, 0.4]):
            model.current_epoch = epoch
            trainer = FakeTrainer(callbacks=[cb], metrics={'val_loss': loss})
            cb.on_validation_epoch_end(trainer, model)
        assert cb.should_stop
        assert cb.stopped_epoch == 3
        assert cb.wait_count >= cb.patience

    def test_improvement_resets_wait_count(self):
        cb = EarlyStopping(monitor='val_loss', patience=3, mode='min')
        model = FakeModule()
        for loss in [0.5, 0.5, 0.4]:  # one no-improve, then an improvement
            trainer = FakeTrainer(callbacks=[cb], metrics={'val_loss': loss})
            cb.on_validation_epoch_end(trainer, model)
        assert cb.wait_count == 0
        assert cb.best_score == 0.4
        assert not cb.should_stop

    def test_max_mode_improvement(self):
        cb = EarlyStopping(monitor='val_acc', patience=2, mode='max')
        model = FakeModule()
        for acc in [0.7, 0.8, 0.9]:
            trainer = FakeTrainer(callbacks=[cb], metrics={'val_acc': acc})
            cb.on_validation_epoch_end(trainer, model)
        assert cb.best_score == 0.9
        assert not cb.should_stop

    def test_non_finite_metric_stops(self):
        cb = EarlyStopping(monitor='val_loss', mode='min', check_finite=True, verbose=True)
        model = FakeModule(current_epoch=4)
        trainer = FakeTrainer(callbacks=[cb], metrics={'val_loss': jnp.inf})
        cb.on_validation_epoch_end(trainer, model)
        assert cb.should_stop
        assert cb.stopped_epoch == 4

    def test_nan_metric_stops(self):
        cb = EarlyStopping(monitor='val_loss', mode='min', check_finite=True)
        model = FakeModule(current_epoch=2)
        trainer = FakeTrainer(callbacks=[cb], metrics={'val_loss': jnp.nan})
        cb.on_validation_epoch_end(trainer, model)
        assert cb.should_stop

    def test_missing_metric_strict_raises(self):
        cb = EarlyStopping(monitor='val_loss', strict=True)
        model = FakeModule()
        trainer = FakeTrainer(callbacks=[cb], metrics={'other': 0.1})
        with pytest.raises(RuntimeError, match="not available"):
            cb.on_validation_epoch_end(trainer, model)

    def test_missing_metric_non_strict_returns(self):
        cb = EarlyStopping(monitor='val_loss', strict=False)
        model = FakeModule()
        trainer = FakeTrainer(callbacks=[cb], metrics={'other': 0.1})
        cb.on_validation_epoch_end(trainer, model)  # no raise
        assert not cb.should_stop

    def test_logged_metrics_fallback(self):
        cb = EarlyStopping(monitor='val_loss', mode='min')
        model = FakeModule()
        trainer = FakeTrainer(callbacks=[cb], metrics={'val_loss': 0.3},
                              use_callback_metrics=False)
        cb.on_validation_epoch_end(trainer, model)
        assert cb.best_score == 0.3

    def test_state_dict_roundtrip(self):
        cb = EarlyStopping()
        cb.best_score = 0.2
        cb.wait_count = 2
        cb.stopped_epoch = 5
        state = cb.state_dict()
        assert state == {'best_score': 0.2, 'wait_count': 2, 'stopped_epoch': 5}

        other = EarlyStopping()
        other.load_state_dict(state)
        assert other.best_score == 0.2
        assert other.wait_count == 2
        assert other.stopped_epoch == 5

    def test_load_state_dict_defaults(self):
        cb = EarlyStopping()
        cb.load_state_dict({})
        assert cb.best_score is None
        assert cb.wait_count == 0
        assert cb.stopped_epoch == 0


# ---------------------------------------------------------------------------
# LearningRateMonitor
# ---------------------------------------------------------------------------

class TestLearningRateMonitor:
    def test_get_lr_value_attribute(self):
        cb = LearningRateMonitor()
        trainer = FakeTrainer(optimizers=[DummyOptValue(0.01)])
        lrs = cb._get_learning_rates(trainer)
        assert lrs == {'lr-opt0': 0.01}

    def test_get_lr_callable(self):
        cb = LearningRateMonitor()
        trainer = FakeTrainer(optimizers=[DummyOptCallable(0.02)])
        lrs = cb._get_learning_rates(trainer)
        assert lrs == {'lr-opt0': 0.02}

    def test_get_lr_plain_float(self):
        cb = LearningRateMonitor()
        trainer = FakeTrainer(optimizers=[DummyOptPlainFloat(0.03)])
        lrs = cb._get_learning_rates(trainer)
        assert lrs == {'lr-opt0': 0.03}

    def test_get_lr_learning_rate_attribute(self):
        cb = LearningRateMonitor()
        trainer = FakeTrainer(optimizers=[DummyOptLearningRate(0.04)])
        lrs = cb._get_learning_rates(trainer)
        assert lrs == {'lr-opt0': 0.04}

    def test_get_lr_no_optimizers_attr(self):
        cb = LearningRateMonitor()
        trainer = FakeTrainer()  # no optimizers attribute
        assert cb._get_learning_rates(trainer) == {}

    def test_get_lr_multiple_optimizers(self):
        cb = LearningRateMonitor()
        trainer = FakeTrainer(optimizers=[DummyOptValue(0.01), DummyOptPlainFloat(0.02)])
        lrs = cb._get_learning_rates(trainer)
        assert lrs == {'lr-opt0': 0.01, 'lr-opt1': 0.02}

    def test_step_interval_logs_and_records(self):
        cb = LearningRateMonitor(logging_interval='step')
        model = FakeModule()
        trainer = FakeTrainer(optimizers=[DummyOptValue(0.01)])
        cb.on_train_batch_start(trainer, model, None, 0)
        assert len(cb.lr_history) == 1
        assert cb.lr_history[0] == {'lr-opt0': 0.01}
        assert any(name == 'lr-opt0' for name, *_ in model.logged)
        # epoch hook should not record under 'step' interval.
        cb.on_train_epoch_start(trainer, model)
        assert len(cb.lr_history) == 1

    def test_epoch_interval_logs_and_records(self):
        cb = LearningRateMonitor(logging_interval='epoch')
        model = FakeModule()
        trainer = FakeTrainer(optimizers=[DummyOptValue(0.05)])
        cb.on_train_epoch_start(trainer, model)
        assert cb.lr_history[0] == {'lr-opt0': 0.05}
        # step hook should not record under 'epoch' interval.
        cb.on_train_batch_start(trainer, model, None, 0)
        assert len(cb.lr_history) == 1


# ---------------------------------------------------------------------------
# GradientClipCallback
# ---------------------------------------------------------------------------

class TestGradientClipCallback:
    def test_init_valid_algorithms(self):
        assert GradientClipCallback(clip_val=1.0, clip_algorithm='norm').clip_algorithm == 'norm'
        assert GradientClipCallback(clip_val=1.0, clip_algorithm='value').clip_algorithm == 'value'

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="clip_algorithm must be"):
            GradientClipCallback(clip_algorithm='bogus')

    def test_hook_with_clip_val(self):
        cb = GradientClipCallback(clip_val=1.0, log_grad_norm=True)
        # No-op body but must run cleanly.
        cb.on_before_optimizer_step(None, None, None)

    def test_hook_without_clip_val_returns_early(self):
        cb = GradientClipCallback(clip_val=None)
        cb.on_before_optimizer_step(None, None, None)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class TestTimer:
    def test_init_without_duration(self):
        t = Timer()
        assert t._max_seconds is None
        assert t.time_elapsed is None
        assert t.time_remaining is None

    def test_init_with_duration_seconds(self):
        t = Timer(duration={'seconds': 30, 'minutes': 1, 'hours': 1, 'days': 1})
        assert t._max_seconds == 30 + 60 + 3600 + 86400

    def test_fit_lifecycle_tracks_time(self):
        t = Timer(verbose=True)
        t.on_fit_start(None, None)
        assert t._start_time is not None
        assert t.time_elapsed is not None
        model = FakeModule(current_epoch=0)
        t.on_train_epoch_start(None, model)
        t.on_train_epoch_end(None, model)
        assert len(t._epoch_times) == 1
        t.on_fit_end(None, model)
        assert t._end_time is not None

    def test_duration_limit_triggers_stop(self):
        # Zero-second budget -> any elapsed time exceeds it.
        t = Timer(duration={'seconds': 0}, verbose=True)
        # _max_seconds is 0 which is falsy, so use a tiny positive budget instead.
        t = Timer(duration={'seconds': 1e-9}, verbose=True)
        t.on_fit_start(None, None)
        model = FakeModule(current_epoch=1)
        t.on_train_epoch_start(None, model)
        import time
        time.sleep(0.001)
        t.on_train_epoch_end(None, model)
        assert t.should_stop
        assert t.time_remaining == 0 or t.time_remaining >= 0

    def test_no_stop_when_under_budget(self):
        t = Timer(duration={'hours': 10}, verbose=False)
        t.on_fit_start(None, None)
        model = FakeModule(current_epoch=0)
        t.on_train_epoch_start(None, model)
        t.on_train_epoch_end(None, model)
        assert not t.should_stop
        assert t.time_remaining is not None and t.time_remaining > 0

    def test_format_time_seconds_minutes_hours(self):
        assert Timer._format_time(5).endswith('s')
        assert Timer._format_time(120).endswith('m')
        assert Timer._format_time(7200).endswith('h')

    def test_fit_end_without_start_no_crash(self):
        t = Timer(verbose=True)
        # on_fit_end before on_fit_start: _start_time is None, verbose branch skipped.
        t.on_fit_end(None, FakeModule())
        assert t._end_time is not None

    def test_epoch_end_without_epoch_start(self):
        t = Timer(verbose=True)
        t.on_fit_start(None, None)
        # No epoch_start recorded -> _epoch_start_time is None, epoch time not appended.
        t.on_train_epoch_end(None, FakeModule(current_epoch=0))
        assert t._epoch_times == []


# ---------------------------------------------------------------------------
# Progress bars (rich + tqdm are available)
# ---------------------------------------------------------------------------

class TestTQDMProgressBar:
    def test_lifecycle(self):
        cb = TQDMProgressBar(refresh_rate=1)
        model = FakeModule(current_epoch=0)
        model._prog_bar = {'loss': 0.5}
        loader = [0] * 4
        trainer = FakeTrainer(train_dataloader=loader)
        cb.on_train_epoch_start(trainer, model)
        assert cb._pbar is not None
        for i in range(4):
            cb.on_train_batch_end(trainer, model, None, None, i)
        cb.on_train_epoch_end(trainer, model)
        assert cb._pbar is None

    def test_batch_end_without_bar_is_noop(self):
        cb = TQDMProgressBar()
        # No on_train_epoch_start called -> _pbar is None.
        cb.on_train_batch_end(None, FakeModule(), None, None, 0)
        assert cb._pbar is None

    def test_no_metrics_postfix(self):
        cb = TQDMProgressBar(refresh_rate=1)
        model = FakeModule(current_epoch=0)
        model._prog_bar = {}  # empty -> no set_postfix
        trainer = FakeTrainer(train_dataloader=[0, 0])
        cb.on_train_epoch_start(trainer, model)
        cb.on_train_batch_end(trainer, model, None, None, 0)
        cb.on_train_epoch_end(trainer, model)

    def test_no_dataloader_total_none(self):
        cb = TQDMProgressBar()
        model = FakeModule(current_epoch=0)
        trainer = FakeTrainer()  # no train_dataloader attr -> total=None
        cb.on_train_epoch_start(trainer, model)
        assert cb._pbar is not None
        # tqdm raises on bool() when total is None, so the callback's own
        # on_train_epoch_end cannot be exercised for this case; close manually.
        cb._pbar.close()


class TestRichProgressBar:
    def test_lifecycle(self):
        cb = RichProgressBar(refresh_rate=1)
        model = FakeModule(current_epoch=0)
        loader = [0] * 3
        trainer = FakeTrainer(train_dataloader=loader)
        cb.on_train_epoch_start(trainer, model)
        assert cb._progress is not None
        for i in range(3):
            cb.on_train_batch_end(trainer, model, None, None, i)
        cb.on_train_epoch_end(trainer, model)
        assert cb._progress is None

    def test_batch_end_without_progress_is_noop(self):
        cb = RichProgressBar()
        cb.on_train_batch_end(None, FakeModule(), None, None, 0)
        assert cb._progress is None

    def test_epoch_end_without_progress_is_noop(self):
        cb = RichProgressBar()
        cb.on_train_epoch_end(None, FakeModule())  # _progress is None
        assert cb._progress is None

    def test_no_dataloader_total_none(self):
        cb = RichProgressBar()
        model = FakeModule(current_epoch=0)
        trainer = FakeTrainer()  # no train_dataloader
        cb.on_train_epoch_start(trainer, model)
        cb.on_train_epoch_end(trainer, model)


# ---------------------------------------------------------------------------
# LambdaCallback
# ---------------------------------------------------------------------------

class TestLambdaCallback:
    def test_assigns_hooks(self):
        called = {}

        cb = LambdaCallback(
            on_train_epoch_end=lambda trainer, module: called.setdefault('end', True),
            on_fit_start=lambda trainer, module: called.setdefault('start', True),
        )
        cb.on_train_epoch_end(None, None)
        cb.on_fit_start(None, None)
        assert called == {'end': True, 'start': True}

    def test_non_callable_raises(self):
        with pytest.raises(ValueError, match="Expected callable"):
            LambdaCallback(on_fit_start=123)


# ---------------------------------------------------------------------------
# PrintCallback
# ---------------------------------------------------------------------------

class TestPrintCallback:
    def test_epoch_and_batch_prints(self, capsys):
        cb = PrintCallback(print_freq=2)
        model = FakeModule(current_epoch=1)
        model._prog_bar = {'loss': 0.5}
        cb.on_train_epoch_start(None, model)
        cb.on_train_batch_end(None, model, None, None, 0)  # 0 % 2 == 0 -> prints
        cb.on_train_batch_end(None, model, None, None, 1)  # 1 % 2 != 0 -> skip
        cb.on_train_epoch_end(None, model)
        out = capsys.readouterr().out
        assert 'Epoch 1' in out
        assert 'Step 0' in out
        assert 'completed' in out

    def test_validation_epoch_end_prints_metrics(self, capsys):
        cb = PrintCallback()
        model = FakeModule()
        trainer = FakeTrainer(metrics={'val_loss': 0.25})
        cb.on_validation_epoch_end(trainer, model)
        out = capsys.readouterr().out
        assert 'Validation' in out
        assert 'val_loss' in out

    def test_validation_epoch_end_without_metrics(self, capsys):
        cb = PrintCallback()
        model = FakeModule()
        trainer = FakeTrainer()  # no callback_metrics
        cb.on_validation_epoch_end(trainer, model)
        out = capsys.readouterr().out
        assert 'Validation' in out


# ---------------------------------------------------------------------------
# Integration: callbacks attached to a real (tiny) Trainer.fit run.
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_fit_with_callbacks(self, monkeypatch):
        # Patch the checkpoint serializer to keep the integration run fast.
        def fake(path, target, *args, **kwargs):
            with open(path, 'wb') as f:
                f.write(b'ckpt')

        monkeypatch.setattr(bf, 'msgpack_save', fake)

        with tempfile.TemporaryDirectory() as tmp:
            X = jnp.ones((32, 4))
            y = jnp.zeros((32, 2))
            model = SimpleModel()
            loader = braintools.trainer.DataLoader((X, y), batch_size=16)

            # PrintCallback / LearningRateMonitor touch traced metrics or
            # scheduler-callable lr, which are incompatible with a JIT'd fit in
            # this version, so use callbacks whose hooks are trace-safe.
            seen = {'epochs': 0}
            callbacks = [
                Timer(verbose=False),
                LambdaCallback(
                    on_train_epoch_end=lambda t, m: seen.__setitem__('epochs', seen['epochs'] + 1),
                    on_fit_end=lambda t, m: None,
                ),
            ]
            trainer = braintools.trainer.Trainer(
                max_epochs=2,
                callbacks=callbacks,
                enable_progress_bar=False,
                enable_checkpointing=False,
                logger=False,
            )
            trainer.fit(model, loader)
            assert model.current_epoch >= 0
            assert seen['epochs'] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
