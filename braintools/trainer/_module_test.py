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

"""Tests for the LightningModule base class and output containers."""

import jax.numpy as jnp
import pytest

import brainstate
import braintools
from braintools.trainer._module import (
    LightningModule,
    TrainOutput,
    EvalOutput,
    _to_scalar,
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


class BareModel(braintools.trainer.LightningModule):
    """Model that does not override optional hooks/steps."""

    def __init__(self):
        super().__init__()
        self.linear = brainstate.nn.Linear(4, 2)

    def __call__(self, x):
        return self.linear(x)


class FlatParamModel(braintools.trainer.LightningModule):
    """Model whose ParamState values are flat arrays (not nested dicts).

    ``print_summary`` calls ``jnp.size`` / accesses ``.shape`` on each
    ParamState value, which only works when the value is an array. The
    brainstate ``Linear`` layer stores its parameters as a nested dict, so
    a flat-array model is used to exercise that code path.
    """

    def __init__(self):
        super().__init__()
        self.w = brainstate.ParamState(jnp.ones((3, 4)))
        self.b = brainstate.ParamState(jnp.zeros((4,)))

    def __call__(self, x):
        return x @ self.w.value + self.b.value


class TestToScalar:
    """Tests for the _to_scalar helper."""

    def test_jax_scalar(self):
        assert _to_scalar(jnp.array(3.0)) == 3.0

    def test_python_float_passthrough(self):
        assert _to_scalar(2.5) == 2.5

    def test_python_int_passthrough(self):
        # int has .item()? no -> returned unchanged
        assert _to_scalar(7) == 7

    def test_numpy_scalar(self):
        import numpy as np
        assert _to_scalar(np.float32(1.5)) == 1.5

    def test_string_passthrough(self):
        assert _to_scalar('hello') == 'hello'

    def test_item_raises_falls_back_to_float(self):
        class WeirdScalar:
            def item(self):
                raise RuntimeError('no item')

            def __float__(self):
                return 4.5

        assert _to_scalar(WeirdScalar()) == 4.5


class TestTrainOutput:
    """Tests for the TrainOutput container."""

    def test_loss_access(self):
        out = TrainOutput(loss=0.5)
        assert out.loss == 0.5
        assert out['loss'] == 0.5
        assert out.get('loss') == 0.5

    def test_metrics_default_empty(self):
        out = TrainOutput(loss=0.1)
        assert out.metrics == {}

    def test_metrics_access(self):
        out = TrainOutput(loss=0.1, metrics={'acc': 0.9})
        assert out['acc'] == 0.9
        assert out.get('acc') == 0.9

    def test_missing_metric_getitem_none(self):
        out = TrainOutput(loss=0.1)
        assert out['missing'] is None

    def test_missing_metric_get_default(self):
        out = TrainOutput(loss=0.1)
        assert out.get('missing', 42) == 42


class TestEvalOutput:
    """Tests for the EvalOutput container."""

    def test_default_empty(self):
        out = EvalOutput()
        assert out.metrics == {}

    def test_metric_access(self):
        out = EvalOutput({'val_loss': 0.3})
        assert out['val_loss'] == 0.3
        assert out.get('val_loss') == 0.3

    def test_missing_metric_getitem_none(self):
        out = EvalOutput({'a': 1})
        assert out['b'] is None

    def test_missing_metric_get_default(self):
        out = EvalOutput({'a': 1})
        assert out.get('b', 'fallback') == 'fallback'


class TestLightningModuleProperties:
    """Tests for property accessors and setters."""

    def test_init_defaults(self):
        model = SimpleModel()
        assert model.current_epoch == 0
        assert model.global_step == 0
        assert model.trainer is None
        assert model.logged_metrics == {}

    def test_trainer_setter(self):
        model = SimpleModel()
        sentinel = object()
        model.trainer = sentinel
        assert model.trainer is sentinel

    def test_current_epoch_setter(self):
        model = SimpleModel()
        model.current_epoch = 12
        assert model.current_epoch == 12

    def test_global_step_setter(self):
        model = SimpleModel()
        model.global_step = 99
        assert model.global_step == 99

    def test_logged_metrics_returns_copy(self):
        model = SimpleModel()
        model.log('a', 1.0)
        snap = model.logged_metrics
        snap['injected'] = 'x'
        # Mutating the returned copy must not affect internal state
        assert 'injected' not in model._logged_metrics

    def test_device_property_no_array_params(self):
        model = SimpleModel()
        # SimpleModel's ParamState values are nested dicts (no .devices()),
        # so the device lookup falls through to None.
        assert model.device is None

    def test_device_property_with_array_param(self):
        # A ParamState holding a raw JAX array exercises the device lookup;
        # ``Array.devices()`` returns a set, so the property must not index it.
        class _ArrModel(braintools.trainer.LightningModule):
            def __init__(self):
                super().__init__()
                self.p = brainstate.ParamState(jnp.ones(3))

        model = _ArrModel()
        assert model.device is not None


class TestLogging:
    """Tests for the logging methods."""

    def test_log_records_full_metadata(self):
        model = SimpleModel()
        model.log('m', 0.5, prog_bar=True, logger=True, on_step=False,
                  on_epoch=True, reduce_fx='sum', sync_dist=True)
        rec = model._logged_metrics['m']
        assert rec['value'] == 0.5
        assert rec['prog_bar'] is True
        assert rec['logger'] is True
        assert rec['on_step'] is False
        assert rec['on_epoch'] is True
        assert rec['reduce_fx'] == 'sum'
        assert rec['sync_dist'] is True

    def test_log_prog_bar_tracked(self):
        model = SimpleModel()
        model.log('pb', 1.0, prog_bar=True)
        assert 'pb' in model._prog_bar_metrics

    def test_log_no_prog_bar_not_tracked(self):
        model = SimpleModel()
        model.log('nopb', 1.0, prog_bar=False)
        assert 'nopb' not in model._prog_bar_metrics

    def test_log_logger_tracked(self):
        model = SimpleModel()
        model.log('lg', 1.0, logger=True)
        assert 'lg' in model._logger_metrics

    def test_log_no_logger_not_tracked(self):
        model = SimpleModel()
        model.log('nolg', 1.0, logger=False)
        assert 'nolg' not in model._logger_metrics

    def test_log_dict(self):
        model = SimpleModel()
        model.log_dict({'a': 0.1, 'b': 0.2}, prog_bar=True)
        assert 'a' in model._logged_metrics
        assert 'b' in model._logged_metrics
        assert 'a' in model._prog_bar_metrics
        assert 'b' in model._prog_bar_metrics

    def test_reset_logged_metrics(self):
        model = SimpleModel()
        model.log('x', 1.0, prog_bar=True)
        assert model._logged_metrics
        model._reset_logged_metrics()
        assert model._logged_metrics == {}
        assert model._prog_bar_metrics == {}
        assert model._logger_metrics == {}

    def test_get_prog_bar_metrics_scalarized(self):
        model = SimpleModel()
        model.log('pb', jnp.array(0.75), prog_bar=True)
        metrics = model._get_prog_bar_metrics()
        assert metrics['pb'] == 0.75
        assert isinstance(metrics['pb'], float)

    def test_get_logger_metrics_scalarized(self):
        model = SimpleModel()
        model.log('lg', jnp.array(0.25), logger=True)
        metrics = model._get_logger_metrics()
        assert metrics['lg'] == 0.25


class TestStepHooks:
    """Tests for the *_step methods and lifecycle hooks."""

    def test_training_step_not_implemented(self):
        model = BareModel()
        with pytest.raises(NotImplementedError):
            model.training_step((jnp.ones((1, 4)), jnp.zeros((1, 2))), 0)

    def test_configure_optimizers_not_implemented(self):
        model = BareModel()
        with pytest.raises(NotImplementedError):
            model.configure_optimizers()

    def test_default_validation_step_returns_none(self):
        model = BareModel()
        assert model.validation_step(None, 0) is None

    def test_default_test_step_returns_none(self):
        model = BareModel()
        assert model.test_step(None, 0) is None

    def test_default_predict_step_returns_none(self):
        model = BareModel()
        assert model.predict_step(None, 0) is None

    def test_overridden_training_step(self):
        model = SimpleModel()
        out = model.training_step((jnp.ones((3, 10)), jnp.zeros((3, 2))), 0)
        assert 'loss' in out
        assert 'train_loss' in model._logged_metrics

    def test_overridden_validation_step(self):
        model = SimpleModel()
        out = model.validation_step((jnp.ones((3, 10)), jnp.zeros((3, 2))), 0)
        assert 'val_loss' in out

    def test_configure_optimizers_returns_optimizer(self):
        model = SimpleModel()
        opt = model.configure_optimizers()
        assert isinstance(opt, braintools.optim.Optimizer)


class TestLifecycleHooks:
    """Tests that default lifecycle hooks are callable no-ops."""

    def test_all_default_hooks_callable(self):
        model = BareModel()
        # No-arg hooks
        for name in [
            'on_fit_start', 'on_fit_end', 'on_train_start', 'on_train_end',
            'on_train_epoch_start', 'on_train_epoch_end',
            'on_validation_start', 'on_validation_end',
            'on_validation_epoch_start', 'on_validation_epoch_end',
            'on_test_start', 'on_test_end',
            'on_test_epoch_start', 'on_test_epoch_end',
            'on_predict_start', 'on_predict_end', 'on_after_backward',
        ]:
            assert getattr(model, name)() is None

    def test_batch_hooks_callable(self):
        model = BareModel()
        batch = (jnp.ones((1, 4)), jnp.zeros((1, 2)))
        assert model.on_train_batch_start(batch, 0) is None
        assert model.on_train_batch_end('out', batch, 0) is None
        assert model.on_validation_batch_start(batch, 0) is None
        assert model.on_validation_batch_end('out', batch, 0) is None
        assert model.on_test_batch_start(batch, 0) is None
        assert model.on_test_batch_end('out', batch, 0) is None
        assert model.on_predict_batch_start(batch, 0) is None
        assert model.on_predict_batch_end('out', batch, 0) is None

    def test_optimizer_and_backward_hooks_callable(self):
        model = SimpleModel()
        opt = braintools.optim.Adam(lr=1e-3)
        assert model.on_before_optimizer_step(opt) is None
        assert model.on_after_optimizer_step(opt) is None
        assert model.on_before_backward(jnp.array(1.0)) is None


class TestStateManagement:
    """Tests for state_dict / load_state_dict."""

    def test_state_dict_contents(self):
        model = SimpleModel()
        sd = model.state_dict()
        assert isinstance(sd, dict)
        assert len(sd) > 0
        # All keys should be strings (stringified state names)
        assert all(isinstance(k, str) for k in sd)

    def test_load_state_dict_round_trip(self):
        src = SimpleModel()
        dst = SimpleModel()
        # Perturb dst so it differs initially
        sd_src = src.state_dict()
        dst.load_state_dict(sd_src)
        sd_dst = dst.state_dict()
        import jax
        for k in sd_src:
            for a, b in zip(jax.tree_util.tree_leaves(sd_src[k]),
                            jax.tree_util.tree_leaves(sd_dst[k])):
                assert jnp.allclose(a, b)

    def test_load_state_dict_ignores_unknown_keys(self):
        model = SimpleModel()
        sd = model.state_dict()
        sd['totally_unknown_key'] = jnp.zeros((2, 2))
        # Should not raise
        model.load_state_dict(sd)


class TestUtilityMethods:
    """Tests for freeze / unfreeze / print_summary."""

    def test_freeze_unfreeze(self):
        model = SimpleModel()
        # These iterate parameters; should not raise regardless of whether
        # ParamState exposes requires_grad.
        model.freeze()
        model.unfreeze()

    def test_print_summary(self, capsys):
        # FlatParamModel exposes flat-array ParamState values, which is what
        # print_summary's size/shape introspection requires.
        model = FlatParamModel()
        model.print_summary()
        captured = capsys.readouterr()
        assert 'FlatParamModel' in captured.out
        assert 'Total parameters' in captured.out
        assert 'Trainable parameters' in captured.out
        # 3*4 weights + 4 biases = 16 parameters
        assert '16' in captured.out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
