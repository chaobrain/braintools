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

"""Tests for ``braintools.trainer._distributed``.

These run on a single device (the standard CI environment). They focus on
strategy construction and dispatch -- in particular that mesh-based strategies
no longer crash at construction time (T-15) -- rather than true multi-host
collectives, which require hardware not present in CI.
"""

import functools
import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import pytest

import brainstate
import braintools
from braintools.trainer._distributed import (
    Strategy,
    SingleDeviceStrategy,
    DataParallelStrategy,
    ShardedDataParallelStrategy,
    FullyShardedDataParallelStrategy,
    AutoStrategy,
    get_strategy,
    all_reduce,
)


# ---------------------------------------------------------------------------
# get_strategy dispatch
# ---------------------------------------------------------------------------

class TestGetStrategy:
    def test_auto(self):
        assert isinstance(get_strategy('auto'), AutoStrategy)

    def test_none_is_auto(self):
        assert isinstance(get_strategy(None), AutoStrategy)

    def test_single(self):
        assert isinstance(get_strategy('single'), SingleDeviceStrategy)
        assert isinstance(get_strategy('single_device'), SingleDeviceStrategy)

    def test_data_parallel_aliases(self):
        for name in ('ddp', 'dp', 'data_parallel'):
            assert isinstance(get_strategy(name), DataParallelStrategy)

    def test_sharded(self):
        for name in ('sdp', 'sharded_data_parallel'):
            assert isinstance(get_strategy(name), ShardedDataParallelStrategy)

    def test_fsdp(self):
        assert isinstance(get_strategy('fsdp'), FullyShardedDataParallelStrategy)

    def test_instance_passthrough(self):
        s = SingleDeviceStrategy()
        assert get_strategy(s) is s

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match='Unknown strategy'):
            get_strategy('not_a_strategy')


# ---------------------------------------------------------------------------
# Strategy base class defaults
# ---------------------------------------------------------------------------

class TestStrategyBaseDefaults:
    def test_reduce_default_identity(self):
        s = SingleDeviceStrategy()
        t = jnp.asarray([1.0, 2.0])
        assert jnp.all(s.reduce(t) == t)

    def test_broadcast_default_identity(self):
        s = SingleDeviceStrategy()
        t = jnp.asarray([1.0, 2.0])
        assert jnp.all(s.broadcast(t) == t)

    def test_barrier_noop(self):
        s = SingleDeviceStrategy()
        assert s.barrier() is None

    def test_is_distributed_single(self):
        assert SingleDeviceStrategy().is_distributed is False


# ---------------------------------------------------------------------------
# SingleDeviceStrategy
# ---------------------------------------------------------------------------

class TestSingleDeviceStrategy:
    def test_properties(self):
        s = SingleDeviceStrategy()
        assert s.name == 'single_device'
        assert s.num_devices == 1
        assert len(s.devices) == 1

    def test_setup_returns_model_and_optimizer(self):
        s = SingleDeviceStrategy()
        model, opt = object(), object()
        out_model, out_opt = s.setup(model, opt)
        assert out_model is model and out_opt is opt

    def test_training_step_updates_params(self):
        class Net(braintools.trainer.LightningModule):
            def __init__(self):
                super().__init__()
                self.lin = brainstate.nn.Linear(3, 1)

        model = Net()
        opt = braintools.optim.SGD(lr=1e-1)
        params = braintools.optim.UniqueStateManager(
            model.states(brainstate.ParamState)).to_pytree()
        opt.register_trainable_weights(params)

        def loss_fn(m, batch):
            x, y = batch
            return jnp.mean((m.lin(x) - y) ** 2)

        before = {k: jnp.array(v) for k, v in model.lin.weight.value.items()}
        batch = (jnp.ones((4, 3)), jnp.zeros((4, 1)))
        s = SingleDeviceStrategy()
        loss, metrics = s.training_step(model, opt, batch, loss_fn, params)
        after = {k: jnp.array(v) for k, v in model.lin.weight.value.items()}
        assert metrics == {}
        assert any(bool(jnp.any(jnp.abs(before[k] - after[k]) > 1e-7)) for k in before)


# ---------------------------------------------------------------------------
# AutoStrategy
# ---------------------------------------------------------------------------

class TestAutoStrategy:
    def test_single_device_selection(self):
        # CI exposes a single device, so Auto must fall back to single-device.
        s = AutoStrategy()
        assert isinstance(s.selected_strategy, SingleDeviceStrategy)
        assert s.num_devices == 1
        assert s.name.startswith('auto(')
        assert len(s.devices) == 1

    def test_setup_delegates(self):
        s = AutoStrategy()
        m, o = object(), object()
        assert s.setup(m, o) == (m, o)


# ---------------------------------------------------------------------------
# Mesh-based strategies must construct without crashing (T-15)
# ---------------------------------------------------------------------------

class TestMeshStrategiesConstruct:
    def test_sharded_data_parallel_builds_mesh(self):
        s = ShardedDataParallelStrategy()
        assert s.name == 'sharded_data_parallel'
        assert s.num_devices == jax.device_count()
        assert s.mesh is not None

    def test_fsdp_1d_mesh(self):
        # Previously ``jax.devices()`` (a list) was passed to Mesh; now wrapped
        # in np.asarray so construction succeeds.
        s = FullyShardedDataParallelStrategy()
        assert s.name == 'fsdp'
        assert s.mesh is not None
        assert s.num_devices == jax.device_count()

    def test_fsdp_2d_mesh_with_model_axis(self):
        # The 2D path calls ``devices.reshape(...)`` which crashed on a plain
        # list before T-15. A (1, 1) mesh is valid on a single device.
        s = FullyShardedDataParallelStrategy(model_axis='model')
        assert s.mesh is not None
        assert s.num_devices == jax.device_count()

    def test_fsdp_param_sharding_specs(self):
        s = FullyShardedDataParallelStrategy(model_axis='model')
        # Scalar, vector, and matrix param shapes map to PartitionSpecs.
        assert s._get_param_sharding(()) is not None
        assert s._get_param_sharding((10,)) is not None
        assert s._get_param_sharding((10, 20)) is not None


# ---------------------------------------------------------------------------
# Collective utility functions
# ---------------------------------------------------------------------------

class TestAllReduce:
    def test_invalid_op_raises(self):
        # The op is validated before any collective context is required.
        with pytest.raises(ValueError, match='Unknown reduction'):
            all_reduce(jnp.asarray(1.0), op='median', axis_name='batch')

    @pytest.mark.parametrize('op', ['mean', 'sum', 'min', 'max'])
    def test_ops_inside_pmap(self, op):
        # Exercise the valid branches within a (single-device) pmap context.
        fn = functools.partial(jax.pmap, axis_name='batch')(
            lambda x: all_reduce(x, op=op, axis_name='batch'))
        out = fn(jnp.arange(jax.device_count() * 2.0).reshape(jax.device_count(), 2))
        assert out.shape[0] == jax.device_count()

    def test_data_parallel_reduce_invalid_op(self):
        s = DataParallelStrategy()
        with pytest.raises(ValueError, match='Unknown reduction'):
            s.reduce(jnp.asarray(1.0), op='bogus')


# ---------------------------------------------------------------------------
# T-D / T-H: distributed collectives on multiple (fake CPU) devices.
# Run in a subprocess so XLA can be configured with >1 host device before
# JAX initialises in this process.
# ---------------------------------------------------------------------------

def _run_multidevice(snippet: str, n_devices: int = 4):
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ)
    env['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={n_devices}"
    env['JAX_PLATFORMS'] = 'cpu'
    env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')
    proc = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(snippet)],
        capture_output=True, text=True, env=env, timeout=300,
    )
    return proc


class TestDistributedCollectivesMultiDevice:
    def test_broadcast_valid_on_multiple_devices(self):
        snippet = """
            import jax, jax.numpy as jnp
            from braintools.trainer import broadcast
            assert jax.device_count() >= 2, jax.device_count()
            x = jnp.arange(jax.device_count() * 1.0)
            g = jax.pmap(lambda v: broadcast(v, src=0, axis_name='batch'), axis_name='batch')
            out = g(x)
            # Every replica must hold device 0's value (x[0] == 0.0).
            assert jnp.allclose(out, x[0]), out
            print('BROADCAST_OK')
        """
        proc = _run_multidevice(snippet)
        assert 'BROADCAST_OK' in proc.stdout, (
            f"stdout={proc.stdout!r}\nstderr={proc.stderr[-2000:]!r}"
        )

    def test_sync_batch_norm_matches_global_stats(self):
        snippet = """
            import jax, jax.numpy as jnp
            from braintools.trainer._distributed import sync_batch_norm
            nd = jax.device_count()
            assert nd >= 2, nd
            # Per-device data with DIFFERENT means so pooled var != mean of vars.
            data = jnp.stack([jnp.arange(4.0) + 10.0 * i for i in range(nd)])  # (nd, 4)
            mean, var = jax.pmap(lambda x: sync_batch_norm(x, axis_name='batch'),
                                 axis_name='batch')(data)
            flat = data.reshape(-1)
            exp_mean = jnp.mean(flat)
            exp_var = jnp.var(flat)
            assert jnp.allclose(mean[0], exp_mean, atol=1e-4), (mean[0], exp_mean)
            assert jnp.allclose(var[0], exp_var, atol=1e-4), (var[0], exp_var)
            print('SBN_OK')
        """
        proc = _run_multidevice(snippet)
        assert 'SBN_OK' in proc.stdout, (
            f"stdout={proc.stdout!r}\nstderr={proc.stderr[-2000:]!r}"
        )


# ---------------------------------------------------------------------------
# T-G: FSDP auto-mesh must give the model axis a real size when devices allow.
# ---------------------------------------------------------------------------

class TestFSDPMeshBalancing:
    def test_model_axis_gets_real_size(self):
        snippet = """
            import jax
            from braintools.trainer import FullyShardedDataParallelStrategy
            assert jax.device_count() == 4, jax.device_count()
            s = FullyShardedDataParallelStrategy(data_axis='data', model_axis='model')
            shape = s.mesh.shape
            # 4 devices -> a balanced 2x2 mesh; model axis must be > 1.
            assert shape['model'] > 1, shape
            assert shape['data'] * shape['model'] == 4, shape
            print('FSDP_OK')
        """
        proc = _run_multidevice(snippet, n_devices=4)
        assert 'FSDP_OK' in proc.stdout, (
            f"stdout={proc.stdout!r}\nstderr={proc.stderr[-2000:]!r}"
        )
