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

"""Regression tests for the 2026-06-19 ``braintools.trainer`` audit.

Each test reproduces a specific issue documented in
``docs/braintools-trainer-issues-found-20260619.md`` (T-A .. T-H).
"""

import os
import subprocess
import sys
import tempfile
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

import brainstate
import braintools
from braintools.trainer import (
    Trainer,
    LightningModule,
    DataLoader,
    Logger,
    LearningRateMonitor,
    ModelCheckpoint,
    CheckpointManager,
)


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


def _loader(n=16, bs=4):
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
        trainer.fit(_Net(), _loader())
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
        trainer.fit(_Net(), _loader())
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
        trainer.fit(_Net(), _loader())
        assert 'train_loss' in spy.keys


# ---------------------------------------------------------------------------
# T-B: ModelCheckpoint must not leave more than save_top_k files on disk
# ---------------------------------------------------------------------------

class TestModelCheckpointTopK:
    def test_save_top_k_enforced_on_disk(self):
        with tempfile.TemporaryDirectory() as d:
            class _Worsening(_Net):
                def validation_step(self, batch, batch_idx):
                    # val_loss strictly increases with epoch -> later epochs worse.
                    loss = jnp.asarray(float(self.current_epoch + 1))
                    self.log('val_loss', loss)
                    return {'val_loss': loss}

            mc = ModelCheckpoint(
                dirpath=d, monitor='val_loss', mode='min', save_top_k=2,
                save_last=False, filename='ck-{epoch:02d}',
            )
            trainer = Trainer(
                max_epochs=6, logger=False, enable_progress_bar=False,
                enable_checkpointing=False, callbacks=[mc],
                check_val_every_n_epoch=1,
            )
            trainer.fit(_Worsening(), _loader(), _loader())

            files = sorted(f for f in os.listdir(d) if f.endswith('.ckpt'))
            assert len(files) == 2, f"save_top_k=2 but found {files}"
            # The two best (lowest val_loss) are epochs 0 and 1.
            assert files == ['ck-00.ckpt', 'ck-01.ckpt']
            assert len(mc.best_k_models) == 2


# ---------------------------------------------------------------------------
# T-C: resuming from a checkpoint whose step is None must not crash
# ---------------------------------------------------------------------------

class TestResumeStepNone:
    def test_resume_with_step_none(self):
        with tempfile.TemporaryDirectory() as d:
            cm = CheckpointManager(d, verbose=False)
            path = cm.save(_Net(), epoch=0, step=None, metrics={'val_loss': 0.5})

            trainer = Trainer(
                max_epochs=1, logger=False, enable_progress_bar=False,
                enable_checkpointing=False,
            )
            # Must not raise; global_step coerced to an int.
            trainer.fit(_Net(), _loader(), ckpt_path=path)
            assert isinstance(trainer.global_step, int)
            assert trainer.global_step >= 0


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
