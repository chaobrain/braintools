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

"""Tests for the checkpointing utilities."""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainstate
import braintools
from braintools.trainer._checkpoint import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    find_checkpoint,
    list_checkpoints,
)


class SimpleModel(braintools.trainer.LightningModule):
    """Simple model for testing checkpointing."""

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
        return {'loss': loss}

    def configure_optimizers(self):
        return braintools.optim.Adam(lr=1e-3)


def _leaves_close(a, b):
    """Check that two pytrees have matching leaves."""
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    return all(jnp.allclose(x, y) for x, y in zip(la, lb))


class TestSaveLoadCheckpoint:
    """Tests for the functional save/load helpers."""

    def test_round_trip_scalars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'ckpt.ckpt')
            state = {'epoch': 7, 'loss': 0.25, 'name': 'run'}
            returned = save_checkpoint(state, filepath)
            assert returned == filepath
            assert os.path.exists(filepath)

            loaded = load_checkpoint(filepath)
            assert loaded['epoch'] == 7
            assert loaded['loss'] == 0.25
            assert loaded['name'] == 'run'

    def test_round_trip_arrays_converted_to_jax(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'arr.ckpt')
            state = {'w': jnp.arange(6.0).reshape(2, 3)}
            save_checkpoint(state, filepath)

            loaded = load_checkpoint(filepath)
            # numpy arrays should be converted back to JAX arrays on load
            assert isinstance(loaded['w'], jax.Array)
            assert jnp.allclose(loaded['w'], state['w'])

    def test_save_creates_nested_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'sub', 'dir', 'ckpt.ckpt')
            save_checkpoint({'a': 1}, filepath)
            assert os.path.exists(filepath)

    def test_save_no_overwrite_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'ckpt.ckpt')
            save_checkpoint({'a': 1}, filepath)
            with pytest.raises(FileExistsError):
                save_checkpoint({'a': 2}, filepath, overwrite=False)

    def test_save_overwrite_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'ckpt.ckpt')
            save_checkpoint({'a': 1}, filepath)
            save_checkpoint({'a': 99}, filepath, overwrite=True)
            loaded = load_checkpoint(filepath)
            assert loaded['a'] == 99

    def test_load_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_checkpoint(os.path.join(tmpdir, 'missing.ckpt'))

    def test_save_numpy_array_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'np.ckpt')
            state = {'w': np.ones((3, 3))}
            save_checkpoint(state, filepath)
            loaded = load_checkpoint(filepath)
            assert jnp.allclose(loaded['w'], jnp.ones((3, 3)))


class TestListCheckpoints:
    """Tests for list_checkpoints."""

    def _touch(self, path, content=b'x'):
        with open(path, 'wb') as f:
            f.write(content)

    def test_missing_dir_returns_empty(self):
        assert list_checkpoints('/nonexistent/path/xyz') == []

    def test_list_sort_by_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for n in ['b.ckpt', 'a.ckpt', 'c.ckpt']:
                self._touch(os.path.join(tmpdir, n))
            result = list_checkpoints(tmpdir, sort_by='name')
            names = [os.path.basename(p) for p in result]
            assert names == ['a.ckpt', 'b.ckpt', 'c.ckpt']

    def test_list_sort_by_time(self):
        import time
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for n in ['first.ckpt', 'second.ckpt', 'third.ckpt']:
                p = os.path.join(tmpdir, n)
                self._touch(p)
                paths.append(p)
                time.sleep(0.01)
            result = list_checkpoints(tmpdir, sort_by='time')
            assert [os.path.basename(p) for p in result] == \
                   ['first.ckpt', 'second.ckpt', 'third.ckpt']

    def test_list_sort_by_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for n in ['model-epoch=10.ckpt', 'model-epoch=2.ckpt', 'model-epoch=5.ckpt']:
                self._touch(os.path.join(tmpdir, n))
            result = list_checkpoints(tmpdir, sort_by='epoch')
            names = [os.path.basename(p) for p in result]
            assert names == ['model-epoch=2.ckpt', 'model-epoch=5.ckpt', 'model-epoch=10.ckpt']

    def test_list_sort_by_epoch_no_match_defaults_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for n in ['noepoch.ckpt', 'alsonone.ckpt']:
                self._touch(os.path.join(tmpdir, n))
            # No epoch in names -> all extract to 0, stable order preserved (no crash)
            result = list_checkpoints(tmpdir, sort_by='epoch')
            assert len(result) == 2

    def test_list_recursive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = os.path.join(tmpdir, 'nested')
            os.makedirs(sub)
            self._touch(os.path.join(tmpdir, 'top.ckpt'))
            self._touch(os.path.join(sub, 'inner.ckpt'))
            result = list_checkpoints(tmpdir, sort_by='name')
            assert len(result) == 2

    def test_list_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._touch(os.path.join(tmpdir, 'a.ckpt'))
            self._touch(os.path.join(tmpdir, 'b.pt'))
            result = list_checkpoints(tmpdir, pattern='*.pt')
            assert len(result) == 1
            assert os.path.basename(result[0]) == 'b.pt'


class TestFindCheckpoint:
    """Tests for find_checkpoint."""

    def _touch(self, path):
        with open(path, 'wb') as f:
            f.write(b'x')

    def test_missing_dir_returns_none(self):
        assert find_checkpoint('/nonexistent/xyz') is None

    def test_empty_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert find_checkpoint(tmpdir) is None

    def test_find_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._touch(os.path.join(tmpdir, 'a.ckpt'))
            found = find_checkpoint(tmpdir)
            assert found is not None
            assert found.endswith('a.ckpt')

    def test_find_best_by_name(self):
        import time
        with tempfile.TemporaryDirectory() as tmpdir:
            self._touch(os.path.join(tmpdir, 'normal.ckpt'))
            time.sleep(0.01)
            self._touch(os.path.join(tmpdir, 'best-model.ckpt'))
            found = find_checkpoint(tmpdir, best=True)
            assert os.path.basename(found) == 'best-model.ckpt'

    def test_find_best_falls_back_to_most_recent(self):
        import time
        with tempfile.TemporaryDirectory() as tmpdir:
            self._touch(os.path.join(tmpdir, 'one.ckpt'))
            time.sleep(0.01)
            self._touch(os.path.join(tmpdir, 'two.ckpt'))
            found = find_checkpoint(tmpdir, best=True)
            # No 'best' in any name -> fall back to most recent (time-sorted last)
            assert os.path.basename(found) == 'two.ckpt'

    def test_find_last_by_name(self):
        import time
        with tempfile.TemporaryDirectory() as tmpdir:
            self._touch(os.path.join(tmpdir, 'epoch1.ckpt'))
            time.sleep(0.01)
            self._touch(os.path.join(tmpdir, 'last.ckpt'))
            time.sleep(0.01)
            self._touch(os.path.join(tmpdir, 'epoch2.ckpt'))
            found = find_checkpoint(tmpdir, last=True)
            assert os.path.basename(found) == 'last.ckpt'

    def test_find_last_falls_back_to_most_recent(self):
        import time
        with tempfile.TemporaryDirectory() as tmpdir:
            self._touch(os.path.join(tmpdir, 'a.ckpt'))
            time.sleep(0.01)
            self._touch(os.path.join(tmpdir, 'b.ckpt'))
            found = find_checkpoint(tmpdir, last=True)
            assert os.path.basename(found) == 'b.ckpt'


class TestCheckpointManagerInit:
    """Tests for CheckpointManager construction."""

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, 'ckpts')
            CheckpointManager(dirpath=target, verbose=False)
            assert os.path.isdir(target)

    def test_invalid_mode_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                CheckpointManager(dirpath=tmpdir, mode='invalid', verbose=False)

    def test_default_properties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, verbose=False)
            assert mgr.latest is None
            assert mgr.best is None
            assert mgr.best_score is None
            assert mgr.checkpoints == []

    def test_is_better_min_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, mode='min', verbose=False)
            assert mgr._is_better(0.1, 0.2) is True
            assert mgr._is_better(0.3, 0.2) is False

    def test_is_better_max_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, mode='max', verbose=False)
            assert mgr._is_better(0.3, 0.2) is True
            assert mgr._is_better(0.1, 0.2) is False


class TestCheckpointManagerSaveLoad:
    """Tests for CheckpointManager save/load."""

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, verbose=False)
            model = SimpleModel()
            path = mgr.save(model, epoch=3, step=12, metrics={'val_loss': 0.5})
            assert path is not None
            assert os.path.exists(path)

            # Load back into a fresh model
            model2 = SimpleModel()
            state = mgr.load(path, model=model2)
            assert state['epoch'] == 3
            assert state['step'] == 12
            assert _leaves_close(model.state_dict(), model2.state_dict())

    def test_save_with_optimizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, verbose=False)
            model = SimpleModel()
            opt = braintools.optim.Adam(lr=1e-3)
            opt.register_trainable_weights(model.states(brainstate.ParamState))
            path = mgr.save(model, optimizer=opt, epoch=1)
            state = load_checkpoint(path)
            assert 'optimizer_state_dict' in state

    def test_save_and_load_optimizer_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, verbose=False)
            model = SimpleModel()
            opt = braintools.optim.Adam(lr=1e-3)
            opt.register_trainable_weights(model.states(brainstate.ParamState))
            path = mgr.save(model, optimizer=opt, epoch=1)

            # Load optimizer state into a fresh, registered optimizer.
            model2 = SimpleModel()
            opt2 = braintools.optim.Adam(lr=1e-3)
            opt2.register_trainable_weights(model2.states(brainstate.ParamState))
            state = mgr.load(path, model=model2, optimizer=opt2)
            assert 'optimizer_state_dict' in state

    def test_save_with_extra_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, verbose=False)
            model = SimpleModel()
            path = mgr.save(model, epoch=0, extra_state={'rng_seed': 42})
            state = load_checkpoint(path)
            assert state['rng_seed'] == 42

    def test_save_model_without_state_dict(self):
        # A plain brainstate module has no state_dict -> exercises the
        # ParamState extraction branch.
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, verbose=False)
            model = brainstate.nn.Linear(4, 3)
            assert not hasattr(model, 'state_dict')
            path = mgr.save(model, epoch=0)
            state = load_checkpoint(path)
            assert 'model_state_dict' in state

    def test_load_into_model_without_load_state_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, verbose=False)
            src = brainstate.nn.Linear(4, 3)
            path = mgr.save(src, epoch=0)
            dst = brainstate.nn.Linear(4, 3)
            mgr.load(path, model=dst)
            # Weights should be copied across
            assert _leaves_close(
                {k: v.value for k, v in src.states(brainstate.ParamState).items()},
                {k: v.value for k, v in dst.states(brainstate.ParamState).items()},
            )

    def test_latest_property_tracks_saves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, max_to_keep=-1, verbose=False)
            model = SimpleModel()
            p0 = mgr.save(model, epoch=0)
            p1 = mgr.save(model, epoch=1)
            assert mgr.latest == p1
            assert p0 != p1

    def test_load_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, max_to_keep=-1, verbose=False)
            model = SimpleModel()
            mgr.save(model, epoch=0)
            mgr.save(model, epoch=5)
            state = mgr.load_latest()
            assert state['epoch'] == 5

    def test_load_none_with_no_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, verbose=False)
            with pytest.raises(FileNotFoundError):
                mgr.load()

    def test_latest_searches_directory_when_no_tracked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save with one manager
            mgr1 = CheckpointManager(dirpath=tmpdir, verbose=False)
            model = SimpleModel()
            mgr1.save(model, epoch=2)
            # Fresh manager has no tracked checkpoints but should find on disk
            mgr2 = CheckpointManager(dirpath=tmpdir, verbose=False)
            assert mgr2.latest is not None


class TestCheckpointManagerBest:
    """Tests for best-checkpoint tracking."""

    def test_best_tracking_min_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                dirpath=tmpdir, max_to_keep=-1, monitor='val_loss',
                mode='min', verbose=False,
            )
            model = SimpleModel()
            mgr.save(model, epoch=0, metrics={'val_loss': 1.0})
            assert mgr.best_score == 1.0
            best_path = mgr.best
            mgr.save(model, epoch=1, metrics={'val_loss': 0.5})
            assert mgr.best_score == 0.5
            assert mgr.best != best_path
            # Worse metric does not update best
            mgr.save(model, epoch=2, metrics={'val_loss': 0.8})
            assert mgr.best_score == 0.5

    def test_load_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                dirpath=tmpdir, max_to_keep=-1, monitor='val_loss',
                mode='min', verbose=False,
            )
            model = SimpleModel()
            mgr.save(model, epoch=0, metrics={'val_loss': 1.0})
            mgr.save(model, epoch=1, metrics={'val_loss': 0.3})
            state = mgr.load_best()
            assert state['epoch'] == 1

    def test_load_best_no_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, verbose=False)
            with pytest.raises(FileNotFoundError):
                mgr.load_best()

    def test_load_best_discovers_from_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Manually place a checkpoint with 'best' in its name
            best_state = {'epoch': 9, 'model_state_dict': {}}
            save_checkpoint(best_state, os.path.join(tmpdir, 'best.ckpt'))
            mgr = CheckpointManager(dirpath=tmpdir, verbose=False)
            # No tracked best; load_best should discover via find_checkpoint
            state = mgr.load_best()
            assert state['epoch'] == 9

    def test_save_best_as(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                dirpath=tmpdir, monitor='val_loss', mode='min', verbose=False,
            )
            model = SimpleModel()
            mgr.save(model, epoch=0, metrics={'val_loss': 0.5})
            dest = os.path.join(tmpdir, 'exported_best.ckpt')
            mgr.save_best_as(dest)
            assert os.path.exists(dest)

    def test_save_best_as_no_best_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, verbose=False)
            with pytest.raises(FileNotFoundError):
                mgr.save_best_as(os.path.join(tmpdir, 'out.ckpt'))


class TestCheckpointManagerSaveBestOnly:
    """Tests for save_best_only behavior."""

    def test_skips_non_improving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                dirpath=tmpdir, save_best_only=True, monitor='val_loss',
                mode='min', verbose=False,
            )
            model = SimpleModel()
            p0 = mgr.save(model, epoch=0, metrics={'val_loss': 1.0})
            assert p0 is not None
            # Worse metric should be skipped (returns None)
            p1 = mgr.save(model, epoch=1, metrics={'val_loss': 2.0})
            assert p1 is None

    def test_saves_on_improvement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                dirpath=tmpdir, save_best_only=True, monitor='val_loss',
                mode='min', verbose=False,
            )
            model = SimpleModel()
            mgr.save(model, epoch=0, metrics={'val_loss': 1.0})
            p1 = mgr.save(model, epoch=1, metrics={'val_loss': 0.4})
            assert p1 is not None

    def test_monitor_metric_missing_still_saves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                dirpath=tmpdir, save_best_only=True, monitor='val_loss',
                mode='min', verbose=False,
            )
            model = SimpleModel()
            # Metric not present -> warning path, but still saves
            path = mgr.save(model, epoch=0, metrics={'other': 0.1})
            assert path is not None


class TestCheckpointManagerCleanup:
    """Tests for old-checkpoint cleanup."""

    def test_cleanup_respects_max_to_keep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, max_to_keep=2, verbose=False)
            model = SimpleModel()
            for e in range(5):
                mgr.save(model, epoch=e)
            assert len(mgr.checkpoints) == 2
            # Only 2 files should remain on disk
            assert len(list_checkpoints(tmpdir)) == 2

    def test_cleanup_keeps_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                dirpath=tmpdir, max_to_keep=2, monitor='val_loss',
                mode='min', verbose=False,
            )
            model = SimpleModel()
            # Epoch 0 has the best (lowest) metric
            best_path = mgr.save(model, epoch=0, metrics={'val_loss': 0.01})
            for e in range(1, 5):
                mgr.save(model, epoch=e, metrics={'val_loss': 1.0 + e})
            # Best checkpoint must not be deleted even though it's oldest
            assert os.path.exists(best_path)
            assert mgr.best == best_path

    def test_max_to_keep_negative_keeps_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, max_to_keep=-1, verbose=False)
            model = SimpleModel()
            for e in range(4):
                mgr.save(model, epoch=e)
            assert len(mgr.checkpoints) == 4

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, max_to_keep=-1, verbose=False)
            model = SimpleModel()
            mgr.save(model, epoch=0)
            mgr.save(model, epoch=1)
            assert len(mgr.checkpoints) == 2
            mgr.clear()
            assert mgr.checkpoints == []
            assert mgr.best is None
            assert mgr.best_score is None
            assert list_checkpoints(tmpdir) == []


class TestFilenameFormatting:
    """Tests for filename template formatting."""

    def test_default_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, verbose=False)
            name = mgr._format_filename(epoch=3)
            assert name == 'checkpoint-epoch=0003'

    def test_template_with_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                dirpath=tmpdir,
                filename_template='ckpt-{epoch}-{val_loss:.2f}',
                verbose=False,
            )
            name = mgr._format_filename(epoch=2, metrics={'val_loss': 0.123})
            assert name == 'ckpt-2-0.12'

    def test_template_bad_key_falls_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                dirpath=tmpdir,
                filename_template='ckpt-{nonexistent}',
                verbose=False,
            )
            name = mgr._format_filename(epoch=4)
            assert name == 'checkpoint-epoch=0004'

    def test_template_metric_name_cleaned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                dirpath=tmpdir,
                filename_template='ckpt-{val_loss}',
                verbose=False,
            )
            # The 'val/loss' key gets cleaned to 'val_loss' in format_dict
            name = mgr._format_filename(epoch=0, metrics={'val/loss': 0.5})
            assert name == 'ckpt-0.5'


class TestVerboseOutput:
    """Tests that exercise the verbose=True print branches end-to-end."""

    def test_verbose_full_lifecycle(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                dirpath=tmpdir, max_to_keep=1, monitor='val_loss',
                mode='min', verbose=True,
            )
            model = SimpleModel()
            # New best + saved messages
            mgr.save(model, epoch=0, metrics={'val_loss': 1.0})
            # New best again + triggers cleanup of the previous (non-best) file
            mgr.save(model, epoch=1, metrics={'val_loss': 0.5})
            # Loading prints a message
            mgr.load_latest()
            # Copy best prints a message
            mgr.save_best_as(os.path.join(tmpdir, 'exported.ckpt'))
            # Clear prints a message
            mgr.clear()
            out = capsys.readouterr().out
            assert 'New best checkpoint' in out
            assert 'Saved checkpoint' in out
            assert 'Loaded checkpoint' in out
            assert 'Copied best checkpoint' in out
            assert 'Cleared all checkpoints' in out

    def test_verbose_skip_and_missing_metric(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                dirpath=tmpdir, save_best_only=True, monitor='val_loss',
                mode='min', verbose=True,
            )
            model = SimpleModel()
            mgr.save(model, epoch=0, metrics={'val_loss': 0.5})
            # Non-improving -> "Skipping checkpoint" message
            mgr.save(model, epoch=1, metrics={'val_loss': 0.9})
            # Monitor metric absent -> warning message
            mgr.save(model, epoch=2, metrics={'other': 0.1})
            out = capsys.readouterr().out
            assert 'Skipping checkpoint' in out
            assert 'not found in metrics' in out

    def test_verbose_cleanup_removes_old(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(dirpath=tmpdir, max_to_keep=1, verbose=True)
            model = SimpleModel()
            mgr.save(model, epoch=0)
            mgr.save(model, epoch=1)
            out = capsys.readouterr().out
            assert 'Removed old checkpoint' in out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
