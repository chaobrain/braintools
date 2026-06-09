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

"""Tests for the progress bar utilities module."""

import builtins
import time

import pytest

from braintools.trainer._progress import (
    ProgressBar,
    SimpleProgressBar,
    TQDMProgressBarWrapper,
    RichProgressBarWrapper,
    ProgressBarPool,
    MetricsDisplay,
    get_progress_bar,
)


def _block_import(monkeypatch, *blocked_prefixes):
    """Force ImportError for the given top-level module names."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        top = name.split('.')[0]
        if top in blocked_prefixes:
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)


class TestProgressBarABC:
    """Tests for the abstract ProgressBar base class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ProgressBar()

    def test_context_manager(self):
        # SimpleProgressBar is concrete and inherits __enter__/__exit__.
        with SimpleProgressBar() as pbar:
            pbar.start(total=5, desc='ctx')
            pbar.update(1)
            assert isinstance(pbar, ProgressBar)


class TestSimpleProgressBar:
    """Tests for SimpleProgressBar."""

    def test_basic_with_total(self):
        pbar = SimpleProgressBar(refresh_rate=0.0)
        pbar.start(total=10, desc='Train', unit='step')
        for _ in range(10):
            pbar.update(1, loss=0.5)
        pbar.close()
        assert pbar._n == 10
        assert pbar._desc == 'Train'

    def test_postfix_float_and_other(self):
        pbar = SimpleProgressBar(refresh_rate=0.0)
        pbar.start(total=4)
        pbar.set_postfix({'loss': 0.1234, 'epoch': 3, 'name': 'abc'})
        assert pbar._postfix['loss'] == 0.1234
        assert pbar._postfix['epoch'] == 3
        pbar.close()

    def test_update_accumulates_postfix(self):
        pbar = SimpleProgressBar(refresh_rate=0.0)
        pbar.start(total=10)
        pbar.update(2, loss=1.0)
        pbar.update(3, acc=0.9)
        assert pbar._n == 5
        assert 'loss' in pbar._postfix
        assert 'acc' in pbar._postfix
        pbar.close()

    def test_no_total(self):
        # Branch where _total is None.
        pbar = SimpleProgressBar(refresh_rate=0.0)
        pbar.start(total=None, desc='stream', unit='samples')
        pbar.update(1)
        pbar.update(1)
        pbar.close()
        assert pbar._n == 2

    def test_refresh_rate_skips_print(self):
        # With a high refresh rate, intermediate updates do not print but still count.
        pbar = SimpleProgressBar(refresh_rate=1000.0)
        pbar.start(total=100)
        for _ in range(5):
            pbar.update(1)
        assert pbar._n == 5
        pbar.close()

    def test_format_time_seconds(self):
        pbar = SimpleProgressBar()
        assert pbar._format_time(12.34).endswith('s')

    def test_format_time_minutes(self):
        pbar = SimpleProgressBar()
        out = pbar._format_time(125.0)
        assert ':' in out and out == '2:05'

    def test_format_time_hours(self):
        pbar = SimpleProgressBar()
        out = pbar._format_time(3725.0)
        assert out.startswith('1:')

    def test_print_bar_with_elapsed_time(self):
        # Exercise the rate/remaining time branch in _print_bar.
        pbar = SimpleProgressBar(refresh_rate=0.0)
        pbar.start(total=10)
        time.sleep(0.01)
        pbar.update(5, loss=0.5)
        pbar.close()
        assert pbar._n == 5

    def test_zero_total_branch(self):
        # total == 0 exercises the "else" branch where percent stays 0.
        pbar = SimpleProgressBar(refresh_rate=0.0)
        pbar.start(total=0, desc='empty')
        pbar.update(1)
        pbar.close()


class TestTQDMProgressBarWrapper:
    """Tests for TQDMProgressBarWrapper (tqdm is installed)."""

    def test_full_lifecycle(self):
        pbar = TQDMProgressBarWrapper(leave=False)
        pbar.start(total=10, desc='Training', unit='it')
        assert pbar._pbar is not None
        for _ in range(10):
            pbar.update(1, loss=0.5)
        pbar.set_postfix({'acc': 0.95})
        pbar.close()
        assert pbar._pbar is None

    def test_update_without_kwargs(self):
        pbar = TQDMProgressBarWrapper(leave=False)
        pbar.start(total=5)
        pbar.update(1)
        pbar.update(2)
        pbar.close()

    def test_set_postfix_standalone(self):
        pbar = TQDMProgressBarWrapper(leave=False)
        pbar.start(total=5, desc='m')
        pbar.set_postfix({'loss': 0.1})
        pbar.close()

    def test_close_when_not_started(self):
        pbar = TQDMProgressBarWrapper()
        # _pbar is None, close should be a no-op.
        pbar.close()
        # update/set_postfix on un-started bar should also be safe no-ops.
        pbar.update(1, loss=0.5)
        pbar.set_postfix({'loss': 0.5})

    def test_fallback_to_simple_when_tqdm_missing(self, monkeypatch):
        # When tqdm import fails, start() falls back to SimpleProgressBar.
        _block_import(monkeypatch, 'tqdm')
        pbar = TQDMProgressBarWrapper()
        pbar.start(total=5, desc='fallback', unit='it')
        from braintools.trainer._progress import SimpleProgressBar as _SPB
        assert isinstance(pbar._pbar, _SPB)
        pbar.update(1, loss=0.5)
        pbar.set_postfix({'loss': 0.5})
        pbar.close()


class TestRichProgressBarWrapper:
    """Tests for RichProgressBarWrapper (rich is installed)."""

    def test_full_lifecycle(self):
        pbar = RichProgressBarWrapper()
        pbar.start(total=10, desc='Training')
        assert pbar._progress is not None
        assert pbar._task_id is not None
        for _ in range(10):
            pbar.update(1)
        pbar.close()
        assert pbar._progress is None
        assert pbar._task_id is None

    def test_set_postfix_updates_description(self):
        pbar = RichProgressBarWrapper()
        pbar.start(total=5, desc='m')
        pbar.set_postfix({'loss': 0.1234, 'epoch': 2})
        pbar.update(1)
        pbar.close()

    def test_close_when_not_started(self):
        pbar = RichProgressBarWrapper()
        pbar.close()  # _progress is None: no-op
        # update/set_postfix on un-started bar are safe no-ops.
        pbar.update(1)
        pbar.set_postfix({'loss': 0.5})

    def test_fallback_to_simple_when_rich_missing(self, monkeypatch):
        # When rich import fails, start() falls back to SimpleProgressBar and
        # _task_id stays None, exercising the non-rich update/close branches.
        _block_import(monkeypatch, 'rich')
        pbar = RichProgressBarWrapper()
        pbar.start(total=5, desc='fallback', unit='it')
        from braintools.trainer._progress import SimpleProgressBar as _SPB
        assert isinstance(pbar._progress, _SPB)
        assert pbar._task_id is None
        # update() with _task_id None routes to the SimpleProgressBar.update path.
        pbar.update(2, loss=0.5)
        # set_postfix is a no-op when _task_id is None.
        pbar.set_postfix({'loss': 0.5})
        # close() hits the elif hasattr(..., 'close') branch (SimpleProgressBar
        # has close() but no stop()).
        pbar.close()
        assert pbar._progress is None


class TestGetProgressBar:
    """Tests for the get_progress_bar factory."""

    def test_simple(self):
        pbar = get_progress_bar('simple')
        assert isinstance(pbar, SimpleProgressBar)

    def test_tqdm(self):
        pbar = get_progress_bar('tqdm')
        assert isinstance(pbar, TQDMProgressBarWrapper)

    def test_rich(self):
        pbar = get_progress_bar('rich')
        assert isinstance(pbar, RichProgressBarWrapper)

    def test_auto_prefers_rich(self):
        # rich is installed, so auto should resolve to the rich wrapper.
        pbar = get_progress_bar('auto')
        assert isinstance(pbar, RichProgressBarWrapper)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            get_progress_bar('nonexistent')

    def test_kwargs_passed_to_simple(self):
        pbar = get_progress_bar('simple', width=20)
        assert pbar.width == 20

    def test_auto_falls_back_to_tqdm_without_rich(self, monkeypatch):
        # rich unavailable -> auto picks tqdm.
        _block_import(monkeypatch, 'rich')
        pbar = get_progress_bar('auto')
        assert isinstance(pbar, TQDMProgressBarWrapper)

    def test_auto_falls_back_to_simple_without_rich_tqdm(self, monkeypatch):
        # Neither rich nor tqdm available -> auto picks simple.
        _block_import(monkeypatch, 'rich', 'tqdm')
        pbar = get_progress_bar('auto')
        assert isinstance(pbar, SimpleProgressBar)


class TestProgressBarPool:
    """Tests for ProgressBarPool."""

    def test_training_validation_testing_epoch(self):
        pool = ProgressBarPool(backend='simple')
        for getter, total in [
            (pool.training, 10),
            (pool.validation, 5),
            (pool.testing, 3),
            (pool.epoch, 2),
        ]:
            pbar = getter(total=total)
            assert isinstance(pbar, SimpleProgressBar)
            pbar.update(1)
        pool.close_all()
        assert pool._bars == {}

    def test_bar_reuse(self):
        pool = ProgressBarPool(backend='simple')
        b1 = pool.training(total=10)
        b2 = pool.training(total=20)
        # Same named bar is reused.
        assert b1 is b2
        pool.close_all()

    def test_default_descriptions(self):
        pool = ProgressBarPool(backend='simple')
        t = pool.training(total=1)
        assert t._desc == 'Training'
        v = pool.validation(total=1)
        assert v._desc == 'Validation'
        pool.close_all()


class TestMetricsDisplay:
    """Tests for MetricsDisplay."""

    def test_format_metric_float_and_other(self):
        display = MetricsDisplay()
        assert display.format_metric('loss', 0.5).startswith('loss:')
        assert display.format_metric('name', 'abc') == 'name: abc'

    def test_format_metrics(self):
        display = MetricsDisplay()
        out = display.format_metrics({'loss': 0.5, 'acc': 0.9})
        assert 'loss' in out and 'acc' in out and ',' in out

    def test_custom_format_spec(self):
        display = MetricsDisplay(format_spec='.2f')
        assert display.format_metric('loss', 0.12345) == 'loss: 0.12'

    def test_print_epoch_summary(self, capsys):
        display = MetricsDisplay(max_width=20)
        display.print_epoch_summary(
            5,
            train_metrics={'loss': 0.5},
            val_metrics={'val_loss': 0.4},
        )
        out = capsys.readouterr().out
        assert 'Epoch 5 Summary' in out
        assert 'Train:' in out
        assert 'Val:' in out

    def test_print_epoch_summary_no_metrics(self, capsys):
        display = MetricsDisplay(max_width=20)
        display.print_epoch_summary(1)
        out = capsys.readouterr().out
        assert 'Epoch 1 Summary' in out
        assert 'Train:' not in out

    def test_print_training_start(self, capsys):
        display = MetricsDisplay(max_width=20)
        display.print_training_start(model_name='Net', num_params=1234, max_epochs=10)
        out = capsys.readouterr().out
        assert 'Starting Training: Net' in out
        assert '1,234' in out
        assert 'Max Epochs: 10' in out

    def test_print_training_start_minimal(self, capsys):
        display = MetricsDisplay(max_width=20)
        display.print_training_start()
        out = capsys.readouterr().out
        assert 'Starting Training: Model' in out
        assert 'Parameters' not in out

    def test_print_training_end(self, capsys):
        display = MetricsDisplay(max_width=20)
        display.print_training_end(best_metrics={'acc': 0.99}, total_time=12.5)
        out = capsys.readouterr().out
        assert 'Training Complete!' in out
        assert 'Best:' in out
        assert 'Total Time: 12.5s' in out

    def test_print_training_end_minimal(self, capsys):
        display = MetricsDisplay(max_width=20)
        display.print_training_end()
        out = capsys.readouterr().out
        assert 'Training Complete!' in out
        assert 'Best:' not in out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
