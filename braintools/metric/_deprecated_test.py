# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Tests for the deprecated ``braintools.metric`` aliases of moved functions."""

import pickle
import warnings

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintools
import braintools.analysis
import braintools.metric
from braintools._deprecation import BraintoolsDeprecationWarning, registered_deprecations
from braintools.metric import _deprecated

_rng = np.random.default_rng(0)
_SPIKES = (_rng.random((40, 4)) < 0.3).astype(float)
_TRAIN_1 = np.array([1.0, 2.0, 3.0])
_TRAIN_2 = np.array([1.1, 2.5])
_T = np.arange(256) * 0.001
_LFP = np.sin(2 * np.pi * 10 * _T) + 0.3 * _rng.standard_normal(256)
_LFP_2 = np.sin(2 * np.pi * 10 * _T + 0.5) + 0.3 * _rng.standard_normal(256)
_LFP_MULTI = np.stack([_LFP, _LFP_2, _rng.standard_normal(256)], axis=1)
_VOLTAGES = _rng.standard_normal((60, 4))
_LAMINAR = _rng.standard_normal((10, 5))
_MAT_A = _rng.standard_normal((4, 4))
_MAT_B = _rng.standard_normal((4, 4))
_X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
_Y = np.array([2.0, 1.0, 4.0, 3.0, 5.0])
_W = np.array([1.0, 1.0, 2.0, 2.0, 1.0])

# (old name, new subpackage, positional args, keyword args)
CASES = [
    ('firing_rate', 'spike', (_SPIKES, 3.0 * u.ms, 1.0 * u.ms), {}),
    ('victor_purpura_distance', 'spike', (_TRAIN_1, _TRAIN_2), {'cost_factor': 2.0}),
    ('van_rossum_distance', 'spike', (_TRAIN_1, _TRAIN_2), {'tau': 0.5}),
    ('spike_train_synchrony', 'spike', (_SPIKES,), {'window_size': 4.0, 'dt': 1.0}),
    ('burst_synchrony_index', 'spike', (_SPIKES,), {'burst_threshold': 2, 'max_isi': 2.0, 'dt': 1.0}),
    ('phase_locking_value', 'spike', (_SPIKES, 10.0, 0.001), {}),
    ('spike_time_tiling_coefficient', 'spike', (_SPIKES,), {'dt': 1.0, 'tau': 2.0}),
    ('correlation_index', 'spike', (_SPIKES,), {'window_size': 4.0, 'dt': 1.0}),
    ('cross_correlation', 'spike', (_SPIKES,), {'bin': 2.0, 'dt': 1.0}),
    ('power_spectral_density', 'spectral', (_LFP, 0.001), {}),
    ('coherence_analysis', 'spectral', (_LFP, _LFP_2, 0.001), {}),
    ('phase_amplitude_coupling', 'spectral', (_LFP, 0.001), {}),
    ('theta_gamma_coupling', 'spectral', (_LFP, 0.001), {}),
    ('spectral_entropy', 'spectral', (_LFP, 0.001), {}),
    ('lfp_phase_coherence', 'spectral', (_LFP_MULTI, 0.001), {}),
    ('unitary_LFP', 'field', (np.arange(40) * 0.5, _SPIKES, 'exc'), {'seed': 1}),
    ('current_source_density', 'field', (_LAMINAR, 0.1), {}),
    ('voltage_fluctuation', 'population', (_VOLTAGES,), {}),
    ('functional_connectivity', 'population', (_VOLTAGES,), {}),
    ('functional_connectivity_dynamics', 'population', (_VOLTAGES,), {'window_size': 20, 'step_size': 10}),
    ('matrix_correlation', 'stats', (_MAT_A, _MAT_B), {}),
    ('weighted_correlation', 'stats', (_X, _Y, _W), {}),
]
MOVED = {name: sub for name, sub, _, _ in CASES}

# Order of the deprecated names in ``braintools.metric.__all__`` before 0.4.
ORIGINAL_ORDER = [
    'cross_correlation', 'voltage_fluctuation', 'matrix_correlation', 'weighted_correlation',
    'functional_connectivity', 'functional_connectivity_dynamics',
    'raster_plot', 'firing_rate', 'victor_purpura_distance', 'van_rossum_distance',
    'spike_train_synchrony', 'burst_synchrony_index', 'phase_locking_value',
    'spike_time_tiling_coefficient', 'correlation_index',
    'unitary_LFP', 'power_spectral_density', 'coherence_analysis', 'phase_amplitude_coupling',
    'theta_gamma_coupling', 'current_source_density', 'spectral_entropy', 'lfp_phase_coherence',
]


def _new(name):
    return getattr(getattr(braintools.analysis, MOVED[name]), name)


def _assert_identical(a, b):
    leaves_a = jax.tree.leaves(a, is_leaf=u.math.is_quantity)
    leaves_b = jax.tree.leaves(b, is_leaf=u.math.is_quantity)
    assert len(leaves_a) == len(leaves_b)
    for x, y in zip(leaves_a, leaves_b):
        assert type(x) is type(y)
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_case_table_covers_every_moved_function():
    assert len(CASES) == 22
    for sub in ('spike', 'spectral', 'field', 'population', 'stats'):
        # Every moved function is exported from its new subpackage (which may
        # also hold functions added after the move).
        exported = set(getattr(braintools.analysis, sub).__all__)
        assert {n for n, s in MOVED.items() if s == sub} <= exported, sub
    # The spec table moves 22 functions; with ``raster_plot`` that is 23 aliases.
    assert set(_deprecated.__all__) == set(MOVED) | {'raster_plot'}
    assert len(_deprecated.__all__) == 23


@pytest.mark.parametrize('name, sub, args, kwargs', CASES, ids=[c[0] for c in CASES])
def test_alias_warns_and_matches_new_function(name, sub, args, kwargs):
    alias = getattr(braintools.metric, name)
    message = (rf'`braintools\.metric\.{name}` is deprecated since 0\.4\.0 and will be removed '
               rf'in 1\.0\.0\. Use `braintools\.analysis\.{sub}\.{name}` instead\.')
    with pytest.warns(BraintoolsDeprecationWarning, match=message) as record:
        old = alias(*args, **kwargs)
    assert record[0].filename == __file__
    new = getattr(getattr(braintools.analysis, sub), name)(*args, **kwargs)
    _assert_identical(old, new)


@pytest.mark.parametrize('name', sorted(MOVED))
def test_alias_metadata(name):
    sub = MOVED[name]
    alias = getattr(braintools.metric, name)
    new = _new(name)
    assert alias is not new
    assert alias.__wrapped__ is new
    assert alias.__name__ == name
    assert alias.__module__ == 'braintools.metric'
    assert alias.__doc__ == new.__doc__
    assert f'braintools.analysis.{sub}.{name}' in alias.__deprecated__
    assert new.__module__ == f'braintools.analysis.{sub}'
    assert name in braintools.metric.__all__


@pytest.mark.parametrize('name', sorted(MOVED))
def test_new_docstring_examples_use_new_paths(name):
    doc = _new(name).__doc__ or ''
    assert 'braintools.metric' not in doc


def test_registry_records():
    records = {r.old_name: r for r in registered_deprecations()
               if r.old_name.startswith('braintools.metric.')}
    assert set(records) == {f'braintools.metric.{n}' for n in list(MOVED) + ['raster_plot']}
    for name, sub in MOVED.items():
        record = records[f'braintools.metric.{name}']
        assert record.kind == 'function'
        assert record.new_name == f'braintools.analysis.{sub}.{name}'
        assert (record.since, record.removed_in) == ('0.4.0', '1.0.0')
        assert record.alias is getattr(braintools.metric, name)
        assert record.target is _new(name)
    raster = records['braintools.metric.raster_plot']
    assert raster.new_name == 'braintools.spike.SpikeTrains.from_dense'
    assert raster.target is _deprecated._raster_plot


def test_access_and_import_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for name in list(MOVED) + ['raster_plot']:
            getattr(braintools.metric, name)
        from braintools.metric import firing_rate, raster_plot  # noqa: F401


def test_metric_all_keeps_names_and_order():
    deprecated = set(ORIGINAL_ORDER)
    assert [n for n in braintools.metric.__all__ if n in deprecated] == ORIGINAL_ORDER
    assert len(braintools.metric.__all__) == len(set(braintools.metric.__all__))


def test_analysis_exposes_only_subpackages():
    assert braintools.analysis.__all__ == ['spike', 'spectral', 'field', 'population', 'stats']
    for name in MOVED:
        assert not hasattr(braintools.analysis, name), name


def test_new_functions_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        braintools.analysis.stats.weighted_correlation(_X, _Y, _W)


def test_alias_pickles_by_reference():
    alias = braintools.metric.firing_rate
    assert pickle.loads(pickle.dumps(alias)) is alias
    new = braintools.analysis.spike.firing_rate
    assert pickle.loads(pickle.dumps(new)) is new


def test_alias_warns_once_at_trace_time_under_jit():
    f = jax.jit(braintools.metric.weighted_correlation)
    with pytest.warns(BraintoolsDeprecationWarning) as record:
        out = f(jnp.asarray(_X), jnp.asarray(_Y), jnp.asarray(_W))
    assert len(record) == 1
    expected = braintools.analysis.stats.weighted_correlation(_X, _Y, _W)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-6)
    # The compiled function is cached: calling again does not re-trace or warn.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        f(jnp.asarray(_X), jnp.asarray(_Y), jnp.asarray(_W))


class TestRasterPlot:
    """``raster_plot`` keeps its implementation but is deprecated."""

    def test_warns_with_spike_trains_replacement(self):
        with pytest.warns(BraintoolsDeprecationWarning,
                          match=r'Use `braintools\.spike\.SpikeTrains\.from_dense` instead'):
            braintools.metric.raster_plot(np.zeros((3, 2)), np.arange(3) * 0.1)

    def test_metadata(self):
        alias = braintools.metric.raster_plot
        assert alias.__name__ == 'raster_plot'
        assert alias.__module__ == 'braintools.metric'
        assert alias.__wrapped__ is _deprecated._raster_plot
        assert '.. deprecated:: 0.4.0' in alias.__doc__

    def test_indices_and_times(self):
        spikes = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 0],
        ])
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        with pytest.warns(BraintoolsDeprecationWarning):
            idx, st = braintools.metric.raster_plot(spikes, times)
        # Spikes occur (sorted by time-step) at: (t0,n1),(t1,n0),(t2,n2),(t3,n1),(t3,n2)
        np.testing.assert_array_equal(idx, np.array([1, 0, 2, 1, 2]))
        np.testing.assert_allclose(np.asarray(st), np.array([0.0, 0.1, 0.2, 0.3, 0.3]))

    def test_matches_nonzero(self):
        t_idx, n_idx = np.nonzero(_SPIKES)
        times = np.arange(_SPIKES.shape[0]) * 0.1
        with pytest.warns(BraintoolsDeprecationWarning):
            idx, st = braintools.metric.raster_plot(_SPIKES, times)
        np.testing.assert_array_equal(idx, n_idx)
        np.testing.assert_array_equal(st, times[t_idx])

    def test_no_spikes(self):
        with pytest.warns(BraintoolsDeprecationWarning):
            idx, st = braintools.metric.raster_plot(np.zeros((4, 3)), np.arange(4) * 1.0)
        assert idx.shape == (0,)
        assert st.shape == (0,)

    def test_preserves_quantity_units(self):
        spikes = np.zeros((5, 2))
        spikes[2, 1] = 1
        times = np.arange(5) * 0.1 * u.ms
        with pytest.warns(BraintoolsDeprecationWarning):
            _, st = braintools.metric.raster_plot(spikes, times)
        assert isinstance(st, u.Quantity)
        assert u.math.allclose(st, jnp.array([0.2]) * u.ms)

    def test_quantity_spike_matrix(self):
        spikes = np.zeros((4, 2))
        spikes[1, 0] = 1
        with pytest.warns(BraintoolsDeprecationWarning):
            idx, st = braintools.metric.raster_plot(spikes * u.UNITLESS, np.arange(4) * 1.0)
        np.testing.assert_array_equal(idx, [0])
        np.testing.assert_array_equal(st, [1.0])
