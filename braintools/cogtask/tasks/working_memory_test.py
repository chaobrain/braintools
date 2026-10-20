# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Per-task semantic tests for the working-memory family.

Each task pins down label conventions (match=1/nonmatch=2, withhold=0 for
no-go, time-varying labels for ReadySetGo, …) and how key hyperparameters
(``num_stimuli``, ``stimulus_encoding``, ``cue_dim``, ``feature_per_direction``)
shape the input/output buffers.
"""

import doctest

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

import braintools.cogtask as ct
from braintools.cogtask.tasks.working_memory import make_encoder, build_cues, OneTwoThreeGo


def _phase_window(info, target):
    for name, s, e in info['phase_history']:
        if name == target:
            return int(s), int(e)
    raise AssertionError(f"no '{target}' phase in {[n for n, _, _ in info['phase_history']]}")


# ---------------------------------------------------------------------------
# make_encoder / build_cues
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["von_mises", "one_hot", "circular", "scalar"])
def test_make_encoder_modes_produce_callables_with_correct_dim(mode):
    enc = make_encoder(mode, "sample_idx")
    f = ct.Feature(8, 'stimulus')
    ctx = ct.Context()
    ctx['sample_idx'] = 3
    out = np.asarray(enc(ctx, f))
    assert out.shape == (8,)


def test_make_encoder_rejects_unknown_mode():
    with pytest.raises(ValueError):
        make_encoder("nonsense", "sample_idx")


def test_make_encoder_population_repeats_one_hot():
    enc = make_encoder("one_hot", "sample_idx", feature_per_direction=3)
    f = ct.Feature(12, 'stimulus')  # 4 dirs * 3 repeats
    ctx = ct.Context()
    ctx['sample_idx'] = 2
    out = np.asarray(enc(ctx, f))
    # Index 2 of base (size 4) repeated 3× gives three contiguous "1.0"s at positions 6, 7, 8.
    np.testing.assert_allclose(out, [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])


def test_make_encoder_population_dim_not_divisible_raises():
    enc = make_encoder("one_hot", "sample_idx", feature_per_direction=3)
    f = ct.Feature(10, 'stimulus')  # 10 / 3 != integer
    ctx = ct.Context()
    ctx['sample_idx'] = 1
    with pytest.raises(ValueError, match="divisible"):
        enc(ctx, f)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_make_encoder_scalar_is_feature_per_direction_agnostic(k):
    # Regression: scalar was only handled in the K==1 branch, so
    # feature_per_direction>1 raised "Unknown mode=scalar". scalar broadcasts a
    # single value across all dims and must work for any K.
    enc = make_encoder("scalar", "sample_value", feature_per_direction=k)
    f = ct.Feature(6, 'stimulus')
    ctx = ct.Context()
    ctx['sample_value'] = 0.7
    out = np.asarray(enc(ctx, f))
    assert out.shape == (6,)
    np.testing.assert_allclose(out, 0.7, atol=1e-6)


def test_build_cues_returns_jax_arrays_with_right_shapes():
    non_resp, resp = build_cues(3)
    assert non_resp.shape == (3,)
    assert resp.shape == (3,)
    np.testing.assert_allclose(np.asarray(non_resp), [0, 0, 0])
    np.testing.assert_allclose(np.asarray(resp), [1, 0, 0])


def test_build_cues_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="Cue"):
        build_cues(3, non_response_cue=[0.0, 0.0])


# ---------------------------------------------------------------------------
# DelayMatchSample
# ---------------------------------------------------------------------------

def test_dms_response_label_is_match_or_nonmatch():
    task = ct.DelayMatchSample(seed=0)
    seen_match = seen_nonmatch = False
    for i in range(30):
        _, Y, info = task.sample_trial(i)
        is_match = bool(info['trial_state']['is_match'])
        s, e = _phase_window(info, 'Response')
        labels = np.asarray(Y)[s:e]
        if is_match:
            assert (labels == 1).all()
            seen_match = True
        else:
            assert (labels == 2).all()
            seen_nonmatch = True
        if seen_match and seen_nonmatch:
            break
    assert seen_match and seen_nonmatch


def test_dms_test_idx_equals_sample_when_match():
    task = ct.DelayMatchSample(seed=0)
    for i in range(20):
        _, _, info = task.sample_trial(i)
        state = info['trial_state']
        if bool(state['is_match']):
            assert int(state['test_idx']) == int(state['sample_idx'])


def test_dms_test_idx_differs_from_sample_when_nonmatch():
    task = ct.DelayMatchSample(seed=0)
    for i in range(20):
        _, _, info = task.sample_trial(i)
        state = info['trial_state']
        if not bool(state['is_match']):
            assert int(state['test_idx']) != int(state['sample_idx'])


def test_dms_feature_per_direction_expands_stim_dim():
    task = ct.DelayMatchSample(seed=0, num_stimuli=4, feature_per_direction=3)
    # input = cue(1) + stim(4*3) = 13
    assert task.num_inputs == 1 + 12


def test_dms_cue_dim_changes_input_and_output_dim():
    task = ct.DelayMatchSample(seed=0, cue_dim=4)
    # input cue dim 4 + stim 8 = 12
    assert task.num_inputs == 4 + 8
    # output cue dim 4 + response 2 = 6
    assert task.num_outputs == 4 + 2


# ---------------------------------------------------------------------------
# DualDelayMatchSample
# ---------------------------------------------------------------------------

def test_dual_dms_test_idx_picks_sample1_or_sample2():
    task = ct.DualDelayMatchSample(seed=0)
    for i in range(10):
        _, _, info = task.sample_trial(i)
        s = info['trial_state']
        gt = int(s['ground_truth'])
        expected = int(s['sample1_idx']) if gt == 0 else int(s['sample2_idx'])
        assert int(s['test_idx']) == expected


# ---------------------------------------------------------------------------
# DelayComparison
# ---------------------------------------------------------------------------

def test_delay_comparison_label_follows_comparison_result():
    task = ct.DelayComparison(seed=0)
    for i in range(10):
        _, Y, info = task.sample_trial(i)
        is_greater = bool(info['trial_state']['comparison_result'])
        s, e = _phase_window(info, 'response')
        labels = np.asarray(Y)[s:e]
        if is_greater:
            assert (labels == 1).all()
        else:
            assert (labels == 2).all()


@pytest.mark.parametrize("encoding,expected_dim", [
    ('scalar', 1),
    ('identity', 1),
    ('gaussian', 10),
])
def test_delay_comparison_value_encoding_dim(encoding, expected_dim):
    task = ct.DelayComparison(seed=0, value_encoding=encoding, num_features=10)
    # cue 1 + stim expected_dim
    assert task.num_inputs == 1 + expected_dim


def test_delay_comparison_unknown_encoding_raises():
    with pytest.raises(ValueError):
        ct.DelayComparison(seed=0, value_encoding='nope')


# ---------------------------------------------------------------------------
# DelayMatchCategory
# ---------------------------------------------------------------------------

def test_dmc_match_means_same_category():
    task = ct.DelayMatchCategory(seed=0)
    for i in range(20):
        _, _, info = task.sample_trial(i)
        s = info['trial_state']
        if bool(s['is_match']):
            assert int(s['sample_category']) == int(s['test_category'])
        else:
            assert int(s['sample_category']) != int(s['test_category'])


def test_dmc_total_stimuli_drives_stim_dim():
    task = ct.DelayMatchCategory(seed=0, num_categories=3, num_exemplars=5)
    assert task.total_stimuli == 15
    assert task.num_inputs == 1 + 15


# ---------------------------------------------------------------------------
# DelayPairedAssociation
# ---------------------------------------------------------------------------

def test_paired_association_ground_truth_equals_pair_idx():
    task = ct.DelayPairedAssociation(seed=0)
    for i in range(8):
        _, _, info = task.sample_trial(i)
        s = info['trial_state']
        assert int(s['ground_truth']) == int(s['pair_idx'])


def test_paired_association_response_label_matches_ground_truth_plus_one():
    task = ct.DelayPairedAssociation(seed=0)
    for i in range(8):
        _, Y, info = task.sample_trial(i)
        gt = int(info['trial_state']['ground_truth'])
        s, e = _phase_window(info, 'response')
        assert (np.asarray(Y)[s:e] == gt + 1).all()


# ---------------------------------------------------------------------------
# GoNoGo
# ---------------------------------------------------------------------------

def test_gonogo_response_label_is_1_on_go_and_0_on_nogo():
    """No-go trials should emit label 0 (withhold), not a distinct 'nogo' label.

    Encoding no-go with label 0 (same as fixation) makes the network learn to
    actually suppress responding, which is the point of go/no-go.
    """
    task = ct.GoNoGo(seed=0)
    found_go = found_nogo = False
    for i in range(40):
        _, Y, info = task.sample_trial(i)
        is_go = bool(info['trial_state']['is_go'])
        s, e = _phase_window(info, 'response')
        resp = np.asarray(Y)[s:e]
        if is_go:
            assert (resp == 1).all()
            found_go = True
        else:
            assert (resp == 0).all()
            found_nogo = True
        if found_go and found_nogo:
            break
    assert found_go and found_nogo


def test_gonogo_stimulus_class_is_zero_on_go_and_one_on_nogo():
    task = ct.GoNoGo(seed=0)
    for i in range(20):
        _, _, info = task.sample_trial(i)
        s = info['trial_state']
        if bool(s['is_go']):
            assert int(s['stimulus_class']) == 0
        else:
            assert int(s['stimulus_class']) == 1


# ---------------------------------------------------------------------------
# IntervalDiscrimination (variable duration)
# ---------------------------------------------------------------------------

def test_interval_discrimination_label_follows_interval_comparison():
    task = ct.IntervalDiscrimination(seed=0)
    for i in range(10):
        _, _, info = task.sample_trial(i)
        s = info['trial_state']
        i1 = float(s['interval1_duration'])
        i2 = float(s['interval2_duration'])
        gt = int(s['ground_truth'])
        # GT==0 when interval1 > interval2, GT==1 otherwise.
        expected = 0 if i1 > i2 else 1
        assert gt == expected


# ---------------------------------------------------------------------------
# PostDecisionWager
# ---------------------------------------------------------------------------

def test_post_decision_wager_emits_decision_and_wager_phases():
    task = ct.PostDecisionWager(seed=0)
    _, _, info = task.sample_trial(0)
    names = [n for n, _, _ in info['phase_history']]
    assert 'decision' in names
    assert 'wager' in names


def test_post_decision_wager_uses_num_choices_plus_two_for_wager_labels():
    task = ct.PostDecisionWager(seed=0)
    for i in range(8):
        _, Y, info = task.sample_trial(i)
        s, e = _phase_window(info, 'wager')
        labels = np.asarray(Y)[s:e]
        wager_gt = int(info['trial_state']['wager_gt'])
        expected = wager_gt + task.num_choices + 1
        assert (labels == expected).all()


# ---------------------------------------------------------------------------
# ReadySetGo
# ---------------------------------------------------------------------------

def test_readysetgo_production_label_transitions_one_to_two():
    """Pre-fix bug: production label was always 1. Post-fix it must transition
    from 1 (hold) to 2 (go) within the response window."""
    task = ct.ReadySetGo(seed=1)
    _, Y, info = task.sample_trial(0)
    s, e = _phase_window(info, 'production')
    window = np.asarray(Y)[s:e]
    assert (window == 1).any() and (window == 2).any()


def test_readysetgo_is_independent_of_dt_unit():
    """``dt`` written as ``1 ms`` or ``0.001 s`` must give the same trial; the
    ms-valued intervals used to be divided by the raw dt mantissa."""
    out = []
    for dt in (1.0 * u.ms, 0.001 * u.second):
        _, Y, info = ct.ReadySetGo(seed=3, dt=dt).sample_trial(0)
        out.append((np.asarray(Y), int(np.asarray(info['mask']).sum())))
    np.testing.assert_array_equal(out[0][0], out[1][0])
    assert out[0][1] == out[1][1]


def test_readysetgo_produce_interval_equals_gain_times_measure():
    task = ct.ReadySetGo(seed=0, gain=2.0)
    _, _, info = task.sample_trial(0)
    s = info['trial_state']
    np.testing.assert_allclose(float(s['produce_interval']), 2.0 * float(s['measure_interval']))


# ---------------------------------------------------------------------------
# DelayDirectionReproduction / ImmediateDirectionReproduction (vector output)
# ---------------------------------------------------------------------------

def test_ddr_response_outputs_match_cos_sin_of_sample_direction():
    task = ct.DelayDirectionReproduction(seed=0)
    for i in range(5):
        _, Y, info = task.sample_trial(i)
        s, e = _phase_window(info, 'Response')
        idx = int(info['trial_state']['sample_idx'])
        theta = 2.0 * np.pi * idx / task.num_stimuli
        # output layout: fixation_out(1) + direction(2)
        Y = np.asarray(Y)
        # During response, fixation_out=0 and direction = [cos, sin].
        np.testing.assert_allclose(Y[s:e, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(Y[s:e, 1], np.cos(theta), atol=1e-5)
        np.testing.assert_allclose(Y[s:e, 2], np.sin(theta), atol=1e-5)


def test_ddr_pre_response_fixation_out_is_one():
    task = ct.DelayDirectionReproduction(seed=0)
    _, Y, info = task.sample_trial(0)
    s, _ = _phase_window(info, 'Response')
    Y = np.asarray(Y)
    np.testing.assert_allclose(Y[:s, 0], 1.0, atol=1e-5)


def test_immediate_direction_reproduction_has_only_fixation_and_response():
    task = ct.ImmediateDirectionReproduction(seed=0)
    _, _, info = task.sample_trial(0)
    names = [n for n, _, _ in info['phase_history'] if n not in ('Sequence', 'Parallel')]
    assert names == ['Fixation', 'Response']


def test_ddr_ifvon_false_uses_cos_sin_encoder_with_2_repeats_stim_dim():
    task = ct.DelayDirectionReproduction(seed=0, IfVon=False, feature_per_direction=3)
    # cue 2 + stim 2*3 = 8
    assert task.num_inputs == 2 + 6


# ---------------------------------------------------------------------------
# DelayDirectionClassification / ImmediateDirectionClassification (categorical)
# ---------------------------------------------------------------------------

def test_ddc_response_label_is_category_plus_one():
    task = ct.DelayDirectionClassification(seed=0, num_dirs=8, num_categories=2)
    for i in range(8):
        _, Y, info = task.sample_trial(i)
        cat = int(info['trial_state']['category'])
        s, e = _phase_window(info, 'Response')
        assert (np.asarray(Y)[s:e] == cat + 1).all()


def test_ddc_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        ct.DelayDirectionClassification(seed=0, num_dirs=0, num_categories=2)
    with pytest.raises(ValueError):
        ct.DelayDirectionClassification(seed=0, num_dirs=8, num_categories=1)


def test_idc_response_label_is_category_plus_one():
    task = ct.ImmediateDirectionClassification(seed=0)
    for i in range(8):
        _, Y, info = task.sample_trial(i)
        cat = int(info['trial_state']['category'])
        s, e = _phase_window(info, 'Response')
        assert (np.asarray(Y)[s:e] == cat + 1).all()


def test_ddc_custom_category_fn_overrides_default():
    task = ct.DelayDirectionClassification(
        seed=0, num_dirs=8, num_categories=2,
        category_fn=lambda idx: jnp.asarray(0, jnp.int32),  # always 0
    )
    for i in range(4):
        _, _, info = task.sample_trial(i)
        assert int(info['trial_state']['category']) == 0


# ---------------------------------------------------------------------------
# OneTwoThreeGo
# ---------------------------------------------------------------------------

_DT10 = 10.0 * u.ms
_123GO_PHASES = ['fixation', 'target', 's1', 'interval1', 's2', 'interval2',
                 's3', 'interval3', 'response']


def _spans(info):
    return {name: (int(s), int(e)) for name, s, e in info['phase_history']}


def test_123go_features_and_buffer_length():
    task = OneTwoThreeGo(seed=0, dt=_DT10)
    assert task.num_inputs == 3
    assert task.num_outputs == 2
    assert task.num_classes == 2
    assert task.is_variable_length
    # max: fixation 800 + target 1500 + 3 * 100 + 3 * 1000 + response 1000 ms.
    assert task.max_trial_duration() == 660
    X, Y, info = task.sample_trial(0)
    assert X.shape == (660, 3)
    assert Y.shape == (660,)
    assert info['mask'].shape == (660,)


def test_123go_phase_order():
    task = OneTwoThreeGo(seed=0, dt=_DT10)
    _, _, info = task.sample_trial(0)
    names = [n for n, _, _ in info['phase_history'] if n != 'Sequence']
    assert names == _123GO_PHASES


def test_123go_target_response_time_is_one_interval_after_third_pulse():
    task = OneTwoThreeGo(seed=1, dt=_DT10)
    for i in range(8):
        _, Y, info = task.sample_trial(i)
        sp = _spans(info)
        interval_steps = int(round(float(info['trial_state']['interval']) / 10.0))
        for k in (1, 2, 3):
            assert sp[f's{k}'][1] - sp[f's{k}'][0] == 10
            assert sp[f'interval{k}'][1] - sp[f'interval{k}'][0] == interval_steps
        # Pulses are evenly spaced by the interval ...
        assert sp['s2'][0] - sp['s1'][1] == interval_steps
        assert sp['s3'][0] - sp['s2'][1] == interval_steps
        # ... and the go label starts one interval after the third pulse.
        go_onset = sp['s3'][1] + interval_steps
        assert sp['response'][0] == go_onset
        Y = np.asarray(Y)
        assert (Y[:go_onset] == 0).all()
        assert (Y[go_onset:sp['response'][1]] == 1).all()
        assert sp['response'][1] - sp['response'][0] == 100
        # The mask covers exactly the trial; padding is zero.
        mask = np.asarray(info['mask'])
        assert mask.sum() == sp['response'][1]
        assert (Y[~mask] == 0).all()


def test_123go_phase_inputs():
    task = OneTwoThreeGo(seed=2, dt=_DT10)
    X, _, info = task.sample_trial(0)
    X = np.asarray(X)
    sp = _spans(info)
    mask = np.asarray(info['mask'])
    fix, stim, target = X[:, 0], X[:, 1], X[:, 2]
    # Fixation stays on for the whole trial (no go cue); padding is zero.
    np.testing.assert_array_equal(fix[mask], 1.0)
    np.testing.assert_array_equal(X[~mask], 0.0)
    # Target appears at the target period and stays on.
    fs, fe = sp['fixation']
    np.testing.assert_array_equal(target[fs:fe], 0.0)
    np.testing.assert_array_equal(target[sp['target'][0]:sp['response'][1]], 1.0)
    # Stimulus is on exactly during the three pulses.
    pulse = np.zeros(X.shape[0], dtype=bool)
    for k in (1, 2, 3):
        s, e = sp[f's{k}']
        pulse[s:e] = True
    np.testing.assert_array_equal(stim[pulse], 1.0)
    np.testing.assert_array_equal(stim[~pulse], 0.0)


def test_123go_sampled_durations_statistics():
    task = OneTwoThreeGo(seed=3, dt=_DT10)
    _, _, meta = task.batch_sample(400, return_meta=True)
    intervals = np.asarray(meta['interval'])
    assert set(np.unique(intervals).tolist()) == {600.0, 700.0, 800.0, 900.0, 1000.0}
    fix = np.asarray(meta['fixation_duration'])
    tgt = np.asarray(meta['target_duration'])
    assert fix.min() >= 100.0 and fix.max() <= 800.0
    assert tgt.min() >= 500.0 and tgt.max() <= 1500.0

    def trunc_exp_mean(mu, a, b):
        ea, eb = np.exp(-a / mu), np.exp(-b / mu)
        return mu + (a * ea - b * eb) / (ea - eb)

    assert fix.mean() == pytest.approx(trunc_exp_mean(400.0, 100.0, 800.0), abs=30.0)
    assert tgt.mean() == pytest.approx(trunc_exp_mean(1000.0, 500.0, 1500.0), abs=40.0)


def test_123go_fixed_periods_and_custom_intervals():
    task = OneTwoThreeGo(seed=4, dt=_DT10, t_fixation=200 * u.ms, t_target=300 * u.ms,
                         intervals=(0.5 * u.second,), t_stimulus=50 * u.ms,
                         t_response=200 * u.ms)
    _, Y, info = task.sample_trial(0)
    state = info['trial_state']
    assert 'fixation_duration' not in state and 'target_duration' not in state
    assert float(state['interval']) == 500.0
    sp = _spans(info)
    assert sp['fixation'] == (0, 20)
    assert sp['target'] == (20, 50)
    assert sp['s1'] == (50, 55)
    assert sp['response'] == (55 + 3 * 50 + 2 * 5, 55 + 3 * 50 + 2 * 5 + 20)
    # Fully determined trial: every valid step is used.
    assert task.max_trial_duration() == sp['response'][1]
    assert int(np.asarray(Y)[sp['response'][0]]) == 1


def test_123go_is_independent_of_dt_unit():
    out = []
    for dt in (10.0 * u.ms, 0.01 * u.second):
        X, Y, info = OneTwoThreeGo(seed=5, dt=dt).sample_trial(0)
        out.append((np.asarray(X), np.asarray(Y), np.asarray(info['mask'])))
    for a, b in zip(*out):
        np.testing.assert_array_equal(a, b)


def test_123go_seeded_reproducibility_and_batch_consistency():
    a = OneTwoThreeGo(seed=6, dt=_DT10)
    b = OneTwoThreeGo(seed=6, dt=_DT10)
    Xa, Ya, ia = a.sample_trial(1)
    Xb, Yb, ib = b.sample_trial(1)
    np.testing.assert_array_equal(np.asarray(Xa), np.asarray(Xb))
    np.testing.assert_array_equal(np.asarray(Ya), np.asarray(Yb))
    assert float(ia['trial_state']['interval']) == float(ib['trial_state']['interval'])

    X, Y, M = a.batch_sample(3, return_mask=True)
    assert X.shape == (660, 3, 3) and Y.shape == (660, 3) and M.shape == (660, 3)
    for i in range(3):
        Xi, Yi, info = a.sample_trial(i)
        np.testing.assert_array_equal(np.asarray(X)[:, i], np.asarray(Xi))
        np.testing.assert_array_equal(np.asarray(Y)[:, i], np.asarray(Yi))
        np.testing.assert_array_equal(np.asarray(M)[:, i], np.asarray(info['mask']))

    Xbf, Ybf = a.batch_sample(3, time_first=False)
    assert Xbf.shape == (3, 660, 3) and Ybf.shape == (3, 660)


@pytest.mark.parametrize("kwargs", [
    dict(t_fixation=(400 * u.ms, 100 * u.ms)),
    dict(t_fixation=(400 * u.ms, 900 * u.ms, 800 * u.ms)),
    dict(t_target=(0 * u.ms, 100 * u.ms, 800 * u.ms)),
    dict(intervals=()),
    dict(intervals=(0 * u.ms, 600 * u.ms)),
])
def test_123go_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        OneTwoThreeGo(**kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(t_fixation=400.0),
    dict(intervals=(600.0, 700.0)),
])
def test_123go_rejects_unitless_durations(kwargs):
    with pytest.raises(Exception):
        OneTwoThreeGo(**kwargs)


def test_123go_docstring_examples(monkeypatch):
    monkeypatch.setattr(ct, 'OneTwoThreeGo', OneTwoThreeGo, raising=False)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in doctest.DocTestFinder(recurse=False).find(OneTwoThreeGo, globs={}):
        runner.run(test)
    assert runner.tries > 0
    assert runner.failures == 0
