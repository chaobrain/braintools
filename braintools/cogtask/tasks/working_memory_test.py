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

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

import braintools.cogtask as ct
from braintools.cogtask.tasks.working_memory import make_encoder, build_cues


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
