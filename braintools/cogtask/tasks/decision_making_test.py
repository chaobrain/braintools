# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Per-task semantic tests for the decision-making family.

Each task verifies:
- the response-phase label convention (label = ground_truth + 1),
- the phase ordering (fixation → ... → response),
- key constructor parameters change the shape of the input/output buffers as
  documented.
"""

import doctest

import brainunit as u
import numpy as np
import pytest

import braintools.cogtask as ct
from braintools.cogtask.tasks.decision_making import MultiSensoryIntegration


def _response_window(info):
    for name, s, e in info['phase_history']:
        if name == 'response':
            return int(s), int(e)
    raise AssertionError(f"no response phase in {[n for n, _, _ in info['phase_history']]}")


# ---------------------------------------------------------------------------
# PerceptualDecisionMaking
# ---------------------------------------------------------------------------

def test_pdm_response_label_matches_ground_truth_plus_one():
    task = ct.PerceptualDecisionMaking(seed=0)
    for i in range(8):
        _, Y, info = task.sample_trial(i)
        gt = int(info['trial_state']['ground_truth'])
        s, e = _response_window(info)
        assert (np.asarray(Y)[s:e] == gt + 1).all()


def test_pdm_phase_order_is_fixation_stimulus_response():
    task = ct.PerceptualDecisionMaking(seed=0)
    _, _, info = task.sample_trial(0)
    # phase_history includes the wrapping Sequence as a final entry — filter it
    # out and keep only the named leaf phases.
    names = [n for n, _, _ in info['phase_history'] if n not in ('Sequence', 'Parallel')]
    assert names == ['fixation', 'stimulus', 'response']


@pytest.mark.parametrize("num_choices", [2, 3, 4])
def test_pdm_num_choices_changes_io_dims(num_choices):
    task = ct.PerceptualDecisionMaking(seed=0, num_choices=num_choices)
    # input: fixation(1) + num_choices*pop_per_choice
    assert task.num_inputs == 1 + num_choices * task.pop_per_choice
    # output: fixation(1) + choice(num_choices)
    assert task.num_outputs == 1 + num_choices


def test_pdm_durations_change_total_trial_length():
    base = ct.PerceptualDecisionMaking(seed=0)
    X0, _, _ = base.sample_trial(0)
    longer = ct.PerceptualDecisionMaking(seed=0, t_stimulus=3000 * u.ms)
    X1, _, _ = longer.sample_trial(0)
    assert X1.shape[0] > X0.shape[0]


def test_pdm_pre_response_label_is_zero():
    task = ct.PerceptualDecisionMaking(seed=0)
    _, Y, info = task.sample_trial(0)
    s, _ = _response_window(info)
    # All labels before the response phase are 0 (fixation/hold).
    assert (np.asarray(Y)[:s] == 0).all()


# ---------------------------------------------------------------------------
# PerceptualDecisionMakingDelayResponse
# ---------------------------------------------------------------------------

def test_pdm_delay_response_includes_delay_phase():
    task = ct.PerceptualDecisionMakingDelayResponse(seed=0)
    _, _, info = task.sample_trial(0)
    names = [n for n, _, _ in info['phase_history'] if n not in ('Sequence', 'Parallel')]
    assert names == ['fixation', 'stimulus', 'delay', 'response']


def test_pdm_delay_response_label_matches_ground_truth():
    task = ct.PerceptualDecisionMakingDelayResponse(seed=0)
    for i in range(4):
        _, Y, info = task.sample_trial(i)
        gt = int(info['trial_state']['ground_truth'])
        s, e = _response_window(info)
        assert (np.asarray(Y)[s:e] == gt + 1).all()


# ---------------------------------------------------------------------------
# ContextDecisionMaking
# ---------------------------------------------------------------------------

def test_context_decision_making_uses_attended_modality_for_label():
    task = ct.ContextDecisionMaking(seed=0)
    for i in range(8):
        _, Y, info = task.sample_trial(i)
        state = info['trial_state']
        gt = int(state['ground_truth'])
        ctx_val = int(state['context'])
        expected = int(state['mod1_gt']) if ctx_val == 0 else int(state['mod2_gt'])
        assert gt == expected
        s, e = _response_window(info)
        assert (np.asarray(Y)[s:e] == gt + 1).all()


def test_context_decision_making_phase_history_has_parallel_modalities():
    task = ct.ContextDecisionMaking(seed=0)
    _, _, info = task.sample_trial(0)
    names = [n for n, _, _ in info['phase_history']]
    assert 'modality1' in names and 'modality2' in names


# ---------------------------------------------------------------------------
# SingleContextDecisionMaking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("context", [0, 1])
def test_single_context_uses_fixed_context(context):
    task = ct.SingleContextDecisionMaking(seed=0, context=context)
    for i in range(4):
        _, _, info = task.sample_trial(i)
        state = info['trial_state']
        gt = int(state['ground_truth'])
        expected = int(state['mod1_gt']) if context == 0 else int(state['mod2_gt'])
        assert gt == expected
        assert int(state['context']) == context


# ---------------------------------------------------------------------------
# PulseDecisionMaking
# ---------------------------------------------------------------------------

def test_pulse_decision_making_label_equals_ground_truth_plus_one():
    task = ct.PulseDecisionMaking(seed=3)
    for i in range(5):
        _, Y, info = task.sample_trial(i)
        gt = int(info['trial_state']['ground_truth'])
        last = int(np.asarray(Y)[-1])
        assert last == gt + 1


def test_pulse_decision_making_phase_history_contains_num_pulses_pulses():
    task = ct.PulseDecisionMaking(seed=0, num_pulses=5)
    _, _, info = task.sample_trial(0)
    names = [n for n, _, _ in info['phase_history']]
    assert names.count('pulse') == 5
    assert names.count('pulse_delay') == 5


def test_pulse_decision_making_ground_truth_follows_evidence_sign():
    task = ct.PulseDecisionMaking(seed=42)
    for i in range(10):
        _, _, info = task.sample_trial(i)
        state = info['trial_state']
        total = float(state['total_evidence'])
        gt = int(state['ground_truth'])
        if total > 0:
            assert gt == 1
        else:
            assert gt == 0


# ---------------------------------------------------------------------------
# MultiSensoryIntegration
# ---------------------------------------------------------------------------

def _windows(info):
    return {name: (int(s), int(e)) for name, s, e in info['phase_history']}


def _tuned(coh, direction, n):
    """NeuroGym's cosine-tuned modality input for one trial."""
    pref = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return 0.5 + coh / 200.0 * np.cos(pref - direction)


def test_msi_default_shapes_match_neurogym():
    task = MultiSensoryIntegration(seed=0)
    # NeuroGym: observation 1 + 2 * dim_ring, action 1 + dim_ring.
    assert task.num_inputs == 5
    assert task.num_outputs == 3
    assert task.num_classes == 3
    assert not task.is_variable_length
    X, Y, _ = task.sample_trial(0)
    # 300 + 750 + 100 ms at dt = 1 ms.
    assert X.shape == (1150, 5)
    assert Y.shape == (1150,)


@pytest.mark.parametrize("num_choices, pop", [(2, 1), (3, 2), (4, 3)])
def test_msi_feature_dims(num_choices, pop):
    task = MultiSensoryIntegration(seed=0, num_choices=num_choices, pop_per_choice=pop)
    assert task.num_inputs == 1 + 2 * num_choices * pop
    assert task.num_outputs == 1 + num_choices
    assert task.input_features['modality1'].num == num_choices * pop
    assert task.input_features['modality2'].num == num_choices * pop


def test_msi_phase_order_and_durations():
    task = MultiSensoryIntegration(seed=0, t_fixation=200 * u.ms, t_stimulus=500 * u.ms,
                                   t_decision=50 * u.ms)
    _, _, info = task.sample_trial(0)
    names = [n for n, _, _ in info['phase_history'] if n != 'Sequence']
    assert names == ['fixation', 'stimulus', 'decision']
    w = _windows(info)
    assert w['fixation'] == (0, 200)
    assert w['stimulus'] == (200, 700)
    assert w['decision'] == (700, 750)


def test_msi_labels_zero_then_ground_truth_plus_one():
    task = MultiSensoryIntegration(seed=1, num_choices=3)
    seen = set()
    for i in range(12):
        _, Y, info = task.sample_trial(i)
        gt = int(info['trial_state']['ground_truth'])
        seen.add(gt)
        s, e = _windows(info)['decision']
        Y = np.asarray(Y)
        assert (Y[:s] == 0).all()
        assert (Y[s:e] == gt + 1).all()
    assert seen == {0, 1, 2}


def test_msi_phase_inputs_noise_free():
    task = MultiSensoryIntegration(seed=3, noise_sigma=0.0 * u.ms ** 0.5, num_choices=4,
                                   pop_per_choice=2)
    for i in range(4):
        X, _, info = task.sample_trial(i)
        X = np.asarray(X)
        st = info['trial_state']
        w = _windows(info)
        m1, m2 = task.input_features['modality1'].i, task.input_features['modality2'].i
        # Fixation: only the fixation channel is on.
        fs, fe = w['fixation']
        np.testing.assert_array_equal(X[fs:fe, 0], 1.0)
        np.testing.assert_array_equal(X[fs:fe, 1:], 0.0)
        # Stimulus: fixation on, both modalities cosine-tuned to the target.
        ss, se = w['stimulus']
        np.testing.assert_array_equal(X[ss:se, 0], 1.0)
        direction = float(st['stimulus_direction'])
        np.testing.assert_allclose(X[ss, m1], _tuned(float(st['mod1_coherence']), direction, 8),
                                   atol=1e-6)
        np.testing.assert_allclose(X[ss, m2], _tuned(float(st['mod2_coherence']), direction, 8),
                                   atol=1e-6)
        # Constant over the stimulus period without noise.
        np.testing.assert_array_equal(X[ss:se, m1], np.broadcast_to(X[ss, m1], X[ss:se, m1].shape))
        # Decision: every input is zero.
        ds, de = w['decision']
        np.testing.assert_array_equal(X[ds:de], 0.0)


def test_msi_evidence_is_split_and_combined_toward_ground_truth():
    task = MultiSensoryIntegration(seed=4, noise_sigma=0.0 * u.ms ** 0.5, num_choices=4,
                                   coherences=(50.0,))
    for i in range(10):
        X, _, info = task.sample_trial(i)
        st = info['trial_state']
        coh_sum = float(st['mod1_coherence']) + float(st['mod2_coherence'])
        np.testing.assert_allclose(coh_sum, float(st['coherence']), rtol=1e-6)
        assert 0.0 <= float(st['coh_prop']) <= 1.0
        assert int(st['modality_condition']) == 0
        ss, _ = _windows(info)['stimulus']
        x = np.asarray(X)[ss]
        combined = (x[task.input_features['modality1'].i] - 0.5
                    + x[task.input_features['modality2'].i] - 0.5)
        # The summed evidence of both modalities peaks at the correct choice and
        # equals that of a single modality at the full coherence.
        assert int(np.argmax(combined)) == int(st['ground_truth'])
        np.testing.assert_allclose(combined.max(), 50.0 / 200.0, atol=1e-6)


def test_msi_unisensory_trials_silence_one_modality():
    task = MultiSensoryIntegration(seed=5, noise_sigma=0.0 * u.ms ** 0.5, p_unisensory=1.0)
    conditions = set()
    for i in range(12):
        X, _, info = task.sample_trial(i)
        st = info['trial_state']
        cond = int(st['modality_condition'])
        conditions.add(cond)
        ss, se = _windows(info)['stimulus']
        x = np.asarray(X)[ss:se]
        m1 = x[:, task.input_features['modality1'].i]
        m2 = x[:, task.input_features['modality2'].i]
        coh = float(st['coherence'])
        if cond == 1:
            np.testing.assert_array_equal(m2, 0.0)
            assert float(st['mod1_coherence']) == pytest.approx(coh)
            assert (float(st['mod1_present']), float(st['mod2_present'])) == (1.0, 0.0)
        else:
            assert cond == 2
            np.testing.assert_array_equal(m1, 0.0)
            assert float(st['mod2_coherence']) == pytest.approx(coh)
            assert (float(st['mod1_present']), float(st['mod2_present'])) == (0.0, 1.0)
    assert conditions == {1, 2}


def test_msi_trial_type_statistics():
    task = MultiSensoryIntegration(seed=6, t_fixation=10 * u.ms, t_stimulus=10 * u.ms,
                                   t_decision=10 * u.ms, p_unisensory=0.5)
    _, _, meta = task.batch_sample(400, return_meta=True)
    cond = np.asarray(meta['modality_condition'])
    frac_uni = np.mean(cond > 0)
    assert 0.4 < frac_uni < 0.6
    multi = cond == 0
    # coh_prop ~ U(0, 1) on multisensory trials.
    assert 0.4 < np.mean(np.asarray(meta['coh_prop'])[multi]) < 0.6
    assert set(np.unique(np.asarray(meta['coherence'])).tolist()) == {5.0, 15.0, 50.0}
    assert 0.4 < np.mean(np.asarray(meta['ground_truth'])) < 0.6


def test_msi_noise_scale():
    task = MultiSensoryIntegration(seed=7, noise_sigma=2.0 * u.ms ** 0.5, t_stimulus=2000 * u.ms)
    clean = MultiSensoryIntegration(seed=7, noise_sigma=0.0 * u.ms ** 0.5, t_stimulus=2000 * u.ms)
    X, _, info = task.sample_trial(0)
    X0, _, _ = clean.sample_trial(0)
    ss, se = _windows(info)['stimulus']
    resid = np.asarray(X)[ss:se, 1:] - np.asarray(X0)[ss:se, 1:]
    # sigma / sqrt(dt) with dt = 1 ms.
    assert np.std(resid) == pytest.approx(2.0, rel=0.05)
    # Fixation channel and other phases are noise free.
    np.testing.assert_array_equal(np.asarray(X)[:, 0], np.asarray(X0)[:, 0])
    np.testing.assert_array_equal(np.asarray(X)[:ss], np.asarray(X0)[:ss])


def test_msi_seeded_reproducibility():
    a = MultiSensoryIntegration(seed=11)
    b = MultiSensoryIntegration(seed=11)
    c = MultiSensoryIntegration(seed=12)
    Xa, Ya, _ = a.sample_trial(3)
    Xb, Yb, _ = b.sample_trial(3)
    Xc, _, _ = c.sample_trial(3)
    np.testing.assert_array_equal(np.asarray(Xa), np.asarray(Xb))
    np.testing.assert_array_equal(np.asarray(Ya), np.asarray(Yb))
    assert not np.array_equal(np.asarray(Xa), np.asarray(Xc))


@pytest.mark.parametrize("time_first", [True, False])
def test_msi_batch_sample_shapes_and_consistency(time_first):
    task = MultiSensoryIntegration(seed=2, t_stimulus=100 * u.ms)
    X, Y = task.batch_sample(3, time_first=time_first)
    if time_first:
        assert X.shape == (500, 3, 5) and Y.shape == (500, 3)
        X_trials = np.moveaxis(np.asarray(X), 1, 0)
        Y_trials = np.asarray(Y).T
    else:
        assert X.shape == (3, 500, 5) and Y.shape == (3, 500)
        X_trials, Y_trials = np.asarray(X), np.asarray(Y)
    for i in range(3):
        Xi, Yi, _ = task.sample_trial(i)
        np.testing.assert_allclose(X_trials[i], np.asarray(Xi), atol=1e-5)
        np.testing.assert_array_equal(Y_trials[i], np.asarray(Yi))


@pytest.mark.parametrize("kwargs", [
    dict(num_choices=1),
    dict(pop_per_choice=0),
    dict(coherences=()),
    dict(p_unisensory=-0.1),
    dict(p_unisensory=1.5),
])
def test_msi_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        MultiSensoryIntegration(**kwargs)


def test_msi_docstring_examples(monkeypatch):
    # Examples use the public ``braintools.cogtask`` name; make it resolvable
    # whether or not the package ``__init__`` exports it yet.
    monkeypatch.setattr(ct, 'MultiSensoryIntegration', MultiSensoryIntegration, raising=False)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in doctest.DocTestFinder(recurse=False).find(MultiSensoryIntegration, globs={}):
        runner.run(test)
    assert runner.tries > 0
    assert runner.failures == 0
