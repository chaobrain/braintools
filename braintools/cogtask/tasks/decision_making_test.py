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

import brainunit as u
import numpy as np
import pytest

import braintools.cogtask as ct


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
