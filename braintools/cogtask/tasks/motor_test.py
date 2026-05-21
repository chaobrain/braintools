# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Per-task semantic tests for the motor / sensorimotor family."""

import numpy as np
import pytest

import braintools.cogtask as ct


def _phase_window(info, target):
    for name, s, e in info['phase_history']:
        if name == target:
            return int(s), int(e)
    raise AssertionError(f"no '{target}' phase in {[n for n, _, _ in info['phase_history']]}")


# ---------------------------------------------------------------------------
# AntiReach
# ---------------------------------------------------------------------------

def test_anti_reach_pro_trial_responds_toward_stimulus():
    """Force anti_prob=0.0 so every trial is a pro trial: response location
    equals stimulus location."""
    task = ct.AntiReach(seed=0, anti_prob=0.0)
    for i in range(5):
        _, _, info = task.sample_trial(i)
        state = info['trial_state']
        assert bool(state['is_anti']) is False
        assert int(state['ground_truth']) == int(state['stim_loc'])


def test_anti_reach_anti_trial_responds_to_opposite_location():
    """Force anti_prob=1.0 so every trial is an anti trial: response location
    is the stimulus's opposite (offset by num_locations//2)."""
    task = ct.AntiReach(seed=0, anti_prob=1.0, num_locations=8)
    for i in range(5):
        _, _, info = task.sample_trial(i)
        state = info['trial_state']
        assert bool(state['is_anti']) is True
        expected = (int(state['stim_loc']) + 4) % 8
        assert int(state['ground_truth']) == expected


def test_anti_reach_rule_input_is_one_on_anti_zero_on_pro():
    task = ct.AntiReach(seed=0)
    for i in range(20):
        _, _, info = task.sample_trial(i)
        state = info['trial_state']
        if bool(state['is_anti']):
            assert int(state['rule']) == 1
        else:
            assert int(state['rule']) == 0


def test_anti_reach_response_label_matches_ground_truth_plus_one():
    task = ct.AntiReach(seed=0)
    for i in range(5):
        _, Y, info = task.sample_trial(i)
        gt = int(info['trial_state']['ground_truth'])
        s, e = _phase_window(info, 'response')
        assert (np.asarray(Y)[s:e] == gt + 1).all()


@pytest.mark.parametrize("num_locations", [4, 6, 8])
def test_anti_reach_num_locations_drives_io_dim(num_locations):
    task = ct.AntiReach(seed=0, num_locations=num_locations)
    # input: fix(1) + stim(num_locations) + rule(2)
    assert task.num_inputs == 1 + num_locations + 2
    # output: fix(1) + response(num_locations)
    assert task.num_outputs == 1 + num_locations


# ---------------------------------------------------------------------------
# Reaching1D
# ---------------------------------------------------------------------------

def test_reaching1d_ground_truth_equals_target_loc():
    task = ct.Reaching1D(seed=0)
    for i in range(5):
        _, _, info = task.sample_trial(i)
        s = info['trial_state']
        assert int(s['ground_truth']) == int(s['target_loc'])


def test_reaching1d_response_label_matches_target_loc_plus_one():
    task = ct.Reaching1D(seed=0)
    for i in range(5):
        _, Y, info = task.sample_trial(i)
        gt = int(info['trial_state']['ground_truth'])
        s, e = _phase_window(info, 'response')
        assert (np.asarray(Y)[s:e] == gt + 1).all()


# ---------------------------------------------------------------------------
# EvidenceAccumulation
# ---------------------------------------------------------------------------

def test_evidence_accumulation_response_label_matches_ground_truth():
    task = ct.EvidenceAccumulation(seed=0)
    for i in range(5):
        _, Y, info = task.sample_trial(i)
        gt = int(info['trial_state']['ground_truth'])
        s, e = _phase_window(info, 'response')
        assert (np.asarray(Y)[s:e] == gt + 1).all()


def test_evidence_accumulation_pop_per_choice_drives_input_dim():
    task = ct.EvidenceAccumulation(seed=0, num_choices=3, pop_per_choice=5)
    # fix(1) + evidence(3*5) = 16
    assert task.num_inputs == 1 + 15
