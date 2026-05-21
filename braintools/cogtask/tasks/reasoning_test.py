# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Per-task semantic tests for the reasoning family."""

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

import braintools.cogtask as ct


def _phase_window(info, target):
    for name, s, e in info['phase_history']:
        if name == target:
            return int(s), int(e)
    raise AssertionError(f"no '{target}' phase in {[n for n, _, _ in info['phase_history']]}")


# ---------------------------------------------------------------------------
# HierarchicalReasoning
# ---------------------------------------------------------------------------

def test_hierarchical_reasoning_show_rule_cue_adds_rule_input_dim():
    no_cue = ct.HierarchicalReasoning(seed=0, show_rule_cue=False)
    with_cue = ct.HierarchicalReasoning(seed=0, show_rule_cue=True)
    assert with_cue.num_inputs == no_cue.num_inputs + 2


def test_hierarchical_reasoning_response_label_matches_ground_truth():
    task = ct.HierarchicalReasoning(seed=0)
    for i in range(5):
        _, Y, info = task.sample_trial(i)
        gt = int(info['trial_state']['ground_truth'])
        s, e = _phase_window(info, 'response')
        assert (np.asarray(Y)[s:e] == gt + 1).all()


def test_hierarchical_reasoning_rule_block_alternates_every_100_trials():
    task = ct.HierarchicalReasoning(seed=0)
    rules = []
    for i in [0, 50, 100, 150, 200]:
        _, _, info = task.sample_trial(i)
        rules.append(int(info['trial_state']['rule']))
    # trial 0..99 → rule 0; 100..199 → rule 1; 200..299 → rule 0
    assert rules == [0, 0, 1, 1, 0]


def test_hierarchical_reasoning_ground_truth_rule_a_short_delay_goes_toward():
    task = ct.HierarchicalReasoning(seed=0, delay_threshold=10000.0)  # always "short"
    # Force trial_index into rule-0 block via index < 100.
    for i in range(5):
        _, _, info = task.sample_trial(i)
        state = info['trial_state']
        assert int(state['rule']) == 0
        # short delay + rule 0 → toward (flash)
        assert int(state['ground_truth']) == int(state['flash2_loc'])


# ---------------------------------------------------------------------------
# ProbabilisticReasoning
# ---------------------------------------------------------------------------

def test_probabilistic_reasoning_ground_truth_follows_total_evidence_sign():
    task = ct.ProbabilisticReasoning(seed=42)
    for i in range(10):
        _, _, info = task.sample_trial(i)
        s = info['trial_state']
        total = float(s['total_evidence'])
        gt = int(s['ground_truth'])
        if total > 0:
            assert gt == 1
        else:
            assert gt == 0


def test_probabilistic_reasoning_repeats_cue_block_num_cues_times():
    task = ct.ProbabilisticReasoning(seed=0, num_cues=4)
    _, _, info = task.sample_trial(0)
    names = [n for n, _, _ in info['phase_history']]
    assert names.count('cue') == 4
    assert names.count('cue_delay') == 4


def test_probabilistic_reasoning_response_label_matches_ground_truth_plus_one():
    task = ct.ProbabilisticReasoning(seed=0)
    for i in range(6):
        _, Y, info = task.sample_trial(i)
        gt = int(info['trial_state']['ground_truth'])
        s, e = _phase_window(info, 'response')
        assert (np.asarray(Y)[s:e] == gt + 1).all()
