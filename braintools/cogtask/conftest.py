# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Shared fixtures and task registries for braintools.cogtask tests.

A package-scoped autouse fixture pins ``brainstate.environ`` to ``dt=1.0 ms``
so phase durations resolve to a stable timestep count across tests. The task
lists below are reused by parametrized tests in ``cogtask_test.py``,
``task_test.py``, and the per-family files under ``tasks/``.
"""

import brainstate
import brainunit as u
import pytest

import braintools.cogtask as ct


@pytest.fixture(scope="package", autouse=True)
def _pin_dt():
    # Use environ.context so the dt change is restored on teardown. Without
    # this, the global mutation leaks into other test packages run later in
    # the same session (e.g. braintools.metric, which expects unitless dt).
    with brainstate.environ.context(dt=1.0 * u.ms):
        yield


# Tasks whose phase durations are fixed at construction time. All trials share
# the same total length, so they can be exercised through ``batch_sample`` and
# the vmap path. Used as the ``parametrize`` input for the "every task is
# constructible" sweep.
FIXED_DURATION_TASKS = [
    ct.PerceptualDecisionMaking,
    ct.PerceptualDecisionMakingDelayResponse,
    ct.ContextDecisionMaking,
    ct.SingleContextDecisionMaking,
    ct.PulseDecisionMaking,
    ct.DelayMatchSample,
    ct.DualDelayMatchSample,
    ct.DelayComparison,
    ct.DelayMatchCategory,
    ct.DelayPairedAssociation,
    ct.GoNoGo,
    ct.PostDecisionWager,
    ct.DelayDirectionReproduction,
    ct.ImmediateDirectionReproduction,
    ct.DelayDirectionClassification,
    ct.ImmediateDirectionClassification,
    ct.ProbabilisticReasoning,
    ct.AntiReach,
    ct.Reaching1D,
    ct.EvidenceAccumulation,
]

# Tasks with phases whose actual duration depends on a value sampled in
# ``trial_init`` (they contain a VariableDuration / conditional phase). They
# run in packed mode: buffers are sized to the static ``max_trial_duration``
# and a per-timestep mask marks the valid steps, so they ARE vmap-safe via
# ``batch_sample(return_mask=True)`` — see
# ``task_variable_length_test.test_migrated_builtin_tasks_packed``. They are
# kept separate only because their valid length varies per trial.
VARIABLE_DURATION_TASKS = [
    ct.HierarchicalReasoning,
    ct.IntervalDiscrimination,
    ct.ReadySetGo,
]

ALL_TASKS = FIXED_DURATION_TASKS + VARIABLE_DURATION_TASKS
