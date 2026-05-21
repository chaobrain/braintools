# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Smoke + correctness tests for braintools.cogtask.

The goals here are (a) "can every pre-built task be constructed and produce
an (X, Y) trial with sensible shapes and label conventions", and (b) "does
the phase engine route compound phases (Sequence, Repeat, Parallel, If,
Switch) correctly". These tests deliberately stop short of training
convergence checks — they only catch wiring breakage.
"""

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

import braintools.cogtask as ct

# All pre-built tasks resolve durations via brainstate.environ.get_dt().
# Set a fixed dt for the whole test module.
brainstate.environ.set(dt=1.0 * u.ms)


# ---------------------------------------------------------------------------
# Pre-built tasks: parameter sets sized so all trials in a batch share shapes.
# Tasks with variable-duration phases (HierarchicalReasoning,
# IntervalDiscrimination, ReadySetGo) are excluded from the batch test below.
# ---------------------------------------------------------------------------
_FIXED_DURATION_TASKS = [
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

_VARIABLE_DURATION_TASKS = [
    ct.HierarchicalReasoning,
    ct.IntervalDiscrimination,
    ct.ReadySetGo,
]


def _construct(cls):
    """Default-construct a task with seed=0 so trials are deterministic."""
    return cls(seed=0)


def test_every_task_constructs_and_yields_a_trial():
    """sample_trial(0) must succeed for every pre-built task with sane shapes."""
    for cls in _FIXED_DURATION_TASKS + _VARIABLE_DURATION_TASKS:
        task = _construct(cls)
        X, Y, info = task.sample_trial(0)
        assert X.ndim == 2, f"{cls.__name__}: X must be (T, n_in), got {X.shape}"
        assert X.shape[1] == task.num_inputs, (
            f"{cls.__name__}: X feature dim {X.shape[1]} != num_inputs {task.num_inputs}"
        )
        if task.output_mode == 'categorical':
            assert Y.ndim == 1, f"{cls.__name__}: categorical Y must be 1D, got {Y.shape}"
        else:
            assert Y.ndim == 2, f"{cls.__name__}: vector Y must be 2D, got {Y.shape}"
            assert Y.shape[1] == task.num_outputs
        assert X.shape[0] == Y.shape[0], f"{cls.__name__}: T mismatch"
        assert 'phase_history' in info and len(info['phase_history']) > 0


def test_seed_reproducibility_per_index():
    """Same (seed, index) must give bit-identical trials."""
    a = ct.PerceptualDecisionMaking(seed=42)
    b = ct.PerceptualDecisionMaking(seed=42)
    Xa, Ya, _ = a.sample_trial(7)
    Xb, Yb, _ = b.sample_trial(7)
    np.testing.assert_array_equal(np.asarray(Ya), np.asarray(Yb))
    np.testing.assert_allclose(np.asarray(Xa), np.asarray(Xb), rtol=1e-5, atol=1e-5)


def test_different_indices_produce_different_trials():
    """Successive trial indices must vary (otherwise batch_sample is useless)."""
    task = ct.PerceptualDecisionMaking(seed=42)
    Y0 = np.asarray(task.sample_trial(0)[1])
    # Across 10 trials at least one ground_truth must differ from index 0's.
    differs = False
    for i in range(1, 10):
        Yi = np.asarray(task.sample_trial(i)[1])
        if not np.array_equal(Y0, Yi):
            differs = True
            break
    assert differs, "All trials produced identical Y across 10 indices"


def test_phase_engine_compound_dispatch():
    """Sequence/Repeat/Parallel/If/Switch must all execute (no silent no-ops).

    Pre-fix bug: If/Switch/While inherited Phase and were treated as leaves —
    their `execute` methods were never called and they silently produced zeros.
    """
    fix = ct.Feature(1, 'fixation')
    a = ct.Feature(2, 'a')
    b = ct.Feature(2, 'b')
    out = ct.Feature(2, 'out')

    # If: a branch must actually run and write into 'a'.
    phases = ct.concat([
        ct.Fixation(10 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0}),
        ct.If(
            lambda ctx: ctx['cond'],
            then=ct.Stimulus(20 * u.ms, inputs={'a': jnp.asarray([1.0, 1.0])}, outputs={'label': 0}),
            else_=ct.Stimulus(20 * u.ms, inputs={'b': jnp.asarray([1.0, 1.0])}, outputs={'label': 0}),
        ),
    ])
    task = ct.Task(
        phases=phases,
        input_features=fix + a + b,
        output_features=fix + out,
        trial_init=lambda ctx: ctx.update(cond=True),
        seed=0,
    )
    X, _, _ = task.sample_trial(0)
    X = np.asarray(X)
    # The 'a' channel must be non-zero in the second half (post-fixation).
    assert X[15:, 1:3].sum() > 0, "If(then=...) branch did not execute — phase engine treated it as a leaf"
    assert X[15:, 3:5].sum() == 0, "If(else_) branch ran when condition was True"


def test_parallel_child_durations_are_respected():
    """Parallel children should encode within their own declared duration,
    not the parent max. Tested by giving a short child and checking the
    tail of the parent slice is zero in the short channel.
    """
    fix = ct.Feature(1, 'fixation')
    short = ct.Feature(1, 'short')
    long_ = ct.Feature(1, 'long')

    phases = (
        ct.Stimulus(5 * u.ms, name='short_phase', inputs={'short': 1.0}, outputs={'label': 0})
        | ct.Stimulus(20 * u.ms, name='long_phase', inputs={'long': 1.0}, outputs={'label': 0})
    )
    task = ct.Task(
        phases=phases,
        input_features=fix + short + long_,
        output_features=fix + ct.Feature(1, 'out'),
        seed=0,
    )
    X, _, _ = task.sample_trial(0)
    X = np.asarray(X)
    # short channel writes 1.0 for the first 5 steps, then 0 afterwards.
    short_col = X[:, 1]
    assert short_col[:5].min() >= 1.0 - 1e-6
    assert short_col[5:].max() <= 1e-6, "Short Parallel child overran into long child's tail"
    # long channel writes 1.0 across all 20 steps.
    long_col = X[:, 2]
    assert long_col.min() >= 1.0 - 1e-6


def test_readysetgo_label_is_time_varying():
    """Pre-fix, the production phase label was always 1 (the time-of-call was
    always 0). Post-fix, the label crosses from 1 to 2 within the response
    window.
    """
    task = ct.ReadySetGo(seed=1)
    _, Y, info = task.sample_trial(0)
    Y = np.asarray(Y)
    # Find the production phase via history.
    prod_slice = None
    for name, start, end in info['phase_history']:
        if name == 'production':
            prod_slice = slice(int(start), int(end))
            break
    assert prod_slice is not None, "production phase not in history"
    window = Y[prod_slice]
    assert (window == 1).any() and (window == 2).any(), (
        f"production labels did not transition 1→2: unique values {np.unique(window)}"
    )


def test_gonogo_nogo_label_is_zero():
    """No-go trials should emit label 0 (withhold), not a distinct 'nogo' label."""
    task = ct.GoNoGo(seed=0)
    found_go = found_nogo = False
    for i in range(40):
        X, Y, info = task.sample_trial(i)
        is_go = bool(info['trial_state']['is_go'])
        # The response phase is the last entry whose name == 'response'.
        for name, s, e in info['phase_history']:
            if name == 'response':
                resp = np.asarray(Y)[int(s):int(e)]
                if is_go:
                    assert (resp == 1).all(), f"go trial {i}: response labels not all 1, got {np.unique(resp)}"
                    found_go = True
                else:
                    assert (resp == 0).all(), f"nogo trial {i}: response labels not all 0, got {np.unique(resp)}"
                    found_nogo = True
        if found_go and found_nogo:
            break
    assert found_go and found_nogo, "Did not sample both go and nogo trials"


def test_feature_helpers():
    """is_feature is a predicate; as_feature is a validator."""
    f = ct.Feature(3, 'x')
    assert ct.is_feature(f) is True
    assert ct.is_feature("not a feature") is False
    assert ct.as_feature(f) is f
    try:
        ct.as_feature("nope")
    except TypeError:
        pass
    else:
        raise AssertionError("as_feature should raise on non-Feature")


def test_pulse_decision_making_runs_through_repeat():
    """PulseDecisionMaking uses Repeat(...) with an encoder reading repeat_index.
    Verify the response label matches ground_truth+1 across multiple trials.
    """
    task = ct.PulseDecisionMaking(seed=3)
    for i in range(5):
        _, Y, info = task.sample_trial(i)
        gt = int(info['trial_state']['ground_truth'])
        # Last value in Y is the response label.
        last = int(np.asarray(Y)[-1])
        assert last == gt + 1, f"trial {i}: response label {last} != ground_truth+1={gt + 1}"


def test_batch_sample_vmap_for_fixed_duration_tasks():
    """A representative fixed-duration task from each family must batch via vmap."""
    for cls in (ct.PerceptualDecisionMaking, ct.DelayMatchSample, ct.GoNoGo, ct.AntiReach):
        task = cls(seed=0)
        X, Y = task.batch_sample(4)
        assert X.shape[1] == 4, f"{cls.__name__}: batch axis wrong, X.shape={X.shape}"
        assert Y.shape[1] == 4
        assert X.shape[2] == task.num_inputs


def test_batch_sample_is_reproducible_with_start_index():
    """Same seed + start_index → same batch; different start_index → different."""
    a = ct.PerceptualDecisionMaking(seed=99)
    b = ct.PerceptualDecisionMaking(seed=99)
    Xa, Ya = a.batch_sample(4, start_index=0)
    Xb, Yb = b.batch_sample(4, start_index=0)
    np.testing.assert_array_equal(np.asarray(Ya), np.asarray(Yb))
    _, Yc = a.batch_sample(4, start_index=100)
    assert not np.array_equal(np.asarray(Ya), np.asarray(Yc)), (
        "batch_sample(start_index=0) and start_index=100 produced identical labels"
    )


def test_hierarchical_reasoning_rule_cue_is_in_inputs():
    """When show_rule_cue=True the input dim grows by 2 (rule one-hot)."""
    no_cue = ct.HierarchicalReasoning(show_rule_cue=False, seed=0)
    with_cue = ct.HierarchicalReasoning(show_rule_cue=True, seed=0)
    assert with_cue.num_inputs == no_cue.num_inputs + 2


if __name__ == '__main__':
    test_every_task_constructs_and_yields_a_trial()
    test_seed_reproducibility_per_index()
    test_different_indices_produce_different_trials()
    test_phase_engine_compound_dispatch()
    test_parallel_child_durations_are_respected()
    test_readysetgo_label_is_time_varying()
    test_gonogo_nogo_label_is_zero()
    test_feature_helpers()
    test_pulse_decision_making_runs_through_repeat()
    test_batch_sample_vmap_for_fixed_duration_tasks()
    test_batch_sample_is_reproducible_with_start_index()
    test_hierarchical_reasoning_rule_cue_is_in_inputs()
    print("All cogtask smoke tests passed.")
