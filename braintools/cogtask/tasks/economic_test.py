# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Per-task semantic tests for the economic (value-based) decision task.

Pins down the offer encoding (NeuroGym's ``fixation, a1, b1, a2, b2, n1, n2``
layout), the value comparison ``relative_value * nA`` vs ``nB`` with random
tie breaking, the label convention (chosen offer slot + 1) and the fixed /
variable offer-period paths.
"""

import doctest

import brainunit as u
import numpy as np
import pytest

import braintools.cogtask as ct
from braintools.cogtask.tasks.economic import EconomicDecisionMaking

_DT10 = 10.0 * u.ms


def _spans(info):
    return {name: (int(s), int(e)) for name, s, e in info['phase_history']}


def _fixed(**kwargs):
    kwargs.setdefault('dt', _DT10)
    kwargs.setdefault('t_offer', 1000.0 * u.ms)
    return EconomicDecisionMaking(**kwargs)


def test_econ_features_and_shapes():
    task = _fixed(seed=0)
    assert task.num_inputs == 7
    assert task.num_outputs == 3
    assert task.num_classes == 3
    assert not task.is_variable_length
    X, Y, _ = task.sample_trial(0)
    # 1500 + 1000 + 750 ms at dt = 10 ms.
    assert X.shape == (325, 7)
    assert Y.shape == (325,)
    names = [f.name for f in task.input_features]
    assert names == ['fixation', 'juice1', 'juice2', 'quantity']


def test_econ_phase_order():
    task = _fixed(seed=0)
    _, _, info = task.sample_trial(0)
    names = [n for n, _, _ in info['phase_history'] if n != 'Sequence']
    assert names == ['fixation', 'offer', 'decision']
    sp = _spans(info)
    assert sp['fixation'] == (0, 150)
    assert sp['offer'] == (150, 250)
    assert sp['decision'] == (250, 325)


@pytest.mark.parametrize("offer, rho, choose_a", [
    ((0, 1), 2.2, True),    # 0B vs 1A
    ((1, 3), 2.2, True),    # 1B vs 3A
    ((1, 1), 2.2, True),    # 1B vs 1A: 2.2 > 1
    ((2, 1), 2.2, True),    # 2B vs 1A: 2.2 > 2
    ((3, 1), 2.2, False),   # 3B vs 1A: 3 > 2.2
    ((6, 1), 2.2, False),
    ((2, 0), 2.2, False),   # 2B vs nothing
    ((3, 1), 4.0, True),    # a stronger preference for A flips the choice
    ((2, 1), 1.5, False),
])
def test_econ_ground_truth_picks_higher_value(offer, rho, choose_a):
    task = _fixed(seed=2, offers=(offer,), relative_value=rho)
    orders = set()
    for i in range(6):
        _, Y, info = task.sample_trial(i)
        s = info['trial_state']
        assert float(s['nB']) == offer[0] and float(s['nA']) == offer[1]
        assert float(s['value_A']) == pytest.approx(rho * offer[1])
        assert float(s['value_B']) == pytest.approx(offer[0])
        assert not bool(s['tie'])
        assert bool(s['choose_A']) is choose_a
        order = int(s['juice_order'])
        orders.add(order)
        slot_of_a = order
        expected_slot = slot_of_a if choose_a else 1 - slot_of_a
        assert int(s['ground_truth']) == expected_slot
        ds, de = _spans(info)['decision']
        Y = np.asarray(Y)
        assert (Y[:ds] == 0).all()
        assert (Y[ds:de] == expected_slot + 1).all()
    # Both spatial assignments of juice A occur.
    assert orders == {0, 1}


def test_econ_ties_are_broken_randomly():
    task = _fixed(seed=2, offers=((2, 1),), relative_value=2.0)
    _, _, meta = task.batch_sample(200, return_meta=True)
    assert np.asarray(meta['tie']).all()
    frac_a = np.mean(np.asarray(meta['choose_A']))
    assert 0.35 < frac_a < 0.65
    # The label always points to the slot holding the chosen juice.
    gt = np.asarray(meta['ground_truth'])
    order = np.asarray(meta['juice_order'])
    choose_a = np.asarray(meta['choose_A'])
    np.testing.assert_array_equal(gt, np.where(choose_a, order, 1 - order))


def test_econ_phase_inputs():
    task = _fixed(seed=3)
    for i in range(6):
        X, _, info = task.sample_trial(i)
        X = np.asarray(X)
        s = info['trial_state']
        sp = _spans(info)
        fs, fe = sp['fixation']
        np.testing.assert_array_equal(X[fs:fe, 0], 1.0)
        np.testing.assert_array_equal(X[fs:fe, 1:], 0.0)
        os_, oe = sp['offer']
        nA, nB = float(s['nA']), float(s['nB'])
        if int(s['juice_order']) == 0:   # offer 1 = A, offer 2 = B
            expected = [1.0, 1.0, 0.0, 0.0, 1.0, nA / 5.0, nB / 5.0]
        else:                            # offer 1 = B, offer 2 = A
            expected = [1.0, 0.0, 1.0, 1.0, 0.0, nB / 5.0, nA / 5.0]
        np.testing.assert_allclose(X[os_:oe], np.broadcast_to(expected, (oe - os_, 7)), atol=1e-6)
        ds, de = sp['decision']
        np.testing.assert_array_equal(X[ds:de], 0.0)


def test_econ_quantity_scale():
    task = _fixed(seed=3, offers=((3, 1),), quantity_scale=1.0)
    X, _, info = task.sample_trial(0)
    os_, _ = _spans(info)['offer']
    q = np.sort(np.asarray(X)[os_, 5:7])
    np.testing.assert_allclose(q, [1.0, 3.0])


def test_econ_noise_only_on_quantities():
    noisy = _fixed(seed=4, noise_sigma=1.0 * u.ms ** 0.5)
    clean = _fixed(seed=4)
    X, _, info = noisy.sample_trial(0)
    X0, _, _ = clean.sample_trial(0)
    X, X0 = np.asarray(X), np.asarray(X0)
    os_, oe = _spans(info)['offer']
    np.testing.assert_array_equal(X[:, :5], X0[:, :5])
    resid = X[os_:oe, 5:7] - X0[os_:oe, 5:7]
    # sigma / sqrt(dt) with dt = 10 ms.
    assert np.std(resid) == pytest.approx(1.0 / np.sqrt(10.0), rel=0.15)
    np.testing.assert_array_equal(X[oe:], X0[oe:])


def test_econ_variable_offer_period():
    task = EconomicDecisionMaking(seed=5, dt=_DT10)
    assert task.is_variable_length
    # 1500 + max 2000 + 750 ms.
    assert task.max_trial_duration() == 425
    X, Y, info = task.sample_trial(0)
    assert X.shape == (425, 7) and Y.shape == (425,)
    s = info['trial_state']
    assert 1000.0 <= float(s['offer_duration']) <= 2000.0
    sp = _spans(info)
    assert sp['offer'][1] - sp['offer'][0] == int(float(s['offer_duration']) / 10.0)
    mask = np.asarray(info['mask'])
    assert mask.sum() == sp['decision'][1]
    assert (np.asarray(Y)[sp['decision'][0]:sp['decision'][1]] == int(s['ground_truth']) + 1).all()

    _, _, meta = task.batch_sample(200, return_meta=True)
    dur = np.asarray(meta['offer_duration'])
    assert dur.min() >= 1000.0 and dur.max() <= 2000.0
    assert dur.mean() == pytest.approx(1500.0, abs=60.0)


def test_econ_offer_statistics():
    task = _fixed(seed=6, t_fixation=10 * u.ms, t_offer=10 * u.ms, t_decision=10 * u.ms)
    _, _, meta = task.batch_sample(400, return_meta=True)
    pairs = set(zip(np.asarray(meta['nB']).tolist(), np.asarray(meta['nA']).tolist()))
    assert pairs == {(0.0, 1.0), (1.0, 3.0), (1.0, 2.0), (1.0, 1.0), (2.0, 1.0),
                     (3.0, 1.0), (4.0, 1.0), (6.0, 1.0), (2.0, 0.0)}
    assert 0.4 < np.mean(np.asarray(meta['juice_order'])) < 0.6
    # With rho = 2.2 the default offers never tie, and A wins on 5 of 9 offers.
    assert not np.asarray(meta['tie']).any()
    assert np.mean(np.asarray(meta['choose_A'])) == pytest.approx(5 / 9, abs=0.08)


def test_econ_seeded_reproducibility_and_batch_consistency():
    a = _fixed(seed=7)
    b = _fixed(seed=7)
    c = _fixed(seed=8)
    Xa, Ya, _ = a.sample_trial(2)
    Xb, Yb, _ = b.sample_trial(2)
    np.testing.assert_array_equal(np.asarray(Xa), np.asarray(Xb))
    np.testing.assert_array_equal(np.asarray(Ya), np.asarray(Yb))
    states = [c.sample_trial(i)[2]['trial_state'] for i in range(8)]
    ref = [a.sample_trial(i)[2]['trial_state'] for i in range(8)]
    key = lambda s: (float(s['nA']), float(s['nB']), int(s['juice_order']))  # noqa: E731
    assert [key(s) for s in states] != [key(s) for s in ref]

    X, Y = a.batch_sample(3)
    assert X.shape == (325, 3, 7) and Y.shape == (325, 3)
    for i in range(3):
        Xi, Yi, _ = a.sample_trial(i)
        np.testing.assert_array_equal(np.asarray(X)[:, i], np.asarray(Xi))
        np.testing.assert_array_equal(np.asarray(Y)[:, i], np.asarray(Yi))
    Xbf, Ybf = a.batch_sample(3, time_first=False)
    assert Xbf.shape == (3, 325, 7) and Ybf.shape == (3, 325)


@pytest.mark.parametrize("kwargs", [
    dict(offers=()),
    dict(offers=((1, 2, 3),)),
    dict(offers=((-1, 2),)),
    dict(relative_value=0.0),
    dict(relative_value=-2.2),
    dict(t_offer=(1000 * u.ms,)),
    dict(t_offer=(2000 * u.ms, 1000 * u.ms)),
    dict(t_offer=(0 * u.ms, 1000 * u.ms)),
])
def test_econ_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        EconomicDecisionMaking(**kwargs)


def test_econ_rejects_unitless_offer_duration():
    with pytest.raises(Exception):
        EconomicDecisionMaking(t_offer=1000.0)


def test_econ_docstring_examples(monkeypatch):
    monkeypatch.setattr(ct, 'EconomicDecisionMaking', EconomicDecisionMaking, raising=False)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in doctest.DocTestFinder(recurse=False).find(EconomicDecisionMaking, globs={}):
        runner.run(test)
    assert runner.tries > 0
    assert runner.failures == 0
