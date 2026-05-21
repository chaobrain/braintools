# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``braintools.cogtask.label``."""

import numpy as np

from braintools.cogtask.context import Context
from braintools.cogtask.label import label, match_label, comparison_label


def _ctx(**state):
    ctx = Context()
    for k, v in state.items():
        ctx[k] = v
    return ctx


def test_label_int_returns_constant():
    f = label(7)
    assert f(_ctx(), None) == 7
    assert f(_ctx(some='garbage'), None) == 7


def test_label_string_reads_from_context():
    f = label('answer')
    assert f(_ctx(answer=3), None) == 3


def test_label_string_missing_key_returns_zero():
    f = label('answer')
    assert f(_ctx(), None) == 0


def test_label_callable_is_invoked_with_context():
    f = label(lambda ctx: ctx['choice'] * 2 + 1)
    assert f(_ctx(choice=4), None) == 9


def test_match_label_returns_match_when_true():
    f = match_label('is_match', match_label=1, nonmatch_label=2)
    assert int(np.asarray(f(_ctx(is_match=True), None))) == 1


def test_match_label_returns_nonmatch_when_false():
    f = match_label('is_match', match_label=1, nonmatch_label=2)
    assert int(np.asarray(f(_ctx(is_match=False), None))) == 2


def test_match_label_default_is_nonmatch_when_key_missing():
    f = match_label('is_match')
    assert int(np.asarray(f(_ctx(), None))) == 2


def test_comparison_label_returns_greater_when_true():
    f = comparison_label('cmp', greater_label=5, less_label=6)
    assert int(np.asarray(f(_ctx(cmp=True), None))) == 5


def test_comparison_label_returns_less_when_false():
    f = comparison_label('cmp', greater_label=5, less_label=6)
    assert int(np.asarray(f(_ctx(cmp=False), None))) == 6


def test_label_name_is_diagnostic():
    assert label(3).__name__ == 'label(3)'
    assert label('foo').__name__ == "label('foo')"
    assert match_label('is_match').__name__ == "match_label('is_match')"
    assert comparison_label('cmp').__name__ == "comparison_label('cmp')"
