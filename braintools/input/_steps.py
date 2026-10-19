# -*- coding: utf-8 -*-

"""Conversion of times to whole numbers of simulation steps."""

import numpy as np

__all__ = ['ceil_steps', 'floor_steps']

# Ratios this close to an integer are treated as that integer, so that e.g.
# 0.3 ms / 0.1 ms (= 2.9999999999999996) is 3 steps, not 2 or 4.
_REL_TOL = 1e-6


def _ratio(t, dt):
    return float(np.asarray(t / dt, dtype=np.float64))


def ceil_steps(t, dt) -> int:
    """Number of steps of size ``dt`` needed to cover ``t`` (ceil, float-tolerant)."""
    r = _ratio(t, dt)
    n = round(r)
    if abs(r - n) <= _REL_TOL * max(1.0, abs(r)):
        return int(n)
    return int(np.ceil(r))


def floor_steps(t, dt) -> int:
    """Index of the step containing time ``t`` (floor, float-tolerant)."""
    r = _ratio(t, dt)
    n = round(r)
    if abs(r - n) <= _REL_TOL * max(1.0, abs(r)):
        return int(n)
    return int(np.floor(r))
