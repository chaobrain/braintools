# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Regression tests for bugs found in the 2026-06-18 ``braintools.quad`` audit.

Each test here reproduces a specific confirmed bug and asserts the *correct*
behaviour, so it fails against the pre-fix code and passes after the fix.
"""

import math

import brainstate
import brainunit as u
import numpy as np

import braintools


# ---------------------------------------------------------------------------
# Bug 1: ode_expeuler_step divided by the Jacobian unit instead of the state
# unit, raising "exprel requires a dimensionless x" for unitful states.
# Exponential Euler is exact for linear ODEs: y1 = y0 * exp(A*dt).
# ---------------------------------------------------------------------------
def test_ode_expeuler_step_with_units():
    tau = 10.0 * u.ms

    def f(v, t):
        return -v / tau

    v0 = -65.0 * u.mV
    with brainstate.environ.context(dt=0.1 * u.ms):
        v1 = braintools.quad.ode_expeuler_step(f, v0, 0.0 * u.ms)

    expected = -65.0 * math.exp(-0.01)  # A*dt = -(1/10ms)*0.1ms = -0.01
    assert u.get_unit(v1) == u.mV
    assert np.allclose(u.get_mantissa(v1), expected, rtol=1e-6)


def test_ode_expeuler_step_dimensionless_unchanged():
    # Guard: the fix must not change the dimensionless result.
    def f(x, t):
        return -x

    with brainstate.environ.context(dt=0.1):
        x1 = braintools.quad.ode_expeuler_step(f, 1.0, 0.0)
    assert np.allclose(x1, math.exp(-0.1), rtol=1e-6)


# ---------------------------------------------------------------------------
# Bug 2 + 3: sde_expeuler_step used randn_like(args[0]) (crashes with no extra
# args) and the wrong unit. With zero diffusion it must equal exp-Euler drift.
# ---------------------------------------------------------------------------
def test_sde_expeuler_step_no_extra_args():
    def df(x, t):
        return -x

    def dg(x, t):
        return 0.0

    with brainstate.environ.context(dt=0.1):
        x1 = braintools.quad.sde_expeuler_step(df, dg, 1.0, 0.0)
    # Zero diffusion -> deterministic exp-Euler drift = exp(-0.1)
    assert np.allclose(x1, math.exp(-0.1), rtol=1e-6)


def test_sde_expeuler_step_with_units():
    tau = 10.0 * u.ms

    def df(v, t):
        return -v / tau

    def dg(v, t):
        return 0.0 * u.mV / u.ms ** 0.5

    v0 = -65.0 * u.mV
    with brainstate.environ.context(dt=0.1 * u.ms):
        v1 = braintools.quad.sde_expeuler_step(df, dg, v0, 0.0 * u.ms)
    assert u.get_unit(v1) == u.mV
    assert np.allclose(u.get_mantissa(v1), -65.0 * math.exp(-0.01), rtol=1e-6)


# ---------------------------------------------------------------------------
# Bug 4: sde_euler_step used jnp.sqrt(dt), which rejects unitful dt.
# With zero diffusion the step is deterministic Euler.
# ---------------------------------------------------------------------------
def test_sde_euler_step_with_units():
    tau = 10.0 * u.ms

    def df(v, t):
        return -v / tau

    def dg(v, t):
        return 0.0 * u.mV / u.ms ** 0.5

    v0 = -65.0 * u.mV
    with brainstate.environ.context(dt=0.1 * u.ms):
        v1 = braintools.quad.sde_euler_step(df, dg, v0, 0.0 * u.ms)
    expected = -65.0 + (65.0 / 10.0) * 0.1  # v0 + (-v0/tau)*dt
    assert u.get_unit(v1) == u.mV
    assert np.allclose(u.get_mantissa(v1), expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Bug 5: sde_tamed_euler_step added a dimensionless 1 to a unitful dt*|f|.
# ---------------------------------------------------------------------------
def test_sde_tamed_euler_step_with_units():
    tau = 10.0 * u.ms

    def df(v, t):
        return -v / tau

    def dg(v, t):
        return 0.0 * u.mV / u.ms ** 0.5

    v0 = -65.0 * u.mV
    dt = 0.1
    with brainstate.environ.context(dt=dt * u.ms):
        v1 = braintools.quad.sde_tamed_euler_step(df, dg, v0, 0.0 * u.ms)
    f_val = 65.0 / 10.0  # mantissa of -v0/tau in mV/ms
    tamed = f_val / (1.0 + dt * abs(f_val))
    expected = -65.0 + tamed * dt
    assert u.get_unit(v1) == u.mV
    assert np.allclose(u.get_mantissa(v1), expected, rtol=1e-6)


def test_sde_tamed_euler_step_dimensionless_unchanged():
    # Guard: dimensionless taming result must be unchanged by the fix.
    def df(y, t):
        return y ** 3

    def dg(y, t):
        return 0.0

    y0 = 10.0
    dt = 0.1
    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.sde_tamed_euler_step(df, dg, y0, 0.0)
    f_val = y0 ** 3
    expected = y0 + (f_val / (1.0 + dt * abs(f_val))) * dt
    assert np.allclose(y1, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Bug 14: the primary (propagated) solution of an embedded method must be
# identical whether or not the error estimate is requested. This characterises
# the FSAL refactor of ode_dopri5_step (the extra stage must not change y).
# ---------------------------------------------------------------------------
def test_embedded_primary_matches_return_error_branch():
    a = -0.9

    def f(y, t):
        return a * y

    methods = [
        braintools.quad.ode_rk45_step,
        braintools.quad.ode_dopri5_step,
        braintools.quad.ode_rkf45_step,
        braintools.quad.ode_rk23_step,
        braintools.quad.ode_dopri8_step,
    ]
    with brainstate.environ.context(dt=0.1):
        for fn in methods:
            y_plain = fn(f, 1.0, 0.0)
            y_err, _ = fn(f, 1.0, 0.0, return_error=True)
            assert np.allclose(y_plain, y_err), fn.__name__


# ---------------------------------------------------------------------------
# Bug 15: a disallowed (non-float) input must raise a clean ValueError, not an
# AttributeError while formatting the message (Python scalars have no .dtype).
# ---------------------------------------------------------------------------
def test_ode_expeuler_step_rejects_integer_scalar():
    def f(x, t):
        return -x

    with brainstate.environ.context(dt=0.1):
        try:
            braintools.quad.ode_expeuler_step(f, 1, 0.0)  # python int
        except ValueError:
            pass
        except Exception as e:  # pragma: no cover - explicit failure path
            raise AssertionError(f"expected ValueError, got {type(e).__name__}: {e}")
        else:
            raise AssertionError("expected ValueError for integer input")


def test_sde_expeuler_step_rejects_integer_scalar():
    def df(x, t):
        return -x

    def dg(x, t):
        return 0.0

    with brainstate.environ.context(dt=0.1):
        try:
            braintools.quad.sde_expeuler_step(df, dg, 1, 0.0)  # python int
        except ValueError:
            pass
        except Exception as e:  # pragma: no cover - explicit failure path
            raise AssertionError(f"expected ValueError, got {type(e).__name__}: {e}")
        else:
            raise AssertionError("expected ValueError for integer input")
