# `braintools.init` — Issues Found & Proposed Solutions (2026-06-18)

This document records issues identified during a senior-developer / JAX-expert
review of the `braintools/init/` module and its documentation, together with the
solutions applied. Each issue lists a minimal reproduction, the root cause, and
the fix.

The module had already been audited previously (the source contains references
to earlier fixes such as `bug C1` and `bug H2/M8`); the issues below are the
*remaining* defects found in this pass. All findings were verified empirically
before fixing, and regression tests were added for every behavioural change.

Severity legend: **High** = wrong result / crash on a documented use case;
**Medium** = crash on a reasonable edge case or contract violation;
**Low** = documentation / cosmetic.

---

## B1 (High) — `TruncatedNormal` crashes on array-valued `std`

**File:** `braintools/init/_init_basic.py` — `TruncatedNormal.__call__`

**Symptom**

```python
import numpy as np, brainunit as u
from braintools.init import TruncatedNormal
init = TruncatedNormal(
    mean=np.array([0., 1.]) * u.mV,
    std=np.array([0.5, 0.2]) * u.mV,
    low=-1. * u.mV, high=2. * u.mV,
)
init(2, rng=np.random.default_rng(0))
# ValueError: The truth value of an array with more than one element is ambiguous.
```

**Root cause**

The degenerate-distribution shortcut used a scalar comparison:

```python
if std == 0:
    samples = jnp.full(size, mean, dtype=float)
else:
    ...
```

When `std` is an array (a legitimate per-element parametrization that every
sibling distribution — `Normal`, `LogNormal`, … — accepts), `std == 0` is an
array and `if` raises `ValueError: truth value ... ambiguous`. Even when no
element is zero, the code path is unreachable for arrays.

**Fix**

Vectorize the computation so the `std == 0` case is handled element-wise via a
"safe" denominator, removing the scalar `if`:

```python
mean_arr = jnp.asarray(mean, dtype=float)
std_arr = jnp.asarray(std, dtype=float)
# Avoid divide-by-zero in the standardized bounds; std==0 contributes 0 below.
safe_std = jnp.where(std_arr == 0, 1.0, std_arr)
a = -jnp.inf if lo is None else (lo - mean_arr) / safe_std
b =  jnp.inf if hi is None else (hi - mean_arr) / safe_std
cdf_a = 0.0 if lo is None else ndtr(a)
cdf_b = 1.0 if hi is None else ndtr(b)
p = jnp.clip(cdf_a + rng.uniform(0., 1., size) * (cdf_b - cdf_a), 1e-7, 1. - 1e-7)
samples = mean_arr + std_arr * ndtri(p)   # std==0 -> exactly mean
```

For `std == 0` the term `std_arr * ndtri(p)` is `0`, so `samples == mean`, then
the existing clip to `[low, high]` clamps the mean into range — identical to the
previous scalar behaviour, now also correct for arrays and mixed (some-zero)
`std`.

---

## B2 (Medium) — `param()` returns inconsistent types for `State` inputs

**File:** `braintools/init/_init_base.py` — `param`

**Symptom**

```python
import numpy as np, brainunit as u
from brainstate import State
from brainstate.nn import Param
from braintools.init import param

param(State(np.ones(10) * u.nS), 10)                       # -> Quantity (bare value!)
param(State(np.ones((4, 10)) * u.nS), 10, batch_size=4)    # -> Quantity (bare value!)
param(Param(np.ones((3, 5))),       5,  batch_size=3)      # -> Param  (container)
```

The `param` docstring states: *"`State`/`Param` inputs are returned with their
value updated."* The `Param` path honours this (always returns the `Param`), but
the `State` path only returns the `State` in the single sub-case where a batch
axis must be *expanded*; in the no-batch and already-batched sub-cases it returns
the bare underlying value. The return type therefore depends on the batch shape,
which is surprising and contradicts the documented contract.

**Root cause**

`State` was wrapped back into the container (`init.restore_value(...)`) only
inside the `param_value.ndim <= len(sizes)` branch:

```python
param_value = init.value if isinstance(init, State) else init
if batch_size is not None and not batch_applied_by_callable:
    if param_value.ndim <= len(sizes):
        ...
        if isinstance(init, State):
            init.restore_value(param_value)
            param_value = init        # <- only here
    else:
        ...                           # already-batched: never re-wraps
return param_value                    # <- no-batch: returns bare value
```

**Fix**

Hoist the `State` re-wrap out of the batch-expansion branch so every path
updates the `State`'s value and returns the `State`, mirroring the `Param` path:

```python
if batch_size is not None and not batch_applied_by_callable:
    if param_value.ndim <= len(sizes):
        param_value = _expand_params_to_match_sizes(param_value, sizes)
        param_value = u.math.repeat(u.math.expand_dims(param_value, 0), batch_size, axis=0)
    else:
        if param_value.shape[0] != batch_size:
            raise ValueError(...)
if isinstance(init, State):
    init.restore_value(param_value)
    return init
return param_value
```

Additionally, the shape-compatibility check ran on `u.math.shape(init)` with
`init` being the `State` *wrapper*, which reports `()` (it does not see into the
wrapped value). As a result the check `_are_broadcastable_shapes((), sizes)`
always passed, so a `State` whose value shape disagreed with `sizes` was *not*
rejected (e.g. `param(State(ones(10) * u.nS), 5)` silently succeeded). The fix
resolves the underlying value *before* the shape check and validates
`u.math.shape(param_value)`.

**Risk:** low. The only in-tree callers of `param` (`braintools.conn`) pass
`Initialization` objects or scalars, never a `State`, and there were no existing
tests for the `State` path.

---

## D1 (Medium) — Broken pipe example in `Initialization` docstring

**File:** `braintools/init/_init_base.py` — `Initialization` class docstring

**Problem**

```python
>>> combined = (Normal(1.0 * u.nS, 0.2 * u.nS) |
...             lambda x: x.clip(0, 2 * u.nS) |
...             lambda x: x * 0.5)
```

Two defects:

1. **Operator precedence.** A bare `lambda` body extends as far right as
   possible, so this parses as
   `Normal(...) | (lambda x: x.clip(0, 2*u.nS) | (lambda x: x * 0.5))`.
   When evaluated, the lambda body computes `array | function`, a bitwise-or
   between an array and a callable → `TypeError`. The intended left-to-right
   pipe is not built.
2. **Unit mismatch.** `x` carries `nS`, so `x.clip(0, 2 * u.nS)` mixes a
   unitless `0` with a `nS` bound → `UnitMismatchError`.

The example only *constructs* the object (it is never called in the docstring),
so doctest does not flag it, yet it is wrong and misleads users.

**Fix**

Parenthesize each lambda and use unit-consistent, executable transforms:

```python
>>> combined = (Normal(1.0 * u.nS, 0.2 * u.nS) |
...             (lambda x: u.math.maximum(x, 0.0 * u.nS)) |
...             (lambda x: x * 0.5))
>>> values = combined(1000, rng=np.random.default_rng(0))
```

---

## D2 (Low) — `KaimingUniform` / `KaimingNormal` omit the `scale` parameter in docs

**File:** `braintools/init/_init_variance_scaling.py`

Both `__init__` accept a leading `scale` argument (default `None`, computed from
`nonlinearity`), but their **Parameters** sections list only `mode`,
`nonlinearity`, and `negative_slope`. The fix documents `scale` and clarifies
that, when `None`, it defaults to the He variance multiplier (`gain ** 2`, i.e.
`2.0` for ReLU).

---

## D3 (Low) — `Identity` 1D behaviour undocumented

**File:** `braintools/init/_init_orthogonal.py` — `Identity`

For a 1-D shape there is no identity matrix, so `Identity` returns a vector of
`scale * ones`. This is reasonable (bias-like) but was undocumented. The fix adds
a **Notes** section describing the 1-D, 2-D, and N-D behaviour.

---

## D4 (Low) — `Orthogonal` / `DeltaOrthogonal` deprecation warning has wrong `stacklevel`

**File:** `braintools/init/_init_orthogonal.py`

`Orthogonal(unit=...)` and `DeltaOrthogonal(unit=...)` emit a `DeprecationWarning`
with the default `stacklevel=1`, so the warning is attributed to
`_init_orthogonal.py` rather than the user's call site. The shared helper
`_resolve_deprecated_unit` in `_init_basic.py` correctly uses an elevated
`stacklevel`. The fix sets `stacklevel=2` so the warning points at user code.

---

## Items reviewed and found correct (no change)

To document the scope of the review, the following were checked and confirmed
correct:

- **Variance-scaling math.** Kaiming (`scale = gain**2`, `variance = scale/fan`),
  Xavier (`fan_avg`, `Var = 2/(fan_in+fan_out)`), and LeCun (`fan_in`,
  `Var = 1/fan_in`) all match their references; the truncated-normal variant
  rescales by the truncated-normal stddev constant to hit the target variance.
- **Distribution parametrizations.** `LogNormal` (linear-space mean/std →
  log-space `mu`/`sigma`), `Gamma`, `Weibull`, `Beta` (rescaled to `[low, high]`),
  and `Exponential` are parametrized correctly.
- **Default RNG backend.** Every distribution works with the default
  `brainstate.random` backend as well as a passed NumPy `Generator`.
- **Unit handling.** `Clipped` / `.clip()` correctly branch on unitless vs
  unit-carrying values (the earlier `H2/M8` fix); `Mixture` / `Conditional`
  reconcile component units and raise on incompatibility.
- **Profile arithmetic.** `ComposedProfile._evaluate`'s `isinstance(obj,
  ArrayLike)` check works on this toolchain (Python `typing.Union` isinstance).
- **Composition guards.** `PipeInit` / `PipeProfile` / `Compose` raise clear
  `TypeError`s when an `Initialization`/`DistanceProfile` is used where a plain
  transform is expected.

Note (by design, not changed): several profiles (`BimodalProfile`, and
`DoGProfile`/`MexicanHatProfile` before their non-negativity clip) can return
values `> 1`. They are *weight-scaling* profiles, so this is intentional; the
downstream connectivity sampler in `braintools.conn` is responsible for
interpreting `probability()` as a sampling probability.
