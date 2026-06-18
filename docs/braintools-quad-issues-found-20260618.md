# `braintools.quad` — Issues Found and Proposed Solutions (2026-06-18)

Audit of the `braintools/quad/` module (ODE / SDE / DDE / IMEX one-step
integrators) and its documentation, performed as a senior Python architect /
JAX expert / BrainX developer.

Scope reviewed:

- `braintools/quad/__init__.py` (public API + module docstring)
- `braintools/quad/_ode_integrator.py`
- `braintools/quad/_sde_integrator.py`
- `braintools/quad/_dde_integrator.py`
- `braintools/quad/_imex_integrator.py`
- Existing tests (`*_test.py`)
- `docs/apis/quad.rst`, `docs/quad/*` (index + notebooks)

Every **correctness** finding below was reproduced empirically against
`jax 0.10.1`, `brainunit 0.5.1`, `brainstate 0.5.0` before being recorded.

Numerical method correctness was cross-checked against the standard Butcher
tableaux: Euler, RK2 (Heun), RK3 (Kutta), RK4, RK4 3/8-rule, Cash–Karp 4(5),
Bogacki–Shampine 2(3)/3(2), Dormand–Prince 5(4) and 8(7) (DOP853), RK–Fehlberg
4(5), SSPRK(3,3), Ralston RK2/RK3, and ARS(2,2,2). **All tableaux are
implemented correctly.** The bugs are concentrated in unit handling, noise
sampling, and documentation.

---

## Severity summary

| #  | Severity | Location | One-line |
|----|----------|----------|----------|
| 1  | High | `_ode_integrator.py:305` | `ode_expeuler_step` divides by Jacobian unit instead of state unit → crashes with units |
| 2  | High | `_sde_integrator.py:270` | `sde_expeuler_step` same unit bug |
| 3  | High | `_sde_integrator.py:275` | `sde_expeuler_step` samples noise from `args[0]` → `IndexError`/wrong shape |
| 4  | High | `_sde_integrator.py:117` | `sde_euler_step` uses `jnp.sqrt(dt)` → crashes for unitful `dt` |
| 5  | High | `_sde_integrator.py:410` | `sde_tamed_euler_step` taming term is dimensionally inconsistent → crashes with units |
| 6  | Medium | `_ode_integrator.py:267-279` | `ode_expeuler_step` docstring: full-width comma, wrong name, SDE-only text |
| 7  | Medium | `__init__.py:129` | `ode_heun_step` example mislabeled "second-order" (it is third-order) |
| 8  | Medium | `__init__.py:191` | SDE example calls `sde_srk3_step` not imported in that block |
| 9  | Medium | `__init__.py:197-207` | `sde_expeuler_step` example uses a non-existent signature |
| 10 | Medium | `__init__.py:246` | `imex_cnab_step` example omits the required `y_prev` argument |
| 11 | Medium | `__init__.py:267-274` | DDE history-init loop iterates an empty `deque` → never initializes |
| 12 | Medium | `_dde_integrator.py:106` | `dde_euler_step` docstring example uses `delay=` (param is `delays`) |
| 13 | Medium | `docs/apis/quad.rst` | DDE integrators entirely missing from the API reference |
| 14 | Low | `_ode_integrator.py:690-698` | `ode_dopri5_step` computes an extra stage even when `return_error=False` |
| 15 | Low | `_ode_integrator.py:301`, `_sde_integrator.py:264` | Exp-Euler dtype error message uses `y.dtype`, fails for Python `float` |
| 16 | Low | `_sde_integrator_test.py:177` | Dead test: the only assertion is commented out |
| 17 | Low | several | `Callable[[A, B, ...], R]` alias mixes positional types with `...` (cosmetic) |

---

## High-severity correctness bugs

### 1 & 2 — Exponential Euler divides by the wrong unit

`ode_expeuler_step` (`_ode_integrator.py:305`) and `sde_expeuler_step`
(`_sde_integrator.py:270`) both contain:

```python
linear, derivative = brainstate.transform.vector_grad(f, argnums=0, return_value=True)(y, t, *args, **kwargs)
linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(linear))
phi = u.math.exprel(dt * linear)
```

`vector_grad` returns the diagonal Jacobian `J = ∂f/∂y`. Physically `J` has unit
`unit(f)/unit(y) = [X]/[T]/[X] = 1/[T]`, so `dt * J` is dimensionless and
`exprel` is well-defined. The code rebuilds `linear` with unit
`unit(derivative)/unit(linear) = ([X]/[T]) / (1/[T]) = [X]` — i.e. it assigns the
**state** unit to the linear coefficient. `dt * linear` then has unit `[T]·[X]`
and `u.math.exprel` rejects it.

This is exactly the failure mode that the canonical `brainstate.nn.exp_euler_step`
documents in an inline comment ("Divide by the *state* unit, not the Jacobian
unit … `u.math.exprel` in saiunit>=0.4.0 rejects"). The braintools port took the
wrong unit.

**Reproduction**

```python
def f(v, t):
    return (-v) / (10.0 * u.ms)
with brainstate.environ.context(dt=0.1 * u.ms):
    braintools.quad.ode_expeuler_step(f, -65.0 * u.mV, 0.0 * u.ms)
# TypeError: exprel requires a dimensionless "x" ... Got Quantity(unit=mV, ...)
```

The existing tests never exercise units, so the bug is invisible to them.

**Fix** — divide by the unit of the state `y`:

```python
linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(y))
```

Verified: with the fix the unitful result equals the dimensionless result times
the unit (`-64.35324 mV`), and matches `ode_euler_step`/`ode_rk4_step`.

### 3 — `sde_expeuler_step` samples noise from `args[0]`

`_sde_integrator.py:275`:

```python
diffusion_part = dg(y, t, *args, **kwargs) * u.math.sqrt(dt) * randn_like(args[0])
```

The Brownian increment shape is taken from the **first extra positional
argument** rather than from the state `y`. Consequences:

- Calling `sde_expeuler_step(df, dg, y, t)` with no extra args raises
  `IndexError: tuple index out of range`.
- When extra args *are* supplied they are also forwarded to `df`/`dg`, so the
  argument is doubly-overloaded; if `args[0]` is not shaped like `y` the noise is
  silently the wrong shape.

Every other SDE stepper samples noise from the state (`randn_like(y)` /
`randn_like(y0)`).

**Fix** — sample from the state and drop the `args[0]` dependency:

```python
diffusion_part = dg(y, t, *args, **kwargs) * u.math.sqrt(dt) * randn_like(y)
```

and remove the docstring sentence claiming the first extra argument fixes the
noise shape.

### 4 — `sde_euler_step` uses `jnp.sqrt` on a unitful `dt`

`_sde_integrator.py:117`:

```python
dt_sqrt = jnp.sqrt(dt)            # all sibling steppers use u.math.sqrt(dt)
```

`jnp.sqrt` cannot consume a `saiunit.Quantity`:

```python
# TypeError: sqrt requires ndarray or scalar arguments, got <class 'saiunit.Quantity'>
```

`sde_euler_step` is the headline SDE method and the one most likely to be called
with a unitful `dt` (e.g. `0.1 * u.ms`). `sde_milstein_step` and the others use
`u.math.sqrt` and work correctly.

**Fix**

```python
dt_sqrt = u.math.sqrt(dt)
```

### 5 — `sde_tamed_euler_step` taming term is dimensionally inconsistent

`_sde_integrator.py:410`:

```python
f_tamed = tree_map(lambda a: a / (1.0 + dt * u.math.abs(a)), f0)
```

`dt * |f|` has unit `[T]·[X]/[T] = [X]` (the state's unit), so `1.0 + dt*|f|`
adds a dimensionless `1` to a unitful quantity → `UnitMismatchError`. The taming
factor must be a pure number. The classical tamed-Euler scheme (Hutzenthaler,
Jentzen & Kloeden 2012) is implicitly written for dimensionless states.

**Fix** — form the taming factor from the dimensionless magnitude of the drift
increment, which reduces *exactly* to the original expression when the state is
dimensionless:

```python
f_tamed = tree_map(lambda a: a / (1.0 + u.get_mantissa(dt * u.math.abs(a))), f0)
```

Verified: dimensionless case reproduces `f/(1+dt|f|)` to full precision; unitful
case integrates cleanly.

---

## Medium-severity documentation bugs

### 6 — `ode_expeuler_step` docstring is corrupted

`_ode_integrator.py:267-279`:

```python
>>> exp_euler_step(fun, x， 0.)      # full-width comma '，' → SyntaxError under doctest;
                                     # also the wrong function name
```

The Examples block additionally references `diffusion_fn` (an SDE concept that
does not apply to the ODE stepper) and shows no expected output. Rewrite using
the real name `ode_expeuler_step`, an ASCII comma, a runnable example, and drop
the diffusion text.

### 7 — `ode_heun_step` mislabeled in the module docstring

`__init__.py:129` lists `ode_heun_step` under the comment "Second-order
methods". The function (and its own docstring) is Heun's **third-order** RK.
Move it to the third-order group.

### 8 — SDE example uses a name it never imports

`__init__.py:159-191`: the import block pulls in
`sde_euler_step, sde_milstein_step, sde_expeuler_step, sde_heun_step,
sde_srk2_step, sde_tamed_euler_step` but the example then calls
`sde_srk3_step`, producing a `NameError` if executed. Add `sde_srk3_step` to the
import list (or drop the call).

### 9 — `sde_expeuler_step` example uses a non-existent signature

`__init__.py:197-207` calls
`sde_expeuler_step(linear_drift, nonlinear_drift, diffusion, V, t)` as though the
drift were split into linear/nonlinear parts. The real signature is
`sde_expeuler_step(df, dg, y, t, *args)` with a single drift `df` linearized
internally. Rewrite the example to `sde_expeuler_step(drift, diffusion, V, t)`.

### 10 — `imex_cnab_step` example omits the required `y_prev`

`__init__.py:246` calls `imex_cnab_step(f_explicit, f_implicit, V, t, ...)`, but
CNAB is a *multistep* method whose signature is
`imex_cnab_step(f_exp, f_imp, y, y_prev, t, *args)`. As written the call raises
`TypeError` (missing `t`). Update the example to pass a previous state and note
that CNAB needs two history points.

### 11 — DDE history initialization loop never runs

`__init__.py:267-274`:

```python
history = deque(maxlen=int(delay / bst.environ.get_dt()) + 1)
...
for i in range(len(history)):    # len(history) == 0 here → body never executes
    history.append(y)
```

The `deque` is freshly created and therefore empty, so `range(len(history))` is
empty and the history is never seeded. Iterate over the intended count
(`history.maxlen`) instead.

### 12 — `dde_euler_step` docstring example uses `delay=`

`_dde_integrator.py:106`: `dde_euler_step(f, y, t, history_fn, delay=1.0)`. The
parameter is named `delays`; `delay=` is swallowed by `**kwargs` and forwarded to
`f`, while the required `delays` argument is missing → `TypeError`. Use
`delays=1.0`.

### 13 — DDE integrators missing from the API reference

`docs/apis/quad.rst` documents ODE, IMEX and SDE steppers but has **no DDE
section**, even though `dde_euler_step`, `dde_heun_step`, `dde_rk4_step`,
`dde_euler_pc_step`, `dde_heun_pc_step` are all public (exported in `__all__`).
Add a "DDE Numerical Integrators" autosummary block.

---

## Low-severity / robustness

### 14 — `ode_dopri5_step` computes a redundant stage

`_ode_integrator.py:690-708`: the 7th stage `y7`/`k7` is computed before the
`return_error` check, and `y7` is byte-for-byte identical to `y5th` (both use the
`b5` weights). When `return_error=False` the extra `tree_map` and `f` evaluation
are dead work (XLA DCE hides this under `jit`, but eager calls pay for it). Use
the FSAL property: compute `k7 = f(y5th, t+dt)` only inside the
`return_error` branch.

### 15 — Exp-Euler dtype error message crashes on Python floats

`_ode_integrator.py:301` and `_sde_integrator.py:264` interpolate `{y.dtype}`
into the `ValueError` message. If a disallowed input is a Python `float` (no
`.dtype`), formatting the message itself raises `AttributeError`, masking the
intended error. Use `u.math.get_dtype(y)`.

### 16 — Dead test

`_sde_integrator_test.py:160-177` (`test_implicit_euler_linear_no_noise`)
computes `y_exact` but its only `assert` is commented out, so the test validates
nothing. Replace with a real assertion (implicit Euler has the closed form
`y1 = y0 / (1 - a·dt)`).

### 17 — Cosmetic type alias

`ODE = Callable[[brainstate.typing.PyTree, float | u.Quantity, ...], ...]` mixes
explicit positional types with `...`, which is not valid `Callable` form for
static checkers (it is accepted at runtime). Harmless; could be simplified to
`Callable[..., brainstate.typing.PyTree]`.

---

## Method-correctness verification (no changes needed)

For the record, these were checked digit-by-digit against published tableaux and
found correct: `ode_euler/rk2/rk3/rk4/rk4_38`, `ode_midpoint`, `ode_heun` (Heun
RK3), `ode_rk45` (Cash–Karp), `ode_rk23`/`ode_bs32` (Bogacki–Shampine),
`ode_dopri5`, `ode_rkf45`, `ode_ssprk33`, `ode_ralston2/3`, `ode_dopri8`
(DOP853 incl. embedded error), all `imex_*`, `sde_milstein` (derivative-free
Platen), `sde_heun`, `sde_srk2/3/4`, and the DDE stage times. ODE/IMEX/DDE all
integrate correctly with `brainunit` quantities.

---

## Remediation plan

1. Fix bugs 1–5 (correctness) and 14–15 (robustness) in the three integrator
   modules; tame on the mantissa for bug 5; sample noise from `y` for bug 3.
2. Fix documentation bugs 6–13 (module docstrings, DDE docstring, `quad.rst`).
3. Repair the dead test (16) and add comprehensive new tests covering: unitful
   ODE/SDE exp-Euler, unitful `sde_euler`/`sde_tamed_euler`, `sde_expeuler`
   no-arg call, embedded-error pairs, IMEX/DDE paths, and PyTree structure — to
   exceed 90 % statement coverage of `braintools/quad/`.
4. Leave 17 as a noted cosmetic item (no behavioral impact).
