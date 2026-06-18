# `braintools.surrogate` — Issues Found and Proposed Solutions (2026-06-19)

Follow-up audit of the `braintools/surrogate/` module (surrogate gradient
functions for spiking neural networks) and its documentation, performed as a
senior Python developer / JAX expert.

This is a second pass. The first audit (2026-06-18, commit `bee02c5`) fixed a
large set of `surrogate_grad` / `surrogate_fun` **formula** bugs and added
`_correctness_test.py` (reference values + function/derivative consistency).
That suite + the others give 834 passing tests at ~100% line coverage.

Scope reviewed:

- `braintools/surrogate/__init__.py` (public API + module docstring)
- `braintools/surrogate/_base.py` (`Surrogate` base class + `heaviside_p` primitive)
- `braintools/surrogate/_impl.py` (all 18 surrogate implementations)
- Tests (`_base_test.py`, `_impl_test.py`, `_impl_extra_test.py`, `_correctness_test.py`)
- `docs/apis/surrogate.rst`

Findings were reproduced empirically against `jax 0.10.1`, `brainstate 0.5.0`,
and cross-checked against the documented math / external references.

## How the headline bug survived the existing suite

The function/derivative consistency test asserts
`d/dx surrogate_fun == surrogate_grad` on
`xs = linspace(-1.7, 1.7, 69) + 1e-3`. A *jump discontinuity located exactly at a
branch break-point* (here `x = 1`) is invisible to this check: JAX
autodiff of `jnp.where(cond, f, g)` returns `where(cond, f', g')`, the
per-branch derivative — it never "sees" the height of the jump between branches.
So a `surrogate_fun` that is the correct antiderivative *within each branch* but
discontinuous *between* branches passes the consistency test while still being a
broken (non-monotone) soft-Heaviside. The monotonicity / continuity check that
would have caught it did not include `LogTailedRelu`.

---

## Severity summary

| # | Severity | Location | One-line |
|---|----------|----------|----------|
| 1 | Medium (correctness, analysis only) | `_impl.py` `LogTailedRelu.surrogate_fun` | `log(x)` for `x>1` is discontinuous at `x=1` (jumps 1.0 → 0.0); should be `1+log(x)` (continuous, C¹). |
| 2 | Low (doc) | `_impl.py` `QPseudoSpike` docstring | `alpha > 1` described as "Compact support"; the gradient is a power-law tail, never compact support. |
| 3 | Low (doc) | `_impl.py` (9 classes) + `_base.py` | Example/`.. plot::` blocks use `braintools.surrogate.X` after only `import brainstate` → `NameError` if copy-pasted. |
| 4 | Low (doc) | `_impl.py` (`S2NN`, `QPseudoSpike`, `LeakyRelu`, `LogTailedRelu` plot blocks) | Set `fn.origin = True` then call `fn(xs)` expecting the smooth function, but `__call__` ignores `origin` → the "Original Function" subplot actually plots the Heaviside step. |

No new *functional* (training-affecting) bug was found: every public surrogate's
`surrogate_grad` reproduces its closed form, the forward pass is the exact
Heaviside step, and all gradients are finite at the dangerous break-points
(re-verified). Issue #1 affects only the optional `surrogate_fun`
(visualization/analysis); the gradient used in backprop is identical before and
after the fix.

---

## Issue 1 — `LogTailedRelu.surrogate_fun` discontinuous at `x = 1` (Medium)

**Location:** `braintools/surrogate/_impl.py`, `LogTailedRelu.surrogate_fun`.

**Current code:**

```python
def surrogate_fun(self, x):
    z = jnp.where(
        x > 1,
        jnp.log(x),                       # <-- log(1) = 0
        jnp.where(x > 0, x, self.alpha * x)  # <-- x = 1 at the same point
    )
    return z
```

**Symptom (reproduced):**

```python
>>> f = braintools.surrogate.LogTailedRelu(alpha=0.1).surrogate_fun
>>> [round(float(f(x)), 4) for x in (0.9, 1.0, 1.0001, 1.1, 2.0)]
[0.9, 1.0, 0.0001, 0.0953, 0.6931]
```

The analysis function jumps **down** from `1.0` to `~0.0` immediately past the
threshold and is therefore non-monotone — an invalid soft-Heaviside / activation
shape.

**Root cause:** the standard *log-tailed ReLU* (Cai et al., *Deep Learning with
Low Precision by Half-Wave Gaussian Quantization*, CVPR 2017 — the reference
already cited in this class' docstring) is

```
f(x) = alpha*x       , x <= 0
f(x) = x             , 0 < x <= 1
f(x) = 1 + log(x)    , x > 1
```

The `+1` makes the tail continuous (`1 + log(1) = 1`, matching the linear branch)
and even `C¹` (`d/dx (1+log x) = 1/x = 1` at `x=1`, matching the unit slope). The
implementation (and docstring) dropped the `+1`. The `surrogate_grad` is `1/x`
for `x>1` either way, so **only the forward analysis function and docstring are
wrong**; backprop is unaffected.

**Proposed fix:** use the continuous antiderivative, with the same
guarded-`where` trick already used in `surrogate_grad` so no `nan` leaks from the
dead branch:

```python
def surrogate_fun(self, x):
    # Guard the dead ``x > 1`` branch so ``log`` stays finite for x <= 1.
    x_safe = jnp.where(x > 1, x, 1.0)
    z = jnp.where(
        x > 1,
        1.0 + jnp.log(x_safe),     # continuous (and C^1) with the x branch at x = 1
        jnp.where(x > 0, x, self.alpha * x)
    )
    return z
```

Also update the docstring math (`\log(x)` → `1 + \log(x)`), the inline Examples
comment, and the test that pinned the old value at `x = e` (`log(e) = 1` →
`1 + log(e) = 2`).

---

## Issue 2 — `QPseudoSpike` docstring: "Compact support" is wrong (Low, doc)

**Location:** `braintools/surrogate/_impl.py`, `QPseudoSpike` Parameters.

The docstring states:

```
- alpha < 1: Heavy-tailed gradient (slower decay)
- alpha = 1: Exponential-like decay
- alpha > 1: Compact support (faster decay)
- alpha = 2: Quadratic decay (default)
```

The backward gradient is `g'(x) = (1 + 2|x|/(alpha+1))^{-alpha}`, a power-law
(polynomial) tail that is **strictly positive for every finite `x` and every
`alpha > 0`**. It never has compact support. ("alpha = 1: Exponential-like" is
also loose — it is still polynomial, `~ (1+|x|)^{-1}` — but the egregious error
is "Compact support".)

**Proposed fix:** reword to describe the true tail behaviour, e.g.

```
- alpha < 1: Heavy (slowly decaying) polynomial tail
- alpha = 1: ~1/(1+|x|) polynomial tail
- alpha > 1: Lighter polynomial tail (faster decay); always non-zero, never compact
- alpha = 2: Quadratic (~|x|^-2) decay (default)
```

---

## Issue 3 — Example blocks are not self-contained (Low, doc)

**Location:** `braintools/surrogate/_impl.py` (the "basic" surrogates: `Sigmoid`,
`PiecewiseQuadratic`, `PiecewiseExp`, `SoftSign`, `Arctan`, `NonzeroSignLog`,
`ERF`, `PiecewiseLeakyRelu`, `SquarewaveFourierSeries`) and
`braintools/surrogate/_base.py`.

These `.. code-block:: python` Examples and `.. plot::` blocks open with

```python
>>> import brainstate
>>> import jax.numpy as jnp
>>> sigmoid = braintools.surrogate.Sigmoid(alpha=4.0)   # NameError: 'braintools'
```

`braintools` is never imported, so the snippet raises `NameError` if a user
copy-pastes it. The project docstring guide (`CLAUDE.md`) requires examples to be
self-contained ("Always include necessary imports … so self-contained"). (These
are not currently run as doctests — no `--doctest-modules` config — so this is a
usability/quality issue, not a CI failure. The *advanced* surrogates already do
`import braintools.surrogate as surrogate`, which is correct.)

**Proposed fix:** add `>>> import braintools` to each affected block.

---

## Issue 4 — `origin = True` toggle in plot examples does nothing (Low, doc)

**Location:** `braintools/surrogate/_impl.py`, the `.. plot::` blocks of `S2NN`,
`QPseudoSpike`, `LeakyRelu`, and `LogTailedRelu`.

Each of these plots a second subplot titled "Original Function" like so:

```python
>>>     s2nn_fn = surrogate.S2NN(alpha=alpha, beta=beta)
>>>     s2nn_fn.origin = True
>>>     ys = jax.vmap(s2nn_fn)(xs)
>>>     ax2.plot(xs, ys, ...)
```

This assumes that setting `.origin = True` makes `__call__` return the smooth
`surrogate_fun`. But `Surrogate.__call__` only ever does
`heaviside_p.bind(x, surrogate_grad(stop_gradient(x)))[0]` — it never reads an
`origin` attribute. So `jax.vmap(s2nn_fn)(xs)` returns the **Heaviside step**
(0/1), and the "Original Function" subplot is wrong (a step, not the smooth
curve). The `origin` flag is a leftover from the upstream `brainpy` API that the
braintools refactor dropped.

**Proposed fix (minimal, no API change):** plot the smooth function directly via
the public `surrogate_fun`, dropping the dead `origin` assignment:

```python
>>>     s2nn_fn = surrogate.S2NN(alpha=alpha, beta=beta)
>>>     ys = jax.vmap(s2nn_fn.surrogate_fun)(xs)
>>>     ax2.plot(xs, ys, ...)
```

---

## Verification performed (non-issues, no change)

- **Break-point continuity** of every `surrogate_fun` with both branches:
  `PiecewiseExp`, `PiecewiseQuadratic`, `PiecewiseLeakyRelu`, `S2NN`,
  `SquarewaveFourierSeries`, `LeakyRelu` are all continuous at their knots
  (`LogTailedRelu` was the only exception — Issue 1).
- **Forward pass** is the exact Heaviside step for all 18 surrogates.
- **Gradient finiteness** at `x = -1` (S2NN), `x = 0` (LogTailedRelu), and across
  `[-2, 2]` for every surrogate, including parameter gradients.
- **API docs** (`docs/apis/surrogate.rst`) list all 18 functions and 17 classes
  (`SlayerGrad` included) — complete.
- **brainstate APIs** referenced in the module docstring (`transform.grad`,
  `optim`, `nn.Linear`, `ParamState`, `augment`) all exist in `brainstate 0.5.0`.
- **JVP parameter-gradient side effect** and the gradient-only `surrogate_fun`
  `NotImplementedError` are already documented (06-18 audit).

## Tests added / updated

- `_impl_extra_test.py::test_log_tailed_relu_fun_regimes` — update the `x = e`
  expectation from `1.0` to `2.0` (now `1 + log(e)`).
- `_correctness_test.py` — add a regression test asserting
  `LogTailedRelu.surrogate_fun` is continuous at `x = 1`, monotone increasing
  over `[-2, 4]`, finite, and still consistent with `surrogate_grad`.
