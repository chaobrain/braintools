# `braintools.surrogate` — Issues Found and Proposed Solutions (2026-06-18)

Audit of the `braintools/surrogate/` module (surrogate gradient functions for
spiking neural networks) and its documentation, performed as a senior Python
architect / JAX expert / BrainX developer.

Scope reviewed:

- `braintools/surrogate/__init__.py` (public API + module docstring)
- `braintools/surrogate/_base.py` (`Surrogate` base class + `heaviside_p` primitive)
- `braintools/surrogate/_impl.py` (all 17 surrogate implementations)
- Existing tests (`_base_test.py`, `_impl_test.py`, `_impl_extra_test.py`)
- `docs/apis/surrogate.rst`, `docs/surrogate/*` (index + notebooks)

Every **correctness** finding below was reproduced empirically against
`jax 0.10.1`, `brainunit 0.5.1`, `brainstate 0.5.0`, cross-checked against
`scipy.stats.norm`, the standard square-wave Fourier series, and the documented
mathematical definitions in each class' own docstring.

## How these bugs survived the existing test suite

The pre-existing suite (709 tests, 100 % line coverage) is **self-referential**:
nearly every assertion is of the form

```python
grad = brainstate.transform.vector_grad(sg)(x)
assert jnp.allclose(grad, sg.surrogate_grad(x))   # compares impl against itself
```

This pins the *plumbing* (forward = Heaviside, backward = `surrogate_grad`) but
can never detect a wrong `surrogate_grad`/`surrogate_fun` **formula**, because it
compares the implementation against itself. The bugs below were found by instead
checking each surrogate against (a) its own documented math, (b) the
function/derivative consistency `d/dx surrogate_fun == surrogate_grad`, and
(c) external references (`scipy`, closed-form integrals).

Most numerical bugs are faithfully inherited from the upstream
`brainpy.math.surrogate` port; in almost every case the braintools **docstring
already states the correct formula**, so the fix is to make the code match its
own documentation.

---

## Severity summary

| #  | Severity | Location | One-line |
|----|----------|----------|----------|
| 1  | High | `_impl.py` `GaussianGrad.surrogate_grad` | Gaussian exponent is `-x²σ²/2` instead of `-x²/(2σ²)` → σ dependence inverted |
| 2  | High | `_impl.py` `Arctan.surrogate_fun` | misuses `arctan2`, returns `arctan(αx/2)+0.5` (range ≈ (−1.07, 2.07)), inconsistent with grad |
| 3  | High | `_impl.py` `ERF.surrogate_fun` | returns `½erf(−αx)` — decreasing, range (−0.5, 0.5); should be `½erfc(−αx)` |
| 4  | High | `_impl.py` `PiecewiseQuadratic.surrogate_grad` | parabola `α−α²x²` (discontinuous at `|x|=1/α`); should be triangle `α−α²|x|` |
| 5  | Medium | `_impl.py` `PiecewiseLeakyRelu.surrogate_grad` | central slope `1/w` ≠ `d/dx surrogate_fun = 1/(2w)` (factor-2) |
| 6  | Medium | `_impl.py` `QPseudoSpike` | `surrogate_fun` (α−1) inconsistent with `surrogate_grad` (α+1) → non-finite; docstring grad formula wrong |
| 7  | Medium | `_impl.py` `SquarewaveFourierSeries` | `range(2, n)` sums `n−1` terms (default `n=2` → 1 term); docstring says `n` terms; doc coefficient `1/π` vs code `2/π` |
| 8  | Low | `_impl.py` `S2NN.surrogate_grad` | `β/(x+1)` evaluated in dead branch → `inf` at `x ≤ −1` (nan-grad hazard) |
| 9  | Low | `_impl.py` `LogTailedRelu.surrogate_grad` | `1/x` evaluated in dead branch → `inf` at `x ≤ 0` (nan-grad hazard) |
| 10 | Low | `_impl.py` `__all__` | exports 16 class/function pairs but `NonzeroSignLog` pair is present while `__init__` re-exports a 17th — verified consistent (no action) |
| 11 | Doc | `_base.py` `_heaviside_jvp` | differentiating output w.r.t. a *parameter* returns `d(surrogate_grad)/d(param)`, not 0 — surprising; undocumented |
| 12 | Doc | `_impl.py` gradient-only surrogates | `ReluGrad/GaussianGrad/MultiGaussianGrad/InvSquareGrad/SlayerGrad` raise `NotImplementedError` from `surrogate_fun` — undocumented |
| 13 | Doc | `_base.py` `_heaviside_transpose` | defined but never registered (JAX derives transpose from the JVP); dead code, undocumented |
| 14 | Doc | `__init__.py` module docstring | `bst.transform.grad`/`bst.augment` examples use APIs that do not match the rest of the codebase; input-unit assumptions unstated |

---

## High-severity correctness bugs

### 1 — `GaussianGrad`: inverted σ in the Gaussian exponent

`_impl.py`, `GaussianGrad.surrogate_grad`:

```python
dx = jnp.exp(-(x ** 2) / 2 * jnp.power(self.sigma, 2)) / (jnp.sqrt(2 * jnp.pi) * self.sigma)
```

By Python precedence `-(x**2) / 2 * sigma**2` evaluates to `-x²·σ²/2`, i.e. the
exponent is **multiplied** by `σ²`. The documented formula (and the standard
Gaussian PDF) is

```
g'(x) = α · 1/(σ√(2π)) · exp(−x²/(2σ²))
```

so the exponent must be **divided** by `2σ²`. The consequences are severe:

- The width is wrong for every `σ ≠ 1/√2`.
- The σ dependence is **inverted**: increasing `σ` makes the gradient *narrower*,
  the opposite of the documented "larger values create smoother gradients".
- The sibling `MultiGaussianGrad.surrogate_grad` computes the central Gaussian
  **correctly** (`-x**2 / (2 * jnp.power(self.sigma, 2))`), so the two Gaussian
  surrogates disagree.

Empirical check (peak at `x=0`, `α=1`) for `σ = 0.3, 0.5, 1.0`:

```
buggy : 0.030, 0.110, 0.282   (increases with σ — wrong)
fixed : 1.330, 0.798, 0.399   (decreases with σ — matches scipy.norm.pdf)
```

**Fix** — divide by `2σ²`:

```python
dx = jnp.exp(-(x ** 2) / (2 * jnp.power(self.sigma, 2))) / (jnp.sqrt(2 * jnp.pi) * self.sigma)
return self.alpha * dx
```

This is a **training-behaviour change** (the gradient values change), but the
current behaviour is unambiguously wrong (contradicts the docstring, the
standard PDF, `scipy`, and `MultiGaussianGrad`).

### 2 — `Arctan.surrogate_fun`: wrong function via `arctan2`

`_impl.py`, `Arctan.surrogate_fun`:

```python
return jnp.arctan2(jnp.pi / 2 * self.alpha * x, jnp.pi) + 0.5
```

`arctan2(y, x) = arctan(y/x)` for `x > 0`, so this computes
`arctan((π/2·αx)/π) + 0.5 = arctan(αx/2) + 0.5`, which

- ranges over ≈ `(−1.07, 2.07)` (a surrogate of the step function must lie in
  `[0, 1]`), and
- is **not** the antiderivative of `surrogate_grad`
  (`d/dx surrogate_fun ≠ surrogate_grad`; max |Δ| ≈ 0.26 over `[−1.5, 1.5]`).

The documented original function is `g(x) = (1/π)·arctan((π/2)αx) + 1/2`.
(The upstream brainpy code called single-argument `jnp.arctan2(...)`, which would
raise `TypeError`; braintools "fixed" the arity but introduced the wrong math.)

**Fix** — use single-argument `arctan` and divide by π:

```python
return jnp.arctan(jnp.pi / 2 * self.alpha * x) / jnp.pi + 0.5
```

Verified: range ⊂ `(0, 1)`, `surrogate_fun(0) = 0.5`, and
`d/dx surrogate_fun == surrogate_grad` (max |Δ| ≈ 4e-8). Forward pass and
`surrogate_grad` (the trained path) are unchanged.

### 3 — `ERF.surrogate_fun`: decreasing, out of `[0, 1]`

`_impl.py`, `ERF.surrogate_fun`:

```python
return sci.special.erf(-self.alpha * x) * 0.5
```

This is **decreasing** in `x` and ranges over `(−0.5, 0.5)` — it is the negative
of a step approximation. The documented original function is

```
g(x) = ½(1 − erf(−αx)) = ½·erfc(−αx)
```

which increases from 0 to 1 with `g(0) = 0.5`, and whose derivative is exactly
the (correct) `surrogate_grad = (α/√π)·exp(−α²x²)`.

**Fix**:

```python
return 0.5 * (1. - sci.special.erf(-self.alpha * x))
```

Verified consistent with `surrogate_grad` (max |Δ| ≈ 6e-8) and range ⊂ `(0, 1)`.
The existing test `test_erf_at_zero` asserts `surrogate_fun(0) == 0.0`; the
correct value is `0.5`, so that test is updated.

### 4 — `PiecewiseQuadratic.surrogate_grad`: discontinuous parabola

`_impl.py`, `PiecewiseQuadratic.surrogate_grad`:

```python
dx = jnp.where(jnp.abs(x) > 1 / self.alpha, 0., (-(self.alpha * x) ** 2 + self.alpha))
```

The inner expression is `α − α²x²` (a downward **parabola**). At the window edge
`|x| = 1/α` it equals `α − 1`, which is **non-zero for `α ≠ 1`**, so the gradient
**jumps discontinuously** to 0 outside the window (e.g. `α=2`: drops from `1` to
`0`). The documented backward formula, and the exact derivative of the
(correct) piecewise-quadratic `surrogate_fun`, is the continuous **triangle**

```
g'(x) = α − α²|x|   for |x| ≤ 1/α,   else 0
```

**Fix**:

```python
dx = jnp.where(jnp.abs(x) > 1 / self.alpha, 0., self.alpha - self.alpha ** 2 * jnp.abs(x))
```

Verified: `d/dx surrogate_fun == surrogate_grad` (max |Δ| ≈ 6e-8) and continuous
at the edge. This is a **training-behaviour change**, but the previous gradient
was discontinuous and contradicted both the docstring and `surrogate_fun`.

---

## Medium-severity correctness bugs

### 5 — `PiecewiseLeakyRelu`: gradient is 2× the slope of its own `surrogate_fun`

`surrogate_fun` is a valid soft-Heaviside that rises **0 → 1 over `[−w, w]`**
(`fun(−w)=0`, `fun(0)=0.5`, `fun(w)=1`), so its central slope is `1/(2w)`. But:

```python
dx = jnp.where(jnp.abs(x) > self.w, self.c, 1 / self.w)   # central slope 1/w
```

uses `1/w`, twice the true derivative (max |Δ| = 0.5 for `w=1`). With break-points
fixed at `±w`, the only `[0,1]` smooth-Heaviside consistent with the gradient is
the one with slope `1/(2w)`; a `1/w` central slope would force `surrogate_fun`
out of `[0, 1]` (it would reach 1.5 at `x=w`). The docstring is itself
internally inconsistent (`g(x)=x/(2w)+½` ⇒ slope `1/(2w)`, yet "Backward
`g'(x)=1/w`").

**Fix** — central slope `1/(2w)`:

```python
dx = jnp.where(jnp.abs(x) > self.w, self.c, 1 / (2 * self.w))
```

This is a **training-behaviour change** (2× smaller central gradient); flagged
prominently. It makes `d/dx surrogate_fun == surrogate_grad` exactly.

### 6 — `QPseudoSpike`: `surrogate_fun` (α−1) vs `surrogate_grad` (α+1)

```python
def surrogate_grad(self, x):
    dx = jnp.power(1 + 2 / (self.alpha + 1) * jnp.abs(x), -self.alpha)   # uses (α+1)
def surrogate_fun(self, x):
    ... jnp.power(1 - 2 / (self.alpha - 1) * jnp.abs(x), 1 - self.alpha) ...   # uses (α−1)
```

The two are **not** a function/derivative pair, and the docstring's "Backward
gradient" prints the `(α−1)` form, contradicting the code. Worse, the
`(α−1)` `surrogate_fun` is non-finite over ordinary ranges (for `α≥1` the base
`1 − 2|x|/(α−1)` goes negative → `nan`; at `α=1` it divides by zero). Empirically
`surrogate_fun` reaches ~2.1e6 / `inf` for `α=2, 3` on `[−1.5, 1.5]`.

The `surrogate_grad` with `(α+1)` is the correct one to keep: it is the only form
finite for the **documented** parameter range (the docstring lists `α<1`
"heavy-tailed" as a valid regime, which the `(α−1)` form cannot represent), it
matches the upstream reference, and it is what training already uses.

**Fix** — keep `surrogate_grad`; correct the docstring backward formula to
`(α+1)`; and replace `surrogate_fun` with the true antiderivative of the
`(α+1)` gradient (centred at `(0, 0.5)`, with the removable `α=1` singularity
handled):

```python
def surrogate_fun(self, x):
    a = self.alpha
    base = 1. + 2. / (a + 1.) * jnp.abs(x)        # always >= 1 for a > 0
    one_minus_a = 1. - a
    safe = jnp.where(one_minus_a == 0., 1., one_minus_a)
    integral = jnp.where(
        one_minus_a == 0.,
        (a + 1.) / 2. * jnp.log(base),                       # limit at a == 1
        (a + 1.) / (2. * safe) * (jnp.power(base, one_minus_a) - 1.),
    )
    return 0.5 + jnp.sign(x) * integral
```

Verified `d/dx surrogate_fun == surrogate_grad` for `α ∈ {0.5, 2, 3}` (max |Δ| ≈
6e-8), all finite, `surrogate_fun(0)=0.5`.

### 7 — `SquarewaveFourierSeries`: off-by-one term count + doc coefficient

```python
dx = jnp.cos(w * x)                       # term i = 1
for i in range(2, self.n):                # i = 2 .. n-1  → only n-1 terms total
    dx += jnp.cos((2 * i - 1.) * w * x)
```

`range(2, self.n)` stops at `n−1`, so the series has `n−1` terms, not `n`. With
the **default `n=2`** the loop is empty and only a single cosine is used — the
docstring states "Number of Fourier terms. Default is 2" with the sum
`Σ_{i=1}^{n}`. The `surrogate_fun` has the same off-by-one. Separately, the
`surrogate_fun` docstring shows coefficient `1/π` while the code (correctly, for
a 0–1 square wave) uses `2/π`.

**Fix** — sum `i = 1 .. n` and correct the doc coefficient:

```python
for i in range(2, self.n + 1):
    dx += jnp.cos((2 * i - 1.) * w * x)
...
for i in range(2, self.n + 1):
    c = (2 * i - 1.)
    ret += jnp.sin(c * w * x) / c
```

`n=1` is unchanged (empty loop → single term); `n≥2` now includes the previously
dropped highest harmonic. `surrogate_fun`/`surrogate_grad` stay a consistent
pair.

---

## Low-severity robustness (JAX sharp edges)

### 8 — `S2NN.surrogate_grad`: `inf` in the dead `where` branch

```python
dx = jnp.where(x < 0., self.alpha * sg * (1. - sg), self.beta / (x + 1.))
```

`β/(x+1)` is evaluated for **all** `x` before `where` selects; at `x = −1` it is
`inf`. The value is correct (the `x<0` branch is selected), but JAX's
`where`-with-non-finite-branch produces `nan` cotangents when this is
differentiated (e.g. the parameter-gradient path), the classic "double where"
hazard.

**Fix** — guard the denominator so the dead branch stays finite:

```python
x_safe = jnp.where(x < 0., 1.0, x)
dx = jnp.where(x < 0., self.alpha * sg * (1. - sg), self.beta / (x_safe + 1.))
```

No change to returned values (only the dead branch is affected).

### 9 — `LogTailedRelu.surrogate_grad`: `inf` in the dead `where` branch

```python
dx = jnp.where(x > 1, 1 / x, jnp.where(x > 0, 1., self.alpha))
```

`1/x` is evaluated for all `x` (→ `inf` at `x=0`). Same nan-gradient hazard.

**Fix**:

```python
x_safe = jnp.where(x > 1, x, 1.0)
dx = jnp.where(x > 1, 1 / x_safe, jnp.where(x > 0, 1., self.alpha))
```

---

## Documentation / design notes (no behaviour change)

### 11 — Parameter-gradient semantics of the custom JVP

`_heaviside_jvp` returns `output_tangent = dx·tx + tdx`. Because `dx =
surrogate_grad(stop_gradient(x))`, differentiating the surrogate output w.r.t.
**`x`** gives the surrogate gradient (correct), but differentiating w.r.t. a
**surrogate parameter** (e.g. `alpha`) returns `d(surrogate_grad)/d(param)`
rather than the mathematically-true `0` (the Heaviside output does not depend on
`alpha`). This is *relied upon* by the existing `test_grad_of_parameters_*`
tests and is therefore retained, but it is surprising and must be documented:
treating a surrogate parameter as a trainable `ParamState` will not produce a
meaningful loss gradient. Added to the `Surrogate` docstring.

### 12 — Gradient-only surrogates lack `surrogate_fun`

`ReluGrad`, `GaussianGrad`, `MultiGaussianGrad`, `InvSquareGrad`, `SlayerGrad`
define only `surrogate_grad`; calling `surrogate_fun` raises
`NotImplementedError` (inherited from `Surrogate`). This is acceptable
(`surrogate_fun` is only used for visualization/analysis) but was undocumented;
a `Notes` sentence is added to each.

### 13 — `_heaviside_transpose` is dead code

The function is defined but its registration is commented out; JAX derives the
transpose automatically from the JVP (the tangent `dx·tx + tdx` is built from
ordinary `mul`/`add` primitives that are themselves transposable). Documented in
a comment so future readers do not assume it is wired in.

### 14 — Module docstring examples

The `__init__.py` examples mix `bst.transform.grad`, `bst.augment.vector_grad`,
and `nn.Module.forward`/`__call__` styles inconsistently and assume
dimensionless input. Left functionally as-is (they are illustrative, not
executed in tests) but a short note is added that surrogate inputs are
dimensionless (`v − v_th` in the same units), since passing a unitful
`brainunit.Quantity` to the `heaviside_p` primitive is unsupported.

---

## Test plan

The new/updated tests target **correctness**, not just plumbing:

1. **Function/derivative consistency** — for every surrogate that implements
   `surrogate_fun`, assert `autodiff(surrogate_fun) ≈ surrogate_grad` over a grid
   (catches #2, #3, #4, #5, #6).
2. **Surrogate-function shape** — for the symmetric surrogates assert
   `surrogate_fun(0) = 0.5`, monotone increasing, and (where bounded) range
   ⊂ `[0, 1]` (catches #2, #3).
3. **Reference values** — `GaussianGrad` vs `scipy.stats.norm.pdf`; `SlayerGrad`
   vs `exp(−α|x|)`; `InvSquareGrad` vs `1/(α|x|+1)²`; `ReluGrad` triangle peak;
   `PiecewiseQuadratic` continuity at `|x|=1/α`; `PiecewiseLeakyRelu` central
   slope `1/(2w)` (catches #1, #4, #5).
4. **`SquarewaveFourierSeries` term count** — assert `n` distinct harmonics are
   present (catches #7).
5. **Numerical robustness** — `S2NN`/`LogTailedRelu` finite gradients (incl.
   parameter gradients) at `x=−1`/`x=0` (catches #8, #9).
6. Retain all existing plumbing/`__repr__`/`__hash__`/batching tests; update the
   two that pinned buggy behaviour (`test_erf_at_zero`).

Target: > 90 % line coverage of `braintools/surrogate/` (baseline already 100 %;
kept at 100 % after the fixes).

---

## References

- Neftci, Mostafa & Zenke (2019), *Surrogate Gradient Learning in SNNs*, IEEE SPM.
- Yin, Corradi & Bohté (2021), *Accurate and efficient time-domain classification…*, Nat. Mach. Intell. (Gaussian surrogates).
- Suetake et al. (2022), *S2NN*, arXiv:2201.10879.
- Herranz-Celotti & Rouat (2022), *Surrogate Gradients Design*, arXiv:2202.00282 (q-PseudoSpike).
- Shrestha & Orchard (2018), *SLAYER*, NeurIPS.
- Upstream reference: `brainpy.math.surrogate._one_input_new` (2.7.8).
