# `braintools/optim/` — Re-audit Issues, Bugs & Edge Cases

**Reviewer role:** Senior Python developer / JAX & Optax expert
**Date:** 2026-06-18
**Scope reviewed:**

- `braintools/optim/__init__.py` (module docstring + public API)
- `braintools/optim/_base.py` (`Optimizer`, `OptimState`)
- `braintools/optim/_optax_optimizer.py` (`OptaxOptimizer` + 24 optimizers)
- `braintools/optim/_optax_lr_scheduler.py` (`LRScheduler` + 18 schedulers)
- `braintools/optim/_scipy_optimizer.py` (`ScipyOptimizer`)
- `braintools/optim/_nevergrad_optimizer.py` (`NevergradOptimizer`)
- `braintools/optim/_sofo_optimizer.py` (`SOFO`, `SOFOScan`)
- `braintools/optim/_state_uniquifier.py` (`UniqueStateManager`)
- `docs/apis/optim.rst`

**Context.** A prior audit (#111, same date) already fixed a large class of correctness
defects (O-1..O-18, S-1..S-17, U-1, F-1..F-3, X-1..X-3, N-1..N-5) and added a comprehensive
regression suite (`_audit_fixes_test.py`). This re-audit re-reads the *current* tree to surface
issues that survived or were not covered. The baseline suite is green: **538 passed**
(`jax 0.10.1`, `optax 0.2.6`, `brainstate 0.5.0`, `scipy 1.17.1`, `nevergrad 1.0.12`,
`torch 2.11.0`).

**Verification.** Every finding below was verified against the actual code plus the relevant
reference (`optax` source via `inspect.getsource`, `torch.optim` algorithms, or numeric repro).
Findings the review *cleared* (no bug) are listed at the end so the negative space is explicit.

---

## Severity index

| ID | Sev | Area | One-line |
|----|-----|------|----------|
| **R-1** | **Critical** | optimizer | `SM3` applies first-moment momentum twice (internal `scale_by_sm3` `b1=0.9` + extra `optax.trace`) |
| **R-2** | **High** | optimizer | `RMSprop(centered=True)` is a silent no-op (always `scale_by_rms`, never `scale_by_stddev`) |
| **R-3** | **High** | optimizer | `Nadam.momentum_decay` is accepted/documented but never applied (`scale_by_adam(nesterov=True)`) |
| **R-4** | **High** | docs | `LBFGS` docstring documents non-existent param `scale_init_hess` (actual: `scale_init_precond`) — examples raise `TypeError` |
| **R-5** | **Medium** | optimizer | `LBFGS` bakes the LR as a float at construction; an `LRScheduler` passed to it never takes effect |
| **R-6** | **Medium** | scheduler | `CosineAnnealingWarmRestarts.step(epoch=...)` ignores `epoch` for the cycle state |
| **R-7** | **Medium** | scheduler | `CyclicLR` / `OneCycleLR` / `ReduceLROnPlateau` enum params validated lazily or not at all |
| **R-8** | **Medium** | optimizer | No hyperparameter validation (negative `lr`/`weight_decay`/clip accepted silently) |
| **R-9** | **Medium** | docs | `Adafactor` example uses wrong-sign `decay_rate=-0.8` (contradicts the documented default `0.8`) |
| **R-10** | **Medium** | docs | `OneCycleLR.find_max_lr` example uses `pct_start=1.0`, which `__init__` rejects (non-runnable) |
| **R-11** | **Medium** | docs | `CyclicLR` `triangular2` example states growing peaks (0.0055/0.00325) instead of decaying (~0.00055/~0.000325) |
| **R-12** | **Low** | optimizer | Base `OptaxOptimizer.default_tx` applies *decoupled* weight decay, inconsistent with `Adam`/`add_param_group` (coupled) |
| **R-13** | **Low** | docs | `ExponentialDecayLR` Notes math uses `transition_steps`; the actual param is `decay_steps` |
| **R-14** | **Low** | docs | `CosineAnnealingWarmRestarts` examples call `.value` on a plain value and reassign the `OptimState` object |
| **R-15** | **Low** | docs | `PiecewiseConstantSchedule` example labels `values` "LR multipliers" — they are absolute LRs |
| **R-16** | **Low** | docs | `add_param_group` silently uses Adam scaling regardless of parent optimizer (undocumented) |
| **R-17** | **Low** | docs | `UniqueStateManager` uses Google-style (`Args:`/`Returns:`) docstrings, violating the project NumPy-doc rule |

---

## Detailed findings

### R-1 — `SM3` applies momentum twice — **Critical**
`braintools/optim/_optax_optimizer.py` (`SM3.default_tx`, ~L4752–4765)

```python
transforms.append(optax.scale_by_sm3(eps=self.eps))   # b1 left at optax default 0.9
if self.momentum > 0:
    transforms.append(optax.trace(decay=self.momentum))  # second momentum stage
```

`optax.scale_by_sm3(b1=0.9, b2=1.0, eps)` *internally* applies a first-moment EMA
(`nu = update_moment(up, state.nu, b1, 1)`), confirmed from `inspect.getsource`. Leaving `b1`
at its default and then chaining `optax.trace(decay=self.momentum)` smooths the gradient
**twice**. Worse, `momentum=0` does **not** disable momentum — the internal `b1=0.9` still runs.
The canonical `optax.sm3(lr, momentum)` is `chain(scale_by_sm3(momentum), scale(-lr))` — one
momentum stage driven by the user's `momentum`.

**Fix.** `optax.scale_by_sm3(b1=self.momentum, b2=1.0, eps=self.eps)` and remove the separate
`optax.trace`. Now `momentum=0` ⇒ no smoothing; `momentum=0.9` ⇒ single smoothing, matching
`optax.sm3`.

### R-2 — `RMSprop(centered=True)` silently ignored — **High**
`braintools/optim/_optax_optimizer.py` (`RMSprop.default_tx`, ~L1770)

`self.centered` is stored (L1744) and the docstring documents a distinct *centered* formula
(`normalizes gradient by variance estimate`), but `default_tx` unconditionally calls
`optax.scale_by_rms(...)` and never branches on `centered`. The existing test only checks the
attribute is stored, so the no-op is untested.

**Fix.** `optax.scale_by_stddev(decay=self.alpha, eps=self.eps)` when `self.centered` else
`optax.scale_by_rms(...)`. `scale_by_stddev` normalizes by `sqrt(E[g²] − E[g]² + eps)` (centered),
matching `optax.rmsprop(centered=True)`.

### R-3 — `Nadam.momentum_decay` never applied — **High**
`braintools/optim/_optax_optimizer.py` (`Nadam.default_tx`, ~L2038)

`Nadam` uses `optax.scale_by_adam(..., nesterov=True)`, which is a *fixed-β* Nesterov-Adam
(≡ `optax.nadam`). The documented `momentum_decay` (ψ, PyTorch's scheduled-momentum term) is
stored but has **no effect** — tuning it changes nothing.

**Fix.** Implement a custom `_scale_by_nadam(b1, b2, eps, momentum_decay)` transform reproducing
PyTorch `torch.optim.NAdam`'s scheduled momentum
(`μ_t = β1·(1 − ½·0.96^{t·ψ})`, running `μ`-product, two-term `m_hat`), verified numerically
against `torch.optim.NAdam`.

### R-4 — `LBFGS` docstring documents a non-existent parameter — **High (docs)**
`braintools/optim/_optax_optimizer.py` (L3225, L3372, L3448)

The Parameters section and two runnable examples reference `scale_init_hess`, but the
constructor parameter is `scale_init_precond` (L3495). `LBFGS(scale_init_hess=False)` raises
`TypeError`. **Fix:** rename all three docstring occurrences to `scale_init_precond`.

### R-5 — `LBFGS` ignores LR schedulers — **Medium**
`braintools/optim/_optax_optimizer.py` (`LBFGS.default_tx`, ~L3536)

`optax.lbfgs(learning_rate=self.current_lr, ...)` captures a concrete float at construction.
Unlike every other optimizer (which reads the live schedule via `scale_by_schedule`), passing
`LBFGS(lr=scheduler)` then calling `scheduler.step()` has no effect — contradicting the
documented `lr : float or LRScheduler`. **Fix:** pass a schedule callable that reads the live
scheduler each step (handling the scheduler's negated-LR convention).

### R-6 — `CosineAnnealingWarmRestarts.step(epoch=...)` ignores `epoch` — **Medium**
`braintools/optim/_optax_lr_scheduler.py` (`step`, ~L4493–4502)

`step(epoch)` sets `last_epoch = epoch` but advances the cycle state with an unconditional
`T_cur += 1`; `last_epoch` is never read by `get_lr` (which depends only on `T_cur`/`T_i`). So
`step(epoch=50)` does **not** place the schedule at epoch 50 (breaks resume-by-epoch / non-unit
stepping). PyTorch recomputes `T_cur`/`T_i` from the absolute epoch.

**Fix.** When `epoch is not None`, recompute via the SGDR closed form: for `T_mult == 1`,
`T_cur = epoch % T_0`, `T_i = T_0`; for `T_mult > 1`,
`n = ⌊log((epoch/T_0)(T_mult−1)+1)/log(T_mult)⌋`, `T_cur = epoch − T_0·(T_mult^n−1)/(T_mult−1)`,
`T_i = T_0·T_mult^n`.

### R-7 — Lazy / missing enum validation in schedulers — **Medium**
`braintools/optim/_optax_lr_scheduler.py`

- `CyclicLR.mode` is only validated inside `get_lr` (~L2235), not in `__init__`; `scale_mode`
  is never validated (an invalid value silently behaves as `'iterations'`).
- `OneCycleLR.anneal_strategy` is validated only inside `get_lr` (~L2641).
- `ReduceLROnPlateau` validates `factor < 1` but not `mode ∈ {min,max}` nor
  `threshold_mode ∈ {rel,abs}`; a typo silently *inverts* the improvement criterion.

**Fix.** Validate these enums in each `__init__` (fail-fast, like `step_size`/`pct_start`).

### R-8 — No hyperparameter validation — **Medium**
`braintools/optim/_optax_optimizer.py` (`OptaxOptimizer.__init__`)

Negative `lr` (float), negative `weight_decay`, and non-positive `grad_clip_norm`/
`grad_clip_value` are accepted silently and diverge/NaN far downstream. PyTorch raises
`ValueError`. **Fix:** validate in the base `__init__` (float `lr ≥ 0`, `weight_decay ≥ 0`,
positive clip thresholds).

### R-9 — `Adafactor` example uses wrong-sign `decay_rate` — **Medium (docs)**
`braintools/optim/_optax_optimizer.py` (L4010)

The "complete configuration" example passes `decay_rate=-0.8`, contradicting the Parameters
text ("Must be positive", default `0.8`). A negative exponent gives a diverging second-moment
decay. **Fix:** `decay_rate=0.8`.

### R-10 — `OneCycleLR.find_max_lr` example is non-runnable — **Medium (docs)**
`braintools/optim/_optax_lr_scheduler.py` (~L2456)

The documented `find_max_lr` helper builds `OneCycleLR(..., pct_start=1.0)`, but `__init__`
enforces `0 < pct_start < 1` → `ValueError`. **Fix:** use `pct_start=0.99`.

### R-11 — `CyclicLR` `triangular2` example numbers wrong — **Medium (docs)**
`braintools/optim/_optax_lr_scheduler.py` (~L2042–2044)

`triangular2` halves amplitude each cycle, so peaks are `0.001 → 0.00055 → 0.000325`. The
example instead claims "Second cycle … 0.0055" (10× too high, *growing*) and "Third … 0.00325".
**Fix:** correct the prose to decaying peaks `~0.00055` and `~0.000325`.

### R-12 — Base `default_tx` weight decay is decoupled (inconsistent) — **Low**
`braintools/optim/_optax_optimizer.py` (`OptaxOptimizer.default_tx`, ~L252–255)

The base fallback orders `scale_by_adam()` *before* `add_decayed_weights()` (decoupled / AdamW),
while the named `Adam` and `add_param_group` apply coupled L2 (decay before scaling). The base
fallback is rarely used directly, but the inconsistency is surprising. **Fix:** make the base
fallback coupled to match `Adam`.

### R-13 — `ExponentialDecayLR` Notes math uses `transition_steps` — **Low (docs)**
`braintools/optim/_optax_lr_scheduler.py` (~L840, L847)

The Notes formula divides by `transition_steps` (optax's internal name); the actual parameter
is `decay_steps`. **Fix:** replace `transition_steps` with `decay_steps` in the math block.

### R-14 — `CosineAnnealingWarmRestarts` example code is non-runnable — **Low (docs)**
`braintools/optim/_optax_lr_scheduler.py` (~L4362–4367, L4435–4445)

`old_T_cur = scheduler.T_cur.value` then `if scheduler.T_cur.value < old_T_cur.value` applies
`.value` to a plain array (`AttributeError`); the state example does
`new_scheduler.T_cur = state['T_cur']`, replacing the `OptimState` object instead of setting
`.value`. **Fix:** compare against `old_T_cur` directly and assign `…T_cur.value = …`.

### R-15 — `PiecewiseConstantSchedule` example mislabels `values` — **Low (docs)**
`braintools/optim/_optax_lr_scheduler.py` (~L4954, L4965)

`get_lr` returns `self.values[idx]` directly (absolute LRs), and the Parameters section says so,
but an example labels them "LR multipliers" and the inline comment computes LRs as if multiplied
by `base_lr`. **Fix:** drop "LR multipliers"; show the absolute `values` as the resulting LRs.

### R-16 — `add_param_group` always uses Adam scaling — **Low (docs)**
`braintools/optim/_optax_optimizer.py` (~L392)

Every added group uses `optax.scale_by_adam()` regardless of the parent optimizer (acknowledged
in a code comment but not user-facing). **Fix:** document this limitation in the
`add_param_group` docstring.

### R-17 — `UniqueStateManager` uses Google-style docstrings — **Low (docs)**
`braintools/optim/_state_uniquifier.py`

The public class/methods use `Args:`/`Returns:` (Google style), violating the project's
NumPy-doc requirement (CLAUDE.md). **Fix:** convert the public docstrings to NumPy style.

---

## Cleared on review (no change needed)

- **Optimizer math:** SGD / Momentum / MomentumNesterov (`optax.trace(nesterov=)`), Adam/AdamW
  coupled-vs-decoupled placement, Adamax, Adagrad (`scale_by_rss`), Adadelta, Lamb/Lars ordering,
  Lion decoupled decay, Yogi, AdaBelief, Rprop, Fromage, Novograd, RAdam — all match optax.
- **Scheduler math:** Step/MultiStep/Exponential/Cosine/Polynomial/Linear/Constant/Warmup/
  OneCycle/Cyclic(`triangular2` amplitude)/SequentialLR(`searchsorted` milestone offsets)/
  ChainedScheduler(multiplicative)/WarmupCosine/Piecewise — verified correct; `ReduceLROnPlateau`
  patience/cooldown/threshold logic correct.
- **JAX safety:** schedules are pure (LR read from `current_lrs`), all branching is on static
  Python attributes (not traced step values), `_write_back` path navigation is trace-safe,
  Lookahead `jnp.where` avoids Python branching.
- **Black-box & forward-mode:** `ScipyOptimizer` (float64 casting, non-finite handling,
  gradient-free Jacobian skip), `NevergradOptimizer` (all-NaN fallback, seeding, budget warning),
  `SOFO`/`SOFOScan` (damping floor, per-step key folding) — all correct.
- **API docs:** `docs/apis/optim.rst` autosummary matches `__init__.__all__` exactly.
