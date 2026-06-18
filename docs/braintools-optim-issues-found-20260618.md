# `braintools/optim/` — Issues, Bugs, Edge Cases & Missing Features

**Reviewer role:** Senior Python architect / JAX & Optax expert / BrainX developer
**Date:** 2026-06-18
**Scope reviewed:**

- `braintools/optim/__init__.py` (module docstring + public API)
- `braintools/optim/_base.py` (`Optimizer`, `OptimState`)
- `braintools/optim/_optax_optimizer.py` (`OptaxOptimizer` + 24 optimizers) (+ test)
- `braintools/optim/_optax_lr_scheduler.py` (`LRScheduler` + 18 schedulers) (+ test)
- `braintools/optim/_scipy_optimizer.py` (`ScipyOptimizer`) (+ test)
- `braintools/optim/_nevergrad_optimizer.py` (`NevergradOptimizer`) (+ test)
- `braintools/optim/_sofo_optimizer.py` (`SOFO`, `SOFOScan`) (+ test)
- `braintools/optim/_state_uniquifier.py` (`UniqueStateManager`) (+ test)
- `docs/apis/optim.rst`, `docs/optim/*.ipynb`

**Test status at review time:** `391 passed` (clean tree, CPU). The suite is green but gives **false confidence**: it almost never calls `.step()` for the exotic optimizers (Lookahead, Fromage, Adafactor, Rprop, Lamb, SM3, Novograd), never exercises `add_param_group` end-to-end, never checks that a scheduler's value is *actually applied to the gradient* (only that `current_lr` reports a number), and the LR-scheduler docstring "worked examples" are not doctested. Consequently a large class of correctness bugs is invisible to CI.

**Verification key:** Findings tagged **[V]** were reproduced directly by the reviewer against the current tree (`jax 0.10.1`, `optax 0.2.6`, `brainstate 0.5.0`, `scipy 1.17.1`, `nevergrad 1.0.12`). Others were established by precise code reading and/or sub-agent repros.

---

## Executive summary

The module is broad and the *happy path* for the mainstream optimizers (SGD, Adam, AdamW, RMSprop) and the basic schedulers (StepLR, ExponentialLR, CosineAnnealingLR) works. But the review surfaced **numerous real correctness defects**, several of which silently produce wrong optimization:

- **Wrong algorithm implementations.** `Adagrad` is actually RMSprop (`scale_by_rms` instead of `scale_by_rss`) **[V]**; `Rprop` steps are scaled by `lr²`; `Lamb` applies its transforms in the wrong order; `Adafactor`'s default `decay_rate=-0.8` is wrong-signed (diverging second moment); `Fromage` is plain momentum-SGD with a fabricated docstring; `Lookahead` **crashes on every `.step()`** **[V]**.
- **Weight decay is silently decoupled** (AdamW-style) for *every* adaptive optimizer, so `Adam(weight_decay=x)` is byte-identical to `AdamW(weight_decay=x)` **[V]** — contradicting both the docstrings (which write coupled L2) and PyTorch.
- **The scheduler→optimizer bridge is broken at the contract level.** `LRScheduler.__call__(count)` ignores `count` and returns the last mutated value, so two schedulers (`SequentialLR`, `CosineAnnealingWarmRestarts`) **apply a stale constant LR to the gradients while *reporting* a different value** **[V]**; the very first optimizer step always uses `base_lr` instead of the scheduled epoch-0 value (defeating warmup); `ChainedScheduler` never combines schedulers and hangs epoch-driven loops; `OneCycleLR`/`SequentialLR` are off-by-one / discontinuous.
- **`add_param_group` + `step()` crashes** **[V]**; nested/list parameter pytrees crash the optimizer **[V]**; `load_state_dict` mis-iterates a dict.
- **Black-box optimizers**: `ScipyOptimizer` returns `None` on a diverging loss **[V]** and crashes on TNC/SLSQP under the default float32 dtype; `NevergradOptimizer` crashes on all-NaN losses and silently ignores `num_workers`.
- **The module-level docstring is badly out of date**: ~13 of its code examples raise `TypeError`/`AttributeError` on copy-paste (a stale `initial_lr`→`base_lr` rename, `bst.Module`→`bst.nn.Module`, wrong Scipy/Nevergrad signatures, etc.) **[V]**.

### Severity index

| ID | Sev | Area | One-line |
|----|-----|------|----------|
| **O-1** | **High** | optimizer | `Adagrad` uses `scale_by_rms` → it is RMSprop, not Adagrad **[V]** |
| **O-2** | **High** | optimizer | `weight_decay` is decoupled for all adaptive opts; `Adam`≡`AdamW`, contradicts docs/PyTorch **[V]** |
| **O-3** | **High** | optimizer | `Lookahead.step()` crashes (`AttributeError`) — double-appends base opt + wrong wrapping **[V]** |
| **O-4** | **High** | optimizer | `add_param_group` + `step()` crashes (`Dict key mismatch`) **[V]** |
| **O-5** | **High** | optimizer | `Rprop` step scaled by `lr²` (LR applied twice) → ~100× too small at defaults |
| **O-6** | **High** | optimizer | `Adafactor` default `decay_rate=-0.8` wrong-signed → diverging second moment **[V]** |
| **O-7** | **High** | optimizer | `Lamb` chains trust-ratio/adam/decay in the wrong order |
| **O-8** | **High** | optimizer | `Fromage` is momentum-SGD mislabeled; docstring fabricates the algorithm |
| **O-9** | Medium | optimizer | `current_lr` setter has no effect on the actual update |
| **O-10** | Medium | optimizer | `LBFGS` ignores LR schedulers (LR frozen at construction) |
| **O-11** | Medium | optimizer | `Novograd` weight-decay not folded into momentum (contradicts own math) |
| **O-12** | Medium | optimizer | `Adagrad.lr_decay` accepted/documented but never used |
| **O-13** | Medium | optimizer | `load_state_dict` iterates a dict's keys for `param_groups_opt_states` **[V]** |
| **O-14** | Medium | optimizer | base docstrings advertise a `closure` param that `step()` doesn't accept |
| **O-15** | Low | optimizer | `assert isinstance(...)` for runtime validation (stripped by `-O`) **[V]** |
| **O-16** | Low | optimizer | dead fallback branch + stale review comments |
| **O-17** | Low | optimizer | docstring examples use nonexistent APIs (`TanhT`, raw `tx.update`) |
| **O-18** | Low | optimizer | `Adafactor(lr=0.0)` silently becomes `1e-3` (`lr or 1e-3`) |
| **S-1** | **High** | scheduler | `LRScheduler.__call__(count)` ignores `count` → not a pure schedule (root cause) **[V]** |
| **S-2** | **High** | scheduler | `SequentialLR`/`CosineAnnealingWarmRestarts` never update `_current_lrs` → stale LR applied **[V]** |
| **S-3** | **High** | scheduler | initial `_current_lrs = base_lr` (not `get_lr@0`) → first step uses wrong LR (warmup defeated) |
| **S-4** | **High** | scheduler | `SequentialLR` passes global epoch to sub-scheduler (no milestone offset) → discontinuity |
| **S-5** | **High** | scheduler | `ChainedScheduler` returns only last scheduler; `last_epoch` never advances → infinite loop |
| **S-6** | **High** | scheduler | `OneCycleLR` off-by-one (`step_num=last_epoch+1`); `initial_lr` never produced |
| **S-7** | **High** | scheduler | divide-by-zero / silent NaN for zero-period params (`PolynomialLR(total_iters=0)` → NaN) |
| **S-8** | Medium | scheduler | `get_last_lr` missing on `LRScheduler` but `optimizer.get_last_lr()` calls it → `AttributeError` |
| **S-9** | Medium | scheduler | `WarmupCosineSchedule` ignores `base_lrs[1:]` (multi-group broken) |
| **S-10** | Medium | scheduler | docstrings say `last_epoch` "Default: -1" but the real default is `0` (≥9×) **[V]** |
| **S-11** | Medium | scheduler | docstring worked-example outputs are off-by-one (StepLR/ExponentialLR/…) |
| **S-12** | Medium | scheduler | `PiecewiseConstantSchedule` uses values as absolute; docstring says multiplier; `base_lr` dead |
| **S-13** | Medium | scheduler | inconsistent `base_lr` naming across schedulers; silent `-1e-3` fallback in `__call__` |
| **S-14** | Low | scheduler | `CosineAnnealingLR` oscillates back up past `T_max` (undocumented) |
| **S-15** | Low | scheduler | `state_dict` drops aux state (ReduceLROnPlateau `best`, WarmRestarts `T_cur`/`T_i`) |
| **S-16** | Low | scheduler | `Chained`/`Sequential` cannot hold a `ReduceLROnPlateau` (signature mismatch) |
| **S-17** | Low | scheduler | `attach_optimizer` error message garbled: `"an Optaxgot ..."` **[V]** |
| **X-1** | **High** | scipy | `TNC`/`SLSQP` crash under default float32 dtype; `TNC` is in the docstring example |
| **X-2** | **High** | scipy | `minimize` returns `None` when all losses are non-finite **[V]** |
| **N-1** | **High** | nevergrad | all-NaN losses crash via `np.nanargmin` (`All-NaN slice`) |
| **X-3** | Medium | scipy | `jac`/`callback` always passed → `RuntimeWarning` for gradient-free methods + wasted tracing |
| **N-2** | Medium | nevergrad | `num_workers` is a documented no-op (no real parallelism) |
| **N-3** | Medium | nevergrad | no seeding / reproducibility control |
| **X-4** | Medium | both | `minimize` methods undocumented; `n_iter` semantics differ & are non-obvious |
| **X-5** | Low | both | black-box optimizers don't honor the `Optimizer` base contract |
| **N-4** | Low | nevergrad | `concat_parameters` docstring overstates broadcasting |
| **N-5** | Low | nevergrad | `budget` silently ignored relative to `n_iter * n_sample` |
| **U-1** | **High** | uniquifier | nested-dict & list/tuple pytrees don't round-trip → optimizer `step()` crashes **[V]** |
| **F-1** | **High** | sofo | `damping=0` → explosive/wrong direction (esp. CE rank-deficient GGN) |
| **U-2** | Medium | uniquifier | `to_dict` path-string collisions silently drop states |
| **F-2** | Medium | sofo | fixed `key` reuses identical random tangents → defeats SOFO's stochastic subspace |
| **F-3** | Medium | sofo | `SOFO.update(inputs, targets)` violates base `update(grads)` contract (LSP) |
| **F-4** | Low | sofo | dead `return_loss` branch + loss computed twice |
| **F-5** | Low | sofo | `_make_grad_fn` rebuilt every `step` (eager-mode perf) |
| **B-1** | Low | base | non-NumPy-doc docstrings in `_base.py` & `_state_uniquifier.py` |
| **U-3** | Low | uniquifier | `merge_with` leaves `pytree_structure` stale |
| **D-1** | **High** | docs | all scheduler examples use `initial_lr=` (real kwarg `base_lr=`) **[V]** |
| **D-2** | **High** | docs | `WarmupScheduler` example: kwargs `warmup_steps/peak_lr/init_lr` don't exist |
| **D-3** | **High** | docs | `WarmupCosineSchedule` example: kwargs `peak_lr/end_lr` don't exist |
| **D-4** | **High** | docs | `ScipyOptimizer` examples omit required `loss_fun`/`bounds`; bogus `optimizer=`/options-only ctor |
| **D-5** | **High** | docs | `NevergradOptimizer` examples use wrong kwarg `optimizer=` + omit required args |
| **D-6** | **High** | docs | `Lookahead` example passes a braintools optimizer + `slow_step_size` |
| **D-7** | **High** | docs | `PolynomialLR` example uses `total_steps` (real `total_iters`) |
| **D-8** | **High** | docs | `Adafactor` example uses nonexistent `min_dim_size_to_factor` |
| **D-9** | **High** | docs | `Momentum(nesterov=True)` — `Momentum` has no `nesterov` (use `MomentumNesterov`/`SGD`) |
| **D-10** | **High** | docs | quick-start `bst.Module` doesn't exist (real `bst.nn.Module`) **[V]** |
| **D-11** | **High** | docs | notebook 05 `Lookahead(braintools.optim.SGD(...))` errors |
| **D-12** | **High** | docs | notebook 05 `LBFGS(..., line_search_fn='zoom')` errors |
| **D-13** | Medium | docs | false claim "Scheduler step is handled automatically" |
| **D-14** | Medium | docs | quick-start examples use `jnp`/`data`/`target` without import/definition |
| **D-15** | Low | docs | `OptimState` exported but missing from `docs/apis/optim.rst` |

---

## High severity

### O-1 — `Adagrad` is RMSprop (wrong optax transform) **[V]**
**Location:** `_optax_optimizer.py:1396`.
**Evidence [V]:** `default_tx` calls `optax.scale_by_rms(initial_scale=self.initial_accumulator_value, eps=self.eps)`. `scale_by_rms` is an *exponentially-decayed* mean of squared grads (RMSprop). Adagrad needs the *monotonic running sum of squares* — `optax.scale_by_rss` (which `optax.adagrad` itself uses). Reproduced:
```
braintools Adagrad deltas: [-0.31623, -0.03315, -0.03473]
optax.adagrad   deltas:    [-0.09535, -0.00949, -0.00945]   MATCH: False
```
The defining "monotonically shrinking effective LR" property is absent.
**Impact:** Every `braintools.optim.Adagrad` user silently gets RMSprop dynamics.
**Fix:** Use `optax.scale_by_rss(initial_accumulator_value=self.initial_accumulator_value, eps=self.eps)`; also implement or drop `lr_decay` (O-12).

### O-2 — `weight_decay` is decoupled for all adaptive optimizers; `Adam` ≡ `AdamW` **[V]**
**Location:** `_optax_optimizer.py:251-254` (Adam/base) and the same pattern in `RMSprop`, `Adagrad`, `Adadelta`, `Adamax`, `Nadam`, `RAdam`, `Yogi`, `AdaBelief`, `SM3`, `Novograd`.
**Evidence [V]:** Every adaptive optimizer chains `scale_by_<adaptive>()` → `add_decayed_weights(wd)` → `scale_by_schedule`. Because `add_decayed_weights` runs *after* the adaptive scaling, `+wd·θ` bypasses the adaptive denominator — that is *decoupled* (AdamW) decay, not the coupled L2 the docstrings/math claim. Reproduced — `Adam(weight_decay=0.1)` and `AdamW(weight_decay=0.1)` give byte-identical updates `[0.890001, 0.890001, 0.890001]`.
**Impact:** Silent wrong regularization vs documentation & vs PyTorch; `Adam` and `AdamW` are indistinguishable; ported training recipes won't reproduce.
**Fix:** For the PyTorch-parity optimizers, apply coupled decay (add `wd·θ` to the gradient *before* the adaptive transform). Keep decoupled only where intended (`AdamW`, `Lamb`). Document the change. (Also feed decay through the momentum buffer for `SGD`/`Momentum` to match PyTorch — see O-2 note / former "SGD-15".)

### O-3 — `Lookahead.step()` crashes on every call **[V]**
**Location:** `_optax_optimizer.py:2799-2801`.
**Evidence [V]:** `default_tx` does `transforms.append(self.base_optimizer)` **and then** `optax.lookahead(self.base_optimizer, …)`. `optax.lookahead` is a *standalone* optimizer whose params must be wrapped in `optax.LookaheadParams` (`.fast`/`.slow`); it cannot be chained after another transform nor consume a plain dict. Reproduced: `Lookahead(optax.sgd(0.1)).step({'w': …})` → `AttributeError: 'dict' object has no attribute 'fast'`. It also double-applies the base optimizer. Tests pass only because they never call `.step()`.
**Impact:** The class is unusable; any training loop crashes.
**Fix:** Special-case Lookahead (don't chain the base optimizer; wrap/extract `LookaheadParams` around the registered values), or remove the class. Add an end-to-end `.step()` test.

### O-4 — `add_param_group` + `step()` crashes **[V]**
**Location:** `_optax_optimizer.py:295-345` (`add_param_group`) and `:470-496` (multi-group `step`).
**Evidence [V]:** `add_param_group` builds a *separate* `UniqueStateManager` and never merges the new params into `self.param_states`. In the multi-group `step` branch, `param_values = self.param_states.to_pytree_value()` lacks the new group's keys, but `all_updates` contains them, so `optax.apply_updates(param_values, all_updates)` raises. Reproduced: `Dict key mismatch; expected keys: ['w1']; present keys: ['w1', 'w2']`.
**Impact:** The advertised multi-group feature crashes on the first step; the class docstring example fails.
**Fix:** Register new-group params into `self.param_states` (and the default group) in `add_param_group`, and build the apply-set from all groups in `step`. Add an end-to-end test.

### O-5 — `Rprop` step scaled by `lr²` (LR applied twice)
**Location:** `_optax_optimizer.py:3623-3632`.
**Evidence:** `default_tx` appends `optax.scale_by_rprop(learning_rate=self.base_lr, …)` (which already bakes the LR into the step) **and then** `optax.scale_by_schedule(self._lr_scheduler)` (multiplies by `−lr` again). Sub-agent repro at `lr=0.01`: braintools `|Δ| = [0, 1e-4, 1.2e-4, …]` vs `scale_by_rprop` alone `[0, 0.01, 0.012, …]` — exactly `lr×` smaller.
**Impact:** Rprop steps ~100× too small at the default `lr=1e-2`; convergence crippled.
**Fix:** Pass `optax.scale_by_rprop(learning_rate=1.0, …)` and let `scale_by_schedule` apply the LR once (so schedulers still work).

### O-6 — `Adafactor` default `decay_rate=-0.8` is wrong-signed **[V]**
**Location:** `_optax_optimizer.py:3835` (default), `:3864` (passed to `scale_by_factored_rms`), docstring `:3656-3658`.
**Evidence [V]:** `optax.adafactor` passes `decay_rate=0.8` (positive); the effective coefficient is `1 − (t)**(−decay_rate)`. With `-0.8` the coefficients are negative and diverging `[0, −0.74, −1.41, …]`; with `0.8` they are `[0, 0.43, 0.59, …] → 1`. Confirmed `decay_rate=-0.8` at both the default and the call site.
**Impact:** Out-of-the-box `Adafactor()` uses an invalid (negative, growing) second-moment decay → unstable/garbage updates.
**Fix:** Default `decay_rate=0.8`, pass through unchanged; fix the docstring formula (`1 − (step+1)^(−decay_rate)`).

### O-7 — `Lamb` chains transforms in the wrong order
**Location:** `_optax_optimizer.py:2326-2332`.
**Evidence:** braintools order = `scale_by_trust_ratio()` → `scale_by_adam()` → `add_decayed_weights()` → schedule. Reference `optax.lamb` = `scale_by_adam()` → `add_decayed_weights()` → `scale_by_trust_ratio()` → `scale_by_learning_rate()`. So LAMB must compute the Adam update + decoupled decay **then** apply the layer-wise trust ratio; braintools applies the trust ratio to the *raw gradient*.
**Impact:** `Lamb` produces incorrect updates; the trust-ratio normalization (the point of LAMB) operates on the wrong quantity.
**Fix:** Reorder to match `optax.lamb`, or delegate to `optax.lamb`.

### O-8 — `Fromage` is momentum-SGD mislabeled; fabricated docstring
**Location:** `_optax_optimizer.py:5020-5030`.
**Evidence:** `default_tx` uses only `optax.trace(decay=momentum)` + `scale_by_schedule` (momentum SGD). The real Fromage (`optax.fromage`) is `chain(scale_by_trust_ratio(min_norm), scale_by_learning_rate(λc·lr), add_decayed_weights(λc−1))`. The docstring also invents a false acronym and update math.
**Impact:** Users selecting `Fromage` get momentum-SGD; results bear no relation to the named algorithm.
**Fix:** Delegate to `optax.fromage` (or remove the class) and rewrite the docstring to match the real reference (Bernstein et al.).

### S-1 — `LRScheduler.__call__(count)` ignores `count` (root cause) **[V]**
**Location:** `_optax_lr_scheduler.py:137-142`.
**Evidence [V]:** `__call__` returns `-self._current_lrs.value[0]` regardless of `count`; `s(0)==s(5)==s(100)`. `optax.scale_by_schedule` is contractually a *pure* `count → scale` function. Here correctness depends entirely on host-side `step()` mutation order. This is the enabler of S-2 and S-3.
**Impact:** Fragile contract; any path that evaluates the schedule at a different `count` (or before `step()`) gets a stale value; cannot be lowered/traced independently.
**Fix:** Make `__call__(count)` compute the LR from `count` directly (evaluate the closed-form `get_lr` at `count`), so the optimizer is independent of host stepping. This fixes the root cause of S-2/S-3.

### S-2 — `SequentialLR` & `CosineAnnealingWarmRestarts` apply a stale LR **[V]**
**Location:** `_optax_lr_scheduler.py:4013` (`SequentialLR.step`), `:4357` (`CosineAnnealingWarmRestarts.step`).
**Evidence [V]:** Both override `step()` and write to `optimizer.param_groups[...]['lr']`/`optimizer.current_lr`, but never call `apply()` and never set `self._current_lrs`. Since the optimizer scales gradients by `__call__ → _current_lrs.value[0]`, the applied LR stays at the initial `base_lr`. Reproduced: after 6 SequentialLR steps, `reported current_lr=1.0` **and** effective `|Δ|=1.0` (should have decayed to ≈0.125).
**Impact:** Training with these two schedulers scales gradients by the constant initial LR; warm restarts / sequential phases are silently ignored. Monitoring `current_lr` won't reveal it.
**Fix:** After updating their internal counters, both `step()` overrides must refresh `_current_lrs` (mirror base `apply()`). Subsumed if S-1 is fixed (then `step()` only needs to advance counters).

### S-3 — First optimizer step uses `base_lr`, not the scheduled epoch-0 LR
**Location:** `_optax_lr_scheduler.py:79` (`self._current_lrs = OptimState(list(self.base_lrs))`).
**Evidence:** PyTorch convention is `optimizer.step(); scheduler.step()`, so the first gradient step runs while `_current_lrs` is still the constructor value. For `LinearLR(start_factor=0.1)` the first applied LR is `base_lr` (1.0), not the warmup start (0.1). Same for `ConstantLR(factor)`, `OneCycleLR`, `WarmupCosineSchedule`, `CyclicLR`.
**Impact:** The first step (and effectively the first epoch) uses the wrong LR; warmup-on-step-0 is defeated.
**Fix:** Subsumed by S-1 (compute from `count`). Otherwise initialize `_current_lrs` from `get_lr()` at the end of `__init__`.

### S-4 — `SequentialLR` passes the global epoch to the sub-scheduler
**Location:** `_optax_lr_scheduler.py:4013-4028`.
**Evidence:** `step` forwards the *global* epoch to the activated sub-scheduler instead of the milestone-relative epoch. With warmup→`ExponentialLR(gamma=0.5)` at milestone 5: epoch 5 → `gamma^5` instead of `gamma^0=1.0` (PyTorch resets the new scheduler at the milestone). Produces an abrupt incorrect drop at every transition.
**Impact:** Every phase transition is discontinuous/incorrect for any non-constant follow-on scheduler.
**Fix:** When stepping sub-scheduler `i>0`, pass `epoch − milestones[i-1]`.

### S-5 — `ChainedScheduler` doesn't combine; `last_epoch` never advances → infinite loop
**Location:** `_optax_lr_scheduler.py:3740-3745`.
**Evidence:** `step` iterates sub-schedulers but never touches `self.last_epoch`; `get_lr` returns only `self.schedulers[-1].get_lr()`. After 16 steps `chained.last_epoch == 0`; `while chained.last_epoch < e: step()` hangs forever. Docstring claims multiplicative combination; only the last scheduler's value is returned (warmup factor dropped).
**Impact:** "Warmup + decay" via `ChainedScheduler` doesn't apply warmup; epoch-driven loops/`state_dict` consumers hang or break.
**Fix:** Implement true multiplicative chaining (multiply each sub-scheduler's factor relative to its `base_lr`) and advance `self.last_epoch` in `step`.

### S-6 — `OneCycleLR` off-by-one; `initial_lr` is never produced
**Location:** `_optax_lr_scheduler.py:2564-2565` (`step_num = self.last_epoch.value + 1`).
**Evidence:** With `max_lr=1.0, div_factor=25` (initial_lr=0.04), `get_lr()` at `last_epoch=0` returns 0.072 (the step-1 value), not 0.04. PyTorch yields `initial_lr` at step 0. The whole curve is shifted one step early.
**Impact:** The documented `initial_lr = max_lr/div_factor` is never applied; peak reached one step early.
**Fix:** Use `step_num = self.last_epoch.value` (0-indexed, consistent with the other schedulers) and re-derive the phase boundaries.

### S-7 — divide-by-zero / silent NaN for zero-period parameters
**Location:** `StepLR.get_lr` `:350`, `CosineAnnealingLR` `:1301`, `PolynomialLR` `:1577`, `WarmupScheduler` `:1868`, `LinearLR` `:3291`, `CyclicLR` `:2170-2171`.
**Evidence:** `PolynomialLR(total_iters=0)` → `get_lr@0 = nan` (silent!); `StepLR(step_size=0)`, `CosineAnnealingLR(T_max=0)`, `WarmupScheduler(warmup_epochs=0)`, `LinearLR(total_iters=0)`, `CyclicLR(step_size_up=0)` → `ZeroDivisionError`.
**Impact:** `PolynomialLR(total_iters=0)` poisons training with NaN; others crash at first `get_lr`. `warmup_epochs=0` ("no warmup") is a plausible request that should be a no-op.
**Fix:** Validate in each `__init__` (`if X <= 0: raise ValueError`) or clamp the denominator with `jnp.maximum(denom, 1)` (as `WarmupCosineSchedule` already does).

### X-1 — `ScipyOptimizer` `TNC`/`SLSQP` crash under default float32
**Location:** `_scipy_optimizer.py:260` (x0), `:275` (jac), `:431-435` (bounds).
**Evidence:** JAX default dtype is float32; scipy's TNC/SLSQP kernels require float64. `TNC` → `ValueError: Buffer dtype mismatch, expected 'float64_t' but got 'float'`; `SLSQP` → `All inputs to slsqp must be of type numpy.float64`. `TNC` appears in the `ScipyOptimizer` docstring (`:360`), so the documented example throws. Casting `x0`, the `jac` output, and `bounds` to float64 fixes both.
**Impact:** Two advertised methods unusable out of the box; documented example fails.
**Fix:** Force float64 for everything scipy touches (`np.asarray(..., dtype=np.float64)` for `x0_flat`, `jac` output, and `_flat_bounds`).

### X-2 — `ScipyOptimizer.minimize` returns `None` on non-finite loss **[V]**
**Location:** `_scipy_optimizer.py:463-482` (`best_fun = np.inf`; `if results.fun < best_fun`).
**Evidence [V]:** `inf < inf` and `nan < inf` are both `False`, so `best_res` is never assigned. Reproduced with an always-`inf` loss → `minimize(...)` returns `None`; caller then hits `AttributeError: 'NoneType' object has no attribute 'x'`.
**Impact:** Easy to trigger when a loss diverges/overflows in float32 or `maxiter=0`; opaque downstream failure.
**Fix:** `if best_res is None or results.fun < best_fun:`; if still `None` after the loop, return the last `results` and/or warn.

### N-1 — `NevergradOptimizer` crashes on all-NaN losses
**Location:** `_nevergrad_optimizer.py:324` (`np.nanargmin`), `:341` (`np.nanmin`).
**Evidence:** If every candidate is NaN, `np.nanargmin` raises `ValueError: All-NaN slice encountered` (nevergrad itself only warns and continues). Common with stiff/divergent ODE simulations.
**Impact:** A single fully-NaN iteration aborts the whole optimization with an opaque error.
**Fix:** Guard: if `np.all(np.isnan(errors))`, fall back to `optimizer.provide_recommendation()` (warn), or replace NaN with `+inf` before `argmin`; guard the `verbose` `nanmin` print too.

### U-1 — Nested-dict & list/tuple pytrees don't round-trip → optimizer `step()` crashes **[V]**
**Location:** `_state_uniquifier.py:112-157` (`_reconstruct_pytree`), surfacing at `_optax_optimizer.py:494-511`.
**Evidence [V]:** `OptaxOptimizer.step` assumes a *flat* `{key: State}` map (`for k in params.keys(): params[k].value = …`), but `UniqueStateManager._reconstruct_pytree` rebuilds the original *nested* structure (and turns lists/tuples into int-keyed dicts). Reproduced with the structure from `UniqueStateManager`'s own docstring: `register_trainable_weights({'layer1': {'weight': s1, 'bias': s2}})` then `step(...)` → `AttributeError: 'dict' object has no attribute 'value'`.
**Impact:** Any user following the `UniqueStateManager` docstring, or passing a nested/list param pytree to *any* optimizer (incl. SOFO), gets a hard crash on the first step. (Tests miss it because `model.states()` returns tuple-keyed flat dicts.)
**Fix:** Make the optimizer operate on the flattened `(path → State)` list consistently end-to-end (e.g. key everything by path-string via `to_dict()`/`to_dict_value()`), and rebuild containers via the stored `treedef`/`jax.tree.unflatten` (this also fixes U-2 and the list/tuple and prefix-collision variants).

### F-1 — SOFO `damping=0` produces an explosively wrong direction
**Location:** `_sofo_optimizer.py:151-153` (`damped_s = s_ + damping * jnp.max(s_)`).
**Evidence:** The relative-damping term vanishes at `damping=0`, so tiny/zero singular values are inverted directly (amplification ~1e7×). For `loss='ce'` the GGN `diag(p)−ppᵀ` is genuinely rank-deficient, so even at moderate settings the direction can be near-orthogonal to the true natural gradient (`cos≈0.004` at `damping=0` vs `0.62–0.91` at `1e-3`). (SOFO/SOFOScan math is otherwise correct — at full rank with zero damping the MSE case recovers the exact natural gradient.)
**Impact:** `damping=0` (a plausible "no regularization" choice the signature permits) silently yields garbage/divergent steps; CE is affected even near the default if ill-conditioned.
**Fix:** Add an absolute floor: `damped_s = s_ + damping*jnp.max(s_) + eps` (or `jnp.maximum(damped_s, eps*jnp.max(s_))`), and/or validate `damping > 0` / use truncated-SVD for rank-deficient directions.

### D-1 … D-12 — Module docstring & notebook examples raise on copy-paste
**Location:** `braintools/optim/__init__.py` (module docstring, lines ~42-326) and `docs/optim/05_advanced_optimizers.ipynb`.
**Evidence [V where noted]:** The big module docstring predates an `initial_lr`→`base_lr` rename and several signature changes. Each of the following raises `TypeError`/`AttributeError`:
- **D-1 [V]** all scheduler examples use `initial_lr=` — real kwarg is `base_lr=` (`:168,171,174,177,180,302,308`, plus `CosineAnnealingLR(initial_lr=…)`/`ExponentialLR(initial_lr=…)` in composite blocks).
- **D-2** `WarmupScheduler(warmup_steps=…, peak_lr=…, init_lr=…)` (`:183-187,300`) — real sig `WarmupScheduler(base_lr, warmup_epochs, warmup_start_lr)`.
- **D-3** `WarmupCosineSchedule(peak_lr=…, end_lr=…)` (`:206-211`) — real `base_lr`/`eta_min`.
- **D-4** `ScipyOptimizer(method=…, options=…)` / `ScipyOptimizer(method=…, bounds=…)` (`:219-236`) — required positional `loss_fun, bounds`.
- **D-5** `NevergradOptimizer(optimizer=…, budget=…, num_workers=…)` (`:244-263`) — kwarg is `method=`; `batched_loss_fun, bounds, n_sample` required.
- **D-6** `Lookahead(base_optimizer=Adam(...), sync_period=5, slow_step_size=0.5)` (`:150-154`) — `slow_step_size`→`alpha`, and base must be an `optax.GradientTransformation`.
- **D-7** `PolynomialLR(initial_lr=…, total_steps=…)` (`:180`) — `base_lr`/`total_iters`.
- **D-8** `Adafactor(lr=…, min_dim_size_to_factor=128)` (`:144`) — no such kwarg.
- **D-9** `Momentum(lr=…, momentum=…, nesterov=True)` (`:99`) — `Momentum` has no `nesterov`; use `MomentumNesterov` or `SGD(..., nesterov=True)`.
- **D-10 [V]** `class SimpleModel(bst.Module)` (`:42`) — real base `bst.nn.Module`.
- **D-11** notebook 05 `Lookahead(braintools.optim.SGD(...))` → `AttributeError 'SGD' object has no attribute 'init'`.
- **D-12** notebook 05 `LBFGS(..., line_search_fn='zoom')` → `TypeError` (kwarg is `linesearch`).
**Impact:** The first thing users copy from the module/quick-start fails immediately.
**Fix:** Rewrite each example against the real signatures (corrected forms enumerated in the audit notes); add doctest-friendly imports (D-14).

---

## Medium severity

### O-9 — `current_lr` setter has no effect on updates
**Location:** `_optax_optimizer.py:283-286` (setter), `:257` (schedule binding). Updates are scaled by `scale_by_schedule(self._lr_scheduler)`, whose `__call__` reads the *scheduler's* `current_lrs`, not `self._current_lr`. Setting `opt.current_lr = 1.0` does not change the step size. **Fix:** write through to the scheduler's `current_lrs` (and param-group lr States), or have the schedule read `self._current_lr`.

### O-10 — `LBFGS` ignores LR schedulers
**Location:** `_optax_optimizer.py:3334-3354`. `default_tx` calls `optax.lbfgs(learning_rate=self.current_lr, …)` (a float read once) and never appends `scale_by_schedule`. `LBFGS(lr=StepLR(...))` freezes at the initial value. **Fix:** plumb the schedule through (mind the sign — `optax.lbfgs` wants a positive schedule), or document that LBFGS ignores schedules.

### O-11 — `Novograd` weight-decay not folded into the momentum
**Location:** `_optax_optimizer.py:4806-4808`. `optax.scale_by_novograd` has its own `weight_decay` (folded into `m_t`, per the paper and braintools' own docstring math); braintools leaves it 0 and appends a separate `add_decayed_weights`. **Fix:** pass `weight_decay=self.weight_decay` into `scale_by_novograd`, drop the separate term.

### O-12 — `Adagrad.lr_decay` accepted/documented but unused
**Location:** `_optax_optimizer.py:1365/1373` (stored), never referenced. PyTorch applies `lr/(1+step·lr_decay)`. **Fix:** implement via a schedule, or remove the parameter + docstring entry.

### O-13 — `load_state_dict` iterates a dict's keys **[V]**
**Location:** `_optax_optimizer.py:574-579`. `state_dict()` stores `param_groups_opt_states` as a dict `{str(i): value}` (`:542-545`), but `load_state_dict` does `for i, s in enumerate(state_dict['param_groups_opt_states'])` — iterating a dict yields its *keys* (`"0"`, …), so `s` becomes a string. **Fix:** iterate `.items()` sorted by int key; restore only scalar leaves of `param_groups` (not param tensors).

### O-14 — `closure` documented but not implemented
**Location:** base docstring `:108`, `:379-388` vs `def step(self, grads)`. `optimizer.step(grads, closure=…)` raises `TypeError`. `Adam.__init__` also accepts deprecated `beta1`/`beta2` that the Parameters section omits. **Fix:** remove `closure` from docstrings (or implement it); document `beta1`/`beta2` as deprecated.

### S-8 — `get_last_lr` missing on `LRScheduler`
**Location:** `_optax_optimizer.py:585-589` calls `self._schedulers[-1].get_last_lr()`; no such method exists. `optimizer.get_last_lr()` raises `AttributeError` whenever a scheduler was registered via `add_scheduler`. **Fix:** add `def get_last_lr(self): return list(self._current_lrs.value)`.

### S-9 — `WarmupCosineSchedule` ignores `base_lrs[1:]`
**Location:** `_optax_lr_scheduler.py:4707,4714,4719`. Uses `base_lrs[0]` then `return [lr for _ in base_lrs]` → multi-group LRs collapse to group 0. **Fix:** compute per `base_lr`.

### S-10 — docstrings say `last_epoch` "Default: -1" but it's `0` **[V]**
**Location:** `_optax_lr_scheduler.py:182-183` and ≥8 more. All `__init__` use `last_epoch=0`. The PyTorch `-1` convention also implies different first-step semantics. **Fix:** replace with "Default: 0" everywhere.

### S-11 — docstring worked-example outputs are off-by-one
**Location:** StepLR `:228-233`, ExponentialLR `:623-626`, CosineAnnealingLR `:1105`, ExponentialDecayLR `:851`. The examples `step()` before printing, so e.g. StepLR prints `Epoch 29 → 0.01` while the doc claims `0.1`. **Fix:** print before stepping (or correct the expected outputs) and apply one convention consistently.

### S-12 — `PiecewiseConstantSchedule` values are absolute, not multipliers
**Location:** `_optax_lr_scheduler.py:5079` (`value = self.values[idx]`, no `base_lr`) vs docstring `η_t = η_base × v_i` (`:4749-4762`). `base_lr` is dead for this scheduler. **Fix:** pick a semantics and align code+docstring (recommend documenting "absolute values" + noting `base_lr` is unused, matching `optax.piecewise_constant_schedule`).

### S-13 — inconsistent `base_lr` naming; silent `-1e-3` fallback
**Location:** `OneCycleLR(max_lr=…)` `:2525`, `CyclicLR(base_lr, max_lr)` `:2143`, others `base_lr`; base `__call__` returns `-1e-3` when `current_lrs` empty `:142`. The meaning of `base_lrs` is overloaded (initial vs peak vs min). **Fix:** document per-scheduler; raise instead of the silent `-1e-3` fallback.

### X-3 — `jac`/`callback` always passed to scipy
**Location:** `_scipy_optimizer.py:270-295`. Gradient-free methods (Nelder-Mead/Powell/COBYLA) emit `RuntimeWarning: Method X does not use gradient information (jac)` every iteration, and a `jit(grad(...))` is built+traced needlessly. **Fix:** only construct/pass `jac` for gradient methods; pass `callback=None` when none set.

### N-2 — `num_workers` is a no-op
**Location:** `_nevergrad_optimizer.py:239,247,291-311`. Only forwarded to the nevergrad constructor; the eval loop is serial Python (parallelism actually comes from JAX vectorization over `n_sample`). **Fix:** document that `n_sample` controls batch size and `num_workers` only configures nevergrad's internal model — or genuinely parallelize.

### N-3 — no seeding / reproducibility
**Location:** `_nevergrad_optimizer.py:153-163,243-271`. No `seed` parameter; runs are non-deterministic. **Fix:** add `seed: Optional[int]`; seed `parametrization.random_state` in `initialize()`.

### X-4 — `minimize` methods undocumented; `n_iter` semantics differ
**Location:** `_scipy_optimizer.py:460`, `_nevergrad_optimizer.py:327` (both `__doc__ is None`). For scipy `n_iter` = random restarts; for nevergrad `n_iter` = ask/tell rounds (`≈ n_iter*n_sample` evals). **Fix:** add NumPy-doc to both, documenting `n_iter`, return types, and the `None`-on-failure behavior (after X-2).

### U-2 — `to_dict` path-string collisions silently drop states
**Location:** `_state_uniquifier.py:186-246`. `_path_to_string` joins with `.`, so a literal key `'a.b'` and the path `a→b` collide → one state overwritten. `to_dict_value()` feeds `register_trainable_weights`/`add_param_group`, so a collision drops a parameter silently. **Fix:** key by the structured path tuple (or unambiguous repr); or detect collisions and raise. (Resolved together with U-1.)

### F-2 — fixed `key` reuses identical random tangents
**Location:** `_sofo_optimizer.py:131`. A fixed `key` freezes SOFO's random reparameterization subspace forever (verified: two consecutive directions identical), defeating the algorithm's stochastic design. Every docstring example passes `key=jax.random.PRNGKey(0)`. **Fix:** treat `key` as a seed and `fold_in` a step counter each iteration; update examples to `key=None` (or seeded-then-folded).

### F-3 — `SOFO.update(inputs, targets)` violates the base contract
**Location:** `_base.py:47` (`update(self, grads)`), `_optax_optimizer.py:375` vs `_sofo_optimizer.py:298-304,449-456`. SOFO overrides `step`/`update` to take *data* instead of precomputed grads → generic loops written against `Optimizer.update(grads)` misbehave. **Fix:** give SOFO a distinct method (e.g. `step_with_data`) and raise from `update(grads)`, or document the divergence prominently.

### D-13 — false claim "Scheduler step is handled automatically"
**Location:** `braintools/optim/__init__.py:84`. The scheduler only advances on explicit `scheduler.step()`; the quick-start loop omits it, so the LR is frozen. **Fix:** add `scheduler.step()` to the loop and correct the comment.

### D-14 — examples use `jnp`/`data`/`target` without import/definition
**Location:** `braintools/optim/__init__.py:45-49,62,65,83`. Violates the project's self-contained-imports rule (`NameError` on copy-paste). **Fix:** add `import jax.numpy as jnp` and define/annotate `data`/`target`.

---

## Low severity / polish

- **O-15 [V]** `assert isinstance(...)` for runtime validation at `_optax_optimizer.py:404,421` (stripped under `python -O`) → use explicit `raise TypeError`.
- **O-16** dead fallback branch (`:435-448`, unreachable since `add_param_group` always sets `'tx'`) and stale comments (`:213-216` "already initialized in parent" — false; `:423,501` "Fix: create dict not set").
- **O-17** docstring examples use nonexistent APIs (`Rprop` `:3546` `brainstate.nn.TanhT()`; `LBFGS` `:3115` raw `optimizer.tx.update(...)`).
- **O-18** `Adafactor.__init__` `lr or 1e-3` (`:3849`) swallows a valid `lr=0.0` → use `1e-3 if lr is None else lr`.
- **S-14** `CosineAnnealingLR` oscillates back up past `T_max` (no clamp); docstring implies it holds at `eta_min`. Clamp `epoch_eff = jnp.minimum(epoch, T_max)` or document the SGDR oscillation.
- **S-15** `state_dict`/`load_state_dict` don't round-trip aux state (`ReduceLROnPlateau.best/num_bad_epochs/cooldown_counter`, `WarmRestarts.T_cur/T_i`) — the tests/docstrings manually copy these, confirming the gap. Add overrides.
- **S-16** `ChainedScheduler`/`SequentialLR` can't contain a `ReduceLROnPlateau` (it needs `metric`); document or special-case.
- **S-17 [V]** `attach_optimizer` message garbled: `"optimizer must be an Optaxgot ..."` (`:89`) → `"... must be an OptaxOptimizer, got ..."`.
- **X-5** `ScipyOptimizer`/`NevergradOptimizer` inherit `register_trainable_weights`/`update` → `NotImplementedError`; they only implement `minimize`. Either document the divergent contract on the base, or split a `BlackBoxOptimizer` base.
- **N-4** `concat_parameters` docstring (`_nevergrad_optimizer.py:36-60`) overstates broadcasting (it stacks; mismatched leaf shapes error). Reword.
- **N-5** `budget` is effectively ignored relative to `n_iter * n_sample` (`:238,291-342`). Document that total evals ≈ `n_iter*n_sample`; optionally warn if it exceeds `budget`.
- **F-4** dead `return_loss` branch + loss computed twice (`_sofo_optimizer.py:140,155-156`) → drop the dead capture or reuse the computed loss.
- **F-5** `_make_grad_fn` rebuilds the `GradientTransform` every `step` (`:277-290,426-440`); eager-mode re-tracing. Build once at registration (pass `targets` as an argument).
- **B-1** non-NumPy-doc docstrings in `_base.py` (`Parameters:` with a colon + dashed underline) and `_state_uniquifier.py` (Google-style `Args:`/`Returns:`) — convert to the mandated NumPy style.
- **U-3** `merge_with` never updates `self.pytree_structure` (`:303-330`) → stale after merge (currently latent). Update/clear it or remove the attribute.
- **D-15** `OptimState` is in `__all__` but absent from `docs/apis/optim.rst` autosummary → add it.

---

## Missing features / hardening (opportunistic)

1. **Per-optimizer numeric regression tests.** A single parametrized test that, for each optimizer, descends a 1-D quadratic and compares the first few updates against the corresponding `optax.<optimizer>` reference would have caught O-1, O-5, O-6, O-7, O-8 and O-2 at once.
2. **Scheduler "effective-LR" tests.** Assert that the LR actually applied to the gradient (not just `current_lr`) matches the closed form — would catch S-1/S-2/S-3/S-6.
3. **`closure` support** for parity with the PyTorch interface the module advertises (or remove the claim).
4. **MATLAB-v7.3-style "unsupported method" guidance** for scipy methods that need float64 (after X-1, surface a clear message rather than a Cython buffer error).
5. **Exception/seed exposure**: promote a `seed` for `NevergradOptimizer`; document `ScipyOptimizer` failure modes.

---

## Suggested remediation order

1. **Scheduler root cause (S-1)** — make `__call__(count)` pure; this collapses S-2, S-3 and de-risks O-9. Small, high-leverage.
2. **Wrong-algorithm optimizers (O-1, O-5, O-6, O-7, O-8, O-3)** — schedule with the new numeric regression tests; these silently corrupt training.
3. **Crashers (O-4, U-1, S-5, S-7, X-2, N-1, S-8)** — each is a hard failure on a documented path.
4. **Weight-decay semantics (O-2, O-11)** — decide coupled vs decoupled, align code+docs+tests; flag as a behavior change.
5. **Contract/medium (O-10, O-13, O-14, S-4, S-6, S-9..S-13, X-1, X-3, N-2, N-3, X-4, U-2, F-1, F-2, F-3)**.
6. **Docs (D-1..D-15)** — rewrite the module docstring + two notebook cells; add `OptimState` to the rst.
7. **Low-severity polish (O-15..O-18, S-14..S-17, X-5, N-4, N-5, F-4, F-5, B-1, U-3)** — opportunistic.

Every **[V]** finding was reproduced against the current tree during this review; the remainder are backed by precise code locations and sub-agent repros.
