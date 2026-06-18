# `braintools/trainer/` — Issues, Bugs, Edge Cases & Missing Features

**Reviewer role:** Senior Python architect / JAX expert / BrainX developer
**Date:** 2026-06-18
**Scope reviewed:**

- `braintools/trainer/__init__.py`
- `braintools/trainer/_module.py` (+ `_module_test.py`)
- `braintools/trainer/_trainer.py` (+ `_trainer_test.py`, `_trainer_extra_test.py`)
- `braintools/trainer/_callbacks.py` (+ `_callbacks_test.py`)
- `braintools/trainer/_loggers.py` (+ `_loggers_test.py`)
- `braintools/trainer/_dataloader.py` (+ `_dataloader_test.py`)
- `braintools/trainer/_distributed.py` (**no test file**)
- `braintools/trainer/_checkpoint.py` (+ `_checkpoint_test.py`)
- `braintools/trainer/_progress.py` (+ `_progress_test.py`)
- Documentation: `docs/apis/` (the trainer module has **no** API page), `docs/index.rst`

**Test status at review time:** `403 passed` in normal mode. Per-file coverage:
`__init__` 100%, `_dataloader` 100%, `_progress` 100%, `_callbacks` 99%, `_checkpoint` 99%,
`_module` 98%, `_trainer` 91%, **`_loggers` 73%**, **`_distributed` (excluded from coverage by `pyproject.toml`)**.

---

## Executive summary

The trainer module is an ambitious PyTorch-Lightning-style framework for JAX/`brainstate`.
It is broad and the individual leaf utilities (datasets, samplers, progress bars,
checkpoint file I/O) are mostly solid. **However, the end-to-end training path and the
callback/metric plumbing contain a number of real correctness bugs**, and several
prominently-documented `Trainer` parameters are silent no-ops.

The reason these survived 403 passing tests is that the test suite is almost entirely
*unit* tests with concrete values; the **integration tests are shallow** (e.g.
`test_trainer_with_callbacks` only asserts `len(trainer.callbacks) == 2`, and
`test_simple_training_loop` runs `fit` with no validation/callbacks/logger and only
asserts `current_epoch >= 0`). No test drives a full `fit` with validation + a metric-
monitoring callback, which is exactly where the bugs live.

Most important findings:

- **`EarlyStopping` `min_delta` is sign-inverted for `mode='min'`** — it counts *worse*
  values as improvements (T-1).
- **Validation/test metrics are double-prefixed** (`val_` + user key `val_loss` →
  `val_val_loss`), so `monitor='val_loss'` never matches and `EarlyStopping` /
  `ModelCheckpoint` silently fail (or raise under `strict=True`) (T-2).
- **`ModelCheckpoint` evaluates its monitored metric at `on_train_epoch_end`, *before*
  validation runs**, so it checkpoints on stale/absent metrics (T-3).
- **The training step runs the forward pass twice** and the logged loss is computed from a
  *different* stochastic draw than the gradients (T-4).
- **`min_epochs` is dead code** and is not enforced against early stopping (T-5).
- **Checkpoint resume silently resets `global_step` to 0** (key name mismatch) (T-6).
- **Shuffling repeats the identical permutation every epoch** (T-7).
- **`accumulate_grad_batches`, `precision`, `deterministic`, `benchmark`, multi-optimizer,
  and `strategy=`-based multi-device training are documented but non-functional** (T-17…T-22).
- **The trainer module has zero rendered API documentation** (T-31).

Each finding carries a location, verified evidence, impact, and proposed solution. The
final section gives the remediation plan and explicit scope decisions (fix fully vs.
make-honest-and-document).

### Severity index

| ID | Severity | Area | One-line |
|----|----------|------|----------|
| T-1 | **High** | callbacks | `EarlyStopping.min_delta` sign-inverted for `mode='min'` |
| T-2 | **High** | trainer/metrics | Validation/test metric keys double-prefixed (`val_val_loss`) |
| T-3 | **High** | callbacks/ordering | `ModelCheckpoint` reads monitored metric before validation runs |
| T-4 | **High** | trainer/jit | Double forward pass; logged loss ≠ loss used for gradients |
| T-5 | **High** | trainer | `min_epochs` not enforced (dead `continue`) |
| T-6 | High | checkpoint | Resume resets `global_step` to 0 (`'step'` vs `'global_step'`) |
| T-7 | Medium | dataloader | Same shuffle every epoch (`reset()` re-seeds; no `set_epoch`) |
| T-8 | Medium | dataloader | `IterableDataset.__getitem__` re-creates iterator → returns elem 0 |
| T-9 | Medium | trainer/logging | Training `self.log` metrics (besides loss) never logged |
| T-10 | Medium | callbacks/jit | Leaked tracers in `PrintCallback`/`TQDMProgressBar` during training |
| T-11 | Medium | callbacks | `LearningRateMonitor` log wiped by `_reset_logged_metrics`; LR mis-extracted |
| T-12 | Medium | callbacks | `CallbackList` never dispatches test/predict/optimizer/backward/ckpt hooks |
| T-13 | Medium | trainer | `on_train_start/end`, `on_*_start/end`, backward/optim hooks never called |
| T-14 | Medium | loggers | `CSVLogger` drops late-appearing columns; `step=0` mishandled |
| T-15 | Medium | distributed | `FSDP` ctor crashes (`list.reshape`); `Mesh(list)` |
| T-16 | Medium | module | `freeze()`/`unfreeze()` are silent no-ops |
| T-17 | Medium | trainer | `accumulate_grad_batches` ignored |
| T-18 | Medium | trainer | `precision` ('16'/'bf16') ignored |
| T-19 | Low | trainer | `deterministic` / `benchmark` ignored |
| T-20 | Medium | trainer | `seed` seeds only numpy, not JAX/brainstate |
| T-21 | Medium | trainer | Multiple optimizers documented but only `optimizers[0]` used |
| T-22 | Medium | distributed | `strategy=` strategies never drive `fit` (no real multi-device) |
| T-23 | Medium | trainer | `val_check_interval` float is ignored |
| T-24 | Low | trainer | `_cleanup` nulls model after `fit` → post-fit `validate/test` break |
| T-25 | Low | trainer | `None` dataloaders → opaque `TypeError` |
| T-26 | Low | module | `print_summary(input_shape=...)` accepted but unused |
| T-27 | Low | module | `_to_scalar` raises on multi-element arrays |
| T-28 | Low | callbacks/ckpt | Filename formatting only catches `KeyError`, not `TypeError`/`ValueError` |
| T-29 | Low | checkpoint | `CheckpointManager` optimizer state format incompatible with trainer |
| T-30 | Low | progress | `RichProgressBarWrapper.set_postfix` clobbers description; off-by-one bar |
| T-31 | **Docs** | docs | No Sphinx API page for the trainer module |
| T-32 | Docs | docs | Docstrings encode broken metric convention & advertise non-working features |
| T-33 | Tests | tests | Shallow integration tests; `_distributed.py` untested |

---

## Detailed findings

### T-1 — `EarlyStopping.min_delta` sign inversion (High)
**Location:** `_callbacks.py:663-671`
```python
if self.mode == 'min':
    self.min_delta *= -1          # <-- bug
...
def _is_improvement(self, current, best):
    if self.mode == 'min':
        return current < best - self.min_delta   # = best + |min_delta|
    return current > best + self.min_delta
```
With `mode='min'` and `min_delta=0.1`, the flip makes the test `current < best + 0.1`.
A value `0.05` *worse* than the best is therefore counted as an "improvement", so the
patience counter never advances and early stopping does not trigger (and `best_score`
drifts upward). **Impact:** `EarlyStopping` is incorrect for the default `mode='min'`
whenever `min_delta > 0`.
**Fix:** Remove the `*= -1` line and make `_is_improvement` use `min_delta` with the
correct sign per mode: `min` → `current < best - min_delta`; `max` → `current > best + min_delta`.

### T-2 — Validation/test metric keys are double-prefixed (High)
**Location:** `_trainer.py:665-669` (val), `_trainer.py:882-885` (test); docstrings `_module.py:293-299`
```python
self.callback_metrics[f'val_{key}'] = ...   # key already == 'val_loss'
```
The documented convention (and the existing tests) is to `self.log('val_loss', ...)` /
`return {'val_loss': loss}` from `validation_step`. The trainer then prepends `val_`,
producing `callback_metrics['val_val_loss']`. Consequently `EarlyStopping(monitor='val_loss')`
and `ModelCheckpoint(monitor='val_loss')` find nothing — silently (returns `None`) or, with
`EarlyStopping(strict=True)` (the default), raising `RuntimeError`. **Impact:** the two most
important callbacks cannot monitor validation metrics using the documented key.
**Fix:** Stop double-prefixing. Treat the names the user logs as authoritative: only add a
`val_`/`test_` prefix to *bare* keys that don't already carry it, and record the metric under
its logged name as well. Align the docstrings/examples to one consistent convention.

### T-3 — `ModelCheckpoint` reads its metric before validation runs (High)
**Location:** `_callbacks.py:519-551`; loop order in `_trainer.py:466-470`
`Trainer._run_fit` runs `_run_train_epoch` (which fires `on_train_epoch_end` → ModelCheckpoint
saves) **before** `_run_validation_epoch`. So a `ModelCheckpoint(monitor='val_loss')` reads the
*previous* epoch's value (or `None` on epoch 0), and `best_model_path`/`best_k_models` tracking
is wrong. **Impact:** "save best by validation metric" does not work.
**Fix:** Perform metric-based checkpointing at validation-epoch end. Add
`on_validation_epoch_end` handling to `ModelCheckpoint`, dedupe so a given epoch is saved once,
and keep `on_train_epoch_end` for the no-validation / `save_on_train_epoch_end=True` case.

### T-4 — Double forward pass; logged loss inconsistent with gradients (High)
**Location:** `_trainer.py:399-410`
```python
loss = loss_fn()                                   # forward #1 (this is the reported loss)
grads = brainstate.transform.grad(loss_fn, ...)()  # forward #2 (gradients)
```
`loss_fn` runs `model.training_step` (and `self.log`) twice per step. Besides ~2× compute, with
stochastic layers (dropout, noise) the reported/ logged loss comes from a *different* random
draw than the gradients. **Fix:** Use `brainstate.transform.grad(..., return_value=True,
has_aux=True)` to compute loss, gradients, and the logged metrics in a single pass.

### T-5 — `min_epochs` not enforced (High)
**Location:** `_trainer.py:457-491`
```python
for epoch in range(self.max_epochs):
    if self._should_stop():       # checked unconditionally
        break
    ...
    if epoch < self.min_epochs - 1:
        continue                  # dead: already the last statement of the loop body
```
`_should_stop()` (EarlyStopping/Timer) is honored regardless of `min_epochs`, and the
`continue` does nothing. **Impact:** training can stop before `min_epochs`. **Fix:** Gate
`_should_stop()` so early termination cannot occur before `min_epochs` is reached; remove the
dead `continue`.

### T-6 — Resume resets `global_step` to 0 (High)
**Location:** `_trainer.py:732-733` reads `state.get('step', 0)`; `_callbacks.py:466-471`
saves the key as `'global_step'`. The names disagree, so resuming always restarts the step
counter at 0 (breaking step-based schedules/`max_steps`). **Fix:** Standardize on
`'global_step'` and read it (fall back to `'step'` for backward-compat).

### T-7 — Identical shuffle every epoch (Medium)
**Location:** `_dataloader.py:582-593`, `286-294`; trainer never calls `set_epoch`.
`DataLoader.__iter__` calls `self.batch_sampler.sampler.reset()` with no argument, which
re-seeds `RandomSampler` to its *original* seed; combined with the trainer never advancing the
epoch, a fixed `seed` yields the **same permutation every epoch**, and `DistributedSampler.epoch`
stays `0`. **Impact:** no real reshuffling → degraded training. **Fix:** Make `DataLoader`
advance a per-iteration epoch and derive each epoch's RNG from `base_seed + epoch`
(reproducible but distinct per epoch); have the trainer call `set_epoch(epoch)`.

### T-8 — `IterableDataset.__getitem__` re-creates the iterator (Medium)
**Location:** `_dataloader.py:190-198`
```python
item = next(iter(self.iterable))   # fresh iterator each call → always first element
```
For any re-iterable input (list, range, ...) random access returns element 0 forever.
**Fix:** Lazily create and cache a single iterator (`self._iterator = iter(self.iterable)`)
and pull successive items from it; raise `IndexError` on exhaustion.

### T-9 — Training metrics logged via `self.log` are never logged (Medium)
**Location:** `_trainer.py:544-576`. The trainer deliberately uses `outputs = {'loss': loss}`
and never forwards `model._get_logger_metrics()` for training, because values captured inside
the JIT-traced `training_step` are stale tracers from trace time. So `self.log('train_acc', …)`
in `training_step` is silently dropped from loggers/progress bar. **Fix:** Return the logged
metrics out of the jitted step (via `has_aux`) so concrete values are available, then forward
them to the loggers, the progress bar, and callbacks.

### T-10 — Leaked tracers in progress callbacks during training (Medium)
**Location:** `_callbacks.py:1119` (`TQDMProgressBar`), `1189-1191` (`PrintCallback`).
Both call `module._get_prog_bar_metrics()` inside `on_train_batch_end`. Those dict values were
stored during JIT tracing (only the first batch traces) and are stale/leaked tracers, so
formatting (`f"{v:.4f}"`) errors or yields garbage. **Fix:** After each step, repopulate the
module's prog-bar/logger metrics with the *concrete* values returned from the jitted step (see
T-9), so all downstream consumers see real numbers.

### T-11 — `LearningRateMonitor` ineffective; LR mis-extracted (Medium)
**Location:** `_trainer.py:534-539` vs `_callbacks.py:787-803`, `771-785`.
The trainer fires `on_train_batch_start` and *then* calls `model._reset_logged_metrics()`,
wiping the LR the monitor just logged. Also `_get_learning_rates` assumes `opt.lr` is a number
or `.value`/callable; with braintools optimizers `opt.lr` is an `LRScheduler` object, so the
extraction is unreliable. **Fix:** Reset logged metrics *before* `on_train_batch_start`; extract
the current LR robustly from braintools optimizers/schedulers.

### T-12 — `CallbackList` dispatches only a subset of hooks (Medium)
**Location:** `_callbacks.py:320-356`. Only `on_fit_*`, `on_train_*`, `on_validation_*` are
forwarded. There is no dispatch for `on_test_*`, `on_predict_*`, `on_before/after_optimizer_step`,
`on_before/after_backward`, `on_save/load_checkpoint`. Consequently `GradientClipCallback` (hooks
`on_before_optimizer_step`) and any user test/predict callbacks never fire, and `Trainer.test()/
predict()` only call the *module* hooks, not callbacks. **Fix:** Add the missing dispatch methods
and invoke them from the trainer's train/val/test/predict/optimizer paths.

### T-13 — Documented lifecycle hooks never invoked (Medium)
**Location:** `_trainer.py` (`_run_fit`, `_run_train_epoch`, `validate/test/predict`).
`LightningModule.on_train_start/on_train_end`, `on_validation_start/on_validation_end`, and
`on_before_backward/on_after_backward/on_before_optimizer_step/on_after_optimizer_step` are
defined and documented but never called. **Fix:** Call the model + callback hooks at the right
points (train start/end, validation start/end, and around the optimizer step).

### T-14 — `CSVLogger` data loss + `step=0` mishandling (Medium)
**Location:** `_loggers.py:583`, `613-629`.
`row = {'step': step or self._step_count}` replaces a legitimate `step=0` with the internal
counter. More seriously, `save()` writes the CSV header only once (when the file doesn't exist)
but uses `extrasaction='ignore'`, so any metric key that first appears *after* the initial flush
(e.g. `val_*` logged later than `train_*`) is **silently dropped** from the file. **Fix:** Use
`step if step is not None else …`; when the set of field names grows beyond the existing header,
rewrite the file with the complete header.

### T-15 — `FullyShardedDataParallelStrategy` construction crashes (Medium)
**Location:** `_distributed.py:507-528`, `398-404`.
`devices = jax.devices()` is a Python `list`; `devices.reshape(mesh_shape)` raises
`AttributeError` (lists have no `.reshape`) whenever `model_axis` is given, and
`Mesh(devices, …)` is passed a list rather than an ndarray. **Fix:** Build the mesh from
`numpy.asarray(jax.devices())` and reshape the array.

### T-16 — `freeze()`/`unfreeze()` are silent no-ops (Medium)
**Location:** `_module.py:678-690`. They guard on `hasattr(state, 'requires_grad')`, but
`brainstate.ParamState` has no such attribute (verified), so the loops never do anything and
the trainer's `grad_states` are unaffected. **Impact:** users believe parameters are frozen
when they are not. **Fix (chosen):** record the frozen/trainable intent on the state and have
the trainer exclude frozen params from `grad_states` *and* from optimizer registration; if the
backend cannot support this safely, emit an explicit warning instead of pretending. (See scope
note in the remediation plan.)

### T-17 — `accumulate_grad_batches` ignored (Medium)
**Location:** `_trainer.py:175`, never read in the loop. Documented gradient accumulation does
nothing. **Fix:** Implement micro-batch gradient accumulation (accumulate grads across N steps,
apply the averaged update every N) on top of the single-pass grad refactor (T-4).

### T-18 — `precision` ignored (Medium)
**Location:** `_trainer.py:176`. `'16'`/`'bf16'` are accepted but never applied (no dtype
casts, no loss scaling). **Fix (chosen):** Validate the value; warn clearly that only `'32'`
is currently effective and document the limitation (full mixed precision is out of scope).

### T-19 — `deterministic` / `benchmark` ignored (Low)
**Location:** `_trainer.py:177-178`. Stored, never used. **Fix:** Document them as advisory and,
for `deterministic`, ensure seeding is applied (see T-20); `benchmark` is a no-op in JAX —
document it.

### T-20 — `seed` seeds only numpy (Medium)
**Location:** `_trainer.py:219-221`. Only `np.random.seed` is called; JAX/`brainstate` RNG is
untouched, so dropout/sampling are not reproducible. **Fix:** Also call
`brainstate.random.seed(seed)` (verified to exist).

### T-21 — Multiple optimizers documented but unused (Medium)
**Location:** `_trainer.py:390` (`optimizer = self.optimizers[0]`); `_module.py:389-394` shows a
GAN example returning `[opt_g, opt_d], []`. Only the first optimizer is ever stepped. **Fix
(chosen):** Warn when `configure_optimizers` yields >1 optimizer that automatic multi-optimizer
training is not supported by `fit` (use manual optimization); document it.

### T-22 — `strategy=` never drives training (Medium)
**Location:** `_trainer.py:324-329`; `_distributed.py`. `fit` builds its own single-device step
and only calls `strategy.setup(...)` (a no-op for most strategies); the strategies'
`training_step` is never used. So `Trainer(strategy='ddp'|'fsdp'|…)` does not parallelize.
**Fix (chosen):** Fix the construction crashes (T-15), keep the strategy utilities correct, and
**document** that `fit` currently performs single-device training and that multi-device
orchestration via `Strategy.training_step` is not wired into `fit`. (Full integration is a large,
CI-untestable change and is out of scope for this audit.)

### T-23 — `val_check_interval` float ignored (Medium)
**Location:** `_trainer.py:695-701`. `_should_validate_batch` returns `False` for any float,
contradicting the docstring ("float = fraction of epoch"). **Fix:** Implement fractional
in-epoch validation: validate every `max(1, int(fraction * len(train_dataloader)))` batches.

### T-24 — `_cleanup` nulls model after `fit` (Low)
**Location:** `_trainer.py:735-740`. After `fit`, `self.model/optimizers` become `None`, so
`trainer.validate()/test()/predict()` raise "No model provided" unless the model is re-passed —
unlike PyTorch Lightning. **Fix:** Keep references to the (trained) model/optimizers after `fit`.

### T-25 — `None` dataloaders give opaque errors (Low)
**Location:** `_trainer.py:531` (`enumerate(None)`), `845`, `939`. Calling `fit`/`test`/`predict`
without a dataloader raises a bare `TypeError`. **Fix:** Validate and raise clear messages.

### T-26 — `print_summary(input_shape=...)` unused (Low)
**Location:** `_module.py:692-700`. The parameter is accepted and documented for "shape
inference" but never used. **Fix:** Either implement a dummy forward pass when provided, or
document it as reserved/unused. (Chosen: document as reserved to avoid changing the signature.)

### T-27 — `_to_scalar` raises on multi-element arrays (Low)
**Location:** `_module.py:37-52`. `value.item()` and `float(value)` both raise for arrays with
>1 element, so an accidental `self.log('x', vector)` crashes during metric retrieval rather than
degrading gracefully. **Fix:** Fall back to returning the value unchanged (or its mean) instead
of raising.

### T-28 — Checkpoint filename formatting only catches `KeyError` (Low)
**Location:** `_callbacks.py:454-460`, `_checkpoint.py:341-347`. A metric value that is an array
or non-numeric with a numeric format spec (`{val_loss:.4f}`) raises `TypeError`/`ValueError`,
which is not caught, crashing the save. **Fix:** Broaden the `except` and coerce metric values to
floats for formatting.

### T-29 — `CheckpointManager` optimizer-state format incompatible with trainer (Low)
**Location:** `_checkpoint.py:417-418` saves a single dict; `_trainer.py:727-730` iterates it as a
list. Cross-loading a `CheckpointManager` checkpoint via the trainer misbehaves. **Fix:** Make
`Trainer._load_checkpoint` accept both a single dict and a list.

### T-30 — `RichProgressBarWrapper` cosmetics (Low)
**Location:** `_progress.py:358-363`, `188-189`. `set_postfix` overwrites the task *description*
(losing "Epoch N"); the 100%-complete bar is one character too long. **Fix:** Append metrics to a
preserved description; clamp the bar fill.

### T-31 — No API documentation for the trainer module (Docs)
There is no `docs/apis/trainer.rst` and the module is absent from the `docs/index.rst` toctree —
every other public module has a page. The entire public trainer API is undocumented in the
rendered docs. **Fix:** Add `docs/apis/trainer.rst` (autosummary of the public API) and wire it
into `docs/index.rst`.

### T-32 — Docstrings encode broken behavior (Docs)
Examples teach the broken metric convention (T-2) and advertise non-working features
(`accumulate_grad_batches`, `precision`, multi-optimizer, distributed). **Fix:** Update
docstrings to the corrected convention and annotate not-yet-supported parameters.

### T-33 — Shallow integration tests; `_distributed` untested (Tests)
The suite lacks end-to-end `fit` tests with validation + monitoring callbacks (which would catch
T-1…T-14), and `_distributed.py` has no test file. **Fix:** Add real end-to-end training tests
(loss decreases, params update, callbacks fire, checkpoints/early-stopping work) and a
`_distributed_test.py` covering all single-device-testable paths.

---

## Remediation plan & scope decisions

**Fix fully (correctness / behavior):** T-1, T-2, T-3, T-4, T-5, T-6, T-7, T-8, T-9, T-10,
T-11, T-12, T-13, T-14, T-15, T-20, T-23, T-24, T-25, T-27, T-28, T-29, T-30.

**Implement:** T-17 (gradient accumulation, built on the T-4 single-pass refactor).

**Implement if backend-safe, else warn + document:** T-16 (parameter freezing).

**Make honest (validate + warn + document) rather than silently ignore:** T-18 (precision),
T-19 (deterministic/benchmark), T-21 (multiple optimizers), T-22 (distributed strategies).

**Docs:** T-26, T-31, T-32.

**Tests:** T-33 — add end-to-end integration tests and `_distributed_test.py`; raise
`_loggers.py` ≥ 90% (mock external backends); keep the whole module > 90% (excluding
`_distributed.py`, which the project omits from coverage).

All fixes follow TDD: a failing test reproduces each behavioral bug first, then the fix makes it
pass, with the full suite kept green.
