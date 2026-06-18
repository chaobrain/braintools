# `braintools.trainer` — Issues Found and Proposed Solutions (2026-06-19)

A senior-developer / JAX-expert audit of the `braintools/trainer/` module and its
documentation (`docs/apis/trainer.rst`). Each issue below was reviewed in source;
the four high/medium issues were additionally **reproduced** with runnable probes
against the current code.

Environment: `jax==0.10.2`, `brainstate==0.5.1`. Baseline before fixes: **489
trainer tests pass**.

Files audited: `_module.py`, `_trainer.py`, `_callbacks.py`, `_loggers.py`,
`_dataloader.py`, `_distributed.py`, `_checkpoint.py`, `_progress.py`.

---

## Summary table

| ID  | Severity | Area | One-line description | Status |
|-----|----------|------|----------------------|--------|
| T-A | High | trainer/callbacks | Metrics logged via `module.log()` inside callbacks/hooks never reach loggers or the progress bar (breaks `LearningRateMonitor`). | Fixed |
| T-B | High | callbacks | `ModelCheckpoint(save_top_k=k)` writes one checkpoint **per epoch** to disk; evicted files are never deleted. | Fixed |
| T-C | Medium | trainer/checkpoint | Resuming from a `CheckpointManager` checkpoint saved with `step=None` sets `global_step=None` and crashes the training loop. | Fixed |
| T-D | Medium | distributed | `broadcast()` / `Strategy.broadcast()` build an invalid `lax.ppermute` and raise on >1 device. | Fixed |
| T-E | Low | callbacks/docs | `GradientClipCallback` is a no-op (even with `log_grad_norm=True`) but its docstring implies it clips/logs. | Fixed (docs) |
| T-F | Low | trainer/docs | Under `accumulate_grad_batches>1`, `max_steps` counts micro-batches, not optimizer steps. | Fixed (docs) |
| T-G | Low | distributed | FSDP auto-mesh with `model_axis` always produces a model axis of size 1. | Fixed |
| T-H | Low | distributed | `sync_batch_norm` averages per-device variances, underestimating the true global variance. | Fixed |
| T-I | Low | dataloader/docs | `create_distributed_batches` silently drops the trailing incomplete batch. | Fixed (docs) |
| T-J | Low | callbacks/docs | `LearningRateMonitor(log_momentum=...)` is accepted but never used. | Fixed (docs) |

---

## T-A (High) — `module.log()` from callbacks/hooks never reaches loggers/progress bar

**Where:** `_trainer.py` `Trainer._create_train_step` / `_run_train_epoch`;
manifests through `_callbacks.py` `LearningRateMonitor`.

**Problem.** The JIT-traced `loss_fn` calls `model._reset_logged_metrics()` and
then runs `training_step`, capturing **only** `training_step`'s logged metrics
into the gradient `aux`:

```python
def loss_fn(batch, batch_idx):
    model._reset_logged_metrics()          # wipes anything logged earlier this batch
    outputs = model.training_step(batch, batch_idx)
    ...
    aux = {'logger': dict(model._logger_metrics), 'prog_bar': dict(model._prog_bar_metrics)}
    return loss, aux
```

The training loop logs/displays from `aux` only. Consequently, anything a
callback or hook logs via `module.log()` during `on_train_batch_start` (e.g.
`LearningRateMonitor` in `'step'` mode) — or during `on_train_epoch_start` for
`'epoch'` mode — is (a) erased by the reset inside `loss_fn`, and (b) never
present in `aux`. The metric silently disappears from loggers and the progress
bar. (`LearningRateMonitor.lr_history` is still populated, which masks the bug.)

**Reproduction.** A spy `Logger` over a 1-epoch fit with a `LearningRateMonitor`
captured only `['loss', 'train_loss']`; `lr-opt0` never appeared, even though
`lr_history == [{'lr-opt0': 0.001}, ...]`.

**Fix.** In `_run_train_epoch`, snapshot the metrics logged outside JIT — at
epoch start and right after the per-batch start hooks — and merge them into the
per-batch `outputs` and progress-bar postfix, with `training_step` metrics taking
precedence on name collisions. These callback/hook values are concrete (logged
outside the traced step), so they are safe to read directly.

---

## T-B (High) — `ModelCheckpoint(save_top_k=k)` leaves orphan checkpoint files

**Where:** `_callbacks.py` `ModelCheckpoint._checkpoint_epoch` / `_update_best_k`.

**Problem.** `_checkpoint_epoch` calls `_update_best_k(score, filepath)` (which
may immediately evict the just-added `filepath` if it is the worst) **before**
`_save_checkpoint(...)`, and `_save_checkpoint` writes unconditionally:

```python
self._update_best_k(current_score, filepath)   # may del filepath from tracking,
                                                # _remove_checkpoint() is a no-op (file not written yet)
self._save_checkpoint(trainer, module, filepath)  # ...but this writes it anyway
```

So a checkpoint that was "evicted" from the top-k is still written to disk and
never tracked or cleaned up. `save_top_k` is not enforced on disk.

**Reproduction.** With `save_top_k=2` over 6 epochs (monotonically worsening
`val_loss`): **6** `.ckpt` files on disk, but only **2** tracked in
`best_k_models`.

**Fix.** Save first, then prune: call `_save_checkpoint(...)` and then
`_update_best_k(...)`. Eviction's `_remove_checkpoint` now finds and deletes the
file, so disk stays at `save_top_k`. The best checkpoint is never the eviction
target (it holds the best score), so it is preserved.

---

## T-C (Medium) — resuming from a `step=None` checkpoint crashes

**Where:** `_trainer.py` `Trainer._load_checkpoint`; `_checkpoint.py`
`CheckpointManager.save`.

**Problem.** `CheckpointManager.save(..., step=None)` stores `'step': None`.
On resume:

```python
self.state.global_step = state.get('global_step', state.get('step', 0))  # -> None
...
state.global_step += 1   # TypeError: unsupported operand type(s) for +=: 'NoneType' and 'int'
```

**Reproduction.** Saving with `step=None` then `trainer.fit(..., ckpt_path=...)`
raised `TypeError: unsupported operand type(s) for +=: 'NoneType' and 'int'`.

**Fix.** Coerce a missing/`None` `global_step` (and `epoch`) to `0` on load.

---

## T-D (Medium) — `broadcast()` / `Strategy.broadcast()` use an invalid `ppermute`

**Where:** `_distributed.py` module-level `broadcast`, `DataParallelStrategy.broadcast`
(and `AutoStrategy.broadcast` which delegates).

**Problem.** The permutation repeats the source index as a source for every
destination:

```python
perm = [(src, i) for i in range(num_devices)]
return lax.ppermute(tensor, axis_name=axis_name, perm=perm)
```

`lax.ppermute` requires unique sources and destinations.

**Reproduction.** On 4 fake CPU devices
(`XLA_FLAGS=--xla_force_host_platform_device_count=4`):
`ValueError: ppermute sources and destinations must be unique, got ((0,0),(0,1),(0,2),(0,3))`.
(`all_reduce` works correctly.)

**Fix.** Implement broadcast with a valid collective:
`lax.all_gather(x, axis_name)[src]` per leaf — every replica gathers all values
and selects the source's, yielding a correct broadcast for any device count.

---

## T-E (Low) — `GradientClipCallback` is a no-op; docstring is misleading

**Where:** `_callbacks.py` `GradientClipCallback`.

`on_before_optimizer_step` is a `pass`; `log_grad_norm` is never read. Gradient
clipping is performed by the `Trainer` (`gradient_clip_val`/`gradient_clip_algorithm`)
inside the JIT-compiled apply step, where a Python callback cannot intercept the
gradients. The class therefore does nothing.

**Fix.** Clarify the docstring: the callback does not currently clip or log;
use `Trainer(gradient_clip_val=...)`. (Behavior unchanged — the Trainer owns
clipping; intercepting traced grads from a callback is out of scope.)

---

## T-F (Low) — `max_steps` semantics under gradient accumulation

**Where:** `_trainer.py` `_run_train_epoch`.

`state.global_step` increments once per **micro-batch**, while the optimizer
steps once per `accumulate_grad_batches` micro-batches. So with
`accumulate_grad_batches>1`, `max_steps` bounds processed batches, not optimizer
updates.

**Fix.** Document this in the `Trainer` docstring (`max_steps` / `accumulate_grad_batches`).

---

## T-G (Low) — FSDP auto-mesh collapses the model axis to size 1

**Where:** `_distributed.py` `FullyShardedDataParallelStrategy.__init__`.

When `model_axis` is given and no `mesh` is supplied, the balancing loop
`for dp in [n_devices, n_devices//2, n_devices//4]` succeeds immediately at
`dp=n_devices, mp=1`, so the model-parallel axis always has size 1 — defeating
the purpose of `model_axis`.

**Fix.** Search for the most balanced 2-D factorization (largest `dp ≤ sqrt(n)`
dividing `n`, with `mp = n/dp`), so the model axis gets a real size when the
device count allows.

---

## T-H (Low) — `sync_batch_norm` underestimates global variance

**Where:** `_distributed.py` `sync_batch_norm` (note: not exported in `__all__`).

It computes per-device variance and then `pmean`s the variances. The mean of
per-device variances ignores the between-device differences in means, so it is
**not** the true pooled variance.

**Fix.** Compute global variance as `pmean(E[x²]) - pmean(E[x])²`.

---

## T-I (Low) — `create_distributed_batches` silently drops the remainder

**Where:** `_dataloader.py` `create_distributed_batches`.

The loop `range(0, n_samples - total_batch_size + 1, total_batch_size)` drops any
trailing samples that do not fill a full `num_devices × batch_size` block, with
no warning. This is reasonable for `pmap` (which needs a fixed leading device
axis), but it is undocumented.

**Fix.** Document the drop-remainder behavior in the function docstring.

---

## T-J (Low) — `LearningRateMonitor(log_momentum=...)` is accepted but unused

**Where:** `_callbacks.py` `LearningRateMonitor`.

`log_momentum` is stored but never read.

**Fix.** Document that momentum logging is not yet implemented.

---

## Other observations (reviewed, no change)

- **JIT metric capture for `training_step`:** metrics logged inside
  `training_step` are correctly returned as concrete `aux` and are re-evaluated
  per batch; the design is sound. Custom callbacks that read `module.logged_metrics`
  directly during `on_train_batch_end` may still observe stale traced values —
  callbacks should use the `outputs` argument (concrete) instead. (Documented
  limitation; not changed.)
- **`DataLoader` epoch reshuffling** (`__iter__` advances `_epoch` before yielding)
  is correct and reproducible for a given `(seed, epoch)`.
- **`CSVLogger`** header-widening / rewrite on newly-appearing metric columns is
  correct.
- **`SimpleProgressBar`** overshoot clamping (T-30) is correct.
