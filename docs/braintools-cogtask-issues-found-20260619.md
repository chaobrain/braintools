# braintools/cogtask — Issues Found (2026-06-19)

Audit of the `braintools/cogtask/` module and its in-code documentation, from
the perspective of a senior Python developer / JAX expert. Each issue below was
reproduced before being listed. Baseline before fixes: `276 passed`.

Severity legend: **[High]** silent wrong results / crashes on supported usage ·
**[Med]** misleading errors or unreachable public API · **[Low]** documentation
examples that do not run.

---

## Issue 1 — [High] `Parallel.execute` silently drops compound children

**File:** `braintools/cogtask/phase.py`, `Parallel.execute`.

The eager (fixed-length) execution path calls `child.encode_inputs(ctx)` /
`child.encode_outputs(ctx)` **directly** on each parallel child:

```python
for i, child in enumerate(self.phases):
    ...
    child.encode_inputs(ctx)
    if i == 0:
        child.encode_outputs(ctx)
```

For *compound* children (`Sequence`, `Repeat`, `If`, `Switch`, `While`, nested
`Parallel`) `encode_inputs`/`encode_outputs` are no-ops — those classes drive
their sub-phases through `execute()`, not through the encode hooks. As a result
a construct like `(A >> B) | C` writes **nothing** for the `A >> B` branch: the
sub-sequence is silently skipped and its input channels stay at zero.

The packed/variable-length path (`Parallel.execute_packed`) does *not* have this
bug — it dispatches every child through `execute_phase_packed`, which handles
both leaves and compounds. So the two execution paths disagree.

**Reproduction**

```python
seqAB = a >> b                  # a, b write input channel 'm'
par   = seqAB | c               # c writes channel 'n'
X, Y, info = task.sample_trial(0)
# eager:  sum(X[:, m]) == 0.0   ← seqAB silently dropped (BUG)
# packed: sum(X[:, m]) == 40.0  ← seqAB executed correctly
```

**Proposed solution.** Dispatch each parallel child through `execute_phase`
(mirroring `execute_packed`), restoring the parent scope afterward. All children
start at `parent_start`; only the first contributes to the output buffer (stash
& restore `ctx.outputs` for `i > 0`). This fixes compound children, preserves
the existing leaf behaviour (the built-in `modality1 | modality2` /
`Fixation | Cue` tasks are unchanged), and makes both paths consistent.

---

## Issue 2 — [Med] `make_encoder(mode="scalar", feature_per_direction>1)` raises a misleading error

**File:** `braintools/cogtask/tasks/working_memory.py`, `make_encoder`.

`"scalar"` is only handled in the `feature_per_direction == 1` branch. When
`feature_per_direction > 1` the function falls through to the population branch,
which only knows `one_hot`/`von_mises`/`circular`, and raises
`ValueError("Unknown mode=scalar")`. But `scalar` broadcasts its value across
*all* `feature.num` dims and is completely agnostic to `feature_per_direction`,
so the call should simply work.

**Reproduction**

```python
make_encoder("scalar", "sample_idx", feature_per_direction=4)
# ValueError: Unknown mode=scalar   ← misleading; scalar is K-agnostic
```

**Proposed solution.** Handle `"scalar"` up front (before the `K == 1`
special-casing) and return `scalar(key)` regardless of `feature_per_direction`.

---

## Issue 3 — [Low] Package-level quickstart docstring uses a raw string label that cannot be encoded

**File:** `braintools/cogtask/__init__.py` (module docstring).

The "Building custom tasks from phases" example ends with:

```python
Response(100 * u.ms, outputs={'label': 'ground_truth'})
```

`DeclarativePhase.encode_outputs` does **not** treat a bare string as a context
key — it passes it to `jnp.asarray(..., dtype=jnp.int32)`, which raises
`ValueError: invalid literal for int() with base 10: 'ground_truth'`. The
`label()` helper is what interprets a string as a context key.

**Reproduction:** running the docstring example verbatim raises the `ValueError`
above.

**Proposed solution.** Use the `label` helper: `outputs={'label':
label('ground_truth')}`, and add `label` to the example's imports.

---

## Issue 4 — [Low] `Task` class docstring examples use APIs that do not exist

**File:** `braintools/cogtask/task.py` (`Task` docstring).

Two examples are broken against the current API:

* Instance-based: `Stimulus(2000 * u.ms, feature=stim, encoder=circular_encoder())`
  — `DeclarativePhase`/`Stimulus` take `inputs=`/`outputs=`, not `feature=`/
  `encoder=`; `circular_encoder` does not exist.
* Class-based: `Feature(1, 40*u.Hz, 'fixation')` — `Feature.__init__(num,
  name=None)` takes no firing-rate argument; this raises
  `TypeError: ... takes from 2 to 3 positional arguments but 4 were given`.

**Proposed solution.** Rewrite both examples against the real API
(`Feature(num, name)`, `Stimulus(duration, inputs=..., outputs=...)`).

---

## Issue 5 — [Low] `create_task` docstring example is broken (wrong arg order + invalid kwarg)

**File:** `braintools/cogtask/task.py` (`create_task` docstring).

```python
input_features=Feature('fix', 1) + Feature('stim', 2),   # name/num swapped
...
num_trial=1000                                            # not a Task kwarg
```

`Feature('fix', 1)` silently produces a nonsensical feature with `num='fix'`
(a string) and `name=1`; `num_trial=1000` is forwarded via `**kwargs` and set as
a stray attribute. The example never runs as intended.

**Proposed solution.** Fix to `Feature(1, 'fix')` etc. and drop the invalid
`num_trial` kwarg.

---

## Issue 6 — [Low] `Feature.__mul__` docstring uses the obsolete 3-argument signature

**File:** `braintools/cogtask/feature.py` (`Feature.__mul__` docstring).

`Example: Feature(1, 30.*u.Hz, 'choice') * 3` — same obsolete firing-rate
signature as Issue 4. Should be `Feature(1, 'choice') * 3`.

---

## Issue 7 — [Low] `Context` docstring constructor & buffer examples do not run

**File:** `braintools/cogtask/context.py` (`Context` docstring).

* `Context(seed=42)` — `Context.__init__` accepts `key=`, not `seed=`.
* `ctx.inputs[ctx.phase_start:ctx.phase_end, :] = stimulus_encoding` — `inputs`
  is a JAX array; item assignment is not supported (`.at[...].set(...)` is).
* Uses `np.pi` without importing `numpy`.

**Proposed solution.** Use `key=` (or no argument), reference the functional
`.at[...].set(...)` update, and use `jax.numpy` consistently.

---

## Issue 8 — [Med] `create_task` is declared public but not exported from the package

**Files:** `braintools/cogtask/task.py` (`__all__` lists `create_task`) vs.
`braintools/cogtask/__init__.py` (neither imports nor re-exports it).

`create_task` has a full public docstring and is in `task.py`'s `__all__`, yet
`braintools.cogtask.create_task` raises `AttributeError`. Either it is public
(then export it) or private (then drop it from `__all__`). Since it is a
documented convenience factory, the right fix is to export it.

**Proposed solution.** Import `create_task` in `__init__.py` and add it to the
package `__all__`.

---

## Observations (documented behaviour / lower risk — not changed)

These are noted for completeness; they are either already documented or carry
behaviour-change risk that outweighs the benefit of a code change:

* **`Task.num_classes` over-reports for DMS-style tasks when `cue_dim > 1`.**
  The categorical label space of DMS tasks is fixed (e.g. `{0, 1, 2}`), but the
  default `num_classes` falls back to `num_outputs = cue_dim + 2`. This is
  explicitly documented on the `Task.num_classes` property ("drifts apart when
  e.g. `cue_dim` changes; pass `num_classes=` explicitly"). Left as documented
  guidance.
* **`ContextDecisionMaking` with `num_contexts > 2`.** Only two stimulus
  modalities exist; the ground-truth selection treats context `0` as "attend
  modality 1" and every other context as "attend modality 2". A reasonable
  interpretation, but worth knowing.
* **`utils.choice(rng, n_total=1, ...)`** samples from an empty set; only
  relevant for the degenerate `num_stimuli == 1` configuration of match tasks.

---

## Test plan

Regression tests are added under the existing `cogtask` test suite:

1. **Parallel compound child (eager)** — `(A >> B) | C` writes the `A`/`B`
   channels; assert the eager and packed paths agree.
2. **`make_encoder` scalar + `feature_per_direction>1`** — returns a working
   encoder and broadcasts the scalar across all dims.
3. **`create_task` export** — importable from `braintools.cogtask` and produces
   a sampleable task.
4. **Fixed docstring behaviours** — `outputs={'label': label('ground_truth')}`
   samples without error.

All pre-existing tests must continue to pass.
