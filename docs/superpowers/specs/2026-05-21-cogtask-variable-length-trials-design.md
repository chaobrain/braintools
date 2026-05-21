# Variable-Length Trial Sequences for `braintools.cogtask`

- **Status**: Proposed
- **Date**: 2026-05-21
- **Branch**: `worktree-tingly-swinging-prism` (rebased onto `cogtask`)
- **Owner**: Chaoming Wang

## 1. Goal

Enable `braintools.cogtask` tasks to generate trials whose total timestep
count varies across samples, while remaining usable inside
`brainstate.transform.jit` and `brainstate.transform.vmap2`. Variable-length
batches must use shape-stable buffers (padding plus a per-timestep mask)
rather than Python-side dynamic shapes inside the JIT-compiled region.

## 2. Background — current behaviour

A snapshot of the audit performed against `braintools/cogtask/` on the
`cogtask` branch:

- **Buffer sizing is Python-side.**
  `Task.sample_trial(index)` calls `_compute_duration(ctx)` which runs the
  phase tree's `get_duration(ctx)` and immediately wraps the result in
  `int(...)`. The returned `total_duration` is a Python `int` and is used to
  allocate static-shape `(T, num_inputs)` and `(T, …)`/`(T,)` buffers.
- **Phase writes use static slices.**
  Leaf phases mutate `ctx.inputs` and `ctx.outputs` via
  `.at[ctx.phase_start : ctx.phase_end, …].set(value)` where
  `phase_start`/`phase_end` are Python ints computed by walking phases in
  order.
- **Batch sampling vmaps a uniform-shape function.**
  `Task.batch_sample(size, …)` is `brainstate.transform.jit`-decorated and
  vmaps `sample_trial` with `brainstate.transform.vmap2`. Per-trial buffers
  must all share the same `T`, so trials with truly variable lengths are
  unsafe under this path.
- **Existing variable-duration tasks bypass JIT.**
  `HierarchicalReasoning`, `IntervalDiscrimination`, and `ReadySetGo` sample
  per-trial durations in `trial_init` (stored as `ctx['delay_duration']`,
  `ctx['interval1_duration']`, `ctx['measure_interval']`, …) and then
  override `get_duration` with a closure that casts those traced JAX scalars
  to Python `int`. This works in eager mode because `int(jnp.asarray(x))`
  returns a concrete value; it raises under tracing. `conftest.py` already
  carves these three tasks out of the parametric vmap test list with the
  comment *"unsafe to vmap because each trial would allocate a different
  total timestep count"*.
- **No mask infrastructure exists today.** Consumers only see rectangular
  `(T, …)` outputs; there is no machinery for ignoring padding in losses.

## 3. Requirements (from the user request)

1. Audit the current trial-sequence generation and batching logic. *(done in
   §2)*
2. Refactor sequence handling so tasks can represent trials with variable
   lengths without relying on Python-side dynamic shapes inside JIT-compiled
   paths.
3. The implementation must work under both `brainstate.transform.jit` and
   `brainstate.transform.vmap2`, using shape-stable representations.
4. Update relevant APIs, tests, and documentation/examples.
5. Add regression tests covering fixed-length behavior, mixed variable-length
   batches, JIT execution, and vmap execution.
6. Preserve existing behavior and public API compatibility wherever
   possible; clearly document any intentional API changes.

## 4. Design decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Padding layout | **Packed with trailing pad** | Densely packs valid timesteps at the front of the buffer; only the tail is masked. Cleaner consumer semantics than per-phase holes. |
| 2 | Activation | **Auto-detect from the phase tree** | Tasks composed entirely of fixed-duration phases keep their existing code path bit-for-bit. Tasks containing any variable-duration phase silently switch into the padded path. No flag for the common case. |
| 3 | Compound coverage | **Full** (`Sequence`, `Repeat`, `Parallel`, `If`, `Switch`, `While`) | A scoped-down version (leaves only) would leave `While`, `If`, and `Switch` unsafe under vmap. Going full removes all "but not under vmap" caveats. |
| 4 | Return shape | **Mask in `info` dict + opt-in `return_mask=True` for `batch_sample`** | `sample_trial` keeps its 3-tuple signature; `batch_sample` keeps its default `(X, Y)` shape. Backward compatible for every existing call site. |

## 5. Architecture

### 5.1 New `Phase` surface

```python
class Phase(ABC):
    is_variable: ClassVar[bool] = False  # subclasses override

    def get_duration(self, ctx: Context) -> int: ...        # unchanged
    def max_steps(self, ctx: Context) -> int: ...           # NEW (static)
    def step_count(self, ctx: Context) -> jax.Array: ...    # NEW (traced)
```

- **`max_steps(ctx)`** returns a **Python `int`** upper bound on the phase's
  length in timesteps. It must be derivable without sampling — typically from
  `max_value()` of a duration sampler (`TruncExp`, `UniformDuration`), the
  branch `max(max_steps_of_branch)` for `If`/`Switch`, or
  `body.max_steps * max_iterations` for `While`. Used exclusively for static
  buffer sizing. Default implementation: `self.get_duration(ctx)`.
- **`step_count(ctx)`** returns a **`jax.Array` `int32` scalar** with the
  actual phase length in timesteps. Defaults to
  `jnp.asarray(self.get_duration(ctx), dtype=jnp.int32)` for fixed phases.
  Variable-duration phases override this to compute the duration from
  `ctx[...]` without any `int(...)` cast.
- **`is_variable`** is a `ClassVar[bool]` propagated by composition.
  `Sequence.is_variable` is `any(child.is_variable for child in children)`,
  and so on for the other compounds.

A free function `phase_tree_is_variable(phase) -> bool` walks the tree once
(used by `Task.__init__`) to decide which execution path to compile.

### 5.2 Buffer model

In variable-length mode `Task.sample_trial` allocates:

```python
T_max = self.phases.max_steps(ctx)            # Python int
ctx.inputs   = jnp.zeros((T_max, self.num_inputs),  dtype=jnp.float32)
ctx.outputs  = (jnp.zeros((T_max,),                  dtype=jnp.int32)
                if categorical else
                jnp.zeros((T_max, self.num_outputs), dtype=jnp.float32))
ctx.mask     = jnp.zeros((T_max,),               dtype=jnp.bool_)
ctx.t_cursor = jnp.int32(0)
```

The fixed-length code path is **unchanged**: `T = int(self._compute_duration(ctx))`,
buffers allocated at `(T, …)`, no mask, no `t_cursor`.

### 5.3 Leaf phase execution under packed mode

For a leaf phase with `max_dur = phase.max_steps(ctx)` and traced
`actual = phase.step_count(ctx)`:

```python
# 1. Build a (max_dur, num_inputs) block in phase-local coordinates.
block = jnp.zeros((max_dur, num_inputs), dtype=ctx.inputs.dtype)
for feat_name, spec in self._input_specs.items():
    feat = ctx.input_features[feat_name]
    value = self._resolve_value(spec, ctx, feat)            # (num,) or (max_dur, num)
    value = jnp.broadcast_to(value, (max_dur, feat.num))
    block = block.at[:, feat.i].set(value)

# 2. Mask out positions t >= actual (so padding tail of this block is zeros).
t_local = jnp.arange(max_dur)
gate = (t_local < actual)[:, None]
block = jnp.where(gate, block, 0)

# 3. Pack into the trial buffer at the current cursor.
ctx.inputs = jax.lax.dynamic_update_slice(ctx.inputs, block, (ctx.t_cursor, 0))

# 4. Mask segment.
mask_segment = jnp.where(t_local < actual, True, False)
ctx.mask = jax.lax.dynamic_update_slice_in_dim(ctx.mask, mask_segment, ctx.t_cursor, axis=0)

# 5. Advance cursor.
ctx.t_cursor = ctx.t_cursor + actual
```

Output buffer is handled analogously — categorical mode uses a `(max_dur,)`
block and writes via `dynamic_update_slice_in_dim`; vector mode uses
`(max_dur, num_outputs)`.

The mask is the source of truth for "real" timesteps. Padded positions in
the input/output buffers are zero by construction. Noise injection happens
**before** the gate, so noise outside `[0, actual)` is masked off too —
unbiased and reproducible.

### 5.4 Compound phases under packed mode

- **`Sequence`** iterates children in Python order; each child runs through
  the leaf-phase recipe, advancing `ctx.t_cursor` as a traced sum.
- **`Repeat`** runs its body `count` times in Python (the body's content
  varies per iteration via `ctx['repeat_index']`); each iteration writes at
  the traced cursor. Body `max_steps` × `count` bounds total length.
- **`Parallel`** writes each child at the same `parent_cursor`, advancing
  the cursor by `max(child.step_count())`. Each child writes into the
  buffer via the leaf recipe; later children may overwrite earlier ones for
  features the earlier child also touched, which matches today's
  "last-writer-wins per feature index" semantics. The trial mask receives
  the OR of all child masks at this slot. The reserved slot is
  `max(child.max_steps)`.
- **`If`** uses `jax.lax.cond` so **only one branch executes**. Both
  branches are wrapped to produce the same pytree shape: a fixed-shape
  `(max_branch_dur, …)` block (where `max_branch_dur = max(then.max_steps,
  else_.max_steps)`), the matching mask segment, and a traced cursor delta.
  The unused branch is *not* run. After `lax.cond` returns, we
  `dynamic_update_slice` the chosen block into the buffer at the current
  cursor and advance by the chosen delta.
- **`Switch`** uses `jax.lax.switch` with the same "every branch produces a
  uniform-shape (block, mask, delta) tuple, only the selected branch runs"
  pattern. Cases are ordered deterministically; integer keys index directly,
  hashable keys are mapped to a stable integer order at construction time.
- **`While`** is bounded by `max_iterations`. We emit a
  `jax.lax.fori_loop(0, max_iterations, body)` whose `body` uses
  `lax.cond(condition(ctx), do_body, no_op)`. Each "real" iteration writes a
  body block (size `body.max_steps`) at the current cursor and advances it;
  each "fake" iteration is a no-op. Total buffer length:
  `body.max_steps * max_iterations`.

All compound `max_steps` mirrors the existing `get_duration` arithmetic but
operates on `max_steps` of children — pure Python `int` math.

### 5.5 Context changes

```python
class Context:
    ...
    # Existing
    inputs: jax.Array
    outputs: jax.Array

    # NEW (only in variable-length mode)
    mask: Optional[jax.Array] = None      # (T_max,) bool
    t_cursor: Optional[jax.Array] = None  # () int32 traced
```

`ctx.phase_start`/`ctx.phase_end` are retained for back-compat with
encoder/label callbacks that may reference them. In variable mode they are
*traced* (the value of `ctx.t_cursor` at phase entry / phase entry + actual).
Encoder callbacks that compare `ctx.phase_start` to a Python int continue to
work because traced arithmetic still produces a `jax.Array`. Encoder
callbacks that *return* a `(duration, …)`-shaped array must instead return
a `(phase.max_steps, …)` array; this is a contract change documented in
§7.3.

### 5.6 Variable-duration samplers

`TruncExp` and `UniformDuration` gain:

```python
def max_value(self) -> Quantity:        # NEW
    return self._max * self._time_unit

@property
def is_variable(self) -> bool:          # NEW class attribute
    return True
```

A new helper phase class `VariableDuration(min_q, max_q, ctx_key, inputs=..., outputs=...)`
replaces the ad-hoc `lambda` overrides currently used inside
`HierarchicalReasoning`, `IntervalDiscrimination`, and `ReadySetGo`. It:

- Declares `is_variable = True`.
- Implements `max_steps(ctx) = ceil(max_q / dt)`.
- Implements `step_count(ctx) = jnp.maximum(1, (ctx[ctx_key] / dt_val).astype(jnp.int32))`.
- Otherwise behaves like a `DeclarativePhase` for inputs/outputs/noise.

The three existing tasks are migrated to `VariableDuration`. Their
`_compute_variable_delay`, `_compute_interval1_duration`,
`_compute_interval2_duration`, and `_compute_measure_interval` helpers are
deleted.

### 5.7 `Task` changes

```python
class Task:
    def __init__(self, ...):
        ...
        self._is_variable_length = phase_tree_is_variable(self.phases)
        # T_max is resolved lazily on first call to sample_trial. It depends
        # only on dt (read from brainstate.environ inside max_steps) and on
        # Python-level static phase metadata, so a stub Context with no
        # trial-init state is sufficient. The result is cached on the Task
        # under a (dt,) key so it survives changes to brainstate.environ
        # between calls.
        self._T_max_cache: Dict[Any, Optional[int]] = {}

    @property
    def is_variable_length(self) -> bool:
        return self._is_variable_length

    @property
    def max_trial_duration(self) -> Optional[int]:
        """Static upper bound on trial length, in timesteps. None for
        fixed-duration tasks (whose actual length is the exact duration).
        Resolved on demand against the current brainstate.environ.dt."""
        if not self._is_variable_length:
            return None
        dt = brainstate.environ.get_dt()
        key = id(dt) if hasattr(dt, 'mantissa') else dt
        if key not in self._T_max_cache:
            stub = Context(key=None)
            self._T_max_cache[key] = int(self.phases.max_steps(stub))
        return self._T_max_cache[key]
```

`sample_trial` dispatches on `self._is_variable_length`:

- Fixed: existing code path, byte-identical to current behavior.
- Variable: the packed flow described in §5.2–§5.4. Returns
  `(X, Y, info)` where `info['mask']` is a `(T_max,)` bool array and
  `info['length']` is a `()` int scalar (= `ctx.t_cursor` at the end of
  execution).

`batch_sample(size, /, time_first=True, return_meta=False, return_mask=False, start_index=0)`:

- Default return is unchanged: `(X, Y)`.
- `return_mask=True` adds a mask of shape `(T_max, batch)` if `time_first`
  else `(batch, T_max)`.
- `return_meta=True` is unchanged.
- `return_mask=True` and `return_meta=True` together return
  `(X, Y, mask, meta)`.

### 5.8 Phase contract for `phase_start`/`phase_end` in variable mode

Inside `execute_phase` we keep computing `ctx.phase_start = ctx.t_cursor`,
`ctx.phase_end = ctx.t_cursor + actual`, but both sides are traced
scalars. Existing leaf phases that only consume them through arithmetic
keep working. **Callbacks** that compute a value of shape
`(ctx.phase_end - ctx.phase_start, …)` must instead derive the local
length from `max_dur` (passed through `ctx` as `ctx.phase_max_steps`, a
new attribute set by the runtime).

`ReadySetGo._production_label` is the only known callback that uses
`ctx.phase_end - ctx.phase_start` to size its output; it is rewritten to
use `ctx.phase_max_steps` and a `where` against `ctx.phase_step_count`.

## 6. API impact

| Symbol | Change |
|--------|--------|
| `Phase.max_steps` | **New** abstract method (default: returns `get_duration`). |
| `Phase.step_count` | **New** method (default: `jnp.asarray(get_duration(ctx))`). |
| `Phase.is_variable` | **New** class flag. |
| `Context.mask`, `Context.t_cursor`, `Context.phase_max_steps`, `Context.phase_step_count` | **New** attributes, `None` outside variable mode. |
| `Task.is_variable_length`, `Task.max_trial_duration` | **New** read-only properties. |
| `Task.batch_sample` | **New** `return_mask: bool = False` kwarg. Default behavior unchanged. |
| `Task.sample_trial` | Variable-mode tasks: `info` dict now contains `mask` and `length` keys. Fixed-mode tasks: unchanged. |
| `TruncExp.max_value`, `UniformDuration.max_value` | **New** accessor. |
| `TruncExp.is_variable`, `UniformDuration.is_variable` | **New** class flag (`True`). |
| `VariableDuration` | **New** declarative phase class. Public, exported. |
| `HierarchicalReasoning`, `IntervalDiscrimination`, `ReadySetGo` | Internal `_compute_…` helpers deleted; `define_phases` rewritten to use `VariableDuration`. **Public behavior preserved.** |
| `conftest.VARIABLE_DURATION_TASKS` | Now safely vmaps; tests are updated to include them in vmap sweeps. |
| All other public symbols | Unchanged. |

**No deprecations or breaking changes** for users who don't write custom
`get_duration` overrides. Users who *do* override `get_duration` to return a
traced JAX value must also override `step_count` if they want vmap safety
(documented in §7.3).

## 7. Testing strategy

### 7.1 Existing tests
Every existing test in `braintools/cogtask/**/*_test.py` must pass without
modification. CI guards regression of the static path.

### 7.2 New regression tests (`task_variable_length_test.py`)

1. `test_fixed_path_byte_identical_to_baseline` — sample one trial via each
   `FIXED_DURATION_TASKS` entry with a fixed seed; compare against a frozen
   snapshot stored under `braintools/cogtask/_baselines/`.
2. `test_variable_sample_trial_shapes` — per variable task, assert
   `X.shape == (T_max, num_inputs)`, `Y.shape[0] == T_max`,
   `info['mask'].shape == (T_max,)`, `info['length'] <= T_max`, and
   `info['mask'].sum() == info['length']`.
3. `test_variable_under_jit` — wrap `task.sample_trial` with
   `brainstate.transform.jit(static_argnums=(0,))`; compare against eager
   call for identity.
4. `test_variable_under_vmap2` — call `batch_sample(8, return_mask=True)`;
   verify mask shape, `mask.sum(axis=0) == lengths`, lengths array has at
   least two distinct values.
5. `test_variable_batch_mixed_lengths` — drive `batch_sample(64)` and assert
   `unique(lengths).size >= 2` for each of `HierarchicalReasoning`,
   `IntervalDiscrimination`, `ReadySetGo`.
6. `test_mask_aware_loss_smoke` — compute a masked MSE on
   `DelayDirectionReproduction` to confirm consumers can apply masks to
   vector outputs.
7. `test_while_phase_under_vmap` — synthetic task with `While(condition,
   body, max_iterations=10)` where condition uses a traced threshold;
   assert it runs under vmap and lengths cluster at expected values.
8. `test_if_switch_under_vmap` — synthetic task with `If(predicate, then,
   else_)` whose predicate is traced; assert both branches contribute to
   the batch.
9. `test_max_trial_duration_static_property` — assert
   `Task.max_trial_duration` is an `int` (or `None` for fixed tasks).

### 7.3 Custom-phase contract test

`test_custom_variable_phase_requires_step_count` — subclassing `Phase` and
overriding `get_duration` to return a traced value (without overriding
`step_count`) raises a clear error at construction time.

## 8. Documentation updates

- **`docs/apis/cogtask.rst`** — new "Variable-length trials" section with
  example showing `batch_sample(32, return_mask=True)`, a masked loss, and a
  note on `max_trial_duration`.
- **Module docstring** of `braintools/cogtask/__init__.py` — short callout
  pointing at the new section.
- **Docstrings** of `HierarchicalReasoning`, `IntervalDiscrimination`,
  `ReadySetGo`, `TruncExp`, `UniformDuration` — add "Compatible with `vmap` /
  `jit`" notes.
- **Docstring** of `VariableDuration` — full reference with worked example.
- **`changelog.md`** — entry under the next version noting the new feature
  and listing the (small) API additions.

## 9. Out of scope

- Per-phase masks exposed in `info` — only a trial-level mask is returned.
  If needed later, individual phase masks can be reconstructed from
  `info['phase_history']` + lengths.
- Variable *channel* counts. Only the time dimension varies.
- Streaming / infinite-length tasks.
- Backwards compatibility shims for users who already overrode
  `get_duration` to return a traced value: they will get a clear error at
  construction and must opt into `step_count`.

## 10. Risks / open questions

- **Compile-time cost of `lax.fori_loop` for `While`**: bounded by
  `max_iterations`. The variable code path is opt-in via auto-detection so
  fixed tasks aren't affected.
- **Memory cost of padding**: a task with `max_iterations=100` over a body
  of `50` timesteps allocates `5000` timesteps even when most trials exit
  after `2` iterations. This is the unavoidable cost of shape stability.
  Documented in §8 of the docs.
- **`Parallel` semantics under padding**: the cursor advances by
  `max(child.step_count)`; child writes use the same start but child masks
  are OR'd. This matches today's semantics.
- **`return_meta` payloads**: only tasks whose `get_trial_meta` returns
  vmap-safe pytrees can use `return_meta=True`. Unchanged by this design.

## 11. Implementation order (preview — for `writing-plans`)

1. Introduce `Phase.max_steps` / `Phase.step_count` / `is_variable`
   defaults; no behavior change yet.
2. Refactor `Context` to hold (optional) `mask`, `t_cursor`,
   `phase_max_steps`, `phase_step_count` attributes.
3. Implement packed execution for leaf phases.
4. Extend `Sequence`, `Repeat`, `Parallel`.
5. Extend `If`, `Switch`, `While`.
6. Add `VariableDuration` declarative phase class.
7. Migrate `HierarchicalReasoning`, `IntervalDiscrimination`, `ReadySetGo`
   off the lambda-override approach.
8. Wire `Task` auto-detection, `is_variable_length`, `max_trial_duration`,
   and the `return_mask` plumbing.
9. Add regression tests.
10. Update docs / changelog.
