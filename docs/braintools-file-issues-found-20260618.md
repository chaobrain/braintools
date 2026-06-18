# `braintools/file/` — Issues, Bugs, Edge Cases & Missing Features

**Reviewer role:** Senior Python architect / JAX expert / BrainX developer
**Date:** 2026-06-18
**Scope reviewed:**

- `braintools/file/__init__.py`
- `braintools/file/_matfile.py` (+ `_matfile_test.py`, `_matfile_extra_test.py`)
- `braintools/file/_msg_checkpoint.py` (+ `_msg_checkpoint_test.py`, `_msg_checkpoint_async_test.py`, `_msg_checkpoint_extra_test.py`)
- `docs/apis/file.rst`, `docs/file/index.md` (+ notebooks)

**Test status at review time:** `99 passed, 5 skipped` in normal mode. Under `-W error::DeprecationWarning` (a common CI setting) the MATLAB tests **fail** (see I-1). The 5 skips are the entire `_matfile_test.py` (see I-19).

---

## Executive summary

The module is generally well-structured and well-tested for the *happy path*, but the review surfaced a number of real defects. The most important:

- **A time-bomb dependency bug** (`scipy.io.matlab.mio5_params` is deprecated and slated for removal in SciPy 2.0.0, and already warns on *every* parse).
- **A silently-broken safety guarantee**: `msgpack_from_state_dict` advertises shape/dtype assertions that do not exist — checkpoints with mismatched array shapes load silently, even with `mismatch='error'`.
- **Two confusing/inverted API contracts** (`header_info`, and several docstrings that contradict the code).
- A handful of **latent correctness bugs** (namedtuple field-name collision, >10 GB checkpoints cannot be reloaded, in-place mutation of the caller's `State`).

Each finding below carries a location, verified evidence, impact, and a proposed solution.

### Severity index

| ID | Severity | Area | One-line |
|----|----------|------|----------|
| I-1 | **High** | matfile | `scipy.io.matlab.mio5_params` deprecated → breaks on SciPy 2.0; warns on every parse |
| I-2 | **High** | checkpoint | Promised array shape/dtype validation is missing; wrong-shape load is silent |
| I-3 | **High** | API/docs | `header_info` parameter semantics are inverted vs. its name |
| I-4 | **Med-High** | checkpoint | >10 GB checkpoints can be saved but never reloaded (`max_size` not plumbed) |
| I-5 | **Medium** | checkpoint | `msgpack_load` docstring says "returns target unchanged" on missing file; code raises |
| I-6 | **Medium** | checkpoint | `msgpack_save` documents a `str` return value but returns `None` |
| I-7 | **Medium** | checkpoint | `msgpack_load(target=...)` mutates the caller's `State` objects in place |
| I-8 | **Medium** | checkpoint | `_restore_namedtuple` heuristic misfires for fields named `name`/`fields`/`values` |
| I-9 | **Medium** | checkpoint | `AsyncManager` async path serializes synchronously; failures can be swallowed |
| I-10 | **Medium** | checkpoint | `wait_previous_save` warns on *normal* waits; stale `braintools.checkpoints` message |
| I-11 | **Medium** | checkpoint | Concurrent/multi-process saves to the same path collide on a fixed `.tmp` name |
| I-12 | **Low-Med** | checkpoint | Registry uses first-match-wins, not most-specific-subclass |
| I-13 | **Low-Med** | checkpoint | `jax.tree_util.Partial.func` is not serialized (silent, undocumented) |
| I-14 | **Low** | checkpoint | `assert` used for runtime key validation (stripped under `python -O`) |
| I-15 | **Low** | checkpoint | Dead code: `_EmptyNode`, unused `wait`, ineffective chunk warning |
| I-16 | **Low** | checkpoint | `from concurrent.futures import thread` (private module); `multiprocessing.cpu_count()` |
| I-17 | **Low** | checkpoint | `parallel=True` default adds pure overhead for the common (<128 MB) case |
| I-18 | **Low** | checkpoint | dict subtype (`OrderedDict`/`defaultdict`) and non-str keys not preserved w/o target |
| I-19 | **Medium** | tests | `_matfile_test.py` is fully skipped *and* references a nonexistent `sio` attribute |
| I-20 | **Low** | docs | `.pkl` extension used for msgpack data in examples; inverted `header_info` example |
| I-21 | **Medium** | features | No v7.3 (HDF5) `.mat` support; no `save_matfile`; exceptions/`max_size` not public |
| I-22 | **Low** | docs/API | Exceptions undocumented; `file.rst`/`index.md` lack narrative on chunking/async/mismatch |
| I-23 | **Low** | code quality | Library defaults to `verbose=True` + raw `print()`; inconsistent `str` vs `PathLike` typing |

---

## High severity

### I-1 — `scipy.io.matlab.mio5_params` is deprecated (breaks on SciPy 2.0; warns on every parse)

**Location:** `braintools/file/_matfile.py:21` (import), `:123` (`mio5_params.mat_struct` attribute access).

**Evidence (verified, SciPy 1.17.1):**

```
DeprecationWarning: Please import `mat_struct` from the `scipy.io.matlab`
namespace; the `scipy.io.matlab.mio5_params` namespace is deprecated and
will be removed in SciPy 2.0.0.
```

Worse, the warning fires not just at import but on **every** `mio5_params.mat_struct` attribute access inside the hot recursive `parse_mat` loop (line 123). Under `-W error::DeprecationWarning` the round-trip tests fail outright at `_matfile.py:123`.

**Impact:** Guaranteed breakage on SciPy 2.0; noisy warnings now; CI failures for any project pinning warnings-as-errors.

**Proposed solution:** Use the public name and resolve the attribute once.

```python
# at top of _matfile.py
try:                                    # SciPy >= 1.8
    from scipy.io.matlab import mat_struct
except ImportError:                     # very old SciPy fallback
    from scipy.io.matlab.mio5_params import mat_struct
```

Then `isinstance(element, mat_struct)` at line 123. (`mat_struct` has been importable from the public `scipy.io.matlab` namespace since SciPy 1.8.)

---

### I-2 — Promised array shape/dtype validation does not exist (silent wrong-shape load)

**Location:** `braintools/file/_msg_checkpoint.py:170-189`. The `msgpack_from_state_dict` docstring states:

> "…as well as lets us add assertions that shapes and dtypes don't change."

But no leaf-level shape/dtype check exists anywhere in the restore path. Array leaves are replaced wholesale by `_msgpack_ext_unpack`.

**Evidence (verified):** saving a `(10,) float32` array and loading into a target expecting `(3,) float64`, with `mismatch='error'`:

```
A) shape mismatch, mismatch=error -> loaded shape: (10,) dtype: float32 | warnings: 0 | RAISED? No
```

The mismatch modes only cover *structural* mismatches (dict keys, list length, namedtuple fields, `Quantity` unit) — never array shape/dtype.

**Impact:** A checkpoint from a differently-shaped model loads silently with the *saved* shapes, corrupting the in-memory model. This is exactly the class of bug the docstring claims to prevent, and `braintools.optim` / `braintools.trainer` rely on load correctness (`_checkpoint.py:125`, `_callbacks.py:493`).

**Proposed solution:** Either (a) remove the false claim from the docstring, or (b) implement a leaf validator honoring `mismatch`. Recommended (b): register a handler for `np.ndarray`/`jax.Array` (or add a leaf check in `msgpack_from_state_dict`) that compares `target.shape`/`target.dtype` against the incoming array and routes through `_handle_mismatch`:

```python
def _restore_array(target, state, mismatch='error'):
    if hasattr(target, 'shape'):
        _handle_mismatch(target.shape != state.shape,
                         f'Shape mismatch: expected {target.shape}, got {state.shape} '
                         f'at path {current_path()}', mismatch)
        _handle_mismatch(target.dtype != state.dtype,
                         f'Dtype mismatch: expected {target.dtype}, got {state.dtype} '
                         f'at path {current_path()}', mismatch)
    return state
```

---

### I-3 — `header_info` parameter semantics are inverted relative to its name

**Location:** `braintools/file/_matfile.py:30` (default `True`), `:143` (`if not header_info or not key.startswith('__')`); echoed in `__init__.py:55`.

**Evidence (verified):**

```
header_info=True (default) keys: ['a']
header_info=False keys        : ['__globals__', '__header__', '__version__', 'a']
```

So `header_info=True` **excludes** header metadata, and `header_info=False` **includes** it — the opposite of what the name implies. The `__init__.py` example even reads `load_matfile('data.mat', header_info=False)  # Include MATLAB metadata`.

**Impact:** A near-certain source of user error; reads as a bug to anyone who hasn't read the docstring.

**Proposed solution:** Rename to an intention-revealing flag and invert the default, keeping `header_info` as a deprecated alias for one release:

```python
def load_matfile(filename, *, include_header=False, header_info=None, ...):
    if header_info is not None:
        warnings.warn("`header_info` is deprecated and inverted; use `include_header`.",
                      DeprecationWarning, stacklevel=2)
        include_header = header_info  # NOTE: old True meant "exclude", so map carefully
    ...
    keep = include_header or not key.startswith('__')
```

At minimum, if the signature must stay frozen, rewrite the docstring/examples to make the inversion unmissable.

---

## Medium–High severity

### I-4 — Checkpoints larger than 10 GB can be saved but never reloaded

**Location:** `_msg_checkpoint.py:658-683` (`_msgpack_restore` hard-codes `max_size = 10 * 1024**3` and raises `ValueError` above it); `:696-715` (`_from_bytes` calls `_msgpack_restore(encoded_bytes)` with no `max_size`); `:1045` (`msgpack_load` likewise). The `max_size` parameter is **not exposed** by any public function.

**Impact:** The chunking machinery (`MAX_CHUNK_SIZE`, `_chunk`) exists specifically to support arrays exceeding msgpack's 2 GB leaf limit (e.g., embedding tables). But a save that succeeds at, say, 12 GB total cannot be reloaded — `msgpack_load` raises `Checkpoint data too large` with no override. Save/load are asymmetric.

**Proposed solution:** Plumb `max_size` through `msgpack_load` → `_from_bytes` → `_msgpack_restore` (default `None` = unbounded, or a generous default). Note the in-memory guard is also weak: it checks `len(encoded_pytree)` *after* the whole file is already read into RAM, so it does not actually protect against OOM during read. Consider checking `os.path.getsize()` in `msgpack_load` before reading instead.

---

## Medium severity

### I-5 — `msgpack_load` docstring contradicts its missing-file behavior

**Location:** `_msg_checkpoint.py:990-997` (docstring) vs. `:1001-1002` (code).

The docstring (ported from Flax) says the function "returns the passed-in `target` unchanged" when the file is not found. The code instead does:

```python
if not os.path.exists(filename):
    raise ValueError(f'Checkpoint not found: {filename}')
```

(`test_load_missing_file_raises` confirms the raise.)

**Impact:** Any consumer that trusted the documented "graceful return" contract (e.g., resume-if-exists logic) will instead get an exception. The wording about "step"/"directory" is also dead Flax terminology — this implementation only handles single files.

**Proposed solution:** Rewrite the Returns/Raises sections to match reality (raises `ValueError` on missing file, `InvalidCheckpointPath` on corrupt data), and delete the step/directory paragraph.

### I-6 — `msgpack_save` documents a `str` return but returns `None`

**Location:** `_msg_checkpoint.py:881` (`-> None`) vs. docstring `:914-917` ("Returns … out: str — Filename of saved checkpoint."). No `return` statement exists.

**Proposed solution:** Either return `filename` (useful, and matches the doc) or delete the Returns block. Returning `filename` is the friendlier fix.

### I-7 — `msgpack_load(target=...)` mutates the caller's `State` in place

**Location:** `_msg_checkpoint.py:421-437` (`_restore_brainstate` does `x.value = ...; return x`).

**Evidence (verified):**

```
B) original State object mutated by load? in-memory st.value = [0. 1. 2. 3.]
   out is same object as target dict value: True
```

After `msgpack_load(fn, target=data)`, the original `State` in `data` has been mutated, and the returned object *is* the same instance.

**Impact:** Contradicts the `msgpack_from_state_dict` contract ("A copy of the object with the restored state", `:188`) and the behavior of every other registered type (dict/list/tuple/namedtuple/Quantity all return new objects). A user writing `restored = msgpack_load(fn, target=model)` and expecting `model` to be untouched is surprised; it also makes the operation non-idempotent under retries.

**Proposed solution:** Document the in-place semantics prominently on `msgpack_load`/`msgpack_from_state_dict` (it is currently only noted on the private helper), or make it opt-in. The mutation is deliberate for live models, so the realistic fix is a clear, top-level docstring warning plus a note that the return value should be used.

### I-8 — `_restore_namedtuple` heuristic misfires on fields `name`/`fields`/`values`

**Location:** `_msg_checkpoint.py:350-352`:

```python
if set(state_dict.keys()) == {'name', 'fields', 'values'}:
    state_dict = {state_dict['fields'][str(i)]: state_dict['values'][str(i)] ...}
```

**Evidence (verified):** a legitimate `namedtuple('NT', ['name','fields','values'])` with real per-field data `{'name': 'hello', 'fields': {...}, 'values': {...}}` is reinterpreted as the special "serialized" form and then raises a spurious:

```
ValueError: The field names of the state dict and the named tuple do not match, got {'x'} and {'values', 'fields', 'name'} at path
```

**Impact:** Any namedtuple whose three fields happen to be exactly `name`, `fields`, `values` cannot be round-tripped. Also note this branch decodes a format that `_namedtuple_state_dict` (`:344-345`) never *produces*, so its provenance is unclear.

**Proposed solution:** Disambiguate with an explicit sentinel rather than a field-name guess (e.g., wrap the alternate form under a reserved key like `'__namedtuple__': True`), or drop the branch if no producer emits it.

### I-9 — `AsyncManager` "async" path serializes synchronously and can swallow failures

**Location:** `_msg_checkpoint.py:947-958`. Only the file *write* (`_save_main_ckpt_file`) is dispatched to the executor; serialization `target = _to_bytes(target)` runs **synchronously** on the calling thread before the submit. For large models the CPU-bound serialization (the expensive part) is not actually backgrounded, contradicting "the save will run without blocking the main thread" (`:907-910`).

Separately, an exception raised inside the background task is stored in the `Future` and only surfaces when `.result()` is later called. If the user never calls `wait_previous_save()`/`save_async()` again, the failure surfaces (if at all) inside `__del__ → close → wait_previous_save → .result()`, which is wrapped in a bare `except Exception: pass` (`:797-799`) — so **async save errors can be silently lost**.

**Proposed solution:** (a) Move `_to_bytes(target)` inside `save_main_ckpt_task` so serialization is also async (note: must snapshot/avoid concurrent mutation of `target`). (b) Don't swallow in `__del__`; at minimum log via `warnings`/`logging` before suppressing, and document that callers should `wait_previous_save()` to observe errors.

### I-10 — `wait_previous_save` warns during normal waits and names the wrong module

**Location:** `_msg_checkpoint.py:812-820`:

```python
warnings.warn('The previous async braintools.checkpoints.save has not finished yet. '
              'Waiting for it to complete before the next save.', UserWarning)
```

Two problems: (1) `braintools.checkpoints.save` is a stale module name — the function is `braintools.file.msgpack_save`. (2) The warning fires on *every* wait where a save is still running, including the perfectly normal `manager.wait_previous_save()` / context-manager exit. Waiting is the intended behavior, not an anomaly.

**Proposed solution:** Only warn in the genuinely surprising case (a *new save* arriving while a prior one is unfinished — i.e., inside `save_async`, not inside the user-invoked `wait_previous_save`). Fix the module name.

### I-11 — Concurrent/multi-process saves to the same path collide on a fixed `.tmp` name

**Location:** `_msg_checkpoint.py:856` — `tmp_filename = filename + '.tmp'`.

If two `AsyncManager`s, two processes, or an interrupted+retried save target the same `filename`, they share one `.tmp` path and can corrupt each other's writes (the Flax original incorporates a process index into the temp name). A single `AsyncManager` serializes its own saves, but nothing protects across managers/processes.

**Proposed solution:** Make the temp name unique, e.g. `f'{filename}.tmp-{os.getpid()}-{id(target)}'` (avoiding `Date.now`/`uuid` only if determinism is required — neither applies here, `uuid4().hex` is fine), then atomically `os.replace` onto `filename`.

### I-19 — `_matfile_test.py` is entirely skipped *and* references a nonexistent attribute

**Location:** `braintools/file/_matfile_test.py:55` (`@pytest.mark.skip(reason="not implemented")`) and `:63/74/94/117/140` (all set/read `matfile_mod.sio`).

The current `_matfile.py` imports `loadmat`/`mio5_params` directly and has **no** `sio` attribute — the test was written against an older `import scipy.io as sio` implementation. So these 5 tests are both dead (skipped) and broken (would `AttributeError`/not patch anything if unskipped). Real coverage lives in `_matfile_extra_test.py`.

**Proposed solution:** Delete `_matfile_test.py` (its intent is fully covered by `_matfile_extra_test.py`), or rewrite it against the current implementation (patch `braintools.file._matfile.loadmat` / `.mat_struct`). Leaving a permanently-skipped file invites bit-rot.

---

## Low–Medium severity

### I-12 — Serialization registry is first-match-wins, not most-specific-subclass

**Location:** `_msg_checkpoint.py:192-197` and `:209-214` iterate `_STATE_DICT_REGISTRY` and take the **first** `issubclass` match. For user class hierarchies (`Base` registered before `Derived`), a `Derived` instance dispatches to `Base`'s handler rather than the more specific one. Insertion order, not specificity, decides.

**Proposed solution:** When multiple registered types match, choose the most derived (`max(..., key=lambda t: len(t.__mro__))` among matches), or document that handlers must be registered most-specific-first.

### I-13 — `jax.tree_util.Partial.func` is not serialized

**Location:** `_msg_checkpoint.py:442-453`. Only `args`/`keywords` are serialized; on restore the function comes from the *target* template (`x.func`). Loading a `Partial` without a matching target yields a bare `{'args':..., 'keywords':...}` dict, silently losing the callable.

**Impact:** Reasonable (functions aren't msgpack-serializable), but undocumented — a quiet footgun.

**Proposed solution:** Document the limitation explicitly (a target with the correct `func` is mandatory to restore a `Partial`).

### I-14 — `assert` used for runtime validation (stripped by `python -O`)

**Location:** `_msg_checkpoint.py:220` — `assert isinstance(key, str), 'A state dict must only have string keys.'`. Under `python -O` this check vanishes, allowing non-string keys to slip into msgpack and fail later with an opaque error.

**Proposed solution:** Replace with `if not isinstance(key, str): raise TypeError(...)`.

### I-15 — Dead / ineffective code

- `_EmptyNode` (`:733-734`) is defined and never used (leftover from the Flax port; its comment even references a `struct.dataclass` that isn't there).
- `wait = list(pool.map(read_chunk, ...))` (`:1041`) — the result list is never read; per-chunk byte counts are discarded, so a short read would silently leave zero-filled gaps with no validation.
- The "inefficient chunking" warning (`:576-582`) is effectively unreachable: `chunksize < 1000` requires `itemsize > ~1 MB`/element, which no real dtype has (`test_chunk_size_warning` confirms it never fires for normal arrays).

**Proposed solution:** Remove `_EmptyNode`; either validate `sum(wait) == file_size` or drop the variable; remove or fix the chunk warning threshold.

### I-16 — Non-idiomatic concurrency imports

**Location:** `_msg_checkpoint.py:28` `from concurrent.futures import thread` (a private submodule) used at `:779` and `:1039`; and `multiprocessing.cpu_count()` at `:1038` (can raise `NotImplementedError`; pulls in the whole `multiprocessing` module).

**Proposed solution:** `from concurrent.futures import ThreadPoolExecutor`; use `os.cpu_count() or 1` (never raises).

### I-17 — `parallel=True` default is pure overhead for the common case

**Location:** `_msg_checkpoint.py:1009-1041`. For files < 128 MB, `num_chunks == 1`, so the "parallel" path opens the file a second time and reads it in one worker thread — strictly more work than `fp.read()`, while the outer handle is opened only to call `.seekable()`. Most checkpoints are well under 128 MB.

**Proposed solution:** Gate the parallel path on a meaningful size threshold (e.g., only when `file_size > buf_size`), and consider defaulting `parallel=False` for small files.

### I-18 — dict subtype and non-string keys are not preserved without a target

**Location:** `_dict_state_dict`/`_restore_dict` (`:287-338`). All keys are stringified on save; `_restore_dict` always builds a plain `dict`. Loading **without** a target therefore turns `{1: x}` into `{'1': x}` and an `OrderedDict`/`defaultdict` into a plain `dict` (losing a `defaultdict` factory). With a target the original key objects/types are recovered, so this only bites the no-target path.

**Proposed solution:** Document that key types and dict subclasses require a `target` to round-trip; optionally reconstruct `type(xs)` in `_restore_dict` when it is a dict subclass.

---

## Low severity / polish

### I-20 — Documentation examples are misleading

- `__init__.py:40, 68, 75` use a `.pkl` (pickle) extension for msgpack data: `msgpack_save('model_checkpoint.pkl', ...)`. Use `.msgpack`/`.msg` to avoid implying pickle.
- `__init__.py:55` reinforces the inverted `header_info` (`header_info=False  # Include MATLAB metadata`) — see I-3.

### I-22 — API docs omit exceptions and key concepts

`docs/apis/file.rst` lists the 7 public callables but not `AlreadyExistsError`/`InvalidCheckpointPath`, and neither it nor `docs/file/index.md` documents the headline behaviors (array chunking, async saving semantics, the `mismatch` modes, the no-target raw-state-dict return). Compared with sibling modules this section is thin.

**Proposed solution:** Add a short narrative section (mismatch modes, async/`AsyncManager` lifecycle, chunking/size limits, save vs. load symmetry) and document the exception types.

### I-23 — Library-unfriendly defaults & inconsistent typing

- `load_matfile` (`verbose=True`), `msgpack_save`/`msgpack_load` (`verbose=True`) print to stdout **by default** using raw `print()`/`sys.stdout.write` (and the two functions even use different mechanisms — `print` at `:921` vs `sys.stdout.write` at `:1004`). Libraries should default to quiet and use the `logging` module.
- `msgpack_save(filename: str, ...)` / `msgpack_load(filename: str, ...)` are type-hinted `str` but accept `os.PathLike` at runtime (and `_matfile` validates `PathLike`); `target` is `brainstate.typing.PyTree` on save but `Optional[Any]` on load. Align the hints (`Union[str, os.PathLike]`).

---

## Missing features (I-21 and related)

1. **No MATLAB v7.3 (HDF5) support.** `scipy.io.loadmat` cannot read v7.3 `.mat` files (HDF5-backed, common for large datasets); such files surface as a generic `ValueError: Failed to load MATLAB file`. Consider detecting the HDF5 magic and delegating to `h5py`/`mat73`, or at least emitting a targeted error message pointing users to a v7.3 reader.
2. **No `save_matfile`.** The module loads `.mat` but cannot write it. `scipy.io.savemat` is used only in tests. A thin `save_matfile` would make the MATLAB story symmetric.
3. **Public surface gaps.** `InvalidCheckpointPath`/`AlreadyExistsError` (which `msgpack_load`/`msgpack_save` raise) are not in `__all__`, so callers must reach into `_msg_checkpoint` to catch them. The `max_size` knob (I-4) is similarly private. Promote the exceptions and expose `max_size`.
4. **No checksum/version header on checkpoints.** Corruption is only detected when msgpack fails to decode; a small magic+version+CRC header would enable integrity checks and forward-compatible format evolution.
5. **No streaming.** Both `load_matfile` and `msgpack_load` read the entire payload into memory; very large artifacts cannot be processed incrementally (relevant given the chunking machinery targets multi-GB arrays).

---

## Suggested remediation order

1. **I-1** (SciPy deprecation) — small, time-sensitive, unblocks strict-warning CI.
2. **I-2** (missing shape/dtype guard) — correctness/data-integrity; pick "fix" or "honest docstring".
3. **I-5 / I-6 / I-10** (docstring/behaviour mismatches & misleading warning) — cheap, high clarity gain.
4. **I-3** (inverted `header_info`) — needs a deprecation cycle; start now.
5. **I-4 / I-7 / I-8 / I-11** (latent correctness) — schedule with tests.
6. **I-19 + remaining low-severity / polish** — opportunistic cleanup.

Every behavioral finding above (I-1, I-2, I-3, I-7, I-8, plus the test/dead-code observations) was reproduced against the current tree during this review.
