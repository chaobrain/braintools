# `braintools.conn` — Issues, Bugs & Edge Cases Audit

**Date:** 2026-06-18
**Scope:** `braintools/conn/` (source + tests) and its documentation (`docs/apis/conn.rst`, `docs/conn/*.ipynb`, module/class docstrings).
**Reviewer perspective:** senior Python architect · JAX expert · computational-neuroscience (BrainX) developer.
**Method:** Static reading of all 8 source files + matching tests, cross-checked against the base contracts, with **runtime verification** of the highest-severity findings (repro snippets included). Findings are tagged **Confirmed** (reproduced or unambiguous from code), **Likely**, or **Suspected**.

---

## Executive summary

The `conn` module is broad and well-featured, but the audit found **systemic correctness problems** that are mostly invisible today because the test-suite is overwhelmingly happy-path (asserts `n_connections > 0`, shapes, and metadata — rarely the actual indices/weights/structure). The most important issues:

| # | Severity | Area | One-line |
|---|----------|------|----------|
| XC-1 / BASE-1 | **Critical** | base | `weight2csr()`/`delay2csr()` attach weights/delays to the **wrong (row,col)** when `pre_indices` aren't pre-sorted (silent data corruption). |
| XC-2 / BASE-2 | **Critical** | base | `Connectivity.__call__` caches the first result and returns it for **all** later calls — different sizes/positions are silently ignored. |
| TOP-1 | **Critical** | topological | `ModularGeneral` crashes (`np.concatenate`) on scalar weights/delays — i.e. on its own documented examples. |
| TOP-2 | **Critical** | topological | `ModularGeneral` reuses cached sub-results → all inter-module blocks are **identical** (no statistical independence). |
| TOP-3 | **Critical** | topological | `ScaleFree` emits **out-of-bounds** neuron indices when `m ≥ N`. |
| KER-1 | **Critical** | kernel | `SobelKernel`/`LaplacianKernel` silently drop **all negative coefficients** → not edge/Laplacian operators at all. |
| CMP-1 | **Critical** | compartment | `DendriticIntegration`/`SynapticClustering` emit massive **duplicate** connections. |
| DOC-1…6 | **Critical** | docs | Multiple module/class docstring examples **crash** on copy-paste (`LogNormal(sigma=)`, `ExponentialProfile(scale=)`, `weight_distribution=`, `DistanceModulated(lambda)`, `DistanceDependent(sigma=,decay=)`). |
| XC-3 | **High** | base/spatial/kernel/compartment | **Autapses** (self-connections) silently created by spatial, kernel, and compartment patterns; `Random` has the inverse bug (drops valid cross-population diagonal). |
| XC-4 | **High** | spatial/topological | **Duplicate edges** from `Ring`, `Grid2d` (periodic, small grids), `SmallWorld` rewiring — corrupt dense/CSR outputs. |
| BASE-3 | **High** | base | `ScaledConnectivity` mutates the base connectivity's **shared cached result** in place. |
| XC-5 | **High** | whole module | **No parameter validation** (`prob`, `exc_ratio`, `sigma`, `m`, `k`, `n_modules`, `density`, …) → silent wrong results or cryptic errors. |
| XC-6 | **Medium** | whole module | **Not JAX-compatible**: NumPy `RandomState`, `scipy.cdist`, Python loops, in-place mutation, data-dependent shapes — undocumented. |
| XC-7 | **Medium** | several | **O(N²)** Python loops / dense `(pre,post)` matrices → memory/CPU blow-ups at scale; advertised `max_distance` gives no savings. |

Counts (approx.): **Critical ≈ 12**, **High ≈ 10**, **Medium ≈ 18**, **Low ≈ 14**.

> **Two single-line fixes eliminate most doc crashes:** `LogNormal/Normal` take `std`, not `sigma`; `ExponentialProfile` takes `decay_constant`, not `scale`.

---

## A. Cross-cutting / systemic issues

These span multiple files; fix them once at the base/convention level.

### XC-1 (BASE-1) — `weight2csr()` / `delay2csr()` misalign data when `pre_indices` are unsorted  ·  **Critical · Confirmed**
- **Location:** `_base.py:177-197` (`weight2csr`, `delay2csr`) + `_base.py:993-1027` (`compute_csr_indices_indptr`).
- **Root cause:** `compute_csr_indices_indptr` sorts by `pre_indices` (`order = np.argsort(pre_indices)`, line 1015) and returns `indices = sorted_post`. But `weight2csr`/`delay2csr` then build `brainevent.CSR((weights, indices, indptr), ...)` passing **`self.weights` in the original (unsorted) order**. The CSR `data` array must be in the same order as `indices`; it isn't. So whenever `pre_indices` is not already ascending, weights/delays are attached to the wrong column positions.
- **Verified:**
  ```python
  # Hand-built:
  r = ConnectionResult([2,0,1],[0,1,2], 3,3, weights=np.array([10.,20.,30.]))
  r.weight2dense()        # correct: (2,0)=10,(0,1)=20,(1,2)=30
  r.weight2csr().todense()# WRONG:   (0,1)=10,(1,2)=20,(2,0)=30   <-- permuted

  # Real generator (ScaleFree emits unsorted pre):
  sf = ScaleFree(m=2, seed=0)(pre_size=12, post_size=12)   # pre_sorted=False
  # weight2dense() == weight2csr().todense()  ->  False
  ```
  `weight2dense`/`delay2matrix` are correct (scatter assignment), so the two conversion paths **disagree** — a particularly nasty silent bug. Generators with unsorted `pre`: `ScaleFree` (confirmed), `Conv2d/Gabor/CustomKernel` (loop over post), `_difference` composites, and any future loop-over-target pattern. (Vectorized ones — `Random`, spatial `np.where`, `GaussianKernel`, union/intersection — happen to emit sorted `pre`, masking the bug.)
- **Fix:** reorder the data by the same permutation used for the indices. Return `order` from `compute_csr_indices_indptr` (or sort inside the converters): `data = (mantissa)[order]`, then `CSR((data*unit, indices, indptr), shape)`. Add a regression test asserting `weight2dense() == weight2csr().todense()` for an unsorted-`pre` result.

### XC-2 (BASE-2) — `__call__` result caching ignores changed arguments  ·  **Critical · Confirmed**
- **Location:** `_base.py:241-262`.
- **Problem:** `__call__` stores `self._cached_result` on the first call and returns it on every subsequent call unless `recompute=True`, **ignoring** new `pre_size`/`post_size`/`pre_positions`/`post_positions`.
- **Verified:**
  ```python
  c = Random(prob=0.5, seed=1)
  c(pre_size=10,  post_size=10).shape   # (10, 10)
  c(pre_size=100, post_size=100).shape  # (10, 10)  <-- stale; same object returned
  ```
- **Impact:** This is a footgun everywhere, but it actively **breaks** composed/looped patterns that legitimately re-call a sub-connectivity with different sizes/positions — see **TOP-2** (`ModularGeneral` produces identical blocks) and the spatial agent's confirmation that reusing a connector with new positions returns stale results.
- **Fix:** make the cache key include `(effective_pre_size, effective_post_size, id/shape of positions)`, or drop caching for the call path and cache only when args are unchanged. At minimum, internal callers (`CompositeConnectivity`, `ModularGeneral`, `SobelKernel('both')`, `ScaledConnectivity`) must call with `recompute=True` or call `generate(...)` directly. Document the behavior either way.

### XC-3 — Autapses (self-connections) handled inconsistently across the module  ·  **High · Confirmed**
The same-population self-connection policy is incoherent:
- **Silently create autapses (no opt-out):**
  - `_spatial.py`: `DistanceDependent`/`Gaussian`/`Exponential` (`:219-222`) and `RadialPatches` (`:1432-1448`) — distance 0 ⇒ probability 1.0 ⇒ every neuron self-connects when `pre_positions is post_positions` (the canonical recurrent case in nearly every example). Verified: 50/50 self-edges.
  - `_kernel.py`: all kernels (e.g. `:331-335`, `:152-176`, `:1135`) — coincident pre/post hits the kernel **peak** weight on the diagonal.
  - `_compartment.py`: `AllToAllCompartments` → `CompartmentSpecific` (`:1642-1656`, gen `:313-324`) — emits SOMA→SOMA on the same neuron, etc.
- **Inverse bug** — `Random`/`FixedProb` (`_random.py:175-176`) skip `i == j` **even when pre and post are different populations**, silently dropping valid cross-population pairs on the index diagonal (see **RND-1**).
- **Fix:** introduce one convention — an `allow_self_connections: bool` parameter (as `_random.py` already exposes) honored by all same-population patterns, and only applied when pre and post are the same population. Mask `pre == post` after generation. Document the default.

### XC-4 — Duplicate edges from several generators corrupt downstream conversions  ·  **High · Confirmed**
- `Ring` (`_spatial.py:836-847`): duplicates + self-loops when `neighbors ≥ n//2` (the docstring's "maximally connected"); verified `Ring(neighbors=5)(5,5)` → 50 edges, self-loops, all pairs duplicated.
- `Grid2d` periodic (`_spatial.py:1101-1118`): duplicates on small grids (≤2 along a dim); verified 2×2 Moore-periodic → 32 edges with 12 duplicates.
- `SmallWorld` rewiring (`_topological.py:146-160`): no check that the new edge already exists ⇒ duplicates; also `k ≥ n` wraps onto self/duplicates.
- `DendriticIntegration`/`SynapticClustering` (`_compartment.py:1196-1224`): cross-cluster duplicates (see **CMP-1**).
- **Interaction with base:** `weight2dense`/`delay2matrix` (`_base.py:173,189`) use scatter assignment, so duplicates are **silently collapsed (last-wins)** — verified `dense[0,1]=7.0` for weights `[5,7]` on the same edge — while `weight2csr` (with the XC-1 fix) would **sum** them. Either way the result is wrong/ambiguous.
- **Fix:** deduplicate `(pre, post[, pre_comp, post_comp])` in each offending generator (mirror `RadialPatches`' set-based dedup), validate neighbor/degree ranges, and decide a documented duplicate policy in `ConnectionResult`.

### XC-5 — Pervasive lack of parameter validation  ·  **High · Confirmed**
No range/precondition checks in most constructors. Examples (all verified or unambiguous):
- `Random(prob=2.0)` ⇒ all-connected; `prob=-0.5` ⇒ empty — no error despite "between 0 and 1".
- `ExcitatoryInhibitory(exc_ratio=1.5)` ⇒ `ValueError: negative dimensions are not allowed` (cryptic).
- `ScaleFree(m=5)(3,3)` ⇒ out-of-bounds indices (**TOP-3**).
- `SmallWorld(k=3)` ⇒ silently degree 2 (odd-k truncation, **TOP-4**); `k ≥ n` ⇒ self/duplicates.
- `ModularRandom(n_modules=5)(3,3)` ⇒ silently collapses to one module (**TOP-6**).
- Spatial `sigma=0` ⇒ divide-by-zero ⇒ NaN ⇒ silently empty; negative `sigma`/`decay_constant` ⇒ probability increasing with distance.
- Compartment classes: `density=5.0`, negative probs, negative `branches_per_axon` (→ cryptic `poisson` error) all pass.
- **Fix:** add explicit `ValueError`s in `__init__` for probabilities `∈[0,1]`, positive scales/degrees, `m<N`, `k` even and `<N`, `n_modules≤N`, etc. Centralize a small validation helper.

### XC-6 — Module is NumPy-only and not JAX `jit`/`vmap`-compatible (undocumented)  ·  **Medium · Confirmed**
- `self.rng` is `np.random.RandomState`/global `np.random` (`_base.py:238`); spatial/compartment use `scipy.spatial.distance.cdist`; many patterns use Python loops, `np.where`-driven data-dependent shapes, and in-place mutation (e.g. distance profiles' `prob[mask]=0`). None of this is traceable/jittable/vmappable.
- In a JAX-first ecosystem this is a legitimate design choice (connectivity is a host-side preprocessing step), but it is **nowhere documented**, and some idioms (in-place mutation) would fail outright on JAX arrays if a user passes them.
- **Fix:** document explicitly that connectivity generation is eager/host/NumPy-only (not `jit`/`vmap`-safe). If on-device generation is a goal, that is a larger design effort (track separately).

### XC-7 — O(N²) Python loops and dense `(pre, post)` allocations  ·  **Medium · Confirmed**
- `Random.generate` (`_random.py:171-180`) is a Python double loop (the other random classes are vectorized) — ~65 ms at 600×600, minutes at 10⁴×10⁴.
- `ModularRandom`/`HierarchicalRandom`/`CorePeripheryRandom` (`_topological.py:666-679,1281-1319,1546-1571`) are O(N²) double loops with per-pair `rng.random()` (their large-N tests are `@skip("too slow")`).
- `GaborKernel`/`Conv2dKernel`/`CustomKernel` loop over post (and pixels) in Python (`_kernel.py:506-532,152-176,1127-1152`).
- `AxonalProjection` topographic branch is a nested Python loop (`_compartment.py:964-967`).
- Spatial `DistanceDependent` builds full `(pre,post)` `cdist` + `random` + `probs` matrices (`_spatial.py:213,219`); `max_distance` is applied **after** the dense compute, so the advertised "improves computational efficiency" is false.
- **Fix:** vectorize (`rng.random((a,b)) < prob` per block + `np.nonzero`; precompute module/level arrays); for spatial/large-N use `cKDTree.query_ball_point`/`query_pairs`; correct the `max_distance` docstrings.

### XC-8 — `generate()` signatures inconsistent with the abstract base  ·  **Low · Confirmed**
- Base declares `generate(self, pre_size, post_size, pre_positions=None, post_positions=None, **kwargs)` (`_base.py:281-291`) and `Random.generate` follows it, but `ClusteredRandom`/`AllToAll`/`OneToOne`/`ExcitatoryInhibitory` and several others use `generate(self, **kwargs)` and fish args from `kwargs`. Works only because `_generate` always calls by keyword; a positional `generate(pre, post)` call (as advertised) raises `KeyError`.
- **Fix:** make all overrides match the base signature.

---

## B. `_base.py`

### BASE-1 — see **XC-1** (CSR/delay misalignment). **Critical · Confirmed**
### BASE-2 — see **XC-2** (caching). **Critical · Confirmed**

### BASE-3 — `ScaledConnectivity.generate` mutates the base's cached result in place  ·  **High · Confirmed**
- **Location:** `_base.py:978-990` (`result.weights = result.weights * self.weight_factor`, same for delays).
- **Problem:** `self.base_connectivity(**kwargs)` returns the base's **cached** `ConnectionResult`; scaling mutates that shared object. So after building/using a scaled view, the base connectivity's own weights are corrupted.
- **Verified:**
  ```python
  base = AllToAll(weight=2.0*u.nS, include_self_connections=True)
  r1 = base(3,3)                     # weights = 2 nS
  scaled = base * 3.0; scaled(3,3)   # scales the *shared* cached result
  base(3,3)                          # weights now 6 nS  <-- base mutated; r1 is the same object
  ```
- **Fix:** copy before scaling — build a new `ConnectionResult` (or `copy.copy` + new `weights`/`delays` arrays) instead of mutating `result` in place.

### BASE-4 — `weight2dense` / `delay2matrix` silently collapse duplicate edges  ·  **Medium · Confirmed**
- **Location:** `_base.py:168-175`, `184-190`. Scatter assignment `matrix[pre, post] = weights` is last-wins for duplicates (verified). Given XC-4, several generators produce duplicates. At minimum document; ideally accumulate (`np.add.at`) or reject duplicates upstream. Note `weight2dense` also loses multi-compartment resolution (collapses to neuron×neuron).

### BASE-5 — `CompositeConnectivity` size assertions reject equivalent tuple/int sizes  ·  **Medium · Likely**
- **Location:** `_base.py:377-378` (`assert conn1.pre_size == conn2.pre_size`). `(10,10)` vs `100` denote the same population but compare unequal (same defect class as TOP-9). Compare `int(np.prod(...))` instead. Also the asserts use `assert` (stripped under `python -O`) — prefer explicit `raise`.

### BASE-6 — Composite `_intersection` weight/delay alignment assumes uniqueness  ·  **Medium · Likely**
- **Location:** `_base.py:700-705`. Alignment of common edges relies on `np.argsort(conn_codes1[mask1])` vs `...2[mask2]` matching one-to-one. If either result contains **duplicate** `(pre,post)` codes, `mask1.sum() != common_codes.size` and the `weights1_common`/`weights2_common` lengths/values misalign. Given XC-4, duplicates are real. Dedup each result's codes first, or join on codes explicitly.

### BASE-7 — `_union` `max_post` can break on `None` post_size  ·  **Low · Suspected**
- **Location:** `_base.py:465-470`. `max(np.max(all_post)+1, result1.post_size, result2.post_size)` raises `TypeError` if a `post_size` is `None` (comparing int and None). The branch logic (`if isinstance(result1.pre_size, tuple)` then uses `post_size`) is also odd. Guard `None` and compute from indices.

### BASE-8 — `rng` is the global `np.random` module when `seed is None`  ·  **Medium · Confirmed**
- **Location:** `_base.py:238` (`self.rng = ... if seed is not None else np.random`). Two problems: (1) results then depend on global RNG state (not isolated/reproducible across the process); (2) `np.random` module vs `RandomState` are not 100% API-identical. Prefer `np.random.default_rng(seed)` (a `Generator`) consistently, created even when `seed is None`. (Note: switching to `Generator` changes method names like `random`/`integers` — coordinate across the module.)

### BASE-9 — Minor: `_validate` delays message uses `len()`; `shape` uses Python `max()`  ·  **Low · Confirmed/Minor**
- `_base.py:135` interpolates `len(self.delays)` while the check uses `u.math.size` (line 132). Not practically triggerable into masking (to fail the check the array has `len`), but inconsistent — use `u.math.size`.
- `_base.py:154,161` use Python `max(self.pre_indices)` on arrays (slow; returns numpy scalar) — use `self.pre_indices.max()` and wrap in `int(...)` for consistency with the other branches.

### BASE-10 — `union` metadata priority contradicts the weight priority  ·  **Low · Confirmed**
- `_base.py:596-604`: `merge_dict(result1.metadata, result2.metadata, {...})` lets **result2** overwrite result1 keys, while weights/delays are **result1**-priority (first-occurrence). Make metadata priority consistent (or document).

---

## C. `_random.py`, `_regular.py`, `_biological.py`

### RND-1 — `Random`/`FixedProb` drop valid cross-population connections (`i==j` skipped across different populations)  ·  **High · Confirmed**
- **Location:** `_random.py:175-176`. The `i == j` autapse guard fires even when pre and post are distinct populations. `Random(prob=1.0, allow_self_connections=False)(pre_size=5, post_size=3)` ⇒ 12 connections instead of 15 (drops `(0,0),(1,1),(2,2)`).
- **Fix:** only apply the `i==j` skip when populations are the same (gate on `pre_num == post_num` / an explicit recurrent flag), as `AllToAll` does (`_regular.py:91`). See XC-3.

### RND-2 — Scalar `weight`/`delay` do **not** get default units, contradicting docstrings  ·  **High · Confirmed**
- **Location:** `_random.py:60-77` (docstring) vs runtime; same in `_regular.py:38-43`. Docstring says a scalar "will use nS/ms units", but `param(2.5, ...)` returns a bare dimensionless float. Downstream `.to(u.nS)` then raises `UnitMismatchError`. Tests always pass `2.5 * u.nS`, so the documented plain-float path is never exercised.
- **Fix:** either attach default units to bare scalars, or correct the docstrings (all three classes) to state scalars remain dimensionless.

### RND-3 — `Random.generate` is an O(N²) Python loop  ·  **Medium · Confirmed** — see **XC-7**.

### REG-1 — `OneToOne(circular=True)` divides by zero on empty population  ·  **Medium · Confirmed**
- **Location:** `_regular.py:189-192`. `pre_indices = np.arange(max(...)) % pre_num` with `pre_num == 0` ⇒ `RuntimeWarning: divide by zero` and 5 bogus connections into an empty population.
- **Fix:** short-circuit to an empty result when either size is 0.

### BIO-1 — `ExcitatoryInhibitory` mixed unit/unitless weights ⇒ cryptic `UnitMismatchError`  ·  **Low · Confirmed**
- **Location:** `_biological.py:210-235,253-278`. `exc_weight=2.0, inh_weight=-3.0*u.nS` fails at generate-time with an opaque message. Validate in `__init__` that exc/inh weights (and delays) share dimensionality.

### BIO-2 / REG-2 — No validation of `prob`/`exc_ratio`/`exc_prob`/`inh_prob`/`cluster_factor`  ·  **Medium · Confirmed** — see **XC-5**. Notably `exc_ratio∉[0,1]` ⇒ `negative dimensions` error; `ClusteredRandom(cluster_factor<0)` silently masked by `np.clip`.

---

## D. `_spatial.py`

### SPA-1 — `Exponential`/`DistanceDependent` docstring examples crash (`ExponentialProfile(scale=)`)  ·  **Critical · Confirmed**
- **Location:** `_spatial.py:151,568,589,604,620,636` and Parameters `:493`. Real param is `decay_constant` (`braintools/init/_distance_impl.py:132`). `ExponentialProfile(scale=…)` ⇒ `TypeError`. See **DOC-5**.

### SPA-2 — Autapses in `DistanceDependent`/`Gaussian`/`Exponential` and `RadialPatches`  ·  **High · Confirmed** — see **XC-3**. (`_spatial.py:219-222`, `:1432-1448`.)

### SPA-3 — `Ring` duplicates + self-loops for large `neighbors`; `Grid2d` periodic duplicates on small grids  ·  **High · Confirmed** — see **XC-4**. (`_spatial.py:836-847`, `:1101-1118`.) Docstrings claim "self-connections excluded" / "exactly 8 neighbors" — both false in these regimes.

### SPA-4 — No validation that `len(positions) == size`; cryptic broadcast error  ·  **Medium · Confirmed**
- **Location:** `_spatial.py:219-220` vs `:187-197`. `random_vals` uses declared sizes; `probs`/`distances` use actual position counts ⇒ `operands could not be broadcast (10,10)(5,5)`. Assert `distances.shape == (pre_num, post_num)` with a clear message, or derive counts from positions.

### SPA-5 — `sigma=0` / negative scale silently mishandled  ·  **Medium · Confirmed** — see **XC-5**. (`sigma=0` ⇒ divide-by-zero ⇒ NaN ⇒ silently empty.)

### SPA-6 — Dense `(pre,post)` matrices; `max_distance` gives no compute/memory savings  ·  **Medium · Confirmed** — see **XC-7**. (`_spatial.py:213,219`.) Docstring efficiency claims are wrong.

### SPA-7 — Caching returns stale results when positions change  ·  **Medium · Confirmed** — instance of **XC-2**; especially damaging because positions are the primary spatial input.

### SPA-8 — Misleading "first 2 dimensions" doc/test vs all-dims `cdist`  ·  **Medium · Confirmed**
- **Location:** `_spatial.py:211-213`; test `_spatial_test.py:170-190`. `cdist` uses all columns (3D distance), but the comment/test say "uses first 2 dimensions" and assert nothing. Fix the doc/test; state N-D is used.

### SPA-9 — No probability clipping (`prob>1` saturates silently)  ·  **Low · Confirmed**
- **Location:** `_spatial.py:220`. Benign for the shipped profiles (cap at 1) but custom/composed/amplitude-scaled profiles can exceed 1. Clip or warn.

### SPA-10 — `Grid2d` misleading error for non-tuple equal sizes  ·  **Low · Confirmed**
- **Location:** `_spatial.py:1079-1085`. `Grid2d()(100,100)` raises "require pre_size == post_size" though they *are* equal — real issue is they must be `(rows,cols)` tuples. Fix the message.

---

## E. `_topological.py`

### TOP-1 — `ModularGeneral` crashes on scalar weights/delays (its documented usage)  ·  **Critical · Confirmed**
- **Location:** `_topological.py:1027-1028` (`np.concatenate(all_weights/all_delays)`). Scalar weights from `param` (e.g. `weight=1.0*u.nS`) make the list elements 0-d ⇒ `ValueError: zero-dimensional arrays cannot be concatenated` / `TypeError: Only dimensionless quantities…`. All docstring examples pass `weight=…*u.nS`.
- **Fix:** broadcast each sub-result's weights/delays to length `len(pre_indices)` and use unit-aware `u.math.concatenate`; define a policy for mixed has/has-no weights.

### TOP-2 — `ModularGeneral` reuses cached sub-results → identical blocks  ·  **Critical · Confirmed** (interacts with XC-2)
- **Location:** `_topological.py:938-943,994-999`. A single `inter_conn` instance is called per module pair; base caching returns the first block for all of them ⇒ all inter-module blocks identical (no independence). Also recurs on the 2nd call of the whole object.
- **Fix:** call sub-connectivities with `recompute=True` or invoke `generate(...)` directly.

### TOP-3 — `ScaleFree` out-of-bounds indices when `m ≥ N`  ·  **Critical · Confirmed**
- **Location:** `_topological.py:308-320`. `m0 = max(m,2)` complete graph built without `m0 ≤ n`; `ScaleFree(m=5)(3,3)` ⇒ `pre_indices.max()==4` with `pre_size==3`. `degree[:m0]=…` hides it. Will crash/corrupt downstream array/CSR ops. Validate `m < n`.

### TOP-4 — `SmallWorld` silently degrades odd `k` to `k-1`  ·  **High · Confirmed**
- **Location:** `_topological.py:139-143`. `k_half = k//2` truncates; metadata still reports the requested `k`. Validate `k` even (or adjust metadata).

### TOP-5 — `SmallWorld` self-loops/duplicates when `k ≥ n`, and rewiring is directed + can duplicate  ·  **High · Confirmed**
- **Location:** `_topological.py:139-160`. No `k<n` check ⇒ wraparound self/duplicate edges even at `p=0`. Rewiring doesn't check existing edges (duplicates) and only moves forward endpoints ⇒ output is **directed**, not canonical (undirected) Watts–Strogatz; docstring claims "exactly n·k connections" and "self-connections avoided" — both false.
- **Fix:** validate `k<n` and even; reject duplicate/self during rewiring; decide+document directed vs undirected (symmetrize if undirected).

### TOP-6 — `ModularRandom` collapses to one module when `n_modules > n`  ·  **High · Confirmed**
- **Location:** `_topological.py:660-663`. `module_size = n // n_modules == 0` ⇒ all neurons land in the last module silently. Also lopsided remainder assignment. Validate `n_modules ≤ n`; use balanced sizing.

### TOP-7 — `ModularGeneral` mutates user sub-conn RNG (`conn.rng = self.rng`)  ·  **High · Confirmed**
- **Location:** `_topological.py:928`. Overwrites a caller-supplied object's RNG (destroys its seeding/reproducibility); applied to intra but not inter conns (inconsistent). Thread randomness without mutating inputs.

### TOP-8 — O(N²) Python loops in `ModularRandom`/`HierarchicalRandom`/`CorePeripheryRandom`  ·  **High · Confirmed** — see **XC-7**. `HierarchicalRandom` additionally rescans level boundaries inside the loop.

### TOP-9 — Square-population check rejects equivalent tuple/int sizes  ·  **Medium · Confirmed**
- **Location:** `_topological.py:135-136,304-305,…` (all classes). `pre_size=(10,10)` vs `post_size=100` raises despite equal counts. Compare `int(np.prod(...))` on both sides.

### TOP-10 — Proportional sizing: `int(ratio*n)` truncation and `isinstance(ratio,int)` misclassification  ·  **Medium · Confirmed**
- **Location:** `_topological.py:872,877,1242,1247`. `int(0.58*100)=57` (use `round`); `isinstance(np.int64,int) is False` and `isinstance(True,int) is True` ⇒ wrong branch. Use `numbers.Integral`/`np.integer` and exclude `bool`.

### TOP-11 — `ModularGeneral` assumes `post_positions` present whenever `pre_positions` is  ·  **Medium · Likely**
- **Location:** `_topological.py:931-936,987-992`. Guards on `pre_positions is not None` then indexes `post_positions[...]` ⇒ `TypeError` if only pre is supplied. Guard on both.

### TOP-12 — Docstrings overstate guarantees  ·  **Low · Confirmed** — "exactly n·k connections", "undirected", connection-count formulas don't hold after the above. Reconcile post-fix.

---

## F. `_kernel.py`

### KER-1 — `SobelKernel`/`LaplacianKernel` drop all negative coefficients (`threshold=0.1`)  ·  **Critical · Confirmed**
- **Location:** `_kernel.py:917,932,940,1024` (hardcoded `threshold=0.1`) → `Conv2dKernel.generate:173` (`if kernel_val > self.threshold`). The defining negative lobes (`-1,-2` Sobel; `-4`/`-8` Laplacian center) are discarded ⇒ net-excitatory filters, not edge/Laplacian operators. Verified: Laplacian keeps only `[1.]`, Sobel keeps `[1.,2.]`.
- **Fix:** threshold on magnitude (`abs(kernel_val) > threshold`) for signed kernels; pass `threshold=0` for Sobel/Laplacian.

### KER-2 — 3D (and `(N,1)`) positions crash `Conv2d`/`Gabor`/`Sobel`/`Laplacian`  ·  **High · Confirmed**
- **Location:** `_kernel.py:165` (`rel_x, rel_y = rel_positions[pre_idx]`), `:518`. Unpacking assumes exactly 2 columns ⇒ `ValueError: too many values to unpack` (3D) / `not enough values` (`(N,1)`). `GaussianKernel` survives (uses `cdist`), so behavior is inconsistent. Verified (GaborKernel on `(8,1)` positions).
- **Fix:** index `[:,0]`/`[:,1]` explicitly (and document a single N-D policy), or validate `positions.shape[1]==2`.

### KER-3 — `Conv2dKernel` uses one `kernel_size` for both axes and a square support box  ·  **High · Confirmed**
- **Location:** `_kernel.py:143-144,159-161`. Non-square kernels (e.g. 5×1) get a square physical footprint; pre offset along a zero-extent axis still connects (snapped via `argmin`). Wrong connections/weights for the orientation-selective case the kernels exist for. Scale each axis by its own pitch and bound per-axis (or require square + document).

### KER-4 — Autapses always created for coincident pre/post  ·  **High · Confirmed** — see **XC-3**. (Peak weight on the diagonal.)

### KER-5 — `Gabor`/`Conv2d`/`Custom` are Python loops; module not JAX-able  ·  **Medium · Confirmed** — see **XC-6/XC-7**. Vectorize like `GaussianKernel`/`DoGKernel`.

### KER-6 — `Conv2dKernel` argmin-snapping is coarse/aliased; no stride/padding/boundary modes  ·  **Medium · Likely**
- **Location:** `_kernel.py:168-176`. Many distinct positions collapse onto one kernel cell with identical weight; box test and snapping are independent so effective support is data-dependent. Docstrings imply convolution but no stride/pad/wrap exist. Define explicit semantics (fractional coord + nearest/bilinear) or document the limitation.

### KER-7 — `GaborKernel.frequency` is not unit-converted  ·  **Medium · Confirmed**
- **Location:** `_kernel.py:526`. `sigma`/`max_distance` are converted to position units but bare-float `frequency` multiplies `x_rot` (now in position units) ⇒ wavelength silently changes 1000× if positions are mm vs um. Require `frequency` as a `1/length` Quantity, or document "cycles per position-unit, not converted".

### KER-8 — `GaussianKernel` normalization hardcodes the 2-D constant  ·  **Medium · Likely**
- **Location:** `_kernel.py:327-328`. `/(2πσ²)` is the 2-D constant; wrong for 1-D/3-D (`(2πσ²)^(n/2)`). Compute from `ndim` or document "2-D only".

### KER-9 — Inconsistent threshold/prune semantics across kernels  ·  **Low · Confirmed**
- **Location:** `_kernel.py:173` (`>`), `:1146` (`abs>`), `:707` (`1e-4`), `:529` (`1e-3`), `:335` (`1e-6`). "threshold" means different things per class and silently decides whether negative (inhibitory) connections survive; cutoffs are hardcoded/undocumented. Standardize on magnitude thresholding; expose/document cutoffs.

### KER-10 — `SobelKernel('both')` shares global RNG when `seed is None`; mutates metadata  ·  **Low · Likely**
- **Location:** `_kernel.py:910-950`. Sub-kernels use `seed`/`seed+1`, but with `seed=None` both draw from global `np.random` (not independent). Derive distinct seeds; avoid mutating returned metadata.

### KER-11 — Stale module path in test docstrings (`_conn_kernel`)  ·  **Low · Confirmed**
- **Location:** `_kernel_test.py:58` etc. import `braintools.conn._conn_kernel` (module is `_kernel`). Would `ImportError` if copied. Fix to `braintools.conn`.

---

## G. `_compartment.py`

### CMP-1 — `DendriticIntegration`/`SynapticClustering` generate massive duplicate connections  ·  **Critical · Confirmed**
- **Location:** `_compartment.py:1196-1224` (and `:1591-1639`). `replace=False` dedups only within a cluster; across clusters the same `(pre,post,AXON,dendrite)` quad repeats with no global dedup. Verified: 120 connections, only 47 unique (~60% dup). Doubles/triples synaptic weight after conversion. Dedup the final quads (or sample without cross-cluster replacement). See XC-4.

### CMP-2 — No-position fallbacks ignore `sigma`/`arborization_radius` (silent dense uniform)  ·  **High · Confirmed**
- **Location:** `_compartment.py:608-615` (`MorphologyDistance` → `CompartmentSpecific(connection_prob=0.1)`), `:1406-1413` (`AxonalArborization` → `density`). A user setting `sigma=1nm` still gets ~10% uniform connectivity; the hardcoded `0.1` is arbitrary/undocumented. Raise or warn when positions are required but missing.

### CMP-3 — `AxonalProjection`/`TopographicProjection` hardcode post compartment to `BASAL_DENDRITE`  ·  **High · Confirmed**
- **Location:** `_compartment.py:998-999`. Unlike sibling axon patterns (basal+apical), every synapse is forced to basal, no override. The test asserts the buggy value, locking it in. Randomize basal/apical or expose `target_compartments`.

### CMP-4 — `DendriticTree` mislabels anatomy and ignores parameters  ·  **High · Confirmed**
- **Location:** `_compartment.py:804-831`. Maps `'proximal'→BASAL`, `'distal'→APICAL` (proximal/distal is a *position*, not basal/apical); sets `pre_compartments=AXON`; `tree_structure`/`distance_dependence`/`branch_length` are accepted but **never used**. Implement morphology or document it as a two-probability wrapper; remove dead params.

### CMP-5 — Scalar prob applied per (source,target) compartment pair doubles effective connectivity  ·  **Medium · Confirmed**
- **Location:** `_compartment.py:1100-1118,296-324`. `AXON:[BASAL,APICAL]` with `prob=0.5` ⇒ ~0.75 aggregate per post and two connections per neuron pair. Arguably intended for multi-compartment, but undocumented/surprising; `BranchSpecific` especially affected. Document; optionally offer normalized mode.

### CMP-6 — `AllToAllCompartments` emits same-neuron same-compartment self-loops  ·  **Medium · Confirmed** — see **XC-3**. (SOMA→SOMA on neuron i.)

### CMP-7 — `cdist` crashes on 1-D position arrays  ·  **Medium · Confirmed**
- **Location:** `_compartment.py:658,974,1426`. 1-D topographies (`shape (N,)`) ⇒ `ValueError: XA must be a 2-dimensional array`. Reshape to `(N,1)` or validate.

### CMP-8 — `MorphologyDistance` uses Euclidean soma distance, not compartment morphology; `morphology_positions` dead  ·  **Medium · Confirmed**
- **Location:** `_compartment.py:547-551` (doc) vs `:661-668`. The class premise (compartment-resolved morphological distance) is unimplemented; `morphology_positions` is stored, never read. Implement or correct the docstring + remove the dead param.

### CMP-9 — No probability/positive-int validation in non-`CompartmentSpecific` classes  ·  **Medium · Confirmed** — see **XC-5**. (`density=5.0`, negative `branches_per_axon` → cryptic `poisson` error, etc.)

### CMP-10 — O(N²) topographic loop and dense `(pre,post)` matrices  ·  **Medium · Confirmed** — see **XC-7**. (`_compartment.py:964-967` + dense allocations throughout.)

### CMP-11 — `CustomCompartment` fragile return parsing  ·  **Low · Confirmed**
- **Location:** `_compartment.py:1667-1705`. Dispatches on `len(result_data)`; scalar single-connection returns yield 0-d arrays; non-sized returns ⇒ opaque `TypeError: len() of unsized object`; a `(4,N)` ndarray is accepted by accident. Validate the contract explicitly.

### CMP-12 — `CompartmentSpecific` dict prob silently drops unlisted mapped pairs  ·  **Low · Confirmed**
- **Location:** `_compartment.py:304-310`. `.get((src,tgt), 0.0)` ⇒ a typo'd key yields an empty compartment with no diagnostic. Warn/error on mapping pairs missing from the dict.

### CMP-13 — `weight_distribution`/`weight_params` documented but nonexistent (crashes)  ·  **Low(doc)/High(impact) · Confirmed** — see **DOC-4**. (`_compartment.py:135-138,446-449,503-507`.)

### CMP-14 — `Proximal/DistalTargeting`/`BranchSpecific` encode an unconfigurable, oversimplified anatomy  ·  **Low · Likely**
- **Location:** `_compartment.py:1032-1045,1100-1107`. Hard-wired `proximal=basal`, `distal=apical`, branch index `<2→basal`. Defensible defaults but presented as morphologically grounded with no override. Document/allow override.

> **Note:** compartment indices in this file are always 0–3, comfortably under the base `max_comp=10` composition assumption — so there is **no** interaction bug with `CompositeConnectivity` here (checked OK).

---

## H. Documentation

### DOC-1 — Module docstring `AxonToDendrite(... weight_distribution=, weight_params=)` crashes  ·  **Critical · Confirmed**
- **Location:** `__init__.py:57-63`. No such params ⇒ they fall to `Connectivity.__init__` ⇒ `TypeError: unexpected keyword argument 'weight_distribution'`. Use `weight=LogNormal(mean=2.0*u.nS, std=0.5*u.nS)`.

### DOC-2 — Module docstring `DistanceDependent(sigma=, decay=, max_prob=)` crashes  ·  **Critical · Confirmed**
- **Location:** `__init__.py:83-88`. Real signature is `DistanceDependent(distance_profile, weight=None, delay=None)`. ⇒ `TypeError: missing required positional 'distance_profile'`. Use `DistanceDependent(GaussianProfile(sigma=100*u.um, max_distance=300*u.um), …)`.

### DOC-3 — `LogNormal(..., sigma=)` crashes (param is `std`)  ·  **Critical · Confirmed**
- **Location:** `__init__.py:61,77` (and `_spatial.py:639`). `LogNormal`/`Normal` take `std`. Systemic — fixing `sigma→std` everywhere clears several crashes.

### DOC-4 — `CompartmentSpecific`/`AxonToDendrite`/`DendriteToDendrite` document non-existent `weight_distribution`/`weight_params`  ·  **Critical · Confirmed**
- **Location:** `_compartment.py:135-138,149-154,503-508,530-534`. Examples crash. Rewrite to `weight=`/`delay=` Initializers.

### DOC-5 — `ExponentialProfile(scale=)` everywhere should be `decay_constant=`  ·  **Critical · Confirmed**
- **Location:** `_spatial.py:150,493,567,588,603,619,635`. Systemic — see SPA-1.

### DOC-6 — `Gaussian`/`Exponential` docstrings misuse `DistanceModulated(lambda d: …)`  ·  **High · Confirmed**
- **Location:** `_spatial.py:411-418,615-624`. Real signature `DistanceModulated(base_dist, distance_profile)` — a bare lambda is neither. ⇒ `TypeError`. Use `DistanceModulated(base_dist=Normal(...), distance_profile=GaussianProfile(...))`.

### DOC-7 — Caching gotcha is undocumented  ·  **Medium · Confirmed** — see **XC-2**. Add a "Caching / `recompute=`" note to `Connectivity` docstrings and `conn.rst`.

### DOC-8 — Notebook `02_spatial_connectivity.ipynb` stale `n_neurons` corrupts degree stats  ·  **Medium · Confirmed**
- In-degree cell uses `n_neurons=300` against 200-neuron Gaussian/Exponential results (`np.bincount(minlength=300)`), inflating CV (~0.79 vs ~0.30) and adding phantom degree-0 neurons. Size each array to its own population.

### DOC-9 — Minor doc hygiene  ·  **Low · Confirmed**
- Several spatial snippets omit `from braintools.init import LogNormal, Normal`. Notebook `02` has an unused `Uniform` import and stale saved outputs (RNG/version drift). `ClusteredRandom` docstring places a `Returns` section on the class and has unverified `See Also`. **Positive:** `docs/apis/conn.rst` is complete/accurate (all 59 `__all__` names documented, no dead/duplicate entries; `Regular`/`Random` names are unambiguous), and `01_basic_connectivity.ipynb` runs clean.

---

## I. Test-suite assessment — false confidence

Across every file the tests are **shape/metadata/`n_connections>0`**-oriented and would **not** catch the bugs above:
- None assert `weight2dense() == weight2csr().todense()` (XC-1 invisible).
- `Random` asymmetric-size tests assert only index bounds, not counts (RND-1 invisible); scalar tests always pass `*u.nS` (RND-2 invisible).
- `ModularGeneral` tests always pass weightless `Random` with distinct per-slot seeds (TOP-1 and TOP-2 invisible).
- `ScaleFree`/`SmallWorld`/`ModularRandom` never tested with `m≥N` / odd-or-`k≥N` / `n_modules>N` (TOP-3/4/5/6 invisible).
- Sobel/Laplacian tests check the *kernel array* but never the resulting *weights* (KER-1 invisible); `test_3d_positions` only runs `GaussianKernel` (KER-2 invisible); `test_single_neuron` treats the autapse as expected (XC-3 invisible).
- Compartment tests assert membership/`>0` only; `test_axonal_projection_local` asserts the buggy BASAL-only behavior (CMP-3); `test_*_no_positions` asserts `>0`, hiding CMP-2; `test_empty_connections_*` resets `pattern.seed` *after* `__init__` (no effect on `self.rng`) — latent false confidence.
- Spatial tests use **Mock** profiles almost everywhere, so real `GaussianProfile`/`ExponentialProfile` paths (and SPA-1/2/3/5) are never exercised.
- Large-N perf tests cap at 500×500 or are `@skip("too slow")` (XC-7 hidden).

**Recommendation:** add **assertion-rich** regression tests for each Critical/High finding (exact counts, no duplicates, no autapses, correct weights on known inputs, structural properties like degree/symmetry/module membership), and a property test `weight2dense == weight2csr.todense()` over randomized unsorted results.

---

## J. Prioritized remediation roadmap

**P0 — silent data corruption / crashes on common or documented usage**
1. XC-1/BASE-1 — fix `weight2csr`/`delay2csr` ordering (+ property test).
2. XC-2/BASE-2 — fix/replace `__call__` caching; make internal recallers use `recompute`/`generate`.
3. TOP-1, TOP-2, TOP-3 — `ModularGeneral` concat + cache reuse; `ScaleFree` bounds.
4. KER-1 — Sobel/Laplacian thresholding.
5. CMP-1 — duplicate connections in `DendriticIntegration`/`SynapticClustering`.
6. DOC-1…6 — fix crashing docstrings (`sigma→std`, `scale→decay_constant`, `weight_distribution→weight`, `DistanceModulated`, module-docstring examples).
7. BASE-3 — `ScaledConnectivity` in-place mutation.

**P1 — wrong results on plausible inputs / important edges**
8. XC-3 — unify autapse policy (`allow_self_connections`) across spatial/kernel/compartment; fix RND-1.
9. XC-4 — dedup `Ring`/`Grid2d`/`SmallWorld`/compartment; define `ConnectionResult` duplicate policy (BASE-4/BASE-6).
10. XC-5 — add parameter validation everywhere (TOP-4/5/6, SPA-5, BIO-2, CMP-9, …).
11. KER-2, KER-3 — kernel dimensionality + non-square support.
12. CMP-2, CMP-3, CMP-4 — fallback semantics, BASAL hardcode, `DendriticTree` anatomy/dead params.
13. RND-2 — scalar unit policy (decide: attach units vs document).

**P2 — performance, JAX clarity, polish**
14. XC-7 — vectorize `Random` + topological/kernel/compartment loops; KDTree for spatial; fix `max_distance` claims.
15. XC-6 — document NumPy/host-only (non-`jit`/`vmap`) nature; or scope a JAX path.
16. BASE-8 — migrate to `np.random.default_rng`/`Generator`; isolate per-instance RNG.
17. Remaining Low/Docs: XC-8 signatures, TOP-9/BASE-5 tuple-vs-int sizes, KER-7/8/9, CMP-5/8/11/12/14, DOC-7/8/9.

---

*Generated from a 1 (base, self) + 6 (parallel subagent) review of `braintools/conn/`. Highest-severity items were reproduced at runtime; line numbers reference the files as of 2026-06-18.*
