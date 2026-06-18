# braintools.visualize — Issues Found (2026-06-19)

Audit of `braintools/visualize/` and its documentation, performed as a senior
Python/JAX reviewer. Each issue below was reproduced before being recorded.
Severity legend: **High** (crash / wrong result in documented usage),
**Medium** (crash / wrong result on a reasonable edge case), **Low** (quality /
documentation).

Environment: numpy 2.4.6, matplotlib 3.10.8, scipy 1.17.1, plotly absent.

---

## 1. `line_plot` reshapes before converting to an array — crashes on list input
**File:** `_plots.py` · **Severity:** High

```python
val_matrix = val_matrix.reshape((val_matrix.shape[0], -1))  # line 138
val_matrix = np.asarray(val_matrix)                          # line 140
```

`reshape` is called on the raw `val_matrix` *before* it is converted with
`np.asarray`. The empty-check on line 121 (`len(val_matrix) == 0`) accepts Python
lists, so a list reaches line 138 and raises
`AttributeError: 'list' object has no attribute 'reshape'`. The same ordering
also strips `brainunit` units silently for Quantity input.

**Repro**
```python
line_plot(list(range(5)), [[1],[2],[3],[4],[5]], plot_ids=[0])
# AttributeError: 'list' object has no attribute 'reshape'
```

**Proposed fix:** Convert with `np.asarray` (stacks lists into an ndarray and
triggers `__array__` for JAX / brainunit) *before* reshaping. Note: `as_numpy`
is unsuitable here because it tree-maps and would leave a nested list a list.

---

## 2. `spike_raster` does not filter a per-spike `color` array — crashes with `time_range`/`neuron_range`
**File:** `_neural.py` · **Severity:** High

When `color` is an array of per-spike colors and a `time_range` or `neuron_range`
filter is applied, `spike_times`/`neuron_ids` are masked but `color` is not, so
`ax.scatter(..., c=color)` raises a length-mismatch error.

**Repro**
```python
st = np.linspace(0, 100, 50); nid = np.arange(50) % 10
spike_raster(st, nid, color=np.random.rand(50), time_range=(0, 50))
# ValueError: 'c' argument has 50 elements, inconsistent with x/y of size 25
```

**Proposed fix:** When `color` is array-like with length equal to the number of
spikes, apply the same mask(s) used on `spike_times`/`neuron_ids`.

---

## 3. `animate_1D` mutates the caller's input dictionaries
**File:** `_plots.py` · **Severity:** Medium

For dict / list-of-dict input, the function writes `var['legend']`, `var['xs']`
and overwrites `var['ys']` on the *caller's* objects, e.g.:

```python
d = {'ys': arr}
animate_1D(d, show=False)
# d now also contains 'xs' and 'legend'
```

**Proposed fix:** Build fresh dictionaries rather than mutating inputs.

---

## 4. `confusion_matrix` normalization divides by zero → NaN
**File:** `_statistical.py` · **Severity:** Medium

```python
cm = cm / cm.sum(axis=0, keepdims=True)   # 'pred'
```

When a class never appears in `y_pred` (column sum 0) — or `y_true` for `'true'` —
the division yields `NaN` and emits a RuntimeWarning, producing blank/incorrect
cells.

**Repro**
```python
confusion_matrix([0,1,2,2,1,0], [0,1,1,1,1,0], normalize='pred')  # class 2 never predicted
# RuntimeWarning: invalid value encountered in divide
```

**Proposed fix:** Divide with a guarded denominator (treat 0 as 0 instead of NaN).

---

## 5. `regression_plot` R² divides by zero for constant `y`
**File:** `_statistical.py` · **Severity:** Medium

```python
ss_tot = np.sum((y - np.mean(y)) ** 2)   # == 0 when y is constant
r_squared = 1 - (ss_res / ss_tot)        # 1 - 0/0 -> NaN + warning
```

**Proposed fix:** Guard `ss_tot == 0` (define R² as 1.0 when the fit is perfect,
else 0.0) and avoid the warning.

---

## 6. `neural_trajectory` raises a cryptic `IndexError` for < 2 features
**File:** `_neural.py` · **Severity:** Medium

With `data` of shape `(T, 1)`, `dims` defaults to `(0, 1)` and `data[:, 1]` raises
`IndexError: index 1 is out of bounds`.

**Proposed fix:** Validate that the data has at least 2 columns (or that the
requested `dims` are in range) and raise a clear `ValueError`.

---

## 7. `correlation_matrix` / `interactive_correlation_matrix` crash on a single feature
**File:** `_statistical.py`, `_interactive.py` · **Severity:** Medium

`np.corrcoef(data.T)` on a single-column input returns a 0-d scalar, so
`imshow` raises `TypeError: Invalid shape () for image data`.

**Proposed fix:** Coerce the correlation result to at least a 2-D array; a single
feature yields the trivial `[[1.0]]` matrix.

---

## 8. `animate_2D` gives a cryptic error for non-2D input / size mismatch
**File:** `_plots.py` · **Severity:** Low/Medium

`num_step, num_neuron = values.shape` raises *"too many values to unpack"* when
`values` is already `(T, H, W)`, and a mismatched `net_size` produces an opaque
reshape error.

**Proposed fix:** Accept either `(T, N)` or `(T, H, W)` input and validate
`N == H*W` with a clear `ValueError`.

---

## 9. `roc_curve` / `precision_recall_curve` AUC not anchored at the curve endpoints
**File:** `_statistical.py` · **Severity:** Low

The AUC/AP trapezoid integrates only over the observed thresholds; the ROC curve
is not anchored at `(0,0)`/`(1,1)`, so when several samples tie at the top score
the leading area is dropped and AUC is underestimated.

**Proposed fix:** Prepend `(fpr=0, tpr=0)` and append `(fpr=1, tpr=1)` for ROC
before integrating; keep the result clamped to `[0, 1]`.

---

## 10. `brain_surface_3d` mis-colors signed data (normalization bug)
**File:** `_three_d.py` · **Severity:** Low/Medium

```python
scaled = face_values / max_val      # only divides by max
```

For signed `values`, negatives map to `< 0` (outside the colormap's `[0, 1]`
domain) and get clamped, so the surface coloring is wrong.

**Proposed fix:** Use min–max normalization `(v - vmin) / (vmax - vmin)` with a
guard for a degenerate (constant) range.

---

## 11. `volume_rendering` ignores its documented `cmap` parameter
**File:** `_three_d.py` · **Severity:** Low

`cmap` is documented and accepted but never applied — `ax.voxels` is called
without face colors, so the rendering is always the matplotlib default.

**Proposed fix:** Map the thresholded volume intensities through `cmap` and pass
the resulting `facecolors` to `voxels`.

---

## 12. Documentation: `fig, ax = neural_network_3d(...)` cannot be unpacked
**File:** `visualize/__init__.py` (module docstring) · **Severity:** Low

`neural_network_3d` returns a single `Axes3D`, but the "Quick Start - Neural
Network 3D" example unpacks it as `fig, ax = neural_network_3d(...)`, which raises
`TypeError: cannot unpack non-iterable Axes3D object` if copy-pasted.

**Proposed fix:** Change the example to `ax = neural_network_3d(...)`.

---

## Non-issues / verified correct
- `as_numpy` uses `jax.tree.map`, so Python lists remain lists — the
  list-of-arrays branches in `spike_raster`, `spike_histogram`, `isi_distribution`,
  `interactive_spike_raster`, and `dashboard_neural_activity` work as intended.
- `_trapezoid = getattr(np, 'trapezoid', None) or np.trapz` correctly handles the
  NumPy 2.x rename/removal of `np.trapz`.
- `apply_style` context-manager rcParams restore and `create_neural_colormap`
  `force=True` re-registration are correct.
- `firing_rate_map` rectangular-grid binning is correct (rows index y, cols x).
- `connectivity_matrix` with all-zero weights handles the degenerate `Normalize`.

## Test plan
A dedicated regression module (`_audit_20260619_test.py`) adds one test per issue
above, run under the Agg backend, alongside the existing suite (231 passed /
31 plotly-skipped at baseline).
