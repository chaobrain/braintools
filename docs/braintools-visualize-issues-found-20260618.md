# `braintools.visualize` — Audit of Issues, Bugs, and Edge Cases (2026-06-18)

**Reviewer role:** senior Python architect / JAX expert / BrainX developer
**Scope:** `braintools/visualize/` source modules and their documentation
(`__init__.py`, `_animation.py`, `_colormaps.py`, `_figures.py`, `_interactive.py`,
`_neural.py`, `_plots.py`, `_statistical.py`, `_three_d.py`, `docs/apis/visualize.rst`).

Every issue below was **reproduced empirically** (matplotlib 3.10.8, numpy 2.4.6,
scipy 1.17.0, brainstate 0.5.0, plotly 6.8.0) before being recorded. Each entry
lists the symptom, the root cause, and the proposed fix.

---

## A. Correctness bugs (wrong output or hard crash)

### A1. `line_plot` writes labels/limits/title to the *wrong* Axes
*File:* `_plots.py`
*Symptom:* When an explicit `ax=` is passed, the data is drawn on `ax` (via
`ax.plot`), but `xlim`, `ylim`, `xlabel`, `ylabel`, and `title` are applied through
the **pyplot state machine** (`plt.xlim(...)`, `plt.xlabel(...)`, …), which targets
the *current* Axes — usually a different subplot. In a multi-panel figure the
labels land on the last-created subplot.

```python
fig, (a1, a2) = plt.subplots(1, 2)
line_plot(t, vals, ax=a1, xlabel='TT')
# a1.get_xlabel() == ''   (wrong)
# a2.get_xlabel() == 'TT' (leaked onto the current axes)
```
*Root cause:* mixing `ax.*` drawing with `plt.*` formatting.
*Fix:* resolve `ax = plt.gca()` when `ax is None`, then use `ax.set_xlim/`
`set_ylim/set_xlabel/set_ylabel/set_title/legend` exclusively.

### A2. `raster_plot` — same wrong-Axes bug
*File:* `_plots.py*` — identical root cause and fix as A1.

### A3. `animate_1D(static_vars=...)` raises `KeyError: 'ys'` at render time
*File:* `_plots.py`
*Symptom:* Passing any `static_vars` (an ndarray, a list of ndarrays, or a
`{'data': ...}` dict) crashes — either in the auto-`ylim` loop
(`var['ys'].max()`) or inside the `frame()` closure (`plt.plot(svar['xs'],
svar['ys'])`).
*Root cause:* static variables are stored inconsistently: the list/ndarray
branches use the key `'data'` (and sometimes omit `'xs'`), while the single-dict
branch uses `'ys'`; the consumers always read `'ys'`/`'xs'`. The existing test
suite *works around* this by always passing an explicit `ylim` and never
rendering the frames (`show=False`, no `save_path`), so the broken path was never
executed.
*Fix:* normalize **every** static var to `{'xs', 'ys', 'legend'}` (accepting
`'data'` or `'ys'` as input, deriving `'xs'` when absent) so both auto-`ylim` and
`frame()` work for all input forms.

### A4. `firing_rate_map` — `IndexError` for non-square `grid_size`
*File:* `_neural.py`
*Symptom:* `firing_rate_map(rates_1d, positions=pos, grid_size=(10, 20))` →
`IndexError: index 16 is out of bounds for axis 0 with size 10`.
*Root cause:* `rate_map`/`count_map` are allocated as `np.zeros(grid_size)` →
shape `(grid_size[0], grid_size[1])`, but written as `rate_map[yi, xi]` where the
bounds check is `0 <= xi < grid_size[0] and 0 <= yi < grid_size[1]`. The row/col
axes are transposed relative to the bounds check, so a non-square grid indexes out
of range (and silently mis-bins for any rectangular grid that doesn't crash).
*Fix:* allocate `(ny, nx)` with `nx, ny = grid_size`, build `x_edges` with `nx`
bins and `y_edges` with `ny` bins, and index `[yi, xi]` consistently.

### A5. `correlation_matrix(method='spearman')` crashes for exactly 2 features
*File:* `_statistical.py` (and `_interactive.py: interactive_correlation_matrix`)
*Symptom:* `correlation_matrix(data_Nx2, method='spearman')` →
`TypeError: Invalid shape () for image data`.
*Root cause:* `scipy.stats.spearmanr(data)[0]` returns a **scalar** when `data` has
exactly two columns (it returns the single pairwise ρ rather than a 2×2 matrix);
`imshow` then receives a 0-d array.
*Fix:* special-case `n_features == 2` and assemble the 2×2 matrix explicitly
(`[[1, ρ], [ρ, 1]]`) in both functions.

### A6. `brain_surface_3d` uses removed-in-3.11 `plt.cm.get_cmap`
*File:* `_three_d.py`
*Symptom:* `MatplotlibDeprecationWarning: The get_cmap function was deprecated in
Matplotlib 3.7 and will be removed in 3.11`. The call becomes a hard `AttributeError`
on matplotlib ≥ 3.11.
*Root cause:* `plt.cm.get_cmap(cmap)(...)`.
*Fix:* use `matplotlib.colormaps[cmap]` (fall back to `plt.get_cmap`), and guard
against divide-by-zero when `np.max(values) == 0`.

---

## B. Robustness / usability bugs

### B1. `animate_2D` / `animate_1D` crash with opaque `KeyError: 'dt'`
*File:* `_plots.py`
*Symptom:* Calling either function with the default `dt=None` while **no**
`brainstate.environ` `dt` is set raises
`KeyError: "Key 'dt' not found in environment."` — surprising for a plain plotting
helper and the reason the module docstring's own animation examples fail.
*Root cause:* `dt = brainstate.environ.get_dt() if dt is None else dt` hard-requires
a global simulation context.
*Fix:* keep honoring an active `environ` dt, but fall back to `dt = 1.0` (time axis
expressed in steps) when none is available, and document this.

### B2. `remove_axis(ax)` (no spine args) is a silent no-op
*File:* `_plots.py`
*Symptom:* The module docstring shows `remove_axis(ax3)` to "remove axis from
decorative panels", but with no positional spine names the `for p in pos:` loop
never executes, so nothing happens. The validation also does a bare
`raise ValueError` with no message.
*Fix:* when called with no spine names, hide all four spines **and** the ticks
(blank the panel — the documented intent); keep per-spine hiding when names are
given; add an informative error message and a docstring.

### B3. `tuning_curve` uses bare `except:`
*File:* `_neural.py` (curve-fit blocks)
*Symptom:* `except:` swallows everything, including `KeyboardInterrupt` /
`SystemExit`.
*Fix:* narrow to `except Exception:`.

### B4. `apply_style` cannot be used as a context manager
*File:* `_colormaps.py`
*Symptom:* The module docstring uses `with apply_style('publication'): ...` (in
two places), but `apply_style` returns `None` →
`TypeError: 'NoneType' object does not support the context manager protocol`.
*Fix:* validate the style name first (so unknown styles still raise `ValueError`),
apply the style immediately (plain calls keep working), and return a small
context-manager object that restores the previous rcParams on `__exit__` —
enabling temporary styling exactly as documented.

---

## C. Documentation bugs — module docstring examples do not run

Every example in the `__init__.py` module docstring below was executed and **raised**
(or silently produced wrong output). These mislead users copying the "Quick Start"
snippets. All are corrected to match the real signatures.

| # | Broken call (in `__init__.py`) | Problem | Correct usage |
|---|--------------------------------|---------|---------------|
| C1 | `population_activity(..., smoothing_window=10)` | no such kwarg | `window_size=10` |
| C2 | `connectivity_matrix(..., colormap='viridis')` | no such kwarg | `cmap='viridis'` |
| C3 | `phase_portrait(v, w, dv, dw, nullclines=True)` | passes λ functions + nonexistent `nullclines` | pass trajectory arrays / `vector_field=True` with `dx,dy` arrays |
| C4 | `roc_curve(..., label='Model')` | duplicate `label` kwarg → `TypeError` | drop `label` |
| C5 | `regression_plot(..., order=1, confidence=0.95)` | wrong kwargs | `fit_line=True, confidence_interval=True` |
| C6 | `raster_plot(spike_times_list, ...)` | needs `ts` + `sp_matrix` (spike matrix) | build a `(time, neuron)` spike matrix |
| C7 | `animate_2D(data_2d, interval=40, vmin=0, vmax=1)` | missing required `net_size`; wrong kwargs | `animate_2D(vals, net_size=(h,w), frame_delay=40, val_min=0, val_max=1, show=False)` |
| C8 | `animate_1D(data_1d, interval=30, ...)` | `interval` → `frame_delay` | `frame_delay=30` |
| C9 | `connectivity_3d(positions, connections, node_size=100)` | needs `source_positions, target_positions, connections`; `node_size`→`node_sizes` | pass both position sets |
| C10 | `trajectory_3d(..., color_by_time=True, tube_radius=0.05)` | `color_by_time`→`time_colors`; no `tube_radius` | `time_colors=True` |
| C11 | `volume_rendering(..., opacity=0.3)` | `opacity`→`alpha` | `alpha=0.3` |
| C12 | `electrode_array_3d(..., electrode_size=50)` | no such kwarg | drop it (fixed marker size) |
| C13 | `dendrite_tree_3d(tree_coords, tree_connections, radius=0.02)` | expects `segments` list of `(start,end)`; no `radius` | pass `[(p0,p1), ...]` |
| C14 | `phase_space_3d(x, y, z, vector_field=True)` | no `vector_field`; x,y,z are a trajectory | drop `vector_field` |
| C15 | `set_default_colors(primary=..., secondary=...)` | takes one `color_dict` | `set_default_colors({'excitatory': ...})` |
| C16 | `interactive_network(adjacency, layout='force')` | no `layout` param | drop `layout` |
| C17 | `interactive_3d_scatter(points, colors=colors)` | needs `x,y,z`; kwarg is `color` | `interactive_3d_scatter(x, y, z, color=...)` |
| C18 | `interactive_surface(X, Y, Z)` | signature is `(z, x=None, y=None)` | `interactive_surface(Z, x=x, y=y)` |
| C19 | `dashboard_neural_activity(spike_2d_matrix)` | expects `spike_times`/`neuron_ids` | pass flat spike times + ids |
| C20 | `neural_trajectory(..., color_by_time=True)` | param is `time_color` | `time_color=True` |
| C21 | `spike_histogram(spike_counts, bin_width=1.0)` | param is `bin_size`; counts aren't times | `spike_histogram(spike_times, bin_size=10.0)` |
| C22 | `distribution_plot(..., kde=True, rug=True)` | no `kde`/`rug` kwargs | `plot_type='both'` |
| C23 | `qq_plot(..., distribution='normal')` | only `'norm'`/`'uniform'`/`'expon'` | `distribution='norm'` |
| C24 | `publication_style(font_family=, font_size=, figure_dpi=)` | wrong kwargs | `fontsize=, figsize=, dpi=` |
| C25 | `dark_style(background=...)` | param is `background_color` | `background_color=...` |
| C26 | `population_activity(rates[np.newaxis, :], time, ...)` (Complete Example) | `(1, N)` vs `N`-length time → broadcast error | pass the 1D `rates` directly |

---

## D. Minor / cosmetic / docstring inaccuracies

* **D1** `_animation.py: animator` docstring claims `:rtype: FuncAnimation` but
  returns `ArtistAnimation`; contains stale references (`bm.random.rand`,
  `splt.animator`, `torch.Tensor`, "celluloid"). The `num_steps=False` sentinel
  means a legitimate `num_steps=0` is silently treated as "all steps". → docstring
  refresh + clarify the sentinel.
* **D2** `_plots.py: animate_1D` docstring documents a non-existent `xticks`
  parameter. → remove from docstring.
* **D3** `_neural.py: neural_trajectory` adds a time colorbar for the 2-D path but
  not for the 3-D path (inconsistent). → add colorbar to 3-D branch.
* **D4** `_statistical.py`: unused loop variable `i` in `distribution_plot`; unused
  `n_subset` in `scatter_matrix`. → clean up.
* **D5** `_interactive.py`: `@set_module_as(...)` is applied to the private helper
  `_check_plotly`. Harmless, but unnecessary. → leave (no behavior impact), noted
  for completeness.

---

## E. Notes on coverage measurement

`pyproject.toml` previously omitted both `_interactive.py` (needs `plotly`) **and**
`_three_d.py` from the project coverage source. However, `_three_d.py` only uses
matplotlib's `mpl_toolkits.mplot3d` — a core dependency — so the "requires pyvista"
omit comment was inaccurate. This audit:

* adds comprehensive tests for the previously thin/omitted `_three_d.py` and
  `_interactive.py` (`_three_d_extra_test.py`, `_interactive_extra_test.py`),
* adds `_audit_fixes_test.py` with a regression test for every bug fixed above,
* **un-omits `_three_d.py`** from the coverage source (it now runs in CI with
  matplotlib alone), keeping only `_interactive.py` omitted (genuinely needs
  `plotly`).

**Result (measured with all files, plotly installed locally):** 99 % statement
coverage — `__init__`, `_animation`, `_colormaps`, `_figures`, `_neural`,
`_plots`, `_three_d` at 100 %; `_statistical` 99 %; `_interactive` 98 %. The only
uncovered lines are defensive `except ImportError` handlers that require an
optional dependency (`scipy`/`plotly`) to be *absent*.

Coverage must be run with the project's `core = "sysmon"` setting (the C tracer
crashes on the numpy 2.x re-import during `braintools` import):
`pytest braintools/visualize/ --cov --cov-report=term-missing`.
