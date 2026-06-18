# `braintools.metric` — Issues Found & Proposed Solutions

**Date:** 2026-06-18
**Auditor role:** Senior Python architect · JAX expert · computational-neuroscience (BrainX) developer
**Scope:** `braintools/metric/` (source + tests) and its documentation (`docs/apis/metric.rst`, `docs/metric/*.ipynb`, `docs/metric/index.md`)
**Method:** **Static analysis only** (no code execution). Headline Critical/High findings were re-verified by direct source reading.
**Reference commit:** `08398d5` (the `metric/` tree and its docs are byte-identical to `origin/main` `cdd8f94`, verified via `git diff`; line numbers below are valid against both).
**Environment present (for reference, not used to run code):** jax 0.10.1, brainstate 0.5.0, brainunit 0.5.1.

---

## 1. Executive summary

The `braintools.metric` package is broad and ambitious (classification, regression, ranking, Fenchel-Young, spike-train, correlation, and LFP metrics). The ML-loss core (classification, regression, ranking, Fenchel-Young) is mostly correct and reasonably tested. The **neuroscience-specific modules (`_lfp`, `_firings`, `_correlation`) and the package/notebook documentation contain numerous correctness, units, and JAX-compatibility defects**, several of which are shipped *and documented as correct*.

Most consequential problems:

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| A17 | **Critical** | `nll_loss` returns the log-prob, not its negative — **sign error** in a shipped loss; the notebook documents the wrong value. | `_classification.py:1073,1077` |
| D11 | **Critical** | `coherence_analysis` computes single-segment magnitude-squared coherence ⇒ **≡ 1 for every input** (scientifically useless). | `_lfp.py:365–374` |
| D3 | **Critical** | The entire **LFP Quick-Start in the package docstring is non-runnable** — wrong kwargs (`fs=`, `phase_freq=`, `spacing=`), wrong arg types, `spike_type='excitatory'` (only `'exc'`/`'inh'` accepted). | `__init__.py:203–253` vs `_lfp.py` |
| DOC-1 | **High** | `jnp.random.*` (which **does not exist** in JAX) used in ~15 docstring/example sites ⇒ every such example raises `AttributeError`. | 5 files (table §8.1) |
| C2 | **High** | `firing_rate` returns Hz only when `width` is a `Quantity`; the **float path silently returns non-Hz values**; kernel normalized by `width` instead of `width1·dt`. | `_firings.py:229–232` |
| D1/D2 | **High** | `power_spectral_density` is a single-window periodogram (not Welch as documented); `noverlap` dead; missing one-sided ×2 and window-power normalization. | `_lfp.py:308–314` |
| B9/B10 | **High** | `l1_loss` default is `'sum'` (docstring & `L1Loss` say `'mean'`) and it computes a per-row **L1 sum, not MAE** despite the name/docstring. | `_regression.py:427,475–477` |
| A8 | **High** | `kl_divergence` has a single-`where` **NaN-gradient** at `targets==0`. | `_classification.py:561` |
| B15/B16 | **High** | Two different `cosine_similarity` functions collide on one public name; the pairwise one is dead (imported as `_cosine_similarity_pairwise`, never exported); file `_pariwise.py` is **misspelled** and has **no test**. | `_pariwise.py`, `__init__.py:399–401` |

Cross-cutting themes: (1) widespread `jnp.random.*` in docs; (2) inconsistent and frequently **broken unit handling** in the neuroscience modules despite the BrainX/`brainunit` ecosystem; (3) many functions are **silently not `jit`/`vmap`/`grad`-able** (Python loops, `len()`, `int()`, boolean-mask indexing, `if` on traced values) with no documentation of that limitation; (4) many tests are **print-only or assert only loose bounds**, so several of the bugs below pass CI.

**Counts (≈):** Critical 4 · High 18 · Medium 31 · Low 27 · Enhancement 22.

Finding IDs use the per-module prefixes used during the audit: **A** = classification, **B** = regression/pairwise/smoothing, **C** = firings, **D** = LFP, **E** = correlation, **F** = ranking/Fenchel-Young, **CC** = cross-cutting, **DOC** = documentation.

---

## 2. Audit scope & method

- **Source:** `_classification.py`, `_regression.py`, `_ranking.py`, `_fenchel_young.py`, `_firings.py`, `_lfp.py`, `_correlation.py`, `_pariwise.py`, `_smoothing.py`, `_util.py`, `__init__.py`.
- **Tests:** all `*_test.py` present (note: **`_pariwise.py` has no test file**).
- **Docs:** `docs/apis/metric.rst`, `docs/metric/index.md`, and notebooks `01`–`07`.
- **Static-only:** no execution. JAX-API correctness (e.g. `jnp.random` does not exist; `jax.random` requires a `PRNGKey`), `brainunit` dimensional algebra, and autodiff/`jit`/`vmap` tracing semantics were reasoned about by reading. Items that cannot be fully decided statically are marked **SUSPECTED**.

---

## 3. Critical issues

### A17 — `nll_loss` returns the log-probability, not the *negative* log-likelihood (sign bug)
**File:** `_classification.py:993,1073,1077` · **Category:** Correctness

The docstring defines `ℓ(x, y) = −x_y` (line 1004), but the body returns the gathered value **without negation**:

```python
loss = input[jnp.arange(len(target)), target]   # line 1073
return loss                                       # batch case  -> NOT negated
...
return input[target]                              # line 1077, scalar case -> NOT negated
```

For the function's own example (`log_probs = jnp.log([0.1,0.7,0.2])`, `target=1`), the correct NLL is `−log(0.7) = +0.3567`, but the function returns `log(0.7) = −0.3567`. NLL must be non-negative; this is a true sign error. Notebook `01_classification_losses.ipynb` (cell `9ee09e0814b9fb0f`) shows `-0.35667497` as the expected output, so the bug is *documented as correct*. There is **no test** for `nll_loss`, which is why it shipped (see TEST-A).

**Fix:** negate both returns: `loss = -input[jnp.arange(len(target)), target]` and `return -input[target]`. Re-run the notebook cell (becomes `0.35667497`). Add a regression test asserting equality with `softmax_cross_entropy_with_integer_labels` on the matching logits.

### D11 — `coherence_analysis` is identically 1 for all inputs
**File:** `_lfp.py:354–379` · **Category:** Correctness

Coherence is computed from a **single** FFT segment:

```python
fft1 = jnp.fft.fft(lfp1[:nperseg] * window, n=n_fft)[:n_fft // 2]
fft2 = jnp.fft.fft(lfp2[:nperseg] * window, n=n_fft)[:n_fft // 2]
coherence = jnp.abs(fft1 * jnp.conj(fft2)) ** 2 / (jnp.abs(fft1)**2 * jnp.abs(fft2)**2 + 1e-12)
```

For a single segment, `|F₁·conj(F₂)|² = |F₁|²·|F₂|²` exactly, so `coherence ≡ 1` at every frequency regardless of the signals. Magnitude-squared coherence is only meaningful when the cross/auto spectra are **averaged over multiple segments**; without averaging it is degenerate. The `clip(0,1)` masks the symptom. This makes the function unusable for its stated purpose.

**Fix:** implement Welch-style segmentation: split into ≥2 overlapping windowed segments, accumulate `⟨P₁₂⟩`, `⟨P₁₁⟩`, `⟨P₂₂⟩`, then `|⟨P₁₂⟩|²/(⟨P₁₁⟩⟨P₂₂⟩)`. Add tests asserting *low* coherence for uncorrelated signals and ≈1 for identical (see TEST-D).

### D3 / D4 — The LFP Quick-Start (package docstring) is entirely non-runnable
**File:** `__init__.py:203–253` vs `_lfp.py` (signatures confirmed) · **Category:** Docs / API

Every LFP example in the package docstring calls an API that does not exist. Confirmed signature-by-signature:

| Docstring call (`__init__.py`) | Actual signature (`_lfp.py`) | Failure |
|---|---|---|
| `unitary_LFP(times, spikes, spike_type='excitatory')` (223) | `unitary_LFP(times, spikes, spike_type, …)`; `spike_type` ∈ `{'exc','inh'}` (223) | `ValueError` (`'excitatory'` rejected) |
| `power_spectral_density(lfp, fs=1000*u.Hz)` (227) | `power_spectral_density(lfp, dt, …)` (267) | `TypeError` (no `fs`); `dt` is seconds |
| `coherence_analysis(s1, s2, fs=1000*u.Hz)` (232) | `coherence_analysis(lfp1, lfp2, dt, …)` (330) | `TypeError` (no `fs`) |
| `phase_amplitude_coupling(lfp, phase_freq=(4,8), amp_freq=(30,80), fs=…)` (235) | `phase_amplitude_coupling(lfp, dt, phase_freq_range, amplitude_freq_range, n_bins)` (383) | `TypeError` (kwargs don't exist) |
| `current_source_density(lfp_channels, spacing=100*u.um)` (247) | `current_source_density(lfp_laminar, electrode_spacing)` (490); mm, positional | `TypeError`/unit error |
| `spectral_entropy(lfp, fs=1000*u.Hz)` (250) | `spectral_entropy(lfp, dt, …)` (524) | `TypeError` (no `fs`) |
| `lfp_phase_coherence(s1, s2, freq_band=(8,12))` (253) | `lfp_phase_coherence(lfp_signals, dt, freq_band)` (567); needs 2-D `(n_time, n_channels)` | `ValueError` (1-D unpack; `s2` taken as `dt`) |

On top of this, the same block uses `jnp.random.randint`/`jnp.random.randn` (DOC-1) and `times.mantissa` on a manually constructed array. **Net: the LFP onboarding path fails immediately for any user who copies it.** None of the LFP functions accept the unit-bearing arguments (`fs=…*u.Hz`, `spacing=…*u.um`) the docstring and the ecosystem imply.

**Fix:** decide the intended public API. Given the BrainX unit convention, the right long-term fix is to make the functions accept `fs: Quantity[Hz]` / `spacing: Quantity[um]` (strip via `.to_decimal(...)`), and unify on `fs` *or* `dt` consistently across the module. Short-term: rewrite the docstring to match the real `dt`-based signatures and valid `spike_type`/`location` values, and replace `jnp.random.*`.

---

## 4. High-severity issues

### Classification (`_classification.py`)

- **A8 — `kl_divergence` NaN gradient at `targets==0`** (line 561). `targets * (jnp.where(targets==0, 0, jnp.log(targets)) - log_predictions)` evaluates `jnp.log(0) = -inf` on the untaken branch, so `jax.grad` w.r.t. `targets` is `NaN` where `targets==0` (the classic single-`where` trap). **Fix:** double-where — `safe_t = jnp.where(targets==0, 1.0, targets); targets * (jnp.where(targets==0, 0.0, jnp.log(safe_t)) - log_predictions)`. Applies analogously to `kl_divergence_with_log_targets` / `convex_kl_divergence`.
- **A16 — `nll_loss` N-D support is documented but unimplemented** (lines 1015,1023,1078–1079). The docstring advertises `(batch, d1, …, dK)` targets; `target.ndim >= 2` hits `assert False`. Also missing `@set_module_as`, type hints, and it shadows builtin `input`. **Fix:** implement via `jnp.take_along_axis` or trim the docstring; add decorator/typing; rename `input → log_probs`.

### Regression / pairwise (`_regression.py`, `_pariwise.py`)

- **B9 — `l1_loss` default contradicts its own docs and wrapper** (line 427 `reduction='sum'`; docstring lines 438/464 say `'mean'`; `L1Loss.__init__` line 415 uses `'mean'`). **Fix:** set default to `'mean'` (or fix all docs to `'sum'`) and add a test pinning it.
- **B10 — `l1_loss` is mislabeled "MAE" but computes a per-row L1 *sum*** (lines 475–477: `jnp.linalg.norm(diff, ord=1, axis=1)` = Σ|·|, never divided by element count). With `reduction='mean'` it is the mean over samples of per-sample sums — still not MAE. **Fix:** if MAE is intended, use `jnp.mean(jnp.abs(diff), axis=1)`; otherwise rename/redocument as "summed L1 norm per sample."
- **B15 — Dead pairwise API / collision.** `_pariwise.cosine_similarity` is imported as `_cosine_similarity_pairwise` (`__init__.py:400`) and is **absent from `__all__`**; `_regression.cosine_similarity` (line 417) takes the public name. The pairwise function is unreachable as `braintools.metric.*`; notebook `04` works around this with `from braintools.metric._pariwise import cosine_similarity` (reaching into a private module). **Fix:** rename to `pairwise_cosine_similarity`, export it, and fix the filename (`_pariwise.py → _pairwise.py`).
- **B16 — `cosine_similarity` semantic collision.** `_regression.cosine_similarity` returns **per-pair** similarity (reduces last axis); `_pariwise.cosine_similarity` returns the **(n,m) pairwise matrix** (sklearn-style). Same qualified name, incompatible shapes; whichever is imported last wins. **Fix:** disambiguate by name (B15); keep `cosine_distance` paired with the elementwise version.
- **B1 — `log_cosh` gradient form** (line 672). Forward value is stable (`jnp.logaddexp(errors, -errors) - log 2`), but the canonical, gradient-stable form is `|x| + softplus(-2|x|) - log 2`. **Fix:** adopt the softplus form. (Forward-value test at `x=500` passes, so the suboptimal gradient is untested.)

### Firings (`_firings.py`)

- **C2 — `firing_rate` units** (lines 229–232). `window = u.math.ones(width1) / width` is converted to Hz **only if `width` is a `Quantity`**; with float `width`/`dt` (the path the tests exercise) no Hz conversion happens, so the result is in `1/[width units]`, not Hz. Also the kernel is normalized by `width` rather than the realized window duration `width1·dt`, introducing a rounding-scale error. **Fix:** `duration = width1 * dt; window = u.math.ones(width1) / duration`; require/validate time units (or document the float-seconds convention) and always convert to Hz.
- **C4/C8/C10/C13/C18/C20 — Spike functions are not `jit`/`vmap`/`grad`-able and don't say so.** `victor_purpura_distance`, `van_rossum_distance`, `spike_train_synchrony`, `burst_synchrony_index`, `spike_time_tiling_coefficient`, `correlation_index` use `len()`, Python `for`, `int()`, `float()`, boolean-mask `jnp.where(...)[0]`, and `if jnp.any(...)`/`if x_var==0` on traced values. They work only with concrete host arrays. **Fix:** document host-side status explicitly and/or provide vectorized variants; replace `if jnp.any(...)` with arithmetic for the ones meant to be traceable.
- **C9 — `spike_train_synchrony` can exceed its documented [0,1] range** (lines 503–513). Coincidences are counted by iterating only over `spikes_i` (asymmetric) and normalized by `min(len_i, len_j)`; when `len_i > len_j` the ratio can exceed 1. It is also not the cited Kreuz SPIKE-synchronization (which uses adaptive, symmetric windows). **Fix:** count symmetric coincidences and normalize by total spikes, or implement/rename appropriately and clamp.

### LFP (`_lfp.py`)

- **D1 — `power_spectral_density` is not Welch and discards most data** (lines 308–314). The code comment admits "Simple periodogram approach…"; it uses only `lfp[:nperseg]` (default `nperseg = n_time//8`, so ~87% of the signal is dropped), and `noverlap` (line 306) is never used. The docstring says "using Welch's method." **Fix:** implement real Welch or rename/redocument as a single-window periodogram over the full signal; remove or wire in `noverlap`.
- **D2 — PSD magnitude is miscalibrated** (line 314). One-sided PSD keeps `[:n_fft//2]` but never doubles interior bins, and normalizes by `fs*nperseg` instead of `fs*Σwindow²` (Hann: `Σw² = 0.375N`). **Fix:** double bins `1:` (excluding DC and, for even `n_fft`, Nyquist) and normalize by `fs*Σwindow²`.
- **D13/D17 — PAC and `lfp_phase_coherence` use a brick-wall mask that yields a (near-)real signal, so `jnp.angle` is degenerate** (`_lfp.py:428–437`, `593–600`). `nyquist/low_cutoff/high_cutoff` are computed and unused (dead "Butterworth" code); the symmetric ±band mask + `ifft` produces a real band-passed signal whose phase is ≈{0,π}, not the Hilbert instantaneous phase. Identical-signal tests pass, masking the defect. **Fix:** build the analytic signal (zero the negative-frequency bins before `ifft`), then take `jnp.angle`/`jnp.abs`. (`jax.scipy.signal.hilbert` availability is uncertain — use the FFT analytic-signal construction for JAX-safety.)

### Correlation (`_correlation.py`)

- **E17/E18 — `weighted_correlation` unguarded degeneracy + a documented exception that is never raised** (lines 537–554). All-zero weights ⇒ 0/0 NaN; zero weighted variance ⇒ NaN/Inf; no clamp to [-1,1]. The `Raises` docstring promises a `ValueError` "if arrays have different lengths," but **no length check exists**. **Fix:** validate `x.shape==y.shape==w.shape`; guard `Σw>0` and zero denominators; optionally `clip(-1,1)`.
- **E22 — `voltage_fluctuation` has a known eager-vs-`jit` discrepancy that the test suite disables** (`_correlation_test.py:130–131`: `# TODO: JIT results are different?` with the assertion commented out). This is a latent correctness bug (likely the both-branch `jnp.where` division at source line 241 plus float32). **Fix:** root-cause (guard the division so the bad branch isn't evaluated; check x64), then re-enable an `assertAlmostEqual`.

### Ranking / Fenchel-Young (`_ranking.py`, `_fenchel_young.py`)

- **F2 — Fenchel-Young `*args`/`**kwargs` forwarding is broken** (`_fenchel_young.py:116–120`). `max_fun` is wrapped once with `jnp.vectorize(max_fun, signature="(n)->()")` and then called as `max_fun_last_dim(scores, *args, **kwargs)`; `vectorize` treats extra positionals as additional core inputs, so the advertised extensibility raises or mis-broadcasts. **Fix:** bind args before vectorizing: `mf = lambda s: max_fun(s, *args, **kwargs); jnp.vectorize(mf, signature="(n)->()")(scores)`.
- **F9 — `ranking_softmax_loss` docstring misdescribes the loss** (lines 186, 283–285). It claims "labels are automatically normalized by the softmax," but labels enter as **raw multiplicative coefficients** on `log_softmax(logits)` (`labels *= weights`); there is no softmax over labels (this is the Rax convention, not ListNet). **Fix:** correct the docstring (labels are raw relevance weights, not normalized); optionally add a true-ListNet `normalize_labels` mode.

---

## 5. Medium / Low issues by module

### 5.1 Classification (`_classification.py`)

| ID | Sev | Issue | Line(s) | Fix |
|----|-----|-------|---------|-----|
| A1/A2 | High | `jnp.random.normal(size=…)` in `ctc_loss*` docstrings (also `size=` is wrong kwarg) | 740, 875 | `jax.random.normal(jax.random.PRNGKey(0), shape)` |
| A14 | Med | `sigmoid_focal_loss` doctest malformed (indented `    >>> print(...)`, no expected output) | 958–961 | de-indent; add expected output |
| A19 | Med | `hinge_loss` docstring expected output wrong: middle element is `0.5`, not `1.5` (`1 − (−0.5)(−1)=0.5`) | 149–153 | change to `[0. 0.5 0.]` |
| A3 | Low | `logits -= jax.lax.stop_gradient(...)` mutates a NumPy caller array in place (JAX arrays immune; NumPy not) | 304 | `logits = logits - …` |
| A13 | Med | `sigmoid_focal_loss` uses `jax.lax.cond(alpha>=0, …)` for a value that is virtually always a static Python float | 988 | plain `if alpha >= 0:` |
| A5 | Low | out-of-range integer `labels` silently clamp (JAX OOB) in `softmax_cross_entropy_with_integer_labels` | 304 | document; optional debug assert |
| A11/A12 | Med/Low | `ctc_loss*`: infeasible alignment → large *finite* (~1e5) loss, not `inf`; label `== blank_id` collision undocumented | 683,777 | document invariants |
| A6 | Med | `assert_is_float/int` use bare `assert` (stripped under `-O`); prefer `Type/ValueError` | 51–56 | raise explicit errors |
| A15 | Low | `\\alpha`/`\\gamma`/`\"…\"` double-escapes inside a raw (`r"""`) docstring render literally | 923–926,974 | single backslashes, plain quotes |
| A10 | Low | no `reduction` arg on any classification loss (inconsistent with `_util._reduce`) | — | see CC4 |

### 5.2 Regression / pairwise / smoothing

| ID | Sev | Issue | Line(s) | Fix |
|----|-----|-------|---------|-----|
| B5 | Med | `_regression.cosine_similarity` default `epsilon=0.` defeats zero-vector safety (docstring claims graceful handling) | 679 | default `1e-8` |
| B3/B4 | Med | `huber_loss`, `log_cosh`, `l2_loss` lack `axis`/`reduction` (notebook `02` mislabels their array output as "(mean)") | 529–534,485 | thread `axis`/`reduction` via `_reduce` |
| B2 | Med | `huber_loss` `jnp.minimum(abs_errors, delta)` breaks on `Quantity` inputs (dimensionless `delta`) | 642–645 | accept `Quantity` delta or document dimensionless-only |
| B17/B19 | Med | `_pariwise.cosine_similarity`: `eps` added to `norm_products` (unit² for Quantity ⇒ dimension error; magnitude-dependent bias) | 112–114 | `jnp.maximum(norm_products, eps)`; unit-safe |
| B18 | Med | `_pariwise.cosine_similarity` `nan_to_num` leaks NaN grads through `0/0` | 114 | floor denominator (B17) |
| B7/B8 | Low | `safe_norm` `axis=None` squeeze edge; **decorated `@set_module_as` but absent from `__all__`** (confirmed) and not re-exported | 127, 27–37 | export or de-publicize; add tests |
| B22/B23 | Med/Low | `smooth_labels` "rows sum to 1" guarantee holds only for valid distributions; no `alpha∈[0,1]` validation | 165,163 | document precondition; assert/clip alpha |
| B11 | Low | `squared_error`/`l2_loss` return unit² for `Quantity` inputs — undocumented | 245 | add Notes |
| B13 | Med | `squared_error`/`absolute_error` docstrings claim "broadcastable" but assert exact `shape` equality | 159,242 | fix docs (or relax to broadcasting) |
| B12/B25/B28 | Low | many examples not doctest-valid (expected output in trailing `# comment`, narrative `print`s); `l1_loss`/`l2_loss`/`l2_norm`/`log_cosh`/`cosine_distance` use Google-style `Args:`/`Returns:` (violates NumPy-doc CLAUDE.md) | various | convert to NumPy-doc + `>>>`/next-line output |

### 5.3 Firings

| ID | Sev | Issue | Line(s) | Fix |
|----|-----|-------|---------|-----|
| C1 | High | `firing_rate` param is `width`, but package docstring/notebooks use `window_size`/`window`/`duration` — three names; keyword calls `TypeError` | 120–122 | settle one name; fix all docs |
| C16/C17 | Med | STTC `T_A`/`T_B` **sum** overlapping windows instead of taking their union (overcount, then `min(1,·)` hides it); div-zero guard is combined so one degenerate term zeros the whole STTC; `tau`/`dt` unit mismatch via `int(tau/dt)` | 773,809–828 | union windows (Cutts–Eglen); guard each term; validate units |
| C19 | Med | `correlation_index` is mean Pearson `corrcoef`, but the name implies the Wong/Meister coincidence-ratio CI (different range/meaning); no reference | 842–923 | rename or implement true CI; add References |
| C5/C26 | Med | time-like params (`tau`,`t_max`,`window_size`,`max_isi`,`reference_freq`,`cost_factor`) typed plain `float`; no `Quantity` support/validation though ecosystem uses units | many | accept/validate `Quantity` uniformly |
| C6/C7 | Med/Low | `van_rossum_distance` normalization deviates from canonical `(1/τ)∫(f₁−f₂)²` (docstring kernel and integral disagree); `5τ` tail truncation undocumented | 407,418,390 | pick/justify a convention; document |
| C14 | Med | `phase_locking_value` is vector strength to an external clock (not Lachaux pairwise PLV); single spike ⇒ PLV=1 inflation; no small-N bias correction | 684–704 | document; add min-spike guard / bias correction |
| C11/C21/C25 | Low | dead `window_steps` (line 482); `raster_plot` drops `Quantity` units via `onp.asarray`; `correlation_index` example overwrites a column with fractional "spikes" | 482,112,878 | remove dead code; preserve units; fix example |
| C22/C24 | High/Low | `jnp.random.random` in `correlation_index` docstring (line 876); compound-statement doctests use `>>>` not `...` for continuations | 876,558–560,668–673,753–755 | `jax.random`/`brainstate.random`; use `...` |

### 5.4 LFP

| ID | Sev | Issue | Line(s) | Fix |
|----|-----|-------|---------|-----|
| D6 | High | docstring lists `location='surface'` and the in-file example uses it, but only `'surface layer'` is accepted ⇒ `NotImplementedError` | 182,249–252 | accept `'surface'`; give `NotImplementedError` a message |
| D9/D10 | High/Med | `current_source_density`: spacing is mm float (doc/`__init__` say `u.um`); no σ (conductivity); silent empty output for `n_electrodes<3`; assumes channels=**last** axis but `__init__` example passes channels-first `(16,1000)` ⇒ derivative over time | 490–518 | unit-aware spacing; add `conductivity`, `axis`, `n≥3` guard |
| D14/D16 | Med | PAC bin-mean uses boolean-mask indexing (non-`jit`, empty-bin→0 biases MI); `spectral_entropy` uses Python `if` on traced sums and mixes channels | 443–448,545–563 | `jnp.digitize`+`segment_mean`; `jnp.where`/`lax.cond`; per-channel entropy |
| D15 | Med | PAC/`theta_gamma_coupling` band tuples are plain Hz; no Nyquist/order validation; `theta_gamma_coupling` annotated `-> float` but returns 0-d array | 386–387,464–467 | unit-aware bands + validation; fix annotation |
| D18/D19 | Low | `lfp_phase_coherence` O(n²) Python double-loop (vectorizable); `unitary_LFP` uses **deprecated** `brainstate.compile.for_loop` and `jnp.where(spikes)` (non-`jit`) | 607–618,259–263 | vectorize PLV; migrate to `brainstate.transform.for_loop` |
| D8 | Low | `unitary_LFP` error messages garbled (`'"spike_type" should be "exc or ""inh".'`, `"E_spikes"`, `times.shape[0]` vs `spikes.shape`, "firs axis"); a test asserts the garbled string | 224–230 | clean messages **and** update the coupled test |
| D20 | Med | `unitary_LFP` `delay = 10.4 + dist/va`: `dist`[mm]/`va`[mm/s]=s added to 10.4 ms (unit inconsistency; numerically masked by tiny default `dist`) | 256,93 | reconcile `va` units; document the 10.4 ms constant |
| D5/D7 | Low | `pos_xs,pos_ys = rng.rand(2,N)*[[xmax],[ymax]]` fragile (works); `exc_amp`/`tts` misnamed in the inhibitory path | 235,261 | clearer per-axis sampling; rename |

### 5.5 Correlation

| ID | Sev | Issue | Line(s) | Fix |
|----|-----|-------|---------|-----|
| E1–E4 | High | `jnp.random.binomial/normal/rand` in docstrings (plus `jnp.fill_diagonal` line 284 — also nonexistent) | 100,210,215,282,284,286,349,350 | `jax.random.*`+`PRNGKey`; `.at[diag].set` |
| E7 | Med | `cross_correlation`: `bin_size=int(bin/dt)=0` when `bin<dt` ⇒ ZeroDivisionError; non-integer ratio truncates (off-by-one binning) | 111–113 | validate `bin≥dt`; `int(round(bin/dt))`, raise if `<1` |
| E8 | Med | `cross_correlation` single neuron ⇒ empty `tril_indices` ⇒ `jnp.mean([])`=NaN (docstring promises a float in [0,1]) | 118,143 | guard `num_neu<2 → 0.0` |
| E13 | Med | `functional_connectivity`: constant column ⇒ `nan_to_num` sets the **diagonal to 0**, contradicting "Diagonal elements are 1.0" | 373–374 | force diagonal to 1 after `nan_to_num` |
| E11/E18 | Med | `matrix_correlation`/`weighted_correlation` lack shape-equality checks (deep, non-obvious failures); `matrix_correlation` NaN for constant inputs (no `nan_to_num`, unlike FC) | 309–316,545–553 | add `shape` checks; unified NaN policy |
| E14/E15 | Med | `functional_connectivity_dynamics`: no JIT note (unlike siblings); `(0,0)` empty-return shape differs from normal path; `n_sig<2` silently returns identity | 435–462 | document static-arg/`vmap` limits; guard `n_sig<2` |
| E6 | Low | `cross_correlation` LaTeX denominator under-parenthesized; code is the correct Wang–Buzsáki coincidence coherence but is labeled "correlation coefficient" | 55–56 | fix LaTeX/wording |
| E9 | Med | `voltage_fluctuation` constant input ⇒ returns `1.0` ("asynchronous"), and both `where` branches evaluated (div-by-zero warning) | 241 | guarded division; document convention |
| E19/E20 | Low | no `Quantity` support (voltages usually mV; `jnp.mean` on Quantity unsupported); return types annotated/doc'd as `float` but are 0-d arrays | 22,81,… | `u.math.*`; `-> jax.Array` |

### 5.6 Ranking / Fenchel-Young

| ID | Sev | Issue | Line(s) | Fix |
|----|-----|-------|---------|-----|
| F3 | Med | `make_fenchel_young_loss` has no `custom_vjp`; correct gradient relies on autodiff of `max_fun` — fine for smooth `Ω*` (logsumexp), wrong/undefined for sparse (sparsemax/entmax) which the docstring implies are supported | 114–122 | document smooth-`max_fun` requirement; offer a `custom_vjp` oracle variant |
| F4 | Med | FY docstring calls `max_fun` "the regularizer Ω" and writes `Ω(θ)−⟨y,θ⟩`; it is actually the **conjugate** `Ω*` (log-partition); gradient is `ŷ(θ)−y` | 45–59 | rewrite as `Ω*(θ)−⟨θ,y⟩` |
| F5 | Low | `__init__.py` FY example `max_fun` uses `jnp.max(scores, axis=-1, keepdims=True)` → shape `(1,)`, violating `vectorize` `"(n)->()"`; the in-file `custom_max` returns a scalar (inconsistent) | `__init__.py` FY block; `_fenchel_young.py:98` | use `jnp.max(scores)` |
| F1 | Low | `jnp.vdot(targets, scores)` conjugates `targets` (latent hazard for complex inputs) | 115,120 | `jnp.sum(targets*scores, axis=-1)` |
| F10/F11 | Low/Med | ranking uses `reduce_fn` callable (rest of package uses `reduction` string + `_util._reduce`); empty-mask NaN guard keyed on `reduce_fn is jnp.mean` identity (fails for `functools.partial`/aliases) | 158,127 | accept `reduction` string; robust mean detection |
| F13/F14 | Low | `labels.astype(...)` assumes array inputs (Python list ⇒ `AttributeError`); masked logits set to `−inf` create `0*NaN` only masked by `where=` in the sum (fragile) | 276,280–294 | `jnp.asarray` inputs; mask CE with `jnp.where(where, …, 0.)` before sum |
| F16/F17 | Low | `See Also` cites nonexistent `jax.nn.softmax_cross_entropy`; "Returns shape `(batch_dims,)`" is actually all leading dims | 264,205–208 | fix references/shape text |

---

## 6. Cross-cutting & architecture

- **CC1 — `cosine_similarity` name collision + dead pairwise export.** See B15/B16. The package docstring (`__init__.py:31`) advertises "Pairwise Metrics: Cosine similarity," but it is unreachable. **Fix:** rename + export `pairwise_cosine_similarity`; add it to `docs/apis/metric.rst` (which currently has **no Pairwise section** at all).
- **CC2 — Misspelled module + missing test.** `_pariwise.py` → `_pairwise.py`; add `_pairwise_test.py`. (Rename touches `__init__.py:399`.)
- **CC3 — `jnp.random.*` is used in ~15 example sites although it does not exist in JAX.** Consolidated in §8.1. Establish a docs convention (`jax.random` + `PRNGKey`, or `brainstate.random`) and add a doctest/CI gate.
- **CC4 — Inconsistent reduction API.** Three conventions coexist: `_util._reduce(reduction='mean'|'sum'|'none')` (regression), `reduce_fn` *callable* (ranking), and **no reduction parameter at all** (classification KL/CTC/focal). **Fix:** standardize on the string `reduction` + `_util._reduce` everywhere, preserving ranking's empty-mask guard.
- **CC5 — Inconsistent / broken unit handling.** `brainunit` is the ecosystem's core, yet `_correlation` and `_fenchel_young` never import it; `_lfp`/`_firings` accept units unevenly and several paths break on `Quantity` (B2, B19, C2, C26, D3, D9, E19). **Fix:** a module-wide units policy — either "dimensionless arrays only" (documented + validated) or first-class `Quantity` support via `u.math`/`u.linalg` with `.to_decimal(...)` at boundaries.
- **CC6 — Silent non-traceability.** Many neuroscience functions cannot be `jit`/`vmap`/`grad`-ed (Python loops, `len()`, `int()`, `float()`, boolean-mask indexing, `if` on traced values): `victor_purpura_distance`, `van_rossum_distance`, `spike_train_synchrony`, `burst_synchrony_index`, `spike_time_tiling_coefficient`, `correlation_index`, `raster_plot`, `functional_connectivity_dynamics`, `unitary_LFP`, PAC, `spectral_entropy`. In a JAX library this is a significant limitation. **Fix:** document host-side status per function; provide vectorized/`lax`-based variants where feasible; add `jit`/`vmap` contract tests (xfail or parity).
- **CC7 — `__all__` / export hygiene.** `safe_norm` is `@set_module_as`-decorated (implying public) but missing from `_regression.__all__` and `__init__` (B8). Audit all `@set_module_as` symbols against the exported surface.

---

## 7. Documentation findings

### 7.1 API reference (`docs/apis/metric.rst`)
- No **Pairwise** section (the pairwise `cosine_similarity` is neither exported nor documented). Add after the rename (CC1).
- Otherwise the autosummary lists match `__all__`. Once functions are renamed/added (pairwise), keep both in sync.

### 7.2 Package docstring (`__init__.py`)
- The **LFP Quick-Start is entirely non-runnable** (D3/D4).
- Contradictory `firing_rate` examples: `firing_rate(spike_times, duration=…)` (line 78) vs `firing_rate(spikes, window_size=…, dt=…)` (line 187) — neither matches the real `firing_rate(spikes, width, dt)` (C1).
- `jnp.random.randint/randn` at lines 222, 246, 274, 278, 288 (DOC-1).
- `times.mantissa` examples assume a constructed `Quantity` path that won't run as written.

### 7.3 Notebooks (`docs/metric/*.ipynb`)
- **01 (classification):** `nll_loss` cell documents the wrong (negative) value (A17/A28).
- **02 (regression):** cells labeled "Huber (mean)", "log-cosh (mean)", "l2_loss (mean)", "l1_loss (… mean by default)" print full arrays / the `'sum'` result — labels are wrong (B3/B4/B9).
- **04 (pairwise):** imports `cosine_similarity` from the **private** `_pariwise` module (symptom of CC1).
- **05/06 (spiking):** reference `cross_correlation`, `voltage_fluctuation`, `functional_connectivity` (these live in `_correlation`, exported — verify each cell imports/qualifies correctly); STTC cell uses `tau=0.01, dt=1.0` ⇒ `int(0.01/1.0)=0` (zero-width window — C16/M4); `firing_rate` unit convention differs between notebooks (Quantity vs float — C2/N3); "Best Practices" heading renders as "est Practices" (dropped "B").
- **07 (LFP):** inherits the coherence (D11), PSD (D1/D2), PAC phase (D13), and CSD spacing/axis (D9/D10) defects in its narrative/outputs.
- Pervasively, notebook example data uses `np.random` (valid — numpy is imported) — *not* a bug; the bug is `jnp.random` in **source docstrings**.

### 7.4 NumPy-doc compliance (per CLAUDE.md)
Many examples violate the project's own docstring rules: expected output placed in trailing `# comments` instead of the next line; compound-statement continuations prefixed `>>>` instead of `...`; some functions use Google-style `Args:`/`Returns:` (B28). A `doctest`/Sphinx build gate would catch most of §7.

### 8.1 Consolidated `jnp.random` / nonexistent-API sites (DOC-1)

| File:line | Offending call | Replace with |
|---|---|---|
| `__init__.py:222` | `jnp.random.randint(0, 2, (100, 1000))` | `jax.random.randint(key, (1000,100), 0, 2)` (note time-first) |
| `__init__.py:246,274,278,288` | `jnp.random.randn(...)` | `jax.random.normal(key, shape)` |
| `_classification.py:740,875` | `jnp.random.normal(size=…)` | `jax.random.normal(key, shape)` |
| `_correlation.py:100` | `jnp.random.binomial(1,0.1,(1000,50))` | `(jax.random.uniform(key,(1000,50))<0.1).astype(float)` |
| `_correlation.py:210,215,349,350` | `jnp.random.normal(...)` | `jax.random.normal(key, shape)` |
| `_correlation.py:282,286` | `jnp.random.rand(5,5)` | `jax.random.uniform(key,(5,5))` |
| `_correlation.py:284` | `jnp.fill_diagonal(...)` (nonexistent; arrays immutable) | `m = m.at[jnp.diag_indices(5)].set(1.0)` |
| `_firings.py:876` | `jnp.random.random((1000,10))` | `(jax.random.uniform(key,(1000,10))<0.1).astype(float)` |

(`brainstate.random.*`, already used in tests, is an equally valid replacement.)

---

## 8. Testing gaps

- **TEST-A — `nll_loss` has no tests** (would have caught the A17 sign bug and A16 N-D assert). Add value tests vs a reference.
- **TEST-KL — KL variants test forward values only**, never gradients (A8 NaN untested).
- **TEST-firing — `firing_rate` tests only `print()`** (`_firings_test.py:30–45`); the C2 unit bug is invisible. Add value/shape assertions and Quantity-vs-float parity.
- **TEST-corr — print-only / disabled tests** (`_correlation_test.py`): `test_cc2–cc5` assert nothing; `test_vf1` has its JIT-equality assertion commented out (E22); several tests enshrine buggy behavior (`isnan or 0.0`, `try/except pass`, tautological `isnan or isfinite`).
- **TEST-lfp — assertions too weak to catch the real bugs:** coherence test never asserts *low* coherence for uncorrelated input (D11 passes); PSD test never checks the peak frequency / power calibration (D1/D2); PAC test never asserts coupled > uncoupled (D13); CSD test uses a linear profile whose 2nd derivative is 0 (passes any code).
- **TEST-spike — correctness vs bounds:** VP/van-Rossum/STTC/synchrony tested with inequalities, not hand-computed values; STTC overcount (C16), asymmetric div-zero (C17), `spike_train_synchrony>1` (C9), PLV single-spike inflation (C14) untested; several random tests are **unseeded** (flaky thresholds).
- **TEST-rank/FY — gradient *correctness* untested** (only NaN-absence); `test_computes_loss_with_extreme_inputs` encodes a non-`logsumexp` formula that passes only by float32 overflow coincidence (F19); FY `*args/**kwargs` (F2) and non-`logsumexp` `max_fun` (F3/F5) untested.
- **General:** add `jit`/`vmap`/`grad` smoke/parity tests across the package (ties to CC6), and a `doctest` gate for examples (ties to §7).

A coverage pass that targets these (rather than line count) is the fastest way to prevent regressions of the Critical/High items. (CLAUDE.md target: >90% with meaningful edge-case/critical-path tests.)

---

## 9. Missing features / enhancements

**Classification/regression:** uniform `reduction`/`axis`; `nll_loss` `weight`/`ignore_index`/N-D (PyTorch parity); `ignore_index`/padding mask for integer-label CE; multiclass (softmax) focal loss; `mean_squared_error`/`mae`/`rmse` aliases (the docstring advertises "MSE, MAE").

**Pairwise:** export `pairwise_cosine_similarity` + a matching `pairwise_cosine_distance`; Euclidean/Manhattan/Mahalanobis pairwise distances.

**Ranking/FY:** RankNet (pairwise logistic), ListMLE, NDCG/ApproxNDCG (graded relevance), pairwise hinge; concrete FY losses (`sparsemax_loss`, `entmax`) and a `custom_vjp` oracle variant.

**Spiking:** ISI/CV, Fano factor, PSTH, Gaussian/exponential-kernel firing rate, SPIKE-distance (Kreuz), true Cutts–Eglen STTC, Wong/Meister coincidence-ratio correlation index, bias-corrected & pairwise (Lachaux) PLV, CCG + jitter/shuffle surrogates.

**LFP:** real Welch (`power_spectral_density`, `coherence_analysis`); unit-aware API (`fs`/`spacing`/bands as `Quantity`); analytic-signal (Hilbert) PAC/phase; conductivity + `axis` + boundary guard for CSD; two-signal `lfp_phase_coherence` overload; `freq_range` for coherence.

**Correlation:** `method=` (Pearson/Spearman/partial/precision); lagged FC and nonzero-lag cross-correlation; significance/surrogate testing; unified degenerate-input (NaN) policy; `Quantity` voltages.

**Cross-cutting:** consistent `Quantity` support; `jit`/`vmap`-safe variants of the host-side functions; `doctest`/Sphinx CI gate.

---

## 10. Prioritized remediation roadmap

1. **Correctness Criticals (ship-blockers):** A17 (`nll_loss` sign), D11 (coherence), C2 (`firing_rate` units), D1/D2 (PSD). Add the regression tests that should have caught each (TEST-A, TEST-lfp, TEST-firing).
2. **Broken public surface:** D3/D4 (LFP Quick-Start) and DOC-1 (`jnp.random` everywhere) — make all documented examples run (add a doctest gate); B15/B16/CC1/CC2 (cosine collision, `_pariwise` rename + test + export + rst section).
3. **High correctness/gradient:** A8 (KL NaN grad), B9/B10 (`l1_loss`), D13/D17 (PAC/phase analytic signal), E17/E18/E22 (`weighted_correlation`, `voltage_fluctuation` JIT), C9/C16/C17 (synchrony/STTC), F2/F9 (FY args / ranking docs).
4. **Consistency/architecture:** CC4 (reduction API), CC5 (units policy), CC6 (traceability docs + tests), CC7 (export hygiene), A-/B-/D-/E-series docstring & validation fixes.
5. **Enhancements (§9):** prioritize unit-aware LFP/Welch and the missing ranking/spiking metrics by user demand.

---

## Appendix A — Investigated and cleared (not bugs)

- `_regression.cosine_similarity` "orthogonal pairs → [0,0,0]" docstring example: recomputed; `[1,1]·[1,−1]=0`, genuinely orthogonal. Correct.
- `voltage_fluctuation` `(time, neurons)` axis handling (population mean over `axis=1`, per-neuron variance via `moveaxis`/`vmap`): axes are consistent and correct (E10).
- `convex_kl_divergence` correction term `Σ(exp(log_pred) − targets)` matches the documented `Σ(Q−P)` and reduces to standard KL for normalized inputs (A9) — suggest only a clarifying Note.
- `cross_correlation` per-pair zero-division is correctly guarded by `lax.cond(sqrt_ij==0, 0, …)` (lines 123–126); the formula matches Wang–Buzsáki.
- Ranking/FY notebook numeric outputs verified to displayed precision; `_ranking.py`/`_fenchel_young.py` contain no `jnp.random` misuse.
- Notebook `np.random.*` usage is valid (numpy is imported); the `np.random` in `firing_rate`'s docstring (lines 179–180) is fine because that example imports numpy.

## Appendix B — Method notes / confidence

- All **Critical** and most **High** findings were re-verified by reading the cited source ranges and signatures during this audit; remaining items are from full-file static reads and are labeled SUSPECTED where runtime confirmation would strengthen them.
- Because verification was static, items depending on `brainunit`/`brainstate` internals (e.g. exact `u.math.is_int` tracing behavior, `rng.rand(2, N)` varargs) are flagged rather than asserted.
- Recommended next step before fixes: stand up a `doctest` + `pytest -q` gate and convert the print-only tests (§8) into assertions; that turns most of this report into red tests to drive the fixes.
