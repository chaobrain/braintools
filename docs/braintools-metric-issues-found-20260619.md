# `braintools.metric` — Issues Found and Proposed Solutions (2026‑06‑19)

A senior‑developer / JAX‑expert audit of every source file in `braintools/metric/`
(`_classification.py`, `_regression.py`, `_firings.py`, `_lfp.py`, `_correlation.py`,
`_ranking.py`, `_fenchel_young.py`, `_pairwise.py`, `_smoothing.py`, `_util.py`,
`__init__.py`) and their docstrings.

Methodology: each file was read in full and inspected for correctness, JAX pitfalls
(NaN gradients, tracer control flow, dtype), numerical stability, edge cases,
`brainunit.Quantity` handling, API consistency, and NumPy‑doc accuracy. **Every
actionable finding below was reproduced with a runnable snippet** before being
recorded. Baseline test state: **332 passing**.

The module is in good shape overall — the core mathematics of every metric was
verified correct against references. The defects are concentrated in (a) a few
`brainunit.Quantity`/edge‑case crashes, (b) one numerically wrong result for
small vectors, (c) one inverted docstring interpretation, and (d) a batch of
docstring example outputs that would fail a strict doctest.

Legend — Severity: **High** (crash / wrong result for normal use), **Medium**
(crash on a supported input class / robustness), **Low** (edge case / docs).

---

## Summary table

| ID | File | Sev | Category | One‑line |
|----|------|-----|----------|----------|
| FIR‑1 | `_firings.py` | High | brainunit | `phase_locking_value` crashes on `Quantity` `dt` (incl. the common `environ.get_dt()` default) |
| INIT‑1 | `__init__.py` / `_regression.py` | High | api | `L1Loss` (documented public class) is unexported → `from braintools.metric import L1Loss` raises `ImportError` |
| PW‑1 | `_pairwise.py` | High | numerical | `pairwise_cosine_similarity` returns wrong values for small (non‑zero) vectors |
| COR‑1 | `_correlation.py` | High | docs | `voltage_fluctuation` Returns interpretation is inverted/impossible |
| COR‑2 | `_correlation.py` | Medium | brainunit | `cross_correlation` crashes on `Quantity` input |
| LFP‑1 | `_lfp.py` | Medium | edge‑case | `lfp_phase_coherence` reports spurious coherence for an empty/zero‑power band |
| SM‑1 | `_smoothing.py` | Medium | edge‑case | `smooth_labels` uses `assert` for input validation (stripped under `python -O`) |
| FIR‑2 | `_firings.py` | Medium | jax/perf | `victor_purpura_distance` builds its DP table with traced JAX arrays → O(n²) device round‑trips |
| REG‑6 | `_regression.py` | Low | edge‑case | `l1_loss` raises `IndexError` on a 0‑d/scalar input |
| LFP‑2 | `_lfp.py` | Low | api | `coherence_analysis` lacks the empty‑`freq_range` fallback that `power_spectral_density` has |
| LFP‑3 | `_lfp.py` | Low | numerical | `coherence_analysis` uses an absolute `1e‑15` floor (amplitude‑dependent) |
| LFP‑5 | `_lfp.py` | Low | numerical | `_analytic_band` doubles the DC bin if a band includes 0 Hz |
| CLS‑2 | `_classification.py` | Low | api | `sigmoid_binary_cross_entropy` raises `AttributeError` for list inputs |
| FIR‑3 | `_firings.py` | Low | correctness | `burst_synchrony_index` strict `> 0` overlap excludes endpoint‑touching bursts |
| FIR‑4 | `_firings.py` | Low | correctness | `correlation_index` silently drops the trailing partial bin; dead clamp code |
| FIR‑6 | `_firings.py` | Low | numerical | `correlation_index` result not clamped to documented `[-1, 1]` |
| COR‑4 | `_correlation.py` | Low | jax | `cross_correlation` `lax.cond` returns a weak‑typed Python `0.` branch |
| PW‑3 | `_pairwise.py` | Low | jax | `_safe_row_norm` hardcodes `axis=1` |
| CLS‑1 | `_classification.py` | Low | docs | `multiclass_hinge_loss` example output wrong (`[0. 0.]` → `[0. 0.6]`) |
| REG‑1..4 | `_regression.py` | Low | docs | `huber_loss`/`log_cosh`/`cosine_similarity`/`safe_norm` example outputs wrong |
| PW‑2 | `_pairwise.py` | Low | docs | `pairwise_cosine_similarity` example diagonal wrong (`1.` → `1.0000001`) |
| REG‑5 | `_regression.py` | Low | docs | `log_cosh` says "no broadcasting" but it does broadcast |
| REG‑7 | `_regression.py` | Low | docs | `log_cosh` rejects `Quantity`; should document dimensionless requirement |
| REG‑9 | `_regression.py` | Low | docs | `squared_error` `axis` param wording implies it works without a reduction |
| FIR‑7 | `_firings.py` | Low | api/docs | `phase_locking_value` `dt` type hint/doc omits `Quantity` and the Hz/seconds contract |
| FIR‑8 | `_firings.py` | Low | docs | `firing_rate` does not state the float‑seconds assumption in Parameters |
| FY‑1..3 | `_fenchel_young.py` | Low | docs | unconditional "non‑negative" claim; misleading complex/`L2` comments |
| COR‑3 | `_correlation.py` | Low | docs | `weighted_correlation`/`functional_connectivity_dynamics` examples not doctestable |
| RNK‑2 | `_ranking.py` | Low | docs | `ranking_softmax_loss` examples print via f‑string with no expected output; module example omits `import jnp` |
| LFP‑4 | `_lfp.py` | Low | docs | PSD "Hann‑windowed" differs from scipy's periodic default (note) |
| SM‑2 | `_smoothing.py` | Low | api | `smooth_labels` return annotation is legacy `jnp.ndarray` |
| UTIL‑1 | `_util.py` | Low | api | `_reduce` silently ignores `axis` when `reduction='none'` (note) |
| CLS‑4 | `_classification.py` | Low | numerical | `sigmoid_focal_loss` has non‑finite gradient for `0<gamma<1` with saturated logits (note) |
| CLS‑3 / CLS‑5 / REG‑8 / RNK‑1 / LFP‑6 | various | Low | note | Acceptable behaviors documented for clarity (no behavior change) |

---

## High severity

### FIR‑1 — `phase_locking_value` crashes on `Quantity` `dt`
**Location:** `_firings.py:745‑753`.
**Evidence:**
```python
m.phase_locking_value(spikes, 10.0, dt=0.1*u.ms)
# TypeError: exp requires ndarray or scalar arguments, got <class 'saiunit.Quantity'>
```
`times = jnp.arange(n_time) * dt` keeps units when `dt` is a `Quantity`, so
`jnp.exp(1j * spike_phases)` fails. This is the *common* path because
`brainstate.environ.get_dt()` returns a `Quantity`. A second latent issue: with
`reference_freq` in Hz, the time base must be **seconds**, but a `0.1*u.ms`
magnitude would be off by 1000×.
**Solution:** convert `dt` to **seconds** (`dt.to_decimal(u.second)` for a
`Quantity`, otherwise assume seconds), document the Hz/seconds contract, and
widen the `dt` type hint (see FIR‑7). Add a regression test with a `Quantity` `dt`.

### PW‑1 — `pairwise_cosine_similarity` wrong for small non‑zero vectors
**Location:** `_pairwise.py:140`.
**Evidence:**
```python
X = jnp.array([[3e-5, 0.], [3e-5, 0.]])
m.pairwise_cosine_similarity(X)[0,0]   # 0.09   (true cosine = 1.0)
```
The denominator floors the **product** of the two norms at `eps=1e‑8`. Whenever
`‖x‖·‖y‖ < 1e‑8` — which happens for perfectly legitimate small vectors — the
floor inflates the denominator and corrupts the (scale‑invariant) cosine. The
floor should apply to each *row* norm individually, which only engages for genuine
near‑zero vectors.
**Solution:** clamp `X_norms`/`Y_norms` at `eps` *before* the outer product:
`norm_products = jnp.maximum(X_norms, eps) @ jnp.maximum(Y_norms, eps).T`. This
preserves the documented zero‑vector → `0` behaviour (a zero row still has dot‑product
`0` in the numerator) and the gradient safety of `_safe_row_norm`. Update the Notes
and `eps` description accordingly.

### INIT‑1 — public class `L1Loss` is not exported
**Location:** `_regression.py` (`__all__`), `__init__.py` (import block + `__all__`).
**Evidence:**
```python
from braintools.metric import L1Loss
# ImportError: cannot import name 'L1Loss' from 'braintools.metric'
```
`L1Loss` is a fully‑documented public class (NumPy‑doc docstring, `Examples` showing
`from braintools.metric import L1Loss`), but it was absent from `_regression.__all__`,
from the `from ._regression import (...)` block in `__init__.py`, and from
`braintools.metric.__all__`. Its own example was therefore non‑importable, and
`braintools.metric.L1Loss` did not exist. (Discovered by a `--doctest-modules` sweep;
`MaxFun` in `_fenchel_young.py` is a typing `Protocol`, intentionally internal, and is
the only other unexported public‑cased class.)
**Solution:** add `'L1Loss'` to `_regression.__all__`, import it in the `_regression`
block of `__init__.py`, and add `'L1Loss'` to `braintools.metric.__all__`. Verified
`from braintools.metric import L1Loss` and `L1Loss().update(x, y)` now work.

### COR‑1 — `voltage_fluctuation` documented interpretation is inverted
**Location:** `_correlation.py:213‑218` (Returns) and the misleading comments in
`_correlation_test.py:147,156`.
**Evidence:** the Golomb χ²(N) = σ²_V / ⟨σ²_Vᵢ⟩ is bounded in approximately
`[1/N, 1]`: it → 1 for full synchrony and → 1/N for asynchrony, and **cannot exceed 1**.
```python
voltage_fluctuation(sync_signal)  # 0.9998   (≈1  → synchronous)
voltage_fluctuation(async_signal) # 0.0168   (≈1/N → asynchronous)
```
The docstring says "Values > 1 indicate synchronized activity, values ≈ 1 indicate
asynchronous activity" — both halves are wrong (the `>1` region is unreachable, and
`≈1` is synchrony, not asynchrony).
**Solution:** rewrite Returns to: *index in approximately `[1/N, 1]`; values near 1
indicate strong synchrony, values near 1/N indicate asynchronous activity; a constant
population returns 1.0 by convention.* Fix the two test comments.

---

## Medium severity

### COR‑2 — `cross_correlation` crashes on `Quantity` input
**Location:** `_correlation.py:117‑130`.
**Evidence:** `cross_correlation(spikes * u.UNITLESS, 5.0, dt=1.0)` →
`TypeError: sum requires ndarray ...`. Every sibling metric strips units with
`u.get_magnitude(...)`; this one operates on the raw input.
**Solution:** `spikes = jnp.asarray(u.get_magnitude(spikes))` at the top.

### LFP‑1 — `lfp_phase_coherence` spurious coherence for empty bands
**Location:** `_lfp.py:650‑667`.
**Evidence:** for a band containing no signal power the band‑limited analytic signal
is ≈ 0; `jnp.angle(0) == 0` makes all unit phasors identical, so channels appear
phase‑locked:
```python
lfp_phase_coherence(sig_10_and_30Hz, 0.001, freq_band=(200,300))[0,1]  # 0.31 (should be ~0)
```
**Solution:** detect channels whose in‑band power is negligible relative to their
broadband power and report their off‑diagonal coherence as `0` (the phase of a
zero signal is undefined). This keeps the documented unit‑phasor PLV unchanged for
channels with real in‑band power. Document the behaviour.
**Test impact:** the pre‑existing `TestLFPPhaseCoherence.test_different_frequency_bands`
asserted `coherence_alpha >= 0.8` for a pure 25 Hz signal evaluated in the 8–12 Hz
band — i.e. it *encoded the spurious‑coherence bug*. Updated it to assert the corrected
semantics (empty band → ~0; the in‑band beta result stays ≥ 0.8).

### SM‑1 — `smooth_labels` validates with `assert`
**Location:** `_smoothing.py:155`.
**Evidence:** under `python -O`, `assert` is stripped and integer labels pass
silently. `assert` is an anti‑pattern for input validation (the sibling `alpha`
check correctly raises `ValueError`).
**Solution:** `if not u.math.is_float(labels): raise TypeError(...)`; update
`test_smooth_labels_assertion_error` to expect `TypeError`.

### FIR‑2 — `victor_purpura_distance` DP uses traced JAX arrays
**Location:** `_firings.py:328‑353`.
**Evidence:** the DP table is a `jnp` array mutated with `.at[i,j].set(...)` and read
with `dp[i-1,j-1]` inside a Python double loop, forcing a device round‑trip per cell
(~1.9 s for a 40×40 pair). The result is correct but pathologically slow. The function
is documented host‑only.
**Solution:** build the DP table with NumPy (`numpy` floats / `min`), converting the
spike times once. Functionally identical, orders of magnitude faster.

---

## Low severity — behaviour fixes

- **REG‑6** `_regression.py:531` — `l1_loss(jnp.array(1.0), jnp.array(1.5))` →
  `IndexError` from `logits.shape[0]`. Fix: `jnp.atleast_1d` the difference before
  the `(N, -1)` reshape so scalars are treated as a one‑sample batch.
- **LFP‑2** `_lfp.py:406‑410` — `coherence_analysis` returns empty arrays for an
  out‑of‑range `freq_range`; mirror the PSD fallback (one zero bin).
- **LFP‑3** `_lfp.py:402` — replace the absolute `+ 1e‑15` denominator floor with a
  scale‑invariant `jnp.where(denom > 0, denom, 1.0)` guard so down‑scaled signals
  are not biased.
- **LFP‑5** `_lfp.py:61‑62` — the analytic‑signal reconstruction doubles every in‑band
  bin including DC; DC (and Nyquist) must not be doubled. Use a per‑bin factor that is
  `1` at `f == 0` and `2` otherwise.
- **CLS‑2** `_classification.py:113` — `sigmoid_binary_cross_entropy` calls `.astype`
  on raw inputs; `jnp.asarray` both arguments first (matches `nll_loss`).
- **FIR‑3** `_firings.py:664` — change `overlap > 0` to `overlap >= 0` so bursts that
  touch at a single instant (the strongest synchrony) are counted; document the convention.
- **FIR‑4 / FIR‑6** `_firings.py:980,999,1003` — remove the dead `min(...)` clamp,
  document that the trailing partial bin is discarded, and clamp the returned mean to
  the documented `[-1, 1]`.
- **COR‑4** `_correlation.py:137,148` — return `jnp.zeros((), dtype=sqrt_ij.dtype)`
  from the `lax.cond` zero branch instead of a weak‑typed Python `0.`.
- **PW‑3** `_pairwise.py:39` — use `axis=-1` in `_safe_row_norm` for robustness.

## Low severity — documentation fixes

- **CLS‑1** `_classification.py:366` — example prints `[0.  0. ]`; actual is
  `[0.         0.5999999]`.
- **REG‑1** `_regression.py:739` — `huber_loss` example `0.00499997` → `0.005`.
- **REG‑2** `_regression.py:836` — `log_cosh` example `0.43378806` → `0.4337808`.
- **REG‑3** `_regression.py:925` — `cosine_similarity` example `1.` → `0.9999999`.
- **REG‑4** `_regression.py:125` — `safe_norm` example `2.2360680` (invalid repr) →
  `2.236068`.
- **PW‑2** `_pairwise.py:118` — diagonal `1.` → `1.0000001`.
- **REG‑5** `_regression.py:794` — `log_cosh` "no broadcasting" → "broadcastable".
- **REG‑7** `_regression.py` — document that `log_cosh` requires dimensionless inputs.
- **REG‑9** `_regression.py:175‑176` — clarify that `axis` only applies when
  `reduction` is `'mean'`/`'sum'`.
- **FIR‑7** `_firings.py:700` — widen `dt` type hint/doc to `float or brainunit.Quantity`
  and state the Hz/seconds contract.
- **FIR‑8** `_firings.py:155‑162` — note that plain‑float `width`/`dt` are assumed seconds.
- **FY‑1..3** `_fenchel_young.py:61,147,171` — qualify the non‑negativity claim
  (holds when `max_fun` is a true conjugate and `y` is in its domain), and fix the
  misleading "L2‑regularized" / complex‑inner‑product comments.
- **COR‑3** `_correlation.py:452,553` — wrap the `weighted_correlation` and
  `functional_connectivity_dynamics` examples in `.. code-block:: python` and add
  expected outputs.
- **RNK‑2** `_ranking.py:281,291,297,304` — `ranking_softmax_loss` examples print with
  `print(f"...: {loss:.4f}")` but show no expected output (a strict doctest would fail);
  added the computed output lines (`Loss: 2.2228`, `Batch loss: 0.4968`,
  `Weighted loss: 0.4076`, `Individual losses: [0.31326163 0.68026966]`). The
  module‑level docstring example used `jnp` without importing it — added
  `import jax.numpy as jnp`.
- **LFP‑4** `_lfp.py:259` — note that `jnp.hanning` is a *symmetric* window, unlike
  `scipy.signal.welch`'s periodic default.
- **SM‑2** `_smoothing.py:31` — return annotation `jnp.ndarray` → `jax.Array`.
- **UTIL‑1** `_util.py` — document that `axis` is ignored when `reduction='none'`.
- **CLS‑4** `_classification.py` — note that fractional `gamma` (`0<gamma<1`) with
  saturated logits yields non‑finite gradients (matches optax/fvcore; default
  `gamma=2` is safe).

## Documented‑only (no behaviour change — recorded for completeness)

- **CLS‑3** `assert_is_float`/`assert_is_int` surface a `ValueError` (from
  `u.math.is_float`) rather than their own `TypeError` for raw list inputs.
- **CLS‑5** `softmax_cross_entropy` yields `NaN` for `-inf` logits at a zero‑label
  position (`0 * -inf`); matches optax. Use the integer‑label variant / finite logits.
- **REG‑8** `cosine_similarity` gradient is finite but can be large (~1e7) for
  near‑zero inputs; the docstring wording is softened.
- **RNK‑1** `_ranking._safe_reduce` computes its NaN guard over the whole array; this
  is harmless for the only (whole‑array mean) call site.
- **LFP‑6** `unitary_LFP` `sig_e` default doc is consistent (`3.15 = 2.1 * 1.5`).

---

## Verification plan

Each behaviour fix gets a focused test (reproducing the bug first, per the
project's TDD agreement) in `braintools/metric/_audit_20260619_test.py`; the
docstring‑output fixes are verified by executing the exact example via `doctest`.

**Result:** the full `braintools/metric` suite is green — **357 passed**
(332 baseline + 25 new audit/regression tests). One pre‑existing test
(`TestLFPPhaseCoherence.test_different_frequency_bands`) asserted the LFP‑1 bug
and was updated to the corrected semantics (see LFP‑1). Doctest spot‑checks pass:
regression 46/0, pairwise 7/0, ranking 18/0, classification (edited fns) 36/0.

## Test infrastructure (per user request)

A repo‑root `conftest.py` forces matplotlib's non‑interactive **Agg** backend for
the entire test suite (`MPLBACKEND=Agg` set before import + `matplotlib.use("Agg",
force=True)`), so running the tests never opens GUI image windows and works
headless/CI. The `braintools/metric` tests themselves use no matplotlib; this
covers sibling suites (e.g. `braintools/visualize`) when the whole project is run.
