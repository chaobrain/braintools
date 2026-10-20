# Migrating to braintools 1.0

Since 0.4, every function that moved has a new home in braintools, and the old
names still work as deprecated aliases. **Braintools 1.0.0 removes all of
them**, together with the deprecated arguments listed below. This page lists
every deprecation and its replacement, so code can be updated before the
upgrade.

## Finding deprecated calls in your code

A deprecated name warns when it is *called* (or a deprecated class is
instantiated), never when it is imported. The warning category is
`braintools._deprecation.BraintoolsDeprecationWarning`, and every message has
the same form:

```text
`braintools.metric.firing_rate` is deprecated since 0.4.0 and will be removed in 1.0.0. Use `braintools.analysis.spike.firing_rate` instead.
```

Since 0.4, `BraintoolsDeprecationWarning` is a subclass of `FutureWarning`
(it used to subclass `DeprecationWarning`). Python shows `FutureWarning` by
default, so the warnings now appear even when the deprecated call is made from
library code rather than from your script. Filters written for
`DeprecationWarning` no longer match them.

To turn every braintools deprecation into an error, so that each remaining
call site fails loudly:

```bash
python -W error::FutureWarning my_script.py
```

or, from Python (for example in a test suite's `conftest.py`):

```python
import warnings
from braintools._deprecation import BraintoolsDeprecationWarning

warnings.simplefilter('error', BraintoolsDeprecationWarning)
```

With pytest, add a filter to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
filterwarnings = [
    'error:`braintools\.[^`]*` is deprecated since:FutureWarning',
]
```

To silence them instead (for example in a downstream package that has not
migrated yet), use
`warnings.filterwarnings('ignore', category=BraintoolsDeprecationWarning)`.

## Every deprecation

The table lists every deprecation registered in braintools. A `function` or
`class` entry is an alias of the replacement. An `argument` entry is a
deprecated argument or argument value of a function that stays. A `feature`
entry is an object that is deprecated without a direct replacement.

| Deprecated | Replacement | Kind | Since | Removed in |
|---|---|---|---|---|
| `braintools.BernoulliEncoder` | `braintools.spike.BernoulliEncoder` | class | 0.4.0 | 1.0.0 |
| `braintools.cogtask.utils.firing_rate` | `braintools.cogtask.utils.rate_to_prob` | function | 0.4.0 | 1.0.0 |
| `braintools.conn.MorphologyDistance(decay_function=str)` | `pass a braintools.init.*Profile instance` | argument | 0.4.0 | 1.0.0 |
| `braintools.DeltaEncoder` | `braintools.spike.DeltaEncoder` | class | 0.4.0 | 1.0.0 |
| `braintools.file.load_matfile(header_info=...)` | `braintools.file.load_matfile(include_header=not header_info)` | argument | 0.2.0 | 1.0.0 |
| `braintools.init.Beta(unit=...)` | `braintools.init.Beta(...) with the unit attached to the value arguments` | argument | 0.1.0 | 1.0.0 |
| `braintools.init.Constant(unit=...)` | `braintools.init.Constant(...) with the unit attached to the value arguments` | argument | 0.1.0 | 1.0.0 |
| `braintools.init.DeltaOrthogonal(unit=...)` | `braintools.init.DeltaOrthogonal(...)(size) * unit` | argument | 0.1.0 | 1.0.0 |
| `braintools.init.Exponential(unit=...)` | `braintools.init.Exponential(...) with the unit attached to the value arguments` | argument | 0.1.0 | 1.0.0 |
| `braintools.init.Gamma(unit=...)` | `braintools.init.Gamma(...) with the unit attached to the value arguments` | argument | 0.1.0 | 1.0.0 |
| `braintools.init.LogNormal(unit=...)` | `braintools.init.LogNormal(...) with the unit attached to the value arguments` | argument | 0.1.0 | 1.0.0 |
| `braintools.init.Normal(unit=...)` | `braintools.init.Normal(...) with the unit attached to the value arguments` | argument | 0.1.0 | 1.0.0 |
| `braintools.init.Orthogonal(unit=...)` | `braintools.init.Orthogonal(...)(size) * unit` | argument | 0.1.0 | 1.0.0 |
| `braintools.init.TruncatedNormal(unit=...)` | `braintools.init.TruncatedNormal(...) with the unit attached to the value arguments` | argument | 0.1.0 | 1.0.0 |
| `braintools.init.Uniform(unit=...)` | `braintools.init.Uniform(...) with the unit attached to the value arguments` | argument | 0.1.0 | 1.0.0 |
| `braintools.init.Weibull(unit=...)` | `braintools.init.Weibull(...) with the unit attached to the value arguments` | argument | 0.1.0 | 1.0.0 |
| `braintools.LatencyEncoder` | `braintools.spike.LatencyEncoder` | class | 0.4.0 | 1.0.0 |
| `braintools.metric.burst_synchrony_index` | `braintools.analysis.spike.burst_synchrony_index` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.coherence_analysis` | `braintools.analysis.spectral.coherence_analysis` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.correlation_index` | `braintools.analysis.spike.correlation_index` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.cross_correlation` | `braintools.analysis.spike.cross_correlation` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.current_source_density` | `braintools.analysis.field.current_source_density` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.firing_rate` | `braintools.analysis.spike.firing_rate` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.functional_connectivity` | `braintools.analysis.population.functional_connectivity` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.functional_connectivity_dynamics` | `braintools.analysis.population.functional_connectivity_dynamics` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.lfp_phase_coherence` | `braintools.analysis.spectral.lfp_phase_coherence` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.matrix_correlation` | `braintools.analysis.stats.matrix_correlation` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.phase_amplitude_coupling` | `braintools.analysis.spectral.phase_amplitude_coupling` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.phase_locking_value` | `braintools.analysis.spike.phase_locking_value` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.power_spectral_density` | `braintools.analysis.spectral.power_spectral_density` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.raster_plot` | `braintools.spike.SpikeTrains.from_dense` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.spectral_entropy` | `braintools.analysis.spectral.spectral_entropy` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.spike_time_tiling_coefficient` | `braintools.analysis.spike.spike_time_tiling_coefficient` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.spike_train_synchrony` | `braintools.analysis.spike.spike_train_synchrony` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.theta_gamma_coupling` | `braintools.analysis.spectral.theta_gamma_coupling` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.unitary_LFP` | `braintools.analysis.field.unitary_LFP` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.van_rossum_distance` | `braintools.analysis.spike.van_rossum_distance` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.victor_purpura_distance` | `braintools.analysis.spike.victor_purpura_distance` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.voltage_fluctuation` | `braintools.analysis.population.voltage_fluctuation` | function | 0.4.0 | 1.0.0 |
| `braintools.metric.weighted_correlation` | `braintools.analysis.stats.weighted_correlation` | function | 0.4.0 | 1.0.0 |
| `braintools.optim.Adam(beta1=...)` | `braintools.optim.Adam(betas=(beta1, beta2))` | argument | 0.1.0 | 1.0.0 |
| `braintools.optim.Adam(beta2=...)` | `braintools.optim.Adam(betas=(beta1, beta2))` | argument | 0.1.0 | 1.0.0 |
| `braintools.PoissonEncoder` | `braintools.spike.PoissonEncoder` | class | 0.4.0 | 1.0.0 |
| `braintools.PopulationEncoder` | `braintools.spike.PopulationEncoder` | class | 0.4.0 | 1.0.0 |
| `braintools.RankOrderEncoder` | `braintools.spike.RankOrderEncoder` | class | 0.4.0 | 1.0.0 |
| `braintools.RateEncoder` | `braintools.spike.RateEncoder` | class | 0.4.0 | 1.0.0 |
| `braintools.spike.PoissonEncoder(time_window=...)` | `braintools.spike.PoissonEncoder() without time_window (it has no effect)` | argument | 0.4.0 | 1.0.0 |
| `braintools.SpikeCountEncoder` | `braintools.spike.SpikeCountEncoder` | class | 0.4.0 | 1.0.0 |
| `braintools.StepCurrentEncoder` | `braintools.spike.StepCurrentEncoder` | class | 0.4.0 | 1.0.0 |
| `braintools.TemporalEncoder` | `braintools.spike.TemporalEncoder` | class | 0.4.0 | 1.0.0 |
| `braintools.trainer.GradientClipCallback` | `Trainer(gradient_clip_val=...)` | feature | 0.4.0 | 1.0.0 |
| `braintools.trainer.Trainer(benchmark=...)` | `the default (XLA autotunes automatically)` | argument | 0.4.0 | 1.0.0 |
| `braintools.trainer.Trainer(deterministic=...)` | `Trainer(seed=...)` | argument | 0.4.0 | 1.0.0 |
| `braintools.trainer.Trainer(precision=...)` | `explicit dtype casts of the model and data` | argument | 0.4.0 | 1.0.0 |
| `braintools.trainer.Trainer(strategy=...)` | `the default strategy='auto'` | argument | 0.4.0 | 1.0.0 |

The top-level `braintools.spike_bitwise*` names are also kept as aliases of
`braintools.spike.bitwise*`. They do not warn yet, because other packages in
the ecosystem bind them at import time, but new code should use
`braintools.spike.bitwise_or` and the other `braintools.spike.bitwise*` names.

## `braintools.metric` → `braintools.analysis`

The measures for analysing neural activity moved out of `braintools.metric`
into the subpackages of `braintools.analysis`. The functions themselves are
unchanged: same arguments, same results. Only the import path changes.

| Old (`braintools.metric.*`) | New |
|---|---|
| `firing_rate`, `victor_purpura_distance`, `van_rossum_distance`, `spike_train_synchrony`, `burst_synchrony_index`, `phase_locking_value`, `spike_time_tiling_coefficient`, `correlation_index`, `cross_correlation` | `braintools.analysis.spike` |
| `power_spectral_density`, `coherence_analysis`, `phase_amplitude_coupling`, `theta_gamma_coupling`, `spectral_entropy`, `lfp_phase_coherence` | `braintools.analysis.spectral` |
| `unitary_LFP`, `current_source_density` | `braintools.analysis.field` |
| `voltage_fluctuation`, `functional_connectivity`, `functional_connectivity_dynamics` | `braintools.analysis.population` |
| `matrix_correlation`, `weighted_correlation` | `braintools.analysis.stats` |

```python
# Before
rate = braintools.metric.firing_rate(spikes, width=5.0)

# After
rate = braintools.analysis.spike.firing_rate(spikes, width=5.0)
```

`braintools.metric.raster_plot(sp_matrix, times)` has no one-to-one
replacement. Use `braintools.spike.SpikeTrains.from_dense` to hold the spikes,
or compute the same arrays directly:

```python
t, i = numpy.nonzero(sp_matrix > 0)
neuron_indices, spike_times = i, times[t]  # == metric.raster_plot(sp_matrix, times)
```

Plotting a raster is `braintools.visualize.raster_plot`, which is not
deprecated.

The training losses and metrics (cross-entropy, regression losses, ranking
losses, `cosine_similarity` and the rest) stay in `braintools.metric`.

## Top-level encoders → `braintools.spike`

The ten spike encoders moved from the top level to `braintools.spike`:
`LatencyEncoder`, `RateEncoder`, `PoissonEncoder`, `PopulationEncoder`,
`BernoulliEncoder`, `DeltaEncoder`, `StepCurrentEncoder`, `SpikeCountEncoder`,
`TemporalEncoder` and `RankOrderEncoder`.

```python
# Before
encoder = braintools.PoissonEncoder()

# After
encoder = braintools.spike.PoissonEncoder()
```

Instances built either way pass `isinstance` checks against both names. The
`time_window` argument of `PoissonEncoder` never had an effect and is
deprecated; drop it.

## `braintools.trainer` is frozen

Since 0.4, `braintools.trainer` gets maintenance fixes only, with no new
features. Features that never did anything are deprecated:

- `GradientClipCallback` never clipped gradients. Use
  `Trainer(gradient_clip_val=...)`.
- `Trainer(precision=...)` other than 32: cast the model and data to the dtype
  you want instead.
- `Trainer(deterministic=...)`: pass `Trainer(seed=...)`.
- `Trainer(benchmark=...)`: drop it; XLA autotunes automatically.
- `Trainer(strategy=...)` other than `'auto'`: drop it.

## `conn.MorphologyDistance(decay_function=str)` → distance profiles

`MorphologyDistance` took the decay function as a string. Pass a
`braintools.init` distance profile instead; it carries its own length scale,
so `sigma` is omitted.

| Old | New |
|---|---|
| `decay_function='gaussian', sigma=s` | `decay_function=braintools.init.GaussianProfile(s)` (or the default, `decay_function=None, sigma=s`) |
| `decay_function='exponential', sigma=s` | `decay_function=braintools.init.ExponentialProfile(s)` |
| `decay_function='linear', sigma=s` | `decay_function=braintools.init.LinearProfile(s)` |

## `cogtask.utils.firing_rate` → `rate_to_prob`

`braintools.cogtask.utils.firing_rate(base, dt)` converts a rate in Hz to a
spike probability per time step (`base * dt / 1000`), not a firing rate. It is
renamed `braintools.cogtask.utils.rate_to_prob`, with the same arguments and
result, so it is no longer confused with `braintools.analysis.spike.firing_rate`.

## Deprecated `unit=` arguments and `load_matfile(header_info=...)`

- The `unit=` argument of `Constant`, `Uniform`, `Normal`, `LogNormal`,
  `Gamma`, `Exponential`, `TruncatedNormal`, `Beta` and `Weibull` in
  `braintools.init` is deprecated: attach the unit to the values instead, for
  example `Normal(0.0 * u.mV, 1.0 * u.mV)` rather than
  `Normal(0.0, 1.0, unit=u.mV)`. `ZeroInit(unit=...)` is not deprecated.
- `Orthogonal(unit=...)` and `DeltaOrthogonal(unit=...)` are deprecated:
  multiply the generated weights by the unit.
- `braintools.file.load_matfile(header_info=...)` meant the inverse of its
  name. Use `include_header=not header_info`.

Before 0.4 these warned with a plain `DeprecationWarning`. They now use
`BraintoolsDeprecationWarning` like every other deprecation.
