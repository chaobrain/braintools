# Release Notes


## Version 0.2.0 (2026-06-18)

This is a codebase-wide correctness, test-coverage, and documentation
release. Nearly every major module — `metric`, `trainer`, `optim`,
`visualize`, `surrogate`, `quad`, `init`, `conn`, and `file` — received a
dedicated static audit; the findings were resolved with fixes and locked in
behind comprehensive new test suites that raise per-module coverage to
roughly 92–100%. The audits uncovered and corrected genuine mathematical and
numerical bugs that had previously gone unnoticed — inverted surrogate-gradient
formulas, a sign error in `nll_loss`, coherence metrics that were identically
one, He-initialization variance off by a factor of two, and unit-handling
crashes in the ODE/SDE integrators under newer `saiunit`. Documentation was
realigned across the board so that docstring examples, notebooks, and the API
reference are runnable and accurate. Alongside the fixes, the release adds new
public API in `file`, `trainer`, `optim`, `metric`, and `init`, and modernizes
the CI and build configuration. The minor-version bump reflects the breadth of
behavioral corrections rather than any intentional break in compatibility.

### Highlights

- **Library-wide audit pass**: every audited module ships corrected behavior
  plus a dedicated regression/correctness test suite, lifting coverage to
  ~92–100% and replacing previously self-referential tests with reference-value
  checks.
- **Real numerical-correctness fixes**: corrected surrogate-gradient formulas
  (`GaussianGrad`, `Arctan`, `ERF`, `PiecewiseQuadratic`, …), an `nll_loss`
  sign error, LFP coherence that was identically `1`, He/Kaiming initialization
  variance that was off by 2×, and integrator unit handling under `saiunit`.
- **New public API**: `braintools.file.save_matfile`, gradient accumulation and
  name-based parameter freezing in `braintools.trainer`, a line-search API for
  `LBFGS` in `braintools.optim`, and exported `safe_norm` / pairwise-cosine
  helpers in `braintools.metric`.
- **Modernized infrastructure**: Python 3.14 configuration, `codecov-action`
  v5 → v7, enforced LF line endings, and removal of the broken `scienceplots`
  integration that was failing CI.

### Added

#### `braintools.file`

- **`save_matfile`**: save a dictionary to a MATLAB ``.mat`` file, the
  counterpart to `load_matfile` (#104).

#### `braintools.trainer`

- **Gradient accumulation**: accumulate gradients across micro-batches,
  numerically equal to a single full-batch step (#112).
- **Name-based parameter freezing**: parameters selected by name are now
  genuinely frozen by the trainer (#112).

#### `braintools.optim`

- **`LBFGS.update(grads, value=, value_fn=)`**: public line-search API
  supporting zoom and backtracking line searches (#111).

#### `braintools.metric`

- **`safe_norm`** is now exported, along with the pairwise helpers
  `pairwise_cosine_similarity` and `pairwise_cosine_distance` (built on a
  gradient-safe norm). `huber_loss`, `log_cosh`, and `l2_loss` gain `axis` and
  `reduction` arguments (#108).

#### `braintools.init`

- **`VarianceScaling`** and the `Initializer` alias are now part of the public
  API, and `ExponentialProfile` gains a `decay_constant` argument (#106).

### Changed

- **`braintools.optim`** (#111):
  - `ChainedScheduler` combines factors multiplicatively, matching PyTorch;
    `PiecewiseConstantSchedule` is documented and treated as absolute LR values;
    `ReduceLROnPlateau`'s incompatibility with `ChainedScheduler` / `SequentialLR`
    is documented.
  - Multi-group `step()` updates the default group through the main `tx` and
    each added group through its own `tx`, while parameters outside any added
    group still update.
  - SciPy backend casts `x0` / `jac` / `bounds` to float64 for the TNC/SLSQP
    Cython kernels and skips `jac` for gradient-free methods; the Nevergrad
    backend gains reproducible seeding and an all-NaN recommendation fallback.
- **`braintools.visualize`** (#110):
  - `line_plot` / `raster_plot` draw onto the passed `Axes` rather than the
    pyplot state machine, so labels, limits, and titles land on the right
    subplot.
  - `animate_1D` / `animate_2D` fall back to `dt=1.0` outside a brainstate `dt`
    context instead of raising `KeyError`; `static_vars` accepts arrays, lists,
    or labeled dicts.
  - `apply_style` validates the style name and returns a context manager that
    restores `rcParams` on exit; `brain_surface_3d` uses `plt.get_cmap`
    (replacing the `plt.cm.get_cmap` removed in Matplotlib 3.11).
- **`braintools.init`** (#106):
  - `TruncatedNormal` uses a jit-traceable, backend-agnostic inverse-CDF
    sampler (`ndtr` / `ndtri`) in place of `scipy.stats.truncnorm`, and
    `VarianceScaling` compensates for the truncated standard deviation so the
    achieved variance matches the target.
  - Distance profiles use unit-aware math throughout; the per-call `unit`
    argument on basic distributions is deprecated in favor of unit-bearing
    quantities.
- **`braintools.conn`** (#105):
  - Per-edge weights and delays are aligned to CSR row order in `weight2csr` /
    `delay2csr`; `__call__` result caching is keyed on `(pre_size, post_size,
    position ids)` so changed sizes recompute, and `ScaledConnectivity` no
    longer mutates the base connectivity's cached result.
  - Spatial/kernel connectivities gain autapse control via
    `allow_self_connections`, and `ExponentialProfile` adopts the
    `decay_constant` API.
- **`braintools.file`** (#104):
  - The matfile reader replaces the deprecated `scipy.io.matlab.mio5_params`
    with the public `mat_struct`, detects MATLAB v7.3 (HDF5) files and raises an
    actionable `NotImplementedError`, and renames the inverted `header_info`
    flag to `include_header` (deprecated alias retained).
  - `msgpack_load` plumbs `max_size` through (None = unlimited, replacing the
    hard 10 GB cap) and `msgpack_save` returns its filename and writes via a
    unique temp file so concurrent saves no longer clobber each other.
- **`braintools.trainer`** (#112): single forward pass per training step via
  `grad(..., has_aux)`; `EarlyStopping` `min_delta` keyed off `mode` and gated
  by `min_epochs`; validation/test metrics prefixed via `_prefixed()`
  (eliminating `val_val_loss`); `seed` seeds both NumPy and brainstate; honest
  validation and warnings for `precision`, `deterministic`/`benchmark`,
  multiple optimizers, and distributed strategies.

### Fixed

- **`braintools.surrogate`** (#109) — formula corrections, caught after the
  self-referential test suite was replaced with reference-value checks:
  - `GaussianGrad` exponent corrected to `-x²/(2σ²)` (the σ dependence was
    inverted); `Arctan.surrogate_fun` rebuilt around `arctan` (it had misused
    `arctan2`, leaving the range outside `[0, 1]`); `ERF.surrogate_fun`
    corrected to be increasing.
  - `PiecewiseQuadratic.surrogate_grad` is now the continuous triangle
    `a − a²|x|`; `PiecewiseLeakyRelu` central slope fixed to `1/(2w)`;
    `QPseudoSpike.surrogate_fun` rewritten as a finite antiderivative;
    `SquarewaveFourierSeries` off-by-one term count fixed.
  - `S2NN` / `LogTailedRelu` guard dead `where`-branch denominators to avoid
    NaN gradients.
- **`braintools.metric`** (#108, #113):
  - `nll_loss` sign error fixed (with N-D support added); KL divergence made
    gradient-safe via the double-`where` pattern; `ctc_loss` uses `jax.random`
    instead of a nonexistent `jnp.random`.
  - LFP fixes: corrected Welch PSD normalization, magnitude-squared coherence
    (which had been identically `1`), Tort PAC, and `current_source_density`
    axis/conductivity/units. `lfp_phase_coherence` is vectorized and its PLV
    output is bounded to `[0, 1]` with an exactly-`1` diagonal, removing float32
    roundoff that pushed values marginally above one (#113).
  - `firing_rate` width/dt scaling corrected; `spike_train_synchrony` made
    symmetric; `cross_correlation`, `voltage_fluctuation`, and the loss
    reductions gained shape / zero-variance / zero-weight guards. The misspelled
    `_pariwise` module was renamed to `_pairwise`.
- **`braintools.quad`** (#107) — all Butcher tableaux verified correct; bugs
  were concentrated in unit handling and noise sampling:
  - `ode_expeuler_step` / `sde_expeuler_step` divide the diagonal Jacobian by
    the *state* unit so `dt·A` is dimensionless (previously crashed under
    `saiunit >= 0.4`); `sde_expeuler_step` samples the Brownian increment from
    the shape of `y`; `sde_euler_step` uses `u.math.sqrt(dt)` for unitful `dt`.
  - `ode_dopri5_step` exploits the FSAL property to drop one redundant stage,
    computing `k7` only when `return_error=True`.
- **`braintools.init`** (#106): `KaimingUniform` / `KaimingNormal` use the
  correct He variance (`scale=2.0` for ReLU, `2/(1+slope²)` for leaky ReLU; it
  had been `sqrt(2)`, giving half the intended variance). `param()` restores the
  `State` value into the returned parameter, and `_to_size` rejects boolean
  sizes.
- **`braintools.conn`** (#102): `CompositeConnectivity._union` uses explicit
  `is not None` checks instead of array truthiness, fixing a
  `ValueError: truth value of an array ... is ambiguous` that broke CI.
- **`braintools.visualize`** (#110): `remove_axis()` with no spine names blanks
  the panel; `firing_rate_map` handles non-square `grid_size` and edge points;
  Spearman correlation builds a 2×2 matrix for exactly two features; ~26 broken
  docstring examples were corrected.
- **`braintools.file`** (#104): checkpoint restore validates array shape and
  honors `mismatch` (wrong-shaped arrays were silently loaded), dispatches to
  the most-specific registered subclass, and stops misfiring the
  namedtuple-envelope heuristic on genuine namedtuples; `AsyncManager` surfaces
  background-save failures instead of swallowing them.

### Infrastructure

- **Dependencies**: bumped `codecov/codecov-action` from v5 to v7 (#101).
- **Python 3.14**: project and tool configuration updated for Python 3.14.
- **Line endings**: enforced LF via `.gitattributes`, with binary rules for
  image/data assets and `eol` rules for `.ps1` / `.sh`; three CRLF-committed
  files were renormalized with no content change.
- **`scienceplots` removal**: the unused `_style.py` scienceplots integration
  was removed — it depended on an undeclared package and on internal Matplotlib
  APIs removed in newer releases, which was breaking CI on Python 3.13 (#103).
- **Audit reports**: each module audit committed its findings under
  `docs/braintools-<module>-issues-found-20260618.md`.


## Version 0.1.10 (2026-06-09)

This release adds two forward-mode second-order optimizers — `SOFO` and
`SOFOScan` — to `braintools.optim`, and hardens the `braintools.cogtask`
task engine so that conditional combinators, categorical labels, and
metadata batching behave correctly under `brainstate.transform.jit` and
`brainstate.transform.vmap2`. The package now ships inline type information
(PEP 561), test coverage is raised to ~92%, and documentation links and
assets are migrated to the new `brainx.chaobrain.com` host.

### Highlights

- **New optimizers `SOFO` and `SOFOScan`**: Second-Order Forward-mode
  Optimization for feedforward and recurrent models. Both build a
  Generalised Gauss-Newton matrix in a random tangent subspace from
  forward-mode JVPs and apply the resulting direction through the standard
  optax update path, so learning-rate schedules, momentum, weight decay, and
  gradient clipping continue to work unchanged.
- **Hardened `cogtask` dispatch**: `Switch` now works in both eager and
  traced execution, `While` fails loudly on unsupported traced conditions,
  and a new `num_classes` parameter decouples categorical-head sizing from
  `num_outputs`.
- **PEP 561 typing**: `braintools` ships a `py.typed` marker and inline
  annotations on its public API, so downstream static type checkers consume
  its types directly.

### Added

#### `braintools.optim` — forward-mode second-order optimizers

- **`SOFO`**: Second-Order Forward-mode Optimization for a model
  `model(inputs) -> predictions` paired with `loss_fn(predictions, targets)`.
  It samples random tangent vectors, takes forward-mode JVPs through the
  model and loss, builds a damped Generalised Gauss-Newton system in the
  random subspace, solves it, and projects the solution back to parameter
  space. Supports `'mse'` and `'ce'` loss forms, a configurable
  `tangent_size` and `damping`, `momentum` / `nesterov`, decoupled
  `weight_decay`, and norm/value gradient clipping.
- **`SOFOScan`**: a recurrent variant for a stateful one-step cell
  `rnn_cell(latent, inputs) -> (new_latent, output)`. The cell is scanned
  over the input sequence with `brainstate.transform.scan`, and forward-mode
  JVPs propagate the tangents through `lax.scan`, accumulating the
  Gauss-Newton matrix over every `(timestep, batch)` sample before a single
  solve. Both optimizers are exported from `braintools.optim` and documented
  in the API reference.

#### `braintools.cogtask`

- **`Task` categorical sizing**: a new `num_classes` argument, decoupled from
  `num_outputs`, sizes categorical output heads independently of the raw
  output dimension.
- **`Task` feature ergonomics**: `Task` now accepts a lone `Feature` in place
  of a `FeatureSet`, and requires features to be supplied whenever `phases`
  are given.
- **`Task` time step**: `Task` and `make_task` accept an optional `dt`
  argument. When set, it is pinned around trial generation via
  `brainstate.environ.context`, so phase durations and buffer sizes are
  computed against that `dt` and the reported `dt` stays consistent
  regardless of the ambient environment. When omitted, the ambient
  `brainstate.environ.get_dt()` is used (unchanged behaviour).

#### Typing

- **PEP 561 support**: a `braintools/py.typed` marker is shipped via package
  data, and the top-level public API — spike bitwise ops, spike encoders
  (with implicit-`Optional` defaults fixed), tree utilities, and `_misc`
  helpers — now carries resolvable inline annotations.

### Changed

- **`cogtask` conditional dispatch**:
  - `Switch` uses dual-mode packed dispatch — a concrete `key in cases`
    lookup in eager mode and a `lax.switch` over ordered branches under
    `jit` / `vmap` — and coerces 0-d concrete array keys (e.g.
    `ctx.rng.choice(...)` selectors) so eager `sample_trial` no longer
    raises `unhashable type: 'ArrayImpl'`.
  - `While` raises a clear `NotImplementedError` for data-dependent (traced)
    conditions under `jit` / `vmap` instead of surfacing a cryptic
    `TracerBoolConversionError`.
- **Documentation links**: chaobrain-ecosystem documentation URLs
  (`brainstate`, `brainunit`, `braincell`, `brainmass`, `brainevent`,
  `braintrace`, `braintools`, and related packages) were rewritten from
  `*.readthedocs.io` to the new `brainx.chaobrain.com` host, stripping
  `/latest`, `/en/latest`, and `/en/stable` path prefixes and
  `?badge=latest` query strings. Third-party ReadTheDocs links are left
  intact.
- **README logo**: the project logo is now served from
  `brainx.chaobrain.com` as WebP instead of a raw GitHub asset.

### Fixed

- **`braintools.cogtask`**:
  - Categorical labels that are statically out of range for the declared
    `num_classes` are now validated and rejected up front.
  - Packed-mode phases expose `phase_start` / `phase_end` before `on_enter`
    runs, matching the contract already provided in fixed-length mode.
  - String leaves are dropped from batched metadata so `return_meta` works
    correctly under `brainstate.transform.vmap2`.
  - Minor fixes to the input encoder and the working-memory task library.
  - Added regression tests covering all of the above.
- **`braintools.trainer`**:
  - `LightningModule.device` no longer raises on array-backed parameters;
    `Array.devices()` returns a set, which is now handled correctly (#92).
  - `ModelCheckpoint` saves through `braintools.file.msgpack_save` instead of
    a state-restore helper, so checkpoints are written correctly (#95).
- **`braintools.visualize`**:
  - `animate_2D` reshapes the value grid before drawing the first frame,
    fixing a `pcolor` crash on the initial step (#93).
  - `correlation_matrix(method='kendall')` builds the matrix pairwise instead
    of passing a 2-D array to `kendalltau` (#94).
  - `remove_axis` uses `ax.spines` instead of the non-existent `ax.spine` (#96).
  - `create_neural_colormap` / `brain_colormaps` register with `force=True`,
    making them idempotent rather than raising on re-use (#97).
  - `roc_curve` / `precision_recall_curve` resolve `np.trapezoid` when
    available (falling back to `np.trapz`), fixing an `AttributeError` on
    NumPy >= 2.4 where `np.trapz` was removed (#99).

### Infrastructure

- **Publish workflow**: reads the package version directly from
  `braintools/_version.py` (the single source of truth) and verifies that the
  release tag matches before publishing.
- **Docs deployment**: the `push: main` trigger was removed; documentation is
  now deployed only on a GitHub release (`released`) or via a manual
  `workflow_dispatch`.
- **Type-check workflow**: a new Type Check workflow runs `mypy` over the
  annotated public surface, backed by a `[tool.mypy]` configuration and a
  `type-check` optional-dependency group.
- **Test coverage**: new test suites cover the previously-untested trainer,
  visualize, file, and surrogate modules, raising overall coverage to ~92%.
  CI runs `pytest` with `--cov` and uploads results to Codecov, and the
  README carries a coverage badge. `tqdm` and `rich` were added to the
  `testing` extra so the progress-bar tests run in CI.


## Version 0.1.9 (2026-05-21)

This release introduces `braintools.cogtask`, a composable framework for
constructing cognitive tasks for neural-network training and computational
neuroscience experiments. It also extends `braintools.init` to accept
`brainstate.nn.Param`, adds official Python 3.14 support, and refreshes
documentation and CI infrastructure.

### Highlights

- **New module `braintools.cogtask`**: a phase-based DSL for building
  trial-structured cognitive tasks, with a library of pre-built paradigms
  drawn from the cognitive-neuroscience literature.
- **Variable-length trials under JIT/vmap**: shape-stable packed-mode trial
  generation enables `batch_sample` to remain compatible with
  `brainstate.transform.jit` and `brainstate.transform.vmap2` for tasks whose
  phases have data-dependent durations.
- **Python 3.14 support**: CI matrix and project metadata updated; minimum
  supported Python remains 3.11.

### Added

#### `braintools.cogtask` — composable cognitive task framework

- **Core API**: `Task`, `TaskConfig`, `Context`, `Phase`, `Sequence`,
  `Repeat`, `Parallel`, conditional combinators `If` / `Switch` / `While`,
  and the `concat` helper for sequential composition.
- **Phase primitives**: `Fixation`, `Delay`, `Stimulus`, `Response`, `Cue`,
  plus the variable-length `VariableDuration` phase whose timestep budget is
  resolved per-trial from a context entry.
- **Feature and label utilities**: `Feature`, `circular`, `one_hot`, and a
  set of input encoders/decoders for constructing task observations and
  targets in a typed, composable way.
- **Pre-built task library** spanning three domains:
  - *Decision making*: `PerceptualDecisionMaking`,
    `PerceptualDecisionMakingDelayResponse`, `ContextDecisionMaking`,
    `SingleContextDecisionMaking`, `PulseDecisionMaking`.
  - *Working memory*: `DelayMatchSample`, `DualDelayMatchSample`,
    `DelayComparison`, `DelayMatchCategory`, `DelayPairedAssociation`,
    `GoNoGo`, `IntervalDiscrimination`, `PostDecisionWager`, `ReadySetGo`,
    `DelayDirectionReproduction`, `ImmediateDirectionReproduction`,
    `DelayDirectionClassification`, `ImmediateDirectionClassification`.
  - *Motor and reasoning*: `AntiReach`, `Reaching1D`, `EvidenceAccumulation`,
    `HierarchicalReasoning`, `ProbabilisticReasoning`.
- **Variable-length trial sequences**:
  - `VariableDuration` phases declare a Python `max_steps` (used as the
    buffer slot size) and report the realised trial length via the traced
    `step_count` field.
  - `Task` auto-detects variable-length phase trees via
    `phase_tree_is_variable(phases)`. When any phase declares
    `is_variable = True`, `sample_trial` allocates buffers of size
    `task.max_trial_duration()`, writes only the front `t_cursor` timesteps,
    and zero-fills the trailing positions while setting the mask to `False`.
  - `Task.batch_sample(..., return_mask=True)` returns `(X, Y, mask)` with
    shape-stable buffers under `brainstate.transform.jit` and
    `brainstate.transform.vmap2`. Fixed-length tasks remain unaffected;
    `return_mask=True` on a fixed task yields an all-`True` mask.
  - `If` uses `jax.lax.cond` so both branches contribute shape-stable
    output; `Switch` and `While` use Python dispatch (concrete keys /
    Python `bool` conditions) and zero-pad to the static maximum.
  - `HierarchicalReasoning`, `IntervalDiscrimination`, and `ReadySetGo`
    have been migrated to `VariableDuration` and are now usable under
    `batch_sample` with masking — previously they were valid only via
    `sample_trial` and were not vmap-safe.
  - Duration samplers `TruncExp` and `UniformDuration` advertise
    `is_variable = True` and expose `max_value()` / `min_value()` so phases
    can size their slots statically from sampler bounds.

#### Other additions

- **`braintools.init.param`**: now accepts `brainstate.nn.Param` instances
  in addition to plain initializers, enabling reuse of pre-built parameter
  objects when constructing layers.
- **`.gitattributes`**: normalises line endings for text files to keep
  diffs and tooling consistent across platforms.

### Changed

- **Python support**: project metadata, CI matrix, and classifiers updated
  to include Python 3.14. Minimum supported Python remains 3.11.
- **Documentation hosting**: docs are now self-hosted at
  `brainx.chaobrain.com/braintools/`; the documentation deployment workflow
  publishes on GitHub release events, while pushes to `main` run a
  build-only verification step.
- **Documentation dependencies**: bumped `sphinx` (`>=5` → `>=9.0.4`),
  `sphinx-book-theme` (`>=1.0.1` → `>=1.2.0`),
  `sphinx-copybutton` (`>=0.5.0` → `>=0.5.2`),
  `jupyter-sphinx` (`>=0.3.2` → `>=0.5.3`), and `brainx-sphinx-header`.

### Fixed

- **`braintools.cogtask` end-to-end correctness pass** (introduced together
  with the module):
  - Renamed `cogtask/typing.py` to `cogtask/_typing.py` so the local module
    no longer shadows the stdlib `typing` when tests run from inside the
    package; updated the absolute import in `feature.py` to the relative
    `from ._typing import Data`.
  - Added the missing `noise_sigma` argument and attribute to
    `PerceptualDecisionMaking`,
    `PerceptualDecisionMakingDelayResponse`, `ContextDecisionMaking`,
    `SingleContextDecisionMaking`, `AntiReach`, `Reaching1D`,
    `EvidenceAccumulation`, `DelayPairedAssociation`, `GoNoGo`, and
    `PostDecisionWager`, which previously raised `AttributeError` as soon
    as `define_phases` ran.
  - Removed a duplicate `Task.__repr__` and an undocumented `TaskLoader`
    symbol from the public docs.
- **Phase engine**: added an `IS_COMPOUND` flag on `Phase` and a uniform
  `children()` traversal hook. `execute_phase` now dispatches
  `Sequence`/`Repeat`/`Parallel`/`If`/`Switch`/`While` to their own
  `execute()` methods; previously, `If`/`Switch`/`While` silently no-op'd.
  `Parallel.execute` now gives each child its own
  `[phase_start, phase_start + duration)` window.
- **Distance-profile tests**: representation-equality checks corrected so
  the test suite is stable across NumPy/JAX representations.

### Infrastructure

- Bumped GitHub Actions: `actions/checkout` 4 → 6,
  `actions/setup-python` 5 → 6, `actions/download-artifact` 5 → 8,
  `actions/upload-artifact` 6 → 7, `appleboy/ssh-action` 1.2.0 → 1.2.5,
  `appleboy/scp-action` 0.1.7 → 1.0.0.
- Refactored version management: a dedicated `braintools/_version.py`
  module is now the single source of truth, and `pyproject.toml` resolves
  the package version via `attr = "braintools._version.__version__"`.


## Version 0.1.7 (2026-01-05)

### Major Features

#### New Training Framework (`braintools.trainer`)
- **PyTorch Lightning-like training API** for JAX-based neural network training with comprehensive features:
  - **LightningModule**: Base class for defining training models with `training_step()`, `validation_step()`, and `configure_optimizers()` hooks
  - **Trainer**: Orchestration class for managing training loops, epochs, and device placement
  - **TrainOutput/EvalOutput**: Structured output types for training and evaluation results

#### Callbacks System
- **10+ built-in callbacks** for customizing training behavior:
  - `ModelCheckpoint`: Automatic model saving based on monitored metrics
  - `EarlyStopping`: Stop training when metrics plateau
  - `LearningRateMonitor`: Track and log learning rate changes
  - `GradientClipCallback`: Gradient clipping for training stability
  - `Timer`: Track training time
  - `RichProgressBar` / `TQDMProgressBar`: Visual progress indicators
  - `LambdaCallback` / `PrintCallback`: Custom callback utilities

#### Logging Backends
- **6 pluggable logging backends**:
  - `TensorBoardLogger`: TensorBoard integration
  - `WandBLogger`: Weights & Biases integration
  - `CSVLogger`: Simple CSV file logging
  - `NeptuneLogger`: Neptune.ai integration
  - `MLFlowLogger`: MLFlow integration
  - `CompositeLogger`: Combine multiple loggers

#### Data Loading Utilities
- **JAX-compatible data loading** with distributed support:
  - `DataLoader` / `DistributedDataLoader`: Efficient batch loading
  - `Dataset`, `ArrayDataset`, `DictDataset`, `IterableDataset`: Dataset abstractions
  - `Sampler`, `RandomSampler`, `SequentialSampler`, `BatchSampler`, `DistributedSampler`: Sampling strategies

#### Distributed Training
- **Multi-device and multi-host training strategies**:
  - `SingleDeviceStrategy`: Single device execution
  - `DataParallelStrategy`: Data parallelism across devices
  - `ShardedDataParallelStrategy` / `FullyShardedDataParallelStrategy`: Memory-efficient sharded training
  - `AutoStrategy`: Automatic strategy selection
  - `all_reduce`, `broadcast`: Distributed communication primitives

#### Checkpointing
- **Comprehensive checkpoint management**:
  - `CheckpointManager`: Manage multiple checkpoints with retention policies
  - `save_checkpoint` / `load_checkpoint`: Save and restore model states
  - `find_checkpoint` / `list_checkpoints`: Checkpoint discovery utilities

#### Progress Bar System
- **Multiple progress bar implementations**:
  - `SimpleProgressBar`: Basic text-based progress
  - `TQDMProgressBarWrapper`: TQDM-based progress
  - `RichProgressBarWrapper`: Rich library-based progress

### Improvements

#### API Documentation
- **Enhanced module documentation**: All public modules now include comprehensive docstrings with examples, parameter descriptions, and usage guidelines directly in `__init__.py` files
- **Reorganized imports**: Cleaner and more consistent import structure across all modules

### Breaking Changes

#### Removed `braintools.param` Module
- **The entire `braintools.param` module has been removed**, including:
  - Data containers (`Data`)
  - Parameter wrappers (`Param`, `Const`)
  - State containers (`ArrayHidden`, `ArrayParam`)
  - Regularization classes (`GaussianReg`, `L1Reg`, `L2Reg`)
  - All transform classes (`SigmoidT`, `SoftplusT`, `AffineT`, etc.)
  - Utility functions (`get_param()`, `get_size()`)
- Users relying on these features should migrate to alternative implementations or pin to version 0.1.6


## Version 0.1.6 (2025-12-25)

### New Features

#### Parameter Management Expansion (`braintools.param`)
- **Hierarchical data container**: Added `Data` for composed state storage and cloning.
- **Parameter wrappers**: Added `Param` and `Const` with built-in transforms and optional regularization.
- **State containers**: Added `ArrayHidden` and `ArrayParam` with transform-aware `.data` access.
- **Regularization priors**: Added `GaussianReg`, `L1Reg`, and `L2Reg` with optional trainable hyperparameters.
- **Utilities**: Added `get_param()` and `get_size()` helpers for parameter/state handling.

#### Transforms
- **New `ReluT` transform** for lower-bounded parameters.
- **Expanded transform suite** now includes `PositiveT`, `NegativeT`, `ScaledSigmoidT`, `PowerT`,
  `OrderedT`, `SimplexT`, and `UnitVectorT`.

### Improvements

#### API Consistency
- **Transform naming cleanup**: Standardized transform class names with the `*T` suffix
  (e.g., `SigmoidT`, `SoftplusT`, `AffineT`, `ChainT`, `MaskedT`, `ClipT`).

#### Documentation
- **Expanded param API docs**: Added sections for data containers, state containers, regularization,
  utilities, and updated transform listings in `docs/apis/param.rst`.
- **API index update**: Added `param` API page to `docs/index.rst`.

#### Tests
- **New test coverage**: Added tests for data containers, modules, regularization, state, transforms,
  and utilities across the param module.

### Breaking Changes
- **Transform API renames**: Transform classes now use the `*T` suffix (e.g., `Sigmoid` -> `SigmoidT`).
- **Custom transform removed**: The `Custom` transform is no longer part of the public API.

### Bug Fixes
- **Initializer RNG**: `TruncatedNormal` now defaults to `numpy.random` when no RNG is provided.


## Version 0.1.5 (2025-12-14)

### New Features

#### Parameter Transformation Module (`braintools.param`)
- **7 new bijective transforms** for constrained optimization and probabilistic modeling:
  - **Positive**: Constrains parameters to (0, +∞) using exponential transformation
  - **Negative**: Constrains parameters to (-∞, 0) using negative softplus
  - **ScaledSigmoid**: Sigmoid with adjustable sharpness/temperature parameter (beta)
  - **Power**: Box-Cox family power transformation for variance stabilization
  - **Ordered**: Ensures monotonically increasing output vectors (useful for cutpoints in ordinal regression)
  - **Simplex**: Stick-breaking transformation for probability vectors summing to 1
  - **UnitVector**: Projects vectors onto the unit sphere (L2 norm = 1)
- **Jacobian computation**: Added `log_abs_det_jacobian()` method to Transform base class and implementations for probabilistic modeling
  - Implemented for: Identity, Sigmoid, Softplus, Log, Exp, Affine, Chain, Positive

#### Surrogate Gradient Enhancements (`braintools.surrogate`)

- Gradient computation of hyperparameters of surrogate gradient functions.
- Fix batching issue in surrogate gradient functions


### Improvements

#### API Enhancements
- **`__repr__` methods**: Added string representations to all Transform classes and Param class for better debugging
- **Enhanced documentation**: Updated `docs/apis/param.rst` with comprehensive API reference
  - Organized sections: Base Classes, Parameter Wrapper, Bounded Transforms, Positive/Negative Transforms, Advanced Transforms, Composition Transforms
  - Descriptive explanations for each transform's use case

#### Code Quality
- **Comprehensive test coverage**: Added 28 new tests for param module (45 total tests passing)
  - Tests for all new transforms: roundtrip, constraints, repr methods
  - Tests for `log_abs_det_jacobian` correctness
  - Tests for edge cases and numerical stability




## Version 0.1.4 (2025-10-31)

### New Features

#### Learning Rate Scheduler Enhancements (`braintools.optim`)
- **New `apply()` method**: Added `apply()` method to all LR schedulers for more flexible learning rate application
  - Allows applying learning rate transformations without stepping the scheduler
  - Useful for custom training loops and learning rate inspection
- **Comprehensive test coverage**: Added 118+ comprehensive tests covering all 17 learning rate schedulers
  - Tests for basic functionality, optimizer integration, JIT compilation, state persistence
  - Full coverage of edge cases and special modes for each scheduler
  - Validates correctness with `@brainstate.transform.jit` compilation

### Improvements

#### Documentation
- **Restructured tutorial organization**: Renamed and reorganized documentation files for better clarity
  - Moved module tutorials into subdirectories (`conn/`, `init/`, `input/`, `file/`, `surrogate/`)
  - Updated table of contents structure across all modules
  - Improved navigation with consolidated index files (`index.md` instead of `toc_*.md`)
- **Enhanced visual branding**: Updated project logo from JPG to high-resolution PNG format
  - Better quality and transparency support
  - Consistent branding across documentation

#### Code Quality
- **Test improvements**: Refactored scheduler tests with better organization and coverage
  - Each scheduler now has 5-10 dedicated tests
  - Tests verify: basic functionality, optimizer integration, JIT compilation, multiple param groups, state dict save/load
  - Discovered and documented key implementation behaviors (epoch counting, initialization patterns)

#### CI/CD
- **Updated GitHub Actions**: Bumped actions to latest versions for improved security and performance
  - `actions/download-artifact`: v5 → v6
  - `actions/upload-artifact`: v4 → v5
  - Better artifact handling in CI pipeline

### Bug Fixes
- Fixed edge cases in learning rate scheduler state management
- Corrected epoch counting behavior in milestone-based schedulers
- Improved JIT compilation compatibility for all schedulers

### Notes
- All 17 learning rate schedulers now have comprehensive test coverage (100%)
- Enhanced reliability for training workflows with thorough validation
- Improved developer experience with better documentation structure


## Version 0.1.0 (2025-10-06)

### Major Features

#### Surrogate Gradients Module (`braintools.surrogate`)
- **New comprehensive surrogate gradient system** for training spiking neural networks (SNNs)
- **18+ surrogate gradient functions** with straight-through estimator support:
  - **Sigmoid-based**: `Sigmoid`, `SoftSign`, `Arctan`, `ERF`
  - **Piecewise**: `PiecewiseQuadratic`, `PiecewiseExp`, `PiecewiseLeakyRelu`
  - **ReLU-based**: `ReluGrad`, `LeakyRelu`, `LogTailedRelu`
  - **Distribution-inspired**: `GaussianGrad`, `MultiGaussianGrad`, `InvSquareGrad`, `SlayerGrad`
  - **Advanced**: `S2NN`, `QPseudoSpike`, `SquarewaveFourierSeries`, `NonzeroSignLog`
- **Customizable hyperparameters** (alpha, sigma, width, etc.) for fine-tuning gradient behavior
- **Comprehensive tutorials**: 2 detailed notebooks covering basics and customization
- Enables gradient-based training of SNNs via backpropagation through time
- Over 2,600 lines of implementation with extensive test coverage

### New Features

#### Learning Rate Schedulers (`braintools.optim`)
- **ExponentialDecayLR scheduler**: Fine-grained exponential decay with step-based control
  - Support for transition steps, staircase mode, delayed start, and bounded decay
  - Better control than epoch-based ExponentialLR for step-level scheduling
  - Compatible with Optax's exponential_decay schedule

### Improvements

#### API Refinements
- **Deprecation warnings** added for future API changes:
  - Deprecated `beta1` and `beta2` parameters in Adam optimizer (use `b1` and `b2` instead)
  - Deprecated `unit` parameter in various initializers (use `UNITLESS` by default)
  - Deprecated `init_call` function replaced with `param` for improved consistency
- **Enhanced state management**: Refactored `UniqueStateManager` to utilize pytree methods
- **Comprehensive tests**: Added extensive tests for `UniqueStateManager` methods and edge cases

#### Documentation
- Updated API documentation for new surrogate gradient module
- Added learning rate scheduler documentation for `ExponentialDecayLR`
- Enhanced optimizer tutorials with updated examples
- Clarified docstrings for `FixedProb` class and variance scaling initializer

### Code Quality

#### Internal Improvements
- Updated copyright information from BDP Ecosystem Limited to BrainX Ecosystem Limited
- Improved consistency across codebase with standardized function signatures
- Better default parameter handling (`UNITLESS` for unit parameters)
- Enhanced test coverage for state management and optimizers

#### Metric Enhancements
- Improved correlation and firing metrics implementation
- Enhanced LFP (Local Field Potential) analysis functions
- Better error handling and validation in metric computations

### Breaking Changes
- **Deprecation notices** (not yet removed, but will be in future versions):
  - `beta1`/`beta2` parameters in Adam optimizer (use `b1`/`b2`)
  - `unit` parameter in initializers (defaults to `UNITLESS`)
  - `init_call` function (use `param` instead)

### Notes
- This release focuses on enabling gradient-based training for spiking neural networks
- The surrogate gradient module is a major addition for neuromorphic computing and SNN research
- Enhanced learning rate scheduling provides more control for training workflows






## Version 0.0.14 (2025-10-04)

### New Features

#### Optimizer Enhancements (`braintools.optim`)
- **Momentum optimizers**: Added `Momentum` and `MomentumNesterov` optimizers with gradient transformations
- **Improved state management**: Refactored optimizer state handling with new `OptimState` class for better encapsulation

#### Initialization Updates (`braintools.init`)
- **ZeroInit initializer**: New zero initialization class for weights and parameters
- **VarianceScaling export**: Added `VarianceScaling` to module exports for easier access

### Improvements
- Enhanced optimizer state management for better performance and maintainability
- Simplified initialization API with additional export options
- Updated documentation for new initialization methods

### Internal Changes
- Refactored test structure for initialization module
- Improved learning rate scheduler implementation


## Version 0.0.13 (2025-10-02)

### Major Features

#### New Initialization Framework (`braintools.init`)
- **Unified initialization API** consolidating all weight and parameter initialization strategies
- **Distance-based initialization**: Support for distance-modulated weight patterns
- **Variance scaling strategies**: Xavier, He, LeCun initialization methods
- **Orthogonal initialization** for improved training stability
- **Composite distributions** for complex initialization patterns
- Simplified API with consistent parameter naming across all initializers

#### Advanced Connectivity Patterns (`braintools.conn`)
- **Topological network patterns**:
  - Small-world and scale-free networks
  - Hierarchical and core-periphery structures
  - Modular and clustered random connectivity
- **Enhanced biological connectivity**:
  - Excitatory-inhibitory balanced networks
  - Distance-dependent connectivity with multiple profiles
  - Compartment-specific connectivity (dendrite, soma, axon)
- **Spatial connectivity improvements**:
  - 2D convolutional kernels for spatial networks
  - Position-based connectivity with normalization
  - Distance modulation using composable profiles

#### Comprehensive Optax Integration (`braintools.optim`)
- **Full Optax optimizer support**: Adam, SGD, RMSProp, AdaGrad, AdaDelta, and more
- **Advanced learning rate schedulers**:
  - Cosine annealing with warm restarts
  - Polynomial decay with warmup
  - Piecewise constant schedules
  - Sequential and chained schedulers
- **Improved optimizer state management** with unique state handling
- **Parameter groups** with per-group learning rates

### Improvements

#### API Enhancements
- Simplified `conn` module API with direct class access
- Refactored initialization calls for consistency
- Improved type annotations throughout
- Better default parameter handling

#### Documentation & Tutorials
- Updated tutorial structure for connectivity patterns
- New examples for topological networks
- Enhanced API documentation with detailed examples
- Improved code readability in tutorials

### Code Quality
- Comprehensive test coverage for new features
- Better error handling and validation
- Consistent naming conventions
- Removed deprecated and redundant code

### Breaking Changes
- Renamed `PointNeuronConnectivity` to `PointConnectivity`
- Renamed `ConvKernel` to `Conv2dKernel`
- Unified initializer names (e.g., `ConstantWeight` → `Constant`)
- Removed `PopulationRateConnectivity` class
- Changed some parameter names for clarity (e.g., unified use of `rng` parameter)


## Version 0.0.12 (2025-09-24)

### Major Features

#### Comprehensive Visualization System
- **New visualization modules** for neural data analysis:
  - `neural.py`: Spike rasters, population activity, connectivity matrices, firing rate maps
  - `three_d.py`: 3D visualizations for neural networks, brain surfaces, trajectories, electrode arrays
  - `statistical.py`: Statistical plotting tools (confusion matrices, ROC curves, correlation plots)
  - `interactive.py`: Interactive visualizations with Plotly support
  - `colormaps.py`: Neural-specific colormaps and publication-ready styling
- **15+ new tutorial notebooks** covering all visualization techniques
- **Brain-specific colormaps** for membrane potential, spike activity, and connectivity

#### Enhanced Numerical Integration
- **New ODE integrators**: 
  - Runge-Kutta methods: RK23, RK45, RKF45, DOP853, DOPRI5, SSPRK33
  - Specialized methods: Midpoint, Heun, RK4(3/8), Ralston RK2/RK3, Bogacki-Shampine
- **New SDE integrators**: Heun, Tamed Euler, Implicit Euler, SRK2, SRK3, SRK4
- **IMEX integrators** for stiff equations: Euler, ARS(2,2,2), CNAB
- **DDE integrators** for delay differential equations
- Comprehensive test coverage and accuracy verification

#### Advanced Spike Processing
- **Spike encoders**: Rate, Poisson, Population, Latency, and Temporal encoders
- **Enhanced spike operations** with bitwise functionality
- **Spike metrics**: Victor-Purpura distance, spike train synchrony, correlation indices
- Tutorial notebooks for spike encoding and analysis

#### New Optimization Framework
- **NevergradOptimizer**: Integration with Nevergrad optimization library
- **ScipyOptimizer**: Enhanced scipy optimization with flexible bounds support
- Refactored optimizer architecture for better extensibility
- Support for dict and sequence parameter bounds

### Improvements

#### File Management
- Enhanced msgpack serialization with mismatch handling options
- Improved checkpoint loading with better error recovery
- Support for handling mismatched keys during state restoration

#### Metrics and Analysis
- **LFP analysis functions**: Power spectral density, coherence analysis, phase-amplitude coupling
- **Functional connectivity**: Dynamic connectivity computation
- **Classification metrics**: Binary, multiclass, focal loss, and smoothing techniques
- **Regression losses**: MSE, MAE, Huber, and quantile losses

### Documentation
- Added comprehensive API documentation for all new modules
- Created tutorials for:
  - ODE/SDE integration methods
  - Classification and regression losses
  - Pairwise and embedding similarity
  - Spiking metrics and LFP analysis
  - Advanced neural visualization techniques
- Updated project description from "brain modeling" to "brain simulation"
- Changed references from BrainPy to BrainTools throughout

### Code Quality
- Added extensive unit tests for all new modules
- Improved type hints and parameter documentation
- Better error handling and validation
- Consistent API design across modules

### Breaking Changes
- Refactored optimizer module structure (moved from single `optimizer.py` to separate modules)
- Removed unused key parameter from spike encoder methods
- Updated some function signatures for clarity

### Bug Fixes
- Fixed Softplus unit scaling issues
- Corrected paths in publish workflow
- Fixed formatting in ODE integrator documentation
- Resolved msgpack checkpoint handling errors






