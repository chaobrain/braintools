# Known issues and follow-ups

Open items found during the 0.4–0.8 roadmap phases (`docs/specs/`). Each entry
says where the problem is and what a fix looks like. Tick an item and link the
PR when it is done.

## Before 1.0

- [ ] **Remove the deprecated aliases.** 1.0.0 removes every name registered
  in `braintools/_deprecation.py` (listed in `docs/migration.md`). The
  registry test fails once the version reaches a record's `removed_in`, so
  the aliases, their tests and the migration-guide rows go together.
- [ ] **Downstream code still calls deprecated names.** brainpy re-exports
  `braintools.metric.*` (for example `cross_correlation`, `unitary_LFP` in
  `brainpy/measure.py`), so brainpy users see `FutureWarning`s for brainpy's
  own API. brainpy and brainmass must move to `braintools.analysis.*` /
  `braintools.fit.*` before braintools 1.0.
- [ ] **Export `BraintoolsDeprecationWarning` publicly.** The migration guide
  and docstrings import it from the private `braintools._deprecation`.
- [ ] **Choose the release version.** `braintools/_version.py` is still
  `0.3.0`.

## Ecosystem

- [ ] **brainstate disables float64 on import.** `import brainstate` sets
  `jax_enable_x64=False` even after `jax.config.update('jax_enable_x64', True)`
  or with `JAX_ENABLE_X64=1` (`brainstate/environ.py`, last line). Fixed
  upstream in https://github.com/chaobrain/brainstate/pull/242; once released,
  raise the `brainstate` floor in `pyproject.toml`. Until then,
  document `brainstate.environ.set(precision=64)` in the `quad` and `fit` docs.
- [ ] **BrainX pins exact versions** (`braintools==0.3.0`, `brainunit==0.5.2`,
  `brainstate==0.5.4`, `jax<=0.11.1`), so `pip install -U braintools` conflicts
  with an installed BrainX. Release BrainX together with braintools, and pin
  compatible ranges (`braintools~=0.8.0`).
- [ ] **saiunit uses private JAX APIs.** JAX 0.11 broke `saiunit.math.einsum`
  (fixed in saiunit 0.5.2). Add a weekly CI job that installs the latest
  jax/brainunit/brainstate/brainevent and runs the suite plus the downstream
  smoke tests, so such breaks show up before users hit them.
- [ ] **Two initializer hierarchies.** `brainstate.nn.init` duplicates
  `braintools.init`, and `brainstate/_state.py` imports `braintools.init.param`
  (a dependency in the wrong direction). Needs a joint decision; the roadmap
  proposes `braintools.init` as canonical.
- [ ] **Two integrator registries.** braincell has its own
  (`braincell/quad/_registry.py`); it could register its solvers through
  `braintools.quad.register_method`.
- [ ] **brainmass test fixtures.** 25 brainmass tests error on fixtures that
  existed before this work (they are counted in the downstream baseline).

## trainer (maintenance mode: bug fixes only)

- [ ] **`ReduceLROnPlateau` raises `TypeError` in `Trainer`.** The trainer
  calls `scheduler.step()` without the monitored metric
  (`braintools/trainer/_trainer.py`, scheduler stepping in the fit loop).
- [ ] **Schedulers passed only as `lr=` lose their epoch on resume.** Only
  schedulers returned by `configure_optimizers` are saved in checkpoints. The
  scheduler state should live in `OptaxOptimizer.state_dict`
  (`braintools/optim/_optax_optimizer.py`).
- [ ] **Mid-epoch checkpoints resume at the next epoch boundary**, and callback
  state is not restored (documented in `Trainer.fit`).
- [ ] **Metric names get a doubled prefix** (`train_train_loss`) when a step
  logs a key that already starts with the stage name.

## optim

- [ ] **`LRScheduler.load_state_dict` leaves the optimizer's reported LR
  stale.** The base class does not push the restored LR to
  `optimizer.current_lr` / the param groups (`SequentialLR` does since 0.8).
  The update itself is correct; only the reported value lags.
- [ ] **`Fromage(momentum=...)`** warns with a plain `UserWarning` that the
  argument is ignored. Decide whether to deprecate the argument through the
  registry.

## quad

- [ ] **`solve_ivp` accepts only the built-in embedded pairs**
  (`_ADAPTIVE` in `braintools/quad/_solve_ivp.py`). A method added with
  `register_method` works with `odeint` but not with `solve_ivp`, and there is
  no way to register an embedded pair with its error order.
- [ ] **Review the `solve_ivp(checkpoint=True)` default.** It bounds memory for
  reverse-mode gradients but recomputes the forward pass.

## Documentation and tests

- [ ] **About 190 public names have no `>>>` example**, mainly in
  `braintools.visualize` (54 names), the `quad.ode_*_step` steppers, the
  surrogate functions (`sigmoid`, `arctan`, ...), `file.msgpack_*` and
  `optim.LRScheduler`. Then add a test that every public name has an example.
- [ ] **Interactive plots are not tested in CI.** plotly is not in the
  `testing` extra, so `braintools/visualize/_interactive_test.py` skips.
- [ ] **mypy runs without `check_untyped_defs`**, so the bodies of unannotated
  functions are not checked.
- [ ] **The wheel ships the co-located tests** (122 `*_test.py` files).
  Exclude them in `[tool.setuptools.packages.find]` / package data if the
  installed package should not contain them.
- [ ] **`visualize.isi_distribution` x-label** does not include the
  `SpikeTrains` time unit, unlike `spike_raster` and `spike_histogram`.
- [ ] **Move the encoder tutorial** `docs/misc/spike_encoding.ipynb` under
  `docs/spike/`; the spike section links to it for now.

## Verification gaps

- [ ] **Check the NeuroGym-style tasks against NeuroGym.**
  `MultiSensoryIntegration`, `OneTwoThreeGo` and `EconomicDecisionMaking`
  (`braintools/cogtask/tasks/`) were written from memory of the NeuroGym
  tasks; neurogym was not installed to compare.
- [ ] **`bernoulli_pairs` on Windows.** The binomial edge count may overflow
  the default integer type on Windows (`braintools/conn/_sampling.py`); untested
  there.
- [ ] **`analysis.graph.spectral_radius`** relies on ARPACK, which can fail to
  converge on some matrices; a dense fallback for small graphs would make it
  robust.
- [ ] **FORCE is seed-sensitive.** The readout tutorial
  (`docs/optim/06_readout_learning.ipynb`) converges for its seed; other seeds
  may need more training steps.
- [ ] **Connectivity beyond 1e6 neurons.** `conn` builds on the host; decide
  between a device backend and pointing users to brainevent's just-in-time
  connectivity.
