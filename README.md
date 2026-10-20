# Modeling tools for brain simulation

<p align="center">
  <img alt="Header image of braintools." src="https://brainx.chaobrain.com/images/braintools.webp" width=50%>
</p>

<p align="center">
  <a href="https://pypi.org/project/braintools/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/braintools"></a>
  <a href="https://github.com/chaobrain/braintools/blob/main/LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://brainx.chaobrain.com/braintools/"><img alt="Documentation Status" src="https://github.com/chaobrain/braintools/actions/workflows/deploy-docs.yml/badge.svg"></a>
  <a href="https://badge.fury.io/py/braintools"><img alt="PyPI version" src="https://badge.fury.io/py/braintools.svg"></a>
  <a href="https://github.com/chaobrain/braintools/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaobrain/braintools/actions/workflows/CI.yml/badge.svg"></a>
  <a href="https://codecov.io/gh/chaobrain/braintools"><img alt="Test Coverage" src="https://codecov.io/gh/chaobrain/braintools/graph/badge.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.17110064"><img src="https://zenodo.org/badge/776629792.svg" alt="DOI"></a>
</p>

`braintools` is a lightweight, JAX-friendly toolbox with practical utilities for brain modeling.

## Highlights

- **Connectivity builders** (`braintools.conn`): declarative, unit-aware patterns for point neurons (random, fixed in-/out-degree, distance-dependent, small-world, modular, ...), multi-compartment morphologies, and convolution kernels; population networks assembled from populations and projections; vectorised, memory-bounded generation that scales to 1e5-neuron networks; export to `brainevent` sparse formats
- **Initializers** (`braintools.init`): parameter initializers (distributions, variance scaling, orthogonal, composites) and distance profiles shared with the connectivity builders
- **Spike data** (`braintools.spike`): the `SpikeTrains` container (a padded, `jit`/`vmap`-friendly spike-time layout with dense/event conversions), spike encoders (latency, rate, Poisson, population, ...), and bitwise spike operations
- **Neural data analysis** (`braintools.analysis`): spike-train statistics (ISI/CV/Fano, PSTHs, correlograms, spike-train distances, synchrony), spectral and LFP analysis (multitaper and wavelet spectra, filtering, phase synchrony, cross-frequency coupling, current source density), population analysis (PCA, decoding, Granger causality, avalanches, functional connectivity), structural graph measures of connectivity, and resampling-based statistics
- **Cognitive task framework** (`braintools.cogtask`): phase-based, composable trial construction for training and analyzing networks on classic systems-neuroscience paradigms, plus a registry of pre-built tasks
- **Numerical integration** (`braintools.quad`): PyTree-aware ODE/SDE/DDE/IMEX one-step integrators ready for `jit`/`vmap`, a fixed-step trajectory driver (`odeint`) and a differentiable adaptive solver (`solve_ivp`)
- **Model fitting** (`braintools.fit`): fit any JAX or `brainstate` model to data with composable objectives, summary-statistic losses, vmapped multi-start optimization, parameter sweeps, and confidence intervals
- **Losses and metrics** (`braintools.metric`): classification, regression, ranking, and pairwise-similarity losses for training
- **Inputs, surrogates, and visualization**: input-current generators (`braintools.input`), surrogate gradients (`braintools.surrogate`), and plotting helpers (`braintools.visualize`)
- **Training helpers**: optimizer wrappers (`braintools.optim`), a multi-start minimizer, online readout learning (RLS, LMS, FORCE), closed-form regressions, and a lightweight training loop (`braintools.trainer`)
- **Data files** (`braintools.file`): checkpoints, MATLAB files, and optional HDF5, NWB, and NEO formats

`braintools` integrates smoothly with the broader ecosystem (e.g., `brainstate`, `brainunit`) while keeping a simple, functional style.

## Installation

```bash
pip install -U braintools
```

Optional extras select a JAX backend:

```bash
pip install -U braintools[cpu]
# CUDA 12.x / CUDA 13.x wheels
pip install -U braintools[cuda12]
pip install -U braintools[cuda13]
# TPU runtime
pip install -U braintools[tpu]
```

and enable optional features:

```bash
pip install -U braintools[viz]        # interactive plotly figures in braintools.visualize
pip install -U braintools[progress]   # tqdm / rich progress bars in braintools.trainer
pip install -U braintools[nevergrad]  # gradient-free optimization in braintools.optim
```

Alternatively, install the curated BrainX bundle that ships with `braintools` and related projects:

```bash
pip install -U BrainX
```

## Cognitive task framework (`braintools.cogtask`)

`braintools.cogtask` lets you build cognitive tasks from composable, named **phases** (fixation, stimulus, delay, response, ...) connected by simple operators:

- `a >> b` runs `b` after `a` (sequential)
- `a * n` repeats `a` `n` times
- `a | b` runs `a` and `b` in parallel (multi-modal stimuli)
- `If`, `Switch`, `While` add trial-by-trial branching

Trials are generated lazily via `task.sample(i)` / `task.batch_sample(B)`, both JIT- and `vmap`-friendly, with per-trial keys derived from `seed` so batches are fully reproducible.

```python
import brainstate
import brainunit as u
from braintools.cogtask import PerceptualDecisionMaking

# Phase durations are expressed in units of the environment timestep, so a
# timestep must be set before a task is sampled.
brainstate.environ.set(dt=1.0 * u.ms)

task = PerceptualDecisionMaking(t_stimulus=1500 * u.ms, num_choices=2, seed=0)
X, Y = task.batch_sample(32)   # X: (T, B, n_in), Y: (T, B) labels
```

The package also ships a registry of canonical tasks: `PerceptualDecisionMaking`, `ContextDecisionMaking`, `DelayMatchSample`, `DelayComparison`, `GoNoGo`, `ReadySetGo`, `PostDecisionWager`, `Reaching1D`, `EvidenceAccumulation`, `HierarchicalReasoning`, and more — see the [API reference](https://brainx.chaobrain.com/braintools/apis/cogtask.html) for the full list.

## Documentation

The full documentation is available at https://brainx.chaobrain.com/braintools/

## Upgrading to 1.0

Names that moved since 0.4 (for example the `metric` analysis functions, now in `braintools.analysis`, and the top-level spike encoders, now in `braintools.spike`) still work but emit a `BraintoolsDeprecationWarning`, a `FutureWarning` shown by default. Version 1.0 removes them. The [migration guide](https://brainx.chaobrain.com/braintools/migration.html) lists every deprecated name with its replacement.

## Ecosystem

`braintools` is one part of our brain simulation ecosystem: https://brainx.chaobrain.com/

## Contributing

Contributions and issue reports are welcome! See `CONTRIBUTING.md` for guidelines.

## License

Apache 2.0. See `LICENSE` for details.

## Citation

If you use `braintools` in your work, please cite the Zenodo DOI: [10.5281/zenodo.17110064](https://doi.org/10.5281/zenodo.17110064)
