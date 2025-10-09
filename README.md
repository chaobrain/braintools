# Modeling tools for brain simulation

<p align="center">
  <img alt="Header image of braintools." src="https://raw.githubusercontent.com/chaobrain/braintools/main/docs/_static/braintools.png" width=50%>
</p>

<p align="center">
  <a href="https://pypi.org/project/braintools/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/braintools"></a>
  <a href="https://github.com/chaobrain/braintools/blob/main/LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href='https://braintools.readthedocs.io/?badge=latest'>
    <img src='https://readthedocs.org/projects/braintools/badge/?version=latest' alt='Documentation Status' />
  </a>
  <a href="https://badge.fury.io/py/braintools"><img alt="PyPI version" src="https://badge.fury.io/py/braintools.svg"></a>
  <a href="https://github.com/chaobrain/braintools/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaobrain/braintools/actions/workflows/CI.yml/badge.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.17110064"><img src="https://zenodo.org/badge/776629792.svg" alt="DOI"></a>
</p>

`braintools` is a lightweight, JAX-friendly toolbox with practical utilities for brain modeling.

## Highlights

- **Composable connectivity**: declarative builders for point, multi-compartment, and population networks with spatial kernels, degree constraints, and unit-aware metadata
- **Visualization suite**: publication plots, interactive dashboards, 3D viewers, and animation helpers in `braintools.visualize`
- **Metrics and solvers**: losses, evaluation metrics, and PyTree-aware ODE/SDE/DDE integrators ready for `jit`/`vmap`
- **Signal and optimization helpers**: reusable generators and lightweight optimizers to prototype models quickly

`braintools` integrates smoothly with the broader ecosystem (e.g., `brainstate`, `brainunit`) while keeping a simple, functional style.

## Installation

```bash
pip install -U braintools
```

Optional extras are published for hardware-specific builds:

```bash
pip install -U braintools[cpu]
# CUDA 12.x wheels
pip install -U braintools[cuda12]
# TPU runtime
pip install -U braintools[tpu]
```

Alternatively, install the curated BrainX bundle that ships with `braintools` and related projects:

```bash
pip install -U BrainX
```

## Documentation

The full documentation is available at https://braintools.readthedocs.io

## Ecosystem

`braintools` is one part of our brain simulation ecosystem: https://brainmodeling.readthedocs.io/

## Contributing

Contributions and issue reports are welcome! See `CONTRIBUTING.md` for guidelines.

## License

Apache 2.0. See `LICENSE` for details.

## Citation

If you use `braintools` in your work, please cite the Zenodo DOI: [10.5281/zenodo.17110064](https://doi.org/10.5281/zenodo.17110064)
