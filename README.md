# Modeling tools for brain simulation

<p align="center">
  	<img alt="Header image of braintools." src="https://raw.githubusercontent.com/chaobrain/braintools/main/docs/_static/braintools.jpg" width=50%>
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


[``braintools``](https://github.com/chaobrain/braintools) is a lightweight, JAX‑friendly toolbox of practical utilities for brain modeling. It focuses on:

- Metric functions for model training/evaluation (classification, regression, ranking, connectivity, LFP helpers)
- Numerical one‑step integrators for ODE/SDE/DDE (PyTree‑aware, JAX‑ready)
- Input generators and small optimization helpers for quick prototyping
- And more to come...

It plays nicely with the rest of our brain simulation ecosystem (e.g., `brainstate`, `brainunit`) and follows a simple, functional style that works with jit/vmap.


## Installation

You can install ``braintools`` via pip:

```bash
pip install braintools --upgrade
```

GPU/TPU builds and extras (see docs for details):

```bash
# CPU (default)
pip install -U braintools[cpu]

# CUDA 12.x wheels
pip install -U braintools[cuda12]

# TPU
pip install -U braintools[tpu]
```

Alternatively, install `BrainX`, which bundles `braintools` with other compatible packages for a comprehensive brain modeling ecosystem:

```bash
pip install BrainX -U
```


## Documentation

The official docs are at: https://braintools.readthedocs.io



## Ecosystem

``braintools`` is part of our brain simulation ecosystem: https://brainmodeling.readthedocs.io/


## Contributing

Contributions and issue reports are welcome! Please see `CONTRIBUTING.md` and open a PR/issue on GitHub.


## License

Apache 2.0. See `LICENSE` for details.


## Citation

If you use ``braintools`` in your work, please use zenodo DOI: [10.5281/zenodo.17110064](https://doi.org/10.5281/zenodo.17110064)

