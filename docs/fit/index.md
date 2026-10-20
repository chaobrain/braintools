# Model Fitting

`braintools.fit` fits any JAX or `brainstate` model to data. You give it a pure `simulate(params)` function or a module with `brainstate.nn.Param` parameters, an objective built from `fit.losses`, and a search space. A `Fitter` then minimises the objective from many starting points at once, by gradient (all starts vmapped), SciPy or Nevergrad.

## What you'll find here

- **Fitting models**: simulating trajectories with `quad.odeint` and `quad.solve_ivp`; composing objectives; a multi-start fit of a Hopf oscillator; why several starts are needed; mapping the loss landscape with `fit.sweep`; confidence intervals from the Hessian; fitting the couplings of a Wilson–Cowan `brainstate` module; differentiable losses for spiking data; and saving results to HDF5.

For the full list of functions and parameters, see the {doc}`API reference <../apis/fit>`.

```{toctree}
:maxdepth: 1

01_fitting_models.ipynb
```
