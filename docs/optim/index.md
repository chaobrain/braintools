# Optimization

`braintools.optim` covers two kinds of optimization: black-box and bounded search for model fitting (`NevergradOptimizer`, `ScipyOptimizer`), and gradient-based training of `brainstate` models with optax optimizers and learning-rate schedules.

## What you'll find here

- **Gradient-free optimization with Nevergrad**: batched losses, bounds as arrays, dicts, and quantities, why `num_workers` should equal `n_sample`, comparing methods on a multimodal function, fitting a neuron model to spike counts, and noisy objectives.
- **Bounded optimization with SciPy**: automatic JAX gradients, units, random restarts, constraints, choosing a method, and why float32 can stop SciPy's line searches.
- **Gradient-based optimizers**: the `register_trainable_weights` / `update` interface, exact agreement with optax, a training loop for a `brainstate` model, learning-rate sensitivity, coupled versus decoupled weight decay, and gradient clipping.
- **Learning-rate schedules**: every built-in schedule, how they compare with PyTorch's, composing schedules, `ReduceLROnPlateau`, and why decaying the rate lets SGD converge.
- **Advanced optimizers**: a learning-rate-tuned comparison, RAdam's early steps, L-BFGS with a line search, optimizer memory, Lookahead, and SOFO.

For the full list of classes and parameters, see the {doc}`API reference <../apis/optim>`.

```{toctree}
:maxdepth: 1

01_nevergrad_optimizer.ipynb
02_scipy_optimizer.ipynb
03_optax_getting_started.ipynb
04_learning_rate_scheduling.ipynb
05_advanced_optimizers.ipynb
```
