# Surrogate Gradients

Spiking neurons emit spikes through a step function, whose derivative is zero almost everywhere. `braintools.surrogate` keeps the exact step in the forward pass and substitutes a smooth, localized derivative in the backward pass, so spiking networks can be trained with gradient descent.

## What you'll find here

- **Basics**: class and functional forms and what their derivatives are, the 18 built-in surrogates compared by peak, gradient mass and support, the steepness parameter, gradients through time in a LIF neuron and the role of the reset, and a first optimization through spikes.
- **Custom surrogates**: writing and checking a new surrogate, what the base class provides compared with `jax.custom_vjp` (forward mode, higher derivatives), surrogate parameters under `jax.jit`, and why the support decides whether silent neurons can learn.
- **Tuning surrogate hyperparameters**: why the training loss does not depend on surrogate parameters, how much the shape and the steepness matter on a spiking classifier, why steep surrogates make gradients explode through the reset, and hypergradients through unrolled training.

For the full list of surrogates and parameters, see the {doc}`API reference <../apis/surrogate>`.

```{toctree}
:maxdepth: 1

01_basics.ipynb
02_customizing.ipynb
03_meta_learning.ipynb
```
