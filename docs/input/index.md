# Input Current

`braintools.input` builds the time series that drive a simulation: injected currents, stimulation protocols, and noise. Every generator reads the time step from `brainstate.environ` and returns one sample per simulation step.

## What you'll find here

- **Introduction**: the functional and composable APIs side by side, the time step and array length, units, and feeding an input into a leaky integrate-and-fire neuron.
- **Functional API**: every generator (piecewise, periodic, pulse, and stochastic), what each parameter means, and numerical checks of the output; `ou_process` (where `sigma` is the stationary standard deviation), `wiener_process` (white-noise increments), and `poisson` statistics.
- **Composable API**: `Input` objects and caching, arithmetic, `&` and `|`, `shift`, `smooth`, `repeat`, `clip`, `apply`, multi-channel inputs, and a family of protocols for an F-I curve.
- **Custom inputs and pipelines**: subclassing `Input`, wrapping functions with `apply`, and a reproducible batch of trials with independent noise run through a model.

For the full list of functions and classes, see the {doc}`API reference <../apis/input>`.

```{toctree}
:maxdepth: 1

01_introduction.ipynb
02_functional_api.ipynb
03_composable_api.ipynb
04_custom_pipelines.ipynb
```
