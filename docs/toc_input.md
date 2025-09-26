# Input Current

Explore how to generate, compose, and customize stimulation protocols with
`braintools.input`. The tutorials below progress from core concepts to advanced
pipelines, each accompanied by runnable notebooks.

- **Introduction to braintools.input** - orientation to the module, environment
  setup, and comparison of the functional vs. composable APIs.
- **Functional API tutorial** - hands-on recipes for section/step/ramp currents,
  pulse generators, and stochastic helpers with unit-aware workflows.
- **Composable API tutorial** - algebraic composition, sequential protocols,
  modulation, and caching behaviour using the object-oriented interface.
- **Custom transformations & pipelines** - extending `Input`, wrapping helpers,
  and building reusable stimulation pipelines for experiments.

```{toctree}
:maxdepth: 1

input_tutorial_01_introduction.ipynb
input_tutorial_02_functional_api.ipynb
input_tutorial_03_composable_api.ipynb
input_tutorial_04_custom_pipelines.ipynb
```
