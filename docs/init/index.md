# Initialization

`braintools.init` describes *how to create* parameter arrays: weights, biases, time constants, initial states. An initializer is a small object that, called with a shape, returns the values, with `brainunit` units carried through.

## What you'll find here

- **Basic distributions**: `Constant`, `ZeroInit`, `Uniform`, `Normal`, `TruncatedNormal`, `LogNormal`, `Gamma`, `Exponential`, `Weibull`, and `Beta`, each checked against its exact mean and standard deviation; reproducible sampling with `rng=` or `brainstate.random.seed`; units; truncation versus clipping.
- **Variance scaling and orthogonal methods**: Kaiming, Xavier, and LeCun as presets of `VarianceScaling`, the `(in, out)` / `(*spatial, in, out)` fan convention shared with `brainstate` layers, a signal-propagation experiment, and `Orthogonal`, `DeltaOrthogonal`, and `Identity` for recurrent and very deep networks.
- **Distance profiles**: the exact formula of every profile, unit handling, why `DoGProfile` and `MexicanHatProfile` are clipped at zero, composing profiles with arithmetic, `.clip`, `.apply`, and `|`, and turning a profile into a connectivity pattern.
- **Composite and distance-modulated initialization**: arithmetic on initializers, `Compose`, `Scaled`, `Clipped`, `Mixture`, `Conditional`, `DistanceModulated` with positions or distances, a spatial E/I weight matrix, and `param`.

For the full list of classes and parameters, see the {doc}`API reference <../apis/init>`.

```{toctree}
:maxdepth: 1

01_basic_distributions.ipynb
02_variance_scaling_orthogonal.ipynb
03_distance_profiles.ipynb
04_composite_distance_modulated.ipynb
```
