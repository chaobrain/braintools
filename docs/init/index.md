# Initialization

Comprehensive tools for initializing neural network weights and parameters with biologically realistic patterns and mathematically principled methods.

## What you'll find here

- **Basic statistical distributions** including Constant, Uniform, Normal, LogNormal, Gamma, Exponential, Weibull, Beta, and TruncatedNormal for diverse weight initialization needs
- **Advanced variance scaling** methods (Kaiming/He, Xavier/Glorot, LeCun) designed for deep networks with different activation functions, plus orthogonal initialization for RNNs
- **Distance-dependent profiles** for spatial neural connectivity, including Gaussian, Exponential, PowerLaw, DoG (Difference of Gaussians), Mexican Hat, and Bimodal patterns
- **Composite strategies** combining mixture distributions, conditional initialization, distance modulation, and profile composition for biologically realistic heterogeneous networks
- Best practices for matching initialization to network architecture, activation functions, and biological constraints with physical units (via BrainUnit)

```{toctree}
:maxdepth: 1

01_basic_distributions.ipynb
02_variance_scaling_orthogonal.ipynb
03_distance_profiles.ipynb
04_composite_distance_modulated.ipynb
```

