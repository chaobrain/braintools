# Surrogate Gradients

Surrogate gradients provide differentiable approximations for non-differentiable spiking neuron operations. This section collects practical guides that show how to work with the surrogate module, from first principles to advanced customization.

- `01_basics.ipynb` walks through the rationale behind surrogate gradients, demonstrates the built-in surrogate functions, and highlights common training patterns for spiking networks.
- `02_customizing.ipynb` dives deeper into extending the library, illustrating how to register custom surrogate kernels, combine them with existing layers, and validate their gradients.

```{toctree}
:maxdepth: 1

01_basics.ipynb
02_customizing.ipynb
```

