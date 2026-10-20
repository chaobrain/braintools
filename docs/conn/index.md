# Synaptic Connectivity

Comprehensive tools for generating neural connectivity patterns across different types of neural models.

## What you'll find here

- A **basic connectivity** tutorial covering `Random` / `FixedProb`, `AllToAll`, and `OneToOne`, the `ConnectionResult` fields, result caching and `recompute=True`, weight and delay initializers with `brainunit` units, and dense/sparse conversion with `weight2dense()`, `delay2matrix()`, and `weight2csr()`.
- A **spatial connectivity** tutorial covering position arrays, distance profiles (`GaussianProfile`, `ExponentialProfile`, `MexicanHatProfile`), the `Gaussian` / `Exponential` / `DistanceDependent` connectors, precomputed distances, `Ring` and `Grid2d` topologies, `ClusteredRandom` and `RadialPatches`, and `result.get_distances()`.
- A **network topologies** tutorial covering `SmallWorld`, `ScaleFree`, `Regular`, `ModularRandom`, `ModularGeneral`, `HierarchicalRandom`, and `CorePeripheryRandom`, with degree, block-density, clustering, path-length, and resilience analyses.
- An **advanced patterns** tutorial covering `ExcitatoryInhibitory` networks, spatial kernels (`GaussianKernel`, `GaborKernel`, `DoGKernel`, `MexicanHat`), convolution kernels (`Conv2dKernel`, `SobelKernel`, `LaplacianKernel`), `CustomKernel`, and composing spatial E/I projections into a single `ConnectionResult`.
- A **composition and multi-compartment** tutorial covering the connector operators (`+` union, `*` intersection, `-` difference, weight/delay scaling) and how they combine weights and delays, plus multi-compartment connectivity: compartment constants, `CompartmentSpecific` and the anatomical shortcuts, `MorphologyDistance`, `DendriticTree`, `AxonalProjection`, `SynapticPlacement`, `AxonalBranching`, `AxonalArborization`, `CustomCompartment`, and per-compartment analysis.
- A **population networks and scale** tutorial covering the fixed-degree connectors (`FixedInDegree`, `FixedOutDegree`, `FixedTotalNum`), `Population` / `Projection` / `Network` block connectivity, export to `brainevent` formats with an event-driven E/I spiking simulation, the structural graph measures of `braintools.analysis.graph` (degree distribution, clustering, path length, small-world index, modularity, motifs, spectral radius), and building 1e5-neuron networks.

For the full list of connectors and their parameters, see the {doc}`API reference <../apis/conn>`.

```{toctree}
:maxdepth: 1

01_basic_connectivity.ipynb
02_spatial_connectivity.ipynb
03_network_topologies.ipynb
04_advanced_patterns.ipynb
05_composition_and_compartments.ipynb
06_networks_and_scale.ipynb
```
