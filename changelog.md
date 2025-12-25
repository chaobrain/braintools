# Release Notes


## Version 0.1.6

### New Features

#### Parameter Management Expansion (`braintools.param`)
- **Hierarchical data container**: Added `Data` for composed state storage and cloning.
- **Parameter wrappers**: Added `Param` and `Const` with built-in transforms and optional regularization.
- **State containers**: Added `ArrayHidden` and `ArrayParam` with transform-aware `.data` access.
- **Regularization priors**: Added `GaussianReg`, `L1Reg`, and `L2Reg` with optional trainable hyperparameters.
- **Utilities**: Added `get_param()` and `get_size()` helpers for parameter/state handling.

#### Transforms
- **New `ReluT` transform** for lower-bounded parameters.
- **Expanded transform suite** now includes `PositiveT`, `NegativeT`, `ScaledSigmoidT`, `PowerT`,
  `OrderedT`, `SimplexT`, and `UnitVectorT`.

### Improvements

#### API Consistency
- **Transform naming cleanup**: Standardized transform class names with the `*T` suffix
  (e.g., `SigmoidT`, `SoftplusT`, `AffineT`, `ChainT`, `MaskedT`, `ClipT`).

#### Documentation
- **Expanded param API docs**: Added sections for data containers, state containers, regularization,
  utilities, and updated transform listings in `docs/apis/param.rst`.
- **API index update**: Added `param` API page to `docs/index.rst`.

#### Tests
- **New test coverage**: Added tests for data containers, modules, regularization, state, transforms,
  and utilities across the param module.

### Breaking Changes
- **Transform API renames**: Transform classes now use the `*T` suffix (e.g., `Sigmoid` -> `SigmoidT`).
- **Custom transform removed**: The `Custom` transform is no longer part of the public API.

### Bug Fixes
- **Initializer RNG**: `TruncatedNormal` now defaults to `numpy.random` when no RNG is provided.


## Version 0.1.5

### New Features

#### Parameter Transformation Module (`braintools.param`)
- **7 new bijective transforms** for constrained optimization and probabilistic modeling:
  - **Positive**: Constrains parameters to (0, +∞) using exponential transformation
  - **Negative**: Constrains parameters to (-∞, 0) using negative softplus
  - **ScaledSigmoid**: Sigmoid with adjustable sharpness/temperature parameter (beta)
  - **Power**: Box-Cox family power transformation for variance stabilization
  - **Ordered**: Ensures monotonically increasing output vectors (useful for cutpoints in ordinal regression)
  - **Simplex**: Stick-breaking transformation for probability vectors summing to 1
  - **UnitVector**: Projects vectors onto the unit sphere (L2 norm = 1)
- **Jacobian computation**: Added `log_abs_det_jacobian()` method to Transform base class and implementations for probabilistic modeling
  - Implemented for: Identity, Sigmoid, Softplus, Log, Exp, Affine, Chain, Positive

#### Surrogate Gradient Enhancements (`braintools.surrogate`)

- Gradient computation of hyperparameters of surrogate gradient functions.
- Fix batching issue in surrogate gradient functions


### Improvements

#### API Enhancements
- **`__repr__` methods**: Added string representations to all Transform classes and Param class for better debugging
- **Enhanced documentation**: Updated `docs/apis/param.rst` with comprehensive API reference
  - Organized sections: Base Classes, Parameter Wrapper, Bounded Transforms, Positive/Negative Transforms, Advanced Transforms, Composition Transforms
  - Descriptive explanations for each transform's use case

#### Code Quality
- **Comprehensive test coverage**: Added 28 new tests for param module (45 total tests passing)
  - Tests for all new transforms: roundtrip, constraints, repr methods
  - Tests for `log_abs_det_jacobian` correctness
  - Tests for edge cases and numerical stability




## Version 0.1.4

### New Features

#### Learning Rate Scheduler Enhancements (`braintools.optim`)
- **New `apply()` method**: Added `apply()` method to all LR schedulers for more flexible learning rate application
  - Allows applying learning rate transformations without stepping the scheduler
  - Useful for custom training loops and learning rate inspection
- **Comprehensive test coverage**: Added 118+ comprehensive tests covering all 17 learning rate schedulers
  - Tests for basic functionality, optimizer integration, JIT compilation, state persistence
  - Full coverage of edge cases and special modes for each scheduler
  - Validates correctness with `@brainstate.transform.jit` compilation

### Improvements

#### Documentation
- **Restructured tutorial organization**: Renamed and reorganized documentation files for better clarity
  - Moved module tutorials into subdirectories (`conn/`, `init/`, `input/`, `file/`, `surrogate/`)
  - Updated table of contents structure across all modules
  - Improved navigation with consolidated index files (`index.md` instead of `toc_*.md`)
- **Enhanced visual branding**: Updated project logo from JPG to high-resolution PNG format
  - Better quality and transparency support
  - Consistent branding across documentation

#### Code Quality
- **Test improvements**: Refactored scheduler tests with better organization and coverage
  - Each scheduler now has 5-10 dedicated tests
  - Tests verify: basic functionality, optimizer integration, JIT compilation, multiple param groups, state dict save/load
  - Discovered and documented key implementation behaviors (epoch counting, initialization patterns)

#### CI/CD
- **Updated GitHub Actions**: Bumped actions to latest versions for improved security and performance
  - `actions/download-artifact`: v5 → v6
  - `actions/upload-artifact`: v4 → v5
  - Better artifact handling in CI pipeline

### Bug Fixes
- Fixed edge cases in learning rate scheduler state management
- Corrected epoch counting behavior in milestone-based schedulers
- Improved JIT compilation compatibility for all schedulers

### Notes
- All 17 learning rate schedulers now have comprehensive test coverage (100%)
- Enhanced reliability for training workflows with thorough validation
- Improved developer experience with better documentation structure


## Version 0.1.0

### Major Features

#### Surrogate Gradients Module (`braintools.surrogate`)
- **New comprehensive surrogate gradient system** for training spiking neural networks (SNNs)
- **18+ surrogate gradient functions** with straight-through estimator support:
  - **Sigmoid-based**: `Sigmoid`, `SoftSign`, `Arctan`, `ERF`
  - **Piecewise**: `PiecewiseQuadratic`, `PiecewiseExp`, `PiecewiseLeakyRelu`
  - **ReLU-based**: `ReluGrad`, `LeakyRelu`, `LogTailedRelu`
  - **Distribution-inspired**: `GaussianGrad`, `MultiGaussianGrad`, `InvSquareGrad`, `SlayerGrad`
  - **Advanced**: `S2NN`, `QPseudoSpike`, `SquarewaveFourierSeries`, `NonzeroSignLog`
- **Customizable hyperparameters** (alpha, sigma, width, etc.) for fine-tuning gradient behavior
- **Comprehensive tutorials**: 2 detailed notebooks covering basics and customization
- Enables gradient-based training of SNNs via backpropagation through time
- Over 2,600 lines of implementation with extensive test coverage

### New Features

#### Learning Rate Schedulers (`braintools.optim`)
- **ExponentialDecayLR scheduler**: Fine-grained exponential decay with step-based control
  - Support for transition steps, staircase mode, delayed start, and bounded decay
  - Better control than epoch-based ExponentialLR for step-level scheduling
  - Compatible with Optax's exponential_decay schedule

### Improvements

#### API Refinements
- **Deprecation warnings** added for future API changes:
  - Deprecated `beta1` and `beta2` parameters in Adam optimizer (use `b1` and `b2` instead)
  - Deprecated `unit` parameter in various initializers (use `UNITLESS` by default)
  - Deprecated `init_call` function replaced with `param` for improved consistency
- **Enhanced state management**: Refactored `UniqueStateManager` to utilize pytree methods
- **Comprehensive tests**: Added extensive tests for `UniqueStateManager` methods and edge cases

#### Documentation
- Updated API documentation for new surrogate gradient module
- Added learning rate scheduler documentation for `ExponentialDecayLR`
- Enhanced optimizer tutorials with updated examples
- Clarified docstrings for `FixedProb` class and variance scaling initializer

### Code Quality

#### Internal Improvements
- Updated copyright information from BDP Ecosystem Limited to BrainX Ecosystem Limited
- Improved consistency across codebase with standardized function signatures
- Better default parameter handling (`UNITLESS` for unit parameters)
- Enhanced test coverage for state management and optimizers

#### Metric Enhancements
- Improved correlation and firing metrics implementation
- Enhanced LFP (Local Field Potential) analysis functions
- Better error handling and validation in metric computations

### Breaking Changes
- **Deprecation notices** (not yet removed, but will be in future versions):
  - `beta1`/`beta2` parameters in Adam optimizer (use `b1`/`b2`)
  - `unit` parameter in initializers (defaults to `UNITLESS`)
  - `init_call` function (use `param` instead)

### Notes
- This release focuses on enabling gradient-based training for spiking neural networks
- The surrogate gradient module is a major addition for neuromorphic computing and SNN research
- Enhanced learning rate scheduling provides more control for training workflows






## Version 0.0.14

### New Features

#### Optimizer Enhancements (`braintools.optim`)
- **Momentum optimizers**: Added `Momentum` and `MomentumNesterov` optimizers with gradient transformations
- **Improved state management**: Refactored optimizer state handling with new `OptimState` class for better encapsulation

#### Initialization Updates (`braintools.init`)
- **ZeroInit initializer**: New zero initialization class for weights and parameters
- **VarianceScaling export**: Added `VarianceScaling` to module exports for easier access

### Improvements
- Enhanced optimizer state management for better performance and maintainability
- Simplified initialization API with additional export options
- Updated documentation for new initialization methods

### Internal Changes
- Refactored test structure for initialization module
- Improved learning rate scheduler implementation


## Version 0.0.13

### Major Features

#### New Initialization Framework (`braintools.init`)
- **Unified initialization API** consolidating all weight and parameter initialization strategies
- **Distance-based initialization**: Support for distance-modulated weight patterns
- **Variance scaling strategies**: Xavier, He, LeCun initialization methods
- **Orthogonal initialization** for improved training stability
- **Composite distributions** for complex initialization patterns
- Simplified API with consistent parameter naming across all initializers

#### Advanced Connectivity Patterns (`braintools.conn`)
- **Topological network patterns**:
  - Small-world and scale-free networks
  - Hierarchical and core-periphery structures
  - Modular and clustered random connectivity
- **Enhanced biological connectivity**:
  - Excitatory-inhibitory balanced networks
  - Distance-dependent connectivity with multiple profiles
  - Compartment-specific connectivity (dendrite, soma, axon)
- **Spatial connectivity improvements**:
  - 2D convolutional kernels for spatial networks
  - Position-based connectivity with normalization
  - Distance modulation using composable profiles

#### Comprehensive Optax Integration (`braintools.optim`)
- **Full Optax optimizer support**: Adam, SGD, RMSProp, AdaGrad, AdaDelta, and more
- **Advanced learning rate schedulers**:
  - Cosine annealing with warm restarts
  - Polynomial decay with warmup
  - Piecewise constant schedules
  - Sequential and chained schedulers
- **Improved optimizer state management** with unique state handling
- **Parameter groups** with per-group learning rates

### Improvements

#### API Enhancements
- Simplified `conn` module API with direct class access
- Refactored initialization calls for consistency
- Improved type annotations throughout
- Better default parameter handling

#### Documentation & Tutorials
- Updated tutorial structure for connectivity patterns
- New examples for topological networks
- Enhanced API documentation with detailed examples
- Improved code readability in tutorials

### Code Quality
- Comprehensive test coverage for new features
- Better error handling and validation
- Consistent naming conventions
- Removed deprecated and redundant code

### Breaking Changes
- Renamed `PointNeuronConnectivity` to `PointConnectivity`
- Renamed `ConvKernel` to `Conv2dKernel`
- Unified initializer names (e.g., `ConstantWeight` → `Constant`)
- Removed `PopulationRateConnectivity` class
- Changed some parameter names for clarity (e.g., unified use of `rng` parameter)


## Version 0.0.12

### Major Features

#### Comprehensive Visualization System
- **New visualization modules** for neural data analysis:
  - `neural.py`: Spike rasters, population activity, connectivity matrices, firing rate maps
  - `three_d.py`: 3D visualizations for neural networks, brain surfaces, trajectories, electrode arrays
  - `statistical.py`: Statistical plotting tools (confusion matrices, ROC curves, correlation plots)
  - `interactive.py`: Interactive visualizations with Plotly support
  - `colormaps.py`: Neural-specific colormaps and publication-ready styling
- **15+ new tutorial notebooks** covering all visualization techniques
- **Brain-specific colormaps** for membrane potential, spike activity, and connectivity

#### Enhanced Numerical Integration
- **New ODE integrators**: 
  - Runge-Kutta methods: RK23, RK45, RKF45, DOP853, DOPRI5, SSPRK33
  - Specialized methods: Midpoint, Heun, RK4(3/8), Ralston RK2/RK3, Bogacki-Shampine
- **New SDE integrators**: Heun, Tamed Euler, Implicit Euler, SRK2, SRK3, SRK4
- **IMEX integrators** for stiff equations: Euler, ARS(2,2,2), CNAB
- **DDE integrators** for delay differential equations
- Comprehensive test coverage and accuracy verification

#### Advanced Spike Processing
- **Spike encoders**: Rate, Poisson, Population, Latency, and Temporal encoders
- **Enhanced spike operations** with bitwise functionality
- **Spike metrics**: Victor-Purpura distance, spike train synchrony, correlation indices
- Tutorial notebooks for spike encoding and analysis

#### New Optimization Framework
- **NevergradOptimizer**: Integration with Nevergrad optimization library
- **ScipyOptimizer**: Enhanced scipy optimization with flexible bounds support
- Refactored optimizer architecture for better extensibility
- Support for dict and sequence parameter bounds

### Improvements

#### File Management
- Enhanced msgpack serialization with mismatch handling options
- Improved checkpoint loading with better error recovery
- Support for handling mismatched keys during state restoration

#### Metrics and Analysis
- **LFP analysis functions**: Power spectral density, coherence analysis, phase-amplitude coupling
- **Functional connectivity**: Dynamic connectivity computation
- **Classification metrics**: Binary, multiclass, focal loss, and smoothing techniques
- **Regression losses**: MSE, MAE, Huber, and quantile losses

### Documentation
- Added comprehensive API documentation for all new modules
- Created tutorials for:
  - ODE/SDE integration methods
  - Classification and regression losses
  - Pairwise and embedding similarity
  - Spiking metrics and LFP analysis
  - Advanced neural visualization techniques
- Updated project description from "brain modeling" to "brain simulation"
- Changed references from BrainPy to BrainTools throughout

### Code Quality
- Added extensive unit tests for all new modules
- Improved type hints and parameter documentation
- Better error handling and validation
- Consistent API design across modules

### Breaking Changes
- Refactored optimizer module structure (moved from single `optimizer.py` to separate modules)
- Removed unused key parameter from spike encoder methods
- Updated some function signatures for clarity

### Bug Fixes
- Fixed Softplus unit scaling issues
- Corrected paths in publish workflow
- Fixed formatting in ODE integrator documentation
- Resolved msgpack checkpoint handling errors






