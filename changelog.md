# Release Notes



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






