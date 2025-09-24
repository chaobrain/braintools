# Release Notes



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






