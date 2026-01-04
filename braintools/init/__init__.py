# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Parameter Initialization System for Neural Networks.

This module provides a comprehensive initialization system for neural network parameters,
including weights, delays, and spatial connectivity patterns. The system is designed with
composability and biological realism in mind.

**Key Features:**

- **Basic Distributions**: Constant, Uniform, Normal, LogNormal, Gamma, Exponential, etc.
- **Variance Scaling**: Xavier, Kaiming/He, LeCun initialization for deep learning
- **Orthogonal Methods**: Orthogonal, DeltaOrthogonal, Identity initialization
- **Composite Patterns**: Mixture, Conditional, Scaled, Clipped distributions
- **Distance-Based**: Spatial connectivity with Gaussian, Exponential, Power-law profiles
- **Composable Design**: Combine and transform initializations with arithmetic operations
- **Unit-Aware**: Full integration with BrainUnit for physical quantities

**Quick Start:**

.. code-block:: python

    import brainunit as u
    from braintools.init import Normal, Uniform, param

    # Basic distributions
    weight_init = Normal(mean=1.0 * u.nS, std=0.2 * u.nS)
    weights = weight_init(1000)

    # Uniform distribution for delays
    delay_init = Uniform(low=1.0 * u.ms, high=3.0 * u.ms)
    delays = delay_init(1000)

    # Use param() helper for flexible initialization
    w = param(Normal(0.5 * u.nS, 0.1 * u.nS), sizes=100)

**Basic Distributions:**

.. code-block:: python

    import numpy as np
    import brainunit as u
    from braintools.init import (
        Constant, ZeroInit, Uniform, Normal, LogNormal,
        Gamma, Exponential, TruncatedNormal, Beta, Weibull
    )

    # Constant values
    const_init = Constant(0.5 * u.nS)
    zero_init = ZeroInit(u.nS)

    # Uniform distribution
    uniform_init = Uniform(0.1 * u.nS, 1.0 * u.nS)

    # Normal distribution
    normal_init = Normal(0.5 * u.nS, 0.1 * u.nS)

    # Log-normal for positive-only values
    lognormal_init = LogNormal(0.5 * u.nS, 0.2 * u.nS)

    # Truncated normal with bounds
    truncated_init = TruncatedNormal(
        mean=0.5 * u.nS,
        std=0.2 * u.nS,
        low=0.0 * u.nS,
        high=1.0 * u.nS
    )

**Variance Scaling (Deep Learning):**

.. code-block:: python

    import numpy as np
    from braintools.init import (
        KaimingUniform, KaimingNormal,
        XavierUniform, XavierNormal,
        LecunUniform, LecunNormal
    )

    # Kaiming/He initialization (for ReLU networks)
    kaiming_init = KaimingNormal(mode='fan_in')
    weights = kaiming_init((256, 128))

    # Xavier/Glorot initialization (for tanh/sigmoid)
    xavier_init = XavierUniform()
    weights = xavier_init((256, 128))

    # LeCun initialization (for SELU networks)
    lecun_init = LecunNormal()
    weights = lecun_init((256, 128))

**Orthogonal Initialization:**

.. code-block:: python

    import numpy as np
    from braintools.init import Orthogonal, DeltaOrthogonal, Identity

    # Orthogonal matrix (for recurrent networks)
    ortho_init = Orthogonal(scale=np.sqrt(2))
    weights = ortho_init((100, 100))

    # Identity initialization
    identity_init = Identity(scale=1.0)
    weights = identity_init((100, 100))

**Composite Patterns:**

.. code-block:: python

    import numpy as np
    import brainunit as u
    from braintools.init import (
        Mixture, Conditional, Scaled, Clipped,
        Normal, Uniform
    )

    # Mixture of distributions
    mixed_init = Mixture(
        distributions=[
            Normal(0.5 * u.nS, 0.1 * u.nS),
            Uniform(0.8 * u.nS, 1.2 * u.nS)
        ],
        weights=[0.7, 0.3]
    )

    # Conditional based on neuron properties
    def is_excitatory(indices):
        return indices < 800

    conditional_init = Conditional(
        condition_fn=is_excitatory,
        true_dist=Normal(0.5 * u.nS, 0.1 * u.nS),
        false_dist=Normal(-0.3 * u.nS, 0.05 * u.nS)
    )

    # Scaled and clipped distributions
    scaled_init = Scaled(Normal(1.0 * u.nS, 0.2 * u.nS), scale_factor=0.5)
    clipped_init = Clipped(
        Normal(0.5 * u.nS, 0.3 * u.nS),
        min_val=0.0 * u.nS,
        max_val=1.0 * u.nS
    )

**Distance-Based Connectivity:**

.. code-block:: python

    import numpy as np
    import brainunit as u
    from braintools.init import (
        GaussianProfile, ExponentialProfile, DistanceModulated,
        Normal
    )

    # Gaussian distance profile
    gaussian_profile = GaussianProfile(
        sigma=50.0 * u.um,
        max_distance=200.0 * u.um
    )

    # Exponential decay
    exp_profile = ExponentialProfile(
        decay_constant=100.0 * u.um,
        max_distance=500.0 * u.um
    )

    # Distance-modulated weights
    positions_pre = np.random.uniform(0, 1000, (100, 2)) * u.um
    positions_post = np.random.uniform(0, 1000, (100, 2)) * u.um

    dist_init = DistanceModulated(
        base_dist=Normal(1.0 * u.nS, 0.2 * u.nS),
        distance_profile=gaussian_profile
    )
    weights = dist_init(
        (100, 100),
        pre_positions=positions_pre,
        post_positions=positions_post
    )

**Composable Initialization:**

.. code-block:: python

    import numpy as np
    import brainunit as u
    from braintools.init import Normal, Compose

    # Arithmetic composition
    init1 = Normal(0.5 * u.nS, 0.1 * u.nS) * 2.0 + 0.1 * u.nS

    # Clipping
    init2 = Normal(0.5 * u.nS, 0.2 * u.nS).clip(0.0 * u.nS, 1.0 * u.nS)

    # Functional composition with pipe operator
    init3 = (
        Normal(1.0 * u.nS, 0.2 * u.nS) |
        (lambda x: u.math.maximum(x, 0 * u.nS)) |
        (lambda x: x * 0.5)
    )

    # Compose class for complex chains
    init4 = Compose(
        Normal(1.0 * u.nS, 0.2 * u.nS),
        lambda x: u.math.maximum(x, 0 * u.nS),
        lambda x: x * 0.5
    )

"""

# Base classes and utilities
from ._init_base import (
    Initialization,
    Initializer,
    param,
    Compose,
)

# Basic distributions
from ._init_basic import (
    Constant,
    ZeroInit,
    Uniform,
    Normal,
    LogNormal,
    Gamma,
    Exponential,
    TruncatedNormal,
    Beta,
    Weibull,
)

# Composite patterns
from ._init_composite import (
    Mixture,
    Conditional,
    Scaled,
    Clipped,
)

# Orthogonal initialization
from ._init_orthogonal import (
    Orthogonal,
    DeltaOrthogonal,
    Identity,
)

# Variance scaling methods
from ._init_variance_scaling import (
    VarianceScaling,
    KaimingUniform,
    KaimingNormal,
    XavierUniform,
    XavierNormal,
    LecunUniform,
    LecunNormal,
)

# Distance-based initialization
from ._init_with_distance import (
    DistanceModulated,
)

# Distance profiles
from ._distance_base import (
    DistanceProfile,
    ComposedProfile,
    ClipProfile,
    ApplyProfile,
    PipeProfile,
)

from ._distance_impl import (
    GaussianProfile,
    ExponentialProfile,
    PowerLawProfile,
    LinearProfile,
    StepProfile,
    SigmoidProfile,
    DoGProfile,
    LogisticProfile,
    BimodalProfile,
    MexicanHatProfile,
)

__all__ = [
    # Base classes and utilities
    'Initialization',
    'Initializer',
    'param',
    'Compose',

    # Basic distributions
    'Constant',
    'ZeroInit',
    'Uniform',
    'Normal',
    'LogNormal',
    'Gamma',
    'Exponential',
    'TruncatedNormal',
    'Beta',
    'Weibull',

    # Composite patterns
    'Mixture',
    'Conditional',
    'Scaled',
    'Clipped',

    # Orthogonal initialization
    'Orthogonal',
    'DeltaOrthogonal',
    'Identity',

    # Variance scaling methods
    'VarianceScaling',
    'KaimingUniform',
    'KaimingNormal',
    'XavierUniform',
    'XavierNormal',
    'LecunUniform',
    'LecunNormal',

    # Distance-based initialization
    'DistanceModulated',

    # Distance profile base classes
    'DistanceProfile',
    'ComposedProfile',
    'ClipProfile',
    'ApplyProfile',
    'PipeProfile',

    # Distance profile implementations
    'GaussianProfile',
    'ExponentialProfile',
    'PowerLawProfile',
    'LinearProfile',
    'StepProfile',
    'SigmoidProfile',
    'DoGProfile',
    'LogisticProfile',
    'BimodalProfile',
    'MexicanHatProfile',
]
