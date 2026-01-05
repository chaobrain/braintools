# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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
Optimization Algorithms and Learning Rate Schedulers.

This module provides a comprehensive collection of optimization algorithms and learning
rate schedulers for training neural networks and spiking neural networks. It includes
modern deep learning optimizers (Adam, SGD, etc.), specialized optimizers for scientific
computing (SciPy, Nevergrad), and flexible learning rate scheduling strategies.

**Key Features:**

- **Gradient-Based Optimizers**: Adam, SGD, RMSprop, Adagrad, and variants
- **Advanced Optimizers**: AdamW, RAdam, Lamb, Lion, AdaBelief, etc.
- **SciPy Integration**: Gradient-free and constrained optimization
- **Nevergrad Integration**: Black-box optimization with evolutionary strategies
- **Learning Rate Schedulers**: Step, exponential, cosine, warmup, and custom schedules
- **PyTorch-like Interface**: Familiar API for PyTorch users
- **JAX/Optax Backend**: High-performance optimization with automatic differentiation

**Quick Start - Basic Optimization:**

.. code-block:: python

    import brainstate as bst
    from braintools.optim import Adam

    # Define a simple model
    class SimpleModel(bst.Module):
        def __init__(self):
            super().__init__()
            self.w = bst.ParamState(jnp.zeros((10, 5)))
            self.b = bst.ParamState(jnp.zeros(5))

        def __call__(self, x):
            return jnp.dot(x, self.w.value) + self.b.value

    # Create model and optimizer
    model = SimpleModel()
    optimizer = Adam(lr=0.001)

    # Register trainable parameters
    optimizer.register_trainable_weights(model.states(bst.ParamState))

    # Training step
    @bst.transform.grad(model.states(bst.ParamState), return_value=True)
    def loss_fn(data, target):
        pred = model(data)
        return jnp.mean((pred - target) ** 2)

    # Update step
    grads, loss = loss_fn(data, target)
    optimizer.update(grads)

**Quick Start - With Learning Rate Scheduler:**

.. code-block:: python

    from braintools.optim import Adam, CosineAnnealingLR

    # Create optimizer with cosine annealing schedule
    scheduler = CosineAnnealingLR(T_max=1000, eta_min=1e-6)
    optimizer = Adam(lr=scheduler, weight_decay=1e-4)

    optimizer.register_trainable_weights(model.states(bst.ParamState))

    # Training loop
    for epoch in range(100):
        grads, loss = loss_fn(data, target)
        optimizer.update(grads)
        # Scheduler step is handled automatically

**Gradient-Based Optimizers:**

.. code-block:: python

    from braintools.optim import (
        SGD, Momentum, Adam, AdamW, RMSprop,
        Adagrad, Adadelta, Nadam, RAdam
    )

    # Stochastic Gradient Descent
    sgd = SGD(lr=0.01, weight_decay=1e-4)

    # Momentum
    momentum = Momentum(lr=0.01, momentum=0.9, nesterov=True)

    # Adam (most popular)
    adam = Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    # AdamW (Adam with decoupled weight decay)
    adamw = AdamW(lr=0.001, weight_decay=0.01)

    # RMSprop
    rmsprop = RMSprop(lr=0.001, alpha=0.99, eps=1e-8)

    # Adagrad (adaptive learning rates)
    adagrad = Adagrad(lr=0.01, eps=1e-10)

    # Adadelta (extension of Adagrad)
    adadelta = Adadelta(lr=1.0, rho=0.9, eps=1e-6)

    # Nadam (Adam + Nesterov momentum)
    nadam = Nadam(lr=0.001, betas=(0.9, 0.999))

    # RAdam (rectified Adam)
    radam = RAdam(lr=0.001, betas=(0.9, 0.999))

**Advanced Optimizers:**

.. code-block:: python

    from braintools.optim import (
        Lamb, Lars, Lion, AdaBelief,
        Adafactor, Yogi, Lookahead
    )

    # Lamb (for large batch training)
    lamb = Lamb(lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)

    # Lars (layer-wise adaptive rate scaling)
    lars = Lars(lr=0.01, momentum=0.9, weight_decay=1e-4)

    # Lion (evolved sign momentum)
    lion = Lion(lr=0.0001, betas=(0.9, 0.99), weight_decay=0.01)

    # AdaBelief (adapting stepsizes by belief in gradient direction)
    adabelief = AdaBelief(lr=0.001, betas=(0.9, 0.999), eps=1e-16)

    # Adafactor (memory-efficient adaptive learning rates)
    adafactor = Adafactor(lr=0.001, min_dim_size_to_factor=128)

    # Yogi (adaptive learning rate with controlled increases)
    yogi = Yogi(lr=0.01, betas=(0.9, 0.999))

    # Lookahead (wrapper for other optimizers)
    lookahead = Lookahead(
        base_optimizer=Adam(lr=0.001),
        sync_period=5,
        slow_step_size=0.5
    )

**Learning Rate Schedulers:**

.. code-block:: python

    from braintools.optim import (
        StepLR, MultiStepLR, ExponentialLR,
        CosineAnnealingLR, PolynomialLR,
        WarmupScheduler, OneCycleLR, CyclicLR,
        WarmupCosineSchedule
    )

    # Step decay
    step_lr = StepLR(initial_lr=0.1, step_size=30, gamma=0.1)

    # Multi-step decay
    multistep_lr = MultiStepLR(initial_lr=0.1, milestones=[30, 60, 90], gamma=0.1)

    # Exponential decay
    exp_lr = ExponentialLR(initial_lr=0.1, gamma=0.95)

    # Cosine annealing
    cosine_lr = CosineAnnealingLR(initial_lr=0.1, T_max=100, eta_min=1e-6)

    # Polynomial decay
    poly_lr = PolynomialLR(initial_lr=0.1, total_steps=1000, power=2.0)

    # Warmup then constant
    warmup_lr = WarmupScheduler(
        warmup_steps=1000,
        peak_lr=0.001,
        init_lr=1e-6
    )

    # One-cycle policy
    onecycle_lr = OneCycleLR(
        max_lr=0.01,
        total_steps=1000,
        pct_start=0.3,
        div_factor=25.0
    )

    # Cyclic learning rate
    cyclic_lr = CyclicLR(
        base_lr=0.001,
        max_lr=0.01,
        step_size_up=2000,
        mode='triangular'
    )

    # Warmup + cosine schedule
    warmup_cosine = WarmupCosineSchedule(
        warmup_steps=1000,
        total_steps=10000,
        peak_lr=0.001,
        end_lr=1e-6
    )

**SciPy Optimization:**

.. code-block:: python

    from braintools.optim import ScipyOptimizer

    # Use SciPy's BFGS for gradient-based optimization
    scipy_opt = ScipyOptimizer(
        method='BFGS',
        options={'maxiter': 1000, 'gtol': 1e-6}
    )

    # Use Nelder-Mead for gradient-free optimization
    nelder_mead = ScipyOptimizer(
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8}
    )

    # Constrained optimization with bounds
    constrained = ScipyOptimizer(
        method='L-BFGS-B',
        bounds=[(0, 1), (-10, 10)],
        options={'maxiter': 1000}
    )

**Nevergrad Optimization:**

.. code-block:: python

    from braintools.optim import NevergradOptimizer

    # Differential evolution
    ng_de = NevergradOptimizer(
        optimizer='TwoPointsDE',
        budget=1000,
        num_workers=4
    )

    # CMA-ES (Covariance Matrix Adaptation)
    ng_cma = NevergradOptimizer(
        optimizer='CMA',
        budget=2000,
        num_workers=1
    )

    # Particle swarm optimization
    ng_pso = NevergradOptimizer(
        optimizer='PSO',
        budget=1000,
        num_workers=8
    )

**Gradient Clipping:**

.. code-block:: python

    from braintools.optim import Adam

    # Clip by global norm
    optimizer = Adam(lr=0.001, grad_clip_norm=1.0)

    # Clip by value
    optimizer = Adam(lr=0.001, grad_clip_value=0.5)

**Weight Decay:**

.. code-block:: python

    from braintools.optim import SGD, AdamW

    # L2 regularization (coupled with gradients)
    sgd = SGD(lr=0.01, weight_decay=1e-4)

    # Decoupled weight decay (better for Adam-like optimizers)
    adamw = AdamW(lr=0.001, weight_decay=0.01)

**Advanced Scheduler Patterns:**

.. code-block:: python

    from braintools.optim import (
        ChainedScheduler, SequentialLR,
        ReduceLROnPlateau, PiecewiseConstantSchedule
    )

    # Chain multiple schedulers
    scheduler = ChainedScheduler([
        WarmupScheduler(warmup_steps=1000, peak_lr=0.001),
        CosineAnnealingLR(initial_lr=0.001, T_max=9000)
    ])

    # Sequential schedulers (switch at milestones)
    sequential = SequentialLR(
        schedulers=[
            ConstantLR(0.001),
            ExponentialLR(initial_lr=0.001, gamma=0.95)
        ],
        milestones=[5000]
    )

    # Reduce on plateau (requires manual metric tracking)
    reduce_plateau = ReduceLROnPlateau(
        initial_lr=0.01,
        factor=0.5,
        patience=10,
        mode='min'
    )

    # Piecewise constant
    piecewise = PiecewiseConstantSchedule(
        boundaries=[1000, 5000, 8000],
        values=[0.1, 0.01, 0.001, 0.0001]
    )

"""

# Base classes
from ._base import (
    Optimizer,
    OptimState,
)

# SciPy optimizer
from ._scipy_optimizer import (
    ScipyOptimizer,
)

# Nevergrad optimizer
from ._nevergrad_optimizer import (
    NevergradOptimizer,
)

# State management utilities
from ._state_uniquifier import (
    UniqueStateManager,
)

# Learning rate schedulers
from ._optax_lr_scheduler import (
    LRScheduler,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    ExponentialDecayLR,
    CosineAnnealingLR,
    PolynomialLR,
    WarmupScheduler,
    CyclicLR,
    OneCycleLR,
    ReduceLROnPlateau,
    LinearLR,
    ConstantLR,
    ChainedScheduler,
    SequentialLR,
    CosineAnnealingWarmRestarts,
    WarmupCosineSchedule,
    PiecewiseConstantSchedule,
)

# Optax-based optimizers
from ._optax_optimizer import (
    OptaxOptimizer,
    SGD,
    Momentum,
    MomentumNesterov,
    Adam,
    AdamW,
    Adagrad,
    Adadelta,
    RMSprop,
    Adamax,
    Nadam,
    RAdam,
    Lamb,
    Lars,
    Lookahead,
    Yogi,
    LBFGS,
    Rprop,
    Adafactor,
    AdaBelief,
    Lion,
    SM3,
    Novograd,
    Fromage,
)

__all__ = [
    # Base classes
    'Optimizer',
    'OptimState',

    # SciPy optimizer
    'ScipyOptimizer',

    # Nevergrad optimizer
    'NevergradOptimizer',

    # State management
    'UniqueStateManager',

    # Learning rate schedulers
    'LRScheduler',
    'StepLR',
    'MultiStepLR',
    'ExponentialLR',
    'ExponentialDecayLR',
    'CosineAnnealingLR',
    'PolynomialLR',
    'WarmupScheduler',
    'CyclicLR',
    'OneCycleLR',
    'ReduceLROnPlateau',
    'LinearLR',
    'ConstantLR',
    'ChainedScheduler',
    'SequentialLR',
    'CosineAnnealingWarmRestarts',
    'WarmupCosineSchedule',
    'PiecewiseConstantSchedule',

    # Optax optimizers
    'OptaxOptimizer',
    'SGD',
    'Momentum',
    'MomentumNesterov',
    'Adam',
    'AdamW',
    'Adagrad',
    'Adadelta',
    'RMSprop',
    'Adamax',
    'Nadam',
    'RAdam',
    'Lamb',
    'Lars',
    'Lookahead',
    'Yogi',
    'LBFGS',
    'Rprop',
    'Adafactor',
    'AdaBelief',
    'Lion',
    'SM3',
    'Novograd',
    'Fromage',
]
