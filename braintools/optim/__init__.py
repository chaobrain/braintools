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

    import jax.numpy as jnp
    import brainstate as bst
    from braintools.optim import Adam

    # Define a simple model
    class SimpleModel(bst.nn.Module):
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

    # Loss returning (grads, value); pass the trainable states via grad_states=
    @bst.transform.grad(grad_states=model.states(bst.ParamState), return_value=True)
    def loss_fn(data, target):
        pred = model(data)
        return jnp.mean((pred - target) ** 2)

    # Example data
    data = jnp.ones((32, 10))
    target = jnp.ones((32, 5))

    # Update step
    grads, loss = loss_fn(data, target)
    optimizer.update(grads)

**Quick Start - With Learning Rate Scheduler:**

.. code-block:: python

    import brainstate as bst
    from braintools.optim import Adam, CosineAnnealingLR

    # Create optimizer with a cosine annealing schedule
    scheduler = CosineAnnealingLR(base_lr=0.001, T_max=1000, eta_min=1e-6)
    optimizer = Adam(lr=scheduler, weight_decay=1e-4)

    optimizer.register_trainable_weights(model.states(bst.ParamState))

    # Training loop
    for epoch in range(100):
        grads, loss = loss_fn(data, target)
        optimizer.update(grads)
        # Advance the schedule explicitly each epoch
        scheduler.step()

**Gradient-Based Optimizers:**

.. code-block:: python

    from braintools.optim import (
        SGD, Momentum, MomentumNesterov, Adam, AdamW, RMSprop,
        Adagrad, Adadelta, Nadam, RAdam
    )

    # Stochastic Gradient Descent
    sgd = SGD(lr=0.01, weight_decay=1e-4)

    # Momentum (note: `Momentum` has no `nesterov` flag)
    momentum = Momentum(lr=0.01, momentum=0.9)

    # Nesterov momentum: use MomentumNesterov or SGD(nesterov=True)
    nesterov = MomentumNesterov(lr=0.01, momentum=0.9)

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

    import optax
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
    adafactor = Adafactor(lr=0.001, decay_rate=0.8)

    # Yogi (adaptive learning rate with controlled increases)
    yogi = Yogi(lr=0.01, betas=(0.9, 0.999))

    # Lookahead wraps an optax GradientTransformation (not a braintools optimizer)
    lookahead = Lookahead(
        base_optimizer=optax.adam(0.001),
        sync_period=5,
        alpha=0.5,
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
    step_lr = StepLR(base_lr=0.1, step_size=30, gamma=0.1)

    # Multi-step decay
    multistep_lr = MultiStepLR(base_lr=0.1, milestones=[30, 60, 90], gamma=0.1)

    # Exponential decay
    exp_lr = ExponentialLR(base_lr=0.1, gamma=0.95)

    # Cosine annealing
    cosine_lr = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=1e-6)

    # Polynomial decay
    poly_lr = PolynomialLR(base_lr=0.1, total_iters=1000, power=2.0)

    # Warmup then base LR (warmup over `warmup_epochs`, starting at `warmup_start_lr`)
    warmup_lr = WarmupScheduler(
        base_lr=0.001,
        warmup_epochs=1000,
        warmup_start_lr=1e-6,
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

    # Warmup + cosine schedule (warmup over `warmup_steps`, decay to `eta_min`)
    warmup_cosine = WarmupCosineSchedule(
        base_lr=0.001,
        warmup_steps=1000,
        total_steps=10000,
        eta_min=1e-6
    )

**SciPy Optimization:**

.. code-block:: python

    from braintools.optim import ScipyOptimizer

    # `loss_fun` and `bounds` are required; the loss signature follows `bounds`
    # (positional for sequence bounds, keyword for dict bounds).
    def loss(x, y):
        return (x - 1.0) ** 2 + (y + 2.0) ** 2

    bounds = [(-5.0, 5.0), (-3.0, 3.0)]

    # Gradient-based optimization (gradients via JAX autodiff)
    scipy_opt = ScipyOptimizer(
        loss_fun=loss,
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 1000},
    )
    result = scipy_opt.minimize(n_iter=1)

    # Gradient-free optimization (no Jacobian is built/passed)
    nelder_mead = ScipyOptimizer(
        loss_fun=loss,
        bounds=bounds,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8},
    )

**Nevergrad Optimization:**

.. code-block:: python

    from braintools.optim import NevergradOptimizer

    # Batched objective: each argument has shape (n_sample,); returns one loss
    # per candidate. The algorithm name is passed via `method=`.
    def batched_loss(x, y):
        return x ** 2 + y ** 2

    bounds = [(-5.0, 5.0), (-3.0, 3.0)]

    # Differential evolution
    ng_de = NevergradOptimizer(
        batched_loss, bounds, n_sample=16, method='TwoPointsDE', budget=1000,
    )
    best = ng_de.minimize(n_iter=10, verbose=False)

    # CMA-ES (Covariance Matrix Adaptation)
    ng_cma = NevergradOptimizer(
        batched_loss, bounds, n_sample=16, method='CMA', budget=2000,
    )

    # Particle swarm optimization
    ng_pso = NevergradOptimizer(
        batched_loss, bounds, n_sample=32, method='PSO', budget=1000,
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
        ChainedScheduler, SequentialLR, ConstantLR, ExponentialLR,
        WarmupScheduler, CosineAnnealingLR,
        ReduceLROnPlateau, PiecewiseConstantSchedule
    )

    # Chain multiple schedulers (all advanced together on each step)
    scheduler = ChainedScheduler([
        WarmupScheduler(base_lr=0.001, warmup_epochs=1000),
        CosineAnnealingLR(base_lr=0.001, T_max=9000)
    ])

    # Sequential schedulers (switch at milestones)
    sequential = SequentialLR(
        schedulers=[
            ConstantLR(base_lr=0.001),
            ExponentialLR(base_lr=0.001, gamma=0.95)
        ],
        milestones=[5000]
    )

    # Reduce on plateau (requires manual metric tracking via scheduler.step(metric))
    reduce_plateau = ReduceLROnPlateau(
        base_lr=0.01,
        factor=0.5,
        patience=10,
        mode='min'
    )

    # Piecewise constant (values are absolute learning rates, not multipliers)
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

# SOFO optimizers
from ._sofo_optimizer import (
    SOFO,
    SOFOScan,
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

    # SOFO optimizers
    'SOFO',
    'SOFOScan',
]
