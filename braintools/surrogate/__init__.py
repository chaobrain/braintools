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
Surrogate Gradient Functions for Spiking Neural Networks.

This module provides a comprehensive collection of surrogate gradient functions
for training spiking neural networks (SNNs). Surrogate gradients replace the
non-differentiable spike function (Heaviside step) with smooth approximations
during backpropagation, enabling gradient-based optimization of SNNs.

**Key Features:**

- **Differentiable Spikes**: Enable backpropagation through discrete spike events
- **Multiple Surrogate Types**: Sigmoid, Gaussian, ReLU-based, piecewise, and more
- **Class and Functional APIs**: Both object-oriented and functional interfaces
- **JAX Integration**: Full JIT compilation, vmap, and autodiff support
- **Customizable**: Easy to create custom surrogate gradient functions
- **Research-Backed**: Implementations from published SNN training literature

**Quick Start - Basic Usage:**

.. code-block:: python

    import brainstate as bst
    import jax.numpy as jnp
    from braintools.surrogate import Sigmoid, sigmoid

    # Class-based API
    spike_fn = Sigmoid(alpha=4.0)
    x = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    spikes = spike_fn(x)  # Forward: step function [0., 0., 1., 1., 1.]
    # Backward: smooth sigmoid gradient

    # Functional API
    spikes = sigmoid(x, alpha=4.0)  # Same result

**Quick Start - Training a Spiking Layer:**

.. code-block:: python

    import brainstate as bst
    import brainunit as u
    import jax.numpy as jnp
    from braintools.surrogate import Sigmoid

    # Define a simple spiking neuron layer
    class LIFLayer(bst.nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = bst.nn.Linear(in_features, out_features)
            self.v = bst.State(jnp.zeros(out_features))
            self.spike_fn = Sigmoid(alpha=4.0)

        def __call__(self, x):
            # Integrate input
            self.v.value = 0.9 * self.v.value + self.linear(x)
            # Generate spikes with surrogate gradient
            spikes = self.spike_fn(self.v.value - 1.0)
            # Reset
            self.v.value = jnp.where(spikes > 0, 0.0, self.v.value)
            return spikes

    # Create and use the layer
    layer = LIFLayer(100, 50)
    x = jnp.ones(100)
    output_spikes = layer(x)

    # Gradients work through the spike function!
    @bst.transform.grad(layer.states(bst.ParamState))
    def loss_fn(x, target):
        return jnp.mean((layer(x) - target) ** 2)

    grads = loss_fn(x, jnp.zeros(50))

**Sigmoid-Based Surrogates:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.surrogate import (
        Sigmoid, SoftSign, Arctan, ERF
    )

    x = jnp.linspace(-2, 2, 100)

    # Standard sigmoid
    sigmoid_fn = Sigmoid(alpha=4.0)
    y1 = sigmoid_fn(x)

    # Softsign (similar to sigmoid but with different tails)
    softsign_fn = SoftSign(alpha=2.0)
    y2 = softsign_fn(x)

    # Arctan (smooth, bounded)
    arctan_fn = Arctan(alpha=3.0)
    y3 = arctan_fn(x)

    # Error function (Gaussian CDF)
    erf_fn = ERF(alpha=2.0)
    y4 = erf_fn(x)

**Piecewise Surrogates:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.surrogate import (
        PiecewiseQuadratic, PiecewiseExp,
        PiecewiseLeakyRelu
    )

    x = jnp.linspace(-2, 2, 100)

    # Piecewise quadratic (triangle-like gradient)
    pq_fn = PiecewiseQuadratic(alpha=1.0)
    y1 = pq_fn(x)

    # Piecewise exponential
    pe_fn = PiecewiseExp(alpha=2.0)
    y2 = pe_fn(x)

    # Piecewise leaky ReLU
    plr_fn = PiecewiseLeakyRelu(c=0.01, alpha=1.0)
    y3 = plr_fn(x)

**ReLU-Based Surrogates:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.surrogate import (
        LeakyRelu, LogTailedRelu, ReluGrad
    )

    x = jnp.linspace(-2, 2, 100)

    # Leaky ReLU gradient
    leaky_fn = LeakyRelu(alpha=0.1, beta=1.0)
    y1 = leaky_fn(x)

    # Log-tailed ReLU (polynomial-like tails)
    log_fn = LogTailedRelu(alpha=1.0)
    y2 = log_fn(x)

    # Simple ReLU gradient
    relu_fn = ReluGrad(alpha=0.5, width=1.0)
    y3 = relu_fn(x)

**Gaussian-Based Surrogates:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.surrogate import (
        GaussianGrad, MultiGaussianGrad, InvSquareGrad
    )

    x = jnp.linspace(-2, 2, 100)

    # Single Gaussian
    gaussian_fn = GaussianGrad(sigma=0.5, alpha=1.0)
    y1 = gaussian_fn(x)

    # Multiple Gaussians (sum of Gaussians at different positions)
    multi_gaussian_fn = MultiGaussianGrad(
        h=0.15,  # height
        s=6.0,   # sigma
        gamma=0.5  # spacing
    )
    y2 = multi_gaussian_fn(x)

    # Inverse square gradient
    inv_square_fn = InvSquareGrad(alpha=1.0)
    y3 = inv_square_fn(x)

**Advanced Surrogates:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.surrogate import (
        SquarewaveFourierSeries, S2NN,
        QPseudoSpike, SlayerGrad, NonzeroSignLog
    )

    x = jnp.linspace(-2, 2, 100)

    # Fourier series approximation
    fourier_fn = SquarewaveFourierSeries(n=4, t_period=8.0)
    y1 = fourier_fn(x)

    # S2NN (Spiking Synaptic Neural Networks)
    s2nn_fn = S2NN(alpha=4.0, beta=1.0)
    y2 = s2nn_fn(x)

    # Q-pseudo spike
    q_fn = QPseudoSpike(alpha=2.0)
    y3 = q_fn(x)

    # SLAYER gradient (SuperSpike-like)
    slayer_fn = SlayerGrad(alpha=1.0)
    y4 = slayer_fn(x)

    # Nonzero sign with logarithmic damping
    log_fn = NonzeroSignLog(alpha=1.0)
    y5 = log_fn(x)

**Custom Surrogate Gradients:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.surrogate import Surrogate

    # Create a custom surrogate gradient
    class MySurrogate(Surrogate):
        def __init__(self, alpha=1.0):
            super().__init__()
            self.alpha = alpha

        def surrogate_fun(self, x):
            # Define the smooth approximation (optional, for analysis)
            return jnp.tanh(self.alpha * x) * 0.5 + 0.5

        def surrogate_grad(self, x):
            # Define the gradient used in backpropagation
            return self.alpha * (1 - jnp.tanh(self.alpha * x) ** 2) * 0.5

    # Use the custom surrogate
    my_fn = MySurrogate(alpha=2.0)
    x = jnp.array([0.0, 1.0, -1.0])
    spikes = my_fn(x)

**Comparing Surrogates:**

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from braintools.surrogate import (
        Sigmoid, GaussianGrad, PiecewiseQuadratic,
        Arctan, ReluGrad
    )

    # Create surrogate functions
    surrogates = {
        'Sigmoid': Sigmoid(alpha=4.0),
        'Gaussian': GaussianGrad(sigma=0.5),
        'Piecewise Quadratic': PiecewiseQuadratic(alpha=1.0),
        'Arctan': Arctan(alpha=3.0),
        'ReLU Grad': ReluGrad(alpha=0.5, width=1.0),
    }

    # Compute gradients
    x = jnp.linspace(-2, 2, 200)

    plt.figure(figsize=(10, 6))
    for name, surrogate in surrogates.items():
        # Use JAX to compute the gradient
        grad_fn = jax.vmap(jax.grad(lambda xi: surrogate(xi)))
        grads = grad_fn(x)
        plt.plot(x, grads, label=name, linewidth=2)

    plt.xlabel('Input')
    plt.ylabel('Surrogate Gradient')
    plt.title('Comparison of Surrogate Gradient Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

**Integration with Spiking Neuron Models:**

.. code-block:: python

    import brainstate as bst
    import jax.numpy as jnp
    from braintools.surrogate import Sigmoid

    # Leaky Integrate-and-Fire neuron
    class LIFNeuron(bst.nn.Module):
        def __init__(self, size, tau=20.0, v_threshold=1.0):
            super().__init__()
            self.size = size
            self.tau = tau
            self.v_threshold = v_threshold
            self.v = bst.State(jnp.zeros(size))
            self.spike_fn = Sigmoid(alpha=4.0)

        def __call__(self, x):
            # LIF dynamics
            dv = (-self.v.value + x) / self.tau
            self.v.value = self.v.value + dv

            # Spike generation with surrogate gradient
            spikes = self.spike_fn(self.v.value - self.v_threshold)

            # Reset
            self.v.value = jnp.where(spikes > 0, 0.0, self.v.value)

            return spikes

    # Adaptive Exponential Integrate-and-Fire
    class AdExNeuron(bst.nn.Module):
        def __init__(self, size):
            super().__init__()
            self.size = size
            self.v = bst.State(jnp.zeros(size) - 70.0)
            self.w = bst.State(jnp.zeros(size))
            self.spike_fn = GaussianGrad(sigma=0.5)

        def __call__(self, I_ext):
            # AdEx dynamics (simplified)
            dv = -self.v.value + jnp.exp(self.v.value + 50) - self.w.value + I_ext
            dw = (self.v.value - self.w.value) * 0.01

            self.v.value = self.v.value + dv * 0.1
            self.w.value = self.w.value + dw * 0.1

            # Spike detection
            spikes = self.spike_fn(self.v.value - (-50.0))

            # Reset
            self.v.value = jnp.where(spikes > 0, -70.0, self.v.value)
            self.w.value = jnp.where(spikes > 0, self.w.value + 10.0, self.w.value)

            return spikes

**Functional vs Class API:**

.. code-block:: python

    import jax.numpy as jnp
    from braintools.surrogate import Sigmoid, sigmoid

    x = jnp.array([0.0, 1.0, -1.0])

    # Class API (reusable object)
    spike_fn = Sigmoid(alpha=4.0)
    y1 = spike_fn(x)
    y2 = spike_fn(x * 2)  # Reuse

    # Functional API (one-time use)
    y3 = sigmoid(x, alpha=4.0)
    y4 = sigmoid(x * 2, alpha=4.0)

    # Both produce identical results
    assert jnp.allclose(y1, y3)

**Parameter Sensitivity:**

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from braintools.surrogate import Sigmoid

    x = jnp.linspace(-2, 2, 200)

    # Different alpha values
    plt.figure(figsize=(12, 4))

    # Forward pass
    plt.subplot(1, 2, 1)
    for alpha in [1.0, 2.0, 4.0, 8.0]:
        spike_fn = Sigmoid(alpha=alpha)
        y = jax.vmap(spike_fn)(x)
        plt.plot(x, y, label=f'alpha={alpha}')
    plt.title('Forward Pass (Step Function)')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Backward pass (gradient)
    plt.subplot(1, 2, 2)
    for alpha in [1.0, 2.0, 4.0, 8.0]:
        spike_fn = Sigmoid(alpha=alpha)
        grad_fn = jax.vmap(jax.grad(lambda xi: spike_fn(xi)))
        grads = grad_fn(x)
        plt.plot(x, grads, label=f'alpha={alpha}')
    plt.title('Backward Pass (Surrogate Gradient)')
    plt.xlabel('Input')
    plt.ylabel('Gradient')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

**Multi-Layer SNN Training:**

.. code-block:: python

    import brainstate as bst
    import jax.numpy as jnp
    from braintools.surrogate import Sigmoid

    # Multi-layer SNN
    class SpikingMLP(bst.nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = []
            for i in range(len(layers) - 1):
                self.layers.append(bst.nn.Linear(layers[i], layers[i+1]))
            self.v = [bst.State(jnp.zeros(size)) for size in layers[1:]]
            self.spike_fn = Sigmoid(alpha=4.0)

        def __call__(self, x, steps=10):
            # Reset states
            for v in self.v:
                v.value = jnp.zeros_like(v.value)

            # Run for multiple time steps
            outputs = []
            for t in range(steps):
                h = x
                for i, (linear, v) in enumerate(zip(self.layers, self.v)):
                    # Integrate
                    v.value = 0.9 * v.value + linear(h)
                    # Spike
                    h = self.spike_fn(v.value - 1.0)
                    # Reset
                    v.value = jnp.where(h > 0, 0.0, v.value)
                outputs.append(h)

            return jnp.mean(jnp.stack(outputs), axis=0)

    # Create and train
    model = SpikingMLP([784, 256, 128, 10])
    optimizer = bst.optim.Adam(lr=1e-3)
    optimizer.register_trainable_weights(model.states(bst.ParamState))

    @bst.transform.grad(model.states(bst.ParamState), return_value=True)
    def loss_fn(x, target):
        output = model(x)
        return jnp.mean((output - target) ** 2)

    # Training step
    x = jnp.ones(784)
    target = jnp.zeros(10)
    grads, loss = loss_fn(x, target)
    optimizer.update(grads)

**Choice of Surrogate Gradient:**

Different surrogate gradients have different properties:

- **Sigmoid**: Smooth, bounded, similar to biological activation
- **Gaussian**: Localized gradient, good for precise spike timing
- **Piecewise**: Triangle-like, simple and effective
- **ReLU-based**: Fast computation, unbounded gradient
- **Arctan**: Smooth with controlled tails
- **Multi-Gaussian**: Multiple gradient peaks for complex dynamics

Choose based on:
- Task requirements (classification, regression, timing)
- Network depth (deeper networks may need smoother gradients)
- Computational budget (simpler surrogates are faster)
- Biological plausibility (if relevant)

"""

# Base class
from ._base import (
    Surrogate,
)

# Surrogate gradient implementations
from ._impl import (
    # Sigmoid-based
    Sigmoid,
    sigmoid,
    SoftSign,
    soft_sign,
    Arctan,
    arctan,
    ERF,
    erf,

    # Piecewise
    PiecewiseQuadratic,
    piecewise_quadratic,
    PiecewiseExp,
    piecewise_exp,
    PiecewiseLeakyRelu,
    piecewise_leaky_relu,

    # ReLU-based
    LeakyRelu,
    leaky_relu,
    LogTailedRelu,
    log_tailed_relu,
    ReluGrad,
    relu_grad,

    # Gaussian-based
    GaussianGrad,
    gaussian_grad,
    InvSquareGrad,
    inv_square_grad,
    MultiGaussianGrad,
    multi_gaussian_grad,

    # Advanced
    SquarewaveFourierSeries,
    squarewave_fourier_series,
    S2NN,
    s2nn,
    QPseudoSpike,
    q_pseudo_spike,
    SlayerGrad,
    slayer_grad,
    NonzeroSignLog,
    nonzero_sign_log,
)

__all__ = [
    # Base class
    'Surrogate',

    # Sigmoid-based surrogates
    'Sigmoid',
    'sigmoid',
    'SoftSign',
    'soft_sign',
    'Arctan',
    'arctan',
    'ERF',
    'erf',

    # Piecewise surrogates
    'PiecewiseQuadratic',
    'piecewise_quadratic',
    'PiecewiseExp',
    'piecewise_exp',
    'PiecewiseLeakyRelu',
    'piecewise_leaky_relu',

    # ReLU-based surrogates
    'LeakyRelu',
    'leaky_relu',
    'LogTailedRelu',
    'log_tailed_relu',
    'ReluGrad',
    'relu_grad',

    # Gaussian-based surrogates
    'GaussianGrad',
    'gaussian_grad',
    'InvSquareGrad',
    'inv_square_grad',
    'MultiGaussianGrad',
    'multi_gaussian_grad',

    # Advanced surrogates
    'SquarewaveFourierSeries',
    'squarewave_fourier_series',
    'S2NN',
    's2nn',
    'QPseudoSpike',
    'q_pseudo_spike',
    'SlayerGrad',
    'slayer_grad',
    'NonzeroSignLog',
    'nonzero_sign_log',
]
