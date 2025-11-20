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

# -*- coding: utf-8 -*-

from typing import Dict, Optional, Union, Callable, Any, List, Sequence

import brainstate.transform
import jax
import jax.numpy as jnp

from ._base import OptimState

__all__ = [
    # Learning Rate Schedulers
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
]


# ============================================================================
# Learning Rate Scheduler Base Class
# ============================================================================

class LRScheduler:
    """Base class for learning rate schedulers.

    Can be used either standalone (passed to optimizer at initialization)
    or attached to an optimizer later.
    """
    __module__ = 'braintools.optim'

    def __init__(self, base_lr: Union[float, List[float]] = 1e-3, last_epoch: int = 0):
        """
        Initialize the scheduler.

        Args:
          base_lr: Base learning rate(s). Can be a float or list of floats for multiple param groups.
          last_epoch: The index of the last epoch.
        """
        self.optimizer = None  # Will be set when attached to optimizer
        self.last_epoch = OptimState(last_epoch)

        # Support both single lr and multiple lrs for param groups
        if isinstance(base_lr, (list, tuple)):
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr]

        # Current learning rates
        self._current_lrs = OptimState(list(self.base_lrs))

    @property
    def current_lrs(self):
        return self._current_lrs

    def attach_optimizer(self, optimizer: 'OptaxOptimizer'):
        """Attach this scheduler to an optimizer."""
        from ._optax_optimizer import OptaxOptimizer
        if not isinstance(optimizer, OptaxOptimizer):
            raise TypeError(f"optimizer must be an Optaxgot {type(optimizer)}")

        self.optimizer = optimizer

        # If optimizer has param groups, ensure we have enough base_lrs
        if len(optimizer.param_groups) > len(self.base_lrs):
            # Extend base_lrs with the last value
            last_lr = self.base_lrs[-1] if self.base_lrs else optimizer.base_lr
            self.base_lrs.extend(
                [last_lr] * (len(optimizer.param_groups) - len(self.base_lrs))
            )
            self.current_lrs.value.extend(
                [last_lr] * (len(optimizer.param_groups) - len(self.current_lrs.value))
            )

    def get_lr(self):
        """Calculate learning rate."""
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None):
        """Update learning rate."""
        if epoch is None:
            self.last_epoch.value += 1
        else:
            self.last_epoch.value = epoch
        self.apply(lambda x: x)

    def apply(self, apply_fn: Callable[[float], float]):
        """Apply a function to modify the current learning rate."""

        values = self.get_lr()
        if not isinstance(values, (list, tuple)):
            values = [values]
        applied_values = [apply_fn(v) for v in values]
        self.current_lrs.value = applied_values

        # If attached to update its learning rates
        if self.optimizer is not None:
            for param_group, lr in zip(self.optimizer.param_groups, applied_values):
                param_group['lr'].value = lr

            # Update the main optimizer lr
            self.optimizer.current_lr = applied_values[0]

    def step_epoch(self):
        """Step the scheduler by one epoch."""
        self.step()

    def __call__(self, count):
        """Make scheduler callable for use with optax.scale_by_schedule.

        This allows the scheduler to be passed directly to the optimizer.
        """
        return -self.current_lrs.value[0] if len(self.current_lrs.value) else -1e-3

    def state_dict(self):
        """Return scheduler state as dictionary."""
        return {
            'last_epoch': self.last_epoch.value,
            'base_lrs': self.base_lrs,
            'current_lrs': self.current_lrs.value,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from dictionary."""
        self.last_epoch.value = state_dict['last_epoch']
        self.base_lrs = state_dict['base_lrs']
        self.current_lrs.value = state_dict.get('current_lrs', list(self.base_lrs))


# ============================================================================
# Learning Rate Scheduler Classes
# ============================================================================

class StepLR(LRScheduler):
    r"""Step learning rate scheduler - Decays learning rate by gamma every step_size epochs.

    StepLR multiplies the learning rate by gamma at regular intervals (every step_size epochs),
    creating a staircase decay pattern. This is one of the most commonly used learning rate
    schedules for training deep neural networks.

    Parameters
    ----------
    base_lr : float or list of float, optional
        Initial learning rate(s). Can be a single float or a list of floats for multiple
        parameter groups. Default: 1e-3.
    step_size : int, optional
        Period of learning rate decay in epochs. The learning rate will be multiplied by
        gamma every step_size epochs. Default: 30.
    gamma : float, optional
        Multiplicative factor of learning rate decay. Must be in range (0, 1].
        Default: 0.1.
    last_epoch : int, optional
        The index of the last epoch. Used for resuming training. Default: -1 (starts
        from beginning).

    Notes
    -----
    The learning rate at epoch :math:`t` is computed as:

    .. math::
        \eta_t = \eta_0 \cdot \gamma^{\lfloor t / \text{step_size} \rfloor}

    where :math:`\eta_0` is the initial learning rate (base_lr), and :math:`\lfloor \cdot \rfloor`
    denotes the floor function.

    **Key characteristics:**

    - Creates discrete "steps" in the learning rate schedule
    - Widely used for training image classification models
    - Simple to tune with only two hyperparameters
    - Works well when combined with momentum-based optimizers

    **Common step_size values:**

    - ImageNet training: step_size=30, total_epochs=90 (decay at epochs 30, 60)
    - CIFAR training: step_size=50, total_epochs=150 (decay at epochs 50, 100)

    Examples
    --------
    **Basic usage with SGD:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> # Create model and scheduler
        >>> model = brainstate.nn.Linear(10, 5)
        >>> scheduler = braintools.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Training loop
        >>> for epoch in range(90):
        ...     # ... training code ...
        ...     scheduler.step()
        ...     if epoch in [0, 29, 30, 59, 60, 89]:
        ...         print(f"Epoch {epoch}: lr = {optimizer.current_lr:.6f}")
        Epoch 0: lr = 0.100000
        Epoch 29: lr = 0.100000
        Epoch 30: lr = 0.010000  # First decay
        Epoch 59: lr = 0.010000
        Epoch 60: lr = 0.001000  # Second decay
        Epoch 89: lr = 0.001000

    **Using with Adam optimizer:**

    .. code-block:: python

        >>> scheduler = braintools.optim.StepLR(base_lr=0.001, step_size=10, gamma=0.5)
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(25):
        ...     # Training step
        ...     scheduler.step()
        # lr decays: 0.001 -> 0.0005 (epoch 10) -> 0.00025 (epoch 20)

    **Custom decay schedule:**

    .. code-block:: python

        >>> # Aggressive decay every 5 epochs
        >>> scheduler = braintools.optim.StepLR(base_lr=0.1, step_size=5, gamma=0.5)
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # After 15 epochs: lr = 0.1 * 0.5^3 = 0.0125

    **Saving and loading scheduler state:**

    .. code-block:: python

        >>> scheduler = braintools.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Train for some epochs
        >>> for epoch in range(50):
        ...     scheduler.step()
        >>>
        >>> # Save checkpoint
        >>> checkpoint = {
        ...     'epoch': 50,
        ...     'model': model.state_dict(),
        ...     'optimizer': optimizer.state_dict(),
        ...     'scheduler': scheduler.state_dict(),
        ... }
        >>>
        >>> # Later, resume training
        >>> new_scheduler = braintools.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> new_scheduler.load_state_dict(checkpoint['scheduler'])
        >>> # Continue from epoch 50

    **Multiple parameter groups:**

    .. code-block:: python

        >>> # Different learning rates for different layers
        >>> scheduler = braintools.optim.StepLR(
        ...     base_lr=[0.1, 0.01],  # Different base lr for each group
        ...     step_size=30,
        ...     gamma=0.1
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> # Both groups decay by gamma every step_size epochs

    **Complete training example:**

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> scheduler = braintools.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> def train_epoch(model, optimizer, data):
        ...     def loss_fn(params):
        ...         # Compute loss
        ...         return loss
        ...     grads = jax.grad(loss_fn)(model.states(brainstate.ParamState))
        ...     optimizer.update(grads)
        >>>
        >>> for epoch in range(90):
        ...     train_epoch(model, optimizer, train_data)
        ...     scheduler.step()
        ...     print(f"Epoch {epoch}: lr = {optimizer.current_lr}")

    See Also
    --------
    MultiStepLR : Decay learning rate at specific milestone epochs
    ExponentialLR : Exponential decay of learning rate
    CosineAnnealingLR : Cosine annealing schedule

    References
    ----------
    .. [1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
           "ImageNet classification with deep convolutional neural networks."
           Advances in neural information processing systems, 25.
    .. [2] He, K., Zhang, X., Ren, S., & Sun, J. (2016).
           "Deep residual learning for image recognition."
           Proceedings of the IEEE conference on computer vision and pattern
           recognition, 770-778.
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        step_size: int = 30,
        gamma: float = 0.1,
        last_epoch: int = 0,
    ):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        factor = self.gamma ** (self.last_epoch.value // self.step_size)
        return [base_lr * factor for base_lr in self.base_lrs]


class MultiStepLR(LRScheduler):
    r"""Multi-step learning rate scheduler - Decays learning rate at specific milestone epochs.

    MultiStepLR reduces the learning rate by a factor of gamma at each epoch specified in
    the milestones list. This provides more flexible control than StepLR, allowing you to
    schedule learning rate drops at arbitrary points during training.

    Parameters
    ----------
    base_lr : float or list of float, optional
        Initial learning rate(s). Can be a single float or a list of floats for multiple
        parameter groups. Default: 1e-3.
    milestones : sequence of int, optional
        List of epoch indices at which to decay the learning rate. Must be increasing.
        Default: (30, 60, 90).
    gamma : float, optional
        Multiplicative factor of learning rate decay. Must be in range (0, 1].
        Default: 0.1.
    last_epoch : int, optional
        The index of the last epoch. Used for resuming training. Default: -1 (starts
        from beginning).

    Notes
    -----
    The learning rate at epoch :math:`t` is computed as:

    .. math::
        \eta_t = \eta_0 \cdot \gamma^{|\{m \in \text{milestones} : m \leq t\}|}

    where :math:`\eta_0` is the initial learning rate (base_lr), and
    :math:`|\{m \in \text{milestones} : m \leq t\}|` counts how many milestones have been reached
    by epoch :math:`t`.

    **Key characteristics:**

    - Provides precise control over when learning rate changes occur
    - Ideal when you know specific epochs where model learning plateaus
    - Commonly used in research papers with fixed training schedules
    - Each milestone multiplies the current lr by gamma

    **Common milestone patterns:**

    - ImageNet (90 epochs): milestones=[30, 60], gamma=0.1
    - CIFAR (200 epochs): milestones=[60, 120, 160], gamma=0.2
    - Fine-tuning: milestones=[10, 20], gamma=0.5

    Examples
    --------
    **Basic usage with predefined milestones:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> scheduler = braintools.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[30, 80],
        ...     gamma=0.1
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # lr schedule:
        >>> # epochs 0-29:  lr = 0.1
        >>> # epochs 30-79: lr = 0.01  (after 1st milestone)
        >>> # epochs 80+:   lr = 0.001 (after 2nd milestone)

    **Using with Adam for fine-tuning:**

    .. code-block:: python

        >>> scheduler = braintools.optim.MultiStepLR(
        ...     base_lr=0.001,
        ...     milestones=[10, 20, 30],
        ...     gamma=0.5
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(40):
        ...     # Training code
        ...     scheduler.step()
        # lr: 0.001 -> 0.0005 (epoch 10) -> 0.00025 (epoch 20) -> 0.000125 (epoch 30)

    **ImageNet-style training schedule:**

    .. code-block:: python

        >>> # Standard ImageNet schedule: 90 epochs with drops at 30 and 60
        >>> scheduler = braintools.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[30, 60],
        ...     gamma=0.1
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(90):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        ...     print(f"Epoch {epoch}: lr = {optimizer.current_lr}")

    **CIFAR training schedule:**

    .. code-block:: python

        >>> # CIFAR-10/100 schedule: 200 epochs
        >>> scheduler = braintools.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[60, 120, 160],
        ...     gamma=0.2
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=5e-4)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(200):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Custom aggressive decay schedule:**

    .. code-block:: python

        >>> # Frequent drops for quick convergence
        >>> scheduler = braintools.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[5, 10, 15, 20, 25],
        ...     gamma=0.5
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # lr rapidly decreases at each milestone

    **Resuming training with state dict:**

    .. code-block:: python

        >>> # Save training state
        >>> scheduler = braintools.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[30, 60, 90],
        ...     gamma=0.1
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(50):
        ...     scheduler.step()
        >>>
        >>> checkpoint = {'scheduler': scheduler.state_dict(), 'epoch': 50}
        >>>
        >>> # Resume later
        >>> new_scheduler = braintools.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[30, 60, 90],
        ...     gamma=0.1
        ... )
        >>> new_scheduler.load_state_dict(checkpoint['scheduler'])
        >>> # Continues from epoch 50 with correct lr

    See Also
    --------
    StepLR : Decay learning rate at regular intervals
    ExponentialLR : Exponential decay of learning rate
    SequentialLR : Switch between different schedulers at milestones

    References
    ----------
    .. [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016).
           "Deep residual learning for image recognition."
           Proceedings of the IEEE conference on computer vision and pattern
           recognition, 770-778.
    .. [2] Zagoruyko, S., & Komodakis, N. (2016).
           "Wide residual networks."
           arXiv preprint arXiv:1605.07146.
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        milestones: Sequence[int] = (30, 60, 90),
        gamma: float = 0.1,
        last_epoch: int = 0,
    ):
        self.milestones = jnp.array(sorted(milestones))
        self.gamma = gamma
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        # Count how many milestones have been reached (JIT-compatible)
        count = jnp.sum(self.last_epoch.value >= self.milestones)
        factor = jnp.power(self.gamma, count)
        return [base_lr * factor for base_lr in self.base_lrs]


class ExponentialLR(LRScheduler):
    r"""Exponential learning rate scheduler - Decays learning rate exponentially.

    ExponentialLR multiplies the learning rate by gamma at every epoch, creating a smooth
    exponential decay. This scheduler is useful when you want a continuous and predictable
    decrease in the learning rate throughout training.

    Parameters
    ----------
    base_lr : float or list of float, optional
        Initial learning rate(s). Can be a single float or a list of floats for multiple
        parameter groups. Default: 1e-3.
    gamma : float
        Multiplicative factor of learning rate decay per epoch. Must be in range (0, 1).
        Typical values: 0.95-0.99 for slow decay, 0.9-0.95 for moderate decay.
    last_epoch : int, optional
        The index of the last epoch. Used for resuming training. Default: -1 (starts
        from beginning).

    Notes
    -----
    The learning rate at epoch :math:`t` is computed as:

    .. math::
        \eta_t = \eta_0 \cdot \gamma^t

    where :math:`\eta_0` is the initial learning rate (base_lr) and :math:`t` is the
    current epoch number.

    **Key characteristics:**

    - Smooth exponential decay every epoch
    - Learning rate decreases continuously
    - Simple one-parameter control (gamma)
    - Decay rate is constant in logarithmic scale

    **Gamma selection guidelines:**

    - gamma=0.95: Moderate decay, lr halves every ~14 epochs
    - gamma=0.96: Gentle decay, lr halves every ~17 epochs
    - gamma=0.98: Slow decay, lr halves every ~35 epochs
    - gamma=0.99: Very slow decay, lr halves every ~69 epochs

    **When to use:**

    - When you want smooth, continuous learning rate reduction
    - For fine-tuning with gradual decay
    - When step-based schedules are too abrupt
    - For long training runs with gradual convergence

    Examples
    --------
    **Basic exponential decay:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> # Decay by 0.95 each epoch
        >>> scheduler = braintools.optim.ExponentialLR(base_lr=0.1, gamma=0.95)
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(20):
        ...     # Training code
        ...     scheduler.step()
        ...     if epoch % 5 == 0:
        ...         print(f"Epoch {epoch}: lr = {optimizer.current_lr:.6f}")
        Epoch 0: lr = 0.100000
        Epoch 5: lr = 0.077378  # lr * 0.95^5
        Epoch 10: lr = 0.059874  # lr * 0.95^10
        Epoch 15: lr = 0.046329  # lr * 0.95^15

    **Slow decay for fine-tuning:**

    .. code-block:: python

        >>> # Very gentle decay with gamma=0.99
        >>> scheduler = braintools.optim.ExponentialLR(base_lr=0.001, gamma=0.99)
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     finetune_epoch(model, optimizer, finetune_loader)
        ...     scheduler.step()
        # After 100 epochs: lr ≈ 0.001 * 0.99^100 ≈ 0.000366

    **Moderate decay for standard training:**

    .. code-block:: python

        >>> # Moderate decay with gamma=0.96
        >>> scheduler = braintools.optim.ExponentialLR(base_lr=0.1, gamma=0.96)
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(50):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        # lr smoothly decreases from 0.1 to ~0.013

    **Combining with warmup:**

    .. code-block:: python

        >>> # Warmup followed by exponential decay
        >>> warmup = braintools.optim.LinearLR(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> decay = braintools.optim.ExponentialLR(base_lr=0.01, gamma=0.95)
        >>> scheduler = braintools.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Using with different optimizers:**

    .. code-block:: python

        >>> # Works with any optimizer
        >>> scheduler = braintools.optim.ExponentialLR(base_lr=0.001, gamma=0.98)
        >>>
        >>> # With Adam
        >>> adam_opt = braintools.optim.Adam(lr=scheduler)
        >>> adam_opt.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Or with RMSprop
        >>> model2 = brainstate.nn.Linear(10, 5)
        >>> scheduler2 = braintools.optim.ExponentialLR(base_lr=0.001, gamma=0.98)
        >>> rmsprop_opt = braintools.optim.RMSprop(lr=scheduler2)
        >>> rmsprop_opt.register_trainable_weights(model2.states(brainstate.ParamState))

    **Saving and loading state:**

    .. code-block:: python

        >>> scheduler = braintools.optim.ExponentialLR(base_lr=0.1, gamma=0.95)
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Train for some epochs
        >>> for epoch in range(50):
        ...     scheduler.step()
        >>>
        >>> # Save checkpoint
        >>> checkpoint = {
        ...     'epoch': 50,
        ...     'model': model.state_dict(),
        ...     'scheduler': scheduler.state_dict(),
        ... }
        >>>
        >>> # Resume training
        >>> new_scheduler = braintools.optim.ExponentialLR(base_lr=0.1, gamma=0.95)
        >>> new_scheduler.load_state_dict(checkpoint['scheduler'])
        >>> # lr will be correctly set to 0.1 * 0.95^50

    **Aggressive decay:**

    .. code-block:: python

        >>> # Fast decay with gamma=0.9
        >>> scheduler = braintools.optim.ExponentialLR(base_lr=0.1, gamma=0.9)
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(30):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        # After 30 epochs: lr ≈ 0.1 * 0.9^30 ≈ 0.00424

    See Also
    --------
    StepLR : Step-wise learning rate decay
    CosineAnnealingLR : Cosine annealing schedule
    MultiStepLR : Multi-step learning rate decay

    References
    ----------
    .. [1] Bottou, L. (2012).
           "Stochastic gradient descent tricks."
           Neural networks: Tricks of the trade, 421-436.
    .. [2] Bengio, Y. (2012).
           "Practical recommendations for gradient-based training of deep architectures."
           Neural networks: Tricks of the trade, 437-478.
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        gamma: float = 0.95,
        last_epoch: int = 0,
    ):
        self.gamma = gamma
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        expon = self.gamma ** self.last_epoch.value
        return [base_lr * expon for base_lr in self.base_lrs]


class ExponentialDecayLR(LRScheduler):
    r"""Exponential decay learning rate scheduler with step-based control.

    ExponentialDecayLR implements optax's exponential_decay schedule, providing more fine-grained
    control compared to ExponentialLR. It supports transition steps, staircase mode, delayed start,
    and bounded decay, making it suitable for step-level (rather than epoch-level) learning rate control.

    Parameters
    ----------
    base_lr : float or list of float, optional
        Initial learning rate(s). Can be a single float or a list of floats for multiple
        parameter groups. Default: 1e-3.
    decay_steps : int
        Number of steps over which to apply the decay. Must be positive.
    decay_rate : float
        The decay rate. Must not be zero. Values < 1 create decay, values > 1 create growth.
        Typical values: 0.96-0.99 for slow decay, 0.9-0.95 for moderate decay.
    transition_begin : int, optional
        Number of steps to wait before starting decay. The learning rate is held at base_lr
        for this many steps. Default: 0.
    staircase : bool, optional
        If True, decay happens at discrete intervals (step-wise). If False, decay is continuous.
        Default: False.
    end_value : float, optional
        Optional bound for the decayed value. When decay_rate < 1, acts as a lower bound.
        When decay_rate > 1, acts as an upper bound. Default: None (no bound).
    last_epoch : int, optional
        The index of the last epoch. Default: 0.

    Notes
    -----
    The learning rate is computed based on step count. When ``step >= transition_begin``:

    **Continuous mode (staircase=False):**

    .. math::
        \text{rate\_factor} = \frac{\text{step} - \text{transition\_begin}}{\text{transition\_steps}}

        \eta = \text{init\_value} \times \text{decay\_rate}^{\text{rate\_factor}}

    **Staircase mode (staircase=True):**

    .. math::
        \text{rate\_factor} = \left\lfloor\frac{\text{step} - \text{transition\_begin}}{\text{transition\_steps}}\right\rfloor

        \eta = \text{init\_value} \times \text{decay\_rate}^{\text{rate\_factor}}

    Before ``transition_begin`` steps, the learning rate is held constant at ``base_lr``.

    **Key differences from ExponentialLR:**

    - Step-based instead of epoch-based control
    - Configurable transition period (decay_steps)
    - Optional delayed start (transition_begin)
    - Staircase mode for discrete decay steps
    - Bounded decay with end_value

    **When to use:**

    - When you need step-level (not epoch-level) learning rate control
    - For fine-grained decay schedules
    - When you want to delay decay start
    - For bounded decay with minimum/maximum values
    - In scenarios requiring staircase (discrete) decay

    Examples
    --------
    **Basic continuous exponential decay:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> # Decay by 0.96 every 1000 steps
        >>> scheduler = braintools.optim.ExponentialDecayLR(
        ...     base_lr=0.1,
        ...     decay_steps=1000,
        ...     decay_rate=0.96
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for step in range(5000):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        ...     if step % 1000 == 0:
        ...         print(f"Step {step}: lr = {optimizer.current_lr:.6f}")
        Step 0: lr = 0.100000
        Step 1000: lr = 0.096000  # 0.1 * 0.96^1
        Step 2000: lr = 0.092160  # 0.1 * 0.96^2
        Step 3000: lr = 0.088474  # 0.1 * 0.96^3
        Step 4000: lr = 0.084935  # 0.1 * 0.96^4

    **Staircase mode (discrete decay steps):**

    .. code-block:: python

        >>> # Decay every 1000 steps with staircase mode
        >>> scheduler = braintools.optim.ExponentialDecayLR(
        ...     base_lr=0.1,
        ...     decay_steps=1000,
        ...     decay_rate=0.5,
        ...     staircase=True
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for step in [0, 500, 1000, 1500, 2000, 2500, 3000]:
        ...     for _ in range(step - scheduler.last_epoch.value):
        ...         scheduler.step()
        ...     print(f"Step {step}: lr = {optimizer.current_lr:.6f}")
        Step 0: lr = 0.100000
        Step 500: lr = 0.100000   # Still in first interval
        Step 1000: lr = 0.050000  # Drops at step 1000
        Step 1500: lr = 0.050000  # Constant until step 2000
        Step 2000: lr = 0.025000  # Drops at step 2000
        Step 2500: lr = 0.025000  # Constant until step 3000
        Step 3000: lr = 0.012500  # Drops at step 3000

    **Delayed decay start:**

    .. code-block:: python

        >>> # Hold LR constant for 2000 steps, then start decay
        >>> scheduler = braintools.optim.ExponentialDecayLR(
        ...     base_lr=0.01,
        ...     decay_steps=1000,
        ...     decay_rate=0.95,
        ...     transition_begin=2000
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for step in [0, 1000, 2000, 3000, 4000]:
        ...     for _ in range(step - scheduler.last_epoch.value):
        ...         scheduler.step()
        ...     print(f"Step {step}: lr = {optimizer.current_lr:.6f}")
        Step 0: lr = 0.010000     # Held constant
        Step 1000: lr = 0.010000  # Held constant
        Step 2000: lr = 0.010000  # Decay starts here
        Step 3000: lr = 0.009500  # 0.01 * 0.95^1
        Step 4000: lr = 0.009025  # 0.01 * 0.95^2

    **Bounded decay with end_value:**

    .. code-block:: python

        >>> # Decay but don't go below 0.001
        >>> scheduler = braintools.optim.ExponentialDecayLR(
        ...     base_lr=0.1,
        ...     decay_steps=500,
        ...     decay_rate=0.9,
        ...     end_value=0.001
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for step in range(0, 5000, 500):
        ...     for _ in range(step - scheduler.last_epoch.value):
        ...         scheduler.step()
        ...     print(f"Step {step}: lr = {optimizer.current_lr:.6f}")
        # LR decays but stops at end_value

    **Fine-tuning with slow decay:**

    .. code-block:: python

        >>> # Very gentle step-based decay for fine-tuning
        >>> scheduler = braintools.optim.ExponentialDecayLR(
        ...     base_lr=1e-4,
        ...     decay_steps=100,
        ...     decay_rate=0.99,
        ...     transition_begin=500
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler, weight_decay=1e-5)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Fine-tune for many steps
        >>> for step in range(10000):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Comparison with ExponentialLR:**

    .. code-block:: python

        >>> # ExponentialLR: epoch-based, simple gamma decay
        >>> exp_lr = braintools.optim.ExponentialLR(base_lr=0.1, gamma=0.95)
        >>>
        >>> # ExponentialDecayLR: step-based, configurable transition
        >>> exp_decay_lr = braintools.optim.ExponentialDecayLR(
        ...     base_lr=0.1,
        ...     decay_steps=1,
        ...     decay_rate=0.95
        ... )
        >>> # These are equivalent when decay_steps=1 and called every epoch

    See Also
    --------
    ExponentialLR : Simple exponential learning rate decay (epoch-based)
    StepLR : Step-wise learning rate decay
    CosineAnnealingLR : Cosine annealing schedule

    References
    ----------
    .. [1] Optax documentation
           https://optax.readthedocs.io/en/latest/api.html#optax.exponential_decay
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        decay_steps: int = 1000,
        decay_rate: float = 0.96,
        transition_begin: int = 0,
        staircase: bool = False,
        end_value: Optional[float] = None,
        last_epoch: int = 0,
    ):
        if decay_steps <= 0:
            raise ValueError(f"decay_steps must be positive, got {decay_steps}")
        if decay_rate == 0:
            raise ValueError("decay_rate must not be zero")
        if transition_begin < 0:
            raise ValueError(f"transition_begin must be non-negative, got {transition_begin}")

        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.transition_begin = transition_begin
        self.staircase = staircase
        self.end_value = end_value
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        step = self.last_epoch.value

        # Calculate decay using JAX operations for JIT compatibility
        steps_since_begin = jnp.maximum(step - self.transition_begin, 0)
        rate_factor = steps_since_begin / self.decay_steps

        # Apply staircase if needed
        rate_factor = jnp.floor(rate_factor) if self.staircase else rate_factor

        lrs = []
        for base_lr in self.base_lrs:
            # Apply exponential decay
            lr = base_lr * (self.decay_rate ** rate_factor)

            # Apply end_value bound if specified
            if self.end_value is not None:
                if self.decay_rate < 1:
                    # Decay: end_value is lower bound
                    lr = jnp.maximum(lr, self.end_value)
                else:
                    # Growth: end_value is upper bound
                    lr = jnp.minimum(lr, self.end_value)

            lrs.append(lr)

        return lrs


class CosineAnnealingLR(LRScheduler):
    r"""Cosine annealing learning rate scheduler - Smoothly anneals learning rate using cosine function.

    CosineAnnealingLR adjusts the learning rate following a cosine curve, starting from the
    initial learning rate and decreasing to a minimum value (eta_min) over T_max epochs.
    This provides a smooth, gradual decay that is popular for training deep neural networks.

    Parameters
    ----------
    base_lr : float or list of float, optional
        Initial learning rate(s). Can be a single float or a list of floats for multiple
        parameter groups. Default: 1e-3.
    T_max : int
        Maximum number of epochs for one annealing cycle. After T_max epochs, the learning
        rate reaches eta_min.
    eta_min : float, optional
        Minimum learning rate. The learning rate will decay from base_lr to eta_min over
        T_max epochs. Default: 0.
    last_epoch : int, optional
        The index of the last epoch. Used for resuming training. Default: -1 (starts
        from beginning).

    Notes
    -----
    The learning rate at epoch :math:`t` is computed as:

    .. math::
        \eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})
        \left(1 + \cos\left(\frac{t}{T_{\max}} \pi\right)\right)

    where :math:`\eta_0` is the initial learning rate (base_lr), :math:`\eta_{\min}` is
    the minimum learning rate, and :math:`T_{\max}` is the maximum number of epochs.

    **Key characteristics:**

    - Smooth cosine curve decay (no abrupt changes)
    - Learning rate starts high, decreases smoothly to eta_min
    - Most decay happens in the middle epochs
    - Popular for training vision models (ResNets, ViTs, etc.)
    - Often combined with warmup for best results

    **Decay pattern:**

    - Early epochs (0-25% of T_max): Slow decay
    - Middle epochs (25-75% of T_max): Fast decay
    - Late epochs (75-100% of T_max): Slow decay approaching eta_min

    **When to use:**

    - Training image classification models
    - When you want smooth learning rate transitions
    - Long training runs (100+ epochs)
    - Combined with warmup for transformer models

    Examples
    --------
    **Basic cosine annealing:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> # Anneal from 0.1 to 0 over 100 epochs
        >>> scheduler = braintools.optim.CosineAnnealingLR(
        ...     base_lr=0.1,
        ...     T_max=100,
        ...     eta_min=0
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        ...     if epoch % 25 == 0:
        ...         print(f"Epoch {epoch}: lr = {optimizer.current_lr:.6f}")
        Epoch 0: lr = 0.100000
        Epoch 25: lr = 0.085355  # Slow decay early
        Epoch 50: lr = 0.050000  # Fast decay middle
        Epoch 75: lr = 0.014645  # Slow decay late

    **With non-zero minimum learning rate:**

    .. code-block:: python

        >>> # Anneal from 0.01 to 0.0001 over 50 epochs
        >>> scheduler = braintools.optim.CosineAnnealingLR(
        ...     base_lr=0.01,
        ...     T_max=50,
        ...     eta_min=0.0001
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(50):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Combined with warmup (recommended):**

    .. code-block:: python

        >>> # Warmup for 5 epochs, then cosine decay
        >>> warmup = braintools.optim.LinearLR(
        ...     start_factor=0.01,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> cosine = braintools.optim.CosineAnnealingLR(
        ...     base_lr=0.1,
        ...     T_max=90,
        ...     eta_min=0
        ... )
        >>> scheduler = braintools.optim.ChainedScheduler([warmup, cosine])
        >>>
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(95):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **CIFAR-10/100 training schedule:**

    .. code-block:: python

        >>> # Standard CIFAR schedule: 200 epochs with cosine decay
        >>> scheduler = braintools.optim.CosineAnnealingLR(
        ...     base_lr=0.1,
        ...     T_max=200,
        ...     eta_min=0
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=5e-4)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(200):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **ImageNet training with cosine decay:**

    .. code-block:: python

        >>> # ImageNet: 90 epochs with warmup + cosine
        >>> warmup = braintools.optim.LinearLR(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> cosine = braintools.optim.CosineAnnealingLR(
        ...     base_lr=0.1,
        ...     T_max=85,
        ...     eta_min=0
        ... )
        >>> scheduler = braintools.optim.ChainedScheduler([warmup, cosine])
        >>>
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(90):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Fine-tuning with gentle cosine decay:**

    .. code-block:: python

        >>> # Gentle decay for fine-tuning: min lr = 10% of base lr
        >>> scheduler = braintools.optim.CosineAnnealingLR(
        ...     base_lr=0.0001,
        ...     T_max=30,
        ...     eta_min=0.00001
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler, weight_decay=1e-5)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(30):
        ...     finetune_epoch(model, optimizer, finetune_loader)
        ...     scheduler.step()

    **Saving and loading state:**

    .. code-block:: python

        >>> scheduler = braintools.optim.CosineAnnealingLR(
        ...     base_lr=0.1,
        ...     T_max=100,
        ...     eta_min=0
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Train for some epochs
        >>> for epoch in range(50):
        ...     scheduler.step()
        >>>
        >>> # Save checkpoint
        >>> checkpoint = {
        ...     'epoch': 50,
        ...     'model': model.state_dict(),
        ...     'scheduler': scheduler.state_dict(),
        ... }
        >>>
        >>> # Resume training
        >>> new_scheduler = braintools.optim.CosineAnnealingLR(
        ...     base_lr=0.1,
        ...     T_max=100,
        ...     eta_min=0
        ... )
        >>> new_scheduler.load_state_dict(checkpoint['scheduler'])
        >>> # Continue from epoch 50 with correct lr

    **Vision Transformer training:**

    .. code-block:: python

        >>> # ViT training schedule
        >>> warmup = braintools.optim.LinearLR(
        ...     start_factor=0.001,
        ...     end_factor=1.0,
        ...     total_iters=10
        ... )
        >>> cosine = braintools.optim.CosineAnnealingLR(
        ...     base_lr=0.001,
        ...     T_max=290,
        ...     eta_min=1e-6
        ... )
        >>> scheduler = braintools.optim.ChainedScheduler([warmup, cosine])
        >>>
        >>> optimizer = braintools.optim.AdamW(lr=scheduler, weight_decay=0.05)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(300):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    See Also
    --------
    CosineAnnealingWarmRestarts : Cosine annealing with periodic restarts
    ExponentialLR : Exponential learning rate decay
    LinearLR : Linear learning rate warmup/cooldown
    WarmupCosineSchedule : Integrated warmup + cosine schedule

    References
    ----------
    .. [1] Loshchilov, I., & Hutter, F. (2016).
           "SGDR: Stochastic gradient descent with warm restarts."
           arXiv preprint arXiv:1608.03983.
    .. [2] He, K., Zhang, X., Ren, S., & Sun, J. (2016).
           "Deep residual learning for image recognition."
           Proceedings of the IEEE conference on computer vision and pattern
           recognition, 770-778.
    .. [3] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2020).
           "An image is worth 16x16 words: Transformers for image recognition at scale."
           arXiv preprint arXiv:2010.11929.
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        T_max: int = 50,
        eta_min: float = 0,
        last_epoch: int = 0,
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        # JIT-compatible cosine annealing computation
        epoch = self.last_epoch.value
        factor = (1 + jnp.cos(jnp.pi * epoch / self.T_max)) / 2
        return [
            self.eta_min + (base_lr - self.eta_min) * factor
            for base_lr in self.base_lrs
        ]


class PolynomialLR(LRScheduler):
    r"""Polynomial learning rate scheduler - Decays learning rate using polynomial function.

    PolynomialLR decreases the learning rate according to a polynomial decay schedule.
    The learning rate is multiplied by a decay factor that follows the formula
    (1 - t/T)^power, where t is the current epoch and T is total_iters. This provides
    smooth decay with controllable rate via the power parameter.

    Parameters
    ----------
    base_lr : float or list of float, optional
        Initial learning rate(s). Can be a single float or a list of floats for multiple
        parameter groups. Default: 1e-3.
    total_iters : int, optional
        Number of epochs over which to decay the learning rate. After total_iters epochs,
        the learning rate becomes 0. Default: 5.
    power : float, optional
        The power of the polynomial. Controls the shape of the decay curve.

        - power=1.0: Linear decay
        - power>1.0: Slower initial decay, faster later
        - power<1.0: Faster initial decay, slower later
        Default: 1.0.
    last_epoch : int, optional
        The index of the last epoch. Used for resuming training. Default: -1 (starts
        from beginning).

    Notes
    -----
    The learning rate at epoch :math:`t` is computed as:

    .. math::
        \eta_t = \eta_0 \cdot \left(1 - \frac{\min(t, T)}{T}\right)^p

    where :math:`\eta_0` is the initial learning rate (base_lr), :math:`T` is total_iters,
    :math:`t` is the current epoch, and :math:`p` is the power parameter.

    **Key characteristics:**

    - Smooth polynomial decay to zero (or near-zero)
    - Decay shape controlled by power parameter
    - Learning rate reaches 0 at total_iters
    - Commonly used in semantic segmentation and detection tasks

    **Power parameter effects:**

    - power=0.5: Square root decay (very fast initial decay)
    - power=1.0: Linear decay (constant rate)
    - power=2.0: Quadratic decay (slow initial, fast final)
    - power=3.0: Cubic decay (very slow initial, very fast final)

    **When to use:**

    - Training semantic segmentation models (DeepLab, FCN)
    - Object detection training (YOLO, RetinaNet)
    - When you want smooth decay to very low learning rates
    - Tasks that benefit from extended low-lr fine-tuning

    Examples
    --------
    **Basic linear decay (power=1.0):**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> # Linear decay from 0.1 to 0 over 100 epochs
        >>> scheduler = braintools.optim.PolynomialLR(
        ...     base_lr=0.1,
        ...     total_iters=100,
        ...     power=1.0
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        # lr decreases linearly: 0.1, 0.099, 0.098, ..., 0.001, 0

    **Quadratic decay (power=2.0):**

    .. code-block:: python

        >>> # Slower initial decay, faster later decay
        >>> scheduler = braintools.optim.PolynomialLR(
        ...     base_lr=0.1,
        ...     total_iters=100,
        ...     power=2.0
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        # lr: epoch 25 ≈ 0.056, epoch 50 ≈ 0.025, epoch 75 ≈ 0.006

    **Square root decay (power=0.5):**

    .. code-block:: python

        >>> # Faster initial decay, slower later decay
        >>> scheduler = braintools.optim.PolynomialLR(
        ...     base_lr=0.01,
        ...     total_iters=50,
        ...     power=0.5
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(50):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Semantic segmentation training (DeepLab style):**

    .. code-block:: python

        >>> # Common setup for semantic segmentation
        >>> scheduler = braintools.optim.PolynomialLR(
        ...     base_lr=0.007,
        ...     total_iters=30000,  # Iterations, not epochs
        ...     power=0.9
        ... )
        >>> optimizer = braintools.optim.SGD(
        ...     lr=scheduler,
        ...     momentum=0.9,
        ...     weight_decay=5e-4
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for iteration in range(30000):
        ...     train_step(model, optimizer, batch)
        ...     scheduler.step()

    **Short training with steep decay:**

    .. code-block:: python

        >>> # Quick decay for fine-tuning
        >>> scheduler = braintools.optim.PolynomialLR(
        ...     base_lr=0.001,
        ...     total_iters=10,
        ...     power=1.0
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler, weight_decay=1e-5)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(10):
        ...     finetune_epoch(model, optimizer, finetune_loader)
        ...     scheduler.step()

    **With warmup:**

    .. code-block:: python

        >>> # Warmup followed by polynomial decay
        >>> warmup = braintools.optim.LinearLR(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> poly_decay = braintools.optim.PolynomialLR(
        ...     base_lr=0.01,
        ...     total_iters=95,
        ...     power=0.9
        ... )
        >>> scheduler = braintools.optim.ChainedScheduler([warmup, poly_decay])
        >>>
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **State persistence:**

    .. code-block:: python

        >>> scheduler = braintools.optim.PolynomialLR(
        ...     base_lr=0.1,
        ...     total_iters=100,
        ...     power=2.0
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Train for some epochs
        >>> for epoch in range(50):
        ...     scheduler.step()
        >>>
        >>> # Save checkpoint
        >>> checkpoint = {
        ...     'epoch': 50,
        ...     'scheduler': scheduler.state_dict(),
        ... }
        >>>
        >>> # Resume training
        >>> new_scheduler = braintools.optim.PolynomialLR(
        ...     base_lr=0.1,
        ...     total_iters=100,
        ...     power=2.0
        ... )
        >>> new_scheduler.load_state_dict(checkpoint['scheduler'])

    **Comparison of power values:**

    .. code-block:: python

        >>> # Visualize different power values
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>>
        >>> powers = [0.5, 1.0, 2.0, 3.0]
        >>> total_iters = 100
        >>> base_lr = 0.1
        >>>
        >>> for power in powers:
        ...     scheduler = braintools.optim.PolynomialLR(
        ...         base_lr=base_lr,
        ...         total_iters=total_iters,
        ...         power=power
        ...     )
        ...     lrs = []
        ...     for _ in range(total_iters):
        ...         lrs.append(scheduler.current_lrs.value[0])
        ...         scheduler.step()
        ...     plt.plot(lrs, label=f'power={power}')
        >>>
        >>> plt.xlabel('Epoch')
        >>> plt.ylabel('Learning Rate')
        >>> plt.legend()
        >>> plt.title('Polynomial LR Decay with Different Powers')
        >>> plt.show()

    See Also
    --------
    LinearLR : Linear learning rate scaling (special case with power=1.0)
    ExponentialLR : Exponential decay
    CosineAnnealingLR : Cosine annealing schedule

    References
    ----------
    .. [1] Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2017).
           "DeepLab: Semantic image segmentation with deep convolutional nets, atrous
           convolution, and fully connected CRFs."
           IEEE transactions on pattern analysis and machine intelligence, 40(4), 834-848.
    .. [2] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
           "Focal loss for dense object detection."
           Proceedings of the IEEE international conference on computer vision, 2980-2988.
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        total_iters: int = 5,
        power: float = 1.0,
        last_epoch: int = 0,
    ):
        self.total_iters = total_iters
        self.power = power
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        decay_factor = ((1 - jnp.minimum(self.last_epoch.value, self.total_iters) / self.total_iters) ** self.power)
        return [base_lr * decay_factor for base_lr in self.base_lrs]


class WarmupScheduler(LRScheduler):
    r"""Warmup learning rate scheduler - Linearly increases learning rate during warmup phase.

    WarmupScheduler gradually increases the learning rate from a small initial value
    (warmup_start_lr) to the base learning rate over a specified number of warmup epochs.
    After the warmup period, the learning rate stays constant at the base learning rate.
    This is commonly used at the beginning of training to stabilize the optimization.

    Parameters
    ----------
    base_lr : float or list of float, optional
        Target learning rate(s) after warmup. Can be a single float or a list of floats
        for multiple parameter groups. Default: 1e-3.
    warmup_epochs : int
        Number of epochs for the warmup phase. The learning rate will increase linearly
        from warmup_start_lr to base_lr over this many epochs.
    warmup_start_lr : float, optional
        Initial learning rate at the start of warmup. Default: 0.0.
    last_epoch : int, optional
        The index of the last epoch. Used for resuming training. Default: -1 (starts
        from beginning).

    Notes
    -----
    The learning rate at epoch :math:`t` is computed as:

    .. math::
        \eta_t = \begin{cases}
            \eta_{\text{start}} + (\eta_{\text{base}} - \eta_{\text{start}}) \cdot \frac{t}{T_{\text{warmup}}}
            & \text{if } t < T_{\text{warmup}} \\
            \eta_{\text{base}} & \text{otherwise}
        \end{cases}

    where :math:`\eta_{\text{start}}` is warmup_start_lr, :math:`\eta_{\text{base}}` is base_lr,
    :math:`T_{\text{warmup}}` is warmup_epochs, and :math:`t` is the current epoch.

    **Key characteristics:**

    - Linear warmup from small initial lr to target lr
    - Prevents instability from large initial gradients
    - Especially important for large batch training
    - Learning rate remains constant after warmup period

    **Common warmup configurations:**

    - Short warmup: 5-10 epochs for standard training
    - Medium warmup: 10-20 epochs for large batch training
    - Long warmup: 30-50 epochs for very large batches or transformers
    - Start lr: Usually 0 or 0.01-0.1 * base_lr

    **When to use:**

    - Training with large batch sizes (>256)
    - Training transformer models (BERT, GPT, ViT)
    - When model shows initial training instability
    - Fine-tuning with aggressive learning rates

    Examples
    --------
    **Basic warmup:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> # Warmup from 0 to 0.1 over 10 epochs
        >>> scheduler = braintools.optim.WarmupScheduler(
        ...     base_lr=0.1,
        ...     warmup_epochs=10,
        ...     warmup_start_lr=0.0
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(50):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        # Epochs 0-9: lr increases linearly from 0 to 0.1
        # Epochs 10+: lr stays at 0.1

    **Warmup with non-zero start:**

    .. code-block:: python

        >>> # Start from 10% of target lr
        >>> scheduler = braintools.optim.WarmupScheduler(
        ...     base_lr=0.01,
        ...     warmup_epochs=5,
        ...     warmup_start_lr=0.001
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(30):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Large batch training:**

    .. code-block:: python

        >>> # Warmup for large batch size (1024+)
        >>> scheduler = braintools.optim.WarmupScheduler(
        ...     base_lr=0.4,  # Linear scaling rule: 0.1 * (batch_size / 256)
        ...     warmup_epochs=20,
        ...     warmup_start_lr=0.0
        ... )
        >>> optimizer = braintools.optim.SGD(
        ...     lr=scheduler,
        ...     momentum=0.9,
        ...     weight_decay=1e-4
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     train_epoch(model, optimizer, large_batch_loader)
        ...     scheduler.step()

    **Transformer training warmup:**

    .. code-block:: python

        >>> # BERT-style warmup
        >>> scheduler = braintools.optim.WarmupScheduler(
        ...     base_lr=0.0001,
        ...     warmup_epochs=10000,  # Often in steps/iterations
        ...     warmup_start_lr=0.0
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler, weight_decay=0.01)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for step in range(100000):
        ...     train_step(model, optimizer, batch)
        ...     scheduler.step()

    **Warmup followed by decay (using ChainedScheduler):**

    .. code-block:: python

        >>> # Warmup then step decay
        >>> warmup = braintools.optim.WarmupScheduler(
        ...     base_lr=0.1,
        ...     warmup_epochs=5,
        ...     warmup_start_lr=0.0
        ... )
        >>> decay = braintools.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> scheduler = braintools.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(90):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Short warmup for fine-tuning:**

    .. code-block:: python

        >>> # Gentle warmup for transfer learning
        >>> scheduler = braintools.optim.WarmupScheduler(
        ...     base_lr=0.0001,
        ...     warmup_epochs=3,
        ...     warmup_start_lr=0.00001
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler, weight_decay=1e-5)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(20):
        ...     finetune_epoch(model, optimizer, finetune_loader)
        ...     scheduler.step()

    **Vision Transformer training:**

    .. code-block:: python

        >>> # ViT warmup schedule
        >>> warmup = braintools.optim.WarmupScheduler(
        ...     base_lr=0.001,
        ...     warmup_epochs=10,
        ...     warmup_start_lr=0.0
        ... )
        >>> cosine = braintools.optim.CosineAnnealingLR(
        ...     base_lr=0.001,
        ...     T_max=290,
        ...     eta_min=1e-6
        ... )
        >>> # Use sequentially: warmup first, then cosine
        >>> optimizer = braintools.optim.AdamW(lr=warmup, weight_decay=0.05)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Warmup phase
        >>> for epoch in range(10):
        ...     optimizer.step(grads)
        ...     warmup.step()
        >>>
        >>> # Switch to cosine after warmup
        >>> cosine.attach_optimizer(optimizer)
        >>> for epoch in range(290):
        ...     optimizer.step(grads)
        ...     cosine.step()

    **State persistence:**

    .. code-block:: python

        >>> scheduler = braintools.optim.WarmupScheduler(
        ...     base_lr=0.1,
        ...     warmup_epochs=10,
        ...     warmup_start_lr=0.0
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Train for some epochs
        >>> for epoch in range(5):
        ...     scheduler.step()
        >>>
        >>> # Save checkpoint
        >>> checkpoint = {
        ...     'epoch': 5,
        ...     'scheduler': scheduler.state_dict(),
        ... }
        >>>
        >>> # Resume training
        >>> new_scheduler = braintools.optim.WarmupScheduler(
        ...     base_lr=0.1,
        ...     warmup_epochs=10,
        ...     warmup_start_lr=0.0
        ... )
        >>> new_scheduler.load_state_dict(checkpoint['scheduler'])

    **Comparison with LinearLR:**

    .. code-block:: python

        >>> # WarmupScheduler: lr increases then stays constant
        >>> warmup_sched = braintools.optim.WarmupScheduler(
        ...     base_lr=0.1,
        ...     warmup_epochs=10,
        ...     warmup_start_lr=0.0
        ... )
        >>>
        >>> # LinearLR: lr increases then CAN decrease or stay constant
        >>> linear_sched = braintools.optim.LinearLR(
        ...     start_factor=0.0,
        ...     end_factor=1.0,
        ...     total_iters=10
        ... )
        >>> # Both achieve similar warmup, but LinearLR is more flexible

    See Also
    --------
    LinearLR : More flexible linear scaling (can warmup or cooldown)
    ConstantLR : Constant factor multiplication
    ChainedScheduler : Combine warmup with other schedules

    References
    ----------
    .. [1] Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L.,
           Kyrola, A., ... & He, K. (2017).
           "Accurate, large minibatch SGD: Training imagenet in 1 hour."
           arXiv preprint arXiv:1706.02677.
    .. [2] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018).
           "BERT: Pre-training of deep bidirectional transformers for language understanding."
           arXiv preprint arXiv:1810.04805.
    .. [3] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2020).
           "An image is worth 16x16 words: Transformers for image recognition at scale."
           arXiv preprint arXiv:2010.11929.
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        warmup_epochs: int = 5,
        warmup_start_lr: float = 0.0,
        last_epoch: int = 0,
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        # JIT-compatible warmup computation using jnp.where
        alpha = jnp.minimum(self.last_epoch.value / self.warmup_epochs, 1.0)
        return [
            self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
            for base_lr in self.base_lrs
        ]


class CyclicLR(LRScheduler):
    r"""Cyclic learning rate scheduler - Oscillates learning rate between bounds.

    CyclicLR implements a learning rate schedule that cyclically varies between a
    minimum (base_lr) and maximum (max_lr) learning rate. This helps the optimizer
    explore different regions of the loss landscape and can lead to better convergence
    and generalization. The policy was originally proposed for faster convergence
    without extensive hyperparameter tuning.

    Parameters
    ----------
    base_lr : float or list of float, optional
        Lower learning rate boundaries in each cycle. This is the minimum learning
        rate during the cycle. Can be a single float or list for multiple parameter
        groups. Default: 1e-3.
    max_lr : float or list of float, optional
        Upper learning rate boundaries in each cycle. The learning rate will oscillate
        between base_lr and max_lr. Can be a single float or list. Default: 1e-2.
    step_size_up : int, optional
        Number of iterations in the increasing half of a cycle. Default: 2000.
    step_size_down : int, optional
        Number of iterations in the decreasing half of a cycle. If None, it's set
        equal to step_size_up. Default: None.
    mode : str, optional
        One of {'triangular', 'triangular2', 'exp_range'}. Values correspond to
        policies detailed below:

        - 'triangular': Basic triangular cycle without amplitude scaling.
        - 'triangular2': Basic triangular cycle that scales amplitude by half each cycle.
        - 'exp_range': Triangular cycle that scales amplitude by gamma^(cycle iterations).

        Default: 'triangular'.
    gamma : float, optional
        Constant used in 'exp_range' mode for multiplicative scaling.
        gamma^(cycle iterations) gives the scaling factor. Default: 1.0.
    scale_fn : callable, optional
        Custom scaling function given y = scale_fn(x), where x is the current
        cycle iteration. Overrides mode parameter. Default: None.
    scale_mode : str, optional
        {'cycle', 'iterations'}. Determines whether scale_fn uses cycle number
        or cycle iterations as input. Default: 'cycle'.
    last_epoch : int, optional
        The index of the last epoch. Used when resuming training. Default: 0.

    Notes
    -----
    **Mathematical Formulation:**

    The learning rate oscillates according to:

    .. math::
        \text{lr} = \text{base_lr} + (\text{max_lr} - \text{base_lr})
                    \times \max(0, 1 - |x - 1|) \times \text{scale}

    where x cycles between 0 and 2, and scale depends on the mode.

    **Modes Explained:**

    1. **triangular**: Constant amplitude oscillation
       - LR oscillates between base_lr and max_lr with fixed amplitude

    2. **triangular2**: Decaying amplitude by half each cycle
       - Amplitude = (max_lr - base_lr) * 0.5^(cycle_number)

    3. **exp_range**: Exponentially decaying amplitude
       - Amplitude = (max_lr - base_lr) * gamma^(iterations)

    **Finding Optimal LR Range:**

    Use the LR range test to find optimal base_lr and max_lr:
    1. Start with very low LR (e.g., 1e-7)
    2. Increase LR exponentially each batch
    3. Plot loss vs LR and find:
       - base_lr: LR where loss starts decreasing
       - max_lr: LR where loss stops decreasing or starts increasing

    **Benefits:**

    - **No manual schedule tuning**: Automatically handles LR scheduling
    - **Escapes saddle points**: Periodic high LR helps escape flat regions
    - **Better generalization**: Oscillation prevents overfitting to sharp minima
    - **Fast convergence**: Can achieve super-convergence with proper range

    Examples
    --------
    **Basic triangular schedule:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> # Basic triangular oscillation
        >>> scheduler = braintools.optim.CyclicLR(
        ...     base_lr=0.001,
        ...     max_lr=0.006,
        ...     step_size_up=2000,  # 2000 iterations to go from base to max
        ...     mode='triangular'
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    **Triangular2 with amplitude decay:**

    .. code-block:: python

        >>> # Amplitude halves each cycle for fine-tuning
        >>> scheduler = braintools.optim.CyclicLR(
        ...     base_lr=0.0001,
        ...     max_lr=0.001,
        ...     step_size_up=1000,
        ...     step_size_down=1000,
        ...     mode='triangular2'  # Amplitude decay
        ... )
        >>>
        >>> # First cycle: LR oscillates between 0.0001 and 0.001
        >>> # Second cycle: LR oscillates between 0.0001 and 0.0055
        >>> # Third cycle: LR oscillates between 0.0001 and 0.00325
        >>> # And so on...

    **Exponential range decay:**

    .. code-block:: python

        >>> # Exponentially decreasing amplitude
        >>> scheduler = braintools.optim.CyclicLR(
        ...     base_lr=0.0001,
        ...     max_lr=0.01,
        ...     step_size_up=500,
        ...     mode='exp_range',
        ...     gamma=0.99994  # Gradual decay
        ... )

    **Asymmetric cycles:**

    .. code-block:: python

        >>> # Spend more time at lower learning rates
        >>> scheduler = braintools.optim.CyclicLR(
        ...     base_lr=0.0001,
        ...     max_lr=0.001,
        ...     step_size_up=500,   # Quick ramp up
        ...     step_size_down=1500  # Slow ramp down
        ... )

    **Custom scaling function:**

    .. code-block:: python

        >>> # Custom amplitude scaling
        >>> def custom_scale(x):
        ...     '''Custom scaling: faster decay initially'''
        ...     return 1 / (1 + 0.0005 * x)
        >>>
        >>> scheduler = braintools.optim.CyclicLR(
        ...     base_lr=0.001,
        ...     max_lr=0.1,
        ...     step_size_up=1000,
        ...     scale_fn=custom_scale,
        ...     scale_mode='iterations'
        ... )

    **LR range test implementation:**

    .. code-block:: python

        >>> # Find optimal LR range
        >>> def lr_range_test(model, data_loader, max_lr=10, num_iter=100):
        ...     scheduler = braintools.optim.CyclicLR(
        ...         base_lr=1e-7,
        ...         max_lr=max_lr,
        ...         step_size_up=num_iter,
        ...         mode='triangular'
        ...     )
        ...     optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        ...
        ...     lrs, losses = [], []
        ...     for i, batch in enumerate(data_loader):
        ...         if i >= num_iter:
        ...             break
        ...         loss = train_step(model, batch, optimizer)
        ...         lrs.append(scheduler.get_lr()[0])
        ...         losses.append(loss)
        ...         scheduler.step()
        ...
        ...     # Plot and find optimal range
        ...     import matplotlib.pyplot as plt
        ...     plt.semilogx(lrs, losses)
        ...     plt.xlabel('Learning Rate')
        ...     plt.ylabel('Loss')
        ...     plt.show()

    **For super-convergence:**

    .. code-block:: python

        >>> # Super-convergence with one cycle
        >>> # Use with large batch sizes and proper regularization
        >>> scheduler = braintools.optim.CyclicLR(
        ...     base_lr=0.08,   # Relatively high base_lr
        ...     max_lr=0.8,     # Very high max_lr
        ...     step_size_up=epochs * len(train_loader) // 2,
        ...     step_size_down=epochs * len(train_loader) // 2,
        ...     mode='triangular'
        ... )
        >>>
        >>> # Combine with strong regularization
        >>> optimizer = braintools.optim.SGD(
        ...     lr=scheduler,
        ...     momentum=0.95,
        ...     weight_decay=1e-4
        ... )

    **Multiple parameter groups:**

    .. code-block:: python

        >>> # Different LR ranges for different layers
        >>> scheduler = braintools.optim.CyclicLR(
        ...     base_lr=[0.0001, 0.001],   # Lower for pretrained layers
        ...     max_lr=[0.001, 0.01],       # Higher for new layers
        ...     step_size_up=1000
        ... )

    **Monitoring cycles:**

    .. code-block:: python

        >>> scheduler = braintools.optim.CyclicLR(
        ...     base_lr=0.001,
        ...     max_lr=0.01,
        ...     step_size_up=100,
        ...     step_size_down=100
        ... )
        >>>
        >>> for iteration in range(1000):
        ...     train_step(...)
        ...     scheduler.step()
        ...
        ...     if iteration % 50 == 0:
        ...         cycle = iteration // (scheduler.step_size_up + scheduler.step_size_down)
        ...         lr = scheduler.get_lr()[0]
        ...         print(f"Iter {iteration}, Cycle {cycle}, LR: {lr:.6f}")

    See Also
    --------
    OneCycleLR : One cycle learning rate policy
    CosineAnnealingLR : Cosine annealing schedule
    CosineAnnealingWarmRestarts : Cosine annealing with restarts
    TriangularLR : Simplified triangular schedule

    References
    ----------
    .. [1] Smith, L. N. (2017).
           "Cyclical learning rates for training neural networks."
           2017 IEEE Winter Conference on Applications of Computer Vision (WACV).
    .. [2] Smith, L. N. (2018).
           "A disciplined approach to neural network hyper-parameters: Part 1 --
           learning rate, batch size, momentum, and weight decay."
           arXiv preprint arXiv:1803.09820.
    .. [3] Smith, L. N., & Topin, N. (2019).
           "Super-convergence: Very fast training of neural networks using large learning rates."
           Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications.
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        max_lr: Union[float, List[float]] = 1e-2,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn: Optional[Callable] = None,
        scale_mode: str = 'cycle',
        last_epoch: int = 0,
    ):
        # Store max_lr separately as it's not part of base class
        if isinstance(max_lr, list):
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr]

        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode

        # Initialize base class with base_lr
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        cycle = jnp.floor(1 + self.last_epoch.value / (self.step_size_up + self.step_size_down))
        x = jnp.abs(self.last_epoch.value / self.step_size_up - 2 * cycle + 1)

        if self.scale_fn is None:
            if self.mode == 'triangular':
                scale = 1.0
            elif self.mode == 'triangular2':
                scale = 1.0 / (2.0 ** (cycle - 1))
            elif self.mode == 'exp_range':
                scale = self.gamma ** self.last_epoch.value
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        else:
            if self.scale_mode == 'cycle':
                scale = self.scale_fn(cycle)
            else:
                scale = self.scale_fn(self.last_epoch.value)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = max_lr - base_lr
            lr = base_lr + base_height * scale * jnp.maximum(0, 1 - x)
            lrs.append(lr)

        return lrs


class OneCycleLR(LRScheduler):
    r"""One cycle learning rate scheduler - Super-convergence training policy.

    OneCycleLR implements the 1cycle learning rate policy, which enables super-convergence
    - training neural networks an order of magnitude faster than with standard methods.
    The policy consists of two phases: first increasing the learning rate from a low value
    to a maximum value, then decreasing it to a value much lower than the initial one.
    This is typically combined with momentum scheduling in the opposite direction.

    Parameters
    ----------
    max_lr : float or list of float, optional
        Upper learning rate boundaries in the cycle. This is the peak learning rate
        that will be reached during training. Can be a single float or list for
        multiple parameter groups. Default: 1e-2.
    total_steps : int, optional
        The total number of steps (batches) in the cycle. Either this or the
        combination of epochs and steps_per_epoch must be provided.
    epochs : int, optional
        The number of epochs to train for. Used with steps_per_epoch to calculate
        total_steps if total_steps is not provided.
    steps_per_epoch : int, optional
        The number of steps (batches) per epoch. Used with epochs to calculate
        total_steps if total_steps is not provided.
    pct_start : float, optional
        The percentage of the cycle spent increasing the learning rate.
        Default: 0.3 (30% of cycle for warmup).
    anneal_strategy : str, optional
        {'cos', 'linear'}. Specifies the annealing strategy:

        - 'cos': Cosine annealing from max_lr to final_lr
        - 'linear': Linear annealing from max_lr to final_lr

        Default: 'cos'.
    div_factor : float, optional
        Determines the initial learning rate via initial_lr = max_lr / div_factor.
        Default: 25.0.
    final_div_factor : float, optional
        Determines the final learning rate via final_lr = max_lr / final_div_factor.
        Default: 1e4.
    last_epoch : int, optional
        The index of the last batch. Used when resuming training. Default: 0.

    Notes
    -----
    **Three Phases of OneCycleLR:**

    1. **Warmup phase** (0 to pct_start):
       - LR increases from initial_lr to max_lr
       - Allows gradients to stabilize

    2. **Annealing phase** (pct_start to 1.0):
       - LR decreases from max_lr to final_lr
       - Uses cosine or linear annealing

    3. **Final phase** (optional extension):
       - LR stays at final_lr for additional training

    **Mathematical Formulation:**

    Initial learning rate:

    .. math::
        \text{initial_lr} = \frac{\text{max_lr}}{\text{div_factor}}

    Final learning rate:

    .. math::
        \text{final_lr} = \frac{\text{max_lr}}{\text{final_div_factor}}

    **Super-Convergence Benefits:**

    - **10x faster training**: Achieve same accuracy in 1/10th the epochs
    - **Better generalization**: Often achieves better final accuracy
    - **Regularization effect**: High LR acts as regularization
    - **Simpler hyperparameter tuning**: Mainly need to find max_lr

    **Finding Optimal max_lr:**

    Use the LR range test:
    1. Start with very small LR
    2. Gradually increase LR each batch
    3. Plot loss vs LR
    4. Choose max_lr slightly less than where loss starts increasing

    **Momentum Scheduling:**

    OneCycleLR works best with momentum scheduling in opposite direction:
    - When LR increases, momentum decreases
    - When LR decreases, momentum increases

    Examples
    --------
    **Basic usage with super-convergence:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> # Training for 5 epochs with 100 batches per epoch
        >>> scheduler = braintools.optim.OneCycleLR(
        ...     max_lr=0.1,
        ...     epochs=5,
        ...     steps_per_epoch=100
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(5):
        ...     for batch in train_loader:
        ...         train_step(batch)
        ...         scheduler.step()

    **With total steps specification:**

    .. code-block:: python

        >>> # Specify total training steps directly
        >>> total_training_steps = 10000
        >>> scheduler = braintools.optim.OneCycleLR(
        ...     max_lr=0.3,
        ...     total_steps=total_training_steps,
        ...     pct_start=0.3,  # 30% warmup
        ...     anneal_strategy='cos'
        ... )

    **Custom phase percentages:**

    .. code-block:: python

        >>> # Longer warmup phase (40% of training)
        >>> scheduler = braintools.optim.OneCycleLR(
        ...     max_lr=0.1,
        ...     total_steps=5000,
        ...     pct_start=0.4,  # 40% for warmup
        ...     div_factor=10,  # Start from 0.01
        ...     final_div_factor=100  # End at 0.001
        ... )

    **For different model sizes:**

    .. code-block:: python

        >>> # Small model/dataset - conservative settings
        >>> scheduler_small = braintools.optim.OneCycleLR(
        ...     max_lr=0.01,
        ...     total_steps=1000,
        ...     pct_start=0.3,
        ...     div_factor=25,
        ...     final_div_factor=1000
        ... )
        >>>
        >>> # Large model - aggressive settings for super-convergence
        >>> scheduler_large = braintools.optim.OneCycleLR(
        ...     max_lr=1.0,  # Very high max_lr
        ...     total_steps=10000,
        ...     pct_start=0.2,  # Shorter warmup
        ...     div_factor=25,
        ...     final_div_factor=10000
        ... )

    **With momentum cycling (recommended):**

    .. code-block:: python

        >>> class OneCycleOptimizer:
        ...     def __init__(self, model, max_lr=0.1, total_steps=1000):
        ...         self.scheduler = braintools.optim.OneCycleLR(
        ...             max_lr=max_lr,
        ...             total_steps=total_steps
        ...         )
        ...         self.base_momentum = 0.85
        ...         self.max_momentum = 0.95
        ...         self.optimizer = braintools.optim.SGD(
        ...             lr=self.scheduler,
        ...             momentum=self.max_momentum
        ...         )
        ...
        ...     def step(self, grads):
        ...         # Update learning rate
        ...         self.scheduler.step()
        ...
        ...         # Cycle momentum in opposite direction
        ...         pct_complete = self.scheduler.last_epoch / self.scheduler.total_steps
        ...         if pct_complete < self.scheduler.pct_start:
        ...             # LR increasing, momentum decreasing
        ...             momentum = self.max_momentum - (self.max_momentum - self.base_momentum) * pct_complete / self.scheduler.pct_start
        ...         else:
        ...             # LR decreasing, momentum increasing
        ...             momentum = self.base_momentum + (self.max_momentum - self.base_momentum) * (pct_complete - self.scheduler.pct_start) / (1 - self.scheduler.pct_start)
        ...
        ...         self.optimizer.momentum = momentum
        ...         self.optimizer.update(grads)

    **LR range test for finding max_lr:**

    .. code-block:: python

        >>> def find_max_lr(model, data_loader, init_lr=1e-7, final_lr=10, num_iter=100):
        ...     '''Find optimal max_lr using LR range test'''
        ...     scheduler = braintools.optim.OneCycleLR(
        ...         max_lr=final_lr,
        ...         total_steps=num_iter,
        ...         div_factor=final_lr/init_lr,
        ...         final_div_factor=1.0,  # Don't decrease at end
        ...         pct_start=1.0  # Only increase
        ...     )
        ...     optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        ...
        ...     lrs, losses = [], []
        ...     for i, batch in enumerate(data_loader):
        ...         if i >= num_iter:
        ...             break
        ...
        ...         loss = compute_loss(model, batch)
        ...         grads = compute_gradients(loss)
        ...         optimizer.update(grads)
        ...
        ...         lrs.append(scheduler.get_lr()[0])
        ...         losses.append(loss.item())
        ...         scheduler.step()
        ...
        ...     # Find LR where loss stops decreasing
        ...     import numpy as np
        ...     smooth_losses = np.convolve(losses, np.ones(5)/5, mode='valid')
        ...     max_lr_idx = np.argmin(smooth_losses) + len(losses) - len(smooth_losses)
        ...     suggested_max_lr = lrs[max_lr_idx]
        ...     print(f"Suggested max_lr: {suggested_max_lr}")
        ...
        ...     return lrs, losses

    **Transfer learning with OneCycle:**

    .. code-block:: python

        >>> # Fine-tuning pretrained model
        >>> scheduler = braintools.optim.OneCycleLR(
        ...     max_lr=0.001,  # Lower max_lr for fine-tuning
        ...     total_steps=2000,
        ...     pct_start=0.1,  # Short warmup
        ...     div_factor=100,  # Very low initial LR
        ...     final_div_factor=1000
        ... )
        >>>
        >>> # Freeze early layers initially
        >>> for param in model.early_layers.parameters():
        ...     param.requires_grad = False
        >>>
        >>> # Unfreeze after warmup
        >>> def unfreeze_callback(epoch):
        ...     if epoch > scheduler.total_steps * scheduler.pct_start:
        ...         for param in model.early_layers.parameters():
        ...             param.requires_grad = True

    **Different annealing strategies:**

    .. code-block:: python

        >>> # Cosine annealing (smoother)
        >>> scheduler_cos = braintools.optim.OneCycleLR(
        ...     max_lr=0.1,
        ...     total_steps=1000,
        ...     anneal_strategy='cos'
        ... )
        >>>
        >>> # Linear annealing (more aggressive)
        >>> scheduler_linear = braintools.optim.OneCycleLR(
        ...     max_lr=0.1,
        ...     total_steps=1000,
        ...     anneal_strategy='linear'
        ... )

    **Monitoring training progress:**

    .. code-block:: python

        >>> scheduler = braintools.optim.OneCycleLR(
        ...     max_lr=0.1,
        ...     total_steps=1000,
        ...     pct_start=0.3
        ... )
        >>>
        >>> for step in range(1000):
        ...     train_step(...)
        ...     scheduler.step()
        ...
        ...     if step % 100 == 0:
        ...         phase = "warmup" if step < 300 else "annealing"
        ...         lr = scheduler.get_lr()[0]
        ...         progress = step / 1000 * 100
        ...         print(f"Step {step} ({progress:.1f}%): {phase} phase, LR={lr:.6f}")

    **Multiple parameter groups:**

    .. code-block:: python

        >>> # Different max_lr for different layers
        >>> scheduler = braintools.optim.OneCycleLR(
        ...     max_lr=[0.001, 0.01],  # Lower for pretrained, higher for new layers
        ...     total_steps=1000,
        ...     pct_start=0.3
        ... )

    See Also
    --------
    CyclicLR : Cyclic learning rate schedules
    CosineAnnealingLR : Cosine annealing schedule
    LinearLR : Linear learning rate schedule
    WarmupScheduler : Simple warmup schedule

    References
    ----------
    .. [1] Smith, L. N., & Topin, N. (2019).
           "Super-convergence: Very fast training of neural networks using large learning rates."
           Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications.
    .. [2] Smith, L. N. (2018).
           "A disciplined approach to neural network hyper-parameters: Part 1 --
           learning rate, batch size, momentum, and weight decay."
           arXiv preprint arXiv:1803.09820.
    .. [3] Howard, J., & Gugger, S. (2020).
           "Fastai: A layered API for deep learning."
           Information, 11(2), 108.
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        max_lr: Union[float, List[float]] = 1e-2,
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = 0,
    ):
        if total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError("You must define either total_steps or both epochs and steps_per_epoch")
        elif total_steps is not None:
            if total_steps <= 0:
                raise ValueError("total_steps must be positive")
            self.total_steps = total_steps
        else:
            if epochs <= 0 or steps_per_epoch <= 0:
                raise ValueError("epochs and steps_per_epoch must be positive")
            self.total_steps = epochs * steps_per_epoch

        # Store max_lr
        if isinstance(max_lr, list):
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr]

        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        # Compute base_lr from max_lr
        base_lrs = [max_lr / div_factor for max_lr in self.max_lrs]
        self.min_lrs = [max_lr / final_div_factor for max_lr in self.max_lrs]

        # Initialize base class with computed base_lr
        super().__init__(base_lrs, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch.value + 1
        warmup_steps = self.pct_start * self.total_steps

        # JIT-compatible computation using jnp.where
        # Compute warmup learning rate
        warmup_pct = jnp.minimum(step_num / warmup_steps, 1.0)
        warmup_lrs = [base_lr + warmup_pct * (max_lr - base_lr)
                      for base_lr, max_lr in zip(self.base_lrs, self.max_lrs)]

        # Compute annealing learning rate
        anneal_pct = jnp.clip((step_num - warmup_steps) / ((1 - self.pct_start) * self.total_steps), 0.0, 1.0)

        if self.anneal_strategy == 'cos':
            anneal_factor = (1 + jnp.cos(jnp.pi * anneal_pct)) / 2
        elif self.anneal_strategy == 'linear':
            anneal_factor = 1 - anneal_pct
        else:
            raise ValueError(f"Unknown anneal_strategy: {self.anneal_strategy}")

        anneal_lrs = [min_lr + anneal_factor * (max_lr - min_lr)
                      for min_lr, max_lr in zip(self.min_lrs, self.max_lrs)]

        # Choose between warmup and annealing phase
        is_warmup = step_num <= warmup_steps
        return [jnp.where(is_warmup, warmup_lr, anneal_lr)
                for warmup_lr, anneal_lr in zip(warmup_lrs, anneal_lrs)]


class ReduceLROnPlateau(LRScheduler):
    r"""Reduce learning rate when a metric has stopped improving - Adaptive LR based on validation metrics.

    ReduceLROnPlateau monitors a validation metric (like loss or accuracy) and reduces the
    learning rate when the metric stops improving for a specified number of epochs (patience).
    This is useful when you don't know in advance when to reduce the learning rate, letting
    the training dynamics determine the schedule.

    Parameters
    ----------
    base_lr : float or list of float, optional
        Initial learning rate(s). Can be a single float or a list of floats for multiple
        parameter groups. Default: 1e-3.
    mode : {'min', 'max'}, optional
        Whether to minimize or maximize the monitored metric.

        - 'min': Reduce lr when metric stops decreasing (e.g., for loss)
        - 'max': Reduce lr when metric stops increasing (e.g., for accuracy)
        Default: 'min'.
    factor : float, optional
        Factor by which to reduce the learning rate. new_lr = lr * factor.
        Must be in range (0, 1). Default: 0.1.
    patience : int, optional
        Number of epochs with no improvement after which learning rate will be reduced.
        For example, if patience=5, the first 5 epochs with no improvement are tolerated,
        and the lr is reduced on the 6th epoch. Default: 10.
    threshold : float, optional
        Threshold for measuring improvement. Only changes greater than threshold are
        considered as improvement. Default: 1e-4.
    threshold_mode : {'rel', 'abs'}, optional
        How to compute the threshold for improvement.

        - 'rel': dynamic threshold = best * (1 ± threshold)
        - 'abs': static threshold = best ± threshold
        Default: 'rel'.
    cooldown : int, optional
        Number of epochs to wait before resuming normal operation after lr has been reduced.
        During cooldown, no further lr reductions occur. Default: 0.
    min_lr : float or list of float, optional
        Minimum learning rate(s). The lr will not be reduced below this value.
        Default: 0.
    eps : float, optional
        Minimal decay applied to lr. If the difference between new and old lr is smaller
        than eps, the update is ignored. Default: 1e-8.
    last_epoch : int, optional
        The index of the last epoch. Used for resuming training. Default: -1.

    Notes
    -----
    The scheduler reduces the learning rate when the monitored metric plateaus:

    .. math::
        \eta_{t+1} = \begin{cases}
            \max(\eta_t \cdot \text{factor}, \eta_{\min}) & \text{if plateau detected} \\
            \eta_t & \text{otherwise}
        \end{cases}

    A plateau is detected when the metric fails to improve for `patience` consecutive epochs.

    For mode='min', improvement is defined as:

    .. math::
        \text{metric}_t < \text{best} \cdot (1 - \text{threshold}) \quad \text{(relative)}

    or

    .. math::
        \text{metric}_t < \text{best} - \text{threshold} \quad \text{(absolute)}

    **Key characteristics:**

    - Adaptive schedule based on training progress
    - No need to pre-specify decay epochs
    - Ideal when optimal schedule is unknown
    - Commonly used for validation-based training

    **Common configurations:**

    - Conservative: patience=10, factor=0.5
    - Moderate: patience=5, factor=0.1
    - Aggressive: patience=3, factor=0.1

    **When to use:**

    - When you don't know the optimal training schedule
    - For validation-driven training
    - When training dynamics are unpredictable
    - For automatic hyperparameter tuning

    Examples
    --------
    **Basic usage with validation loss:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> scheduler = braintools.optim.ReduceLROnPlateau(
        ...     base_lr=0.1,
        ...     mode='min',
        ...     factor=0.5,
        ...     patience=5,
        ...     min_lr=0.001
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     # Training
        ...     optimizer.step(grads)
        ...
        ...     # Validation
        ...     val_loss = validate(model, val_loader)
        ...
        ...     # Update learning rate based on validation loss
        ...     scheduler.step(val_loss)
        ...
        ...     print(f"Epoch {epoch}: lr={optimizer.current_lr:.6f}, val_loss={val_loss:.4f}")

    **With validation accuracy (maximize mode):**

    .. code-block:: python

        >>> scheduler = braintools.optim.ReduceLROnPlateau(
        ...     base_lr=0.01,
        ...     mode='max',  # Maximize accuracy
        ...     factor=0.1,
        ...     patience=10,
        ...     threshold=0.01
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(200):
        ...     optimizer.step(grads)
        ...     val_acc = evaluate_accuracy(model, val_loader)
        ...     scheduler.step(val_acc)

    **Conservative schedule for stable training:**

    .. code-block:: python

        >>> # Reduce lr by half when no improvement for 10 epochs
        >>> scheduler = braintools.optim.ReduceLROnPlateau(
        ...     base_lr=0.1,
        ...     mode='min',
        ...     factor=0.5,
        ...     patience=10,
        ...     threshold=1e-3,
        ...     cooldown=5  # Wait 5 epochs after reduction
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    **Aggressive schedule for quick adaptation:**

    .. code-block:: python

        >>> # Reduce lr by 90% when no improvement for 3 epochs
        >>> scheduler = braintools.optim.ReduceLROnPlateau(
        ...     base_lr=0.01,
        ...     mode='min',
        ...     factor=0.1,
        ...     patience=3,
        ...     threshold=1e-4,
        ...     min_lr=1e-6
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    **With absolute threshold mode:**

    .. code-block:: python

        >>> # Use absolute threshold for improvement
        >>> scheduler = braintools.optim.ReduceLROnPlateau(
        ...     base_lr=0.1,
        ...     mode='min',
        ...     factor=0.5,
        ...     patience=5,
        ...     threshold=0.001,
        ...     threshold_mode='abs'  # Absolute improvement threshold
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    **Complete training loop with early stopping:**

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> scheduler = braintools.optim.ReduceLROnPlateau(
        ...     base_lr=0.01,
        ...     mode='min',
        ...     factor=0.5,
        ...     patience=10,
        ...     min_lr=1e-5
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> best_loss = float('inf')
        >>> patience_counter = 0
        >>> early_stop_patience = 20
        >>>
        >>> for epoch in range(200):
        ...     # Training
        ...     optimizer.step(grads)
        ...
        ...     # Validation
        ...     val_loss = validate(model, val_loader)
        ...
        ...     # Update learning rate
        ...     old_lr = optimizer.current_lr
        ...     scheduler.step(val_loss)
        ...     if optimizer.current_lr < old_lr:
        ...         print(f"Epoch {epoch}: Reduced LR to {optimizer.current_lr:.6f}")
        ...
        ...     # Early stopping
        ...     if val_loss < best_loss:
        ...         best_loss = val_loss
        ...         patience_counter = 0
        ...         # Save best model
        ...     else:
        ...         patience_counter += 1
        ...         if patience_counter >= early_stop_patience:
        ...             print(f"Early stopping at epoch {epoch}")
        ...             break

    **State persistence:**

    .. code-block:: python

        >>> scheduler = braintools.optim.ReduceLROnPlateau(
        ...     base_lr=0.1,
        ...     mode='min',
        ...     factor=0.5,
        ...     patience=5
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Train for some epochs
        >>> for epoch in range(50):
        ...     val_loss = train_and_validate(model, optimizer)
        ...     scheduler.step(val_loss)
        >>>
        >>> # Save checkpoint
        >>> checkpoint = {
        ...     'epoch': 50,
        ...     'model': model.state_dict(),
        ...     'optimizer': optimizer.state_dict(),
        ...     'scheduler': scheduler.state_dict(),
        ...     'best_metric': scheduler.best
        ... }
        >>>
        >>> # Resume training
        >>> new_scheduler = braintools.optim.ReduceLROnPlateau(
        ...     base_lr=0.1,
        ...     mode='min',
        ...     factor=0.5,
        ...     patience=5
        ... )
        >>> new_scheduler.load_state_dict(checkpoint['scheduler'])
        >>> new_scheduler.best = checkpoint['best_metric']

    **Multiple metrics monitoring:**

    .. code-block:: python

        >>> # Monitor different metrics for different purposes
        >>> val_scheduler = braintools.optim.ReduceLROnPlateau(
        ...     base_lr=0.01,
        ...     mode='min',
        ...     factor=0.5,
        ...     patience=5
        ... )
        >>> optimizer = braintools.optim.Adam(lr=val_scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     optimizer.step(grads)
        ...     val_loss = validate(model, val_loader)
        ...
        ...     # Use validation loss for lr scheduling
        ...     val_scheduler.step(val_loss)
        ...
        ...     # Could also track other metrics separately
        ...     val_acc = evaluate_accuracy(model, val_loader)
        ...     print(f"Epoch {epoch}: lr={optimizer.current_lr:.6f}, "
        ...           f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    See Also
    --------
    StepLR : Fixed step-based learning rate decay
    ExponentialLR : Exponential decay
    CosineAnnealingLR : Cosine annealing schedule
    OneCycleLR : One cycle learning rate policy

    References
    ----------
    .. [1] Smith, L. N. (2017).
           "Cyclical learning rates for training neural networks."
           2017 IEEE winter conference on applications of computer vision (WACV), 464-472.
    .. [2] Loshchilov, I., & Hutter, F. (2016).
           "SGDR: Stochastic gradient descent with warm restarts."
           arXiv preprint arXiv:1608.03983.
    .. [3] PyTorch documentation on ReduceLROnPlateau.
           https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr: Union[float, List[float]] = 0,
        eps: float = 1e-8,
        last_epoch: int = 0,
    ):
        super().__init__(base_lr=base_lr, last_epoch=last_epoch)
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0")

        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.eps = eps

        # Store min_lr
        if isinstance(min_lr, list):
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr]

        self.cooldown_counter = OptimState(0)
        self.best = OptimState(jnp.inf if self.mode == 'min' else -jnp.inf)
        self.num_bad_epochs = OptimState(0)
        self.mode_worse = float('inf') if mode == 'min' else -float('inf')

    def step(self, metric: float, epoch: Optional[int] = None):
        """
        Step with metric value (JIT-compatible).

        Args:
          metric: The metric value to monitor.
          epoch: Optional epoch number.
        """
        # Handle epoch update
        if epoch is None:
            epoch = self.last_epoch.value + 1
        self.last_epoch.value = epoch

        # Convert metrics to JAX array for compatibility
        metric = jnp.asarray(metric)

        # Get current state values
        cooldown_counter = self.cooldown_counter.value
        best = self.best.value
        num_bad_epochs = self.num_bad_epochs.value

        # Check if in cooldown period
        in_cooldown = cooldown_counter > 0

        # Update cooldown counter
        new_cooldown = jnp.where(
            in_cooldown,
            cooldown_counter - 1,
            cooldown_counter
        )

        # Check if current metric is better
        is_better = self._is_better_jax(metric, best)

        # Update best value and bad epochs counter when not in cooldown
        new_best = jnp.where(
            jnp.logical_and(jnp.logical_not(in_cooldown), is_better),
            metric,
            best
        )

        new_num_bad_epochs = jnp.where(
            in_cooldown,
            num_bad_epochs,  # Don't change during cooldown
            jnp.where(
                is_better,
                0,  # Reset on improvement
                num_bad_epochs + 1  # Increment on no improvement
            )
        )

        # Check if we should reduce learning rate
        should_reduce = jnp.logical_and(
            jnp.logical_not(in_cooldown),
            new_num_bad_epochs > self.patience
        )

        # Update states
        self.cooldown_counter.value = jnp.where(
            should_reduce,
            self.cooldown,
            new_cooldown
        )
        self.best.value = new_best
        self.num_bad_epochs.value = jnp.where(
            should_reduce,
            0,  # Reset after reduction
            new_num_bad_epochs
        )

        # Conditionally reduce learning rate (JIT-compatible)
        self._reduce_lr_conditional(should_reduce)

    def _is_better(self, a, b):
        """Python version for non-JIT contexts."""
        if self.mode == 'min':
            if self.threshold_mode == 'rel':
                return a < b * (1 - self.threshold)
            else:
                return a < b - self.threshold
        else:
            if self.threshold_mode == 'rel':
                return a > b * (1 + self.threshold)
            else:
                return a > b + self.threshold

    def _is_better_jax(self, a, b):
        """JAX-compatible version of _is_better."""
        # Convert to JAX arrays
        a = jnp.asarray(a)
        b = jnp.asarray(b)

        if self.mode == 'min':
            if self.threshold_mode == 'rel':
                return a < b * (1 - self.threshold)
            else:
                return a < (b - self.threshold)
        else:
            if self.threshold_mode == 'rel':
                return a > b * (1 + self.threshold)
            else:
                return a > (b + self.threshold)

    def _reduce_lr_conditional(self, should_reduce):
        """Conditionally reduce learning rate in a JIT-compatible way."""
        # Get current learning rates as array
        current_lrs_array = jnp.array(self._current_lrs.value)

        # Create min_lrs array with proper broadcasting
        min_lrs_array = jnp.array(self.min_lrs)
        if len(min_lrs_array) == 1:
            min_lrs_array = jnp.full_like(current_lrs_array, min_lrs_array[0])
        elif len(min_lrs_array) < len(current_lrs_array):
            # Pad with the last value
            padding = jnp.full((len(current_lrs_array) - len(min_lrs_array),), min_lrs_array[-1])
            min_lrs_array = jnp.concatenate([min_lrs_array, padding])

        # Compute reduced learning rates
        reduced_lrs = jnp.maximum(
            current_lrs_array * self.factor,
            min_lrs_array
        )

        # Conditionally update using jnp.where
        new_lrs_array = jnp.where(
            should_reduce,
            reduced_lrs,
            current_lrs_array
        )

        # Update stored learning rates
        self._current_lrs.value = list(new_lrs_array)

        # Update optimizer if attached
        if self.optimizer is not None:
            for i, param_group in enumerate(self.optimizer.param_groups):
                if i < len(new_lrs_array):
                    param_group['lr'].value = new_lrs_array[i]
            # Update the main optimizer lr
            self.optimizer.current_lr = new_lrs_array[0]

    def _reduce_lr(self):
        """Direct learning rate reduction (for non-JIT contexts)."""
        # This is called when we know for sure we want to reduce
        self._reduce_lr_conditional(True)

    def get_lr(self):
        # Return current learning rates
        return list(self._current_lrs.value)


class LinearLR(LRScheduler):
    r"""Linear learning rate scheduler - Linearly scales learning rate between two factors.

    LinearLR multiplies the base learning rate by a factor that changes linearly from
    start_factor to end_factor over total_iters epochs. This is commonly used for learning
    rate warmup or cooldown phases in training.

    Parameters
    ----------
    start_factor : float, optional
        Multiplicative factor for the learning rate at the start (epoch 0).
        The initial lr will be base_lr * start_factor. Must be in range (0, 1].
        Default: 1/3.
    end_factor : float, optional
        Multiplicative factor for the learning rate at the end (after total_iters).
        The final lr will be base_lr * end_factor. Must be in range (0, 1].
        Default: 1.0.
    total_iters : int, optional
        Number of epochs over which to linearly transition from start_factor to
        end_factor. Default: 5.
    last_epoch : int, optional
        The index of the last epoch. Used for resuming training. Default: -1 (starts
        from beginning).

    Notes
    -----
    The learning rate at epoch :math:`t` is computed as:

    .. math::
        \eta_t = \begin{cases}
            \eta_0 \cdot s & \text{if } t = 0 \\
            \eta_0 \cdot e & \text{if } t > T \\
            \eta_0 \cdot \left(s + (e - s) \cdot \frac{t}{T}\right) & \text{otherwise}
        \end{cases}

    where :math:`\eta_0` is the base learning rate, :math:`s` is start_factor,
    :math:`e` is end_factor, :math:`T` is total_iters, and :math:`t` is the current epoch.

    **Key characteristics:**

    - Smooth linear transition between two learning rate values
    - Most commonly used for warmup (start_factor < end_factor)
    - Can also be used for cooldown (start_factor > end_factor)
    - Simple and predictable learning rate schedule

    **Common usage patterns:**

    - Warmup: start_factor=0.01, end_factor=1.0, total_iters=5-10
    - Cooldown: start_factor=1.0, end_factor=0.1, total_iters=10-20
    - Gradual increase: start_factor=0.1, end_factor=1.0, total_iters=100

    Examples
    --------
    **Learning rate warmup:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> # Warmup from 0.001 * 0.1 = 0.0001 to 0.001 over 10 epochs
        >>> scheduler = braintools.optim.LinearLR(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=10
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(15):
        ...     # Training code
        ...     scheduler.step()
        ...     if epoch < 11:
        ...         print(f"Epoch {epoch}: lr ≈ {optimizer.current_lr:.6f}")
        # lr gradually increases from 0.0001 to 0.001

    **Standard warmup with default parameters:**

    .. code-block:: python

        >>> # Default: warmup from base_lr/3 to base_lr over 5 epochs
        >>> scheduler = braintools.optim.LinearLR()
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # lr increases from ~0.00033 to 0.001 over 5 epochs

    **Learning rate cooldown:**

    .. code-block:: python

        >>> # Linearly decrease lr from base_lr to base_lr*0.01 over 20 epochs
        >>> scheduler = braintools.optim.LinearLR(
        ...     start_factor=1.0,
        ...     end_factor=0.01,
        ...     total_iters=20
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(30):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        # lr decreases from 0.001 to 0.00001 over first 20 epochs, then stays at 0.00001

    **Combining with StepLR for warmup + decay:**

    .. code-block:: python

        >>> # Warmup for 5 epochs, then step decay
        >>> warmup = braintools.optim.LinearLR(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> decay = braintools.optim.StepLR(base_lr=0.01, step_size=30, gamma=0.1)
        >>> scheduler = braintools.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(90):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Gradual learning rate increase:**

    .. code-block:: python

        >>> # Start with very small lr and gradually increase
        >>> scheduler = braintools.optim.LinearLR(
        ...     start_factor=0.01,
        ...     end_factor=1.0,
        ...     total_iters=100
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # lr increases from 0.00001 to 0.001 over 100 epochs

    **Fine-tuning with gentle start:**

    .. code-block:: python

        >>> # Start at 30% of base lr, reach full lr in 3 epochs
        >>> scheduler = braintools.optim.LinearLR(
        ...     start_factor=0.3,
        ...     end_factor=1.0,
        ...     total_iters=3
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(20):
        ...     finetune_epoch(model, optimizer, finetune_loader)
        ...     scheduler.step()

    See Also
    --------
    ConstantLR : Multiply learning rate by constant factor
    WarmupScheduler : Alternative warmup implementation
    ChainedScheduler : Combine multiple schedulers

    References
    ----------
    .. [1] Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L.,
           Kyrola, A., ... & He, K. (2017).
           "Accurate, large minibatch SGD: Training imagenet in 1 hour."
           arXiv preprint arXiv:1706.02677.
    .. [2] He, T., Zhang, Z., Zhang, H., Zhang, Z., Xie, J., & Li, M. (2019).
           "Bag of tricks for image classification with convolutional neural networks."
           Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
           Recognition, 558-567.
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = 0,
    ):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        # JIT-compatible conditional logic using jnp.where
        epoch = self.last_epoch.value

        # Compute the interpolation factor
        interpolation = jnp.clip(epoch / self.total_iters, 0.0, 1.0)
        factor = self.start_factor + (self.end_factor - self.start_factor) * interpolation

        return [base_lr * factor for base_lr in self.base_lrs]


class ConstantLR(LRScheduler):
    r"""Constant learning rate scheduler - Multiplies learning rate by a constant factor.

    ConstantLR multiplies the base learning rate by a constant factor for a specified
    number of epochs (total_iters), then returns to the original base learning rate.
    This is useful for implementing warmup phases or temporary learning rate adjustments.

    Parameters
    ----------
    base_lr : float or list of float, optional
        Initial learning rate(s). Can be a single float or a list of floats for multiple
        parameter groups. Default: 1e-3.
    factor : float, optional
        Multiplicative factor applied to base_lr for the first total_iters epochs.
        Must be in range (0, 1]. Default: 1/3.
    total_iters : int, optional
        Number of epochs to apply the factor. After total_iters epochs, the learning
        rate returns to base_lr. Default: 5.
    last_epoch : int, optional
        The index of the last epoch. Used for resuming training. Default: -1 (starts
        from beginning).

    Notes
    -----
    The learning rate at epoch :math:`t` is computed as:

    .. math::
        \eta_t = \begin{cases}
            \eta_0 \cdot \text{factor} & \text{if } t < \text{total_iters} \\
            \eta_0 & \text{otherwise}
        \end{cases}

    where :math:`\eta_0` is the base learning rate.

    **Key characteristics:**

    - Simple two-phase learning rate schedule
    - Commonly used for warmup with constant reduced lr
    - Automatically returns to base_lr after warmup period
    - No gradual transition (step change at total_iters)

    **Comparison with LinearLR:**

    - ConstantLR: Instant jump from (factor * base_lr) to base_lr at total_iters
    - LinearLR: Smooth linear transition from start_factor to end_factor

    Examples
    --------
    **Basic constant warmup:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> # Use 0.5 * base_lr for first 10 epochs, then full base_lr
        >>> scheduler = braintools.optim.ConstantLR(
        ...     base_lr=0.001,
        ...     factor=0.5,
        ...     total_iters=10
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Epochs 0-9:  lr = 0.0005
        >>> # Epochs 10+:  lr = 0.001

    **Default warmup configuration:**

    .. code-block:: python

        >>> # Default: lr = base_lr/3 for 5 epochs, then lr = base_lr
        >>> scheduler = braintools.optim.ConstantLR()
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(10):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        ...     print(f"Epoch {epoch}: lr = {optimizer.current_lr}")
        # First 5 epochs: lr ≈ 0.000333
        # Remaining epochs: lr = 0.001

    **Short warmup for fine-tuning:**

    .. code-block:: python

        >>> # Use 20% of base_lr for first 3 epochs
        >>> scheduler = braintools.optim.ConstantLR(
        ...     base_lr=0.0001,
        ...     factor=0.2,
        ...     total_iters=3
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Epochs 0-2:  lr = 0.00002
        >>> # Epochs 3+:   lr = 0.0001

    **Combining with StepLR:**

    .. code-block:: python

        >>> # Warmup, then step decay
        >>> warmup = braintools.optim.ConstantLR(
        ...     base_lr=0.1,
        ...     factor=0.1,
        ...     total_iters=5
        ... )
        >>> decay = braintools.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> scheduler = braintools.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(90):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        # Epochs 0-4:   lr = 0.01 (warmup)
        # Epochs 5-29:  lr = 0.1  (after warmup)
        # Epochs 30-59: lr = 0.01 (first decay)
        # Epochs 60+:   lr = 0.001 (second decay)

    **Conservative start for transfer learning:**

    .. code-block:: python

        >>> # Start with very low lr for stability
        >>> scheduler = braintools.optim.ConstantLR(
        ...     base_lr=0.001,
        ...     factor=0.01,
        ...     total_iters=10
        ... )
        >>> optimizer = braintools.optim.Adam(lr=scheduler, weight_decay=1e-5)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # First 10 epochs: lr = 0.00001 (conservative)
        >>> # Remaining epochs: lr = 0.001 (normal training)

    **Multiple parameter groups:**

    .. code-block:: python

        >>> # Different base_lr for different layers
        >>> scheduler = braintools.optim.ConstantLR(
        ...     base_lr=[0.1, 0.01],
        ...     factor=0.1,
        ...     total_iters=5
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> # Both groups use factor=0.1 for first 5 epochs

    **Complete training workflow:**

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> scheduler = braintools.optim.ConstantLR(
        ...     base_lr=0.01,
        ...     factor=0.1,
        ...     total_iters=5
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(50):
        ...     # Training step
        ...     for batch in train_loader:
        ...         loss = compute_loss(model, batch)
        ...         grads = jax.grad(compute_loss)(model.states(brainstate.ParamState))
        ...         optimizer.update(grads)
        ...
        ...     scheduler.step()
        ...     if epoch in [0, 4, 5, 10]:
        ...         print(f"Epoch {epoch}: lr = {optimizer.current_lr}")

    See Also
    --------
    LinearLR : Linearly scale learning rate (smooth transition)
    WarmupScheduler : Alternative warmup implementation
    ChainedScheduler : Combine multiple schedulers

    References
    ----------
    .. [1] Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L.,
           Kyrola, A., ... & He, K. (2017).
           "Accurate, large minibatch SGD: Training imagenet in 1 hour."
           arXiv preprint arXiv:1706.02677.
    .. [2] Smith, L. N. (2017).
           "Cyclical learning rates for training neural networks."
           2017 IEEE winter conference on applications of computer vision (WACV), 464-472.
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        factor: float = 1.0 / 3,
        total_iters: int = 5,
        last_epoch: int = 0,
    ):
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        # JIT-compatible: use jnp.where instead of if-else
        factor = jnp.where(self.last_epoch.value < self.total_iters, self.factor, 1.0)
        return [base_lr * factor for base_lr in self.base_lrs]


class ChainedScheduler(LRScheduler):
    r"""Chain multiple schedulers together - Applies multiple schedulers simultaneously.

    ChainedScheduler allows you to apply multiple learning rate schedulers at the same time.
    All schedulers are stepped together at each epoch, and their effects are combined
    multiplicatively. This is particularly useful for implementing complex learning rate
    schedules like warmup followed by decay.

    Parameters
    ----------
    schedulers : list of LRScheduler
        List of scheduler instances to chain together. All schedulers must operate on
        the same optimizer. The schedulers will be stepped in the order provided.

    Notes
    -----
    When multiple schedulers are chained:

    - Each scheduler computes its own learning rate adjustment
    - All schedulers are stepped simultaneously
    - The final learning rate is determined by the last scheduler in the chain
    - State management is handled individually for each scheduler

    **Key characteristics:**

    - Enables complex multi-phase learning rate schedules
    - Common pattern: warmup + decay
    - All schedulers share the same epoch counter
    - Useful for combining complementary scheduling strategies

    **Common patterns:**

    - Warmup + StepLR: Gradual increase followed by step decay
    - Warmup + CosineAnnealing: Linear warmup then smooth cosine decay
    - Multiple decay stages: ConstantLR + MultiStepLR

    Examples
    --------
    **Warmup followed by step decay:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Create individual schedulers
        >>> warmup = braintools.optim.LinearLR(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> decay = braintools.optim.StepLR(base_lr=0.01, step_size=30, gamma=0.1)
        >>>
        >>> # Chain them together
        >>> scheduler = braintools.optim.ChainedScheduler([warmup, decay])
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Training loop
        >>> for epoch in range(90):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        # Epochs 0-4:   warmup from 0.001 to 0.01
        # Epochs 5-29:  lr = 0.01
        # Epochs 30-59: lr = 0.001 (first decay)
        # Epochs 60+:   lr = 0.0001 (second decay)

    **Constant warmup + multi-step decay:**

    .. code-block:: python

        >>> # Start with reduced lr, then schedule decays
        >>> warmup = braintools.optim.ConstantLR(factor=0.1, total_iters=5)
        >>> decay = braintools.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[30, 60, 80],
        ...     gamma=0.1
        ... )
        >>> scheduler = braintools.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Multiple warmup phases:**

    .. code-block:: python

        >>> # Two-stage warmup
        >>> warmup1 = braintools.optim.ConstantLR(
        ...     base_lr=0.01,
        ...     factor=0.01,
        ...     total_iters=3
        ... )
        >>> warmup2 = braintools.optim.LinearLR(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=7
        ... )
        >>> scheduler = braintools.optim.ChainedScheduler([warmup1, warmup2])
        >>>
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        # Epochs 0-2:   lr = 0.0001 (constant low)
        # Epochs 3-9:   lr increases from ~0.001 to 0.01 (linear)
        # Epochs 10+:   lr = 0.01 (normal)

    **Saving and loading chained scheduler state:**

    .. code-block:: python

        >>> warmup = braintools.optim.LinearLR(start_factor=0.1, end_factor=1.0, total_iters=5)
        >>> decay = braintools.optim.StepLR(base_lr=0.01, step_size=30, gamma=0.1)
        >>> scheduler = braintools.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Train for some epochs
        >>> for epoch in range(50):
        ...     scheduler.step()
        >>>
        >>> # Save state
        >>> checkpoint = {'scheduler': scheduler.state_dict(), 'epoch': 50}
        >>>
        >>> # Later, resume training
        >>> new_warmup = braintools.optim.LinearLR(start_factor=0.1, end_factor=1.0, total_iters=5)
        >>> new_decay = braintools.optim.StepLR(base_lr=0.01, step_size=30, gamma=0.1)
        >>> new_scheduler = braintools.optim.ChainedScheduler([new_warmup, new_decay])
        >>> new_scheduler.load_state_dict(checkpoint['scheduler'])
        >>> # Continue from epoch 50

    **ImageNet-style training schedule:**

    .. code-block:: python

        >>> # Standard ImageNet: warmup + step decay
        >>> warmup = braintools.optim.LinearLR(
        ...     start_factor=0.01,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> decay = braintools.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[30, 60],
        ...     gamma=0.1
        ... )
        >>> scheduler = braintools.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = braintools.optim.SGD(
        ...     lr=scheduler,
        ...     momentum=0.9,
        ...     weight_decay=1e-4
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(90):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Fine-tuning with conservative start:**

    .. code-block:: python

        >>> # Conservative warmup for transfer learning
        >>> warmup = braintools.optim.ConstantLR(
        ...     base_lr=0.001,
        ...     factor=0.1,
        ...     total_iters=3
        ... )
        >>> decay = braintools.optim.MultiStepLR(
        ...     base_lr=0.001,
        ...     milestones=[10, 20],
        ...     gamma=0.5
        ... )
        >>> scheduler = braintools.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(30):
        ...     finetune_epoch(model, optimizer, finetune_loader)
        ...     scheduler.step()

    See Also
    --------
    SequentialLR : Switch between different schedulers at specific milestones
    LinearLR : Linear learning rate warmup/cooldown
    StepLR : Step learning rate decay
    MultiStepLR : Multi-step learning rate decay

    References
    ----------
    .. [1] Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L.,
           Kyrola, A., ... & He, K. (2017).
           "Accurate, large minibatch SGD: Training imagenet in 1 hour."
           arXiv preprint arXiv:1706.02677.
    .. [2] He, T., Zhang, Z., Zhang, H., Zhang, Z., Xie, J., & Li, M. (2019).
           "Bag of tricks for image classification with convolutional neural networks."
           Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
           Recognition, 558-567.
    """
    __module__ = 'braintools.optim'

    def __init__(self, schedulers: List[LRScheduler]):
        self.schedulers = schedulers
        super().__init__()
        self.optimizer = schedulers[0].optimizer if schedulers else None
        for sch in schedulers:
            assert isinstance(sch, LRScheduler), f'All elements must be LRScheduler, got {type(sch)}'

        # Get base_lrs from first scheduler for compatibility with attach_optimizer
        if schedulers:
            self.base_lrs = schedulers[0].base_lrs
        else:
            self.base_lrs = [1e-3]

    def attach_optimizer(self, optimizer):
        """Attach optimizer to all schedulers."""
        self.optimizer = optimizer
        for scheduler in self.schedulers:
            if isinstance(scheduler, LRScheduler):
                scheduler.attach_optimizer(optimizer)

    def step(self, *args, **kwargs):
        for scheduler in self.schedulers:
            scheduler.step(*args, **kwargs)

    def get_lr(self):
        return self.schedulers[-1].get_lr()

    def state_dict(self):
        return {
            'schedulers': [s.state_dict() for s in self.schedulers]
        }

    def load_state_dict(self, state_dict):
        for scheduler, s_dict in zip(self.schedulers, state_dict['schedulers']):
            scheduler.load_state_dict(s_dict)


class SequentialLR(LRScheduler):
    r"""Sequential learning rate scheduler - Chains multiple schedulers based on epoch milestones.

    SequentialLR allows you to chain multiple learning rate schedulers, with each scheduler
    being active during specific epoch ranges defined by milestones. This is particularly
    useful for complex training strategies that require different learning rate policies
    at different stages of training.

    Parameters
    ----------
    schedulers : List[LRScheduler]
        List of schedulers to be sequentially applied. The number of schedulers should
        be ``len(milestones) + 1``.
    milestones : List[int]
        List of epoch indices that define when to switch schedulers. Must be in
        ascending order. For n milestones, you need n+1 schedulers.
    last_epoch : int, optional
        The index of the last epoch. Default: 0.

    Notes
    -----
    **Scheduler Switching Logic:**

    Given milestones [m1, m2, ..., mn] and schedulers [s0, s1, ..., sn]:

    - Epochs [0, m1): uses scheduler s0
    - Epochs [m1, m2): uses scheduler s1
    - ...
    - Epochs [mn, ∞): uses scheduler sn

    **JIT Compatibility:**

    This implementation is JIT-compatible through the use of JAX operations for
    scheduler selection. The scheduler index is computed using ``jnp.searchsorted``
    for efficient milestone-based switching.

    **Important Considerations:**

    1. Each scheduler should be initialized with the appropriate ``base_lr`` that
       matches your intended learning rate at the transition point.
    2. The ``last_epoch`` parameter of individual schedulers is managed internally.
    3. When saving/loading state, all schedulers' states are preserved.

    Examples
    --------
    **Basic usage with warmup and decay:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> # Warmup for 5 epochs, then exponential decay
        >>> warmup = braintools.optim.LinearLR(
        ...     base_lr=0.1,
        ...     start_factor=0.01,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> decay = braintools.optim.ExponentialLR(
        ...     base_lr=0.1,
        ...     gamma=0.95
        ... )
        >>> scheduler = braintools.optim.SequentialLR(
        ...     schedulers=[warmup, decay],
        ...     milestones=[5]
        ... )
        >>>
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step(epoch)

    **Three-phase training (warmup → training → fine-tuning):**

    .. code-block:: python

        >>> # Phase 1: Warmup (epochs 0-5)
        >>> warmup = braintools.optim.LinearLR(
        ...     base_lr=0.001,
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>>
        >>> # Phase 2: Main training (epochs 5-80)
        >>> main_training = braintools.optim.CosineAnnealingLR(
        ...     base_lr=0.001,
        ...     T_max=75,
        ...     eta_min=0.0001
        ... )
        >>>
        >>> # Phase 3: Fine-tuning (epochs 80+)
        >>> fine_tuning = braintools.optim.ConstantLR(
        ...     base_lr=0.0001,
        ...     factor=1.0
        ... )
        >>>
        >>> scheduler = braintools.optim.SequentialLR(
        ...     schedulers=[warmup, main_training, fine_tuning],
        ...     milestones=[5, 80]
        ... )

    **Complex schedule for transformer training:**

    .. code-block:: python

        >>> # Transformer training schedule
        >>> # 1. Linear warmup
        >>> warmup = braintools.optim.LinearLR(
        ...     base_lr=0.0005,
        ...     start_factor=0.0,
        ...     end_factor=1.0,
        ...     total_iters=4000  # 4000 steps
        ... )
        >>>
        >>> # 2. Constant learning rate
        >>> constant = braintools.optim.ConstantLR(
        ...     base_lr=0.0005,
        ...     factor=1.0
        ... )
        >>>
        >>> # 3. Cosine decay to near zero
        >>> cosine_decay = braintools.optim.CosineAnnealingLR(
        ...     base_lr=0.0005,
        ...     T_max=20000,
        ...     eta_min=1e-6
        ... )
        >>>
        >>> scheduler = braintools.optim.SequentialLR(
        ...     schedulers=[warmup, constant, cosine_decay],
        ...     milestones=[4000, 10000]
        ... )

    **State persistence across training sessions:**

    .. code-block:: python

        >>> # Save scheduler state
        >>> scheduler = braintools.optim.SequentialLR(
        ...     schedulers=[scheduler1, scheduler2],
        ...     milestones=[50]
        ... )
        >>> # ... train for some epochs ...
        >>> checkpoint = {
        ...     'epoch': epoch,
        ...     'scheduler': scheduler.state_dict(),
        ...     'optimizer': optimizer.state_dict(),
        ... }
        >>> save(checkpoint, 'checkpoint.pkl')
        >>>
        >>> # Resume training
        >>> scheduler = braintools.optim.SequentialLR(
        ...     schedulers=[scheduler1, scheduler2],
        ...     milestones=[50]
        ... )
        >>> scheduler.load_state_dict(checkpoint['scheduler'])
        >>> # Continue training from saved epoch

    **Using with different optimizers:**

    .. code-block:: python

        >>> # Works with any optimizer
        >>> scheduler = braintools.optim.SequentialLR(
        ...     schedulers=[warmup_sched, main_sched],
        ...     milestones=[10]
        ... )
        >>>
        >>> # With SGD
        >>> sgd_opt = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>>
        >>> # With Adam
        >>> adam_opt = braintools.optim.Adam(lr=scheduler)
        >>>
        >>> # With LAMB for large batch training
        >>> lamb_opt = braintools.optim.LAMB(lr=scheduler)

    **Monitoring scheduler transitions:**

    .. code-block:: python

        >>> scheduler = braintools.optim.SequentialLR(
        ...     schedulers=[sched1, sched2, sched3],
        ...     milestones=[10, 20]
        ... )
        >>>
        >>> for epoch in range(30):
        ...     scheduler.step(epoch)
        ...     current_lr = scheduler.get_lr()
        ...     active_scheduler = scheduler.current_scheduler_idx
        ...     print(f"Epoch {epoch}: LR={current_lr[0]:.6f}, "
        ...           f"Active scheduler: {active_scheduler}")
        ...
        ...     # Detect transitions
        ...     if epoch in scheduler.milestones:
        ...         print(f"  -> Switching to scheduler {active_scheduler}")

    See Also
    --------
    ChainedScheduler : Applies multiple schedulers simultaneously
    LinearLR : Linear learning rate schedule (good for warmup)
    CosineAnnealingLR : Cosine annealing schedule
    ExponentialLR : Exponential decay schedule
    StepLR : Step-wise decay schedule

    References
    ----------
    .. [1] Goyal, P., et al. (2017).
           "Accurate, large minibatch SGD: Training ImageNet in 1 hour."
           arXiv preprint arXiv:1706.02677.
    .. [2] Liu, L., et al. (2019).
           "On the variance of the adaptive learning rate and beyond."
           International Conference on Learning Representations.
    .. [3] You, Y., et al. (2019).
           "Large batch optimization for deep learning: Training BERT in 76 minutes."
           International Conference on Learning Representations.
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        schedulers: List[LRScheduler],
        milestones: List[int],
        last_epoch: int = 0,
    ):

        # Get base_lr from first scheduler
        base_lr = schedulers[0].base_lrs if len(schedulers) else [1e-3]
        super().__init__(base_lr=base_lr, last_epoch=last_epoch)
        if len(schedulers) != len(milestones) + 1:
            raise ValueError("Number of schedulers should be len(milestones) + 1")

        self.schedulers = schedulers
        self.milestones = jnp.array(milestones)
        self.n_schedulers = len(schedulers)

        # Store current scheduler index as OptimState for JIT compatibility
        self._current_scheduler_idx = OptimState(0)
        self._update_scheduler_idx(last_epoch)

    def _update_scheduler_idx(self, epoch):
        """Update the current scheduler index in a JIT-compatible way."""
        # Create extended milestones array with infinity at the end
        milestones_extended = jnp.concatenate([self.milestones, jnp.array([1e10])])

        # Use searchsorted to find the current scheduler index
        idx = jnp.searchsorted(milestones_extended, epoch, side='right')
        self._current_scheduler_idx.value = idx

    @property
    def current_scheduler_idx(self):
        return self._current_scheduler_idx.value

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch.value + 1

        self.last_epoch.value = epoch

        # Update which scheduler to use
        self._update_scheduler_idx(epoch)

        # Step all schedulers (only the active one will actually update)
        # This is necessary for JIT compatibility since we can't index dynamically
        brainstate.transform.switch(
            self._current_scheduler_idx.value,
            [sch.step for sch in self.schedulers],
            epoch
        )

    def get_lr(self):
        """Get learning rate from the current scheduler."""
        # For JIT compatibility, we compute all LRs and select the right one
        all_lrs = []
        for i, scheduler in enumerate(self.schedulers):
            lr = scheduler.get_lr()
            all_lrs.append(lr)

        # Select the current scheduler's LR
        # In non-JIT mode, we can just index directly
        current_idx = self._current_scheduler_idx.value
        return all_lrs[current_idx]

    def state_dict(self):
        return {
            'schedulers': [s.state_dict() for s in self.schedulers],
            'milestones': self.milestones,
            'last_epoch': self.last_epoch.value,
            'current_scheduler_idx': self.current_scheduler_idx,
        }

    def load_state_dict(self, state_dict):
        self.milestones = jnp.array(state_dict['milestones'])
        self.last_epoch.value = state_dict['last_epoch']
        self._current_scheduler_idx.value = state_dict['current_scheduler_idx']
        for scheduler, s_dict in zip(self.schedulers, state_dict['schedulers']):
            scheduler.load_state_dict(s_dict)


class CosineAnnealingWarmRestarts(LRScheduler):
    r"""Cosine annealing with warm restarts - SGDR (Stochastic Gradient Descent with Warm Restarts).

    CosineAnnealingWarmRestarts implements a learning rate schedule where the learning rate
    decreases following a cosine annealing schedule, then periodically restarts from the
    initial value. This creates a series of cosine waves with potentially increasing periods,
    allowing the model to escape local minima and explore different regions of the loss landscape.

    Parameters
    ----------
    base_lr : float or list of float, optional
        Initial learning rate(s). This is the maximum learning rate at the start of each
        cosine annealing cycle. Can be a single float or a list for multiple parameter
        groups. Default: 1e-3.
    T_0 : int, optional
        Number of epochs for the first restart cycle. This defines the initial period
        before the first restart. Default: 10.
    T_mult : int, optional
        Factor by which the period increases after each restart. If T_mult=1, all cycles
        have the same length. If T_mult=2, each cycle is twice as long as the previous.
        Default: 1.
    eta_min : float, optional
        Minimum learning rate. The learning rate will never go below this value during
        the cosine annealing. Default: 0.
    last_epoch : int, optional
        The index of the last epoch. Used when resuming training. Default: 0.

    Notes
    -----
    **Mathematical Formulation:**

    Within each cosine annealing cycle, the learning rate follows:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})
                 \left(1 + \cos\left(\frac{T_{cur}}{T_i}\pi\right)\right)

    where:
    - :math:`\eta_{max}` is the base learning rate
    - :math:`\eta_{min}` is the minimum learning rate
    - :math:`T_{cur}` is the number of epochs since the last restart
    - :math:`T_i` is the current cycle length

    **Restart Schedule:**

    The cycle lengths follow the pattern:
    - First cycle: :math:`T_0` epochs
    - Second cycle: :math:`T_0 \times T_{mult}` epochs
    - Third cycle: :math:`T_0 \times T_{mult}^2` epochs
    - And so on...

    **Benefits of Warm Restarts:**

    1. **Escape Local Minima**: Periodic restarts help the optimizer escape sharp minima
    2. **Ensemble Effect**: Each restart produces a different model, creating an implicit ensemble
    3. **Fast Convergence**: Combines rapid initial progress with fine-tuning
    4. **Exploration**: Allows exploring different regions of the parameter space

    **JIT Compatibility:**

    This implementation is JIT-compatible through the use of `jnp.where` for conditional
    updates in the restart logic.

    Examples
    --------
    **Basic usage with fixed-length cycles:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> # Restart every 50 epochs with same cycle length
        >>> scheduler = braintools.optim.CosineAnnealingWarmRestarts(
        ...     base_lr=0.1,
        ...     T_0=50,
        ...     T_mult=1,  # Fixed cycle length
        ...     eta_min=0.001
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(200):
        ...     train_epoch(...)
        ...     scheduler.step()
        ...     # LR will restart at epochs 50, 100, 150

    **Increasing cycle lengths (recommended):**

    .. code-block:: python

        >>> # Cycles of increasing length: 10, 20, 40, 80, ...
        >>> scheduler = braintools.optim.CosineAnnealingWarmRestarts(
        ...     base_lr=0.1,
        ...     T_0=10,
        ...     T_mult=2,  # Double cycle length each time
        ...     eta_min=0.0001
        ... )
        >>>
        >>> # Training schedule:
        >>> # Epochs [0, 10): First cycle (10 epochs)
        >>> # Epochs [10, 30): Second cycle (20 epochs)
        >>> # Epochs [30, 70): Third cycle (40 epochs)
        >>> # And so on...

    **For transformer training:**

    .. code-block:: python

        >>> # Transformer models often benefit from warm restarts
        >>> scheduler = braintools.optim.CosineAnnealingWarmRestarts(
        ...     base_lr=0.0005,
        ...     T_0=1000,  # First cycle: 1000 steps
        ...     T_mult=2,   # Increasing cycles
        ...     eta_min=1e-6
        ... )
        >>> optimizer = braintools.optim.AdamW(lr=scheduler, weight_decay=0.01)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    **Fine-tuning with short cycles:**

    .. code-block:: python

        >>> # Fine-tuning with frequent restarts for exploration
        >>> scheduler = braintools.optim.CosineAnnealingWarmRestarts(
        ...     base_lr=0.001,  # Lower LR for fine-tuning
        ...     T_0=5,          # Short initial cycle
        ...     T_mult=1,       # Keep cycles short
        ...     eta_min=1e-5
        ... )
        >>>
        >>> # This creates rapid oscillations for better exploration
        >>> for epoch in range(50):
        ...     fine_tune_epoch(...)
        ...     scheduler.step()

    **Snapshot ensembling:**

    .. code-block:: python

        >>> # Save model at each restart for ensemble
        >>> scheduler = braintools.optim.CosineAnnealingWarmRestarts(
        ...     base_lr=0.1,
        ...     T_0=25,
        ...     T_mult=1,
        ...     eta_min=0
        ... )
        >>>
        >>> snapshots = []
        >>> for epoch in range(100):
        ...     train_epoch(...)
        ...     scheduler.step()
        ...
        ...     # Save snapshot at minimum LR (just before restart)
        ...     if scheduler.T_cur.value == scheduler.T_i.value - 1:
        ...         snapshot = copy.deepcopy(model.state_dict())
        ...         snapshots.append(snapshot)
        ...         print(f"Saved snapshot at epoch {epoch}")

    **Monitoring restarts:**

    .. code-block:: python

        >>> scheduler = braintools.optim.CosineAnnealingWarmRestarts(
        ...     base_lr=0.1,
        ...     T_0=10,
        ...     T_mult=2,
        ...     eta_min=0.001
        ... )
        >>>
        >>> for epoch in range(100):
        ...     old_T_cur = scheduler.T_cur.value
        ...     scheduler.step()
        ...     current_lr = scheduler.get_lr()[0]
        ...
        ...     # Detect restart
        ...     if scheduler.T_cur.value < old_T_cur.value:
        ...         print(f"Restart at epoch {epoch}! LR reset to {current_lr:.6f}")
        ...
        ...     if epoch % 10 == 0:
        ...         print(f"Epoch {epoch}: LR={current_lr:.6f}, "
        ...               f"T_cur={scheduler.T_cur.value}, T_i={scheduler.T_i.value}")

    **Custom restart schedule with T_mult > 1:**

    .. code-block:: python

        >>> # Create a schedule with specific restart points
        >>> # Restarts at: 0, 100, 300, 700, 1500, ...
        >>> scheduler = braintools.optim.CosineAnnealingWarmRestarts(
        ...     base_lr=0.01,
        ...     T_0=100,    # First cycle: 100 epochs
        ...     T_mult=2,   # Each cycle doubles
        ...     eta_min=0.0001
        ... )
        >>>
        >>> # Calculate when restarts will occur
        >>> def get_restart_epochs(T_0, T_mult, n_restarts):
        ...     epochs = [0]
        ...     T_i = T_0
        ...     for i in range(n_restarts):
        ...         epochs.append(epochs[-1] + T_i)
        ...         T_i = T_i * T_mult
        ...     return epochs
        >>>
        >>> restart_epochs = get_restart_epochs(100, 2, 5)
        >>> print(f"Restarts at epochs: {restart_epochs}")
        >>> # Output: [0, 100, 300, 700, 1500, 3100]

    **Combining with other techniques:**

    .. code-block:: python

        >>> # Combine with gradient clipping and weight decay
        >>> scheduler = braintools.optim.CosineAnnealingWarmRestarts(
        ...     base_lr=0.1,
        ...     T_0=30,
        ...     T_mult=2,
        ...     eta_min=0.001
        ... )
        >>>
        >>> optimizer = braintools.optim.AdamW(
        ...     lr=scheduler,
        ...     weight_decay=1e-4,
        ...     clip_norm=1.0  # Gradient clipping
        ... )

    **State persistence for long training:**

    .. code-block:: python

        >>> # Save and restore scheduler state
        >>> scheduler = braintools.optim.CosineAnnealingWarmRestarts(
        ...     base_lr=0.1, T_0=50, T_mult=2
        ... )
        >>>
        >>> # Train for some epochs
        >>> for epoch in range(75):
        ...     scheduler.step()
        >>>
        >>> # Save state
        >>> state = {
        ...     'epoch': 75,
        ...     'scheduler': scheduler.state_dict(),
        ...     'T_cur': scheduler.T_cur,
        ...     'T_i': scheduler.T_i
        ... }
        >>>
        >>> # Later: restore and continue
        >>> new_scheduler = braintools.optim.CosineAnnealingWarmRestarts(
        ...     base_lr=0.1, T_0=50, T_mult=2
        ... )
        >>> new_scheduler.load_state_dict(state['scheduler'])
        >>> new_scheduler.T_cur = state['T_cur']
        >>> new_scheduler.T_i = state['T_i']

    See Also
    --------
    CosineAnnealingLR : Standard cosine annealing without restarts
    OneCycleLR : One cycle learning rate policy
    CyclicLR : Cyclic learning rates between bounds
    ChainedScheduler : Chain multiple schedulers together

    References
    ----------
    .. [1] Loshchilov, I., & Hutter, F. (2016).
           "SGDR: Stochastic gradient descent with warm restarts."
           International Conference on Learning Representations (ICLR 2017).
           arXiv preprint arXiv:1608.03983.
    .. [2] Huang, G., et al. (2017).
           "Snapshot ensembles: Train 1, get M for free."
           International Conference on Learning Representations.
    .. [3] Smith, L. N., & Topin, N. (2019).
           "Super-convergence: Very fast training of neural networks using large learning rates."
           Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications.
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        T_0: int = 10,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = 0,
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = OptimState(0)
        self.T_i = OptimState(T_0)
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + jnp.cos(jnp.pi * self.T_cur.value / self.T_i.value)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch.value + 1
        self.last_epoch.value = epoch

        # JIT-compatible: use jnp.where for conditional updates
        self.T_cur.value = self.T_cur.value + 1
        should_restart = self.T_cur.value >= self.T_i.value
        self.T_cur.value = jnp.where(should_restart, 0, self.T_cur.value)
        self.T_i.value = jnp.where(should_restart, self.T_i.value * self.T_mult, self.T_i.value)

        values = self.get_lr()
        if self.optimizer is not None:
            for param_group, lr in zip(self.optimizer.param_groups, values):
                param_group['lr'].value = lr
            self.optimizer.current_lr = values[0]


class WarmupCosineSchedule(LRScheduler):
    r"""Warmup + Cosine annealing schedule for smooth training transitions.

    WarmupCosineSchedule combines linear warmup with cosine annealing to create a
    smooth learning rate schedule that's particularly effective for training deep
    neural networks from scratch. The schedule starts with a low learning rate,
    linearly increases to the base rate during warmup, then follows a cosine decay
    to a minimum value.

    This scheduler is widely used in:
    - Vision Transformers (ViT) and other transformer architectures
    - Self-supervised learning (SimCLR, BYOL, MAE)
    - Large-scale distributed training
    - Fine-tuning pre-trained models

    Parameters
    ----------
    base_lr : float or list of float, optional
        Peak learning rate(s) reached at the end of warmup. This is the maximum
        learning rate in the schedule. Can be a single float or list for multiple
        parameter groups. Default: 1e-3.
    warmup_steps : int, optional
        Number of steps for the linear warmup phase. During this phase, the
        learning rate linearly increases from warmup_start_lr to base_lr.
        Default: 1000.
    total_steps : int, optional
        Total number of training steps. The cosine annealing phase spans from
        warmup_steps to total_steps. Default: 10000.
    warmup_start_lr : float, optional
        Starting learning rate for the warmup phase. Set to 0 for linear warmup
        from zero, or a small value (e.g., 1e-6) for stability. Default: 0.0.
    eta_min : float, optional
        Minimum learning rate at the end of cosine annealing. The learning rate
        will not go below this value. Default: 0.0.
    last_epoch : int, optional
        The index of the last epoch. Used when resuming training. Default: 0.

    Notes
    -----
    **Mathematical Formulation:**

    The learning rate schedule consists of two phases:

    1. **Linear Warmup Phase** (step < warmup_steps):

    .. math::
        \eta_t = \eta_{warmup_start} + \frac{t}{T_{warmup}}
                 \cdot (\eta_{base} - \eta_{warmup_start})

    2. **Cosine Annealing Phase** (step >= warmup_steps):

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{base} - \eta_{min})
                 \cdot \left(1 + \cos\left(\pi \cdot \frac{t - T_{warmup}}
                 {T_{total} - T_{warmup}}\right)\right)

    where:
    - :math:`t` is the current step
    - :math:`T_{warmup}` is the number of warmup steps
    - :math:`T_{total}` is the total number of steps
    - :math:`\eta_{base}` is the peak learning rate
    - :math:`\eta_{min}` is the minimum learning rate

    **Benefits of Warmup:**

    1. **Stability**: Prevents divergence in early training with large learning rates
    2. **Gradient Statistics**: Allows optimizer to gather statistics before full LR
    3. **Weight Initialization**: Gives random weights time to organize
    4. **Large Batch Training**: Essential for stable training with large batches

    **JIT Compatibility:**

    This implementation is fully JIT-compatible through the use of JAX operations
    and conditional selection with `jnp.where`.

    Examples
    --------
    **Basic Vision Transformer training:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> # Standard ViT training schedule
        >>> scheduler = braintools.optim.WarmupCosineSchedule(
        ...     base_lr=0.001,           # Peak learning rate
        ...     warmup_steps=10000,      # 10k warmup steps
        ...     total_steps=100000,      # 100k total steps
        ...     warmup_start_lr=1e-6,    # Start from small LR
        ...     eta_min=1e-5             # End at small LR
        ... )
        >>> optimizer = braintools.optim.AdamW(lr=scheduler, weight_decay=0.05)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for step in range(100000):
        ...     loss = train_step(...)
        ...     scheduler.step()
        ...     if step % 1000 == 0:
        ...         lr = scheduler.get_lr()[0]
        ...         print(f"Step {step}: LR = {lr:.6f}")

    **Self-supervised learning (SimCLR/BYOL style):**

    .. code-block:: python

        >>> # Self-supervised learning benefits from longer warmup
        >>> scheduler = braintools.optim.WarmupCosineSchedule(
        ...     base_lr=0.3,             # High LR for contrastive learning
        ...     warmup_steps=4000,       # ~10 epochs warmup
        ...     total_steps=40000,       # 100 epochs total
        ...     warmup_start_lr=0.0,     # Start from zero
        ...     eta_min=0.0              # Decay to zero
        ... )
        >>>
        >>> # Scale learning rate with batch size (linear scaling rule)
        >>> batch_size = 4096
        >>> base_batch_size = 256
        >>> scaled_lr = 0.3 * (batch_size / base_batch_size)
        >>> scheduler.base_lrs = [scaled_lr]

    **Fine-tuning pre-trained models:**

    .. code-block:: python

        >>> # Shorter warmup for fine-tuning
        >>> scheduler = braintools.optim.WarmupCosineSchedule(
        ...     base_lr=5e-5,            # Lower LR for fine-tuning
        ...     warmup_steps=500,        # Quick warmup
        ...     total_steps=10000,       # Shorter total training
        ...     warmup_start_lr=0.0,
        ...     eta_min=1e-6
        ... )
        >>> optimizer = braintools.optim.AdamW(lr=scheduler, weight_decay=0.01)

    **Distributed training with large batches:**

    .. code-block:: python

        >>> # Large batch training needs careful warmup
        >>> world_size = 8  # 8 GPUs
        >>> batch_per_gpu = 64
        >>> total_batch = batch_per_gpu * world_size  # 512
        >>>
        >>> # Linear scaling rule with warmup
        >>> base_lr = 0.001
        >>> scaled_lr = base_lr * (total_batch / 128)  # Scale from base batch 128
        >>>
        >>> scheduler = braintools.optim.WarmupCosineSchedule(
        ...     base_lr=scaled_lr,
        ...     warmup_steps=2000,       # Longer warmup for large batch
        ...     total_steps=50000,
        ...     warmup_start_lr=base_lr / 100,  # Start at 1% of base
        ...     eta_min=scaled_lr * 0.01
        ... )

    **MAE-style masked autoencoder training:**

    .. code-block:: python

        >>> # MAE uses specific warmup schedule
        >>> scheduler = braintools.optim.WarmupCosineSchedule(
        ...     base_lr=1.5e-4,          # Base LR for batch 4096
        ...     warmup_steps=40 * 312,   # 40 epochs of warmup
        ...     total_steps=1600 * 312,  # 1600 epochs total
        ...     warmup_start_lr=0.0,
        ...     eta_min=0.0
        ... )
        >>>
        >>> # Combined with specific optimizer settings
        >>> optimizer = braintools.optim.AdamW(
        ...     lr=scheduler,
        ...     betas=(0.9, 0.95),  # MAE-specific betas
        ...     weight_decay=0.05
        ... )

    **BERT-style transformer training:**

    .. code-block:: python

        >>> # BERT uses fraction-based warmup
        >>> total_steps = 1000000
        >>> warmup_fraction = 0.1  # 10% warmup
        >>>
        >>> scheduler = braintools.optim.WarmupCosineSchedule(
        ...     base_lr=1e-4,
        ...     warmup_steps=int(total_steps * warmup_fraction),
        ...     total_steps=total_steps,
        ...     warmup_start_lr=0.0,
        ...     eta_min=1e-5
        ... )

    **Monitoring warmup progression:**

    .. code-block:: python

        >>> scheduler = braintools.optim.WarmupCosineSchedule(
        ...     base_lr=0.001,
        ...     warmup_steps=1000,
        ...     total_steps=10000,
        ...     warmup_start_lr=1e-5,
        ...     eta_min=1e-4
        ... )
        >>>
        >>> for step in range(10000):
        ...     scheduler.step()
        ...     lr = scheduler.get_lr()[0]
        ...
        ...     # Track phase transitions
        ...     if step == 0:
        ...         print(f"Starting warmup from LR = {lr:.6f}")
        ...     elif step == scheduler.warmup_steps - 1:
        ...         print(f"Ending warmup at LR = {lr:.6f}")
        ...     elif step == scheduler.warmup_steps:
        ...         print(f"Starting cosine decay from LR = {lr:.6f}")
        ...     elif step == scheduler.total_steps - 1:
        ...         print(f"Training complete at LR = {lr:.6f}")

    **Custom warmup strategies:**

    .. code-block:: python

        >>> # Aggressive warmup (reach peak quickly)
        >>> fast_warmup = braintools.optim.WarmupCosineSchedule(
        ...     base_lr=0.01,
        ...     warmup_steps=100,        # Very short warmup
        ...     total_steps=10000,
        ...     warmup_start_lr=0.001,   # Start at 10% of peak
        ...     eta_min=0.0001
        ... )
        >>>
        >>> # Conservative warmup (gradual increase)
        >>> slow_warmup = braintools.optim.WarmupCosineSchedule(
        ...     base_lr=0.01,
        ...     warmup_steps=5000,       # 50% of training for warmup
        ...     total_steps=10000,
        ...     warmup_start_lr=1e-6,    # Start very low
        ...     eta_min=0.001
        ... )

    **Combining with gradient accumulation:**

    .. code-block:: python

        >>> # Gradient accumulation affects effective batch size
        >>> accumulation_steps = 4
        >>> per_step_batch = 32
        >>> effective_batch = per_step_batch * accumulation_steps  # 128
        >>>
        >>> # Adjust learning rate and warmup accordingly
        >>> scheduler = braintools.optim.WarmupCosineSchedule(
        ...     base_lr=0.001 * (effective_batch / 32),
        ...     warmup_steps=2000 // accumulation_steps,  # Adjust for accumulation
        ...     total_steps=50000 // accumulation_steps,
        ...     warmup_start_lr=1e-5,
        ...     eta_min=1e-5
        ... )

    **State persistence for checkpointing:**

    .. code-block:: python

        >>> # Save scheduler state
        >>> scheduler = braintools.optim.WarmupCosineSchedule(
        ...     base_lr=0.001,
        ...     warmup_steps=1000,
        ...     total_steps=10000
        ... )
        >>> # ... train for some steps ...
        >>> checkpoint = {
        ...     'step': current_step,
        ...     'scheduler': scheduler.state_dict(),
        ...     'optimizer': optimizer.state_dict(),
        ...     'model': model.state_dict()
        ... }
        >>> save(checkpoint, 'checkpoint.pkl')
        >>>
        >>> # Resume training
        >>> scheduler = braintools.optim.WarmupCosineSchedule(
        ...     base_lr=0.001,
        ...     warmup_steps=1000,
        ...     total_steps=10000
        ... )
        >>> scheduler.load_state_dict(checkpoint['scheduler'])
        >>> # Continue from saved step

    See Also
    --------
    CosineAnnealingLR : Pure cosine annealing without warmup
    LinearLR : Linear learning rate schedule (can be used for warmup)
    OneCycleLR : Another schedule combining warmup with annealing
    PolynomialLR : Polynomial decay schedule

    References
    ----------
    .. [1] Dosovitskiy, A., et al. (2020).
           "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale."
           International Conference on Learning Representations.
    .. [2] He, K., et al. (2022).
           "Masked Autoencoders Are Scalable Vision Learners."
           Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
    .. [3] Goyal, P., et al. (2017).
           "Accurate, large minibatch SGD: Training ImageNet in 1 hour."
           arXiv preprint arXiv:1706.02677.
    .. [4] Chen, T., et al. (2020).
           "A Simple Framework for Contrastive Learning of Visual Representations."
           International Conference on Machine Learning.
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = 0,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        # JIT-compatible: use jnp.where instead of if-else
        epoch = self.last_epoch.value
        is_warmup = epoch < self.warmup_steps

        # Warmup phase calculation
        alpha = jnp.clip(epoch / jnp.maximum(self.warmup_steps, 1), 0.0, 1.0)
        warmup_lr = self.warmup_start_lr + (jnp.array(self.base_lrs[0]) - self.warmup_start_lr) * alpha

        # Cosine annealing phase calculation
        progress = jnp.clip(
            (epoch - self.warmup_steps) / jnp.maximum(self.total_steps - self.warmup_steps, 1),
            0.0, 1.0
        )
        cosine_lr = self.eta_min + (jnp.array(self.base_lrs[0]) - self.eta_min) * \
                    (1 + jnp.cos(jnp.pi * progress)) / 2

        # Select based on phase
        lr_value = jnp.where(is_warmup, warmup_lr, cosine_lr)
        return [lr_value for _ in self.base_lrs]


class PiecewiseConstantSchedule(LRScheduler):
    r"""Piecewise constant learning rate schedule with step-wise transitions.

    PiecewiseConstantSchedule implements a learning rate schedule where the learning
    rate remains constant within specified intervals and changes abruptly at predefined
    boundaries. This creates a step function that's useful for stage-based training
    where different phases require different learning rates.

    This scheduler is particularly effective for:
    - Multi-stage training pipelines
    - Transfer learning with progressive unfreezing
    - Training with curriculum learning
    - Reproducing specific research papers with fixed schedules
    - Budget-constrained training with predetermined phases

    Parameters
    ----------
    base_lr : float or list of float, optional
        Base learning rate(s) that will be scaled by the values parameter.
        Can be a single float or list for multiple parameter groups. This serves
        as a reference that gets multiplied by the values at each stage.
        Default: 1e-3.
    boundaries : list of int, optional
        Step indices where the learning rate changes. Must be sorted in ascending
        order. The schedule will have len(boundaries) + 1 distinct phases.
        Default: [1000, 2000].
    values : list of float, optional
        Multiplicative factors for the base learning rate in each phase. Must have
        exactly len(boundaries) + 1 elements. The i-th value applies from
        boundary[i-1] to boundary[i]. Default: [1.0, 0.1, 0.01].
    last_epoch : int, optional
        The index of the last epoch. Used when resuming training. Default: 0.

    Notes
    -----
    **Mathematical Formulation:**

    The learning rate at step t is defined as:

    .. math::
        \eta_t = \eta_{base} \times v_i

    where :math:`v_i` is determined by:

    .. math::
        v_i = \begin{cases}
            \text{values}[0] & \text{if } t < \text{boundaries}[0] \\
            \text{values}[1] & \text{if } \text{boundaries}[0] \leq t < \text{boundaries}[1] \\
            ... & ... \\
            \text{values}[n] & \text{if } t \geq \text{boundaries}[n-1]
        \end{cases}

    **Schedule Structure:**

    Given boundaries [b1, b2, ..., bn] and values [v0, v1, ..., vn]:

    - Steps [0, b1): learning_rate = base_lr × v0
    - Steps [b1, b2): learning_rate = base_lr × v1
    - Steps [b2, b3): learning_rate = base_lr × v2
    - ...
    - Steps [bn, ∞): learning_rate = base_lr × vn

    **JIT Compatibility:**

    This implementation is JIT-compatible through the use of `jnp.searchsorted`
    for efficient boundary-based value selection without Python conditionals.

    Examples
    --------
    **Classic ImageNet training schedule:**

    .. code-block:: python

        >>> import braintools
        >>> import brainstate
        >>>
        >>> # ResNet50 on ImageNet schedule
        >>> scheduler = braintools.optim.PiecewiseConstantSchedule(
        ...     base_lr=0.1,
        ...     boundaries=[30, 60, 80],  # Epochs to decrease LR
        ...     values=[1.0, 0.1, 0.01, 0.001]  # LR multipliers
        ... )
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Train for 90 epochs total
        >>> for epoch in range(90):
        ...     train_epoch(...)
        ...     scheduler.step()
        ...     lr = scheduler.get_lr()[0]
        ...     print(f"Epoch {epoch}: LR = {lr:.6f}")
        ...     # LR: 0.1 (epochs 0-29), 0.01 (30-59), 0.001 (60-79), 0.0001 (80-89)

    **Transfer learning with progressive unfreezing:**

    .. code-block:: python

        >>> # Unfreeze layers progressively with different LRs
        >>> scheduler = braintools.optim.PiecewiseConstantSchedule(
        ...     base_lr=1e-3,
        ...     boundaries=[5, 10, 15],  # Unfreeze stages
        ...     values=[0.01, 0.1, 0.5, 1.0]  # Gradual increase
        ... )
        >>>
        >>> # Stage 1 (0-4): Only train head, very low LR
        >>> # Stage 2 (5-9): Unfreeze top layers
        >>> # Stage 3 (10-14): Unfreeze middle layers
        >>> # Stage 4 (15+): Full model training
        >>>
        >>> for epoch in range(20):
        ...     if epoch == 5:
        ...         unfreeze_top_layers(model)
        ...     elif epoch == 10:
        ...         unfreeze_middle_layers(model)
        ...     elif epoch == 15:
        ...         unfreeze_all_layers(model)
        ...
        ...     train_epoch(...)
        ...     scheduler.step()

    **Budget-aware training schedule:**

    .. code-block:: python

        >>> # Training with computational budget constraints
        >>> # Fast initial training, then careful fine-tuning
        >>> scheduler = braintools.optim.PiecewiseConstantSchedule(
        ...     base_lr=0.01,
        ...     boundaries=[100, 500, 800],
        ...     values=[10.0, 1.0, 0.1, 0.01]  # Aggressive start, careful end
        ... )
        >>>
        >>> # Steps 0-99: Fast exploration (LR=0.1)
        >>> # Steps 100-499: Normal training (LR=0.01)
        >>> # Steps 500-799: Fine-tuning (LR=0.001)
        >>> # Steps 800+: Final refinement (LR=0.0001)

    **Multi-phase curriculum learning:**

    .. code-block:: python

        >>> # Different learning rates for different curriculum stages
        >>> scheduler = braintools.optim.PiecewiseConstantSchedule(
        ...     base_lr=1e-3,
        ...     boundaries=[1000, 3000, 6000, 9000],
        ...     values=[0.1, 0.5, 1.0, 0.5, 0.1]
        ... )
        >>>
        >>> curriculum_difficulties = [0.2, 0.4, 0.6, 0.8, 1.0]
        >>>
        >>> for step in range(10000):
        ...     # Determine curriculum difficulty
        ...     stage = sum(step >= b for b in scheduler.boundaries)
        ...     difficulty = curriculum_difficulties[stage]
        ...
        ...     # Train with appropriate difficulty
        ...     batch = get_curriculum_batch(difficulty)
        ...     train_step(batch)
        ...     scheduler.step()

    **Reproducing paper schedules:**

    .. code-block:: python

        >>> # WideResNet schedule from paper
        >>> scheduler = braintools.optim.PiecewiseConstantSchedule(
        ...     base_lr=0.1,
        ...     boundaries=[60, 120, 160],  # Specific to WideResNet
        ...     values=[1.0, 0.2, 0.04, 0.008]
        ... )
        >>>
        >>> # CIFAR training schedule from "Bag of Tricks"
        >>> scheduler = braintools.optim.PiecewiseConstantSchedule(
        ...     base_lr=0.1,
        ...     boundaries=[150, 250],
        ...     values=[1.0, 0.1, 0.01]
        ... )

    **Step-based (not epoch-based) scheduling:**

    .. code-block:: python

        >>> # Define boundaries in terms of training steps
        >>> steps_per_epoch = len(train_loader)
        >>> epoch_boundaries = [30, 60, 80]  # Desired epoch boundaries
        >>> step_boundaries = [e * steps_per_epoch for e in epoch_boundaries]
        >>>
        >>> scheduler = braintools.optim.PiecewiseConstantSchedule(
        ...     base_lr=0.1,
        ...     boundaries=step_boundaries,
        ...     values=[1.0, 0.1, 0.01, 0.001]
        ... )
        >>>
        >>> global_step = 0
        >>> for epoch in range(90):
        ...     for batch in train_loader:
        ...         train_step(batch)
        ...         scheduler.step(global_step)
        ...         global_step += 1

    **Combining with warmup:**

    .. code-block:: python

        >>> # Add warmup to piecewise schedule
        >>> warmup_steps = 500
        >>> main_boundaries = [5000, 10000, 15000]
        >>>
        >>> # Combine warmup with main schedule
        >>> all_boundaries = [warmup_steps] + main_boundaries
        >>> all_values = [0.01, 1.0, 0.1, 0.01, 0.001]  # Low start for warmup
        >>>
        >>> scheduler = braintools.optim.PiecewiseConstantSchedule(
        ...     base_lr=0.01,
        ...     boundaries=all_boundaries,
        ...     values=all_values
        ... )

    **Dynamic monitoring and adjustment:**

    .. code-block:: python

        >>> scheduler = braintools.optim.PiecewiseConstantSchedule(
        ...     base_lr=0.1,
        ...     boundaries=[1000, 2000, 3000],
        ...     values=[1.0, 0.5, 0.1, 0.01]
        ... )
        >>>
        >>> for step in range(4000):
        ...     old_lr = scheduler.get_lr()[0]
        ...     scheduler.step()
        ...     new_lr = scheduler.get_lr()[0]
        ...
        ...     # Detect LR changes
        ...     if old_lr != new_lr:
        ...         print(f"Step {step}: LR changed from {old_lr:.6f} to {new_lr:.6f}")
        ...         # Optionally reset momentum or other optimizer states
        ...         reset_optimizer_momentum(optimizer)
        ...
        ...     if step % 100 == 0:
        ...         print(f"Step {step}: LR = {new_lr:.6f}")

    **Research experimentation with multiple schedules:**

    .. code-block:: python

        >>> # Compare different decay strategies
        >>> schedules = {
        ...     'aggressive': braintools.optim.PiecewiseConstantSchedule(
        ...         base_lr=0.1,
        ...         boundaries=[10, 20],
        ...         values=[1.0, 0.01, 0.0001]
        ...     ),
        ...     'conservative': braintools.optim.PiecewiseConstantSchedule(
        ...         base_lr=0.1,
        ...         boundaries=[30, 60],
        ...         values=[1.0, 0.5, 0.1]
        ...     ),
        ...     'multi_stage': braintools.optim.PiecewiseConstantSchedule(
        ...         base_lr=0.1,
        ...         boundaries=[10, 20, 30, 40],
        ...         values=[1.0, 0.8, 0.4, 0.1, 0.01]
        ...     )
        ... }
        >>>
        >>> # Run experiments
        >>> for name, scheduler in schedules.items():
        ...     print(f"Testing schedule: {name}")
        ...     model = create_model()
        ...     optimizer = braintools.optim.SGD(lr=scheduler)
        ...     results = train_model(model, optimizer)
        ...     log_results(name, results)

    **State persistence and checkpointing:**

    .. code-block:: python

        >>> # Save and restore schedule state
        >>> scheduler = braintools.optim.PiecewiseConstantSchedule(
        ...     base_lr=0.1,
        ...     boundaries=[1000, 2000],
        ...     values=[1.0, 0.1, 0.01]
        ... )
        >>>
        >>> # Train for some steps
        >>> for step in range(1500):
        ...     train_step(...)
        ...     scheduler.step()
        >>>
        >>> # Save checkpoint
        >>> checkpoint = {
        ...     'step': 1500,
        ...     'scheduler_state': scheduler.state_dict(),
        ...     'model_state': model.state_dict()
        ... }
        >>> save(checkpoint, 'checkpoint.pkl')
        >>>
        >>> # Later: restore and continue
        >>> scheduler = braintools.optim.PiecewiseConstantSchedule(
        ...     base_lr=0.1,
        ...     boundaries=[1000, 2000],
        ...     values=[1.0, 0.1, 0.01]
        ... )
        >>> scheduler.load_state_dict(checkpoint['scheduler_state'])
        >>> # Continue from step 1500 with correct LR

    See Also
    --------
    StepLR : Exponential decay at regular intervals
    MultiStepLR : Similar concept with multiplicative decay
    CosineAnnealingLR : Smooth transitions instead of step changes
    SequentialLR : Chain multiple schedulers sequentially

    References
    ----------
    .. [1] He, K., et al. (2016).
           "Deep Residual Learning for Image Recognition."
           Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
    .. [2] Zagoruyko, S., & Komodakis, N. (2016).
           "Wide Residual Networks."
           British Machine Vision Conference.
    .. [3] Smith, L. N. (2017).
           "Cyclical Learning Rates for Training Neural Networks."
           IEEE Winter Conference on Applications of Computer Vision.
    .. [4] He, T., et al. (2019).
           "Bag of Tricks for Image Classification with Convolutional Neural Networks."
           Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        boundaries: List[int] = None,
        values: List[float] = None,
        last_epoch: int = 0,
    ):
        if boundaries is None:
            boundaries = [1000, 2000]
        if values is None:
            values = [1.0, 0.1, 0.01]

        if len(boundaries) != len(values) - 1:
            raise ValueError("boundaries must have one less element than values")

        self.boundaries = boundaries
        self.values = jnp.array(values)
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        # JIT-compatible: use jnp.searchsorted to find the appropriate value
        # searchsorted returns the index where epoch would be inserted to maintain order
        epoch = self.last_epoch.value
        boundaries_array = jnp.array(self.boundaries)

        # Find which segment we're in
        idx = jnp.searchsorted(boundaries_array, epoch, side='right')
        value = self.values[idx]

        return [value for _ in self.base_lrs]
