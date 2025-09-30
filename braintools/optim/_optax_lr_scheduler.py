# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from __future__ import annotations

from typing import Dict, Optional, Union, Callable, Any, List, Sequence

import jax.numpy as jnp
from brainstate import LongTermState

__all__ = [
    # Learning Rate Schedulers
    'LRScheduler',
    'StepLR',
    'MultiStepLR',
    'ExponentialLR',
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

    def __init__(self, base_lr: Union[float, List[float]] = 1e-3, last_epoch: int = 0):
        """
        Initialize the scheduler.

        Args:
          base_lr: Base learning rate(s). Can be a float or list of floats for multiple param groups.
          last_epoch: The index of the last epoch.
        """
        self.optimizer = None  # Will be set when attached to optimizer
        self.last_epoch = LongTermState(last_epoch)

        # Support both single lr and multiple lrs for param groups
        if isinstance(base_lr, (list, tuple)):
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr]

        # Current learning rates
        self._current_lrs = LongTermState(list(self.base_lrs))

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

        values = self.get_lr()
        if not isinstance(values, (list, tuple)):
            values = [values]

        self.current_lrs.value = list(values)

        # If attached to update its learning rates
        if self.optimizer is not None:
            for param_group, lr in zip(self.optimizer.param_groups, values):
                if isinstance(param_group.get('lr'), LongTermState):
                    param_group['lr'].value = lr
                else:
                    param_group['lr'] = lr

            # Update the main optimizer lr
            self.optimizer.current_lr = values[0]

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
        \eta_t = \eta_0 \cdot \gamma^{\lfloor t / \text{step\_size} \rfloor}

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

        >>> import braintools as bts
        >>> import brainstate as bst
        >>>
        >>> # Create model and scheduler
        >>> model = bst.nn.Linear(10, 5)
        >>> scheduler = bts.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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

        >>> scheduler = bts.optim.StepLR(base_lr=0.001, step_size=10, gamma=0.5)
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(25):
        ...     # Training step
        ...     scheduler.step()
        # lr decays: 0.001 -> 0.0005 (epoch 10) -> 0.00025 (epoch 20)

    **Custom decay schedule:**

    .. code-block:: python

        >>> # Aggressive decay every 5 epochs
        >>> scheduler = bts.optim.StepLR(base_lr=0.1, step_size=5, gamma=0.5)
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> # After 15 epochs: lr = 0.1 * 0.5^3 = 0.0125

    **Saving and loading scheduler state:**

    .. code-block:: python

        >>> scheduler = bts.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
        >>> new_scheduler = bts.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> new_scheduler.load_state_dict(checkpoint['scheduler'])
        >>> # Continue from epoch 50

    **Multiple parameter groups:**

    .. code-block:: python

        >>> # Different learning rates for different layers
        >>> scheduler = bts.optim.StepLR(
        ...     base_lr=[0.1, 0.01],  # Different base lr for each group
        ...     step_size=30,
        ...     gamma=0.1
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> # Both groups decay by gamma every step_size epochs

    **Complete training example:**

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> model = bst.nn.Linear(10, 5)
        >>> scheduler = bts.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> def train_epoch(model, optimizer, data):
        ...     def loss_fn(params):
        ...         # Compute loss
        ...         return loss
        ...     grads = jax.grad(loss_fn)(model.states(bst.ParamState))
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
        return [
            base_lr * self.gamma ** (self.last_epoch.value // self.step_size)
            for base_lr in self.base_lrs
        ]


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

        >>> import braintools as bts
        >>> import brainstate as bst
        >>>
        >>> model = bst.nn.Linear(10, 5)
        >>> scheduler = bts.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[30, 80],
        ...     gamma=0.1
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> # lr schedule:
        >>> # epochs 0-29:  lr = 0.1
        >>> # epochs 30-79: lr = 0.01  (after 1st milestone)
        >>> # epochs 80+:   lr = 0.001 (after 2nd milestone)

    **Using with Adam for fine-tuning:**

    .. code-block:: python

        >>> scheduler = bts.optim.MultiStepLR(
        ...     base_lr=0.001,
        ...     milestones=[10, 20, 30],
        ...     gamma=0.5
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(40):
        ...     # Training code
        ...     scheduler.step()
        # lr: 0.001 -> 0.0005 (epoch 10) -> 0.00025 (epoch 20) -> 0.000125 (epoch 30)

    **ImageNet-style training schedule:**

    .. code-block:: python

        >>> # Standard ImageNet schedule: 90 epochs with drops at 30 and 60
        >>> scheduler = bts.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[30, 60],
        ...     gamma=0.1
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(90):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        ...     print(f"Epoch {epoch}: lr = {optimizer.current_lr}")

    **CIFAR training schedule:**

    .. code-block:: python

        >>> # CIFAR-10/100 schedule: 200 epochs
        >>> scheduler = bts.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[60, 120, 160],
        ...     gamma=0.2
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=5e-4)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(200):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Custom aggressive decay schedule:**

    .. code-block:: python

        >>> # Frequent drops for quick convergence
        >>> scheduler = bts.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[5, 10, 15, 20, 25],
        ...     gamma=0.5
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> # lr rapidly decreases at each milestone

    **Resuming training with state dict:**

    .. code-block:: python

        >>> # Save training state
        >>> scheduler = bts.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[30, 60, 90],
        ...     gamma=0.1
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(50):
        ...     scheduler.step()
        >>>
        >>> checkpoint = {'scheduler': scheduler.state_dict(), 'epoch': 50}
        >>>
        >>> # Resume later
        >>> new_scheduler = bts.optim.MultiStepLR(
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

        >>> import braintools as bts
        >>> import brainstate as bst
        >>>
        >>> model = bst.nn.Linear(10, 5)
        >>> # Decay by 0.95 each epoch
        >>> scheduler = bts.optim.ExponentialLR(base_lr=0.1, gamma=0.95)
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
        >>> scheduler = bts.optim.ExponentialLR(base_lr=0.001, gamma=0.99)
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     finetune_epoch(model, optimizer, finetune_loader)
        ...     scheduler.step()
        # After 100 epochs: lr ≈ 0.001 * 0.99^100 ≈ 0.000366

    **Moderate decay for standard training:**

    .. code-block:: python

        >>> # Moderate decay with gamma=0.96
        >>> scheduler = bts.optim.ExponentialLR(base_lr=0.1, gamma=0.96)
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(50):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        # lr smoothly decreases from 0.1 to ~0.013

    **Combining with warmup:**

    .. code-block:: python

        >>> # Warmup followed by exponential decay
        >>> warmup = bts.optim.LinearLR(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> decay = bts.optim.ExponentialLR(base_lr=0.01, gamma=0.95)
        >>> scheduler = bts.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Using with different optimizers:**

    .. code-block:: python

        >>> # Works with any optimizer
        >>> scheduler = bts.optim.ExponentialLR(base_lr=0.001, gamma=0.98)
        >>>
        >>> # With Adam
        >>> adam_opt = bts.optim.Adam(lr=scheduler)
        >>> adam_opt.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> # Or with RMSprop
        >>> model2 = bst.nn.Linear(10, 5)
        >>> scheduler2 = bts.optim.ExponentialLR(base_lr=0.001, gamma=0.98)
        >>> rmsprop_opt = bts.optim.RMSprop(lr=scheduler2)
        >>> rmsprop_opt.register_trainable_weights(model2.states(bst.ParamState))

    **Saving and loading state:**

    .. code-block:: python

        >>> scheduler = bts.optim.ExponentialLR(base_lr=0.1, gamma=0.95)
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
        >>> new_scheduler = bts.optim.ExponentialLR(base_lr=0.1, gamma=0.95)
        >>> new_scheduler.load_state_dict(checkpoint['scheduler'])
        >>> # lr will be correctly set to 0.1 * 0.95^50

    **Aggressive decay:**

    .. code-block:: python

        >>> # Fast decay with gamma=0.9
        >>> scheduler = bts.optim.ExponentialLR(base_lr=0.1, gamma=0.9)
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        gamma: float = 0.95,
        last_epoch: int = 0,
    ):
        self.gamma = gamma
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** self.last_epoch.value
                for base_lr in self.base_lrs]


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

        >>> import braintools as bts
        >>> import brainstate as bst
        >>>
        >>> model = bst.nn.Linear(10, 5)
        >>> # Anneal from 0.1 to 0 over 100 epochs
        >>> scheduler = bts.optim.CosineAnnealingLR(
        ...     base_lr=0.1,
        ...     T_max=100,
        ...     eta_min=0
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
        >>> scheduler = bts.optim.CosineAnnealingLR(
        ...     base_lr=0.01,
        ...     T_max=50,
        ...     eta_min=0.0001
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(50):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Combined with warmup (recommended):**

    .. code-block:: python

        >>> # Warmup for 5 epochs, then cosine decay
        >>> warmup = bts.optim.LinearLR(
        ...     start_factor=0.01,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> cosine = bts.optim.CosineAnnealingLR(
        ...     base_lr=0.1,
        ...     T_max=90,
        ...     eta_min=0
        ... )
        >>> scheduler = bts.optim.ChainedScheduler([warmup, cosine])
        >>>
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(95):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **CIFAR-10/100 training schedule:**

    .. code-block:: python

        >>> # Standard CIFAR schedule: 200 epochs with cosine decay
        >>> scheduler = bts.optim.CosineAnnealingLR(
        ...     base_lr=0.1,
        ...     T_max=200,
        ...     eta_min=0
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=5e-4)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(200):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **ImageNet training with cosine decay:**

    .. code-block:: python

        >>> # ImageNet: 90 epochs with warmup + cosine
        >>> warmup = bts.optim.LinearLR(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> cosine = bts.optim.CosineAnnealingLR(
        ...     base_lr=0.1,
        ...     T_max=85,
        ...     eta_min=0
        ... )
        >>> scheduler = bts.optim.ChainedScheduler([warmup, cosine])
        >>>
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(90):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Fine-tuning with gentle cosine decay:**

    .. code-block:: python

        >>> # Gentle decay for fine-tuning: min lr = 10% of base lr
        >>> scheduler = bts.optim.CosineAnnealingLR(
        ...     base_lr=0.0001,
        ...     T_max=30,
        ...     eta_min=0.00001
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler, weight_decay=1e-5)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(30):
        ...     finetune_epoch(model, optimizer, finetune_loader)
        ...     scheduler.step()

    **Saving and loading state:**

    .. code-block:: python

        >>> scheduler = bts.optim.CosineAnnealingLR(
        ...     base_lr=0.1,
        ...     T_max=100,
        ...     eta_min=0
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
        >>> new_scheduler = bts.optim.CosineAnnealingLR(
        ...     base_lr=0.1,
        ...     T_max=100,
        ...     eta_min=0
        ... )
        >>> new_scheduler.load_state_dict(checkpoint['scheduler'])
        >>> # Continue from epoch 50 with correct lr

    **Vision Transformer training:**

    .. code-block:: python

        >>> # ViT training schedule
        >>> warmup = bts.optim.LinearLR(
        ...     start_factor=0.001,
        ...     end_factor=1.0,
        ...     total_iters=10
        ... )
        >>> cosine = bts.optim.CosineAnnealingLR(
        ...     base_lr=0.001,
        ...     T_max=290,
        ...     eta_min=1e-6
        ... )
        >>> scheduler = bts.optim.ChainedScheduler([warmup, cosine])
        >>>
        >>> optimizer = bts.optim.AdamW(lr=scheduler, weight_decay=0.05)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + jnp.cos(jnp.pi * epoch / self.T_max)) / 2
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

        >>> import braintools as bts
        >>> import brainstate as bst
        >>>
        >>> model = bst.nn.Linear(10, 5)
        >>> # Linear decay from 0.1 to 0 over 100 epochs
        >>> scheduler = bts.optim.PolynomialLR(
        ...     base_lr=0.1,
        ...     total_iters=100,
        ...     power=1.0
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        # lr decreases linearly: 0.1, 0.099, 0.098, ..., 0.001, 0

    **Quadratic decay (power=2.0):**

    .. code-block:: python

        >>> # Slower initial decay, faster later decay
        >>> scheduler = bts.optim.PolynomialLR(
        ...     base_lr=0.1,
        ...     total_iters=100,
        ...     power=2.0
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        # lr: epoch 25 ≈ 0.056, epoch 50 ≈ 0.025, epoch 75 ≈ 0.006

    **Square root decay (power=0.5):**

    .. code-block:: python

        >>> # Faster initial decay, slower later decay
        >>> scheduler = bts.optim.PolynomialLR(
        ...     base_lr=0.01,
        ...     total_iters=50,
        ...     power=0.5
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(50):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Semantic segmentation training (DeepLab style):**

    .. code-block:: python

        >>> # Common setup for semantic segmentation
        >>> scheduler = bts.optim.PolynomialLR(
        ...     base_lr=0.007,
        ...     total_iters=30000,  # Iterations, not epochs
        ...     power=0.9
        ... )
        >>> optimizer = bts.optim.SGD(
        ...     lr=scheduler,
        ...     momentum=0.9,
        ...     weight_decay=5e-4
        ... )
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for iteration in range(30000):
        ...     train_step(model, optimizer, batch)
        ...     scheduler.step()

    **Short training with steep decay:**

    .. code-block:: python

        >>> # Quick decay for fine-tuning
        >>> scheduler = bts.optim.PolynomialLR(
        ...     base_lr=0.001,
        ...     total_iters=10,
        ...     power=1.0
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler, weight_decay=1e-5)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(10):
        ...     finetune_epoch(model, optimizer, finetune_loader)
        ...     scheduler.step()

    **With warmup:**

    .. code-block:: python

        >>> # Warmup followed by polynomial decay
        >>> warmup = bts.optim.LinearLR(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> poly_decay = bts.optim.PolynomialLR(
        ...     base_lr=0.01,
        ...     total_iters=95,
        ...     power=0.9
        ... )
        >>> scheduler = bts.optim.ChainedScheduler([warmup, poly_decay])
        >>>
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **State persistence:**

    .. code-block:: python

        >>> scheduler = bts.optim.PolynomialLR(
        ...     base_lr=0.1,
        ...     total_iters=100,
        ...     power=2.0
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
        >>> new_scheduler = bts.optim.PolynomialLR(
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
        ...     scheduler = bts.optim.PolynomialLR(
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

        >>> import braintools as bts
        >>> import brainstate as bst
        >>>
        >>> model = bst.nn.Linear(10, 5)
        >>> # Warmup from 0 to 0.1 over 10 epochs
        >>> scheduler = bts.optim.WarmupScheduler(
        ...     base_lr=0.1,
        ...     warmup_epochs=10,
        ...     warmup_start_lr=0.0
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(50):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        # Epochs 0-9: lr increases linearly from 0 to 0.1
        # Epochs 10+: lr stays at 0.1

    **Warmup with non-zero start:**

    .. code-block:: python

        >>> # Start from 10% of target lr
        >>> scheduler = bts.optim.WarmupScheduler(
        ...     base_lr=0.01,
        ...     warmup_epochs=5,
        ...     warmup_start_lr=0.001
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(30):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Large batch training:**

    .. code-block:: python

        >>> # Warmup for large batch size (1024+)
        >>> scheduler = bts.optim.WarmupScheduler(
        ...     base_lr=0.4,  # Linear scaling rule: 0.1 * (batch_size / 256)
        ...     warmup_epochs=20,
        ...     warmup_start_lr=0.0
        ... )
        >>> optimizer = bts.optim.SGD(
        ...     lr=scheduler,
        ...     momentum=0.9,
        ...     weight_decay=1e-4
        ... )
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     train_epoch(model, optimizer, large_batch_loader)
        ...     scheduler.step()

    **Transformer training warmup:**

    .. code-block:: python

        >>> # BERT-style warmup
        >>> scheduler = bts.optim.WarmupScheduler(
        ...     base_lr=0.0001,
        ...     warmup_epochs=10000,  # Often in steps/iterations
        ...     warmup_start_lr=0.0
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler, weight_decay=0.01)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for step in range(100000):
        ...     train_step(model, optimizer, batch)
        ...     scheduler.step()

    **Warmup followed by decay (using ChainedScheduler):**

    .. code-block:: python

        >>> # Warmup then step decay
        >>> warmup = bts.optim.WarmupScheduler(
        ...     base_lr=0.1,
        ...     warmup_epochs=5,
        ...     warmup_start_lr=0.0
        ... )
        >>> decay = bts.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> scheduler = bts.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(90):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Short warmup for fine-tuning:**

    .. code-block:: python

        >>> # Gentle warmup for transfer learning
        >>> scheduler = bts.optim.WarmupScheduler(
        ...     base_lr=0.0001,
        ...     warmup_epochs=3,
        ...     warmup_start_lr=0.00001
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler, weight_decay=1e-5)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(20):
        ...     finetune_epoch(model, optimizer, finetune_loader)
        ...     scheduler.step()

    **Vision Transformer training:**

    .. code-block:: python

        >>> # ViT warmup schedule
        >>> warmup = bts.optim.WarmupScheduler(
        ...     base_lr=0.001,
        ...     warmup_epochs=10,
        ...     warmup_start_lr=0.0
        ... )
        >>> cosine = bts.optim.CosineAnnealingLR(
        ...     base_lr=0.001,
        ...     T_max=290,
        ...     eta_min=1e-6
        ... )
        >>> # Use sequentially: warmup first, then cosine
        >>> optimizer = bts.optim.AdamW(lr=warmup, weight_decay=0.05)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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

        >>> scheduler = bts.optim.WarmupScheduler(
        ...     base_lr=0.1,
        ...     warmup_epochs=10,
        ...     warmup_start_lr=0.0
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
        >>> new_scheduler = bts.optim.WarmupScheduler(
        ...     base_lr=0.1,
        ...     warmup_epochs=10,
        ...     warmup_start_lr=0.0
        ... )
        >>> new_scheduler.load_state_dict(checkpoint['scheduler'])

    **Comparison with LinearLR:**

    .. code-block:: python

        >>> # WarmupScheduler: lr increases then stays constant
        >>> warmup_sched = bts.optim.WarmupScheduler(
        ...     base_lr=0.1,
        ...     warmup_epochs=10,
        ...     warmup_start_lr=0.0
        ... )
        >>>
        >>> # LinearLR: lr increases then CAN decrease or stay constant
        >>> linear_sched = bts.optim.LinearLR(
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
    """Cyclic learning rate scheduler."""

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

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = max_lr - base_lr

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

            lr = base_lr + base_height * scale * jnp.maximum(0, 1 - x)
            lrs.append(lr)

        return lrs


class OneCycleLR(LRScheduler):
    """One cycle learning rate scheduler."""

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

        >>> import braintools as bts
        >>> import brainstate as bst
        >>>
        >>> model = bst.nn.Linear(10, 5)
        >>> scheduler = bts.optim.ReduceLROnPlateau(
        ...     base_lr=0.1,
        ...     mode='min',
        ...     factor=0.5,
        ...     patience=5,
        ...     min_lr=0.001
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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

        >>> scheduler = bts.optim.ReduceLROnPlateau(
        ...     base_lr=0.01,
        ...     mode='max',  # Maximize accuracy
        ...     factor=0.1,
        ...     patience=10,
        ...     threshold=0.01
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(200):
        ...     optimizer.step(grads)
        ...     val_acc = evaluate_accuracy(model, val_loader)
        ...     scheduler.step(val_acc)

    **Conservative schedule for stable training:**

    .. code-block:: python

        >>> # Reduce lr by half when no improvement for 10 epochs
        >>> scheduler = bts.optim.ReduceLROnPlateau(
        ...     base_lr=0.1,
        ...     mode='min',
        ...     factor=0.5,
        ...     patience=10,
        ...     threshold=1e-3,
        ...     cooldown=5  # Wait 5 epochs after reduction
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))

    **Aggressive schedule for quick adaptation:**

    .. code-block:: python

        >>> # Reduce lr by 90% when no improvement for 3 epochs
        >>> scheduler = bts.optim.ReduceLROnPlateau(
        ...     base_lr=0.01,
        ...     mode='min',
        ...     factor=0.1,
        ...     patience=3,
        ...     threshold=1e-4,
        ...     min_lr=1e-6
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))

    **With absolute threshold mode:**

    .. code-block:: python

        >>> # Use absolute threshold for improvement
        >>> scheduler = bts.optim.ReduceLROnPlateau(
        ...     base_lr=0.1,
        ...     mode='min',
        ...     factor=0.5,
        ...     patience=5,
        ...     threshold=0.001,
        ...     threshold_mode='abs'  # Absolute improvement threshold
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))

    **Complete training loop with early stopping:**

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> model = bst.nn.Linear(10, 5)
        >>> scheduler = bts.optim.ReduceLROnPlateau(
        ...     base_lr=0.01,
        ...     mode='min',
        ...     factor=0.5,
        ...     patience=10,
        ...     min_lr=1e-5
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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

        >>> scheduler = bts.optim.ReduceLROnPlateau(
        ...     base_lr=0.1,
        ...     mode='min',
        ...     factor=0.5,
        ...     patience=5
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
        >>> new_scheduler = bts.optim.ReduceLROnPlateau(
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
        >>> val_scheduler = bts.optim.ReduceLROnPlateau(
        ...     base_lr=0.01,
        ...     mode='min',
        ...     factor=0.5,
        ...     patience=5
        ... )
        >>> optimizer = bts.optim.Adam(lr=val_scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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

        self.cooldown_counter = LongTermState(0)
        self.best = LongTermState(jnp.inf if self.mode == 'min' else -jnp.inf)
        self.num_bad_epochs = LongTermState(0)
        self.mode_worse = float('inf') if mode == 'min' else -float('inf')

    def step(self, metrics: float, epoch: Optional[int] = None):
        """
        Step with metric value.

        Args:
          metrics: The metric value to monitor.
          epoch: Optional epoch number.
        """
        if epoch is None:
            epoch = self.last_epoch.value + 1
        self.last_epoch.value = epoch

        if self.cooldown_counter.value > 0:
            self.cooldown_counter.value -= 1
            return

        if self._is_better(metrics, self.best.value):
            self.best.value = metrics
            self.num_bad_epochs.value = 0
        else:
            self.num_bad_epochs.value += 1

        if self.num_bad_epochs.value > self.patience:
            self._reduce_lr()
            self.cooldown_counter.value = self.cooldown
            self.num_bad_epochs.value = 0

    def _is_better(self, a, b):
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

    def _reduce_lr(self):
        # Reduce current learning rates using JAX operations
        current_lrs_array = jnp.array(self.current_lrs.value)

        # Create min_lrs array with proper broadcasting
        min_lrs_array = jnp.array(self.min_lrs)
        if len(min_lrs_array) == 1:
            min_lrs_array = jnp.full_like(current_lrs_array, min_lrs_array[0])
        elif len(min_lrs_array) < len(current_lrs_array):
            # Pad with the last value
            padding = jnp.full((len(current_lrs_array) - len(min_lrs_array),), min_lrs_array[-1])
            min_lrs_array = jnp.concatenate([min_lrs_array, padding])

        # Compute new learning rates
        new_lrs_array = jnp.maximum(
            current_lrs_array * self.factor,
            min_lrs_array
        )

        # Convert back to list for storage
        self.current_lrs.value = list(new_lrs_array)

        # Update optimizer if attached
        if self.optimizer is not None:
            for i, param_group in enumerate(self.optimizer.param_groups):
                if i < len(new_lrs_array):
                    param_group['lr'] = new_lrs_array[i]
            # Update the main optimizer lr
            self.optimizer.current_lr = new_lrs_array[0]

    def get_lr(self):
        # Return current learning rates
        return list(self.current_lrs.value)


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

        >>> import braintools as bts
        >>> import brainstate as bst
        >>>
        >>> model = bst.nn.Linear(10, 5)
        >>> # Warmup from 0.001 * 0.1 = 0.0001 to 0.001 over 10 epochs
        >>> scheduler = bts.optim.LinearLR(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=10
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
        >>> scheduler = bts.optim.LinearLR()
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> # lr increases from ~0.00033 to 0.001 over 5 epochs

    **Learning rate cooldown:**

    .. code-block:: python

        >>> # Linearly decrease lr from base_lr to base_lr*0.01 over 20 epochs
        >>> scheduler = bts.optim.LinearLR(
        ...     start_factor=1.0,
        ...     end_factor=0.01,
        ...     total_iters=20
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(30):
        ...     optimizer.step(grads)
        ...     scheduler.step()
        # lr decreases from 0.001 to 0.00001 over first 20 epochs, then stays at 0.00001

    **Combining with StepLR for warmup + decay:**

    .. code-block:: python

        >>> # Warmup for 5 epochs, then step decay
        >>> warmup = bts.optim.LinearLR(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> decay = bts.optim.StepLR(base_lr=0.01, step_size=30, gamma=0.1)
        >>> scheduler = bts.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(90):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Gradual learning rate increase:**

    .. code-block:: python

        >>> # Start with very small lr and gradually increase
        >>> scheduler = bts.optim.LinearLR(
        ...     start_factor=0.01,
        ...     end_factor=1.0,
        ...     total_iters=100
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> # lr increases from 0.00001 to 0.001 over 100 epochs

    **Fine-tuning with gentle start:**

    .. code-block:: python

        >>> # Start at 30% of base lr, reach full lr in 3 epochs
        >>> scheduler = bts.optim.LinearLR(
        ...     start_factor=0.3,
        ...     end_factor=1.0,
        ...     total_iters=3
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
            \eta_0 \cdot \text{factor} & \text{if } t < \text{total\_iters} \\
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

        >>> import braintools as bts
        >>> import brainstate as bst
        >>>
        >>> model = bst.nn.Linear(10, 5)
        >>> # Use 0.5 * base_lr for first 10 epochs, then full base_lr
        >>> scheduler = bts.optim.ConstantLR(
        ...     base_lr=0.001,
        ...     factor=0.5,
        ...     total_iters=10
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> # Epochs 0-9:  lr = 0.0005
        >>> # Epochs 10+:  lr = 0.001

    **Default warmup configuration:**

    .. code-block:: python

        >>> # Default: lr = base_lr/3 for 5 epochs, then lr = base_lr
        >>> scheduler = bts.optim.ConstantLR()
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
        >>> scheduler = bts.optim.ConstantLR(
        ...     base_lr=0.0001,
        ...     factor=0.2,
        ...     total_iters=3
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> # Epochs 0-2:  lr = 0.00002
        >>> # Epochs 3+:   lr = 0.0001

    **Combining with StepLR:**

    .. code-block:: python

        >>> # Warmup, then step decay
        >>> warmup = bts.optim.ConstantLR(
        ...     base_lr=0.1,
        ...     factor=0.1,
        ...     total_iters=5
        ... )
        >>> decay = bts.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> scheduler = bts.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
        >>> scheduler = bts.optim.ConstantLR(
        ...     base_lr=0.001,
        ...     factor=0.01,
        ...     total_iters=10
        ... )
        >>> optimizer = bts.optim.Adam(lr=scheduler, weight_decay=1e-5)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> # First 10 epochs: lr = 0.00001 (conservative)
        >>> # Remaining epochs: lr = 0.001 (normal training)

    **Multiple parameter groups:**

    .. code-block:: python

        >>> # Different base_lr for different layers
        >>> scheduler = bts.optim.ConstantLR(
        ...     base_lr=[0.1, 0.01],
        ...     factor=0.1,
        ...     total_iters=5
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> # Both groups use factor=0.1 for first 5 epochs

    **Complete training workflow:**

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> model = bst.nn.Linear(10, 5)
        >>> scheduler = bts.optim.ConstantLR(
        ...     base_lr=0.01,
        ...     factor=0.1,
        ...     total_iters=5
        ... )
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(50):
        ...     # Training step
        ...     for batch in train_loader:
        ...         loss = compute_loss(model, batch)
        ...         grads = jax.grad(compute_loss)(model.states(bst.ParamState))
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

        >>> import braintools as bts
        >>> import brainstate as bst
        >>>
        >>> model = bst.nn.Linear(10, 5)
        >>>
        >>> # Create individual schedulers
        >>> warmup = bts.optim.LinearLR(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> decay = bts.optim.StepLR(base_lr=0.01, step_size=30, gamma=0.1)
        >>>
        >>> # Chain them together
        >>> scheduler = bts.optim.ChainedScheduler([warmup, decay])
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
        >>> warmup = bts.optim.ConstantLR(factor=0.1, total_iters=5)
        >>> decay = bts.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[30, 60, 80],
        ...     gamma=0.1
        ... )
        >>> scheduler = bts.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Multiple warmup phases:**

    .. code-block:: python

        >>> # Two-stage warmup
        >>> warmup1 = bts.optim.ConstantLR(
        ...     base_lr=0.01,
        ...     factor=0.01,
        ...     total_iters=3
        ... )
        >>> warmup2 = bts.optim.LinearLR(
        ...     start_factor=0.1,
        ...     end_factor=1.0,
        ...     total_iters=7
        ... )
        >>> scheduler = bts.optim.ChainedScheduler([warmup1, warmup2])
        >>>
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        # Epochs 0-2:   lr = 0.0001 (constant low)
        # Epochs 3-9:   lr increases from ~0.001 to 0.01 (linear)
        # Epochs 10+:   lr = 0.01 (normal)

    **Saving and loading chained scheduler state:**

    .. code-block:: python

        >>> warmup = bts.optim.LinearLR(start_factor=0.1, end_factor=1.0, total_iters=5)
        >>> decay = bts.optim.StepLR(base_lr=0.01, step_size=30, gamma=0.1)
        >>> scheduler = bts.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = bts.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> # Train for some epochs
        >>> for epoch in range(50):
        ...     scheduler.step()
        >>>
        >>> # Save state
        >>> checkpoint = {'scheduler': scheduler.state_dict(), 'epoch': 50}
        >>>
        >>> # Later, resume training
        >>> new_warmup = bts.optim.LinearLR(start_factor=0.1, end_factor=1.0, total_iters=5)
        >>> new_decay = bts.optim.StepLR(base_lr=0.01, step_size=30, gamma=0.1)
        >>> new_scheduler = bts.optim.ChainedScheduler([new_warmup, new_decay])
        >>> new_scheduler.load_state_dict(checkpoint['scheduler'])
        >>> # Continue from epoch 50

    **ImageNet-style training schedule:**

    .. code-block:: python

        >>> # Standard ImageNet: warmup + step decay
        >>> warmup = bts.optim.LinearLR(
        ...     start_factor=0.01,
        ...     end_factor=1.0,
        ...     total_iters=5
        ... )
        >>> decay = bts.optim.MultiStepLR(
        ...     base_lr=0.1,
        ...     milestones=[30, 60],
        ...     gamma=0.1
        ... )
        >>> scheduler = bts.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = bts.optim.SGD(
        ...     lr=scheduler,
        ...     momentum=0.9,
        ...     weight_decay=1e-4
        ... )
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
        >>>
        >>> for epoch in range(90):
        ...     optimizer.step(grads)
        ...     scheduler.step()

    **Fine-tuning with conservative start:**

    .. code-block:: python

        >>> # Conservative warmup for transfer learning
        >>> warmup = bts.optim.ConstantLR(
        ...     base_lr=0.001,
        ...     factor=0.1,
        ...     total_iters=3
        ... )
        >>> decay = bts.optim.MultiStepLR(
        ...     base_lr=0.001,
        ...     milestones=[10, 20],
        ...     gamma=0.5
        ... )
        >>> scheduler = bts.optim.ChainedScheduler([warmup, decay])
        >>>
        >>> optimizer = bts.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(bst.ParamState))
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
    """Sequential learning rate scheduler."""

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
        self.milestones = milestones

        # JIT-compatible: Find which scheduler to use using searchsorted
        milestones_array = jnp.array(milestones + [float('inf')])
        self._current_scheduler_idx = int(jnp.searchsorted(milestones_array, last_epoch, side='right'))

    @property
    def current_scheduler_idx(self):
        return self._current_scheduler_idx

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch.value + 1

        self.last_epoch.value = epoch

        # JIT-compatible: Find current scheduler index using searchsorted
        milestones_array = jnp.array(self.milestones + [float('inf')])
        self._current_scheduler_idx = int(jnp.searchsorted(milestones_array, epoch, side='right'))

        # Step the current scheduler
        self.schedulers[self._current_scheduler_idx].step(epoch)

    def get_lr(self):
        return self.schedulers[self._current_scheduler_idx].get_lr()

    def state_dict(self):
        return {
            'schedulers': [s.state_dict() for s in self.schedulers],
            'milestones': self.milestones,
            'last_epoch': self.last_epoch,
            '_current_scheduler_idx': self._current_scheduler_idx,
        }

    def load_state_dict(self, state_dict):
        self.milestones = state_dict['milestones']
        self.last_epoch = state_dict['last_epoch']
        self._current_scheduler_idx = state_dict['_current_scheduler_idx']
        for scheduler, s_dict in zip(self.schedulers, state_dict['schedulers']):
            scheduler.load_state_dict(s_dict)


class CosineAnnealingWarmRestarts(LRScheduler):
    """Cosine annealing with warm restarts."""

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
        self.T_cur = 0
        self.T_i = T_0
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + jnp.cos(jnp.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch.value + 1

        # JIT-compatible: use jnp.where for conditional updates
        self.T_cur = self.T_cur + 1
        should_restart = self.T_cur >= self.T_i
        self.T_cur = jnp.where(should_restart, 0, self.T_cur)
        self.T_i = jnp.where(should_restart, self.T_i * self.T_mult, self.T_i)

        self.last_epoch.value = epoch

        values = self.get_lr()
        if self.optimizer is not None:
            for param_group, lr in zip(self.optimizer.param_groups, values):
                param_group['lr'] = lr
            self.optimizer.current_lr = values[0]


class WarmupCosineSchedule(LRScheduler):
    """Warmup + Cosine annealing schedule."""

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
    """Piecewise constant learning rate schedule."""

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
        self.values = values
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        # JIT-compatible: use jnp.searchsorted to find the appropriate value
        # searchsorted returns the index where epoch would be inserted to maintain order
        epoch = self.last_epoch.value
        boundaries_array = jnp.array(self.boundaries)
        values_array = jnp.array(self.values)

        # Find which segment we're in
        idx = jnp.searchsorted(boundaries_array, epoch, side='right')
        value = values_array[idx]

        return [value for _ in self.base_lrs]
