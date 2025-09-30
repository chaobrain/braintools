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

    def __init__(self, base_lr: Union[float, List[float]] = 1e-3, last_epoch: int = -1):
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

        # Initialize learning rates
        if last_epoch == -1:
            self.step()

    def attach_optimizer(self, optimizer: 'OptaxOptimizer'):
        """Attach this scheduler to an optimizer."""
        from ._optax_optimizer import OptaxOptimizer
        if not isinstance(optimizer, OptaxOptimizer):
            raise TypeError(f"optimizer must be an Optaxgot {type(optimizer)}")

        self.optimizer = optimizer

        # If optimizer has param groups, ensure we have enough base_lrs
        if len(optimizer.param_groups) > len(self.base_lrs):
            # Extend base_lrs with the last value
            last_lr = self.base_lrs[-1] if self.base_lrs else optimizer._base_lr
            self.base_lrs.extend([last_lr] * (len(optimizer.param_groups) - len(self.base_lrs)))
            self._current_lrs.value.extend(
                [last_lr] * (len(optimizer.param_groups) - len(self._current_lrs.value))
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

        self._current_lrs.value = list(values)

        # If attached to update its learning rates
        if self.optimizer is not None:
            for param_group, lr in zip(self.optimizer.param_groups, values):
                if isinstance(param_group.get('lr'), LongTermState):
                    param_group['lr'].value = lr
                else:
                    param_group['lr'] = lr

            # Update the main optimizer lr
            self.optimizer.lr = values[0]

    def step_epoch(self):
        """Step the scheduler by one epoch."""
        self.step()

    def __call__(self, count):
        """Make scheduler callable for use with optax.scale_by_schedule.

        This allows the scheduler to be passed directly to the optimizer.
        """
        return -self._current_lrs.value[0] if self._current_lrs.value else -1e-3

    def state_dict(self):
        """Return scheduler state as dictionary."""
        return {
            'last_epoch': self.last_epoch.value,
            'base_lrs': self.base_lrs,
            '_current_lrs': self._current_lrs.value,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from dictionary."""
        self.last_epoch.value = state_dict['last_epoch']
        self.base_lrs = state_dict['base_lrs']
        self._current_lrs.value = state_dict.get('_current_lrs', list(self.base_lrs))


# ============================================================================
# Learning Rate Scheduler Classes
# ============================================================================

class StepLR(LRScheduler):
    """Step learning rate scheduler."""

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        step_size: int = 30,
        gamma: float = 0.1,
        last_epoch: int = -1,
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
    """Multi-step learning rate scheduler."""

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        milestones: Sequence[int] = (30, 60, 90),
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        factor = 1.0
        for milestone in self.milestones:
            if self.last_epoch.value >= milestone:
                factor *= self.gamma
            else:
                break
        return [base_lr * factor for base_lr in self.base_lrs]


class ExponentialLR(LRScheduler):
    """Exponential learning rate scheduler."""

    def __init__(
        self,
        gamma: float,
        last_epoch: int = -1,
    ):
        self.gamma = gamma
        super().__init__(last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** self.last_epoch.value
                for base_lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler."""

    def __init__(
        self,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(last_epoch)

    def get_lr(self):
        if self.last_epoch.value == 0:
            return self.base_lrs
        elif (self.last_epoch.value - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - jnp.cos(jnp.pi / self.T_max)) / 2
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        else:
            return [(1 + jnp.cos(jnp.pi * self.last_epoch.value / self.T_max)) /
                    (1 + jnp.cos(jnp.pi * (self.last_epoch.value - 1) / self.T_max)) *
                    (group['lr'] - self.eta_min) + self.eta_min
                    for group in self.optimizer.param_groups]


class PolynomialLR(LRScheduler):
    """Polynomial learning rate scheduler."""

    def __init__(
        self,
        total_iters: int = 5,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
        self.total_iters = total_iters
        self.power = power
        super().__init__(last_epoch)

    def get_lr(self):
        decay_factor = ((1 - min(self.last_epoch.value, self.total_iters) / self.total_iters)
                        ** self.power)
        return [base_lr * decay_factor for base_lr in self.base_lrs]


class WarmupScheduler(LRScheduler):
    """Warmup learning rate scheduler."""

    def __init__(
        self,
        warmup_epochs: int,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        super().__init__(last_epoch)

    def get_lr(self):
        if self.last_epoch.value < self.warmup_epochs:
            alpha = self.last_epoch.value / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        return self.base_lrs


class CyclicLR(LRScheduler):
    """Cyclic learning rate scheduler."""

    def __init__(
        self,
        optimizer: 'OptaxOptimizer',
        base_lr: Union[float, List[float]],
        max_lr: Union[float, List[float]],
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn: Optional[Callable] = None,
        scale_mode: str = 'cycle',
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer

        if isinstance(base_lr, list):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("base_lr must have len equal to param_groups")
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("max_lr must have len equal to param_groups")
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode

        super().__init__(last_epoch)

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

            lr = base_lr + base_height * scale * max(0, 1 - x)
            lrs.append(lr)

        return lrs


class OneCycleLR(LRScheduler):
    """One cycle learning rate scheduler."""

    def __init__(
        self,
        optimizer: 'OptaxOptimizer',
        max_lr: Union[float, List[float]],
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
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

        if isinstance(max_lr, list):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("max_lr must have len equal to param_groups")
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        self.base_lrs = [max_lr / div_factor for max_lr in self.max_lrs]
        self.min_lrs = [max_lr / final_div_factor for max_lr in self.max_lrs]

        super().__init__(last_epoch)

    def get_lr(self):
        step_num = self.last_epoch.value + 1

        if step_num > self.total_steps:
            return self.min_lrs

        if step_num <= self.pct_start * self.total_steps:
            # Warmup phase
            pct = step_num / (self.pct_start * self.total_steps)
            return [base_lr + pct * (max_lr - base_lr)
                    for base_lr, max_lr in zip(self.base_lrs, self.max_lrs)]
        else:
            # Annealing phase
            pct = (step_num - self.pct_start * self.total_steps) / \
                  ((1 - self.pct_start) * self.total_steps)

            if self.anneal_strategy == 'cos':
                pct = (1 + jnp.cos(jnp.pi * pct)) / 2
            elif self.anneal_strategy == 'linear':
                pct = 1 - pct
            else:
                raise ValueError(f"Unknown anneal_strategy: {self.anneal_strategy}")

            return [min_lr + pct * (max_lr - min_lr)
                    for min_lr, max_lr in zip(self.min_lrs, self.max_lrs)]


class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when a metric has stopped improving."""

    def __init__(
        self,
        optimizer: 'OptaxOptimizer',
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr: Union[float, List[float]] = 0,
        eps: float = 1e-8,
    ):
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0")

        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.eps = eps

        if isinstance(min_lr, list):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("min_lr must have len equal to param_groups")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.cooldown_counter = 0
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = float('inf') if mode == 'min' else -float('inf')
        self.last_epoch = -1

    def step(self, metrics: float, epoch: Optional[int] = None):
        """
        Step with metric value.

        Args:
          metrics: The metric value to monitor.
          epoch: Optional epoch number.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        if self.best is None:
            self.best = metrics
        elif self._is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

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
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            param_group['lr'] = new_lr

        # Update the main optimizer lr
        self.optimizer.lr = self.optimizer.param_groups[0]['lr']

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class LinearLR(LRScheduler):
    """Linear learning rate scheduler."""

    def __init__(
        self,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(last_epoch)

    def get_lr(self):
        if self.last_epoch.value == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]
        elif self.last_epoch.value > self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
        else:
            factor = self.start_factor + (self.end_factor - self.start_factor) * \
                     (self.last_epoch.value / self.total_iters)
            return [base_lr * factor for base_lr in self.base_lrs]


class ConstantLR(LRScheduler):
    """Constant learning rate scheduler."""

    def __init__(
        self,
        base_lr: Union[float, List[float]] = 1e-3,
        factor: float = 1.0 / 3,
        total_iters: int = 5,
        last_epoch: int = -1,
    ):
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(base_lr, last_epoch)

    def get_lr(self):
        if self.last_epoch.value < self.total_iters:
            return [base_lr * self.factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class ChainedScheduler:
    """Chain multiple schedulers together."""

    def __init__(self, schedulers: List[LRScheduler]):
        self.schedulers = schedulers
        self.optimizer = schedulers[0].optimizer

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


class SequentialLR:
    """Sequential learning rate scheduler."""

    def __init__(
        self,
        optimizer: 'OptaxOptimizer',
        schedulers: List[LRScheduler],
        milestones: List[int],
        last_epoch: int = -1,
    ):
        if len(schedulers) != len(milestones) + 1:
            raise ValueError("Number of schedulers should be len(milestones) + 1")

        self.optimizer = optimizer
        self.schedulers = schedulers
        self.milestones = milestones
        self.last_epoch = last_epoch
        self._current_scheduler_idx = 0

        # Find which scheduler to use
        for i, milestone in enumerate(milestones):
            if last_epoch < milestone:
                self._current_scheduler_idx = i
                break
        else:
            self._current_scheduler_idx = len(milestones)

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        # Check if we need to switch scheduler
        for i, milestone in enumerate(self.milestones):
            if self.last_epoch < milestone:
                if self._current_scheduler_idx != i:
                    self._current_scheduler_idx = i
                break
        else:
            self._current_scheduler_idx = len(self.milestones)

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
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        super().__init__(last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + jnp.cos(jnp.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch.value + 1

        self.T_cur = self.T_cur + 1
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult

        self.last_epoch = epoch

        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self.optimizer.lr = values[0]


class WarmupCosineSchedule(LRScheduler):
    """Warmup + Cosine annealing schedule."""

    def __init__(
        self,
        warmup_steps: int,
        total_steps: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(last_epoch)

    def get_lr(self):
        if self.last_epoch.value < self.warmup_steps:
            # Warmup phase
            alpha = self.last_epoch.value / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch.value - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + jnp.cos(jnp.pi * progress)) / 2
                    for base_lr in self.base_lrs]


class PiecewiseConstantSchedule(LRScheduler):
    """Piecewise constant learning rate schedule."""

    def __init__(
        self,
        boundaries: List[int],
        values: List[float],
        last_epoch: int = -1,
    ):
        if len(boundaries) != len(values) - 1:
            raise ValueError("boundaries must have one less element than values")

        self.boundaries = boundaries
        self.values = values
        super().__init__(last_epoch)

    def get_lr(self):
        for i, boundary in enumerate(self.boundaries):
            if self.last_epoch.value < boundary:
                return [self.values[i] for _ in self.base_lrs]
        return [self.values[-1] for _ in self.base_lrs]
