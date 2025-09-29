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

from __future__ import annotations

from typing import Dict, Optional, Union, Callable, Any, List, Tuple

import jax.tree
import optax
from brainstate import LongTermState, State, maybe_state
from brainstate.typing import PyTree

from braintools.file._msg_checkpoint import msgpack_from_state_dict
from ._base import Optimizer
from ._optax_lr_scheduler import LRScheduler
from ._state_uniquifier import UniqueStateManager

MaskOrFn = Optional[Union[Any, Callable]]

__all__ = [
    'OptaxOptimizer',
    # Main Optimizers
    'SGD',
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


class OptaxOptimizer(Optimizer):
    """
    Enhanced train state with support for Optax optimizers and learning rate schedulers.

    This class provides a PyTorch-like interface for JAX/Optax optimizers with support for:
    - Multiple parameter groups with different hyperparameters
    - Learning rate schedulers
    - State dict for saving/loading
    - Gradient clipping
    - Weight decay

    Example usage::

      >>> import jax
      >>> import jax.numpy as jnp
      >>> import brainstate
      >>> import braintools

      >>> class Model(brainstate.nn.Module):
      ...   def __init__(self):
      ...     super().__init__()
      ...     self.linear1 = brainstate.nn.Linear(2, 3)
      ...     self.linear2 = brainstate.nn.Linear(3, 4)
      ...   def __call__(self, x):
      ...     return self.linear2(self.linear1(x))

      >>> model = Model()
      >>> optimizer = braintools.optim.Adam(lr=1e-3, betas=(0.9, 0.999))
      >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

      >>> # With learning rate scheduler
      >>> scheduler = braintools.optim.CosineAnnealingLR(optimizer, T_max=100)

      >>> for epoch in range(100):
      ...   # Training loop
      ...   grads = compute_gradients(...)
      ...   optimizer.step(grads)
      ...   scheduler.step()

    Attributes:
      param_states: PyTree of brainstate.State objects representing trainable parameters.
      tx: An Optax gradient transformation.
      lr: Current learning rate (can be modified by schedulers).
      step_count: Number of optimization steps taken.
      param_groups: List of parameter groups with their own hyperparameters.
    """

    param_states: UniqueStateManager  # Container for PyTree of brainstate.State objects
    opt_state: Optional[LongTermState]
    step_count: LongTermState
    _base_lr: float
    _current_lr: LongTermState
    param_groups: List[Dict[str, Any]]
    param_groups_opt_states: List[LongTermState]
    _schedulers: List[LRScheduler]

    def __init__(
        self,
        tx: Optional[optax.GradientTransformation] = None,
        lr: Union[float, LRScheduler] = 1e-3,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        """
        Initialize the optimizer with enhanced features.

        Args:
          tx: An Optax gradient transformation. If None, will be created based on other parameters.
          lr: Learning rate (float) or LRScheduler instance.
          weight_decay: Weight decay (L2 penalty).
          grad_clip_norm: Maximum gradient norm for clipping.
          grad_clip_value: Maximum gradient value for clipping.
        """
        super().__init__()

        # param_states is already initialized in parent class as StateDictManager
        # which will hold our pytree of State objects

        self.param_states = UniqueStateManager()

        # Handle lr as either float or scheduler
        if isinstance(lr, LRScheduler):
            self._lr_scheduler = lr
            self._base_lr = lr.base_lrs[0] if lr.base_lrs else 1e-3
            self._current_lr = LongTermState(self._base_lr)
            lr.attach_optimizer(self)
        else:
            self._lr_scheduler = None
            self._base_lr = lr
            self._current_lr = LongTermState(lr)

        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_value = grad_clip_value
        self.step_count = LongTermState(0)
        self.param_groups = []
        self.param_groups_opt_states = []  # Changed to list
        self._schedulers = []

        if tx is not None:
            if not isinstance(tx, optax.GradientTransformation):
                raise TypeError(f"tx must be an instance of optax.GradientTransformation, got {tx}")
            self.tx = tx
        else:
            self.tx = self._create_default_tx()

        self.opt_state = None

    def _create_default_tx(self):
        """Create default gradient transformation with clipping and weight decay."""
        transforms = []

        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))

        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))

        transforms.append(optax.scale_by_adam())

        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))

        # Use a schedule function that reads from the State or scheduler
        if self._lr_scheduler is not None:
            # Use the scheduler's __call__ method
            transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        else:
            # Use a simple function that reads from the State
            def lr_schedule(count):
                return -self.lr

            transforms.append(optax.scale_by_schedule(lr_schedule))

        return optax.chain(*transforms)

    @property
    def lr(self):
        """Get current learning rate."""
        return self._current_lr.value

    @lr.setter
    def lr(self, value: float):
        """Set learning rate (will be used by schedulers)."""
        self._current_lr.value = value

    def _get_leaf_value(self, v):
        if not isinstance(v, State):
            raise TypeError(
                f"All params values must be brainstate.State, got {type(v)}"
            )
        return v.value

    def add_param_group(self, params: PyTree[State], **kwargs):
        """
        Add a parameter group with specific hyperparameters.

        Args:
            params: A pytree (dict) of brainstate.State objects.
            **kwargs: Additional hyperparameters for this group.
        """
        # Validate that params is a dict of State objects
        jax.tree.map(self._get_leaf_value, params, is_leaf=lambda x: isinstance(x, State))

        # Create UniqueStateManager for this group
        manager = UniqueStateManager()
        manager.merge_with(params)
        param_values = manager.to_dict_value()
        group_lr_state = LongTermState(kwargs.get('lr', self._base_lr))

        group = {
            'params': manager.to_dict(),
            'lr': group_lr_state,
            'weight_decay': kwargs.get('weight_decay', self.weight_decay),
        }
        group.update(kwargs)
        self.param_groups.append(group)

        # Initialize optimizer state for this param group if needed
        group_weight_decay = group['weight_decay']

        # Create group-specific transformation
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        transforms.append(optax.scale_by_adam())  # Use default Adam scaling
        if group_weight_decay > 0:
            transforms.append(optax.add_decayed_weights(group_weight_decay))

        # Use a schedule function that reads from the group's LR State
        def group_lr_schedule(count):
            return -group_lr_state.value

        transforms.append(optax.scale_by_schedule(group_lr_schedule))
        group_tx = optax.chain(*transforms)

        # Store the transformation for this group
        group['tx'] = group_tx

        # Initialize and store the optimizer state for this group
        group_opt_state = LongTermState(group_tx.init(param_values))
        self.param_groups_opt_states.append(group_opt_state)

    def register_trainable_weights(self, param_states: PyTree[State]):
        """Register trainable weights and initialize optimizer state.

        Args:
            param_states: A pytree (dict) of brainstate.State objects representing parameters.
        """
        jax.tree.map(self._get_leaf_value, param_states, is_leaf=lambda x: isinstance(x, State))

        # Update the param_states pytree (StateDictManager handles State objects)
        self.param_states.merge_with(param_states)

        # Initialize optimizer state using values from State objects
        param_values = self.param_states.to_pytree_value()
        self.opt_state = LongTermState(self.tx.init(param_values))

        # Create a default param group with all registered parameters
        # This maintains compatibility with PyTorch-like behavior
        if not self.param_groups:
            self.param_groups = [
                {
                    'params': self.param_states.to_pytree(),
                    'lr': self._base_lr,
                    'weight_decay': self.weight_decay,
                }
            ]

        return self

    def update(self, grads: Dict[str, Any]):
        """Update the model states with gradients (backward compatibility)."""
        return self.step(grads)

    def step(self, grads: Optional[Dict[str, Any]] = None, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.

        Args:
          grads: Gradients for parameters. If None, closure must be provided.
          closure: A closure that reevaluates the model and returns the loss.

        Returns:
          Optional loss value if closure is provided.
        """
        if self.opt_state is None:
            raise ValueError("register_trainable_weights must be called before step.")

        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            if closure is None:
                raise ValueError("Either grads or closure must be provided.")
            # Compute gradients using closure if needed
            # This would require additional implementation
            raise NotImplementedError("Automatic gradient computation from closure not yet implemented.")

        # Only use param_groups logic if multiple groups have been configured
        # (more than just the single default group)
        if self.param_groups and len(self.param_groups) > 1:

            # Process each parameter group separately with its own hyperparameters
            all_updates = {}
            processed_params = set()

            # First, handle the default group (index 0) which uses the main optimizer state
            if self.param_groups:
                default_group = self.param_groups[0]
                default_params = default_group['params']
                assert isinstance(default_params, dict)

                # Extract gradients and values for default group
                default_grads = {k: grads[k] for k in default_params.keys() if k in grads}
                default_param_values = {k: v.value for k, v in default_params.items()}

                if default_grads:
                    # Use the main optimizer state for the default group
                    updates, new_opt_state = self.tx.update(default_grads, self.opt_state.value, default_param_values)
                    self.opt_state.value = new_opt_state
                    all_updates.update(updates)
                    processed_params.update(default_params.keys())

            # Then handle additional parameter groups with custom hyperparameters
            for group_idx in range(1, len(self.param_groups)):
                group = self.param_groups[group_idx]
                group_params = group['params']
                assert isinstance(group_params, dict)

                # Extract gradients and values for this group (fix: create dict not set)
                group_grads = {k: grads[k] for k in group_params.keys() if k in grads}
                group_param_values = {k: v.value for k, v in group_params.items()}

                # Skip this group if no gradients are provided for its parameters
                if not group_grads:
                    continue

                # Use the pre-stored transformation for this group
                if 'tx' in group:
                    group_tx = group['tx']
                else:
                    # Fallback: create transformation if not stored (for backward compatibility)
                    group_lr = maybe_state(group.get('lr', self.lr))
                    group_weight_decay = group.get('weight_decay', self.weight_decay)

                    transforms = []
                    if self.grad_clip_norm is not None:
                        transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
                    if self.grad_clip_value is not None:
                        transforms.append(optax.clip(self.grad_clip_value))
                    transforms.append(optax.scale_by_adam())
                    if group_weight_decay > 0:
                        transforms.append(optax.add_decayed_weights(group_weight_decay))
                    transforms.append(optax.scale(-group_lr))
                    group_tx = optax.chain(*transforms)

                # Use pre-initialized group optimizer state (group_idx - 1 because we skip default group)
                opt_state_idx = group_idx - 1
                if opt_state_idx < len(self.param_groups_opt_states):
                    # Apply group-specific transformation
                    updates, new_group_opt_state = group_tx.update(
                        group_grads,
                        self.param_groups_opt_states[opt_state_idx].value,
                        group_param_values
                    )
                    self.param_groups_opt_states[opt_state_idx].value = new_group_opt_state

                    # Accumulate updates
                    all_updates.update(updates)
                    processed_params.update(group_params.keys())
                else:
                    raise ValueError(
                        f'Optimizer state for parameter group index '
                        f'{group_idx} not initialized.'
                    )

            # Handle any remaining parameters not in any param_group
            params = self.param_states.to_dict()
            param_values = self.param_states.to_dict_value()
            unprocessed_params = set(params.keys()) - processed_params

            if unprocessed_params:
                # Get gradients for unprocessed parameters
                unprocessed_grads = {k: grads[k] for k in unprocessed_params if k in grads}
                unprocessed_values = {k: param_values[k] for k in unprocessed_params}

                if unprocessed_grads:
                    # Use main optimizer for unprocessed parameters
                    updates, new_opt_state = self.tx.update(unprocessed_grads, self.opt_state.value, unprocessed_values)
                    self.opt_state.value = new_opt_state
                    all_updates.update(updates)

            # Apply all accumulated updates to parameters
            new_params = optax.apply_updates(param_values, all_updates)

            # Update parameters in the State objects
            for k in params.keys():
                if k in new_params:
                    params[k].value = new_params[k]
        else:
            # Original implementation for backward compatibility
            param_states = self.param_states.to_dict()
            param_values = self.param_states.to_dict_value()
            # Fix: create dict not set
            filtered_grads = {k: grads[k] for k in param_values.keys() if k in grads}

            # Apply gradient transformations
            updates, new_opt_state = self.tx.update(filtered_grads, self.opt_state.value, param_values)
            new_params = optax.apply_updates(param_values, updates)

            # Update parameters in the State objects
            for k in param_values.keys():
                if k in new_params:
                    param_states[k].value = new_params[k]

            # Update optimizer state
            self.opt_state.value = new_opt_state

        # Increment step counter
        self.step_count.value += 1

        return loss

    def state_dict(self):
        """
        Return the state of the optimizer as a dictionary.

        Returns:
          Dictionary containing optimizer state, step count, and hyperparameters.
        """
        # Prepare param_groups for serialization
        serializable_groups = dict()
        for i, group in enumerate(self.param_groups):
            group_dict = jax.tree.map(
                lambda x: (x.value if isinstance(x, State) else x),
                group,
                is_leaf=lambda x: isinstance(x, State)
            )
            group_dict.pop('tx', None)
            serializable_groups[str(i)] = group_dict

        state_dict = {
            'step_count': self.step_count.value,
            'lr': self.lr,
            'base_lr': self._base_lr,
            'param_groups': serializable_groups,
            'param_groups_opt_states': {
                str(i): s.value
                for i, s in enumerate(self.param_groups_opt_states)
            },
            'opt_state': self.opt_state.value
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load optimizer state from a dictionary.

        Args:
          state_dict: Dictionary containing optimizer state.
        """
        self.step_count.value = state_dict['step_count']
        self.lr = msgpack_from_state_dict(self.lr, state_dict['lr'])
        self._base_lr = state_dict['base_lr']

        # Load param_groups and restore lr_state for groups that have it
        self.param_groups = msgpack_from_state_dict(
            self.param_groups,
            state_dict['param_groups']
        )

        if 'opt_state' in state_dict:
            if self.opt_state is None:
                self.opt_state = LongTermState(state_dict['opt_state'])
            else:
                self.opt_state.value = state_dict['opt_state']

        # Load param group optimizer states
        if 'param_groups_opt_states' in state_dict:
            for i, s in enumerate(state_dict['param_groups_opt_states']):
                if i < len(self.param_groups_opt_states):
                    self.param_groups_opt_states[i].value = s
                else:
                    self.param_groups_opt_states.append(LongTermState(s))

    def add_scheduler(self, scheduler: LRScheduler):
        """Add a learning rate scheduler."""
        self._schedulers.append(scheduler)

    def get_last_lr(self) -> List[float]:
        """Get last computed learning rates from schedulers."""
        if self._schedulers:
            return self._schedulers[-1].get_last_lr()
        return [self.lr]


# Optimizer implementations

class SGD(OptaxOptimizer):
    """Stochastic Gradient Descent optimizer with optional momentum and weight decay."""

    def __init__(
        self,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        transforms = []

        if grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(grad_clip_norm))

        if grad_clip_value is not None:
            transforms.append(optax.clip(grad_clip_value))

        if momentum > 0:
            if nesterov:
                transforms.append(optax.trace(decay=momentum, nesterov=True))
            else:
                transforms.append(optax.trace(decay=momentum, nesterov=False))

        if weight_decay > 0:
            transforms.append(optax.add_decayed_weights(weight_decay))

        transforms.append(optax.scale(-lr))

        tx = optax.chain(*transforms) if transforms else optax.sgd(lr)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.momentum = momentum
        self.nesterov = nesterov


class Adam(OptaxOptimizer):
    """Adam optimizer with adaptive learning rates."""

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        # Store Adam-specific parameters
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad

        # Don't build tx here, let the base class handle it
        super().__init__(
            tx=None,  # Will be created by base class
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def _create_default_tx(self):
        """Create Adam-specific gradient transformation."""
        transforms = []

        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))

        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))

        if self.amsgrad:
            transforms.append(optax.scale_by_amsgrad(b1=self.betas[0], b2=self.betas[1], eps=self.eps))
        else:
            transforms.append(optax.scale_by_adam(b1=self.betas[0], b2=self.betas[1], eps=self.eps))

        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))

        # Use a schedule function that reads from the State or scheduler
        if hasattr(self, '_lr_scheduler') and self._lr_scheduler is not None:
            # Use the scheduler's __call__ method
            transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        else:
            # Use a simple function that reads from the State
            def lr_schedule(count):
                return -self.lr

            transforms.append(optax.scale_by_schedule(lr_schedule))

        return optax.chain(*transforms)


class AdamW(OptaxOptimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.adamw(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps, weight_decay=weight_decay)

        if grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.betas = betas
        self.eps = eps


class Adagrad(OptaxOptimizer):
    """Adagrad optimizer with adaptive learning rates."""

    def __init__(
        self,
        lr: float = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.adagrad(
            learning_rate=lr,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps
        )

        if weight_decay > 0 or grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            if weight_decay > 0:
                transforms.append(optax.add_decayed_weights(weight_decay))
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.lr_decay = lr_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps


class Adadelta(OptaxOptimizer):
    """Adadelta optimizer."""

    def __init__(
        self,
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        transforms = []

        if grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(grad_clip_norm))

        if grad_clip_value is not None:
            transforms.append(optax.clip(grad_clip_value))

        transforms.append(optax.scale_by_adadelta(rho=rho, eps=eps))

        if weight_decay > 0:
            transforms.append(optax.add_decayed_weights(weight_decay))

        transforms.append(optax.scale(-lr))

        tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.rho = rho
        self.eps = eps


class RMSprop(OptaxOptimizer):
    """RMSprop optimizer."""

    def __init__(
        self,
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.rmsprop(
            learning_rate=lr,
            decay=alpha,
            eps=eps,
            momentum=momentum,
            centered=centered
        )

        if weight_decay > 0 or grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            if weight_decay > 0:
                transforms.append(optax.add_decayed_weights(weight_decay))
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.centered = centered


class Adamax(OptaxOptimizer):
    """Adamax optimizer (variant of Adam with infinity norm)."""

    def __init__(
        self,
        lr: float = 2e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.adamax(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps)

        if weight_decay > 0 or grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            if weight_decay > 0:
                transforms.append(optax.add_decayed_weights(weight_decay))
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.betas = betas
        self.eps = eps


class Nadam(OptaxOptimizer):
    """Nadam optimizer (Adam with Nesterov momentum)."""

    def __init__(
        self,
        lr: float = 2e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 4e-3,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.nadam(
            learning_rate=lr,
            b1=betas[0],
            b2=betas[1],
            eps=eps,
            momentum_decay=momentum_decay
        )

        if weight_decay > 0 or grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            if weight_decay > 0:
                transforms.append(optax.add_decayed_weights(weight_decay))
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.betas = betas
        self.eps = eps
        self.momentum_decay = momentum_decay


class RAdam(OptaxOptimizer):
    """RAdam optimizer (Rectified Adam)."""

    def __init__(
        self,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.radam(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps)

        if weight_decay > 0 or grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            if weight_decay > 0:
                transforms.append(optax.add_decayed_weights(weight_decay))
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.betas = betas
        self.eps = eps


class Lamb(OptaxOptimizer):
    """LAMB optimizer (Layer-wise Adaptive Moments)."""

    def __init__(
        self,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.lamb(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps, weight_decay=weight_decay)

        if grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.betas = betas
        self.eps = eps


class Lars(OptaxOptimizer):
    """LARS optimizer (Large Batch Training)."""

    def __init__(
        self,
        lr: float = 1.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.lars(
            learning_rate=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coefficient=trust_coefficient
        )

        if grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.momentum = momentum
        self.trust_coefficient = trust_coefficient
        self.eps = eps


class Lookahead(OptaxOptimizer):
    """Lookahead optimizer wrapper."""

    def __init__(
        self,
        base_optimizer: optax.GradientTransformation,
        k: int = 5,
        alpha: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.lookahead(base_optimizer, slow_step_size=alpha, period=k)

        if weight_decay > 0 or grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            if weight_decay > 0:
                transforms.append(optax.add_decayed_weights(weight_decay))
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.k = k
        self.alpha = alpha


class Yogi(OptaxOptimizer):
    """Yogi optimizer (improvement over Adam)."""

    def __init__(
        self,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.yogi(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps)

        if weight_decay > 0 or grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            if weight_decay > 0:
                transforms.append(optax.add_decayed_weights(weight_decay))
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.betas = betas
        self.eps = eps


class LBFGS(OptaxOptimizer):
    """L-BFGS optimizer (Limited-memory BFGS)."""

    def __init__(
        self,
        lr: float = 1.0,
        memory_size: int = 10,
        scale_init_hess: bool = True,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.lbfgs(
            learning_rate=lr,
            memory_size=memory_size,
            scale_init_hess=scale_init_hess
        )

        if grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            tx = optax.chain(*transforms)

        super().__init__(tx=tx, lr=lr,
                         grad_clip_norm=grad_clip_norm, grad_clip_value=grad_clip_value)
        self.memory_size = memory_size
        self.scale_init_hess = scale_init_hess


class Rprop(OptaxOptimizer):
    """Rprop optimizer."""

    def __init__(
        self,
        lr: float = 1e-2,
        etas: Tuple[float, float] = (0.5, 1.2),
        step_sizes: Tuple[float, float] = (1e-6, 50.0),
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.rprop(
            learning_rate=lr,
            eta_minus=etas[0],
            eta_plus=etas[1],
            min_step_size=step_sizes[0],
            max_step_size=step_sizes[1]
        )

        if grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.etas = etas
        self.step_sizes = step_sizes


class Adafactor(OptaxOptimizer):
    """Adafactor optimizer (memory-efficient variant of Adam)."""

    def __init__(
        self,
        lr: Optional[float] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        factored: bool = True,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.adafactor(
            learning_rate=lr,
            min_dim_size_to_factor=128 if factored else None,
            decay_rate=decay_rate,
            eps=eps[0],
            clip_threshold=clip_threshold,
            beta1=beta1,
            weight_decay=weight_decay
        )

        if grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr or 1e-3,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.eps = eps
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.beta1 = beta1
        self.factored = factored


class AdaBelief(OptaxOptimizer):
    """AdaBelief optimizer (adapts step size according to belief in gradient direction)."""

    def __init__(
        self,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.adabelief(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps)

        if weight_decay > 0 or grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            if weight_decay > 0:
                transforms.append(optax.add_decayed_weights(weight_decay))
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.betas = betas
        self.eps = eps


class Lion(OptaxOptimizer):
    """Lion optimizer (discovered through program search)."""

    def __init__(
        self,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.lion(learning_rate=lr, b1=betas[0], b2=betas[1], weight_decay=weight_decay)

        if grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.betas = betas


class SM3(OptaxOptimizer):
    """SM3 optimizer (memory-efficient adaptive optimizer)."""

    def __init__(
        self,
        lr: float = 1.0,
        momentum: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.sm3(learning_rate=lr, momentum=momentum, eps=eps)

        if weight_decay > 0 or grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            if weight_decay > 0:
                transforms.append(optax.add_decayed_weights(weight_decay))
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.momentum = momentum
        self.eps = eps


class Novograd(OptaxOptimizer):
    """Novograd optimizer (normalized gradient descent)."""

    def __init__(
        self,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.novograd(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps, weight_decay=weight_decay)

        if grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.betas = betas
        self.eps = eps


class Fromage(OptaxOptimizer):
    """Fromage optimizer (memory-efficient learning rate free optimizer)."""

    def __init__(
        self,
        lr: float = 1.0,
        momentum: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        tx = optax.fromage(learning_rate=lr, momentum=momentum)

        if grad_clip_norm is not None or grad_clip_value is not None:
            transforms = []
            if grad_clip_norm is not None:
                transforms.append(optax.clip_by_global_norm(grad_clip_norm))
            if grad_clip_value is not None:
                transforms.append(optax.clip(grad_clip_value))
            transforms.append(tx)
            tx = optax.chain(*transforms)

        super().__init__(
            tx=tx,
            lr=lr,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )
        self.momentum = momentum
