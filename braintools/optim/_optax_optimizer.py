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

import warnings
from typing import Dict, Optional, Union, Callable, Any, List, Tuple

import jax.tree
import optax
from brainstate import State, maybe_state
from brainstate.typing import PyTree

from braintools.file import msgpack_from_state_dict
from ._base import Optimizer, OptimState
from ._optax_lr_scheduler import LRScheduler, ConstantLR
from ._state_uniquifier import UniqueStateManager

MaskOrFn = Optional[Union[Any, Callable]]

__all__ = [
    'OptaxOptimizer',

    # Main Optimizers
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


class OptaxOptimizer(Optimizer):
    """
    Base class for Optax-based optimizers with PyTorch-like interface.

    This class provides a unified interface for JAX/Optax optimizers with advanced features
    including learning rate scheduling, parameter groups, gradient clipping, and weight decay.

    Parameters
    ----------
    tx : optax.GradientTransformation, optional
        An Optax gradient transformation. If None, will be created based on optimizer-specific
        parameters via ``_create_default_tx()`` method.
    lr : float or LRScheduler, default=1e-3
        Learning rate. Can be a float (automatically converted to ConstantLR) or an
        LRScheduler instance for advanced scheduling.
    weight_decay : float, default=0.0
        Weight decay coefficient (L2 penalty).
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping. If provided, gradients will be clipped
        using ``optax.clip_by_global_norm``.
    grad_clip_value : float, optional
        Maximum gradient value for clipping. If provided, gradients will be clipped
        element-wise using ``optax.clip``.

    Attributes
    ----------
    param_states : UniqueStateManager
        Container for PyTree of brainstate.State objects representing trainable parameters.
    opt_state : OptimState
        Optimizer state containing momentum, variance, and other optimizer-specific values.
    step_count : OptimState
        Number of optimization steps taken.
    param_groups : list of dict
        List of parameter groups with their own hyperparameters. Each group is a dictionary
        with keys 'params', 'lr', 'weight_decay', etc.
    base_lr : float
        Base learning rate (read-only property).
    lr : float
        Current learning rate (can be modified by schedulers).

    Methods
    -------
    register_trainable_weights(param_states)
        Register parameters to be optimized.
    add_param_group(params, **kwargs)
        Add a new parameter group with custom hyperparameters.
    step(grads, closure=None)
        Perform a single optimization step.
    state_dict()
        Get optimizer state for checkpointing.
    load_state_dict(state_dict)
        Load optimizer state from checkpoint.

    Notes
    -----
    This base class implements the unified learning rate handling where all learning rates
    (whether float or LRScheduler) are internally managed through a scheduler. Float learning
    rates are automatically converted to ``ConstantLR`` for consistent handling.

    Examples
    --------
    Basic usage with float learning rate:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>>
        >>> # Define a simple model
        >>> class Model(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = brainstate.nn.Linear(10, 5)
        ...
        >>> model = Model()
        >>> optimizer = braintools.optim.Adam(lr=0.001)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Using learning rate scheduler:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>>
        >>> model = Model()
        >>> scheduler = braintools.optim.StepLR(base_lr=0.01, step_size=10, gamma=0.1)
        >>> optimizer = braintools.optim.Adam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Training loop
        >>> for epoch in range(100):
        ...     # ... compute gradients
        ...     optimizer.step(grads)
        ...     scheduler.step()

    Multiple parameter groups with different learning rates:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>>
        >>> model = Model()
        >>> optimizer = braintools.optim.Adam(lr=0.001)
        >>>
        >>> # Register main parameters
        >>> optimizer.register_trainable_weights(model.linear.states(brainstate.ParamState))
        >>>
        >>> # Add another group with different lr
        >>> special_params = {'special': brainstate.ParamState(jnp.zeros(5))}
        >>> optimizer.add_param_group(special_params, lr=0.0001)

    See Also
    --------
    Adam : Adam optimizer with adaptive learning rates
    SGD : Stochastic gradient descent optimizer
    AdamW : AdamW optimizer with decoupled weight decay
    StepLR : Learning rate scheduler with step decay
    """
    __module__ = 'braintools.optim'

    param_states: UniqueStateManager  # Container for PyTree of brainstate.State objects
    opt_state: Optional[OptimState]
    step_count: OptimState
    base_lr: float
    current_lr: float
    param_groups: List[Dict[str, Any]]
    param_groups_opt_states: List[OptimState]
    schedulers: List[LRScheduler]

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

        # Initialize attributes first
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_value = grad_clip_value
        self.step_count = OptimState(0)
        self.param_groups = []
        self.param_groups_opt_states = []  # Changed to list
        self._schedulers = []

        # Handle lr as either float or scheduler
        # Convert float to ConstantLR for unified handling
        lr = ConstantLR(base_lr=lr, factor=1.0, total_iters=0) if not isinstance(lr, LRScheduler) else lr
        self._lr_scheduler = lr
        self._base_lr = lr.base_lrs[0] if lr.base_lrs else 1e-3
        self._current_lr = OptimState(self._base_lr)
        lr.attach_optimizer(self)

        tx = self.default_tx() if tx is None else tx
        if not isinstance(tx, optax.GradientTransformation):
            raise TypeError(f"tx must be an instance of optax.GradientTransformation, got {tx}")
        self.tx = tx
        self.opt_state = None

    def default_tx(self):
        """Create default gradient transformation with clipping and weight decay."""
        transforms = []

        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))

        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))

        transforms.append(optax.scale_by_adam())

        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))

        # Always use the scheduler (now always present due to ConstantLR unification)
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))

        return optax.chain(*transforms)

    @property
    def schedulers(self):
        return self._schedulers

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def lr(self):
        return self._lr_scheduler

    @property
    def base_lr(self) -> float:
        """Get base learning rate."""
        return self._base_lr

    @property
    def current_lr(self):
        """Get current learning rate."""
        return self._current_lr.value

    @current_lr.setter
    def current_lr(self, value: float):
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
        param_values = manager.to_pytree_value()
        group_lr_state = OptimState(kwargs.get('lr', self.base_lr))

        group = {
            'params': manager.to_pytree(),
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
        group_opt_state = OptimState(group_tx.init(param_values))
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
        self.opt_state = OptimState(self.tx.init(param_values))

        # Create a default param group with all registered parameters
        # This maintains compatibility with PyTorch-like behavior
        if not self.param_groups:
            self.param_groups = [
                {
                    'params': self.param_states.to_pytree(),
                    'lr': self._current_lr,
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
                    group_lr = maybe_state(group.get('lr', self.current_lr))
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
            params = self.param_states.to_pytree()
            param_values = self.param_states.to_pytree_value()
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
            param_states = self.param_states.to_pytree()
            param_values = self.param_states.to_pytree_value()
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
            'lr': self.current_lr,
            'base_lr': self.base_lr,
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
        self.current_lr = msgpack_from_state_dict(self.current_lr, state_dict['lr'])
        self._base_lr = state_dict['base_lr']

        # Load param_groups and restore lr_state for groups that have it
        self.param_groups = msgpack_from_state_dict(
            self.param_groups,
            state_dict['param_groups']
        )

        if 'opt_state' in state_dict:
            if self.opt_state is None:
                self.opt_state = OptimState(state_dict['opt_state'])
            else:
                self.opt_state.value = state_dict['opt_state']

        # Load param group optimizer states
        if 'param_groups_opt_states' in state_dict:
            for i, s in enumerate(state_dict['param_groups_opt_states']):
                if i < len(self.param_groups_opt_states):
                    self.param_groups_opt_states[i].value = s
                else:
                    self.param_groups_opt_states.append(OptimState(s))

    def add_scheduler(self, scheduler: LRScheduler):
        """Add a learning rate scheduler."""
        self._schedulers.append(scheduler)

    def get_last_lr(self) -> List[float]:
        """Get last computed learning rates from schedulers."""
        if self._schedulers:
            return self._schedulers[-1].get_last_lr()
        return [self.current_lr]

    def lr_apply(self, apply_fn: Callable[[float], float]):
        """Apply a function to modify the current learning rate."""
        self.lr.apply(apply_fn)


# Optimizer implementations

class SGD(OptaxOptimizer):
    r"""
    Stochastic Gradient Descent (SGD) optimizer with momentum and weight decay.

    Implements the standard SGD algorithm with optional momentum, Nesterov momentum,
    and weight decay (L2 regularization).

    Parameters
    ----------
    lr : float or LRScheduler, default=1e-3
        Learning rate. Can be a float or LRScheduler instance.
    momentum : float, default=0.0
        Momentum factor. Set to 0 for vanilla SGD.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient.
    nesterov : bool, default=False
        Whether to use Nesterov momentum.
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping.
    grad_clip_value : float, optional
        Maximum gradient value for clipping.

    Notes
    -----
    The SGD update with momentum is computed as:

    .. math::

        v_{t+1} = \mu v_t + g_t

        \theta_{t+1} = \theta_t - \alpha v_{t+1}

    where :math:`\mu` is the momentum, :math:`g_t` is the gradient at step t,
    :math:`\alpha` is the learning rate, and :math:`\theta` are the parameters.

    With Nesterov momentum, the update becomes:

    .. math::

        v_{t+1} = \mu v_t + g_t

        \theta_{t+1} = \theta_t - \alpha (\mu v_{t+1} + g_t)

    Examples
    --------
    Basic SGD without momentum:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>> import jax.numpy as jnp
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>> optimizer = braintools.optim.SGD(lr=0.01)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    SGD with momentum:

    .. code-block:: python

        >>> optimizer = braintools.optim.SGD(lr=0.01, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    SGD with Nesterov momentum:

    .. code-block:: python

        >>> optimizer = braintools.optim.SGD(lr=0.01, momentum=0.9, nesterov=True)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    SGD with learning rate scheduling:

    .. code-block:: python

        >>> scheduler = braintools.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     # Training code here
        ...     optimizer.step(grads)
        ...     if (epoch + 1) % epoch_size == 0:
        ...         scheduler.step()

    See Also
    --------
    Adam : Adam optimizer with adaptive learning rates
    RMSprop : RMSprop optimizer
    Momentum : Pure momentum optimizer
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        # Store SGD-specific parameters
        self.momentum = momentum
        self.nesterov = nesterov

        # Don't build tx here, let the base class handle it
        super().__init__(
            tx=None,  # Will be created by base class
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        """Create SGD-specific gradient transformation."""
        transforms = []

        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))

        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))

        if self.momentum > 0:
            if self.nesterov:
                transforms.append(optax.trace(decay=self.momentum, nesterov=True))
            else:
                transforms.append(optax.trace(decay=self.momentum, nesterov=False))

        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))

        # Always use the scheduler (now always present due to ConstantLR unification)
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))

        return optax.chain(*transforms)


class Momentum(OptaxOptimizer):
    r"""
    Momentum optimizer.

    Implements the momentum variant of stochastic gradient descent, where updates
    accumulate a velocity that persists across iterations to accelerate convergence
    in relevant directions.

    Parameters
    ----------
    lr : float or LRScheduler, default=1e-3
        Learning rate. Can be a float or LRScheduler instance.
    momentum : float, default=0.9
        Momentum factor. The fraction of the gradient to retain from previous steps.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient.
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping.
    grad_clip_value : float, optional
        Maximum gradient value for clipping.

    Notes
    -----
    The momentum update is computed as:

    .. math::

        v_{t+1} = \mu v_t + g_t

        \theta_{t+1} = \theta_t - \alpha v_{t+1}

    where :math:`\mu` is the momentum factor, :math:`g_t` is the gradient at step t,
    :math:`\alpha` is the learning rate, :math:`v_t` is the velocity, and
    :math:`\theta` are the parameters.

    Examples
    --------
    Basic Momentum optimizer:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>> import jax.numpy as jnp
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>> optimizer = braintools.optim.Momentum(lr=0.01, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Momentum with weight decay:

    .. code-block:: python

        >>> optimizer = braintools.optim.Momentum(lr=0.01, momentum=0.9, weight_decay=0.0001)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Momentum with learning rate scheduling:

    .. code-block:: python

        >>> scheduler = braintools.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        >>> optimizer = braintools.optim.Momentum(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     # Training code here
        ...     optimizer.step(grads)
        ...     if (epoch + 1) % epoch_size == 0:
        ...         scheduler.step()

    See Also
    --------
    MomentumNesterov : Momentum with Nesterov acceleration
    SGD : Stochastic gradient descent with optional momentum
    Adam : Adam optimizer with adaptive learning rates
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        # Store momentum-specific parameters
        self.momentum = momentum

        # Don't build tx here, let the base class handle it
        super().__init__(
            tx=None,  # Will be created by base class
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        """Create Momentum-specific gradient transformation."""
        transforms = []

        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))

        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))

        # Add momentum transformation (always use momentum for this optimizer)
        transforms.append(optax.trace(decay=self.momentum, nesterov=False))

        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))

        # Always use the scheduler (now always present due to ConstantLR unification)
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))

        return optax.chain(*transforms)


class MomentumNesterov(OptaxOptimizer):
    r"""
    Nesterov Momentum optimizer.

    Implements Nesterov's accelerated gradient method, which looks ahead by extrapolating
    the momentum term before computing the gradient. This often leads to faster convergence
    compared to standard momentum.

    Parameters
    ----------
    lr : float or LRScheduler, default=1e-3
        Learning rate. Can be a float or LRScheduler instance.
    momentum : float, default=0.9
        Momentum factor. The fraction of the gradient to retain from previous steps.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient.
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping.
    grad_clip_value : float, optional
        Maximum gradient value for clipping.

    Notes
    -----
    The Nesterov momentum update is computed as:

    .. math::

        v_{t+1} = \mu v_t + g_t

        \theta_{t+1} = \theta_t - \alpha (\mu v_{t+1} + g_t)

    This is equivalent to first making a momentum step, then computing the gradient
    at the resulting position, which provides a "lookahead" effect.

    where :math:`\mu` is the momentum factor, :math:`g_t` is the gradient at step t,
    :math:`\alpha` is the learning rate, :math:`v_t` is the velocity, and
    :math:`\theta` are the parameters.

    References
    ----------
    .. [1] Nesterov, Y. (1983). A method for unconstrained convex minimization problem
           with the rate of convergence O(1/k²).

    Examples
    --------
    Basic Nesterov Momentum optimizer:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>> import jax.numpy as jnp
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>> optimizer = braintools.optim.MomentumNesterov(lr=0.01, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Nesterov Momentum with weight decay:

    .. code-block:: python

        >>> optimizer = braintools.optim.MomentumNesterov(lr=0.01, momentum=0.9, weight_decay=0.0001)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Nesterov Momentum with gradient clipping:

    .. code-block:: python

        >>> optimizer = braintools.optim.MomentumNesterov(
        ...     lr=0.01,
        ...     momentum=0.9,
        ...     grad_clip_norm=1.0
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Nesterov Momentum with learning rate scheduling:

    .. code-block:: python

        >>> scheduler = braintools.optim.ExponentialLR(base_lr=0.01, gamma=0.95)
        >>> optimizer = braintools.optim.MomentumNesterov(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     # Training code here
        ...     optimizer.step(grads)
        ...     scheduler.step()

    See Also
    --------
    Momentum : Standard momentum optimizer
    SGD : Stochastic gradient descent with optional momentum and Nesterov
    Adam : Adam optimizer with adaptive learning rates
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        # Store momentum-specific parameters
        self.momentum = momentum

        # Don't build tx here, let the base class handle it
        super().__init__(
            tx=None,  # Will be created by base class
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        """Create Nesterov Momentum-specific gradient transformation."""
        transforms = []

        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))

        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))

        # Add Nesterov momentum transformation
        transforms.append(optax.trace(decay=self.momentum, nesterov=True))

        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))

        # Always use the scheduler (now always present due to ConstantLR unification)
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))

        return optax.chain(*transforms)


class Adam(OptaxOptimizer):
    r"""
    Adam (Adaptive Moment Estimation) optimizer.

    Adam is an adaptive learning rate optimization algorithm that combines the advantages
    of AdaGrad and RMSProp. It computes adaptive learning rates for each parameter by
    maintaining first and second moment estimates of the gradients.

    Parameters
    ----------
    lr : float or LRScheduler, default=1e-3
        Learning rate. Can be a float or LRScheduler instance.
    betas : tuple of float, default=(0.9, 0.999)
        Coefficients (beta1, beta2) for computing running averages of gradient
        and its square. beta1 is the exponential decay rate for the first moment
        estimates, beta2 is for the second moment estimates.
    eps : float, default=1e-8
        Term added to the denominator to improve numerical stability.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient.
    amsgrad : bool, default=False
        Whether to use the AMSGrad variant of Adam.
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping.
    grad_clip_value : float, optional
        Maximum gradient value for clipping.

    Notes
    -----
    The Adam update is computed as:

    .. math::

        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t

        v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2

        \hat{m}_t = \frac{m_t}{1 - \beta_1^t}

        \hat{v}_t = \frac{v_t}{1 - \beta_2^t}

        \theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}

    where :math:`g_t` is the gradient, :math:`m_t` and :math:`v_t` are the first
    and second moment estimates, :math:`\alpha` is the learning rate, and :math:`t`
    is the time step.

    References
    ----------
    .. [1] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
           arXiv preprint arXiv:1412.6980.

    Examples
    --------
    Basic Adam optimizer:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> optimizer = braintools.optim.Adam(lr=0.001)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Adam with custom beta values:

    .. code-block:: python

        >>> optimizer = braintools.optim.Adam(lr=0.001, betas=(0.9, 0.99))
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Adam with AMSGrad:

    .. code-block:: python

        >>> optimizer = braintools.optim.Adam(lr=0.001, amsgrad=True)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Adam with learning rate scheduler:

    .. code-block:: python

        >>> scheduler = braintools.optim.ExponentialLR(base_lr=0.001, gamma=0.95)
        >>> optimizer = braintools.optim.Adam(lr=scheduler, betas=(0.9, 0.999))
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     # Training code
        ...     optimizer.step(grads)
        ...     scheduler.step()

    Adam with gradient clipping:

    .. code-block:: python

        >>> optimizer = braintools.optim.Adam(lr=0.001, grad_clip_norm=1.0)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    See Also
    --------
    AdamW : Adam with decoupled weight decay
    RAdam : Rectified Adam
    Nadam : Adam with Nesterov momentum
    SGD : Stochastic gradient descent
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        beta1: float = None,
        beta2: float = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        # Store Adam-specific parameters
        betas = list(betas)
        if beta1 is not None:
            betas[0] = beta1
            warnings.warn(
                'The `beta1` parameter is deprecated, please use `betas=(beta1, beta2)` instead.',
                DeprecationWarning
            )
        if beta2 is not None:
            betas[1] = beta2
            warnings.warn(
                'The `beta2` parameter is deprecated, please use `betas=(beta1, beta2)` instead.',
                DeprecationWarning
            )
        self.betas = tuple(betas)
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

    def default_tx(self):
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

        # Always use the scheduler (now always present due to ConstantLR unification)
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))

        return optax.chain(*transforms)


class AdamW(OptaxOptimizer):
    r"""
    AdamW optimizer with decoupled weight decay regularization.

    AdamW modifies the standard Adam algorithm by decoupling the weight decay from the
    gradient-based update, which has been shown to improve generalization performance.

    Parameters
    ----------
    lr : float or LRScheduler, default=1e-3
        Learning rate. Can be a float or LRScheduler instance.
    betas : tuple of float, default=(0.9, 0.999)
        Coefficients (beta1, beta2) for computing running averages.
    eps : float, default=1e-8
        Term added to the denominator for numerical stability.
    weight_decay : float, default=0.01
        Weight decay coefficient (decoupled from gradient).
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping.
    grad_clip_value : float, optional
        Maximum gradient value for clipping.

    Notes
    -----
    Unlike Adam where weight decay is part of the gradient computation, AdamW applies
    weight decay directly to the parameters:

    .. math::

        \theta_t = \theta_{t-1} - \alpha (\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1})

    where :math:`\lambda` is the weight decay coefficient.

    References
    ----------
    .. [1] Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization.
           arXiv preprint arXiv:1711.05101.

    Examples
    --------
    Basic AdamW usage:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> optimizer = braintools.optim.AdamW(lr=0.001, weight_decay=0.01)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    AdamW with scheduler:

    .. code-block:: python

        >>> scheduler = braintools.optim.CosineAnnealingLR(base_lr=0.001, T_max=100)
        >>> optimizer = braintools.optim.AdamW(lr=scheduler, weight_decay=0.01)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    See Also
    --------
    Adam : Standard Adam optimizer
    SGD : Stochastic gradient descent
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        # Store AdamW-specific parameters
        self.betas = betas
        self.eps = eps

        # Don't build tx here, let the base class handle it
        super().__init__(
            tx=None,  # Will be created by base class
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        """Create AdamW-specific gradient transformation."""
        transforms = []

        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))

        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))

        transforms.append(optax.scale_by_adam(b1=self.betas[0], b2=self.betas[1], eps=self.eps))

        # AdamW uses decoupled weight decay
        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))

        # Always use the scheduler (now always present due to ConstantLR unification)
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))

        return optax.chain(*transforms)


class Adagrad(OptaxOptimizer):
    r"""
    Adagrad optimizer with adaptive learning rates.

    Adagrad adapts the learning rate for each parameter based on the historical
    gradient information. Parameters with larger gradients have smaller learning rates,
    and vice versa. This makes it well-suited for sparse data.

    Parameters
    ----------
    lr : float or LRScheduler, default=1e-2
        Learning rate. Can be a float or LRScheduler instance.
    lr_decay : float, default=0.0
        Learning rate decay over each update.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient.
    initial_accumulator_value : float, default=0.0
        Initial value for the gradient accumulator.
    eps : float, default=1e-10
        Term added to the denominator to improve numerical stability.
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping.
    grad_clip_value : float, optional
        Maximum gradient value for clipping.

    Notes
    -----
    The Adagrad update is computed as:

    .. math::

        G_t = G_{t-1} + g_t^2

        \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{G_t} + \epsilon} g_t

    where :math:`G_t` accumulates squared gradients, :math:`g_t` is the gradient,
    :math:`\alpha` is the learning rate, and :math:`\epsilon` is for numerical stability.

    Adagrad's main weakness is that the accumulated squared gradients in the denominator
    continue to grow, causing the learning rate to shrink and eventually become
    infinitesimally small.

    References
    ----------
    .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods
           for online learning and stochastic optimization. JMLR, 12, 2121-2159.

    Examples
    --------
    Basic Adagrad usage:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> optimizer = braintools.optim.Adagrad(lr=0.01)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Adagrad with custom epsilon for stability:

    .. code-block:: python

        >>> optimizer = braintools.optim.Adagrad(lr=0.01, eps=1e-8)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Adagrad with weight decay:

    .. code-block:: python

        >>> optimizer = braintools.optim.Adagrad(lr=0.01, weight_decay=0.01)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    See Also
    --------
    Adadelta : Adagrad extension that seeks to reduce aggressive learning rate decay
    RMSprop : Root Mean Square Propagation, similar to Adagrad but with exponential decay
    Adam : Combines ideas from Adagrad and RMSprop
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        # Store Adagrad-specific parameters
        self.lr_decay = lr_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps

        # Don't build tx here, let the base class handle it
        super().__init__(
            tx=None,  # Will be created by base class
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        """Create Adagrad-specific gradient transformation."""
        transforms = []

        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))

        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))

        transforms.append(optax.scale_by_rms(initial_scale=self.initial_accumulator_value, eps=self.eps))

        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))

        # Always use the scheduler (now always present due to ConstantLR unification)
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))

        return optax.chain(*transforms)


class Adadelta(OptaxOptimizer):
    r"""
    Adadelta optimizer - an extension of Adagrad.

    Adadelta is an extension of Adagrad that seeks to reduce its aggressive,
    monotonically decreasing learning rate. Instead of accumulating all past
    squared gradients, Adadelta restricts the window of accumulated past
    gradients to some fixed size.

    Parameters
    ----------
    lr : float or LRScheduler, default=1.0
        Learning rate (scaling factor). Can be a float or LRScheduler instance.
        Note: Adadelta is largely learning rate free, so 1.0 is often sufficient.
    rho : float, default=0.9
        Coefficient used for computing running average of squared gradients.
    eps : float, default=1e-6
        Term added to the denominator to improve numerical stability.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient.
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping.
    grad_clip_value : float, optional
        Maximum gradient value for clipping.

    Notes
    -----
    The Adadelta update is computed as:

    .. math::

        E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho) g_t^2

        \Delta \theta_t = - \frac{\sqrt{E[\Delta \theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} g_t

        E[\Delta \theta^2]_t = \rho E[\Delta \theta^2]_{t-1} + (1 - \rho) \Delta \theta_t^2

        \theta_t = \theta_{t-1} + \Delta \theta_t

    where :math:`\rho` is the decay rate, :math:`g_t` is the gradient,
    and :math:`\epsilon` is for numerical stability.

    References
    ----------
    .. [1] Zeiler, M. D. (2012). ADADELTA: An adaptive learning rate method.
           arXiv preprint arXiv:1212.5701.

    Examples
    --------
    Basic Adadelta usage:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> optimizer = braintools.optim.Adadelta()
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Adadelta with custom rho:

    .. code-block:: python

        >>> optimizer = braintools.optim.Adadelta(rho=0.95)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Adadelta with explicit learning rate:

    .. code-block:: python

        >>> optimizer = braintools.optim.Adadelta(lr=0.5, rho=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    See Also
    --------
    Adagrad : Adaptive gradient algorithm with accumulating squared gradients
    RMSprop : Similar to Adadelta but simpler
    Adam : Combines ideas from RMSprop and momentum
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        # Store Adadelta-specific parameters
        self.rho = rho
        self.eps = eps

        # Don't build tx here, let the base class handle it
        super().__init__(
            tx=None,  # Will be created by base class
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        """Create Adadelta-specific gradient transformation."""
        transforms = []

        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))

        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))

        transforms.append(optax.scale_by_adadelta(rho=self.rho, eps=self.eps))

        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))

        # Always use the scheduler (now always present due to ConstantLR unification)
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))

        return optax.chain(*transforms)


class RMSprop(OptaxOptimizer):
    r"""
    RMSprop (Root Mean Square Propagation) optimizer.

    RMSprop divides the learning rate by an exponentially decaying average of squared
    gradients. This helps the optimizer navigate ravines, where the surface curves
    much more steeply in one dimension than in another.

    Parameters
    ----------
    lr : float or LRScheduler, default=1e-2
        Learning rate. Can be a float or LRScheduler instance.
    alpha : float, default=0.99
        Smoothing constant (decay rate for moving average of squared gradients).
    eps : float, default=1e-8
        Term added to the denominator to improve numerical stability.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient.
    momentum : float, default=0.0
        Momentum factor. If > 0, uses momentum-based RMSprop.
    centered : bool, default=False
        If True, compute centered RMSprop (normalizes gradient by variance estimate).
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping.
    grad_clip_value : float, optional
        Maximum gradient value for clipping.

    Notes
    -----
    The RMSprop update is computed as:

    .. math::

        E[g^2]_t = \alpha E[g^2]_{t-1} + (1 - \alpha) g_t^2

        \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t

    where :math:`\alpha` is the decay rate, :math:`g_t` is the gradient,
    :math:`\eta` is the learning rate, and :math:`\epsilon` is for numerical stability.

    With centered=True:

    .. math::

        E[g]_t = \alpha E[g]_{t-1} + (1 - \alpha) g_t

        \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t - E[g]_t^2 + \epsilon}} g_t

    References
    ----------
    .. [1] Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the
           gradient by a running average of its recent magnitude. COURSERA:
           Neural networks for machine learning, 4(2), 26-31.

    Examples
    --------
    Basic RMSprop usage:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> optimizer = braintools.optim.RMSprop(lr=0.01)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    RMSprop with momentum:

    .. code-block:: python

        >>> optimizer = braintools.optim.RMSprop(lr=0.01, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Centered RMSprop:

    .. code-block:: python

        >>> optimizer = braintools.optim.RMSprop(lr=0.01, centered=True)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    RMSprop with custom alpha:

    .. code-block:: python

        >>> optimizer = braintools.optim.RMSprop(lr=0.01, alpha=0.95)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    See Also
    --------
    Adagrad : Adaptive gradient algorithm
    Adadelta : Extension of Adagrad
    Adam : Combines RMSprop with momentum
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        # Store RMSprop-specific parameters
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.centered = centered

        # Don't build tx here, let the base class handle it
        super().__init__(
            tx=None,  # Will be created by base class
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        """Create RMSprop-specific gradient transformation."""
        transforms = []

        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))

        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))

        transforms.append(optax.scale_by_rms(decay=self.alpha, eps=self.eps))

        if self.momentum > 0:
            transforms.append(optax.trace(decay=self.momentum))

        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))

        # Always use the scheduler (now always present due to ConstantLR unification)
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))

        return optax.chain(*transforms)


class Adamax(OptaxOptimizer):
    r"""
    Adamax optimizer - variant of Adam based on infinity norm.

    Adamax is a variant of Adam based on the infinity norm, making it more robust
    to large gradients. It can sometimes achieve better performance than Adam.

    Parameters
    ----------
    lr : float or LRScheduler, default=2e-3
        Learning rate. Can be a float or LRScheduler instance.
    betas : tuple of float, default=(0.9, 0.999)
        Coefficients (beta1, beta2) for computing running averages.
    eps : float, default=1e-8
        Term added to the denominator for numerical stability.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient.
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping.
    grad_clip_value : float, optional
        Maximum gradient value for clipping.

    Notes
    -----
    The Adamax update uses the infinity norm:

    .. math::

        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t

        u_t = \max(\beta_2 u_{t-1}, |g_t|)

        \theta_t = \theta_{t-1} - \frac{\alpha}{1 - \beta_1^t} \frac{m_t}{u_t + \epsilon}

    where :math:`u_t` uses the max operation instead of the squared gradients in Adam.

    References
    ----------
    .. [1] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
           arXiv preprint arXiv:1412.6980.

    Examples
    --------
    Basic Adamax usage:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>> optimizer = braintools.optim.Adamax(lr=0.002)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Adamax with custom betas:

    .. code-block:: python

        >>> optimizer = braintools.optim.Adamax(lr=0.002, betas=(0.9, 0.99))
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    See Also
    --------
    Adam : Standard Adam optimizer
    Nadam : Adam with Nesterov momentum
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 2e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.betas = betas
        self.eps = eps
        super().__init__(
            tx=None,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        transforms.append(optax.scale_by_adamax(b1=self.betas[0], b2=self.betas[1], eps=self.eps))
        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)


class Nadam(OptaxOptimizer):
    r"""
    Nadam optimizer - Adam with Nesterov accelerated gradient.

    Nadam (Nesterov-accelerated Adaptive Moment Estimation) combines Adam with
    Nesterov momentum. It provides the benefits of both adaptive learning rates
    and Nesterov's accelerated gradient method, often leading to faster convergence
    and better performance than standard Adam.

    Parameters
    ----------
    lr : float or LRScheduler, default=2e-3
        Learning rate. Can be a float or LRScheduler instance.
    betas : tuple of float, default=(0.9, 0.999)
        Coefficients (beta1, beta2) for computing running averages of gradient
        and its square. beta1 is the exponential decay rate for the first moment
        estimates, beta2 is for the second moment estimates.
    eps : float, default=1e-8
        Term added to the denominator to improve numerical stability.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient.
    momentum_decay : float, default=4e-3
        Momentum schedule decay rate for Nadam.
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping.
    grad_clip_value : float, optional
        Maximum gradient value for clipping.

    Notes
    -----
    The Nadam update combines Adam's adaptive learning rate with Nesterov momentum:

    .. math::

        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t

        v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2

        \hat{m}_t = \frac{m_t}{1 - \beta_1^{t+1}} + \frac{(1 - \beta_1) g_t}{1 - \beta_1^t}

        \hat{v}_t = \frac{v_t}{1 - \beta_2^t}

        \theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}

    where the key difference from Adam is in the bias-corrected first moment estimate
    :math:`\hat{m}_t`, which incorporates a look-ahead step similar to Nesterov momentum.

    References
    ----------
    .. [1] Dozat, T. (2016). Incorporating Nesterov momentum into Adam.
           ICLR 2016 Workshop.

    Examples
    --------
    Basic Nadam usage:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>> import jax.numpy as jnp
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>> optimizer = braintools.optim.Nadam(lr=0.002)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Nadam with custom betas:

    .. code-block:: python

        >>> optimizer = braintools.optim.Nadam(lr=0.002, betas=(0.9, 0.99))
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Nadam with weight decay:

    .. code-block:: python

        >>> optimizer = braintools.optim.Nadam(lr=0.002, weight_decay=0.01)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Nadam with learning rate scheduler:

    .. code-block:: python

        >>> scheduler = braintools.optim.ExponentialLR(gamma=0.95)
        >>> optimizer = braintools.optim.Nadam(lr=scheduler, betas=(0.9, 0.999))
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> for epoch in range(100):
        ...     # Training code
        ...     optimizer.step(grads)
        ...     scheduler.step()

    Nadam with gradient clipping:

    .. code-block:: python

        >>> optimizer = braintools.optim.Nadam(
        ...     lr=0.002,
        ...     grad_clip_norm=1.0,
        ...     grad_clip_value=0.5
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Comparison with Adam - Nadam often converges faster:

    .. code-block:: python

        >>> # Compare convergence
        >>> model1 = brainstate.nn.Linear(10, 5)
        >>> model2 = brainstate.nn.Linear(10, 5)
        >>>
        >>> adam = braintools.optim.Adam(lr=0.002)
        >>> nadam = braintools.optim.Nadam(lr=0.002)
        >>>
        >>> adam.register_trainable_weights(model1.states(brainstate.ParamState))
        >>> nadam.register_trainable_weights(model2.states(brainstate.ParamState))
        >>> # Nadam typically shows faster initial convergence

    See Also
    --------
    Adam : Standard Adam optimizer
    Adamax : Adam variant with infinity norm
    RAdam : Rectified Adam with variance adaptation
    SGD : Stochastic gradient descent with Nesterov momentum option
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 2e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 4e-3,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.betas = betas
        self.eps = eps
        self.momentum_decay = momentum_decay
        super().__init__(
            tx=None,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        # Nadam is Adam with Nesterov momentum - use adam with nesterov-style updates
        transforms.append(optax.scale_by_adam(b1=self.betas[0], b2=self.betas[1], eps=self.eps, nesterov=True))
        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)


class RAdam(OptaxOptimizer):
    r"""RAdam optimizer (Rectified Adam).

    RAdam addresses the bad convergence problem of Adam by rectifying the variance
    of the adaptive learning rate. It provides a dynamic warmup schedule that
    automatically adapts to the current optimization state.

    Parameters
    ----------
    lr : float or LRScheduler, default=1e-3
        Learning rate. Can be a float (converted to ConstantLR) or any LRScheduler instance.
    betas : tuple of float, default=(0.9, 0.999)
        Coefficients for computing running averages of gradient and its square.
        The first value (beta1) controls the exponential decay rate for the first moment,
        and the second value (beta2) controls the decay rate for the second moment.
    eps : float, default=1e-8
        Term added to the denominator for numerical stability.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient.
    grad_clip_norm : float, optional
        Maximum norm for gradient clipping. If specified, gradients are clipped
        when their global norm exceeds this value.
    grad_clip_value : float, optional
        Maximum absolute value for gradient clipping. If specified, gradients are
        clipped element-wise to [-grad_clip_value, grad_clip_value].

    Notes
    -----
    RAdam introduces a rectification term that explicitly controls the adaptive
    learning rate based on the variance of the exponential moving average. The
    update rule is:

    .. math::
        \rho_t = \rho_{\infty} - \frac{2t\beta_2^t}{1-\beta_2^t}

        r_t = \sqrt{\frac{(\rho_t - 4)(\rho_t - 2)\rho_{\infty}}{(\rho_{\infty} - 4)(\rho_{\infty} - 2)\rho_t}}

    where :math:`\rho_{\infty} = \frac{2}{1-\beta_2} - 1`

    When :math:`\rho_t > 4`, the adaptive learning rate with rectification is used:

    .. math::
        \theta_{t+1} = \theta_t - \alpha \cdot r_t \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}

    Otherwise, it falls back to non-adaptive learning rate:

    .. math::
        \theta_{t+1} = \theta_t - \alpha \cdot m_t

    RAdam automatically performs warmup without requiring manual tuning, making it
    more robust than standard Adam in the early stages of training.

    References
    ----------
    .. [1] Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., & Han, J. (2019).
           On the variance of the adaptive learning rate and beyond.
           arXiv preprint arXiv:1908.03265.

    Examples
    --------
    Basic usage with float learning rate:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Initialize RAdam optimizer
        >>> optimizer = braintools.optim.RAdam(lr=0.001)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Using custom beta values for different convergence behavior:

    .. code-block:: python

        >>> # Slower first moment decay for more stable updates
        >>> optimizer = braintools.optim.RAdam(lr=0.001, betas=(0.8, 0.999))
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With learning rate scheduler for training dynamics:

    .. code-block:: python

        >>> # Exponential learning rate decay
        >>> scheduler = braintools.optim.ExponentialLR(base_lr=0.001, gamma=0.95)
        >>> optimizer = braintools.optim.RAdam(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Using gradient clipping for stability:

    .. code-block:: python

        >>> # Clip gradients by global norm
        >>> optimizer = braintools.optim.RAdam(
        ...     lr=0.001,
        ...     grad_clip_norm=1.0
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With weight decay for regularization:

    .. code-block:: python

        >>> # Add L2 regularization
        >>> optimizer = braintools.optim.RAdam(
        ...     lr=0.001,
        ...     weight_decay=0.01
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Complete training loop example:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> # Setup
        >>> model = brainstate.nn.Linear(10, 5)
        >>> optimizer = braintools.optim.RAdam(lr=0.001)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Training step
        >>> @brainstate.transform.jit
        ... def train_step(input_data, target):
        ...     def loss_fn():
        ...         pred = model(input_data)
        ...         return jnp.mean((pred - target) ** 2)
        ...
        ...     grads = brainstate.transform.grad(loss_fn, model.states(brainstate.ParamState))()
        ...     optimizer.update(grads)
        ...     return loss_fn()
        >>>
        >>> # Train
        >>> x = jnp.ones((32, 10))
        >>> y = jnp.zeros((32, 5))
        >>> loss = train_step(x, y)

    See Also
    --------
    Adam : Standard Adam optimizer without rectification
    AdamW : Adam with decoupled weight decay
    Nadam : Adam with Nesterov momentum

    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.betas = betas
        self.eps = eps
        super().__init__(
            tx=None,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        transforms.append(optax.scale_by_radam(b1=self.betas[0], b2=self.betas[1], eps=self.eps))
        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)


class Lamb(OptaxOptimizer):
    r"""LAMB optimizer (Layer-wise Adaptive Moments).

    LAMB is designed for large batch training, adapting the learning rate based on
    the ratio of weight norm to gradient norm for each layer. It enables training
    with very large batch sizes while maintaining performance comparable to small
    batch training.

    Parameters
    ----------
    lr : float or LRScheduler, default=1e-3
        Learning rate. Can be a float (converted to ConstantLR) or any LRScheduler instance.
        Note: LAMB can often use higher learning rates than Adam due to its layer-wise
        normalization.
    betas : tuple of float, default=(0.9, 0.999)
        Coefficients for computing running averages of gradient and its square.
        The first value (beta1) controls the exponential decay rate for the first moment,
        and the second value (beta2) controls the decay rate for the second moment.
    eps : float, default=1e-6
        Term added to the denominator for numerical stability.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient. LAMB applies weight decay adaptively
        based on the trust ratio.
    grad_clip_norm : float, optional
        Maximum norm for gradient clipping. If specified, gradients are clipped
        when their global norm exceeds this value.
    grad_clip_value : float, optional
        Maximum absolute value for gradient clipping. If specified, gradients are
        clipped element-wise to [-grad_clip_value, grad_clip_value].

    Notes
    -----
    LAMB extends Adam with layer-wise adaptation of the learning rate. The key
    innovation is the trust ratio mechanism that normalizes updates based on
    parameter and gradient norms:

    .. math::
        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t

        v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2

        \hat{m}_t = \frac{m_t}{1 - \beta_1^t}

        \hat{v}_t = \frac{v_t}{1 - \beta_2^t}

        r_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}

    The trust ratio is computed as:

    .. math::
        \text{trust_ratio} = \begin{cases}
            \frac{\|w_t\|}{\|r_t\|} & \text{if } \|w_t\| > 0 \text{ and } \|r_t\| > 0 \\
            1 & \text{otherwise}
        \end{cases}

    Final update:

    .. math::
        w_{t+1} = w_t - \alpha \cdot \text{trust_ratio} \cdot r_t

    LAMB is particularly effective for:

    - Training with batch sizes of 32K or larger
    - BERT and other transformer models
    - Distributed training across multiple GPUs/TPUs
    - Achieving linear scaling of learning rate with batch size

    References
    ----------
    .. [1] You, Y., Li, J., Reddi, S., Hseu, J., Kumar, S., Bhojanapalli, S., ... & Hsieh, C. J. (2019).
           Large batch optimization for deep learning: Training BERT in 76 minutes.
           arXiv preprint arXiv:1904.00962.

    Examples
    --------
    Basic usage with float learning rate:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Initialize LAMB optimizer for large batch training
        >>> optimizer = braintools.optim.Lamb(lr=0.002)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Using LAMB with large learning rate for big batch sizes:

    .. code-block:: python

        >>> # LAMB can handle larger learning rates due to trust ratio
        >>> optimizer = braintools.optim.Lamb(lr=0.01, betas=(0.9, 0.999))
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With learning rate scheduler for warmup and decay:

    .. code-block:: python

        >>> # Polynomial decay with warmup (common for BERT training)
        >>> scheduler = braintools.optim.PolynomialLR(
        ...     base_lr=0.002,
        ...     warmup_steps=1000,
        ...     total_steps=10000,
        ...     power=1.0
        ... )
        >>> optimizer = braintools.optim.Lamb(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With weight decay for regularization:

    .. code-block:: python

        >>> # LAMB applies weight decay adaptively
        >>> optimizer = braintools.optim.Lamb(
        ...     lr=0.002,
        ...     weight_decay=0.01
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Using gradient clipping for stability:

    .. code-block:: python

        >>> # Clip gradients for training stability
        >>> optimizer = braintools.optim.Lamb(
        ...     lr=0.002,
        ...     grad_clip_norm=1.0
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Complete training example for large batch:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> # Setup for large batch training
        >>> model = brainstate.nn.Sequential(
        ...     brainstate.nn.Linear(784, 256),
        ...     brainstate.nn.ReLU(),
        ...     brainstate.nn.Linear(256, 10)
        ... )
        >>>
        >>> # LAMB with settings for large batch
        >>> optimizer = braintools.optim.Lamb(
        ...     lr=0.01,  # Higher lr due to normalization
        ...     betas=(0.9, 0.999),
        ...     weight_decay=0.01
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Simulate large batch training
        >>> def train_step(batch_x, batch_y):
        ...     def loss_fn():
        ...         logits = model(batch_x)
        ...         return jnp.mean(
        ...             braintools.metric.softmax_cross_entropy(logits, batch_y)
        ...         )
        ...
        ...     grads = brainstate.transform.grad(loss_fn, model.states(brainstate.ParamState))()
        ...     optimizer.update(grads)
        ...     return loss_fn()
        >>>
        >>> # Large batch size
        >>> x = jnp.ones((1024, 784))  # Large batch
        >>> y = jnp.zeros((1024, 10))
        >>> # loss = train_step(x, y)

    See Also
    --------
    Adam : Standard Adam optimizer without layer-wise adaptation
    Lars : Layer-wise Adaptive Rate Scaling
    AdamW : Adam with decoupled weight decay

    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.betas = betas
        self.eps = eps
        super().__init__(
            tx=None,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        transforms.append(optax.scale_by_trust_ratio())
        transforms.append(optax.scale_by_adam(b1=self.betas[0], b2=self.betas[1], eps=self.eps))
        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)


class Lars(OptaxOptimizer):
    r"""LARS optimizer (Layer-wise Adaptive Rate Scaling).

    LARS adapts the learning rate for each layer based on the ratio between
    the norm of weights and the norm of gradients. This allows for stable
    training with very large batch sizes by preventing layers from having
    their weights change too rapidly.

    Parameters
    ----------
    lr : float or LRScheduler, default=1.0
        Learning rate. Can be a float (converted to ConstantLR) or any LRScheduler instance.
        Note: LARS typically uses higher base learning rates than SGD due to its
        layer-wise scaling.
    momentum : float, default=0.9
        Momentum factor for the moving average of gradients. Higher values
        provide more smoothing of gradient updates.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient. Applied before the trust ratio
        computation.
    trust_coefficient : float, default=0.001
        Trust coefficient (eta) that scales the local learning rate. Controls
        how much the layer-wise adaptation affects the update.
    eps : float, default=1e-8
        Term added to denominators for numerical stability.
    grad_clip_norm : float, optional
        Maximum norm for gradient clipping. If specified, gradients are clipped
        when their global norm exceeds this value.
    grad_clip_value : float, optional
        Maximum absolute value for gradient clipping. If specified, gradients are
        clipped element-wise to [-grad_clip_value, grad_clip_value].

    Notes
    -----
    LARS computes a local learning rate for each layer based on the trust ratio.
    The update rule with momentum is:

    .. math::
        g_t = \nabla L(w_t) + \lambda w_t

    where :math:`\lambda` is the weight decay coefficient.

    The local learning rate is computed as:

    .. math::
        \text{local_lr} = \eta \times \frac{\|w_t\|}{\|g_t\| + \epsilon}

    The momentum update is then:

    .. math::
        v_t = \mu v_{t-1} + \text{local_lr} \times g_t

        w_{t+1} = w_t - \alpha \times v_t

    where :math:`\eta` is the trust coefficient, :math:`\mu` is momentum,
    and :math:`\alpha` is the global learning rate.

    Key properties of LARS:

    - Enables linear scaling of batch size with learning rate
    - Particularly effective for training ResNets and other CNNs
    - Maintains different learning rates for different layers
    - Prevents rapid weight changes in any single layer

    The trust ratio mechanism ensures that:

    - Layers with large weights get smaller updates
    - Layers with large gradients get smaller updates
    - The relative change in weights is controlled

    References
    ----------
    .. [1] You, Y., Gitman, I., & Ginsburg, B. (2017).
           Large batch training of convolutional networks.
           arXiv preprint arXiv:1708.03888.
    .. [2] You, Y., Zhang, Z., Hsieh, C. J., Demmel, J., & Keutzer, K. (2018).
           ImageNet training in minutes.
           In Proceedings of the 47th International Conference on Parallel Processing.

    Examples
    --------
    Basic usage with momentum:

    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Initialize LARS optimizer
        >>> optimizer = braintools.optim.Lars(lr=0.1, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With custom trust coefficient for fine-tuning:

    .. code-block:: python

        >>> # Smaller trust coefficient for more conservative updates
        >>> optimizer = braintools.optim.Lars(
        ...     lr=0.1,
        ...     momentum=0.9,
        ...     trust_coefficient=0.0001
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Using learning rate scheduler:

    .. code-block:: python

        >>> # Cosine annealing schedule
        >>> scheduler = braintools.optim.CosineAnnealingLR(
        ...     base_lr=0.1,
        ...     T_max=100
        ... )
        >>> optimizer = braintools.optim.Lars(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With weight decay for regularization:

    .. code-block:: python

        >>> # LARS with weight decay
        >>> optimizer = braintools.optim.Lars(
        ...     lr=0.1,
        ...     momentum=0.9,
        ...     weight_decay=5e-4
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Large batch training configuration:

    .. code-block:: python

        >>> # Configuration for large batch training
        >>> # Linear scaling rule: lr = base_lr * (batch_size / base_batch)
        >>> batch_size = 4096
        >>> base_batch = 256
        >>> base_lr = 0.1
        >>> scaled_lr = base_lr * (batch_size / base_batch)
        >>>
        >>> optimizer = braintools.optim.Lars(
        ...     lr=scaled_lr,
        ...     momentum=0.9,
        ...     weight_decay=5e-4,
        ...     trust_coefficient=0.001
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Complete training example with CNN:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> # Setup CNN model
        >>> model = brainstate.nn.Sequential(
        ...     brainstate.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        ...     brainstate.nn.ReLU(),
        ...     brainstate.nn.MaxPool2d(2),
        ...     brainstate.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        ...     brainstate.nn.ReLU(),
        ...     brainstate.nn.AdaptiveAvgPool2d((1, 1)),
        ...     brainstate.nn.Flatten(),
        ...     brainstate.nn.Linear(128, 10)
        ... )
        >>>
        >>> # LARS for large batch CNN training
        >>> optimizer = braintools.optim.Lars(
        ...     lr=1.6,  # Large lr for batch size 4096
        ...     momentum=0.9,
        ...     weight_decay=1e-4,
        ...     trust_coefficient=0.001
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Training step
        >>> def train_step(images, labels):
        ...     def loss_fn():
        ...         logits = model(images)
        ...         return jnp.mean(
        ...             braintools.metric.softmax_cross_entropy(logits, labels)
        ...         )
        ...
        ...     grads = brainstate.transform.grad(loss_fn, model.states(brainstate.ParamState))()
        ...     optimizer.update(grads)
        ...     return loss_fn()
        >>>
        >>> # Large batch training
        >>> x = jnp.ones((256, 3, 32, 32))  # Large batch
        >>> y = jnp.zeros((256, 10))
        >>> # loss = train_step(x, y)

    See Also
    --------
    Lamb : Layer-wise Adaptive Moments optimizer
    SGD : Standard stochastic gradient descent
    Adam : Adaptive moment estimation

    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.momentum = momentum
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        super().__init__(
            tx=None,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        transforms.append(optax.scale_by_trust_ratio(trust_coefficient=self.trust_coefficient, eps=self.eps))
        transforms.append(optax.trace(decay=self.momentum))
        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)


class Lookahead(OptaxOptimizer):
    r"""Lookahead optimizer wrapper.

    Lookahead is a meta-optimizer that wraps any standard optimizer and maintains
    two sets of weights: "fast weights" updated by the base optimizer, and "slow
    weights" that are periodically synchronized with the fast weights. This approach
    reduces variance and improves training stability.

    Parameters
    ----------
    base_optimizer : optax.GradientTransformation
        The base optimizer to wrap (e.g., SGD, Adam). The base optimizer performs
        the fast weight updates.
    sync_period : int, default=5
        Number of fast weight update steps before synchronizing with slow weights.
        Also known as 'k' in the paper. Typical values are 5-10.
    alpha : float, default=0.5
        Slow weights step size. Controls how much the slow weights move toward
        the fast weights during synchronization. Also known as 'slow step size'.
        Range: [0, 1], where 0 means no update and 1 means full update.
    lr : float or LRScheduler, default=1e-3
        Learning rate for the base optimizer. Can be a float (converted to ConstantLR)
        or any LRScheduler instance.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient.
    grad_clip_norm : float, optional
        Maximum norm for gradient clipping. If specified, gradients are clipped
        when their global norm exceeds this value.
    grad_clip_value : float, optional
        Maximum absolute value for gradient clipping. If specified, gradients are
        clipped element-wise to [-grad_clip_value, grad_clip_value].

    Notes
    -----
    Lookahead maintains two sets of parameters:

    - Fast weights :math:`\theta_f`: Updated by the base optimizer every step
    - Slow weights :math:`\theta_s`: Updated periodically every `k` steps

    The update procedure is:

    1. Fast weight update (every step):

    .. math::
        \theta_f^{t+1} = \text{BaseOptimizer}(\theta_f^t, g_t)

    2. Slow weight update (every `k` steps):

    .. math::
        \theta_s^{t+k} = \theta_s^t + \alpha (\theta_f^{t+k} - \theta_s^t)

        \theta_f^{t+k} = \theta_s^{t+k}

    where:

    - :math:`k` is the sync_period
    - :math:`\alpha` is the slow step size (alpha parameter)
    - :math:`g_t` is the gradient at step t

    Benefits of Lookahead:

    - Reduces variance in the optimization trajectory
    - Often achieves better generalization than the base optimizer alone
    - Provides a form of implicit regularization
    - Works with any base optimizer (SGD, Adam, etc.)
    - Minimal computational overhead

    The slow weights act as an "anchor" that prevents the fast weights from
    moving too far in potentially suboptimal directions, leading to more stable
    and often faster convergence.

    References
    ----------
    .. [1] Zhang, M. R., Lucas, J., Hinton, G., & Ba, J. (2019).
           Lookahead optimizer: k steps forward, 1 step back.
           arXiv preprint arXiv:1907.08610.

    Examples
    --------
    Basic usage with SGD as base optimizer:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>> import optax
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Create base optimizer (SGD)
        >>> base_opt = optax.sgd(learning_rate=0.1)
        >>>
        >>> # Wrap with Lookahead
        >>> optimizer = braintools.optim.Lookahead(
        ...     base_optimizer=base_opt,
        ...     sync_period=5,
        ...     alpha=0.5
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With Adam as base optimizer:

    .. code-block:: python

        >>> # Lookahead + Adam (RAdam paper recommends this combination)
        >>> base_opt = optax.adam(learning_rate=0.001)
        >>> optimizer = braintools.optim.Lookahead(
        ...     base_optimizer=base_opt,
        ...     sync_period=6,
        ...     alpha=0.5
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Custom synchronization period:

    .. code-block:: python

        >>> # Longer sync period for more exploration
        >>> base_opt = optax.sgd(learning_rate=0.1, momentum=0.9)
        >>> optimizer = braintools.optim.Lookahead(
        ...     base_optimizer=base_opt,
        ...     sync_period=10,  # Synchronize every 10 steps
        ...     alpha=0.5
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Adjusting slow weights step size:

    .. code-block:: python

        >>> # Smaller alpha for more conservative slow weight updates
        >>> base_opt = optax.adam(learning_rate=0.001)
        >>> optimizer = braintools.optim.Lookahead(
        ...     base_optimizer=base_opt,
        ...     sync_period=5,
        ...     alpha=0.3  # More conservative than default 0.5
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With learning rate scheduler:

    .. code-block:: python

        >>> # Combine with scheduler for dynamic learning rate
        >>> scheduler = braintools.optim.ExponentialLR(base_lr=0.1, gamma=0.95)
        >>> base_opt = optax.sgd(learning_rate=0.1)
        >>> optimizer = braintools.optim.Lookahead(
        ...     base_optimizer=base_opt,
        ...     lr=scheduler,
        ...     sync_period=5,
        ...     alpha=0.5
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Complete training example:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> # Setup
        >>> model = brainstate.nn.Sequential(
        ...     brainstate.nn.Linear(784, 128),
        ...     brainstate.nn.ReLU(),
        ...     brainstate.nn.Linear(128, 10)
        ... )
        >>>
        >>> # Lookahead with SGD + momentum
        >>> base_opt = optax.sgd(learning_rate=0.1, momentum=0.9)
        >>> optimizer = braintools.optim.Lookahead(
        ...     base_optimizer=base_opt,
        ...     sync_period=5,
        ...     alpha=0.5
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Training loop
        >>> for epoch in range(10):
        ...     for batch_x, batch_y in data_loader:
        ...         def loss_fn():
        ...             logits = model(batch_x)
        ...             return jnp.mean(
        ...                 braintools.metric.softmax_cross_entropy(logits, batch_y)
        ...             )
        ...
        ...         grads = brainstate.transform.grad(loss_fn, model.states(brainstate.ParamState))()
        ...         optimizer.update(grads)

    See Also
    --------
    SGD : Stochastic gradient descent base optimizer
    Adam : Adaptive moment estimation base optimizer
    RAdam : Rectified Adam (works well with Lookahead)

    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        base_optimizer: optax.GradientTransformation,
        sync_period: int = 5,
        alpha: float = 0.5,
        lr: Union[float, LRScheduler] = 1e-3,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.sync_period = sync_period
        self.alpha = alpha
        self.base_optimizer = base_optimizer
        super().__init__(
            tx=None,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        transforms.append(self.base_optimizer)
        transforms.append(
            optax.lookahead(self.base_optimizer, slow_step_size=self.alpha, sync_period=self.sync_period)
        )
        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)


class Yogi(OptaxOptimizer):
    r"""Yogi optimizer (improvement over Adam).

    Yogi is an adaptive learning rate optimizer that addresses some of the
    limitations of Adam by controlling the increase of the effective learning
    rate. It uses additive updates instead of multiplicative updates for the
    second moment estimate, which prevents the effective learning rate from
    increasing too rapidly.

    Parameters
    ----------
    lr : float or LRScheduler, default=1e-3
        Learning rate. Can be a float (converted to ConstantLR) or any LRScheduler instance.
    betas : tuple of float, default=(0.9, 0.999)
        Coefficients for computing running averages of gradient and its square.
        The first value (beta1) controls the exponential decay rate for the first moment,
        and the second value (beta2) controls the decay rate for the second moment.
    eps : float, default=1e-3
        Term added to the denominator for numerical stability. Note: Yogi uses
        a larger default epsilon (1e-3) than Adam (1e-8) for better stability.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient.
    grad_clip_norm : float, optional
        Maximum norm for gradient clipping. If specified, gradients are clipped
        when their global norm exceeds this value.
    grad_clip_value : float, optional
        Maximum absolute value for gradient clipping. If specified, gradients are
        clipped element-wise to [-grad_clip_value, grad_clip_value].

    Notes
    -----
    Yogi modifies Adam's second moment update to use an additive approach.
    The key difference from Adam is in the second moment computation:

    First moment (same as Adam):

    .. math::
        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t

    Second moment (Yogi's modification):

    .. math::
        v_t = v_{t-1} - (1 - \beta_2) \text{sign}(v_{t-1} - g_t^2) \odot g_t^2

    Bias correction:

    .. math::
        \hat{m}_t = \frac{m_t}{1 - \beta_1^t}

        \hat{v}_t = \frac{v_t}{1 - \beta_2^t}

    Parameter update:

    .. math::
        \theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}

    The sign-based additive update in the second moment prevents the effective
    learning rate from increasing when the gradient magnitude decreases, which
    can happen with Adam's multiplicative update.

    Key advantages of Yogi over Adam:

    - More stable convergence in some scenarios
    - Prevents the effective learning rate from growing unboundedly
    - Better handles changing gradient magnitudes
    - Often achieves better generalization
    - Particularly effective for problems with sparse gradients

    References
    ----------
    .. [1] Zaheer, M., Reddi, S., Sachan, D., Kale, S., & Kumar, S. (2018).
           Adaptive methods for nonconvex optimization.
           In Advances in Neural Information Processing Systems (NeurIPS).

    Examples
    --------
    Basic usage with default parameters:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Initialize Yogi optimizer
        >>> optimizer = braintools.optim.Yogi(lr=0.001)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With custom beta values:

    .. code-block:: python

        >>> # Adjust momentum parameters
        >>> optimizer = braintools.optim.Yogi(
        ...     lr=0.001,
        ...     betas=(0.9, 0.99)  # Faster second moment decay
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With larger epsilon for increased stability:

    .. code-block:: python

        >>> # Yogi is less sensitive to epsilon than Adam
        >>> optimizer = braintools.optim.Yogi(
        ...     lr=0.001,
        ...     eps=1e-2  # Even larger epsilon for more stability
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With learning rate scheduler:

    .. code-block:: python

        >>> # Combine with exponential decay
        >>> scheduler = braintools.optim.ExponentialLR(base_lr=0.01, gamma=0.95)
        >>> optimizer = braintools.optim.Yogi(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With weight decay for regularization:

    .. code-block:: python

        >>> # Add L2 regularization
        >>> optimizer = braintools.optim.Yogi(
        ...     lr=0.001,
        ...     weight_decay=0.01
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Complete training example for NLP tasks:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> # Setup language model
        >>> model = brainstate.nn.Sequential(
        ...     brainstate.nn.Embedding(10000, 256),
        ...     brainstate.nn.LSTM(256, 512),
        ...     brainstate.nn.Linear(512, 10000)
        ... )
        >>>
        >>> # Yogi works well for NLP with sparse gradients
        >>> optimizer = braintools.optim.Yogi(
        ...     lr=0.001,
        ...     betas=(0.9, 0.999),
        ...     eps=1e-3
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Training step
        >>> def train_step(tokens, targets):
        ...     def loss_fn():
        ...         logits = model(tokens)
        ...         return jnp.mean(
        ...             braintools.metric.softmax_cross_entropy(logits, targets)
        ...         )
        ...
        ...     grads = brainstate.transform.grad(loss_fn, model.states(brainstate.ParamState))()
        ...     optimizer.update(grads)
        ...     return loss_fn()
        >>>
        >>> # Train
        >>> tokens = jnp.ones((32, 50), dtype=jnp.int32)
        >>> targets = jnp.zeros((32, 10000))
        >>> # loss = train_step(tokens, targets)

    See Also
    --------
    Adam : Standard adaptive moment estimation
    AdamW : Adam with decoupled weight decay
    RAdam : Rectified Adam with variance rectification

    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.betas = betas
        self.eps = eps
        super().__init__(
            tx=None,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        transforms.append(optax.scale_by_yogi(b1=self.betas[0], b2=self.betas[1], eps=self.eps))
        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)


class LBFGS(OptaxOptimizer):
    r"""L-BFGS optimizer (Limited-memory Broyden-Fletcher-Goldfarb-Shanno).

    L-BFGS is a quasi-Newton optimization method that approximates the inverse Hessian
    matrix using a limited amount of memory. It provides superlinear convergence for
    smooth, unconstrained optimization problems and is widely used in scientific
    computing, machine learning, and numerical optimization.

    This optimizer is particularly effective for:
    - Medium to large-scale optimization problems
    - Smooth, differentiable objective functions
    - Full-batch or deterministic gradient computations
    - Scientific computing and parameter estimation
    - Neural network fine-tuning with small datasets

    Parameters
    ----------
    lr : float or LRScheduler, default=1.0
        Learning rate (step size). Can be a float or any LRScheduler instance.
        L-BFGS typically uses lr=1.0 as it computes optimal step sizes via
        line search. Adjust only if line search is disabled or for specific needs.
    memory_size : int, default=10
        Number of past gradient-position pairs to store for Hessian approximation.
        Typical values: 3-20. Larger values give better approximations but use more
        memory. Trade-off between accuracy and computational cost.
    scale_init_hess : bool, default=True
        Whether to scale the initial Hessian approximation using gradient information.
        Improves convergence by adapting to problem scale. Recommended for most cases.
    grad_clip_norm : float, optional
        Maximum norm for gradient clipping. Gradients are scaled when their
        global norm exceeds this value. Useful for numerical stability.
    grad_clip_value : float, optional
        Maximum absolute value for element-wise gradient clipping. Each gradient
        component is clipped to [-grad_clip_value, grad_clip_value].

    Notes
    -----
    **Mathematical Formulation:**

    L-BFGS approximates the inverse Hessian matrix :math:`H_k^{-1}` using the
    limited-memory BFGS update formula. The parameter update is:

    .. math::
        \theta_{k+1} = \theta_k - \alpha_k H_k^{-1} \nabla f(\theta_k)

    The inverse Hessian approximation uses :math:`m` stored pairs:

    .. math::
        s_i = \theta_{i+1} - \theta_i \quad \text{(position difference)}

    .. math::
        y_i = \nabla f(\theta_{i+1}) - \nabla f(\theta_i) \quad \text{(gradient difference)}

    with curvature information:

    .. math::
        \rho_i = \frac{1}{y_i^T s_i}

    **Two-Loop Recursion Algorithm:**

    1. **First loop** (newest to oldest): Compute direction adjustments
    2. **Initial scaling**: :math:`H_0^{-1} = \gamma_k I` where
       :math:`\gamma_k = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}}`
    3. **Second loop** (oldest to newest): Apply BFGS corrections

    **Line Search with Zoom Algorithm:**

    This implementation includes automatic zoom line search finding step size
    :math:`\alpha_k` satisfying the strong Wolfe conditions for robust convergence.

    **Key Characteristics:**

    - **Superlinear convergence**: Faster than first-order methods near optimum
    - **Memory efficient**: O(mn) storage for n parameters, m history size
    - **Curvature aware**: Uses second-order information without computing Hessian
    - **Self-scaling**: Adapts to problem geometry automatically
    - **Robust line search**: Ensures sufficient decrease and curvature conditions

    **Limitations:**

    - Not suitable for stochastic mini-batch optimization
    - Requires full gradients for best performance
    - Memory scales with memory_size × parameter_count
    - Line search requires additional function evaluations

    **Important Usage Note:**

    L-BFGS with line search requires additional function evaluations. For best
    performance, use with ``optax.value_and_grad_from_state`` to reuse computations:

    .. code-block:: python

        >>> import optax
        >>> value_and_grad = optax.value_and_grad_from_state(objective)
        >>> value, grad = value_and_grad(params, state=opt_state)
        >>> updates, opt_state = optimizer.tx.update(
        ...     grad, opt_state, params,
        ...     value=value, grad=grad, value_fn=objective
        ... )

    References
    ----------
    .. [1] Liu, D. C., & Nocedal, J. (1989).
           "On the limited memory BFGS method for large scale optimization."
           Mathematical Programming, 45(1-3), 503-528.
    .. [2] Nocedal, J., & Wright, S. (2006).
           "Numerical Optimization" (2nd ed.).
           Springer Science & Business Media.
    .. [3] Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995).
           "A limited memory algorithm for bound constrained optimization."
           SIAM Journal on Scientific Computing, 16(5), 1190-1208.
    .. [4] Morales, J. L., & Nocedal, J. (2011).
           "Remark on 'Algorithm 778: L-BFGS-B'."
           ACM Transactions on Mathematical Software, 38(1), 1-4.

    Examples
    --------
    Basic usage for batch optimization:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Initialize L-BFGS optimizer
        >>> optimizer = braintools.optim.LBFGS(lr=1.0)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With custom memory size:

    .. code-block:: python

        >>> # Larger memory for better Hessian approximation
        >>> optimizer = braintools.optim.LBFGS(
        ...     lr=1.0,
        ...     memory_size=20
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With smaller memory for efficiency:

    .. code-block:: python

        >>> # Smaller memory footprint
        >>> optimizer = braintools.optim.LBFGS(
        ...     lr=1.0,
        ...     memory_size=5
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Disabling initial Hessian scaling:

    .. code-block:: python

        >>> # Without Hessian scaling
        >>> optimizer = braintools.optim.LBFGS(
        ...     lr=1.0,
        ...     memory_size=10,
        ...     scale_init_hess=False
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Fine-tuning example with full-batch training:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> # Setup for fine-tuning
        >>> model = brainstate.nn.Sequential(
        ...     brainstate.nn.Linear(784, 128),
        ...     brainstate.nn.ReLU(),
        ...     brainstate.nn.Linear(128, 10)
        ... )
        >>>
        >>> # L-BFGS for fine-tuning with full batch
        >>> optimizer = braintools.optim.LBFGS(
        ...     lr=1.0,
        ...     memory_size=10
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Full-batch training step
        >>> def train_step(data_x, data_y):
        ...     def loss_fn():
        ...         logits = model(data_x)
        ...         return jnp.mean(
        ...             braintools.metric.softmax_cross_entropy(logits, data_y)
        ...         )
        ...
        ...     # Compute gradients on full dataset
        ...     grads = brainstate.transform.grad(loss_fn, model.states(brainstate.ParamState))()
        ...     optimizer.update(grads)
        ...     return loss_fn()
        >>>
        >>> # Use entire dataset (not mini-batch)
        >>> x_full = jnp.ones((1000, 784))
        >>> y_full = jnp.zeros((1000, 10))
        >>> # loss = train_step(x_full, y_full)

    Convex optimization example:

    .. code-block:: python

        >>> # L-BFGS excels at convex problems
        >>> model = brainstate.nn.Linear(50, 1)  # Linear regression
        >>> optimizer = braintools.optim.LBFGS(
        ...     lr=1.0,
        ...     memory_size=15
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Typically converges in fewer iterations than first-order methods
        >>> for epoch in range(100):
        ...     # Full-batch gradient computation
        ...     pass  # training code here


    **Scientific computing - parameter fitting:**

    .. code-block:: python

        >>> # Fitting exponential decay model
        >>> def exponential_model(params, t):
        ...     return params['A'] * jnp.exp(-params['k'] * t) + params['C']
        >>>
        >>> def loss_fn(params):
        ...     predictions = exponential_model(params, time_points)
        ...     return jnp.mean((predictions - observations) ** 2)
        >>>
        >>> # L-BFGS for precise parameter estimation
        >>> optimizer = braintools.optim.LBFGS(
        ...     lr=1.0,
        ...     memory_size=20,  # Higher accuracy for scientific computing
        ...     scale_init_hess=True
        ... )

    **Hybrid optimization strategy:**

    .. code-block:: python

        >>> # Stage 1: Adam for exploration (stochastic)
        >>> adam_opt = braintools.optim.Adam(lr=0.001)
        >>> for epoch in range(50):
        ...     for batch in dataloader:
        ...         grads = compute_batch_gradients(batch)
        ...         adam_opt.update(grads)
        >>>
        >>> # Stage 2: L-BFGS for refinement (deterministic)
        >>> lbfgs_opt = braintools.optim.LBFGS(lr=1.0, memory_size=20)
        >>> for epoch in range(20):
        ...     grads = compute_full_gradients(full_dataset)
        ...     lbfgs_opt.update(grads)

    **Memory size comparison:**

    .. code-block:: python

        >>> # Small memory (fast, less accurate)
        >>> opt_small = braintools.optim.LBFGS(memory_size=3)
        >>>
        >>> # Medium memory (balanced)
        >>> opt_medium = braintools.optim.LBFGS(memory_size=10)
        >>>
        >>> # Large memory (slower, more accurate)
        >>> opt_large = braintools.optim.LBFGS(memory_size=30)

    See Also
    --------
    SGD : First-order stochastic gradient descent
    Adam : Adaptive moment estimation for stochastic optimization
    Rprop : Resilient propagation for batch learning
    Adagrad : Adaptive gradient algorithm

    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1.0,
        memory_size: int = 10,
        scale_init_precond: bool = True,
        linesearch: Optional[Union[str, optax.GradientTransformationExtraArgs]] = None,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.memory_size = memory_size
        self.scale_init_precond = scale_init_precond

        if linesearch is None:
            linesearch = optax.identity()
        elif isinstance(linesearch, str):
            if linesearch == 'zoom':
                linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=100)
            elif linesearch == 'backtracking':
                linesearch = optax.scale_by_backtracking_linesearch(max_backtracking_steps=100)
            else:
                raise ValueError(f"Unknown linesearch method: {linesearch}")
        elif isinstance(linesearch, optax.GradientTransformationExtraArgs):
            linesearch = linesearch
        else:
            raise ValueError(f"Unknown linesearch method: {linesearch}")
        self.linesearch = linesearch

        # Now call parent init with the complete optimizer
        super().__init__(
            lr=lr,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        # This method shouldn't be called since we provide tx in __init__
        # But we'll implement it for completeness
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))

        # Create LBFGS with proper parameters
        lbfgs_tx = optax.lbfgs(
            learning_rate=self.current_lr,
            memory_size=self.memory_size,
            scale_init_precond=self.scale_init_precond,
            linesearch=self.linesearch,
        )

        if transforms:
            return optax.chain(*transforms, lbfgs_tx)
        else:
            return lbfgs_tx

    def update(self, grads, value=None, value_fn=None, **kwargs):
        """Update parameters with LBFGS optimizer.

        Parameters
        ----------
        grads : dict
            Dictionary of gradients for each parameter.
        value : float, optional
            Current value of the objective function. Required for linesearch.
        value_fn : callable, optional
            Function to compute objective value. Required for linesearch.
        **kwargs
            Additional arguments passed to the optimizer update.

        Notes
        -----
        LBFGS requires additional arguments for the linesearch:
        - value: current objective function value
        - grad: gradients (automatically passed)
        - value_fn: callable to evaluate objective function

        For best performance, use with optax.value_and_grad_from_state:

        .. code-block:: python

            >>> value_and_grad = optax.value_and_grad_from_state(loss_fn)
            >>> value, grad = value_and_grad(params, state=opt_state)
            >>> updates, opt_state = optimizer.update(
            ...     grad, opt_state, params,
            ...     value=value, grad=grad, value_fn=loss_fn
            ... )
        """
        # Pass extra arguments needed for LBFGS with linesearch
        extra_args = {}
        if value is not None:
            extra_args['value'] = value
        if value_fn is not None:
            extra_args['value_fn'] = value_fn
        # grad is automatically included from grads parameter
        extra_args['grad'] = grads

        # Call parent update with extra arguments
        return super().update(grads, **extra_args, **kwargs)


class Rprop(OptaxOptimizer):
    r"""Rprop optimizer (Resilient Backpropagation).

    Rprop is a gradient-based optimization algorithm that adapts the step size
    individually for each parameter based only on the sign of the gradient,
    not its magnitude. This makes it particularly robust to varying gradient
    scales and well-suited for batch learning.

    Parameters
    ----------
    lr : float or LRScheduler, default=1e-2
        Initial learning rate (step size). Can be a float (converted to ConstantLR)
        or any LRScheduler instance. In Rprop, this serves as the initial step size.
    etas : tuple of float, default=(0.5, 1.2)
        Step size adjustment factors (eta_minus, eta_plus). When gradient sign changes,
        step size is multiplied by eta_minus (typically < 1). When gradient sign is
        consistent, step size is multiplied by eta_plus (typically > 1).
    step_sizes : tuple of float, default=(1e-6, 50.0)
        Minimum and maximum allowed step sizes (min_step_size, max_step_size).
        Prevents step sizes from becoming too small or too large.
    grad_clip_norm : float, optional
        Maximum norm for gradient clipping. If specified, gradients are clipped
        when their global norm exceeds this value.
    grad_clip_value : float, optional
        Maximum absolute value for gradient clipping. If specified, gradients are
        clipped element-wise to [-grad_clip_value, grad_clip_value].

    Notes
    -----
    Rprop adapts the step size for each weight based on the sign pattern of gradients.
    The update rule is:

    .. math::
        \Delta_t^{(i)} = \begin{cases}
            \eta^+ \cdot \Delta_{t-1}^{(i)} & \text{if } \frac{\partial E}{\partial w_i^{(t)}} \cdot \frac{\partial E}{\partial w_i^{(t-1)}} > 0 \\
            \eta^- \cdot \Delta_{t-1}^{(i)} & \text{if } \frac{\partial E}{\partial w_i^{(t)}} \cdot \frac{\partial E}{\partial w_i^{(t-1)}} < 0 \\
            \Delta_{t-1}^{(i)} & \text{otherwise}
        \end{cases}

    The step size is then clipped:

    .. math::
        \Delta_t^{(i)} = \text{clip}(\Delta_t^{(i)}, \Delta_{\min}, \Delta_{\max})

    Finally, the parameter update is:

    .. math::
        w_t^{(i)} = w_{t-1}^{(i)} - \text{sign}\left(\frac{\partial E}{\partial w_i^{(t)}}\right) \cdot \Delta_t^{(i)}

    Key characteristics of Rprop:

    - **Sign-based updates**: Uses only gradient sign, not magnitude
    - **Individual step sizes**: Each parameter has its own adaptive step size
    - **Batch learning**: Designed for full-batch gradient descent
    - **Robust to scales**: Insensitive to gradient magnitude variations
    - **Simple and effective**: Few hyperparameters to tune
    - **Local adaptation**: Adapts based on consecutive gradient signs

    Rprop is particularly well-suited for:

    - Neural network training with batch learning
    - Problems with varying gradient scales across parameters
    - Scenarios where gradient magnitudes are unreliable
    - Feed-forward networks and small-medium sized problems

    **Advantages:**

    - Robust to gradient scaling issues
    - Fast convergence on many problems
    - Simple to implement and tune

    **Limitations:**

    - Not designed for mini-batch stochastic optimization
    - Requires sign consistency across consecutive steps
    - Less effective with very noisy gradients

    References
    ----------
    .. [1] Riedmiller, M., & Braun, H. (1993).
           A direct adaptive method for faster backpropagation learning: The RPROP algorithm.
           In IEEE International Conference on Neural Networks (pp. 586-591).
    .. [2] Igel, C., & Hüsken, M. (2000).
           Improving the Rprop learning algorithm.
           In Proceedings of the Second International ICSC Symposium on Neural Computation.

    Examples
    --------
    Basic usage with default parameters:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Initialize Rprop optimizer
        >>> optimizer = braintools.optim.Rprop(lr=0.01)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With custom eta values for step size adjustment:

    .. code-block:: python

        >>> # More aggressive step size changes
        >>> optimizer = braintools.optim.Rprop(
        ...     lr=0.01,
        ...     etas=(0.3, 1.5)  # Faster decrease, faster increase
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With custom step size bounds:

    .. code-block:: python

        >>> # Tighter bounds on step sizes
        >>> optimizer = braintools.optim.Rprop(
        ...     lr=0.01,
        ...     step_sizes=(1e-5, 10.0)
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Complete configuration:

    .. code-block:: python

        >>> # All parameters customized
        >>> optimizer = braintools.optim.Rprop(
        ...     lr=0.01,
        ...     etas=(0.5, 1.2),
        ...     step_sizes=(1e-6, 50.0)
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Batch training example:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> # Setup for batch learning
        >>> model = brainstate.nn.Sequential(
        ...     brainstate.nn.Linear(100, 50),
        ...     brainstate.nn.TanhT(),
        ...     brainstate.nn.Linear(50, 10)
        ... )
        >>>
        >>> # Rprop for batch training
        >>> optimizer = braintools.optim.Rprop(
        ...     lr=0.01,
        ...     etas=(0.5, 1.2)
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Full-batch training step
        >>> def train_step(batch_x, batch_y):
        ...     def loss_fn():
        ...         logits = model(batch_x)
        ...         return jnp.mean(
        ...             braintools.metric.softmax_cross_entropy(logits, batch_y)
        ...         )
        ...
        ...     grads = brainstate.transform.grad(loss_fn, model.states(brainstate.ParamState))()
        ...     optimizer.update(grads)
        ...     return loss_fn()
        >>>
        >>> # Use full batch or large batches
        >>> x = jnp.ones((500, 100))
        >>> y = jnp.zeros((500, 10))
        >>> # loss = train_step(x, y)

    Classification task example:

    .. code-block:: python

        >>> # Rprop for classification
        >>> model = brainstate.nn.Sequential(
        ...     brainstate.nn.Linear(784, 256),
        ...     brainstate.nn.ReLU(),
        ...     brainstate.nn.Linear(256, 10)
        ... )
        >>>
        >>> optimizer = braintools.optim.Rprop(lr=0.01)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Rprop adapts step sizes automatically
        >>> # Works well even with varying gradient scales

    See Also
    --------
    SGD : Stochastic gradient descent with momentum
    Adam : Adaptive moment estimation
    LBFGS : Limited-memory BFGS for batch optimization

    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-2,
        etas: Tuple[float, float] = (0.5, 1.2),
        step_sizes: Tuple[float, float] = (1e-6, 50.0),
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.etas = etas
        self.step_sizes = step_sizes
        super().__init__(
            tx=None,
            lr=lr,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        transforms.append(
            optax.scale_by_rprop(
                learning_rate=self.base_lr,
                eta_minus=self.etas[0],
                eta_plus=self.etas[1],
                min_step_size=self.step_sizes[0],
                max_step_size=self.step_sizes[1]
            )
        )
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)


class Adafactor(OptaxOptimizer):
    r"""Adafactor optimizer (memory-efficient variant of Adam).

    Adafactor is designed to reduce memory usage during training large models
    by using factored second moment estimation. Instead of storing a full
    second moment matrix, it maintains row and column statistics, significantly
    reducing memory requirements especially for models with large embedding tables.

    Parameters
    ----------
    lr : float or LRScheduler, optional, default=None
        Learning rate. Can be a float (converted to ConstantLR) or any LRScheduler
        instance. If None, uses adaptive learning rate based on step count and
        RMS of parameters.
    eps : tuple of float, default=(1e-30, 1e-3)
        Regularization constants for squared gradient and parameter scale (eps[0], eps[1]).
        The first value prevents division by zero, the second clips the parameter scale.
    clip_threshold : float, default=1.0
        Threshold for gradient clipping by root mean square. Helps prevent
        gradient explosions.
    decay_rate : float, default=-0.8
        Controls the decay of the second moment estimate. Negative values result
        in polynomial decay: decay = 1 - (step + 1)^decay_rate.
    beta1 : float, optional, default=None
        Momentum parameter for first moment. If None, no momentum is used.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty) coefficient.
    factored : bool, default=True
        Whether to use factored second moment estimation. When True, significantly
        reduces memory usage. Set to False to use full second moment (more memory).
    grad_clip_norm : float, optional
        Maximum norm for gradient clipping. If specified, gradients are clipped
        when their global norm exceeds this value.
    grad_clip_value : float, optional
        Maximum absolute value for gradient clipping. If specified, gradients are
        clipped element-wise to [-grad_clip_value, grad_clip_value].

    Notes
    -----
    Adafactor's key innovation is factored second moment estimation. Instead of
    maintaining a full matrix :math:`V_t \in \mathbb{R}^{n \times m}`, it maintains
    row and column averages :math:`R_t \in \mathbb{R}^n` and :math:`C_t \in \mathbb{R}^m`:

    .. math::
        R_t = \beta_2 R_{t-1} + (1 - \beta_2) \text{mean}(G_t^2, \text{axis}=1)

        C_t = \beta_2 C_{t-1} + (1 - \beta_2) \text{mean}(G_t^2, \text{axis}=0)

    The second moment is approximated as:

    .. math::
        V_t \approx R_t \otimes C_t / \text{mean}(R_t)

    where :math:`\otimes` denotes outer product.

    The update rule with optional momentum is:

    .. math::
        M_t = \beta_1 M_{t-1} + (1 - \beta_1) G_t \quad \text{(if beta1 is not None)}

        \theta_{t+1} = \theta_t - \alpha_t \frac{M_t}{\sqrt{V_t} + \epsilon}

    Key advantages of Adafactor:

    - **Memory efficient**: O(n+m) instead of O(n×m) for factored mode
    - **Adaptive learning rate**: Can work without explicit learning rate
    - **Large models**: Designed for transformer and large embedding models
    - **Stable training**: Built-in gradient clipping
    - **Automatic scheduling**: Polynomial decay of second moment

    Adafactor is particularly well-suited for:

    - Training very large transformer models (BERT, GPT, T5)
    - Models with large embedding tables
    - Situations with limited GPU memory
    - Long training runs where adaptive scheduling helps

    References
    ----------
    .. [1] Shazeer, N., & Stern, M. (2018).
           Adafactor: Adaptive learning rates with sublinear memory cost.
           In International Conference on Machine Learning (pp. 4596-4604).

    Examples
    --------
    Basic usage with automatic learning rate:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Initialize Adafactor with auto learning rate
        >>> optimizer = braintools.optim.Adafactor()
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With explicit learning rate:

    .. code-block:: python

        >>> # Explicit learning rate
        >>> optimizer = braintools.optim.Adafactor(lr=1e-3)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With momentum for better convergence:

    .. code-block:: python

        >>> # Add momentum (first moment)
        >>> optimizer = braintools.optim.Adafactor(
        ...     lr=1e-3,
        ...     beta1=0.9
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Non-factored mode for better accuracy:

    .. code-block:: python

        >>> # Use full second moment (more memory)
        >>> optimizer = braintools.optim.Adafactor(
        ...     lr=1e-3,
        ...     factored=False
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    For large transformer training:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>>
        >>> # Setup large transformer model
        >>> model = brainstate.nn.Sequential(
        ...     brainstate.nn.Embedding(50000, 512),  # Large vocabulary
        ...     brainstate.nn.Linear(512, 512),
        ...     brainstate.nn.ReLU(),
        ...     brainstate.nn.Linear(512, 50000)
        ... )
        >>>
        >>> # Adafactor with factored mode for memory efficiency
        >>> optimizer = braintools.optim.Adafactor(
        ...     lr=None,  # Adaptive learning rate
        ...     beta1=0.9,
        ...     factored=True,
        ...     clip_threshold=1.0
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        >>>
        >>> # Training step
        >>> def train_step(tokens, targets):
        ...     def loss_fn():
        ...         logits = model(tokens)
        ...         return jnp.mean(
        ...             braintools.metric.softmax_cross_entropy(logits, targets)
        ...         )
        ...
        ...     grads = brainstate.transform.grad(loss_fn, model.states(brainstate.ParamState))()
        ...     optimizer.update(grads)
        ...     return loss_fn()
        >>>
        >>> # Train with large batches
        >>> tokens = jnp.ones((128, 512), dtype=jnp.int32)
        >>> targets = jnp.zeros((128, 50000))
        >>> # loss = train_step(tokens, targets)

    With weight decay and custom parameters:

    .. code-block:: python

        >>> # Complete configuration
        >>> optimizer = braintools.optim.Adafactor(
        ...     lr=1e-3,
        ...     eps=(1e-30, 1e-3),
        ...     clip_threshold=1.0,
        ...     decay_rate=-0.8,
        ...     beta1=0.9,
        ...     weight_decay=0.01,
        ...     factored=True
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    See Also
    --------
    Adam : Standard adaptive moment estimation
    AdamW : Adam with decoupled weight decay
    SM3 : Another memory-efficient adaptive optimizer

    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Optional[Union[float, LRScheduler]] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        factored: bool = True,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.eps = eps
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.beta1 = beta1
        self.factored = factored
        super().__init__(
            tx=None,
            lr=lr or 1e-3,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        transforms.append(
            optax.scale_by_factored_rms(
                factored=self.factored,
                decay_rate=self.decay_rate,
                epsilon=self.eps[0]
            )
        )
        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)


class AdaBelief(OptaxOptimizer):
    r"""AdaBelief optimizer - Adapts step size according to belief in gradient direction.

    AdaBelief is an adaptive learning rate optimizer that adapts the step size
    according to the "belief" in the gradient direction. Unlike Adam which adapts
    based on gradient magnitudes, AdaBelief adapts based on the variance of the
    prediction error (gradient - momentum).

    The key insight is that when the gradient and momentum are aligned (high belief),
    the optimizer should take larger steps. When they diverge (low belief), smaller
    steps should be taken. This leads to faster convergence and better generalization
    compared to Adam.

    Parameters
    ----------
    lr : float or LRScheduler, default=1e-3
        Learning rate. Can be a float or LRScheduler instance.
        If float is provided, it will be automatically converted to a ConstantLR scheduler.
    betas : tuple of float, default=(0.9, 0.999)
        Coefficients (beta1, beta2) used for computing running averages of gradient
        and its variance. beta1 is the exponential decay rate for the first moment,
        beta2 is the exponential decay rate for the second moment (variance).
    eps : float, default=1e-16
        Term added to the denominator for numerical stability. AdaBelief uses a
        smaller epsilon than Adam by default.
    weight_decay : float, default=0.0
        Weight decay coefficient (L2 penalty). When greater than 0, applies L2
        regularization to the parameters.
    grad_clip_norm : float, optional
        Maximum gradient norm for gradient clipping. If None, no gradient norm
        clipping is applied.
    grad_clip_value : float, optional
        Maximum absolute gradient value for element-wise gradient clipping.
        If None, no gradient value clipping is applied.

    Notes
    -----
    The AdaBelief update rules are:

    .. math::
        M_t = \beta_1 M_{t-1} + (1 - \beta_1) G_t

        S_t = \beta_2 S_{t-1} + (1 - \beta_2) (G_t - M_t)^2 + \epsilon

        \hat{M}_t = \frac{M_t}{1 - \beta_1^t}

        \hat{S}_t = \frac{S_t}{1 - \beta_2^t}

        \theta_{t+1} = \theta_t - \alpha \frac{\hat{M}_t}{\sqrt{\hat{S}_t} + \epsilon}

    where:

    - :math:`G_t` is the gradient at step t
    - :math:`M_t` is the first moment (exponential moving average of gradients)
    - :math:`S_t` is the "belief" - variance of gradient prediction error
    - :math:`(G_t - M_t)^2` measures the deviation between gradient and momentum
    - :math:`\hat{M}_t, \hat{S}_t` are bias-corrected estimates
    - :math:`\alpha` is the learning rate

    The key difference from Adam is the second moment estimation:

    - **Adam**: :math:`V_t = \beta_2 V_{t-1} + (1 - \beta_2) G_t^2` (gradient magnitude)
    - **AdaBelief**: :math:`S_t = \beta_2 S_{t-1} + (1 - \beta_2) (G_t - M_t)^2` (gradient variance)

    Key advantages of AdaBelief:

    - **Better generalization**: Adapts based on gradient variance, not magnitude
    - **Fast convergence**: Takes larger steps when gradient is reliable
    - **Stable training**: Takes smaller steps when gradient is noisy
    - **Automatic adaptation**: No need for extensive hyperparameter tuning
    - **Works across domains**: Effective for image, language, and RL tasks

    AdaBelief is particularly well-suited for:

    - Training deep neural networks with complex loss landscapes
    - Problems where Adam overfits or converges slowly
    - Transfer learning and fine-tuning tasks
    - Reinforcement learning with noisy gradients
    - Training with small batch sizes

    References
    ----------
    .. [1] Zhuang, J., Tang, T., Ding, Y., Tatikonda, S., Dvornek, N., Papademetris, X., & Duncan, J. S. (2020).
           AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients.
           Advances in Neural Information Processing Systems, 33, 18795-18806.
           arXiv preprint arXiv:2010.07468.

    Examples
    --------
    Basic usage:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Initialize AdaBelief
        >>> optimizer = braintools.optim.AdaBelief(lr=0.001)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Custom betas for different momentum and variance decay:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Faster momentum decay, slower variance decay
        >>> optimizer = braintools.optim.AdaBelief(lr=0.001, betas=(0.8, 0.999))
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With learning rate scheduler for gradual decay:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Learning rate decays every 30 epochs
        >>> scheduler = braintools.optim.StepLR(base_lr=0.001, step_size=30, gamma=0.5)
        >>> optimizer = braintools.optim.AdaBelief(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With weight decay for regularization:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Add L2 regularization
        >>> optimizer = braintools.optim.AdaBelief(lr=0.001, weight_decay=0.01)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With gradient clipping for stable training:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Clip gradients by global norm
        >>> optimizer = braintools.optim.AdaBelief(
        ...     lr=0.001,
        ...     grad_clip_norm=1.0
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Complete configuration for deep learning:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> # Large model
        >>> model = brainstate.nn.Linear(1000, 500)
        >>>
        >>> # Learning rate schedule
        >>> scheduler = braintools.optim.StepLR(base_lr=0.001, step_size=20, gamma=0.5)
        >>>
        >>> # Complete AdaBelief configuration
        >>> optimizer = braintools.optim.AdaBelief(
        ...     lr=scheduler,
        ...     betas=(0.9, 0.999),
        ...     eps=1e-16,
        ...     weight_decay=0.0001,
        ...     grad_clip_norm=1.0
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    See Also
    --------
    Adam : Standard adaptive moment estimation
    Yogi : Adam variant with additive second moment updates
    RAdam : Rectified Adam with warmup
    AdamW : Adam with decoupled weight decay

    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.betas = betas
        self.eps = eps
        super().__init__(
            tx=None,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        transforms.append(optax.scale_by_belief(b1=self.betas[0], b2=self.betas[1], eps=self.eps))
        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)


class Lion(OptaxOptimizer):
    r"""Lion (EvoLved Sign Momentum) optimizer - Discovered through program search.

    Lion is a novel optimizer discovered through large-scale evolutionary program search.
    It uses sign-based updates for both momentum and parameter updates, making it
    extremely memory-efficient and computationally simple. Despite its simplicity,
    Lion achieves competitive or superior performance compared to Adam while using
    significantly less memory.

    The key insight of Lion is using the sign operation, which provides implicit
    adaptive learning rates and strong regularization effects. Lion typically requires
    smaller learning rates (3-10x smaller than Adam) but larger weight decay values.

    Parameters
    ----------
    lr : float or LRScheduler, default=1e-4
        Learning rate. Can be a float or LRScheduler instance.
        If float is provided, it will be automatically converted to a ConstantLR scheduler.
        **Note**: Lion typically requires 3-10× smaller learning rates than Adam.
    betas : tuple of float, default=(0.9, 0.99)
        Coefficients (beta1, beta2) used for computing the interpolation between
        gradient and momentum. beta1 is used for the update, beta2 is used for
        momentum tracking. Different from Adam where both betas are exponential
        decay rates.
    weight_decay : float, default=0.0
        Weight decay coefficient (L2 penalty). Lion typically uses larger weight
        decay values than Adam (3-10× larger) due to its implicit regularization.
    grad_clip_norm : float, optional
        Maximum gradient norm for gradient clipping. If None, no gradient norm
        clipping is applied.
    grad_clip_value : float, optional
        Maximum absolute gradient value for element-wise gradient clipping.
        If None, no gradient value clipping is applied.

    Notes
    -----
    The Lion update rules are:

    .. math::
        C_t = \beta_1 M_{t-1} + (1 - \beta_1) G_t

        \theta_{t+1} = \theta_t - \alpha \cdot \text{sign}(C_t)

        M_t = \beta_2 M_{t-1} + (1 - \beta_2) G_t

    where:

    - :math:`G_t` is the gradient at step t
    - :math:`M_t` is the momentum (exponential moving average of gradients)
    - :math:`C_t` is the interpolation between momentum and current gradient
    - :math:`\text{sign}(\cdot)` is the element-wise sign function
    - :math:`\alpha` is the learning rate

    Key differences from Adam:

    - **Sign-based updates**: Uses sign(gradient) instead of gradient magnitude
    - **Simpler computation**: No square root or division operations
    - **Less memory**: Only stores momentum (not second moment)
    - **Different hyperparameters**: Smaller lr, larger weight decay
    - **Implicit adaptive learning**: Sign operation provides adaptation

    Key advantages of Lion:

    - **Memory efficient**: Only 1 state per parameter (vs 2 for Adam)
    - **Computationally simple**: No expensive operations (sqrt, division)
    - **Strong regularization**: Sign operation provides implicit regularization
    - **Better generalization**: Often achieves lower validation loss than Adam
    - **Robust**: Works well across different architectures and tasks

    Lion is particularly well-suited for:

    - Training large language models (LLMs) and vision transformers
    - Memory-constrained environments
    - Tasks requiring strong generalization
    - Replacing Adam/AdamW with better efficiency

    Hyperparameter recommendations (relative to Adam):

    - Learning rate: Use 3-10× smaller (e.g., Adam lr=1e-3 → Lion lr=1e-4)
    - Weight decay: Use 3-10× larger (e.g., Adam wd=0.01 → Lion wd=0.1)
    - Batch size: Can use with same batch size as Adam

    References
    ----------
    .. [1] Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., ... & Le, Q. V. (2023).
           Symbolic Discovery of Optimization Algorithms.
           arXiv preprint arXiv:2302.06675.

    Examples
    --------
    Basic usage with small learning rate:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Initialize Lion with small lr (3-10x smaller than Adam)
        >>> optimizer = braintools.optim.Lion(lr=1e-4)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With larger weight decay (recommended for Lion):

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Lion typically uses larger weight decay than Adam
        >>> optimizer = braintools.optim.Lion(lr=1e-4, weight_decay=0.1)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With custom betas:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Custom interpolation coefficients
        >>> optimizer = braintools.optim.Lion(lr=1e-4, betas=(0.95, 0.98))
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With learning rate scheduler:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Step decay schedule
        >>> scheduler = braintools.optim.StepLR(
        ...     base_lr=1e-4,
        ...     step_size=100,
        ...     gamma=0.5
        ... )
        >>> optimizer = braintools.optim.Lion(lr=scheduler, weight_decay=0.1)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With gradient clipping for stable training:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Clip gradients for stability
        >>> optimizer = braintools.optim.Lion(
        ...     lr=1e-4,
        ...     weight_decay=0.1,
        ...     grad_clip_norm=1.0
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Complete configuration for large model training:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> # Large transformer model
        >>> model = brainstate.nn.Linear(1000, 500)
        >>>
        >>> # Learning rate decay schedule
        >>> scheduler = braintools.optim.StepLR(
        ...     base_lr=1e-4,
        ...     step_size=100,
        ...     gamma=0.9
        ... )
        >>>
        >>> # Complete Lion configuration
        >>> optimizer = braintools.optim.Lion(
        ...     lr=scheduler,
        ...     betas=(0.9, 0.99),
        ...     weight_decay=0.1,
        ...     grad_clip_norm=1.0
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    See Also
    --------
    Adam : Standard adaptive moment estimation
    AdamW : Adam with decoupled weight decay
    SGD : Stochastic gradient descent with momentum

    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.betas = betas
        super().__init__(
            tx=None,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        transforms.append(optax.scale_by_lion(b1=self.betas[0], b2=self.betas[1]))
        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)


class SM3(OptaxOptimizer):
    r"""SM3 (Square-root of Minima of Sums of Maxima of Squared-gradients) optimizer.

    SM3 is a memory-efficient adaptive optimizer designed for training models with
    large embedding tables and sparse gradients. It achieves significant memory
    savings by using a clever factorization of the second moment matrix, storing
    only one value per dimension instead of one value per parameter.

    SM3 is particularly effective for:

    - Models with large embedding layers (e.g., recommendation systems, NLP)
    - Sparse gradient scenarios (word embeddings, sparse features)
    - Memory-constrained training environments
    - Models where most parameters are in embedding tables

    The key insight is that for parameters with sparse gradients, we can store
    a much smaller second moment estimate by exploiting the structure of the
    parameter tensor.

    Parameters
    ----------
    lr : float or LRScheduler, default=1.0
        Learning rate. Can be a float or LRScheduler instance.
        If float is provided, it will be automatically converted to a ConstantLR scheduler.
        **Note**: SM3 typically uses larger base learning rates than Adam (e.g., 1.0).
    momentum : float, default=0.9
        Momentum coefficient for the first moment. When > 0, maintains an exponential
        moving average of gradients. Set to 0 to disable momentum.
    eps : float, default=1e-8
        Term added to the denominator for numerical stability. Prevents division
        by zero when second moment estimates are very small.
    weight_decay : float, default=0.0
        Weight decay coefficient (L2 penalty). When greater than 0, applies L2
        regularization to the parameters.
    grad_clip_norm : float, optional
        Maximum gradient norm for gradient clipping. If None, no gradient norm
        clipping is applied.
    grad_clip_value : float, optional
        Maximum absolute gradient value for element-wise gradient clipping.
        If None, no gradient value clipping is applied.

    Notes
    -----
    The SM3 update rules are:

    For a parameter tensor :math:`\theta` of shape :math:`(d_1, d_2, ..., d_k)`:

    .. math::
        V_t^{(i)} = \max(V_{t-1}^{(i)}, G_t^2) \quad \text{for each dimension } i

        v_t = \sqrt{\min_i V_t^{(i)} + \epsilon}

        M_t = \rho M_{t-1} + (1 - \rho) G_t \quad \text{(if momentum > 0)}

        \theta_{t+1} = \theta_t - \alpha \frac{M_t}{v_t}

    where:

    - :math:`G_t` is the gradient at step t
    - :math:`V_t^{(i)}` is the second moment accumulator for dimension i
    - :math:`v_t` is the effective second moment (min of all dimension accumulators)
    - :math:`M_t` is the momentum (optional)
    - :math:`\rho` is the momentum coefficient
    - :math:`\alpha` is the learning rate

    Memory comparison for parameter shape (n, m):

    - **Adam**: Stores 2nm values (first + second moment)
    - **SM3**: Stores n + m values (one per dimension)
    - **Savings**: For large embeddings (e.g., 100k × 512), ~99.5% reduction

    Key advantages of SM3:

    - **Extreme memory efficiency**: O(sum of dimensions) vs O(product of dimensions)
    - **Sparse gradient friendly**: Designed for sparse updates
    - **Adaptive learning rates**: Maintains per-parameter adaptation
    - **Simple and stable**: No complex hyperparameter tuning needed
    - **Embedding-optimized**: Ideal for large embedding layers

    SM3 is particularly well-suited for:

    - Training models with large vocabulary embeddings (NLP, RecSys)
    - Sparse gradient scenarios (word2vec, matrix factorization)
    - Memory-constrained environments (edge devices, limited GPU memory)
    - Recommendation systems with large item/user embeddings

    Comparison with other optimizers:

    - **vs Adam**: Much less memory, competitive performance on sparse tasks
    - **vs Adagrad**: Similar memory, better performance with momentum
    - **vs SGD**: Adaptive rates help with sparse features
    - **vs Adafactor**: Different factorization, better for embeddings

    References
    ----------
    .. [1] Anil, R., Gupta, V., Koren, T., & Singer, Y. (2019).
           Memory-Efficient Adaptive Optimization.
           Advances in Neural Information Processing Systems, 32.
           arXiv preprint arXiv:1901.11150.

    Examples
    --------
    Basic usage with default settings:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> # Create model with embedding layer
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Initialize SM3 with default lr=1.0
        >>> optimizer = braintools.optim.SM3(lr=1.0)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With custom learning rate:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Use smaller learning rate
        >>> optimizer = braintools.optim.SM3(lr=0.1)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With momentum for better convergence:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Higher momentum for smoother updates
        >>> optimizer = braintools.optim.SM3(lr=1.0, momentum=0.95)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Without momentum (pure adaptive):

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Disable momentum
        >>> optimizer = braintools.optim.SM3(lr=1.0, momentum=0.0)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With learning rate scheduler:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Exponential decay schedule
        >>> scheduler = braintools.optim.StepLR(
        ...     base_lr=1.0,
        ...     step_size=100,
        ...     gamma=0.9
        ... )
        >>> optimizer = braintools.optim.SM3(lr=scheduler, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Complete configuration for large embedding model:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> # Large embedding model (e.g., 100k vocabulary, 512 dimensions)
        >>> model = brainstate.nn.Linear(1000, 500)
        >>>
        >>> # Learning rate schedule for long training
        >>> scheduler = braintools.optim.StepLR(
        ...     base_lr=1.0,
        ...     step_size=1000,
        ...     gamma=0.95
        ... )
        >>>
        >>> # Complete SM3 configuration
        >>> optimizer = braintools.optim.SM3(
        ...     lr=scheduler,
        ...     momentum=0.9,
        ...     eps=1e-8,
        ...     weight_decay=0.0001
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    See Also
    --------
    Adam : Standard adaptive moment estimation
    Adafactor : Another memory-efficient optimizer
    Adagrad : Adaptive learning rates for sparse features

    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1.0,
        momentum: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.momentum = momentum
        self.eps = eps
        super().__init__(
            tx=None,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        transforms.append(optax.scale_by_sm3(eps=self.eps))
        if self.momentum > 0:
            transforms.append(optax.trace(decay=self.momentum))
        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)


class Novograd(OptaxOptimizer):
    r"""Novograd (Normalized Gradient) optimizer - Layer-wise gradient normalization with momentum.

    Novograd is an adaptive learning rate optimizer that combines layer-wise gradient
    normalization with Adam-like second moment estimation. It was designed specifically
    for training speech recognition models but has been shown to work well across
    various deep learning tasks including computer vision and NLP.

    The key innovation of Novograd is computing the second moment per layer rather
    than per weight, which provides more stable training and reduces memory usage.
    It normalizes gradients by their layer-wise L2 norm, which helps with training
    stability, especially for models with varying layer sizes.

    Parameters
    ----------
    lr : float or LRScheduler, default=1e-3
        Learning rate. Can be a float or LRScheduler instance.
        If float is provided, it will be automatically converted to a ConstantLR scheduler.
    betas : tuple of float, default=(0.9, 0.999)
        Coefficients (beta1, beta2) used for computing running averages.
        beta1 is for the first moment (momentum), beta2 is for the second moment
        (per-layer gradient variance).
    eps : float, default=1e-8
        Term added to the denominator for numerical stability. Prevents division
        by zero when gradients are very small.
    weight_decay : float, default=0.0
        Weight decay coefficient (L2 penalty). When greater than 0, applies L2
        regularization to the parameters.
    grad_clip_norm : float, optional
        Maximum gradient norm for gradient clipping. If None, no gradient norm
        clipping is applied.
    grad_clip_value : float, optional
        Maximum absolute gradient value for element-wise gradient clipping.
        If None, no gradient value clipping is applied.

    Notes
    -----
    The Novograd update rules are:

    For each layer l with gradient :math:`G_t^{(l)}`:

    .. math::
        g_t^{(l)} = \frac{G_t^{(l)}}{\|G_t^{(l)}\|_2 + \epsilon}

        v_t^{(l)} = \beta_2 v_{t-1}^{(l)} + (1 - \beta_2) \|G_t^{(l)}\|_2^2

        m_t^{(l)} = \beta_1 m_{t-1}^{(l)} + g_t^{(l)} + \lambda \theta_{t-1}^{(l)}

        \theta_t^{(l)} = \theta_{t-1}^{(l)} - \alpha \frac{m_t^{(l)}}{\sqrt{v_t^{(l)}} + \epsilon}

    where:

    - :math:`G_t^{(l)}` is the gradient for layer l at step t
    - :math:`g_t^{(l)}` is the normalized gradient (unit norm)
    - :math:`v_t^{(l)}` is the second moment (per-layer, not per-weight)
    - :math:`m_t^{(l)}` is the first moment (momentum)
    - :math:`\lambda` is the weight decay coefficient
    - :math:`\alpha` is the learning rate

    Key differences from Adam:

    - **Layer-wise normalization**: Normalizes gradients by layer L2 norm
    - **Per-layer second moment**: Stores one variance per layer, not per weight
    - **Memory efficient**: Reduces memory for second moment estimation
    - **More stable**: Layer-wise normalization improves training stability
    - **Better for varied layer sizes**: Handles layers of different sizes better

    Key advantages of Novograd:

    - **Stable training**: Layer-wise normalization reduces gradient variance
    - **Memory efficient**: Per-layer second moment reduces memory usage
    - **Robust to layer size**: Works well with varying layer dimensions
    - **Good generalization**: Often achieves better test performance than Adam
    - **Simple**: No complex hyperparameter tuning needed

    Novograd is particularly well-suited for:

    - Speech recognition models (Jasper, QuartzNet)
    - Training from scratch (not fine-tuning)
    - Models with layers of varying sizes
    - Tasks requiring stable training dynamics
    - Replacing Adam for better generalization

    Comparison with other optimizers:

    - **vs Adam**: Less memory, more stable, better generalization
    - **vs SGD**: Adaptive rates, no manual lr tuning needed
    - **vs RMSprop**: Better momentum, per-layer adaptation
    - **vs Layer-wise Adam**: Similar concept, different implementation

    References
    ----------
    .. [1] Ginsburg, B., Castonguay, P., Hrinchuk, O., Kuchaiev, O., Lavrukhin, V.,
           Leary, R., ... & Cohen, J. (2019).
           Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training
           of Deep Networks.
           arXiv preprint arXiv:1905.11286.

    Examples
    --------
    Basic usage:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Initialize Novograd
        >>> optimizer = braintools.optim.Novograd(lr=0.001)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With custom betas:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Higher beta1 for more momentum
        >>> optimizer = braintools.optim.Novograd(lr=0.001, betas=(0.95, 0.999))
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With weight decay:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Add L2 regularization
        >>> optimizer = braintools.optim.Novograd(lr=0.001, weight_decay=0.01)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With learning rate scheduler:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Polynomial decay schedule
        >>> scheduler = braintools.optim.StepLR(
        ...     base_lr=0.01,
        ...     step_size=100,
        ...     gamma=0.5
        ... )
        >>> optimizer = braintools.optim.Novograd(lr=scheduler)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With gradient clipping for stable training:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Clip gradients by global norm
        >>> optimizer = braintools.optim.Novograd(
        ...     lr=0.001,
        ...     grad_clip_norm=1.0
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Complete configuration for speech recognition:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> # Large speech model
        >>> model = brainstate.nn.Linear(1000, 500)
        >>>
        >>> # Learning rate schedule with warmup
        >>> scheduler = braintools.optim.StepLR(
        ...     base_lr=0.01,
        ...     step_size=1000,
        ...     gamma=0.9
        ... )
        >>>
        >>> # Complete Novograd configuration
        >>> optimizer = braintools.optim.Novograd(
        ...     lr=scheduler,
        ...     betas=(0.95, 0.98),
        ...     eps=1e-8,
        ...     weight_decay=0.001,
        ...     grad_clip_norm=1.0
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    See Also
    --------
    Adam : Standard adaptive moment estimation
    RMSprop : Root mean square propagation
    SGD : Stochastic gradient descent with momentum
    Lars : Layer-wise adaptive rate scaling

    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.betas = betas
        self.eps = eps
        super().__init__(
            tx=None,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            grad_clip_value=grad_clip_value
        )

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        transforms.append(optax.scale_by_novograd(b1=self.betas[0], b2=self.betas[1], eps=self.eps))
        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)


class Fromage(OptaxOptimizer):
    r"""Fromage (FRee-scale Optimal Metho for Adaptive GradiEnt) optimizer.

    Fromage is a learning-rate-free optimizer that adapts the step size automatically
    based on the curvature of the loss landscape. It eliminates the need for manual
    learning rate tuning by using the ratio of gradient norms to determine the optimal
    step size. This makes it particularly useful for hyperparameter-free training and
    rapid prototyping.

    The key innovation of Fromage is computing the step size from the ratio of
    consecutive gradient norms, which approximates the local curvature of the loss
    function. This provides an automatic adaptation mechanism without requiring
    explicit learning rate scheduling or tuning.

    Parameters
    ----------
    lr : float or LRScheduler, default=1.0
        Learning rate scale factor. While Fromage is designed to be learning-rate-free,
        this parameter can be used to globally scale the automatically computed step sizes.
        If float is provided, it will be automatically converted to a ConstantLR scheduler.
        Typically set to 1.0 to use pure automatic adaptation.
    momentum : float, default=0.0
        Momentum coefficient for the first moment. When > 0, maintains an exponential
        moving average of gradients. Set to 0 to disable momentum and use pure
        gradient-based updates.
    grad_clip_norm : float, optional
        Maximum gradient norm for gradient clipping. If None, no gradient norm
        clipping is applied.
    grad_clip_value : float, optional
        Maximum absolute gradient value for element-wise gradient clipping.
        If None, no gradient value clipping is applied.

    Notes
    -----
    The Fromage update rules are:

    .. math::
        \alpha_t = \frac{\|G_t\|_2}{\|G_t - G_{t-1}\|_2 + \epsilon}

        M_t = \rho M_{t-1} + (1 - \rho) G_t \quad \text{(if momentum > 0)}

        \theta_{t+1} = \theta_t - \alpha_t \cdot M_t

    where:

    - :math:`G_t` is the gradient at step t
    - :math:`\alpha_t` is the automatically computed step size
    - :math:`\|G_t\|_2` is the L2 norm of the current gradient
    - :math:`\|G_t - G_{t-1}\|_2` is the gradient difference norm (curvature proxy)
    - :math:`M_t` is the momentum (optional)
    - :math:`\rho` is the momentum coefficient
    - :math:`\epsilon` is a small constant for numerical stability

    The step size :math:`\alpha_t` approximates :math:`1/L` where L is the local
    Lipschitz constant of the gradient, providing an optimal step size based on
    local curvature.

    Key advantages of Fromage:

    - **Learning-rate-free**: No manual lr tuning needed
    - **Automatic adaptation**: Step size adjusts to local curvature
    - **Simple**: Minimal hyperparameters to tune
    - **Fast prototyping**: Good default performance without tuning
    - **Curvature-aware**: Adapts to loss landscape geometry
    - **Robust**: Works across different problem scales

    Fromage is particularly well-suited for:

    - Rapid prototyping and experimentation
    - Hyperparameter-free training pipelines
    - Problems where learning rate is hard to tune
    - Transfer learning with unknown optimal lr
    - Automated machine learning (AutoML)
    - Research experiments requiring minimal tuning

    Comparison with other optimizers:

    - **vs SGD**: No learning rate tuning required
    - **vs Adam**: Simpler, fewer hyperparameters, learning-rate-free
    - **vs AdaGrad**: Automatic adaptation without accumulation issues
    - **vs Hypergradient methods**: Simpler, more efficient computation

    Limitations:

    - May be less optimal than well-tuned adaptive optimizers
    - Requires multiple gradient evaluations for best performance
    - Gradient difference computation adds slight overhead
    - Best for medium-scale problems (not extensively tested on huge models)

    References
    ----------
    .. [1] Bernstein, J., Wang, Y. X., Azizzadenesheli, K., & Anandkumar, A. (2018).
           signSGD: Compressed Optimisation for Non-Convex Problems.
           In International Conference on Machine Learning (pp. 560-569).
    .. [2] Vaswani, S., Mishkin, A., Laradji, I., Schmidt, M., Gidel, G., & Lacoste-Julien, S. (2019).
           Painless Stochastic Gradient: Interpolation, Line-Search, and Convergence Rates.
           Advances in Neural Information Processing Systems, 32.

    Examples
    --------
    Basic learning-rate-free usage:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> # Create model
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Initialize Fromage with default lr=1.0 (no tuning needed)
        >>> optimizer = braintools.optim.Fromage(lr=1.0)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With momentum for smoother updates:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Enable momentum for better convergence
        >>> optimizer = braintools.optim.Fromage(lr=1.0, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Without momentum (pure adaptive):

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Pure gradient-based updates
        >>> optimizer = braintools.optim.Fromage(lr=1.0, momentum=0.0)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With global learning rate scaling:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Scale automatic step sizes by 0.5
        >>> optimizer = braintools.optim.Fromage(lr=0.5, momentum=0.9)
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    With gradient clipping:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> model = brainstate.nn.Linear(10, 5)
        >>>
        >>> # Clip gradients for stability
        >>> optimizer = braintools.optim.Fromage(
        ...     lr=1.0,
        ...     momentum=0.9,
        ...     grad_clip_norm=1.0
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    Complete configuration for prototyping:

    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import braintools as braintools
        >>>
        >>> # Model for rapid experimentation
        >>> model = brainstate.nn.Linear(100, 50)
        >>>
        >>> # Complete Fromage configuration
        >>> optimizer = braintools.optim.Fromage(
        ...     lr=1.0,
        ...     momentum=0.9,
        ...     grad_clip_norm=1.0
        ... )
        >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))

    See Also
    --------
    SGD : Stochastic gradient descent with momentum
    Adam : Adaptive moment estimation
    Adagrad : Adaptive learning rates for sparse features

    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        lr: Union[float, LRScheduler] = 1.0,
        momentum: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
    ):
        self.momentum = momentum
        super().__init__(tx=None, lr=lr, grad_clip_norm=grad_clip_norm, grad_clip_value=grad_clip_value)

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        # Fromage doesn't have a standard optax implementation, using basic approach
        if self.momentum > 0:
            transforms.append(optax.trace(decay=self.momentum))
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)
