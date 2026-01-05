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
Distributed training strategies for JAX.

This module provides different strategies for distributed training including
data parallelism (pmap), model parallelism (sharding), and FSDP-like strategies.
"""

import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import brainstate
from brainstate import State
from brainstate.typing import PyTree

__all__ = [
    'Strategy',
    'SingleDeviceStrategy',
    'DataParallelStrategy',
    'ShardedDataParallelStrategy',
    'FullyShardedDataParallelStrategy',
    'AutoStrategy',
    'get_strategy',
    'all_reduce',
    'broadcast',
]


class Strategy(ABC):
    """
    Abstract base class for distributed training strategies.

    Strategies define how model parameters and computations are distributed
    across devices.
    """
    __module__ = 'braintools.trainer'

    def __init__(self):
        self._is_setup = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    @property
    @abstractmethod
    def num_devices(self) -> int:
        """Number of devices used by this strategy."""
        pass

    @property
    @abstractmethod
    def devices(self) -> List[Any]:
        """List of devices used by this strategy."""
        pass

    @property
    def is_distributed(self) -> bool:
        """Whether this is a distributed strategy."""
        return self.num_devices > 1

    @abstractmethod
    def setup(self, model: Any, optimizer: Any) -> Tuple[Any, Any]:
        """
        Set up the model and optimizer for distributed training.

        Parameters
        ----------
        model : Any
            The model to distribute.
        optimizer : Any
            The optimizer to distribute.

        Returns
        -------
        Tuple[Any, Any]
            The distributed model and optimizer.
        """
        pass

    @abstractmethod
    def training_step(
        self,
        model: Any,
        optimizer: Any,
        batch: Any,
        loss_fn: Callable,
        param_states: PyTree,
    ) -> Tuple[Any, PyTree]:
        """
        Execute a single training step.

        Parameters
        ----------
        model : Any
            The model.
        optimizer : Any
            The optimizer.
        batch : Any
            The input batch.
        loss_fn : Callable
            Loss function to compute.
        param_states : PyTree
            Parameter states for gradient computation.

        Returns
        -------
        Tuple[Any, PyTree]
            Loss value and metrics.
        """
        pass

    def reduce(
        self,
        tensor: Any,
        op: str = 'mean',
        axis_name: str = 'batch',
    ) -> Any:
        """
        Reduce tensor across devices.

        Parameters
        ----------
        tensor : Any
            Tensor to reduce.
        op : str
            Reduction operation ('mean', 'sum', 'min', 'max').
        axis_name : str
            Name of the axis for reduction.

        Returns
        -------
        Any
            Reduced tensor.
        """
        return tensor

    def broadcast(self, tensor: Any, src: int = 0) -> Any:
        """
        Broadcast tensor from source to all devices.

        Parameters
        ----------
        tensor : Any
            Tensor to broadcast.
        src : int
            Source device index.

        Returns
        -------
        Any
            Broadcasted tensor.
        """
        return tensor

    def barrier(self):
        """Synchronization barrier across devices."""
        pass


class SingleDeviceStrategy(Strategy):
    """
    Single device strategy (no distribution).

    This is the default strategy when only one device is available
    or distribution is not needed.

    Parameters
    ----------
    device : Any, optional
        The device to use. Default is the first available device.

    Examples
    --------
    >>> strategy = SingleDeviceStrategy()
    >>> trainer = Trainer(strategy=strategy)
    """
    __module__ = 'braintools.trainer'

    def __init__(self, device: Optional[Any] = None):
        super().__init__()
        self._device = device or jax.devices()[0]

    @property
    def name(self) -> str:
        return 'single_device'

    @property
    def num_devices(self) -> int:
        return 1

    @property
    def devices(self) -> List[Any]:
        return [self._device]

    def setup(self, model: Any, optimizer: Any) -> Tuple[Any, Any]:
        self._is_setup = True
        return model, optimizer

    def training_step(
        self,
        model: Any,
        optimizer: Any,
        batch: Any,
        loss_fn: Callable,
        param_states: PyTree,
    ) -> Tuple[Any, PyTree]:
        """Single device training step using brainstate transforms."""

        def compute_loss():
            return loss_fn(model, batch)

        loss = compute_loss()
        grads = brainstate.transform.grad(compute_loss, grad_states=param_states)()
        optimizer.step(grads)

        return loss, {}


class DataParallelStrategy(Strategy):
    """
    Data parallel strategy using pmap.

    Replicates the model across all devices and splits data batches.
    Gradients are synchronized using all-reduce.

    Parameters
    ----------
    devices : List[Any], optional
        List of devices to use. Default is all available devices.
    axis_name : str, default='batch'
        Name of the batch axis for collective operations.

    Examples
    --------
    >>> strategy = DataParallelStrategy()
    >>> trainer = Trainer(strategy=strategy)
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        devices: Optional[List[Any]] = None,
        axis_name: str = 'batch',
    ):
        super().__init__()
        self._devices = devices or jax.devices()
        self._axis_name = axis_name
        self._pmap_fn = None

    @property
    def name(self) -> str:
        return 'data_parallel'

    @property
    def num_devices(self) -> int:
        return len(self._devices)

    @property
    def devices(self) -> List[Any]:
        return self._devices

    def setup(self, model: Any, optimizer: Any) -> Tuple[Any, Any]:
        """Replicate model parameters across devices."""
        self._is_setup = True
        return model, optimizer

    def _replicate_params(self, params: PyTree) -> PyTree:
        """Replicate parameters across all devices."""
        return jax.device_put_replicated(params, self._devices)

    def _unreplicate_params(self, params: PyTree) -> PyTree:
        """Get parameters from first device."""
        return jax.tree.map(lambda x: x[0], params)

    def training_step(
        self,
        model: Any,
        optimizer: Any,
        batch: Any,
        loss_fn: Callable,
        param_states: PyTree,
    ) -> Tuple[Any, PyTree]:
        """
        Parallel training step.

        The batch should already be sharded with shape (num_devices, batch_size, ...).
        """
        axis_name = self._axis_name

        @functools.partial(jax.pmap, axis_name=axis_name)
        def pmap_train_step(batch_shard):
            def compute_loss():
                return loss_fn(model, batch_shard)

            loss = compute_loss()
            grads = brainstate.transform.grad(compute_loss, grad_states=param_states)()

            # Average gradients across devices
            grads = jax.tree.map(
                lambda g: lax.pmean(g, axis_name=axis_name),
                grads
            )

            return loss, grads

        # Run parallel computation
        losses, grads = pmap_train_step(batch)

        # Update optimizer (on host)
        avg_grads = self._unreplicate_params(grads)
        optimizer.step(avg_grads)

        # Return averaged loss
        avg_loss = jnp.mean(losses)

        return avg_loss, {}

    def reduce(
        self,
        tensor: Any,
        op: str = 'mean',
        axis_name: str = None,
    ) -> Any:
        """Reduce tensor across devices."""
        axis_name = axis_name or self._axis_name

        if op == 'mean':
            return lax.pmean(tensor, axis_name=axis_name)
        elif op == 'sum':
            return lax.psum(tensor, axis_name=axis_name)
        elif op == 'min':
            return lax.pmin(tensor, axis_name=axis_name)
        elif op == 'max':
            return lax.pmax(tensor, axis_name=axis_name)
        else:
            raise ValueError(f"Unknown reduction operation: {op}")

    def broadcast(self, tensor: Any, src: int = 0) -> Any:
        """Broadcast tensor from source device."""
        return lax.ppermute(
            tensor,
            axis_name=self._axis_name,
            perm=[(src, i) for i in range(self.num_devices)]
        )


class ShardedDataParallelStrategy(Strategy):
    """
    Sharded data parallel strategy using jax.sharding.

    Uses modern JAX sharding APIs for more flexible data distribution.

    Parameters
    ----------
    mesh : Mesh, optional
        Device mesh for sharding. Default creates a 1D mesh.
    data_axis : str, default='data'
        Name of the data parallel axis in the mesh.

    Examples
    --------
    >>> strategy = ShardedDataParallelStrategy()
    >>> trainer = Trainer(strategy=strategy)
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        mesh: Optional[Mesh] = None,
        data_axis: str = 'data',
    ):
        super().__init__()
        self._data_axis = data_axis

        if mesh is None:
            devices = jax.devices()
            self._mesh = Mesh(devices, (data_axis,))
        else:
            self._mesh = mesh

        self._devices = list(self._mesh.devices.flat)

    @property
    def name(self) -> str:
        return 'sharded_data_parallel'

    @property
    def num_devices(self) -> int:
        return len(self._devices)

    @property
    def devices(self) -> List[Any]:
        return self._devices

    @property
    def mesh(self) -> Mesh:
        """The device mesh."""
        return self._mesh

    def setup(self, model: Any, optimizer: Any) -> Tuple[Any, Any]:
        """Set up sharded training."""
        self._is_setup = True
        return model, optimizer

    def _shard_data(self, data: Any) -> Any:
        """Shard data across the data axis."""
        sharding = NamedSharding(self._mesh, PartitionSpec(self._data_axis))
        return jax.device_put(data, sharding)

    def _replicate(self, params: PyTree) -> PyTree:
        """Replicate parameters across all devices."""
        sharding = NamedSharding(self._mesh, PartitionSpec())
        return jax.tree.map(lambda x: jax.device_put(x, sharding), params)

    def training_step(
        self,
        model: Any,
        optimizer: Any,
        batch: Any,
        loss_fn: Callable,
        param_states: PyTree,
    ) -> Tuple[Any, PyTree]:
        """Sharded training step."""

        # Shard the batch
        sharded_batch = jax.tree.map(self._shard_data, batch)

        @jax.jit
        def train_step(batch):
            def compute_loss():
                return loss_fn(model, batch)

            with self._mesh:
                loss = compute_loss()
                grads = brainstate.transform.grad(compute_loss, grad_states=param_states)()

                # Average gradients
                grads = jax.tree.map(
                    lambda g: jax.lax.pmean(g, axis_name=self._data_axis),
                    grads
                )

            return loss, grads

        loss, grads = train_step(sharded_batch)
        optimizer.step(grads)

        return loss, {}


class FullyShardedDataParallelStrategy(Strategy):
    """
    Fully Sharded Data Parallel (FSDP) strategy.

    Shards both model parameters and gradients across devices,
    similar to PyTorch's FSDP.

    Parameters
    ----------
    mesh : Mesh, optional
        Device mesh for sharding.
    data_axis : str, default='data'
        Name of the data parallel axis.
    model_axis : str, optional
        Name of the model parallel axis (for tensor parallelism).

    Examples
    --------
    >>> strategy = FullyShardedDataParallelStrategy()
    >>> trainer = Trainer(strategy=strategy)
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        mesh: Optional[Mesh] = None,
        data_axis: str = 'data',
        model_axis: Optional[str] = None,
    ):
        super().__init__()
        self._data_axis = data_axis
        self._model_axis = model_axis

        if mesh is None:
            devices = jax.devices()
            if model_axis:
                # 2D mesh for data + model parallelism
                n_devices = len(devices)
                # Try to create a balanced mesh
                for dp in [n_devices, n_devices // 2, n_devices // 4]:
                    mp = n_devices // dp
                    if dp * mp == n_devices:
                        break
                mesh_shape = (dp, mp)
                self._mesh = Mesh(
                    devices.reshape(mesh_shape),
                    (data_axis, model_axis)
                )
            else:
                # 1D mesh for data parallelism only
                self._mesh = Mesh(devices, (data_axis,))
        else:
            self._mesh = mesh

        self._devices = list(self._mesh.devices.flat)

    @property
    def name(self) -> str:
        return 'fsdp'

    @property
    def num_devices(self) -> int:
        return len(self._devices)

    @property
    def devices(self) -> List[Any]:
        return self._devices

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def _get_param_sharding(self, param_shape: Tuple[int, ...]) -> PartitionSpec:
        """Determine sharding for a parameter based on its shape."""
        # Shard the largest dimension
        if len(param_shape) == 0:
            return PartitionSpec()
        elif len(param_shape) == 1:
            return PartitionSpec(self._data_axis)
        else:
            # Shard along first dimension (typically features/neurons)
            if self._model_axis:
                return PartitionSpec(self._data_axis, self._model_axis)
            return PartitionSpec(self._data_axis, None)

    def setup(self, model: Any, optimizer: Any) -> Tuple[Any, Any]:
        """Set up FSDP by sharding model parameters."""
        self._is_setup = True

        # Shard parameters
        param_states = model.states(brainstate.ParamState)
        for state in param_states.values():
            if hasattr(state, 'value'):
                spec = self._get_param_sharding(state.value.shape)
                sharding = NamedSharding(self._mesh, spec)
                state.value = jax.device_put(state.value, sharding)

        return model, optimizer

    def training_step(
        self,
        model: Any,
        optimizer: Any,
        batch: Any,
        loss_fn: Callable,
        param_states: PyTree,
    ) -> Tuple[Any, PyTree]:
        """FSDP training step with parameter gathering and gradient scattering."""

        # Shard batch data
        data_sharding = NamedSharding(
            self._mesh, PartitionSpec(self._data_axis)
        )
        sharded_batch = jax.tree.map(
            lambda x: jax.device_put(x, data_sharding), batch
        )

        @jax.jit
        def fsdp_train_step(batch):
            def compute_loss():
                return loss_fn(model, batch)

            with self._mesh:
                loss = compute_loss()
                grads = brainstate.transform.grad(compute_loss, grad_states=param_states)()

                # Reduce gradients
                grads = jax.tree.map(
                    lambda g: jax.lax.pmean(g, axis_name=self._data_axis),
                    grads
                )

            return loss, grads

        loss, grads = fsdp_train_step(sharded_batch)
        optimizer.step(grads)

        return loss, {}


class AutoStrategy(Strategy):
    """
    Automatically select the best strategy based on available devices.

    This strategy inspects the available hardware and selects an appropriate
    distribution strategy.

    Parameters
    ----------
    prefer_fsdp : bool, default=False
        Whether to prefer FSDP over simple data parallelism when multiple
        devices are available.

    Examples
    --------
    >>> strategy = AutoStrategy()
    >>> trainer = Trainer(strategy=strategy)
    """
    __module__ = 'braintools.trainer'

    def __init__(self, prefer_fsdp: bool = False):
        super().__init__()
        self._prefer_fsdp = prefer_fsdp
        self._inner_strategy: Optional[Strategy] = None
        self._select_strategy()

    def _select_strategy(self):
        """Select the appropriate strategy based on available devices."""
        devices = jax.devices()
        num_devices = len(devices)

        if num_devices == 1:
            self._inner_strategy = SingleDeviceStrategy(devices[0])
        elif self._prefer_fsdp and num_devices >= 4:
            self._inner_strategy = FullyShardedDataParallelStrategy()
        elif num_devices > 1:
            self._inner_strategy = DataParallelStrategy(devices)
        else:
            self._inner_strategy = SingleDeviceStrategy()

    @property
    def name(self) -> str:
        return f'auto({self._inner_strategy.name})'

    @property
    def num_devices(self) -> int:
        return self._inner_strategy.num_devices

    @property
    def devices(self) -> List[Any]:
        return self._inner_strategy.devices

    @property
    def selected_strategy(self) -> Strategy:
        """The automatically selected strategy."""
        return self._inner_strategy

    def setup(self, model: Any, optimizer: Any) -> Tuple[Any, Any]:
        return self._inner_strategy.setup(model, optimizer)

    def training_step(
        self,
        model: Any,
        optimizer: Any,
        batch: Any,
        loss_fn: Callable,
        param_states: PyTree,
    ) -> Tuple[Any, PyTree]:
        return self._inner_strategy.training_step(
            model, optimizer, batch, loss_fn, param_states
        )

    def reduce(self, tensor: Any, op: str = 'mean', axis_name: str = 'batch') -> Any:
        return self._inner_strategy.reduce(tensor, op, axis_name)

    def broadcast(self, tensor: Any, src: int = 0) -> Any:
        return self._inner_strategy.broadcast(tensor, src)


def get_strategy(
    strategy: Union[str, Strategy, None] = 'auto',
    **kwargs,
) -> Strategy:
    """
    Get a distributed training strategy by name.

    Parameters
    ----------
    strategy : str or Strategy, default='auto'
        Strategy name or instance.
        Options: 'auto', 'single', 'ddp', 'dp', 'fsdp', 'sdp'
    **kwargs
        Additional arguments passed to the strategy constructor.

    Returns
    -------
    Strategy
        The requested strategy instance.

    Examples
    --------
    >>> strategy = get_strategy('ddp')
    >>> strategy = get_strategy('fsdp', prefer_fsdp=True)
    """
    if isinstance(strategy, Strategy):
        return strategy

    if strategy is None or strategy == 'auto':
        return AutoStrategy(**kwargs)
    elif strategy in ('single', 'single_device'):
        return SingleDeviceStrategy(**kwargs)
    elif strategy in ('ddp', 'dp', 'data_parallel'):
        return DataParallelStrategy(**kwargs)
    elif strategy in ('sdp', 'sharded_data_parallel'):
        return ShardedDataParallelStrategy(**kwargs)
    elif strategy == 'fsdp':
        return FullyShardedDataParallelStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# Utility Functions
# =============================================================================

def all_reduce(
    tensor: Any,
    op: str = 'mean',
    axis_name: str = 'batch',
) -> Any:
    """
    All-reduce tensor across devices.

    Parameters
    ----------
    tensor : Any
        Tensor to reduce.
    op : str
        Reduction operation ('mean', 'sum', 'min', 'max').
    axis_name : str
        Name of the axis for reduction (used with pmap).

    Returns
    -------
    Any
        Reduced tensor.

    Examples
    --------
    >>> @functools.partial(jax.pmap, axis_name='batch')
    ... def parallel_fn(x):
    ...     return all_reduce(x, op='mean', axis_name='batch')
    """
    if op == 'mean':
        return lax.pmean(tensor, axis_name=axis_name)
    elif op == 'sum':
        return lax.psum(tensor, axis_name=axis_name)
    elif op == 'min':
        return lax.pmin(tensor, axis_name=axis_name)
    elif op == 'max':
        return lax.pmax(tensor, axis_name=axis_name)
    else:
        raise ValueError(f"Unknown reduction operation: {op}")


def broadcast(
    tensor: Any,
    src: int = 0,
    axis_name: str = 'batch',
    num_devices: Optional[int] = None,
) -> Any:
    """
    Broadcast tensor from source device to all devices.

    Parameters
    ----------
    tensor : Any
        Tensor to broadcast.
    src : int
        Source device index.
    axis_name : str
        Name of the axis for broadcast (used with pmap).
    num_devices : int, optional
        Number of devices.

    Returns
    -------
    Any
        Broadcasted tensor.
    """
    if num_devices is None:
        num_devices = jax.device_count()

    perm = [(src, i) for i in range(num_devices)]
    return lax.ppermute(tensor, axis_name=axis_name, perm=perm)


def sync_batch_norm(
    x: Any,
    axis_name: str = 'batch',
) -> Tuple[Any, Any]:
    """
    Compute synchronized batch normalization statistics.

    Parameters
    ----------
    x : Any
        Input tensor.
    axis_name : str
        Name of the axis for synchronization.

    Returns
    -------
    Tuple[Any, Any]
        Synchronized mean and variance.
    """
    # Compute local statistics
    mean = jnp.mean(x, axis=0)
    var = jnp.var(x, axis=0)

    # Synchronize across devices
    mean = lax.pmean(mean, axis_name=axis_name)
    var = lax.pmean(var, axis_name=axis_name)

    return mean, var
