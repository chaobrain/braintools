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
JAX-compatible DataLoader for training.

This module provides a DataLoader class that works seamlessly with JAX,
supporting shuffling, batching, prefetching, and distributed training.
"""

import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    'DataLoader',
    'DistributedDataLoader',
    'Dataset',
    'ArrayDataset',
    'DictDataset',
    'IterableDataset',
    'Sampler',
    'RandomSampler',
    'SequentialSampler',
    'BatchSampler',
    'DistributedSampler',
]


class Dataset:
    """
    Abstract base class for datasets.

    Subclass this to create custom datasets.
    """
    __module__ = 'braintools.trainer'

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        """Return the sample at the given index."""
        raise NotImplementedError


class ArrayDataset(Dataset):
    """
    Dataset wrapping arrays.

    Parameters
    ----------
    *arrays : Array-like
        Arrays to wrap. All arrays must have the same length (first dimension).

    Examples
    --------
    >>> X = jnp.ones((100, 10))
    >>> y = jnp.zeros((100,))
    >>> dataset = ArrayDataset(X, y)
    >>> len(dataset)
    100
    >>> x_sample, y_sample = dataset[0]
    """
    __module__ = 'braintools.trainer'

    def __init__(self, *arrays):
        if len(arrays) == 0:
            raise ValueError("At least one array must be provided")

        # Validate all arrays have the same length
        self.arrays = arrays
        self._length = len(arrays[0])

        for i, arr in enumerate(arrays):
            if len(arr) != self._length:
                raise ValueError(
                    f"All arrays must have the same length. "
                    f"Array 0 has length {self._length}, array {i} has length {len(arr)}"
                )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: Union[int, slice, np.ndarray]) -> Tuple:
        if isinstance(index, int):
            return tuple(arr[index] for arr in self.arrays)
        elif isinstance(index, (slice, np.ndarray, jnp.ndarray)):
            return tuple(arr[index] for arr in self.arrays)
        else:
            raise TypeError(f"Invalid index type: {type(index)}")


class DictDataset(Dataset):
    """
    Dataset wrapping a dictionary of arrays.

    Parameters
    ----------
    data : Dict[str, Array-like]
        Dictionary mapping keys to arrays.

    Examples
    --------
    >>> data = {'x': jnp.ones((100, 10)), 'y': jnp.zeros((100,))}
    >>> dataset = DictDataset(data)
    >>> len(dataset)
    100
    >>> sample = dataset[0]
    >>> sample['x'].shape
    (10,)
    """
    __module__ = 'braintools.trainer'

    def __init__(self, data: Dict[str, Any]):
        if not data:
            raise ValueError("Data dictionary cannot be empty")

        self.data = data
        self.keys = list(data.keys())

        # Get length from first array
        first_key = self.keys[0]
        self._length = len(data[first_key])

        # Validate all arrays have the same length
        for key in self.keys:
            if len(data[key]) != self._length:
                raise ValueError(
                    f"All arrays must have the same length. "
                    f"'{first_key}' has length {self._length}, "
                    f"'{key}' has length {len(data[key])}"
                )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: Union[int, slice, np.ndarray]) -> Dict[str, Any]:
        return {key: self.data[key][index] for key in self.keys}


class IterableDataset(Dataset):
    """
    Dataset wrapping an iterable.

    Parameters
    ----------
    iterable : Iterable
        The iterable to wrap.
    length : int, optional
        The length of the iterable (if known).

    Examples
    --------
    >>> def data_generator():
    ...     for i in range(100):
    ...         yield {'x': np.random.randn(10), 'y': i}
    >>> dataset = IterableDataset(data_generator(), length=100)
    """
    __module__ = 'braintools.trainer'

    def __init__(self, iterable: Any, length: Optional[int] = None):
        self.iterable = iterable
        self._length = length
        self._cache: List[Any] = []

    def __len__(self) -> int:
        if self._length is not None:
            return self._length
        raise TypeError("This dataset does not have a known length")

    def __iter__(self):
        return iter(self.iterable)

    def __getitem__(self, index: int) -> Any:
        # Cache items for random access
        while len(self._cache) <= index:
            try:
                item = next(iter(self.iterable))
                self._cache.append(item)
            except StopIteration:
                raise IndexError(f"Index {index} out of range")
        return self._cache[index]


class Sampler:
    """
    Base class for samplers.

    Samplers define the order in which dataset samples are accessed.
    """
    __module__ = 'braintools.trainer'

    def __iter__(self) -> Iterator[int]:
        """Return an iterator over sample indices."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of samples."""
        raise NotImplementedError


class SequentialSampler(Sampler):
    """
    Samples elements sequentially.

    Parameters
    ----------
    data_source : Dataset
        Dataset to sample from.

    Examples
    --------
    >>> sampler = SequentialSampler(dataset)
    >>> list(sampler)
    [0, 1, 2, ..., n-1]
    """
    __module__ = 'braintools.trainer'

    def __init__(self, data_source: Dataset):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """
    Samples elements randomly.

    Parameters
    ----------
    data_source : Dataset
        Dataset to sample from.
    replacement : bool, default=False
        Whether to sample with replacement.
    num_samples : int, optional
        Number of samples to draw. Default is len(data_source).
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> sampler = RandomSampler(dataset, seed=42)
    >>> indices = list(sampler)  # Random permutation
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        data_source: Dataset,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.replacement:
            indices = self._rng.integers(0, n, size=self.num_samples).tolist()
        else:
            indices = self._rng.permutation(n).tolist()
            if self.num_samples != n:
                indices = indices[:self.num_samples]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def reset(self, seed: Optional[int] = None):
        """Reset the random state."""
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)


class BatchSampler(Sampler):
    """
    Wraps a sampler to yield batches of indices.

    Parameters
    ----------
    sampler : Sampler
        Base sampler to wrap.
    batch_size : int
        Size of each batch.
    drop_last : bool, default=False
        Whether to drop the last incomplete batch.

    Examples
    --------
    >>> base_sampler = SequentialSampler(dataset)
    >>> batch_sampler = BatchSampler(base_sampler, batch_size=32)
    >>> for batch_indices in batch_sampler:
    ...     print(len(batch_indices))  # 32 (or less for last batch)
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return math.ceil(len(self.sampler) / self.batch_size)


class DistributedSampler(Sampler):
    """
    Sampler for distributed training.

    Splits data across multiple replicas (devices/processes).

    Parameters
    ----------
    data_source : Dataset
        Dataset to sample from.
    num_replicas : int, optional
        Number of distributed replicas. Default is number of JAX devices.
    rank : int, optional
        Rank of current replica. Default is 0.
    shuffle : bool, default=True
        Whether to shuffle the data.
    seed : int, default=0
        Random seed for shuffling.
    drop_last : bool, default=False
        Whether to drop samples that don't divide evenly.

    Examples
    --------
    >>> sampler = DistributedSampler(dataset, num_replicas=4, rank=0)
    >>> # Each replica gets 1/4 of the data
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        data_source: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.data_source = data_source
        self.num_replicas = num_replicas or len(jax.devices())
        self.rank = rank or 0
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Calculate number of samples per replica
        total_size = len(data_source)
        if self.drop_last and total_size % self.num_replicas != 0:
            self.num_samples = total_size // self.num_replicas
        else:
            self.num_samples = math.ceil(total_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        # Deterministic shuffling based on epoch
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            indices = rng.permutation(len(self.data_source)).tolist()
        else:
            indices = list(range(len(self.data_source)))

        # Pad indices to make evenly divisible
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]

        # Subsample for this replica
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set the epoch for deterministic shuffling."""
        self.epoch = epoch


def default_collate_fn(batch: List[Any]) -> Any:
    """
    Default collate function for batching samples.

    Parameters
    ----------
    batch : List[Any]
        List of samples to collate.

    Returns
    -------
    Any
        Collated batch (stacked arrays or dict of stacked arrays).
    """
    if not batch:
        return batch

    elem = batch[0]

    # Handle dictionaries
    if isinstance(elem, dict):
        return {key: default_collate_fn([d[key] for d in batch]) for key in elem}

    # Handle tuples
    if isinstance(elem, tuple):
        return tuple(default_collate_fn(samples) for samples in zip(*batch))

    # Handle lists
    if isinstance(elem, list):
        return [default_collate_fn(samples) for samples in zip(*batch)]

    # Handle arrays
    if isinstance(elem, (np.ndarray, jnp.ndarray)):
        return jnp.stack(batch)

    # Handle scalars
    if isinstance(elem, (int, float)):
        return jnp.array(batch)

    # Default: return as-is
    return batch


class DataLoader:
    """
    JAX-compatible data loader.

    Parameters
    ----------
    dataset : Dataset or Array-like
        Dataset or array(s) to load from.
    batch_size : int, default=32
        Number of samples per batch.
    shuffle : bool, default=False
        Whether to shuffle the data each epoch.
    sampler : Sampler, optional
        Custom sampler. Mutually exclusive with shuffle.
    batch_sampler : BatchSampler, optional
        Custom batch sampler. Mutually exclusive with batch_size, shuffle, sampler.
    num_workers : int, default=0
        Number of worker processes for data loading (not implemented in JAX).
    collate_fn : Callable, optional
        Function to collate samples into batches.
    drop_last : bool, default=False
        Whether to drop the last incomplete batch.
    prefetch : int, default=2
        Number of batches to prefetch (for device transfer).
    seed : int, optional
        Random seed for shuffling.

    Examples
    --------
    Basic usage with arrays:

    >>> X = jnp.ones((1000, 784))
    >>> y = jnp.zeros((1000,))
    >>> loader = DataLoader((X, y), batch_size=32, shuffle=True)
    >>> for batch in loader:
    ...     x_batch, y_batch = batch
    ...     print(x_batch.shape)  # (32, 784)

    With a dataset:

    >>> dataset = DictDataset({'x': X, 'y': y})
    >>> loader = DataLoader(dataset, batch_size=32)
    >>> for batch in loader:
    ...     print(batch['x'].shape)  # (32, 784)
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        dataset: Union[Dataset, Any],
        batch_size: int = 32,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = False,
        prefetch: int = 2,
        seed: Optional[int] = None,
    ):
        # Convert raw arrays to dataset
        if not isinstance(dataset, Dataset):
            if isinstance(dataset, dict):
                dataset = DictDataset(dataset)
            elif isinstance(dataset, tuple):
                dataset = ArrayDataset(*dataset)
            elif hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__'):
                # Assume it's array-like
                dataset = ArrayDataset(dataset)
            else:
                raise TypeError(
                    f"dataset must be a Dataset, dict, tuple of arrays, or array-like. "
                    f"Got {type(dataset)}"
                )

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn or default_collate_fn
        self.drop_last = drop_last
        self.prefetch = prefetch
        self.seed = seed

        # Set up sampler
        if batch_sampler is not None:
            if batch_size != 32 or shuffle or sampler is not None:
                raise ValueError(
                    "batch_sampler is mutually exclusive with "
                    "batch_size, shuffle, and sampler"
                )
            self.batch_sampler = batch_sampler
        else:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset, seed=seed)
                else:
                    sampler = SequentialSampler(dataset)
            self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self._epoch = 0

    def __iter__(self) -> Iterator[Any]:
        """Iterate over batches."""
        # Reset random sampler if shuffling
        if hasattr(self.batch_sampler.sampler, 'reset'):
            self.batch_sampler.sampler.reset()

        for batch_indices in self.batch_sampler:
            # Get samples
            samples = [self.dataset[i] for i in batch_indices]
            # Collate into batch
            batch = self.collate_fn(samples)
            yield batch

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.batch_sampler)

    def set_epoch(self, epoch: int):
        """Set the epoch (for distributed samplers)."""
        self._epoch = epoch
        if hasattr(self.batch_sampler.sampler, 'set_epoch'):
            self.batch_sampler.sampler.set_epoch(epoch)

    @property
    def num_samples(self) -> int:
        """Return the total number of samples."""
        return len(self.dataset)


class DistributedDataLoader(DataLoader):
    """
    DataLoader for distributed training.

    Automatically shards data across multiple devices/processes.

    Parameters
    ----------
    dataset : Dataset or Array-like
        Dataset to load from.
    batch_size : int, default=32
        Batch size per replica.
    num_replicas : int, optional
        Number of replicas. Default is number of JAX devices.
    rank : int, optional
        Rank of this replica.
    shuffle : bool, default=True
        Whether to shuffle data.
    seed : int, default=0
        Random seed for shuffling.
    drop_last : bool, default=False
        Whether to drop incomplete batches.
    **kwargs
        Additional arguments passed to DataLoader.

    Examples
    --------
    >>> loader = DistributedDataLoader(dataset, batch_size=32)
    >>> # Data is automatically split across devices
    """
    __module__ = 'braintools.trainer'

    def __init__(
        self,
        dataset: Union[Dataset, Any],
        batch_size: int = 32,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        **kwargs,
    ):
        # Convert raw arrays to dataset
        if not isinstance(dataset, Dataset):
            if isinstance(dataset, dict):
                dataset = DictDataset(dataset)
            elif isinstance(dataset, tuple):
                dataset = ArrayDataset(*dataset)
            else:
                dataset = ArrayDataset(dataset)

        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            **kwargs,
        )

        self.num_replicas = sampler.num_replicas
        self.rank = sampler.rank

    def set_epoch(self, epoch: int):
        """Set the epoch for the distributed sampler."""
        super().set_epoch(epoch)


def create_distributed_batches(
    data: Any,
    batch_size: int,
    devices: Optional[List[Any]] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> Iterator[Any]:
    """
    Create batches suitable for distributed training with pmap.

    Parameters
    ----------
    data : Any
        Input data (array, tuple of arrays, or dict of arrays).
    batch_size : int
        Batch size per device.
    devices : List, optional
        List of devices. Default is all JAX devices.
    shuffle : bool, default=False
        Whether to shuffle the data.
    seed : int, optional
        Random seed for shuffling.

    Yields
    ------
    Any
        Batches with shape (num_devices, batch_size, ...).

    Examples
    --------
    >>> for batch in create_distributed_batches(X, batch_size=32):
    ...     # batch.shape == (num_devices, 32, ...)
    ...     outputs = pmap_train_step(batch)
    """
    if devices is None:
        devices = jax.devices()
    num_devices = len(devices)

    # Handle different data types
    if isinstance(data, dict):
        n_samples = len(next(iter(data.values())))
        get_item = lambda i: {k: v[i] for k, v in data.items()}
    elif isinstance(data, tuple):
        n_samples = len(data[0])
        get_item = lambda i: tuple(arr[i] for arr in data)
    else:
        n_samples = len(data)
        get_item = lambda i: data[i]

    # Create indices
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)

    # Calculate batch size for all devices
    total_batch_size = batch_size * num_devices

    # Iterate in chunks
    for start in range(0, n_samples - total_batch_size + 1, total_batch_size):
        batch_indices = indices[start:start + total_batch_size]

        # Get batch data
        if isinstance(data, dict):
            batch = {
                k: jnp.stack([v[i] for i in batch_indices])
                for k, v in data.items()
            }
            # Reshape for pmap: (num_devices, batch_size, ...)
            batch = {
                k: v.reshape(num_devices, batch_size, *v.shape[1:])
                for k, v in batch.items()
            }
        elif isinstance(data, tuple):
            batch = tuple(
                jnp.stack([arr[i] for i in batch_indices]).reshape(
                    num_devices, batch_size, *arr.shape[1:]
                )
                for arr in data
            )
        else:
            batch = jnp.stack([data[i] for i in batch_indices])
            batch = batch.reshape(num_devices, batch_size, *data.shape[1:])

        yield batch
