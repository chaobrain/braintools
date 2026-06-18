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

"""Tests for the JAX-compatible DataLoader module."""

import jax.numpy as jnp
import numpy as np
import pytest

from braintools.trainer._dataloader import (
    DataLoader,
    DistributedDataLoader,
    Dataset,
    ArrayDataset,
    DictDataset,
    IterableDataset,
    Sampler,
    RandomSampler,
    SequentialSampler,
    BatchSampler,
    DistributedSampler,
    default_collate_fn,
    create_distributed_batches,
)


class TestDataset:
    """Tests for the abstract Dataset base class."""

    def test_base_not_implemented(self):
        dataset = Dataset()
        with pytest.raises(NotImplementedError):
            len(dataset)
        with pytest.raises(NotImplementedError):
            dataset[0]


class TestArrayDataset:
    """Tests for ArrayDataset."""

    def test_len_and_int_index(self):
        X = jnp.arange(100).reshape(100, 1)
        y = jnp.arange(100)
        dataset = ArrayDataset(X, y)
        assert len(dataset) == 100

        sample = dataset[3]
        assert len(sample) == 2
        assert int(sample[0][0]) == 3
        assert int(sample[1]) == 3

    def test_slice_index(self):
        X = jnp.arange(100).reshape(100, 1)
        dataset = ArrayDataset(X)
        sub = dataset[0:5]
        assert sub[0].shape == (5, 1)

    def test_ndarray_index(self):
        X = jnp.arange(100).reshape(100, 1)
        dataset = ArrayDataset(X)
        idx = np.array([1, 3, 5])
        sub = dataset[idx]
        assert sub[0].shape == (3, 1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ArrayDataset()

    def test_mismatched_lengths_raises(self):
        X = jnp.ones((10, 2))
        y = jnp.ones((5, 2))
        with pytest.raises(ValueError):
            ArrayDataset(X, y)

    def test_invalid_index_type_raises(self):
        dataset = ArrayDataset(jnp.ones((10, 2)))
        with pytest.raises(TypeError):
            dataset["bad"]


class TestDictDataset:
    """Tests for DictDataset."""

    def test_len_and_index(self):
        data = {'x': jnp.ones((50, 3)), 'y': jnp.zeros((50,))}
        dataset = DictDataset(data)
        assert len(dataset) == 50

        sample = dataset[0]
        assert set(sample.keys()) == {'x', 'y'}
        assert sample['x'].shape == (3,)

    def test_slice_index(self):
        data = {'x': jnp.ones((50, 3))}
        dataset = DictDataset(data)
        sub = dataset[0:10]
        assert sub['x'].shape == (10, 3)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            DictDataset({})

    def test_mismatched_lengths_raises(self):
        data = {'x': jnp.ones((10, 2)), 'y': jnp.ones((5, 2))}
        with pytest.raises(ValueError):
            DictDataset(data)


class TestIterableDataset:
    """Tests for IterableDataset."""

    def test_known_length(self):
        items = [{'x': i} for i in range(10)]
        dataset = IterableDataset(items, length=10)
        assert len(dataset) == 10

    def test_unknown_length_raises(self):
        dataset = IterableDataset(iter([1, 2, 3]))
        with pytest.raises(TypeError):
            len(dataset)

    def test_iteration(self):
        items = [1, 2, 3, 4]
        dataset = IterableDataset(items, length=4)
        assert list(iter(dataset)) == [1, 2, 3, 4]

    def test_getitem_caches(self):
        # A one-shot iterator is consumed and cached as items are requested.
        dataset = IterableDataset(iter([10, 20, 30]), length=3)
        assert dataset[0] == 10
        assert dataset[1] == 20
        # Cached access again returns the already-seen item.
        assert dataset[0] == 10
        assert dataset._cache == [10, 20]

    def test_getitem_out_of_range_raises(self):
        dataset = IterableDataset(iter([1, 2]), length=2)
        with pytest.raises(IndexError):
            dataset[10]

    def test_getitem_reiterable_distinct_items(self):
        # A re-iterable source (e.g. a list) must yield distinct items by
        # index. The previous implementation created a fresh iterator on every
        # access and so returned the first element for every index.
        dataset = IterableDataset([10, 20, 30], length=3)
        assert dataset[0] == 10
        assert dataset[1] == 20
        assert dataset[2] == 30

    def test_getitem_negative_index_raises(self):
        dataset = IterableDataset([1, 2, 3], length=3)
        with pytest.raises(IndexError):
            dataset[-1]


class TestSampler:
    """Tests for the abstract Sampler base class."""

    def test_base_not_implemented(self):
        sampler = Sampler()
        with pytest.raises(NotImplementedError):
            iter(sampler)
        with pytest.raises(NotImplementedError):
            len(sampler)


class TestSequentialSampler:
    """Tests for SequentialSampler."""

    def test_order_and_len(self):
        dataset = ArrayDataset(jnp.arange(10).reshape(10, 1))
        sampler = SequentialSampler(dataset)
        assert list(sampler) == list(range(10))
        assert len(sampler) == 10


class TestRandomSampler:
    """Tests for RandomSampler."""

    def test_permutation(self):
        dataset = ArrayDataset(jnp.arange(20).reshape(20, 1))
        sampler = RandomSampler(dataset, seed=42)
        indices = list(sampler)
        assert sorted(indices) == list(range(20))
        assert len(sampler) == 20

    def test_reproducible_with_seed(self):
        dataset = ArrayDataset(jnp.arange(20).reshape(20, 1))
        s1 = RandomSampler(dataset, seed=7)
        s2 = RandomSampler(dataset, seed=7)
        assert list(s1) == list(s2)

    def test_replacement(self):
        dataset = ArrayDataset(jnp.arange(5).reshape(5, 1))
        sampler = RandomSampler(dataset, replacement=True, num_samples=20, seed=1)
        indices = list(sampler)
        assert len(indices) == 20
        assert all(0 <= i < 5 for i in indices)
        assert len(sampler) == 20

    def test_num_samples_subsample(self):
        dataset = ArrayDataset(jnp.arange(20).reshape(20, 1))
        sampler = RandomSampler(dataset, num_samples=5, seed=3)
        indices = list(sampler)
        assert len(indices) == 5
        assert sampler.num_samples == 5

    def test_num_samples_default(self):
        dataset = ArrayDataset(jnp.arange(8).reshape(8, 1))
        sampler = RandomSampler(dataset)
        assert sampler.num_samples == 8

    def test_reset(self):
        dataset = ArrayDataset(jnp.arange(20).reshape(20, 1))
        sampler = RandomSampler(dataset, seed=11)
        first = list(sampler)
        sampler.reset()
        assert list(sampler) == first
        sampler.reset(seed=99)
        assert sampler.seed == 99


class TestBatchSampler:
    """Tests for BatchSampler."""

    def test_grouping_no_drop(self):
        base = SequentialSampler(ArrayDataset(jnp.arange(10).reshape(10, 1)))
        bs = BatchSampler(base, batch_size=3, drop_last=False)
        batches = list(bs)
        assert [len(b) for b in batches] == [3, 3, 3, 1]
        assert len(bs) == 4

    def test_grouping_drop_last(self):
        base = SequentialSampler(ArrayDataset(jnp.arange(10).reshape(10, 1)))
        bs = BatchSampler(base, batch_size=3, drop_last=True)
        batches = list(bs)
        assert [len(b) for b in batches] == [3, 3, 3]
        assert len(bs) == 3

    def test_exact_division(self):
        base = SequentialSampler(ArrayDataset(jnp.arange(9).reshape(9, 1)))
        bs = BatchSampler(base, batch_size=3)
        assert len(list(bs)) == 3
        assert len(bs) == 3

    def test_invalid_batch_size_raises(self):
        base = SequentialSampler(ArrayDataset(jnp.arange(5).reshape(5, 1)))
        with pytest.raises(ValueError):
            BatchSampler(base, batch_size=0)


class TestDistributedSampler:
    """Tests for DistributedSampler on a single replica."""

    def test_single_replica_full_coverage(self):
        dataset = ArrayDataset(jnp.arange(10).reshape(10, 1))
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
        indices = list(sampler)
        assert indices == list(range(10))
        assert len(sampler) == 10

    def test_shuffle_deterministic(self):
        dataset = ArrayDataset(jnp.arange(10).reshape(10, 1))
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=0)
        a = list(sampler)
        b = list(sampler)
        assert a == b
        assert sorted(a) == list(range(10))

    def test_set_epoch_changes_order(self):
        dataset = ArrayDataset(jnp.arange(10).reshape(10, 1))
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=0)
        order0 = list(sampler)
        sampler.set_epoch(5)
        order5 = list(sampler)
        assert sorted(order5) == list(range(10))
        assert order0 != order5

    def test_padding_when_not_divisible(self):
        # 10 samples over 3 replicas -> ceil(10/3)=4 per replica, total 12 with padding
        dataset = ArrayDataset(jnp.arange(10).reshape(10, 1))
        sampler = DistributedSampler(dataset, num_replicas=3, rank=0, shuffle=False, drop_last=False)
        assert sampler.num_samples == 4
        assert sampler.total_size == 12
        indices = list(sampler)
        assert len(indices) == 4

    def test_drop_last_when_not_divisible(self):
        # 10 samples over 3 replicas with drop_last -> 10 // 3 = 3 per replica
        dataset = ArrayDataset(jnp.arange(10).reshape(10, 1))
        sampler = DistributedSampler(dataset, num_replicas=3, rank=0, shuffle=False, drop_last=True)
        assert sampler.num_samples == 3
        assert sampler.total_size == 9
        indices = list(sampler)
        assert len(indices) == 3

    def test_default_num_replicas(self):
        dataset = ArrayDataset(jnp.arange(8).reshape(8, 1))
        sampler = DistributedSampler(dataset, shuffle=False)
        # Single JAX device available
        assert sampler.num_replicas == 1
        assert sampler.rank == 0


class TestDefaultCollate:
    """Tests for default_collate_fn."""

    def test_empty(self):
        assert default_collate_fn([]) == []

    def test_arrays(self):
        batch = [jnp.ones((3,)), jnp.ones((3,))]
        out = default_collate_fn(batch)
        assert out.shape == (2, 3)

    def test_tuples(self):
        batch = [(jnp.ones((2,)), jnp.zeros((1,))), (jnp.ones((2,)), jnp.zeros((1,)))]
        out = default_collate_fn(batch)
        assert isinstance(out, tuple)
        assert out[0].shape == (2, 2)
        assert out[1].shape == (2, 1)

    def test_dicts(self):
        batch = [{'x': jnp.ones((2,))}, {'x': jnp.ones((2,))}]
        out = default_collate_fn(batch)
        assert out['x'].shape == (2, 2)

    def test_lists(self):
        batch = [[jnp.ones((2,)), jnp.zeros((2,))], [jnp.ones((2,)), jnp.zeros((2,))]]
        out = default_collate_fn(batch)
        assert isinstance(out, list)
        assert out[0].shape == (2, 2)

    def test_scalars(self):
        out = default_collate_fn([1, 2, 3])
        assert out.shape == (3,)
        out_f = default_collate_fn([1.0, 2.0])
        assert out_f.shape == (2,)

    def test_numpy_arrays(self):
        batch = [np.ones((4,)), np.zeros((4,))]
        out = default_collate_fn(batch)
        assert out.shape == (2, 4)

    def test_unknown_type_returns_as_is(self):
        batch = ['a', 'b']
        out = default_collate_fn(batch)
        assert out == ['a', 'b']


class TestDataLoader:
    """Tests for DataLoader."""

    def test_iteration_tuple_input(self):
        X = jnp.ones((100, 10))
        y = jnp.zeros((100, 2))
        loader = DataLoader((X, y), batch_size=32)
        assert len(loader) == 4
        batches = list(loader)
        assert len(batches) == 4
        assert batches[0][0].shape == (32, 10)
        # Last partial batch
        assert batches[-1][0].shape == (4, 10)

    def test_dict_input(self):
        data = {'x': jnp.ones((50, 3)), 'y': jnp.zeros((50,))}
        loader = DataLoader(data, batch_size=10)
        batch = next(iter(loader))
        assert batch['x'].shape == (10, 3)

    def test_dataset_input(self):
        dataset = DictDataset({'x': jnp.ones((20, 4))})
        loader = DataLoader(dataset, batch_size=5)
        assert len(loader) == 4

    def test_array_like_input(self):
        X = jnp.ones((20, 4))
        loader = DataLoader(X, batch_size=5)
        batch = next(iter(loader))
        assert batch[0].shape == (5, 4)

    def test_shuffle(self):
        X = jnp.arange(100).reshape(100, 1)
        loader = DataLoader((X,), batch_size=100, shuffle=True, seed=42)
        batch = next(iter(loader))[0]
        assert not jnp.all(batch[:10, 0] == jnp.arange(10))

    def test_shuffle_varies_between_epochs(self):
        # Successive epochs must reshuffle, otherwise every epoch sees the data
        # in the exact same order (the classic "shuffle does nothing" bug).
        X = jnp.arange(20).reshape(20, 1)
        loader = DataLoader((X,), batch_size=20, shuffle=True, seed=5)
        epoch1 = next(iter(loader))[0]
        epoch2 = next(iter(loader))[0]
        assert not jnp.all(epoch1 == epoch2)

    def test_shuffle_reproducible_across_loaders(self):
        # The per-epoch ordering must be reproducible: a fresh loader with the
        # same seed replays the identical sequence of epoch orderings.
        X = jnp.arange(20).reshape(20, 1)
        loader = DataLoader((X,), batch_size=20, shuffle=True, seed=5)
        epoch1 = next(iter(loader))[0]
        epoch2 = next(iter(loader))[0]

        loader2 = DataLoader((X,), batch_size=20, shuffle=True, seed=5)
        assert jnp.all(next(iter(loader2))[0] == epoch1)
        assert jnp.all(next(iter(loader2))[0] == epoch2)

    def test_shuffle_epoch_counter_advances(self):
        # Even a partially-consumed iterator advances the epoch counter so the
        # next pass reshuffles.
        X = jnp.arange(20).reshape(20, 1)
        loader = DataLoader((X,), batch_size=5, shuffle=True, seed=7)
        assert loader._epoch == 0
        next(iter(loader))  # pull only the first batch
        assert loader._epoch == 1

    def test_drop_last(self):
        X = jnp.ones((100, 10))
        loader = DataLoader((X,), batch_size=32, drop_last=True)
        assert len(loader) == 3
        assert len(list(loader)) == 3

    def test_custom_sampler(self):
        dataset = ArrayDataset(jnp.arange(10).reshape(10, 1))
        sampler = SequentialSampler(dataset)
        loader = DataLoader(dataset, batch_size=5, sampler=sampler)
        assert len(loader) == 2

    def test_custom_collate_fn(self):
        called = {'n': 0}

        def my_collate(batch):
            called['n'] += 1
            return default_collate_fn(batch)

        X = jnp.ones((10, 2))
        loader = DataLoader((X,), batch_size=5, collate_fn=my_collate)
        list(loader)
        assert called['n'] == 2

    def test_batch_sampler_argument(self):
        dataset = ArrayDataset(jnp.arange(10).reshape(10, 1))
        base = SequentialSampler(dataset)
        bs = BatchSampler(base, batch_size=4)
        loader = DataLoader(dataset, batch_sampler=bs)
        assert len(loader) == 3

    def test_batch_sampler_conflict_raises(self):
        dataset = ArrayDataset(jnp.arange(10).reshape(10, 1))
        base = SequentialSampler(dataset)
        bs = BatchSampler(base, batch_size=4)
        with pytest.raises(ValueError):
            DataLoader(dataset, batch_sampler=bs, shuffle=True)

    def test_invalid_dataset_type_raises(self):
        with pytest.raises(TypeError):
            DataLoader(12345, batch_size=4)

    def test_num_samples_property(self):
        X = jnp.ones((30, 2))
        loader = DataLoader((X,), batch_size=10)
        assert loader.num_samples == 30

    def test_set_epoch_plain_sampler(self):
        X = jnp.ones((30, 2))
        loader = DataLoader((X,), batch_size=10)
        # SequentialSampler has no set_epoch; should be a no-op without error
        loader.set_epoch(3)
        assert loader._epoch == 3


class TestDistributedDataLoader:
    """Tests for DistributedDataLoader on a single replica."""

    def test_tuple_input(self):
        X = jnp.ones((20, 4))
        y = jnp.zeros((20,))
        loader = DistributedDataLoader((X, y), batch_size=5, num_replicas=1, rank=0, shuffle=False)
        assert loader.num_replicas == 1
        assert loader.rank == 0
        batches = list(loader)
        assert batches[0][0].shape == (5, 4)

    def test_dict_input(self):
        data = {'x': jnp.ones((20, 4))}
        loader = DistributedDataLoader(data, batch_size=5, num_replicas=1, rank=0, shuffle=False)
        batch = next(iter(loader))
        assert batch['x'].shape == (5, 4)

    def test_array_input(self):
        X = jnp.ones((20, 4))
        loader = DistributedDataLoader(X, batch_size=5, num_replicas=1, rank=0, shuffle=False)
        batch = next(iter(loader))
        assert batch[0].shape == (5, 4)

    def test_dataset_input(self):
        dataset = ArrayDataset(jnp.ones((20, 4)))
        loader = DistributedDataLoader(dataset, batch_size=5, num_replicas=1, rank=0, shuffle=False)
        assert loader.num_samples == 20

    def test_set_epoch(self):
        X = jnp.ones((20, 4))
        loader = DistributedDataLoader(X, batch_size=5, num_replicas=1, rank=0, shuffle=True)
        loader.set_epoch(2)
        assert loader.batch_sampler.sampler.epoch == 2


class TestCreateDistributedBatches:
    """Tests for create_distributed_batches on a single device."""

    def test_array_data(self):
        X = jnp.arange(40).reshape(40, 1)
        batches = list(create_distributed_batches(X, batch_size=10))
        assert len(batches) == 4
        # Single device: (num_devices=1, batch_size=10, 1)
        assert batches[0].shape == (1, 10, 1)

    def test_tuple_data(self):
        X = jnp.arange(40).reshape(40, 1)
        y = jnp.arange(40).reshape(40, 1)
        batches = list(create_distributed_batches((X, y), batch_size=10))
        assert isinstance(batches[0], tuple)
        assert batches[0][0].shape == (1, 10, 1)
        assert batches[0][1].shape == (1, 10, 1)

    def test_dict_data(self):
        data = {'x': jnp.arange(40).reshape(40, 1)}
        batches = list(create_distributed_batches(data, batch_size=10))
        assert batches[0]['x'].shape == (1, 10, 1)

    def test_shuffle(self):
        X = jnp.arange(40).reshape(40, 1)
        batches = list(create_distributed_batches(X, batch_size=10, shuffle=True, seed=1))
        assert len(batches) == 4
        # Shuffled, so the first batch shouldn't be 0..9 in order
        flat = batches[0].reshape(-1)
        assert not jnp.all(flat == jnp.arange(10))

    def test_explicit_devices(self):
        import jax
        devices = jax.devices()
        n_dev = len(devices)
        X = jnp.arange(40).reshape(40, 1)
        batches = list(create_distributed_batches(X, batch_size=10, devices=devices))
        # total_batch_size = 10 * n_dev; 40 samples -> 40 // (10*n_dev) batches.
        assert len(batches) == 40 // (10 * n_dev)
        assert batches[0].shape == (n_dev, 10, 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
