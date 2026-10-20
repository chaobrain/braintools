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

"""Feature encoding system for cognitive task inputs and outputs."""

import copy
from typing import List, Optional, Tuple, Union, overload

import jax.numpy as jnp

from ._typing import Data

__all__ = [
    'Feature',
    'FeatureSet',
    'CircleFeature',
    'is_feature',
    'as_feature',
]


def is_feature(x) -> bool:
    """Return True iff ``x`` is a :class:`Feature` instance."""
    return isinstance(x, Feature)


def as_feature(x) -> 'Feature':
    """Return ``x`` if it is a :class:`Feature`; otherwise raise ``TypeError``."""
    if isinstance(x, Feature):
        return x
    raise TypeError(f'Expected a {Feature.__name__} instance, got {type(x).__name__}.')


class _FeatureBase:
    """
    Base class for features with index tracking.
    """

    def __init__(self, num: int):
        self._num = num
        self._start = 0
        self._end = self._num

    @property
    def num(self) -> int:
        """Number of dimensions for this feature."""
        return self._num

    @property
    def i(self) -> slice:
        """Index slice for this feature."""
        return slice(self._start, self._end)

    def __add__(self, other: Union['Feature', 'FeatureSet']) -> 'FeatureSet':
        raise NotImplementedError

    def shift(self, num: int):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


def _placed_copy(ft: 'Feature', start: int) -> 'Feature':
    """Copy ``ft`` and move the copy so that its slice begins at ``start``."""
    c = ft.copy()
    c.shift(start - c._start)
    return c


def _placed_copies(fs: 'FeatureSet', start: int) -> List['Feature']:
    """Copy the features of ``fs``, keeping their layout, so the set begins at ``start``.

    The layout is rebased on the smallest feature start rather than on the set's
    own ``_start``: a set built directly from features taken out of another set
    (``FeatureSet(fs['b'])``) keeps ``_start == 0`` while its features carry
    their old offsets.
    """
    base = min((ft._start for ft in fs.fts), default=fs._start)
    return [_placed_copy(ft, start + ft._start - base) for ft in fs.fts]


class FeatureSet(_FeatureBase):
    """
    Collection of features with automatic index management.

    Features in a set have their indices adjusted automatically
    to avoid overlap. Supports composition via +, -, | operators.

    Parameters
    ----------
    *fts
        Features to include in the set.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.cogtask import Feature
        >>> features = Feature(1, 'fixation') + Feature(8, 'stimulus')
        >>> features
        FeatureSet(names=['fixation', 'stimulus'], nums=[1, 8])
        >>> features['fixation'].i, features['stimulus'].i
        (slice(0, 1, None), slice(1, 9, None))
        >>> 'stimulus' in features, len(features), features.num
        (True, 2, 9)
        >>>
        >>> # Removing a feature re-packs the remaining slices
        >>> rest = (features + Feature(2, 'choice')) - 'stimulus'
        >>> rest['choice'].i
        slice(1, 3, None)
    """

    def __init__(self, *fts):
        self.fts = fts
        self._s = dict()
        for ft in fts:
            name = ft.name
            if name in self._s:
                raise ValueError(f'Duplicate feature name {name}')
            else:
                self._s[name] = ft
        num = sum([ft.num for ft in fts])
        super().__init__(num)

    def shift(self, num: int):
        """Shift all indices by the given amount.

        Warning
        -------
        This method mutates the FeatureSet in-place, unlike `__add__` which creates a copy.
        If you need to preserve the original FeatureSet, create a copy before calling shift().

        Parameters
        ----------
        num : int
            The amount to shift indices by (can be positive or negative).
        """
        self._start += num
        self._end += num
        for ft in self.fts:
            ft._start += num
            ft._end += num

    @overload
    def __getitem__(self, item: Union[str, int]) -> 'Feature': ...

    @overload
    def __getitem__(self, item: slice) -> Tuple['Feature', ...]: ...

    def __getitem__(self, item: Union[str, int, slice]) -> Union['Feature', Tuple['Feature', ...]]:
        """
        Access features by name or index.

        Parameters
        ----------
        item : str or int or slice
            Feature name, index, or slice.

        Returns
        -------
        Feature or tuple of Features
        """
        if isinstance(item, str):
            return self._s[item]
        elif isinstance(item, (int, slice)):
            return tuple(self._s.values())[item]
        else:
            raise ValueError

    def __add__(self, other: Union['Feature', 'FeatureSet']) -> 'FeatureSet':
        """
        Compose this feature set with another feature or feature set.

        This operation is IMMUTABLE - all features are copied before
        any modifications are made. The result is laid out contiguously from
        index 0, whatever offsets the operands carried inside another set.
        """
        # Copy all features from self, laid out from index 0
        self_copies = _placed_copies(self, 0)

        if isinstance(other, Feature):
            return FeatureSet(*self_copies, _placed_copy(other, self._num))
        elif isinstance(other, FeatureSet):
            return FeatureSet(*self_copies, *_placed_copies(other, self._num))
        else:
            raise TypeError(
                f'Only support addition with {Feature.__name__} or {FeatureSet.__name__}, '
                f'but got {type(other).__name__}'
            )

    def __sub__(self, other: Union[str, 'Feature']) -> Union['Feature', 'FeatureSet']:
        """
        Remove a feature from this set by name.

        Returns a new FeatureSet without the specified feature, with the
        remaining slices re-packed from index 0. When only one feature
        remains, that feature is returned on its own.
        """
        if isinstance(other, str):
            name_to_remove: Optional[str] = other
        elif isinstance(other, Feature):
            name_to_remove = other.name
        else:
            raise TypeError(f"Cannot subtract {type(other).__name__} from FeatureSet")

        # Zero-based copies: the originals carry their slices inside ``self``.
        remaining = [_placed_copy(ft, 0) for ft in self.fts if ft.name != name_to_remove]
        if len(remaining) == len(self.fts):
            raise KeyError(f"Feature '{name_to_remove}' not found in set")
        if len(remaining) == 0:
            raise ValueError("Cannot remove the last feature from set")

        # Recalculate indices by re-composing
        result: Union['Feature', 'FeatureSet'] = remaining[0]
        for ft in remaining[1:]:
            result = result + ft
        return result

    def __or__(self, other: Union['Feature', 'FeatureSet']) -> 'FeatureSet':
        """Union operator as alias for composition (+)."""
        return self.__add__(other)

    def __repr__(self):
        names = [ft.name for ft in self.fts]
        nums = [ft.num for ft in self.fts]
        return f'{self.__class__.__name__}(names={names}, nums={nums})'

    def copy(self) -> 'FeatureSet':
        """Create a deep copy of this feature set.

        The copy keeps the layout of the original: its features and its own
        slice ``i`` sit at the same (possibly shifted) indices.
        """
        copied_features = [ft.copy() for ft in self.fts]
        new_set = FeatureSet(*copied_features)
        new_set._start = self._start
        new_set._end = self._end
        return new_set

    def __iter__(self):
        """Iterate over features in this set."""
        return iter(self.fts)

    def __len__(self) -> int:
        """Return the number of features in this set."""
        return len(self.fts)

    def __contains__(self, item: str) -> bool:
        """Check if a feature with the given name exists."""
        return item in self._s


class Feature(_FeatureBase):
    """
    Individual feature encoder for cognitive task inputs/outputs.

    A Feature defines the encoding dimensions for a single input or output
    channel in a cognitive task.

    Parameters
    ----------
    num : int
        Number of dimensions for this feature.
    name : str, optional
        Feature name. Required for composition.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.cogtask import Feature
        >>> fix = Feature(1, 'fixation')
        >>> stim = Feature(8, 'stimulus')
        >>> fix.num, fix.i
        (1, slice(0, 1, None))
        >>>
        >>> # Compose features into a FeatureSet
        >>> all_features = fix + stim
        >>> all_features.num
        9
        >>>
        >>> # Repeat a feature with indexed names
        >>> Feature(2, 'choice') * 3
        FeatureSet(names=['choice_0', 'choice_1', 'choice_2'], nums=[2, 2, 2])
    """

    def __init__(
        self,
        num: int,
        name: Optional[str] = None,
    ):
        self.name = name
        super().__init__(num)

    def shift(self, num: int):
        """Shift indices by the given amount."""
        self._start += num
        self._end += num

    def __add__(self, other: Union['Feature', 'FeatureSet']) -> 'FeatureSet':
        """
        Compose this feature with another feature or feature set.

        This operation is IMMUTABLE - both operands are copied before
        any modifications are made. The result is laid out contiguously from
        index 0, whatever offsets the operands carried inside another set.
        """
        # Copy self to avoid mutation, laid out from index 0 (a feature taken
        # out of a set still carries its offset inside that set).
        self_copy = _placed_copy(self, 0)

        if isinstance(other, Feature):
            return FeatureSet(self_copy, _placed_copy(other, self_copy.num))
        elif isinstance(other, FeatureSet):
            return FeatureSet(self_copy, *_placed_copies(other, self_copy.num))
        else:
            raise TypeError(
                f'Only support addition with {Feature.__name__} or {FeatureSet.__name__}, '
                f'but got {type(other).__name__}'
            )

    def __or__(self, other: Union['Feature', 'FeatureSet']) -> 'FeatureSet':
        """Union operator as alias for composition (+)."""
        return self.__add__(other)

    def __mul__(self, count: int) -> Union['Feature', 'FeatureSet']:
        """
        Create multiple copies of this feature with indexed names.

        Example: Feature(1, 'choice') * 3
        Creates: choice_0, choice_1, choice_2
        """
        if not isinstance(count, int) or count < 1:
            raise ValueError("Repeat count must be a positive integer")

        base_name = self.name or 'feature'
        features = []
        for i in range(count):
            ft = _placed_copy(self, 0)
            ft.name = f"{base_name}_{i}"
            features.append(ft)

        result: Union[Feature, FeatureSet] = features[0]
        for ft in features[1:]:
            result = result + ft
        return result

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.name}", num={self.num})'

    def copy(self) -> 'Feature':
        """Create a shallow copy of this feature."""
        return copy.copy(self)


class CircleFeature(Feature):
    """
    Circular feature for angular/directional data.

    Extends Feature with value range constraints, useful for
    encoding circular quantities like angles or directions.

    Parameters
    ----------
    num : int
        Number of dimensions for this feature.
    limits : tuple
        (min, max) value range for the circular feature.
    name : str, optional
        Feature name.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.cogtask import CircleFeature
        >>> direction = CircleFeature(8, limits=(0, 2 * np.pi), name='direction')
        >>> direction
        CircleFeature("direction", num=8)
        >>> direction.v_min, round(direction.v_range, 4)
        (0, 6.2832)
    """

    def __init__(
        self,
        num: int,
        limits: Tuple[Data, Data] = (0., jnp.pi * 2),
        name: Optional[str] = None,
    ):
        super().__init__(num, name)
        assert len(limits) == 2
        self.v_min = limits[0]
        self.v_max = limits[1]
        self.v_range = limits[1] - limits[0]


# Backwards compatibility alias
_FeatSet = FeatureSet
