"""
Hierarchical data containers for parameter and state management.

This module provides the ``Data`` class, a flexible container for hierarchical
data structures that supports dictionary-like and attribute-style access,
cloning, serialization, and composition.
"""

import dataclasses
from typing import Any, Dict

import brainstate
from brainstate.util.struct import dataclass

Array = brainstate.typing.ArrayLike

__all__ = [
    'Data',
]


def is_dataclass(cls):
    return hasattr(cls, '_brainstate_dataclass')


@dataclass
class Data:
    """
    Hierarchical state container for composed dynamics.

    Stores child states in a dictionary where keys match the attribute names
    of child dynamics in the parent dynamics class.

    Supports two initialization styles:
        - Data(children={'key1': data1, 'key2': data2})
        - Data(key1=data1, key2=data2)

    And two access styles:
        - cd['key1'] or cd.key1

    Attributes:
        children: Dict mapping child names to their states.
    """

    children: Dict[str, Any] = dataclasses.field(default_factory=dict, kw_only=True)

    def __init__(self, children: Dict[str, Any] = None, **kwargs):
        object.__setattr__(self, 'children', dict(children) if children is not None else {})
        self.children.update(kwargs)

    def __getattr__(self, key: str) -> Any:
        """Get child state by attribute name."""
        try:
            return self.children[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def __getitem__(self, key: str) -> Any:
        """Get child state by name."""
        return self.children[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set child state by name."""
        self.children[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if child exists."""
        return key in self.children

    def keys(self):
        """Return child keys."""
        return self.children.keys()

    def items(self):
        """Return child items."""
        return self.children.items()

    def values(self):
        """Return child values."""
        return self.children.values()

    def clone(self) -> 'Data':
        """
        Create a deep copy of the state, recursively cloning children.

        Returns:
            New state instance with cloned tensors.
        """
        cloned_children = {}
        for k, v in self.children.items():
            if v is None:
                cloned_children[k] = None
            elif hasattr(v, 'clone'):
                cloned_children[k] = v.clone()
            else:
                cloned_children[k] = v
        return self.__class__(children=cloned_children)

    @property
    def state_size(self) -> int:
        """Number of state variables per node."""
        total = 0
        for v in self.children.values():
            if isinstance(v, Data):
                total = total + v.state_size
            elif v is not None:
                total += 1
        return total

    @property
    def dtype(self):
        """Return dtype of first array child."""
        for v in self.children.values():
            if v is None:
                continue
            if isinstance(v, Data):
                try:
                    return v.dtype
                except ValueError:
                    continue
            if hasattr(v, 'dtype'):
                return v.dtype
        raise ValueError("No array children found to determine dtype")

    def add(self, *args, **updates) -> 'Data':
        children = {k: v for k, v in self.children.items()}
        for arg in args:
            assert isinstance(arg, (Data, dict)), 'Argument must be of type Data or Dict, got {}'.format(type(arg))
            for k, v in arg.items():
                children[k] = v
        for k in updates:
            children[k] = updates[k]
        return Data(children=children)

    def pop(self, *args) -> 'Data':
        children = {k: v for k, v in self.children.items()}
        for arg in args:
            children.pop(arg)
        return Data(children=children)

    def replace(self, **updates) -> 'Data':
        """
        Apply partial updates to child states.

        Args:
            updates: Dictionary of child states to update.

        Returns:
            New state instance with updated children.
        """
        children = {k: v for k, v in self.children.items()}
        for k in updates:
            children[k] = updates[k]
        return self.__class__(children=children)

    def to_dict(self) -> Dict:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary mapping state variable names to tensors.
        """
        return {k: d.to_dict() if isinstance(d, Data) else d for k, d in self.children.items()}

    @classmethod
    def from_dict(cls, d: Dict) -> 'Data':
        """
        Create state from dictionary.

        Args:
            d: Dictionary mapping state variable names to tensors.

        Returns:
            State instance.
        """
        return cls(children={k: cls.from_dict(v) if isinstance(v, dict) else v for k, v in d.items()})
