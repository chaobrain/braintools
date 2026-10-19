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


from typing import Callable, Any, Union, Optional

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

# ``random.normal`` accepts only float/complex dtypes, so non-float inputs fall
# back to the default float rather than being passed through.
_FLOAT_DTYPES = (jnp.float16, jnp.float32, jnp.float64, jnp.bfloat16)


def set_module_as(module: str) -> Callable[[Callable], Callable]:
    def wrapper(fun: Callable) -> Callable:
        fun.__module__ = module
        return fun

    return wrapper


def tree_map(f: Callable[..., Any], tree: Any, *rest: Any) -> Any:
    return jax.tree.map(f, tree, *rest, is_leaf=u.math.is_quantity)


def randn_like(y: brainstate.typing.ArrayLike) -> jax.Array:
    """``N(0, 1)`` samples with ``y``'s shape and float dtype.

    Preserving the dtype matters: with ``jax_enable_x64`` on, a float64 state must
    receive float64 noise (a float32 draw silently caps the precision of every SDE
    step that uses this), and a float16/bfloat16 state must not be promoted to
    float32. Matches ``brainstate.random.randn_like``, which also preserves dtype.
    """
    dtype = u.math.get_dtype(y)
    return brainstate.random.randn(
        *u.math.shape(y),
        dtype=dtype if dtype in _FLOAT_DTYPES else None,
    )


Initializer = Union[brainstate.typing.ArrayLike, Callable]
InitializerOrNot = Optional[Union[brainstate.typing.ArrayLike, Callable]]
