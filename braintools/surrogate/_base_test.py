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

"""
Tests for the base ``Surrogate`` class and the heaviside primitive plumbing
defined in ``braintools/surrogate/_base.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintools.surrogate as surrogate
from jax.interpreters import ad

from braintools.surrogate._base import (
    Surrogate,
    heaviside_p,
    _heaviside_abstract,
    _heaviside_imp,
    _heaviside_batching,
    _heaviside_transpose,
)


class _TanhSurrogate(Surrogate):
    """A concrete surrogate used to exercise the base-class machinery."""

    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_fun(self, x):
        return jnp.tanh(self.alpha * x) * 0.5 + 0.5

    def surrogate_grad(self, x):
        return self.alpha * (1 - jnp.tanh(self.alpha * x) ** 2) * 0.5


class TestHeavisidePrimitiveImpl:
    """Test the low-level heaviside primitive implementation helpers."""

    def test_abstract_eval_returns_input_shape(self):
        x = jnp.zeros((3, 4))
        dx = jnp.ones((3, 4))
        out = _heaviside_abstract(x, dx)
        assert isinstance(out, list)
        assert len(out) == 1
        assert out[0] is x

    @pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 0.5, 2.0])
    def test_imp_is_heaviside(self, x):
        x = jnp.array(x)
        dx = jnp.ones_like(x)
        out = _heaviside_imp(x, dx)
        assert isinstance(out, list)
        expected = jnp.asarray(x >= 0, dtype=x.dtype)
        assert jnp.allclose(out[0], expected)

    def test_imp_preserves_dtype(self):
        x = jnp.array([-1.0, 1.0], dtype=jnp.float32)
        dx = jnp.ones_like(x)
        out = _heaviside_imp(x, dx)
        assert out[0].dtype == x.dtype

    def test_primitive_is_multiple_results(self):
        assert heaviside_p.multiple_results is True

    def test_bind_returns_tuple(self):
        x = jnp.array([-1.0, 0.0, 1.0])
        dx = jnp.ones_like(x)
        out = heaviside_p.bind(x, dx)
        assert isinstance(out, (tuple, list))
        assert jnp.allclose(out[0], jnp.array([0.0, 1.0, 1.0]))


class TestSurrogateForward:
    """Test the forward (Heaviside) pass of the base ``__call__``."""

    @pytest.mark.parametrize("x", [-1.5, -0.5, 0.0, 0.5, 1.5])
    def test_forward_is_step(self, x):
        x = jnp.array(x)
        sg = _TanhSurrogate(alpha=2.0)
        y = sg(x)
        assert jnp.allclose(y, jnp.asarray(x >= 0, dtype=x.dtype))

    def test_forward_array(self):
        x = jnp.linspace(-3, 3, 21)
        sg = _TanhSurrogate(alpha=1.5)
        y = sg(x)
        assert jnp.allclose(y, jnp.asarray(x >= 0, dtype=x.dtype))

    def test_forward_under_jit(self):
        x = jnp.linspace(-2, 2, 11)
        sg = _TanhSurrogate(alpha=1.0)
        y = jax.jit(sg)(x)
        assert jnp.allclose(y, jnp.asarray(x >= 0, dtype=x.dtype))


class TestSurrogateGradient:
    """Test that backprop uses the custom surrogate gradient (JVP plumbing)."""

    @pytest.mark.parametrize("x", [-1.0, -0.25, 0.0, 0.25, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_grad_matches_surrogate_grad(self, x, alpha):
        x = jnp.array(x)
        sg = _TanhSurrogate(alpha=alpha)
        grad = jax.grad(lambda xi: sg(xi).sum())(x)
        assert jnp.allclose(grad, sg.surrogate_grad(x))

    def test_vmap_grad_matches(self):
        xs = jnp.linspace(-2, 2, 41)
        sg = _TanhSurrogate(alpha=2.0)
        grads = jax.vmap(jax.grad(lambda xi: sg(xi)))(xs)
        assert jnp.allclose(grads, sg.surrogate_grad(xs))

    def test_jit_grad_matches(self):
        xs = jnp.linspace(-2, 2, 21)
        sg = _TanhSurrogate(alpha=1.0)
        grads = jax.jit(jax.vmap(jax.grad(lambda xi: sg(xi))))(xs)
        assert jnp.allclose(grads, sg.surrogate_grad(xs))

    def test_grad_is_zero_when_no_cotangent(self):
        # Output does not depend on input -> Zero tangent path in the JVP.
        x = jnp.array([0.5, -0.5])
        sg = _TanhSurrogate(alpha=1.0)

        def f(xi):
            # discard the surrogate output so its tangent is Zero
            sg(xi)
            return (xi * 2.0).sum()

        grad = jax.grad(f)(x)
        assert jnp.allclose(grad, jnp.array([2.0, 2.0]))

    def test_second_order_grad_runs(self):
        # The straight-through estimator gives a constant surrogate grad block;
        # the second derivative is well-defined for the tanh surrogate.
        x = jnp.array(0.3)
        sg = _TanhSurrogate(alpha=1.0)
        g2 = jax.grad(jax.grad(lambda xi: sg(xi)))(x)
        assert jnp.isfinite(g2)


class TestHeavisideBatching:
    """Exercise the batching rule branches via ``jax.vmap``."""

    def test_vmap_over_x_only(self):
        # dx depends on x (closed over), x is the batched arg -> x_axis branch.
        xs = jnp.linspace(-2, 2, 8)
        sg = _TanhSurrogate(alpha=1.0)
        ys = jax.vmap(sg)(xs)
        assert jnp.allclose(ys, jnp.asarray(xs >= 0, dtype=xs.dtype))

    def test_vmap_both_args_same_axis(self):
        # Bind heaviside_p directly with both args batched on axis 0.
        x = jnp.linspace(-1, 1, 5)
        dx = jnp.ones_like(x)
        out = jax.vmap(lambda a, b: heaviside_p.bind(a, b)[0])(x, dx)
        assert jnp.allclose(out, jnp.asarray(x >= 0, dtype=x.dtype))

    def test_vmap_both_args_different_axes(self):
        # x batched on axis 0, dx batched on axis 1 -> moveaxis branch.
        x = jnp.linspace(-1, 1, 4)  # batched on axis 0
        dx = jnp.ones((3, 4))  # batched on axis 1

        def f(a, b):
            return heaviside_p.bind(a, b)[0]

        out = jax.vmap(f, in_axes=(0, 1))(x, dx)
        assert jnp.allclose(out, jnp.asarray(x >= 0, dtype=x.dtype))

    def test_vmap_only_dx_batched(self):
        # x is a scalar closed over (not batched), dx is batched -> repeat branch.
        x_scalar = jnp.array(0.5)
        dx = jnp.ones((6,))

        def f(b):
            return heaviside_p.bind(x_scalar, b)[0]

        out = jax.vmap(f)(dx)
        assert out.shape == (6,)
        assert jnp.allclose(out, jnp.ones((6,)))

    def test_vmap_only_dx_batched_negative_x(self):
        x_scalar = jnp.array(-0.5)
        dx = jnp.ones((6,))

        def f(b):
            return heaviside_p.bind(x_scalar, b)[0]

        out = jax.vmap(f)(dx)
        assert jnp.allclose(out, jnp.zeros((6,)))

    def test_batching_rule_neither_axis_batched(self):
        # Directly exercise the (None, None) branch: neither argument batched.
        x = jnp.array([-1.0, 0.0, 1.0])
        dx = jnp.ones_like(x)
        result, out_axes = _heaviside_batching((x, dx), (None, None))
        assert out_axes == (None,)
        assert jnp.allclose(result[0], jnp.array([0.0, 1.0, 1.0]))


class TestHeavisideTranspose:
    """Directly exercise the (unregistered) transpose rule for completeness."""

    def test_transpose_with_defined_dx(self):
        ct = (jnp.array([1.0, 2.0, 3.0]),)
        x = jnp.array([0.1, 0.2, 0.3])
        dx = jnp.array([0.5, 0.5, 0.5])
        cot_tx, cot_tdx = _heaviside_transpose(ct, x, dx)
        assert jnp.allclose(cot_tx, dx * ct[0])
        assert jnp.allclose(cot_tdx, ct[0])

    def test_transpose_with_undefined_dx(self):
        ct = (jnp.array([1.0, 2.0]),)
        x = jnp.array([0.1, 0.2])
        # An UndefinedPrimal residual -> the dx*ct branch returns a Zero.
        undefined_dx = ad.UndefinedPrimal(jax.ShapeDtypeStruct((2,), jnp.float32))
        cot_tx, cot_tdx = _heaviside_transpose(ct, x, undefined_dx)
        assert isinstance(cot_tx, ad.Zero)
        assert jnp.allclose(cot_tdx, ct[0])


class TestSurrogateDunders:
    """Test base/concrete repr, hashing and equality behaviour."""

    def test_repr_of_concrete(self):
        sg = surrogate.Sigmoid(alpha=4.0)
        assert repr(sg) == 'Sigmoid(alpha=4.0)'

    def test_pretty_repr_base(self):
        # PrettyObject provides a default repr for subclasses without __repr__.
        sg = _TanhSurrogate(alpha=1.0)
        r = repr(sg)
        assert isinstance(r, str)
        assert '_TanhSurrogate' in r

    def test_hash_is_stable(self):
        a = surrogate.Sigmoid(alpha=4.0)
        b = surrogate.Sigmoid(alpha=4.0)
        assert hash(a) == hash(b)

    def test_hash_differs_with_param(self):
        a = surrogate.Sigmoid(alpha=4.0)
        c = surrogate.Sigmoid(alpha=2.0)
        assert hash(a) != hash(c)

    def test_usable_as_dict_key(self):
        key = surrogate.Sigmoid(alpha=4.0)
        d = {key: 'x'}
        assert d[key] == 'x'


class TestBaseNotImplemented:
    """The abstract base methods must raise when not overridden."""

    def test_surrogate_fun_not_implemented(self):
        base = Surrogate()
        with pytest.raises(NotImplementedError):
            base.surrogate_fun(jnp.array(0.0))

    def test_surrogate_grad_not_implemented(self):
        base = Surrogate()
        with pytest.raises(NotImplementedError):
            base.surrogate_grad(jnp.array(0.0))

    def test_calling_base_raises(self):
        base = Surrogate()
        with pytest.raises(NotImplementedError):
            base(jnp.array(0.0))


class TestSurrogateFunConsistency:
    """The smooth surrogate_fun should be consistent with surrogate_grad."""

    def test_surrogate_fun_autodiff_matches_surrogate_grad(self):
        xs = jnp.linspace(-2, 2, 41)
        sg = _TanhSurrogate(alpha=1.5)
        auto = jax.vmap(jax.grad(sg.surrogate_fun))(xs)
        assert jnp.allclose(auto, sg.surrogate_grad(xs), atol=1e-5)

    def test_surrogate_fun_values(self):
        sg = _TanhSurrogate(alpha=1.0)
        # tanh(0)*0.5 + 0.5 == 0.5
        assert np.allclose(float(sg.surrogate_fun(jnp.array(0.0))), 0.5)
