"""
Comprehensive tests for _init.py

Tests cover:
- Basic Initialization class functionality
- Arithmetic operations (+, -, *, /)
- Composition operations (pipe, Compose)
- Transformation methods (clip, add, multiply, apply)
- init_call helper function
- Edge cases and error handling
"""

import brainunit as u
import numpy as np
import pytest

from braintools.init._init_base import (
    Initialization,
    param,
    Compose,
)


class SimpleInit(Initialization):
    """Simple test initialization that returns constant value."""

    def __init__(self, value):
        self.value = value

    def __call__(self, size, **kwargs):
        if isinstance(size, int):
            return np.full(size, self.value)
        return np.full(size, self.value)

    def __repr__(self):
        return f"SimpleInit({self.value})"


class QuantityInit(Initialization):
    """Test initialization that returns quantity values."""

    def __init__(self, value):
        self.value = value

    def __call__(self, size, **kwargs):
        if isinstance(size, int):
            return u.math.full(size, self.value)
        return u.math.full(size, self.value)

    def __repr__(self):
        return f"QuantityInit({self.value})"


class RandomInit(Initialization):
    """Test initialization that returns random values."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, size, **kwargs):
        rng = kwargs.get('rng', np.random)
        return rng.normal(self.mean, self.std, size)

    def __repr__(self):
        return f"RandomInit({self.mean}, {self.std})"


class TestBasicInitialization:
    """Test basic Initialization class functionality."""

    def test_simple_initialization(self):
        """Test basic initialization creation and calling."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)

        result = init(10, rng=rng)
        assert result.shape == (10,)
        assert np.all(result == 5.0)

    def test_quantity_initialization(self):
        """Test initialization with brainunit quantities."""
        rng = np.random.default_rng(42)
        init = QuantityInit(2.5 * u.nS)

        result = init(10, rng=rng)
        assert result.shape == (10,)
        assert isinstance(result, u.Quantity)
        assert np.all(result.to(u.nS).mantissa == 2.5)

    def test_random_initialization(self):
        """Test initialization with random values."""
        rng = np.random.default_rng(42)
        init = RandomInit(0.0, 1.0)

        result = init(100, rng=rng)
        assert result.shape == (100,)
        assert -3.0 < result.mean() < 3.0
        assert 0.5 < result.std() < 1.5

    def test_kwargs_passing(self):
        """Test that kwargs are properly passed to initialization."""

        class KwargsInit(Initialization):

            def __call__(self, size, custom_param=None, **kwargs):
                if custom_param is not None:
                    return np.full(size, custom_param)
                return np.zeros(size)

        rng = np.random.default_rng(42)
        init = KwargsInit()

        result = init(5, custom_param=7.0, rng=rng)
        assert np.all(result == 7.0)


class TestArithmeticOperations:
    """Test arithmetic operations on Initialization objects."""

    def test_addition_with_scalar(self):
        """Test adding a scalar to an initialization."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)
        result_init = init + 3.0

        result = result_init(10, rng=rng)
        assert np.all(result == 8.0)

    def test_addition_with_quantity(self):
        """Test adding a quantity to an initialization."""
        rng = np.random.default_rng(42)
        init = QuantityInit(2.0 * u.nS)
        result_init = init + 1.0 * u.nS

        result = result_init(10, rng=rng)
        assert np.all(result.to(u.nS).mantissa == 3.0)

    def test_addition_with_initialization(self):
        """Test adding two initializations."""
        rng = np.random.default_rng(42)
        init1 = SimpleInit(5.0)
        init2 = SimpleInit(3.0)
        result_init = init1 + init2

        result = result_init(10, rng=rng)
        assert np.all(result == 8.0)

    def test_right_addition(self):
        """Test right addition (scalar + init)."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)
        result_init = 3.0 + init

        result = result_init(10, rng=rng)
        assert np.all(result == 8.0)

    def test_subtraction_with_scalar(self):
        """Test subtracting a scalar from an initialization."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)
        result_init = init - 2.0

        result = result_init(10, rng=rng)
        assert np.all(result == 3.0)

    def test_subtraction_with_initialization(self):
        """Test subtracting two initializations."""
        rng = np.random.default_rng(42)
        init1 = SimpleInit(5.0)
        init2 = SimpleInit(2.0)
        result_init = init1 - init2

        result = result_init(10, rng=rng)
        assert np.all(result == 3.0)

    def test_right_subtraction(self):
        """Test right subtraction (scalar - init)."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)
        result_init = 10.0 - init

        result = result_init(10, rng=rng)
        assert np.all(result == 5.0)

    def test_multiplication_with_scalar(self):
        """Test multiplying an initialization by a scalar."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)
        result_init = init * 3.0

        result = result_init(10, rng=rng)
        assert np.all(result == 15.0)

    def test_multiplication_with_initialization(self):
        """Test multiplying two initializations."""
        rng = np.random.default_rng(42)
        init1 = SimpleInit(5.0)
        init2 = SimpleInit(3.0)
        result_init = init1 * init2

        result = result_init(10, rng=rng)
        assert np.all(result == 15.0)

    def test_right_multiplication(self):
        """Test right multiplication (scalar * init)."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)
        result_init = 3.0 * init

        result = result_init(10, rng=rng)
        assert np.all(result == 15.0)

    def test_division_with_scalar(self):
        """Test dividing an initialization by a scalar."""
        rng = np.random.default_rng(42)
        init = SimpleInit(10.0)
        result_init = init / 2.0

        result = result_init(10, rng=rng)
        assert np.all(result == 5.0)

    def test_division_with_initialization(self):
        """Test dividing two initializations."""
        rng = np.random.default_rng(42)
        init1 = SimpleInit(10.0)
        init2 = SimpleInit(2.0)
        result_init = init1 / init2

        result = result_init(10, rng=rng)
        assert np.all(result == 5.0)

    def test_right_division(self):
        """Test right division (scalar / init)."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)
        result_init = 20.0 / init

        result = result_init(10, rng=rng)
        assert np.all(result == 4.0)

    def test_chained_arithmetic(self):
        """Test chaining multiple arithmetic operations."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)
        result_init = (init + 3.0) * 2.0 - 1.0

        result = result_init(10, rng=rng)
        assert np.all(result == 15.0)

    def test_arithmetic_with_quantities(self):
        """Test arithmetic operations with quantities."""
        rng = np.random.default_rng(42)
        init = QuantityInit(2.0 * u.nS)
        result_init = init * 3.0 + 1.0 * u.nS

        result = result_init(10, rng=rng)
        assert np.all(result.to(u.nS).mantissa == 7.0)


class TestTransformationMethods:
    """Test transformation methods on Initialization objects."""

    def test_clip_both_bounds(self):
        """Test clipping with both min and max bounds."""
        rng = np.random.default_rng(42)
        init = RandomInit(0.0, 5.0)
        clipped = init.clip(-2.0, 2.0)

        result = clipped(100, rng=rng)
        assert np.all(result >= -2.0)
        assert np.all(result <= 2.0)

    def test_clip_min_only(self):
        """Test clipping with only minimum bound."""
        rng = np.random.default_rng(42)
        init = RandomInit(0.0, 5.0)
        clipped = init.clip(min_val=0.0)

        result = clipped(100, rng=rng)
        assert np.all(result >= 0.0)

    def test_clip_max_only(self):
        """Test clipping with only maximum bound."""
        rng = np.random.default_rng(42)
        init = RandomInit(0.0, 5.0)
        clipped = init.clip(max_val=2.0)

        result = clipped(100, rng=rng)
        assert np.all(result <= 2.0)

    def test_clip_with_quantities(self):
        """Test clipping with quantity values."""
        rng = np.random.default_rng(42)
        init = QuantityInit(5.0 * u.nS)
        clipped = init.clip(1.0 * u.nS, 3.0 * u.nS)

        result = clipped(10, rng=rng)
        result_vals = result.to(u.nS).mantissa
        assert np.all(result_vals == 3.0)

    def test_add_method(self):
        """Test .add() method."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)
        result_init = init.add(3.0)

        result = result_init(10, rng=rng)
        assert np.all(result == 8.0)

    def test_multiply_method(self):
        """Test .multiply() method."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)
        result_init = init.multiply(3.0)

        result = result_init(10, rng=rng)
        assert np.all(result == 15.0)

    def test_apply_method(self):
        """Test .apply() method with custom function."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)
        result_init = init.apply(lambda x: x ** 2)

        result = result_init(10, rng=rng)
        assert np.all(result == 25.0)

    def test_apply_with_numpy_function(self):
        """Test .apply() with numpy function."""
        rng = np.random.default_rng(42)
        init = RandomInit(1.0, 0.1)
        result_init = init.apply(np.abs)

        result = result_init(100, rng=rng)
        assert np.all(result >= 0.0)

    def test_chained_transformations(self):
        """Test chaining multiple transformation methods."""
        rng = np.random.default_rng(42)
        init = RandomInit(0.0, 2.0)
        transformed = init.clip(-1.0, 1.0).add(5.0).multiply(2.0)

        result = transformed(100, rng=rng)
        assert np.all(result >= 8.0)
        assert np.all(result <= 12.0)


class TestPipeOperation:
    """Test pipe operator for functional composition."""

    def test_pipe_with_function(self):
        """Test pipe operator with simple function."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0) | (lambda x: x * 2)

        result = init(10, rng=rng)
        assert np.all(result == 10.0)

    def test_pipe_with_multiple_functions(self):
        """Test chaining multiple pipes."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0) | (lambda x: x * 2) | (lambda x: x + 3)

        result = init(10, rng=rng)
        assert np.all(result == 13.0)

    def test_pipe_with_numpy_functions(self):
        """Test pipe with numpy functions."""
        rng = np.random.default_rng(42)
        init = RandomInit(0.0, 1.0) | np.abs | np.sqrt

        result = init(100, rng=rng)
        assert np.all(result >= 0.0)

    def test_pipe_with_quantity_function(self):
        """Test pipe with quantity operations."""
        rng = np.random.default_rng(42)
        init = QuantityInit(2.0 * u.nS) | (lambda x: u.math.maximum(x, 1.0 * u.nS))

        result = init(10, rng=rng)
        assert np.all(result.to(u.nS).mantissa == 2.0)

    def test_pipe_error_handling(self):
        """Test that pipe raises error for non-callable."""
        init = SimpleInit(5.0)
        with pytest.raises(TypeError):
            result_init = init | 5.0
            rng = np.random.default_rng(42)
            result_init(10, rng=rng)


class TestComposeClass:
    """Test Compose class for functional composition."""

    def test_compose_single_init(self):
        """Test Compose with single initialization."""
        rng = np.random.default_rng(42)
        init = Compose(SimpleInit(5.0))

        result = init(10, rng=rng)
        assert np.all(result == 5.0)

    def test_compose_multiple_functions(self):
        """Test Compose with multiple functions."""
        rng = np.random.default_rng(42)
        init = Compose(
            SimpleInit(5.0),
            lambda x: x * 2,
            lambda x: x + 3
        )

        result = init(10, rng=rng)
        assert np.all(result == 13.0)

    def test_compose_mixed_init_and_functions(self):
        """Test Compose with mix of initializations and functions."""
        rng = np.random.default_rng(42)
        init = Compose(
            RandomInit(0.0, 1.0),
            np.abs,
            lambda x: x * 10
        )

        result = init(100, rng=rng)
        assert np.all(result >= 0.0)

    def test_compose_with_quantities(self):
        """Test Compose with quantity operations."""
        rng = np.random.default_rng(42)
        init = Compose(
            QuantityInit(2.0 * u.nS),
            lambda x: u.math.maximum(x, 1.0 * u.nS),
            lambda x: x * 2.0
        )

        result = init(10, rng=rng)
        assert np.all(result.to(u.nS).mantissa == 4.0)

    def test_compose_empty_error(self):
        """Test that Compose raises error when empty."""
        with pytest.raises(ValueError):
            Compose()

    def test_compose_repr(self):
        """Test Compose string representation."""
        init = Compose(SimpleInit(5.0), lambda x: x * 2)
        repr_str = repr(init)
        assert "Compose" in repr_str


class TestInitCall:
    """Test init_call helper function."""

    def test_init_call_with_initialization(self):
        """Test init_call with Initialization object."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)

        result = param(init, 10, rng=rng)
        assert result.shape == (10,)
        assert np.all(result == 5.0)

    def test_init_call_with_none(self):
        """Test init_call with None returns None."""
        rng = np.random.default_rng(42)
        result = param(None, 10, rng=rng)
        assert result is None

    def test_init_call_with_float(self):
        """Test init_call with float scalar."""
        rng = np.random.default_rng(42)
        result = param(5.0, 10, rng=rng)
        assert result == 5.0

    def test_init_call_with_int(self):
        """Test init_call with int scalar."""
        rng = np.random.default_rng(42)
        result = param(5, 10, rng=rng)
        assert result == 5

    def test_init_call_with_scalar_quantity(self):
        """Test init_call with scalar quantity."""
        rng = np.random.default_rng(42)
        result = param(5.0 * u.nS, 10, rng=rng)
        assert isinstance(result, u.Quantity)
        assert result.to(u.nS).mantissa == 5.0

    def test_init_call_with_array_correct_size(self):
        """Test init_call with array of correct size."""
        rng = np.random.default_rng(42)
        arr = np.ones(10)
        result = param(arr, 10, rng=rng)
        assert np.array_equal(result, arr)

    def test_init_call_with_quantity_array_correct_size(self):
        """Test init_call with quantity array of correct size."""
        rng = np.random.default_rng(42)
        arr = np.ones(10) * u.nS
        result = param(arr, 10, rng=rng)
        assert u.math.allclose(result, arr)

    def test_init_call_passes_kwargs(self):
        """Test that init_call passes kwargs to initialization."""

        class KwargsInit(Initialization):

            def __call__(self, size, custom_param=None, **kwargs):
                if custom_param is not None:
                    return np.full(size, custom_param)
                return np.zeros(size)

        rng = np.random.default_rng(42)
        init = KwargsInit()
        result = param(init, 10, custom_param=7.0, rng=rng)
        assert np.all(result == 7.0)


class TestInternalClasses:
    """Test internal composition classes."""

    def test_binary_op_init_repr(self):
        """Test string representations of binary operations."""
        init1 = SimpleInit(5.0)
        init2 = SimpleInit(3.0)

        add = init1 + init2
        assert "+" in repr(add)

        sub = init1 - init2
        assert "-" in repr(sub)

        mul = init1 * init2
        assert "*" in repr(mul)

        div = init1 / init2
        assert "/" in repr(div)

    def test_clip_init_repr(self):
        """Test ClipInit string representation."""
        init = SimpleInit(5.0)
        clipped = init.clip(0.0, 10.0)
        assert "clip" in repr(clipped)

    def test_apply_init_repr(self):
        """Test ApplyInit string representation."""
        init = SimpleInit(5.0)
        applied = init.apply(lambda x: x * 2)
        assert "apply" in repr(applied)

    def test_pipe_init_repr(self):
        """Test PipeInit string representation."""
        init = SimpleInit(5.0) | (lambda x: x * 2)
        assert "|" in repr(init)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_size_array(self):
        """Test initialization with size 0."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)
        result = init(0, rng=rng)
        assert result.shape == (0,)

    def test_large_array(self):
        """Test initialization with large size."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)
        result = init(100000, rng=rng)
        assert result.shape == (100000,)
        assert np.all(result == 5.0)

    def test_multidimensional_size(self):
        """Test initialization with tuple size."""
        rng = np.random.default_rng(42)
        init = SimpleInit(5.0)
        result = init((10, 5), rng=rng)
        assert result.shape == (10, 5)
        assert np.all(result == 5.0)

    def test_negative_values(self):
        """Test initialization with negative values."""
        rng = np.random.default_rng(42)
        init = SimpleInit(-5.0)
        result = init(10, rng=rng)
        assert np.all(result == -5.0)

    def test_very_small_values(self):
        """Test initialization with very small values."""
        rng = np.random.default_rng(42)
        init = SimpleInit(1e-10)
        result = init(10, rng=rng)
        assert np.allclose(result, 1e-10)

    def test_very_large_values(self):
        """Test initialization with very large values."""
        rng = np.random.default_rng(42)
        init = SimpleInit(1e10)
        result = init(10, rng=rng)
        assert np.allclose(result, 1e10)

    def test_complex_nested_operations(self):
        """Test deeply nested operations."""
        rng = np.random.default_rng(42)
        init = SimpleInit(1.0)
        complex_init = ((init + 1) * 2 - 3) / 4
        complex_init = complex_init.clip(-10, 10).add(5).multiply(0.5)

        result = complex_init(10, rng=rng)
        assert result.shape == (10,)

    def test_operations_preserve_units(self):
        """Test that operations preserve units correctly."""
        rng = np.random.default_rng(42)
        init = QuantityInit(2.0 * u.nS)

        result1 = (init * 2.0)(10, rng=rng)
        assert result1.unit == u.nS

        result2 = (init + 1.0 * u.nS)(10, rng=rng)
        assert result2.unit == u.nS

        result3 = init.clip(1.0 * u.nS, 5.0 * u.nS)(10, rng=rng)
        assert result3.unit == u.nS


class TestDocumentationExamples:
    """Test examples from documentation."""

    def test_basic_composition_example(self):
        """Test basic composition example from docs."""
        rng = np.random.default_rng(42)

        init = SimpleInit(1.0) * 2.0 + 0.1
        result = init(10, rng=rng)
        assert np.allclose(result, 2.1)

    def test_pipe_example(self):
        """Test pipe example from docs."""
        rng = np.random.default_rng(42)

        init = (SimpleInit(1.0)
                | (lambda x: x * 2)
                | (lambda x: x + 1))
        result = init(10, rng=rng)
        assert np.allclose(result, 3.0)

    def test_compose_example(self):
        """Test Compose example from docs."""
        rng = np.random.default_rng(42)

        init = Compose(
            SimpleInit(1.0),
            lambda x: x * 10
        )
        result = init(10, rng=rng)
        assert np.allclose(result, 10.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
