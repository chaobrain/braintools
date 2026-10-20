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

import copy

import pytest

import braintools.quad as quad
from braintools.quad import _registry
from braintools.quad._registry import get_method, list_methods, register_method


@pytest.fixture
def clean_registry():
    saved = copy.deepcopy({k: dict(v) for k, v in _registry._REGISTRY.items()})
    yield
    _registry._REGISTRY.clear()
    _registry._REGISTRY.update(saved)


def _dummy(f, y, t, *args, **kwargs):
    return y


class TestBuiltins:
    @pytest.mark.parametrize('kind', _registry.KINDS)
    def test_every_public_stepper_is_registered(self, kind):
        public = [n for n in quad.__all__ if n.startswith(f'{kind}_') and n.endswith('_step')]
        assert public, kind
        for full in public:
            short = full[len(kind) + 1:-len('_step')]
            fn = getattr(quad, full)
            assert get_method(short, kind) is fn
            assert get_method(full, kind) is fn
            assert short in list_methods(kind)

    def test_short_names(self):
        assert get_method('euler') is quad.ode_euler_step
        assert get_method('rk4') is quad.ode_rk4_step
        assert get_method('dopri5') is quad.ode_dopri5_step
        assert get_method('milstein', 'sde') is quad.sde_milstein_step
        assert get_method('euler', 'sde') is quad.sde_euler_step
        assert get_method('ars222', 'imex') is quad.imex_ars222_step
        assert get_method('heun_pc', 'dde') is quad.dde_heun_pc_step

    def test_list_methods(self):
        everything = list_methods()
        assert set(everything) == set(_registry.KINDS)
        for kind, names in everything.items():
            assert names == sorted(names)
            assert names == list_methods(kind)
        assert list_methods('imex') == ['ars222', 'cnab', 'euler']
        # the returned list is a copy
        list_methods('ode').append('zzz')
        assert 'zzz' not in list_methods('ode')


class TestLookupErrors:
    def test_unknown_name_lists_close_matches(self):
        with pytest.raises(KeyError) as info:
            get_method('rk5')
        msg = str(info.value)
        assert 'Did you mean' in msg and "'rk4'" in msg and 'Available methods' in msg

    def test_unknown_name_without_close_match(self):
        with pytest.raises(KeyError) as info:
            get_method('qqqqqqqq', 'imex')
        assert 'Did you mean' not in str(info.value)

    def test_name_of_other_kind(self):
        with pytest.raises(KeyError):
            get_method('milstein', 'ode')

    def test_bad_prefix_not_normalized(self):
        with pytest.raises(KeyError):
            get_method('sde_milstein_step', 'ode')
        with pytest.raises(KeyError):
            get_method('ode__step', 'ode')

    @pytest.mark.parametrize('call', [
        lambda: get_method('rk4', 'pde'),
        lambda: list_methods('pde'),
        lambda: register_method('x', _dummy, 'pde'),
    ])
    def test_unknown_kind(self, call):
        with pytest.raises(ValueError, match='Unknown integrator kind'):
            call()

    def test_non_string_name(self):
        with pytest.raises(TypeError):
            get_method(4)


class TestRegister:
    def test_register_and_get(self, clean_registry):
        assert register_method('my_step', _dummy) is _dummy
        assert get_method('my_step') is _dummy
        assert 'my_step' in list_methods('ode')
        register_method('my_sde', _dummy, kind='sde')
        assert get_method('my_sde', 'sde') is _dummy
        with pytest.raises(KeyError):
            get_method('my_sde', 'ode')

    def test_duplicate_requires_overwrite(self, clean_registry):
        def other(f, y, t, *args, **kwargs):
            return y

        register_method('dup', _dummy)
        register_method('dup', _dummy)  # same function again: no-op
        with pytest.raises(ValueError, match='overwrite=True'):
            register_method('dup', other)
        with pytest.raises(ValueError, match='already registered'):
            register_method('rk4', other)
        register_method('dup', other, overwrite=True)
        assert get_method('dup') is other

    def test_overwrite_builtin(self, clean_registry):
        register_method('rk4', _dummy, overwrite=True)
        assert get_method('rk4') is _dummy

    def test_restored_after_fixture(self):
        assert get_method('rk4') is quad.ode_rk4_step
        with pytest.raises(KeyError):
            get_method('my_step')

    @pytest.mark.parametrize('name', ['', None, 3])
    def test_bad_name(self, name, clean_registry):
        with pytest.raises(TypeError, match='non-empty string'):
            register_method(name, _dummy)

    def test_not_callable(self, clean_registry):
        with pytest.raises(TypeError, match='callable'):
            register_method('bad', 1.0)
