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
from braintools.quad._registry import MethodInfo, get_method, list_methods, method_info, register_method


@pytest.fixture
def clean_registry():
    saved = {
        table: copy.deepcopy({k: dict(v) for k, v in getattr(_registry, table).items()})
        for table in ('_REGISTRY', '_META', '_ALIASES')
    }
    yield
    for table, value in saved.items():
        getattr(_registry, table).clear()
        getattr(_registry, table).update(value)


def _dummy(f, y, t, *args, **kwargs):
    return y


class TestBuiltins:
    # 'module' has no built-in steppers: it is a namespace for stateful hosts.
    @pytest.mark.parametrize('kind', [k for k in _registry.KINDS if k != 'module'])
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

    def test_module_kind_starts_empty(self):
        assert 'module' in _registry.KINDS
        assert list_methods('module') == []

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


class TestMetadata:
    def test_builtin_ode_metadata(self):
        info = method_info('rk4')
        assert isinstance(info, MethodInfo)
        assert (info.name, info.kind, info.fn) == ('rk4', 'ode', quad.ode_rk4_step)
        assert (info.order, info.category) == (4, 'explicit')
        assert method_info('dopri5').category == 'embedded'
        assert method_info('expeuler').category == 'exponential'
        assert method_info('ralston4').order == 4

    def test_every_builtin_ode_method_has_an_order(self):
        for name in list_methods('ode'):
            info = method_info(name)
            assert info.order is not None, name
            assert info.category in ('explicit', 'embedded', 'exponential'), name

    def test_documented_orders_of_other_kinds(self):
        assert method_info('euler', 'sde').order == 0.5
        assert method_info('milstein', 'sde').order == 1.0
        assert method_info('ars222', 'imex').order == 2
        assert method_info('rk4', 'dde').order == 4
        assert method_info('srk2', 'sde').order is None

    def test_aliases(self):
        assert get_method('heun3') is quad.ode_heun_step
        assert get_method('ssprk3') is quad.ode_ssprk33_step
        assert get_method('heun2') is quad.ode_rk2_step
        assert method_info('ssprk3').name == 'ssprk33'
        assert method_info('heun').aliases == ('heun3',)
        assert method_info('rk2').aliases == ('heun2',)
        # aliases are not listed as names
        assert 'heun3' not in list_methods('ode')
        # no category name is used as an alias
        with pytest.raises(KeyError):
            get_method('explicit')

    def test_full_name_of_new_stepper(self):
        assert get_method('ode_ralston4_step') is quad.ode_ralston4_step
        assert method_info('ode_ralston4_step').name == 'ralston4'

    def test_info_is_a_frozen_snapshot(self, clean_registry):
        register_method('snap', _dummy, order=1)
        info = method_info('snap')
        register_method('snap', _dummy, order=2)
        assert info.order == 1 and method_info('snap').order == 2
        with pytest.raises(AttributeError):
            info.order = 3

    def test_register_with_metadata(self, clean_registry):
        register_method('my_rk', _dummy, order=2, category='explicit',
                        aliases=('my_rk_alias',), description='A test method.')
        info = method_info('my_rk_alias')
        assert info == MethodInfo('my_rk', 'ode', _dummy, 2, 'explicit', ('my_rk_alias',), 'A test method.')
        assert get_method('my_rk_alias') is _dummy

    def test_plain_reregistration_keeps_metadata(self, clean_registry):
        register_method('rk4', quad.ode_rk4_step)
        assert method_info('rk4').order == 4

    def test_reregistration_replaces_aliases(self, clean_registry):
        register_method('m', _dummy, aliases=('a1',))
        register_method('m', _dummy, aliases=('a2',))
        assert method_info('m').aliases == ('a2',)
        with pytest.raises(KeyError):
            get_method('a1')

    def test_module_kind(self, clean_registry):
        def step(target, *args):
            return None

        register_method('m', step, kind='module', order=1, aliases=('m2',))
        assert get_method('m2', 'module') is step
        assert method_info('m', 'module').kind == 'module'
        assert list_methods('module') == ['m']
        assert list_methods('imex') == ['ars222', 'cnab', 'euler']
        with pytest.raises(KeyError):
            get_method('m')  # not an 'ode' method

    def test_module_kind_is_not_accepted_by_odeint(self, clean_registry):
        register_method('only_module', _dummy, kind='module')
        with pytest.raises(KeyError):
            quad.odeint(lambda y, t: -y, 1.0, 0.0, 1.0, 0.1, method='only_module')


class TestMetadataErrors:
    def test_alias_clashing_with_a_name(self, clean_registry):
        with pytest.raises(ValueError, match='already the name'):
            register_method('x', _dummy, aliases=('rk4',))
        with pytest.raises(ValueError, match='already the name'):
            register_method('x', _dummy, aliases=('rk4',), overwrite=True)

    def test_alias_clashing_with_an_alias(self, clean_registry):
        with pytest.raises(ValueError, match="refers to the 'ode' method 'heun'"):
            register_method('x', _dummy, aliases=('heun3',))
        register_method('x', _dummy, aliases=('heun3',), overwrite=True)
        assert get_method('heun3') is _dummy
        assert method_info('heun').aliases == ()

    def test_name_clashing_with_an_alias(self, clean_registry):
        with pytest.raises(ValueError, match="alias of the 'ode' method 'rk2'"):
            register_method('heun2', _dummy)
        register_method('heun2', _dummy, overwrite=True)
        assert get_method('heun2') is _dummy
        assert method_info('rk2').aliases == ()

    def test_same_alias_in_another_kind_is_fine(self, clean_registry):
        register_method('x', _dummy, kind='sde', aliases=('heun3',))
        assert get_method('heun3', 'sde') is _dummy
        assert get_method('heun3') is quad.ode_heun_step

    @pytest.mark.parametrize('kwargs, error', [
        (dict(order=0), TypeError),
        (dict(order=-1), TypeError),
        (dict(order=True), TypeError),
        (dict(order='2'), TypeError),
        (dict(category=2), TypeError),
        (dict(description=None), TypeError),
        (dict(aliases=('',)), TypeError),
        (dict(aliases=(3,)), TypeError),
        (dict(aliases=('bad', 'bad')), ValueError),
        (dict(aliases=('bad',)), ValueError),
    ])
    def test_bad_metadata(self, kwargs, error, clean_registry):
        with pytest.raises(error):
            register_method('bad', _dummy, **kwargs)
        with pytest.raises(KeyError):
            get_method('bad')

    def test_string_alias_is_one_alias(self, clean_registry):
        register_method('s', _dummy, aliases='s_alias')
        assert method_info('s').aliases == ('s_alias',)

    def test_unknown_name(self):
        with pytest.raises(KeyError, match='Did you mean'):
            method_info('ralston5')
        with pytest.raises(ValueError, match='Unknown integrator kind'):
            method_info('rk4', 'modul')
        with pytest.raises(TypeError):
            method_info(None)

    def test_close_matches_include_aliases(self):
        with pytest.raises(KeyError) as info:
            get_method('heun33')
        assert "'heun3'" in str(info.value)
