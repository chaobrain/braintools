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

import importlib
import pathlib
import types

from autosummary_names import case_collision_filename_map, documented_modules

APIS_DIR = pathlib.Path(__file__).resolve().parents[1] / 'apis'


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _stub_names(module, mapping):
    names = [n for n in dir(module)
             if not n.startswith('_') and not isinstance(getattr(module, n), types.ModuleType)]
    return [mapping.get(f'{module.__name__}.{n}', f'{module.__name__}.{n}') for n in names]


def test_function_renamed_class_kept():
    class Step:
        pass

    def step():
        pass

    mapping = case_collision_filename_map([_module('pkg', Step=Step, step=step)])
    assert mapping == {'pkg.step': 'pkg.step-function'}


def test_no_collision_gives_empty_map():
    mapping = case_collision_filename_map([_module('pkg', Step=object, ramp=len)])
    assert mapping == {}


def test_three_way_collision_stays_unique():
    class ERF:
        pass

    class Erf:
        pass

    def erf():
        pass

    module = _module('pkg', ERF=ERF, Erf=Erf, erf=erf)
    mapping = case_collision_filename_map([module])
    stubs = _stub_names(module, mapping)
    assert len({s.lower() for s in stubs}) == 3
    # Exactly one name keeps its plain stub file, so its URL stays stable.
    assert len(mapping) == 2


def test_submodules_ignored():
    class Context:
        pass

    module = _module('pkg', Context=Context, context=_module('pkg.context'))
    assert case_collision_filename_map([module]) == {}


def test_documented_modules_parses_currentmodule(tmp_path):
    (tmp_path / 'a.rst').write_text('.. currentmodule:: pkg.a \n')
    (tmp_path / 'b.rst').write_text('Title\n=====\n\n.. currentmodule:: pkg.b\n.. currentmodule:: pkg.a\n')
    assert documented_modules(tmp_path) == ['pkg.a', 'pkg.b']


def test_documented_api_stub_names_are_case_insensitively_unique():
    # Guards against pages overwriting each other on Windows / macOS.
    names = documented_modules(APIS_DIR)
    assert 'braintools.input' in names and 'braintools.surrogate' in names
    modules = [importlib.import_module(n) for n in names]
    mapping = case_collision_filename_map(modules)
    for module in modules:
        stubs = _stub_names(module, mapping)
        assert len({s.lower() for s in stubs}) == len(stubs), module.__name__
    assert mapping['braintools.input.step'] == 'braintools.input.step-function'
    assert 'braintools.input.Step' not in mapping
