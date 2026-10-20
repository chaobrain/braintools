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

"""Helpers that keep autosummary stub file names unique on any filesystem."""

import inspect
import pathlib
import re

_CURRENTMODULE = re.compile(r'^\.\. currentmodule::\s*(\S+)\s*$', re.MULTILINE)


def documented_modules(apis_dir):
    """Return the modules named by ``.. currentmodule::`` in ``apis_dir/*.rst``.

    Parameters
    ----------
    apis_dir : str or pathlib.Path
        Directory holding the API reference ``.rst`` files.

    Returns
    -------
    list of str
        Sorted, de-duplicated module names.
    """
    names = set()
    for path in pathlib.Path(apis_dir).glob('*.rst'):
        names.update(_CURRENTMODULE.findall(path.read_text(encoding='utf-8')))
    return sorted(names)


def case_collision_filename_map(modules):
    """Build an ``autosummary_filename_map`` for names differing only in case.

    Functions such as ``braintools.input.step`` and classes such as
    ``braintools.input.Step`` would share one stub file on case-insensitive
    filesystems (Windows, macOS), so one page silently overwrites the other.
    In each colliding group a class keeps the plain file name and the other
    names get a ``-function`` (or ``-class`` / ``-object``) suffix.

    Parameters
    ----------
    modules : iterable of module
        Modules whose public names are documented with autosummary.

    Returns
    -------
    dict of str to str
        Maps fully qualified names to their stub file names.
    """
    mapping = {}
    for module in modules:
        groups = {}
        for name in sorted(n for n in dir(module) if not n.startswith('_')):
            # Submodules are not documented through autosummary stubs.
            if not inspect.ismodule(getattr(module, name)):
                groups.setdefault(name.lower(), []).append(name)
        for group in groups.values():
            if len(group) < 2:
                continue
            # Classes first, so a class keeps the plain (stable) file name.
            group.sort(key=lambda n: (not inspect.isclass(getattr(module, n)), n))
            taken = {f'{module.__name__}.{group[0]}'.lower()}
            for name in group[1:]:
                full = f'{module.__name__}.{name}'
                obj = getattr(module, name)
                kind = 'class' if inspect.isclass(obj) else 'function' if callable(obj) else 'object'
                candidate, counter = f'{full}-{kind}', 1
                while candidate.lower() in taken:
                    counter += 1
                    candidate = f'{full}-{kind}{counter}'
                taken.add(candidate.lower())
                mapping[full] = candidate
    return mapping
