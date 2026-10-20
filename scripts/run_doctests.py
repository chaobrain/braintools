# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Run the docstring examples of the public ``braintools`` API.

For every public module (``braintools`` and each module reachable through the
``__all__`` lists), the examples of the module docstring and of every exported
name, including methods and properties, are run with ``doctest`` under
``NORMALIZE_WHITESPACE | ELLIPSIS``. Each module runs in its own Python
process, which bounds peak memory and keeps state from leaking between
modules. Run from the repository root::

    python scripts/run_doctests.py                    # every public module
    python scripts/run_doctests.py braintools.conn    # selected modules
    python scripts/run_doctests.py --list             # print the module list

The exit status is non-zero if any example fails or a docstring cannot be
parsed.
"""

import argparse
import doctest
import importlib
import inspect
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Force the non-interactive backend, as ``conftest.py`` does for pytest, so that
# examples calling ``plt.show()`` never open a window or block. Set before any
# import of matplotlib; the module subprocesses inherit it through ``env``.
os.environ['MPLBACKEND'] = 'Agg'


def public_modules():
    """Return the dotted names of all public ``braintools`` modules, sorted."""
    import braintools

    seen = set()
    stack = [braintools]
    while stack:
        mod = stack.pop()
        if mod.__name__ in seen:
            continue
        seen.add(mod.__name__)
        for attr in getattr(mod, '__all__', ()):
            obj = getattr(mod, attr, None)
            if inspect.ismodule(obj) and obj.__name__.startswith('braintools.'):
                stack.append(obj)
    return sorted(seen)


def run_module(name: str) -> int:
    """Run the doctests of one module in this process; return the number of failures."""
    module = importlib.import_module(name)
    finder = doctest.DocTestFinder(recurse=True)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
    objects = [(name, module)]
    for attr in getattr(module, '__all__', ()):
        obj = getattr(module, attr, None)
        if not inspect.ismodule(obj):
            objects.append((f'{name}.{attr}', obj))
    errors = 0
    seen = set()
    for obj_name, obj in objects:
        try:
            tests = finder.find(obj, obj_name, module=False, globs={})
        except ValueError as exc:  # malformed docstring (e.g. inconsistent indentation)
            print(f'ERROR {obj_name}: {exc}')
            errors += 1
            continue
        for test in tests:
            if test.examples and test.name not in seen:
                seen.add(test.name)
                runner.run(test)
    result = runner.summarize(verbose=False)
    print(f'{name}: {result.attempted} examples, {result.failed} failed, {errors} unparseable')
    return result.failed + errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('modules', nargs='*', help='modules to check (default: every public module)')
    parser.add_argument('--list', action='store_true', help='print the public modules and exit')
    parser.add_argument('--inline', action='store_true', help='run in this process (one module only)')
    args = parser.parse_args(argv)

    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    if args.list:
        print('\n'.join(public_modules()))
        return 0
    if args.inline:
        if len(args.modules) != 1:
            parser.error('--inline takes exactly one module')
        return 1 if run_module(args.modules[0]) else 0

    modules = args.modules or public_modules()
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [_REPO_ROOT, env.get('PYTHONPATH')]))
    failed = []
    for name in modules:
        start = time.perf_counter()
        status = subprocess.call([sys.executable, os.path.abspath(__file__), '--inline', name], env=env)
        print(f'  ({time.perf_counter() - start:.0f} s)', flush=True)
        if status:
            failed.append(name)
    if failed:
        print(f'doctest failures in: {", ".join(failed)}')
        return 1
    print(f'all doctests passed in {len(modules)} modules')
    return 0


if __name__ == '__main__':
    sys.exit(main())
