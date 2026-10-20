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

"""Time and memory benchmark for building large ``braintools.conn`` networks.

Each case is built in its own Python process, so the reported peak resident
set size (``ru_maxrss``, the number ``/usr/bin/time -f %M`` prints) belongs to
that case alone. Run from the repository root::

    python benchmarks/conn_build.py            # full sizes (N = 1e5)
    python benchmarks/conn_build.py --quick    # N = 1e4 smoke run
    python benchmarks/conn_build.py --case c --n 30000

Cases
-----
a  ``FixedProb(0.001)`` on N neurons (about 1e-3 N^2 edges).
b  ``FixedInDegree(100)`` on N neurons, exported to ``brainevent.FixedNumPerPost``.
c  N neurons uniformly placed on a 2-D sheet (10 um mean spacing), ``Gaussian``
   profile with sigma = 40 um and ``max_distance`` = 120 um (about 100 edges
   per neuron).
d  E/I ``Network`` of N / 10 neurons (80 % E, 20 % I) with ``FixedInDegree``
   projections (80 from E, 20 from I), exported to one block CSR.

Cases b and d need ``braintools.conn.FixedInDegree`` / ``Network``; they are
reported as ``skipped`` when those names are not available.

For every case the script prints one line::

    RESULT case=<c> n=<N> edges=<E> time_s=<t> peak_rss_mb=<m> status=<ok|skipped:...>

followed (unless ``--inline``) by a Markdown table of all cases. The time
covers building the connectivity (and the export, where the case has one),
not the interpreter start-up or the imports.
"""

import argparse
import os
import resource
import subprocess
import sys
import time

# Benchmark the checkout this script lives in, not an installed braintools.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CASES = ('a', 'b', 'c', 'd')
DESCRIPTIONS = {
    'a': 'FixedProb(0.001)',
    'b': 'FixedInDegree(100) -> FixedNumPerPost',
    'c': '2-D sheet Gaussian(sigma=40um, max_distance=120um)',
    'd': 'E/I Network (N/10 neurons, FixedInDegree) -> block CSR',
}


class Skip(Exception):
    """Raised when a case cannot run with the installed braintools."""


def _peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB, the unit of ``/usr/bin/time -f %M``.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _setup_a(n):
    from braintools.conn import FixedProb

    def build():
        return FixedProb(0.001, seed=0)(pre_size=n, post_size=n).n_connections

    return build


def _setup_b(n):
    import braintools.conn as bc
    if not hasattr(bc, 'FixedInDegree'):
        raise Skip('braintools.conn.FixedInDegree not available')

    def build():
        result = bc.FixedInDegree(100, seed=0)(pre_size=n, post_size=n)
        if not hasattr(result, 'to_fixed_num_per_post'):
            raise Skip('ConnectionResult.to_fixed_num_per_post not available')
        result.to_fixed_num_per_post()
        return result.n_connections

    return build


def _setup_c(n):
    import brainunit as u
    import numpy as np
    from braintools.conn import Gaussian
    from braintools.init import GaussianProfile

    side = float(np.sqrt(n)) * 10.0
    positions = np.random.RandomState(0).uniform(0.0, side, (n, 2)) * u.um
    profile = GaussianProfile(sigma=40.0 * u.um, max_distance=120.0 * u.um)
    # Warm-up outside the timed region: the first unit-aware math call triggers
    # one-off lazy imports (array-backend detection) that are not build cost.
    profile.probability(np.linspace(0.0, 200.0, 8) * u.um)

    def build():
        conn = Gaussian(profile, seed=0)
        result = conn(pre_size=n, post_size=n, pre_positions=positions, post_positions=positions)
        return result.n_connections

    return build


def _setup_d(n):
    import braintools.conn as bc
    for name in ('FixedInDegree', 'Population', 'Projection', 'Network'):
        if not hasattr(bc, name):
            raise Skip(f'braintools.conn.{name} not available')
    n_total = max(n // 10, 10)
    n_e = int(0.8 * n_total)
    n_i = n_total - n_e

    def build():
        e = bc.Population(n_e, name='E')
        i = bc.Population(n_i, name='I')
        projections = [
            bc.Projection(e, e, bc.FixedInDegree(80, seed=1)),
            bc.Projection(e, i, bc.FixedInDegree(80, seed=2)),
            bc.Projection(i, e, bc.FixedInDegree(20, seed=3)),
            bc.Projection(i, i, bc.FixedInDegree(20, seed=4)),
        ]
        net = bc.Network([e, i], projections)
        built = net.build()
        net.to_csr()
        return sum(r.n_connections for r in built.values())

    return build


SETUPS = {'a': _setup_a, 'b': _setup_b, 'c': _setup_c, 'd': _setup_d}


def run_case(case: str, n: int) -> str:
    """Build one case in this process and return its ``RESULT`` line."""
    try:
        build = SETUPS[case](n)
        t0 = time.perf_counter()
        edges = build()
        elapsed = f'{time.perf_counter() - t0:.3f}'
        status = 'ok'
    except Skip as exc:
        edges, elapsed, status = -1, 'nan', 'skipped:' + str(exc).replace(' ', '_')
    return (f'RESULT case={case} n={n} edges={edges} time_s={elapsed} '
            f'peak_rss_mb={_peak_rss_mb():.1f} status={status}')


def parse_result(line: str) -> dict:
    """Parse a ``RESULT`` line into a ``{field: str}`` dict."""
    return dict(tok.split('=', 1) for tok in line.split()[1:])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Benchmark braintools.conn network construction.')
    parser.add_argument('--quick', action='store_true', help='use N = 1e4 (smoke test)')
    parser.add_argument('--n', type=int, default=None, help='override the population size N')
    parser.add_argument('--case', choices=CASES, action='append',
                        help='case to run (repeatable; default: all)')
    parser.add_argument('--inline', action='store_true',
                        help='run in this process instead of one subprocess per case')
    args = parser.parse_args(argv)

    n = args.n if args.n is not None else (10_000 if args.quick else 100_000)
    cases = args.case or list(CASES)

    lines = []
    for case in cases:
        if args.inline:
            line = run_case(case, n)
        else:
            cmd = [sys.executable, os.path.abspath(__file__), '--inline', '--n', str(n), '--case', case]
            env = dict(os.environ)
            env.setdefault('JAX_PLATFORMS', 'cpu')
            env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            found = [ln for ln in proc.stdout.splitlines() if ln.startswith('RESULT ')]
            if proc.returncode != 0 or not found:
                sys.stderr.write(proc.stderr)
                line = (f'RESULT case={case} n={n} edges=-1 time_s=nan peak_rss_mb=nan '
                        f'status=error:exit_{proc.returncode}')
            else:
                line = found[-1]
        print(line, flush=True)
        lines.append(line)

    if not args.inline:
        print()
        print('| case | connectivity | N | edges | time (s) | peak RSS (MB) | status |')
        print('|---|---|---|---|---|---|---|')
        for line in lines:
            f = parse_result(line)
            print(f"| {f['case']} | {DESCRIPTIONS[f['case']]} | {f['n']} | {f['edges']} | "
                  f"{f['time_s']} | {f['peak_rss_mb']} | {f['status']} |")
    return 1 if any('status=error' in ln for ln in lines) else 0


if __name__ == '__main__':
    sys.exit(main())
