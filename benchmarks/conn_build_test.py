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

"""Smoke test of ``benchmarks/conn_build.py``: run ``--quick`` and check the output format."""

import importlib.util
import math
import os
import subprocess
import sys
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conn_build.py')


def _load_module():
    spec = importlib.util.spec_from_file_location('conn_build', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestConnBuildBenchmark(unittest.TestCase):

    def _check_line(self, fields, case, n):
        self.assertEqual(fields['case'], case)
        self.assertEqual(int(fields['n']), n)
        status = fields['status']
        self.assertTrue(status == 'ok' or status.startswith('skipped:'), status)
        rss = float(fields['peak_rss_mb'])
        self.assertTrue(0 < rss < 4096, rss)
        if status == 'ok':
            self.assertGreater(int(fields['edges']), 0)
            self.assertGreaterEqual(float(fields['time_s']), 0.0)
        else:
            self.assertEqual(int(fields['edges']), -1)
            self.assertTrue(math.isnan(float(fields['time_s'])))

    def test_quick_run_output_format(self):
        env = dict(os.environ, JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false')
        proc = subprocess.run([sys.executable, SCRIPT, '--quick'], capture_output=True, text=True,
                              env=env, timeout=900)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        lines = proc.stdout.splitlines()
        results = [ln for ln in lines if ln.startswith('RESULT ')]
        self.assertEqual(len(results), 4)
        module = _load_module()
        parsed = [module.parse_result(ln) for ln in results]
        for fields, case in zip(parsed, 'abcd'):
            self._check_line(fields, case, 10_000)
        # Cases a and c only need classes that exist in every version.
        self.assertEqual(parsed[0]['status'], 'ok')
        self.assertEqual(parsed[2]['status'], 'ok')
        # FixedProb(0.001) on 1e4 x 1e4 neurons: about 1e5 edges.
        self.assertLess(abs(int(parsed[0]['edges']) - 99_990), 5 * 316)
        # The Markdown summary table follows the result lines.
        header = '| case | connectivity | N | edges | time (s) | peak RSS (MB) | status |'
        self.assertIn(header, lines)
        for case in 'abcd':
            self.assertEqual(sum(ln.startswith(f'| {case} |') for ln in lines), 1)

    def test_inline_single_case_and_parser(self):
        module = _load_module()
        line = module.run_case('a', 500)
        fields = module.parse_result(line)
        self._check_line(fields, 'a', 500)
        self.assertEqual(module.parse_result('RESULT case=x n=3 status=ok'),
                         {'case': 'x', 'n': '3', 'status': 'ok'})


if __name__ == '__main__':
    unittest.main()
