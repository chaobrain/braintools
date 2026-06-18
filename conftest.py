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

"""Project-wide pytest configuration.

Force a non-interactive ("Agg") matplotlib backend for the whole test suite so
that running the tests never opens GUI image windows and works in headless / CI
environments. ``conftest.py`` is imported by pytest before any test module, so
this takes effect even for modules that import ``matplotlib.pyplot`` lazily.

We set ``MPLBACKEND`` *before* importing matplotlib: that environment variable is
read at matplotlib import time and cannot be silently overridden by an
interactive default, making it the most robust way to guarantee a headless
backend regardless of which test module imports matplotlib first. The explicit
``matplotlib.use("Agg")`` call is belt-and-suspenders for an already-imported
matplotlib.
"""

import os

os.environ["MPLBACKEND"] = "Agg"

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except ImportError:  # matplotlib is optional; nothing to configure without it.
    pass
