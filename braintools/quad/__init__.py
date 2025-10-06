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

"""
Lightweight one-step integrators for ODEs, SDEs, and DDEs.

This submodule provides compact, JAX-friendly stepping functions for ordinary
differential equations (ODEs), stochastic differential equations (SDEs), and
delay differential equations (DDEs) that operate directly on JAX PyTrees. All
steppers use the global time step ``dt`` from ``brainstate.environ`` so they can
be dropped into simulation loops with minimal boilerplate.

Available
---------
- ODE: ``ode_euler_step``, ``ode_rk2_step``, ``ode_rk3_step``, ``ode_rk4_step``,
  ``ode_expeuler_step``.
- SDE: ``sde_euler_step`` (Euler–Maruyama), ``sde_milstein_step``,
  ``sde_expeuler_step``.
- IMEX: ``imex_euler_step``, ``imex_ars222_step``, ``imex_cnab_step``.
- DDE: ``dde_euler_step``, ``dde_heun_step``, ``dde_rk4_step``,
  ``dde_euler_pc_step``, ``dde_heun_pc_step``.

Notes
-----
- Steppers accept arbitrary PyTrees as state and return an updated PyTree with
  the same structure.
- Where applicable, units are handled via ``brainunit``; pass times and states
  as quantities to keep dimensionality consistent.
- DDE steppers require a history function to evaluate delayed terms.
"""

from ._dde_integrator import *
from ._dde_integrator import __all__ as dde_all
from ._imex_integrator import *
from ._imex_integrator import __all__ as imex_all
from ._ode_integrator import *
from ._ode_integrator import __all__ as ode_all
from ._sde_integrator import *
from ._sde_integrator import __all__ as sde_all

__all__ = ode_all + sde_all + imex_all + dde_all
