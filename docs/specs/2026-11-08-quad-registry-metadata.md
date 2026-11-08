# quad registry: metadata, a `module` kind, `ralston4` and aliases (QUAD-2a)

## Goal

braincell has its own integrator registry with aliases, categories and orders, and
its `rk2` is a different method from braintools' `rk2`. The goal is to keep one registry, the one in
`braintools.quad`, which braincell then uses. braintools needs three things for that:
per-method metadata, somewhere to store stateful module steps, and the methods and
names braincell relies on but braintools lacks.

## Decisions

- **Metadata is additive.** `register_method(name, fn, kind='ode', *, overwrite=False,
  order=None, category=None, aliases=(), description='')`. New
  `method_info(name, kind='ode') -> MethodInfo`, where `MethodInfo` is a frozen dataclass with fields `name, kind,
  fn, order, category, aliases, description`. `get_method` and `method_info` resolve
  aliases and full function names. `list_methods` still lists canonical names only.
  Existing names, signatures and behaviour do not change. Reason: brainmass and other
  packages look the frozen `ode_*`/`sde_*`/`imex_*`/`dde_*` names up by string.
- **Alias conflicts raise.** An alias that clashes with a name or alias of the same
  kind raises `ValueError` unless `overwrite=True`. Re-registering the same function
  object stays a no-op.
- **A `module` kind.** It holds steps of the form `step(target, *args) -> None` that read `t`/`dt` from
  `brainstate.environ` and advance a stateful module in place. braintools registers
  nothing there, and `odeint`/`solve_ivp` do not accept it. It is only a namespace with
  metadata for stateful hosts such as braincell, so the core stays functional.
- **`ode_ralston4_step`** uses Ralston's (1962) exact coefficients, the closed forms in √5.
- **Aliases:** `heun3 -> heun`, `ssprk3 -> ssprk33`, `heun2 -> rk2`. There is no `explicit` alias, because
  `explicit` is a category name.
- **Orders and categories.** Every built-in ODE method gets an `order` and a `category`
  (`explicit`, `embedded` or `exponential`). An embedded pair's order is that of the solution it
  returns. For SDE, IMEX and DDE methods, the order is filled in only where the docstring states one.

## Steps

1. Registry: metadata, aliases, `method_info`/`MethodInfo`, `module` kind, tests.
2. `ode_ralston4_step`: implementation, registry entry, order-condition and convergence tests.
3. Docs (`docs/apis/quad.rst`, the trajectories notebook output), changelog, version 0.5.0.
4. `downstream.yml`: add braincell (git main), running `braincell.quad` and
   `braincell._single_compartment`.

## Verification

`pytest braintools/quad braintools/__init___test.py`, `mypy`, `python scripts/run_doctests.py`
and `sphinx-build -W` all pass. The new tests cover metadata lookups, aliases, the
`module` kind, conflict errors, and `ralston4`: its order conditions hold to ≤ 1e-14, and its
local error shrinks by about 32× when dt is halved.
