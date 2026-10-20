``braintools.quad`` module
============================

.. currentmodule:: braintools.quad 
.. automodule:: braintools.quad 

Time-stepping schemes for deterministic, stochastic, and implicit-explicit
equations used in neural simulations.

Trajectories and Method Registry
--------------------------------

:func:`odeint` drives any registered ODE stepper over a fixed time grid as one
compiled ``scan`` (``checkpoint=True`` bounds memory for backpropagation through
time). :func:`solve_ivp` integrates with an adaptive step size, using the
embedded error estimate of ``dopri5`` / ``rk45`` / ``rkf45`` / ``rk23`` /
``bs32`` / ``dopri8`` and a PI step-size controller; its bounded loop is
reverse-mode differentiable. Steppers are looked up by short name through the
registry (``'euler'``, ``'rk4'``, ``'dopri5'``, ``'milstein'``, ...), which also
accepts user-defined methods.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    odeint
    solve_ivp
    register_method
    get_method
    list_methods

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ODESolution


ODE Numerical Integrators
-------------------------

Explicit integrators for deterministic dynamics, ranging from Euler to
higher-order Runge-Kutta variants.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    ode_euler_step
    ode_rk2_step
    ode_rk3_step
    ode_rk4_step
    ode_expeuler_step
    ode_midpoint_step
    ode_heun_step
    ode_rk4_38_step
    ode_rk45_step
    ode_rk23_step
    ode_dopri5_step
    ode_rk45_dopri_step
    ode_rkf45_step
    ode_ssprk33_step
    ode_dopri8_step
    ode_rk87_dopri_step
    ode_bs32_step
    ode_ralston2_step
    ode_ralston3_step



IMEX Numerical Integrators
--------------------------

Hybrid implicit-explicit solvers suited for stiff systems that mix fast and
slow processes.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    imex_euler_step
    imex_ars222_step
    imex_cnab_step


SDE Numerical Integrators
-------------------------

Stochastic integrators that support noise-driven dynamics and diffusion
processes.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    sde_euler_step
    sde_milstein_step
    sde_expeuler_step
    sde_heun_step
    sde_tamed_euler_step
    sde_implicit_euler_step
    sde_srk2_step
    sde_srk3_step
    sde_srk4_step


DDE Numerical Integrators
-------------------------

Steppers for delay differential equations, where the right-hand side depends on
one or more delayed states ``y(t - τ)`` supplied by a user history function.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    dde_euler_step
    dde_heun_step
    dde_rk4_step
    dde_euler_pc_step
    dde_heun_pc_step


