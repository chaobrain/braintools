``braintools.quad`` module
============================

.. currentmodule:: braintools.quad 
.. automodule:: braintools.quad 

Time-stepping schemes for deterministic, stochastic, and implicit-explicit
equations used in neural simulations.

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


