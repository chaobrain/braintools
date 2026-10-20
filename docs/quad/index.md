# Numerical Integration

`braintools.quad` provides one-step integrators for ordinary, stochastic, implicit–explicit, and delay differential equations. Each call advances a JAX PyTree state (optionally with `brainunit` units) by one step of size `dt` taken from `brainstate.environ`; you write the loop, usually with a scan. For whole trajectories, `quad.odeint` runs any registered stepper over a fixed time grid as one compiled `scan`, and `quad.solve_ivp` integrates with an adaptive step size (both are used in the {doc}`model-fitting tutorial <../fit/01_fitting_models>`).

## What you'll find here

- **ODE integration**: the calling convention, measured convergence orders of all fixed-step methods, embedded error estimates driving an adaptive loop, stiffness and exponential Euler, units and PyTree states, and a Hodgkin–Huxley neuron compared across methods and step sizes.
- **SDE integration**: random numbers and brainstate transformations, Itô versus Stratonovich, strong convergence measured on shared Brownian paths, scheme-dependent stationary statistics, tamed and implicit Euler for superlinear drifts, and a population of noisy integrate-and-fire neurons.
- **IMEX and delay equations**: implicit–explicit splitting with Newton-solved implicit stages, L-stability, a FitzHugh–Nagumo cable with implicit diffusion, history interpolation for delay equations, and oscillations caused by delayed inhibition.
- **Trajectories and adaptive integration**: `odeint` over fixed-step methods and their measured orders, `solve_ivp` with adaptive Dormand–Prince steps on a Van der Pol oscillator checked against `scipy.integrate.solve_ivp`, output at `t_eval` times, embedded pairs of different orders, gradients and parameter sensitivities through both drivers with `checkpoint=True`, and the method registry (`list_methods`, `get_method`, `register_method` with a custom Runge–Kutta stepper).

For the full list of steppers and parameters, see the {doc}`API reference <../apis/quad>`.

```{toctree}
:maxdepth: 1

ode_integration.ipynb
sde_integration.ipynb
imex_and_dde_integration.ipynb
trajectories_and_adaptive.ipynb
```
