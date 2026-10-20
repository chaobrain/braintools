``braintools.fit`` module
=========================

.. currentmodule:: braintools.fit
.. automodule:: braintools.fit

.. seealso::

   For a runnable walkthrough, see the tutorial :doc:`../fit/01_fitting_models`.


Objectives
----------

A :class:`Term` wraps one loss ``fn(prediction, target)`` with a weight and an
optional key that selects one output of a dict-valued prediction. Terms compose
with ``+`` and scalar ``*`` into an :class:`Objective`, whose
:meth:`Objective.breakdown` reports each term separately.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Term
   Objective


Fitting
-------

:class:`Fitter` fits either a pure ``simulate(params)`` function over a search
space, or a ``brainstate`` module through :meth:`Fitter.from_module` (its
``brainstate.nn.Param(..., fit=True)`` parameters and their bounding
transforms). Backends: ``'grad'`` (all starts minimized at once by
:func:`braintools.optim.multistart_minimize`), ``'scipy'`` and ``'nevergrad'``.
In module mode the best parameters are written back to the model only when
:meth:`Fitter.fit` finishes.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Fitter
   FitResult


Parameter Sweeps and Uncertainty
--------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   sweep
   hessian_ci

.. autosummary::
   :toctree: generated/
   :nosignatures:

   HessianCI


Losses
------

.. currentmodule:: braintools.fit.losses
.. automodule:: braintools.fit.losses

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   mse
   rmse
   mae
   correlation_loss
   fc_correlation_loss
   fc_rmse
   fcd_ks
   fcd_wasserstein
   ks_distance
   wasserstein_1d
   psd_log_distance
   rate_loss
   cv_loss
   van_rossum_loss
