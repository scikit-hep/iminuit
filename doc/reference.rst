.. include:: bibliography.txt

.. _api:

Reference
=========

.. currentmodule:: iminuit

Quick Summary
-------------
These methods and properties you will probably use a lot:

.. autosummary::
    Minuit
    Minuit.migrad
    Minuit.hesse
    Minuit.minos
    Minuit.values
    Minuit.errors
    Minuit.merrors
    Minuit.fixed
    Minuit.limits
    Minuit.covariance
    Minuit.valid
    Minuit.accurate
    Minuit.fval
    Minuit.nfit
    Minuit.mnprofile
    Minuit.mncontour
    Minuit.visualize
    Minuit.draw_mnmatrix


Main interface
--------------

.. autoclass:: Minuit
    :members:
    :undoc-members:

Scipy-like interface
--------------------

.. autofunction:: minimize

Cost functions
--------------

.. automodule:: iminuit.cost
    :members:
    :inherited-members:

Utilities
---------

.. automodule:: iminuit.util
    :exclude-members: Matrix
    :inherited-members:

.. autoclass:: iminuit.util.Matrix
    :members:
    :no-inherited-members:
