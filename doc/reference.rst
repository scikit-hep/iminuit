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
    Minuit.draw_mnprofile


Minuit
------

.. autoclass:: Minuit
    :members:

Scipy-like interface
--------------------

.. autofunction:: minimize

Cost functions
--------------

.. automodule:: iminuit.cost
    :members:

Utilities
---------

.. automodule:: iminuit.util
    :members:
