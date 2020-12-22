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

minimize
--------

The :func:`minimize` function provides the same interface as :func:`scipy.optimize.minimize`.
If you are familiar with the latter, this allows you to use Minuit with a quick start.
Eventually, you still may want to learn the interface of the :class:`Minuit` class,
as it provides more functionality if you are interested in parameter uncertainties.

.. autofunction:: minimize

iminuit.cost
------------

.. automodule:: iminuit.cost
    :members:

iminuit.util
------------

.. currentmodule:: iminuit.util

.. automodule:: iminuit.util
    :members:
