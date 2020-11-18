.. include:: references.txt

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

Cost functions
--------------

.. automodule:: iminuit.cost
    :members:

.. currentmodule:: iminuit

minimize
--------

The :func:`minimize` function provides the same interface as :func:`scipy.optimize.minimize`.
If you are familiar with the latter, this allows you to use Minuit with a quick start.
Eventually, you still may want to learn the interface of the :class:`Minuit` class,
as it provides more functionality if you are interested in parameter uncertainties.

.. autofunction:: minimize

Utility Functions
-----------------

.. currentmodule:: util

The module :mod:`util` provides the :func:`describe` function and various function to manipulate
fit arguments. Most of these functions (apart from describe) are for internal use. You should not rely
on them in your code. We list the ones that are for the public.

.. automodule:: iminuit.util
    :members:


.. _function-sig-label:

Function Signature Extraction Ordering
--------------------------------------

    1. Using ``f.func_code.co_varnames``, ``f.func_code.co_argcount``
       All functions that are defined like::

        def f(x, y):
            return (x - 2) ** 2 + (y - 3) ** 2

       or::

        f = lambda x, y: (x - 2) ** 2 + (y - 3) ** 2

       Have these two attributes.

    2. Using ``f.__call__.func_code.co_varnames``, ``f.__call__.co_argcount``.
       Minuit knows how to skip the ``self`` parameter. This allow you to do
       things like encapsulate your data with in a fitting algorithm::

        class MyLeastSquares:
            def __init__(self, data_x, data_y, data_yerr):
                self.x = data_x
                self.y = data_y
                self.ye = data_yerr

            def __call__(self, a, b):
                result = 0.0
                for x, y, ye in zip(self.x, self.y, self.ye):
                    y_predicted = a * x + b
                    residual = (y - y_predicted) / ye
                    result += residual ** 2
                return result

    3. If all fails, Minuit will try to read the function signature from the
       docstring to get function signature.


    This order is very similar to PyMinuit signature detection. Actually,
    it is a superset of PyMinuit signature detection.
    The difference is that it allows you to fake function
    signature by having a ``func_code`` attribute in the object. This allows you
    to make a generic functor of your custom cost function. This is explained
    in the **Advanced Tutorial** in the docs.


    .. note::

        If you are unsure what iminuit will parse your function signature, you can use
        :func:`describe` to check which argument names are detected.
