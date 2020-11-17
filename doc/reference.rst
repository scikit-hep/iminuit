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

Data objects
------------

.. currentmodule:: iminuit

iminuit uses various data objects as return values. This section lists them.

.. _function-minimum-sruct:

Function Minimum Data Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A data object with extra information about the fit result. It has the attributes:

    * *fval*: Value of the cost function at the minimum.

    * *edm*: Estimated Distance to Minimum.

    * *nfcn*: Number of function call in last Migrad call

    * *up*: Equal to the value of ``errordef`` when Migrad ran.

    * *is_valid*: Whether the function minimum is ok, defined as

        * has_valid_parameters
        * and not has_reached_call_limit
        * and not is_above_max_edm

    * *has_valid_parameters*: Validity of parameters. This means:

        1. The parameters must have valid error(if it's not fixed).
           Valid error is not necessarily accurate.
        2. The parameters value must be valid

    * *has_accurate_covariance*: Whether covariance matrix is accurate.

    * *has_pos_def_covar*: Positive definiteness of covariance matrix.
      Must be true if the extremum is a minimum.

    * *has_made_posdef_covar*: Whether Migrad has to force covariance matrix
      to be positive definite by adding a diagonal matrix (should not happen!).

    * *hesse_failed*: Whether a call to Hesse after Migrad was successful.

    * *has_covaraince*: Has Covariance.

    * *is_above_max_edm*: Is estimated distance to minimum above its goal?
      This is the convergence criterion of Migrad, if it is violated, Migrad did not
      converge.

    * *has_reached_call_limit*: Whether Migrad exceeded the allowed number of
      function calls.

.. _minos-error-struct:

Minos Data Object
~~~~~~~~~~~~~~~~~

A data object with extra information about the Minos result. It has the attributes:

    * *lower*: lower error value

    * *upper*: upper error value

    * *is_valid*: Validity of minos error value. This means `lower_valid`
      and `upper_valid`

    * *lower_valid*: Validity of lower error

    * *upper_valid*: Validity of upper error

    * *at_lower_limit*: minos calculation hits the lower limit on parameters

    * *at_upper_limit*: minos calculation hits the upper limit on parameters

    * *lower_new_min*: found a new minimum while scanning cost function for
      lower error value

    * *upper_new_min*: found a new minimum while scanning cost function for
      upper error value

    * *nfn*: number of call to FCN in the last minos scan

    * *min*: the value of the parameter at the minimum

.. _minuit-param-struct:

Parameter Data Object
~~~~~~~~~~~~~~~~~~~~~

A data object with information about a fitted parameter. It has the attributes:

    * *number*: parameter number

    * *name*: parameter name

    * *value*: parameter value

    * *error*: parameter parabolic error(like those from hesse)

    * *is_fixed*: is the parameter fixed

    * *is_const*: is the parameter a constant(We do not support const but
      you can alway use fixing parameter instead)

    * *has_limits*: parameter has limits set

    * *has_lower_limit*: parameter has lower limit set. We do not support one
      sided limit though.

    * *has_upper_limit*: parameter has upper limit set.

    * *lower_limit*: value of lower limit for this parameter

    * *upper_limit*: value of upper limit for this parameter


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
