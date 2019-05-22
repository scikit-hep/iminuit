.. include:: references.txt

.. _api:

Reference
=========

.. currentmodule:: iminuit

Quick Summary
-------------
These are the things you will use a lot:

.. autosummary::
    Minuit
    Minuit.from_array_func
    Minuit.migrad
    Minuit.minos
    Minuit.values
    Minuit.args
    Minuit.errors
    Minuit.get_merrors
    Minuit.fval
    Minuit.fitarg
    Minuit.mnprofile
    Minuit.draw_mnprofile
    Minuit.mncontour
    Minuit.draw_mncontour
    minimize
    util.describe

Minuit
------

.. autoclass:: Minuit
    :members:
    :undoc-members:
    :exclude-members: migrad, hesse, minos, args, values, errors,
        merror, fval, edm, fitarg, covariance, gcc, errordef,
        fcn, pedantic, throw_nan, tol

    .. automethod:: migrad

    .. automethod:: hesse

    .. automethod:: minos

    .. autoattribute:: args

    .. autoattribute:: values

    .. autoattribute:: errors

    .. autoattribute:: fitarg

    .. autoattribute:: merrors

    .. autoattribute:: fval

    .. autoattribute:: edm

    .. autoattribute:: covariance

    .. autoattribute:: gcc

    .. autoattribute:: errordef

    .. autoattribute:: tol


minimize
--------

The :func:`iminuit.minimize` function provides the same interface as :func:`scipy.optimize.minimize`.
If you are familiar with the latter, this allows you to use Minuit with a quick start.
Eventually, you still may want to learn the interface of the :class:`iminuit.Minuit` class,
as it provides more functionality if you are interested in parameter uncertainties.

.. autofunction:: iminuit.minimize

Utility Functions
-----------------

.. currentmodule:: iminuit.util

The module :mod:`iminuit.util` provides the :func:`describe` function and various function to manipulate
fit arguments. Most of these functions (apart from describe) are for internal use. You should not rely
on them in your code. We list the ones that are for the public.

.. automodule:: iminuit.util
    :members:
    :undoc-members:
    :exclude-members: arguments_from_docstring, true_param, param_name,
        extract_iv, extract_error, extract_fix, extract_limit


Data objects
------------

.. currentmodule:: iminuit

iminuit uses various data objects as return values. This section lists them.

.. _function-minimum-sruct:

Function Minimum Data Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass of NamedTuple that stores information about the fit result. It is returned by
:meth:`Minuit.get_fmin` and :meth:`Minuit.migrad`.
It has the following attributes:

    * *fval*: FCN minimum value

    * *edm*: Estimated Distance to Minimum

    * *nfcn*: Number of function call in last mimizier call

    * *up*: UP parameter. This determine how minimizer define 1 :math:`\sigma`
      error

    * *is_valid*: Validity of function minimum. This is defined as

        * has_valid_parameters
        * and not has_reached_call_limit
        * and not is_above_max_edm

    * *has_valid_parameters*: Validity of parameters. This means:

        1. The parameters must have valid error(if it's not fixed).
           Valid error is not necessarily accurate.
        2. The parameters value must be valid

    * *has_accurate_covariance*: Boolean indicating whether covariance matrix
      is accurate.

    * *has_pos_def_covar*: Positive definiteness of covariance

    * *has_made_posdef_covar*: Whether minimizer has to force covariance matrix
      to be positive definite by adding diagonal matrix.

    * *hesse_failed*: Successfulness of the hesse call after minimizer.

    * *has_covaraince*: Has Covariance.

    * *is_above_max_edm*: Is EDM above 0.0001*tolerance*up? The convergence
      of migrad is defined by EDM being below this number.

    * *has_reached_call_limit*: Whether the last minimizer exceeds number of
      FCN calls it is allowd.

.. _minos-error-struct:

Minos Data Object
~~~~~~~~~~~~~~~~~

Subclass of NamedTuple which stores information about the Minos result. It is returned by :meth:`Minuit.minos`
(as part of a dictionary from parameter name -> data object). You can get it also from :meth:`Minuit.get_merrors`. It has the following attributes:

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

Subclass of NamedTuple which stores the fit parameter state. It is returned by :meth:`Minuit.hesse` and as part of the :meth:`Minuit.migrad` result. You can access the latest parameter state by calling
:meth:`Minuit.get_param_states`, and the initial state via :meth:`Minuit.get_initial_param_states`. It has the following attrubutes:

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

        def f(x,y):
            return (x-2)**2+(y-3)**2

       or::

        f = lambda x,y: (x-2)**2+(y-3)**2

       Have these two attributes.

    2. Using ``f.__call__.func_code.co_varnames``, ``f.__call__.co_argcount``.
       Minuit knows how to skip the `self` parameter. This allow you to do
       things like encapsulate your data with in a fitting algorithm::

        class MyChi2:
            def __init__(self, x, y):
                self.x, self.y = (x,y)
            def f(self, x, m, c):
                return m*x + c
            def __call__(self,m,c):
                return sum([(self.f(x,m,c)-y)**2
                           for x,y in zip(self.x ,self.y)])

    3. If all fails, Minuit will try to read the function signature from the
       docstring to get function signature.


    This order is very similar to PyMinuit signature detection. Actually,
    it is a superset of PyMinuit signature detection.
    The difference is that it allows you to fake function
    signature by having a func_code attribute in the object. This allows you
    to make a generic functor of your custom cost function. This is explained
    in the **Advanced Tutorial** in the docs.


    .. note::

        If you are unsure what minuit will parse your function signature as
        , you can use :func:`describe` which returns tuple of argument names
        minuit will use as call signature.
