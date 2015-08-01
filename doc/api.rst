.. _api-doc:

Full API Documentation
======================

.. currentmodule:: iminuit

Quick Summary
-------------
These are the things you will use a lot:

.. autosummary::
    util.describe
    Minuit
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
    Minuit.mncontour_grid
    Minuit.draw_mncontour

Minuit
------

.. autoclass:: Minuit
    :members:
    :undoc-members:
    :exclude-members: migrad, hesse, minos, args, values, errors,
        merror, fval, edm, fitarg, covariance, gcc, errordef,
        fcn, frontend, pedantic, throw_nan, tol

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


Utility Functions
-----------------

:mod:util module provides describe function and various function to manipulate
fitarguments.

.. automodule:: iminuit.util
    :members:
    :undoc-members:

Return Value Struct
-------------------

iminuit uses various structs as return value. This section lists the struct
and all it's field

.. _function-minimum-sruct:

Function Minimum Struct
-----------------------

They are usually return value from :meth:`Minuit.get_fmin`
and :meth:`Minuit.migrad`
Function Mimum Struct has the following attributes:

    * *fval*: FCN minimum value

    * *edm*: `Estimated Distance to Minimum
      <http://en.wikipedia.org/wiki/Minimum_distance_estimation>`_.

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

Minos Error Struct
------------------

Minos Error Struct is used in return value from :meth:`Minuit.minos`.
You can also call :meth:`Minuit.get_merrors` to get accumulated dictionary
all minos errors that has been calculated. It contains various minos status:

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

Minuit Parameter Struct
-----------------------
Minuit Parameter Struct is return value from :meth:`Minuit.hesse`
You can, however, access the latest parameter by calling
:meth:`Minuit.get_param_states`. Minuit Parameter Struct has the following
attrubutes:

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
       All functions that is defined like::

        def f(x,y):
            return (x-2)**2+(y-3)**2

       or::

        f = lambda x,y: (x-2)**2+(y-3)**2

       Has these two attributes.

    2. Using ``f.__call__.func_code.co_varnames``, ``f.__call__.co_argcount``.
       Minuit knows how to dock off self parameter. This allow you to do
       things like encapsulate your data with in fitting algorithm::

        class MyChi2:
            def __init__(self, x, y):
                self.x, self.y = (x,y)
            def f(self, x, m, c):
                return m*x + c
            def __call__(self,m,c):
                return sum([(self.f(x,m,c)-y)**2
                           for x,y in zip(self.x ,self.y)])

    3. If all fail, Minuit will call ``inspect.getargspec`` for
       function signature.
       Builtin C functions will fall into this category since
       they have no signature information. ``inspect.getargspec`` will parse
       docstring to get function signature.


    This order is very similar to PyMinuit signature detection. Actually,
    it is a superset of PyMinuit signature detection.
    The difference is that it allows you to fake function
    signature by having func_code attribute in the object. This allows you
    to make a generic functor of your custom cost function. This is how
    `probfit <http://github.com/iminuit/probfit>`_ was written::

        f = lambda x,m,c: m*x+c
        #the beauty here is that all you need to build
        #a Chi^2 is just your function and data
        class GenericChi2:
            def __init__(self, f, x, y):
                self.f = f
                args = describe(f)#extract function signature
                self.func_code = Struct(
                        co_varnames = args[1:],#dock off independent param
                        co_argcount = len(args)-1
                    )
            def __call__(self, *arg):
                return sum((self.f(x,*arg)-y)**2 for x,y in zip(self.x, self.y))

        m = Minuit(GenericChi2(f,x,y))


    .. note::

        If you are unsure what minuit will parse your function signature as
        , you can use :func:`describe` which returns tuple of argument names
        minuit will use as call signature.