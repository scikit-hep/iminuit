# cython: embedsignature=True, c_string_type=str, c_string_encoding=ascii, language_level=2
# distutils: language = c++
"""IPython Minuit class definition."""
from warnings import warn
from libc.math cimport sqrt
from libcpp.string cimport string
from libcpp.cast cimport dynamic_cast
from cython.operator cimport dereference as deref
from iminuit import util as mutil
from iminuit._deprecated import deprecated
from iminuit.latex import LatexFactory
from iminuit import _minuit_methods
from collections import OrderedDict

include "Minuit2.pxi"
include "Minuit2Struct.pxi"

cimport numpy as np
import numpy as np
np.import_array()

__all__ = ['Minuit']

# Pointer types
ctypedef FCNGradientBase* FCNGradientBasePtr
ctypedef IMinuitMixin* IMinuitMixinPtr
ctypedef PythonGradientFCN* PythonGradientFCNPtr
ctypedef MnUserParameterState* MnUserParameterStatePtr
ctypedef const MnUserParameterState* MnUserParameterStateConstPtr

# Helper functions
cdef set_parameter_state(MnUserParameterStatePtr state, object parameters, dict fitarg):
    """Construct parameter state from user input.

    Caller is responsible for cleaning up the pointer.
    """
    cdef double inf = float("infinity")
    cdef double val
    cdef double err
    cdef double lb
    cdef double ub
    for i, pname in enumerate(parameters):
        val = fitarg[pname]
        err = fitarg['error_' + pname]
        state.Add(pname, val, err)

        lim = fitarg['limit_' + pname]
        if lim is not None:
            lb, ub = lim
            if lb == ub:
                state.SetValue(i, lb)
                state.Fix(i)
            else:
                if lb == -inf and ub == inf:
                    pass
                elif ub == inf:
                    state.SetLowerLimit(i, lb)
                elif lb == -inf:
                    state.SetUpperLimit(i, ub)
                else:
                    state.SetLimits(i, lb, ub)
                # need to set value again so that MINUIT can
                # correct internal/external transformation;
                # also use opportunity to correct a starting value outside of limit
                val = max(val, lb)
                val = min(val, ub)
                state.SetValue(i, val)
                state.SetError(i, err)

        if fitarg['fix_' + pname]:
            state.Fix(i)


cdef check_extra_args(parameters, kwd):
    """Check keyword arguments to find unwanted/typo keyword arguments"""
    fixed_param = set('fix_' + p for p in parameters)
    limit_param = set('limit_' + p for p in parameters)
    error_param = set('error_' + p for p in parameters)
    for k in kwd.keys():
        if k not in parameters and \
                        k not in fixed_param and \
                        k not in limit_param and \
                        k not in error_param:
            raise RuntimeError(
                ('Cannot understand keyword %s. May be a typo?\n'
                 'The parameters are %r') % (k, parameters))


cdef states_equal(n, MnUserParameterStateConstPtr a, MnUserParameterStateConstPtr b):
    result = False
    for i in range(n):
        result |= a.Parameter(i).Value() != b.Parameter(i).Value()
        result |= a.Parameter(i).Error() != b.Parameter(i).Error()
        result |= a.Parameter(i).IsFixed() != b.Parameter(i).IsFixed()
        result |= a.Parameter(i).HasLowerLimit() != b.Parameter(i).HasLowerLimit()
        result |= a.Parameter(i).HasUpperLimit() != b.Parameter(i).HasUpperLimit()
        result |= a.Parameter(i).LowerLimit() != b.Parameter(i).LowerLimit()
        result |= a.Parameter(i).UpperLimit() != b.Parameter(i).UpperLimit()
    return result


def is_number(value):
    return isinstance(value, (int, long, float))

def is_int(value):
    return isinstance(value, (int, long))

# Helper classes
cdef class BasicView:
    """Dict-like view of parameter state.

    Derived classes need to implement methods _set and _get to access
    specific properties of the parameter state."""
    cdef object _minuit
    cdef MnUserParameterStatePtr _state

    def __init__(self, minuit):
        self._minuit = minuit

    def __iter__(self):
        return self._minuit.pos2var.__iter__()

    def __len__(self):
        return len(self._minuit.pos2var)

    def keys(self):
        return [k for k in self]

    def items(self):
        return [(name, self._get(k)) for (k, name) in enumerate(self)]

    def values(self):
        return [self._get(k) for k in range(len(self))]

    def __getitem__(self, key):
        if isinstance(key, slice):
            ind = range(*key.indices(len(self)))
            return [self._get(i) for i in ind]
        i = key if is_int(key) else self._minuit.var2pos[key]
        if i < 0:
            i += len(self)
        if i >= len(self):
            raise IndexError
        return self._get(i)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            ind = range(*key.indices(len(self)))
            if hasattr(value, "__getitem__") and hasattr(value, "__len__"):
                if len(value) != len(ind):
                    raise ValueError("length of argument does not match slice")
                for i, v in zip(ind, value):
                    self._set(i, v)
            else: # basic broadcasting
                for i in ind:
                    self._set(i, value)
            return
        i = key if is_int(key) else self._minuit.var2pos[key]
        if i < 0:
            i += len(self)
        if i >= len(self):
            raise IndexError
        self._set(i, value)

    def __repr__(self):
        s = "<%s of Minuit at %x>" % (self.__class__.__name__, id(self._minuit))
        for (k, v) in self.items():
            s += "\n  {0}: {1}".format(k, v)
        return s


cdef class ArgsView:
    """List-like view of parameter values."""
    cdef object _minuit
    cdef MnUserParameterStatePtr _state

    def __init__(self, minuit):
        self._minuit = minuit

    def __len__(self):
        return len(self._minuit.pos2var)

    def __getitem__(self, key):
        cdef int i
        if isinstance(key, slice):
            return [self._state.Parameter(i).Value() for i in range(*key.indices(len(self)))]
        i = key
        if i < 0:
            i += len(self)
        if i >= len(self):
            raise IndexError
        return self._state.Parameter(i).Value()

    def __setitem__(self, key, value):
        cdef int i
        if isinstance(key, slice):
            for i, v in zip(range(*key.indices(len(self))), value):
                self._state.SetValue(i, v)
        else:
            i = key
            if i < 0:
                i += len(self)
            if i >= len(self):
                raise IndexError
            self._state.SetValue(i, value)

    def __repr__(self):
        s = "<ArgsView of Minuit at %x>" % id(self._minuit)
        for v in self:
            s += "\n  {0}".format(v)
        return s


cdef class ValueView(BasicView):
    """Dict-like view of parameter values."""
    def _get(self, unsigned int i):
        return self._state.Parameter(i).Value()

    def _set(self, unsigned int i, double value):
        self._state.SetValue(i, value)


cdef class ErrorView(BasicView):
    """Dict-like view of parameter errors."""
    def _get(self, unsigned int i):
        return self._state.Parameter(i).Error()

    def _set(self, unsigned int i, double value):
        self._state.SetError(i, value)


cdef class FixedView(BasicView):
    """Dict-like view of whether parameters are fixed."""
    def _get(self, unsigned int i):
        return self._state.Parameter(i).IsFixed()

    def _set(self, unsigned int i, bint fix):
        if fix:
            self._state.Fix(i)
        else:
            self._state.Release(i)


cdef class Minuit:
    # error definition constants
    LEAST_SQUARES = 1.0
    LIKELIHOOD = 0.5

    # Standard stuff

    cdef readonly object fcn
    """Cost function (usually a chi^2 or likelihood function)."""

    cdef readonly object grad
    """Gradient function of the cost function."""

    cdef readonly bint use_array_call
    """Boolean. Whether to pass parameters as numpy array to cost function."""

    cdef readonly tuple pos2var
    """Map variable position to name"""

    cdef readonly object var2pos
    """Map variable name to position"""

    # C++ object state
    cdef FCNBase*pyfcn  #:FCN
    cdef MnApplication*minimizer  #:migrad
    cdef FunctionMinimum*cfmin  #:last migrad result
    #:initial parameter state
    cdef MnUserParameterState initial_upst
    #:last parameter state(from hesse/migrad)
    cdef MnUserParameterState last_upst

    # PyMinuit compatible fields

    @property
    def errordef(self):
        """FCN increment above the minimum that corresponds to one standard deviation.

        Default value is 1.0. `errordef` should be 1.0 for a least-squares cost
        function and 0.5 for negative log-likelihood function. See page 37 of
        http://hep.fi.infn.it/minuit.pdf. This parameter is sometimes called
        ``UP`` in the MINUIT docs.

        To make user code more readable, we provided two named constants::

            from iminuit import Minuit
            assert Minuit.LEAST_SQUARES == 1
            assert Minuit.LIKELIHOOD == 0.5

            Minuit(a_least_squares_function, errordef=Minuit.LEAST_SQUARES)
            Minuit(a_likelihood_function, errordef=Minuit.LIKELIHOOD)
        """
        return self.pyfcn.Up()

    @errordef.setter
    def errordef(self, value):
        self.pyfcn.SetErrorDef(value)
        if self.cfmin:
            self.cfmin.SetErrorDef(value)


    cdef public double tol
    """Tolerance for convergence.

    The main convergence criteria of MINUIT is ``edm < edm_max``, where ``edm_max`` is
    calculated as ``edm_max = 0.002 * tol * errordef`` and EDM is the *estimated distance
    to minimum*, as described in the `MINUIT paper`_.
    """

    cdef public unsigned int strategy
    """Current minimization strategy.

    **0**: Fast. Does not check a user-provided gradient. Does not improve Hesse matrix
    at minimum. Extra call to :meth:`hesse` after :meth:`migrad` is always needed for
    good error estimates. If you pass a user-provided gradient to MINUIT,
    convergence is **faster**.

    **1**: Default. Checks user-provided gradient against numerical gradient. Checks and
    usually improves Hesse matrix at minimum. Extra call to :meth:`hesse` after
    :meth:`migrad` is usually superfluous. If you pass a user-provided gradient to
    MINUIT, convergence is **slower**.

    **2**: Careful. Like 1, but does extra checks of intermediate Hessian matrix during
    minimization. The effect in benchmarks is a somewhat improved accuracy at the cost
    of more function evaluations. A similar effect can be achieved by reducing the
    tolerance attr:`tol` for convergence at any strategy level.
    """

    cdef int _print_level

    @property
    def print_level(self):
        """Current print level.

        - 0: quiet
        - 1: print minimal debug messages to terminal
        - 2: print more debug messages to terminal
        - 3: print even more debug messages to terminal

        Note: Setting the level to 3 has a global side effect on all current instances of Minuit (this is an issue in C++ MINUIT2).
        """
        return self._print_level

    @print_level.setter
    def print_level(self, level):
        if level < 0: level = 0
        self._print_level = level
        if level >= 3 or level < MnPrint.Level():
            warn("Setting print_level >=3 has the side-effect of setting the level "
                 "globally for all Minuit instances",
                 mutil.IMinuitWarning)
            MnPrint.SetLevel(level)
        if self.minimizer:
            self.minimizer.Minimizer().Builder().SetPrintLevel(level)

    cdef readonly bint throw_nan
    """Boolean. Whether to raise runtime error if function evaluate to nan."""

    # PyMinuit compatible interface

    cdef readonly object parameters
    """Parameter name tuple"""

    cdef public ArgsView args
    """Parameter values in a list-like object.

    See :attr:`values` for details.

    .. seealso:: :attr:`values`, :attr:`errors`, :attr:`fixed`
    """

    cdef public ValueView values
    """Parameter values in a dict-like object.

    Use to read or write current parameter values based on the parameter index or the
    parameter name as a string. If you change a parameter value and run :meth:`migrad`,
    the minimization will start from that value, similar for :meth:`hesse` and
    :meth:`minos`.

    .. seealso:: :attr:`errors`, :attr:`fixed`
    """

    cdef public ErrorView errors
    """Parameter parabolic errors in a dict-like object.

    Like :attr:`values`, but instead of reading or writing the values, you read or write
    the errors (which double as step sizes for MINUITs numerical gradient estimation).

    .. seealso:: :attr:`values`, :attr:`fixed`
    """

    cdef public FixedView fixed
    """Access fixation state of a parameter in a dict-like object.

    Use to read or write the fixation state of a parameter based on the parameter index
    or the parameter name as a string. If you change the state and run :meth:`migrad`,
    :meth:`hesse`, or :meth:`minos`, the new state is used.

    In case of complex fits, it can help to fix some parameters first and only minimize
    the function with respect to the other parameters, then release the fixed parameters
    and minimize again starting from that state.

    .. seealso:: :attr:`values`, :attr:`errors`
    """

    cdef readonly object covariance
    """Covariance matrix (dict (name1, name2) -> covariance).

    .. seealso:: :meth:`matrix`
    """

    cdef readonly object merrors
    """MINOS errors."""

    cdef readonly int ncalls
    """Number of FCN call of last MIGRAD / MINOS / HESSE run."""

    cdef readonly int ngrads
    """Number of Gradient calls of last MIGRAD / MINOS / HESSE run."""

    cdef readonly object gcc
    """Global correlation coefficients (dict : name -> gcc)."""

    cdef readonly object fitarg
    """Current Minuit state in form of a dict.

    * name -> value
    * error_name -> error
    * fix_name -> fix
    * limit_name -> (lower_limit, upper_limit)

    This is very useful when you want to save the fit parameters and
    re-use them later. For example::

        m = Minuit(f, x=1)
        m.migrad()
        fitarg = m.fitarg

        m2 = Minuit(f, **fitarg)
    """

    @property
    def narg(self):
        """Number of parameters."""
        return len(self.parameters)

    @property
    def nfit(self):
        """Number of fitted parameters (fixed parameters not counted)."""
        nfit = 0
        for v in self.fixed.values():
            nfit += not v
        return nfit

    def __init__(self, fcn, grad=None, errordef=None,
                 print_level=0, name=None,
                 pedantic=True, throw_nan=False,
                 use_array_call=False,
                 **kwds):
        """
        Construct minuit object from given *fcn*

        **Arguments:**

            **fcn**, the function to be optimized, is the only required argument.

            Two kinds of function signatures are understood.

            a) Parameters passed as positional arguments

            The function has several positional arguments, one for each fit
            parameter. Example::

                def func(a, b, c): ...

            The parameters a, b, c must accept a real number.

            iminuit automagically detects parameters names in this case.
            More information about how the function signature is detected can
            be found in :ref:`function-sig-label`

            b) Parameters passed as Numpy array

            The function has a single argument which is a Numpy array.
            Example::

                def func(x): ...

            Pass the keyword `use_array_call=True` to use this signature. For
            more information, see "Parameter Keyword Arguments" further down.

            If you work with array parameters a lot, have a look at the static
            initializer method :meth:`from_array_func`, which adds some
            convenience and safety to this use case.

        **Builtin Keyword Arguments:**

            - **throw_nan**: set fcn to raise RuntimeError when it
              encounters *nan*. (Default False)

            - **pedantic**: warns about parameters that do not have initial
              value or initial error/stepsize set.

            - **name**: sequence of strings. If set, this is used to detect
              parameter names instead of iminuit's function signature detection.

            - **print_level**: set the print_level for this Minuit. 0 is quiet.
              1 print out at the end of MIGRAD/HESSE/MINOS. 2 prints debug messages.

            - **errordef**: Optional. See :attr:`errordef` for details on
              this parameter. If set to `None` (the default), Minuit will try to call
              `fcn.errordef` and `fcn.default_errordef()` (deprecated) to set the error
              definition. If this fails, a warning is raised and use a value appropriate
              for a least-squares function is used.

            - **grad**: Optional. Provide a function that calculates the
              gradient analytically and returns an iterable object with one
              element for each dimension. If None is given MINUIT will
              calculate the gradient numerically. (Default None)

            - **use_array_call**: Optional. Set this to true if your function
              signature accepts a single numpy array of the parameters. You
              need to also pass the `name` keyword then to
              explicitly name the parameters.

        **Parameter Keyword Arguments:**

            iminuit allows user to set initial value, initial stepsize/error, limits of
            parameters and whether the parameter should be fixed by passing keyword
            arguments to Minuit.

            This is best explained through examples::

                def f(x, y):
                    return (x-2)**2 + (y-3)**2

            * Initial value (varname)::

                #initial value for x and y
                m = Minuit(f, x=1, y=2)

            * Initial step size (fix_varname)::

                #initial step size for x and y
                m = Minuit(f, error_x=0.5, error_y=0.5)

            * Limits (limit_varname=tuple)::

                #limits x and y
                m = Minuit(f, limit_x=(-10,10), limit_y=(-20,20))

            * Fixing parameters::

                #fix x but vary y
                m = Minuit(f, fix_x=True)

            .. note::

                You can use dictionary expansion to programmatically change parameters.::

                    kwargs = dict(x=1., error_x=0.5)
                    m = Minuit(f, **kwargs)

                You can also obtain fit arguments from Minuit object for later reuse.
                *fitarg* will be automatically updated to the minimum value and the
                corresponding error when you ran migrad/hesse.::

                    m = Minuit(f, x=1, error_x=0.5)
                    my_fitarg = m.fitarg
                    another_fit = Minuit(f, **my_fitarg)

        """

        if name is None:
            name = kwds.get("forced_parameters", None)
            if name is not None:
                warn("Using keyword `forced_parameters` is deprecated. Use `name` instead",
                     DeprecationWarning,
                     stacklevel=2)
                del kwds["forced_parameters"]

        if use_array_call and name is None:
            raise KeyError("`use_array_call=True` requires that `name` is set")

        args = mutil.describe(fcn) if name is None \
            else name

        self.parameters = args
        check_extra_args(args, kwds)

        # Maintain 2 dictionaries to easily convert between
        # parameter names and position
        self.pos2var = tuple(args)
        self.var2pos = {k: i for i, k in enumerate(args)}

        if errordef is None:
            if hasattr(fcn, 'errordef'):
                errordef = fcn.errordef
            elif hasattr(fcn, 'default_errordef'):
                warn("Using .default_errordef() is deprecated. Use .errordef instead",
                     DeprecationWarning,
                     stacklevel=2)
                errordef = fcn.default_errordef()

        if errordef is not None and (is_number(errordef) == False or errordef <= 0):
            raise ValueError("errordef must be a positive number")

        if pedantic:
            _minuit_methods.pedantic(self, args, kwds, errordef)

        if errordef is None:
            errordef = 1.0

        self.fcn = fcn
        self.grad = grad
        self.use_array_call = use_array_call

        self.tol = 0.1
        self.strategy = 1
        self.print_level = print_level
        self.throw_nan = throw_nan

        if self.grad is None:
            self.pyfcn = new PythonFCN(
                self.fcn,
                self.use_array_call,
                errordef,
                self.parameters,
                self.throw_nan,
            )
        else:
            self.pyfcn = new PythonGradientFCN(
                self.fcn,
                self.grad,
                self.use_array_call,
                errordef,
                self.parameters,
                self.throw_nan,
            )

        self.fitarg = {}
        for x in args:
            lim = mutil._normalize_limit(kwds.get('limit_' + x, None))
            val = kwds.get(x, mutil._guess_initial_value(lim))
            err = kwds.get('error_' + x, mutil._guess_initial_step(val))
            fix = kwds.get('fix_' + x, False)
            self.fitarg[unicode(x)] = val
            self.fitarg['error_' + x] = err
            self.fitarg['limit_' + x] = lim
            self.fitarg['fix_' + x] = fix

        self.minimizer = NULL
        self.cfmin = NULL
        set_parameter_state(&self.initial_upst, self.parameters, self.fitarg)
        self.last_upst = self.initial_upst

        self.args = ArgsView(self)
        self.args._state = &self.last_upst
        self.values = ValueView(self)
        self.values._state = &self.last_upst
        self.errors = ErrorView(self)
        self.errors._state = &self.last_upst
        self.fixed = FixedView(self)
        self.fixed._state = &self.last_upst
        self.covariance = None
        self.ncalls = 0
        self.ngrads = 0
        self.merrors = mutil.MErrors()
        self.gcc = None



    @classmethod
    def from_array_func(cls, fcn, start, error=None, limit=None, fix=None,
                        name=None, **kwds):
        """Construct Minuit object from given *fcn* and start sequence.

        This is an alternative named constructor for the minuit object. It is
        more convenient to use for functions that accept a numpy array.

        **Arguments:**

            **fcn**: The function to be optimized. Must accept a single
            parameter that is a numpy array.

                def func(x): ...

            **start**: Sequence of numbers. Starting point for the
            minimization.

        **Keyword arguments:**

            **error**: Optional sequence of numbers. Initial step sizes.
            Scalars are automatically broadcasted to the length of the
            start sequence.

            **limit**: Optional sequence of limits that restrict the range in
            which a parameter is varied by minuit. Limits can be set in
            several ways. With inf = float("infinity") we get:

            - No limit: None, (-inf, inf), (None, None)

            - Lower limit: (x, None), (x, inf) [replace x with a number]

            - Upper limit: (None, x), (-inf, x) [replace x with a number]

            A single limit is automatically broadcasted to the length of the
            start sequence.

            **fix**: Optional sequence of boolean values. Whether to fix a
            parameter to the starting value.

            **name**: Optional sequence of parameter names. If names are not
            specified, the parameters are called x0, ..., xN.

            All other keywords are forwarded to :class:`Minuit`, see
            its documentation.

        **Example:**

            A simple example function is passed to Minuit. It accept a numpy
            array of the parameters. Initial starting values and error
            estimates are given::

                import numpy as np

                def f(x):
                    mu = (2, 3)
                    return np.sum((x-mu)**2)

                # error is automatically broadcasted to (0.5, 0.5)
                m = Minuit.from_array_func(f, (2, 3),
                                           error=0.5)

        """
        npar = len(start)
        pnames = name if name is not None else ["x%i"%i for i in range(npar)]
        kwds["name"] = pnames
        kwds["use_array_call"] = True
        if error is not None:
            if np.isscalar(error):
                error = np.ones(npar) * error
            else:
                if len(error) != npar:
                    raise RuntimeError("length of error sequence does "
                                       "not match start sequence")
        if limit is not None:
            if (len(limit) == 2 and
                np.isscalar(limit[0]) and
                np.isscalar(limit[1])):
                limit = [limit for i in range(npar)]
            else:
                if len(limit) != npar:
                    raise RuntimeError("length of limit sequence does "
                                       "not match start sequence")
        for i, name in enumerate(pnames):
            kwds[name] = start[i]
            if error is not None:
                kwds["error_" + name] = error[i]
            if limit is not None:
                kwds["limit_" + name] = limit[i]
            if fix is not None:
                kwds["fix_" + name] = fix[i]
        return Minuit(fcn, **kwds)


    def migrad(self, unsigned ncall=0, resume=True, int nsplit=1, precision=None):
        """Run MIGRAD.

        MIGRAD is a robust minimisation algorithm which earned its reputation
        in 40+ years of almost exclusive usage in high-energy physics. How
        MIGRAD works is described in the `MINUIT paper`_.

        **Arguments:**

            * **ncall**: integer (approximate) maximum number of call before
              MIGRAD will stop trying. Default: 0 (indicates to use MIGRAD's
              internal heuristic). Using nsplit > 1 requires ncall > 0.
              Note: MIGRAD may slightly violate this limit,
              because it checks the condition only after a full iteration of the
              algorithm, which usually performs several function calls.

            * **resume**: boolean indicating whether MIGRAD should resume from
              the previous minimiser attempt(True) or should start from the
              beginning(False). Default True.

            * **nsplit**: split MIGRAD in to *split* runs. Max fcn call
              for each run is ncall/nsplit. MIGRAD stops when it found the
              function minimum to be valid or ncall is reached. This is useful
              for getting progress. However, you need to make sure that
              ncall/nsplit is large enough. Otherwise, MIGRAD will think
              that the minimum is invalid due to exceeding max call
              (ncall/nsplit). Default 1(no split).

            * **precision**: override miniut own's internal precision.

        **Return:**

            :ref:`function-minimum-sruct`, list of :ref:`minuit-param-struct`
        """
        if nsplit > 1 and ncall == 0:
            raise ValueError("ncall > 0 is required for nsplit > 1")

        #construct new fcn and migrad if
        #it's a clean state or resume=False
        cdef MnStrategy*strat = NULL

        if not resume:
            self.last_upst = self.initial_upst

        if self.minimizer is not NULL:
            del self.minimizer
            self.minimizer = NULL
        strat = new MnStrategy(self.strategy)

        if self.grad is None:
            self.minimizer = new MnMigrad(
                deref(<FCNBase*> self.pyfcn),
                self.last_upst, deref(strat)
            )
        else:
            self.minimizer = new MnMigrad(
                deref(<FCNGradientBase*> self.pyfcn),
                self.last_upst, deref(strat)
            )

        del strat
        strat = NULL

        self.minimizer.Minimizer().Builder().SetPrintLevel(self.print_level)
        if precision is not None:
            self.minimizer.SetPrecision(precision)

        cdef PythonGradientFCNPtr grad_ptr = NULL
        if not resume:
            dynamic_cast[IMinuitMixinPtr](self.pyfcn).resetNumCall()
            grad_ptr = dynamic_cast[PythonGradientFCNPtr](self.pyfcn)
            if grad_ptr:
                grad_ptr.resetNumGrad()

        #this returns a real object need to copy
        ncall_round = round(1.0 * ncall / nsplit)
        assert (nsplit == 1 or ncall_round > 0)

        def total_calls():
            return self.ncalls_total - self.ncalls

        self.ncalls = self.ncalls_total
        self.ngrads = self.ngrads_total

        while total_calls() == 0 or (not self.cfmin.IsValid() and total_calls() < ncall):
            if self.cfmin:  # delete existing
                del self.cfmin
            self.cfmin = call_mnapplication_wrapper(
                deref(self.minimizer), ncall_round, self.tol)
            self.last_upst = self.cfmin.UserState()
            if self.print_level > 1 and nsplit != 1:
                print(self.fmin)

        self.last_upst = self.cfmin.UserState()
        self.refresh_internal_state()

        if self.print_level > 0:
            print(self.fmin)

        return mutil.MigradResult(self.fmin, self.params)


    def hesse(self, unsigned ncall=0, **kwargs):
        """Run HESSE to compute parabolic errors.

        HESSE estimates the covariance matrix by inverting the matrix of
        `second derivatives (Hesse matrix) at the minimum
        <http://en.wikipedia.org/wiki/Hessian_matrix>`_. This covariance
        matrix is valid if your :math:`\chi^2` or likelihood profile looks
        like a hyperparabola around the the minimum. This is usually the case,
        especially when you fit many observations (in the limit of infinite
        samples this is always the case). If you want to know how your
        parameters are correlated, you also need to use HESSE.

        Also see :meth:`minos`, which computes the uncertainties in a
        different way.

        **Arguments:**
            - **ncall**: limit the number of calls made by MINOS.
              Default: 0 (uses an internal heuristic by C++ MINUIT).

        **Returns:**

            list of :ref:`minuit-param-struct`
        """

        if "maxcall" in kwargs:
            warn("using `maxcall` keyword is deprecated, use `ncall` keyword instead",
                 DeprecationWarning,
                 stacklevel=2);
            ncall = kwargs.pop("maxcall")

        if kwargs:
            raise KeyError("keyword(s) not recognized: " + " ".join(kwargs))

        self.ncalls = self.ncalls_total
        self.ngrads = self.ngrads_total

        # must be allocated with new to avoid random crashes
        cdef MnHesse* hesse = new MnHesse(self.strategy)

        if self.cfmin:
            if states_equal(len(self.parameters), &self.last_upst, &self.cfmin.UserState()):
                # last_upst has been modified, cannot just update last cfmin
                self.last_upst = hesse.call(
                    deref(<FCNBase*> self.pyfcn),
                    self.last_upst,
                    ncall
                )
            else:
                # last_upst not modified, can update cfmin which is more efficient
                hesse.call(
                    deref(<FCNBase*> self.pyfcn),
                    deref(self.cfmin),
                    ncall
                )
                self.last_upst = self.cfmin.UserState()

            if not self.last_upst.HasCovariance():
                warn("HESSE Failed. Covariance and GlobalCC will not be available",
                     mutil.HesseFailedWarning)
        else:
            self.last_upst = hesse.call(
                deref(<FCNBase*> self.pyfcn),
                self.last_upst,
                ncall
            )

        del hesse

        self.refresh_internal_state()

        if not self.cfmin:
            # if cfmin does not exist and HESSE fails, we raise an exception
            if not self.last_upst.HasCovariance():
                raise RuntimeError("HESSE Failed")

        return self.params


    def minos(self, var=None, sigma=1., unsigned int maxcall=0):
        """Run MINOS to compute asymmetric confidence intervals.

        MINOS uses the profile likelihood method to compute (asymmetric)
        confidence intervals. It scans the negative log-likelihood or
        (equivalently) the least-squares cost function around the minimum
        to construct an asymmetric confidence interval. This interval may
        be more reasonable when a parameter is close to one of its
        parameter limits. As a rule-of-thumb: when the confidence intervals
        computed with HESSE and MINOS differ strongly, the MINOS intervals
        are to be preferred. Otherwise, HESSE intervals are preferred.

        Running MINOS is computationally expensive when there are many
        fit parameters. Effectively, it scans over *var* in small steps
        and runs MIGRAD to minimise the FCN with respect to all other free
        parameters at each point. This is requires many more FCN evaluations
        than running HESSE.

        **Arguments:**

            - **var**: optional variable name to compute the error for.
              If var is not given, MINOS is run for every variable.
            - **sigma**: number of :math:`\sigma` error. Default 1.0.
            - **maxcall**: limit the number of calls made by MINOS.
              Default: 0 (uses an internal heuristic by C++ MINUIT).

        **Returns:**

            Dictionary of varname to :ref:`minos-error-struct`, containing
            all up to now computed errors, including the current request.

        """
        if self.cfmin is NULL:
            raise RuntimeError('MINOS require function to be at the minimum.'
                               ' Run MIGRAD first.')

        cdef MnMinos*minos = NULL
        cdef MinosError mnerror
        cdef double oldup = self.pyfcn.Up()
        self.pyfcn.SetErrorDef(oldup * sigma * sigma)
        if not self.cfmin.IsValid():
            raise RuntimeError(('Function minimum is not valid. Make sure '
                                'MIGRAD converged first'))
        if var is not None and var not in self.parameters:
            raise RuntimeError('Specified parameters(%r) cannot be found '
                               'in parameter list :' % var + str(self.parameters))

        varlist = [var] if var is not None else self.parameters
        fixed_param = [k for (k, v) in self.fixed.items() if v]

        self.ncalls = self.ncalls_total
        self.ngrads = self.ngrads_total

        for vname in varlist:
            if vname in fixed_param:
                if var is not None:  #specifying vname but it's fixed
                    warn('Specified variable name for minos is set to fixed',
                         mutil.IMinuitWarning)
                    return None
                continue
            if self.grad is None:
                minos = new MnMinos(
                    deref(<FCNBase*> self.pyfcn),
                    deref(self.cfmin), self.strategy
                )
            else:
                minos = new MnMinos(
                    deref(dynamic_cast[FCNGradientBasePtr](self.pyfcn)),
                    deref(self.cfmin), self.strategy
                )

            mnerror = minos.Minos(self.var2pos[vname], maxcall)
            self.merrors[vname] = minoserror2struct(vname, mnerror)


        self.refresh_internal_state()
        del minos
        self.pyfcn.SetErrorDef(oldup)
        return self.merrors


    def matrix(self, correlation=False, skip_fixed=True):
        """Error or correlation matrix in tuple or tuples format."""
        if not self.last_upst.HasCovariance():
            raise RuntimeError(
                "Covariance is not valid. May be the last Hesse call failed?")

        cdef MnUserCovariance mncov = self.last_upst.Covariance()
        cdef vector[MinuitParameter] mp = self.last_upst.MinuitParameters()

        # When some parameters are fixed, mncov is a sub-matrix. If skip-fixed
        # is false, we need to expand the sub-matrix back into the full form.
        # This requires a translation between sub-index und full-index.
        if skip_fixed:
            npar = 0
            for i in range(mp.size()):
                if not mp[i].IsFixed():
                    npar += 1
            ind = range(npar)
            def cov(i, j):
                return mncov.get(i, j)
        else:
            ext2int = {}
            iint = 0
            for i in range(mp.size()):
                if not mp[i].IsFixed():
                    ext2int[i] = iint
                    iint += 1
            ind = range(mp.size())
            def cov(i, j):
                if i not in ext2int or j not in ext2int:
                    return 0.0
                return mncov.get(ext2int[i], ext2int[j])

        names = [k for (k, v) in self.fixed.items() if not (skip_fixed and v)]
        if correlation:
            def cor(i, j):
                return cov(i, j) / (sqrt(cov(i, i) * cov(j, j)) + 1e-100)
            ret = mutil.Matrix(names, ((cor(i, j) for i in ind) for j in ind))
        else:
            ret = mutil.Matrix(names, ((cov(i, j) for i in ind) for j in ind))
        return ret

    def latex_matrix(self):
        """Build :class:`LatexFactory` object with correlation matrix."""
        matrix = self.matrix(correlation=True, skip_fixed=True)
        return LatexFactory.build_matrix(matrix.names, matrix)

    def np_matrix(self, **kwds):
        """Covariance or correlation matrix in numpy array format.

        Keyword arguments are forwarded to :meth:`matrix`.

        The name of this function was chosen to be analogous to :meth:`matrix`,
        it returns the same information in a different format. For
        documentation on the arguments, please see :meth:`matrix`.

        **Returns:**

            2D ``numpy.ndarray`` of shape (N,N) (not a ``numpy.matrix``).
        """
        matrix = self.matrix(**kwds)
        return np.array(matrix, dtype=np.double)

    def np_values(self):
        """Parameter values in numpy array format.

        Fixed parameters are included, the order follows :attr:`parameters`.

        **Returns:**

            ``numpy.ndarray`` of shape (N,).
        """
        return np.array(self.args, dtype=np.double)

    def np_errors(self):
        """Hesse parameter errors in numpy array format.

        Fixed parameters are included, the order follows :attr:`parameters`.

        **Returns:**

            ``numpy.ndarray`` of shape (N,).
        """
        a = np.empty(len(self.parameters), dtype=np.double)
        for i, k in enumerate(self.parameters):
            a[i] = self.errors[k]
        return a

    def np_merrors(self):
        """MINOS parameter errors in numpy array format.

        Fixed parameters are included, the order follows :attr:`parameters`.

        The format of the produced array follows matplotlib conventions, as
        in ``matplotlib.pyplot.errorbar``. The shape is (2, N) for N
        parameters. The first row represents the downward error as a positive
        offset from the center. Likewise, the second row represents the
        upward error as a positive offset from the center.

        **Returns:**

            ``numpy.ndarray`` of shape (2, N).
        """
        # array format follows matplotlib conventions, see pyplot.errorbar
        a = np.empty((2, len(self.parameters)), dtype=np.double)
        for i, k in enumerate(self.parameters):
            me = self.merrors[k]
            a[0, i] = -me.lower
            a[1, i] = me.upper
        return a

    def np_covariance(self):
        """Covariance matrix in numpy array format.

        Fixed parameters are included, the order follows :attr:`parameters`.

        **Returns:**

            ``numpy.ndarray`` of shape (N,N) (not a ``numpy.matrix``).
        """
        return self.np_matrix(correlation=False, skip_fixed=False)

    def latex_param(self):
        """build :class:`iminuit.latex.LatexTable` for current parameter"""
        return LatexFactory.build_param_table(self.params, self.merrors)

    def latex_initial_param(self):
        """Build :class:`iminuit.latex.LatexTable` for initial parameter"""
        params = self.get_initial_param_states()
        return LatexFactory.build_param_table(params, {})

    @property
    def fmin(self):
        """Current function minimum data object"""
        fmin = None
        if self.cfmin is not NULL:
            fmin = cfmin2struct(self.cfmin, self.tol, self.ncalls_total)
        return fmin

    @property
    def fval(self):
        """Last evaluated FCN value

        .. seealso:: :meth:`fmin`
        """
        fmin = self.fmin
        return fmin.fval if fmin else None

    @property
    def params(self):
        """List of current parameter data objects"""
        cdef vector[MinuitParameter] vmps = self.last_upst.MinuitParameters()
        return mutil.Params((minuitparam2struct(vmps[i]) for i in range(vmps.size())),
                            self.merrors)

    @property
    def init_params(self):
        """List of current parameter data objects set to the initial fit state"""
        cdef vector[MinuitParameter] vmps = self.initial_upst.MinuitParameters()
        return mutil.Params((minuitparam2struct(vmps[i]) for i in range(vmps.size())),
                            None)

    @property
    def ncalls_total(self):
        """Total number of calls to FCN (not just the last operation)"""
        cdef IMinuitMixinPtr ptr = dynamic_cast[IMinuitMixinPtr](self.pyfcn)
        return ptr.getNumCall() if ptr else 0

    @property
    def ngrads_total(self):
        """Total number of calls to Gradient (not just the last operation)"""
        cdef PythonGradientFCNPtr ptr = dynamic_cast[PythonGradientFCNPtr](self.pyfcn)
        return ptr.getNumGrad() if ptr else 0

    @property
    def valid(self):
        """Check if function minimum is valid."""
        return self.cfmin is not NULL and self.cfmin.IsValid()

    @property
    def accurate(self):
        """Check if covariance (of the last MIGRAD run) is accurate."""
        return self.cfmin is not NULL and self.cfmin.HasAccurateCovar()

    # Various utility functions

    def is_clean_state(self):
        """Check if minuit is in a clean state, ie. no MIGRAD call"""
        return self.minimizer is NULL and self.cfmin is NULL

    cdef void clear_cobj(self):
        # clear C++ internal state
        del self.pyfcn
        self.pyfcn = NULL
        del self.minimizer
        self.minimizer = NULL
        del self.cfmin
        self.cfmin = NULL

    def __dealloc__(self):
        self.clear_cobj()

    def mnprofile(self, vname, bins=30, bound=2, subtract_min=False):
        """Calculate MINOS profile around the specified range.

        Scans over **vname** and minimises FCN over the other parameters in each point.

        **Arguments:**

            * **vname** name of variable to scan

            * **bins** number of scanning bins. Default 30.

            * **bound**
              If bound is tuple, (left, right) scanning bound.
              If bound is\\ a number, it specifies how many :math:`\sigma`
              symmetrically from minimum (minimum+- bound* :math:`\sigma`).
              Default 2

            * **subtract_min** subtract_minimum off from return value. This
              makes it easy to label confidence interval. Default False.

        **Returns:**

            bins(center point), value, MIGRAD results
        """
        if vname not in self.parameters:
            raise ValueError('Unknown parameter %s' % vname)

        if is_number(bound):
            if not self.accurate:
                warn('Specify nsigma bound but error '
                     'but error matrix is not accurate.',
                     mutil.IMinuitWarning)
            start = self.values[vname]
            sigma = self.errors[vname]
            bound = (start - bound * sigma, start + bound * sigma)

        values = np.linspace(bound[0], bound[1], bins, dtype=np.double)
        results = np.empty(bins, dtype=np.double)
        migrad_status = np.empty(bins, dtype=np.bool)
        cdef double vmin = float("infinity")
        for i, v in enumerate(values):
            fitparam = self.fitarg.copy()
            fitparam[vname] = v
            fitparam['fix_%s' % vname] = True
            m = Minuit(self.fcn, print_level=0,
                       pedantic=False, forced_parameters=self.parameters,
                       use_array_call=self.use_array_call,
                       **fitparam)
            m.migrad()
            migrad_status[i] = m.valid
            if not m.valid:
                warn('MIGRAD fails to converge for %s=%f' % (vname, v),
                     mutil.IMinuitWarning)
            results[i] = m.fval
            if m.fval < vmin:
                vmin = m.fval

        if subtract_min:
            results -= vmin

        return values, results, migrad_status

    def draw_mnprofile(self, vname, bins=30, bound=2, subtract_min=False,
                       band=True, text=True):
        """Draw MINOS profile in the specified range.

        It is obtained by finding MIGRAD results with **vname** fixed
        at various places within **bound**.

        **Arguments:**

            * **vname** variable name to scan

            * **bins** number of scanning bin. Default 30.

            * **bound**
              If bound is tuple, (left, right) scanning bound.
              If bound is a number, it specifies how many :math:`\sigma`
              symmetrically from minimum (minimum+- bound* :math:`\sigma`).
              Default 2.

            * **subtract_min** subtract_minimum off from return value. This
              makes it easy to label confidence interval. Default False.

            * **band** show green band to indicate the increase of fcn by
              *errordef*. Default True.

            * **text** show text for the location where the fcn is increased
              by *errordef*. This is less accurate than :meth:`minos`.
              Default True.

        **Returns:**

            bins(center point), value, migrad results

        .. plot:: pyplots/draw_mnprofile.py
            :include-source:
        """
        x, y, s = self.mnprofile(vname, bins, bound, subtract_min)
        return _minuit_methods.draw_profile(self, vname, x, y, band, text)

    def profile(self, vname, bins=100, bound=2, subtract_min=False, **kwargs):
        """Calculate cost function profile around specify range.

        **Arguments:**

            * **vname** variable name to scan

            * **bins** number of scanning bin. Default 100.

            * **bound**
              If bound is tuple, (left, right) scanning bound.
              If bound is a number, it specifies how many :math:`\sigma`
              symmetrically from minimum (minimum+- bound* :math:`\sigma`).
              Default: 2.

            * **subtract_min** subtract_minimum off from return value. This
              makes it easy to label confidence interval. Default False.

        **Returns:**

            bins(center point), value

        .. seealso::

            :meth:`mnprofile`
        """
        if "args" in kwargs:
            warn("The args keyword has been removed.",
                 DeprecationWarning,
                 stacklevel=2)
            del kwargs["args"]

        if len(kwargs):
            raise ValueError("Unrecognized keywords: " + " ".join(kwargs))

        if subtract_min and self.cfmin is NULL:
            raise RuntimeError("Request for minimization "
                               "subtraction but no minimization has been done. "
                               "Run MIGRAD first.")

        if is_number(bound):
            start = self.values[vname]
            sigma = self.errors[vname]
            bound = (start - bound * sigma, start + bound * sigma)

        return _minuit_methods.profile(self, vname, bins, bound, subtract_min)

    def draw_profile(self, vname, bins=100, bound=2,
                     subtract_min=False, band=True, text=True, **kwargs):
        """A convenient wrapper for drawing profile using matplotlib.

        A 1D scan of the cost function around the minimum, useful to inspect the
        minimum and the FCN around the minimum for defects.

        For a fit with several free parameters this is not the same as the MINOS
        profile computed by :meth:`draw_mncontour`. Use :meth:`mnprofile` or
        :meth:`draw_mnprofile` to compute confidence intervals.

        If a function minimum was found in a previous MIGRAD call, a vertical line
        indicates the parameter value. An optional band indicates the uncertainty
        interval of the parameter computed by HESSE or MINOS.

        **Arguments:**

            In addition to argument listed on :meth:`profile`. draw_profile
            take these addition argument:

            * **band** show green band to indicate the increase of fcn by
              *errordef*. Note again that this is NOT minos error in general.
              Default True.

            * **text** show text for the location where the fcn is increased
              by *errordef*. This is less accurate than :meth:`minos`
              Note again that this is NOT minos error in general. Default True.

        .. seealso::
            :meth:`mnprofile`
            :meth:`draw_mnprofile`
            :meth:`profile`
        """

        if "args" in kwargs:
            warn("The args keyword has been removed.",
                 DeprecationWarning,
                 stacklevel=2)
            del kwargs["args"]

        if len(kwargs):
            raise ValueError("Unrecognized keywords: " + " ".join(kwargs))

        x, y = self.profile(vname, bins, bound, subtract_min)
        return _minuit_methods.draw_profile(self, vname, x, y, band, text)

    def contour(self, x, y, bins=50, bound=2, subtract_min=False, **kwargs):
        """2D contour scan.

        Return the contour of a function scan over **x** and **y**, while keeping
        all other parameters fixed.

        The related :meth:`mncontour` works differently: for new pair of **x** and **y**
        in the scan, it minimises the function with the respect to the other parameters.

        This method is useful to inspect the function near the minimum to detect issues
        (the contours should look smooth). Use :meth:`mncontour` to create confidence
        regions for the parameters. If the fit has only two free parameters, you can
        use this instead of :meth:`mncontour`.

        **Arguments:**

            - **x** variable name for X axis of scan

            - **y** variable name for Y axis of scan

            - **bound**
              If bound is 2x2 array, [[v1min,v1max],[v2min,v2max]].
              If bound is a number, it specifies how many :math:`\sigma`
              symmetrically from minimum (minimum+- bound*:math:`\sigma`).
              Default: 2.

            - **subtract_min** subtract_minimum off from return values. Default False.

        **Returns:**

            x_bins, y_bins, values

            values[y, x] <-- this choice is so that you can pass it
            to through matplotlib contour()

        .. seealso::

            :meth:`mncontour`
            :meth:`mnprofile`

        """

        if "args" in kwargs:
            warn("The args keyword has been removed.",
                 DeprecationWarning,
                 stacklevel=2)
            del kwargs["args"]

        if len(kwargs):
            raise ValueError("Unrecognized keywords: " + " ".join(kwargs))

        if subtract_min and self.cfmin is NULL:
            raise RuntimeError("Request for minimization "
                               "subtraction but no minimization has been done. "
                               "Run MIGRAD first.")

        if is_number(bound):
            x_start = self.values[x]
            x_sigma = self.errors[x]
            x_bound = (x_start - bound * x_sigma, x_start + bound * x_sigma)
            y_start = self.values[y]
            y_sigma = self.errors[y]
            y_bound = (y_start - bound * y_sigma, y_start + bound * y_sigma)
        else:
            x_bound = bound[0]
            y_bound = bound[1]

        x_val = np.linspace(x_bound[0], x_bound[1], bins)
        y_val = np.linspace(y_bound[0], y_bound[1], bins)

        cdef int x_pos = self.var2pos[x]
        cdef int y_pos = self.var2pos[y]

        cdef list arg = list(self.args)

        result = np.empty((bins, bins), dtype=np.double)
        if self.use_array_call:
            varg = np.array(arg, dtype=np.double)
            for i, x in enumerate(x_val):
                varg[x_pos] = x
                for j, y in enumerate(y_val):
                    varg[y_pos] = y
                    result[i, j] = self.fcn(varg)
        else:
            for i, x in enumerate(x_val):
                arg[x_pos] = x
                for j, y in enumerate(y_val):
                    arg[y_pos] = y
                    result[i, j] = self.fcn(*arg)


        if subtract_min:
            result -= self.cfmin.Fval()

        return x_val, y_val, result

    def mncontour(self, x, y, int numpoints=100, sigma=1.0):
        """Two-dimensional MINOS contour scan.

        This scans over **x** and **y** and minimises all other free
        parameters in each scan point. This works as if **x** and **y** are
        fixed, while the other parameters are minimised by MIGRAD.

        This scan produces a statistical confidence region with the `profile
        likelihood method <https://en.wikipedia.org/wiki/Likelihood_function#Profile_likelihood>`_.
        The contour line represents the values of **x** and **y** where the
        function passes the threshold that corresponds to `sigma` standard
        deviations (note that 1 standard deviations in two dimensions has a
        smaller coverage probability than 68 %).

        The calculation is expensive since it has to run MIGRAD at various
        points.

        **Arguments:**

            - **x** string variable name of the first parameter

            - **y** string variable name of the second parameter

            - **numpoints** number of points on the line to find. Default 20.

            - **sigma** number of sigma for the contour line. Default 1.0.

        **Returns:**

            x MINOS error struct, y MINOS error struct, contour line

            contour line is a list of the form
            [[x1,y1]...[xn,yn]]

        .. seealso::

            :meth:`contour`
            :meth:`mnprofile`

        """
        if self.cfmin is NULL:
            raise ValueError('Run MIGRAD first')

        cdef unsigned int ix = self.var2pos[x]
        cdef unsigned int iy = self.var2pos[y]

        vary_param = [k for (k, v) in self.fixed.items() if not v]

        if x not in vary_param or y not in vary_param:
            raise ValueError('mncontour has to be run on vary parameters.')

        cdef double oldup = self.pyfcn.Up()
        self.pyfcn.SetErrorDef(oldup * sigma * sigma)

        cdef MinosErrorHolder meh
        meh = get_minos_error(deref(<FCNBase *> self.pyfcn),
                              deref(self.cfmin),
                              self.strategy,
                              ix, iy, numpoints)

        xminos = minoserror2struct(x, meh.x)
        yminos = minoserror2struct(y, meh.y)

        self.pyfcn.SetErrorDef(oldup)

        return xminos, yminos, meh.points  # using type coersion here

    def draw_mncontour(self, x, y, nsigma=2, numpoints=100):
        """Draw MINOS contour.

        **Arguments:**

            - **x**, **y** parameter name

            - **nsigma** number of sigma contours to draw

            - **numpoints** number of points to calculate for each contour

        **Returns:**

            contour

        .. seealso::

            :meth:`mncontour`

        """
        return _minuit_methods.draw_mncontour(self, x, y, nsigma, numpoints)

    def draw_contour(self, x, y, bins=50, bound=2, **kwargs):
        """Convenience wrapper for drawing contours.

        The arguments are the same as :meth:`contour`.

        Please read the docs of :meth:`contour` and :meth:`mncontour` to understand the
        difference between the two.

        .. seealso::

            :meth:`contour`
            :meth:`draw_mncontour`

        """
        if "show_sigma" in kwargs:
            warn("The show_sigma keyword has been removed due to potential confusion. "
                 "Use draw_mncontour to draw sigma contours.",
                 DeprecationWarning,
                 stacklevel=2)
            del kwargs["show_sigma"]
        if "args" in kwargs:
            warn("The args keyword is unused and has been removed.",
                 DeprecationWarning,
                 stacklevel=2)
            del kwargs["args"]
        if len(kwargs):
            raise ValueError("Invalid keyword(s): " + " ".join(kwargs))

        return _minuit_methods.draw_contour(self, x, y, bins, bound)

    cdef refresh_internal_state(self):
        """Refresh internal state attributes.

        These attributes should be in a function instead
        but kept here for PyMinuit compatibility
        """
        cdef vector[MinuitParameter] mpv
        cdef MnUserCovariance cov
        cdef double tmp = 0
        mpv = self.last_upst.MinuitParameters()
        self.fitarg.update({unicode(k): v for k, v in self.values.items()})
        self.fitarg.update({'error_' + k: v for k, v in self.errors.items()})
        vary_param = [k for (k, v) in self.fixed.items() if not v]
        if self.last_upst.HasCovariance():
            cov = self.last_upst.Covariance()
            self.covariance = \
                {(v1, v2): cov.get(i, j) \
                 for i, v1 in enumerate(vary_param) \
                 for j, v2 in enumerate(vary_param)}
        else:
            self.covariance = None
        self.ncalls = self.ncalls_total - self.ncalls
        self.ngrads = self.ngrads_total - self.ngrads
        self.gcc = None
        if self.last_upst.HasGlobalCC() and self.last_upst.GlobalCC().IsValid():
            self.gcc = {v: self.last_upst.GlobalCC().GlobalCC()[i] for \
                        i, v in enumerate(vary_param)}

    @property
    def edm(self):
        warn(
             ":attr:`edm` is deprecated: Use `this_object.fmin.edm` instead",
             DeprecationWarning,
             stacklevel=2,
        )
        fmin = self.fmin
        return fmin.edm if fmin else None

    @property
    def merrors_struct(self):
        warn(
             ":attr:`merrors_struct` is deprecated: Use `this_object.merrors` instead",
             DeprecationWarning,
             stacklevel=2,
        )
        return self.merrors

    @deprecated("use `this_object.merrors` instead")
    def get_merrors(self):
        return self.merrors

    @deprecated("use `this_object.fmin` instead")
    def get_fmin(self):
        return self.fmin

    @deprecated("Use `this_object.params` instead")
    def get_param_states(self):
        return self.params

    @deprecated("Use `this_object.init_params` instead")
    def get_initial_param_states(self):
        return self.init_params

    @deprecated("Use `this_object.ncalls_total` instead")
    def get_num_call_fcn(self):
        return self.ncalls_total

    @deprecated("Use `this_object.ngrads_total` instead")
    def get_num_call_grad(self):
        return self.ngrads_total

    @deprecated("use `this_object.fixed` instead")
    def is_fixed(self, vname):
        return self.fixed[vname]

    @deprecated("Use `this_object.valid` instead")
    def migrad_ok(self):
        return self.valid

    @deprecated("use `print(this_object.matrix())` instead")
    def print_matrix(self):
        print(self.matrix(correlation=True, skip_fixed=True))

    @deprecated("use `print(this_object.fmin)` instead")
    def print_fmin(self):
        print(self.fmin)

    @deprecated("use `print(this_object.merrors)` instead")
    def print_all_minos(self):
        print(self.merrors)

    @deprecated("use `print(this_object.params)` instead")
    def print_param(self, **kwds):
        print(self.params)

    @deprecated("use `print(this_object.init_params)` instead")
    def print_initial_param(self, **kwds):
        print(self.get_initial_param_states())

    @deprecated("use `this_object.errordef = value` instead")
    def set_errordef(self, value):
        self.errordef = value

    @deprecated("use `this_object.errordef = value` instead")
    def set_up(self, value):
        self.errordef = value

    @deprecated("use `this_object.strategy` instead")
    def set_strategy(self, value):
        self.strategy = value

    @deprecated("use `this_object.print_level` instead")
    def set_print_level(self, value):
        self.print_level = value

    @deprecated("use `[name for (name,fix) in this_object.fixed.items() if fix]`")
    def list_of_fixed_param(self):
        return [name for (name, is_fixed) in self.fixed.items() if is_fixed]

    @deprecated("use `[name for (name,fix) in this_object.fixed.items() if not fix]`")
    def list_of_vary_param(self):
        return [name for (name, is_fixed) in self.fixed.items() if not is_fixed]

    @deprecated("use `this_object.accurate`")
    def matrix_accurate(self):
        return self.accurate
