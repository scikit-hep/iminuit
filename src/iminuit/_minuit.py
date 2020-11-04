from warnings import warn
from . import util as mutil, _minuit_methods
from .latex import LatexFactory
from ._core import (
    FCN,
    MnContours,
    MnHesse,
    MnMigrad,
    MnMinos,
    MnPrint,
    MnStrategy,
    MnUserParameterState,
)
import numpy as np

__all__ = ["Minuit"]


def minoserror2struct(name, m):
    return mutil.MError(
        name,
        m.is_valid,
        m.lower,
        m.upper,
        m.lower_valid,
        m.upper_valid,
        m.at_lower_limit,
        m.at_upper_limit,
        m.at_lower_max_fcn,
        m.at_upper_max_fcn,
        m.lower_new_min,
        m.upper_new_min,
        m.nfcn,
        m.min,
    )


def fmin2struct(fmin, up, tolerance, ncalls, ncalls_total, ngrads, ngrads_total):
    has_parameters_at_limit = False
    for mp in fmin.state:
        if not mp.has_limits:
            continue
        v = mp.value
        e = mp.error
        lb = mp.lower_limit
        ub = mp.upper_limit
        # the 0.5 error threshold is somewhat arbitrary
        has_parameters_at_limit |= min(v - lb, ub - v) < 0.5 * e

    return mutil.FMin(
        fmin.fval,
        fmin.edm,
        tolerance,
        ncalls,
        ncalls_total,
        up,
        fmin.is_valid,
        fmin.has_valid_parameters,
        fmin.has_accurate_covar,
        fmin.has_posdef_covar,
        fmin.has_made_posdef_covar,
        fmin.hesse_failed,
        fmin.has_covariance,
        fmin.is_above_max_edm,
        fmin.has_reached_call_limit,
        has_parameters_at_limit,
        ngrads,
        ngrads_total,
    )


def get_params(mps, merrors):
    return mutil.Params(
        (
            mutil.Param(
                mp.number,
                mp.name,
                mp.value,
                mp.error,
                mp.is_const,
                mp.is_fixed,
                mp.has_limits,
                mp.has_lower_limit,
                mp.has_upper_limit,
                mp.lower_limit if mp.has_lower_limit else None,
                mp.upper_limit if mp.has_upper_limit else None,
            )
            for mp in mps
        ),
        merrors,
    )


def is_int(value):
    return isinstance(value, int)


# Helper classes
class BasicView:
    """Dict-like view of parameter state.

    Derived classes need to implement methods _set and _get to access
    specific properties of the parameter state."""

    _minuit = None

    def __init__(self, minuit):
        self._minuit = minuit

    def __iter__(self):
        return self._minuit.pos2var.__iter__()

    def __len__(self):
        return len(self._minuit.pos2var)

    def keys(self):
        return self._minuit.pos2var

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
            else:  # basic broadcasting
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


class ArgsView:
    """List-like view of parameter values."""

    _minuit = None

    def __init__(self, minuit):
        self._minuit = minuit

    def __len__(self):
        return len(self._minuit._pos2var)

    def __getitem__(self, key):
        if isinstance(key, slice):
            ind = range(*key.indices(len(self)))
            return [self._minuit._last_state[i].value for i in ind]
        i = key
        if i < 0:
            i += len(self)
        if i >= len(self):
            raise IndexError
        return self._minuit._last_state[i].value

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            ind = range(*key.indices(len(self)))
            for i, v in zip(ind, value):
                self._minuit._last_state.set_value(i, v)
        else:
            i = key
            if i < 0:
                i += len(self)
            if i >= len(self):
                raise IndexError
            self._minuit._last_state.set_value(i, value)

    def __repr__(self):
        s = "<ArgsView of Minuit at %x>" % id(self._minuit)
        for v in self:
            s += "\n  {0}".format(v)
        return s


class ValueView(BasicView):
    """Dict-like view of parameter values."""

    def _get(self, i):
        return self._minuit._last_state[i].value

    def _set(self, i, value):
        self._minuit._last_state.set_value(i, value)


class ErrorView(BasicView):
    """Dict-like view of parameter errors."""

    def _get(self, i):
        return self._minuit._last_state[i].error

    def _set(self, i, value):
        self._minuit._last_state.set_error(i, value)


class FixedView(BasicView):
    """Dict-like view of whether parameters are fixed."""

    def _get(self, i):
        return self._minuit._last_state[i].is_fixed

    def _set(self, i, fix):
        if fix:
            self._minuit._last_state.fix(i)
        else:
            self._minuit._last_state.release(i)


class Minuit:
    LEAST_SQUARES = 1.0
    """Set `:attr:errordef` to this constant for a least-squares cost function."""

    LIKELIHOOD = 0.5
    """Set `:attr:errordef` to this constant for a negative log-likelihood function."""

    @property
    def fcn(self):
        """Cost function (usually a chi^2 or likelihood function)."""
        return self._fcn

    @property
    def grad(self):
        """Gradient function of the cost function."""
        return self._grad

    @property
    def use_array_call(self):
        """Boolean. Whether to pass parameters as numpy array to cost function."""
        return self._fcn.use_array_call

    @property
    def pos2var(self):
        """Map variable position to name"""
        return self._pos2var

    @property
    def var2pos(self):
        """Map variable name to position"""
        return self._var2pos

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
        return self._fcn.up

    @errordef.setter
    def errordef(self, value):
        self._fcn.up = value
        if self._fmin:
            self._fmin.up = value

    tol = 0.1
    """Tolerance for convergence.

    The main convergence criteria of MINUIT is ``edm < edm_max``, where ``edm_max`` is
    calculated as ``edm_max = 0.002 * tol * errordef`` and EDM is the *estimated distance
    to minimum*, as described in the `MINUIT paper`_.
    """

    _strategy = None

    @property
    def strategy(self):
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
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        self._strategy.strategy = value

    @property
    def print_level(self):
        """Current print level.

        - 0: quiet
        - 1: print minimal debug messages to terminal
        - 2: print more debug messages to terminal
        - 3: print even more debug messages to terminal

        Note: Setting the level to 3 has a global side effect on all current instances
        of Minuit (this is an issue in C++ MINUIT2).
        """
        return self._print_level

    @print_level.setter
    def print_level(self, level):
        if level < 0:
            level = 0
        self._print_level = level
        if level >= 3 or level < MnPrint.global_level:
            warn(
                "Setting print_level >=3 has the side-effect of setting the level "
                "globally for all Minuit instances",
                mutil.IMinuitWarning,
            )
            MnPrint.global_level = level

    throw_nan = False
    """Boolean. Whether to raise runtime error if function evaluate to nan."""

    args = None
    """Parameter values in a list-like object.

    See :attr:`values` for details.

    .. seealso:: :attr:`values`, :attr:`errors`, :attr:`fixed`
    """

    values = None
    """Parameter values in a dict-like object.

    Use to read or write current parameter values based on the parameter index or the
    parameter name as a string. If you change a parameter value and run :meth:`migrad`,
    the minimization will start from that value, similar for :meth:`hesse` and
    :meth:`minos`.

    .. seealso:: :attr:`errors`, :attr:`fixed`
    """

    errors = None
    """Parameter parabolic errors in a dict-like object.

    Like :attr:`values`, but instead of reading or writing the values, you read or write
    the errors (which double as step sizes for MINUITs numerical gradient estimation).

    .. seealso:: :attr:`values`, :attr:`fixed`
    """

    fixed = None
    """Access fixation state of a parameter in a dict-like object.

    Use to read or write the fixation state of a parameter based on the parameter index
    or the parameter name as a string. If you change the state and run :meth:`migrad`,
    :meth:`hesse`, or :meth:`minos`, the new state is used.

    In case of complex fits, it can help to fix some parameters first and only minimize
    the function with respect to the other parameters, then release the fixed parameters
    and minimize again starting from that state.

    .. seealso:: :attr:`values`, :attr:`errors`
    """

    merrors = None
    """MINOS errors."""

    @property
    def fitarg(self):
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

        kwargs = {}
        for mp in self._last_state:
            kwargs[mp.name] = mp.value
            kwargs[f"error_{mp.name}"] = mp.error
            if mp.is_fixed:
                kwargs[f"fix_{mp.name}"] = mp.is_fixed
            has_lower = mp.has_lower_limit
            has_upper = mp.has_upper_limit
            if has_lower or has_upper:
                kwargs[f"limit_{mp.name}"] = (
                    mp.lower_limit if has_lower else -np.inf,
                    mp.upper_limit if has_upper else np.inf,
                )
        return kwargs

    @property
    def parameters(self):
        """Parameter name tuple"""
        return self._pos2var

    @property
    def narg(self):
        """Number of parameters."""
        return len(self._init_state)

    @property
    def nfit(self):
        """Number of fitted parameters (fixed parameters not counted)."""
        return self.narg - sum(self.fixed.values())

    @property
    def covariance(self):
        """Covariance matrix (dict (name1, name2) -> covariance).

        .. seealso:: :meth:`matrix`
        """
        free = tuple(self._free_parameters())
        cov = self._last_state.covariance
        if self._last_state.has_covariance:
            return {
                (v1, v2): cov[i, j]
                for i, v1 in enumerate(free)
                for j, v2 in enumerate(free)
            }

    @property
    def gcc(self):
        """Global correlation coefficients (dict : name -> gcc)."""
        free = self._free_parameters()
        if self._last_state.has_globalcc:
            gcc = self._last_state.globalcc
            if gcc:
                return {v: gcc[i] for i, v in enumerate(free)}

    _print_level = 0
    _ncalls = 0
    _ngrads = 0
    _fmin = None

    def __init__(
        self,
        fcn,
        grad=None,
        errordef=None,
        print_level=0,
        name=None,
        pedantic=True,
        throw_nan=False,
        use_array_call=False,
        **kwds,
    ):
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
                warn(
                    "Using keyword `forced_parameters` is deprecated. Use `name` instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                del kwds["forced_parameters"]

        if use_array_call and name is None:
            raise KeyError("`use_array_call=True` requires that `name` is set")

        args = mutil.describe(fcn) if name is None else name

        # Maintain 2 dictionaries to easily convert between
        # parameter names and position
        self._pos2var = tuple(args)
        self._var2pos = {k: i for i, k in enumerate(args)}
        _minuit_methods.check_extra_args(args, kwds)

        if errordef is None:
            if hasattr(fcn, "errordef"):
                errordef = fcn.errordef
            elif hasattr(fcn, "default_errordef"):
                warn(
                    "Using .default_errordef() is deprecated. Use .errordef instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                errordef = fcn.default_errordef()

        if errordef is not None:
            errordef = float(errordef)
            if errordef <= 0:
                raise ValueError("errordef must be a positive number")

        if pedantic:
            _minuit_methods.pedantic(self, args, kwds, errordef)

        if errordef is None:
            errordef = 1.0

        self.throw_nan = throw_nan
        self.print_level = print_level
        self._strategy = MnStrategy(1)

        self._fcn = FCN(fcn, grad, use_array_call, errordef)

        self._init_state = self._make_init_state(kwds)
        self._last_state = self._init_state

        self.args = ArgsView(self)
        self.values = ValueView(self)
        self.errors = ErrorView(self)
        self.fixed = FixedView(self)
        self.merrors = mutil.MErrors()

    def _make_init_state(self, kwds):
        state = MnUserParameterState()
        for i, x in enumerate(self.pos2var):
            lim = mutil._normalize_limit(kwds.get(f"limit_{x}", None))
            val = kwds.get(x, mutil._guess_initial_value(lim))
            err = kwds.get(f"error_{x}", mutil._guess_initial_step(val))
            fix = kwds.get(f"fix_{x}", False)
            if lim is None:
                state.add(x, val, err)
            else:
                lb, ub = lim
                if lb == ub:
                    state.add(x, lb, err)
                    state.fix(i)
                elif lb == -np.inf and ub == np.inf:
                    state.add(x, val, err)
                elif ub == np.inf:
                    state.add(x, val, err)
                    state.set_lower_limit(i, lb)
                elif lb == -np.inf:
                    state.add(x, val, err)
                    state.set_upper_limit(i, ub)
                else:
                    state.add(x, val, err, lb, ub)
            if fix:
                state.fix(i)
        return state

    @classmethod
    def from_array_func(
        cls, fcn, start, error=None, limit=None, fix=None, name=None, **kwds
    ):
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
        pnames = name if name is not None else [f"x{i}" for i in range(npar)]
        kwds["name"] = pnames
        kwds["use_array_call"] = True
        if error is not None:
            if np.isscalar(error):
                error = np.ones(npar) * error
            else:
                if len(error) != npar:
                    raise RuntimeError(
                        "length of error sequence does " "not match start sequence"
                    )
        if limit is not None:
            if len(limit) == 2 and np.isscalar(limit[0]) and np.isscalar(limit[1]):
                limit = [limit for i in range(npar)]
            else:
                if len(limit) != npar:
                    raise RuntimeError(
                        "length of limit sequence does " "not match start sequence"
                    )
        for i, name in enumerate(pnames):
            kwds[name] = start[i]
            if error is not None:
                kwds["error_" + name] = error[i]
            if limit is not None:
                kwds["limit_" + name] = limit[i]
            if fix is not None:
                kwds["fix_" + name] = fix[i]
        return cls(fcn, **kwds)

    def migrad(
        self, ncall=None, resume=True, precision=None, iterate=5, **deprecated_kwargs
    ):
        """Run MIGRAD.

        MIGRAD is a robust minimisation algorithm which earned its reputation
        in 40+ years of almost exclusive usage in high-energy physics. How
        MIGRAD works is described in the `MINUIT paper`_.

        **Arguments:**

            * **ncall**: integer or None, optional; (approximate)
              maximum number of call before MIGRAD will stop trying. Default: None
              (indicates to use MIGRAD's internal heuristic). Note: MIGRAD may slightly
              violate this limit, because it checks the condition only after a full
              iteration of the algorithm, which usually performs several function calls.

            * **resume**: boolean indicating whether MIGRAD should resume from
              the previous minimiser attempt(True) or should start from the
              beginning(False). Default True.

            * **precision**: override Minuit precision estimate for the cost function.
              Default: None (= use epsilon of a C++ double). If the cost function has a
              lower precision (e.g. of a C++ float), setting this to a lower value will
              accelerate convergence and reduce the rate of unsuccessful convergence.

            * **iterate**: automatically call Migrad up to N times if convergence
              was not reached. Default: 5. This simple heuristic makes Migrad converge
              more often even if the numerical precision of the cost function is low.
              Setting this to 1 disables the feature.

        **Return:**

            :ref:`function-minimum-sruct`, list of :ref:`minuit-param-struct`
        """
        if ncall is None:
            ncall = 0  # tells C++ Minuit to use its internal heuristic

        if iterate < 1:
            raise ValueError("iterate must be at least 1")

        if "nsplit" in deprecated_kwargs:
            warn(
                "`nsplit` keyword has been removed and is ignored",
                RuntimeWarning,
                stacklevel=2,
            )
            del deprecated_kwargs["nsplit"]

        if deprecated_kwargs:
            raise KeyError("keyword(s) not recognized: " + " ".join(deprecated_kwargs))

        # construct new fcn and migrad if
        # it's a clean state or resume=False

        if not resume:
            self._last_state = self._init_state
            self._fmin = None
            self._fcn.nfcn = 0
            self._fcn.ngrad = 0

        migrad = MnMigrad(self._fcn, self._last_state, self.strategy)
        migrad.set_print_level(self.print_level)
        if precision is not None:
            migrad.precision = precision

        ncalls_total_before = self._fcn.nfcn
        ngrads_total_before = self._fcn.ngrad

        # Automatically call Migrad up to `iterate` times if minimum is not valid.
        # This simple heuristic makes Migrad converge more often.
        for _ in range(iterate):
            self._fmin = migrad(ncall, self.tol)
            if self._fmin.is_valid or self._fmin.has_reached_call_limit:
                break

        self._last_state = self._fmin.state
        self._ncalls = self._fcn.nfcn - ncalls_total_before
        self._ngrads = self._fcn.ngrad - ngrads_total_before

        if self.print_level > 1:
            print(self.fmin)

        return mutil.MigradResult(self.fmin, self.params)

    def hesse(self, ncall=None, **deprecated_kwargs):
        """Run HESSE to compute parabolic errors.

        HESSE estimates the covariance matrix by inverting the matrix of
        `second derivatives (Hesse matrix) at the minimum
        <http://en.wikipedia.org/wiki/Hessian_matrix>`_. This covariance
        matrix is valid if your :math:`\\chi^2` or likelihood profile looks
        like a hyperparabola around the the minimum. This is usually the case,
        especially when you fit many observations (in the limit of infinite
        samples this is always the case). If you want to know how your
        parameters are correlated, you also need to use HESSE.

        Also see :meth:`minos`, which computes the uncertainties in a
        different way.

        **Arguments:**
            - **ncall**: integer or None, limit the number of calls made by MINOS.
              Default: None (uses an internal heuristic by C++ MINUIT).

        **Returns:**

            list of :ref:`minuit-param-struct`
        """

        if "maxcall" in deprecated_kwargs:
            warn(
                "using `maxcall` keyword is deprecated, use `ncall` keyword instead",
                DeprecationWarning,
                stacklevel=2,
            )
            ncall = deprecated_kwargs.pop("maxcall")

        if deprecated_kwargs:
            raise KeyError("keyword(s) not recognized: " + " ".join(deprecated_kwargs))

        ncall = 0 if ncall is None else int(ncall)

        ncalls_total_before = self._fcn.nfcn
        ngrads_total_before = self._fcn.ngrad

        hesse = MnHesse(self.strategy)

        if self._fmin and self._fmin.state == self._last_state:
            # _last_state not modified, can update _fmin which is more efficient
            hesse(self._fcn, self._fmin, ncall)
            self._last_state = self._fmin.state
        else:
            # _fmin does not exist or _last_state was modified,
            # so we cannot just update last _fmin
            self._last_state = hesse(self._fcn, self._last_state, ncall)

        if not self._last_state.has_covariance:
            warn(
                "HESSE Failed. Covariance and GlobalCC will not be available",
                mutil.HesseFailedWarning,
            )

        self._ncalls = self.ncalls_total - ncalls_total_before
        self._ngrads = self.ngrads_total - ngrads_total_before

        if not self._fmin and not self._last_state.has_covariance:
            raise RuntimeError("HESSE Failed")

        return self.params

    def minos(self, var=None, sigma=1.0, ncall=None, **deprecated_kwargs):
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
            - **sigma**: number of :math:`\\sigma` error. Default 1.0.
            - **ncall**: integer or None, limit the number of calls made by MINOS.
              Default: None (uses an internal heuristic by C++ MINUIT).

        **Returns:**

            Dictionary of varname to :ref:`minos-error-struct`, containing
            all up to now computed errors, including the current request.

        """
        if not self._fmin:
            raise RuntimeError(
                "MINOS require function to be at the minimum." " Run MIGRAD first."
            )

        if "maxcall" in deprecated_kwargs:
            warn(
                "using `maxcall` keyword is deprecated, use `ncall` keyword instead",
                DeprecationWarning,
                stacklevel=2,
            )
            ncall = deprecated_kwargs.pop("maxcall")

        if deprecated_kwargs:
            raise KeyError("keyword(s) not recognized: " + " ".join(deprecated_kwargs))

        ncall = 0 if ncall is None else int(ncall)

        # FIXME there should be a guard for up
        oldup = self._fcn.up
        self._fcn.up = oldup * sigma * sigma
        if not self._fmin.is_valid:
            raise RuntimeError(
                ("Function minimum is not valid. Make sure " "MIGRAD converged first")
            )
        if var is not None and var not in self.pos2var:
            raise RuntimeError(f"Unknown parameter {var}")

        ncalls_total_before = self._fcn.nfcn
        ngrads_total_before = self._fcn.ngrad

        minos = MnMinos(self._fcn, self._fmin, self.strategy)

        vnames = self.pos2var if var is None else [var]
        for vname in vnames:
            if self.fixed[vname]:
                if var is not None and var == vname:
                    warn(
                        f"Cannot scan parameter {var}, it is fixed",
                        mutil.IMinuitWarning,
                    )
                    return None
                continue
            mnerror = minos(self.var2pos[vname], ncall, self.tol)
            self.merrors[vname] = minoserror2struct(vname, mnerror)

        self._ncalls = self._fcn.nfcn - ncalls_total_before
        self._ngrads = self._fcn.ngrad - ngrads_total_before

        # FIXME should be done by a guard
        self._fcn.up = oldup
        return self.merrors

    def matrix(self, correlation=False, skip_fixed=True):
        """Error or correlation matrix in tuple or tuples format."""
        if not self._last_state.has_covariance:
            raise RuntimeError(
                "Covariance is not valid. Maybe the last Hesse call failed?"
            )

        mncov = self._last_state.covariance

        # When some parameters are fixed, mncov is a sub-matrix. If skip-fixed
        # is false, we need to expand the sub-matrix back into the full form.
        # This requires a translation between sub-index und full-index.
        if skip_fixed:
            npar = sum(not mp.is_fixed for mp in self._last_state)
            ind = range(npar)

            def cov(i, j):
                return mncov[i, j]

        else:
            ext2int = {}
            iint = 0
            for mp in self._last_state:
                if not mp.is_fixed:
                    ext2int[mp.number] = iint
                    iint += 1
            ind = range(self.narg)

            def cov(i, j):
                if i not in ext2int or j not in ext2int:
                    return 0.0
                return mncov[ext2int[i], ext2int[j]]

        names = [k for (k, v) in self.fixed.items() if not (skip_fixed and v)]
        if correlation:

            def cor(i, j):
                return cov(i, j) / ((cov(i, i) * cov(j, j)) ** 0.5 + 1e-100)

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
        a = np.empty(self.narg, dtype=np.double)
        for i in range(self.narg):
            a[i] = self.errors[i]
        return a

    def np_merrors(self):
        """MINOS parameter errors in numpy array format.

        Fixed parameters are included (zeros are returned), the order follows
        :attr:`parameters`.

        The format of the produced array follows matplotlib conventions, as
        in ``matplotlib.pyplot.errorbar``. The shape is (2, N) for N
        parameters. The first row represents the downward error as a positive
        offset from the center. Likewise, the second row represents the
        upward error as a positive offset from the center.

        **Returns:**

            ``numpy.ndarray`` of shape (2, N).
        """
        # array format follows matplotlib conventions, see pyplot.errorbar
        a = np.zeros((2, self.narg))
        for me in self.merrors.values():
            i = self.var2pos[me.name]
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
        return LatexFactory.build_param_table(self.init_params, {})

    @property
    def fmin(self):
        """Current function minimum data object"""
        if self._fmin:
            return fmin2struct(
                self._fmin,
                self._fcn.up,
                self.tol,
                self._ncalls,
                self.ncalls_total,
                self._ngrads,
                self.ngrads_total,
            )

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
        return get_params(self._last_state, self.merrors)

    @property
    def init_params(self):
        """List of current parameter data objects set to the initial fit state"""
        return get_params(self._init_state, None)

    @property
    def ncalls_total(self):
        """Total number of calls to FCN (not just the last operation)"""
        return self._fcn.nfcn

    @property
    def ngrads_total(self):
        """Total number of calls to Gradient (not just the last operation)"""
        return self._fcn.ngrad

    @property
    def valid(self):
        """Check if function minimum is valid."""
        return self._fmin and self._fmin.is_valid

    @property
    def accurate(self):
        """Check if covariance (of the last MIGRAD run) is accurate."""
        return self._fmin and self._fmin.has_accurate_covar

    # Various utility functions

    def is_clean_state(self):
        """Check if minuit is in a clean state, ie. no MIGRAD call"""
        return not self._fmin

    def mnprofile(self, vname, bins=30, bound=2, subtract_min=False):
        """Calculate MINOS profile around the specified range.

        Scans over **vname** and minimises FCN over the other parameters in each point.

        **Arguments:**

            * **vname** name of variable to scan

            * **bins** number of scanning bins. Default 30.

            * **bound**
              If bound is tuple, (left, right) scanning bound.
              If bound is a number, it specifies how many :math:`\\sigma`
              symmetrically from minimum (minimum+- bound* :math:`\\sigma`).
              Default 2

            * **subtract_min** subtract_minimum off from return value. This
              makes it easy to label confidence interval. Default False.

        **Returns:**

            bins(center point), value, MIGRAD results
        """
        if vname not in self.pos2var:
            raise ValueError("Unknown parameter %s" % vname)

        bound = self._normalize_bound(vname, bound)
        values = np.linspace(bound[0], bound[1], bins, dtype=np.double)
        results = np.empty(bins, dtype=np.double)
        status = np.empty(bins, dtype=np.bool)
        state = MnUserParameterState(self._last_state)  # copy
        ipar = self.var2pos[vname]
        state.fix(ipar)
        for i, v in enumerate(values):
            state.set_value(ipar, v)
            migrad = MnMigrad(self._fcn, state, self.strategy)
            fm = migrad(0, self.tol)
            if not fm.is_valid:
                warn(
                    "MIGRAD fails to converge for %s=%f" % (vname, v),
                    mutil.IMinuitWarning,
                )
            status[i] = fm.is_valid
            results[i] = fm.fval
        vmin = np.min(results)

        if subtract_min:
            results -= vmin

        return values, results, status

    def draw_mnprofile(
        self, vname, bins=30, bound=2, subtract_min=False, band=True, text=True
    ):
        """Draw MINOS profile in the specified range.

        It is obtained by finding MIGRAD results with **vname** fixed
        at various places within **bound**.

        **Arguments:**

            * **vname** variable name to scan

            * **bins** number of scanning bin. Default 30.

            * **bound**
              If bound is tuple, (left, right) scanning bound.
              If bound is a number, it specifies how many :math:`\\sigma`
              symmetrically from minimum (minimum+- bound* :math:`\\sigma`).
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

        .. plot:: plots/mnprofile.py
            :include-source:
        """
        x, y, s = self.mnprofile(vname, bins, bound, subtract_min)
        return _minuit_methods.draw_profile(self, vname, x, y, band, text)

    def profile(
        self, vname, bins=100, bound=2, subtract_min=False, **deprecated_kwargs
    ):
        """Calculate cost function profile around specify range.

        **Arguments:**

            * **vname** variable name to scan

            * **bins** number of scanning bin. Default 100.

            * **bound**
              If bound is tuple, (left, right) scanning bound.
              If bound is a number, it specifies how many :math:`\\sigma`
              symmetrically from minimum (minimum+- bound* :math:`\\sigma`).
              Default: 2.

            * **subtract_min** subtract_minimum off from return value. This
              makes it easy to label confidence interval. Default False.

        **Returns:**

            bins(center point), value

        .. seealso::

            :meth:`mnprofile`
        """
        if "args" in deprecated_kwargs:
            warn("The args keyword has been removed.", DeprecationWarning, stacklevel=2)
            del deprecated_kwargs["args"]

        if len(deprecated_kwargs):
            raise ValueError("Unrecognized keywords: " + " ".join(deprecated_kwargs))

        if subtract_min and not self._fmin:
            raise RuntimeError(
                "Request for minimization "
                "subtraction but no minimization has been done. "
                "Run MIGRAD first."
            )

        bound = self._normalize_bound(vname, bound)
        return _minuit_methods.profile(self, vname, bins, bound, subtract_min)

    def draw_profile(
        self,
        vname,
        bins=100,
        bound=2,
        subtract_min=False,
        band=True,
        text=True,
        **deprecated_kwargs,
    ):
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

        if "args" in deprecated_kwargs:
            warn("The args keyword has been removed.", DeprecationWarning, stacklevel=2)
            del deprecated_kwargs["args"]

        if len(deprecated_kwargs):
            raise ValueError("Unrecognized keywords: " + " ".join(deprecated_kwargs))

        bound = self._normalize_bound(vname, bound)
        x, y = self.profile(vname, bins, bound, subtract_min)
        return _minuit_methods.draw_profile(self, vname, x, y, band, text)

    def contour(self, x, y, bins=50, bound=2, subtract_min=False, **deprecated_kwargs):
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
              If bound is a number, it specifies how many :math:`\\sigma`
              symmetrically from minimum (minimum+- bound*:math:`\\sigma`).
              Default: 2.

            - **subtract_min** Subtract minimum off from return values. Default False.

        **Returns:**

            x_bins, y_bins, values

            values[y, x] <-- this choice is so that you can pass it
            to through matplotlib contour()

        .. seealso::

            :meth:`mncontour`
            :meth:`mnprofile`

        """

        if "args" in deprecated_kwargs:
            warn("The args keyword has been removed.", DeprecationWarning, stacklevel=2)
            del deprecated_kwargs["args"]

        if len(deprecated_kwargs):
            raise ValueError("Unrecognized keywords: " + " ".join(deprecated_kwargs))

        if subtract_min and not self._fmin:
            raise RuntimeError(
                "Request for minimization "
                "subtraction but no minimization has been done. "
                "Run MIGRAD first."
            )

        try:
            n = float(bound)
            x_bound = self._normalize_bound(x, n)
            y_bound = self._normalize_bound(y, n)
        except ValueError:
            x_bound = self._normalize_bound(x, bound[0])
            y_bound = self._normalize_bound(y, bound[1])

        x_val = np.linspace(x_bound[0], x_bound[1], bins)
        y_val = np.linspace(y_bound[0], y_bound[1], bins)

        x_pos = self.var2pos[x]
        y_pos = self.var2pos[y]

        arg = list(self.args)

        result = np.empty((bins, bins), dtype=np.double)
        varg = np.array(arg, dtype=np.double)
        for i, x in enumerate(x_val):
            varg[x_pos] = x
            for j, y in enumerate(y_val):
                varg[y_pos] = y
                result[i, j] = self._fcn(varg)

        if subtract_min:
            result -= self._fmin.fval

        return x_val, y_val, result

    def mncontour(self, x, y, numpoints=100, sigma=1.0):
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
        if not self._fmin:
            raise ValueError("Run MIGRAD first")

        ix = self.var2pos[x]
        iy = self.var2pos[y]

        vary = self._free_parameters()
        if x not in vary or y not in vary:
            raise ValueError("mncontour has to be run on vary parameters.")

        # FIXME this should be done with a guard
        oldup = self._fcn.up
        self._fcn.up = oldup * sigma * sigma

        mnc = MnContours(self._fcn, self._fmin, self.strategy)
        mex, mey, ce = mnc(ix, iy, numpoints)

        self._fcn.up = oldup

        return mex, mey, ce

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

        .. plot:: plots/mncontour.py
            :include-source:
        """
        return _minuit_methods.draw_mncontour(self, x, y, nsigma, numpoints)

    def draw_contour(self, x, y, bins=50, bound=2, **deprecated_kwargs):
        """Convenience wrapper for drawing contours.

        The arguments are the same as :meth:`contour`.

        Please read the docs of :meth:`contour` and :meth:`mncontour` to understand the
        difference between the two.

        .. seealso::

            :meth:`contour`
            :meth:`draw_mncontour`

        """
        if "show_sigma" in deprecated_kwargs:
            warn(
                "The show_sigma keyword has been removed due to potential confusion. "
                "Use draw_mncontour to draw sigma contours.",
                DeprecationWarning,
                stacklevel=2,
            )
            del deprecated_kwargs["show_sigma"]
        if "args" in deprecated_kwargs:
            warn(
                "The args keyword is unused and has been removed.",
                DeprecationWarning,
                stacklevel=2,
            )
            del deprecated_kwargs["args"]
        if len(deprecated_kwargs):
            raise ValueError("Invalid keyword(s): " + " ".join(deprecated_kwargs))

        return _minuit_methods.draw_contour(self, x, y, bins, bound)

    def _free_parameters(self):
        return (mp.name for mp in self._last_state if not mp.is_fixed)

    def _normalize_bound(self, vname, bound):
        try:
            n = float(bound)
            if not self.accurate:
                warn(
                    "Specified nsigma bound, but error matrix is not accurate",
                    mutil.IMinuitWarning,
                )
            start = self.values[vname]
            sigma = self.errors[vname]
            bound = (start - n * sigma, start + n * sigma)
        except ValueError:
            pass
        return bound

    # All deprecated stuff goes through __getattr__, which is only called if
    # normal attribute access returns AttributeError.
    # Warning: that this will hide AttributeErrors inside methods.
    # def __getattr__(self, key):
    #     return _minuit_methods.deprecated(self, key)
