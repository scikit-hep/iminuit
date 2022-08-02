"""Minuit class."""

import warnings
from . import util as mutil
from ._core import (
    FCN,
    MnContours,
    MnHesse,
    MnMachinePrecision,
    MnMigrad,
    MnMinos,
    MnPrint,
    MnSimplex,
    MnStrategy,
    MnUserParameterState,
    FunctionMinimum,
)
import numpy as np
import typing as _tp

# Better use numpy.typing.ArrayLike in the future, but this
# requires dropping Python-3.6 support
_ArrayLike = _tp.Collection

MnPrint.global_level = 0

__all__ = ["Minuit"]


class Minuit:
    """Function minimizer and error computer."""

    __slots__ = (
        "_fcn",
        "_strategy",
        "_tolerance",
        "_precision",
        "_values",
        "_errors",
        "_merrors",
        "_fixed",
        "_limits",
        "_fmin",
        "_covariance",
        "_var2pos",
        "_pos2var",
        "_init_state",
        "_last_state",
    )

    LEAST_SQUARES = 1.0
    """Set errordef to this for a least-squares cost function."""  # pragma: nocover

    LIKELIHOOD = 0.5
    """Set errordef to this for a negated log-likelihood function."""  # pragma: nocover

    @property
    def fcn(self) -> FCN:
        """Get cost function (usually a least-squares or likelihood function)."""
        return self._fcn

    @property
    def grad(self) -> _tp.Callable[[np.ndarray], np.ndarray]:
        """Get gradient function of the cost function."""
        return self._fcn.gradient  # type:ignore

    @property
    def pos2var(self) -> _tp.Tuple[str, ...]:
        """Map variable index to name."""
        return self._pos2var

    @property
    def var2pos(self) -> _tp.Dict[str, int]:
        """Map variable name to index."""
        return self._var2pos

    @property
    def parameters(self) -> _tp.Tuple[str, ...]:
        """
        Get tuple of parameter names.

        This is an alias for :attr:`pos2var`.
        """
        return self._pos2var

    @property
    def errordef(self) -> float:
        """
        Access FCN increment above the minimum that corresponds to one standard deviation.

        Default value is 1.0. `errordef` should be 1.0 for a least-squares cost
        function and 0.5 for a negative log-likelihood function. See section 1.5.1 on page
        6 of the :download:`MINUIT2 User's Guide <mnusersguide.pdf>`. This parameter is
        also called *UP* in MINUIT documents.

        To make user code more readable, we provided two named constants::

            m_lsq = Minuit(a_least_squares_function)
            m_lsq.errordef = Minuit.LEAST_SQUARES  # == 1

            m_nll = Minuit(a_likelihood_function)
            m_nll.errordef = Minuit.LIKELIHOOD     # == 0.5
        """
        return self._fcn._errordef  # type: ignore

    @errordef.setter
    def errordef(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"errordef={value} must be a positive number")
        self._fcn._errordef = value
        if self._fmin:
            self._fmin._src.errordef = value

    @property
    def precision(self) -> _tp.Optional[float]:
        """
        Access estimated precision of the cost function.

        Default: None. If set to None, Minuit assumes the cost function is computed in
        double precision. If the precision of the cost function is lower (because it
        computes in single precision, for example) set this to some multiple of the
        smallest relative change of a parameter that still changes the function.
        """
        return self._precision

    @precision.setter
    def precision(self, value: _tp.Optional[float]) -> None:
        if value is not None and not (value > 0):
            raise ValueError("precision must be a positive number or None")
        self._precision = value

    @property
    def tol(self) -> float:
        """
        Access tolerance for convergence with the EDM criterion.

        Minuit detects converge with the EDM criterion. EDM stands for *Estimated
        Distance to Minimum*, it is mathematically described in the `MINUIT paper`_.
        The EDM criterion is well suited for statistical cost functions,
        since it stops the minimization when parameter improvements become small
        compared to parameter uncertainties.

        The convergence is detected when `edm < edm_max`, where `edm_max` is calculated
        as

            * Migrad: edm_max = 0.002 * tol * errordef
            * Simplex: edm_max = tol * errordef

        Users can set `tol` (default: 0.1) to a different value to either speed up
        convergence at the cost of a larger error on the fitted parameters and possibly
        invalid estimates for parameter uncertainties or smaller values to get more
        accurate parameter values, although this should never be necessary as the
        default is fine.

        If the tolerance is set to a very small value or zero, Minuit will use an
        internal lower limit for the tolerance. To restore the default use, one can
        assign `None`.

        Under some circumstances, Migrad is allowed to violate edm_max by a factor of
        10. Users should not try to detect convergence by comparing edm with edm_max,
        but query :attr:`iminuit.util.FMin.is_above_max_edm`.
        """
        return self._tolerance

    @tol.setter
    def tol(self, value: _tp.Optional[float]) -> None:
        if value is None:  # used to reset tolerance
            value = 0.1
        elif value < 0:
            raise ValueError("tolerance must be non-negative")
        self._tolerance = value

    @property
    def strategy(self) -> MnStrategy:
        """
        Access current minimization strategy.

        You can assign an integer:

        - 0: Fast. Does not check a user-provided gradient. Does not improve Hesse matrix
          at minimum. Extra call to :meth:`hesse` after :meth:`migrad` is always needed
          for good error estimates. If you pass a user-provided gradient to MINUIT,
          convergence is **faster**.
        - 1: Default. Checks user-provided gradient against numerical gradient. Checks and
          usually improves Hesse matrix at minimum. Extra call to :meth:`hesse` after
          :meth:`migrad` is usually superfluous. If you pass a user-provided gradient to
          MINUIT, convergence is **slower**.
        - 2: Careful. Like 1, but does extra checks of intermediate Hessian matrix during
          minimization. The effect in benchmarks is a somewhat improved accuracy at the
          cost of more function evaluations. A similar effect can be achieved by reducing
          the tolerance :attr:`tol` for convergence at any strategy level.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, value: int) -> None:
        self._strategy.strategy = value

    @property
    def print_level(self) -> int:
        """
        Access current print level.

        You can assign an integer:

        - 0: quiet (default)
        - 1: print minimal debug messages to terminal
        - 2: print more debug messages to terminal
        - 3: print even more debug messages to terminal

        Warnings
        --------
        Setting print_level has the unwanted side-effect of setting the level
        globally for all Minuit instances in the current Python session.
        """
        return MnPrint.global_level  # type: ignore

    @print_level.setter
    def print_level(self, level: int) -> None:
        MnPrint.global_level = level

    @property
    def throw_nan(self) -> bool:
        """
        Access whether to raise runtime error if the function evaluates to NaN.

        If you set this to True, an error is raised whenever the function evaluates
        to NaN.
        """
        return self._fcn._throw_nan  # type: ignore

    @throw_nan.setter
    def throw_nan(self, value: bool) -> None:
        self._fcn._throw_nan = value

    @property
    def values(self) -> mutil.ValueView:
        """Access parameter values via an array-like view.

        Use to read or write current parameter values based on the parameter index
        or the parameter name as a string. If you change a parameter value and run
        :meth:`migrad`, the minimization will start from that value, similar for
        :meth:`hesse` and :meth:`minos`.

        See Also
        --------
        errors, fixed, limits
        """
        return self._values

    @values.setter
    def values(self, args: _tp.Iterable) -> None:
        self._values[:] = args

    @property
    def errors(self) -> mutil.ErrorView:
        """Access parameter parabolic errors via an array-like view.

        Like :attr:`values`, but instead of reading or writing the values, you read
        or write the errors (which double as step sizes for MINUITs numerical gradient
        estimation). Only positive values are accepted when assigning to errors.

        See Also
        --------
        values, fixed, limits
        """
        return self._errors

    @errors.setter
    def errors(self, args: _tp.Iterable) -> None:
        self._errors[:] = args

    @property
    def fixed(self) -> mutil.FixedView:
        """Access whether parameters are fixed via an array-like view.

        Use to read or write the fixation state of a parameter based on the parameter
        index or the parameter name as a string. If you change the state and run
        :meth:`migrad`, :meth:`hesse`, or :meth:`minos`, the new state is used.

        In case of complex fits, it can help to fix some parameters first and only
        minimize the function with respect to the other parameters, then release the
        fixed parameters and minimize again starting from that state.

        See Also
        --------
        values, errors, limits
        """
        return self._fixed

    @fixed.setter
    def fixed(self, args: _tp.Iterable) -> None:
        self._fixed[:] = args

    @property
    def limits(self) -> mutil.LimitView:
        """Access parameter limits via a array-like view.

        Use to read or write the limits of a parameter based on the parameter index
        or the parameter name as a string. If you change the limits and run
        :meth:`migrad`, :meth:`hesse`, or :meth:`minos`, the new state is used.

        In case of complex fits, it can help to limit some parameters first, run Migrad,
        then remove the limits and run Migrad again. Limits will bias the result only if
        the best fit value is outside the limits, not if it is inside. Limits will affect
        the estimated Hesse uncertainties if the parameter is close to a limit. They do
        not affect the Minos uncertainties, because those are invariant to
        transformations and limits are implemented via a variable transformation.

        See Also
        --------
        values, errors, fixed
        """
        return self._limits

    @limits.setter
    def limits(self, args: _tp.Iterable) -> None:
        self._limits[:] = args

    @property
    def merrors(self) -> mutil.MErrors:
        """
        Return a dict-like with Minos data objects.

        The Minos data objects contain the full status information of the Minos run.

        See Also
        --------
        util.MError
        util.MErrors
        """
        return self._merrors

    @property
    def covariance(self) -> _tp.Optional[mutil.Matrix]:
        r"""
        Return covariance matrix.

        The square-root of the diagonal elements of the covariance matrix correspond to
        a standard deviation for each parameter with 68 % coverage probability in the
        asymptotic limit (large samples). To get k standard deviations, multiply the
        covariance matrix with k^2.

        The submatrix formed by two parameters describes an ellipse. The asymptotic
        coverage probabilty of the standard ellipse is lower than 68 %. It can be computed
        from the :math:`\chi^2` distribution with 2 degrees of freedom. In general, to
        obtain a (hyper-)ellipsoid with coverage probability CL, one has to multiply the
        submatrix of the corresponding k parameters with a factor. For k = 1,2,3 and
        CL = 0.99 ::

            from scipy.stats import chi2

            chi2(1).ppf(0.99) # 6.63...
            chi2(2).ppf(0.99) # 9.21...
            chi2(3).ppf(0.99) # 11.3...

        See Also
        --------
        util.Matrix
        """
        return self._covariance

    @property
    def npar(self) -> int:
        """Get number of parameters."""
        return len(self._last_state)

    @property
    def nfit(self) -> int:
        """Get number of fitted parameters (fixed parameters not counted)."""
        return self.npar - sum(self.fixed)

    @property
    def ndof(self) -> int:
        """
        Get number of degrees of freedom if cost function supports this.

        To support this feature, the cost function has to report the number of data
        points with a property called ``ndata``. Unbinned cost functions should return
        infinity.
        """
        return self._fcn._ndata() - self.nfit  # type: ignore

    @property
    def fmin(self) -> _tp.Optional[mutil.FMin]:
        """
        Get function minimum data object.

        See Also
        --------
        util.FMin
        """
        return self._fmin

    @property
    def fval(self) -> _tp.Optional[float]:
        """
        Get function value at minimum.

        This is an alias for :attr:`iminuit.util.FMin.fval`.

        See Also
        --------
        util.FMin
        """
        fm = self._fmin
        return fm.fval if fm else None

    @property
    def params(self) -> mutil.Params:
        """
        Get list of current parameter data objects.

        See Also
        --------
        init_params, util.Params
        """
        return _get_params(self._last_state, self._merrors)

    @property
    def init_params(self) -> mutil.Params:
        """
        Get list of current parameter data objects set to the initial fit state.

        See Also
        --------
        params, util.Params
        """
        return _get_params(self._init_state, mutil.MErrors())

    @property
    def valid(self) -> bool:
        """
        Return True if the function minimum is valid.

        This is an alias for :attr:`iminuit.util.FMin.is_valid`.

        See Also
        --------
        util.FMin
        """
        return self._fmin.is_valid if self._fmin else False

    @property
    def accurate(self) -> bool:
        """
        Return True if the covariance matrix is accurate.

        This is an alias for :attr:`iminuit.util.FMin.has_accurate_covar`.

        See Also
        --------
        util.FMin
        """
        return self._fmin.has_accurate_covar if self._fmin else False

    @property
    def nfcn(self) -> int:
        """Get total number of function calls."""
        return self._fcn._nfcn  # type:ignore

    @property
    def ngrad(self) -> int:
        """Get total number of gradient calls."""
        return self._fcn._ngrad  # type:ignore

    def __init__(
        self,
        fcn: _tp.Callable,
        *args: _tp.Union[float, _ArrayLike[float]],
        grad: _tp.Optional[_tp.Callable] = None,
        name: _tp.Optional[_tp.Collection[str]] = None,
        **kwds: float,
    ):
        """
        Initialize Minuit object.

        This does not start the minimization or perform any other work yet. Algorithms
        are started by calling the corresponding methods.

        Parameters
        ----------
        fcn :
            Function to minimize. See notes for details on what kind of functions are
            accepted.
        *args :
            Starting values for the minimization as positional arguments.
            See notes for details on how to set starting values.
        grad :
            Function that calculates the gradient and returns an iterable object with one
            entry for each parameter, which is the derivative for that parameter.
            If None (default), Minuit will calculate the gradient numerically.
        name :
            If it is set, it overrides iminuit's function signature detection.
        **kwds :
            Starting values for the minimization as keyword arguments.
            See notes for details on how to set starting values.

        Notes
        -----
        *Callables*

        By default, Minuit assumes that the callable `fcn` behaves like chi-square
        function, meaning that the function minimum in repeated identical random
        experiments is chi-square distributed up to an arbitrary additive constant. This
        is important for the correct error calculation. If `fcn` returns a log-likelihood,
        one should multiply the result with -2 to adapt it. If the function returns the
        negated log-likelihood, one can alternatively set the attribute
        `fcn.errordef` = :attr:`Minuit.LIKELIHOOD` or
        :attr:`Minuit.errordef` = :attr:`Minuit.LIKELIHOOD` after initialization to make
        Minuit calculate errors properly.

        Minuit reads the function signature of `fcn` to detect the number and names of the
        function parameters. Two kinds of function signatures are understood.

        a) Function with positional arguments.

            The function has positional arguments, one for each fit
            parameter. Example::

                def fcn(a, b, c): ...

            The parameters a, b, c must accept a real number.

            iminuit automatically detects the parameters names in this case.
            More information about how the function signature is detected can
            be found in :func:`iminuit.util.describe`.

        b) Function with arguments passed as a single Numpy array.

            The function has a single argument which is a Numpy array.
            Example::

                def fcn_np(x): ...

            To use this form, starting values need to be passed to Minuit in form as
            an array-like type, e.g. a numpy array, tuple or list. For more details,
            see "Parameter Keyword Arguments" further down.

        In some cases, the detection fails, for example, for a function like this::

                def difficult_fcn(*args): ...

        To use such a function, set the `name` keyword as described further below.

        *Parameter initialization*

        Initial values for the minimization can be set with positional arguments or
        via keywords. This is best explained through an example::

            def fcn(x, y):
                return (x - 2) ** 2 + (y - 3) ** 2

        The following ways of passing starting values are equivalent::

            Minuit(fcn, x=1, y=2)
            Minuit(fcn, y=2, x=1) # order is irrelevant when keywords are used ...
            Minuit(fcn, 1, 2)     # ... but here the order matters

        Positional arguments can also be used if the function has no signature::

            def fcn_no_sig(*args):
                # ...

            Minuit(fcn_no_sig, 1, 2)

        If the arguments are explicitly named with the `name` keyword described
        further below, keywords can be used for initialization::

            Minuit(fcn_no_sig, x=1, y=2, name=("x", "y"))  # this also works

        If the function accepts a single Numpy array, then the initial values
        must be passed as a single array-like object::

            def fcn_np(x):
                return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

            Minuit(fcn_np, (1, 2))

        Setting the values with keywords is not possible in this case. Minuit
        deduces the number of parameters from the length of the initialization
        sequence.

        See Also
        --------
        migrad, hesse, minos, scan, simplex
        """
        array_call = False
        if len(args) == 1 and isinstance(args[0], _tp.Collection):
            array_call = True
            start = np.array(args[0])
        else:
            start = np.array(args)
        del args

        if name is None:
            name = mutil.describe(fcn)
            if len(name) == 0 or (array_call and len(name) == 1):
                name = tuple(f"x{i}" for i in range(len(start)))

        if len(start) == 0 and len(kwds) == 0:
            raise RuntimeError(
                "starting value(s) are required"
                + (f" for {' '.join(name)}" if name else "")
            )

        # Maintain two dictionaries to easily convert between
        # parameter names and position
        self._pos2var = tuple(name)
        self._var2pos = {k: i for i, k in enumerate(name)}

        # set self.tol to default value
        self.tol = None  # type:ignore
        self._strategy = MnStrategy(1)
        self._fcn = FCN(
            fcn,
            getattr(fcn, "grad", grad),
            array_call,
            getattr(fcn, "errordef", 1.0),
        )

        self._init_state = _make_init_state(self._pos2var, start, kwds)
        self._values = mutil.ValueView(self)
        self._errors = mutil.ErrorView(self)
        self._fixed = mutil.FixedView(self)
        self._limits = mutil.LimitView(self)

        self.precision = getattr(fcn, "precision", None)

        self.reset()

    def reset(self) -> "Minuit":  # requires from __future__ import annotations
        """
        Reset minimization state to initial state.

        Leaves :attr:`strategy`, :attr:`precision`, :attr:`tol`, :attr:`errordef`,
        :attr:`print_level` unchanged.
        """
        self._last_state = self._init_state
        self._fmin: _tp.Optional[mutil.FMin] = None
        self._fcn._nfcn = 0
        self._fcn._ngrad = 0
        self._merrors = mutil.MErrors()
        self._covariance: _tp.Optional[mutil.Matrix] = None
        return self  # return self for method chaining and to autodisplay current state

    def migrad(self, ncall: _tp.Optional[int] = None, iterate: int = 5) -> "Minuit":
        """
        Run Migrad minimization.

        Migrad from the Minuit2 library is a robust minimisation algorithm which earned
        its reputation in 40+ years of almost exclusive usage in high-energy physics. How
        Migrad works is described in the `Minuit paper`_. It uses first and approximate
        second derivatives to achieve quadratic convergence near the minimum.

        Parameters
        ----------
        ncall :
            Approximate maximum number of calls before minimization will be aborted.
            If set to None, use the adaptive heuristic from the Minuit2 library
            (Default: None). Note: The limit may be slightly violated, because the
            condition is checked only after a full iteration of the algorithm, which
            usually performs several function calls.

        iterate :
            Automatically call Migrad up to N times if convergence was not reached
            (Default: 5). This simple heuristic makes Migrad converge more often even if
            the numerical precision of the cost function is low. Setting this to 1
            disables the feature.

        See Also
        --------
        simplex, scan
        """
        if ncall is None:
            ncall = 0  # tells C++ Minuit to use its internal heuristic

        if iterate < 1:
            raise ValueError("iterate must be at least 1")

        migrad = MnMigrad(self._fcn, self._last_state, self.strategy)

        t = mutil._Timer(self._fmin)
        with t:
            # Automatically call Migrad up to `iterate` times if minimum is not valid.
            # This simple heuristic makes Migrad converge more often.
            for _ in range(iterate):
                # workaround: precision must be set again after each call to MnMigrad
                if self._precision is not None:
                    migrad.precision = self._precision
                fm = migrad(ncall, self._tolerance)
                if fm.is_valid or fm.has_reached_call_limit:
                    break

        self._last_state = fm.state

        self._fmin = mutil.FMin(
            fm,
            "Migrad",
            self.nfcn,
            self.ngrad,
            self.ndof,
            self._edm_goal(migrad_factor=True),
            t.value,
        )
        self._make_covariance()

        return self  # return self for method chaining and to autodisplay current state

    def simplex(self, ncall: _tp.Optional[int] = None) -> "Minuit":
        """
        Run Simplex minimization.

        Simplex from the Minuit2 C++ library is a variant of the Nelder-Mead algorithm to
        find the minimum of a function. It does not make use of derivatives.
        `The Wikipedia has a good article on the Nelder-Mead method
        <https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method>`_.

        Parameters
        ----------
        ncall :
            Approximate maximum number of calls before minimization will be aborted.
            If set to None, use the adaptive heuristic from the Minuit2 library
            (Default: None). Note: The limit may be slightly violated, because the
            condition is checked only after a full iteration of the algorithm, which
            usually performs several function calls.

        Notes
        -----
        The Simplex method usually converges more slowly than Migrad, but performs better
        in certain cases, the Rosenbrock function is a notable example. Unlike Migrad, the
        Simplex method does not have quadratic convergence near the minimum, so it is a
        good approach to run Migrad after Simplex to obtain an accurate solution in fewer
        steps. Simplex may also be useful to get close to the minimum from an unsuitable
        starting point.

        The convergence criterion for Simplex is also based on EDM, but the threshold
        is much more lax than that of Migrad (see :attr:`Minuit.tol` for details).
        This was made so that Simplex stops early when getting near the minimum, to give
        the user a chance to switch to the more efficient Migrad algorithm to finish the
        minimization. Early stopping can be avoided by setting Minuit.tol to an
        accordingly smaller value, however.
        """
        if ncall is None:
            ncall = 0  # tells C++ Minuit to use its internal heuristic

        simplex = MnSimplex(self._fcn, self._last_state, self.strategy)
        if self._precision is not None:
            simplex.precision = self._precision

        t = mutil._Timer(self._fmin)
        with t:
            fm = simplex(ncall, self._tolerance)
        self._last_state = fm.state

        self._fmin = mutil.FMin(
            fm,
            "Simplex",
            self.nfcn,
            self.ngrad,
            self.ndof,
            self._edm_goal(),
            t.value,
        )
        self._covariance = None
        self._merrors = mutil.MErrors()

        return self  # return self for method chaining and to autodisplay current state

    def scan(self, ncall: _tp.Optional[int] = None) -> "Minuit":
        """
        Brute-force minimization.

        Scans the function on a regular hypercube grid, whose bounds are defined either
        by parameter limits if present or by Minuit.values +/- Minuit.errors.
        Minuit.errors are initialized to very small values by default, too small for this
        scan. They should be increased before running scan or limits should be set. The
        scan evaluates the function exactly at the limit boundary, so the function should
        be defined there.

        Parameters
        ----------
        ncall :
            Approximate number of function calls to spend on the scan. The
            actual number will be close to this, the scan uses ncall^(1/npar) steps per
            cube dimension. If no value is given, a heuristic is used to set ncall.

        Notes
        -----
        The scan can return an invalid minimum, this is not a cause for alarm. It just
        minimizes the cost function, the EDM value is only computed after the scan found
        a best point. If the best point still has a bad EDM value, the minimum is
        considered invalid. But even if it is considered valid, it is probably not
        accurate, since the tolerance is very lax. One should always run :meth:`migrad`
        after the scan.

        This implementation here does a full scan of the hypercube in Python. Originally,
        this was supposed to use MnScan from C++ Minuit2, but MnScan is unsuitable.
        It does a 1D scan with 41 steps (not configurable) for each parameter in sequence,
        so it is not actually scanning the full hypercube. It first scans one parameter,
        then starts the scan of the second parameter from the best value of the first and
        so on. This fails easily when the parameters are correlated.
        """
        # Implementation notes:
        # Returning a valid FunctionMinimum object was a major challenge, because C++
        # Minuit2 does not allow one to initialize data objects with data, it forces one
        # to go through algorithm objects. Because of that design, the Minuit2 C++
        # interface forces one to compute the gradient and second derivatives for the
        # starting values, even though these are not used in a scan. We turn a
        # disadvantage into an advantage here by tricking Minuit2 into computing updates
        # of the step sizes and to estimate the EDM value.

        # Running MnScan would look like this:
        #  scan = MnScan(self._fcn, self._last_state, self.strategy)
        #  fm = scan(0, 0)  # args are ignored
        #  self._last_state = fm.state
        #  self._fmin = mutil.FMin(fm, self._fcn.nfcn, self._fcn.ngrad, self._tolerance)

        n = self.nfit
        if ncall is None:
            ncall = self._migrad_maxcall()
        nstep = int(ncall ** (1 / n))

        if self._last_state == self._init_state:
            # avoid overriding initial state
            self._last_state = MnUserParameterState(self._last_state)

        x = np.empty(self.npar + 1)
        x[self.npar] = np.inf
        lims = list(self.limits)
        for i, (low, up) in enumerate(lims):
            v = self.values[i]
            e = self.errors[i]
            if self.fixed[i]:
                lims[i] = v, v
            else:
                lims[i] = (
                    v - e if low == -np.inf else low,
                    v + e if up == np.inf else up,
                )

        def run(ipar: int) -> None:
            if ipar == self.npar:
                r = self.fcn(x[: self.npar])
                if r < x[self.npar]:
                    x[self.npar] = r
                    self.values = x[: self.npar]
                return
            low, up = lims[ipar]
            if low == up:
                x[ipar] = low
                run(ipar + 1)
            else:
                for xi in np.linspace(low, up, nstep):
                    x[ipar] = xi
                    run(ipar + 1)

        t = mutil._Timer(self._fmin)
        with t:
            run(0)

        edm_goal = self._edm_goal()
        fm = FunctionMinimum(self._fcn, self._last_state, self.strategy, edm_goal)
        self._last_state = fm.state
        self._fmin = mutil.FMin(
            fm,
            "Scan",
            self.nfcn,
            self.ngrad,
            self.ndof,
            edm_goal,
            t.value,
        )
        self._covariance = None
        self._merrors = mutil.MErrors()

        return self  # return self for method chaining and to autodisplay current state

    def scipy(
        self,
        method: _tp.Optional[_tp.Union[str, _tp.Callable]] = None,
        ncall: _tp.Optional[int] = None,
        hess: _tp.Optional[_tp.Any] = None,
        hessp: _tp.Optional[_tp.Any] = None,
        constraints: _tp.Optional[_tp.Iterable[_tp.Any]] = None,
    ) -> "Minuit":
        """
        Minimize with SciPy algorithms.

        Parameters
        ----------
        method : str or Callable, optional
            Which scipy method to use.
        ncall : int, optional
            Function call limit.
        hess : Callable, optional
            Function that computes the Hessian matrix. It must use the exact same
            calling conversion as the original fcn (several arguments which are numbers
            or a single array argument).
        hessp : Callable, optional
            Function that computes the product of the Hessian matrix with a vector.
            It must use the same calling conversion as the original fcn (several
            arguments which are numbers or a single array argument) and end with another
            argument which is an arbitrary vector.
        constraints : scipy.optimize.LinearConstraint or
                      scipy.optimize.NonlinearConstraint, optional
            Linear or non-linear constraints, see docs of :func:`scipy.optimize.minimize`
            look for the `constraints` parameter. The function used in the constraint
            must use the exact same calling convention as the original fcn, see hess
            parameter for details. No parameters may be omitted in the signature, even
            if those parameters are not used in the constraint.

        Notes
        -----
        The call limit may be violated since many algorithms checks the call limit only
        after a full iteraction of their algorithm, which consists of several function
        calls. Some algorithms do not check the number of function calls at all, in this
        case the call limit acts on the number of iterations of the algorithm. This
        issue should be fixed in scipy.

        The SciPy minimizers use their own internal rule for convergence. The EDM
        criterion is evaluated only after the original algorithm already stopped. This
        means that usually SciPy minimizers will use more iterations than Migrad and
        the tolerance :attr:`tol` has no effect on SciPy minimizers.
        """
        try:
            from scipy.optimize import (
                minimize,
                Bounds,
                NonlinearConstraint,
                LinearConstraint,
            )
        except ImportError as exc:
            exc.msg += "\n\nPlease install scipy to use scipy minimizers in iminuit."
            raise

        if ncall is None:
            ncall = self._migrad_maxcall()

        cfree = ~np.array(self.fixed[:], dtype=bool)
        cpar = np.array(self.values[:])
        no_fixed_parameters = self.nfit == self.npar

        if no_fixed_parameters:

            class Wrapped:
                __slots__ = ("fcn",)

                def __init__(self, fcn):
                    self.fcn = fcn

                if self.fcn._array_call:

                    def __call__(self, par):
                        return self.fcn(par)

                else:

                    def __call__(self, par):
                        return self.fcn(*par)

            WrappedGrad = Wrapped
            WrappedHess = Wrapped

            class WrappedHessp:
                __slots__ = ("fcn",)

                def __init__(self, fcn):
                    self.fcn = fcn

                if self.fcn._array_call:

                    def __call__(self, par, v):
                        return self.fcn(par, v)

                else:

                    def __call__(self, par, v):
                        return self.fcn(*par, v)

        else:

            class Wrapped:  # type:ignore
                __slots__ = ("fcn", "free", "par")

                def __init__(self, fcn):
                    self.fcn = fcn
                    self.free = cfree
                    self.par = cpar

                if self.fcn._array_call:

                    def __call__(self, par):
                        self.par[self.free] = par
                        return self.fcn(self.par)

                else:

                    def __call__(self, par):
                        self.par[self.free] = par
                        return self.fcn(*self.par)

            class WrappedGrad(Wrapped):  # type:ignore
                def __call__(self, par):
                    g = super().__call__(par)
                    return np.atleast_1d(g)[self.free]

            class WrappedHess(Wrapped):  # type:ignore
                def __init__(self, fcn):
                    super().__init__(fcn)
                    self.freem = np.outer(self.free, self.free)
                    n = np.sum(self.free)
                    self.shape = n, n

                def __call__(self, par):
                    h = super().__call__(par)
                    return np.atleast_2d(h)[self.freem].reshape(self.shape)

            class WrappedHessp:  # type:ignore
                __slots__ = ("fcn", "free", "par", "vec")

                def __init__(self, fcn):
                    self.fcn = fcn
                    self.free = cfree
                    self.par = cpar
                    self.vec = np.zeros_like(self.par)

                if self.fcn._array_call:

                    def __call__(self, par, v):
                        self.par[self.free] = par
                        self.vec[self.free] = v
                        return self.fcn(self.par, self.vec)[self.free]

                else:

                    def __call__(self, par, v):
                        self.par[self.free] = par
                        self.vec[self.free] = v
                        return self.fcn(*self.par, self.vec)[self.free]

        fcn = Wrapped(self.fcn._fcn)

        grad = self.fcn._grad
        grad = WrappedGrad(grad) if grad else None

        if hess:
            hess = WrappedHess(hess)

        if hessp:
            hessp = WrappedHessp(hessp)

        if constraints is not None:
            if isinstance(constraints, dict):
                raise ValueError("setting constraints with dicts is not supported")

            if not isinstance(constraints, _tp.Iterable):
                constraints = [constraints]

            for i, c in enumerate(constraints):
                if isinstance(c, NonlinearConstraint):
                    c.fun = Wrapped(c.fun)
                elif isinstance(c, LinearConstraint):
                    if no_fixed_parameters == False:
                        x = cpar.copy()
                        x[cfree] = 0
                        shift = np.dot(c.A, x)
                        lb = c.lb - shift
                        ub = c.ub - shift
                        A = np.atleast_1d(c.A)[:, cfree]
                        constraints[i] = LinearConstraint(A, lb, ub, c.keep_feasible)
                else:
                    raise ValueError(
                        "setting constraints with dicts is not supported, use "
                        "LinearConstraint or NonlinearConstraint from scipy.optimize."
                    )

        pr = self._mnprecision()

        # Limits for scipy need to be a little bit tighter than the ones for Minuit
        # so that the Jacobian of the transformation is not zero or infinite.
        start = []
        lower_bound = []
        upper_bound = []
        has_limits = False
        for p in self.params:
            if p.is_fixed:
                continue
            has_limits |= p.has_limits
            # ensure lower < x < upper for Minuit
            ai = -np.inf if p.lower_limit is None else p.lower_limit
            bi = np.inf if p.upper_limit is None else p.upper_limit
            if ai > 0:
                ai *= 1 + pr.eps2
            elif ai < 0:
                ai *= 1 - pr.eps2
            else:
                ai = pr.eps2
            if bi > 0:
                bi *= 1 - pr.eps2
            elif bi < 0:
                bi *= 1 + pr.eps2
            else:
                bi = -pr.eps2
            xi = np.clip(p.value, ai, bi)
            lower_bound.append(ai)
            upper_bound.append(bi)
            start.append(xi)

        if method is None:
            # like in scipy.optimize.minimize
            if constraints:
                method = "SLSQP"
            elif has_limits:
                method = "L-BFGS-B"
            else:
                method = "BFGS"

        # various workarounds for API inconsistencies in scipy.optimize.minimize
        options = {"maxiter": ncall}
        if method in (
            "Nelder-Mead",
            "Powell",
        ):
            options["maxfev"] = ncall
            del options["maxiter"]

        if method in ("L-BFGS-B", "TNC"):
            options["maxfun"] = ncall
            del options["maxiter"]

        if method in ("COBYLA", "SLSQP", "trust-constr") and constraints is None:
            constraints = ()

        t = mutil._Timer(self._fmin)
        with t:
            r = minimize(
                fcn,
                start,
                method=method,
                bounds=Bounds(lower_bound, upper_bound, keep_feasible=True)
                if has_limits
                else None,
                jac=grad,
                hess=hess,
                hessp=hessp,
                constraints=constraints,
                options=options,
            )
        if self.print_level > 0:
            print(r)

        self.fcn._nfcn += r["nfev"]
        if grad:
            self.fcn._ngrad += r.get("njev", 0)

        # Get inverse Hesse matrix, working around many inconsistencies in scipy.
        # Try in order:
        # 1) If hess_inv is returned as full matrix as result, use that.
        # 2) If hess is returned as full matrix, invert it and use that.
        # - These two are approximations to the exact Hessian. -
        # 3) If externally computed hessian was passed to method, use that.
        #    Hessian is considered accurate then.

        matrix = None
        needs_invert = False
        if "hess_inv" in r:
            matrix = r.hess_inv
        elif "hess" in r:
            matrix = r.hess
            needs_invert = True
        # hess_inv is a function, need to convert to full matrix
        if isinstance(matrix, _tp.Callable):  # type:ignore
            matrix = matrix(np.eye(self.nfit))  # type:ignore

        accurate_covar = bool(hess) or bool(hessp)

        # Newton-CG neither returns hessian nor inverted hessian
        if matrix is None:
            if accurate_covar:
                if hessp:
                    matrix = [hessp(r.x, ei) for ei in np.eye(self.nfit)]
                else:
                    matrix = hess(r.x)
                needs_invert = True

        if needs_invert:
            matrix = np.linalg.inv(matrix)

        # Last resort: use parameter step sizes as "errors"
        if matrix is None:
            matrix = np.zeros((self.nfit, self.nfit))
            i = 0
            for p in self.params:
                if p.is_fixed:
                    continue
                matrix[i, i] = p.error**2
                i += 1

        if "grad" in r:  # trust-constr has "grad" and "jac", but "grad" is "jac"!
            jac = r.grad
        elif "jac" in r:
            jac = r.jac
        else:
            tol = 1e-2
            dx = np.sqrt(np.diag(matrix) * tol)
            jac = mutil._jacobi(fcn, r.x, dx, tol)[1][0]

        edm_goal = self._edm_goal(migrad_factor=True)
        fm = FunctionMinimum(
            self._last_state.trafo,
            r.x,
            matrix,
            jac,
            r.fun,
            self.errordef,
            edm_goal,
            self.nfcn,
            ncall,
            accurate_covar,
        )

        self._last_state = fm.state
        self._fmin = mutil.FMin(
            fm,
            f"SciPy[{method}]",
            self.nfcn,
            self.ngrad,
            self.ndof,
            edm_goal,
            t.value,
        )

        if accurate_covar:
            self._make_covariance()
        else:
            if self.strategy.strategy > 0:
                self.hesse()

        return self

    def hesse(self, ncall: _tp.Optional[int] = None) -> "Minuit":
        """
        Run Hesse algorithm to compute asymptotic errors.

        The Hesse method estimates the covariance matrix by inverting the matrix of
        `second derivatives (Hesse matrix) at the minimum
        <http://en.wikipedia.org/wiki/Hessian_matrix>`_. To get parameters correlations,
        you need to use this. The Minos algorithm is another way to estimate parameter
        uncertainties, see :meth:`minos`.

        Parameters
        ----------
        ncall :
            Approximate upper limit for the number of calls made by the Hesse algorithm.
            If set to None, use the adaptive heuristic from the Minuit2 library
            (Default: None).

        Notes
        -----
        The covariance matrix is asymptotically (in large samples) valid. By valid we mean
        that confidence intervals constructed from the errors contain the true value with
        a well-known coverage probability (68 % for each interval). In finite samples,
        this is likely to be true if your cost function looks like a hyperparabola around
        the minimum.

        In practice, the errors very likely have correct coverage if the results from
        Minos and Hesse methods agree. It is possible to construct artifical functions
        where this rule is violated, but in practice it should always work.

        See Also
        --------
        minos
        """
        ncall = 0 if ncall is None else int(ncall)

        # Should be fixed upstream: workaround for segfault in MnHesse when all
        # parameters are fixed
        if self.nfit == 0:
            warnings.warn(
                "Hesse called with all parameters fixed",
                mutil.IMinuitWarning,
                stacklevel=2,
            )
            return self

        if self._fmin_does_not_exist_or_last_state_was_modified():
            # create a seed minimum
            edm_goal = self._edm_goal(migrad_factor=True)
            fm = FunctionMinimum(
                self._fcn,
                self._last_state,
                self._strategy,
                edm_goal,
            )
            self._fmin = mutil.FMin(
                fm, "External", self.nfcn, self.ngrad, self.ndof, edm_goal, 0
            )
            self._merrors = mutil.MErrors()

        assert self._fmin is not None
        fm = self._fmin._src

        # update _fmin with Hesse
        hesse = MnHesse(self.strategy)

        t = mutil._Timer(self._fmin)
        with t:
            hesse(self._fcn, fm, ncall, self._fmin.edm_goal)

        self._last_state = fm.state
        self._fmin = mutil.FMin(
            fm,
            self._fmin.algorithm,
            self.nfcn,
            self.ngrad,
            self.ndof,
            self._fmin.edm_goal,
            t.value,
        )

        self._make_covariance()

        return self  # return self for method chaining and to autodisplay current state

    def minos(
        self,
        *parameters: str,
        cl: _tp.Optional[float] = None,
        ncall: _tp.Optional[int] = None,
    ) -> "Minuit":
        """
        Run Minos algorithm to compute confidence intervals.

        The Minos algorithm uses the profile likelihood method to compute (generally
        asymmetric) confidence intervals. It scans the negative log-likelihood or
        (equivalently) the least-squares cost function around the minimum to construct a
        confidence interval.

        Parameters
        ----------
        *parameters :
            Names of parameters to generate Minos errors for. If no positional
            arguments are given, Minos is run for each parameter.
        cl : float or None, optional
            Confidence level for the confidence interval. If not set or None, a
            standard 68.3 % confidence interval is produced. Setting this to another
            value requires the scipy module to be installed. If 0 < cl < 1, the value
            is interpreted as the confidence level (a probability). If cl >= 1, it is
            interpreted as number of standard deviations. For example, cl=3 produces a
            3 sigma interval. Values other than 0.68, 0.9, 0.95, 0.99, 1, 2, 3, 4, 5
            require the scipy module.
        ncall : int or None, optional
            Limit the number of calls made by Minos. If None, an adaptive internal
            heuristic of the Minuit2 library is used (Default: None).

        Notes
        -----
        Asymptotically (large samples), the Minos interval has a coverage probability
        equal to the given confidence level. The coverage probability is the probility for
        the interval to contain the true value in repeated identical experiments.

        The interval is invariant to transformations and thus not distorted by parameter
        limits, unless the limits intersect with the confidence interval. As a
        rule-of-thumb: when the confidence intervals computed with the Hesse and Minos
        algorithms differ strongly, the Minos intervals are preferred. Otherwise, Hesse
        intervals are preferred.

        Running Minos is computationally expensive when there are many fit parameters.
        Effectively, it scans over one parameter in small steps and runs a full
        minimisation for all other parameters of the cost function for each scan point.
        This requires many more function evaluations than running the Hesse algorithm.
        """
        ncall = 0 if ncall is None else int(ncall)

        factor = _cl_to_errordef(cl, 1, 1.0)

        if self._fmin_does_not_exist_or_last_state_was_modified():
            self.hesse()  # creates self._fmin

        assert self._fmin is not None
        fm = self._fmin._src

        if not self.valid:
            raise RuntimeError(f"Function minimum is not valid: {repr(self._fmin)}")

        if len(parameters) == 0:
            pars = [par for par in self.parameters if not self.fixed[par]]
        else:
            pars = []
            for par in parameters:
                if par not in self._var2pos:
                    raise RuntimeError(f"Unknown parameter {par}")
                if self.fixed[par]:
                    warnings.warn(
                        f"Cannot scan over fixed parameter {par}",
                        mutil.IMinuitWarning,
                    )
                else:
                    pars.append(par)

        t = mutil._Timer(self._fmin)
        with t:
            with _TemporaryErrordef(self._fcn, factor):
                minos = MnMinos(self._fcn, fm, self.strategy)
                for par in pars:
                    me = minos(self._var2pos[par], ncall, self._tolerance)
                    self._merrors[par] = mutil.MError(
                        me.number,
                        par,
                        me.lower,
                        me.upper,
                        me.is_valid,
                        me.lower_valid,
                        me.upper_valid,
                        me.at_lower_limit,
                        me.at_upper_limit,
                        me.at_lower_max_fcn,
                        me.at_upper_max_fcn,
                        me.lower_new_min,
                        me.upper_new_min,
                        me.nfcn,
                        me.min,
                    )

        if self._fmin:
            self._fmin._nfcn = self.nfcn
            self._fmin._ngrad = self.ngrad
            self._fmin._time = t.value

        return self  # return self for method chaining and to autodisplay current state

    def mnprofile(
        self,
        vname: str,
        *,
        size: int = 30,
        bound: _tp.Union[float, _tp.Collection[float]] = 2,
        grid: _tp.Optional[_ArrayLike[float]] = None,
        subtract_min: bool = False,
    ) -> _tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Get Minos profile over a specified interval.

        Scans over one parameter and minimises the function with respect to all other
        parameters for each scan point.

        Parameters
        ----------
        vname : str
            Parameter to scan over.
        size : int, optional
            Number of scanning points (Default: 100). Ignored if grid is set.
        bound : tuple of float or float, optional
            If bound is tuple, (left, right) scanning bound.
            If bound is a number, it specifies an interval of N :math:`\sigma`
            symmetrically around the minimum (Default: 2). Ignored if grid is set.
        grid : array-like, optional
            Parameter values on which to compute the profile. If grid is set, size and
            bound are ignored.
        subtract_min : bool, optional
            If true, subtract offset so that smallest value is zero (Default: False).

        Returns
        -------
        array of float
            Parameter values where the profile was computed.
        array of float
            Profile values.
        array of bool
            Whether minimisation in each point succeeded or not.
        """
        if vname not in self._pos2var:
            raise ValueError("Unknown parameter %s" % vname)

        if grid is not None:
            x = np.array(grid, dtype=float)
            if x.ndim != 1:
                raise ValueError("grid must be 1D array-like")
        else:
            a, b = self._normalize_bound(vname, bound)
            x = np.linspace(a, b, size, dtype=float)

        y = np.empty_like(x)
        status = np.empty(len(x), dtype=bool)

        state = MnUserParameterState(self._last_state)  # copy
        ipar = self._var2pos[vname]
        state.fix(ipar)
        for i, v in enumerate(x):
            state.set_value(ipar, v)
            migrad = MnMigrad(self._fcn, state, self.strategy)
            fm = migrad(0, self._tolerance)
            if not fm.is_valid:
                warnings.warn(
                    f"MIGRAD fails to converge for {vname}={v}", mutil.IMinuitWarning
                )
            status[i] = fm.is_valid
            y[i] = fm.fval

        if subtract_min:
            y -= np.min(y)

        return x, y, status

    def draw_mnprofile(
        self, vname: str, *, band: bool = True, text: bool = True, **kwargs
    ) -> _tp.Tuple[_tp.Collection[float], _tp.Collection[float]]:
        r"""
        Draw Minos profile over a specified interval (requires matplotlib).

        See :meth:`mnprofile` for details and shared arguments. The following additional
        arguments are accepted.

        Parameters
        ----------
        band : bool, optional
            If true, show a band to indicate the Hesse error interval (Default: True).

        text : bool, optional
            If true, show text a title with the function value and the Hesse error
            (Default: True).

        Examples
        --------
        .. plot:: plots/mnprofile.py
            :include-source:

        See Also
        --------
        mnprofile, profile, draw_profile
        """
        if "subtract_min" not in kwargs:
            kwargs["subtract_min"] = True
        x, y, _ = self.mnprofile(vname, **kwargs)
        return self._draw_profile(vname, x, y, band, text)

    def profile(
        self,
        vname: str,
        *,
        size: int = 100,
        bound: _tp.Union[float, _tp.Collection[float]] = 2,
        grid: _tp.Optional[_ArrayLike[float]] = None,
        subtract_min: bool = False,
    ) -> _tp.Tuple[np.ndarray, np.ndarray]:
        r"""
        Calculate 1D cost function profile over a range.

        A 1D scan of the cost function around the minimum, useful to inspect the
        minimum. For a fit with several free parameters this is not the same as the Minos
        profile computed by :meth:`mncontour`.

        Parameters
        ----------
        vname : str
            Parameter to scan over.
        size : int, optional
            Number of scanning points (Default: 100). Ignored if grid is set.
        bound : tuple of float or float, optional
            If bound is tuple, (left, right) scanning bound.
            If bound is a number, it specifies an interval of N :math:`\sigma`
            symmetrically around the minimum (Default: 2). Ignored if grid is set.
        grid : array-like, optional
            Parameter values on which to compute the profile. If grid is set, size and
            bound are ignored.
        subtract_min : bool, optional
            If true, subtract offset so that smallest value is zero (Default: False).

        Returns
        -------
        array of float
            Parameter values.
        array of float
            Function values.

        See Also
        --------
        mnprofile
        """
        if grid is not None:
            x = np.array(grid, dtype=float)
            if x.ndim != 1:
                raise ValueError("grid must be 1D array-like")
        else:
            a, b = self._normalize_bound(vname, bound)
            x = np.linspace(a, b, size, dtype=float)

        y = np.empty_like(x)
        values = np.array(self.values)
        ipar = self._var2pos[vname]
        for i, vi in enumerate(x):
            values[ipar] = vi
            y[i] = self.fcn(values)

        if subtract_min:
            y -= np.min(y)

        return x, y

    def draw_profile(
        self, vname: str, *, band: bool = True, text: bool = True, **kwargs
    ) -> _tp.Tuple[np.ndarray, np.ndarray]:
        """
        Draw 1D cost function profile over a range (requires matplotlib).

        See :meth:`profile` for details and shared arguments. The following additional
        arguments are accepted.

        Parameters
        ----------
        band : bool, optional
            If true, show a band to indicate the Hesse error interval (Default: True).

        text : bool, optional
            If true, show text a title with the function value and the Hesse error
            (Default: True).

        See Also
        --------
        profile, mnprofile, draw_mnprofile
        """
        if "subtract_min" not in kwargs:
            kwargs["subtract_min"] = True
        x, y = self.profile(vname, **kwargs)
        return self._draw_profile(vname, x, y, band, text)

    def _draw_profile(
        self, vname: str, x: np.ndarray, y: np.ndarray, band: bool, text: bool
    ) -> _tp.Tuple[np.ndarray, np.ndarray]:
        from matplotlib import pyplot as plt

        plt.plot(x, y)
        plt.xlabel(vname)
        plt.ylabel("FCN")

        v = self.values[vname]
        plt.axvline(v, color="k", linestyle="--")

        vmin = None
        vmax = None
        if vname in self.merrors:
            vmin = v + self.merrors[vname].lower
            vmax = v + self.merrors[vname].upper
        else:
            vmin = v - self.errors[vname]
            vmax = v + self.errors[vname]

        if vmin is not None and band:
            plt.axvspan(vmin, vmax, facecolor="0.8")

        if text:
            plt.title(
                (f"{vname} = {v:.3g}")
                if vmin is None
                else (
                    "{} = {:.3g} - {:.3g} + {:.3g}".format(vname, v, v - vmin, vmax - v)
                ),
                fontsize="large",
            )

        return x, y

    def contour(
        self,
        x: str,
        y: str,
        *,
        size: int = 50,
        bound: _tp.Union[
            float, _tp.Tuple[_tp.Tuple[float, float], _tp.Tuple[float, float]]
        ] = 2,
        grid: _tp.Optional[_tp.Tuple[_ArrayLike, _ArrayLike]] = None,
        subtract_min: bool = False,
    ) -> _tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Get a 2D contour of the function around the minimum.

        It computes the contour via a function scan over two parameters, while keeping
        all other parameters fixed. The related :meth:`mncontour` works differently: for
        each pair of parameter values in the scan, it minimises the function with the
        respect to all other parameters.

        This method is useful to inspect the function near the minimum to detect issues
        (the contours should look smooth). It is not a confidence region unless the
        function only has two parameters. Use :meth:`mncontour` to compute confidence
        regions.

        Parameters
        ----------
        x : str
            First parameter for scan.
        y : str
            Second parameter for scan.
        size : int or tuple of int, optional
            Number of scanning points per parameter (Default: 50). A tuple is
            interpreted as the number of scanning points per parameter.
            Ignored if grid is set.
        bound : float or tuple of floats, optional
            If bound is 2x2 array, [[v1min,v1max],[v2min,v2max]].
            If bound is a number, it specifies how many :math:`\sigma`
            symmetrically from minimum (minimum+- bound*:math:`\sigma`).
            (Default: 2). Ignored if grid is set.
        grid : tuple of array-like, optional
            Grid points to scan over. If grid is set, size and bound are ignored.
        subtract_min :
            Subtract minimum from return values (Default: False).

        Returns
        -------
        array of float
            Parameter values of first parameter.
        array of float
            Parameter values of second parameter.
        2D array of float
            Function values.

        See Also
        --------
        mncontour, mnprofile
        """
        if grid is not None:
            xg, yg = grid
            xv = np.array(xg, dtype=float)
            yv = np.array(yg, dtype=float)
            if xv.ndim != 1 or yv.ndim != 1:
                raise ValueError("grid per parameter must be 1D array-like")
        else:
            if isinstance(bound, tuple):
                xb, yb = bound
                xrange = self._normalize_bound(x, xb)
                yrange = self._normalize_bound(y, yb)
            else:
                n = float(bound)
                xrange = self._normalize_bound(x, n)
                yrange = self._normalize_bound(y, n)
            if isinstance(size, _tp.Iterable):
                xsize, ysize = size
            else:
                xsize = size
                ysize = size
            xv = np.linspace(xrange[0], xrange[1], xsize)
            yv = np.linspace(yrange[0], yrange[1], ysize)
        zv = np.empty((len(xv), len(yv)), dtype=float)

        ipar = self._var2pos[x]
        jpar = self._var2pos[y]
        values = np.array(self.values)
        for i, xi in enumerate(xv):
            values[ipar] = xi
            for j, yi in enumerate(yv):
                values[jpar] = yi
                zv[i, j] = self._fcn(values)

        if subtract_min:
            zv -= np.min(zv)

        return xv, yv, zv

    def draw_contour(
        self,
        x: str,
        y: str,
        **kwargs,
    ) -> _tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Draw 2D contour around minimum (requires matplotlib).

        See :meth:`contour` for details on parameters and interpretation. Please also read
        the docs of :meth:`mncontour` to understand the difference between the two.

        See Also
        --------
        contour, mncontour, draw_mncontour
        """
        from matplotlib import pyplot as plt

        if "subtract_min" not in kwargs:
            kwargs["subtract_min"] = True
        vx, vy, vz = self.contour(x, y, **kwargs)

        v = [self.errordef * (i + 1) for i in range(4)]

        CS = plt.contour(vx, vy, vz.T, v)
        plt.clabel(CS, v)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.axhline(self.values[y], color="k", ls="--")
        plt.axvline(self.values[x], color="k", ls="--")
        return vx, vy, vz

    def mncontour(
        self,
        x: str,
        y: str,
        *,
        cl: _tp.Optional[float] = None,
        size: int = 100,
        interpolated: int = 0,
    ) -> np.ndarray:
        """
        Get 2D Minos confidence region.

        This scans over two parameters and minimises all other free parameters for each
        scan point. This scan produces a statistical confidence region according to the
        `profile likelihood method <https://en.wikipedia.org/wiki/Likelihood_function>`_
        with a confidence level `cl`, which is asymptotically equal to the coverage
        probability of the confidence region.

        The calculation is expensive since a numerical minimisation has to be performed
        at various points.

        Parameters
        ----------
        x : str
            Variable name of the first parameter.
        y : str
            Variable name of the second parameter.
        cl : float or None, optional
            Confidence level of the contour. If not set or None, a standard 68 %
            contour is computed (default). If 0 < cl < 1, the value is interpreted as the
            confidence level (a probability). If cl >= 1, it is interpreted as number of
            standard deviations. For example, cl=3 produces a 3 sigma contour. Values
            other than 0.68, 0.9, 0.95, 0.99, 1, 2, 3, 4, 5 require the scipy module.
        size : int, optional
            Number of points on the contour to find (default: 100). Increasing this makes
            the contour smoother, but requires more computation time.
        interpolated : int, optional
            Number of interpolated points on the contour (default: 0). If you set this to
            a value larger than size, cubic spline interpolation is used to generate a
            smoother curve and the interpolated coordinates are returned. Values smaller
            than size are ignored. Good results can be obtained with size=20,
            interpolated=200. This requires scipy.

        Returns
        -------
        array of float (N x 2)
            Contour points of the form [[x1, y1]...[xn, yn]].
            Note that N = size + 1, the last point [xn, yn] is identical to [x1, y1].
            This makes it easier to draw a closed contour.

        See Also
        --------
        contour, mnprofile
        """
        factor = _cl_to_errordef(cl, 2, 0.68)

        if self._fmin_does_not_exist_or_last_state_was_modified():
            self.hesse()  # creates self._fmin

        if not self.valid:
            raise RuntimeError(f"Function minimum is not valid: {repr(self._fmin)}")

        pars = set((x, y)) - self._free_parameters()
        if pars:
            raise ValueError(
                f"mncontour can only be run on free parameters, not on {pars}"
            )

        ix = self._var2pos[x]
        iy = self._var2pos[y]
        with _TemporaryErrordef(self._fcn, factor):
            assert self._fmin is not None
            mnc = MnContours(self._fcn, self._fmin._src, self.strategy)
            ce = mnc(ix, iy, size)[2]

        pts = np.array(ce)
        # add starting point at end to close the contour
        pts = np.append(pts, pts[:1], axis=0)

        if interpolated > size:
            try:
                from scipy.interpolate import CubicSpline

                x = np.linspace(0, 1, len(pts))
                spl = CubicSpline(x, pts, bc_type="periodic")
                pts = spl(np.linspace(0, 1, interpolated))

            except ModuleNotFoundError:
                warnings.warn(
                    "Interpolation requires scipy. Please install scipy.",
                    mutil.IMinuitWarning,
                )
        return pts

    def draw_mncontour(
        self,
        x: str,
        y: str,
        *,
        cl: _tp.Optional[_tp.Union[float, _ArrayLike[float]]] = None,
        size: int = 100,
        interpolated: int = 0,
    ) -> _tp.Any:
        """
        Draw 2D Minos confidence region (requires matplotlib).

        See :meth:`mncontour` for details on the interpretation of the region.

        Parameters
        ----------
        See :meth:`mncontour`.

        Examples
        --------
        .. plot:: plots/mncontour.py
            :include-source:

        Returns
        -------
        ContourSet
            Instance of a ContourSet class from matplot.contour.

        See Also
        --------
        mncontour
        """
        from matplotlib import __version__ as mpl_version
        from matplotlib import pyplot as plt
        from matplotlib.path import Path
        from matplotlib.contour import ContourSet

        mpl_version = tuple(map(int, mpl_version.split(".")))

        cls = [mutil._replace_none(x, 0.68) for x in mutil._iterate(cl)]

        c_val = []
        c_pts = []
        codes = []
        for cl in cls:
            pts = self.mncontour(x, y, cl=cl, size=size, interpolated=interpolated)
            n_lineto = len(pts) - 2
            if mpl_version < (3, 5):
                n_lineto -= 1  # pragma: no cover
            c_val.append(cl)
            c_pts.append([pts])  # level can have more than one contour in mpl
            codes.append([[Path.MOVETO] + [Path.LINETO] * n_lineto + [Path.CLOSEPOLY]])
        assert len(c_val) == len(codes), f"{len(c_val)} {len(codes)}"
        cs = ContourSet(plt.gca(), c_val, c_pts, codes)
        plt.clabel(cs)
        plt.xlabel(x)
        plt.ylabel(y)

        return cs

    def draw_mnmatrix(
        self,
        *,
        cl: _tp.Optional[_tp.Union[float, _ArrayLike[float]]] = None,
        size: int = 100,
        figsize=None,
    ) -> _tp.Any:
        """
        Draw matrix of Minos scans (requires matplotlib).

        This draws a matrix of Minos likelihood scans, meaning that the likelihood is
        minimized with respect to the parameters that are not scanned over. The diagonal
        cells of the matrix show the 1D scan, the off-diagonal cells show 2D scans for
        all unique pairs of parameters. The 2D scans show confidence regions. See
        :meth:`mncontour` for details on the interpretation of these regions.

        Parameters
        ----------
        cl : float or collection of floats, optional
            See :meth:`mncontour`.

        size : int, optional
            See :meth:`mncontour`

        Examples
        --------
        .. plot:: plots/mnmatrix.py
            :include-source:

        Returns
        -------
        fig, ax
            Figure and axes instances generated by matplotlib.

        See Also
        --------
        mncontour
        """
        if not self.valid:
            raise RuntimeError(f"Function minimum is not valid: {repr(self._fmin)}")

        pars = [p for p in self.parameters if not self.fixed[p]]
        npar = len(pars)

        if npar == 0:
            raise RuntimeError("all parameters are fixed")

        cls = [mutil._replace_none(x, 0.68) for x in mutil._iterate(cl)]
        if len(cls) == 0:
            raise ValueError("cl must have at least one value")

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(
            npar,
            npar,
            figsize=figsize,
            constrained_layout=True,
            squeeze=False,
        )

        try:
            from progressbar import ProgressBar
        except ModuleNotFoundError:
            ProgressBar = mutil._ProgressBar

        prange = {p: (np.inf, -np.inf) for p in pars}

        with ProgressBar(
            max_value=npar + (npar * (npar + 1) // 2 - npar) * len(cls)
        ) as bar:
            for i, par1 in enumerate(pars):
                plt.sca(ax[i, i])
                fmax = 0
                for k, cl in enumerate(cls):
                    f = _cl_to_errordef(cl, 1, 0.68)
                    fmax = max(fmax, f)
                    plt.axhline(f, color=f"C{k}")
                bound = fmax**0.5 + 1
                for iter in range(5):
                    x, y, ok = self.mnprofile(par1, bound=bound, subtract_min=True)
                    x = x[ok]
                    y = y[ok]
                    if y[0] > fmax and y[-1] > fmax:
                        break
                    bound *= 1.6
                bar += 1
                plt.plot(x, y, "k")
                a, b = prange[par1]
                extremes = []
                for k, (xk, yk) in enumerate(zip(x, y)):
                    if yk < fmax and y[k - 1] > fmax:
                        extremes.append(x[k - 1])
                    if yk > fmax and y[k - 1] < fmax:
                        extremes.append(xk)
                a = min(*extremes, a)
                b = max(*extremes, b)
                prange[par1] = (a, b)
                plt.ylim(0, fmax + 0.5)
                for j in range(i):
                    par2 = pars[j]
                    plt.sca(ax[i, j])
                    plt.plot(self.values[par2], self.values[par1], "+", color="k")
                    for k, cl in enumerate(cls):
                        pts = self.mncontour(par1, par2, cl=cl, size=size)
                        bar += 1
                        if len(pts) > 0:
                            x, y = np.transpose(pts)
                            plt.plot(y, x, color=f"C{k}")
                            for r, p in ((x, par1), (y, par2)):
                                a, b = prange[p]
                                a = min(np.min(r), a)
                                b = max(np.max(r), b)
                                prange[p] = (a, b)
                    ax[j, i].set_visible(False)

        for i, par1 in enumerate(pars):
            ax[i, i].set_xlim(*prange[par1])
            if i > 0:
                ax[i, 0].set_ylabel(par1)
            ax[-1, i].set_xlabel(par1)
            for j in range(i):
                par2 = pars[j]
                ax[j, i].set_xlim(*prange[par1])
                ax[j, i].set_ylim(*prange[par2])

        return fig, ax

    def interactive(
        self,
        plot: _tp.Optional[_tp.Callable] = None,
        raise_on_exception=False,
        **kwargs,
    ):
        """
        Return fitting widget (requires ipywidgets, IPython, matplotlib).

        A fitting widget is returned which can be displayed and manipulated in a
        Jupyter notebook to find good starting parameters and to debug the fit.

        Parameters
        ----------
        plot, optional : Callable or None
            To visualize the fit, interactive tries to access the visualize method on
            the cost function, which accepts the current model parameters as an array-like
            and potentially further keyword arguments, and draws a visualization into the
            current matplotlib axes. If the cost function does not provide a visualize
            method or if you want to override it, pass the function here.
        raise_on_exception : bool, optional
            The default is to catch exceptions in the plot function and convert them
            into a plotted message. In unit tests, raise_on_exception should be set to
            True to allow detecting errors.
        **kwargs :
            Any other keyword arguments are forwarded to the plot function.

        Examples
        --------
        .. plot:: plots/interactive.py
            :include-source:
        """
        try:
            from ipywidgets import (
                HBox,
                VBox,
                Output,
                FloatSlider,
                Button,
                ToggleButton,
                Layout,
                Dropdown,
            )
            from IPython.display import clear_output
            from matplotlib import pyplot as plt
        except ModuleNotFoundError as e:
            e.msg += (
                "\n\nPlease install ipywidgets, IPython, and matplotlib to "
                "enable interactive"
            )
            raise

        pyfcn = self.fcn._fcn

        if plot is None:
            if hasattr(pyfcn, "visualize"):
                plot = pyfcn.visualize
            else:
                raise ValueError(
                    f"class {pyfcn.__class__.__name__} has no visualize method, "
                    "please use the plot argument to pass a visualization function"
                )

        def plot_with_frame(args, from_fit, report_success):
            trans = plt.gca().transAxes
            try:
                with warnings.catch_warnings():
                    if self.fcn._array_call:
                        plot([args], **kwargs)  # prevent unpacking of array
                    else:
                        plot(args, **kwargs)
            except Exception:
                if raise_on_exception:
                    raise

                import traceback

                plt.figtext(
                    0.01,
                    0.5,
                    traceback.format_exc(),
                    ha="left",
                    va="center",
                    transform=trans,
                    color="r",
                )
                return
            if from_fit:
                fval = self.fmin.fval
            else:
                fval = self.fcn(args)
            plt.text(
                0.05,
                1.05,
                f"FCN = {fval:.3f}",
                transform=trans,
                fontsize="x-large",
            )
            if from_fit and report_success:
                plt.text(
                    0.95,
                    1.05,
                    f"{'success' if self.valid and self.accurate else 'FAILURE'}",
                    transform=trans,
                    fontsize="x-large",
                    ha="right",
                )

        class ParameterBox(HBox):
            def __init__(self, par, val, min, max, step, fix):
                self.par = par
                self.slider = FloatSlider(
                    val,
                    min=a,
                    max=b,
                    step=step,
                    description=par,
                    continuous_update=True,
                    layout=Layout(min_width="70%"),
                )
                self.fix = ToggleButton(
                    fix, description="Fix", layout=Layout(width="3.1em")
                )
                self.opt = ToggleButton(
                    False, description="Opt", layout=Layout(width="3.5em")
                )
                self.opt.observe(self.on_opt_toggled, "value")
                super().__init__([self.slider, self.fix, self.opt])

            def on_opt_toggled(self, change):
                self.slider.disabled = self.opt.value
                on_slider_change(None)

        def fit():
            if algo_choice.value == "Migrad":
                self.migrad()
            elif algo_choice.value == "Scipy":
                self.scipy()
            elif algo_choice.value == "Simplex":
                self.simplex()
                return False
            else:
                assert False  # pragma: no cover, should never happen
            return True

        def on_slider_change(change):
            if out.block:
                return
            args = [x.slider.value for x in parameters]
            from_fit = False
            report_success = False
            if any(x.opt.value for x in parameters):
                save = self.fixed[:]
                self.fixed = [not x.opt.value for x in parameters]
                self.values = args
                report_success = fit()
                args = self.values[:]
                out.block = True
                for x, val in zip(parameters, args):
                    x.slider.value = val
                out.block = False
                self.fixed = save
                from_fit = True
            # mutil._show_inline_matplotlib_plots()
            with out:
                clear_output(wait=True)
                plot_with_frame(args, from_fit, report_success)
                mutil._show_inline_matplotlib_plots()

        def on_fit_button_clicked(change):
            for x in parameters:
                self.values[x.par] = x.slider.value
                self.fixed[x.par] = x.fix.value
            report_success = fit()
            out.block = True
            for x in parameters:
                val = self.values[x.par]
                if val < x.slider.min:
                    x.slider.min = val
                elif val > x.slider.max:
                    x.slider.max = val
                x.slider.value = val
            out.block = False
            with out:
                clear_output(wait=True)
                plot_with_frame(self.values, True, report_success)
                mutil._show_inline_matplotlib_plots()

        def on_update_button_clicked(change):
            for x in parameters:
                x.slider.continuous_update = not x.slider.continuous_update

        def on_reset_button_clicked(change):
            self.reset()
            out.block = True
            for x in parameters:
                x.slider.value = self.values[x.par]
            out.block = False
            on_slider_change(None)

        parameters = []
        for par in self.parameters:
            val = self.values[par]
            step = mutil._guess_initial_step(val)
            a, b = self.limits[par]
            # safety margin to avoid overflow warnings
            a = a + 1e-300 if np.isfinite(a) else val - 100 * step
            b = b - 1e-300 if np.isfinite(b) else val + 100 * step
            parameters.append(ParameterBox(par, val, a, b, step, self.fixed[par]))

        fit_button = Button(description="Fit")
        fit_button.on_click(on_fit_button_clicked)

        update_button = ToggleButton(True, description="Continuous")
        update_button.observe(on_update_button_clicked)

        reset_button = Button(description="Reset")
        reset_button.on_click(on_reset_button_clicked)

        algo_choice = Dropdown(options=["Migrad", "Scipy", "Simplex"], value="Migrad")

        ui = VBox(
            [
                HBox([fit_button, update_button, reset_button, algo_choice]),
                VBox(parameters),
            ]
        )

        out = Output()
        out.block = False

        for x in parameters:
            x.slider.observe(on_slider_change, "value")
        # mutil._show_inline_matplotlib_plots()
        on_slider_change(None)

        return HBox([out, ui])

    def _free_parameters(self) -> _tp.Set[str]:
        return set(mp.name for mp in self._last_state if not mp.is_fixed)

    def _mnprecision(self) -> MnMachinePrecision:
        pr = MnMachinePrecision()
        if self._precision is not None:
            pr.eps = self._precision
        return pr

    def _normalize_bound(
        self, vname: str, bound: _tp.Union[float, _tp.Collection[float]]
    ) -> _tp.Tuple[float, float]:
        if isinstance(bound, _tp.Collection):
            return mutil._normalize_limit(bound)

        if not self.accurate:
            warnings.warn(
                "Specified nsigma bound, but error matrix is not accurate",
                mutil.IMinuitWarning,
            )
        start = self.values[vname]
        sigma = self.errors[vname]
        return (start - bound * sigma, start + bound * sigma)

    def _copy_state_if_needed(self):
        # If FunctionMinimum exists, _last_state may be a reference to its user state.
        # The state is read-only in C++, but mutable in Python. To not violate
        # invariants, we need to make a copy of the state when the user requests a
        # modification. If a copy was already made (_last_state is already a copy),
        # no further copy has to be made.
        #
        # If FunctionMinimum does not exist, we don't want to copy. We want to
        # implicitly modify _init_state; _last_state is an alias for _init_state, then.
        if self._fmin and self._last_state == self._fmin._src.state:
            self._last_state = MnUserParameterState(self._last_state)

    def _make_covariance(self) -> None:
        if self._last_state.has_covariance:
            cov = self._last_state.covariance
            m = mutil.Matrix(self._var2pos)
            n = len(m)
            if cov.nrow < self.npar:
                ext2int = {}
                k = 0
                for mp in self._last_state:
                    if not mp.is_fixed:
                        ext2int[mp.number] = k
                        k += 1
                m.fill(0)
                for e, i in ext2int.items():
                    for f, j in ext2int.items():
                        m[e, f] = cov[i, j]
            else:
                n = len(m)
                for i in range(n):
                    for j in range(n):
                        m[i, j] = cov[i, j]
            self._covariance = m
        else:
            self._covariance = None

    def _edm_goal(self, migrad_factor=False) -> float:
        # EDM goal
        # - taken from the source code, see VariableMeticBuilder::Minimum and
        #   ModularFunctionMinimizer::Minimize
        # - goal is used to detect convergence but violations by 10x are also accepted;
        #   see VariableMetricBuilder.cxx:425
        edm_goal = max(
            self.tol * self.errordef, self._mnprecision().eps2  # type:ignore
        )
        if migrad_factor:
            edm_goal *= 2e-3
        return edm_goal

    def _migrad_maxcall(self) -> int:
        n = self.nfit
        return 200 + 100 * n + 5 * n * n

    def _fmin_does_not_exist_or_last_state_was_modified(self) -> bool:
        return not self._fmin or self._fmin._src.state is not self._last_state

    def __repr__(self):
        """Get detailed text representation."""
        s = []
        if self.fmin is not None:
            s.append(repr(self.fmin))
        s.append(repr(self.params))
        if self.merrors:
            s.append(repr(self.merrors))
        if self.covariance is not None:
            s.append(repr(self.covariance))
        return "\n".join(s)

    def __str__(self):
        """Get user-friendly text representation."""
        s = []
        if self.fmin is not None:
            s.append(str(self.fmin))
        s.append(str(self.params))
        if self.merrors:
            s.append(str(self.merrors))
        if self.covariance is not None:
            s.append(str(self.covariance))
        return "\n".join(s)

    def _repr_html_(self):
        s = ""
        if self.fmin is not None:
            s += self.fmin._repr_html_()
        s += self.params._repr_html_()
        if self.merrors:
            s += self.merrors._repr_html_()
        if self.covariance is not None:
            s += self.covariance._repr_html_()
        return s

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("<Minuit ...>")
        else:
            p.text(str(self))


def _make_init_state(
    pos2var: _tp.Tuple[str, ...], args: np.ndarray, kwds: _tp.Dict[str, float]
) -> MnUserParameterState:
    nargs = len(args)
    # check kwds
    if nargs:
        if kwds:
            raise RuntimeError(
                f"positional arguments cannot be mixed with "
                f"parameter keyword arguments {kwds}"
            )
    else:
        for kw in kwds:
            if kw not in pos2var:
                raise RuntimeError(
                    f"{kw} is not one of the parameters [{' '.join(pos2var)}]"
                )
        nargs = len(kwds)

    if len(pos2var) != nargs:
        raise RuntimeError(
            f"{nargs} values given for {len(pos2var)} function parameter(s)"
        )

    state = MnUserParameterState()
    for i, x in enumerate(pos2var):
        val = kwds[x] if kwds else args[i]
        err = mutil._guess_initial_step(val)
        state.add(x, val, err)
    return state


def _get_params(mps: MnUserParameterState, merrors: mutil.MErrors) -> mutil.Params:
    def get_me(name: str) -> _tp.Optional[_tp.Tuple[float, float]]:
        if name in merrors:
            me = merrors[name]
            return me.lower, me.upper
        return None

    return mutil.Params(
        (
            mutil.Param(
                mp.number,
                mp.name,
                mp.value,
                mp.error,
                get_me(mp.name),
                mp.is_const,
                mp.is_fixed,
                mp.lower_limit if mp.has_lower_limit else None,
                mp.upper_limit if mp.has_upper_limit else None,
            )
            for mp in mps
        ),
    )


class _TemporaryErrordef:
    def __init__(self, fcn: FCN, factor: float):
        self.saved = fcn._errordef
        self.fcn = fcn
        self.fcn._errordef *= factor

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: object) -> None:
        self.fcn._errordef = self.saved


def _cl_to_errordef(cl, npar, default):
    assert 0 < npar < 3
    cl = float(default if cl is None else cl)
    if cl <= 0:
        raise ValueError("cl must be positive")

    if npar == 1:
        if cl >= 1.0:
            factor = cl**2
        else:
            factor = {
                0.68: 0.988946481478023,  # chi2(1).ppf(0.68)
                0.90: 2.705543454095404,  # chi2(1).ppf(0.9)
                0.95: 3.841458820694124,  # chi2(1).ppf(0.95)
                0.99: 6.634896601021215,  # chi2(1).ppf(0.99)
            }.get(cl, 0.0)
    else:
        factor = {
            0.68: 2.27886856637673,  # chi2(2).ppf(0.68)
            0.90: 4.605170185988092,  # chi2(2).ppf(0.9)
            0.95: 5.991464547107979,  # chi2(2).ppf(0.95)
            0.99: 9.21034037197618,  # chi2(2).ppf(0.99)
            1.0: 2.295748928898636,  # chi2(2).ppf(chi2(1).cdf(1))
            2.0: 6.180074306244168,  # chi2(2).ppf(chi2(1).cdf(2 ** 2))
            3.0: 11.829158081900795,  # chi2(2).ppf(chi2(1).cdf(3 ** 2))
            4.0: 19.333908611934685,  # chi2(2).ppf(chi2(1).cdf(4 ** 2))
            5.0: 28.743702426935496,  # chi2(2).ppf(chi2(1).cdf(5 ** 2))
        }.get(cl, 0.0)

    if factor == 0.0:
        try:
            from scipy.stats import chi2

        except ImportError as exc:
            exc.msg += (
                "\n\n"
                "You set an uncommon cl value, "
                "scipy is needed to process it. Please install scipy."
            )
            raise

        if cl >= 1.0:
            cl = chi2(1).cdf(cl**2)  # convert sigmas into confidence level
        factor = chi2(npar).ppf(cl)  # convert confidence level to errordef

    return factor
