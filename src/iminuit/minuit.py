"""Minuit class."""

from warnings import warn
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
from collections.abc import Iterable
from typing import Tuple, Dict, Callable, Sequence, Optional, Union

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
    """Set :attr:`errordef` to this constant for a least-squares cost function."""

    LIKELIHOOD = 0.5
    """Set :attr:`errordef` to this constant for a negative log-likelihood function."""

    @property
    def fcn(self) -> Callable:
        """Get cost function (usually a least-squares or likelihood function)."""
        return self._fcn

    @property
    def grad(self) -> Callable:
        """Get gradient function of the cost function."""
        return self._fcn._grad

    @property
    def pos2var(self) -> Tuple[str]:
        """Map variable index to name."""
        return self._pos2var

    @property
    def var2pos(self) -> Dict[str, int]:
        """Map variable name to index."""
        return self._var2pos

    @property
    def parameters(self) -> Tuple[str]:
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
        return self._fcn._errordef

    @errordef.setter
    def errordef(self, value: float):
        if value <= 0:
            raise ValueError(f"errordef={value} must be a positive number")
        self._fcn._errordef = value
        if self._fmin:
            self._fmin._src.errordef = value

    @property
    def precision(self) -> float:
        """
        Access estimated precision of the cost function.

        Default: None. If set to None, Minuit assumes the cost function is computed in
        double precision. If the precision of the cost function is lower (because it
        computes in single precision, for example) set this to some multiple of the
        smallest relative change of a parameter that still changes the function.
        """
        return self._precision

    @precision.setter
    def precision(self, value: float):
        if value is not None and not (value > 0):
            raise ValueError("precision must be a positive number or None")
        self._precision = value

    @property
    def tol(self) -> float:
        """
        Access tolerance for convergence.

        The main convergence criteria of MINUIT is `edm < edm_max`, where `edm_max`
        is calculated as `edm_max = 0.002 * tol * errordef` in case of the MIGRAD
        algorithm and as `edm_max = tol * errordef` in case of the SIMPLEX algorithm.
        EDM stands for *Estimated Distance to Minimum*, which is described in the
        `MINUIT paper`_. The EDM criterion is well suited for statistical cost functions,
        since it stops the minimization when parameter improvements become small
        compared to parameter uncertainties.
        """
        return self._tolerance

    @tol.setter
    def tol(self, value: float):
        if value <= 0:
            raise ValueError("tolerance must be positive")
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
    def strategy(self, value: int):
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
        return MnPrint.global_level

    @print_level.setter
    def print_level(self, level: int):
        MnPrint.global_level = level

    @property
    def throw_nan(self) -> bool:
        """
        Access whether to raise runtime error if the function evaluates to NaN.

        If you set this to True, an error is raised whenever the function evaluates
        to NaN.
        """
        return self._fcn._throw_nan

    @throw_nan.setter
    def throw_nan(self, value: bool):
        self._fcn._throw_nan = value

    @property
    def values(self) -> mutil.ValueView:
        """Access parameter values via an array-like view.

        Use to read or write current parameter values based on the parameter index or the
        parameter name as a string. If you change a parameter value and run :meth:`migrad`,
        the minimization will start from that value, similar for :meth:`hesse` and
        :meth:`minos`.

        See Also
        --------
        errors, fixed, limits
        """
        return self._values

    @values.setter
    def values(self, args):
        self._values[:] = args

    @property
    def errors(self) -> mutil.ErrorView:
        """Access parameter parabolic errors via an array-like view.

        Like :attr:`values`, but instead of reading or writing the values, you read or write
        the errors (which double as step sizes for MINUITs numerical gradient estimation).

        See Also
        --------
        values, fixed, limits
        """
        return self._errors

    @errors.setter
    def errors(self, args):
        self._errors[:] = args

    @property
    def fixed(self) -> mutil.FixedView:
        """Access whether parameters are fixed via an array-like view.

        Use to read or write the fixation state of a parameter based on the parameter index
        or the parameter name as a string. If you change the state and run :meth:`migrad`,
        :meth:`hesse`, or :meth:`minos`, the new state is used.

        In case of complex fits, it can help to fix some parameters first and only minimize
        the function with respect to the other parameters, then release the fixed parameters
        and minimize again starting from that state.

        See Also
        --------
        values, errors, limits
        """
        return self._fixed

    @fixed.setter
    def fixed(self, args):
        self._fixed[:] = args

    @property
    def limits(self) -> mutil.LimitView:
        """Access parameter limits via a array-like view.

        Use to read or write the limits of a parameter based on the parameter index
        or the parameter name as a string. If you change the limits and run :meth:`migrad`,
        :meth:`hesse`, or :meth:`minos`, the new state is used.

        In case of complex fits, it can help to limit some parameters first, run Migrad,
        then remove the limits and run Migrad again. Limits will bias the result only if
        the best fit value is outside the limits, not if it is inside. Limits will affect
        the estimated HESSE uncertainties if the parameter is close to a limit. They do
        not affect the MINOS uncertainties, because those are invariant to
        transformations and limits are implemented via a variable transformation.

        See Also
        --------
        values, errors, fixed
        """
        return self._limits

    @limits.setter
    def limits(self, args):
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
    def covariance(self) -> mutil.Matrix:
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
    def fmin(self) -> mutil.FMin:
        """
        Get function minimum data object.

        See Also
        --------
        util.FMin
        """
        return self._fmin

    @property
    def fval(self) -> float:
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
        return _get_params(self._init_state, {})

    @property
    def valid(self) -> bool:
        """
        Return True if the function minimum is valid.

        This is an alias for :attr:`iminuit.util.FMin.is_valid`.

        See Also
        --------
        util.FMin
        """
        return self._fmin and self._fmin.is_valid

    @property
    def accurate(self) -> bool:
        """
        Return True if the covariance matrix is accurate.

        This is an alias for :attr:`iminuit.util.FMin.has_accurate_covar`.

        See Also
        --------
        util.FMin
        """
        return self._fmin and self._fmin.has_accurate_covar

    @property
    def nfcn(self) -> int:
        """Get total number of function calls."""
        return self._fcn._nfcn

    @property
    def ngrad(self) -> int:
        """Get total number of gradient calls."""
        return self._fcn._ngrad

    def __init__(
        self,
        fcn: Callable,
        *args: Union[float, Sequence[float]],
        grad: Optional[Callable] = None,
        name: Optional[Sequence[str]] = None,
        **kwds,
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
        *Accepted callables*

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

        In some cases, the detection fails, for example for a function like this::

                def difficult_fcn(*args): ...

        To use such a function, set `name`.

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
        if len(args) == 1 and isinstance(args[0], Iterable):
            array_call = True
            args = args[0]

        if name is None:
            name = mutil.describe(fcn)
            if len(name) == 0 or (array_call and len(name) == 1):
                name = tuple(f"x{i}" for i in range(len(args)))

        if len(args) == 0 and len(kwds) == 0:
            raise RuntimeError(
                "starting value(s) are required"
                + (f" for {' '.join(name)}" if name else "")
            )

        # Maintain two dictionaries to easily convert between
        # parameter names and position
        self._pos2var = tuple(name)
        self._var2pos = {k: i for i, k in enumerate(name)}

        self._tolerance = 0.1
        self._strategy = MnStrategy(1)
        self._fcn = FCN(
            fcn,
            getattr(fcn, "grad", grad),
            array_call,
            getattr(fcn, "errordef", 0.0),
        )

        self._init_state = _make_init_state(self._pos2var, args, kwds)
        self._values = mutil.ValueView(self)
        self._errors = mutil.ErrorView(self)
        self._fixed = mutil.FixedView(self)
        self._limits = mutil.LimitView(self)

        self.precision = getattr(fcn, "precision", None)

        self.reset()

    def reset(self):
        """
        Reset minimization state to initial state.

        Leaves :attr:`strategy`, :attr:`precision`, :attr:`tol`, :attr:`errordef`,
        :attr:`print_level` unchanged.
        """
        self._last_state = self._init_state
        self._fmin = None
        self._fcn._nfcn = 0
        self._fcn._ngrad = 0
        self._merrors = mutil.MErrors()
        self._covariance = None
        return self  # return self for method chaining and to autodisplay current state

    def migrad(self, ncall: Optional[int] = None, iterate: int = 5):
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

        edm_goal = self._migrad_edm_goal()
        self._fmin = mutil.FMin(fm, self.nfcn, self.ngrad, edm_goal)
        self._make_covariance()

        return self  # return self for method chaining and to autodisplay current state

    def simplex(self, ncall: Optional[int] = None):
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

        fm = simplex(ncall, self._tolerance)
        self._last_state = fm.state

        edm_goal = max(self._tolerance * fm.errordef, simplex.precision.eps2)
        self._fmin = mutil.FMin(fm, self.nfcn, self.ngrad, edm_goal)
        self._covariance = None
        self._merrors = mutil.MErrors()

        return self  # return self for method chaining and to autodisplay current state

    def scan(self, ncall: Optional[int] = None):
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
        Originally this was supposed to use MnScan from C++ Minuit2, but MnScan is broken.
        It does a 1D scan with 41 steps for each parameter in sequence, so it is not
        actually scanning the full hypercube. It first scans one parameter, then starts
        the scan of the second parameter from the best value of the first and so on.
        Other issues: One cannot configure the number of steps. A gradient and second
        derivatives are computed for the starting values only to be discarded.

        This implementation here does a full scan of the hypercube in Python. Returning a
        valid FunctionMinimum object was a major challenge, because C++ Minuit2 does not
        allow one to initialize data objects with data, it forces one to go through
        algorithm objects. Because of that design, the Minuit2 C++ interface forces one to
        compute the gradient and second derivatives for the starting values, even though
        these are not used in a scan. We turn a disadvantage into an advantage by tricking
        Minuit2 into computing updates of the step sizes and to estimate the EDM value.
        """
        # Running MnScan would look like this:
        # scan = MnScan(self._fcn, self._last_state, self.strategy)
        # fm = scan(0, 0)  # args are ignored
        # self._last_state = fm.state
        # self._fmin = mutil.FMin(fm, self._fcn.nfcn, self._fcn.ngrad, self._tolerance)

        n = self.nfit
        if ncall is None:
            ncall = 200 + 100 * n + 5 * n * n
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

        def run(ipar):
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

        run(0)

        edm_goal = self._tolerance * self._fcn._errordef
        fm = FunctionMinimum(self._fcn, self._last_state, self.strategy, edm_goal)
        self._last_state = fm.state
        self._fmin = mutil.FMin(fm, self.nfcn, self.ngrad, edm_goal)
        self._covariance = None
        self._merrors = mutil.MErrors()

        return self  # return self for method chaining and to autodisplay current state

    def hesse(self, ncall: Optional[int] = None):
        r"""
        Run HESSE algorithm to compute asymptotic errors.

        HESSE estimates the covariance matrix by inverting the matrix of
        `second derivatives (Hesse matrix) at the minimum
        <http://en.wikipedia.org/wiki/Hessian_matrix>`_. This covariance matrix is valid
        if your :math:`\chi^2` or likelihood profile looks like a hyperparabola around
        the minimum. Asymptotically (large samples) this is always the case, but in small
        samples the uncertainty estimate is approximate. If you want to know how your
        parameters are correlated, you also need to use this. The MINOS algorithm is
        another way to estimate parameter uncertainties, see :meth:`minos`.

        Parameters
        ----------
        ncall :
            Approximate upper limit for the number of calls made by the HESSE algorithm.
            If set to None, use the adaptive heuristic from the Minuit2 library
            (Default: None).

        See Also
        --------
        minos
        """
        ncall = 0 if ncall is None else int(ncall)

        # Should be fixed upstream: workaround for segfault in MnHesse when all
        # parameters are fixed
        if self.nfit == 0:
            warn(
                "HESSE called with all parameters fixed",
                mutil.IMinuitWarning,
                stacklevel=2,
            )
            return self

        hesse = MnHesse(self.strategy)

        fm = self._fmin._src if self._fmin else None
        if fm and fm.state is self._last_state:
            # fmin exists and _last_state not modified,
            # can update _fmin which is more efficient
            hesse(self._fcn, fm, ncall)
            self._last_state = fm.state
            self._fmin = mutil.FMin(fm, self.nfcn, self.ngrad, self._fmin.edm_goal)
        else:
            # _fmin does not exist or _last_state was modified,
            # so we cannot just update last _fmin
            self._last_state = hesse(self._fcn, self._last_state, ncall)

        if self._last_state.has_covariance is False:
            if not self._fmin:
                raise RuntimeError("HESSE Failed")

        self._make_covariance()

        return self  # return self for method chaining and to autodisplay current state

    def minos(
        self, *parameters: str, cl: Optional[float] = None, ncall: Optional[int] = None
    ):
        """
        Run Minos algorithm to compute confidence intervals.

        The Minos algorithm uses the profile likelihood method to compute (generally
        asymmetric) confidence intervals. It scans the negative log-likelihood or
        (equivalently) the least-squares cost function around the minimum to construct a
        confidence interval.

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

        Parameters
        ----------
        *parameters :
            Names of parameters to generate Minos errors for. If no positional
            arguments are given, Minos is run for each parameter.
        cl :
            Confidence level for the confidence interval. If None, a standard 68.3 %
            confidence interval is produced (Default: None). Setting this to another
            value requires the scipy module to be installed.
        ncall :
            Limit the number of calls made by Minos. If None, an adaptive internal
            heuristic of the Minuit2 library is used (Default: None).
        """
        ncall = 0 if ncall is None else int(ncall)

        if cl is None:
            factor = 1.0
        else:
            try:
                from scipy.stats import chi2
            except ImportError:  # pragma: no cover
                raise ImportError("setting cl requires scipy")  # pragma: no cover
            factor = chi2(1).ppf(cl)

        if not self._fmin:
            # create a seed minimum for MnMinos
            fm = FunctionMinimum(
                self._fcn, self._last_state, self._strategy, self._tolerance
            )
            # running MnHesse on seed is necessary for MnMinos to work
            hesse = MnHesse(self.strategy)
            hesse(self._fcn, fm, ncall)
            self._last_state = fm.state
            self._make_covariance()
        else:
            fm = self._fmin._src

        if not fm.is_valid:
            raise RuntimeError("Function minimum is not valid.")

        if len(parameters) == 0:
            pars = [par for par in self.parameters if not self.fixed[par]]
        else:
            pars = []
            for par in parameters:
                if par not in self._var2pos:
                    raise RuntimeError(f"Unknown parameter {par}")
                if self.fixed[par]:
                    warn(
                        f"Cannot scan over fixed parameter {par}",
                        mutil.IMinuitWarning,
                    )
                else:
                    pars.append(par)

        with TemporaryErrordef(self._fcn, factor):
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

        return self  # return self for method chaining and to autodisplay current state

    def mnprofile(
        self,
        vname: str,
        *,
        size: int = 30,
        bound: Union[int, Sequence[int]] = 2,
        subtract_min: bool = False,
    ) -> Tuple[Sequence[float], Sequence[float], Sequence[bool]]:
        r"""
        Get Minos profile over a specified interval.

        Scans over one parameter and minimises the function with respect to all other
        parameters for each scan point.

        Parameters
        ----------
        vname :
            Name of parameter to scan.
        size :
            Number of scanning points (Default: 30).
        bound :
            If bound is tuple, (left, right) scanning bound. If bound is a number, it
            specifies how many :math:`\sigma` symmetrically from minimum (minimum +/-
            bound * :math:`\sigma`) (Default: 2).
        subtract_min :
            Subtract minimum from return values (Default: False).

        Returns
        -------
        locations : array of float
            Parameter values where the profile was computed.
        fvals: array of float
            Profile values.
        status: array of bool
            Whether minimisation in each point succeeded or not.
        """
        if vname not in self._pos2var:
            raise ValueError("Unknown parameter %s" % vname)

        bound = self._normalize_bound(vname, bound)

        x = np.linspace(bound[0], bound[1], size, dtype=np.double)
        y = np.empty(size, dtype=np.double)
        status = np.empty(size, dtype=np.bool)

        state = MnUserParameterState(self._last_state)  # copy
        ipar = self._var2pos[vname]
        state.fix(ipar)
        for i, v in enumerate(x):
            state.set_value(ipar, v)
            migrad = MnMigrad(self._fcn, state, self.strategy)
            fm = migrad(0, self._tolerance)
            if not fm.is_valid:
                warn(f"MIGRAD fails to converge for {vname}={v}", mutil.IMinuitWarning)
            status[i] = fm.is_valid
            y[i] = fm.fval

        if subtract_min:
            y -= np.min(y)

        return x, y, status

    def draw_mnprofile(
        self,
        vname: str,
        *,
        size: int = 30,
        bound: Union[int, Sequence[Sequence[int]]] = 2,
        subtract_min: bool = False,
        band: bool = True,
        text: bool = True,
    ) -> Tuple[Sequence[float], Sequence[float]]:
        r"""
        Draw Minos profile over a specified interval (requires matplotlib).

        See :meth:`mnprofile` for details and shared arguments. The following arguments
        are accepted.

        Parameters
        ----------
        band :
            If true, show a band to indicate the Hesse error interval (Default: True).

        text :
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
        x, y, _ = self.mnprofile(
            vname, size=size, bound=bound, subtract_min=subtract_min
        )
        return self._draw_profile(vname, x, y, band, text)

    def profile(
        self,
        vname: str,
        *,
        size: int = 100,
        bound: Union[int, Tuple[int, int]] = 2,
        subtract_min: bool = False,
    ) -> Tuple[Sequence[float], Sequence[float]]:
        r"""
        Calculate 1D cost function profile over a range.

        A 1D scan of the cost function around the minimum, useful to inspect the
        minimum. For a fit with several free parameters this is not the same as the Minos
        profile computed by :meth:`mncontour`.

        Parameters
        ----------
        vname :
            Parameter to scan over.
        size :
            Number of scanning points (Default: 100).
        bound :
            If bound is tuple, (left, right) scanning bound.
            If bound is a number, it specifies an interval of N :math:`\sigma`
            symmetrically around the minimum (Default: 2).
        subtract_min :
            If true, subtract offset so that smallest value is zero (Default: False).

        Returns
        -------
        x : array of float
            Parameter values.
        y : array of float
            Function values.

        See Also
        --------
        mnprofile
        """
        bound = self._normalize_bound(vname, bound)

        ipar = self._var2pos[vname]
        x = np.linspace(bound[0], bound[1], size, dtype=np.double)
        y = np.empty(size, dtype=np.double)
        values = np.array(self.values)
        for i, vi in enumerate(x):
            values[ipar] = vi
            y[i] = self.fcn(values)

        if subtract_min:
            y -= np.min(y)

        return x, y

    def draw_profile(
        self,
        vname: str,
        *,
        size: int = 100,
        bound: Union[int, Tuple[int, int]] = 2,
        subtract_min: bool = False,
        band: bool = True,
        text: bool = True,
    ) -> Tuple[Sequence[float], Sequence[float]]:
        """
        Draw 1D cost function profile over a range (requires matplotlib).

        See :meth:`profile` for details and shared arguments. The following additional
        arguments are accepted.

        Parameters
        ----------
        band :
            If true, show a band to indicate the Hesse error interval (Default: True).

        text :
            If true, show text a title with the function value and the Hesse error
            (Default: True).

        See Also
        --------
        profile, mnprofile, draw_mnprofile
        """
        x, y = self.profile(vname, size=size, bound=bound, subtract_min=subtract_min)
        return self._draw_profile(vname, x, y, band, text)

    def _draw_profile(self, vname, x, y, band, text):
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
        bound: Union[int, Sequence[Sequence[int]]] = 2,
        subtract_min: bool = False,
    ) -> Tuple[Sequence[float], Sequence[float], Sequence[Sequence[float]]]:
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
        x :
            First parameter for scan.
        y :
            Second parameter for scan.
        size :
            Number of scanning points (Default: 50).
        bound :
            If bound is 2x2 array, [[v1min,v1max],[v2min,v2max]].
            If bound is a number, it specifies how many :math:`\sigma`
            symmetrically from minimum (minimum+- bound*:math:`\sigma`).
            (Default: 2).
        subtract_min :
            Subtract minimum from return values (Default: False).

        Returns
        -------
        x : array of float
            Parameter values of first parameter.
        y : array of float
            Parameter values of second parameter.
        fval : 2D array of float
            Function values.

        See Also
        --------
        mncontour, mnprofile
        """
        try:
            n = float(bound)
            in_sigma = True
        except TypeError:
            in_sigma = False

        if in_sigma:
            xrange = self._normalize_bound(x, n)
            yrange = self._normalize_bound(y, n)
        else:
            xrange = self._normalize_bound(x, bound[0])
            yrange = self._normalize_bound(y, bound[1])

        ipar = self._var2pos[x]
        jpar = self._var2pos[y]

        x = np.linspace(xrange[0], xrange[1], size)
        y = np.linspace(yrange[0], yrange[1], size)

        z = np.empty((size, size), dtype=np.double)
        values = np.array(self.values)
        for i, xi in enumerate(x):
            values[ipar] = xi
            for j, yi in enumerate(y):
                values[jpar] = yi
                z[i, j] = self._fcn(values)

        if subtract_min:
            z -= np.min(z)

        return x, y, z

    def draw_contour(
        self,
        x: str,
        y: str,
        *,
        size: int = 50,
        bound: Union[int, Sequence[Sequence[int]]] = 2,
    ) -> Tuple[Sequence[float], Sequence[float], Sequence[Sequence[float]]]:
        """
        Draw 2D contour around minimum (required matplotlib).

        See :meth:`contour` for details on parameters and interpretation. Please also read
        the docs of :meth:`mncontour` to understand the difference between the two.

        See Also
        --------
        contour, mncontour, draw_mncontour
        """
        from matplotlib import pyplot as plt

        vx, vy, vz = self.contour(x, y, size=size, bound=bound, subtract_min=True)

        v = [self.errordef * (i + 1) for i in range(4)]

        CS = plt.contour(vx, vy, vz, v)
        plt.clabel(CS, v)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.axhline(self.values[y], color="k", ls="--")
        plt.axvline(self.values[x], color="k", ls="--")
        return vx, vy, vz

    def mncontour(
        self, x: str, y: str, *, cl: Optional[float] = None, size: int = 100
    ) -> Sequence[Sequence[float]]:
        """
        Get 2D MINOS confidence region.

        This scans over two parameters and minimises all other free parameters for each
        scan point. This scan produces a statistical confidence region according to the
        `profile likelihood method <https://en.wikipedia.org/wiki/Likelihood_function>`_
        with a confidence level `cl`, which is asymptotically equal to the coverage
        probability of the confidence region.

        The calculation is expensive since a numerical minimisation has to be performed
        at various points.

        Parameters
        ----------
        x :
            Variable name of the first parameter.
        y :
            Variable name of the second parameter.
        cl :
            Confidence level of the contour. If None, a standard 68 % contour is computed
            (Default: None). Setting this to another value requires the scipy module to
            be installed.
        size :
            Number of points on the contour to find. Default 100. Increasing this makes
            the contour smoother, but requires more computation time.

        Returns
        -------
        points : array of float (N x 2)
            Contour points of the form [[x1, y1]...[xn, yn]].

        See Also
        --------
        contour, mnprofile
        """
        if cl is None:
            factor = 2.27886856637673  # chi2(2).ppf(0.68)
        else:
            try:
                from scipy.stats import chi2

                factor = chi2(2).ppf(float(cl))
            except ImportError:  # pragma: no cover
                raise ImportError("setting cl requires scipy")  # pragma: no cover

        if not self._fmin:
            raise ValueError("Run MIGRAD first")

        ix = self._var2pos[x]
        iy = self._var2pos[y]

        vary = self._free_parameters()
        if x not in vary or y not in vary:
            raise ValueError("mncontour cannot be run on fixed parameters.")

        with TemporaryErrordef(self._fcn, factor):
            mnc = MnContours(self._fcn, self._fmin._src, self.strategy)
            ce = mnc(ix, iy, size)[2]

        return np.array(ce)

    def draw_mncontour(
        self, x: str, y: str, *, cl: Optional[float] = None, size: int = 100
    ) -> Sequence[Sequence[float]]:
        """
        Draw 2D Minos confidence region (requires matplotlib).

        See :meth:`mncontour` for details on parameters and interpretation.

        Examples
        --------
        .. plot:: plots/mncontour.py
            :include-source:

        See Also
        --------
        mncontour
        """
        from matplotlib import pyplot as plt
        from matplotlib.contour import ContourSet

        cls = [None] if cl is None else cl

        c_val = []
        c_pts = []
        for cl in cls:
            pts = self.mncontour(x, y, cl=cl, size=size)
            # close curve
            pts = list(pts)
            pts.append(pts[0])
            c_val.append(cl if cl is not None else 0.68)
            c_pts.append([pts])  # level can have more than one contour in mpl
        cs = ContourSet(plt.gca(), c_val, c_pts)
        plt.clabel(cs)
        plt.xlabel(x)
        plt.ylabel(y)

        return cs

    def _free_parameters(self):
        return (mp.name for mp in self._last_state if not mp.is_fixed)

    def _normalize_bound(self, vname, bound):
        try:
            n = float(bound)
            in_sigma = True
        except TypeError:
            in_sigma = False
            pass

        if in_sigma:
            if not self.accurate:
                warn(
                    "Specified nsigma bound, but error matrix is not accurate",
                    mutil.IMinuitWarning,
                )
            start = self.values[vname]
            sigma = self.errors[vname]
            bound = (start - n * sigma, start + n * sigma)

        return bound

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

    def _make_covariance(self):
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

    def _migrad_edm_goal(self):
        pr = MnMachinePrecision()
        if self.precision is not None:
            pr.eps = self.precision
        # EDM goal
        # - taken from the source code, see VariableMeticBuilder::Minimum and
        #   ModularFunctionMinimizer::Minimize
        # - goal is used to detect convergence but violations by 10x are also accepted;
        #   see VariableMetricBuilder.cxx:425
        return 2e-3 * max(self.tol * self.errordef, pr.eps2)

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


def _make_init_state(pos2var, args, kwds):
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


def _get_params(mps, merrors):
    def get_me(name):
        if name in merrors:
            me = merrors[name]
            return me.lower, me.upper

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
                mp.has_limits,
                mp.has_lower_limit,
                mp.has_upper_limit,
                mp.lower_limit if mp.has_lower_limit else None,
                mp.upper_limit if mp.has_upper_limit else None,
            )
            for mp in mps
        ),
    )


class TemporaryErrordef:
    def __init__(self, fcn, factor):
        self.saved = fcn._errordef
        self.fcn = fcn
        self.fcn._errordef *= factor

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.fcn._errordef = self.saved
