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

MnPrint.global_level = 0

__all__ = ["Minuit"]


class Minuit:

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
    def fcn(self):
        """Cost function (usually a least-squares or likelihood function)."""
        return self._fcn

    @property
    def grad(self):
        """Gradient function of the cost function."""
        return self._fcn._grad

    @property
    def pos2var(self):
        """Map variable index to name."""
        return self._pos2var

    @property
    def var2pos(self):
        """Map variable name to index."""
        return self._var2pos

    @property
    def parameters(self):
        """Tuple of parameter names, an alias for :attr:`pos2var`."""
        return self._pos2var

    @property
    def errordef(self):
        """FCN increment above the minimum that corresponds to one standard deviation.

        Default value is 1.0. `errordef` should be 1.0 for a least-squares cost
        function and 0.5 for negative log-likelihood function. See section 1.5.1 on page 6
        of the :download:`MINUIT2 User's Guide <mnusersguide.pdf>`. This parameter is
        also called *UP* in MINUIT documents.

        To make user code more readable, we provided two named constants::

            m_lsq = Minuit(a_least_squares_function)
            m_lsq.errordef = Minuit.LEAST_SQUARES  # == 1

            m_nll = Minuit(a_likelihood_function)
            m_nll.errordef = Minuit.LIKELIHOOD     # == 0.5
        """
        return self._fcn._errordef

    @errordef.setter
    def errordef(self, value):
        if value <= 0:
            raise ValueError(f"errordef={value} must be a positive number")
        self._fcn._errordef = value
        if self._fmin:
            self._fmin._src.errordef = value

    @property
    def precision(self):
        """Estimated precision of the cost function.

        Default: None. If set to None, Minuit assumes the cost function is computed in
        double precision. If the precision of the cost function is lower (because it
        computes in single precision, for example) set this to some multiple of the
        smallest relative change of a parameter that still changes the function.
        """
        return self._precision

    @precision.setter
    def precision(self, value):
        if value is not None and not (value > 0):
            raise ValueError("precision must be a positive number or None")
        self._precision = value

    @property
    def tol(self):
        """Tolerance for convergence.

        The main convergence criteria of MINUIT is ``edm < edm_max``, where ``edm_max``
        is calculated as ``edm_max = 0.002 * tol * errordef`` in case of the MIGRAD
        algorithm and as ``edm_max = tol * errordef`` in case of the SIMPLEX algorithm.
        EDM stands for *estimated distance to minimum*, which is described in the
        `MINUIT paper`_. The EDM criterion is well suited for statistical cost functions,
        since it stops the minimization when parameter improvements become small
        compared to parameter uncertainties.
        """
        return self._tolerance

    @tol.setter
    def tol(self, value):
        if value <= 0:
            raise ValueError("tolerance must be positive")
        self._tolerance = value

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
        tolerance :attr:`tol` for convergence at any strategy level.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        self._strategy.strategy = value

    @property
    def print_level(self):
        """Current print level.

        - 0: quiet (default)
        - 1: print minimal debug messages to terminal
        - 2: print more debug messages to terminal
        - 3: print even more debug messages to terminal

        Warning: Setting print_level has the unwanted side-effect of setting the level
        globally for all Minuit instances in the current Python session.
        """
        return MnPrint.global_level

    @print_level.setter
    def print_level(self, level):
        MnPrint.global_level = level

    @property
    def throw_nan(self):
        """Boolean. Whether to raise runtime error if function evaluate to nan."""
        return self._fcn._throw_nan

    @throw_nan.setter
    def throw_nan(self, value):
        self._fcn._throw_nan = value

    @property
    def values(self):
        """Parameter values in a array-like object.

        Use to read or write current parameter values based on the parameter index or the
        parameter name as a string. If you change a parameter value and run :meth:`migrad`,
        the minimization will start from that value, similar for :meth:`hesse` and
        :meth:`minos`.

        .. seealso:: :attr:`errors`, :attr:`fixed`, :attr:`limits`
        """
        return self._values

    @values.setter
    def values(self, args):
        self._values[:] = args

    @property
    def errors(self):
        """Parameter parabolic errors in a array-like object.

        Like :attr:`values`, but instead of reading or writing the values, you read or write
        the errors (which double as step sizes for MINUITs numerical gradient estimation).

        .. seealso:: :attr:`values`, :attr:`fixed`, :attr:`limits`
        """
        return self._errors

    @errors.setter
    def errors(self, args):
        self._errors[:] = args

    @property
    def fixed(self):
        """Access fixation state of a parameter in a array-like object.

        Use to read or write the fixation state of a parameter based on the parameter index
        or the parameter name as a string. If you change the state and run :meth:`migrad`,
        :meth:`hesse`, or :meth:`minos`, the new state is used.

        In case of complex fits, it can help to fix some parameters first and only minimize
        the function with respect to the other parameters, then release the fixed parameters
        and minimize again starting from that state.

        .. seealso:: :attr:`values`, :attr:`errors`, :attr:`limits`
        """
        return self._fixed

    @fixed.setter
    def fixed(self, args):
        self._fixed[:] = args

    @property
    def limits(self):
        """Access parameter limits in a array-like object.

        Use to read or write the limits of a parameter based on the parameter index
        or the parameter name as a string. If you change the limits and run :meth:`migrad`,
        :meth:`hesse`, or :meth:`minos`, the new state is used.

        In case of complex fits, it can help to limit some parameters first and then
        remove the limits. Limits will bias the result only if the best fit value is
        outside the limits, not if it is inside. Limits will affect the estimated
        HESSE uncertainties if the parameter is close to a limit.

        .. seealso:: :attr:`values`, :attr:`errors`, :attr:`fixed`
        """
        return self._limits

    @limits.setter
    def limits(self, args):
        self._limits[:] = args

    @property
    def merrors(self):
        """Returns a dict with data objects that contain the full status information
        of the Minos run.

        .. seealso:: :class:`iminuit.util.MError`, :class:`iminuit.util.MErrors`
        """
        return self._merrors

    @property
    def covariance(self):
        """Returns the covariance matrix.

        The square-root of the diagonal elements of the covariance matrix correspond to
        a standard deviation for each parameter with 68 % coverage probability in the
        asymptotic limit (large samples). To get k standard deviations, multiply the
        covariance matrix with k^2.

        The submatrix formed by two parameters describes an ellipse. The asymptotic
        coverage probabilty of the ellipse is lower than 68 %. It can be computed from the
        :math:`\\chi^2` distribution with 2 degrees of freedom. In general, to obtain a
        (hyper-)ellipsoid with coverage probability CL, one has to multiply the
        submatrix of the corresponding k parameters with a factor. For k = 1,2,3 and
        CL = 0.99 ::

            from scipy.stats import chi2

            chi2(1).ppf(0.99) # 6.63...
            chi2(2).ppf(0.99) # 9.21...
            chi2(3).ppf(0.99) # 11.3...

        .. seealso:: :class:`iminuit.util.Matrix`
        """
        return self._covariance

    @property
    def npar(self):
        """Number of parameters."""
        return len(self._last_state)

    @property
    def nfit(self):
        """Number of fitted parameters (fixed parameters not counted)."""
        return self.npar - sum(self.fixed)

    @property
    def fmin(self):
        """Current function minimum.

        .. seealso:: :class:`iminuit.util.FMin`
        """
        return self._fmin

    @property
    def fval(self):
        """Function minimum value (alias for Minuit.fmin.fval).

        .. seealso:: :class:`iminuit.util.FMin`
        """
        fm = self._fmin
        return fm.fval if fm else None

    @property
    def params(self):
        """List of current parameter data objects.

        .. seealso:: :class:`iminuit.util.Params`
        """
        return _get_params(self._last_state, self._merrors)

    @property
    def init_params(self):
        """List of current parameter data objects set to the initial fit state.

        .. seealso:: :class:`iminuit.util.Params`
        """
        return _get_params(self._init_state, {})

    @property
    def valid(self):
        """Whether the function minimum is valid (alias for Minuit.fmin.is_valid).

        .. seealso:: :class:`iminuit.util.FMin`
        """
        return self._fmin and self._fmin.is_valid

    @property
    def accurate(self):
        """Whether the covariance matrix is accurate (alias for Minuit.fmin.has_accurate_covar).

        .. seealso:: :class:`iminuit.util.FMin`
        """
        return self._fmin and self._fmin.has_accurate_covar

    @property
    def nfcn(self):
        """Total number of function calls."""
        return self._fcn._nfcn

    @property
    def ngrad(self):
        """Total number of gradient calls."""
        return self._fcn._ngrad

    def __init__(
        self,
        fcn,
        *args,
        grad=None,
        name=None,
        **kwds,
    ):
        """
        Construct minuit object from given *fcn*.

        **Accepted functions**

            ``fcn``, the function to be optimized, is the only required argument.

            Two kinds of function signatures are understood.

            a) Parameters passed as positional arguments

                The function has positional arguments, one for each fit
                parameter. Example::

                    def fcn(a, b, c): ...

                The parameters a, b, c must accept a real number.

                iminuit automatically detects the parameters names in this case.
                More information about how the function signature is detected can
                be found in :func:`iminuit.util.describe`.

            b) Parameters passed as Numpy array

                The function has a single argument which is a Numpy array.
                Example::

                    def fcn_np(x): ...

                To use this form, starting values need to be passed to Minuit in form as
                an array-like type, e.g. a numpy array, tuple or list. For more details,
                see "Parameter Keyword Arguments" further down.

        **Parameter initialization**

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

            If the arguments are explicitly named with the ``name`` keyword described
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

        **Other keyword arguments**
            - **grad**: Optional. Provide a function that calculates the
              gradient analytically and returns an iterable object with one
              element for each dimension. If None is given MINUIT will
              calculate the gradient numerically. (Default None)

            - **name**: sequence of strings. If set, this is used to detect
              parameter names instead of iminuit's function signature detection.
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
        """Reset minimization state to initial state.

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

    def migrad(self, ncall=None, iterate=5):
        """Run MnMigrad from the Minuit2 library.

        MIGRAD is a robust minimisation algorithm which earned its reputation
        in 40+ years of almost exclusive usage in high-energy physics. How
        MIGRAD works is described in the `MINUIT paper`_.

        **Arguments:**

            * **ncall**: integer or None, optional; (approximate)
              maximum number of calls before minimization will be aborted. Default: None
              (indicates to use an internal heuristic). Note: The limit may be slightly
              violated, because the condition is checked only after a full iteration of
              the algorithm, which usually performs several function calls.

            * **iterate**: automatically call Migrad up to N times if convergence
              was not reached. Default: 5. This simple heuristic makes Migrad converge
              more often even if the numerical precision of the cost function is low.
              Setting this to 1 disables the feature.

        **Return:**

            self
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

    def simplex(self, ncall=None):
        """Run MnSimplex from the Minuit2 C++ library.

        Uses a variant of the Nelder-Mead algorithm to find the minimum. It does not
        make use of derivatives. `The Wikipedia has a good article on the method
        <https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method>`_. The method usually
        converges more slowly than MIGRAD, but may perform better in some special cases.
        The Rosenbrock function is one of those. The SIMPLEX method does not have
        quadratic convergerence near the minimum that MIGRAD offers, but it may better
        at getting close to the minimum from an unsuitable starting point.

        The convergence criterion for MnSimplex is also based on EDM, but the threshold
        is much more lax than that of MIGRAD (see :attr:`Minuit.tol` for details).
        This was made so that SIMPLEX stops early when getting near the minimum. The
        idea is to switch to the more efficient MIGRAD algorithm to finish the
        minimization. Early stopping can be avoided by setting Minuit.tol to an
        accordingly smaller value, however.

        **Arguments:**

            * **ncall**: integer or None, optional; (approximate)
              maximum number of calls before minimization will be aborted. Default: None
              (indicates to use an internal heuristic). Note: The limit may be slightly
              violated, because the condition is checked only after a full iteration of
              the algorithm, which usually performs several function calls.

        **Return:**

            self
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

    def scan(self, ncall=None):
        """Scan over a regular hypercube grid to find the best minimum.

        Scans the function on a regular hypercube grid, whose bounds are defined either
        by parameter limits if present or by Minuit.values +/- Minuit.errors.
        Minuit.errors are initialized to very small values by default, too small for this
        scan. They should be increased before running scan or limits should be set. The
        scan evaluates the function exactly at the limit boundary, so the function should
        be defined there.

        **Arguments:**

            * **ncall**: Approximate number of function calls to spend on the scan. The
              actual number will be close to this, the scan uses ncall^(1/npar) steps per
              cube dimension. If no value is given, a heuristic is used to set ncall.

        **Return:**

            self

        **Notes**

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

    def hesse(self, ncall=None):
        """Run HESSE algorithm to compute asymptotic errors.

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

            self
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
        if fm and fm.state == self._last_state:
            # _last_state not modified, can update _fmin which is more efficient
            hesse(self._fcn, fm, ncall)
            self._last_state = fm.state
            self._fmin = mutil.FMin(fm, self.nfcn, self.ngrad, self._tolerance)
        else:
            # _fmin does not exist or _last_state was modified,
            # so we cannot just update last _fmin
            self._last_state = hesse(self._fcn, self._last_state, ncall)

        if self._last_state.has_covariance is False:
            if not self._fmin:
                raise RuntimeError("HESSE Failed")

        self._make_covariance()

        return self  # return self for method chaining and to autodisplay current state

    def minos(self, *parameters, cl=None, ncall=None):
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

        **Positional arguments**

            Names of parameters to generate Minos errors for. If no positional
            arguments are given, MINOS is run for each parameter.

        **Keyword arguments**

            - **cl**: confidence level for the error interval. Default: None, which
              produces standard 68.3 % confidence intervals. Setting this to another
              value requires the scipy module to be installed. Asymptotically, intervals
              with confidence level cl obtained from repeated identical experiments
              cover the true value with a probability equal to cl.
            - **ncall**: integer or None, limit the number of calls made by MINOS.
              Default: None (uses an internal heuristic by C++ MINUIT).

        **Returns:**

            self
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

    def mnprofile(self, vname, *, size=30, bound=2, subtract_min=False):
        """Calculate MINOS profile around the specified range.

        Scans over **vname** and minimises FCN over the other parameters in each point.

        **Arguments:**

            * **vname** name of variable to scan

            * **size** number of scanning points. Default: 30.

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
        if vname not in self._pos2var:
            raise ValueError("Unknown parameter %s" % vname)

        bound = self._normalize_bound(vname, bound)

        values = np.linspace(bound[0], bound[1], size, dtype=np.double)
        results = np.empty(size, dtype=np.double)
        status = np.empty(size, dtype=np.bool)

        state = MnUserParameterState(self._last_state)  # copy
        ipar = self._var2pos[vname]
        state.fix(ipar)
        for i, v in enumerate(values):
            state.set_value(ipar, v)
            migrad = MnMigrad(self._fcn, state, self.strategy)
            fm = migrad(0, self._tolerance)
            if not fm.is_valid:
                warn(f"MIGRAD fails to converge for {vname}={v}", mutil.IMinuitWarning)
            status[i] = fm.is_valid
            results[i] = fm.fval
        vmin = np.min(results)

        if subtract_min:
            results -= vmin

        return values, results, status

    def draw_mnprofile(
        self, vname, *, size=30, bound=2, subtract_min=False, band=True, text=True
    ):
        """Draw MINOS profile in the specified range.

        It is obtained by finding MIGRAD results with **vname** fixed
        at various places within **bound**.

        **Arguments:**

            * **vname** variable name to scan

            * **size** number of scanning points. Default: 30.

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
        x, y, s = self.mnprofile(
            vname, size=size, bound=bound, subtract_min=subtract_min
        )
        return self._draw_profile(vname, x, y, band, text)

    def profile(self, vname, *, size=100, bound=2, subtract_min=False):
        """Calculate cost function profile around specify range.

        **Arguments:**

            * **vname** variable name to scan

            * **size** number of scanning points. Default: 100.

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
        if subtract_min and not self._fmin:
            raise RuntimeError(
                "Request for minimization "
                "subtraction but no minimization has been done. "
                "Run MIGRAD first."
            )

        bound = self._normalize_bound(vname, bound)

        ipar = self._var2pos[vname]
        scan = np.linspace(bound[0], bound[1], size, dtype=np.double)
        result = np.empty(size, dtype=np.double)
        values = np.array(self.values)
        for i, vi in enumerate(scan):
            values[ipar] = vi
            result[i] = self.fcn(values)
        if subtract_min:
            result -= self.fval
        return scan, result

    def draw_profile(
        self, vname, *, size=100, bound=2, subtract_min=False, band=True, text=True
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

    def contour(self, x, y, *, size=50, bound=2, subtract_min=False):
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

            * **size** number of scanning points. Default: 50.

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
        if subtract_min and not self._fmin:
            raise RuntimeError(
                "Request for minimization "
                "subtraction but no minimization has been done. "
                "Run MIGRAD first."
            )

        try:
            n = float(bound)
            in_sigma = True
        except TypeError:
            in_sigma = False

        if in_sigma:
            x_bound = self._normalize_bound(x, n)
            y_bound = self._normalize_bound(y, n)
        else:
            x_bound = self._normalize_bound(x, bound[0])
            y_bound = self._normalize_bound(y, bound[1])

        x_val = np.linspace(x_bound[0], x_bound[1], size)
        y_val = np.linspace(y_bound[0], y_bound[1], size)

        x_pos = self._var2pos[x]
        y_pos = self._var2pos[y]

        result = np.empty((size, size), dtype=np.double)
        varg = np.array(self.values)
        for i, x in enumerate(x_val):
            varg[x_pos] = x
            for j, y in enumerate(y_val):
                varg[y_pos] = y
                result[i, j] = self._fcn(varg)

        if subtract_min:
            result -= self.fval

        return x_val, y_val, result

    def mncontour(self, x, y, *, cl=None, size=100):
        """Two-dimensional MINOS contour scan.

        This scans over **x** and **y** and minimises all other free
        parameters in each scan point. This works as if **x** and **y** are
        fixed, while the other parameters are minimised by MIGRAD.

        This scan produces a statistical confidence region with the `profile
        likelihood method <https://en.wikipedia.org/wiki/Likelihood_function>`_.
        The contour line represents the values of **x** and **y** where the
        function passes the threshold that corresponds to `sigma` standard
        deviations (note that 1 standard deviations in two dimensions has a
        smaller coverage probability than 68 %).

        The calculation is expensive since it has to run MIGRAD at various
        points.

        **Arguments:**

            - **x** string variable name of the first parameter

            - **y** string variable name of the second parameter

            - **cl** confidence level of the contour. Default: None, which produces a
              68 % contour. Setting this to another value requires the scipy module to
              be installed. Asymptotically, contours with confidence level cl obtained
              from repeated identical experiments cover the true value with a
              probability equal to cl.

            - **size** number of points on the contour to find. Default 100. Increasing
              this makes the contour smoother, but requires more computation time.

        **Returns:**

            contour line, a numpy array of the form [[x1,y1]...[xn,yn]]

        .. seealso::

            :meth:`contour`
            :meth:`mnprofile`

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

    def draw_mncontour(self, x, y, *, cl=None, size=100):
        """Draw MINOS contour.

        **Arguments:**

            - **x**, **y** parameter name

            - **cl** confidence level of the contour. Default: None, which computes a
              68 % contour. Setting this to another value requires the scipy module to
              be installed. Passing a sequence of confidence levels is also allowed.

            - **size** number of points to calculate for each contour. Default: 100.

        **Returns:**

            contour

        .. seealso::

            :meth:`mncontour`

        .. plot:: plots/mncontour.py
            :include-source:
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

    def draw_contour(self, x, y, *, size=50, bound=2):
        """Convenience wrapper for drawing contours.

        The arguments are the same as :meth:`contour`.

        Please read the docs of :meth:`contour` and :meth:`mncontour` to understand the
        difference between the two.

        .. seealso::

            :meth:`contour`
            :meth:`draw_mncontour`

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
