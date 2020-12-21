"""Scipy interface for Minuit."""

from .minuit import Minuit
import warnings
import numpy as np


def minimize(
    fun,
    x0,
    args=(),
    method=None,
    jac=None,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=None,
    tol=None,
    callback=None,
    options=None,
):
    """
    Interface to MIGRAD using the ``scipy.optimize.minimize`` API.

    For a general description of the arguments, see ``scipy.optimize.minimize``.

    The ``method`` argument is ignored. The optimisation is always done using MIGRAD.

    The `options` argument can be used to pass special settings to Minuit.
    All are optional.

    **Options:**

      - *disp* (bool): Set to true to print convergence messages. Default: False.
      - *tol* (float): Tolerance for convergence. Default: None.
      - *maxfun* (int): Maximum allowed number of iterations. Default: None.
      - *maxfev* (int): Deprecated alias for *maxfun*.
      - *eps* (sequence): Initial step size to numerical compute derivative.
        Minuit automatically refines this in subsequent iterations and is very
        insensitive to the initial choice. Default: 1.

    **Returns: OptimizeResult** (dict with attribute access)
      - *x* (ndarray): Solution of optimization.
      - *fun* (float): Value of objective function at minimum.
      - *message* (str): Description of cause of termination.
      - *hess_inv* (ndarray): Inverse of Hesse matrix at minimum (may not be exact).
      - nfev (int): Number of function evaluations.
      - njev (int): Number of jacobian evaluations.
      - minuit (object): Minuit object internally used to do the minimization.
        Use this to extract more information about the parameter errors.
    """
    from scipy.optimize import OptimizeResult, Bounds

    x0 = np.atleast_1d(x0)

    if method not in {None, "migrad"}:
        warnings.warn("method argument is ignored")

    if constraints is not None:
        raise ValueError("Constraints are not supported by Minuit, only bounds")

    if hess or hessp:
        warnings.warn("hess and hessp arguments cannot be handled and are ignored")

    def wrapped(func, args, callback=None):
        if callback is None:
            return lambda x: func(x, *args)

        def f(x):
            callback(x)
            return func(x, *args)

        return f

    wrapped_fun = wrapped(fun, args, callback)
    wrapped_fun.errordef = 0.5  # so that hesse is really second derivative

    if bool(jac):
        if jac is True:
            raise ValueError("jac=True is not supported, only jac=callable")
        assert hasattr(jac, "__call__")
        wrapped_grad = wrapped(jac, args)
    else:
        wrapped_grad = None

    m = Minuit(wrapped_fun, x0, grad=wrapped_grad)
    if bounds is not None:
        if isinstance(bounds, Bounds):
            m.limits = [(a, b) for a, b in zip(bounds.lb, bounds.ub)]
        else:
            m.limits = bounds
    if tol:
        m.tol = tol

    ncall = 0
    if options:
        if "disp" in options:
            m.print_level = 2
        if "maxiter" in options:
            warnings.warn("maxiter not supported, acts like maxfun instead")
            ncall = options["maxiter"]
        if "maxfev" in options:
            warnings.warn(
                "maxfev is deprecated, use maxfun instead", DeprecationWarning
            )
            ncall = options["maxfev"]
        if "maxfun" in options:
            ncall = options["maxfun"]
        if "eps" in options:
            m.errors = options["eps"]

    m.migrad(ncall=ncall)

    message = "Optimization terminated successfully."
    if not m.valid:
        message = "Optimization failed."
        fmin = m.fmin
        if fmin.has_reached_call_limit:
            message += " Call limit was reached."
        if fmin.is_above_max_edm:
            message += " Estimated distance to minimum too large."

    n = len(x0)
    return OptimizeResult(
        x=np.array(m.values),
        success=m.valid,
        fun=m.fval,
        hess_inv=m.covariance if m.covariance is not None else np.ones((n, n)),
        message=message,
        nfev=m.nfcn,
        njev=m.ngrad,
        minuit=m,
    )
