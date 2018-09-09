from iminuit import Minuit
import warnings
import numpy as np


def minimize(fun, x0, args=(), method=None,
             jac=None, hess=None, hessp=None,
             bounds=None, constraints=None,
             tol=None, callback=None, options=None):
    """An interface to MIGRAD using the ``scipy.optimize.minimize`` API.

    For a general description of the arguments, see ``scipy.optimize.minimize``.

    The ``method`` argument is ignored. The optimisation is always done using MIGRAD.

    The `options` argument can be used to pass special settings to Minuit.
    All are optional.

    **Options:**

      - *disp* (bool): Set to true to print convergence messages. Default: False.
      - *maxfev* (int): Maximum allowed number of iterations. Default: 10000.
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
    from scipy.optimize import OptimizeResult

    if method not in {None, 'migrad'}:
        warnings.warn("method argument is ignored")

    if constraints is not None:
        raise ValueError("Constraints are not supported by Minuit, only bounds")

    if hess or hessp:
        warnings.warn("hess and hessp arguments cannot be handled and are ignored")

    if tol:
        warnings.warn("tol argument has no effect on Minuit")

    def wrapped(func, args, callback=None):
        if callback is None:
            return lambda x: func(x, *args)
        else:
            def f(x):
                callback(x)
                return func(x, *args)
            return f
    wrapped_fun = wrapped(fun, args, callback)

    maxfev = 10000
    error = None
    kwargs = {'print_level': 0, 'errordef': 1}
    if options:
        if "disp" in options:
            kwargs['print_level'] = 1
        if "maxiter" in options:
            warnings.warn("maxiter not supported, acts like maxfev instead")
            maxfev = options["maxiter"]
        if "maxfev" in options:
            maxfev = options["maxfev"]
        if "eps" in options:
            error = options["eps"]

    # prevent warnings from Minuit about missing initial step
    if error is None:
        error = np.ones_like(x0)

    if bool(jac):
        if jac is True:
            raise ValueError("jac=True is not supported, only jac=callable")
        assert hasattr(jac, "__call__")
        wrapped_grad = wrapped(jac, args)
    else:
        wrapped_grad = None

    m = Minuit.from_array_func(wrapped_fun,
                               x0,
                               error=error,
                               limit=bounds,
                               grad=wrapped_grad,
                               **kwargs)
    m.migrad(ncall=maxfev)

    message = "Optimization terminated successfully."
    success = m.migrad_ok()
    if not success:
        message = "Optimization failed."
        fmin = m.get_fmin()
        if fmin.has_reached_call_limit:
            message += " Call limit was reached."
        if fmin.is_above_max_edm:
            message += " Estimated distance to minimum too large."

    return OptimizeResult(
        x=m.np_values(),
        success=success,
        fun=m.fval,
        hess_inv=m.np_covariance(),
        message=message,
        nfev=m.get_num_call_fcn(),
        njev=m.get_num_call_grad(),
        minuit=m,
    )
