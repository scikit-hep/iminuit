from iminuit import Minuit
import warnings


class OptimizeResult(dict):
    """Imitation of scipy.optimize.OptimizeResult."""
    def __init__(self, **kwargs):
        dict.__init__(self, **kwargs)
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def minimize(fun, x0, args=(), method=None,
             jac=None, hess=None, hessp=None,
             bounds=None, constraints=(),
             tol=None, callback=None, options=None):
    """Imitates the interface of the SciPy method of the same name in scipy.optimize.

    For a general description of the arguments, see scipy.optimize.minimize.

    The only supported method is 'Migrad'.

    The `options` argument can be used to pass special settings to Minuit.
    All are optional.

    **Options**

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
    import numpy as np

    if method:
        m = method.lower()
        if m != "migrad":
            raise ValueError("Unknown solver " + m)

    nfev = 0
    njev = 0

    class Wrapped:
        fun = None
        jac = None
        args = ()
        callback = None
        nfev = 0
        njev = 0
        np_array = np.array
        def __init__(self, fun, jac, args, callback):
            self.fun = fun
            self.jac = jac
            self.args = args
            self.callback = callback
        def func(self, *args):
            self.nfev += 1
            x = self.np_array(args)
            if self.callback:
                self.callback(x)
            return self.fun(x, *self.args)
        def grad(self, *args):
            self.njev += 1
            x = self.np_array(args)
            return self.jac(x, *self.args)

    wrapped = Wrapped(fun, jac, args, callback)

    if hess or hessp:
        warnings.warn("hess and hessp arguments cannot be handled and are ignored")

    if constraints:
        raise ValueError("Minuit only supports bounds, not constraints")

    if tol:
        warnings.warn("tol argument has no effect on Minuit")

    kwargs = dict(errordef=1, print_level=0)
    if jac:
        kwargs['grad_fcn'] = wrapped.grad

    maxfev = 10000
    eps = None
    if options:
        if "disp" in options:
            kwargs['print_level'] = 1
        if "maxiter" in options:
            warnings.warn("maxiter not supported, acts like maxfev instead")
            maxfev = options["maxiter"]
        if "maxfev" in options:
            maxfev = options["maxfev"]
        if "eps" in options:
            eps = options["eps"]
    if eps is None:
        eps = np.ones(len(x0))

    names = ['%i'%i for i in range(len(x0))]
    for i, x in enumerate(names):
        kwargs[x] = x0[i]
        kwargs['error_'+x] = eps[i]
        if bounds:
            kwargs['limit_'+x] = bounds[i]
    kwargs['forced_parameters'] = names

    m = Minuit(wrapped.func, **kwargs)
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

    return OptimizeResult(x=m.np_values(),
                          success=success,
                          fun=m.fval,
                          hess_inv=m.np_covariance(),
                          message=message,
                          nfev=wrapped.nfev,
                          njev=wrapped.njev,
                          minuit=m,
                          )
