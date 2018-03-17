from iminuit import Minuit
from scipy.optimize import OptimizeResult
from numpy import array

def minimize(fun, x0, method=None, jac=None, bounds=None, callback=None, options=None):
    """Imitates the interface of the SciPy method of the same name in scipy.optimize.

    For argument description, see scipy.optimize.minimize.

    The only supported method is 'Migrad'.
    """
    if method:
        m = method.lower()
        if m != "migrad":
            raise ValueError("Unknown solver " + m)
    names = ['%i'%i for i in range(len(x0))]
    kwargs = {}
    for i, x in enumerate(names):
        kwargs[x] = x0[i]
        kwargs['error_'+x] = 0.1 if x0[i] == 0 else abs(0.1 * x0[i])
        if bounds:
            kwargs['limit_'+x] = bounds[i]

    nfev = 0
    njev = 0

    class Wrapped:
        fun = None
        jac = None
        callback = None
        nfev = 0
        njev = 0
        def __init__(self, fun, jac, callback):
            self.fun = fun
            self.jac = jac
            self.callback = callback
        def func(self, *args):
            self.nfev += 1
            x = array(args)
            if self.callback:
                self.callback(x)
            return self.fun(x)
        def grad(self, *args):
            self.njev += 1
            x = array(args)
            return self.jac(x)

    wrapped = Wrapped(fun, jac, callback)

    print_level = 0
    maxiter = 10000
    if options:
        if "disp" in options:
            print_level = 1
        if "maxiter" in options:
            maxiter = options["maxiter"]

    m = Minuit(wrapped.func,
               grad_fcn=wrapped.grad if jac else None,
               forced_parameters=names,
               print_level=0, errordef=1,
               **kwargs)
    m.migrad(ncall=maxiter)

    return OptimizeResult(x=m.np_values(),
                          success=m.migrad_ok(),
                          fun=m.fval,
                          hess_inv=m.np_covariance(),
                          nfev=wrapped.nfev,
                          njev=wrapped.njev)
