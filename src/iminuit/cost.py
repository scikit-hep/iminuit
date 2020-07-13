"""
Standard cost functions to minimize.
"""

from .util import describe, make_func_code
import numpy as np


def _safe_log(x):
    # does not return NaN for x >= 0
    log_const = 1e-323
    return np.log(x + log_const)


def _sum_log_x(x):
    return np.sum(_safe_log(x))


def _sum_n_log_mu(n, mu):
    return np.sum(n * _safe_log(mu))


def _sum_log_poisson(n, mu):
    return np.sum(mu - n * _safe_log(mu))


def _z_squared(y, ye, ym):
    z = y - ym
    z /= ye
    return z * z


def _sum_z_squared(y, ye, ym):
    return np.sum(_z_squared(y, ye, ym))


def _sum_z_squared_soft_l1(y, ye, ym):
    z = _z_squared(y, ye, ym)
    return np.sum(2 * (np.sqrt(1.0 + z) - 1.0))


try:
    import numba as nb

    jit = nb.njit(nogil=True, parallel=True, cache=True)

    _safe_log = jit(_safe_log)
    _sum_log_x = jit(_sum_log_x)
    _sum_n_log_mu = jit(_sum_n_log_mu)
    _sum_log_poisson = jit(_sum_log_poisson)
    _z_squared = jit(_z_squared)
    _sum_z_squared = jit(_sum_z_squared)
    _sum_z_squared_soft_l1 = jit(_sum_z_squared_soft_l1)

    del jit
    del nb
except ImportError:
    pass


class Cost:
    """Common base class for cost functions.

    **Attributes**

    mask : array-like or None
        If not None, only values selected by the mask are considered. The mask acts on
        the first dimension of a value array, i.e. values[mask]. Default is None.
    verbose : int
        Verbosity level. Default is 0.
    errordef : int
        Error definition constant used by Minuit. For internal use.
    """

    mask = None
    verbose = 0

    @property
    def errordef(self):
        return self._errordef

    def __init__(self, args, verbose, errordef):
        self.func_code = make_func_code(args)
        self.verbose = verbose
        self._errordef = errordef


class UnbinnedNLL(Cost):
    """Unbinned negative log-likelihood.

    Use this if only the shape of the fitted PDF is of interest and the original
    unbinned data is available.
    """

    def __init__(self, data, pdf, verbose=0):
        """
        **Parameters**

        data: array-like
            Sample of observations.

        pdf: callable
            Probability density function of the form f(data, par0, par1, ..., parN),
            where `data` is the data sample and par0, ... parN are model parameters.

        verbose: int, optional
            Verbosity level

            - 0: is no output (default)
            - 1: print current args and negative log-likelihood value
        """
        self.data = np.atleast_1d(data)
        self.pdf = pdf
        Cost.__init__(self, describe(self.pdf)[1:], verbose, 0.5)

    def __call__(self, *args):
        data = self.data if self.mask is None else self.data[self.mask]
        r = -_sum_log_x(self.pdf(data, *args))
        if self.verbose >= 1:
            print(args, "->", r)
        return r


class ExtendedUnbinnedNLL(Cost):
    """Unbinned extended negative log-likelihood.

    Use this if shape and normalization of the fitted PDF are of interest and the
    original unbinned data is available.
    """

    def __init__(self, data, scaled_pdf, verbose=0):
        """
        **Parameters**

        data: array-like
            Sample of observations.

        scaled_pdf: callable
            Scaled probability density function of the form f(data, par0, par1, ...,
            parN), where `data` is the data sample and par0, ... parN are model
            parameters. Must return a tuple (<integral over f in data range>,
            <f evaluated at data points>).

        verbose: int, optional
            Verbosity level

            - 0: is no output (default)
            - 1: print current args and negative log-likelihood value
        """
        self.data = np.atleast_1d(data)
        self.scaled_pdf = scaled_pdf
        Cost.__init__(self, describe(self.scaled_pdf)[1:], verbose, 0.5)

    def __call__(self, *args):
        data = self.data if self.mask is None else self.data[self.mask]
        ns, s = self.scaled_pdf(data, *args)
        r = ns - _sum_log_x(s)
        if self.verbose >= 1:
            print(args, "->", r)
        return r


class BinnedNLL(Cost):
    """Binned negative log-likelihood.

    Use this if only the shape of the fitted PDF is of interest and the data is binned.
    """

    def __init__(self, n, xe, cdf, verbose=0):
        """
        **Parameters**

        n: array-like
            Histogram counts.

        xe: array-like
            Bin edge locations, must be len(n) + 1.

        cdf: callable
            Cumulative density function of the form f(xe, par0, par1, ..., parN),
            where `xe` is a bin edge and par0, ... parN are model parameters.

        verbose: int, optional
            Verbosity level

            - 0: is no output (default)
            - 1: print current args and negative log-likelihood value
        """
        n = np.atleast_1d(n)
        xe = np.atleast_1d(xe)

        if np.any((np.array(n.shape) + 1) != xe.shape):
            raise ValueError("n and xe have incompatible shapes")

        self.n = n
        self.xe = xe
        self.cdf = cdf
        Cost.__init__(self, describe(self.cdf)[1:], verbose, 0.5)

    def __call__(self, *args):
        prob = np.diff(self.cdf(self.xe, *args))
        ma = self.mask
        if ma is None:
            n = self.n
        else:
            n = self.n[ma]
            prob = prob[ma]
        mu = np.sum(n) * prob
        # + np.sum(mu) can be skipped, it is effectively constant
        r = -_sum_n_log_mu(n, mu)
        if self.verbose >= 1:
            print(args, "->", r)
        return r


class ExtendedBinnedNLL(Cost):
    """Binned extended negative log-likelihood.

    Use this if shape and normalization of the fitted PDF are of interest and the data
    is binned.
    """

    def __init__(self, n, xe, scaled_cdf, verbose=0):
        """
        **Parameters**

        n: array-like
            Histogram counts.

        xe: array-like
            Bin edge locations, must be len(n) + 1.

        scaled_cdf: callable
            Scaled Cumulative density function of the form f(xe, par0, par1, ..., parN),
            where `xe` is a bin edge and par0, ... parN are model parameters.

        verbose: int, optional
            Verbosity level

            - 0: is no output (default)
            - 1: print current args and negative log-likelihood value
        """
        n = np.atleast_1d(n)
        xe = np.atleast_1d(xe)

        if np.any((np.array(n.shape) + 1) != xe.shape):
            raise ValueError("n and xe have incompatible shapes")

        self.n = n
        self.xe = xe
        self.scaled_cdf = scaled_cdf
        Cost.__init__(self, describe(self.scaled_cdf)[1:], verbose, 0.5)

    def __call__(self, *args):
        mu = np.diff(self.scaled_cdf(self.xe, *args))
        ma = self.mask
        if ma is None:
            n = self.n
        else:
            n = self.n[ma]
            mu = mu[ma]
        r = _sum_log_poisson(n, mu)
        if self.verbose >= 1:
            print(args, "->", r)
        return r


class LeastSquares(Cost):
    """Least-squares cost function (aka chisquare function).

    Use this if you have data of the form (x, y +/- yerror).
    """

    _loss = None
    _cost = None

    def __init__(self, x, y, yerror, model, loss="linear", verbose=0):
        """
        **Parameters**

        x: array-like
            Locations where the model is evaluated.

        y: array-like
            Observed values. Must have the same length as `x`.

        yerror: array-like or float
            Estimated uncertainty of observed values. Must have same shape as `y` or
            be a scalar, which is then broadcasted to same shape as `y`.

        model: callable
            Function of the form f(x, par0, par1, ..., parN) whose output is compared
            to observed values, where `x` is the location and par0, ... parN are model
            parameters.

        loss: str or callable, optional
            The loss function can be modified to make the fit robust against outliers,
            see scipy.optimize.least_squares for details. Only "linear" (default) and
            "soft_l1" are currently implemented, but users can pass any loss function
            as this argument. It should be a monotonic, twice differentiable function,
            which accepts the squared residual and returns a modified squared residual.

            .. plot:: plots/loss.py

        verbose: int, optional
            Verbosity level

            - 0: is no output (default)
            - 1: print current args and negative log-likelihood value
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        if np.ndim(yerror) == 0:
            yerror = yerror * np.ones_like(y)
        else:
            yerror = np.asarray(yerror)
            if yerror.shape != y.shape:
                raise ValueError("y and yerror must have same shape")

        self.x = x
        self.y = y
        self.yerror = yerror
        self.model = model
        self.loss = loss
        Cost.__init__(self, describe(self.model)[1:], verbose, 1.0)

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss):
        self._loss = loss
        if hasattr(loss, "__call__"):
            self._cost = lambda y, ye, ym: np.sum(loss(_z_squared(y, ye, ym)))
        elif loss == "linear":
            self._cost = _sum_z_squared
        elif loss == "soft_l1":
            self._cost = _sum_z_squared_soft_l1
        else:
            raise ValueError("unknown loss type: " + loss)

    def __call__(self, *args):
        ma = self.mask
        if ma is None:
            x = self.x
            y = self.y
            yerror = self.yerror
        else:
            x = self.x[ma]
            y = self.y[ma]
            yerror = self.yerror[ma]
        ym = self.model(x, *args)
        r = self._cost(y, yerror, ym)
        if self.verbose >= 1:
            print(args, "->", r)
        return r
