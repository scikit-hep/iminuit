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


class UnbinnedNLL:
    """Unbinned negative log-likelihood.

    Use this if only the shape of the fitted PDF is of interest and the original
    unbinned data is available.
    """

    verbose = False
    errordef = 0.5

    def __init__(self, data, pdf, verbose=0):
        """
        Parameters
        ----------
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
        self.verbose = verbose
        self.func_code = make_func_code(describe(self.pdf)[1:])

    def __call__(self, *args):
        r = -_sum_log_x(self.pdf(self.data, *args))
        if self.verbose >= 1:
            print(args, "->", r)
        return r


class ExtendedUnbinnedNLL:
    """Unbinned extended negative log-likelihood.

    Use this if shape and normalization of the fitted PDF are of interest and the
    original unbinned data is available.
    """

    verbose = False
    errordef = 0.5

    def __init__(self, data, scaled_pdf, verbose=0):
        """
        Parameters
        ----------
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
        self.data = data
        self.scaled_pdf = scaled_pdf
        self.verbose = verbose
        self.func_code = make_func_code(describe(self.scaled_pdf)[1:])

    def __call__(self, *args):
        ns, s = self.scaled_pdf(self.data, *args)
        r = ns - _sum_log_x(s)
        if self.verbose >= 1:
            print(args, "->", r)
        return r


class BinnedNLL:
    """Binned negative log-likelihood.

    Use this if only the shape of the fitted PDF is of interest and the data is binned.
    """

    verbose = False
    errordef = 0.5

    def __init__(self, n, xe, cdf, verbose=0):
        """
        Parameters
        ----------
        n: array-like
            Histogram counts.
        xe: array-like
            Bin edge locations, must be len(n) + 1.
        cdf: callable
            Cumulative density function of the form f(x, par0, par1, ..., parN),
            where `x` is the observation value and par0, ... parN are model parameters.
        verbose: int, optional
            Verbosity level
            - 0: is no output (default)
            - 1: print current args and negative log-likelihood value
        """
        self.n = n
        self.xe = xe
        self.tot = np.sum(n)
        self.cdf = cdf
        self.verbose = verbose
        self.func_code = make_func_code(describe(self.cdf)[1:])

    def __call__(self, *args):
        c = self.cdf(self.xe, *args)
        mu = self.tot * np.diff(c)
        # + np.sum(mu) can be skipped, it is effectively constant
        r = -_sum_n_log_mu(self.n, mu)
        if self.verbose >= 1:
            print(args, "->", r)
        return r


class ExtendedBinnedNLL:
    """Binned extended negative log-likelihood.

    Use this if shape and normalization of the fitted PDF are of interest and the data
    is binned.
    """

    verbose = False
    errordef = 0.5

    def __init__(self, n, xe, scaled_cdf, verbose=0):
        """
        Parameters
        ----------
        n: array-like
            Histogram counts.
        xe: array-like
            Bin edge locations, must be len(n) + 1.
        scaled_cdf: callable
            Scaled Cumulative density function of the form f(x, par0, par1, ..., parN),
            where `x` is the observation value and par0, ... parN are model parameters.
        verbose: int, optional
            Verbosity level
            - 0: is no output (default)
            - 1: print current args and negative log-likelihood value
        """
        self.n = n
        self.xe = xe
        self.scaled_cdf = scaled_cdf
        self.verbose = verbose
        self.func_code = make_func_code(describe(self.scaled_cdf)[1:])

    def __call__(self, *args):
        mu = np.diff(self.scaled_cdf(self.xe, *args))
        r = _sum_log_poisson(self.n, mu)
        if self.verbose >= 1:
            print(args, "->", r)
        return r


class LeastSquares:
    """Least-squares cost function (aka chisquare function).

    Use this if you have data of the form (x, y +/- yerror).
    """

    verbose = False
    errordef = 1.0

    def __init__(self, x, y, yerror, model, loss="linear", verbose=0):
        """
        Parameters
        ----------
        x: array-like
            Locations where the model is evaluated.
        y: array-like
            Observed values.
        yerror: array-like
            Estimated uncertainty of observed values.
        model: callable
            Function of the form f(x, par0, par1, ..., parN) whose output is compared
            to observed values, where `x` is the location and par0, ... parN are model
            parameters.
        loss: str or callable, optional
            See scipyl.optimize.least_squares.
            Only "soft_l1" and "linear" (default) are currently recognized.
        verbose: int, optional
            Verbosity level
            - 0: is no output (default)
            - 1: print current args and negative log-likelihood value
        """
        self.x = x
        self.y = y
        self.yerror = yerror
        self.model = model
        if hasattr(loss, "__call__"):
            self.cost = lambda y, ye, ym: np.sum(loss(_z_squared(y, ye, ym)))
        elif loss == "linear":
            self.cost = _sum_z_squared
        elif loss == "soft_l1":
            self.cost = _sum_z_squared_soft_l1
        else:
            raise ValueError("unknown loss type: " + loss)
        self.verbose = verbose
        self.func_code = make_func_code(describe(self.model)[1:])

    def __call__(self, *args):
        r = self.cost(self.y, self.yerror, self.model(self.x, *args))
        if self.verbose >= 1:
            print(args, "->", r)
        return r
