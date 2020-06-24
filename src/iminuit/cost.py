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


try:
    import numba as nb

    jit = nb.njit(nogil=True, parallel=True, cache=True)

    _safe_log = jit(_safe_log)
    _sum_log_x = jit(_sum_log_x)
    _sum_n_log_mu = jit(_sum_n_log_mu)
    _sum_log_poisson = jit(_sum_log_poisson)

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
        self.func_code = make_func_code(describe(self.pdf)[1:])
        self.verbose = verbose

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
        self.func_code = make_func_code(describe(self.scaled_pdf)[1:])
        self.verbose = verbose

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
        self.func_code = make_func_code(describe(self.cdf)[1:])
        self.verbose = verbose

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
        self.func_code = make_func_code(describe(self.scaled_cdf)[1:])
        self.verbose = verbose

    def __call__(self, *args):
        mu = np.diff(self.scaled_cdf(self.xe, *args))
        r = _sum_log_poisson(self.n, mu)
        if self.verbose >= 1:
            print(args, "->", r)
        return r
