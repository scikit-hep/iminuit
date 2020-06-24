from .util import describe, make_func_code
import numpy as np


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
            - 1: print args and negative log-likelihood value
            - 2: Like 1, but also show all log-probabilities.
        """
        self.data = np.atleast_1d(data)
        self.pdf = pdf
        self.func_code = make_func_code(describe(self.pdf)[1:])
        self.verbose = verbose

    def __call__(self, *args):
        log_p = np.log(self.pdf(self.data, *args))
        r = -np.sum(log_p)
        if self.verbose >= 1:
            print(args, "->", r)
        if self.verbose == 2:
            print(log_p)
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
            - 1: print args and negative log-likelihood value
            - 2: Like 1, but also show all expectations per bin.
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
        r = -np.sum(self.n * np.log(mu))
        if self.verbose >= 1:
            print(args, "->", r)
        if self.verbose == 2:
            print(mu)
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
            parameters.
        verbose: int, optional
            Verbosity level
            - 0: is no output (default)
            - 1: print args and negative log-likelihood value
            - 2: Like 1, but also show all log-probabilities.
        """
        self.data = data
        self.scaled_pdf = scaled_pdf
        self.func_code = make_func_code(describe(self.scaled_pdf)[1:])
        self.verbose = verbose

    def __call__(self, *args):
        log_s = np.log(self.scaled_pdf(self.data, *args))
        r = self.scaled_pdf.integral(*args) - np.sum(log_s)
        if self.verbose >= 1:
            print(args, "->", r)
        if self.verbose == 2:
            print(log_p)
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
            - 1: print args and negative log-likelihood value
            - 2: Like 1, but also show all expectations per bin.
        """
        self.n = n
        self.xe = xe
        self.scaled_cdf = scaled_cdf
        self.func_code = make_func_code(describe(self.scaled_cdf)[1:])
        self.verbose = verbose

    def __call__(self, *args):
        c = self.scaled_cdf(self.xe, *args)
        mu_tot = self.scaled_cdf(float("infinity"), *args)
        mu = np.diff(c)
        r = mu_tot - np.sum(self.n * np.log(mu))
        if self.verbose >= 1:
            print(args, "->", r)
        if self.verbose == 2:
            print(mu)
        return r
