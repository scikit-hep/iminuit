"""
Standard cost functions to minimize.
"""

from .util import describe, make_func_code
import numpy as np


def _safe_log(x):
    # does not return NaN for x == 0
    log_const = 1e-323  # pragma: no cover
    return np.log(x + log_const)  # pragma: no cover


def _sum_log_x(x):
    return np.sum(_safe_log(x))  # pragma: no cover


def _neg_sum_n_log_mu(n, mu):
    # subtract n log(n) to keep sum small, required to not loose accuracy in Minuit
    return np.sum(n * _safe_log(n / (mu + 1e-323)))  # pragma: no cover


def _sum_log_poisson(n, mu):
    # subtract n - n log(n) to keep sum small, required to not loose accuracy in Minuit
    return np.sum(mu - n + n * _safe_log(n / (mu + 1e-323)))  # pragma: no cover


def _z_squared(y, ye, ym):
    z = y - ym  # pragma: no cover
    z /= ye  # pragma: no cover
    return z * z  # pragma: no cover


def _sum_z_squared(y, ye, ym):
    return np.sum(_z_squared(y, ye, ym))  # pragma: no cover


def _sum_z_squared_soft_l1(y, ye, ym):
    z = _z_squared(y, ye, ym)  # pragma: no cover
    return np.sum(2 * (np.sqrt(1.0 + z) - 1.0))  # pragma: no cover


try:
    import numba as nb

    jit = nb.njit(nogil=True, parallel=True, cache=True)

    _safe_log = jit(_safe_log)
    _sum_log_x = jit(_sum_log_x)
    _neg_sum_n_log_mu = jit(_neg_sum_n_log_mu)
    _sum_log_poisson = jit(_sum_log_poisson)
    _z_squared = jit(_z_squared)
    _sum_z_squared = jit(_sum_z_squared)
    _sum_z_squared_soft_l1 = jit(_sum_z_squared_soft_l1)

    del jit
    del nb
except ImportError:  # pragma: no cover
    pass  # pragma: no cover


class Cost:
    """Common base class for cost functions.

    **Attributes**

    verbose : int
        Verbosity level. Default is 0.
    errordef : int
        Error definition constant used by Minuit. For internal use.
    """

    __slots__ = "_func_code", "verbose"

    @property
    def errordef(self):
        return 1.0

    @property
    def func_code(self):
        return self._func_code

    def __init__(self, args, verbose):
        self._func_code = make_func_code(args)
        self.verbose = verbose

    def __add__(self, rhs):
        return CostSum(self, rhs)

    def __call__(self, *args):
        r = self._call(args)
        if self.verbose >= 1:
            print(args, "->", r)
        return r


class MaskedCost(Cost):
    """Common base class for cost functions.

    **Attributes**

    mask : array-like or None
        If not None, only values selected by the mask are considered. The mask acts on
        the first dimension of a value array, i.e. values[mask]. Default is None.
    """

    __slots__ = "_mask", "_masked"

    def __init__(self, args, verbose):
        super().__init__(args, verbose)
        self.mask = None

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask
        self._masked = self._make_masked()

    def _make_masked(self):
        return self.data if self._mask is None else self.data[self._mask]


class CostSum(Cost):
    __slots__ = "_items", "_maps"

    def __init__(self, *items):
        tmp = []
        for item in items:
            if isinstance(item, CostSum):
                tmp += item._items
            else:
                tmp.append(item)
        items = tmp
        args, self._maps = self._join_args(items)
        self._items = list(items)
        super().__init__(args, max(c.verbose for c in items))

    def _call(self, args):
        r = 0.0
        for c, m in zip(self._items, self._maps):
            a = tuple(args[mi] for mi in m)
            r += c._call(a)
        return r

    def _join_args(self, costs):
        out_args = []
        in_args = tuple(c._func_code.co_varnames for c in costs)
        for args in in_args:
            for arg in args:
                if arg not in out_args:
                    out_args.append(arg)
        maps = []
        for args in in_args:
            pos = tuple(out_args.index(arg) for arg in args)
            maps.append(pos)
        return tuple(out_args), tuple(maps)


class UnbinnedNLL(MaskedCost):
    """Unbinned negative log-likelihood.

    Use this if only the shape of the fitted PDF is of interest and the original
    unbinned data is available.
    """

    __slots__ = "_data", "_pdf"

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data[:] = value

    @property
    def pdf(self):
        return self._pdf

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
        self._data = _norm(data)
        self._pdf = pdf
        super().__init__(describe(self._pdf)[1:], verbose)

    def _call(self, args):
        data = self._masked
        return -2.0 * _sum_log_x(self._pdf(data, *args))


class ExtendedUnbinnedNLL(MaskedCost):
    """Unbinned extended negative log-likelihood.

    Use this if shape and normalization of the fitted PDF are of interest and the
    original unbinned data is available.
    """

    __slots__ = "_data", "_scaled_pdf"

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data[:] = value

    @property
    def scaled_pdf(self):
        return self._scaled_pdf

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
        self._data = _norm(data)
        self._scaled_pdf = scaled_pdf
        super().__init__(describe(self._scaled_pdf)[1:], verbose)

    def _call(self, args):
        data = self._masked
        ns, s = self._scaled_pdf(data, *args)
        return 2.0 * (ns - _sum_log_x(s))


class BinnedNLL(MaskedCost):
    """Binned negative log-likelihood.

    Use this if only the shape of the fitted PDF is of interest and the data is binned.
    """

    __slots__ = "_n", "_xe", "_cdf"

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n[:] = value

    @property
    def xe(self):
        return self._xe

    @xe.setter
    def xe(self, value):
        self.xe[:] = value

    @property
    def cdf(self):
        return self._cdf

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
        n = _norm(n)
        xe = _norm(xe)

        if np.any((np.array(n.shape) + 1) != xe.shape):
            raise ValueError("n and xe have incompatible shapes")

        self._n = n
        self._xe = xe
        self._cdf = cdf
        super().__init__(describe(self._cdf)[1:], verbose)

    def _call(self, args):
        prob = np.diff(self._cdf(self._xe, *args))
        n = self._masked
        ma = self._mask
        if ma is not None:
            prob = prob[ma]
        mu = np.sum(n) * prob
        # + np.sum(mu) can be skipped, it is effectively constant
        return 2.0 * _neg_sum_n_log_mu(n, mu)

    def _make_masked(self):
        return self._n if self._mask is None else self._n[self._mask]


class ExtendedBinnedNLL(MaskedCost):
    """Binned extended negative log-likelihood.

    Use this if shape and normalization of the fitted PDF are of interest and the data
    is binned.
    """

    __slots__ = "_n", "_xe", "_scaled_cdf"

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n[:] = value

    @property
    def xe(self):
        return self._xe

    @xe.setter
    def xe(self, value):
        self.xe[:] = value

    @property
    def scaled_cdf(self):
        return self._scaled_cdf

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
        n = _norm(n)
        xe = _norm(xe)

        if np.any((np.array(n.shape) + 1) != xe.shape):
            raise ValueError("n and xe have incompatible shapes")

        self._n = n
        self._xe = xe
        self._scaled_cdf = scaled_cdf
        super().__init__(describe(self._scaled_cdf)[1:], verbose)

    def _call(self, args):
        mu = np.diff(self._scaled_cdf(self._xe, *args))
        ma = self._mask
        n = self._masked
        if ma is not None:
            mu = mu[ma]
        return 2.0 * _sum_log_poisson(n, mu)

    def _make_masked(self):
        return self._n if self._mask is None else self._n[self._mask]


class LeastSquares(MaskedCost):
    """Least-squares cost function (aka chisquare function).

    Use this if you have data of the form (x, y +/- yerror).
    """

    __slots__ = "_loss", "_cost", "_x", "_y", "_yerror", "_model"

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
        x = _norm(x)
        y = _norm(y)
        yerror = np.asarray(yerror, dtype=float)

        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        if yerror.ndim == 0:
            yerror = yerror * np.ones_like(y)
        elif yerror.shape != y.shape:
            raise ValueError("y and yerror must have same shape")

        self._x = x
        self._y = y
        self._yerror = yerror
        self._model = model
        self.loss = loss
        super().__init__(describe(self._model)[1:], verbose)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x[:] = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y[:] = value

    @property
    def yerror(self):
        return self._yerror

    @yerror.setter
    def yerror(self, value):
        self._yerror[:] = value

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

    def _call(self, args):
        x, y, yerror = self._masked
        ym = self._model(x, *args)
        return self._cost(y, yerror, ym)

    def _make_masked(self):
        ma = self._mask
        if ma is None:
            return self._x, self._y, self._yerror
        else:
            return self._x[ma], self._y[ma], self._yerror[ma]


class NormalConstraint(Cost):

    __slots__ = "_value", "_cov", "_covinv"

    def __init__(self, args, value, error):
        if isinstance(args, str):
            args = [args]
        self._value = _norm(value)
        self._cov = _norm(error)
        if self._cov.ndim < 2:
            self._cov **= 2
        self._covinv = _covinv(self._cov)
        super().__init__(args, False)

    @property
    def covariance(self):
        return self._cov

    @covariance.setter
    def covariance(self, value):
        self._cov[:] = value
        self._covinv = _covinv(self._cov)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value[:] = value

    def _call(self, args):
        delta = self._value - args
        if self._covinv.ndim < 2:
            return np.sum(delta ** 2 * self._covinv)
        return np.einsum("i,ij,j", delta, self._covinv, delta)


def _norm(value):
    return np.atleast_1d(np.asarray(value, dtype=float))


def _covinv(array):
    return np.linalg.inv(array) if array.ndim == 2 else 1.0 / array
