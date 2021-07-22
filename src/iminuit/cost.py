"""
Standard cost functions to minimize.

The cost functions defined here should be preferred over custom implementations. They
have been optimized with knowledge about implementation details of Minuit to give the
highest accucary and the most robust results. They are partially accelerated with numba,
if numba is available.
"""

from .util import describe, make_func_code, merge_signatures, PerformanceWarning
import numpy as np
from collections.abc import Sequence
from typing import Tuple, Callable, Union
import warnings


def _sum_log_x(x):
    return np.sum(np.log(x + 1e-323))


def _spd_transform(n, mu):
    # Scaled Poisson distribution from Bohm and Zech, NIMA 748 (2014) 1-6
    v, var = np.transpose(n)
    s = v / (var + 1e-323)
    return v * s, mu * s


def _log_poisson_part(n, mu):
    # add n log(n) to keep sum small, required to not loose accuracy in Minuit
    return n * (np.log(n + 1e-323) - np.log(mu + 1e-323))


def _sum_log_poisson_part(n, mu):
    if n.ndim == 2:
        n2, mu2 = _spd_transform(n, mu)
    else:
        n2, mu2 = n, mu
    return np.sum(_log_poisson_part(n2, mu2))


def _sum_log_poisson(n, mu):
    if n.ndim == 2:
        n2, mu2 = _spd_transform(n, mu)
    else:
        n2, mu2 = n, mu
    # subtract n to keep sum small, required to not loose accuracy in Minuit
    return np.sum(mu2 - n2 + _log_poisson_part(n2, mu2))


def _z_squared(y, ye, ym):
    z = (y - ym) / ye
    return z * z


def _sum_z_squared(y, ye, ym):
    return np.sum(_z_squared(y, ye, ym))


def _sum_z_squared_soft_l1(y, ye, ym):
    z = _z_squared(y, ye, ym)
    return np.sum(2 * (np.sqrt(1.0 + z) - 1.0))


# If numba is available, use it to accelerate computations in float32 and float64
# precision. Fall back to plain numpy for float128 which is not currently supported
# by numba.
try:
    import numba as nb
    from numba.extending import overload

    _spd_transform_np = _spd_transform

    @overload(_spd_transform, inline="always")
    def _spd_transform_ol(n, mu):
        return _spd_transform_np  # pragma: no cover

    _log_poisson_part_np = _log_poisson_part

    @overload(_log_poisson_part, inline="always")
    def _log_poisson_part_ol(n, mu):
        return _log_poisson_part_np  # pragma: no cover

    _z_squared_np = _z_squared

    @overload(_z_squared, inline="always")
    def _z_squared_ol(y, ye, ym):
        return _z_squared_np  # pragma: no cover

    _sum_log_x_np = _sum_log_x
    _sum_log_x_nb = nb.njit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_sum_log_x_np)

    def _sum_log_x(x):
        if x.dtype in (np.float32, np.float64):
            return _sum_log_x_nb(x)
        return _sum_log_x_np(x)

    _sum_log_poisson_part_np = _sum_log_poisson_part
    _sum_log_poisson_part_nb = nb.njit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_sum_log_poisson_part_np)

    def _sum_log_poisson_part(n, mu):
        if mu.dtype in (np.float32, np.float64):
            return _sum_log_poisson_part_nb(n, mu)
        return _sum_log_poisson_part_np(n, mu)

    _sum_log_poisson_np = _sum_log_poisson
    _sum_log_poisson_nb = nb.njit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_sum_log_poisson_np)

    def _sum_log_poisson(n, mu):
        if mu.dtype in (np.float32, np.float64):
            return _sum_log_poisson_nb(n, mu)
        # fallback to numpy for float128
        return _sum_log_poisson_np(n, mu)

    _sum_z_squared_np = _sum_z_squared
    _sum_z_squared_nb = nb.njit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_sum_z_squared_np)

    def _sum_z_squared(y, ye, ym):
        if ym.dtype in (np.float32, np.float64):
            return _sum_z_squared_nb(y, ye, ym)
        # fallback to numpy for float128
        return _sum_z_squared_np(y, ye, ym)

    _sum_z_squared_soft_l1_np = _sum_z_squared_soft_l1
    _sum_z_squared_soft_l1_nb = nb.njit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_sum_z_squared_soft_l1_np)

    def _sum_z_squared_soft_l1(y, ye, ym):
        if ym.dtype in (np.float32, np.float64):
            return _sum_z_squared_soft_l1_nb(y, ye, ym)
        # fallback to numpy for float128
        return _sum_z_squared_soft_l1_np(y, ye, ym)

    del nb
    del overload
except ModuleNotFoundError:  # pragma: no cover
    pass


class Cost:
    """Base class for all cost functions."""

    __slots__ = "_func_code", "_verbose", "_ndata"

    @property
    def errordef(self):
        """For internal use."""
        return 1.0

    @property
    def func_code(self):
        """For internal use."""
        return self._func_code

    @property
    def verbose(self):
        """
        Access verbosity level.

        Set this to 1 to print all function calls with input and output.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value: int):
        self._verbose = value

    @property
    def ndata(self):
        """
        Return number of points in least-squares fits or bins in a binned fit.

        Infinity is returned if the cost function is unbinned.
        """
        return self._ndata

    def __init__(self, args: Tuple[str], ndata: float, verbose: int):
        """For internal use."""
        self._func_code = make_func_code(args)
        self._verbose = verbose
        self._ndata = ndata

    def __add__(self, rhs):
        """
        Add two cost functions to form a combined cost function.

        Returns
        -------
        CostSum
        """
        return CostSum(self, rhs)

    def __radd__(self, lhs):
        """
        Add two cost functions to form a combined cost function.

        Returns
        -------
        CostSum
        """
        return CostSum(lhs, self)

    def __call__(self, *args):
        """
        Evaluate the cost function.

        If verbose >= 1, print arguments and result.

        Parameters
        ----------
        *args : float
            Parameter values.

        Returns
        -------
        float
        """
        r = self._call(args)
        if self.verbose >= 1:
            print(args, "->", r)
        return r


class Constant(Cost):
    """
    Cost function that represents a constant.

    If your cost function produces results that are far away from O(1), adding a
    constant that brings the value closer to zero may improve the numerical stability.
    """

    __slots__ = "value"

    def __init__(self, value: float):
        """Initialize constant with a value."""
        self.value = value
        super().__init__((), 0.0, False)

    def _call(self, args):
        return self.value


class MaskedCost(Cost):
    """Base class for cost functions that support data masking."""

    __slots__ = "_mask", "_masked"

    def __init__(self, args, ndata, verbose):
        """For internal use."""
        super().__init__(args, ndata, verbose)
        self.mask = None

    @property
    def mask(self):
        """Boolean array, array of indices, or None.

        If not None, only values selected by the mask are considered. The mask acts on
        the first dimension of a value array, i.e. values[mask]. Default is None.
        """
        return self._mask

    @property
    def ndata(self):
        """
        Return number of points in least-squares fits or bins in a binned fit.

        Infinity is returned if the cost function is unbinned.
        """
        n = np.sum(~self._mask) if self._mask is not None else 0
        return self._ndata - n

    @mask.setter
    def mask(self, mask):
        self._mask = None if mask is None else np.asarray(mask)
        self._masked = self._make_masked()

    def _make_masked(self):
        return self._data if self._mask is None else self._data[self._mask]


class CostSum(Cost, Sequence):
    """Sum of cost functions.

    Users do not need to create objects of this class themselves. They should just add
    cost functions, for example::

        nll = UnbinnedNLL(...)
        lsq = LeastSquares(...)
        ncs = NormalConstraint(...)
        csum = nll + lsq + ncs

    CostSum is used to combine data from different experiments or to combine normal cost
    functions with penalty terms (see NormalConstraint).

    The parameters of CostSum are the union of all parameters of its constituents.

    Supports the sequence protocol to access the constituents.

    Warnings
    --------
    CostSum does not work very well with cost functions that accept arrays, because the
    function signature does not allow one to determine how many parameters are accepted
    by the function and which parameters overlap between different cost functions.

    CostSum works with cost functions that accept arrays only under the condition that
    all cost functions accept the very same array parameter:

    1) All array must have the same name in all constituent cost functions.
    2) All arrays must have the same length.
    3) The positions in each array must correspond to the same model parameters.
    """

    __slots__ = "_items", "_maps"

    def __init__(self, *items):
        """Initialize with cost functions.

        Parameters
        ----------
        *items : Cost
            Cost functions. May also be other CostSum functions.
        """
        self._items = []
        for item in items:
            if isinstance(item, CostSum):
                self._items += item._items
            elif isinstance(item, (int, float)):
                if item != 0:
                    self._items.append(Constant(item))
            else:
                self._items.append(item)
        args, self._maps = merge_signatures(self._items)
        super().__init__(
            args, sum(c.ndata for c in self._items), max(c.verbose for c in self._items)
        )

    def _call(self, args):
        r = 0.0
        for c, map in zip(self._items, self._maps):
            c_args = tuple(args[i] for i in map)
            r += c._call(c_args)
        return r

    def __len__(self):
        """Return number of constituent cost functions."""
        return self._items.__len__()

    def __getitem__(self, key):
        """Get constituent cost function by index."""
        return self._items.__getitem__(key)


class UnbinnedCost(MaskedCost):
    """Base class for unbinned cost functions."""

    __slots__ = "_data", "_model"

    @property
    def data(self):
        """Unbinned samples."""
        return self._data

    @data.setter
    def data(self, value):
        self._data[:] = value

    def __init__(self, data, model: Callable, verbose):
        """For internal use."""
        self._data = _norm(data)
        self._model = model
        super().__init__(describe(model)[1:], np.inf, verbose)


class UnbinnedNLL(UnbinnedCost):
    """Unbinned negative log-likelihood.

    Use this if only the shape of the fitted PDF is of interest and the original
    unbinned data is available.
    """

    __slots__ = ()

    @property
    def pdf(self):
        """Get probability density model."""
        return self._model

    def __init__(self, data, pdf: Callable, verbose: int = 0):
        """
        Initialize UnbinnedNLL with data and model.

        Parameters
        ----------
        data : array-like
            Sample of observations.
        pdf : callable
            Probability density function of the form f(data, par0, [par1, ...]),
            where `data` is the data sample and `parN` are model parameters.
        verbose : int, optional
            Verbosity level. 0: is no output (default).
            1: print current args and negative log-likelihood value.
        """
        super().__init__(data, pdf, verbose)

    def _call(self, args):
        data = self._masked
        pdf = self._model(data, *args)
        pdf = _check_model_output(pdf)
        return -2.0 * _sum_log_x(pdf)


class ExtendedUnbinnedNLL(UnbinnedCost):
    """Unbinned extended negative log-likelihood.

    Use this if shape and normalization of the fitted PDF are of interest and the
    original unbinned data is available.
    """

    __slots__ = ()

    @property
    def scaled_pdf(self):
        """Get density model."""
        return self._model

    def __init__(self, data, scaled_pdf: Callable, verbose: int = 0):
        """
        Initialize cost function with data and model.

        Parameters
        ----------
        data : array-like
            Sample of observations.
        scaled_pdf : callable
            Density function of the form f(data, par0, [par1, ...]), where `data` is
            the data sample and `parN` are model parameters. Must return a tuple
            (<integral over f in data range>, <f evaluated at data points>).
        verbose : int, optional
            Verbosity level. 0: is no output (default).
            1: print current args and negative log-likelihood value.
        """
        super().__init__(data, scaled_pdf, verbose)

    def _call(self, args):
        data = self._masked
        ns, spdf = self._model(data, *args)
        spdf = _check_model_output(
            spdf, "Model should return numpy array in second position"
        )
        return 2.0 * (ns - _sum_log_x(spdf))


class BinnedCost(MaskedCost):
    """Base class for binned cost functions."""

    __slots__ = "_n", "_xe", "_model"

    @property
    def n(self):
        """Access bin counts."""
        return self._n

    @n.setter
    def n(self, value):
        self._n[:] = value

    @property
    def xe(self):
        """Access bin edges."""
        return self._xe

    @xe.setter
    def xe(self, value):
        self.xe[:] = value

    def __init__(self, n, xe, model, verbose):
        """For internal use."""
        self._n = _norm(n)
        self._xe = _norm(xe)
        self._model = model

        if self._n.ndim > 2:
            raise ValueError("n must be at most 2-dimensional")

        if self._n.ndim == 2 and self._n.shape[1] != 2:
            raise ValueError("n must shape (N x 2) if 2-dimensional")

        if np.any((np.array(self._n.shape[0]) + 1) != self._xe.shape):
            raise ValueError("n and xe have incompatible shapes")

        super().__init__(describe(model)[1:], len(n), verbose)


class BinnedNLL(BinnedCost):
    """Binned negative log-likelihood.

    Use this if only the shape of the fitted PDF is of interest and the data is binned.
    """

    __slots__ = ()

    @property
    def cdf(self):
        """Get cumulative density function."""
        return self._model

    def __init__(self, n, xe, cdf: Callable, verbose: int = 0):
        """
        Initialize cost function with data and model.

        Parameters
        ----------
        n : array-like
            Histogram counts. If this is an array N x 2, it is interpreted as pairs of
            sum of weights and sum of weights squared.
        xe : array-like
            Bin edge locations, must be len(n) + 1.
        cdf : callable
            Cumulative density function of the form f(xe, par0, par1, ..., parN),
            where `xe` is a bin edge and par0, ... parN are model parameters. Must be
            normalized to unity over the range (xe[0], xe[-1]).
        verbose : int, optional
            Verbosity level. 0: is no output (default).
            1: print current args and negative log-likelihood value.
        """
        super().__init__(n, xe, cdf, verbose)

    def _call(self, args):
        cdf = self._model(self._xe, *args)
        cdf = _check_model_output(cdf)
        prob = np.diff(cdf)
        n = self._masked
        ma = self._mask
        if ma is not None:
            prob = prob[ma]
        mu = np.sum(n) * prob
        # + np.sum(mu) can be skipped, it is effectively constant
        return 2.0 * _sum_log_poisson_part(n, mu)

    def _make_masked(self):
        return self._n if self._mask is None else self._n[self._mask]


class ExtendedBinnedNLL(BinnedCost):
    """Binned extended negative log-likelihood.

    Use this if shape and normalization of the fitted PDF are of interest and the data
    is binned.
    """

    __slots__ = ()

    @property
    def scaled_cdf(self):
        """Get integrated density model."""
        return self._model

    def __init__(self, n, xe, scaled_cdf: Callable, verbose: int = 0):
        """
        Initialize cost function with data and model.

        Parameters
        ----------
        n : array-like
            Histogram counts. If this is an array N x 2, it is interpreted as pairs of
            sum of weights and sum of weights squared.
        xe : array-like
            Bin edge locations, must be len(n) + 1.
        scaled_cdf : callable
            Scaled Cumulative density function of the form f(xe, par0, [par1, ...]),
            where `xe` is a bin edge and `parN` are model parameters.
        verbose : int, optional
            Verbosity level. 0: is no output (default).
            1: print current args and negative log-likelihood value.
        """
        super().__init__(n, xe, scaled_cdf, verbose)

    def _call(self, args):
        scdf = self._model(self._xe, *args)
        scdf = _check_model_output(scdf)
        mu = np.diff(scdf)
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

    @property
    def x(self):
        """Get explanatory variables."""
        return self._x

    @x.setter
    def x(self, value):
        self._x[:] = value

    @property
    def y(self):
        """Get samples."""
        return self._y

    @y.setter
    def y(self, value):
        self._y[:] = value

    @property
    def yerror(self):
        """Get sample uncertainties."""
        return self._yerror

    @yerror.setter
    def yerror(self, value):
        self._yerror[:] = value

    @property
    def model(self):
        """Get model of the form y = f(x, par0, [par1, ...])."""
        return self._model

    @property
    def loss(self):
        """Get loss function."""
        return self._loss

    @loss.setter
    def loss(self, loss: Union[str, Callable]):
        from typing import Callable

        self._loss = loss
        if isinstance(loss, Callable):
            self._cost = lambda y, ye, ym: np.sum(loss(_z_squared(y, ye, ym)))
        elif loss == "linear":
            self._cost = _sum_z_squared
        elif loss == "soft_l1":
            self._cost = _sum_z_squared_soft_l1
        else:
            raise ValueError("unknown loss type: " + loss)

    def __init__(
        self,
        x,
        y,
        yerror,
        model: Callable,
        loss: Union[str, Callable] = "linear",
        verbose: int = 0,
    ):
        """
        Initialize cost function with data and model.

        Parameters
        ----------
        x : array-like
            Locations where the model is evaluated.
        y : array-like
            Observed values. Must have the same length as `x`.
        yerror : array-like or float
            Estimated uncertainty of observed values. Must have same shape as `y` or
            be a scalar, which is then broadcasted to same shape as `y`.
        model : callable
            Function of the form f(x, par0, [par1, ...]) whose output is compared
            to observed values, where `x` is the location and `parN` are model
            parameters.
        loss : str or callable, optional
            The loss function can be modified to make the fit robust against outliers,
            see scipy.optimize.least_squares for details. Only "linear" (default) and
            "soft_l1" are currently implemented, but users can pass any loss function
            as this argument. It should be a monotonic, twice differentiable function,
            which accepts the squared residual and returns a modified squared residual.
        verbose : int, optional
            Verbosity level. 0: is no output (default).
            1: print current args and negative log-likelihood value.

        Notes
        -----
        Alternative loss functions make the fit more robust against outliers by weakening
        the pull of outliers. The mechanical analog of a least-squares fit is a system
        with attractive forces. The points pull the model towards them with a force whose
        potential is given by :math:`rho(z)` for a squared-offset :math:`z`. The plot
        shows the standard potential in comparison with the weaker soft-l1 potential, in
        which outliers act with a constant force independent of their distance.

        .. plot:: plots/loss.py
        """
        x = _norm(x)
        y = _norm(y)

        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        yerror = np.asarray(yerror, dtype=float)
        if yerror.ndim == 0:
            yerror = yerror * np.ones_like(y)
        elif yerror.shape != y.shape:
            raise ValueError("y and yerror must have same shape")

        self._x = x
        self._y = y
        self._yerror = yerror
        self._model = model
        self.loss = loss
        super().__init__(describe(self._model)[1:], len(x), verbose)

    def _call(self, args):
        x, y, yerror = self._masked
        ym = self._model(x, *args)
        ym = _check_model_output(ym)
        return self._cost(y, yerror, ym)

    def _make_masked(self):
        ma = self._mask
        if ma is None:
            return self._x, self._y, self._yerror
        else:
            return self._x[ma], self._y[ma], self._yerror[ma]


class NormalConstraint(Cost):
    """
    Gaussian penalty for one or several parameters.

    The Gaussian penalty acts like a pseudo-measurement of the parameter itself, based
    on a (multi-variate) normal distribution. Penalties can be set for one or
    several parameters at once (which is more efficient). When several parameter are
    constrained, one can specify the full covariance matrix of the parameters.

    Notes
    -----
    It is sometimes necessary to add a weak penalty on a parameter to avoid
    instabilities in the fit. A typical example in high-energy physics is the fit of a
    signal peak above some background. If the amplitude of the peak vanishes, the shape
    parameters of the peak become unconstrained and the fit becomes unstable. This can
    be avoided by adding weak (large uncertainty) penalty on the shape parameters whose
    pull is negligible if the peak amplitude is non-zero.

    This class can also be used to approximately include external measurements of some
    parameters, if the original cost function is not available or too costly to compute.
    If the external measurement was performed in the asymptotic limit with a large
    sample, a Gaussian penalty is an accurate statistical representation of the
    external result.
    """

    __slots__ = "_value", "_cov", "_covinv"

    def __init__(self, args, value, error):
        """
        Initialize the normal constraint with expected value(s) and error(s).

        Parameters
        ----------
        args : str or sequence of str
            Parameter name(s).
        value : float or array-like
            Expected value(s). Must have same length as `args`.
        error : float or array-like
            Expected error(s). If 1D, must have same length as `args`. If 2D, must be
            the covariance matrix of the parameters.
        """
        if isinstance(args, str):
            args = [args]
        self._value = _norm(value)
        self._cov = _norm(error)
        if self._cov.ndim < 2:
            self._cov **= 2
        self._covinv = _covinv(self._cov)
        super().__init__(args, len(self._value), False)

    @property
    def covariance(self):
        """
        Get expected covariance of parameters.

        Can be 1D (diagonal of covariance matrix) or 2D (full covariance matrix).
        """
        return self._cov

    @covariance.setter
    def covariance(self, value):
        self._cov[:] = value
        self._covinv = _covinv(self._cov)

    @property
    def value(self):
        """Get expected parameter values."""
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
    value = np.atleast_1d(value)
    dtype = value.dtype
    if dtype.kind != "f":
        value = value.astype(np.float64)
    return value


def _covinv(array):
    return np.linalg.inv(array) if array.ndim == 2 else 1.0 / array


def _check_model_output(x, msg="Model should return numpy array"):
    if not isinstance(x, np.ndarray):
        warnings.warn(
            f"{msg}, but returns {type(x)}",
            PerformanceWarning,
        )
        return np.array(x)
    return x


del Sequence
del Callable
del Tuple
del Union
