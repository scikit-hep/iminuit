"""
Standard cost functions to minimize for statistical fits.

We provide these for convenience, so that you do not have to write your own for standard
fits. The cost functions optionally use Numba to accelerate some calculations, if Numba is
installed.

What to use when
----------------
- Fit a normalised probability density to data

    - Data are not binned: :class:`UnbinnedNLL`
    - Data are binned: :class:`BinnedNLL`, also supports histogram of weighted samples

- Fit a density to data, density is not normalised

    - Data are not binned: :class:`ExtendedUnbinnedNLL`
    - Data are binned: :class:`ExtendedBinnedNLL`, also supports
      histogram of weighted samples

- Fit a template to binned data with bin-wise uncertainties on the template:
  :class:`BarlowBeestonLite`, which also supports weighted data and weighted templates

- Fit of a function f(x) to (x, y, yerror) pairs with normal-distributed fluctuations. x
  is one- or multi-dimensional, y is one-dimensional.

    - y values contain no outliers: :class:`LeastSquares`
    - y values contain outliers: :class:`LeastSquares` with loss function set to "soft_l1"

- Include constraints from external fits or apply regularisation:
  :class:`NormalConstraint`

Combining cost functions
------------------------
All cost functions can be added, which generates a new combined cost function. Parameters
with the same name are shared between component cost functions. Use this to constrain one
or several parameters with different data sets and using different statistical models for
each data set. Gaussian penalty terms can also be added to the cost function to introduce
external knowledge about a parameter.

Notes
-----
The cost functions defined here have been optimized with knowledge about implementation
details of Minuit to give the highest accucary and the most robust results, so they should
perform well. If you have trouble with your own implementations, try these.

The binned versions of the log-likelihood fits support weighted samples. For each bin of
the histogram, the sum of weights and the sum of squared weights is needed then, see class
documentation for details.
"""

from .util import (
    describe,
    make_func_code,
    merge_signatures,
    PerformanceWarning,
    _smart_sampling,
)
import numpy as np
from collections.abc import Sequence
import abc
import typing as _tp
import warnings

# correct ArrayLike from numpy.typing generates horrible looking signatures
# python's help(), so we use this as a workaround
_ArrayLike = _tp.Collection


def _safe_log(x):
    # guard against x = 0
    return np.log(x + 1e-323)


def _unbinned_nll(x):
    return -np.sum(_safe_log(x))


def _z_squared(y, ye, ym):
    z = (y - ym) / ye
    return z * z


def _soft_l1_loss(z_sqr):
    return np.sum(2 * (np.sqrt(1 + z_sqr) - 1))


def _soft_l1_cost(y, ye, ym):
    return _soft_l1_loss(_z_squared(y, ye, ym))


def _replace_none(x, replacement):
    if x is None:
        return replacement
    return x


class BohmZechTransform:
    """
    Apply Bohm-Zech transform.

    See Bohm and Zech, NIMA 748 (2014) 1-6.
    """

    def __init__(self, val: _ArrayLike, var: _ArrayLike):
        """
        Initialize transformer with data value and variance.

        Parameters
        ----------
        val : array-like
            Observed values.
        var : array-like
            Estimated variance of observed values.
        """
        val, var = np.atleast_1d(val, var)
        self._scale = val / (var + 1e-323)
        self._obs = val * self._scale

    def __call__(self, val: _ArrayLike, var: _tp.Optional[_ArrayLike] = None):
        """
        Return precomputed scaled data and scaled prediction.

        Parameters
        ----------
        val : array-like
            Predicted values.
        var : array-like, optional
            Predicted variance.

        Returns
        -------
        (obs, pred) or (obs, pred, pred_var)
        """
        s = self._scale
        if var is None:
            return self._obs, val * s
        return self._obs, val * s, var * s**2


def chi2(y: _ArrayLike, ye: _ArrayLike, ym: _ArrayLike):
    """
    Compute (potentially) chi2-distributed cost.

    The value returned by this function is chi2-distributed, if the observed values are
    normally distributed around the expected values with the provided standard deviations.

    Parameters
    ----------
    y : array-like
        Observed values.
    ye : array-like
        Uncertainties of values.
    ym : array-like
        Expected values.

    Returns
    -------
    float
        Const function value.
    """
    y, ye, ym = np.atleast_1d(y, ye, ym)
    return np.sum(_z_squared(y, ye, ym))


def multinominal_chi2(n: _ArrayLike, mu: _ArrayLike):
    """
    Compute asymptotically chi2-distributed cost for binomially-distributed data.

    See Baker & Cousins, NIM 221 (1984) 437-442.

    Parameters
    ----------
    n : array-like
        Observed counts.
    mu : array-like
        Expected counts per bin. Must satisfy sum(mu) == sum(n).

    Returns
    -------
    float
        Cost function value.

    Notes
    -----
    The implementation makes the result asymptotically chi2-distributed and
    keeps the sum small near the minimum, which helps to maximise the numerical
    accuracy for Minuit.
    """
    n, mu = np.atleast_1d(n, mu)
    return 2 * np.sum(n * (_safe_log(n) - _safe_log(mu)))


def poisson_chi2(n: _ArrayLike, mu: _ArrayLike):
    """
    Compute asymptotically chi2-distributed cost for Poisson-distributed data.

    See Baker & Cousins, NIM 221 (1984) 437-442.

    Parameters
    ----------
    n : array-like
        Observed counts.
    mu : array-like
        Expected counts.

    Returns
    -------
    float
        Cost function value.

    Notes
    -----
    The implementation makes the result asymptotically chi2-distributed,
    which helps to maximise the numerical accuracy for Minuit.
    """
    n, mu = np.atleast_1d(n, mu)
    return 2 * np.sum(mu - n + n * (_safe_log(n) - _safe_log(mu)))


def barlow_beeston_lite_chi2_jsc(n: _ArrayLike, mu: _ArrayLike, mu_var: _ArrayLike):
    """
    Compute asymptotically chi2-distributed cost for a template fit.

    J.S. Conway, PHYSTAT 2011, https://doi.org/10.48550/arXiv.1103.0354

    Parameters
    ----------
    n : array-like
        Observed counts.
    mu : array-like
        Expected counts. This is the sum of the normalised templates scaled with
        the component yields. Must be positive everywhere.
    mu_var : array-like
        Expected variance of mu. Must be positive everywhere.

    Returns
    -------
    float
        Cost function value.

    Notes
    -----
    The implementation deviates slightly from the paper by making the result
    asymptotically chi2-distributed, which helps to maximise the numerical
    accuracy for Minuit.
    """
    beta_var = mu_var / mu**2

    # need to solve quadratic equation b^2 + (mu beta_var - 1) b - n beta_var = 0
    p = mu * beta_var - 1
    q = -n * beta_var
    beta = 0.5 * (-p + np.sqrt(p**2 - 4 * q))

    return poisson_chi2(n, mu * beta) + np.sum((beta - 1) ** 2 / beta_var)


def barlow_beeston_lite_chi2_hpd(n: _ArrayLike, mu: _ArrayLike, mu_var: _ArrayLike):
    """
    Compute asymptotically chi2-distributed cost for a template fit.

    H.P. Dembinski, https://doi.org/10.48550/arXiv.2206.12346

    Parameters
    ----------
    n : array-like
        Observed counts.
    mu : array-like
        Expected counts. This is the sum of the normalised templates scaled
        with the component yields.
    mu_var : array-like
        Expected variance of mu. Must be positive everywhere.

    Returns
    -------
    float
        Cost function value.
    """
    k = mu**2 / mu_var
    beta = (n + k) / (mu + k)
    return poisson_chi2(n, mu * beta) + poisson_chi2(k, k * beta)


# If numba is available, use it to accelerate computations in float32 and float64
# precision. Fall back to plain numpy for float128 which is not currently supported
# by numba.
try:
    from numba import njit as _njit
    from numba.extending import overload as _overload

    @_overload(_safe_log, inline="always")
    def _ol_safe_log(x):
        return _safe_log

    @_overload(_z_squared, inline="always")
    def _ol_z_squared(y, ye, ym):
        return _z_squared

    _unbinned_nll_np = _unbinned_nll
    _unbinned_nll_nb = _njit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_unbinned_nll_np)

    def _unbinned_nll(x):
        if x.dtype in (np.float32, np.float64):
            return _unbinned_nll_nb(x)
        # fallback to numpy for float128
        return _unbinned_nll_np(x)

    _multinominal_chi2_np = multinominal_chi2
    _multinominal_chi2_nb = _njit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_multinominal_chi2_np)

    def multinominal_chi2(n, p):  # noqa
        if p.dtype in (np.float32, np.float64):
            return _multinominal_chi2_nb(n, p)
        # fallback to numpy for float128
        return _multinominal_chi2_np(n, p)

    multinominal_chi2.__doc__ = _multinominal_chi2_np.__doc__

    _poisson_chi2_np = poisson_chi2
    _poisson_chi2_nb = _njit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_poisson_chi2_np)

    def poisson_chi2(n, mu):  # noqa
        if mu.dtype in (np.float32, np.float64):
            return _poisson_chi2_nb(n, mu)
        # fallback to numpy for float128
        return _poisson_chi2_np(n, mu)

    poisson_chi2.__doc__ = _poisson_chi2_np.__doc__

    _chi2_np = chi2
    _chi2_nb = _njit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_chi2_np)

    def chi2(y, ye, ym):  # noqa
        if ym.dtype in (np.float32, np.float64):
            return _chi2_nb(y, ye, ym)
        # fallback to numpy for float128
        return _chi2_np(y, ye, ym)

    chi2.__doc__ = _chi2_np.__doc__

    _soft_l1_loss_np = _soft_l1_loss
    _soft_l1_loss_nb = _njit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_soft_l1_loss_np)

    def _soft_l1_loss(z_sqr):
        if z_sqr.dtype in (np.float32, np.float64):
            return _soft_l1_loss_nb(z_sqr)
        # fallback to numpy for float128
        return _soft_l1_loss_np(z_sqr)

    @_overload(_soft_l1_loss, inline="always")
    def _ol_soft_l1_loss(z_sqr):
        return _soft_l1_loss_np

    _soft_l1_cost_np = _soft_l1_cost
    _soft_l1_cost_nb = _njit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_soft_l1_cost_np)

    def _soft_l1_cost(y, ye, ym):
        if ym.dtype in (np.float32, np.float64):
            return _soft_l1_cost_nb(y, ye, ym)
        # fallback to numpy for float128
        return _soft_l1_cost_np(y, ye, ym)

except ModuleNotFoundError:
    pass


class Cost(abc.ABC):
    """Base class for all cost functions."""

    __slots__ = ("_func_code", "_verbose")

    @property
    def errordef(self):
        """For internal use."""
        return 1.0

    @property
    def func_code(self):
        """For internal use."""
        return self._func_code

    @property
    @abc.abstractmethod
    def ndata(self):
        """
        Return number of points in least-squares fits or bins in a binned fit.

        Infinity is returned if the cost function is unbinned. This is used by Minuit to
        compute the reduced chi2, a goodness-of-fit estimate.
        """
        NotImplemented  # pragma: no cover

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

    def __init__(self, args: _tp.Tuple[str, ...], verbose: int):
        """For internal use."""
        self._func_code = make_func_code(args)
        self._verbose = verbose

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

    If your cost function produces results that are far away from O(1), adding a constant
    that brings the value closer to zero may improve the numerical stability.
    """

    __slots__ = "value"

    def __init__(self, value: float):
        """Initialize constant with a value."""
        self.value = value
        super().__init__((), False)

    @Cost.ndata.getter  # type:ignore
    def ndata(self):
        """See Cost.ndata."""
        return 0

    def _call(self, args):
        return self.value


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
    CostSum does not support cost functions that accept a parameter array, because the
    function signature does not allow one to determine how many parameters are accepted by
    the function and which parameters overlap between different cost functions.
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
        super().__init__(args, max(c.verbose for c in self._items))

    def _split(self, args):
        for component, cmap in zip(self._items, self._maps):
            component_args = tuple(args[i] for i in cmap)
            yield component, component_args

    def _call(self, args):
        r = 0.0
        for comp, cargs in self._split(args):
            r += comp._call(cargs)
        return r

    @Cost.ndata.getter  # type:ignore
    def ndata(self):
        """See Cost.ndata."""
        return sum(c.ndata for c in self._items)

    def __len__(self):
        """Return number of constituent cost functions."""
        return self._items.__len__()

    def __getitem__(self, key):
        """Get constituent cost function by index."""
        return self._items.__getitem__(key)

    def visualize(
        self, args: _ArrayLike, component_kwargs: _tp.Dict[int, _tp.Dict] = None
    ):
        """
        Visualize data and model agreement (requires matplotlib).

        The visualization is drawn with matplotlib.pyplot into the current figure.
        Subplots are created to visualize each part of the cost function, the figure
        height is increased accordingly. Parts without a visualize method are silently
        ignored.

        Parameters
        ----------
        args : array-like
            Parameter values.
        component_kwargs : dict of dicts, optional
            Dict that maps an index to dict of keyword arguments. This can be
            used to pass keyword arguments to a visualize method of a component with
            that index.
        **kwargs :
            Other keyword arguments are forwarded to all components.
        """
        from matplotlib import pyplot as plt

        args = np.atleast_1d(args)

        n = sum(hasattr(comp, "visualize") for comp in self)
        fig = plt.gcf()
        if n > 1:
            fig.set_figheight(n * fig.get_figheight())

        if component_kwargs is None:
            component_kwargs = {}
        i = 0
        for k, (comp, cargs) in enumerate(self._split(args)):
            if hasattr(comp, "visualize"):
                i += 1
                plt.subplot(n, 1, i)
                kwargs = component_kwargs.get(k, {})
                comp.visualize(cargs, **kwargs)


class MaskedCost(Cost):
    """Base class for cost functions that support data masking."""

    __slots__ = "_data", "_mask", "_masked"

    def __init__(self, args, data, verbose):
        """For internal use."""
        self._data = data
        self._mask = None
        self._update_cache()
        Cost.__init__(self, args, verbose)

    @property
    def mask(self):
        """Boolean array, array of indices, or None.

        If not None, only values selected by the mask are considered. The mask acts on
        the first dimension of a value array, i.e. values[mask]. Default is None.
        """
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = None if mask is None else np.asarray(mask)
        self._update_cache()

    @property
    def data(self):
        """Return data samples."""
        return self._data

    @data.setter
    def data(self, value):
        self._data[...] = value
        self._update_cache()

    def _update_cache(self):
        self._masked = self._data[_replace_none(self._mask, ...)]


class UnbinnedCost(MaskedCost):
    """Base class for unbinned cost functions."""

    __slots__ = "_model", "_log"

    @Cost.ndata.getter  # type:ignore
    def ndata(self):
        """See Cost.ndata."""
        # unbinned likelihoods have infinite degrees of freedom
        return np.inf

    def __init__(self, data, model: _tp.Callable, verbose: int, log: bool):
        """For internal use."""
        self._model = model
        self._log = log
        super().__init__(describe(model)[1:], _norm(data), verbose)

    def visualize(self, args: _ArrayLike, model_points: int = 0):
        """
        Visualize data and model agreement (requires matplotlib).

        The visualization is drawn with matplotlib.pyplot into the current axes.

        Parameters
        ----------
        args : array-like
            Parameter values.
        model_points : int, optional
            How many points to use to draw the model. Default is 0, in this case
            an smart sampling algorithm selects the number of points.
        """
        from matplotlib import pyplot as plt

        if self.data.ndim > 1:
            raise ValueError("visualize is not implemented for multi-dimensional data")

        n, xe = np.histogram(self.data, bins=50)
        cx = 0.5 * (xe[1:] + xe[:-1])
        plt.errorbar(cx, n, n**0.5, fmt="ok")
        if model_points > 0:
            if xe[0] > 0 and xe[-1] / xe[0] > 1e2:
                xm = np.geomspace(xe[0], xe[-1], model_points)
            else:
                xm = np.linspace(xe[0], xe[-1], model_points)
            ym = self.scaled_pdf(xm, *args)
        else:
            xm, ym = _smart_sampling(lambda x: self.scaled_pdf(x, *args), xe[0], xe[-1])
        dx = xe[1] - xe[0]
        plt.fill_between(xm, 0, ym * dx, fc="C0")


class UnbinnedNLL(UnbinnedCost):
    """Unbinned negative log-likelihood.

    Use this if only the shape of the fitted PDF is of interest and the original
    unbinned data is available. The data can be one- or multi-dimensional.
    """

    __slots__ = ()

    @property
    def pdf(self):
        """Get probability density model."""
        if self._log:
            return lambda *args: np.exp(self._model(*args))
        return self._model

    @property
    def scaled_pdf(self):
        """Get probability density model."""
        scale = np.prod(self.data.shape)
        if self._log:
            return lambda *args: scale * np.exp(self._model(*args))
        return lambda *args: scale * self._model(*args)

    def __init__(
        self,
        data: _ArrayLike,
        pdf: _tp.Callable,
        verbose: int = 0,
        log: bool = False,
    ):
        """
        Initialize UnbinnedNLL with data and model.

        Parameters
        ----------
        data : array-like
            Sample of observations. If the observations are multidimensional, data must
            have the shape (D, N), where D is the number of dimensions and N the number of
            data points.
        pdf : callable
            Probability density function of the form f(data, par0, [par1, ...]), where
            data is the data sample and par0, ... are model parameters. If the data are
            multivariate, data passed to f has shape (D, N), where D is the number of
            dimensions and N the number of data points.
        verbose : int, optional
            Verbosity level. 0: is no output (default). 1: print current args and negative
            log-likelihood value.
        log : bool, optional
            Distributions of the exponential family (normal, exponential, poisson, ...)
            allow one to compute the logarithm of the pdf directly, which is more accurate
            and efficient than effectively doing ``log(exp(logpdf))``. Set this to True,
            if the model returns the logarithm of the pdf instead of the pdf. Default is
            False.
        """
        super().__init__(data, pdf, verbose, log)

    def _call(self, args):
        data = self._masked
        x = self._model(data, *args)
        x = _normalize_model_output(x)
        if self._log:
            return -2.0 * np.sum(x)
        return 2.0 * _unbinned_nll(x)


class ExtendedUnbinnedNLL(UnbinnedCost):
    """Unbinned extended negative log-likelihood.

    Use this if shape and normalization of the fitted PDF are of interest and the original
    unbinned data is available. The data can be one- or multi-dimensional.
    """

    __slots__ = ()

    @property
    def pdf(self):
        """Get probability density model."""
        if self._log:

            def fn(*args):
                n, x = self._model(*args)
                return np.exp(x) / n

        else:

            def fn(*args):
                n, x = self._model(*args)
                return x / n

        return fn

    @property
    def scaled_pdf(self):
        """Get density model."""
        if self._log:
            return lambda *args: np.exp(self._model(*args)[1])
        return lambda *args: self._model(*args)[1]

    def __init__(
        self,
        data: _ArrayLike,
        scaled_pdf: _tp.Callable,
        verbose: int = 0,
        log: bool = False,
    ):
        """
        Initialize cost function with data and model.

        Parameters
        ----------
        data : array-like
            Sample of observations. If the observations are multidimensional, data must
            have the shape (D, N), where D is the number of dimensions and N the number of
            data points.
        scaled_pdf : callable
            Density function of the form f(data, par0, [par1, ...]), where data is the
            sample and par0, ... are model parameters. Must return a tuple
            (<integral over f in data window>, <f evaluated at data points>). The first
            value is the density integrated over the data window, the interval that we
            consider for the fit. For example, if the data are exponentially distributed,
            but we fit only the interval (0, 5), then the first value is the density
            integrated from 0 to 5. If the data are multivariate, data passed to f has
            shape (D, N), where D is the number of dimensions and N the number of data
            points.
        verbose : int, optional
            Verbosity level. 0: is no output (default). 1: print current args and negative
            log-likelihood value.
        log : bool, optional
            Distributions of the exponential family (normal, exponential, poisson, ...)
            allow one to compute the logarithm of the pdf directly, which is more accurate
            and efficient than effectively doing ``log(exp(logpdf))``. Set this to True,
            if the model returns the logarithm of the density as the second argument
            instead of the density. Default is False.
        """
        super().__init__(data, scaled_pdf, verbose, log)

    def _call(self, args):
        data = self._masked
        ns, x = self._model(data, *args)
        x = _normalize_model_output(
            x, "Model should return numpy array in second position"
        )
        if self._log:
            return 2 * (ns - np.sum(x))
        return 2 * (ns + _unbinned_nll(x))


class BinnedCost(MaskedCost):
    """Base class for binned cost functions."""

    __slots__ = "_xe", "_ndim", "_bztrafo"

    n = MaskedCost.data

    @property
    def xe(self):
        """Access bin edges."""
        return self._xe

    @Cost.ndata.getter  # type:ignore
    def ndata(self):
        """See Cost.ndata."""
        return np.prod(self._masked.shape[: self._ndim])

    def __init__(self, args, n, xe, verbose, *updater):
        """For internal use."""
        if not isinstance(xe, _tp.Iterable):
            raise ValueError("xe must be iterable")

        shape = _shape_from_xe(xe)
        self._ndim = len(shape)
        self._xe = _norm(xe) if self._ndim == 1 else tuple(_norm(xei) for xei in xe)

        n = _norm(n)
        is_weighted = n.ndim > self._ndim

        if n.ndim != (self._ndim + int(is_weighted)):
            raise ValueError("n must either have same dimension as xe or one extra")

        for i, xei in enumerate([self._xe] if self._ndim == 1 else self._xe):
            if len(xei) != n.shape[i] + 1:
                raise ValueError(
                    f"n and xe have incompatible shapes along dimension {i}, "
                    "xe must be longer by one element along each dimension"
                )

        if is_weighted:
            if n.shape[-1] != 2:
                raise ValueError("n must have shape (..., 2)")
            self._bztrafo = BohmZechTransform(n[..., 0], n[..., 1])
        else:
            self._bztrafo = None

        super().__init__(args, n, verbose)

    def _update_cache(self):
        super()._update_cache()
        if self._bztrafo:
            ma = _replace_none(self._mask, ...)
            self._bztrafo = BohmZechTransform(self._data[ma, 0], self._data[ma, 1])

    def visualize(self, args: _ArrayLike):
        """
        Visualize data and model agreement (requires matplotlib).

        The visualization is drawn with matplotlib.pyplot into the current axes.

        Parameters
        ----------
        args : array-like
            Parameter values.
        """
        from matplotlib import pyplot as plt

        args = np.atleast_1d(args)

        if self._ndim > 1:
            raise ValueError("visualize is not implemented for multi-dimensional data")

        n = self._masked[..., 0] if self._bztrafo else self._masked
        ne = (self._masked[..., 1] if self._bztrafo else self._masked) ** 0.5
        xe = self.xe
        cx = 0.5 * (xe[1:] + xe[:-1])
        if self.mask is not None:
            cx = cx[self.mask]
        plt.errorbar(cx, n, ne, fmt="ok")
        mu = self._pred(args)  # implemented in derived
        plt.stairs(mu, xe, fill=True, color="C0")


class BinnedCostWithModel(BinnedCost):
    """Base class for binned cost functions."""

    __slots__ = "_xe_shape", "_model", "_model_arg"

    def __init__(self, n, xe, model, verbose):
        """For internal use."""
        self._model = model

        super().__init__(describe(model)[1:], n, xe, verbose)

        if self._ndim == 1:
            self._xe_shape = None
            self._model_arg = _norm(self.xe)
        else:
            self._xe_shape = tuple(len(xei) for xei in self.xe)
            self._model_arg = np.row_stack(
                [x.flatten() for x in np.meshgrid(*self.xe, indexing="ij")]
            )

    def _pred(self, args):
        d = self._model(self._model_arg, *args)
        d = _normalize_model_output(d)
        if self._xe_shape is not None:
            d = d.reshape(self._xe_shape)
        for i in range(self._ndim):
            d = np.diff(d, axis=i)
        # differences can come out negative due to round-off error in subtraction,
        # we set negative values to zero
        d[d < 0] = 0
        return d


class BarlowBeestonLite(BinnedCost):
    """
    Binned cost function for a template fit with uncertainties on the template.

    Compared to the original Beeston-Barlow method, the lite methods uses one nuisance
    parameter per bin instead of one nuisance parameter per component per bin, which
    is an approximation. This class offers two different lite methods. The default
    method used is the one which performs better on average.

    The cost function works for both weighted data and weighted templates. The cost
    function assumes that the weights are independent of the data. This is not the
    case for sWeights, and the uncertaintes for results obtained with sWeights will
    only be approximately correct, see C. Langenbruch, Eur.Phys.J.C 82 (2022) 5, 393.

    Barlow and Beeston, Comput.Phys.Commun. 77 (1993) 219-228,
    https://doi.org/10.1016/0010-4655(93)90005-W)
    J.S. Conway, PHYSTAT 2011, https://doi.org/10.48550/arXiv.1103.0354
    """

    __slots__ = "_bbl_data", "_impl"

    def __init__(
        self,
        n: _ArrayLike,
        xe: _ArrayLike,
        templates: _tp.Sequence[_tp.Sequence],
        name: _tp.Collection[str] = None,
        verbose: int = 0,
        method: str = "hpd",
    ):
        """
        Initialize cost function with data and model.

        Parameters
        ----------
        n : array-like
            Histogram counts. If this is an array with dimension D+1, where D is the
            number of histogram axes, then the last dimension must have two elements
            and is interpreted as pairs of sum of weights and sum of weights squared.
        xe : array-like or collection of array-like
            Bin edge locations, must be len(n) + 1, where n is the number of bins.
            If the histogram has more than one axis, xe must be a collection of the
            bin edge locations along each axis.
        templates : collection of array-like
            Collection of arrays, which contain the histogram counts of each template.
            The template histograms must use the same axes as the data histogram. If
            the counts are represented by an array with dimension D+1, where D is the
            number of histogram axes, then the last dimension must have two elements
            and is interpreted as pairs of sum of weights and sum of weights squared.
        name : collection of str, optional
            Optional name for the yield of each template. Must have length K.
        verbose : int, optional
            Verbosity level. 0: is no output (default).
            1: print current args and negative log-likelihood value.
        method : {"jsc", "hpd"}, optional
            Which version of the lite method to use. jsc: Method developed by
            J.S. Conway, PHYSTAT 2011, https://doi.org/10.48550/arXiv.1103.0354.
            hpd: Method developed by H.P. Dembinski. Default is "hpd", which seems to
            perform slightly better on average. The default may change in the future
            when more practical experience with both method is gained. Set this
            parameter explicitly to ensure that a particular method is used now and
            in the future.
        """
        M = len(templates)
        if M < 1:
            raise ValueError("at least one template is required")
        if name is None:
            name = [f"x{i}" for i in range(M)]
        else:
            if len(name) != M:
                raise ValueError("number of names must match number of templates")

        shape = _shape_from_xe(xe)
        ndim = len(shape)
        temp = []
        temp_var = []
        for ti in templates:
            t = _norm(ti)
            if t.ndim > ndim:
                # template is weighted
                if t.ndim != ndim + 1 or t.shape[:-1] != shape:
                    raise ValueError("shapes of n and templates do not match")
                temp.append(t[..., 0])
                temp_var.append(t[..., 1])
            else:
                if t.ndim != ndim or t.shape != shape:
                    raise ValueError("shapes of n and templates do not match")
                temp.append(t)
                temp_var.append(t)

        nt = []
        nt_var = []
        for t, tv in zip(temp, temp_var):
            f = 1 / np.sum(t)
            nt.append(t * f)
            nt_var.append(tv * f**2)
        self._bbl_data = (nt, nt_var)

        if method == "jsc":
            self._impl = barlow_beeston_lite_chi2_jsc
        elif method == "hpd":
            self._impl = barlow_beeston_lite_chi2_hpd
        else:
            raise ValueError(
                f"method {method} is not understood, allowed values: {{'jsc', 'hpd'}}"
            )

        super().__init__(name, n, xe, verbose)

    def _pred(self, args):
        ntemp, ntemp_var = self._bbl_data
        mu = 0
        mu_var = 0
        for a, nt, vnt in zip(args, ntemp, ntemp_var):
            mu += a * nt
            mu_var += a**2 * vnt
        return mu, mu_var

    def _call(self, args):
        mu, mu_var = self._pred(args)

        ma = self.mask
        if ma is not None:
            mu = mu[ma]
            mu_var = mu_var[ma]

        if self._bztrafo:
            n, mu, mu_var = self._bztrafo(mu, mu_var)
        else:
            n = self._masked

        ma = mu > 0
        return self._impl(n[ma], mu[ma], mu_var[ma])

    def visualize(self, args: _ArrayLike):
        """
        Visualize data and model agreement (requires matplotlib).

        The visualization is drawn with matplotlib.pyplot into the current axes.

        Parameters
        ----------
        args : array-like
            Parameter values.
        """
        from matplotlib import pyplot as plt

        if self._ndim > 1:
            raise ValueError("visualize is not implemented for multi-dimensional data")

        args = np.atleast_1d(args)

        n = self._masked[..., 0] if self._bztrafo else self._masked
        ne = (self._masked[..., 1] if self._bztrafo else self._masked) ** 0.5

        xe = self.xe
        cx = 0.5 * (xe[1:] + xe[:-1])
        if self.mask is not None:
            cx = cx[self.mask]
        plt.errorbar(cx, n, ne, fmt="ok")

        mu, mu_var = self._pred(args)
        mu_err = mu_var**0.5
        plt.stairs(mu + mu_err, xe, baseline=mu - mu_err, fill=True, color="C0")


class BinnedNLL(BinnedCostWithModel):
    """
    Binned negative log-likelihood.

    Use this if only the shape of the fitted PDF is of interest and the data is binned.
    This cost function works with normal and weighted histograms. The histogram can be
    one- or multi-dimensional.

    The cost function has a minimum value that is asymptotically chi2-distributed. It is
    constructed from the log-likelihood assuming a multivariate-normal distribution and
    using the saturated model as a reference.
    """

    __slots__ = ()

    @property
    def cdf(self):
        """Get cumulative density function."""
        return self._model

    def __init__(
        self, n: _ArrayLike, xe: _ArrayLike, cdf: _tp.Callable, verbose: int = 0
    ):
        """
        Initialize cost function with data and model.

        Parameters
        ----------
        n : array-like
            Histogram counts. If this is an array with dimension D+1, where D is the
            number of histogram axes, then the last dimension must have two elements
            and is interpreted as pairs of sum of weights and sum of weights squared.
        xe : array-like or collection of array-like
            Bin edge locations, must be len(n) + 1, where n is the number of bins.
            If the histogram has more than one axis, xe must be a collection of the
            bin edge locations along each axis.
        cdf : callable
            Cumulative density function of the form f(xe, par0, par1, ..., parN),
            where xe is a bin edge and par0, ... are model parameters. The corresponding
            density must be normalized to unity over the space covered by the histogram.
            If the model is multivariate, xe must be an array-like with shape (D, N),
            where D is the dimension and N is the number of points where the model is
            evaluated.
        verbose : int, optional
            Verbosity level. 0: is no output (default).
            1: print current args and negative log-likelihood value.
        """
        super().__init__(n, xe, cdf, verbose)

    def _pred(self, args):
        p = super()._pred(args)
        ma = self.mask
        if ma is not None:
            p /= np.sum(p[ma])  # normalise probability of remaining bins
        scale = np.sum(self._masked[..., 0] if self._bztrafo else self._masked)
        return p * scale

    def _call(self, args):
        mu = self._pred(args)
        ma = self.mask
        if ma is not None:
            mu = mu[ma]
        if self._bztrafo:
            n, mu = self._bztrafo(mu)
        else:
            n = self._masked
        return multinominal_chi2(n, mu)


class ExtendedBinnedNLL(BinnedCostWithModel):
    """
    Binned extended negative log-likelihood.

    Use this if shape and normalization of the fitted PDF are of interest and the data is
    binned. This cost function works with normal and weighted histograms. The histogram
    can be one- or multi-dimensional.

    The cost function works for both weighted data. The cost function assumes that
    the weights are independent of the data. This is not the case for sWeights, and
    the uncertaintes for results obtained with sWeights will only be approximately
    correct, see C. Langenbruch, Eur.Phys.J.C 82 (2022) 5, 393.

    The cost function has a minimum value that is asymptotically chi2-distributed. It is
    constructed from the log-likelihood assuming a poisson distribution and using the
    saturated model as a reference.
    """

    __slots__ = ()

    @property
    def scaled_cdf(self):
        """Get integrated density model."""
        return self._model

    def __init__(
        self,
        n: _ArrayLike,
        xe: _ArrayLike,
        scaled_cdf: _tp.Callable,
        verbose: int = 0,
    ):
        """
        Initialize cost function with data and model.

        Parameters
        ----------
        n : array-like
            Histogram counts. If this is an array with dimension D+1, where D is the
            number of histogram axes, then the last dimension must have two elements
            and is interpreted as pairs of sum of weights and sum of weights squared.
        xe : array-like or collection of array-like
            Bin edge locations, must be len(n) + 1, where n is the number of bins.
            If the histogram has more than one axis, xe must be a collection of the
            bin edge locations along each axis.
        scaled_cdf : callable
            Scaled Cumulative density function of the form f(xe, par0, [par1, ...]), where
            xe is a bin edge and par0, ... are model parameters.  If the model is
            multivariate, xe must be an array-like with shape (D, N), where D is the
            dimension and N is the number of points where the model is evaluated.
        verbose : int, optional
            Verbosity level. 0: is no output (default). 1: print current args and negative
            log-likelihood value.
        """
        super().__init__(n, xe, scaled_cdf, verbose)

    def _call(self, args):
        mu = self._pred(args)
        ma = self.mask
        if ma is not None:
            mu = mu[ma]
        if self._bztrafo:
            n, mu = self._bztrafo(mu)
        else:
            n = self._masked
        return poisson_chi2(n, mu)


class LeastSquares(MaskedCost):
    """Least-squares cost function (aka chisquare function).

    Use this if you have data of the form (x, y +/- yerror), where x can be
    one-dimensional or multi-dimensional, but y is always one-dimensional. See
    :meth:`__init__` for details on how to use a multivariate model.
    """

    __slots__ = "_loss", "_cost", "_model", "_ndim"

    @property
    def x(self):
        """Get explanatory variables."""
        if self._ndim == 1:
            return self.data[:, 0]
        return self.data.T[: self._ndim]

    @x.setter
    def x(self, value):
        if self._ndim == 1:
            self.data[:, 0] = _norm(value)
        else:
            self.data[:, : self._ndim] = _norm(value).T
        self._update_cache()

    @property
    def y(self):
        """Get samples."""
        return self.data[:, self._ndim]

    @y.setter
    def y(self, value):
        self.data[:, self._ndim] = _norm(value)
        self._update_cache()

    @property
    def yerror(self):
        """Get sample uncertainties."""
        return self.data[:, self._ndim + 1]

    @yerror.setter
    def yerror(self, value):
        self.data[:, self._ndim + 1] = _norm(value)
        self._update_cache()

    @property
    def model(self):
        """Get model of the form y = f(x, par0, [par1, ...])."""
        return self._model

    @property
    def loss(self):
        """Get loss function."""
        return self._loss

    @loss.setter
    def loss(self, loss: _tp.Union[str, _tp.Callable]):
        self._loss = loss
        if isinstance(loss, str):
            if loss == "linear":
                self._cost = chi2
            elif loss == "soft_l1":
                self._cost = _soft_l1_cost
            else:
                raise ValueError("unknown loss type: " + loss)
        else:
            self._cost = lambda y, ye, ym: np.sum(
                loss(_z_squared(y, ye, ym))  # type:ignore
            )

    @Cost.ndata.getter  # type:ignore
    def ndata(self):
        """See Cost.ndata."""
        return len(self._masked)

    def __init__(
        self,
        x: _ArrayLike,
        y: _ArrayLike,
        yerror: _ArrayLike,
        model: _tp.Callable,
        loss: _tp.Union[str, _tp.Callable] = "linear",
        verbose: int = 0,
    ):
        """
        Initialize cost function with data and model.

        Parameters
        ----------
        x : array-like
            Locations where the model is evaluated. If the model is multivariate, x must
            have shape (D, N), where D is the number of dimensions and N the number of
            data points.
        y : array-like
            Observed values. Must have the same length as x.
        yerror : array-like or float
            Estimated uncertainty of observed values. Must have same shape as y or be a
            scalar, which is then broadcasted to same shape as y.
        model : callable
            Function of the form f(x, par0, [par1, ...]) whose output is compared to
            observed values, where x is the location and par0, ... are model parameters.
            If the model is multivariate, x has shape (D, N), where D is the number
            of dimensions and N the number of data points.
        loss : str or callable, optional
            The loss function can be modified to make the fit robust against outliers, see
            scipy.optimize.least_squares for details. Only "linear" (default) and
            "soft_l1" are currently implemented, but users can pass any loss function as
            this argument. It should be a monotonic, twice differentiable function, which
            accepts the squared residual and returns a modified squared residual.
        verbose : int, optional
            Verbosity level. 0: is no output (default). 1: print current args and negative
            log-likelihood value.

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
        assert x.ndim >= 1  # guaranteed by _norm

        self._ndim = x.ndim
        self._model = model
        self.loss = loss

        x = np.atleast_2d(x)
        data = np.column_stack(np.broadcast_arrays(*x, y, yerror))

        super().__init__(describe(self._model)[1:], data, verbose)

    def _call(self, args):
        x = self._masked.T[0] if self._ndim == 1 else self._masked.T[: self._ndim]
        y, yerror = self._masked.T[self._ndim :]
        ym = self._model(x, *args)
        ym = _normalize_model_output(ym)
        return self._cost(y, yerror, ym)

    def visualize(self, args: _ArrayLike, model_points: int = 0):
        """
        Visualize data and model agreement (requires matplotlib).

        The visualization is drawn with matplotlib.pyplot into the current axes.

        Parameters
        ----------
        args : array-like
            Parameter values.

        model_points : int, optional
            How many points to use to draw the model. Default is 0, in this case
            an smart sampling algorithm selects the number of points.
        """
        from matplotlib import pyplot as plt

        if self._ndim > 1:
            raise ValueError("visualize is not implemented for multi-dimensional data")

        # TODO
        # - make linear or log-spacing configurable

        x, y, ye = self._masked.T
        plt.errorbar(x, y, ye, fmt="ok")
        if model_points > 0:
            if x[0] > 0 and x[-1] / x[0] > 1e2:
                xm = np.geomspace(x[0], x[-1], model_points)
            else:
                xm = np.linspace(x[0], x[-1], model_points)
            ym = self.model(xm, *args)
        else:
            xm, ym = _smart_sampling(lambda x: self.model(x, *args), x[0], x[-1])
        plt.plot(xm, ym)


class NormalConstraint(Cost):
    """
    Gaussian penalty for one or several parameters.

    The Gaussian penalty acts like a pseudo-measurement of the parameter itself, based on
    a (multi-variate) normal distribution. Penalties can be set for one or several
    parameters at once (which is more efficient). When several parameter are constrained,
    one can specify the full covariance matrix of the parameters.

    Notes
    -----
    It is sometimes necessary to add a weak penalty on a parameter to avoid instabilities
    in the fit. A typical example in high-energy physics is the fit of a signal peak above
    some background. If the amplitude of the peak vanishes, the shape parameters of the
    peak become unconstrained and the fit becomes unstable. This can be avoided by adding
    weak (large uncertainty) penalty on the shape parameters whose pull is negligible if
    the peak amplitude is non-zero.

    This class can also be used to approximately include external measurements of some
    parameters, if the original cost function is not available or too costly to compute.
    If the external measurement was performed in the asymptotic limit with a large sample,
    a Gaussian penalty is an accurate statistical representation of the external result.
    """

    __slots__ = "_value", "_cov", "_covinv"

    def __init__(
        self,
        args: _tp.Union[str, _tp.Iterable[str]],
        value: _ArrayLike,
        error: _ArrayLike,
    ):
        """
        Initialize the normal constraint with expected value(s) and error(s).

        Parameters
        ----------
        args : str or sequence of str
            Parameter name(s).
        value : float or array-like
            Expected value(s). Must have same length as `args`.
        error : float or array-like
            Expected error(s). If 1D, must have same length as `args`. If 2D, must be the
            covariance matrix of the parameters.
        """
        self._value = _norm(value)
        self._cov = _norm(error)
        if self._cov.ndim < 2:
            self._cov **= 2
        self._covinv = _covinv(self._cov)
        tp_args = (args,) if isinstance(args, str) else tuple(args)
        super().__init__(tp_args, False)

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
            return np.sum(delta**2 * self._covinv)
        return np.einsum("i,ij,j", delta, self._covinv, delta)

    @Cost.ndata.getter  # type:ignore
    def ndata(self):
        """See Cost.ndata."""
        return len(self._value)

    def visualize(self, args: _ArrayLike):
        """
        Visualize data and model agreement (requires matplotlib).

        The visualization is drawn with matplotlib.pyplot into the current axes.

        Parameters
        ----------
        args : array-like
            Parameter values.
        """
        from matplotlib import pyplot as plt

        args = np.atleast_1d(args)

        par = self.func_code.co_varnames
        val = self.value
        cov = self.covariance
        if cov.ndim == 2:
            cov = np.diag(cov)
        err = np.sqrt(cov)

        n = len(par)

        i = 0
        max_pull = 0
        for v, e, a in zip(val, err, args):
            pull = (a - v) / e
            max_pull = max(abs(pull), max_pull)
            plt.errorbar(pull, -i, 0, 1, fmt="o", color="C0")
            i += 1
        plt.axvline(0, color="k")
        plt.xlim(-max_pull - 1.1, max_pull + 1.1)
        yaxis = plt.gca().yaxis
        yaxis.set_ticks(-np.arange(n))
        yaxis.set_ticklabels(par)
        plt.ylim(-n + 0.5, 0.5)


def _norm(value: _ArrayLike) -> np.ndarray:
    value = np.atleast_1d(value)
    dtype = value.dtype
    if dtype.kind != "f":
        value = value.astype(np.float64)
    return value


def _covinv(array):
    return np.linalg.inv(array) if array.ndim == 2 else 1.0 / array


def _normalize_model_output(x, msg="Model should return numpy array"):
    if not isinstance(x, np.ndarray):
        warnings.warn(
            f"{msg}, but returns {type(x)}",
            PerformanceWarning,
        )
        x = np.array(x)
        if x.dtype.kind != "f":
            return x.astype(float)
    return x


def _shape_from_xe(xe):
    if isinstance(xe[0], _tp.Iterable):
        return tuple(len(xei) - 1 for xei in xe)
    return (len(xe) - 1,)
