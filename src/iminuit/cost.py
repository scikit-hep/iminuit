"""
Standard cost functions to minimize for statistical fits.

We provide these for convenience, so that you do not have to write your own for standard
fits. The cost functions optionally use Numba to accelerate some calculations, if Numba
is installed.

**There is no need** to set :attr:`iminuit.Minuit.errordef` manually for any of these
cost functions. :class:`iminuit.Minuit` automatically uses the correct value, which is
provided by each cost function with the attribute ``Cost.errordef``.

What to use when
----------------
- Fit a normalised probability density to data

    - Data are not binned: :class:`UnbinnedNLL`
    - Data are binned: :class:`BinnedNLL`, also supports histogram of weighted samples

- Fit a density to data, density is not normalised

    - Data are not binned: :class:`ExtendedUnbinnedNLL`
    - Data are binned: :class:`ExtendedBinnedNLL`, also supports
      histogram of weighted samples

- Fit a template to binned data with bin-wise uncertainties on the template

    - :class:`Template`, also supports weighted data and weighted template histograms

- Fit of a function f(x) to (x, y, yerror) pairs with normal-distributed fluctuations. x
  is one- or multi-dimensional, y is one-dimensional.

    - y values contain no outliers: :class:`LeastSquares`
    - y values contain outliers: :class:`LeastSquares` with loss function set to
      "soft_l1"

- Include constraints from external fits or apply regularisation

    - :class:`NormalConstraint`

Combining cost functions
------------------------
All cost functions can be added, which generates a new combined cost function.
Parameters with the same name are shared between component cost functions. Use this to
constrain one or several parameters with different data sets and using different
statistical models for each data set. Gaussian penalty terms can also be added to the
cost function to introduce external knowledge about a parameter.

Model parameter limits
----------------------
The Minuit algorithms support box constrains in parameter space. A user-defined model
can declare that a parameter is only valid over an interval on the real line with the
``Annotated`` type annotation, see :class:`iminuit.Minuit` for details. A typical
example is the sigma parameter of a normal distribution, which must be positive. The
cost functions defined here propagate this information to :class:`iminuit.Minuit`.

Note: The :class:`Template` declares that the template amplitudes must be non-negative,
which is usually the right choice, however, it may be desirable to fit templates which
can have negative amplitudes. To achieve this, simply reset the limits with
:attr:`iminuit.Minuit.limits` after creating the Minuit instance.

User-defined gradients
----------------------
If the user provides a model gradient, the cost functions defined here except
:class:`Template` will then also make their gradient available, which is then
automatically used by :class:`iminuit.Minuit` (see the constructor for details) to
potentially improve the fit (improve convergence  or robustness).

Note that it is perfectly normal to use Minuit without a user-defined gradient, and
Minuit does not always benefit from a user-defined gradient. If the gradient is
expensive to compute, the time to converge may increase. If you have trouble with the
fitting process, it is unlikely that the issues are resolved by a user-defined gradient.

Notes
-----
The cost functions defined here have been optimized with knowledge about implementation
details of Minuit to give the highest accucary and the most robust results, so they
should perform well. If you have trouble with your own implementations, try these.

The binned versions of the log-likelihood fits support weighted samples. For each bin of
the histogram, the sum of weights and the sum of squared weights is needed then, see
class documentation for details.
"""

from __future__ import annotations

from .util import (
    describe,
    merge_signatures,
    PerformanceWarning,
    _smart_sampling,
    _detect_log_spacing,
    is_positive_definite,
)
from .typing import Model, ModelGradient, LossFunction
import numpy as np
from numpy.typing import NDArray, ArrayLike
from collections.abc import Sequence as ABCSequence
import abc
from typing import (
    List,
    Tuple,
    Union,
    Sequence,
    Collection,
    Dict,
    Any,
    Iterable,
    Optional,
    overload,
    TypeVar,
    Callable,
    cast,
)
import warnings
from ._deprecated import deprecated_parameter, deprecated

__all__ = [
    "CHISQUARE",
    "NEGATIVE_LOG_LIKELIHOOD",
    "chi2",
    "multinominal_chi2",
    "poisson_chi2",
    "template_chi2_jsc",
    "template_chi2_da",
    "template_nll_asy",
    "Cost",
    "CostSum",
    "Constant",
    "BinnedNLL",
    "UnbinnedNLL",
    "ExtendedBinnedNLL",
    "ExtendedUnbinnedNLL",
    "Template",
    "LeastSquares",
]

T = TypeVar("T", float, NDArray)

CHISQUARE = 1.0
NEGATIVE_LOG_LIKELIHOOD = 0.5

_TINY_FLOAT = np.finfo(float).tiny


def _safe_log(x):
    # guard against x = 0
    return np.log(x + _TINY_FLOAT)


def _unbinned_nll(x):
    return -np.sum(_safe_log(x))


def _z_squared(y, ye, ym):
    z = (y - ym) / ye
    return z * z


def _replace_none(x, replacement):
    if x is None:
        return replacement
    return x


@deprecated("The class is deprecated and will be removed without replacement")
class BohmZechTransform:
    """
    Apply Bohm-Zech transform.

    See Bohm and Zech, NIMA 748 (2014) 1-6.

    :meta private:
    """

    __slots__ = "_obs", "_scale"

    _obs: NDArray
    _scale: NDArray

    def __init__(self, val: ArrayLike, var: ArrayLike):
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

        self._scale = np.ones_like(val)
        np.divide(val, var, out=self._scale, where=var > 0)
        self._obs = val * self._scale

    @overload
    def __call__(
        self, val: ArrayLike
    ) -> Tuple[NDArray, NDArray]: ...  # pragma: no cover

    @overload
    def __call__(
        self, val: ArrayLike, var: ArrayLike
    ) -> Tuple[NDArray, NDArray, NDArray]: ...  # pragma: no cover

    def __call__(self, val, var=None):
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
        val = np.atleast_1d(val)
        s = self._scale
        if var is None:
            return self._obs, val * s
        var = np.atleast_1d(var)
        return self._obs, val * s, var * s**2


def chi2(y: ArrayLike, ye: ArrayLike, ym: ArrayLike) -> float:
    """
    Compute (potentially) chi2-distributed cost.

    The value returned by this function is chi2-distributed, if the observed values are
    normally distributed around the expected values with the provided standard
    deviations.

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


def _chi2_grad(y: NDArray, ye: NDArray, ym: NDArray, ymg: NDArray) -> NDArray:
    return -2 * np.sum((y - ym) * ymg * ye**-2, axis=1)


def _soft_l1_cost(y: NDArray, ye: NDArray, ym: NDArray) -> float:
    z_sqr = _z_squared(y, ye, ym)
    return 2 * np.sum(np.sqrt(1 + z_sqr) - 1)


def _soft_l1_cost_grad(y: NDArray, ye: NDArray, ym: NDArray, ymg: NDArray) -> NDArray:
    inv_ye = 1 / ye
    z = (y - ym) * inv_ye
    return -2 * np.sum(z * ymg * inv_ye * (1 + z**2) ** -0.5, axis=1)


def multinominal_chi2(n: ArrayLike, mu: ArrayLike) -> float:
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


def _multinominal_chi2_grad(n: NDArray, mu: NDArray, gmu: NDArray) -> NDArray:
    return -2 * np.sum(n * gmu / mu, axis=tuple(range(1, gmu.ndim)))


def poisson_chi2(n: ArrayLike, mu: ArrayLike) -> float:
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


def _poisson_chi2_grad(n: NDArray, mu: NDArray, gmu: NDArray) -> NDArray:
    return 2 * np.sum(gmu * (1.0 - n / mu), axis=tuple(range(1, gmu.ndim)))


def template_chi2_jsc(n: ArrayLike, mu: ArrayLike, mu_var: ArrayLike) -> float:
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
        Asymptotically chi-square-distributed test statistic.

    Notes
    -----
    The implementation deviates slightly from the paper by making the result
    asymptotically chi2-distributed, which helps to maximise the numerical
    accuracy for Minuit.
    """
    n, mu, mu_var = np.atleast_1d(n, mu, mu_var)

    beta_var = mu_var / mu**2

    # Eq. 15 from https://doi.org/10.48550/arXiv.2206.12346
    p = 0.5 - 0.5 * mu * beta_var
    beta = p + np.sqrt(p**2 + n * beta_var)

    return poisson_chi2(n, mu * beta) + np.sum((beta - 1) ** 2 / beta_var)


def template_chi2_da(n: ArrayLike, mu: ArrayLike, mu_var: ArrayLike) -> float:
    """
    Compute asymptotically chi2-distributed cost for a template fit.

    H.P. Dembinski, A. Abdelmotteleb, https://doi.org/10.48550/arXiv.2206.12346

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
        Asymptotically chi-square-distributed test statistic.
    """
    n, mu, mu_var = np.atleast_1d(n, mu, mu_var)
    k = mu**2 / mu_var
    # avoid divide by zero
    beta = (n + k) / (mu + k + _TINY_FLOAT)
    return poisson_chi2(n, mu * beta) + poisson_chi2(k, k * beta)


def template_nll_asy(n: ArrayLike, mu: ArrayLike, mu_var: ArrayLike) -> float:
    """
    Compute marginalized negative log-likelikihood for a template fit.

    This is the negative logarithm of equation 3.15 of the paper by
    C.A. Argüelles, A. Schneider, T. Yuan,
    https://doi.org/10.1007/JHEP06(2019)030.

    The authors use a Bayesian approach and integrate over the nuisance
    parameters. Like the other Barlow-Beeston-lite methods, this is an
    approximation. The resulting likelihood cannot be turned into an
    asymptotically chi-square distributed test statistic as detailed
    in Baker & Cousins, NIM 221 (1984) 437-442.

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
        Negative log-likelihood function value.
    """
    from scipy.special import loggamma as lg

    n, mu, mu_var = np.atleast_1d(n, mu, mu_var)

    alpha = mu**2 / mu_var + 1
    beta = mu / mu_var
    return -np.sum(
        alpha * np.log(beta)
        + lg(n + alpha)
        - (lg(n + 1) + (n + alpha) * np.log(1 + beta) + lg(alpha))
    )


# If numba is available, use it to accelerate computations in float32 and float64
# precision. Fall back to plain numpy for float128 which is not currently supported
# by numba.
try:
    from numba import njit as jit
    from numba.extending import overload as nb_overload

    @nb_overload(_safe_log, inline="always")
    def _ol_safe_log(x):
        return _safe_log  # pragma: no cover

    @nb_overload(_z_squared, inline="always")
    def _ol_z_squared(y, ye, ym):
        return _z_squared  # pragma: no cover

    _unbinned_nll_np = _unbinned_nll
    _unbinned_nll_nb = jit(
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
    _multinominal_chi2_nb = jit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_multinominal_chi2_np)

    def multinominal_chi2(n: ArrayLike, mu: ArrayLike) -> float:  # noqa
        n, mu = np.atleast_1d(n, mu)
        if mu.dtype in (np.float32, np.float64):
            return _multinominal_chi2_nb(n, mu)
        # fallback to numpy for float128
        return _multinominal_chi2_np(n, mu)

    multinominal_chi2.__doc__ = _multinominal_chi2_np.__doc__

    _poisson_chi2_np = poisson_chi2
    _poisson_chi2_nb = jit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_poisson_chi2_np)

    def poisson_chi2(n: ArrayLike, mu: ArrayLike) -> float:  # noqa
        n, mu = np.atleast_1d(n, mu)
        if mu.dtype in (np.float32, np.float64):
            return _poisson_chi2_nb(n, mu)
        # fallback to numpy for float128
        return _poisson_chi2_np(n, mu)

    poisson_chi2.__doc__ = _poisson_chi2_np.__doc__

    _chi2_np = chi2
    _chi2_nb = jit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_chi2_np)

    def chi2(y: ArrayLike, ye: ArrayLike, ym: ArrayLike) -> float:  # noqa
        y, ye, ym = np.atleast_1d(y, ye, ym)
        if ym.dtype in (np.float32, np.float64):
            return _chi2_nb(y, ye, ym)
        # fallback to numpy for float128
        return _chi2_np(y, ye, ym)

    chi2.__doc__ = _chi2_np.__doc__

    _soft_l1_cost_np = _soft_l1_cost
    _soft_l1_cost_nb = jit(
        nogil=True,
        cache=True,
        error_model="numpy",
    )(_soft_l1_cost_np)

    def _soft_l1_cost(y: NDArray, ye: NDArray, ym: NDArray) -> float:
        if ym.dtype in (np.float32, np.float64):
            return _soft_l1_cost_nb(y, ye, ym)
        # fallback to numpy for float128
        return _soft_l1_cost_np(y, ye, ym)

except ModuleNotFoundError:
    pass


class Cost(abc.ABC):
    """
    Base class for all cost functions.

    :meta private:
    """

    __slots__ = ("_parameters", "_verbose")

    _parameters: Dict[str, Optional[Tuple[float, float]]]
    _verbose: int

    @property
    def errordef(self):
        """
        For internal use.

        :meta private:
        """
        return self._errordef()

    def _errordef(self):
        return CHISQUARE

    @property
    def ndata(self):
        """
        Return number of points in least-squares fits or bins in a binned fit.

        Infinity is returned if the cost function is unbinned. This is used by Minuit to
        compute the reduced chi2, a goodness-of-fit estimate.
        """
        return self._ndata()

    @property
    def npar(self):
        """Return total number of model parameters."""
        return len(self._parameters)

    @abc.abstractmethod
    def _ndata(self):
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

    def __init__(
        self, parameters: Dict[str, Optional[Tuple[float, float]]], verbose: int
    ):
        """For internal use."""
        self._parameters = parameters
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

    def __call__(self, *args: float) -> float:
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
        r = self._value(args)
        if self.verbose >= 1:
            print(args, "->", r)
        return r

    def grad(self, *args: float) -> NDArray:
        """
        Compute gradient of the cost function.

        This requires that a model gradient is provided.

        Parameters
        ----------
        *args : float
            Parameter values.

        Returns
        -------
        ndarray of float
            The length of the array is equal to the length of args.
        """
        return self._grad(args)

    @property
    def has_grad(self) -> bool:
        """Return True if cost function can compute a gradient."""
        return self._has_grad()

    @abc.abstractmethod
    def _value(self, args: Sequence[float]) -> float: ...  # pragma: no cover

    @abc.abstractmethod
    def _grad(self, args: Sequence[float]) -> NDArray: ...  # pragma: no cover

    @abc.abstractmethod
    def _has_grad(self) -> bool: ...  # pragma: no cover


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
        super().__init__({}, False)

    def _ndata(self):
        return 0

    def _value(self, args: Sequence[float]) -> float:
        return self.value

    def _grad(self, args: Sequence[float]) -> NDArray:
        return np.zeros(0)

    @staticmethod
    def _has_grad():
        return True


class CostSum(Cost, ABCSequence):
    """
    Sum of cost functions.

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
    function signature does not allow one to determine how many parameters are accepted
    by the function and which parameters overlap between different cost functions.
    """

    __slots__ = "_items", "_maps"

    def __init__(self, *items: Union[Cost, float]):
        """
        Initialize with cost functions.

        Parameters
        ----------
        *items : Cost
            Cost functions. May also be other CostSum functions.
        """
        self._items: List[Cost] = []
        for item in items:
            if isinstance(item, CostSum):
                self._items += item._items
            elif isinstance(item, (int, float)):
                if item != 0:
                    self._items.append(Constant(item))
            else:
                self._items.append(item)
        signatures, self._maps = merge_signatures(self._items, annotations=True)
        super().__init__(signatures, max(c.verbose for c in self._items))

    def _split(self, args: Sequence[float]):
        for component, cmap in zip(self._items, self._maps):
            component_args = tuple(args[i] for i in cmap)
            yield component, component_args

    def _value(self, args: Sequence[float]) -> float:
        r = 0.0
        for component, component_args in self._split(args):
            r += component._value(component_args) / component.errordef
        return r

    def _grad(self, args: Sequence[float]) -> NDArray:
        r = np.zeros(self.npar)
        for component, indices in zip(self._items, self._maps):
            component_args = tuple(args[i] for i in indices)
            r[indices] += component._grad(component_args) / component.errordef
        return r

    def _has_grad(self) -> bool:
        return all(component.has_grad for component in self._items)

    def _ndata(self):
        return sum(c.ndata for c in self._items)

    def __len__(self):
        """Return number of constituent cost functions."""
        return self._items.__len__()

    def __getitem__(self, key):
        """Get constituent cost function by index."""
        return self._items.__getitem__(key)

    def visualize(
        self, args: Sequence[float], component_kwargs: Dict[int, Dict[str, Any]] = None
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

        n = sum(hasattr(comp, "visualize") for comp in self)

        fig = plt.gcf()
        fig.set_figwidth(n * fig.get_figwidth() / 1.5)
        _, ax = plt.subplots(1, n, num=fig.number)

        if component_kwargs is None:
            component_kwargs = {}

        i = 0
        for k, (comp, cargs) in enumerate(self._split(args)):
            if not hasattr(comp, "visualize"):
                continue
            kwargs = component_kwargs.get(k, {})
            plt.sca(ax[i])
            comp.visualize(cargs, **kwargs)
            i += 1


class MaskedCost(Cost):
    """
    Base class for cost functions that support data masking.

    :meta private:
    """

    __slots__ = "_data", "_mask", "_masked"

    _mask: Optional[NDArray]

    def __init__(
        self,
        parameters: Dict[str, Optional[Tuple[float, float]]],
        data: NDArray,
        verbose: int,
    ):
        """For internal use."""
        self._data = data
        self._mask = None
        self._update_cache()
        Cost.__init__(self, parameters, verbose)

    @property
    def mask(self):
        """
        Boolean array, array of indices, or None.

        If not None, only values selected by the mask are considered. The mask acts on
        the first dimension of a value array, i.e. values[mask]. Default is None.
        """
        return self._mask

    @mask.setter
    def mask(self, mask: Optional[ArrayLike]):
        self._mask = None if mask is None else np.asarray(mask)
        self._update_cache()

    @property
    def data(self):
        """Return data samples."""
        return self._data

    @data.setter
    def data(self, value: ArrayLike):
        self._data[...] = value
        self._update_cache()

    def _update_cache(self):
        self._masked = self._data[_replace_none(self._mask, ...)]


class MaskedCostWithPulls(MaskedCost):
    """
    Base class for cost functions with pulls.

    :meta private:
    """

    def pulls(self, args: Sequence[float]) -> NDArray:
        """
        Return studentized residuals (aka pulls).

        Parameters
        ----------
        args : sequence of float
            Parameter values.

        Returns
        -------
        array
            Array of pull values. If the cost function is masked, the array contains NaN
            values where the mask value is False.

        Notes
        -----
        Pulls allow one to estimate how well a model fits the data. A pull is a value
        computed for each data point. It is given by (observed - predicted) /
        standard-deviation. If the model is correct, the expectation value of each pull
        is zero and its variance is one in the asymptotic limit of infinite samples.
        Under these conditions, the chi-square statistic is computed from the sum of
        pulls squared has a known probability distribution if the model is correct. It
        therefore serves as a goodness-of-fit statistic.

        Beware: the sum of pulls squared in general is not identical to the value
        returned by the cost function, even if the cost function returns a chi-square
        distributed test-statistic. The cost function is computed in a slightly
        differently way that makes the return value approach the asymptotic chi-square
        distribution faster than a test statistic based on sum of pulls squared. In
        summary, only use pulls for plots. Compute the chi-square test statistic
        directly from the cost function.
        """
        return self._pulls(args)

    def _ndata(self):
        return np.prod(self._masked.shape[: self._ndim])

    @abc.abstractmethod
    def _pulls(self, args: Sequence[float]) -> NDArray: ...  # pragma: no cover


class UnbinnedCost(MaskedCost):
    """
    Base class for unbinned cost functions.

    :meta private:
    """

    __slots__ = "_model", "_model_grad", "_log"

    def __init__(
        self,
        data,
        model: Model,
        verbose: int,
        log: bool,
        grad: Optional[ModelGradient],
        name: Optional[Sequence[str]],
    ):
        """For internal use."""
        self._model = model
        self._log = log
        self._model_grad = grad
        super().__init__(_model_parameters(model, name), _norm(data), verbose)

    @abc.abstractproperty
    def pdf(self):
        """Get probability density model."""
        ...  # pragma: no cover

    @abc.abstractproperty
    def scaled_pdf(self):
        """Get number density model."""
        ...  # pragma: no cover

    def _ndata(self):
        # unbinned likelihoods have infinite degrees of freedom
        return np.inf

    def _npoints(self):
        # cannot use len(self._masked) because multi-dimensional data has format
        # (K, N) with K dimensions and N points
        return self._masked.shape[-1]

    @deprecated_parameter(bins="nbins")
    def visualize(
        self,
        args: Sequence[float],
        model_points: Union[int, Sequence[float]] = 0,
        bins: int = 50,
    ):
        """
        Visualize data and model agreement (requires matplotlib).

        The visualization is drawn with matplotlib.pyplot into the current axes.

        Parameters
        ----------
        args : array-like
            Parameter values.
        model_points : int or array-like, optional
            How many points to use to draw the model. Default is 0, in this case
            an smart sampling algorithm selects the number of points. If array-like,
            it is interpreted as the point locations.
        bins : int, optional
            number of bins. Default is 50 bins.
        """
        from matplotlib import pyplot as plt

        x = np.sort(self.data)

        if x.ndim > 1:
            raise ValueError("visualize is not implemented for multi-dimensional data")

        # this implementation only works with a histogram with linear spacing

        if isinstance(model_points, Iterable):
            xm = np.array(model_points)
            ym = self.scaled_pdf(xm, *args)
        elif model_points > 0:
            if _detect_log_spacing(x):
                xm = np.geomspace(x[0], x[-1], model_points)
            else:
                xm = np.linspace(x[0], x[-1], model_points)
            ym = self.scaled_pdf(xm, *args)
        else:
            xm, ym = _smart_sampling(lambda x: self.scaled_pdf(x, *args), x[0], x[-1])

        # use xm for range, which may be narrower or wider than x range
        n, xe = np.histogram(x, bins=bins, range=(xm[0], xm[-1]))
        cx = 0.5 * (xe[1:] + xe[:-1])
        dx = xe[1] - xe[0]

        plt.errorbar(cx, n, n**0.5, fmt="ok")
        plt.fill_between(xm, 0, ym * dx, fc="C0")

    def fisher_information(self, *args: float) -> NDArray:
        """
        Estimate Fisher information for model and sample.

        The estimated Fisher information is only meaningful if the arguments provided
        are estimates of the true values.

        Parameters
        ----------
        *args: float
            Estimates of model parameters.
        """
        g = self._pointwise_score(args)
        return np.einsum("ji,ki->jk", g, g)

    def covariance(self, *args: float) -> NDArray:
        """
        Estimate covariance of the parameters with the sandwich estimator.

        This requires that the model gradient is provided, and that the arguments are
        the maximum-likelihood estimates. The sandwich estimator is only asymptotically
        correct.

        Parameters
        ----------
        *args : float
            Maximum-likelihood estimates of the parameter values.

        Returns
        -------
        ndarray of float
            The array has shape (K, K) for K arguments.
        """
        return np.linalg.inv(self.fisher_information(*args))

    @abc.abstractmethod
    def _pointwise_score(
        self, args: Sequence[float]
    ) -> NDArray: ...  # pragma: no cover

    def _has_grad(self) -> bool:
        return self._model_grad is not None


class UnbinnedNLL(UnbinnedCost):
    """
    Unbinned negative log-likelihood.

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
        """Get number density model."""
        scale = np.prod(self.data.shape)
        if self._log:
            return lambda *args: scale * np.exp(self._model(*args))
        return lambda *args: scale * self._model(*args)

    def __init__(
        self,
        data: ArrayLike,
        pdf: Model,
        *,
        verbose: int = 0,
        log: bool = False,
        grad: Optional[ModelGradient] = None,
        name: Optional[Sequence[str]] = None,
    ):
        """
        Initialize UnbinnedNLL with data and model.

        Parameters
        ----------
        data : array-like
            Sample of observations. If the observations are multidimensional, data must
            have the shape (D, N), where D is the number of dimensions and N the number
            of data points.
        pdf : callable
            Probability density function of the form f(data, par0, [par1, ...]), where
            data is the data sample and par0, ... are model parameters. If the data are
            multivariate, data passed to f has shape (D, N), where D is the number of
            dimensions and N the number of data points. Must return an array with the
            shape (N,).
        verbose : int, optional
            Verbosity level. 0: is no output (default). 1: print current args and
            negative log-likelihood value.
        log : bool, optional
            Distributions of the exponential family (normal, exponential, poisson, ...)
            allow one to compute the logarithm of the pdf directly, which is more
            accurate and efficient than numerically computing ``log(pdf)``. Set this
            to True, if the model returns the logpdf instead of the pdf.
            Default is False.
        grad : callable or None, optional
            Optionally pass the gradient of the pdf. Has the same calling signature like
            the pdf, but must return an array with the shape (K, N), where N is the
            number of data points and K is the number of parameters. If `log` is True,
            the function must return the gradient of the logpdf instead of the pdf. The
            gradient can be used by Minuit to improve or speed up convergence and to
            compute the sandwich estimator for the variance of the parameter estimates.
            Default is None.
        name : sequence of str or None, optional
            Optional names for each parameter of the model (in order). Must have the
            same length as there are model parameters. Default is None.
        """
        super().__init__(data, pdf, verbose, log, grad, name)

    def _value(self, args: Sequence[float]) -> float:
        f = self._eval_model(args)
        if self._log:
            return -2.0 * np.sum(f)
        return 2.0 * _unbinned_nll(f)

    def _grad(self, args: Sequence[float]) -> NDArray:
        g = self._pointwise_score(args)
        return -2.0 * np.sum(g, axis=1)

    def _pointwise_score(self, args: Sequence[float]) -> NDArray:
        g = self._eval_model_grad(args)
        if self._log:
            return g
        f = self._eval_model(args)
        return g / f

    def _eval_model(self, args: Sequence[float]) -> float:
        data = self._masked
        return _normalize_output(self._model(data, *args), "model", self._npoints())

    def _eval_model_grad(self, args: Sequence[float]) -> NDArray:
        if self._model_grad is None:
            raise ValueError("no gradient available")  # pragma: no cover
        data = self._masked
        return _normalize_output(
            self._model_grad(data, *args), "model gradient", self.npar, self._npoints()
        )


class ExtendedUnbinnedNLL(UnbinnedCost):
    """
    Unbinned extended negative log-likelihood.

    Use this if shape and normalization of the fitted PDF are of interest and the
    original unbinned data is available. The data can be one- or multi-dimensional.
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
        data: ArrayLike,
        scaled_pdf: Model,
        *,
        verbose: int = 0,
        log: bool = False,
        grad: Optional[ModelGradient] = None,
        name: Optional[Sequence[str]] = None,
    ):
        """
        Initialize cost function with data and model.

        Parameters
        ----------
        data : array-like
            Sample of observations. If the observations are multidimensional, data must
            have the shape (D, N), where D is the number of dimensions and N the number
            of data points.
        scaled_pdf : callable
            Density function of the form f(data, par0, [par1, ...]), where data is the
            sample and par0, ... are model parameters. Must return a tuple (<integral
            over f in data window>, <f evaluated at data points>). The first value is
            the density integrated over the data window, the interval that we consider
            for the fit. For example, if the data are exponentially distributed, but we
            fit only the interval (0, 5), then the first value is the density integrated
            from 0 to 5. If the data are multivariate, data passed to f has shape (D,
            N), where D is the number of dimensions and N the number of data points.
        verbose : int, optional
            Verbosity level. 0: is no output (default). 1: print current args and
            negative log-likelihood value.
        log : bool, optional
            Distributions of the exponential family (normal, exponential, poisson, ...)
            allow one to compute the logarithm of the pdf directly, which is more
            accurate and efficient than effectively doing ``log(exp(logpdf))``. Set this
            to True, if the model returns the logarithm of the density as the second
            argument instead of the density. Default is False.
        grad : callable or None, optional
            Optionally pass the gradient of the density function. Has the same calling
            signature like the density function, but must return two arrays. The first
            array has shape (K,) where K are the number of parameters, while the second
            has shape (K, N), where N is the number of data points. The first array is
            the gradient of the integrated density. The second array is the gradient of
            the density itself. If `log` is True, the second array must be the gradient
            of the log-density instead. The gradient can be used by Minuit to improve or
            speed up convergence and to compute the sandwich estimator for the variance
            of the parameter estimates. Default is None.
        name : sequence of str or None, optional
            Optional names for each parameter of the model (in order). Must have the
            same length as there are model parameters. Default is None.
        """
        super().__init__(data, scaled_pdf, verbose, log, grad, name)

    def _value(self, args: Sequence[float]) -> float:
        fint, f = self._eval_model(args)
        if self._log:
            return 2 * (fint - np.sum(f))
        return 2 * (fint + _unbinned_nll(f))

    def _grad(self, args: Sequence[float]) -> NDArray:
        g = self._pointwise_score(args)
        return -2 * np.sum(g, axis=1)

    def _pointwise_score(self, args: Sequence[float]) -> NDArray:
        gint, g = self._eval_model_grad(args)
        m = self._npoints()
        if self._log:
            return g - (gint / m)[:, np.newaxis]
        _, f = self._eval_model(args)
        return g / f - (gint / m)[:, np.newaxis]

    def _eval_model(self, args: Sequence[float]) -> Tuple[float, float]:
        data = self._masked
        fint, f = self._model(data, *args)
        f = _normalize_output(f, "model", self._npoints(), msg="in second position")
        return fint, f

    def _eval_model_grad(self, args: Sequence[float]) -> Tuple[NDArray, NDArray]:
        if self._model_grad is None:
            raise ValueError("no gradient available")  # pragma: no cover
        data = self._masked
        gint, g = self._model_grad(data, *args)
        gint = _normalize_output(
            gint, "model gradient", self.npar, msg="in first position"
        )
        g = _normalize_output(
            g, "model gradient", self.npar, self._npoints(), msg="in second position"
        )
        return gint, g


class BinnedCost(MaskedCostWithPulls):
    """
    Base class for binned cost functions to support histograms filled with weights.

    Histograms filled with weights are supported by applying the Bohm-Zech transform.
    See Bohm and Zech, NIMA 748 (2014) 1-6.

    :meta private:
    """

    __slots__ = "_xe", "_ndim", "_bohm_zech_scale", "_bohm_zech_n"

    _xe: Union[NDArray, Tuple[NDArray, ...]]
    _ndim: int
    _bohm_zech_scale: Optional[NDArray]
    _bohm_zech_n: Optional[NDArray]

    n = MaskedCost.data

    @property
    def xe(self):
        """Access bin edges."""
        return self._xe

    def __init__(
        self,
        parameters: Dict[str, Optional[Tuple[float, float]]],
        n: ArrayLike,
        xe: Union[ArrayLike, Sequence[ArrayLike]],
        verbose: int,
    ):
        """For internal use."""
        if not isinstance(xe, Iterable):
            raise ValueError("xe must be iterable")

        shape = _shape_from_xe(xe)
        self._ndim = len(shape)
        if self._ndim == 1:
            self._xe = _norm(cast(ArrayLike, xe))
        else:
            self._xe = tuple(_norm(xei) for xei in xe)

        n = _norm(n)

        is_weighted = n.ndim > self._ndim and n.shape[-1] == 2

        if n.ndim != (self._ndim + int(is_weighted)):
            raise ValueError("n must either have same dimension as xe or one extra")

        xei: NDArray
        for i, xei in enumerate([self._xe] if self._ndim == 1 else self._xe):
            if len(xei) != n.shape[i] + 1:
                raise ValueError(
                    f"n and xe have incompatible shapes along dimension {i}, "
                    "xe must be longer by one element along each dimension"
                )

        self._bohm_zech_scale = None
        self._bohm_zech_n = None
        self._set_bohm_zech(n, is_weighted)
        super().__init__(parameters, n, verbose)

    def prediction(
        self, args: Sequence[float]
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Return the bin-wise expectation for the fitted model.

        Parameters
        ----------
        args : array-like
            Parameter values.

        Returns
        -------
        NDArray
            Model prediction for each bin. The expectation is always returned for all
            bins, even if some bins are temporarily masked.
        """
        return self._pred(args)

    def visualize(self, args: Sequence[float]) -> None:
        """
        Visualize data and model agreement (requires matplotlib).

        The visualization is drawn with matplotlib.pyplot into the current axes.

        Parameters
        ----------
        args : sequence of float
            Parameter values.

        Notes
        -----
        The automatically provided visualization for multi-dimensional data set is often
        not very pretty, but still helps to judge whether the fit is reasonable. Since
        there is no obvious way to draw higher dimensional data with error bars in
        comparison to a model, the visualization shows all data bins as a single
        sequence.
        """
        return self._visualize(args)

    def _visualize(self, args: Sequence[float]) -> None:
        from matplotlib import pyplot as plt

        n, ne = self._n_err()
        mu = self.prediction(args)
        assert not isinstance(mu, tuple)

        if self._ndim > 1:
            # flatten higher-dimensional data
            n = n.reshape(-1)
            ne = ne.reshape(-1)
            mu = mu.reshape(-1)
            # just use bin numbers instead of original values
            xe = np.arange(len(n) + 1) - 0.5
            cx = np.arange(len(n)).astype(float)
        else:
            xe = self.xe
            cx = 0.5 * (xe[1:] + xe[:-1])
        plt.errorbar(cx, n, ne, fmt="ok")
        plt.stairs(mu, xe, fill=True, color="C0")

    @abc.abstractmethod
    def _pred(
        self, args: Sequence[float]
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]: ...  # pragma: no cover

    def _n_err(self) -> Tuple[NDArray, NDArray]:
        d = self.data
        if self._bohm_zech_scale is None:
            n = d.copy()
            err = d**0.5
        else:
            n = d[..., 0].copy()
            err = d[..., 1] ** 0.5
        # mask values where error is zero
        ma = err == 0
        if self.mask is not None:
            ma = ~self.mask
        n[ma] = np.nan
        err[ma] = np.nan
        return n, err

    def _pulls(self, args: Sequence[float]) -> NDArray:
        mu = self.prediction(args)
        n, ne = self._n_err()
        return (n - mu) / ne

    def _set_bohm_zech(self, n: NDArray, is_weighted: bool):
        if not is_weighted:
            return
        val = n[..., 0]
        var = n[..., 1]
        self._bohm_zech_scale = np.ones_like(val)
        np.divide(val, var, out=self._bohm_zech_scale, where=var > 0)
        self._bohm_zech_n = val * self._bohm_zech_scale

    def _update_cache(self):
        super()._update_cache()
        self._set_bohm_zech(self._masked, self._bohm_zech_scale is not None)

    @overload
    def _transformed(
        self, val: NDArray
    ) -> Tuple[NDArray, NDArray]: ...  # pragma: no cover

    @overload
    def _transformed(
        self, val: NDArray, var: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray]: ...  # pragma: no cover

    def _transformed(self, val, var=None):
        s = self._bohm_zech_scale
        ma = self.mask
        if ma is not None:
            val = val[ma]
            if var is not None:
                var = var[ma]
        if s is None:
            if var is None:
                return self._masked, val
            return self._masked, val, var
        if var is None:
            return self._bohm_zech_n, val * s
        return self._bohm_zech_n, val * s, var * s**2

    def _counts(self):
        if self._bohm_zech_scale is None:
            return self._masked
        return self._masked[..., 0]


class BinnedCostWithModel(BinnedCost):
    """
    Base class for binned cost functions with parametric model.

    :meta private:
    """

    __slots__ = (
        "_xe_shape",
        "_model",
        "_model_xe",
        "_model_xm",
        "_model_dx",
        "_model_len",
        "_model_grad",
        "_pred_impl",
    )

    _model_xe: np.ndarray
    _xe_shape: Union[Tuple[int], Tuple[int, ...]]

    def __init__(self, n, xe, model, verbose, grad, use_pdf, name):
        """For internal use."""
        self._model = model
        self._model_grad = grad

        if use_pdf and grad:
            raise ValueError("keywords use_pdf and grad cannot be used together")

        if use_pdf == "approximate":
            self._pred_impl = self._pred_approximate
        elif use_pdf == "numerical":
            self._pred_impl = self._pred_numerical
        elif use_pdf == "":
            self._pred_impl = self._pred_cdf
        else:
            msg = (
                f"use_pdf={use_pdf} is not understood, "
                "allowed values are '', 'approximate', or 'numerical'"
            )
            raise ValueError(msg)

        super().__init__(_model_parameters(model, name), n, xe, verbose)

        if self._ndim == 1:
            self._xe_shape = (len(self.xe),)
            self._model_xe = _norm(self.xe)
            if use_pdf:
                dx = np.diff(self._model_xe)
                self._model_dx = dx
                self._model_xm = self._model_xe[:-1] + 0.5 * dx
        else:
            self._xe_shape = tuple(len(xei) for xei in self.xe)
            self._model_xe = np.vstack(
                [x.flatten() for x in np.meshgrid(*self.xe, indexing="ij")]
            )
            if use_pdf == "approximate":
                dx = [np.diff(xe) for xe in self.xe]
                xm = [xei[:-1] + 0.5 * dxi for (xei, dxi) in zip(self.xe, dx)]
                xm = np.meshgrid(*xm, indexing="ij")
                dx = np.meshgrid(*dx, indexing="ij")
                self._model_xm = np.array(xm)
                self._model_dx = np.prod(dx, axis=0)
            elif use_pdf == "numerical":
                raise ValueError(
                    'use_pdf="numerical" is not supported for '
                    "multidimensional histograms"
                )

        self._model_len = np.prod(self._xe_shape)

    def _pred(self, args: Sequence[float]) -> NDArray:
        return self._pred_impl(args)

    def _pred_cdf(self, args: Sequence[float]) -> NDArray:
        d = self._model(self._model_xe, *args)
        d = _normalize_output(d, "model", self._model_len)
        if self._ndim > 1:
            d = d.reshape(self._xe_shape)
        for i in range(self._ndim):
            d = np.diff(d, axis=i)
        # differences can come out negative due to round-off error in subtraction,
        # we set negative values to zero
        d[d < 0] = 0
        return d

    def _pred_approximate(self, args: Sequence[float]) -> NDArray:
        y = self._model(self._model_xm, *args)
        return y * self._model_dx

    def _pred_numerical(self, args: Sequence[float]) -> NDArray:
        from scipy.integrate import quad

        assert self._ndim == 1

        d = np.empty(self._model_len - 1)
        for i in range(self._model_len - 1):
            a = self._model_xe[i]
            b = self._model_xe[i + 1]
            d[i] = quad(lambda x: self._model(x, *args), a, b)[0]
        return d

    def _pred_grad(self, args: Sequence[float]) -> NDArray:
        d = self._model_grad(self._model_xe, *args)
        d = _normalize_output(d, "model gradient", self.npar, self._model_len)
        if self._ndim > 1:
            d = d.reshape((self.npar, *self._xe_shape))
        for i in range(1, self._ndim + 1):
            d = np.diff(d, axis=i)
        return d

    def _has_grad(self) -> bool:
        return self._model_grad is not None


class Template(BinnedCost):
    """
    Binned cost function for a template fit with uncertainties on the template.

    This cost function is for a mixture of components. Use this if the sample originate
    from two or more components and you are interested in estimating the yield that
    originates from one or more components. In high-energy physics, one component is
    often a peaking signal over a smooth background component. A component can be
    described by a parametric model or a template.

    A parametric model is accepted in form of a scaled cumulative density function,
    while a template is a non-parametric shape estimate obtained by histogramming a
    Monte-Carlo simulation. Even if the Monte-Carlo simulation is asymptotically
    correct, estimating the shape from a finite simulation sample introduces some
    uncertainty. This cost function takes that additional uncertainty into account.

    There are several ways to fit templates and take the sampling uncertainty into
    account. Barlow and Beeston [1]_ found an exact likelihood for this problem, with
    one nuisance parameter per component per bin. Solving this likelihood is somewhat
    challenging though. The Barlow-Beeston likelihood also does not handle the
    additional uncertainty in weighted templates unless the weights per bin are all
    equal.

    Other works [2]_ [3]_ [4]_ describe likelihoods that use only one nuisance parameter
    per bin, which is an approximation. Some marginalize over the nuisance parameters
    with some prior, while others profile over the nuisance parameter. This class
    implements several of these methods. The default method is the one which performs
    best under most conditions, according to current knowledge. The default may change
    if this assessment changes.

    The cost function returns an asymptotically chi-square distributed test statistic,
    except for the method "asy", where it is the negative logarithm of the marginalised
    likelihood instead. The standard transform [5]_ which we use convert likelihoods
    into test statistics only works for (profiled) likelihoods, not for likelihoods
    marginalized over a prior.

    All methods implemented here have been generalized to work with both weighted data
    and weighted templates, under the assumption that the weights are independent of the
    data. This is not the case for sWeights, and the uncertaintes for results obtained
    with sWeights will only be approximately correct [6]_. The methods have been further
    generalized to allow fitting a mixture of parametric models and templates.

    .. [1] Barlow and Beeston, Comput.Phys.Commun. 77 (1993) 219-228
    .. [2] Conway, PHYSTAT 2011 proceeding, https://doi.org/10.48550/arXiv.1103.0354
    .. [3] Argüelles, Schneider, Yuan, JHEP 06 (2019) 030
    .. [4] Dembinski and Abdelmotteleb, https://doi.org/10.48550/arXiv.2206.12346
    .. [5] Baker and Cousins, NIM 221 (1984) 437-442
    .. [6] Langenbruch, Eur.Phys.J.C 82 (2022) 5, 393
    """

    __slots__ = "_model_data", "_model_xe", "_xe_shape", "_impl", "_model_len"

    _model_data: List[
        Union[
            Tuple[NDArray, NDArray],
            Tuple[Model, float],
        ]
    ]
    _model_xe: np.ndarray
    _xe_shape: Union[Tuple[int], Tuple[int, ...]]

    def __init__(
        self,
        n: ArrayLike,
        xe: Union[ArrayLike, Sequence[ArrayLike]],
        model_or_template: Collection[Union[Model, ArrayLike]],
        *,
        name: Optional[Sequence[str]] = None,
        verbose: int = 0,
        method: str = "da",
    ):
        """
        Initialize cost function with data and model.

        Parameters
        ----------
        n : array-like
            Histogram counts. If this is an array with dimension D+1, where D is the
            number of histogram axes, then the last dimension must have two elements and
            is interpreted as pairs of sum of weights and sum of weights squared.
        xe : array-like or collection of array-like
            Bin edge locations, must be len(n) + 1, where n is the number of bins. If
            the histogram has more than one axis, xe must be a collection of the bin
            edge locations along each axis.
        model_or_template : collection of array-like or callable
            Collection of models or arrays. An array represent the histogram counts of a
            template. The template histograms must use the same axes as the data
            histogram. If the counts are represented by an array with dimension D+1,
            where D is the number of histogram axes, then the last dimension must have
            two elements and is interpreted as pairs of sum of weights and sum of
            weights squared. Callables must return the model cdf evaluated as xe.
        name : sequence of str or None, optional
            Optional name for the yield of each template and the parameter of each model
            (in order). Must have the same length as there are templates and model
            parameters in templates_or_model. Default is None.
        verbose : int, optional
            Verbosity level. 0: is no output (default). 1: print current args and
            negative log-likelihood value.
        method : {"jsc", "asy", "da"}, optional
            Which method to use. "jsc": Conway's method [2]_. "asy": ASY method [3]_.
            "da": DA method [4]_. Default is "da", which to current knowledge offers the
            best overall performance. The default may change in the future, so please
            set this parameter explicitly in code that has to be stable. For all methods
            except the "asy" method, the minimum value is chi-square distributed.
        """
        M = len(model_or_template)
        if M < 1:
            raise ValueError("at least one template or model is required")

        shape = _shape_from_xe(xe)
        ndim = len(shape)

        npar = 0
        annotated: Dict[str, Optional[Tuple[float, float]]] = {}
        self._model_data = []
        for i, t in enumerate(model_or_template):
            if isinstance(t, Collection):
                tt = _norm(t)
                if tt.ndim > ndim:
                    # template is weighted
                    if tt.ndim != ndim + 1 or tt.shape[:-1] != shape:
                        raise ValueError("shapes of n and templates do not match")
                    t1 = tt[..., 0].copy()
                    t2 = tt[..., 1].copy()
                else:
                    if tt.ndim != ndim or tt.shape != shape:
                        raise ValueError("shapes of n and templates do not match")
                    t1 = tt.copy()
                    t2 = tt.copy()
                # normalize to unity
                f = 1 / np.sum(t1)
                t1 *= f
                t2 *= f**2
                self._model_data.append((t1, t2))
                annotated[f"x{i}"] = (0.0, np.inf)
            elif isinstance(t, Model):
                ann = _model_parameters(t, None)
                npar = len(ann)
                self._model_data.append((t, npar))
                for k in ann:
                    annotated[f"x{i}_{k}"] = ann[k]
            else:
                raise ValueError(
                    "model_or_template must be a collection of array-likes "
                    "and/or Model types"
                )

        if name is not None:
            if len(annotated) != len(name):
                raise ValueError(
                    "number of names must match number of templates and "
                    "model parameters"
                )
            annotated = {new: annotated[old] for (old, new) in zip(annotated, name)}

        known_methods = {
            "jsc": template_chi2_jsc,
            "asy": template_nll_asy,
            "hpd": template_chi2_da,
            "da": template_chi2_da,
        }
        try:
            self._impl = known_methods[method]
        except KeyError:
            raise ValueError(
                f"method {method} is not understood, allowed values: {known_methods}"
            )

        if method == "hpd":
            warnings.warn(
                "key 'hpd' is deprecated, please use 'da' instead",
                category=FutureWarning,
                stacklevel=2,
            )

        super().__init__(annotated, n, xe, verbose)

        if self._ndim == 1:
            self._xe_shape = (len(self.xe),)
            self._model_xe = _norm(self.xe)
        else:
            self._xe_shape = tuple(len(xei) for xei in self.xe)
            self._model_xe = np.vstack(
                [x.flatten() for x in np.meshgrid(*self.xe, indexing="ij")]
            )
        self._model_len = np.prod(self._xe_shape)

    def _pred(self, args: Sequence[float]) -> Tuple[NDArray, NDArray]:
        mu: NDArray = 0  # type:ignore
        mu_var: NDArray = 0  # type:ignore
        i = 0
        for t1, t2 in self._model_data:
            if isinstance(t1, np.ndarray) and isinstance(t2, np.ndarray):
                a = args[i]
                mu += a * t1
                mu_var += a**2 * t2
                i += 1
            elif isinstance(t1, Model) and isinstance(t2, int):
                d = t1(self._model_xe, *args[i : i + t2])
                d = _normalize_output(d, "model", self._model_len)
                if self._ndim > 1:
                    d = d.reshape(self._xe_shape)
                for j in range(self._ndim):
                    d = np.diff(d, axis=j)
                # differences can come out negative due to round-off error in
                # subtraction, we set negative values to zero
                d[d < 0] = 0
                mu += d
                mu_var += np.ones_like(mu) * 1e-300
                i += t2
            else:  # never arrive here
                assert False  # pragma: no cover
        return mu, mu_var

    def _value(self, args: Sequence[float]) -> float:
        mu, mu_var = self._pred(args)
        n, mu, mu_var = self._transformed(mu, mu_var)
        ma = mu > 0
        return self._impl(n[ma], mu[ma], mu_var[ma])

    def _grad(self, args: Sequence[float]) -> NDArray:
        raise NotImplementedError  # pragma: no cover

    def _has_grad(self) -> bool:
        return False

    def _errordef(self) -> float:
        return NEGATIVE_LOG_LIKELIHOOD if self._impl is template_nll_asy else CHISQUARE

    def prediction(self, args: Sequence[float]) -> Tuple[NDArray, NDArray]:
        """
        Return the fitted template and its standard deviation.

        This returns the prediction from the templates, the sum over the products of the
        template yields with the normalized templates. The standard deviation is
        returned as the second argument, this is the estimated uncertainty of the fitted
        template alone. It is obtained via error propagation, taking the statistical
        uncertainty in the template into account, but regarding the yields as parameters
        without uncertainty.

        Parameters
        ----------
        args : array-like
            Parameter values.

        Returns
        -------
        y, yerr : NDArray, NDArray
            Template prediction and its standard deviation, based on the statistical
            uncertainty of the template only.
        """
        mu, mu_var = self._pred(args)
        return mu, np.sqrt(mu_var)

    def _visualize(self, args: Sequence[float]) -> None:
        from matplotlib import pyplot as plt

        n, ne = self._n_err()
        mu, mue = self.prediction(args)  # type: ignore

        # see implementation notes in BinnedCost.visualize
        if self._ndim > 1:
            n = n.reshape(-1)
            ne = ne.reshape(-1)
            mu = mu.reshape(-1)
            mue = mue.reshape(-1)
            xe = np.arange(len(n) + 1) - 0.5
            cx = np.arange(len(n)).astype(float)
        else:
            xe = self.xe
            cx = 0.5 * (xe[1:] + xe[:-1])

        plt.errorbar(cx, n, ne, fmt="ok")

        # need fill=True and fill=False so that bins with mue=0 show up
        for fill in (False, True):
            plt.stairs(mu + mue, xe, baseline=mu - mue, fill=fill, color="C0")

    def _pulls(self, args: Sequence[float]) -> NDArray:
        mu, mue = self.prediction(args)
        n, ne = self._n_err()
        return (n - mu) / (mue**2 + ne**2) ** 0.5


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
        self,
        n: ArrayLike,
        xe: Union[ArrayLike, Sequence[ArrayLike]],
        cdf: Model,
        *,
        verbose: int = 0,
        grad: Optional[ModelGradient] = None,
        use_pdf: str = "",
        name: Optional[Sequence[str]] = None,
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
        grad: callable or None, optional
            Optionally pass the gradient of the cdf (Default is None). Has the same
            calling signature like the cdf, but must return an array with the shape (K,
            N), where N is the number of data points and K is the number of parameters.
            The gradient can be used by Minuit to improve or speed up convergence.
        use_pdf: str, optional
            Either "", "numerical", or "approximate" (Default is ""). If the model cdf
            is not available, but the model pdf is, this option can be set to
            "numerical" or "approximate" to compute the integral of the pdf over the bin
            patch. The option "numerical" uses numerical integration, which is accurate
            but computationally expensive and only supported for 1D histograms. The
            option "approximate" uses the zero-order approximation of evaluating the pdf
            at the bin center, multiplied with the bin area. This is fast and works in
            higher dimensions, but can lead to biased results if the curvature of the
            pdf inside the bin is significant.
        name : sequence of str or None, optional
            Optional names for each parameter of the model (in order). Must have the
            same length as there are model parameters. Default is None.
        """
        super().__init__(n, xe, cdf, verbose, grad, use_pdf, name)

    def _pred(self, args: Sequence[float]) -> NDArray:
        # must return array of full length, mask not applied yet
        p = super()._pred(args)
        # normalise probability of remaining bins
        ma = self.mask
        if ma is not None:
            p /= np.sum(p[ma])
        # scale probabilities with total number of entries of unmasked bins in histogram
        return p * np.sum(self._counts())

    def _value(self, args: Sequence[float]) -> float:
        mu = self._pred(args)
        n, mu = self._transformed(mu)
        return multinominal_chi2(n, mu)

    def _grad(self, args: Sequence[float]) -> NDArray:
        # pg and p must be arrays of full length, mask not applied yet
        pg = super()._pred_grad(args)
        p = super()._pred(args)
        ma = self.mask
        # normalise probability of remaining bins
        if ma is not None:
            scale = np.sum(p[ma])
            pg = pg / scale - p * np.sum(pg[:, ma]) / scale**2
            p /= scale
        # scale probabilities with total number of entries of unmasked bins in histogram
        scale = np.sum(self._counts())
        mu = p * scale
        gmu = pg * scale
        ma = self.mask
        if ma is not None:
            mu = mu[ma]
            gmu = gmu[:, ma]
        # don't need to scale mu and gmu, because scale factor cancels
        n = self._masked if self._bohm_zech_scale is None else self._bohm_zech_n
        return _multinominal_chi2_grad(n, mu, gmu)


class ExtendedBinnedNLL(BinnedCostWithModel):
    """
    Binned extended negative log-likelihood.

    Use this if shape and normalization of the fitted PDF are of interest and the data
    is binned. This cost function works with normal and weighted histograms. The
    histogram can be one- or multi-dimensional.

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
        n: ArrayLike,
        xe: Union[ArrayLike, Sequence[ArrayLike]],
        scaled_cdf: Model,
        *,
        verbose: int = 0,
        grad: Optional[ModelGradient] = None,
        use_pdf: str = "",
        name: Optional[Sequence[str]] = None,
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
            Scaled Cumulative density function of the form f(xe, par0, [par1, ...]),
            where xe is a bin edge and par0, ... are model parameters.  If the model is
            multivariate, xe must be an array-like with shape (D, N), where D is the
            dimension and N is the number of points where the model is evaluated.
        verbose : int, optional
            Verbosity level. 0: is no output (default). 1: print current args and
            negative log-likelihood value.
        grad: callable or None, optional
            Optionally pass the gradient of the cdf (Default is None). Has the same
            calling signature like the cdf, but must return an array with the shape (K,
            N), where N is the number of data points and K is the number of parameters.
            The gradient can be used by Minuit to improve or speed up convergence.
        use_pdf: str, optional
            Either "", "numerical", or "approximate". If the model cdf is not available,
            but the model pdf is, this option can be set to "numerical" or "approximate"
            to compute the integral of the pdf over the bin patch. The option
            "numerical" uses numerical integration, which is accurate but
            computationally expensive and only supported for 1D histograms. The option
            "approximate" uses the zero-order approximation of evaluating the pdf at the
            bin center, multiplied with the bin area. This is fast and works in higher
            dimensions, but can lead to biased results if the curvature of the pdf
            inside the bin is significant.
        name : sequence of str or None, optional
            Optional names for each parameter of the model (in order). Must have the
            same length as there are model parameters. Default is None.
        """
        super().__init__(n, xe, scaled_cdf, verbose, grad, use_pdf, name)

    def _value(self, args: Sequence[float]) -> float:
        mu = self._pred(args)
        n, mu = self._transformed(mu)
        return poisson_chi2(n, mu)

    def _grad(self, args: Sequence[float]) -> NDArray:
        mu = self._pred(args)
        gmu = self._pred_grad(args)
        ma = self.mask
        if ma is not None:
            mu = mu[ma]
            gmu = gmu[:, ma]
        n = self._counts()
        s = self._bohm_zech_scale
        if s is None:
            return _poisson_chi2_grad(n, mu, gmu)
        # use original n and mu because Bohm-Zech scale factor cancels
        return _poisson_chi2_grad(n, mu, s * gmu)


class LeastSquares(MaskedCostWithPulls):
    """
    Least-squares cost function (aka chisquare function).

    Use this if you have data of the form (x, y +/- yerror), where x can be
    one-dimensional or multi-dimensional, but y is always one-dimensional. See
    :meth:`__init__` for details on how to use a multivariate model.
    """

    __slots__ = "_loss", "_cost", "_cost_grad", "_model", "_model_grad", "_ndim"

    _loss: Union[str, LossFunction]
    _cost: Callable[[ArrayLike, ArrayLike, ArrayLike], float]
    _cost_grad: Optional[Callable[[NDArray, NDArray, NDArray, NDArray], NDArray]]
    _model: Model
    _model_grad: Optional[ModelGradient]
    _ndim: int

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
        if len(self._parameters) == 1:
            return lambda x, *args: (
                self._model(x, args) if len(args) > 1 else self._model(x, *args)
            )
        else:
            return self._model

    @property
    def loss(self):
        """Get loss function."""
        return self._loss

    @loss.setter
    def loss(self, loss: Union[str, LossFunction]):
        self._loss = loss
        if isinstance(loss, str):
            if loss == "linear":
                self._cost = chi2
                self._cost_grad = _chi2_grad
            elif loss == "soft_l1":
                self._cost = _soft_l1_cost  # type: ignore
                self._cost_grad = _soft_l1_cost_grad
            else:
                raise ValueError(f"unknown loss {loss!r}")
        elif isinstance(loss, LossFunction):
            self._cost = lambda y, ye, ym: np.sum(
                loss(_z_squared(y, ye, ym))  # type:ignore
            )
            self._cost_grad = None
        else:
            raise ValueError("loss must be str or LossFunction")

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        yerror: ArrayLike,
        model: Model,
        *,
        loss: Union[str, LossFunction] = "linear",
        verbose: int = 0,
        grad: Optional[ModelGradient] = None,
        name: Optional[Sequence[str]] = None,
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
            The loss function can be modified to make the fit robust against outliers,
            see scipy.optimize.least_squares for details. Only "linear" (default) and
            "soft_l1" are currently implemented, but users can pass any loss function as
            this argument. It should be a monotonic, twice differentiable function,
            which accepts the squared residual and returns a modified squared residual.
        verbose : int, optional
            Verbosity level. 0: is no output (default). 1: print current args and
            negative log-likelihood value.

        Notes
        -----
        Alternative loss functions make the fit more robust against outliers by
        weakening the pull of outliers. The mechanical analog of a least-squares fit is
        a system with attractive forces. The points pull the model towards them with a
        force whose potential is given by :math:`rho(z)` for a squared-offset :math:`z`.
        The plot shows the standard potential in comparison with the weaker soft-l1
        potential, in which outliers act with a constant force independent of their
        distance.

        .. plot:: plots/loss.py
        """
        x = _norm(x)
        y = _norm(y)
        assert x.ndim >= 1  # guaranteed by _norm

        self._ndim = x.shape[0] if x.ndim > 1 else 1
        self._model = model
        self._model_grad = grad
        self.loss = loss

        x = np.atleast_2d(x)
        data = np.column_stack(np.broadcast_arrays(*x, y, yerror))
        super().__init__(_model_parameters(model, name), data, verbose)

    def _ndata(self):
        return len(self._masked)

    def visualize(
        self, args: ArrayLike, model_points: Union[int, Sequence[float]] = 0
    ) -> Tuple[Tuple[NDArray, NDArray, NDArray], Tuple[NDArray, NDArray]]:
        """
        Visualize data and model agreement (requires matplotlib).

        The visualization is drawn with matplotlib.pyplot into the current axes.

        Parameters
        ----------
        args : array-like
            Parameter values.

        model_points : int or array-like, optional
            How many points to use to draw the model. Default is 0, in this case
            an smart sampling algorithm selects the number of points. If array-like,
            it is interpreted as the point locations.
        """
        from matplotlib import pyplot as plt

        if self._ndim > 1:
            raise ValueError("visualize is not implemented for multi-dimensional data")

        x, y, ye = self._masked.T
        plt.errorbar(x, y, ye, fmt="ok")
        if isinstance(model_points, Iterable):
            xm = np.array(model_points)
            ym = self.model(xm, *args)
        elif model_points > 0:
            if _detect_log_spacing(x):
                xm = np.geomspace(x[0], x[-1], model_points)
            else:
                xm = np.linspace(x[0], x[-1], model_points)
            ym = self.model(xm, *args)
        else:
            xm, ym = _smart_sampling(lambda x: self.model(x, *args), x[0], x[-1])
        plt.plot(xm, ym)
        return (x, y, ye), (xm, ym)

    def prediction(self, args: Sequence[float]) -> NDArray:
        """
        Return the prediction from the fitted model.

        Parameters
        ----------
        args : array-like
            Parameter values.

        Returns
        -------
        NDArray
            Model prediction for each bin.
        """
        return self.model(self.x, *args)

    def _pulls(self, args: Sequence[float]) -> NDArray:
        y = self.y.copy()
        ye = self.yerror.copy()
        ym = self.prediction(args)

        if self.mask is not None:
            ma = ~self.mask
            y[ma] = np.nan
            ye[ma] = np.nan
        return (y - ym) / ye

    def _pred(self, args: Sequence[float]) -> NDArray:
        x = self._masked.T[0] if self._ndim == 1 else self._masked.T[: self._ndim]
        ym = self._model(x, *args)
        return _normalize_output(ym, "model", self._ndata())

    def _pred_grad(self, args: Sequence[float]) -> NDArray:
        if self._model_grad is None:
            raise ValueError("no gradient available")  # pragma: no cover
        x = self._masked.T[0] if self._ndim == 1 else self._masked.T[: self._ndim]
        ymg = self._model_grad(x, *args)
        return _normalize_output(ymg, "model gradient", self.npar, self._ndata())

    def _value(self, args: Sequence[float]) -> float:
        y, ye = self._masked.T[self._ndim :]
        ym = self._pred(args)
        return self._cost(y, ye, ym)

    def _grad(self, args: Sequence[float]) -> NDArray:
        if self._cost_grad is None:
            raise ValueError("no cost gradient available")  # pragma: no cover
        y, ye = self._masked.T[self._ndim :]
        ym = self._pred(args)
        ymg = self._pred_grad(args)
        return self._cost_grad(y, ye, ym, ymg)

    def _has_grad(self) -> bool:
        return self._model_grad is not None and self._cost_grad is not None


class NormalConstraint(Cost):
    """
    Gaussian penalty for one or several parameters.

    The Gaussian penalty acts like a pseudo-measurement of the parameter itself, based
    on a (multi-variate) normal distribution. Penalties can be set for one or several
    parameters at once (which is more efficient). When several parameter are
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
    sample, a Gaussian penalty is an accurate statistical representation of the external
    result.
    """

    __slots__ = "_expected", "_cov", "_covinv"

    def __init__(
        self,
        args: Union[str, Iterable[str]],
        value: ArrayLike,
        error: ArrayLike,
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
            Expected error(s). If 1D, must have same length as `args`. If 2D, must be
            the covariance matrix of the parameters.
        """
        tp_args = (args,) if isinstance(args, str) else tuple(args)
        nargs = len(tp_args)
        self._expected = _norm(value)
        if self._expected.ndim > 1:
            raise ValueError("value must be a scalar or one-dimensional")
        # args can be a vector of values, in this case we have nargs == 1
        if nargs > 1 and len(self._expected) != nargs:
            raise ValueError("size of value does not match size of args")
        self._cov = _norm(error)
        if len(self._cov) != len(self._expected):
            raise ValueError("size of error does not match size of value")
        if self._cov.ndim < 2:
            self._cov **= 2
        elif self._cov.ndim == 2:
            if not is_positive_definite(self._cov):
                raise ValueError("covariance matrix is not positive definite")
        else:
            raise ValueError("covariance matrix cannot have more than two dimensions")
        self._covinv = _covinv(self._cov)
        super().__init__({k: None for k in tp_args}, False)

    @property
    def covariance(self):
        """
        Get expected covariance of parameters.

        Can be 1D (diagonal of covariance matrix) or 2D (full covariance matrix).
        """
        return self._cov

    @covariance.setter
    def covariance(self, value):
        value = np.asarray(value)
        if value.ndim == 2 and not is_positive_definite(value):
            raise ValueError("covariance matrix is not positive definite")
        self._cov[:] = value
        self._covinv = _covinv(self._cov)

    @property
    def value(self):
        """Get expected parameter values."""
        return self._expected

    @value.setter
    def value(self, value):
        self._expected[:] = value

    def _value(self, args: Sequence[float]) -> float:
        delta = args - self._expected
        if self._covinv.ndim < 2:
            return np.sum(delta**2 * self._covinv)
        return np.einsum("i,ij,j", delta, self._covinv, delta)

    def _grad(self, args: Sequence[float]) -> NDArray:
        delta = args - self._expected
        if self._covinv.ndim < 2:
            return 2 * delta * self._covinv
        return 2 * self._covinv @ delta

    def _has_grad(self) -> bool:
        return True

    def _ndata(self):
        return len(self._expected)

    def visualize(self, args: ArrayLike):
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

        par = self._parameters
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


def _norm(value: ArrayLike) -> NDArray:
    value = np.atleast_1d(value)
    dtype = value.dtype
    if dtype.kind != "f":
        value = value.astype(np.float64)
    return value


def _covinv(array):
    return np.linalg.inv(array) if array.ndim == 2 else 1.0 / array


def _normalize_output(x, kind, *shape, msg=None):
    if not isinstance(x, np.ndarray):
        if msg is None:
            msg = f"{kind} should return numpy array, but returns {type(x)}"
        else:
            msg = f"{kind} should return numpy array {msg}, but returns {type(x)}"
        warnings.warn(msg, PerformanceWarning)
        x = np.array(x)
        if x.dtype.kind != "f":
            return x.astype(float)
    if x.ndim < len(shape):
        return x.reshape(*shape)
    elif x.shape != shape:
        # NumPy 2 uses a numpy int here
        pretty_shape = tuple(int(i) for i in shape)
        msg = (
            f"output of {kind} has shape {x.shape!r}, but {pretty_shape!r} is required"
        )
        raise ValueError(msg)
    return x


def _shape_from_xe(xe):
    if isinstance(xe[0], Iterable):
        return tuple(len(xei) - 1 for xei in xe)
    return (len(xe) - 1,)


def _model_parameters(model, name):
    # strip first argument from model
    ann = describe(model, annotations=True)
    args = iter(ann)
    next(args)
    params = {k: ann[k] for k in args}
    if name:
        if len(params) == len(name):
            params = {n: att for (n, att) in zip(name, params.values())}
        elif len(params) > 0:
            raise ValueError("length of name does not match number of model parameters")
        else:
            params = {n: None for n in name}
    return params


_deprecated_content = {
    "BarlowBeestonLite": ("Template", Template),
    "barlow_beeston_lite_chi2_jsc": ("template_chi2_jsc", template_chi2_jsc),
    "barlow_beeston_lite_chi2_hpd": ("template_chi2_da", template_chi2_da),
}


def __getattr__(name: str) -> Any:
    if name in _deprecated_content:
        new_name, obj = _deprecated_content[name]
        warnings.warn(
            f"{name} was renamed to {new_name}, please import {new_name} instead",
            FutureWarning,
            stacklevel=2,
        )
        return obj

    raise AttributeError
