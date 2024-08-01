"""
Data classes and utilities used by :class:`iminuit.Minuit`.

You can look up the interface of data classes that iminuit uses here.
"""

from __future__ import annotations
import inspect
from collections import OrderedDict
from argparse import Namespace
from iminuit import _repr_html, _repr_text, _deprecated
from iminuit.typing import Key, UserBound, Cost, CostGradient
from iminuit.warnings import IMinuitWarning, HesseFailedWarning, PerformanceWarning
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import (
    overload,
    Any,
    List,
    Union,
    Dict,
    Iterable,
    Generator,
    Tuple,
    Optional,
    Callable,
    Collection,
    Sequence,
    TypeVar,
    Annotated,
    get_args,
    get_origin,
)
import abc
from time import monotonic
import warnings
import sys

T = TypeVar("T")

__all__ = (
    "IMinuitWarning",
    "HesseFailedWarning",
    "PerformanceWarning",
    "ValueView",
    "FixedView",
    "LimitView",
    "Matrix",
    "FMin",
    "Param",
    "Params",
    "MError",
    "MErrors",
    "make_func_code",
    "make_with_signature",
    "merge_signatures",
    "describe",
    "gradient",
    "is_positive_definite",
)


class BasicView(abc.ABC):
    """
    Array-like view of parameter state.

    Derived classes need to implement methods _set and _get to access
    specific properties of the parameter state.

    :meta private:
    """

    __slots__ = ("_minuit", "_ndim")

    def __init__(self, minuit: Any, ndim: int = 0):
        # Users should not call this __init__, instances are created by the library
        self._minuit = minuit
        self._ndim = ndim

    def __iter__(self) -> Generator:
        """Get iterator over values."""
        for i in range(len(self)):
            yield self._get(i)

    def __len__(self) -> int:
        """Get number of paramters."""
        return self._minuit.npar  # type: ignore

    @abc.abstractmethod
    def _get(self, idx: int) -> Any:
        return NotImplemented  # pragma: no cover

    @abc.abstractmethod
    def _set(self, idx: int, value: Any) -> None:
        NotImplemented  # pragma: no cover

    def __getitem__(self, key: Key) -> Any:
        """
        Get values of the view.

        Parameters
        ----------
        key: int, str, slice, list of int or str
            If the key is an int or str, return corresponding value.
            If it is a slice, list of int or str, return the corresponding subset.
        """
        index = _key2index(self._minuit._var2pos, key)
        if isinstance(index, list):
            return [self._get(i) for i in index]
        return self._get(index)

    def __setitem__(self, key: Key, value: Any) -> None:
        """Assign new value at key, which can be index, parameter name, or slice."""
        self._minuit._copy_state_if_needed()
        index = _key2index(self._minuit._var2pos, key)
        if isinstance(index, list):
            if _ndim(value) == self._ndim:  # support basic broadcasting
                for i in index:
                    self._set(i, value)
            else:
                if len(value) != len(index):
                    raise ValueError("length of argument does not match slice")
                for i, v in zip(index, value):
                    self._set(i, v)
        else:
            self._set(index, value)

    def __eq__(self, other: object) -> bool:
        """Return true if all values are equal."""
        a, b = np.broadcast_arrays(self, other)  # type:ignore
        return bool(np.all(a == b))

    def __repr__(self) -> str:
        """Get detailed text representation."""
        s = f"<{self.__class__.__name__}"
        for k, v in zip(self._minuit._pos2var, self):
            s += f" {k}={v}"
        s += ">"
        return s

    def to_dict(self) -> Dict[str, float]:
        """Obtain dict representation."""
        return {k: self._get(i) for i, k in enumerate(self._minuit._pos2var)}


def _ndim(obj: Any) -> int:
    nd = 0
    while isinstance(obj, Iterable):
        nd += 1
        for x in obj:
            if x is not None:
                obj = x
                break
        else:
            break
    return nd


class ValueView(BasicView):
    """Array-like view of parameter values."""

    def _get(self, i: int) -> float:
        return self._minuit._last_state[i].value  # type:ignore

    def _set(self, i: int, value: float) -> None:
        self._minuit._last_state.set_value(i, value)


class ErrorView(BasicView):
    """Array-like view of parameter errors."""

    def _get(self, i: int) -> float:
        return self._minuit._last_state[i].error  # type:ignore

    def _set(self, i: int, value: float) -> None:
        if value <= 0:
            warnings.warn(
                "Assigned errors must be positive. "
                "Non-positive values are replaced by a heuristic.",
                IMinuitWarning,
            )
            value = _guess_initial_step(value)
        self._minuit._last_state.set_error(i, value)


class FixedView(BasicView):
    """Array-like view of whether parameters are fixed."""

    def _get(self, i: int) -> bool:
        return self._minuit._last_state[i].is_fixed  # type:ignore

    def _set(self, i: int, fix: bool) -> None:
        if fix:
            self._minuit._last_state.fix(i)
        else:
            self._minuit._last_state.release(i)

    def __invert__(self) -> list:
        """Return list with inverted elements."""
        return [not self._get(i) for i in range(len(self))]


class LimitView(BasicView):
    """Array-like view of parameter limits."""

    def __init__(self, minuit: Any):
        # Users should not call this __init__, instances are created by the library
        super(LimitView, self).__init__(minuit, 1)

    def _get(self, i: int) -> Tuple[float, float]:
        p = self._minuit._last_state[i]
        return (
            p.lower_limit if p.has_lower_limit else -np.inf,
            p.upper_limit if p.has_upper_limit else np.inf,
        )

    def _set(self, i: int, arg: UserBound) -> None:
        state = self._minuit._last_state
        val = state[i].value
        err = state[i].error
        # changing limits is a cheap operation, start from clean state
        state.remove_limits(i)
        low, high = _normalize_limit(arg)
        if low != -np.inf and high != np.inf:  # both must be set
            if low == high:
                state.fix(i)
            else:
                state.set_limits(i, low, high)
        elif low != -np.inf:  # lower limit must be set
            state.set_lower_limit(i, low)
        elif high != np.inf:  # lower limit must be set
            state.set_upper_limit(i, high)
        # bug in Minuit2: must set parameter value and error again after changing limits
        if val < low:
            val = low
        elif val > high:
            val = high
        state.set_value(i, val)
        state.set_error(i, err)


def _normalize_limit(lim: Optional[Iterable]) -> Tuple[float, float]:
    if lim is None:
        return (-np.inf, np.inf)
    a, b = lim
    if a is None:
        a = -np.inf
    if b is None:
        b = np.inf
    if a > b:
        raise ValueError(f"limit {lim!r} is invalid")
    return a, b


class Matrix(np.ndarray):
    """
    Enhanced Numpy ndarray.

    Works like a normal ndarray in computations, but also supports pretty printing in
    ipython and Jupyter notebooks. Elements can be accessed via indices or parameter
    names.
    """

    __slots__ = ("_var2pos",)

    def __new__(cls, parameters: Union[Dict, Tuple]) -> Any:  # noqa D102
        # Users should not call __new__, instances are created by the library
        if isinstance(parameters, dict):
            var2pos = parameters
        elif isinstance(parameters, tuple):
            var2pos = {x: i for i, x in enumerate(parameters)}
        else:
            raise TypeError("parameters must be tuple or dict")
        n = len(parameters)
        obj = super().__new__(cls, (n, n))
        obj._var2pos = var2pos
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        """For internal use."""
        if obj is None:
            self._var2pos: Dict[str, int] = {}
        else:
            self._var2pos = getattr(obj, "_var2pos", {})

    def __getitem__(  # type:ignore
        self,
        key: Union[Key, Tuple[Key, Key], Iterable[Key], NDArray],
    ) -> NDArray:
        """Get matrix element at key."""
        var2pos = self._var2pos

        def trafo(key):
            if isinstance(key, str):
                return var2pos[key]
            if isinstance(key, slice):
                return slice(trafo(key.start), trafo(key.stop), key.step)
            if isinstance(key, tuple):
                return tuple(trafo(k) for k in key)
            return key

        if isinstance(key, slice):
            # slice returns square matrix
            sl = trafo(key)
            return super().__getitem__((sl, sl))
        if isinstance(key, (str, tuple)):
            return super().__getitem__(trafo(key))
        if isinstance(key, Iterable) and not isinstance(key, np.ndarray):
            # iterable returns square matrix
            index2 = [trafo(k) for k in key]  # type:ignore
            t = super().__getitem__(index2).T
            return np.ndarray.__getitem__(t, index2).T
        return super().__getitem__(key)

    def to_dict(self) -> Dict[Tuple[str, str], float]:
        """
        Convert matrix to dict.

        Since the matrix is symmetric, the dict only contains the upper triangular
        matrix.
        """
        names = tuple(self._var2pos)
        d = {}
        for i, pi in enumerate(names):
            for j in range(i, len(names)):
                pj = names[j]
                d[pi, pj] = float(self[i, j])
        return d

    def to_table(self) -> Tuple[List[List[str]], Tuple[str, ...]]:
        """
        Convert matrix to tabular format.

        The output is consumable by the external
        `tabulate <https://pypi.org/project/tabulate>`_ module.

        Examples
        --------
        >>> import tabulate as tab
        >>> from iminuit import Minuit
        >>> m = Minuit(lambda x, y: x ** 2 + y ** 2, x=1, y=2).migrad()
        >>> tab.tabulate(*m.covariance.to_table())
              x    y
        --  ---  ---
        x     1   -0
        y    -0    4
        """
        names = tuple(self._var2pos)  # type:ignore
        nums = _repr_text.matrix_format(self)
        tab = []
        n = len(self)
        for i, name in enumerate(names):
            tab.append([name] + [nums[n * i + j] for j in range(n)])
        return tab, names

    def correlation(self):
        """
        Compute and return correlation matrix.

        If the matrix is already a correlation matrix, this effectively returns a copy
        of the original matrix.
        """
        a = self.copy()
        d = np.diag(a) ** 0.5
        a /= np.outer(d, d) + 1e-100
        return a

    def __repr__(self):
        """Get detailed text representation."""
        return super(Matrix, self).__str__()

    def __str__(self):
        """Get user-friendly text representation."""
        if self.ndim != 2:
            return repr(self)
        return _repr_text.matrix(self)

    def _repr_html_(self):
        return _repr_html.matrix(self)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("<Matrix ...>")
        else:
            p.text(str(self))

    # ndarray uses __reduce__ for pickling instead of __getstate__
    def __reduce__(self):
        """Get representation for pickling and copying."""
        restore, args, state = super().__reduce__()
        return restore, args, (state, self._var2pos)

    def __setstate__(self, state):
        """Restore from pickled state."""
        state, self._var2pos = state
        super().__setstate__(state)


class FMin:
    """
    Function minimum view.

    This object provides detailed metadata about the function minimum. Inspect this to
    check what exactly happened if the fit did not converge. Use the attribute
    :attr:`iminuit.Minuit.fmin` to get the best fit values, their uncertainties, or the
    function value at the minimum. For convenience, you can also get a basic OK from
    :class:`iminuit.Minuit` with the methods :attr:`iminuit.Minuit.valid` and
    :attr:`iminuit.Minuit.accurate`.

    See Also
    --------
    :attr:`iminuit.Minuit.values`
    :attr:`iminuit.Minuit.errors`
    :attr:`iminuit.Minuit.merrors`
    :attr:`iminuit.Minuit.covariance`
    :attr:`iminuit.Minuit.fval`
    :attr:`iminuit.Minuit.valid`
    :attr:`iminuit.Minuit.accurate`
    """

    __slots__ = (
        "_src",
        "_algorithm",
        "_has_parameters_at_limit",
        "_nfcn",
        "_ngrad",
        "_ndof",
        "_edm_goal",
        "_time",
    )

    def __init__(
        self,
        fmin: Any,
        algorithm: str,
        nfcn: int,
        ngrad: int,
        ndof: int,
        edm_goal: float,
        time: float,
    ):
        # Users should not call this __init__, instances are created by the library
        self._src = fmin
        self._algorithm = algorithm
        self._has_parameters_at_limit = False
        for mp in fmin.state:
            if mp.is_fixed or not mp.has_limits:
                continue
            v = mp.value
            e = mp.error
            lb = mp.lower_limit if mp.has_lower_limit else -np.inf
            ub = mp.upper_limit if mp.has_upper_limit else np.inf
            # the 0.5 error threshold is somewhat arbitrary
            self._has_parameters_at_limit |= min(v - lb, ub - v) < 0.5 * e
        self._nfcn = nfcn
        self._ngrad = ngrad
        self._ndof = ndof
        self._edm_goal = edm_goal
        self._time = time

    @property
    def algorithm(self) -> str:
        """Get algorithm that was used to compute the function minimum."""
        return self._algorithm

    @property
    def edm(self) -> float:
        """
        Get Estimated Distance to Minimum.

        Minuit uses this criterion to determine whether the fit converged. It depends
        on the gradient and the Hessian matrix. It measures how well the current
        second order expansion around the function minimum describes the function, by
        taking the difference between the predicted (based on gradient and Hessian)
        function value at the minimum and the actual value.
        """
        return self._src.edm  # type:ignore

    @property
    def edm_goal(self) -> float:
        """
        Get EDM threshold value for stopping the minimization.

        The threshold is allowed to be violated by a factor of 10 in some situations.
        """
        return self._edm_goal

    @property
    def fval(self) -> float:
        """Get cost function value at the minimum."""
        return self._src.fval  # type:ignore

    @property
    def reduced_chi2(self) -> float:
        """
        Get χ²/ndof of the fit.

        This returns NaN if the cost function is unbinned, errordef is not 1,
        or if the cost function does not report the degrees of freedom.
        """
        if np.isfinite(self._ndof) and self._ndof > 0 and self.errordef == 1:
            return self.fval / self._ndof
        return np.nan

    @property
    def has_parameters_at_limit(self) -> bool:
        """
        Return whether any bounded parameter was fitted close to a bound.

        The estimated error for the affected parameters is usually off. May be an
        indication to remove or loosen the limits on the affected parameter.
        """
        return self._has_parameters_at_limit

    @property
    def nfcn(self) -> int:
        """Get number of function calls so far."""
        return self._nfcn

    @property
    def ngrad(self) -> int:
        """Get number of function gradient calls so far."""
        return self._ngrad

    @property
    def is_valid(self) -> bool:
        """
        Return whether Migrad converged successfully.

        For it to return True, the following conditions need to be fulfilled:

          - :attr:`has_reached_call_limit` is False
          - :attr:`is_above_max_edm` is False

        Note: The actual verdict is computed inside the Minuit2 C++ code, so we
        cannot guarantee that is_valid is exactly equivalent to these conditions.
        """
        valid = self._src.is_valid
        if valid and self.edm < 0:
            return False
        return valid

    @property
    def has_valid_parameters(self) -> bool:
        """
        Return whether parameters are valid.

        This is the same as :attr:`is_valid` and only kept for backward compatibility.
        """
        return self.is_valid

    @property
    def has_accurate_covar(self) -> bool:
        """
        Return whether the covariance matrix is accurate.

        While Migrad runs, it computes an approximation to the current Hessian
        matrix. If the strategy is set to 0 or if the fit did not converge, the
        inverse of this approximation is returned instead of the inverse of the
        accurately computed Hessian matrix. This property returns False if the
        approximation has been returned instead of an accurate matrix computed by
        the Hesse method.
        """
        return self._src.has_accurate_covar  # type:ignore

    @property
    def has_posdef_covar(self) -> bool:
        """
        Return whether the Hessian matrix is positive definite.

        This must be the case if the extremum is a minimum, otherwise it is a saddle
        point. If it returns False, the fitted result may be correct, but the reported
        uncertainties are false. This may affect some parameters or all of them.
        Possible causes:

            * Model contains redundanted parameters that are 100% correlated. Fix:
              remove the parameters that are 100% correlated.
            * Cost function is not computed in double precision. Fix: try adjusting
              :attr:`iminuit.Minuit.precision` or change the cost function to compute
              in double precision.
            * Cost function is not analytical near the minimum. Fix: change the cost
              function to something analytical. Functions are not analytical if:

                * It does computations based on (pseudo)random numbers.
                * It contains vertical steps, for example from code like this::

                      if cond:
                          return value1
                      else:
                          return value2
        """
        return self._src.has_posdef_covar  # type:ignore

    @property
    def has_made_posdef_covar(self) -> bool:
        """
        Return whether the matrix was forced to be positive definite.

        While Migrad runs, it computes an approximation to the current Hessian matrix.
        It can happen that this approximation is not positive definite, but that is
        required to compute the next Newton step. Migrad then adds an appropriate
        diagonal matrix to enforce positive definiteness.

        If the fit has converged successfully, this should always return False. If
        Minuit forced the matrix to be positive definite, the parameter uncertainties
        are false, see :attr:`has_posdef_covar` for more details.
        """
        return self._src.has_made_posdef_covar  # type:ignore

    @property
    def hesse_failed(self) -> bool:
        """Return whether the last call to Hesse failed."""
        return self._src.hesse_failed  # type:ignore

    @property
    def has_covariance(self) -> bool:
        """
        Return whether a covariance matrix was computed at all.

        This is false if the Simplex minimization algorithm was used instead of
        Migrad, in which no approximation to the Hessian is computed.
        """
        return self._src.has_covariance  # type:ignore

    @property
    def is_above_max_edm(self) -> bool:
        """
        Return whether the EDM value is below the convergence threshold.

        Returns True, if the fit did not converge; otherwise returns False.
        """
        return self._src.is_above_max_edm  # type:ignore

    @property
    def has_reached_call_limit(self) -> bool:
        """
        Return whether Migrad exceeded the allowed number of function calls.

        Returns True true, the fit was stopped before convergence was reached;
        otherwise returns False.
        """
        return self._src.has_reached_call_limit  # type:ignore

    @property
    def errordef(self) -> float:
        """Equal to the value of :attr:`iminuit.Minuit.errordef` when Migrad ran."""
        return self._src.errordef  # type:ignore

    @property
    def time(self) -> float:
        """Runtime of the last algorithm."""
        return self._time

    def __eq__(self, other: object) -> bool:
        """Return True if all attributes are equal."""

        def relaxed_equal(k: str, a: object, b: object) -> bool:
            a = getattr(a, k)
            b = getattr(b, k)
            if isinstance(a, float) and np.isnan(a):
                return np.isnan(b)  # type:ignore
            return a == b  # type:ignore

        return all(relaxed_equal(k, self, other) for k in self.__slots__)

    def __repr__(self) -> str:
        """Get detailed text representation."""
        s = "<FMin"
        for key in sorted(dir(self)):
            if key.startswith("_"):
                continue
            val = getattr(self, key)
            s += f" {key}={val!r}"
        s += ">"
        return s

    def __str__(self) -> str:
        """Get user-friendly text representation."""
        return _repr_text.fmin(self)  # type:ignore

    def _repr_html_(self) -> str:
        return _repr_html.fmin(self)  # type:ignore

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("<FMin ...>")
        else:
            p.text(str(self))


class Param:
    """Data object for a single Parameter."""

    __slots__ = (
        "number",
        "name",
        "value",
        "error",
        "merror",
        "is_const",
        "is_fixed",
        "lower_limit",
        "upper_limit",
    )

    def __init__(
        self,
        *args: Union[int, str, float, Optional[Tuple[float, float]], bool],
    ):
        # Users should not call this __init__, instances are created by the library
        assert len(args) == len(self.__slots__)
        for k, arg in zip(self.__slots__, args):
            setattr(self, k, arg)

    def __eq__(self, other: object) -> bool:
        """Return True if all values are equal."""
        return all(getattr(self, k) == getattr(other, k) for k in self.__slots__)

    def __repr__(self) -> str:
        """Get detailed text representation."""
        pairs = []
        for k in self.__slots__:
            v = getattr(self, k)
            pairs.append(f"{k}={v!r}")
        return "Param(" + ", ".join(pairs) + ")"

    @property
    def has_limits(self):
        """Query whether the parameter has an lower or upper limit."""
        return self.has_lower_limit or self.has_upper_limit

    @property
    def has_lower_limit(self):
        """Query whether parameter has a lower limit."""
        return self.lower_limit is not None

    @property
    def has_upper_limit(self):
        """Query whether parameter has an upper limit."""
        return self.upper_limit is not None

    def __str__(self) -> str:
        """Get user-friendly text representation."""
        return _repr_text.params([self])  # type:ignore

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("Param(...)")
        else:
            p.text(str(self))


class Params(tuple):
    """Tuple-like holder of parameter data objects."""

    __slots__ = ()

    def _repr_html_(self):
        return _repr_html.params(self)

    def to_table(self):
        """
        Convert parameter data to a tabular format.

        The output is consumable by the external
        `tabulate <https://pypi.org/project/tabulate>`_ module.

        Examples
        --------
        >>> import tabulate as tab
        >>> from iminuit import Minuit
        >>> m = Minuit(lambda x, y: x ** 2 + (y / 2) ** 2 + 1, x=0, y=0)
        >>> m.fixed["x"] = True
        >>> m.migrad().minos()
        >>> tab.tabulate(*m.params.to_table())
          pos  name      value    error  error-    error+    limit-    limit+    fixed
        -----  ------  -------  -------  --------  --------  --------  --------  -------
            0  x             0      0.1                                          yes
            1  y             0      1.4  -1.0      1.0
        """
        header = [
            "pos",
            "name",
            "value",
            "error",
            "error-",
            "error+",
            "limit-",
            "limit+",
            "fixed",
        ]
        tab = []
        for i, mp in enumerate(self):
            name = mp.name
            row = [i, name]
            me = mp.merror
            if me:
                val, err, mel, meu = _repr_text.pdg_format(mp.value, mp.error, *me)
            else:
                val, err = _repr_text.pdg_format(mp.value, mp.error)
                mel = ""
                meu = ""
            row += [
                val,
                err,
                mel,
                meu,
                f"{mp.lower_limit}" if mp.lower_limit is not None else "",
                f"{mp.upper_limit}" if mp.upper_limit is not None else "",
                "yes" if mp.is_fixed else "",
            ]
            tab.append(row)
        return tab, header

    def __getitem__(self, key):
        """Get item at key, which can be an index or a parameter name."""
        if isinstance(key, str):
            for i, p in enumerate(self):
                if p.name == key:
                    break
            key = i
        return super(Params, self).__getitem__(key)

    def __str__(self):
        """Get user-friendly text representation."""
        return _repr_text.params(self)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("Params(...)")
        else:
            p.text(str(self))


class MError:
    """
    Minos data object.

    Attributes
    ----------
    number : int
        Parameter index.
    name : str
        Parameter name.
    lower : float
        Lower error.
    upper : float
        Upper error.
    is_valid : bool
        Whether Minos computation was successful.
    lower_valid : bool
        Whether downward scan was successful.
    upper_valid : bool
        Whether upward scan was successful.
    at_lower_limit : bool
        Whether scan reached lower limit.
    at_upper_limit : bool
        Whether scan reached upper limit.
    at_lower_max_fcn : bool
        Whether allowed number of function evaluations was exhausted.
    at_upper_max_fcn : bool
        Whether allowed number of function evaluations was exhausted.
    lower_new_min : float
        Parameter value for new minimum, if one was found in downward scan.
    upper_new_min : float
        Parameter value for new minimum, if one was found in upward scan.
    nfcn : int
        Number of function calls.
    min : float
        Function value at the new minimum.
    """

    __slots__ = (
        "number",
        "name",
        "lower",
        "upper",
        "is_valid",
        "lower_valid",
        "upper_valid",
        "at_lower_limit",
        "at_upper_limit",
        "at_lower_max_fcn",
        "at_upper_max_fcn",
        "lower_new_min",
        "upper_new_min",
        "nfcn",
        "min",
    )

    def __init__(self, *args: Union[int, str, float, bool]):
        # Users should not call this __init__, instances are created by the library
        assert len(args) == len(self.__slots__)
        for k, arg in zip(self.__slots__, args):
            setattr(self, k, arg)

    def __eq__(self, other: object) -> bool:
        """Return True if all values are equal."""
        return all(getattr(self, k) == getattr(other, k) for k in self.__slots__)

    def __repr__(self) -> str:
        """Get detailed text representation."""
        s = "<MError"
        for idx, k in enumerate(self.__slots__):
            v = getattr(self, k)
            s += f" {k}={v!r}"
        s += ">"
        return s

    def __str__(self) -> str:
        """Get user-friendly text representation."""
        return _repr_text.merrors({None: self})  # type:ignore

    def _repr_html_(self) -> str:
        return _repr_html.merrors({None: self})  # type:ignore

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("<MError ...>")
        else:
            p.text(str(self))


class MErrors(OrderedDict):
    """Dict-like map from parameter name to Minos result object."""

    __slots__ = ()

    def _repr_html_(self):
        return _repr_html.merrors(self)

    def __repr__(self):
        """Get detailed text representation."""
        return "<MErrors\n  " + ",\n  ".join(repr(x) for x in self.values()) + "\n>"

    def __str__(self):
        """Get user-friendly text representation."""
        return _repr_text.merrors(self)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("<MErrors ...>")
        else:
            p.text(str(self))

    def __getitem__(self, key):
        """Get item at key, which can be an index or a parameter name."""
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("index out of range")
            for i, k in enumerate(self):
                if i == key:
                    break
            key = k
        return OrderedDict.__getitem__(self, key)


@_deprecated.deprecated("use jacobi.propagate instead from jacobi library")
def propagate(
    fn: Callable,
    x: Collection[float],
    cov: Collection[Collection[float]],
) -> Tuple[NDArray, NDArray]:
    """
    Numerically propagates the covariance into a new space.

    This function is deprecated and will be removed. Please use jacobi.propagate from
    the jacobi library, which is more accurate. The signatures of the two functions are
    compatible, so it is a drop-in replacement.

    Parameters
    ----------
    fn: callable
        Vectorized function that computes y = fn(x).
    x: array-like with shape (N,)
        Input vector.
    cov: array-like with shape (N, N)
        Covariance matrix of input vector.

    Returns
    -------
    y, ycov
        y is the result of fn(x)
        ycov is the propagated covariance matrix.
    """
    from scipy.optimize import approx_fprime

    vx = np.atleast_1d(x)  # type:ignore
    if np.ndim(cov) != 2:  # type:ignore
        raise ValueError("cov must be 2D array-like")
    vcov = np.atleast_2d(cov)  # type:ignore
    if vcov.shape[0] != vcov.shape[1]:
        raise ValueError("cov must have shape (N, N)")
    tol = 1e-10
    dx = (np.diag(vcov) * tol) ** 0.5
    if not np.all(dx >= 0):
        raise ValueError("diagonal elements of covariance matrix must be non-negative")
    y = fn(vx)
    jac = np.atleast_2d(approx_fprime(vx, fn, dx))
    ycov = np.einsum("ij,kl,jl", jac, jac, vcov)
    return y, np.squeeze(ycov) if np.ndim(y) == 0 else ycov


class _Timer:
    def __init__(self, fmin):
        self.value = fmin.time if fmin else 0.0

    def __enter__(self):
        self.value += monotonic()

    def __exit__(self, *args):
        self.value = monotonic() - self.value


@_deprecated.deprecated(
    "Use of ``func_code`` attribute to declare parameters is deprecated. "
    "Use ``_parameters`` instead, which is a dict of parameter names to limits."
)
def make_func_code(params: Collection[str]) -> Namespace:
    """
    Make a func_code object to fake a function signature.

    Example::

        def f(a, b): ...

        f.func_code = make_func_code(["x", "y"])
    """
    return Namespace(co_varnames=tuple(params), co_argcount=len(params))


def make_with_signature(
    callable: Callable, *varnames: str, **replacements: str
) -> Callable:
    """
    Return new callable with altered signature.

    Parameters
    ----------
    callable:
        Original callable with positional arguments, whose names shall be changed.
    *varnames: sequence of str
        Replace the first N argument names with these.
    **replacements: mapping of str to str, optional
        Replace old argument name (key) with new argument name (value).

    Returns
    -------
    Callable with positional-only arguments.
    """
    pars = describe(callable, annotations=True)
    if pars:
        args = list(pars)
        n = len(varnames)
        if n > len(args):
            raise ValueError("varnames longer than original signature")
        args[:n] = varnames
        for old, new in replacements.items():
            i = args.index(old)
            args[i] = new
        pars = {new: pars[old] for (new, old) in zip(args, pars)}

    class Caller:
        def __init__(self, parameters):
            self._parameters = parameters

        def __call__(self, *args: Any) -> Any:
            return callable(*args)

    return Caller(pars)


def merge_signatures(
    callables: Iterable[Callable], annotations: bool = False
) -> Tuple[Any, List[List[int]]]:
    """
    Merge signatures of callables with positional arguments.

    This is best explained by an example::

        def f(x, y, z): ...

        def g(x, p): ...

        parameters, mapping = merge_signatures(f, g)
        # parameters is ('x', 'y', 'z', 'p')
        # mapping is ([0, 1, 2], [0, 3])

    Parameters
    ----------
    callables :
        Callables whose parameters can be extracted with :func:`describe`.
    annotations : bool, optional
        Whether to return the annotions. Default is False, for backward
        compatibility.

    Returns
    -------
    tuple(parameters, mapping)
        parameters is the tuple of the merged parameter names.
        mapping contains the mapping of parameters indices from the merged signature to
        the original signatures.
    """
    args: List[str] = []
    anns: List[Optional[Tuple[float, float]]] = []
    mapping = []

    for f in callables:
        amap = []
        for i, (k, ann) in enumerate(describe(f, annotations=True).items()):
            if k in args:
                amap.append(args.index(k))
            else:
                amap.append(len(args))
                args.append(k)
                anns.append(ann)
        mapping.append(amap)

    if annotations:
        return {k: a for (k, a) in zip(args, anns)}, mapping
    return args, mapping


@overload
def describe(callable: Callable) -> List[str]: ...  # pragma: no cover


@overload
def describe(
    callable: Callable, annotations: bool
) -> Dict[str, Optional[Tuple[float, float]]]: ...  # pragma: no cover


def describe(callable, *, annotations=False):
    """
    Attempt to extract the function argument names and annotations.

    Parameters
    ----------
    callable : callable
        Callable whose parameter names should be extracted.
    annotations : bool, optional
        Whether to also extract annotations. Default is false.

    Returns
    -------
    list or dict
        If annotate is False, return list of strings with the parameters names if
        successful and an empty list otherwise. If annotations is True, return a dict,
        which maps parameter names to annotations. For parameters without annotations,
        the dict maps to None.

    Notes
    -----
    Parameter names are extracted with the following three methods, which are attempted
    in order. The first to succeed determines the result.

    1.  Using ``obj._parameters``, which is a dict that maps parameter names to
        parameter limits or None if the parameter has to limits. Users are encouraged
        to use this mechanism to provide signatures for objects that otherwise would
        not have a detectable signature. Example::

            def f(*args):  # no signature
                x, y = args
                return (x - 2) ** 2 + (y - 3) ** 2

            f._parameters = {"x": None, "y": (1, 4)}

        Here, the first parameter is declared to have no limits (values from minus to
        plus infinity are allowed), while the second parameter is declared to have
        limits, it cannot be smaller than 1 or larger than 4.

        Note: In the past, the ``func_code`` attribute was used for a similar purpose as
        ``_parameters``. It is still supported for legacy code, but should not be used
        anymore in new code, since it does not support declaring parameter limits.

        If an objects has a ``func_code`` attribute, it is used to detect the
        parameters. Example::

            from iminuit.util import make_func_code

            def f(*args): # no signature
                x, y = args
                return (x - 2) ** 2 + (y - 3) ** 2

            # deprecated, make_func_code will raise a warning
            f.func_code = make_func_code(("x", "y"))

    2.  Using :func:`inspect.signature`. The :mod:`inspect` module provides a general
        function to extract the signature of a Python callable. It works on most
        callables, including Functors like this::

            class MyLeastSquares:
                def __call__(self, a, b):
                    ...

        Limits are supported via annotations, using the Annotated type that was
        introduced in Python-3.9 and can be obtained from the external package
        typing_extensions in Python-3.8. In the following example, the second parameter
        has a lower limit at 0, because it must be positive::

            from typing import Annotated

            def my_cost_function(a: float, b: Annotated[float, 0:]):
                ...

        There are no standard annotations yet at the time of this writing. iminuit
        supports the annotations Gt, Ge, Lt, Le from the external package
        annotated-types, and interprets slice notation as limits.

    3.  Using the docstring. The docstring is textually parsed to detect the parameter
        names. This requires that a docstring is present which follows the Python
        standard formatting for function signatures. Here is a contrived example which
        is parsed correctly: fn(a, b: float, int c, d=1, e: float=2.2, double e=3).

    Ambiguous cases with positional and keyword argument are handled in the following
    way::

        def fcn(a, b, *args, **kwargs): ...
        # describe returns [a, b];
        # *args and **kwargs are ignored

        def fcn(a, b, c=1): ...
        # describe returns [a, b, c];
        # positional arguments with default values are detected
    """
    if _address_of_cfunc(callable) != 0:
        return {} if annotations else []

    args = (
        getattr(callable, "_parameters", {})
        or _describe_impl_func_code(callable)
        or _describe_impl_inspect(callable)
        or _describe_impl_docstring(callable)
    )

    if annotations:
        return args
    # for backward-compatibility
    return list(args)


def _describe_impl_func_code(callable):
    # Check (faked) f.func_code; for backward-compatibility with iminuit-1.x
    if hasattr(callable, "func_code"):
        # cannot warn about deprecation here, since numba.njit also uses .func_code
        fc = callable.func_code
        return {x: None for x in fc.co_varnames[: fc.co_argcount]}
    return {}


def _describe_impl_inspect(callable):
    try:
        signature = inspect.signature(callable)
    except ValueError:  # raised when used on built-in function
        return {}

    r = {}
    for name, par in signature.parameters.items():
        # stop when variable number of arguments is encountered
        if par.kind is inspect.Parameter.VAR_POSITIONAL:
            break
        # stop when keyword argument is encountered
        if par.kind is inspect.Parameter.VAR_KEYWORD:
            break
        r[name] = _get_limit(par.annotation)
    return r


def _describe_impl_docstring(callable):
    doc = inspect.getdoc(callable)

    if doc is None:
        return {}

    # Examples of strings we want to parse:
    #   min(iterable, *[, default=obj, key=func]) -> value
    #   min(arg1, arg2, *args, *[, key=func]) -> value
    #   Foo.bar(self, int ncall_me =10000, [resume=True, int nsplit=1])
    #   Foo.baz(self: Foo, ncall_me: int =10000)

    try:
        # function wrapper functools.partial does not offer __name__,
        # we cannot extract the signature in this case
        name = callable.__name__
    except AttributeError:
        return {}

    token = name + "("
    start = doc.find(token)
    if start < 0:
        return {}
    start += len(token)

    nbrace = 1
    for ich, ch in enumerate(doc[start:]):
        if ch == "(":
            nbrace += 1
        elif ch == ")":
            nbrace -= 1
        if nbrace == 0:
            break
    items = [x.strip(" []") for x in doc[start : start + ich].split(",")]

    # strip self if callable is a class method
    if inspect.ismethod(callable):
        items = items[1:]

    #   "iterable", "*", "default=obj", "key=func"
    #   "arg1", "arg2", "*args", "*", "key=func"
    #   "int ncall_me =10000", "resume=True", "int nsplit=1"
    #   "ncall_me: int =10000"

    try:
        i = items.index("*args")
        items = items[:i]
    except ValueError:
        pass

    #   "iterable", "*", "default=obj", "key=func"
    #   "arg1", "arg2", "*", "key=func"
    #   "int ncall_me =10000", "resume=True", "int nsplit=1"
    #   "ncall_me: int =10000"

    def extract(s: str) -> str:
        i = s.find("=")
        if i >= 0:
            s = s[:i]
        i = s.find(":")
        if i >= 0:
            return s[:i].strip()
        s = s.strip()
        i = s.find(" ")
        if i >= 0:
            return s[i:].strip()
        return s

    #   "iterable", "*", "default", "key"
    #   "arg1", "arg2", "key"
    #   "ncall_me", "resume", "nsplit"
    #   "ncall_me"
    return {extract(x): None for x in items if x != "*"}


def _get_limit(
    annotation: Union[type, Annotated[float, Any], str],
) -> Optional[Tuple[float, float]]:
    from iminuit import typing

    if isinstance(annotation, str):
        # This provides limited support for string annotations, which
        # have a lot of problems, see https://peps.python.org/pep-0649.
        try:
            annotation = eval(annotation, None, typing.__dict__)
        except NameError:
            # We ignore unknown annotations to fix issue #846.
            # I cannot replicate here what inspect.signature(..., eval_str=True) does.
            # I need a dict with the global objects at the call site of describe, but
            # it is not globals(). Anyway, when using strings, only the annotations
            # from the package annotated-types are supported.
            pass

    if annotation == inspect.Parameter.empty:
        return None

    if get_origin(annotation) is not Annotated:
        return None

    tp, *constraints = get_args(annotation)
    assert tp is float
    lower = -np.inf
    upper = np.inf
    for c in constraints:
        if isinstance(c, slice):
            if c.start is not None:
                lower = c.start
            if c.stop is not None:
                upper = c.stop
            continue
        if isinstance(c, Sequence):
            lower, upper = c
            continue

        # Minuit does not distinguish between closed and open intervals.
        # We use a chain of ifs so that the code also works with the
        # `Interval` class, which contains several of those attributes
        # and which can be None.
        gt = getattr(c, "gt", None)
        ge = getattr(c, "ge", None)
        lt = getattr(c, "lt", None)
        le = getattr(c, "le", None)
        if gt is not None:
            lower = gt
        if ge is not None:
            lower = ge
        if lt is not None:
            upper = lt
        if le is not None:
            upper = le
    return lower, upper


def _guess_initial_step(val: float) -> float:
    return 1e-2 * abs(val) if val != 0 else 1e-1  # heuristic


def _key2index_from_slice(var2pos: Dict[str, int], key: slice) -> List[int]:
    start = var2pos[key.start] if isinstance(key.start, str) else key.start
    stop = var2pos[key.stop] if isinstance(key.stop, str) else key.stop
    start, stop, step = slice(start, stop, key.step).indices(len(var2pos))
    return list(range(start, stop, step))


def _key2index_item(var2pos: Dict[str, int], key: Union[str, int]) -> int:
    if isinstance(key, str):
        return var2pos[key]
    i = key
    if i < 0:
        i += len(var2pos)
    if i < 0 or i >= len(var2pos):
        raise IndexError
    return i


def _key2index(
    var2pos: Dict[str, int],
    key: Key,
) -> Union[int, List[int]]:
    if key is ...:
        return list(range(len(var2pos)))
    if isinstance(key, slice):
        return _key2index_from_slice(var2pos, key)
    if not isinstance(key, str) and isinstance(key, Iterable):
        # convert boolean masks into list of indices
        if isinstance(key[0], bool):
            key = [k for k in range(len(var2pos)) if key[k]]
        return [_key2index_item(var2pos, k) for k in key]
    return _key2index_item(var2pos, key)


def _address_of_cfunc(fcn: Any) -> int:
    from ctypes import (
        c_void_p,
        c_double,
        c_uint32,
        POINTER,
        CFUNCTYPE,
        cast as ctypes_cast,
    )

    c_sig = CFUNCTYPE(c_double, c_uint32, POINTER(c_double))

    fcn = getattr(fcn, "ctypes", None)
    if isinstance(fcn, c_sig):
        return ctypes_cast(fcn, c_void_p).value  # type: ignore
    return 0


def _iterate(x):
    if not isinstance(x, Iterable):
        yield x
    else:
        for xi in x:
            yield xi


def _replace_none(x: Optional[T], v: T) -> T:
    if x is None:
        return v
    return x


# poor-mans progressbar
class ProgressBar:
    """
    Simple progress bar.

    Renders as nice HTML progressbar in Jupyter notebooks.
    """

    value: int = 0
    max_value: int

    def _update(self, fraction):
        self._out.write(f"\r{100 * fraction:.0f} %")
        self._out.flush()

    def _finish(self):
        self._out.write("\r     ")
        self._out.flush()

    def __init__(self, max_value):
        """
        Initialize bar.

        Parameters
        ----------
        max_value: int
            Total number of entries.
        """
        self.max_value = max_value
        self._out = sys.stdout

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                # DeprecationWarning: Jupyter is migrating its paths ...
                from ipykernel.iostream import OutStream

            if isinstance(self._out, OutStream):
                from IPython.display import display, HTML

                self._update = lambda v: display(
                    HTML(
                        f"<progress value='{100 * v:.0f}' max='100'></progress> "
                        f"{100 * v:.0f} %"
                    ),
                    clear=True,
                )
                self._finish = lambda: display(HTML(""), clear=True)
        except ModuleNotFoundError:
            pass

        self._update(self.value)

    def __enter__(self, *args):
        """Noop."""
        return self

    def __exit__(self, *args):
        """Clean up the bar."""
        self._finish()

    def __add__(self, v):
        """Increment progress."""
        self.value += v
        self._update(self.value / self.max_value)
        return self


def _histogram_segments(mask, xe, masked):
    assert masked.ndim == 1

    if mask is None:
        return [(masked, xe)]

    segments = []
    a = 0
    b = 0
    am = 0
    n = len(mask)
    while a < n:
        if not mask[a]:
            a += 1
            continue
        b = a + 1
        while b < n and mask[b]:
            b += 1
        segments.append((masked[am : am + b - a], xe[a : b + 1]))
        am += b - a
        a = b + 1
    return segments


def _smart_sampling(f, xmin, xmax, start=5, tol=5e-3, maxiter=20, maxtime=10):
    t0 = monotonic()
    x = np.linspace(xmin, xmax, start)
    ynew = f(x)
    ymin = np.min(ynew)
    ymax = np.max(ynew)
    y = {xi: yi for (xi, yi) in zip(x, ynew)}
    a = x[:-1]
    b = x[1:]
    niter = 0
    while len(a):
        niter += 1
        if niter > maxiter:
            msg = (
                f"Iteration limit {maxiter} in smart sampling reached, "
                f"produced {len(y)} points"
            )
            warnings.warn(msg, RuntimeWarning)
            break
        if monotonic() - t0 > maxtime:
            msg = (
                f"Time limit {maxtime} in smart sampling reached, "
                f"produced {len(y)} points"
            )
            warnings.warn(msg, RuntimeWarning)
            break
        xnew = 0.5 * (a + b)
        ynew = f(xnew)
        ymin = min(ymin, np.min(ynew))
        ymax = max(ymax, np.max(ynew))
        for xi, yi in zip(xnew, ynew):
            y[xi] = yi
        yint = 0.5 * (
            np.fromiter((y[ai] for ai in a), float)
            + np.fromiter((y[bi] for bi in b), float)
        )
        dy = np.abs(ynew - yint)
        dx = np.abs(b - a)

        # in next iteration, handle intervals which do not
        # pass interpolation test and are not too narrow
        mask = (dy > tol * (ymax - ymin)) & (dx > tol * abs(xmax - xmin))
        a = a[mask]
        b = b[mask]
        xnew = xnew[mask]
        a = np.append(a, xnew)
        b = np.append(xnew, b)

    xy = list(y.items())
    xy.sort()
    return np.transpose(xy)


def _detect_log_spacing(x: NDArray) -> bool:
    # x should never contain NaN
    x = np.sort(x)
    if x[0] <= 0:
        return False
    d_lin = np.diff(x)
    d_log = np.diff(np.log(x))
    lin_rel_std = np.std(d_lin) / np.mean(d_lin)
    log_rel_std = np.std(d_log) / np.mean(d_log)
    return log_rel_std < lin_rel_std


def gradient(fcn: Cost) -> Optional[CostGradient]:
    """
    Return a callable which computes the gradient of fcn or None.

    Parameters
    ----------
    fcn: Cost
        Cost function which may provide a callable gradient. How the gradient
        is detected is specified below in the Notes.

    Notes
    -----
    This function checks whether the following attributes exist: `fcn.grad` and
    `fcn.has_grad`. If `fcn.grad` exists and is a CostGradient, it is returned unless
    `fcn.has_grad` exists and is False. If no useable gradient is detected, None is
    returned.

    Returns
    -------
    callable or None
        The gradient function or None
    """
    grad = getattr(fcn, "grad", None)
    has_grad = getattr(fcn, "has_grad", True)
    if grad and isinstance(grad, CostGradient) and has_grad:
        return grad
    return None


def is_positive_definite(m: ArrayLike) -> bool:
    """
    Return True if argument is a positive definite matrix.

    This test is somewhat expensive, because we attempt a cholesky decomposition.

    Parameters
    ----------
    m : array-like
        Matrix to be tested.

    Returns
    -------
    bool
        True if matrix is positive definite and False otherwise.
    """
    m = np.atleast_2d(m)
    if np.all(m.T == m):
        # maybe check this first https://en.wikipedia.org/wiki/Diagonally_dominant_matrix
        # and only try cholesky if that fails
        try:
            np.linalg.cholesky(m)
        except np.linalg.LinAlgError:
            return False
        return True
    return False
