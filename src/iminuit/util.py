"""iminuit utility functions and classes."""
import re
from collections import OrderedDict, namedtuple
from collections.abc import Iterable
from . import _repr_html
from . import _repr_text

inf = float("infinity")


class IMinuitWarning(RuntimeWarning):
    """iminuit warning."""


class InitialParamWarning(IMinuitWarning):
    """Initial parameter warning."""


class HesseFailedWarning(IMinuitWarning):
    """HESSE failed warning."""


class BasicView:
    """Array-like view of parameter state.

    Derived classes need to implement methods _set and _get to access
    specific properties of the parameter state."""

    _minuit = None
    _ndim = 0

    def __init__(self, minuit, ndim=0):
        self._minuit = minuit
        self._ndim = ndim

    def __iter__(self):
        for i in range(len(self)):
            yield self._get(i)

    def __len__(self):
        return self._minuit.narg

    def __getitem__(self, key):
        if isinstance(key, slice):
            ind = range(*key.indices(len(self)))
            return [self._get(i) for i in ind]
        i = key if isinstance(key, int) else self._minuit._var2pos[key]
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError
        return self._get(i)

    def __setitem__(self, key, value):
        self._minuit._copy_state_if_needed()
        if isinstance(key, slice):
            ind = range(*key.indices(len(self)))
            if _ndim(value) == self._ndim:  # basic broadcasting
                for i in ind:
                    self._set(i, value)
            else:
                if len(value) != len(ind):
                    raise ValueError("length of argument does not match slice")
                for i, v in zip(ind, value):
                    self._set(i, v)
        else:
            i = key if isinstance(key, int) else self._minuit._var2pos[key]
            if i < 0:
                i += len(self)
            if i < 0 or i >= len(self):
                raise IndexError
            self._set(i, value)

    def __eq__(self, other):
        return len(self) == len(other) and all(x == y for x, y in zip(self, other))

    def __repr__(self):
        s = "<{} of Minuit at {:x}>".format(self.__class__.__name__, id(self._minuit))
        for (k, v) in zip(self._minuit._pos2var, self):
            s += f"\n  {k}: {v}"
        return s


def _ndim(obj):
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

    def _get(self, i):
        return self._minuit._last_state[i].value

    def _set(self, i, value):
        self._minuit._last_state.set_value(i, value)


class ErrorView(BasicView):
    """Array-like view of parameter errors."""

    def _get(self, i):
        return self._minuit._last_state[i].error

    def _set(self, i, value):
        self._minuit._last_state.set_error(i, value)


class FixedView(BasicView):
    """Array-like view of whether parameters are fixed."""

    def _get(self, i):
        return self._minuit._last_state[i].is_fixed

    def _set(self, i, fix):
        if fix:
            self._minuit._last_state.fix(i)
        else:
            self._minuit._last_state.release(i)


class LimitView(BasicView):
    """Array-like view of parameter limits."""

    def _get(self, i):
        p = self._minuit._last_state[i]
        return (
            p.lower_limit if p.has_lower_limit else -inf,
            p.upper_limit if p.has_upper_limit else inf,
        )

    def _set(self, i, args):
        state = self._minuit._last_state
        val = state[i].value
        err = state[i].error
        # changing limits is a cheap operation, start from clean state
        state.remove_limits(i)
        low, high = _normalize_limit(args)
        if low != -inf and high != inf:  # both must be set
            state.set_limits(i, low, high)
            if low == high:
                state.fix(i)
        elif low != -inf:  # lower limit must be set
            state.set_lower_limit(i, low)
        else:  # lower limit must be set
            state.set_upper_limit(i, high)
        # bug in Minuit2: must set parameter value and error again after changing limits
        if val < low:
            val = low
        elif val > high:
            val = high
        state.set_value(i, val)
        state.set_error(i, err)


def _normalize_limit(lim):
    if lim is None:
        return (-inf, inf)
    lim = list(lim)
    if lim[0] is None:
        lim[0] = -inf
    if lim[1] is None:
        lim[1] = inf
    if lim[0] > lim[1]:
        raise ValueError("limit " + str(lim) + " is invalid")
    return tuple(lim)


class Matrix(tuple):
    """Matrix data object (tuple of tuples)."""

    __slots__ = ()

    def __new__(self, names, data):
        """Create new matrix."""
        self.names = names
        return tuple.__new__(Matrix, (tuple(x) for x in data))

    def __str__(self):
        """Return string suitable for terminal."""
        return _repr_text.matrix(self)

    def to_table(self):
        args = []
        for mi in self:
            for mj in mi:
                args.append(mj)
        nums = _repr_text.matrix_format(*args)
        tab = []
        n = len(self)
        for i, name in enumerate(self.names):
            tab.append([name] + [nums[n * i + j] for j in range(n)])
        return tab, self.names

    def _repr_html_(self):
        return _repr_html.matrix(self)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("Matrix(...)")
        else:
            p.text(str(self))


class Param:
    """Data object for a single Parameter."""

    __slots__ = (
        "number",
        "name",
        "value",
        "error",
        "is_const",
        "is_fixed",
        "has_limits",
        "has_lower_limit",
        "has_upper_limit",
        "lower_limit",
        "upper_limit",
    )

    def __init__(
        self,
        number,
        name,
        value,
        error,
        is_const,
        is_fixed,
        has_limits,
        has_lower_limit,
        has_upper_limit,
        lower_limit,
        upper_limit,
    ):
        self.number = number
        self.name = name
        self.value = value
        self.error = error
        self.is_const = is_const
        self.is_fixed = is_fixed
        self.has_limits = has_limits
        self.has_lower_limit = has_lower_limit
        self.has_upper_limit = has_upper_limit
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def __str__(self):
        pairs = []
        for k in self.__slots__:
            v = getattr(self, k)
            pairs.append(f"{k}={v!r}")
        return "Param(" + ", ".join(pairs) + ")"

    def __eq__(self, other):
        return (
            self.number == other.number
            and self.name == other.name
            and self.value == other.value
            and self.error == other.error
            and self.is_const == other.is_const
            and self.is_fixed == other.is_fixed
            and self.has_limits == other.has_limits
            and self.has_lower_limit == other.has_lower_limit
            and self.has_upper_limit == other.has_upper_limit
            and self.lower_limit == other.lower_limit
            and self.upper_limit == other.upper_limit
        )


class FMin:
    """Function minimum view."""

    __slots__ = (
        "_src",
        "_has_parameters_at_limit",
        "_nfcn",
        "_ngrad",
        "_tolerance",
    )

    def __init__(self, fmin, nfcn, ngrad, tol):
        self._src = fmin
        self._has_parameters_at_limit = False
        for mp in fmin.state:
            if mp.is_fixed or not mp.has_limits:
                continue
            v = mp.value
            e = mp.error
            lb = mp.lower_limit if mp.has_lower_limit else -inf
            ub = mp.upper_limit if mp.has_upper_limit else inf
            # the 0.5 error threshold is somewhat arbitrary
            self._has_parameters_at_limit |= min(v - lb, ub - v) < 0.5 * e
        self._nfcn = nfcn
        self._ngrad = ngrad
        self._tolerance = tol

    def __str__(self):
        """Return string suitable for terminal."""
        return _repr_text.fmin(self)

    def _repr_html_(self):
        return _repr_html.fmin(self)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("FMin(...)")
        else:
            p.text(str(self))

    @property
    def edm(self):
        """Estimated Distance to Minimum.

        Minuit uses this criterion to determine whether the fit converged. It depends
        on the gradient and the Hessian matrix. It measures how well the current
        second order expansion around the function minimum describes the function, by
        taking the difference between the predicted (based on gradient and Hessian)
        function value at the minimum and the actual value.
        """
        return self._src.edm

    @property
    def fval(self):
        """Value of the cost function at the minimum."""
        return self._src.fval

    @property
    def has_parameters_at_limit(self):
        """Whether any bounded parameter was fitted close to a bound.

        The estimated error for the affected parameters is usually off. May be an
        indication to remove or loosen the limits on the affected parameter.
        """
        return self._has_parameters_at_limit

    @property
    def nfcn(self):
        """Number of function calls so far."""
        return self._nfcn

    @property
    def ngrad(self):
        """Number of function gradient calls so far."""
        return self._ngrad

    @property
    def tolerance(self):
        """Equal to the tolerance value when Migrad ran."""
        return self._tolerance

    @property
    def is_valid(self):
        """Whether Migrad converged successfully.

        For it to return True, the following conditions need to be fulfilled:
        - has_valid_parameters is True
        - has_reached_call_limit is False
        - is_above_max_edm is False

        Note: The actual verdict is computed inside the Minuit2 C++ code, so we
        cannot guarantee that is_valid is exactly equivalent to these conditions.
        """
        return self._src.is_valid

    @property
    def has_valid_parameters(self):
        """Whether parameters are valid.

        For it to return True, the following conditions need to be fulfilled:
        - has_reached_call_limit is False
        - is_above_max_edm is False

        Note: The actual verdict is computed inside the Minuit2 C++ code, so we
        cannot guarantee that is_valid is exactly equivalent to these conditions.
        """
        return self._src.has_valid_parameters

    @property
    def has_accurate_covar(self):
        """Whether the covariance matrix is accurate.

        While Migrad runs, it computes an approximation to the current Hessian
        matrix. If the strategy is set to 0 or if the fit did not converge, the
        inverse of this approximation is returned instead of the inverse of the
        accurately computed Hessian matrix. This property returns False if the
        approximation has been returned instead of an accurate matrix.
        """
        return self._src.has_accurate_covar

    @property
    def has_posdef_covar(self):
        """Whether the Hessian matrix is positive definite.

        This must be the case if the extremum is a minimum. Otherwise it is a
        maximum or a saddle point.

        If the fit has converged, this should always be true. It may be false if the
        fit did not converge or was stopped prematurely.
        """
        return self._src.has_posdef_covar

    @property
    def has_made_posdef_covar(self):
        """Whether the matrix was forced to be positive definite.

        While Migrad runs, it computes an approximation to the current Hessian matrix.
        It can happen that this approximation is not positive definite, but that is
        required to compute the next Newton step. Migrad then adds an appropriate
        diagonal matrix to enforce positive definiteness.

        If the fit has converged, this should always be false. It may be true if the
        fit did not converge or was stopped prematurely.
        """
        return self._src.has_made_posdef_covar

    @property
    def hesse_failed(self):
        """Whether the last call to Hesse failed."""
        return self._src.hesse_failed

    @property
    def has_covariance(self):
        """Whether a covariance matrix was computed at all.

        This is false if the Simplex minimization algorithm was used instead of
        Migrad, in which no approximation to the Hessian is computed.
        """
        return self._src.has_covariance

    @property
    def is_above_max_edm(self):
        """Whether the EDM value is below the convergence threshold.

        If this is true, the fit did not converge; otherwise this is false.
        """
        return self._src.is_above_max_edm

    @property
    def has_reached_call_limit(self):
        """Whether Migrad exceeded the allowed number of function calls.

        If this is true, the fit was stopped before convergence was reached.
        """
        return self._src.has_reached_call_limit

    @property
    def up(self):
        """Equal to the value of ``Minuit.errordef`` when Migrad ran."""
        return self._src.up


class Params(list):
    """List-like holder of parameter data objects."""

    def __init__(self, seq, merrors):
        """Make Params from sequence of Param objects and MErrors object."""
        list.__init__(self, seq)
        self.merrors = merrors

    def _repr_html_(self):
        return _repr_html.params(self)

    def to_table(self):
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
        mes = self.merrors
        tab = []
        for i, mp in enumerate(self):
            name = mp.name
            row = [i, name]
            if mes and name in mes:
                me = mes[name]
                val, err, mel, meu = _repr_text.pdg_format(
                    mp.value, mp.error, me.lower, me.upper
                )
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

    def __str__(self):
        """Return string suitable for terminal."""
        return _repr_text.params(self)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("[...]")
        else:
            p.text(str(self))


class MError:
    """Minos result object."""

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

    def __init__(self, name, minos_error):
        self.number = minos_error.number
        self.name = name
        self.lower = minos_error.lower
        self.upper = minos_error.upper
        self.is_valid = minos_error.is_valid
        self.lower_valid = minos_error.lower_valid
        self.upper_valid = minos_error.upper_valid
        self.at_lower_limit = minos_error.at_lower_limit
        self.at_upper_limit = minos_error.at_upper_limit
        self.at_lower_max_fcn = minos_error.at_lower_max_fcn
        self.at_upper_max_fcn = minos_error.at_upper_max_fcn
        self.lower_new_min = minos_error.lower_new_min
        self.upper_new_min = minos_error.upper_new_min
        self.nfcn = minos_error.nfcn
        self.min = minos_error.min

    def _repr_html_(self):
        return _repr_html.merrors([self])

    def __str__(self):
        """Return string suitable for terminal."""
        return _repr_text.merrors([self])

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("MError(...)")
        else:
            p.text(str(self))


class MErrors(OrderedDict):
    """Dict-like map from parameter name to Minos result object."""

    __slots__ = ()

    def _repr_html_(self):
        return _repr_html.merrors(self.values())

    def __str__(self):
        """Return string suitable for terminal."""
        return _repr_text.merrors(self.values())

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("MErrors(...)")
        else:
            p.text(str(self))

    def __getitem__(self, key):
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


def make_func_code(params):
    """Make a func_code object to fake function signature.

    You can make a funccode from describable object by::

        make_func_code(["x", "y"])
    """
    return namedtuple("FuncCode", "co_varnames co_argcount")(params, len(params))


def describe(callable):
    """Try to extract the function argument names."""
    return (
        _arguments_from_func_code(callable)
        or _arguments_from_inspect(callable)
        or _arguments_from_docstring(callable.__call__.__doc__)
        or _arguments_from_docstring(callable.__doc__)
    )


def _arguments_from_func_code(obj):
    # Check (faked) f.func_code; for backward-compatibility with iminuit-1.x
    if hasattr(obj, "func_code"):
        fc = obj.func_code
        nargs = fc.co_argcount
        return fc.co_varnames[:nargs]


def _arguments_from_inspect(f):
    # Check inspect.signature for arguemnts
    import inspect

    signature = inspect.signature(f)
    args = []
    for name, par in signature.parameters.items():
        # Variable number of arguments is not supported
        if par.kind is inspect.Parameter.VAR_POSITIONAL:
            return None
        if par.kind is inspect.Parameter.VAR_KEYWORD:
            break
        args.append(name)
    return args


def _arguments_from_docstring(doc):
    """Parse first line of docstring for argument name.

    Docstring should be of the form ``min(iterable[, key=func])``.

    It can also parse cython docstring of the form
    ``Minuit.migrad(self[, int ncall_me =10000, foo=True, int bar=1])``
    """
    if doc is None:
        return None

    doc = doc.lstrip()

    # care only the firstline
    # docstring can be long
    line = doc.split("\n", 1)[0]  # get the firstline
    if line.startswith("('...',)"):
        line = doc.split("\n", 2)[1]  # get the second line
    p = re.compile(r"^[\w|\s.]+\(([^)]*)\).*")
    # 'min(iterable[, key=func])\n' -> 'iterable[, key=func]'
    sig = p.search(line)
    if sig is None:
        return None
    # iterable[, key=func]' -> ['iterable[' ,' key=func]']
    sig = sig.groups()[0].split(",")
    ret = []
    for s in sig:
        # get the last one after all space after =
        # ex: int x= True
        tmp = s.split("=")[0].split()[-1]
        # clean up non _+alphanum character
        tmp = "".join([x for x in tmp if x.isalnum() or x == "_"])
        ret.append(tmp)
        # re.compile(r'[\s|\[]*(\w+)(?:\s*=\s*.*)')
        # ret += self.docstring_kwd_re.findall(s)
    ret = list(filter(lambda x: x != "", ret))

    if ret[0] == "self":
        ret = ret[1:]

    return ret


def _guess_initial_step(val):
    return 1e-2 * val if val != 0 else 1e-1  # heuristic
