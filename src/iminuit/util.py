"""iminuit utility functions and classes."""
import re
from collections import OrderedDict, namedtuple
from . import repr_html
from . import repr_text

inf = float("infinity")


class IMinuitWarning(RuntimeWarning):
    """iminuit warning."""


class InitialParamWarning(IMinuitWarning):
    """Initial parameter warning."""


class HesseFailedWarning(IMinuitWarning):
    """HESSE failed warning."""


class Matrix(tuple):
    """Matrix data object (tuple of tuples)."""

    __slots__ = ()

    def __new__(self, names, data):
        """Create new matrix."""
        self.names = names
        return tuple.__new__(Matrix, (tuple(x) for x in data))

    def __str__(self):
        """Return string suitable for terminal."""
        return repr_text.matrix(self)

    def _repr_html_(self):
        return repr_html.matrix(self)

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
    """Function minimum status object."""

    __slots__ = (
        "_src",
        "_fcn",
        "has_parameters_at_limit",
        "nfcn",
        "ngrad",
        "tolerance",
    )

    def __init__(self, fmin, fcn, nfcn, ngrad, tol):
        self._src = fmin
        self._fcn = fcn
        self.has_parameters_at_limit = False
        for mp in fmin.state:
            if mp.is_fixed or not mp.has_limits:
                continue
            v = mp.value
            e = mp.error
            lb = mp.lower_limit if mp.has_lower_limit else -inf
            ub = mp.upper_limit if mp.has_upper_limit else inf
            # the 0.5 error threshold is somewhat arbitrary
            self.has_parameters_at_limit |= min(v - lb, ub - v) < 0.5 * e
        self.nfcn = nfcn
        self.ngrad = ngrad
        self.tolerance = tol

    def __str__(self):
        """Return string suitable for terminal."""
        return repr_text.fmin(self)

    def _repr_html_(self):
        return repr_html.fmin(self)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("FMin(...)")
        else:
            p.text(str(self))

    @property
    def nfcn_total(self):
        return self._fcn.nfcn

    @property
    def ngrad_total(self):
        return self._fcn.ngrad

    @property
    def edm(self):
        return self._src.edm

    @property
    def fval(self):
        return self._src.fval

    @property
    def is_valid(self):
        return self._src.is_valid

    @property
    def has_valid_parameters(self):
        return self._src.has_valid_parameters

    @property
    def has_accurate_covar(self):
        return self._src.has_accurate_covar

    @property
    def has_posdef_covar(self):
        return self._src.has_posdef_covar

    @property
    def has_made_posdef_covar(self):
        return self._src.has_made_posdef_covar

    @property
    def hesse_failed(self):
        return self._src.hesse_failed

    @property
    def has_covariance(self):
        return self._src.has_covariance

    @property
    def is_above_max_edm(self):
        return self._src.is_above_max_edm

    @property
    def has_reached_call_limit(self):
        return self._src.has_reached_call_limit

    @property
    def up(self):
        return self._src.up


class Params(list):
    """List of parameter data objects."""

    def __init__(self, seq, merrors):
        """Make Params from sequence of Param objects and MErrors object."""
        list.__init__(self, seq)
        self.merrors = merrors

    def _repr_html_(self):
        return repr_html.params(self)

    def __str__(self):
        """Return string suitable for terminal."""
        return repr_text.params(self)

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
        return repr_html.merrors([self])

    def __str__(self):
        """Return string suitable for terminal."""
        return repr_text.merrors([self])

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("MError(...)")
        else:
            p.text(str(self))


class MErrors(OrderedDict):
    """Dict from parameter name to Minos result object."""

    __slots__ = ()

    def _repr_html_(self):
        return repr_html.merrors(self.values())

    def __str__(self):
        """Return string suitable for terminal."""
        return repr_text.merrors(self.values())

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


# MigradResult used to be a tuple, so we don't add the dict interface
class MigradResult(namedtuple("_MigradResult", "fmin params")):
    """Holds the Migrad result."""

    __slots__ = ()

    def __str__(self):
        """Return string suitable for terminal."""
        return str(self.fmin) + "\n" + str(self.params)

    def _repr_html_(self):
        return self.fmin._repr_html_() + self.params._repr_html_()

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("MigradResult(...)")
        else:
            p.text(str(self))


def arguments_from_docstring(doc):
    """Parse first line of docstring for argument name.

    Docstring should be of the form ``min(iterable[, key=func])``.

    It can also parse cython docstring of the form
    ``Minuit.migrad(self[, int ncall_me =10000, resume=True, int nsplit=1])``
    """
    if doc is None:
        return False, []

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
        return False, []
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
    return bool(ret), ret


def _is_bound(f):
    return getattr(f, "__self__", None) is not None


def make_func_code(params):
    """Make a func_code object to fake function signature.

    You can make a funccode from describable object by::

        make_func_code(["x", "y"])
    """
    return namedtuple("FuncCode", "co_varnames co_argcount")(params, len(params))


def _fc_or_c(f):
    if hasattr(f, "func_code"):
        return f.func_code
    if hasattr(f, "__code__"):
        return f.__code__
    return make_func_code([])


def arguments_from_funccode(f):
    """Check f.funccode for arguments."""
    fc = _fc_or_c(f)
    nargs = fc.co_argcount
    # bound method and fake function will be None
    if nargs == 0:
        # Function has variable number of arguments
        return False, []
    args = fc.co_varnames[:nargs]
    if _is_bound(f):
        args = args[1:]
    return bool(args), list(args)


def arguments_from_call_funccode(f):
    """Check f.__call__.func_code for arguments."""
    fc = _fc_or_c(f.__call__)
    nargs = fc.co_argcount
    args = list(fc.co_varnames[1:nargs])
    return bool(args), args


def arguments_from_inspect(f):
    """Check inspect.signature for arguemnts"""
    import inspect

    signature = inspect.signature(f)
    ok = True
    for name, par in signature.parameters.items():
        # Variable number of arguments is not supported
        if par.kind is inspect.Parameter.VAR_POSITIONAL:
            ok = False
        if par.kind is inspect.Parameter.VAR_KEYWORD:
            ok = False
    return ok, list(signature.parameters)


def describe(f, verbose=False):
    """Try to extract the function argument names."""
    # using funccode
    ok, args = arguments_from_funccode(f)
    if ok:
        return args
    if verbose:
        print("Failed to extract arguments from f.func_code/__code__")

    # using __call__ funccode
    ok, args = arguments_from_call_funccode(f)
    if ok:
        return args
    if verbose:
        print("Failed to extract arguments from f.__call__.func_code/__code__")

    # now we are parsing __call__.__doc__
    # we assume that __call__.__doc__ doesn't have self
    # this is what cython gives
    ok, args = arguments_from_docstring(f.__call__.__doc__)
    if ok:
        if args[0] == "self":
            args = args[1:]
        return args
    if verbose:
        print("Failed to parse __call__.__doc__")

    # how about just __doc__
    ok, args = arguments_from_docstring(f.__doc__)
    if ok:
        if args[0] == "self":
            args = args[1:]
        return args
    if verbose:
        print("Failed to parse __doc__")

    ok, args = arguments_from_inspect(f)
    if ok:
        return args
    if verbose:
        print(
            "Failed to parse inspect.signature(f). Perhaps you are using"
            " a variable number of arguments. This is not supported."
        )

    raise TypeError("Unable to obtain function signature")


def fitarg_rename(fitarg, ren):
    """Rename variable names in ``fitarg`` with rename function.

    ::

        #simple renaming
        fitarg_rename({'x':1, 'limit_x':1, 'fix_x':1, 'error_x':1},
            lambda pname: 'y' if pname=='x' else pname)
        #{'y':1, 'limit_y':1, 'fix_y':1, 'error_y':1},

        #prefixing
        figarg_rename({'x':1, 'limit_x':1, 'fix_x':1, 'error_x':1},
            lambda pname: 'prefix_'+pname)
        #{'prefix_x':1, 'limit_prefix_x':1, 'fix_prefix_x':1, 'error_prefix_x':1}

    """
    if isinstance(ren, str):
        s = ren
        ren = lambda x: s + "_" + x  # noqa: E731

    ret = {}
    prefix = ["limit_", "fix_", "error_"]
    for k, v in fitarg.items():
        vn = k
        pf = ""
        for p in prefix:
            if k.startswith(p):
                i = len(p)
                vn = k[i:]
                pf = p
        newvn = pf + ren(vn)
        ret[newvn] = v
    return ret


def _normalize_limit(lim):
    if lim is None:
        return None
    lim = list(lim)
    if lim[0] is None:
        lim[0] = -inf
    if lim[1] is None:
        lim[1] = inf
    if lim[0] > lim[1]:
        raise ValueError("limit " + str(lim) + " is invalid")
    return tuple(lim)


def _guess_initial_value(lim):
    if lim is None:
        return 0.0
    if lim[1] == inf:
        return lim[0] + 1.0
    if lim[0] == -inf:
        return lim[1] - 1.0
    return 0.5 * (lim[0] + lim[1])


def _guess_initial_step(val):
    step = 1e-2 * val if val != 0 else 1e-1  # heuristic
    return step
