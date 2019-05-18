"""iminuit utility functions and classes.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import re
import types
from collections import OrderedDict, namedtuple
from . import repr_html
from . import repr_text
from operator import itemgetter


class Matrix(tuple):
    """Matrix data object (tuple of tuples)."""
    __slots__ = ()

    def __new__(self, names, data):
        self.names = names
        return tuple.__new__(Matrix, (tuple(x) for x in data))

    def _repr_html_(self):
        return repr_html.matrix(self)

    def __str__(self):
        return repr_text.matrix(self)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("Matrix(...)")
        else:
            p.text(str(self))


class dict_interface_mixin(object):
    "Provides a dict-like interface for a namedtuple."

    __slots__ = ()

    def __getitem__(self, key):
        base = super(dict_interface_mixin, self)
        if isinstance(key, int):
            return base.__getitem__(key)
        else:
            return base.__getattribute__(key)

    def __contains__(self, key):
        return key in self.keys()

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        return self._fields

    def values(self):
        base = super(dict_interface_mixin, self)
        return tuple(base.__getitem__(i) for i in range(len(self)))

    def items(self):
        keys = self.keys()
        values = self.values()
        return tuple((keys[i], values[i]) for i in range(len(self)))

    def __str__(self):
        return self.__class__.__name__ + "(" + ", ".join("{}={}".format(k, repr(v)) for (k, v) in self.items()) + ")"


class Param(dict_interface_mixin, namedtuple("Param",
    "number name value error is_const is_fixed has_limits "
    "has_lower_limit has_upper_limit lower_limit upper_limit")):
    """Data object for a single Parameter."""

    __slots__ = ()


class Params(list):
    """List of parameter data objects."""

    def __init__(self, seq, merrors):
        list.__init__(self, seq)
        self.merrors = merrors

    def _repr_html_(self):
        return repr_html.params(self)

    def __str__(self):
        return repr_text.params(self)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("[...]")
        else:
            p.text(str(self))


class MError(dict_interface_mixin, namedtuple("MError",
    "name is_valid lower upper lower_valid upper_valid at_lower_limit at_upper_limit "
    "at_lower_max_fcn at_upper_max_fcn lower_new_min upper_new_min nfcn min")):
    """Minos result object."""

    __slots__ = ()

    def _repr_html_(self):
        return repr_html.merror(self)

    def __str__(self):
        return repr_text.merror(self)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("MError(...)")
        else:
            p.text(str(self))


class MErrors(OrderedDict):
    """Dict from parameter name to Minos result object."""

    def _repr_html_(self):
        return "\n".join([x._repr_html_() for x in self.values()])

    def __str__(self):
        return "\n".join([str(x) for x in self.values()])

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("MErrors(...)")
        else:
            p.text(str(self))


class FMin(dict_interface_mixin, namedtuple("FMin",
    "fval edm tolerance nfcn ncalls up is_valid has_valid_parameters has_accurate_covar "
    "has_posdef_covar has_made_posdef_covar hesse_failed has_covariance is_above_max_edm "
    "has_reached_call_limit")):
    """Function minimum status object."""

    __slots__ = ()

    def _repr_html_(self):
        return repr_html.fmin(self)

    def __str__(self):
        return repr_text.fmin(self)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("FMin(...)")
        else:
            p.text(str(self))


# MigradResult used to be a tuple, so we don't add the dict interface
class MigradResult(namedtuple("MigradResult", "fmin params")):
    """Holds the Migrad result."""

    __slots__ = ()

    def __str__(self):
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
        raise RuntimeError('__doc__ is None')

    doc = doc.lstrip()

    # care only the firstline
    # docstring can be long
    line = doc.split('\n', 1)[0]  # get the firstline
    if line.startswith("('...',)"):
        line = doc.split('\n', 2)[1]  # get the second line
    p = re.compile(r'^[\w|\s.]+\(([^)]*)\).*')
    # 'min(iterable[, key=func])\n' -> 'iterable[, key=func]'
    sig = p.search(line)
    if sig is None:
        return []
    # iterable[, key=func]' -> ['iterable[' ,' key=func]']
    sig = sig.groups()[0].split(',')
    ret = []
    for s in sig:
        # get the last one after all space after =
        # ex: int x= True
        tmp = s.split('=')[0].split()[-1]
        # clean up non _+alphanum character
        tmp = ''.join([x for x in tmp if x.isalnum() or x == '_'])
        ret.append(tmp)
        # re.compile(r'[\s|\[]*(\w+)(?:\s*=\s*.*)')
        # ret += self.docstring_kwd_re.findall(s)
    ret = list(filter(lambda x: x != '', ret))

    if len(ret) == 0:
        raise RuntimeError('Your doc is unparsable\n' + doc)

    return ret


def fc_or_c(f):
    if hasattr(f, 'func_code'):
        return f.func_code
    else:
        return f.__code__


def arguments_from_funccode(f):
    """Check f.funccode for arguments
    """
    fc = fc_or_c(f)
    vnames = fc.co_varnames
    nargs = fc.co_argcount
    # bound method and fake function will be None
    args = vnames[1 if is_bound(f) else 0:nargs]
    if not args:
        raise RuntimeError('Function has variable number of arguments')
    return list(args)


def arguments_from_call_funccode(f):
    """Check f.__call__.func_code for arguments
    """
    fc = fc_or_c(f.__call__)
    argcount = fc.co_argcount
    args = list(fc.co_varnames[1:argcount])
    if not args:
        raise RuntimeError('Function has variable number of arguments')
    return args


def is_bound(f):
    """Test whether ``f`` is a bound function.
    """
    return getattr(f, '__self__', None) is not None


def dock_if_bound(f, v):
    """Dock off ``self`` if a bound function is passed.
    """
    return v[1:] if is_bound(f) else v


def better_arg_spec(f, verbose=False):
    """Extract function signature.

    ..seealso::

        :ref:`function-sig-label`
    """
    # using funccode
    try:
        return arguments_from_funccode(f)
    except Exception as e:
        if verbose:
            print(e)  # TODO: this might not be such a good idea.
            print("Extracting arguments from f.func_code/__code__ fails")

    # using __call__ funccode
    try:
        return arguments_from_call_funccode(f)
    except Exception as e:
        if verbose:
            print(e)  # TODO: this might not be such a good idea.
            print("Extracting arguments from f.__call__.func_code/__code__ fails")

    # try:
    #     return list(inspect.getargspec(f.__call__)[0][1:])
    # except Exception as e:
    #     if verbose:
    #         print(e)
    #         print("inspect.getargspec(f)[0] fails")

    # try:
    #     return list(inspect.getargspec(f)[0])
    # except Exception as e:
    #     if verbose:
    #         print(e)
    #         print("inspect.getargspec(f)[0] fails")

    # now we are parsing __call__.__doc__
    # we assume that __call__.__doc__ doesn't have self
    # this is what cython gives
    try:
        t = arguments_from_docstring(f.__call__.__doc__)
        if t[0] == 'self':
            t = t[1:]
        return t
    except Exception as e:
        if verbose:
            print(e)
            print("fail parsing __call__.__doc__")

    # how about just __doc__
    try:
        t = arguments_from_docstring(f.__doc__)
        if t[0] == 'self':
            t = t[1:]
        return t
    except Exception as e:
        if verbose:
            print(e)
            print("fail parsing __doc__")

    raise TypeError("Unable to obtain function signature")


def describe(f, verbose=False):
    """Try to extract the function argument names.

    .. seealso::

        :ref:`function-sig-label`
    """
    return better_arg_spec(f, verbose)


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
    tmp = ren
    if isinstance(ren, str):
        ren = lambda x: tmp + '_' + x
    ret = {}
    prefix = ['limit_', 'fix_', 'error_', ]
    for k, v in fitarg.items():
        vn = k
        pf = ''
        for p in prefix:
            if k.startswith(p):
                vn = k[len(p):]
                pf = p
        newvn = pf + ren(vn)
        ret[newvn] = v
    return ret


def true_param(p):
    """Check if ``p`` is a parameter name, not a limit/error/fix attributes.
    """
    return (not p.startswith('limit_') and
            not p.startswith('error_') and
            not p.startswith('fix_'))


def param_name(p):
    """Extract parameter name from attributes.

    Examples:

    - ``fix_x`` -> ``x``
    - ``error_x`` -> ``x``
    - ``limit_x`` -> ``x``
    """
    prefix = ['limit_', 'error_', 'fix_']
    for prf in prefix:
        if p.startswith(prf):
            return p[len(prf):]
    return p


def extract_iv(b):
    """Extract initial value from fitargs dictionary."""
    return dict((k, v) for k, v in b.items() if true_param(k))


def extract_limit(b):
    """Extract limit from fitargs dictionary."""
    return dict((k, v) for k, v in b.items() if k.startswith('limit_'))


def extract_error(b):
    """Extract error from fitargs dictionary."""
    return dict((k, v) for k, v in b.items() if k.startswith('error_'))


def extract_fix(b):
    """extract fix attribute from fitargs dictionary"""
    return dict((k, v) for k, v in b.items() if k.startswith('fix_'))


def remove_var(b, exclude):
    """Exclude variable in exclude list from b."""
    return dict((k, v) for k, v in b.items() if param_name(k) not in exclude)


def make_func_code(params):
    """Make a func_code object to fake function signature.

    You can make a funccode from describable object by::

        make_func_code(describe(f))
    """
    class FuncCode(object):
        __slots__ = ('co_varnames', 'co_argcount')
    fc = FuncCode()
    fc.co_varnames = params
    fc.co_argcount = len(params)
    return fc


def format_exception(etype, evalue, tb):
    # work around for https://bugs.python.org/issue17413
    # the issue is not fixed in Python-3.7
    import traceback
    s = "".join(traceback.format_tb(tb))
    return "%s: %s\n%s" % (etype.__name__, evalue, s)
