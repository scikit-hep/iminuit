"""iminuit utility functions and classes.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import re
from .py23_compat import is_string

__all__ = [
    'describe',
    'Struct',
    'fitarg_rename',
    'true_param',
    'param_name',
    'extract_iv',
    'extract_limit',
    'extract_error',
    'extract_fix',
    'remove_var',
    'arguments_from_docstring',
]


class Struct(dict):
    """A Struct is a Python dict with tab completion.

    Example:

    >>> s = Struct(a=42)
    >>> s['a']
    42
    >>> s.a
    42
    """
    def __setattr__(self, key, value):
        try:
            self[key] = value
        except KeyError:
            raise AttributeError

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError


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
    if is_string(ren):
        def ren(x):
            return tmp + '_' + x
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
