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
]
import inspect

class Struct:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__str__()


def better_arg_spec(f, verbose):
    """extract function signature

    ..seealso::

        :ref:`function-sig-label`
    """
    try:
        return f.func_code.co_varnames[:f.func_code.co_argcount]
    except Exception as e:
        if verbose:
            print e #this might not be such a good dea.
            print "f.func_code.co_varnames[:f.func_code.co_argcount] fails"
        #using __call__ funccode
    try:
        #vnames = f.__call__.func_code.co_varnames
        return f.__call__.func_code.co_varnames[1:f.__call__.func_code.co_argcount]
    except Exception as e:
        if verbose:
            print e #this too
            print "f.__call__.func_code.co_varnames[1:f.__call__.func_code.co_argcount] fails"

    return inspect.getargspec(f)[0]


def describe(f,verbose=False):
    """try to extract function arguements name

    ..seealso::

        :ref:`function-sig-label`
    """
    return better_arg_spec(f, verbose)


def fitarg_rename(fitarg, ren):
    """
    rename variable names in fitarg with rename function

    ::

        #simple renaming
        fitarg_rename({'x':1, 'limit_x':1, 'fix_x':1, 'error_x':1},
            lambda pname: 'y' if pname=='x' else pname)
        #{'y':1, 'limit_y':1, 'fix_y':1, 'error_y':1},

    ::
    
        #prefixing
        figarg_rename(fitarg_rename({'x':1, 'limit_x':1, 'fix_x':1, 'error_x':1},
            lambda pname: 'prefix_'+pname)
        #{'prefix_x':1, 'limit_prefix_x':1, 'fix_prefix_x':1, 'error_prefix_x':1}
    """
    tmp = ren
    if isinstance(ren, basestring): ren = lambda x: tmp + '_' + x
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
    """check if p is parameter name not a limit/error/fix attributes"""
    return not p.startswith('limit_') and\
        not p.startswith('error_') and\
        not p.startswith('fix_')


def param_name(p):
    """
    extract parameter name from attributes eg

        fix_x -> x
        error_x -> x
        limit_x -> x
    """
    prefix = ['limit_', 'error_', 'fix_']
    for prf in prefix:
        if p.startswith(prf):
            return p[len(prf):]
    return p


def extract_iv(b):
    """extract initial value from fitargs dictionary"""
    return {k: v for k, v in b.items() if true_param(k)}


def extract_limit(b):
    """extract limit from fitargs dictionary"""
    return {k: v for k, v in b.items() if k.startswith('limit_')}


def extract_error(b):
    """extract error from fitargs dictionary"""
    return {k: v for k, v in b.items() if k.startswith('error_')}


def extract_fix(b):
    """extract fix attribute from fitargs dictionary"""
    return {k: v for k, v in b.items() if k.startswith('fix_')}


def remove_var(b, exclude):
    """exclude variable in exclude list from b"""
    return {k: v for k, v in b.items() if param_name(k) not in exclude}
