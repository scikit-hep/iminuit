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
import inspect
import StringIO
import re


class Struct:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, s):
        return self.__dict__[s]


def arguments_from_docstring(doc):
    """Parse first line of docstring for argument name

    Docstring should be of the form ``min(iterable[, key=func])``.

    It can also parse cython docstring of the form
    ``Minuit.migrad(self[, int ncall_me =10000, resume=True, int nsplit=1])``
    """
    if doc is None:
        raise RuntimeError('__doc__ is None')
    sio = StringIO.StringIO(doc.lstrip())
    #care only the firstline
    #docstring can be long
    line = sio.readline()
    if line.startswith("('...',)"):
        line=sio.readline()#stupid cython
    p = re.compile(r'^[\w|\s.]+\(([^)]*)\).*')
    #'min(iterable[, key=func])\n' -> 'iterable[, key=func]'
    sig = p.search(line)
    if sig is None:
        return []
    # iterable[, key=func]' -> ['iterable[' ,' key=func]']
    sig = sig.groups()[0].split(',')
    ret = []
    for s in sig:
        #print s
        #get the last one after all space after =
        #ex: int x= True
        tmp = s.split('=')[0].split()[-1]
        #clean up non _+alphanum character
        ret.append(''.join(filter(lambda x: str.isalnum(x) or x=='_', tmp)))
        #re.compile(r'[\s|\[]*(\w+)(?:\s*=\s*.*)')
        #ret += self.docstring_kwd_re.findall(s)
    ret = filter(lambda x: x!='', ret)

    if len(ret)==0:
        raise RuntimeError('Your doc is unparsable\n'+doc)

    return ret


def is_bound(f):
    """test whether f is bound function"""
    return getattr(f, 'im_self', None) is not None


def dock_if_bound(f, v):
    """dock off self if bound function is passed"""
    return v[1:] if is_bound(f) else v


def better_arg_spec(f, verbose=False):
    """extract function signature

    ..seealso::

        :ref:`function-sig-label`
    """

    try:
        vnames = f.func_code.co_varnames
        #bound method and fake function will be None
        if is_bound(f):
            #bound method dock off self
            return list(vnames[1:f.func_code.co_argcount])
        else:
            #unbound and fakefunc
            return list(vnames[:f.func_code.co_argcount])
    except Exception as e:
        if verbose:
            print e #this might not be such a good dea.
            print "f.func_code.co_varnames[:f.func_code.co_argcount] fails"
        #using __call__ funccode

    try:
        #vnames = f.__call__.func_code.co_varnames
        return list(f.__call__.func_code.co_varnames[1:f.__call__.func_code.co_argcount])
    except Exception as e:
        if verbose:
            print e #this too
            print "f.__call__.func_code.co_varnames[1:f.__call__.func_code.co_argcount] fails"

    try:
        return list(inspect.getargspec(f.__call__)[0][1:])
    except Exception as e:
        if verbose:
            print e
            print "inspect.getargspec(f)[0] fails"

    try:
        return list(inspect.getargspec(f)[0])
    except Exception as e:
        if verbose:
            print e
            print "inspect.getargspec(f)[0] fails"

    #now we are parsing __call__.__doc__
    #we assume that __call__.__doc__ doesn't have self
    #this is what cython gives
    try:
        t = arguments_from_docstring(f.__call__.__doc__)
        if t[0]=='self':
            t = t[1:]
        return t
    except Exception as e:
        if verbose:
            print e
            print "fail parsing __call__.__doc__"

    #how about just __doc__
    try:
        t = arguments_from_docstring(f.__doc__)
        if t[0]=='self':
            t = t[1:]
        return t
    except Exception as e:
        if verbose:
            print e
            print "fail parsing __doc__"

    raise TypeError("Unable to obtain function signature")
    return None

def describe(f, verbose=False):
    """try to extract function arguements name

    .. seealso::

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

        #prefixing
        figarg_rename({'x':1, 'limit_x':1, 'fix_x':1, 'error_x':1},
            lambda pname: 'prefix_'+pname)
        #{'prefix_x':1, 'limit_prefix_x':1, 'fix_prefix_x':1, 'error_prefix_x':1}

    """
    tmp = ren
    if isinstance(ren, basestring):
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
    return dict((k, v) for k, v in b.items() if true_param(k))


def extract_limit(b):
    """extract limit from fitargs dictionary"""
    return dict((k, v) for k, v in b.items() if k.startswith('limit_'))


def extract_error(b):
    """extract error from fitargs dictionary"""
    return dict((k, v) for k, v in b.items() if k.startswith('error_'))


def extract_fix(b):
    """extract fix attribute from fitargs dictionary"""
    return dict((k, v) for k, v in b.items() if k.startswith('fix_'))


def remove_var(b, exclude):
    """exclude variable in exclude list from b"""
    return dict((k, v) for k, v in b.items() if param_name(k) not in exclude)


def make_func_code(params=None):
    """make a func_code object to fake function signature

        you can make a funccode from describeable object by::

            make_func_code(describe(f))
    """
    return Struct(co_varnames=params, co_argcount=len(params))
