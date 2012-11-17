import inspect

class Struct:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__str__()


def better_arg_spec(f):
    #built-in funccode
    try:
        return f.func_code.co_varnames[:f.func_code.co_argcount]
    except Exception as e:
        #print e
        pass
        #using __call__ funccode
    try:
        vnames = f.__call__.func_code.co_varnames
        return f.__call__.func_code.co_varnames[1:f.__call__.func_code.co_argcount]
    except Exception as e:
        #print e
        pass
    return inspect.getargspec(f)[0]


def describe(f,quiet=True):
    if not quiet: print better_arg_spec(f)
    return better_arg_spec(f)


def fitarg_rename(fitarg, ren):
    """
    rename variable names in fitarg with rename function ren
    taking care of limit_, fix_, error_
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
    return not p.startswith('limit_') and not p.startswith('error_') and not p.startswith('fix_')


def param_name(p):
    prefix = ['limit_', 'error_', 'fix_']
    for prf in prefix:
        if p.startswith(prf):
            return p[len(prf):]
    return p


def extract_iv(b):
    return {k: v for k, v in b.items() if true_param(k)}


def extract_limit(b):
    return {k: v for k, v in b.items() if k.startswith('limit_')}


def extract_error(b):
    return {k: v for k, v in b.items() if k.startswith('error_')}


def extract_fix(b):
    return {k: v for k, v in b.items() if k.startswith('fix_')}


def remove_var(b, exclude):
    return {k: v for k, v in b.items() if param_name(k) not in exclude}