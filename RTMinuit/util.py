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

def fitarg_rename(fitarg, ren):
    """
    rename variable names in fitarg with rename function ren
    taking care of limit_, fix_, error_
    """
    tmp = ren
    if isinstance(ren, basestring): ren = lambda x: tmp+'_'+x
    ret = {}
    prefix = ['limit_','fix_','error_',]
    for k,v in fitarg.items():
        vn = k
        pf = ''
        for p in prefix:
            if k.startswith(p): 
                vn = k[len(p):]
                pf = p
        newvn = pf+ren(vn)
        ret[newvn] = v
    return ret
