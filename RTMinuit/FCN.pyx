from util import better_arg_spec
from cpython cimport bool
cdef class FCN:
    cdef f
    cdef int narg
    cdef bool verbose
    cdef varnames
    def __init__(self,f,verbose=False):
        self.f = f
        args = better_arg_spec(f)
        self.varnames = args
        self.narg = len(args)
        self.verbose = verbose
        
    def __call__(self,npar,gin,f,par,flag):
        #FCN(Int_t&npar, Double_t*gin, Double_t&f, Double_t*par, Int_t flag)
        p = tuple((par[i] for i in range(self.narg)))
        #print p
        ret =self.f(*p)
        if self.verbose: self.print_call(p,ret)
        f[0] = ret
        
    def print_call(self,p,ret):
        fmt = '%5.2f | '%ret
        for i,par in enumerate(p):
            token = '%6s=%7.4f '%(self.varnames[i],par)
            fmt += token
        print fmt
