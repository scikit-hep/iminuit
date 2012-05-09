from util import better_arg_spec
cdef class FCN:
    cdef f
    cdef int narg
    def __init__(self,f):
        self.f = f
        args = better_arg_spec(f)
        narg = len(args)
        self.narg = narg
    def __call__(self,npar,gin,f,par,flag):
        #FCN(Int_t&npar, Double_t*gin, Double_t&f, Double_t*par, Int_t flag)
        p = tuple((par[i] for i in range(self.narg)))
        #print p
        ret =self.f(*p)
        f[0] = ret