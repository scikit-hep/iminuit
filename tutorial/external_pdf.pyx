cimport cython
import numpy as np
cimport numpy as np #overwritten those from python with cython
from libc.math cimport exp, M_PI, sqrt, log

@cython.embedsignature(True)
cpdef external_mypdf(double x, double mu, double sigma):
    #cpdef means generate both c function and python function
    cdef double norm = 1./(sqrt(2*M_PI)*sigma)
    cdef double ret = exp(-1*(x-mu)*(x-mu)/(2.*sigma*sigma))*norm
    return ret

cdef class External_LogLH:#cdef is here to reduce name lookup for __call__
    cdef np.ndarray data
    cdef int ndata
    cdef public func_code
    cdef object pdf

    def __init__(self, data):
        self.data = data
        self.ndata = len(data)

    #@cython.boundscheck(False)#you can turn off bound checking
    @cython.embedsignature(True)#you need this for describe to work
    def compute(self, double mu, double sigma):
        cdef np.ndarray[np.double_t, ndim=1] mydata = self.data
        cdef double loglh = 0.
        cdef double thisdata
        for i in range(self.ndata):
            thisdata = mydata[i]
            loglh -= log(external_mypdf(mydata[i],mu,sigma))
        return loglh 
