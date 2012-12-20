# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Hard Core Tutorial
# ==================
# 
# Typically in fitting, performance matters. Python is slow since it does tons of extra stuff(name lookup etc.). We can fix that with cython and numpy. This tutorial will demonstate how one would write a model which can take data and fit to the data. We will be demonstrating two ways: fastway and generic way.

# <markdowncell>

# ##Basic Cython
# Before we go on lets talk about how to use cython efficiently. Cython speed things up by using static type where it can. Generally the more type information you tell them the better it can generate C code.
# 
# Cython has a very handy option call annotate which lets you know which line of code is static type which one make a call to python object.

# <codecell>

%pylab inline
%load_ext cythonmagic
from iminuit import Minuit

# <codecell>

%%cython --annotate

def slow_f(n):
    x = 100.
    for i in range(n):
        x+=n
    return x

#you tell it more type information like this
def fast_f(int n):
    cdef double x=100.
    cdef int i
    for i in range(n):
        x+=n
    return x
        

# <markdowncell>

# You can see that there is yellow code line and white code line.
# 
# - yellow code line means calling to python code
# - white code line means native C
# 
# Basically, your goal is to get as many white lines as possible. By telling as much type information as possible.
# 
# You can also click on each line to see what code cython actually generate for you. (You many need to double click it)

# <codecell>

%timeit -n10 -r10 slow_f(100)

# <codecell>

%timeit -n10 -r10 fast_f(100)

# <markdowncell>

# ###Quick And Dirty way
# Let's look at how to write a cython cost function

# <codecell>

%pylab inline
%load_ext cythonmagic
from iminuit import Minuit

# <codecell>

data = randn(1e6)*3+2 #mu=2, sigma=3
hist(data,bins=100, histtype='step');

# <codecell>

%%cython --force
#use --annotate if you wonder what kind of code it generates
cimport cython
import numpy as np
cimport numpy as np #overwritten those from python with cython
from libc.math cimport exp, M_PI, sqrt, log
from iminuit.util import describe, make_func_code

@cython.embedsignature(True)#dump the signatre so describe works
cpdef mypdf(double x, double mu, double sigma):
    #cpdef means generate both c function and python function
    cdef double norm = 1./(sqrt(2*M_PI)*sigma)
    cdef double ret = exp(-1*(x-mu)*(x-mu)/(2.*sigma*sigma))*norm
    return ret

cdef class QuickAndDirtyLogLH:#cdef is here to reduce name lookup for __call__
    cdef np.ndarray data
    cdef int ndata
    
    def __init__(self, data):
        self.data = data
        self.ndata = len(data)
    
    @cython.embedsignature(True)#you need this to dump function signature in docstring
    def compute(self, double mu, double sigma):
        cdef np.ndarray[np.double_t, ndim=1] mydata = self.data
        cdef double loglh = 0.
        cdef tuple t
        cdef double thisdata
        for i in range(self.ndata):
            thisdata = mydata[i]
            loglh -= log(mypdf(mydata[i],mu,sigma))
        return loglh

# <codecell>

describe(mypdf)

# <codecell>

lh = QuickAndDirtyLogLH(data)
describe(lh.compute)

# <codecell>

m = Minuit(lh.compute, mu=1.5, sigma=2.5, error_mu=0.1, 
    error_sigma=0.1, limit_sigma=(0.1,10.0))

# <codecell>

%timeit -n1 -r1 m.migrad()

# <markdowncell>

# Have your cython PDF in a separate file
# ---------------------------------------
# 
# Lots of time your stuff is incredibly complicated and doesn't fit in ipython notebook. Or you may want to reuse your PDF in many notebooks. We have external_pdf.pyx in the same directory as this tutorial. This is how you load it.

# <codecell>

%pylab inline
%load_ext cythonmagic
from iminuit import Minuit

# <codecell>

import pyximport;
pyximport.install(
    setup_args=dict(
        include_dirs=[np.get_include()],#include directory
    #    libraries = ['m']#'stuff you need to link (no -l)
    #    library_dirs ['some/dir']#library dir
    #    extra_compile_args = ['-g','-O2'],
    #    extra_link_args=['-some-link-flags'],
    ),
    reload_support=True,#you may also find this useful
) #if anything funny is going on look at your console
import external_pdf

# <codecell>

#reload(external_pdf) #you may find this useful for reloading your module

# <codecell>

data = randn(1e6)
lh = external_pdf.External_LogLH(data)

# <codecell>

m = Minuit(lh.compute,mu=1.5, sigma=2.5, error_mu=0.1, 
    error_sigma=0.1, limit_sigma=(0.1,10.0))

# <codecell>

%timeit -r1 -n1 m.migrad()

# <markdowncell>

# ###Generic Reusable Cost Function
# 
# Sometime we want to write a cost function that will take in any pdf and data and compute appropriate
# cost function. This is slower than the previous example but will make your code much more reusable.

# <codecell>

%pylab inline
%load_ext cythonmagic
from iminuit import Minuit

# <codecell>

data = randn(1e6)*3+2 #mu=2, sigma=3
hist(data,bins=100, histtype='step');

# <markdowncell>

# This is how you do it

# <codecell>

%%cython
#use --annotate if you wonder what kind of code it generates
cimport cython
import numpy as np
cimport numpy as np #overwritten those from python with cython
from iminuit.util import make_func_code, describe
from libc.math cimport log

cdef class LogLH:#cdef is here to reduce name lookup for __call__
    cdef np.ndarray data
    cdef int ndata
    cdef public func_code
    cdef object pdf
    
    def __init__(self, pdf, data):
        self.data = data
        self.ndata = len(data)
        #the important line is here
        self.func_code = make_func_code(describe(pdf)[1:])#1: dock off independent param
        self.pdf = pdf
    
    #@cython.boundscheck(False)#you can turn off bound checking
    def __call__(self, *arg):
        cdef np.ndarray[np.double_t, ndim=1] mydata = self.data
        cdef double loglh = 0.
        cdef tuple t
        for i in range(self.ndata):
            #it's slower because we need to do so many python stuff
            #to do generic function call
            #if you are python hacker and know how to get around this
            #please let us know
            t = (mydata[i],) + arg
            loglh -= log(self.pdf(*t))
        return loglh

# <markdowncell>

# And your favorite PDF

# <codecell>

%%cython
#use --annotate if you wonder what kind of code it generates
from libc.math cimport exp, M_PI, sqrt, log
cimport cython

@cython.binding(True)
def mypdf(double x, double mu, double sigma): 
    #cpdef means generate both c function and python function
    cdef double norm = 1./(sqrt(2*M_PI)*sigma)
    cdef double ret = exp(-1*(x-mu)*(x-mu)/(2.*sigma*sigma))*norm
    return ret

# <codecell>

mylh = LogLH(mypdf,data)

# <codecell>

print describe(mypdf)
print describe(mylh)

# <codecell>

m=Minuit(mylh, mu=1.5, sigma=2.5, error_mu=0.1, 
    error_sigma=0.1, limit_sigma=(0.1,10.0))

# <codecell>

describe(mypdf)

# <codecell>

%timeit -n1 -r1 m.migrad() #you can feel it's much slower than before

# <codecell>

#before
x = linspace(-10,12,100)
before = np.fromiter((mypdf(xx,1.5,2.5) for xx in x), float);
after = np.fromiter((mypdf(xx,m.values['mu'],m.values['sigma']) for xx in x), float);
plot(x,before, label='before')
plot(x,after, label='after')
hist(data, normed=True, bins=100,histtype='step',label='data')
legend();

# <markdowncell>

# ###Parallel Computing With Cython and OpenMP
# *For this tutorial you will need a compiler with openmp support. GCC has one. However, clang does NOT support it.*
# 
# Computer nowadays are multi-core machines so it makes sense to utilize all of them. This method is fast but quite restricted and cubersome since you need to write function such that cython can figure out its reentrant-ness. And you need some understanding of thread-local and thread-share variable.
# 
# You can read [prange](http://wiki.cython.org/enhancements/prange) from cython wiki for more information and how to gain a more complete control over paralelization. The official documentation is [here](http://docs.cython.org/src/userguide/parallelism.html)

# <codecell>

%load_ext cythonmagic
from iminuit import Minuit

# <codecell>

%%cython -f -c-fopenmp --link-args=-fopenmp -c-g
#use --annotate if you wonder what kind of code it generates
cimport cython
import numpy as np
cimport numpy as np #overwritten those from python with cython
from libc.math cimport exp, M_PI, sqrt, log
from iminuit.util import describe, make_func_code
import multiprocessing
from cython.parallel import prange


#notice nogil a the end (no global intepreter lock)
#cython doesn't do a super thorough check for this
#so make sure your function is reentrant this means approximately 
#just simple function compute simple stuff based on local stuff and no read/write to global 
@cython.embedsignature(True)#dump the signatre so describe works
@cython.cdivision(True)
cpdef double mypdf(double x, double mu, double sigma) nogil:
    #cpdef means generate both c function and python function
    cdef double norm
    cdef double ret
    norm = 1./(sqrt(2*M_PI)*sigma)
    ret = exp(-1*(x-mu)*(x-mu)/(2.*sigma*sigma))*norm
    return ret

cdef class ParallelLogLH:#cdef is here to reduce name lookup for __call__
    cdef np.ndarray data
    cdef int ndata
    cdef int njobs
    cdef np.ndarray buf#buffer for storing result from each job
    def __init__(self, data, njobs=None):
        self.data = data
        self.ndata = len(data)
        self.njobs = njobs if njobs is not None else multiprocessing.cpu_count()
        self.buf = np.empty(njobs)
    
    @cython.boundscheck(False)
    @cython.embedsignature(True)#you need this to dump function signature in docstring
    def compute(self, double mu, double sigma):
        cdef np.ndarray[np.double_t, ndim=1] mydata = self.data
        cdef double loglh = 0.
        cdef tuple t
        cdef double thisdata
        cdef int i=0
        #in parallel computing you need to be careful which variable is
        #thread private which variable is shared between thread
        #otherwise you will get into hard to detect racing condition
        #cython rule of thumb(guess rule) is
        # 1) assigned before use is thread private
        # 2) read-only is thread-shared
        # 3) inplace modification only is thread shared
        cdef int njobs = self.njobs
        cdef double tmp
        with nogil:
            for i in prange(self.ndata, 
                            num_threads=njobs, 
                            chunksize=10000, 
                            schedule='dynamic'):#split into many threads
                thisdata = mydata[i] #this is assigned before read so it's thread private
                tmp  = mypdf(thisdata,mu,sigma) #also here assigne before read
                loglh -= log(tmp) #inplace modification so loglh is thread shared
        return loglh

# <codecell>

data = randn(1e7)#10 millions

# <codecell>

plh = ParallelLogLH(data)

# <codecell>

plh.compute(1.5,2.0)

# <codecell>

describe(plh.compute)

# <codecell>

m=Minuit(plh.compute,mu=1.5, sigma=2.5, error_mu=0.1, 
    error_sigma=0.1, limit_sigma=(0.1,10.0))

# <codecell>

%%timeit -n1 -r1
m.migrad()

# <codecell>


