# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Hard Core Tutorial
# ------------------
# 
# Typically in fitting, performance matters. Python is slow since it does tons of extra stuff(name lookup etc.). We can fix that with cython and numpy. This tutorial will demonstate how one would write a model which can take data and fit to the data. We will be demonstrating two ways: fastway and generic way.

# <codecell>

#otherwise import those missing library
%pylab inline

# <codecell>

%load_ext cythonmagic

# <codecell>

from iminuit import *

# <markdowncell>

# ##Basic Cython
# Before we go on lets talk about how to use cython efficiently. Cython speed things up by using static type where it can. Generally the more type information you tell them the better it can generate C code.
# 
# Cython has a very handy option call annotate which lets you know which line of code is static type which one make a call to python object.

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

%timeit slow_f(100)

# <codecell>

%timeit fast_f(100)

# <markdowncell>

# ###Quick And Dirty way
# Let's look at how to write a cython cost function

# <codecell>

data = randn(1e6)*3+2 #mu=2, sigma=3
hist(data,bins=100, histtype='step');

# <codecell>

%%cython
#use --annotate if you wonder what kind of code it generates
cimport cython
import numpy as np
cimport numpy as np #overwritten those from python with cython
from libc.math cimport exp, M_PI, sqrt, log

@cython.embedsignature(True)
cpdef mypdf(double x, double mu, double sigma):
    #cpdef means generate both c function and python function
    cdef double norm = 1./(sqrt(2*M_PI)*sigma)
    cdef double ret = exp(-1*(x-mu)*(x-mu)/(2.*sigma*sigma))*norm
    return ret

cdef class QuickAndDirtyLogLH:#cdef is here to reduce name lookup for __call__
    cdef np.ndarray data
    cdef int ndata
    cdef public func_code
    cdef object pdf
    
    def __init__(self, data):
        self.data = data
        self.ndata = len(data)
    
    #@cython.boundscheck(False)#you can turn off bound checking   
    @cython.embedsignature(True)#you need this for describe to work
    def __call__(self, double mu, double sigma):
        """__call__(self, mu, sigma)"""
        cdef np.ndarray[np.double_t, ndim=1] mydata = self.data
        cdef double loglh = 0.
        cdef tuple t
        cdef double thisdata
        for i in range(self.ndata):
            thisdata = mydata[i]
            loglh -= log(mypdf(mydata[i],mu,sigma))
        return loglh

# <codecell>

lh = QuickAndDirtyLogLH(data)

# <codecell>

describe(mypdf)

# <codecell>

describe(lh, verbose=True)

# <codecell>

m = Minuit(lh)

# <codecell>


# <markdowncell>

# ###Generic Reusable Cost Function
# 
# Sometime we want to write a cost function that will take in any pdf and data and compute appropriate
# cost function. This is slower than the previous example but will make your code much more reusable.
# 
# Let's say we want to do a unbinned likelihood fit of a million data points to a guassian.
# This is to show how dist_fit was written. You are welcome to contribute to dist_fit with your favorite likelihood function.

# <codecell>

data = randn(1e6)*3+2 #mu=2, sigma=3
hist(data,bins=100, histtype='step');

# <codecell>


# <codecell>


# <codecell>


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
        self.func_code = make_func_code(describe(pdf)[1:])#for auto signature extraction
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

describe(mylh)

# <codecell>

m=Minuit(mylh, mu=1.5, sigma=2.5, error_mu=0.1, 
    error_sigma=0.1, limit_sigma=(0.1,10.0), forced_parameters=['mu','sigma'])

# <codecell>

describe(mypdf)

# <codecell>

%timeit -n1 -r1 m.migrad() #you can feel it's much slower than before

# <codecell>

#before
x = linspace(-10,12,100)
before = np.fromiter((mypdf(xx,1.5,2.5) for xx in x), float);
after = np.fromiter((mypdf(xx,m.values['mu'],m.values['sigma']) for xx in x), float);

# <codecell>

plot(x,before, label='before')
plot(x,after, label='after')
hist(data, normed=True, bins=100,histtype='step',label='data')
legend();

# <codecell>


# <markdowncell>

# Have your cython PDF in a separate file
# ---------------------------------------
# 
# Lots of time your stuff is incredibly complicated and doesn't fit in ipython notebook. Or you may want to reuse your PDF in many notebooks. Here is how you do it.

# <codecell>


# <codecell>


# <codecell>

import StringIO
import re
def arguments_from_docstring(doc):
    """Parse first line of docstring for argument name

    Docstring should be of the form 'min(iterable[, key=func])\n'.
    It can also parse cython docstring of the form
    Minuit.migrad(self[, int ncall_me =10000, resume=True, int nsplit=1])
    """
    if doc is None:
        raise RuntimeError('__doc__ is None')
    sio = StringIO.StringIO(doc.lstrip())
    #care only the firstline
    #docstring can be long
    line = sio.readline()
    p = re.compile(r'^[\w|\s.]+\(([^)]*)\).*')
    #'min(iterable[, key=func])\n' -> 'iterable[, key=func]'
    sig = p.search(line)
    if sig is None:
        return []
    print sig.groups()
    # iterable[, key=func]' -> ['iterable[' ,' key=func]']
    sig = sig.groups()[0].split(',')
    ret = []
    for s in sig:
        #print s
        tmp = s.split('=')[0].split()[-1]
        ret.append(''.join(filter(lambda x :str.isalnum(x) or x=='_', tmp)))
        #re.compile(r'[\s|\[]*(\w+)(?:\s*=\s*.*)')
        #ret += self.docstring_kwd_re.findall(s)
    return ret

# <codecell>

s = 'Minuit.migrad(self[, int ncall_me =10000, resume=True, int nsplit=1])'
arguments_from_docstring(s)

# <codecell>

filter(lambda x :str.isalnum(x) or x=='_', s)

# <codecell>


# <codecell>


# <codecell>


