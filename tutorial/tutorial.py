# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab inline
from iminuit import Minuit, describe, Struct

# <markdowncell>

# ##Really Quick Start
# Let go through a quick course about how to minimize things. If you use PyMinuit before you will find that iminuit is very similar to PyMinuit.
# 
# Our first example is about trying to minimize a simple function:
# 
# $f(x,y,z) = (x-1)^2 + (y-2)^2 + (z-3)^2 - 1$
# 
# We know easily that the answer has is
# $x=1$, $y=2$, $z=3$ and the minimum value should be $-1$

# <codecell>

def f(x,y,z):
    return (x-1.)**2 + (y-2*x)**2 + (z-3.*x)**2 -1.

# <markdowncell>

# iminuit relies on Python introspection. If you wonder what kind of function iminuit sees, you can use

# <codecell>

describe(f) 

# <markdowncell>

# ###Contruct Minuit Object
# To minimize we need to construct a Minuit object

# <codecell>

m=Minuit(f, x=2, error_x=0.2, limit_x=(-10.,10.), y=3., fix_y=True, print_level=1)

# <markdowncell>

# The initial value/error are optional but it's nice to do it
# and here is how to use it
# 
# - `x=2` set intial value of `x` to 2
# - `error_x=0.2` set the initial stepsize
# - `limit_x = (-1,1)` set the range for `x`
# - `y=2`, `fix_y=True` fix `y` value to 2
# 
# We did not put any constain on z. Minuit will howerver warn you about missig initial error/step(using python builtin warning).

# <markdowncell>

# ####Check that initial setting you put in is what you want

# <codecell>

m.print_param()#or call print_initial_param
#bonus: if you click the + button on the top left corner it will show latex table
#which you can copy and paste to your beamer/latex document

# <markdowncell>

# 
# 
# ###Run Migrad
# 
# Migrad performs Variable Metric Minimization. In a nutshell, it combines steepest descends algorithm along with line search strategy. Migrad is very popular in high energy physics field because of its robustness.

# <codecell>

#Minimize
m.migrad();
#notice also in your prompt it prints out progress

# <markdowncell>

# migrad summary table give you a nice overview of fit status. 
# 
# - All blocks should be green.
# - Red means something bad. 
# - Yellow is a caution(fit is good but you ran over call limit)
# 
# You can use the return value of `migrad()` to check fit status. Most important field is `is_valid`.

# <markdowncell>

# ###Accessing values and errors
# ####Accessing Values

# <codecell>

#and this is how you get the the value
print 'parameters', m.parameters
print 'args', m.args
print 'value', m.values

# <markdowncell>

# ####Error(parabolic)

# <codecell>

#and the error
print 'error', m.errors

# <markdowncell>

# ####Function minimum

# <codecell>

#and function value at the minimum
print 'fval', m.fval
#Tip: you can also obtain value at current state by
print 'current state', f(*m.args)

# <markdowncell>

# ####Correlation and Covariance Matrix

# <codecell>

#covariance, correlation matrix
#remember y is fixed
print 'covariance', m.covariance
print 'matrix()', m.matrix() #covariance
print 'matrix(correlation=True)', m.matrix(correlation=True) #correlation
m.print_matrix() #correlation
#again click the + button on the top left corner for latex code

# <markdowncell>

# ####Fit status

# <codecell>

#get mimization status
print m.get_fmin()
print m.get_fmin().is_valid

# <markdowncell>

# ###Contour and $\chi^2$/Likelihood profile
# $\chi^2$ and contour can be obtained easily

# <codecell>

#minos contour
xminos, zminos, ctr = m.mncontour('x','z')
fill(*zip(*ctr)) #looks kinda ugly, right?

# <codecell>

#drawing it nicely
m.draw_mncontour('x','z', nsigma=4);

# <codecell>

#or you can get the gridded data
x, y, g, r = m.mncontour_grid('x','z', nsigma=3) # r is the raw data
pcolormesh(x,y,g)
colorbar()

# <codecell>

#1D value Scan
x,y = m.profile('x',subtract_min=True);
plot(x,y) #if you have matplotlib

# <codecell>

#we also provide convenience wrapper for drawing it
m.draw_profile('x');

# <codecell>

#2d contour NOT minos contour
x,y,z = m.contour('x','y',subtract_min=True)
cs = contour(x,y,z)
clabel(cs)

# <codecell>

#or a convenience wrapper
m.draw_contour('x','z');

# <markdowncell>

# ###Hesse and Minos

# <markdowncell>

# ####Hesse
# Hesse find the error by finding the inverse of second derivative matrix(hessian). The error assume parabolic shape at the minimum. Hesse error is symmetric by construct. Hesse is always called at the end of migrad to get the error. You normally don't need to call it manually.

# <codecell>

m.hesse()

# <markdowncell>

# ####minos
# 
# minos multidimensionally scan likelihood/$\chi^2$ until to find the contour where the value of the cost function increase by `UP`(see `set_up`). It takes really long time but give the correct error(unless it fails). 

# <codecell>

m.minos() #call m.minos('x') if you need minos error for just 1 variable
print m.get_merrors()['x']
print m.get_merrors()['x'].lower
print m.get_merrors()['x'].upper

# <markdowncell>

# ###Printing Out Nice Tables
# you can force use print_* to do various html display

# <codecell>

m.print_param()
m.print_matrix()

# <markdowncell>

# ##Alternative Ways to define function
# ###Cython
# If you want speed with minimal code change this is the way to do it. This is a quick way to use cython. For a hard core cython see hard-core-tutorial.ipynb.

# <codecell>

#sometimes we want speeeeeeed
%load_ext cythonmagic
%pylab inline
from iminuit import Minuit, describe, Struct

# <codecell>

%%cython --force
cimport cython
import numpy as np
cimport numpy as np #overwritten those from python with cython

#@cython.binding(True) this works too
@cython.embedsignature(True)#dump signature in pydoc so describe can extract signature
def cython_f(double x,double y,double z):
    return (x-1.)**2 + (y-2.)**2 + (z-3.)**2 -1.

# <codecell>

#you can always see what iminuit will see
print describe(cython_f)

# <codecell>

m = Minuit(cython_f)
m.migrad()
print m.values

# <markdowncell>

# ###Callable object ie: __call__
# This is useful if you need to bound your object to some data

# <codecell>

%pylab inline
from iminuit import Minuit, describe, Struct

# <codecell>

x = [1,2,3,4,5]
y = [2,4,6,8,10]# y=2x
class StraightLineChi2:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __call__(self,m,c): #lets try to find slope and intercept
        chi2 = sum((y - m*x+c)**2 for x,y in zip(self.x,self.y))
        return chi2

# <codecell>

chi2 = StraightLineChi2(x,y)
describe(chi2)

# <codecell>

m = Minuit(chi2)
m.migrad()
print m.values

# <markdowncell>

# ###Faking a function signature
# This is missing from PyMinuit. iminuit allows you to take funciton sinature by using `func_code.co_varnames` and `func_code.co_argcount`. This is very useful for making a higher order function that takes PDF and data in to calculate appropriate cost function.

# <codecell>

%pylab inline
from iminuit import Minuit, describe, Struct

# <codecell>

#this is very useful if you want to build a generic cost functor
#this is actually how probfit is implemented
from iminuit.util import make_func_code
x = [1,2,3,4,5]
y = [2,4,6,8,10]# y=2x
class Chi2Functor:
    def __init__(self,f,x,y):
        self.f = f
        self.x = x
        self.y = y
        f_sig = describe(f)
        #this is how you fake function 
        #signature dynamically
        self.func_code = make_func_code(f_sig[1:])#docking off independent variable
        self.func_defaults = None #this keeps np.vectorize happy
    def __call__(self,*arg):
        #notice that it accept variable length
        #positional arguments
        chi2 = sum((y-self.f(x,*arg))**2 for x,y in zip(self.x,self.y))
        return chi2

# <codecell>

def linear(x,m,c):
    return m*x+c

def parabola(x,a,b,c):
    return a*x**2 + b*x + c 

# <codecell>

linear_chi2 = Chi2Functor(linear,x,y)
describe(linear_chi2)

# <codecell>

m = Minuit(linear_chi2)
m.migrad();
print m.values

# <codecell>

#now here is the beauty
#you can use the same Chi2Functor now for parabola
parab_chi2 = Chi2Functor(parabola,x,y)
describe(parab_chi2)

# <codecell>

m = Minuit(parab_chi2,x,y)
m.migrad()
print m.values

# <markdowncell>

# ####Last Resort: Forcing function signature
# 
# built-in function normally do not have signature. Function from swig belongs in this categories. Python intro spection will fails and we have to force function signature.

# <codecell>

%pylab inline
from iminuit import Minuit, describe, Struct

# <codecell>

%%cython
#sometimes you get a function with absolutely no signature
def nosig_f(x,y):
    return x**2+(y-4)**2

# <codecell>

#something from swig will give you a function with no
#signature information
try:
    describe(nosig_f)#it raise error
except Exception as e:
    print e

# <codecell>

#Use forced_parameters
m = Minuit(nosig_f, forced_parameters=('x','y'))

# <codecell>

m.migrad()
print m.values

# <codecell>

print f

# <markdowncell>

# ###Frontend
# Frontend affects how the output from migrad/minos etc are displayed. iminuit is shipped with two frontends. ConsoleFrontend print in text format and HtmlFrontend print html object to Ipython notebook. When you construct Minuit object the front end is selected automatically. If you are in IPython it will use Html frontend; otherwise, it will use console fronend. You can force Minuit to use frontend of your choice too.

# <codecell>

%pylab inline
from iminuit import Minuit, describe, Struct

# <codecell>

#this is just showing off console frontend (you can force it)
from iminuit.ConsoleFrontend import ConsoleFrontend
def f(x,y,z):
    return (x-1.)**2 + (y-2.)**2 + (z-3.)**2 -1.
m = Minuit(f, frontend=ConsoleFrontend())

# <codecell>

m.migrad();

# <codecell>


